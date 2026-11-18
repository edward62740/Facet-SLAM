#!/usr/bin/env python
"""Rendering and depth metrics of a finished run.

Usage: python scripts/eval_render_metrics.py RUN_DIR [--stride 5] [--max_frames N] [--renders 12]
Writes RUN_DIR/render_metrics.json, depth_l1.json and renders/.
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

from mesh_splatting.arguments import OptimizationParams, PipelineParams  # noqa: E402
from mesh_splatting.render_utils import get_psnr  # noqa: E402
from mesh_splatting.scene.triangle_model import TriangleModel  # noqa: E402
from mesh_splatting.triangle_renderer import render  # noqa: E402
from mesh_splatting.utils.graphics_utils import getProjectionMatrix2  # noqa: E402
from mesh_splatting.utils.loss_utils import ssim  # noqa: E402
from utils.cameras import Camera  # noqa: E402
from utils.dataset import load_dataset  # noqa: E402


def latest_iteration(mesh_dir: Path) -> Path:
    its = [p for p in mesh_dir.glob("iteration_*") if (p / "point_cloud_state_dict.pt").exists()]
    its = [p for p in its if p.name.split("_")[1].isdigit()]
    return max(its, key=lambda p: int(p.name.split("_")[1]))


def save_panels(path, gt, img, pkg, cam, soup, W2C, K, psnr):
    """Save ground truth, render, raw triangles and normals as one image."""
    H, W = gt.shape[1:]

    def bgr(x):
        return cv2.cvtColor((x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    n = pkg["rend_normal"].permute(1, 2, 0) @ cam.world_view_transform[:3, :3]
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    n[..., 2] = -n[..., 2].abs()
    n = (n * torch.tensor([1.0, -1.0, -1.0], device=n.device) + 1) / 2
    n = torch.where(pkg["rend_alpha"][0][..., None] > 0.5, n, torch.ones_like(n))
    verts, colour = soup
    S = 2
    Vc = verts @ W2C[:3, :3].T + W2C[:3, 3]
    z = Vc[:, :, 2]
    u = S * (K[0] * Vc[:, :, 0] / np.maximum(z, 1e-3) + K[2])
    v = S * (K[1] * Vc[:, :, 1] / np.maximum(z, 1e-3) + K[3])
    ok = (z.min(1) > 0.05) & (u.max(1) > 0) & (u.min(1) < S * W) & (v.max(1) > 0) & (v.min(1) < S * H)
    normal = np.cross(Vc[:, 1] - Vc[:, 0], Vc[:, 2] - Vc[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    shade = 0.65 + 0.35 * np.abs(normal @ (np.array([0.3, -0.8, -0.5]) / np.linalg.norm([0.3, -0.8, -0.5])))
    canvas = np.full((S * H, S * W, 3), 255, np.uint8)
    for i in np.flatnonzero(ok)[np.argsort(-z.mean(1)[ok])]:
        pts = np.stack([u[i], v[i]], 1).astype(np.int32).reshape(-1, 1, 2)
        c = (colour[i][::-1] * shade[i] * 255).astype(int).tolist()
        cv2.fillPoly(canvas, [pts], c, cv2.LINE_AA)
        cv2.polylines(canvas, [pts], True, [max(x - 70, 0) for x in c], 1, cv2.LINE_AA)
    panels = [bgr(gt), bgr(img), cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA), bgr(n.permute(2, 0, 1))]
    for panel, label in zip(panels, ("Ground truth", "Render  %.2f dB" % psnr, "Raw triangles", "Normals")):
        cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--gsicp_protocol", action="store_true",
                    help="also report PSNR as GS-ICP SLAM computes it: both images zeroed where the sensor depth is 0, "
                         "mean over all pixels")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--renders", type=int, default=12,
                    help="frames, evenly spaced over the run at the estimated poses, saved to RUN_DIR/renders as "
                         "ground truth | render over raw triangles | normals; 0: none")
    ap.add_argument("--traj", default="trajectory_all.json",
                    help="trajectory file in the run dir (e.g. trajectory_all_online.json: the poses as tracked, "
                         "before the re-anchoring to the optimised keyframe poses)")
    ap.add_argument("--tag", default="",
                    help="non-empty: write render_metrics_<tag>_stride<N>.json and leave the standard files untouched")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    config = yaml.safe_load((run_dir / "config.yml").read_text())
    config.setdefault("Results", {})["use_gui"] = False
    ds_path = config["Dataset"]["dataset_path"]
    if not Path(ds_path).is_absolute():
        ds_path = str(_root / ds_path)
    dataset = load_dataset(SimpleNamespace(distributed=False), ds_path, config)
    proj = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=dataset.fx, fy=dataset.fy, cx=dataset.cx, cy=dataset.cy,
                                W=dataset.width, H=dataset.height).transpose(0, 1)
    opt = OptimizationParams(ArgumentParser())
    for key, value in config.get("opt_params", {}).items():
        if hasattr(opt, key):
            setattr(opt, key, value)
    pipe = PipelineParams(ArgumentParser())
    it_dir = latest_iteration(run_dir / "mesh")
    sh_degree = int(config.get("Training", {}).get("sh_degree", 0) or 0)      # the run's colour bands
    triangles = TriangleModel(sh_degree=sh_degree, config=config)
    triangles.load_parameters_for_slam_resume(str(it_dir), device="cuda")
    triangles.scaling = opt.upscaling_factor
    background = torch.zeros(3, device="cuda")
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")

    traj = json.loads((run_dir / args.traj).read_text()) if (run_dir / args.traj).exists() else []
    by_uid = {r["uid"]: r for r in traj}
    kf_uids = sorted(r["uid"] for r in traj if r["is_kf"])
    if not kf_uids:
        payload = torch.load(str(it_dir / "keyframes_state.pt"), map_location="cpu")
        kf_uids = sorted(int(s["uid"]) for s in payload.get("keyframes", []))
    n = len(dataset.poses) if args.max_frames is None else min(args.max_frames, len(dataset.poses))
    end = min(n, max(by_uid) + 1) if by_uid else n      # a run stopped early: its own frames only
    sampled = list(range(0, end, args.stride))
    # keyframes skipped, as stock eval_rendering and MonoGS
    kf_set = set(kf_uids)
    nonkf = [u for u in range(0, end - 1, args.stride) if u not in kf_set]

    # raw triangles for save_panels
    faces = triangles._triangle_indices.long()
    faces = faces[torch.sigmoid(triangles.vertex_weight.float()).reshape(-1)[faces].mean(1) >= 0.09]
    with torch.no_grad():
        soup = (triangles.vertices.float()[faces].cpu().numpy().astype(float),
                (0.5 + 0.28209479177387814 * triangles._features_dc.float()[:, 0, :]).clamp(0, 1)[faces].mean(1)
                .cpu().numpy())
    renders_dir = run_dir / ("renders_" + args.tag if args.tag else "renders")

    def metrics_for(uid, R, T, save=False):
        cam = Camera.init_from_dataset(dataset, uid, proj)
        cam.update_RT(R.to("cuda").float(), T.to("cuda").float())
        with torch.no_grad():
            pkg = render(cam, triangles, pipe, background)
        img = torch.clamp(pkg["render"], 0, 1)
        gt = cam.original_image.cuda()
        mask = gt > 0
        out = dict(
            psnr=float(get_psnr(img[mask].unsqueeze(0), gt[mask].unsqueeze(0))),
            ssim=float(ssim(img.unsqueeze(0), gt.unsqueeze(0))),
            lpips=float(lpips_fn(img.unsqueeze(0), gt.unsqueeze(0))),
        )
        if args.gsicp_protocol and cam.depth is not None:
            dm = (torch.as_tensor(cam.depth, device="cuda", dtype=torch.float32) > 0).float()[None]
            mse = ((img * dm - gt * dm) ** 2).mean()
            out["psnr_gsicp"] = float(-10.0 * torch.log10(mse))
        if cam.depth is not None:
            gtd = torch.as_tensor(cam.depth, device="cuda", dtype=torch.float32)
            valid = gtd > 0.01
            channels = (("depth_l1_expected_covered_cm", "expected_depth"),
                        ("depth_l1_median_covered_cm", "render_depth_median"))
            for key, ch in channels:
                d = pkg[ch][0]
                m = valid & (d > 0)
                out[key] = float((d[m] - gtd[m]).abs().mean() * 100) if m.any() else float("nan")
            # holes count as error, as stock
            d, m = pkg["render_depth_median"][0], gtd > 0
            out["depth_l1_cm"] = float((d[m] - gtd[m]).abs().mean() * 100) if m.any() else float("nan")
            out["depth_coverage"] = float(((d > 0) & valid).float().sum() / valid.float().sum())
        if save:
            W2C = np.eye(4)
            W2C[:3, :3], W2C[:3, 3] = np.asarray(R, dtype=float), np.asarray(T, dtype=float)
            save_panels(renders_dir / f"frame_{uid:05d}.jpg", gt, img, pkg, cam, soup, W2C,
                        (dataset.fx, dataset.fy, dataset.cx, dataset.cy), out["psnr"])
        return out

    # align the GT poses to the map frame (Umeyama on the keyframe centres)
    def c2w_from(R, T):
        M = np.eye(4)
        M[:3, :3] = np.asarray(R)
        M[:3, 3] = np.asarray(T)
        return np.linalg.inv(M)
    if by_uid and len(kf_uids) >= 3:
        est_c = np.array([c2w_from(by_uid[u]["R_kf"], by_uid[u]["T_kf"])[:3, 3] for u in kf_uids])
        gt_c = np.array([np.linalg.inv(dataset.poses[u])[:3, 3] for u in kf_uids])
        ms, md = gt_c.mean(0), est_c.mean(0)
        Hm = (gt_c - ms).T @ (est_c - md)
        U, _, Vt = np.linalg.svd(Hm)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R_a = Vt.T @ D @ U.T
        t_a = md - R_a @ ms
        T_a_inv = np.linalg.inv(np.block([[R_a, t_a[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
        align_rmse_cm = float(np.sqrt((((gt_c @ R_a.T + t_a) - est_c) ** 2).sum(1).mean()) * 100)
    else:
        T_a_inv = np.eye(4)
        align_rmse_cm = float("nan")

    def gt_pose_in_map(u):
        W = np.asarray(dataset.poses[u]) @ T_a_inv
        return torch.tensor(W[:3, :3]), torch.tensor(W[:3, 3])

    results = {"align_rmse_cm": align_rmse_cm}
    modes = {
        "gt_kf": [(u, *gt_pose_in_map(u)) for u in kf_uids],
        "gt_all": [(u, *gt_pose_in_map(u)) for u in sampled],
        "gt_nonkf": [(u, *gt_pose_in_map(u)) for u in nonkf],
    }
    if by_uid:
        def tracked(uids):
            return [(u, torch.tensor(by_uid[u]["R_kf"]), torch.tensor(by_uid[u]["T_kf"])) for u in uids if u in by_uid]
        modes["tracked_kf"] = tracked(kf_uids)
        modes["tracked_all"] = tracked(sampled)
        modes["tracked_nonkf"] = tracked(nonkf)
    save_mode = "tracked_all" if modes.get("tracked_all") else "gt_all"
    uids = [f[0] for f in modes[save_mode]]
    picks = np.linspace(0, len(uids) - 1, min(args.renders, len(uids))).round().astype(int) if uids else []
    save = {uids[i] for i in picks}
    if save:
        renders_dir.mkdir(exist_ok=True)
    # one bar over all modes, on stderr
    bar = tqdm(total=sum(len(f) for f in modes.values()), desc="Evaluating renders", unit="frame", dynamic_ncols=True)
    for name, frames in modes.items():
        if not frames:
            continue
        bar.set_postfix_str(name)
        per = []
        for u, R, T in frames:
            per.append(metrics_for(u, R, T, name == save_mode and u in save))
            bar.update()
        agg = {}
        for k in per[0]:
            vals = np.array([p[k] for p in per], dtype=float)
            agg[k] = float(np.nanmean(vals))
        agg["n_frames"] = len(per)
        results[name] = agg
        tqdm.write(name + " " + json.dumps({k: round(v, 4) for k, v in agg.items()}))
    bar.close()
    if save:
        print(f"renders: {len(save)} frames of {save_mode} in {renders_dir}")
    results["iteration"] = it_dir.name
    results["num_triangles"] = int(triangles._triangle_indices.shape[0])
    if args.tag:
        (run_dir / f"render_metrics_{args.tag}_stride{args.stride}.json").write_text(json.dumps(results, indent=1))
        return
    if args.gsicp_protocol:
        # comparison run: keep the standard files
        (run_dir / f"render_metrics_gsicp_stride{args.stride}.json").write_text(json.dumps(results, indent=1))
        return
    (run_dir / "render_metrics.json").write_text(json.dumps(results, indent=1))
    # depth L1 as stock: keyframes, median depth, holes included
    src_mode = "tracked_kf" if "tracked_kf" in results else "gt_kf"
    src = results[src_mode]
    poses = "their SLAM" if src_mode == "tracked_kf" else "aligned ground-truth"
    (run_dir / "depth_l1.json").write_text(json.dumps({
        "depth_l1_cm": src.get("depth_l1_cm"), "depth_l1_covered_cm": src.get("depth_l1_median_covered_cm"),
        "protocol": f"{src_mode}: keyframes at {poses} poses, "
                    "median depth; depth_l1_cm: mean over gt > 0, holes count (scripts/eval_depth_l1.py); "
                    "depth_l1_covered_cm: rendered pixels only"}))
    print("align_rmse_cm", round(align_rmse_cm, 3))


if __name__ == "__main__":
    main()
