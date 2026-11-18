#!/usr/bin/env python
"""Write the results row (eval_row.json) of one run directory.

Usage: python scripts/icp_eval.py RUN_DIR [--sequence NAME] [--csv results.csv] [--run_id X] [--notes TEXT]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

COLUMNS = ("run_id,commit,config_hash,sequence,seed,tracker,ate_all_cm,ate_kf_cm,rpe_trans_cm,rpe_rot_deg,"
           "track_ms_median,psnr,ssim,lpips,depth_l1_cm,system_fps,gpu,notes").split(",")


# --------------------------------------------------------------------------- #
# SE(3) helpers (numpy)
# --------------------------------------------------------------------------- #
def w2c(R, T):
    M = np.eye(4)
    M[:3, :3] = np.asarray(R)
    M[:3, 3] = np.asarray(T)
    return M


def umeyama_se3(src, dst):
    """R, t with dst ~ R src + t (no scale)."""
    ms, md = src.mean(0), dst.mean(0)
    H = (src - ms).T @ (dst - md)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = md - R @ ms
    return R, t


def ate_rmse_cm(c2w_est, c2w_gt):
    pe = np.array([m[:3, 3] for m in c2w_est])
    pg = np.array([m[:3, 3] for m in c2w_gt])
    if len(pe) < 3:
        return float("nan")
    R, t = umeyama_se3(pe, pg)
    pa = pe @ R.T + t
    return float(np.sqrt(((pa - pg) ** 2).sum(1).mean()) * 100)


def log_so3(R):
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = math.acos(cos)
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def rpe(w2c_est, w2c_gt):
    """Relative pose error between consecutive frames."""
    dt, dr = [], []
    for k in range(1, len(w2c_est)):
        rel_e = w2c_est[k] @ np.linalg.inv(w2c_est[k - 1])
        rel_g = w2c_gt[k] @ np.linalg.inv(w2c_gt[k - 1])
        E = np.linalg.inv(rel_g) @ rel_e
        dt.append(np.linalg.norm(E[:3, 3]))
        dr.append(np.linalg.norm(log_so3(E[:3, :3])))
    if not dt:
        return float("nan"), float("nan")
    return float(np.sqrt(np.mean(np.square(dt))) * 100), float(np.degrees(np.sqrt(np.mean(np.square(dr)))))


# --------------------------------------------------------------------------- #
def gpu_name():
    try:
        query = ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        out = subprocess.check_output(query).decode().strip().split("\n")[0]
        return out.replace(", ", "/")
    except Exception:
        return "unknown"


def evaluate(run_dir: Path, args):
    cfg = {}
    cfg_path = run_dir / "config.yml"
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    res = cfg.get("Results", {})
    tr = cfg.get("Training", {})
    row = dict.fromkeys(COLUMNS, "")
    row["run_id"] = args.run_id or run_dir.parent.name + "/" + run_dir.name
    row["commit"] = res.get("commit", "")
    row["config_hash"] = res.get("config_hash", "")
    row["sequence"] = args.sequence or run_dir.parent.name
    row["seed"] = res.get("seed", "")
    acfg = tr.get("async", {}) or {}
    kind = acfg.get("tracker", "gicp_cpu") if acfg.get("enabled") else "photometric"   # stock: the photometric tracker
    row["tracker"] = kind
    row["gpu"] = gpu_name()
    row["notes"] = args.notes

    traj_path = run_dir / "trajectory_all.json"
    if traj_path.exists():
        rows = json.loads(traj_path.read_text())
        est = [w2c(r["R"], r["T"]) for r in rows]
        gt = [w2c(r["R_gt"], r["T_gt"]) for r in rows]
        c2w_e = [np.linalg.inv(m) for m in est]
        c2w_g = [np.linalg.inv(m) for m in gt]
        row["ate_all_cm"] = round(ate_rmse_cm(c2w_e, c2w_g), 3)
        kf = [i for i, r in enumerate(rows) if r["is_kf"]]
        if len(kf) >= 3:
            est_kf = [np.linalg.inv(w2c(rows[i]["R_kf"], rows[i]["T_kf"])) for i in kf]
            gt_kf = [c2w_g[i] for i in kf]
            row["ate_kf_cm"] = round(ate_rmse_cm(est_kf, gt_kf), 3)
        rt, rr = rpe(est, gt)
        row["rpe_trans_cm"] = round(rt, 3)
        row["rpe_rot_deg"] = round(rr, 4)
    timing_path = run_dir / "timing.json"
    if timing_path.exists():
        tj = json.loads(timing_path.read_text())
        row["system_fps"] = round(tj.get("train_fps", float("nan")), 4)
        prof = tj.get("profiler", {})
        if kind == "photometric" and "frontend_total" in prof:
            row["track_ms_median"] = prof["frontend_total"]["avg_ms"]
            row["notes"] = (row["notes"] + " track_ms=frontend_avg").strip()
        if "ate_rmse_m" in tj and row["ate_kf_cm"] == "":
            row["ate_kf_cm"] = round(tj["ate_rmse_m"] * 100, 3)
    rm = run_dir / "render_metrics.json"
    src = json.loads(rm.read_text()).get("tracked_all") if rm.exists() else None
    if src:
        # every 5th frame at the estimated poses, keyframes included
        for k in ("psnr", "ssim", "lpips"):
            if src.get(k) is not None:
                row[k] = round(src[k], 4)
        row["notes"] = (row["notes"] + " psnr=tracked_all").strip()
    else:
        # no fallback to other poses
        print(f"warning: no tracked_all rendering metrics in {rm} (run scripts/eval_render_metrics.py); "
              "psnr, ssim, lpips left empty", file=sys.stderr)
        row["notes"] = (row["notes"] + " psnr=missing").strip()
    p = run_dir / "depth_l1.json"
    if p.exists():
        v = json.loads(p.read_text())["depth_l1_cm"]
        row["depth_l1_cm"] = "" if v is None else round(float(v), 4)      # None: the run had no depth
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--sequence", default=None)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    row = evaluate(run_dir, args)
    print(json.dumps(row, indent=1))
    (run_dir / "eval_row.json").write_text(json.dumps(row, indent=1))
    if args.csv:
        import fcntl
        p = Path(args.csv)
        lockf = open(str(p) + ".lock", "a")
        fcntl.flock(lockf, fcntl.LOCK_EX)
        rows = []
        if p.exists():
            def same_run(r):
                return (r["run_id"] == row["run_id"] and r["sequence"] == row["sequence"]
                        and str(r["seed"]) == str(row["seed"]))
            with open(p, newline="") as f:
                rows = [r for r in csv.DictReader(f) if not same_run(r)]
        rows.append(row)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in COLUMNS})
        tmp.replace(p)
        fcntl.flock(lockf, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
