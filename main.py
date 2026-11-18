"""Facet-SLAM: RGB-D triangle-splatting SLAM with a CPU-based GICP tracker.

  Usage: python main.py --dataset fr1_desk --max_fps 30 --eval

Please see --help and README.md.
"""
import argparse
import collections
import difflib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime


def _clean_ld_library_path():
    """Drop CUDA toolkit dirs from LD_LIBRARY_PATH and re-exec, so torch loads its own CUDA libraries."""
    entries = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    keep = [p for p in entries if not re.search(r"(^|/)cuda(-?[\d.]+)?(/|$)", p.lower())]   # .../cuda-12.4/lib64
    if keep != entries and not os.environ.get("FACET_LD_CLEAN"):
        os.environ["LD_LIBRARY_PATH"] = ":".join(keep)
        os.environ["FACET_LD_CLEAN"] = "1"
        os.execv(sys.executable, [sys.executable] + getattr(sys, "orig_argv", [None] + sys.argv)[1:])


_clean_ld_library_path()

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
TUM = {"fr1_desk": "freiburg1_desk", "fr1_desk2": "freiburg1_desk2", "fr1_xyz": "freiburg1_xyz",
       "fr1_room": "freiburg1_room", "fr1_360": "freiburg1_360", "fr1_rpy": "freiburg1_rpy", "fr2_xyz": "freiburg2_xyz",
       "fr3_office": "freiburg3_long_office_household", "fr3_stfar": "freiburg3_structure_texture_far",
       "fr3_stnear": "freiburg3_structure_texture_near"}
REPLICA = ("room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4")
# optional Training.async keys
ASYNC_KEYS = ("enabled", "tracker", "kf_burst_iters", "window_k", "topology_interval", "publish_interval", "max_fps")

parser = argparse.ArgumentParser(description="Facet-SLAM: RGB-D triangle-splatting SLAM with a CPU GICP tracker")
parser.add_argument("--dataset", type=str, default=None,
                    help=f"Named sequence: {' '.join(TUM)} {' '.join(REPLICA)} scene<id> (ScanNet, in TUM format under "
                         "datasets/scannet/) d435_<recording>")
parser.add_argument("--dataset_path", type=str, default=None,
                    help="Path to the sequence (default: from --dataset, else the config's Dataset.dataset_path)")
parser.add_argument("--config", type=str, default=None,
                    help="Per-sequence config, e.g. configs/rgbd/tum/fr1_desk.yaml (default: from --dataset)")
parser.add_argument("--preset", type=str, default="auto", choices=["auto", "tum", "replica", "scannet", "none"],
                    help="Final-configuration overlay from configs/facet/ (auto: scannet for configs/rgbd/scannet, else by "
                         "Dataset.type; none: the per-sequence config alone)")
parser.add_argument("--seed", type=int, default=0, help="Seed of torch / numpy / random")
parser.add_argument("--max_fps", type=float, default=0,
                    help="Pace the tracker at this camera rate (0: uncapped; the paper's TUM protocol uses 30)")
parser.add_argument("--num_frames", type=int, default=None, help="Stop after N frames (smoke tests)")
parser.add_argument("--output_dir", type=str, default="output",
                    help="Results go to OUTPUT_DIR/<parent>_<sequence>/<timestamp>/ (the last two components of the "
                         "dataset path)")
parser.add_argument("--eval", action="store_true",
                    help="After SLAM, render every 5th frame at the estimated poses: PSNR, SSIM, LPIPS over the "
                         "non-keyframes (as stock and MonoGS) and over all of them, depth L1, and twelve frames saved "
                         "to the run directory's renders/ (scripts/eval_render_metrics.py)")
parser.add_argument("--verbose", action="store_true",
                    help="Show the log of the SLAM processes (default: it goes to OUTPUT_DIR/<name>_seed<seed>.log)")
parser.add_argument("--dry_run", action="store_true", help="Print the merged configuration and exit (no GPU)")
parser.add_argument("--viewer", choices=["map", "render", "both"], default=None,
                    help="Live viewer (Rerun, live_viewer.py), off by default. map: the triangle map as a coloured mesh "
                         "with the trajectory, the keyframes and the camera (as GS-ICP SLAM shows its map); render: the "
                         "mapper's render of the map at the newest tracked pose beside the input image; both. The pose "
                         "follows every frame, the render --viewer_render_hz, the mesh --viewer_hz; the 3D view is "
                         "interactive")
parser.add_argument("--viewer_sink", choices=["web", "spawn", "connect"], default="web",
                    help="web: serve a web viewer on localhost:VIEWER_WEB_PORT (no display needed; forward the two "
                         "ports from a remote machine); spawn: the native Rerun viewer; connect: a viewer already "
                         "running at --viewer_url")
parser.add_argument("--viewer_port", type=int, default=9876, help="gRPC port the viewer data goes through")
parser.add_argument("--viewer_web_port", type=int, default=9090, help="web sink: the port of the web viewer")
parser.add_argument("--viewer_url", type=str, default=None,
                    help="connect sink: the viewer's gRPC url (default rerun+http://127.0.0.1:VIEWER_PORT/proxy)")
parser.add_argument("--viewer_hz", type=float, default=2.0, help="Most map-mesh refreshes per second")
parser.add_argument("--viewer_render_hz", type=float, default=15.0,
                    help="Most render and input-image refreshes per second (the camera pose follows every frame)")
parser.add_argument("--viewer_depth_range", type=float, nargs=2, default=[0.25, 8.0], metavar=("NEAR", "FAR"),
                    help="Render depth colour map: turbo on a log scale between NEAR and FAR metres")
parser.add_argument("--viewer_map_refresh", type=float, default=10.0,
                    help="Seconds between full re-sends of the map mesh (in between, only new or changed keyframe "
                         "pieces are sent)")
parser.add_argument("--viewer_layout", choices=["grid", "map"], default="grid",
                    help="--viewer both: grid (default): input, render, map and render depth as four equal panels with "
                         "the viewer's own panels hidden; map: the map large, the images in a column")
parser.add_argument("--viewer_image_scale", type=float, default=None,
                    help="Scale of the images sent to the viewer (default 1.0 for the spawn sink, 0.5 otherwise)")
parser.add_argument("--viewer_keep", action="store_true",
                    help="web sink: keep serving after the run until Ctrl-C (the browser keeps what it already has)")
parser.add_argument("--d435", action="store_true",
                    help="Live input from an Intel RealSense D435 / D435i instead of a sequence (d435_capture.py, "
                         "d435_live.py): colour and aligned depth at --d435_size and --d435_fps, the TUM overlay. The "
                         "tracker takes the camera's newest frame each time; every frame is also recorded. Ctrl-C stops "
                         "the camera, the run then finishes and saves as usual")
parser.add_argument("--d435_size", type=str, default="640x480", metavar="WxH",
                    help="Stream resolution, colour and depth alike (640x480, the TUM format; 848x480 is the D435's "
                         "native depth mode)")
parser.add_argument("--d435_fps", type=int, default=30, help="Stream rate (30, as TUM; the D435 also offers 15 and 6)")
parser.add_argument("--d435_max_depth", type=float, default=5.0,
                    help="Depth beyond this many metres is dropped at capture (0: keep all)")
parser.add_argument("--d435_seconds", type=float, default=0, help="Stop the camera after this long (0: until Ctrl-C)")
parser.add_argument("--d435_serial", type=str, default=None, help="Camera serial number, with several attached")
parser.add_argument("--d435_spool", type=str, default=None,
                    help="Directory the frames are recorded to (default OUTPUT_DIR/d435_live/<time>; uncompressed, about 2.8 GB per "
                         "minute at 640x480 and 30 Hz)")
parser.add_argument("--d435_replay", type=str, default=None, metavar="RECORDING",
                    help="Feed a recorded sequence (datasets/d435i/<recording>, with its caminfo.txt) through the live "
                         "path at its recorded rate instead of a camera")
parser.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE",
                    help="Override a config value, dot notation (e.g. -o Training.async.gicp.stride=4). An unknown key "
                         "is an error; +KEY=VALUE adds a key no config defines")


def resolve_dataset(name):
    """--dataset NAME -> (config, dataset path)."""
    if name in TUM:
        return f"configs/rgbd/tum/{name}.yaml", os.path.join(ROOT, "datasets/tum", "rgbd_dataset_" + TUM[name])
    if name.startswith(("room", "office")):
        return f"configs/rgbd/replica/{name}.yaml", os.path.join(ROOT, "datasets/replica", name)
    if name.startswith("scene"):        # ScanNet scenes, in TUM format
        return f"configs/rgbd/scannet/{name}.yaml", os.path.join(ROOT, "datasets/scannet", name)
    if name.startswith("d435_"):        # RealSense recordings under datasets/d435i/
        return f"configs/rgbd/d435i/{name[5:]}.yaml", os.path.join(ROOT, "datasets/d435i", name[5:])
    parser.error(f"unknown --dataset {name}")


def start_d435(args):
    """Start the camera process and write the run's config."""
    spool = os.path.abspath(args.d435_spool) if args.d435_spool else os.path.join(
        args.output_dir, "d435_live", datetime.now().strftime("%Y%m%d_%H%M%S"))
    if os.path.exists(spool):
        parser.error(f"--d435_spool {spool} exists: give a new directory")
    os.makedirs(spool)
    width, height = (int(v) for v in args.d435_size.lower().split("x"))
    cmd = [sys.executable, os.path.join(ROOT, "d435_capture.py"), "--out", spool, "--width", str(width), "--height",
           str(height), "--fps", str(args.d435_fps), "--max_depth", str(args.d435_max_depth), "--seconds",
           str(args.d435_seconds)]
    cmd += ["--serial", args.d435_serial] if args.d435_serial else []
    cmd += ["--replay", os.path.abspath(args.d435_replay)] if args.d435_replay else []
    log = os.path.join(spool, "capture.log")
    process = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    calibration, t0 = os.path.join(spool, "calibration.json"), time.time()
    while not os.path.isfile(calibration):
        if process.poll() is not None or time.time() - t0 > 60:
            process.terminate()
            sys.exit(f"the camera did not start (d435_capture.py):\n{open(log).read()[-2000:]}")
        time.sleep(0.05)
    c = json.load(open(calibration))
    keys = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "width", "height", "depth_scale", "distorted")
    dataset = {"dataset_path": spool, "Calibration": {k: c[k] for k in keys}}
    head = f"# main.py --d435, {c['source']}, depth dropped beyond {c['max_depth']} m at capture\n"
    for name, live in (("config.yaml", True), ("replay.yaml", False)):
        with open(os.path.join(spool, name), "w") as f:
            f.write(head + yaml.safe_dump({"inherit_from": "configs/rgbd/tum/base_config.yaml",
                                           "Dataset": dict(dataset, live=True) if live else dataset}))
    return os.path.join(spool, "config.yaml"), spool, process, c


def stop_d435(capture, spool):
    """Stop the camera process."""
    open(os.path.join(spool, "STOP"), "w").close()
    try:
        capture.wait(timeout=15)
    except subprocess.TimeoutExpired:
        capture.kill()


def leaves(d, prefix=""):
    out = {}
    for k, v in d.items():
        out.update(leaves(v, prefix + k + ".") if isinstance(v, dict) else {prefix + k: v})
    return out


def lookup(config, key):
    d = config
    for k in key.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def apply_override(config, key, value):
    """as slam.py applies -o"""
    keys = key.split(".")
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def yaml_value(v):
    """Value as an -o string."""
    s = yaml.safe_dump(v, default_flow_style=True, width=1 << 30).strip()
    s = s[:-3].strip() if s.endswith("...") else s
    assert yaml.safe_load(s) == v, (v, s)
    return s


def known_keys(config):
    """Keys that -o may set."""
    from argparse import ArgumentParser
    from mesh_splatting.arguments import OptimizationParams
    from slam_mp import _gicp_cfg
    keys = set(leaves(config))
    keys |= {"Training.async." + k for k in ASYNC_KEYS}
    keys |= {"Training.async.gicp." + k for k in _gicp_cfg({"Dataset": {"type": "tum"}})}
    keys |= {"opt_params." + k for k in vars(OptimizationParams(ArgumentParser()))}
    return keys


def build_config(args):
    """Merge the configs and the -o overrides."""
    from utils.config_utils import load_config
    cfg, ds = resolve_dataset(args.dataset) if args.dataset else (None, None)
    cfg = args.config or cfg
    if cfg is None:
        parser.error("give --dataset NAME, or --config CFG (with --dataset_path PATH)")
    if not os.path.isfile(cfg):
        parser.error(f"config not found: {cfg}")
    config = load_config(cfg)
    ds = args.dataset_path or ds or config.get("Dataset", {}).get("dataset_path")
    if not ds or not os.path.isdir(ds):
        parser.error(f"dataset not found: {ds}")
    ds = os.path.abspath(ds)
    preset = args.preset
    if preset == "auto" and "/scannet/" in os.path.abspath(cfg).replace(os.sep, "/"):
        preset = "scannet"                            # ScanNet loads as TUM but has its own preset
    if preset == "auto":
        preset = {"tum": "tum", "replica": "replica"}.get(config.get("Dataset", {}).get("type"))
        if preset is None:
            dataset_type = config.get("Dataset", {}).get("type")
            parser.error(f"no final configuration for Dataset.type {dataset_type!r}: pass --preset")
    overlay = None if preset == "none" else f"configs/facet/{preset}.yaml"
    pairs = [("Results.use_gui", False), ("Results.save_dir", args.output_dir)]   # use_gui: as run_tum_sequences.sh
    if overlay:
        pairs += [(k, v) for k, v in leaves(load_config(overlay)).items() if k != "inherit_from"]
    pairs.append(("Dataset.dataset_path", ds))        # read by scripts/eval_render_metrics.py
    if args.viewer:
        pairs += [("Results.viewer.mode", args.viewer), ("Results.viewer.sink", args.viewer_sink),
                  ("Results.viewer.port", args.viewer_port), ("Results.viewer.web_port", args.viewer_web_port),
                  ("Results.viewer.hz", args.viewer_hz), ("Results.viewer.render_hz", args.viewer_render_hz),
                  ("Results.viewer.depth_range", list(args.viewer_depth_range)),
                  ("Results.viewer.map_refresh_s", args.viewer_map_refresh), ("Results.viewer.layout", args.viewer_layout)]
        pairs += [("Results.viewer.image_scale", args.viewer_image_scale)] if args.viewer_image_scale is not None else []
        pairs += [("Results.viewer.url", args.viewer_url)] if args.viewer_url else []
    if args.max_fps > 0:
        pairs.append(("Training.async.max_fps", int(args.max_fps) if args.max_fps.is_integer() else args.max_fps))
    for k, v in pairs:
        apply_override(config, k, v)
    out = [(k, yaml_value(v)) for k, v in pairs]
    known = known_keys(config)
    for o in args.override:
        key, eq, s = o.partition("=")
        key = key.strip()
        if not eq or not key.lstrip("+"):
            parser.error(f"-o must be KEY=VALUE, got: {o}")
        if key.startswith("+"):
            key = key[1:]
        elif key not in known:
            if isinstance(lookup(config, key), dict):
                parser.error(f"-o {key}: a section, not a key")
            near = difflib.get_close_matches(key, sorted(known), n=3, cutoff=0.6)
            parser.error(f"-o {key}: unknown config key" + (f" (did you mean {', '.join(near)}?)" if near else "")
                         + "; +KEY=VALUE adds a key no config defines")
        try:
            v = yaml.safe_load(s)
        except yaml.YAMLError as e:
            parser.error(f"-o {key}: cannot read the value {s!r} ({str(e).splitlines()[0]})")
        if isinstance(v, dict):
            parser.error(f"-o {key}: the value {s!r} reads as a mapping; set its keys one by one")
        if isinstance(v, str) and re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+", v):
            v, s = float(v), yaml_value(float(v))        # YAML 1.1 reads 5e-2 as a string
        apply_override(config, key, v)
        out.append((key, s))
    return cfg, ds, out, config, overlay


def progress_hook(every=10, window=30):
    """tqdm progress bar over the frames."""
    from tqdm import tqdm
    tqdm.monitor_interval = 0                     # no monitor thread next to the tracker
    state = dict(bar=None, t0=0.0, t=collections.deque(maxlen=window + 1))

    def hook(i, total, n_kf, mapper):
        now = time.perf_counter()
        state["t"].append(now)
        if state["bar"] is None:
            state["bar"] = tqdm(total=total, unit="frame", dynamic_ncols=True,
                                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]")
            state["t0"] = now
        if total > state["bar"].total:            # live camera: the total grows
            state["bar"].total = total
        if i % every and i != total:
            return
        t, bar = state["t"], state["bar"]
        current = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else 0.0
        average = (i - 1) / (now - state["t0"]) if i > 1 else 0.0
        bar.set_postfix_str(f"fps {current:.1f}, avg {average:.1f}, kf {n_kf}, tris {mapper.get('triangles', 0)}",
                            refresh=False)
        bar.update(i - bar.n)

    def close():
        if state["bar"] is not None:
            state["bar"].close()
    hook.close = close
    return hook

# print the metrics and stuff at the end
def summarise(run_dir, process_s):
    def read(name):
        p = os.path.join(run_dir, name)
        return json.load(open(p)) if os.path.isfile(p) else {}
    timing = read("timing.json")
    row = read("eval_row.json")
    depth_l1 = read("depth_l1.json")
    render_metrics = read("render_metrics.json")
    nonkf = render_metrics.get("tracked_nonkf", {})
    all_frames = render_metrics.get("tracked_all", {})
    has_gt = any(any(abs(x) > 1e-9 for x in r["T_gt"]) for r in (read("trajectory_all.json") or []))
    n = timing.get("num_frames", 0)
    mesh = os.path.join(run_dir, "mesh")
    if has_gt:
        # this is the window opt. pose ATE RMSE
        ate_final = row.get("ate_all_cm")
        ate_online = 100 * timing["ate_all_online_m"] if "ate_all_online_m" in timing else row.get("ate_all_cm")
    else:
        ate_final = ate_online = None
    tracking_ms = 1e3 * timing["tracking_s"] / max(n - 1, 1) if "tracking_s" in timing else None   # frame 1 is not tracked
    s = dict(frames=n, wall_time_s=timing.get("train_total_s"), fps=timing.get("train_fps"),
             tracking_ms_per_frame=tracking_ms,
             keyframes=timing.get("keyframes"), tracking_keyframes=timing.get("tracking_keyframes"),
             triangles=timing.get("num_triangles"),
             peak_gpu_gb=timing.get("mapper", {}).get("peak_gpu_gb", timing.get("peak_gpu_gb")),
             map_mb=sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(mesh) for f in fs) / 1e6,
             ate_all_final_cm=ate_final, ate_all_online_cm=ate_online, ate_reanchored="ate_all_online_m" in timing and timing.get("gicp_cfg", {}).get("window_opt", True))
    if nonkf:
        # every 5th, skip train (kf)
        s.update(psnr=nonkf["psnr"], ssim=nonkf["ssim"], lpips=nonkf["lpips"], eval_frames=nonkf.get("n_frames"))
    if all_frames:
        s.update(psnr_all=all_frames["psnr"], ssim_all=all_frames["ssim"], lpips_all=all_frames["lpips"],
                 eval_frames_all=all_frames.get("n_frames"))
    if depth_l1:
        # depth L1 over the keyframes
        s.update(depth_l1_cm=depth_l1.get("depth_l1_cm"), depth_l1_covered_cm=depth_l1.get("depth_l1_covered_cm"))
    s.update(process_s=process_s, run_dir=run_dir)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    return s


def print_summary(s, out):
    def show(*values):
        print(*values, file=out)
    show(s["frames"], "frames processed")
    if s["wall_time_s"] is not None:
        show(f"Total time: {s['wall_time_s']:.2f} s")
        show(f"Average FPS: {s['fps']:.2f}")
    if s["tracking_ms_per_frame"] is not None:
        show(f"Average tracking time per frame: {s['tracking_ms_per_frame']:.2f} ms")
        show(f"Keyframes: {s['keyframes']} ({s['tracking_keyframes']} tracking keyframes)")
    show(f"Triangles: {s['triangles']}")
    show(f"Peak GPU memory: {s['peak_gpu_gb']} GB")
    show(f"Map size on disk: {s['map_mb']:.1f} MB")
    if s["ate_all_final_cm"] is None:
        if s.get("eval_ok", True):
            show("ATE RMSE: no ground truth")
        else:
            show("ATE RMSE: unavailable (scripts/icp_eval.py failed)")
    elif s["ate_reanchored"]:
        show(f"ATE RMSE (all frames): online {s['ate_all_online_cm']:.3f} cm, final {s['ate_all_final_cm']:.3f} cm")
    else:
        show(f"ATE RMSE (all frames): {s['ate_all_final_cm']:.3f} cm")
    if "psnr" in s:
        show(f"PSNR: {s['psnr']:.2f}  SSIM: {s['ssim']:.3f}  LPIPS: {s['lpips']:.3f}  (keyframes excluded)")
    if "psnr_all" in s:
        show(f"PSNR: {s['psnr_all']:.2f}  SSIM: {s['ssim_all']:.3f}  LPIPS: {s['lpips_all']:.3f}  (keyframes included)")
    if s.get("depth_l1_cm") is not None:
        show(f"Depth L1: {s['depth_l1_cm']:.2f} cm")
    show("Results saved to", s["run_dir"])


def main():
    args = parser.parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    if args.dataset_path:
        args.dataset_path = os.path.abspath(args.dataset_path)
    if args.config and not os.path.isfile(os.path.join(ROOT, args.config)):
        args.config = os.path.abspath(args.config)
    for flag in ("d435_spool", "d435_replay"):
        if getattr(args, flag):
            setattr(args, flag, os.path.abspath(getattr(args, flag)))
    os.chdir(ROOT)                                # configs inherit_from paths relative to the repository
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")    # as scripts/run_tum_sequences.sh
    os.environ.setdefault("WANDB_MODE", "disabled")
    capture = camera = None
    if args.d435 or args.d435_replay:
        for flag in ("dataset", "config", "dataset_path", "eval", "dry_run", "max_fps", "num_frames"):
            if getattr(args, flag):
                parser.error(f"--{flag} does not go with --d435 (for --eval, run the recording again: "
                             "--config <spool>/replay.yaml)")
        # ign
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        args.config, args.dataset_path, capture, camera = start_d435(args)
    try:
        cfg, ds, overrides, config, overlay = build_config(args)
    except BaseException: #added for robustness, we must have the logs save on ctrl c
        if capture:
            stop_d435(capture, args.dataset_path)
        raise
    name = args.dataset or ("d435_live" if capture else os.path.basename(ds.rstrip("/")))
    argv = ["slam.py", "--config", cfg, "--dataset_path", ds, "--seed", str(args.seed)]
    argv += ["--num_frames", str(args.num_frames)] if args.num_frames else []
    for k, s in overrides:
        argv += ["-o", f"{k}={s}"]
    if args.dry_run:
        config.setdefault("Results", {})["seed"] = args.seed
        print(f"# {name}: {cfg} + {overlay or 'no overlay'}, dataset {ds}")
        print("# slam.py command:", " ".join(argv[1:]))
        print(yaml.dump(config), end="")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    out = sys.stdout
    if not args.verbose:
        # stdout of all processes goes to the log; stderr stays
        log = os.path.join(args.output_dir, f"{name}_seed{args.seed}.log")
        sys.stdout.flush()
        out = os.fdopen(os.dup(1), "w")
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(fd, 1)
        os.close(fd)
        print(" ".join(sys.argv), flush=True)
    print(f"Running {name} ({cfg} + {overlay or 'no overlay'}), seed {args.seed}", file=out, flush=True)
    if capture:
        print(f"Camera: {camera['source']}, {camera['width']}x{camera['height']}, fx {camera['fx']:.1f} fy "
              f"{camera['fy']:.1f}; recording to {ds}\nKeep the camera still until tracking starts. Ctrl-C stops the camera; the run then "
              "finishes and saves",
              file=out, flush=True)
    if args.viewer and args.viewer_sink == "web":
        print(f"Viewer: http://localhost:{args.viewer_web_port}/?url=rerun%2Bhttp://localhost:{args.viewer_port}/proxy"
              f"  (from a remote machine forward ports {args.viewer_web_port} and {args.viewer_port})", file=out,
              flush=True)
    t0 = time.time()
    import slam
    import slam_mp
    slam_mp.frame_hook = progress_hook()
    sys.argv = argv
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")   # slam.py names the run directory by the time
    try:
        slam.main()
    finally:
        slam_mp.frame_hook.close()
        if capture:                                # also when the run failed
            stop_d435(capture, ds)
            signal.signal(signal.SIGINT, signal.default_int_handler)

    # the run directory slam.py made
    path = ds.rstrip("/").split("/")
    base = os.path.join(config["Results"]["save_dir"], path[-2] + "_" + path[-1] if len(path) >= 2 else "run")
    runs = [d for d in (os.listdir(base) if os.path.isdir(base) else []) if d >= stamp]
    if not runs:
        sys.exit(f"no run directory under {base}: nothing was saved (Results.save_results is off?)")
    run_dir = os.path.join(base, min(runs))
    if args.eval and subprocess.run([sys.executable, "scripts/eval_render_metrics.py", run_dir]).returncode:
        print("render metrics failed", file=sys.stderr)
    ate_eval = subprocess.run([sys.executable, "scripts/icp_eval.py", run_dir, "--sequence", name])   # -> eval_row.json
    eval_ok = ate_eval.returncode == 0
    if not eval_ok:
        print("icp_eval failed", file=sys.stderr)
    s = summarise(run_dir, time.time() - t0)
    s["eval_ok"] = eval_ok
    print_summary(s, out)
    if capture:
        frames = open(os.path.join(ds, "END")).read().strip() if os.path.isfile(os.path.join(ds, "END")) else "?"
        print(f"Camera frames: {frames} recorded, {s.get('frames', '?')} tracked; run it again offline with "
              f"--config {os.path.join(ds, 'replay.yaml')} --dataset_path {ds}", file=out, flush=True)
    if args.viewer and args.viewer_sink == "web" and args.viewer_keep:
        print("Viewer still serving; Ctrl-C to exit", file=out, flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    if not eval_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
