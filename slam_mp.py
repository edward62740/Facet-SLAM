"""The three processes of Facet-SLAM and what they share.

  Tracker (main process, CPU)   tracker.py
  Mapper (spawned, GPU)         mapper.py
  WindowOptProcess (spawned)    window_opt.py, and is optional

message passing protocol
  tracker -> mapper   ("kf", uid, R, T, view_iteration, "track" | "map"), ("correct", id, {uid: 4x4}), ("end", payload)
  mapper -> tracker   ("ready" | "snap" | "done", version, last keyframe uid, target, stats)
  tracker -> window   ("kf", uid, anchor uid, relative pose), ("end",)
  window -> tracker   ("ready",), ("corr" | "final", {uid: C2W})
"""
import queue
import random

import numpy as np
import torch

frame_hook = None    # set by main.py, progress callback/frame

# bunch of hyperparameters, empirically set
DIVERGED_STEP_CM = 25          # inter-frame step logged as diverged (cm)
DIVERGED_STEP_DEG = 30         # same, rotation (deg)
IDENTITY_TOLERANCE = 1e-7      # smaller corrections are not sent
FINAL_POSES_TIMEOUT_S = 600    # wait for the window's final poses (s)
MAPPER_EXIT_TIMEOUT_S = 600    # wait at shutdown (s)
HIDDEN_VERTEX_WEIGHT = -30.0   # vertex weight that hides a triangle
MIN_SOURCE_POINTS = 20         # fewer depth samples: frame not registered
REMAP_MAX_POINTS = 32766       # cv2.remap's point limit
PARENT_CHECK_S = 2.0           # parent-alive check interval (s)


def _log(*values, tag):
    from utils.eval_utils import Log      # heavy import, kept out of the window process
    Log(*values, tag=tag)


def _async_cfg(config):
    return dict(config.get("Training", {}).get("async", {}) or {})

# for reproducibility
def _seed(seed):
    """Seed a spawned process (None: unseeded)."""
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def _gicp_cfg(config):
    """tracker settings, similar to GS-ICP SLAM"""
    tum = config["Dataset"]["type"] == "tum"
    settings = dict(

        stride=5 if tum else 10,                    # subsampling of the depth image
        corr=0.03 if tum else 0.02,                 # maximum correspondence distance (m)
        overlap_th=1e-3 if tum else 5e-4,           # covered if within this squared distance (m^2)
        keyframe_th=0.81 if tum else 0.7,           # tracking keyframe below this covered fraction
        depth_trunc=3.0 if tum else 100.0,          # depth beyond this is ignored (m)
        trackable_opacity=0.09 if tum else 0.05,    # minimum opacity of a triangle in the target

        threads=16,                                 # GICP threads (for CPU, very important!!)
        knn=10,                                     # neighbours of  B_i (the GICP dst approx.)
        cov_floor=0.002,                            # depth-noise floor added to both covariances (m)
        kf_min_gap=0,                               # minimum frames between tracking keyframes
        mapping_kf_every=0,                         # every Nth frame as a hole-filling keyframe
        sbi_levels=[40, 80], sbi_iters=10, sbi_blur=0.75,     # rotation initialisation (_SBIRotation)
        prefetch_workers=1, prefetch_ahead=2,                 # decode ahead so the pipeline doesnt stall
        
        window_opt=True,                            # keyframe pose refinement on / off
        window_size=16, window_iters=3, window_nbrs=8, window_points=1000, window_color_points=400,
        window_threads=4)                                     # keyframe pose refinement (window_opt.py)
    settings.update(dict(_async_cfg(config).get("gicp", {}) or {}))
    return settings


def _pack_poses(rows):
    """Stack per-frame poses into numpy arrays for the queue."""
    keys = ("R", "T", "R_gt", "T_gt")
    return dict(uid=np.array([row[0] for row in rows], np.int64),
                **{key: np.stack([row[i].numpy() for row in rows]) for i, key in enumerate(keys, 1)})


def _unpack_poses(packed):
    return [(int(uid), *(torch.from_numpy(packed[key][i]) for key in ("R", "T", "R_gt", "T_gt")))
            for i, uid in enumerate(packed["uid"])]


def _wait_ready(from_child, process, name):
    """block until the child process has sent "ready" """
    while True:
        try:
            msg = from_child.get(timeout=1.0)
        except queue.Empty:
            if not process.is_alive():
                raise RuntimeError(f"{name} process died")
            continue
        assert msg[0] == "ready", msg[0]
        return


# ----------------------------------------------------------------------------------------------------------------------
# process ENTRY points
# ----------------------------------------------------------------------------------------------------------------------
def mapper_main(args, config, dataset_path, config_path, opt, pipe, num_frames, kf_q, snap_q, done_ev, viewer_link=None):
    from mapper import Mapper
    Mapper(args, config, dataset_path, config_path, opt, pipe, num_frames, kf_q, snap_q, done_ev, viewer_link).run()


def window_main(config, dataset_path, in_q, out_q, seed=None):
    from window_opt import WindowOptProcess
    WindowOptProcess(config, dataset_path, in_q, out_q, seed).run()


def run_async_cpu(args, config, dataset_path, config_path, opt, pipe, num_frames):
    """run SLAM: the tracker here, the mapper and the window optimisation in child processes."""
    from tracker import Tracker
    Tracker(args, config, dataset_path, config_path, opt, pipe, num_frames).run()
