"""GICP tracker on CPU, see mp.py"""
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

import slam_mp
from utils.dataset import load_dataset
from slam_mp import (
    DIVERGED_STEP_CM, DIVERGED_STEP_DEG, FINAL_POSES_TIMEOUT_S, IDENTITY_TOLERANCE, MAPPER_EXIT_TIMEOUT_S,
    MIN_SOURCE_POINTS, REMAP_MAX_POINTS, _async_cfg, _gicp_cfg, _log, _pack_poses, _wait_ready,
)


def _import_pygicp():
    """ensure the proper gicp gets imported"""
    import ctypes
    import glob
    import importlib
    import sys
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native", "pygicp")
    for build_dir in glob.glob(os.path.join(root, "build", "lib.*")):
        if build_dir not in sys.path:
            sys.path.insert(0, build_dir)
    try:
        return importlib.import_module("pygicp")
    except ImportError:
        for library in sorted(glob.glob(os.path.join(root, "libs", "*.so*"))):
            ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
        return importlib.import_module("pygicp")


_SOURCE_FILTERS = {}


def reg_filter(n):
    """Point filter that keeps every point."""
    if n not in _SOURCE_FILTERS:
        _SOURCE_FILTERS[n] = np.arange(1, n + 1, dtype=np.int32)
    return _SOURCE_FILTERS[n]


def _correct_target(rows, corrections):
    """Apply pending window corrections to target rows."""
    ids = np.round(rows[:, 3]).astype(np.int64)
    for correction in corrections:
        for uid, D in correction.items():
            moved = ids == int(uid)
            if moved.any():
                rows[moved, :3] = rows[moved, :3] @ D[:3, :3].T.astype(rows.dtype) + D[:3, 3].astype(rows.dtype)
    return rows


class _SBIRotation:
    """Rotation between consecutive frames from small blurred grey images; initialises GICP."""

    def __init__(self, dataset, levels, iters, blur, huber=0.3):
        self.W = int(dataset.width)
        self.H = int(dataset.height)
        self.iters = int(iters)
        self.blur = float(blur)
        self.huber = float(huber)
        self.levels = [(int(w), int(round(int(w) * self.H / self.W))) for w in levels]
        if max((w - 2) * (h - 2) for w, h in self.levels) > REMAP_MAX_POINTS:
            raise ValueError(f"Training.async.gicp.sbi_levels {list(levels)}: more than {REMAP_MAX_POINTS} pixels")
        self.base = (4 * self.levels[-1][0], int(round(4 * self.levels[-1][0] * self.H / self.W)))
        self.undistort_map = None
        if getattr(dataset, "disorted", False):
            # undistortion maps at the base size
            scale = self.W / self.base[0]
            map_x = cv2.resize(dataset.map1x.astype(np.float32), self.base, interpolation=cv2.INTER_LINEAR)
            map_y = cv2.resize(dataset.map1y.astype(np.float32), self.base, interpolation=cv2.INTER_LINEAR)
            self.undistort_map = ((map_x + 0.5) / scale - 0.5, (map_y + 0.5) / scale - 0.5)
        self.geometry = []     # per level: rays, Jacobian, intrinsics, size
        for w, h in self.levels:
            scale = w / self.W
            fx, fy = dataset.fx * scale, dataset.fy * scale
            cx, cy = (dataset.cx + 0.5) * scale - 0.5, (dataset.cy + 0.5) * scale - 0.5
            vv, uu = np.mgrid[1:h - 1, 1:w - 1]
            x, y = ((uu - cx) / fx).reshape(-1), ((vv - cy) / fy).reshape(-1)
            rays = np.stack([x, y, np.ones_like(x)], 1)
            # d pixel / d rotation at the identity
            J_u = np.stack([-fx * x * y, fx * (1 + x * x), -fx * y], 1)
            J_v = np.stack([-fy * (1 + y * y), fy * x * y, fy * x], 1)
            self.geometry.append((rays, J_u, J_v, fx, fy, cx, cy, w, h))

    def pyramid(self, grey):
        base = cv2.resize(grey, self.base, interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.undistort_map is not None:
            base = cv2.remap(base, self.undistort_map[0], self.undistort_map[1], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        out = []
        for w, h in self.levels:
            small = cv2.GaussianBlur(cv2.resize(base, (w, h), interpolation=cv2.INTER_AREA), (0, 0), self.blur)
            out.append((small - small.mean()) / max(float(small.std()), 1e-6))
        return out

    def _warp(self, reference, geometry, R):
        rays, _, _, fx, fy, cx, cy, w, h = geometry
        q = rays @ R.T
        z = np.maximum(q[:, 2], 1e-6)
        u, v = fx * q[:, 0] / z + cx, fy * q[:, 1] / z + cy
        ok = (q[:, 2] > 0.1) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        warped = cv2.remap(reference, u.astype(np.float32).reshape(-1, 1), v.astype(np.float32).reshape(-1, 1),
                           cv2.INTER_LINEAR).reshape(-1)
        return warped, ok

    def _cost(self, e):
        if not e.size:
            return float("inf")
        a = np.abs(e)
        return float(np.where(a <= self.huber, 0.5 * e * e, self.huber * a - 0.5 * self.huber ** 2).mean())

    def align(self, reference, current):
        """Rotation of the current camera in the previous camera's frame."""
        R = np.eye(3)
        for geometry, I_ref, I_cur in zip(self.geometry, reference, current):
            _, J_u, J_v, *_ = geometry
            # inverse compositional: template and Jacobian fixed per level
            gx = (cv2.Sobel(I_cur, cv2.CV_32F, 1, 0, ksize=3) / 8.0)[1:-1, 1:-1].reshape(-1, 1)
            gy = (cv2.Sobel(I_cur, cv2.CV_32F, 0, 1, ksize=3) / 8.0)[1:-1, 1:-1].reshape(-1, 1)
            J = gx * J_u + gy * J_v
            template = I_cur[1:-1, 1:-1].reshape(-1)
            for _ in range(self.iters):
                warped, ok = self._warp(I_ref, geometry, R)
                if ok.sum() < 0.2 * len(template):
                    break
                e, J_ok = (warped - template)[ok], J[ok]
                a = np.abs(e)
                J_weighted = J_ok * np.where(a <= self.huber, 1.0, self.huber / np.maximum(a, 1e-12))[:, None]
                d = np.linalg.solve(J_weighted.T @ J_ok + 1e-6 * np.eye(3), J_weighted.T @ e)
                R = R @ cv2.Rodrigues(-d.reshape(3, 1))[0]
                if np.linalg.norm(d) < 1e-4:
                    break
        geometry, I_ref, I_cur = self.geometry[-1], reference[-1], current[-1]
        template = I_cur[1:-1, 1:-1].reshape(-1)
        identity_warp, identity_ok = self._warp(I_ref, geometry, np.eye(3))
        aligned_warp, aligned_ok = self._warp(I_ref, geometry, R)
        if self._cost((aligned_warp - template)[aligned_ok]) < self._cost((identity_warp - template)[identity_ok]):
            return R
        return np.eye(3)


class Tracker:
    """Registers every depth frame (obs A_i) to the map (dst B_i) and runs the frame loop."""

    def __init__(self, args, config, dataset_path, config_path, opt, pipe, num_frames):
        self.trainer_args = (args, config, dataset_path, config_path, opt, pipe, num_frames)   # the mapper's trainer
        self.args = args
        self.config = config
        self.dataset_path = dataset_path
        self.async_cfg = _async_cfg(config)
        self.max_fps = float(self.async_cfg.get("max_fps", 0) or 0)
        self.dataset = load_dataset(args, dataset_path, config)
        self.view = None                                   # live viewer
        if (config.get("Results", {}) or {}).get("viewer", {}).get("mode") in ("map", "render", "both"):
            from live_viewer import TrackerView
            pose = None
            if config["Results"]["viewer"]["mode"] in ("render", "both"):
                pose = mp.get_context("spawn").Array("d", 13, lock=False)   # the newest pose, for the mapper's render
                pose[0] = -1
            self.view = TrackerView(config, self.dataset, pose)
        total = min(num_frames or len(self.dataset.poses), len(self.dataset.poses))
        self.frames = list(range(getattr(args, "start_frame", 0) or 0, total, config["Training"]["frame_stride"]))
        if hasattr(self.dataset, "live_frames"):           # live camera: always the newest frame
            self.frames = self.dataset.live_frames(num_frames)
        self.cfg = SimpleNamespace(**_gicp_cfg(config))
        if int(self.cfg.window_points) > REMAP_MAX_POINTS:
            raise ValueError(f"Training.async.gicp.window_points {self.cfg.window_points}: at most {REMAP_MAX_POINTS}")
        self._setup_registration()
        self.pending = {}                                  # frame index -> prefetch future
        self.pool = ThreadPoolExecutor(int(self.cfg.prefetch_workers))   # cv2 decodes without the GIL
        self.sbi = _SBIRotation(self.dataset, self.cfg.sbi_levels, self.cfg.sbi_iters, self.cfg.sbi_blur)
        self.previous_pyramid = None
        self.target = None                                 # the published target rows (trackable_centroids)
        self.c2w = None
        self.c2w_prev = None
        self.overlap = 0.0                                 # covered fraction of the last frame
        # newest target from the mapper
        self.target_version = 0
        self.target_keyframe = -1
        self.mapper_stats = {}
        self.keyframe_order = []        # keyframes sent to the window optimisation
        self.keyframe_pose = {}         # their C2W in the working frame
        self.corrections = []           # corrections sent to the mapper
        self.relative_poses = []        # per frame: (anchor keyframe, relative pose)
        self.poses = []                 # per frame: (uid, R, T, R_gt, T_gt), as tracked
        self.frames_since_keyframe = 0
        self.num_keyframes = 0          # frames posted to the mapper
        self.num_tracking_keyframes = 0
        self.wait_s = 0.0
        self.track_s = 0.0
        self.t_start = 0.0

    def _setup_registration(self):
        """Set up fastgicp and the depth sampling grid."""
        pygicp = _import_pygicp()
        self.reg = pygicp.FastGICP()
        self.reg.set_num_threads(int(self.cfg.threads))
        self.reg.set_max_correspondence_distance(float(self.cfg.corr))
        self.reg.set_max_knn_distance(99999.0)
        self.reg.set_correspondence_randomness(int(self.cfg.knn))
        self.reg.set_regularization_method("none")
        self.reg.set_covariance_floor(float(self.cfg.cov_floor))
        dataset = self.dataset
        v, u = np.mgrid[0:int(dataset.height):int(self.cfg.stride), 0:int(dataset.width):int(self.cfg.stride)]
        self.grid = (v.reshape(-1), u.reshape(-1))
        self.ray_x = ((u.reshape(-1) - dataset.cx) / dataset.fx).astype(np.float64)
        self.ray_y = ((v.reshape(-1) - dataset.cy) / dataset.fy).astype(np.float64)

    # ---- frame data -----------------------------------------------------------------------------------------------
    def load(self, idx, upcoming=()):
        """Load a frame's subsampled depth and image pyramid."""
        for j in upcoming:
            if j not in self.pending:
                self.pending[j] = self.pool.submit(self._read, j)
        future = self.pending.pop(idx, None)
        return future.result() if future is not None else self._read(idx)

    def _read(self, idx):
        depth = cv2.imread(self.dataset.depth_paths[idx], cv2.IMREAD_ANYDEPTH)
        grey = cv2.imread(self.dataset.color_paths[idx], cv2.IMREAD_GRAYSCALE)
        return dict(z=depth[self.grid] / float(self.dataset.depth_scale), pyr=self.sbi.pyramid(grey))

    def ground_truth(self, idx):
        pose = self.dataset.poses[idx]
        pose = pose if isinstance(pose, torch.Tensor) else torch.from_numpy(np.asarray(pose))
        return pose[:3, :3].clone(), pose[:3, 3].clone()

    # ---- registration ---------------------------------------------------------------------------------------------
    def _grid_covs(self, z, ok):
        """scatter covariance B_i of each depth sample with its grid neighbours, approximately."""
        v, u = self.grid
        nv, nu = len(np.unique(v)), len(np.unique(u))
        zg = z.reshape(nv, nu)
        valid = (zg > 0) & (zg <= self.cfg.depth_trunc)
        P = np.stack([(self.ray_x * z).reshape(nv, nu), (self.ray_y * z).reshape(nv, nu), zg], -1)   # (nv, nu, 3)
        P_pad = np.pad(P, ((1, 1), (1, 1), (0, 0)))
        valid_pad = np.pad(valid, 1)
        z_pad = np.pad(zg, 1)
        weight_sum = np.zeros((nv, nu))
        point_sum = np.zeros((nv, nu, 3))
        outer_sum = np.zeros((nv, nu, 3, 3))
        for dv in (-1, 0, 1):
            for du in (-1, 0, 1):
                rows, cols = slice(1 + dv, 1 + dv + nv), slice(1 + du, 1 + du + nu)
                P_n = P_pad[rows, cols]
                same_surface = np.abs(z_pad[rows, cols] - zg) <= 0.03 * np.maximum(zg, 1e-3)
                w = (valid_pad[rows, cols] & valid & same_surface).astype(np.float64)
                weight_sum += w
                point_sum += w[..., None] * P_n
                outer_sum += w[..., None, None] * (P_n[..., :, None] * P_n[..., None, :])
        count = np.maximum(weight_sum, 1.0)
        mean = point_sum / count[..., None]
        C = outer_sum / count[..., None, None] - mean[..., :, None] * mean[..., None, :]
        C = C + float(self.cfg.cov_floor) ** 2 * np.eye(3)
        footprint = int(self.cfg.stride) * zg / self.dataset.fx                 # of a sample
        few = weight_sum < 3
        C[few] = (footprint[few] ** 2)[:, None, None] * np.eye(3)
        C = C.reshape(-1, 3, 3)[ok]
        out = np.zeros((C.shape[0], 4, 4))
        out[:, :3, :3] = C
        return out

    def set_target(self, rows):
        """rows: target rows (trackable_centroids)"""
        if rows.shape[0] == 0:
            raise RuntimeError("the tracking target is empty: the map has no trackable triangles")
        self.target = rows.numpy()
        self.reg.set_input_target(self.target[:, :3].astype(np.float64))
        self.reg.calculate_target_covariance()              # the raw k-NN scatter of the centroids

    def set_pose(self, R, T):
        W2C = np.eye(4)
        W2C[:3, :3] = R.double().numpy()
        W2C[:3, 3] = T.double().numpy()
        self.c2w = np.linalg.inv(W2C)

    def _align(self, points, init, covariances):
        self.reg.set_input_source(points)
        # all points pass the filter
        self.reg.set_source_filter(points.shape[0], reg_filter(points.shape[0]))
        self.reg.set_source_covariances(covariances)             # used as given (set_input_source clears them)
        return self.reg.align(init)

    def track(self, frame):
        """Register one frame. ret (is_keyframe, R, T)."""
        z = frame["z"]
        ok = (z > 0) & (z <= self.cfg.depth_trunc)
        self.registered = int(ok.sum()) >= MIN_SOURCE_POINTS
        if not self.registered:        # no depth in range: keep the pose
            _log("Frame not registered:", int(ok.sum()), "depth samples within depth_trunc; pose kept", tag="GICP")
            self.previous_pyramid = frame["pyr"]
            self.c2w_prev = self.c2w
            W2C = np.linalg.inv(self.c2w)
            return False, torch.from_numpy(W2C[:3, :3]).float(), torch.from_numpy(W2C[:3, 3]).float()
        points = np.stack([self.ray_x[ok] * z[ok], self.ray_y[ok] * z[ok], z[ok]], 1)
        covariances = self._grid_covs(z, ok)
        init = self.c2w
        if self.previous_pyramid is not None:
            rotation = np.eye(4)
            rotation[:3, :3] = self.sbi.align(self.previous_pyramid, frame["pyr"])
            init = self.c2w @ rotation
        self.previous_pyramid = frame["pyr"]
        self.c2w_prev = self.c2w
        self.c2w = self._align(points, init, covariances)
        W2C = np.linalg.inv(self.c2w)
        _, d2 = self.reg.get_source_correspondence()
        d2 = np.asarray(d2)
        self.overlap = float((d2 < self.cfg.overlap_th).mean()) if d2.size else 0.0
        is_keyframe = self.overlap < self.cfg.keyframe_th
        return is_keyframe, torch.from_numpy(W2C[:3, :3]).float(), torch.from_numpy(W2C[:3, 3]).float()

    def _warn_if_diverged(self, frame_idx):
        """log if lm step diverge"""
        step = np.linalg.inv(self.c2w_prev) @ self.c2w
        step_cm = 100 * float(np.linalg.norm(step[:3, 3]))
        step_deg = float(np.degrees(np.arccos(np.clip((np.trace(step[:3, :3]) - 1) / 2, -1, 1))))
        if self.overlap < 0.5 * self.cfg.keyframe_th or step_cm > DIVERGED_STEP_CM or step_deg > DIVERGED_STEP_DEG:
            _log("GICP alignment looks diverged: frame", frame_idx, "overlap", round(self.overlap, 3), "step",
                 round(step_cm, 1), "cm", round(step_deg, 1), "deg", tag="GICP")

    # ---- the run --------------------------------------------------------------------------------------------------
    def run(self):
        """main  frame loop ->final poses and outputs"""
        self._start_processes()
        self.t_start = time.perf_counter()
        # per frame: index, start, processing, tracking, waiting, keyframe code
        self.frame_log = []
        for view_iteration, frame_idx in enumerate(self.frames, start=1):
            t_frame = time.perf_counter()
            before = (self.track_s, self.wait_s, self.num_keyframes, self.num_tracking_keyframes)
            if view_iteration == 1:
                R, T = self._first_frame(frame_idx)
            else:
                R, T = self._track_frame(view_iteration, frame_idx)
            anchor = self.keyframe_order[-1]
            self.relative_poses.append((anchor, np.linalg.inv(self.keyframe_pose[anchor]) @ self.c2w))
            self.poses.append((int(frame_idx), R.clone(), T.clone()) + self.ground_truth(frame_idx))
            if self.view is not None:
                self.view.frame(frame_idx, self.c2w, R.numpy(), T.numpy(), self.num_keyframes, self.poses[-1][3:5])
            if slam_mp.frame_hook is not None:               # inside the frame's max_fps budget
                slam_mp.frame_hook(view_iteration, len(self.frames), self.num_keyframes, self.mapper_stats)
            # 0 none, 1 mapping keyframe, 2 tracking keyframe
            self.frame_log.append((int(frame_idx), t_frame - self.t_start, time.perf_counter() - t_frame,
                                   self.track_s - before[0], self.wait_s - before[1],
                                   int(self.num_keyframes > before[2]) + int(self.num_tracking_keyframes > before[3])))
            if self.max_fps > 0:
                while time.perf_counter() - t_frame < 1.0 / self.max_fps:
                    time.sleep(1e-4)
        self._finish()

    def _start_processes(self):
        """Spawn the mapper and the window optimisation."""
        ctx = mp.get_context("spawn")
        self.kf_q, self.snap_q, self.done_ev = ctx.Queue(), ctx.Queue(), ctx.Event()
        mapper_args = self.trainer_args + (self.kf_q, self.snap_q, self.done_ev,
                                           self.view.link() if self.view is not None else None)
        self.mapper_proc = ctx.Process(target=slam_mp.mapper_main, args=mapper_args, daemon=True)
        self.mapper_proc.start()
        self.window_proc = None
        if not self.cfg.window_opt:
            _wait_ready(self.snap_q, self.mapper_proc, "mapper")
            return
        self.opt_in, self.opt_out = ctx.Queue(), ctx.Queue()
        thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        saved = {key: os.environ.get(key) for key in thread_vars}
        for key in thread_vars:
            os.environ[key] = str(int(self.cfg.window_threads))   # inherited by the spawned window process only
        window_args = (self.config, self.dataset_path, self.opt_in, self.opt_out, getattr(self.args, "seed", None))
        self.window_proc = ctx.Process(target=slam_mp.window_main, args=window_args, daemon=True)
        self.window_proc.start()
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _wait_ready(self.snap_q, self.mapper_proc, "mapper")
        _wait_ready(self.opt_out, self.window_proc, "window optimisation")

    def _first_frame(self, frame_idx):
        """global frame is the first frame."""
        R, T = torch.eye(3), torch.zeros(3)
        self.kf_q.put(("kf", frame_idx, R, T, 1))
        self._wait_for_target(frame_idx)
        self.set_pose(R, T)
        self._post_to_window(frame_idx)
        self.num_keyframes += 1
        return R, T

    def _track_frame(self, view_iteration, frame_idx):
        self._poll_mapper()
        self._poll_window()
        frame = self.load(frame_idx, self.frames[view_iteration:view_iteration + int(self.cfg.prefetch_ahead)])
        t_track = time.perf_counter()
        is_keyframe, R, T = self.track(frame)
        self.track_s += time.perf_counter() - t_track
        self._warn_if_diverged(frame_idx)
        if is_keyframe and self.frames_since_keyframe + 1 < int(self.cfg.kf_min_gap):
            is_keyframe = False
        self.frames_since_keyframe = 0 if is_keyframe else self.frames_since_keyframe + 1
        self.num_tracking_keyframes += int(is_keyframe)
        mapping_every = int(self.cfg.mapping_kf_every)
        since = self.frames_since_keyframe
        is_mapping_keyframe = self.registered and mapping_every > 0 and since > 0 and since % mapping_every == 0
        if is_keyframe or is_mapping_keyframe:
            self.kf_q.put(("kf", frame_idx, R, T, view_iteration, "track" if is_keyframe else "map"))
            self.num_keyframes += 1
            if is_keyframe:
                self._wait_for_target(frame_idx)
            self._post_to_window(frame_idx)
        return R, T

    def _wait_for_target(self, frame_idx):
        """Wait for the target that includes the new keyframe."""
        t_wait = time.perf_counter()
        self._poll_mapper(block_until_kf=frame_idx)
        self.wait_s += time.perf_counter() - t_wait

    # ---- the mapper's targets -------------------------------------------------------------------------------------
    def _poll_mapper(self, block_until_kf=None, until_done=False):
        """take the mapper's messages."""
        while True:
            try:
                blocking = block_until_kf is not None or until_done
                msg = self.snap_q.get(timeout=0.5) if blocking else self.snap_q.get_nowait()
            except queue.Empty:
                if block_until_kf is None and not until_done:
                    return
                if not self.mapper_proc.is_alive():
                    raise RuntimeError("mapper process died")
                continue
            kind, version, keyframe_done, target, stats = msg
            if kind == "done":
                self.mapper_stats = stats
                return
            if version > self.target_version:
                self.target_version, self.target_keyframe, self.mapper_stats = version, keyframe_done, stats
                self._set_published_target(target)
            if block_until_kf is not None and self.target_keyframe >= block_until_kf:
                block_until_kf = None
                if not until_done:
                    self._poll_mapper()
                    return

    def _set_published_target(self, target):
        """Use a published target."""
        centroids = target["__centroids__"]
        applied = int(target.get("__corr__", 0) or 0)
        if applied < len(self.corrections):
            centroids = torch.from_numpy(_correct_target(centroids.numpy().copy(), self.corrections[applied:]))
        self.set_target(centroids)

    # ---- the window optimisation ----------------------------------------------------------------------------------
    def _post_to_window(self, uid):
        """Send a keyframe to the wopt."""
        c2w = self.c2w.copy()
        if self.window_proc is None:
            pass
        elif self.keyframe_order:
            previous = self.keyframe_order[-1]
            self.opt_in.put(("kf", int(uid), previous, np.linalg.inv(self.keyframe_pose[previous]) @ c2w))
        else:
            self.opt_in.put(("kf", int(uid), None, c2w))
        self.keyframe_order.append(int(uid))
        self.keyframe_pose[int(uid)] = c2w

    def _poll_window(self, block=False):
        """Apply optimised poses; with block, wait for the final ones."""
        if self.window_proc is None:
            return {} if block else None
        deadline = time.perf_counter() + FINAL_POSES_TIMEOUT_S
        while True:
            try:
                msg = self.opt_out.get(timeout=1.0) if block else self.opt_out.get_nowait()
            except queue.Empty:
                if not block:
                    return None
                if self.window_proc.is_alive() and time.perf_counter() < deadline:
                    continue
                _log("No final poses from the window optimisation: its process",
                     "did not answer" if self.window_proc.is_alive() else "has died",
                     "- the keyframe poses stay as they are", tag="Eval")
                self.opt_in.cancel_join_thread()     # nobody reads the queue any more
                return {}
            if msg[0] == "final":
                return msg[1]
            self._apply_window_poses(msg[1])

    def _apply_window_poses(self, optimised):
        """Apply corrections to the pose, the target and the mapper."""
        optimised = {uid: pose for uid, pose in optimised.items() if uid in self.keyframe_pose}
        if not optimised:
            return
        correction = {}
        for uid, pose in optimised.items():
            correction[uid] = pose @ np.linalg.inv(self.keyframe_pose[uid])
            self.keyframe_pose[uid] = pose
        newest = max(optimised, key=self.keyframe_order.index)
        newest_correction = correction[newest]
        for uid in self.keyframe_order[self.keyframe_order.index(newest) + 1:]:
            self.keyframe_pose[uid] = newest_correction @ self.keyframe_pose[uid]
            correction[uid] = newest_correction
        correction = {uid: D for uid, D in correction.items() if np.abs(D - np.eye(4)).max() > IDENTITY_TOLERANCE}
        if not correction:
            return
        self.corrections.append(correction)
        self.kf_q.put(("correct", len(self.corrections), correction))
        self._apply_correction(newest_correction, correction)

    def _apply_correction(self, newest_correction, correction):
        """Move the current pose and the target."""
        self.c2w = newest_correction @ self.c2w
        if self.c2w_prev is not None:
            self.c2w_prev = newest_correction @ self.c2w_prev
        if self.target is not None:
            self.set_target(torch.from_numpy(_correct_target(self.target.copy(), [correction])))

    # ---- the end of the run ---------------------------------------------------------------------------------------
    def _finish(self):
        """Final poses, ATE, then let the mapper save."""
        total_s = time.perf_counter() - self.t_start
        fps = len(self.frames) / total_s
        _log("Async (CPU tracker): frames", len(self.frames), "keyframes", self.num_keyframes, "FPS", round(fps, 2),
             "tracking s", round(self.track_s, 2), "waiting for the mapper s", round(self.wait_s, 2), tag="Eval")
        timing = dict(num_frames=len(self.frames), train_total_s=total_s, train_fps=fps, tracking_s=self.track_s,
                      wait_s=self.wait_s, keyframes=self.num_keyframes, tracking_keyframes=self.num_tracking_keyframes,
                      async_cfg=self.async_cfg, gicp_cfg=vars(self.cfg))
        if self.window_proc is not None:
            self.opt_in.put(("end",))
        self._apply_final_poses(self._poll_window(block=True))
        poses_online = self.poses
        poses = self._reanchored_poses()
        timing.update(ate_all_online_m=self._ate_m(poses_online), ate_all_final_m=self._ate_m(poses),
                      window_corrections=len(self.corrections))
        _log("Keyframe window optimisation: corrections", len(self.corrections), "ATE all frames online",
             round(100 * timing["ate_all_online_m"], 3), "cm, final", round(100 * timing["ate_all_final_m"], 3), "cm",
             tag="Eval")
        if self.window_proc is not None:
            self.window_proc.join(timeout=60)
        self.kf_q.put(("end", dict(poses=_pack_poses(poses), timing=timing, poses_online=_pack_poses(poses_online),
                                   frame_log=self.frame_log)))
        self._poll_mapper(until_done=True)
        self.done_ev.set()
        self.mapper_proc.join(timeout=MAPPER_EXIT_TIMEOUT_S)
        if self.view is not None:
            self.view.close()

    def _apply_final_poses(self, final):
        """the saved map and keyframe cameras follow the final poses"""
        known = {uid: pose for uid, pose in final.items() if uid in self.keyframe_pose}
        correction = {uid: pose @ np.linalg.inv(self.keyframe_pose[uid]) for uid, pose in known.items()}
        correction = {uid: D for uid, D in correction.items() if np.abs(D - np.eye(4)).max() > IDENTITY_TOLERANCE}
        self.keyframe_pose.update(known)
        if correction:
            self.corrections.append(correction)
            self.kf_q.put(("correct", len(self.corrections), correction))

    def _reanchored_poses(self):
        """Every frame placed relative to its keyframe's final pose."""
        poses = []
        for (uid, _, _, R_gt, T_gt), (anchor, relative) in zip(self.poses, self.relative_poses):
            W2C = np.linalg.inv(self.keyframe_pose[anchor] @ relative)
            poses.append((uid, torch.from_numpy(W2C[:3, :3]).float(), torch.from_numpy(W2C[:3, 3]).float(), R_gt, T_gt))
        return poses

    @staticmethod
    def _ate_m(rows):
        """ATE RMSE in meter after standard SE(3) alignment."""
        centres = np.array([-(row[1].double().numpy().T @ row[2].double().numpy()) for row in rows])
        centres_gt = np.array([-(row[3].double().numpy().T @ row[4].double().numpy()) for row in rows])
        finite = np.isfinite(centres).all(1) & np.isfinite(centres_gt).all(1)     # ScanNet marks lost poses with inf
        if finite.sum() < 3:
            return float("nan")
        centres, centres_gt = centres[finite], centres_gt[finite]
        U, _, Vt = np.linalg.svd((centres - centres.mean(0)).T @ (centres_gt - centres_gt.mean(0)))
        reflection = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        R = (U @ reflection @ Vt).T
        t = centres_gt.mean(0) - R @ centres.mean(0)
        return float(np.sqrt((((centres @ R.T + t) - centres_gt) ** 2).sum(1).mean()))
