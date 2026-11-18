"""Pose-only window optimisation of the keyframe poses, solved by Ceres (native/ceres_window_opt).

Optional pose window opt. using a past buffer of frames, marginally improves robustness in some cases.
"""
import os
import queue
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch.multiprocessing as mp

from slam_mp import PARENT_CHECK_S, _gicp_cfg, _seed
from utils.dataset import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "native", "ceres_window_opt"))
import ceres_window_opt  # noqa: E402  (native/ceres_window_opt/build.sh)


def _quat_xyzw(R):
    """Rotation matrix -> quaternion (x, y, z, w)."""
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R)))
    j = (i + 1) % 3
    k = (i + 2) % 3
    s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


def _rotation(q):
    """unit quaternion (x, y, z, w) -> rotation matrix"""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _concat(arrays, empty_shape):
    return np.concatenate(arrays) if arrays else np.zeros(empty_shape)


class WindowOpt:
    HUBER = 1.345             # Huber scale of the whitened residuals
    SIGMA_GREY = 0.05         # photometric noise (grey levels)
    PRIOR_SIGMA = 1.0         # the exposure prior
    MIN_OVERLAP = 0.3         # min fraction of samples for a keyframe to observe another
    MIN_PAIR_SAMPLES = 30     # min samples for a pair to count
    MAX_PAIR_DISTANCE = 2.0   # m between the two camera centres
    MIN_PAIR_COS = 0.3        # cosine between the two optical axes
    PHOTO_DEPTH_GATE = 0.03   # depth agreement for a photometric sample (m)
    DEPTH_JUMP = 0.03         # relative depth jump that marks a discontinuity
    NORMAL_MIN_VALID = 20     # valid pixels needed in a 5 x 5 patch
    BORDER = 3                # border without samples (px)

    def __init__(self, fx, fy, cx, cy, H, W, sigma, corr, points=1000, color_points=400, window=16, nbrs=8, iters=3,
                 threads=4, seed=0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.H = int(H)
        self.W = int(W)
        self.sigma = sigma                # point-to-plane noise (m)
        self.corr = corr                  # projective association gate (m)
        self.points = points
        self.color_points = color_points
        self.window = window
        self.nbrs = nbrs
        self.iters = iters
        self.threads = threads
        self.rng = np.random.default_rng(seed)
        vv, uu = np.mgrid[0:self.H, 0:self.W]
        self.ray_x = ((uu - cx) / fx).astype(np.float32)
        self.ray_y = ((vv - cy) / fy).astype(np.float32)
        self.uids = []         # keyframes in order
        self.P = {}            # uid -> camera-to-world pose
        self.X = {}            # uid -> sample points
        self.N = {}            # uid -> sample normals
        self.c = {}            # uid -> sample grey levels (nan: none)
        self.obs = {}          # uid -> (depth, grey, gradients) of the window keyframes

    # ---- keyframe intake: sample selection -------------------------------------------------------------------------
    def add(self, uid, C2W, depth, grey):
        """Add a keyframe."""
        z = depth.astype(np.float32)
        normals, normal_norm, valid_count = self._normals(z)
        v, u = self._sample_pixels(z, valid_count, normal_norm)
        X = np.stack([self.ray_x[v, u] * z[v, u], self.ray_y[v, u] * z[v, u], z[v, u]], 1).astype(np.float64)
        G = grey.astype(np.float32)
        self.uids.append(uid)
        self.P[uid] = np.asarray(C2W, np.float64).copy()
        self.X[uid] = X
        self.N[uid] = normals[v, u].astype(np.float64)
        self.c[uid] = self._sample_grey(X, G)
        self.obs[uid] = (z, G, self._gradients(G))
        for old in self.uids[:-self.window]:   # only window keyframes observe
            self.obs.pop(old, None)

    def _normals(self, z):
        """Normals from 5 x 5 patches."""
        points = np.stack([self.ray_x * z, self.ray_y * z, z], -1)
        valid = (z > 0).astype(np.float32)
        valid_count = cv2.boxFilter(valid, -1, (5, 5), normalize=False)
        point_sum = cv2.boxFilter(points * valid[..., None], -1, (5, 5), normalize=False)
        mean = point_sum / np.maximum(valid_count, 1e-6)[..., None]
        du = np.zeros_like(mean)
        dv = np.zeros_like(mean)
        du[:, 2:-2] = mean[:, 4:] - mean[:, :-4]
        dv[2:-2] = mean[4:] - mean[:-4]
        normals = np.cross(du, dv)
        norm = np.linalg.norm(normals, axis=2)
        normals = normals / np.maximum(norm, 1e-12)[..., None]
        normals *= np.where((normals * mean).sum(2, keepdims=True) > 0, -1.0, 1.0)
        return normals, norm, valid_count

    def _sample_pixels(self, z, valid_count, normal_norm):
        """Random pixels with valid depth and normal."""
        z_nan = np.where(z > 0, z, np.nan)
        discontinuity = np.zeros(z.shape, bool)
        with np.errstate(invalid="ignore"):
            for shift in ((0, 2), (0, -2), (2, 0), (-2, 0)):
                jump = np.abs(np.roll(z_nan, shift, (0, 1)) - z_nan)
                discontinuity |= ~(jump < self.DEPTH_JUMP * np.nan_to_num(z_nan, nan=1.0))
        ok = (z > 0) & (valid_count >= self.NORMAL_MIN_VALID) & (normal_norm > 1e-9) & ~discontinuity
        ok[:self.BORDER] = False
        ok[-self.BORDER:] = False
        ok[:, :self.BORDER] = False
        ok[:, -self.BORDER:] = False
        v, u = np.nonzero(ok)
        if len(v):
            chosen = self.rng.choice(len(v), size=min(len(v), self.points), replace=False)
        else:
            chosen = np.zeros(0, int)
        return v[chosen], u[chosen]

    @staticmethod
    def _gradients(G):
        return (cv2.Sobel(G, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(G, cv2.CV_32F, 0, 1, ksize=3) / 8.0)

    def _sample_grey(self, X, G):
        """Grey level at each sample's projection."""
        c = np.full(len(X), np.nan)
        u, v = self._project(X)
        ok = (X[:, 2] > 0.1) & (u > 1) & (u < self.W - 2) & (v > 1) & (v < self.H - 2)
        if ok.any():   # no valid depth: no samples
            c[ok] = self._remap(G, u[ok], v[ok])
        return c

    def _project(self, x):
        """pixel coordinates of camera-frame points (depth clamped away from zero)"""
        z = np.maximum(x[:, 2], 1e-3)
        return self.fx * x[:, 0] / z + self.cx, self.fy * x[:, 1] / z + self.cy

    @staticmethod
    def _remap(image, u, v):
        """bilinear samples of image at (u, v)"""
        return cv2.remap(image, u.astype(np.float32).reshape(-1, 1), v.astype(np.float32).reshape(-1, 1),
                         cv2.INTER_LINEAR).ravel()

    # ---- association ----------------------------------------------------------------------------------------------
    def _associate(self, T, index, window):
        """Geometric and photometric observations of each window keyframe."""
        offset = np.cumsum([0] + [len(self.X[u]) for u in self.uids])
        geometric = []       # per pair: sample ids, observer, points, normals
        photometric = []     # per pair: sample ids, observer, grey levels, exposure
        for u_k in window:
            k = index[u_k]
            depth_k, grey_k, gradients_k = self.obs[u_k]
            for _, h, associated, x, u, v, d in self._candidates(T, k, depth_k):
                u_h = self.uids[h]
                du, dv, dd = u[associated], v[associated], d[associated]
                y = np.stack([self.ray_x[dv, du] * dd, self.ray_y[dv, du] * dd, dd], 1)
                geometric.append((offset[h] + associated, np.full(len(associated), k), y, self.N[u_h][associated]))
                photo = self._photometric_pair(x[associated], dd, self.c[u_h][associated], grey_k, gradients_k)
                if photo is not None:
                    chosen, c, exposure = photo
                    photometric.append((offset[h] + associated[chosen], np.full(len(chosen), k), c, exposure))
        pair_ids = [np.full(len(rows[0]), pair) for pair, rows in enumerate(photometric)]
        return dict(geo_sample=_concat([rows[0] for rows in geometric], 0).astype(int),
                    geo_kf=_concat([rows[1] for rows in geometric], 0).astype(int),
                    geo_y=_concat([rows[2] for rows in geometric], (0, 3)),
                    geo_n=_concat([rows[3] for rows in geometric], (0, 3)),
                    photo_sample=_concat([rows[0] for rows in photometric], 0).astype(int),
                    photo_kf=_concat([rows[1] for rows in photometric], 0).astype(int),
                    photo_pair=_concat(pair_ids, 0).astype(int),
                    photo_c=_concat([rows[2] for rows in photometric], 0),
                    exposure=np.array([rows[3] for rows in photometric], np.float64).reshape(-1, 2))

    def _candidates(self, T, k, depth_k):
        """Keyframes that see most of k's samples."""
        T_k_inv = np.linalg.inv(T[k])
        candidates = []
        for h, u_h in enumerate(self.uids):
            far = np.linalg.norm(T[h][:3, 3] - T[k][:3, 3]) > self.MAX_PAIR_DISTANCE
            if h == k or far or T[h][:3, 2] @ T[k][:3, 2] < self.MIN_PAIR_COS:
                continue
            x = (self.X[u_h] @ T[h][:3, :3].T + T[h][:3, 3]) @ T_k_inv[:3, :3].T + T_k_inv[:3, 3]
            u_f, v_f = self._project(x)
            u = np.round(u_f).astype(int)
            v = np.round(v_f).astype(int)
            inside = (x[:, 2] > 0.1) & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
            d = np.zeros(len(x), np.float32)
            d[inside] = depth_k[v[inside], u[inside]]
            associated = (d > 0) & (np.abs(d - x[:, 2]) < self.corr)
            fraction = float(associated[::4].mean()) if len(associated) else 0.0   # of every 4th sample
            if fraction >= self.MIN_OVERLAP and associated.sum() >= self.MIN_PAIR_SAMPLES:
                candidates.append((fraction, h, np.flatnonzero(associated), x, u, v, d))
        return sorted(candidates, key=lambda candidate: -candidate[0])[:self.nbrs]

    def _photometric_pair(self, x, d, c, grey_k, gradients_k):
        """Photometric samples of a pair, with exposure."""
        u, v = self._project(x)
        usable = ((np.abs(x[:, 2] - d) < self.PHOTO_DEPTH_GATE) & (x[:, 2] > 0.1) & (u > 2) & (u < self.W - 3)
                  & (v > 2) & (v < self.H - 3) & np.isfinite(c))
        if usable.sum() < self.MIN_PAIR_SAMPLES:
            return None
        u, v = u[usable], v[usable]
        gradient = np.hypot(self._remap(gradients_k[0], u, v), self._remap(gradients_k[1], u, v))
        strongest = np.argsort(-gradient)[:self.color_points]
        chosen = np.flatnonzero(usable)[strongest]
        c = c[usable][strongest]
        intensity = self._remap(grey_k, u[strongest], v[strongest]).astype(np.float64)
        exposure = np.linalg.lstsq(np.stack([c, np.ones_like(c)], 1), intensity, rcond=None)[0]
        return chosen, c, exposure

    # ---- solve ----------------------------------------------------------------------------------------------------
    def optimise(self):
        """Solve the window. Returns {uid: C2W}."""
        window = [u for u in self.uids[-self.window:] if u != self.uids[0]]
        if not window:
            return {}
        index = {u: i for i, u in enumerate(self.uids)}
        T = np.stack([self.P[u] for u in self.uids])
        observations = self._associate(T, index, window)
        if len(observations["geo_sample"]) or len(observations["photo_sample"]):
            self._solve(T, index, window, observations)
        out = {u: T[index[u]].copy() for u in window}
        self.P.update(out)
        return out

    def _solve(self, T, index, window, observations):
        """Run the Ceres solver and write the poses back."""
        samples = np.concatenate([self.X[u] for u in self.uids])
        host = np.concatenate([np.full(len(self.X[u]), i) for i, u in enumerate(self.uids)])
        sample_ids = np.concatenate([observations["geo_sample"], observations["photo_sample"]])
        used = np.unique(sample_ids)
        position = np.searchsorted(used, sample_ids)
        free = np.zeros(len(self.uids), int)
        free[[index[u] for u in window]] = 1
        n_geo = len(observations["geo_sample"])
        photo = None
        if len(observations["photo_sample"]):
            observers = sorted(set(observations["photo_kf"].tolist()))
            kf_image = np.full(len(self.uids), -1)
            kf_image[observers] = np.arange(len(observers))
            photo = dict(lm=position[n_geo:], kf=observations["photo_kf"], pair=observations["photo_pair"],
                         c=observations["photo_c"], images=np.stack([self.obs[self.uids[i]][1] for i in observers]),
                         kf_image=kf_image, ab=observations["exposure"],
                         K=np.array([self.fx, self.fy, self.cx, self.cy]), sigma=self.SIGMA_GREY,
                         prior_sigma=self.PRIOR_SIGMA)
        q = np.array([_quat_xyzw(pose[:3, :3]) for pose in T])
        t = T[:, :3, 3].copy()
        ceres_window_opt.solve(q, t, free, samples[used], host[used], position[:n_geo], observations["geo_kf"],
                               observations["geo_y"], observations["geo_n"], self.sigma, self.iters, self.threads,
                               self.HUBER, photo)
        for u in window:
            T[index[u]][:3, :3] = _rotation(q[index[u]])
            T[index[u]][:3, 3] = t[index[u]]


class WindowOptProcess:
    """Process that refines the newest keyframe poses."""

    def __init__(self, config, dataset_path, in_q, out_q, seed=None):
        _seed(seed)
        cfg = SimpleNamespace(**_gicp_cfg(config))
        tum = config["Dataset"]["type"] == "tum"
        self.in_q = in_q
        self.out_q = out_q
        self.dataset = load_dataset(SimpleNamespace(distributed=False), dataset_path, config)
        self.depth_trunc = float(cfg.depth_trunc)
        dataset = self.dataset
        self.opt = WindowOpt(dataset.fx, dataset.fy, dataset.cx, dataset.cy, dataset.height, dataset.width,
                             sigma=0.01 if tum else 0.002, corr=0.05 if tum else 0.02, points=int(cfg.window_points),
                             color_points=int(cfg.window_color_points), window=int(cfg.window_size),
                             nbrs=int(cfg.window_nbrs), iters=int(cfg.window_iters), threads=int(cfg.window_threads),
                             seed=0 if seed is None else seed)

    def run(self):
        self.out_q.put(("ready",))
        while True:
            added = False
            for msg in self._drain():
                if msg[0] == "end":
                    if added:
                        self.opt.optimise()     # keyframes drained together with "end"
                    self.out_q.put(("final", dict(self.opt.P)))
                    return
                self._add_keyframe(*msg[1:])
                added = True
            if added:
                self.out_q.put(("corr", self.opt.optimise()))

    def _drain(self):
        """Take all waiting messages."""
        parent = mp.parent_process()
        while True:
            try:
                msgs = [self.in_q.get(timeout=PARENT_CHECK_S)]
                break
            except queue.Empty:
                if parent is not None and not parent.is_alive():
                    return [("end",)]
        while True:
            try:
                msgs.append(self.in_q.get_nowait())
            except queue.Empty:
                return msgs

    def _add_keyframe(self, uid, anchor, relative):
        """Load a keyframe's depth and grey image."""
        dataset = self.dataset
        depth = cv2.imread(dataset.depth_paths[uid], cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth /= float(dataset.depth_scale)
        depth[(depth <= 0) | (depth > self.depth_trunc)] = 0
        grey = cv2.cvtColor(cv2.imread(dataset.color_paths[uid]), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        self.opt.add(uid, relative if anchor is None else self.opt.P[anchor] @ relative, depth, grey)
