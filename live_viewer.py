"""rerun live viewer, enabled by main.py --viewer. PLEASE LEAVE THIS OFF FOR THE FULL FPS (ESP. JETSON)."""
import copy
import functools
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np

APP = "FacetSLAM"
SH_C0 = 0.28209479177387814
DEFAULTS = dict(mode=None, sink="web", port=9876, web_port=9090, url=None, hz=2.0, render_hz=15.0, min_opacity=0.1,
                image_scale=None, depth_range=[0.25, 8.0], map_refresh_s=10.0, layout="grid")
STRIP = 100            # points per trajectory piece
MODES = ("map", "render", "both")


def log_depth_colours(depth, near, far, bar_cache={}):
    """Depth image as turbo colours on a log scale."""
    import cv2
    span = np.log(far / near)
    t = np.log(np.clip(depth, near, far) / near) / span
    rgb = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
    rgb[depth <= 0] = 0
    H = rgb.shape[0]
    key = (H, near, far)
    if key not in bar_cache:
        ramp = np.linspace(1, 0, H)[:, None].repeat(10, 1)              # far at the top
        bar = np.zeros((H, 44, 3), np.uint8)
        bar[:, 2:12] = cv2.applyColorMap((ramp * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
        tick = near
        while tick <= far * 1.0001:
            y = int(round((1 - np.log(tick / near) / span) * (H - 1)))
            cv2.line(bar, (12, y), (15, y), (255, 255, 255), 1)
            label = f"{tick:g}m"
            cv2.putText(bar, label, (16, min(max(y + 4, 9), H - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1,
                        cv2.LINE_AA)
            tick *= 2
        bar_cache[key] = bar
    return np.concatenate([rgb, bar_cache[key]], 1)


def guarded(method):
    """A viewer error never stops SLAM: report once, then switch the viewer off."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "failed", False):
            return None
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:                     # noqa: BLE001 - deliberately broad
            self.failed = True
            print(f"[viewer] {type(self).__name__}.{method.__name__} failed, viewer off for the rest of the run: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            return None
    return wrapper


class Rate:
    """due() is True at most `hz` times a second."""

    def __init__(self, hz):
        self.period = 1.0 / max(float(hz), 1e-3)
        self.next = None

    def due(self, now=None):
        now = time.perf_counter() if now is None else now
        if self.next is None:
            self.next = now + self.period
            return True
        if now < self.next:
            return False
        self.next = max(self.next + self.period, now - self.period)
        return True


def settings(config):
    s = dict(DEFAULTS)
    s.update((config.get("Results", {}) or {}).get("viewer", {}) or {})
    if s["mode"] not in MODES:            # YAML reads `off` as False
        s["mode"] = None
    if s["image_scale"] is None:          # full size locally, half over the network
        s["image_scale"] = 1.0 if s["sink"] == "spawn" else 0.5
    return s


def _blueprint(mode, layout="grid"):
    import rerun.blueprint as rrb
    world = rrb.Spatial3DView(origin="world", name="map")
    images = [rrb.Spatial2DView(origin="world/camera/image", name="input")]
    if mode in ("render", "both"):
        images += [rrb.Spatial2DView(origin="render/rgb", name="render"),
                   rrb.Spatial2DView(origin="render/depth", name="render depth")]
    if mode == "both" and layout == "grid":
        # four equal panels
        return rrb.Blueprint(rrb.Grid(images[0], images[1], world, images[2], grid_columns=2),
                             rrb.TopPanel(state="hidden"), rrb.TimePanel(state="hidden"),
                             rrb.BlueprintPanel(state="hidden"), rrb.SelectionPanel(state="hidden"))
    if mode == "render":
        return rrb.Blueprint(rrb.Horizontal(rrb.Vertical(*images[1:]), rrb.Vertical(images[0], world)),
                             collapse_panels=True)
    return rrb.Blueprint(rrb.Horizontal(world, rrb.Vertical(*images), column_shares=[3, 1]), collapse_panels=True)


class TrackerView:
    """Tracker side: camera pose and trajectory."""

    def __init__(self, config, dataset, pose_array=None):
        import rerun as rr
        self.rr = rr
        self.s = s = settings(config)
        self.dataset = dataset
        self.rid = str(uuid.uuid4())
        rr.init(APP, recording_id=self.rid, spawn=False)
        blueprint = _blueprint(s["mode"], s["layout"])
        if s["sink"] == "spawn":
            rr.spawn(port=int(s["port"]), memory_limit="25%", default_blueprint=blueprint)
            self.url = f"rerun+http://127.0.0.1:{int(s['port'])}/proxy"
        elif s["sink"] == "connect":
            self.url = s["url"] or f"rerun+http://127.0.0.1:{int(s['port'])}/proxy"
            rr.connect_grpc(self.url, default_blueprint=blueprint)
        else:
            self.url = rr.serve_grpc(grpc_port=int(s["port"]), default_blueprint=blueprint, server_memory_limit="25%")
            rr.serve_web_viewer(web_port=int(s["web_port"]), open_browser=False, connect_to=self.url)
            print(f"[viewer] open http://localhost:{int(s['web_port'])}/?url={self.url.replace('+', '%2B')}  "
                  f"(forward ports {int(s['web_port'])} and {int(s['port'])} from a remote machine)", flush=True)
        # also the active one, or a stored layout would win
        rr.send_blueprint(blueprint, make_active=True, make_default=True)
        self.pose = pose_array                       # newest pose, shared with the mapper
        self.scale = float(s["image_scale"])
        fx, fy, cx, cy = (float(getattr(dataset, k)) for k in ("fx", "fy", "cx", "cy"))
        W, H = int(dataset.width), int(dataset.height)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)      # x right, y down, z ahead
        rr.log("world/camera/image", rr.Pinhole(focal_length=[fx * self.scale, fy * self.scale],
                                                principal_point=[cx * self.scale, cy * self.scale],
                                                width=round(W * self.scale), height=round(H * self.scale),
                                                image_plane_distance=0.15), static=True)
        self.centres, self.kf_centres = [], []
        self.gt_centres, self.gt_align = [], None     # ground truth in the map frame
        self.gt_moved = False
        self.last_kf = 0
        self.image_rate = Rate(s["render_hz"])
        self.reader = ThreadPoolExecutor(1)          # decode and log off the tracker's thread
        self.pending = None

    def link(self):
        """what the mapper needs to join: (recording id, url, shared pose)"""
        return self.rid, self.url, self.pose

    @guarded
    def frame(self, frame_idx, c2w, R, T, num_keyframes, gt=None):
        """gt: (R_gt, T_gt) world-to-camera of the dataset, or None"""
        rr = self.rr
        if self.pose is not None:
            self.pose[1:10] = R.reshape(-1).tolist()
            self.pose[10:13] = T.reshape(-1).tolist()
            self.pose[0] = frame_idx
        rr.set_time("frame", sequence=int(frame_idx))
        rr.log("world/camera", rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]))
        self.centres.append(c2w[:3, 3].copy())
        if num_keyframes != self.last_kf:
            self.last_kf = num_keyframes
            self.kf_centres.append(c2w[:3, 3].copy())
            rr.log("world/keyframes", rr.Points3D(np.array(self.kf_centres), radii=0.01, colors=[255, 140, 0]),
                   static=True)
        self._log_strip("world/trajectory", self.centres, [30, 120, 255])
        if gt is not None:
            self._log_ground_truth(c2w, gt)
        if (self.pending is None or self.pending.done()) and self.image_rate.due():
            self.pending = self.reader.submit(self._log_image, int(frame_idx))

    def _log_ground_truth(self, c2w, gt):
        """Log the ground-truth trajectory in the map frame."""
        rr = self.rr
        W = np.eye(4)
        W[:3, :3] = np.asarray(gt[0], dtype=np.float64)
        W[:3, 3] = np.asarray(gt[1], dtype=np.float64).reshape(-1)
        if not np.isfinite(W).all():
            return
        G = np.linalg.inv(W)
        if self.gt_align is None:
            self.gt_align = c2w @ np.linalg.inv(G)
        self.gt_centres.append((self.gt_align @ G)[:3, 3])
        if len(self.gt_centres) < 2 or (self.gt_moved is False and
                                        np.abs(self.gt_centres[-1] - self.gt_centres[0]).max() < 1e-6):
            return                                   # no ground truth
        self.gt_moved = True
        self._log_strip("world/ground_truth", self.gt_centres, [40, 200, 70])
        rr.log("world/ground_truth_now", rr.Points3D(self.gt_centres[-1][None], radii=0.012, colors=[40, 200, 70]),
               static=True)

    def _log_strip(self, path, points, colour):
        """Log the newest piece of a polyline."""
        n = len(points)
        k = (n - 1) // STRIP
        self.rr.log(f"{path}/{k:04d}", self.rr.LineStrips3D([np.array(points[max(k * STRIP - 1, 0):n])], radii=0.003,
                                                           colors=[colour]), static=True)

    def _log_image(self, frame_idx):
        import cv2
        rr = self.rr
        bgr = cv2.imread(self.dataset.color_paths[frame_idx], cv2.IMREAD_COLOR)
        if bgr is None:
            return
        if self.scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        rr.set_time("frame", sequence=frame_idx)     # per thread
        rr.log("world/camera/image", rr.Image(bgr[..., ::-1]).compress(jpeg_quality=80))

    def close(self):
        self.reader.shutdown(wait=True)


class MapperView:
    """Mapper side: the map and the render."""

    def __init__(self, config, link, mapper):
        import rerun as rr
        self.rr = rr
        self.s = settings(config)
        rid, url, self.pose = link
        rr.init(APP, recording_id=rid, spawn=False)
        rr.connect_grpc(url)
        self.mapper = mapper
        self.map_rate = Rate(self.s["hz"])
        self.refresh_rate = Rate(1.0 / max(float(self.s["map_refresh_s"]), 1e-3))
        self.sent = set()                            # keyframe uids whose piece has been sent
        self.render_rate = Rate(self.s["render_hz"])
        self.sender = ThreadPoolExecutor(1)          # compress and log off the mapper's thread
        self.mesh_sender = ThreadPoolExecutor(1)     # the mesh has its own thread
        self.pending = None
        self.mesh_pending = None

    @guarded
    def publish(self, force=False):
        """Refresh the map mesh."""
        if self.s["mode"] not in ("map", "both"):
            return
        if self.mesh_pending is not None and not self.mesh_pending.done():
            if not force:
                return
            self.mesh_pending.result()
        if force or self.map_rate.due():
            self._log_map(full=force or self.refresh_rate.due())

    @guarded
    def tick(self, force=False):
        """Refresh the render."""
        if self.s["mode"] not in ("render", "both"):
            return
        if self.pending is not None and not self.pending.done():
            if not force:
                return                               # previous render still in flight: skip
            self.pending.result()
        if force or self.render_rate.due():
            self._log_render()

    @guarded
    def final(self):
        self.publish(force=True)
        self.tick(force=True)
        for pending in (self.pending, self.mesh_pending):
            if pending is not None:
                pending.result()
        self.sender.shutdown(wait=True)
        self.mesh_sender.shutdown(wait=True)

    def _log_map(self, full=False):
        """Send the map, one piece per keyframe."""
        import torch
        rr = self.rr
        triangles = self.mapper.trainer.triangles
        with torch.no_grad():
            faces = triangles._triangle_indices.long()
            if faces.numel() == 0:
                return
            ids = triangles.unique_kfIDs.to(faces.device).long().reshape(-1)
            opacity = torch.sigmoid(triangles.vertex_weight.float()).reshape(-1)
            keep = opacity[faces].mean(1) >= float(self.s["min_opacity"])
            faces, ids = faces[keep], ids[keep]
            uids, counts = torch.unique(ids, return_counts=True)
            counts = dict(zip(uids.tolist(), counts.tolist()))
            send = [u for u in counts if full or u not in self.sent]      # new keyframes; everything on a refresh
            gone = [u for u in self.sent if u not in counts]
            self.sent = set(counts)
            if not send:
                if gone:
                    self.mesh_pending = self.mesh_sender.submit(
                        lambda: [rr.log(f"world/map/kf{int(u):05d}", rr.Clear(recursive=False), static=True)
                                 for u in gone])
                return
            pick = torch.isin(ids, torch.tensor(send, device=ids.device, dtype=ids.dtype))
            faces, ids = faces[pick], ids[pick]
            used, inverse = torch.unique(faces, return_inverse=True)
            vertices = triangles.vertices[used].float().cpu().numpy()
            dc = triangles._features_dc[used].float().reshape(len(used), -1)[:, :3]
            colors = (torch.clamp(SH_C0 * dc + 0.5, 0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            faces = inverse.reshape(-1, 3).cpu().numpy()
            ids = ids.cpu().numpy()

        def send_pieces():                           # split per keyframe on this thread
            order = np.argsort(ids, kind="stable")
            f_sorted, i_sorted = faces[order], ids[order]
            bounds = np.flatnonzero(np.diff(i_sorted)) + 1
            for start, stop in zip(np.r_[0, bounds], np.r_[bounds, len(f_sorted)]):
                if stop <= start:
                    continue
                v_used, f_local = np.unique(f_sorted[start:stop], return_inverse=True)
                rr.log(f"world/map/kf{int(i_sorted[start]):05d}",
                       rr.Mesh3D(vertex_positions=vertices[v_used],
                                 triangle_indices=f_local.reshape(-1, 3).astype(np.uint32),
                                 vertex_colors=colors[v_used]), static=True)
            for u in gone:
                rr.log(f"world/map/kf{int(u):05d}", rr.Clear(recursive=False), static=True)
        self.mesh_pending = self.mesh_sender.submit(send_pieces)

    def _log_render(self):
        import torch
        rr = self.rr
        if self.pose is None or self.mapper.newest is None or self.pose[0] < 0:
            return
        values = list(self.pose[:13])
        frame_idx = int(values[0])
        cam = copy.copy(self.mapper.newest)          # newest keyframe's intrinsics, newest tracked pose
        device = self.mapper.trainer.triangles.vertices.device
        cam.update_RT(torch.tensor(values[1:10], dtype=torch.float32, device=device).reshape(3, 3),
                      torch.tensor(values[10:13], dtype=torch.float32, device=device))
        trainer = self.mapper.trainer
        with torch.no_grad():
            pkg = self.mapper.render(cam, trainer.triangles, trainer.pipe, trainer.background)
            rgb = (pkg["render"].clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            alpha = pkg["rend_alpha"][0]
            depth = (pkg["expected_depth"][0] / alpha.clamp_min(1e-3)) * (alpha > 0.5)
        scale = float(self.s["image_scale"])
        step = 1 if scale >= 1.0 else 2                                    # depth at the colour image's size
        depth = depth[::step, ::step].float().cpu().numpy()                # m
        near, far = (float(x) for x in self.s["depth_range"])

        def send():
            import cv2
            small = rgb if scale == 1.0 else cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            rr.set_time("frame", sequence=frame_idx)     # per thread
            rr.log("render/rgb", rr.Image(small).compress(jpeg_quality=85))
            rr.log("render/depth", rr.Image(log_depth_colours(depth, near, far)).compress(jpeg_quality=90))
        self.pending = self.sender.submit(send)
