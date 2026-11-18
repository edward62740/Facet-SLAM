"""Triangle mapper process."""

import json
import os
import queue
import random
import time

import torch
import torch.multiprocessing as mp

from slam_mp import (
    HIDDEN_VERTEX_WEIGHT, MAPPER_EXIT_TIMEOUT_S, PARENT_CHECK_S, _async_cfg, _gicp_cfg, _seed, _unpack_poses,
)


def trackable_centroids(triangles, opacity_th, track_uids):
    """Means of the triangles the tracker registers against, with their keyframe uid."""
    with torch.no_grad():
        faces = triangles._triangle_indices.long()
        keep = torch.sigmoid(triangles.vertex_weight.float()).reshape(-1)[faces].mean(1) >= opacity_th
        ids = triangles.unique_kfIDs
        tracked = torch.tensor(sorted(track_uids), device=keep.device, dtype=ids.dtype)
        keep &= torch.isin(ids.to(keep.device), tracked)
        centroids = triangles.vertices.float()[faces[keep]].mean(1)
        return torch.cat([centroids, ids.to(centroids.device)[keep].float()[:, None]], 1).cpu()


class Mapper:
    """Owns the map: adds triangles at keyframes and trains in between."""

    def __init__(self, args, config, dataset_path, config_path, opt, pipe, num_frames, kf_q, snap_q, done_ev,
                 viewer_link=None):
        _seed(getattr(args, "seed", None))
        torch.cuda.set_device(0)
        from slam import MeshSLAMTrainer                       # spawn: import inside the child
        from mesh_splatting.triangle_renderer import render
        self.render = render
        self.config = config
        self.kf_q = kf_q
        self.snap_q = snap_q
        self.done_ev = done_ev
        async_cfg = _async_cfg(config)
        self.window_k = int(async_cfg.get("window_k", 1))                   # keyframes per iteration between keyframes
        self.topology_every = int(async_cfg.get("topology_interval", 100))  # iterations between prune / reset / densify
        self.publish_every = int(async_cfg.get("publish_interval", 100))    # iterations between published targets
        self.burst_iters = int(async_cfg.get("kf_burst_iters", 10))         # iterations right after a tracking keyframe
        self.trackable_opacity = float(_gicp_cfg(config)["trackable_opacity"])
        self.track_uids = set()               # keyframes in the tracking target
        self.trainer = MeshSLAMTrainer(args, config, dataset_path, config_path, opt, pipe, num_frames=num_frames)
        if int(_gicp_cfg(config)["window_size"]) > self.trainer.triangles.FUSE_KEEP:
            raise ValueError(f"Training.async.gicp.window_size: at most {self.trainer.triangles.FUSE_KEEP} "
                             "(TriangleModel.FUSE_KEEP, the keyframes whose observations a correction can still move)")
        self.backend = self.trainer.backend
        self.keyframe_buffer = self.backend.keyframe_buffer
        training = config["Training"]
        self.first_keyframe_iters = training["iterations_per_view"]
        self.replay_size = training["mapping_replay_size"]
        self.pose_window = training["initial_ba"]["pose_window"]
        self.pose_lr_scale = training.get("mapping_pose_lr_scale", 0.5)
        self.stats = dict(keyframes=0, topology_rounds=0)   # sent to the tracker with every target
        self.version = 0                      # of the published target
        self.corrections_applied = 0          # id of the last window correction applied
        self.newest = None                    # newest tracking keyframe
        self.newest_iteration = 0
        self.last_keyframe = -1
        self.accumulated = None               # visibility and pixel counts since the last spawn
        self.loss_history = []                # losses since the last topology update
        self.iteration = 0                    # continuous iterations
        self.ending = False
        self.end_payload = None
        self.stash = []                       # messages read ahead
        self.view = None                      # live viewer
        if viewer_link is not None:
            from live_viewer import MapperView
            self.view = MapperView(config, viewer_link, self)

    def run(self):
        torch.cuda.synchronize()
        self.snap_q.put(("ready", 0, -1, {}, {}))        # the tracker starts its clock now
        parent, checked = mp.parent_process(), time.monotonic()
        while True:
            if parent is not None and time.monotonic() - checked > PARENT_CHECK_S:
                checked = time.monotonic()
                if not parent.is_alive():
                    return                    # the tracker is gone: exit
            if self.view is not None:
                self.view.tick()              # live viewer render
            msg = self._next_message()
            if msg is not None and msg[0] == "end":
                self.ending = True
                self.end_payload = msg[1]
            elif msg is not None and msg[0] == "correct":
                self._on_correction(msg[1], msg[2])
                continue
            elif msg is not None and msg[0] == "kf":
                self._on_keyframe(msg)
                continue
            if self.ending:
                self._finish()      # all keyframes are in: final map
                return
            if self.newest is not None:
                self._continuous_round()

    def _next_message(self):
        if self.stash:
            return self.stash.pop(0)
        try:
            if not self.ending and self.newest is None:
                return self.kf_q.get(timeout=0.05)        # nothing to train yet
            return self.kf_q.get_nowait()
        except queue.Empty:
            return None

    def _keyframe_pending(self):
        """True if another keyframe is waiting."""
        try:
            following = self.kf_q.get_nowait()
        except queue.Empty:
            return False
        if following[0] == "end":
            self.ending = True
            self.end_payload = following[1]
            return False
        self.stash.append(following)
        return following[0] == "kf"

    # ---- keyframes ------------------------------------------------------------------------------------------------
    def _on_keyframe(self, msg):
        _, uid, R, T, view_iteration = msg[:5]
        mapping_keyframe = len(msg) > 5 and msg[5] == "map"      # similar to GS ICP SLAM
        cam = self.trainer._get_camera(uid)
        cam.update_RT(R.to(cam.device), T.to(cam.device))
        first = self.newest is None
        if mapping_keyframe and not first:
            self._fill_holes(cam, uid)
            return
        self.track_uids.add(int(uid))
        if first:
            self._first_keyframe(cam, uid, view_iteration)
        else:
            self._tracking_keyframe(cam, uid, view_iteration)
        self.accumulated = None       # spawning resets the statistics
        cam = self._buffered_camera(uid, cam)
        self.newest, self.newest_iteration, self.last_keyframe = cam, view_iteration, uid
        self.stats["keyframes"] += 1

    def _buffered_camera(self, uid, cam):
        """Use the keyframe buffer's CPU copy of the camera."""
        buffered = self.keyframe_buffer.get_by_uid(uid)
        if buffered is not None and buffered is not cam:
            self.trainer.cameras[uid] = buffered
            return buffered
        return cam

    def _fill_holes(self, cam, uid):
        """Mapping keyframe: add triangles where nothing is rendered."""
        with torch.no_grad():
            render_pkg = self.render(cam, self.trainer.triangles, self.trainer.pipe, self.trainer.background)
        self.trainer._keyframe_map_update(cam, render_pkg, holes_only=True)
        self._buffered_camera(uid, cam)
        self.accumulated = None

    def _first_keyframe(self, cam, uid, view_iteration):
        """covered by the initial model; the synchronous loop's view_iteration 1"""
        self.keyframe_buffer.add(cam)
        self._publish(uid)                # first target: the initial map
        self._mapping_round(cam, view_iteration, self.first_keyframe_iters)

    def _tracking_keyframe(self, cam, uid, view_iteration):
        """Tracking keyframe: spawn, publish, train a burst."""
        render_pkg = self._render_tracked_only(cam)
        self.trainer._keyframe_map_update(cam, render_pkg)
        self._publish(uid)                # publish before the burst
        if self._keyframe_pending():
            return
        visibility, _, pixel_counts, _, frustum = self._mapping_round(cam, view_iteration, self.burst_iters)
        if cam.uid > 1:
            self._topology_update(visibility, frustum, pixel_counts, uid)

    def _render_tracked_only(self, cam):
        """Render the tracking target only."""
        triangles = self.trainer.triangles
        ids = triangles.unique_kfIDs
        faces = triangles._triangle_indices.long()
        with torch.no_grad():
            tracked = torch.isin(ids, torch.tensor(sorted(self.track_uids), device=ids.device, dtype=ids.dtype))
            hidden = faces[~tracked].reshape(-1)
            saved = triangles.vertex_weight.data[hidden].clone()
            triangles.vertex_weight.data[hidden] = HIDDEN_VERTEX_WEIGHT
            render_pkg = self.render(cam, triangles, self.trainer.pipe, self.trainer.background)
            triangles.vertex_weight.data[hidden] = saved
        return render_pkg

    # ---- mapping --------------------------------------------------------------------------------------------------
    def _mapping_round(self, cam, view_iteration, iters, window=None):
        out = self.backend.run_mapping_iterations(num_iters=iters, viewpoint_cam=cam, replay_size=self.replay_size,
                                                  pose_window=self.pose_window, pose_lr_scale=self.pose_lr_scale,
                                                  view_iteration=view_iteration, total_views=self.trainer.total_frames,
                                                  window_override=window,
                                                  tick_callback=self.view.tick if self.view is not None else None)
        self.loss_history.extend(out[3])      # one loss per iteration
        return out

    def _continuous_round(self):
        """One mapping iteration on random keyframes."""
        window = random.sample(list(self.keyframe_buffer), min(self.window_k, len(self.keyframe_buffer)))
        visibility, _, pixel_counts, _, frustum = self._mapping_round(self.newest, self.newest_iteration, 1, window)
        acc = self.accumulated
        if acc is None or acc["vis"].numel() != visibility.numel():
            acc = dict(vis=torch.zeros_like(visibility, dtype=torch.bool), pix=torch.zeros_like(pixel_counts))
        acc["vis"] = acc["vis"] | visibility.bool()
        acc["pix"] = acc["pix"] + pixel_counts
        acc["fr"] = frustum
        self.accumulated = acc
        self.iteration += 1
        if self.topology_every > 0 and self.iteration % self.topology_every == 0 and self.newest.uid > 1:
            self._topology_update(acc["vis"], acc["fr"], acc["pix"], self.newest.uid)
            self.accumulated = None
        if self.publish_every > 0 and self.iteration % self.publish_every == 0:
            self._publish(self.last_keyframe)

    def _topology_update(self, visibility, frustum, pixel_counts, uid):
        """Prune, reset weights, densify."""
        if visibility.numel() != self.trainer.triangles._triangle_indices.shape[0]:
            return
        self.stats["topology_rounds"] += 1
        self.trainer._post_mapping_topology(visibility, frustum, pixel_counts, self.loss_history,
                                            self.stats["topology_rounds"], uid)
        del self.loss_history[:]

    # ---- the target and the corrections ---------------------------------------------------------------------------
    def _publish(self, keyframe_done):
        """Send the tracking target to the tracker."""
        self.version += 1
        torch.cuda.synchronize()
        self.stats["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        self.stats["triangles"] = int(self.trainer.triangles._triangle_indices.shape[0])
        target = {"__centroids__": trackable_centroids(self.trainer.triangles, self.trackable_opacity, self.track_uids),
                  "__corr__": self.corrections_applied}       # window corrections already applied to this target
        self.snap_q.put(("snap", self.version, keyframe_done, target, dict(self.stats)))
        if self.view is not None:
            self.view.publish()               # live viewer mesh

    def _on_correction(self, correction_id, correction):
        self._apply_corrections(correction)
        self.corrections_applied = correction_id
        self._publish(self.last_keyframe)

    def _apply_corrections(self, correction):
        """Move corrected keyframes and their triangles."""
        triangles = self.trainer.triangles
        with torch.no_grad():
            for uid, D in correction.items():
                D = torch.as_tensor(D, dtype=torch.float32, device=triangles.vertices.device)
                triangles.fuse_correct(uid, D)
                cameras = (self.keyframe_buffer.get_by_uid(uid), self.trainer.cameras.get(uid))
                for cam in {id(c): c for c in cameras if c is not None}.values():
                    W2C = torch.eye(4, device=cam.R.device)
                    W2C[:3, :3] = cam.R.float()
                    W2C[:3, 3] = cam.T.float()
                    W2C_new = W2C @ torch.linalg.inv(D.to(W2C.device))                   # W2C_new = W2C_old D^-1
                    cam.update_RT(W2C_new[:3, :3].to(cam.R.dtype), W2C_new[:3, 3].to(cam.T.dtype))
        self.stats["window_corrections"] = self.stats.get("window_corrections", 0) + len(correction)

    # ---- the end of the run ---------------------------------------------------------------------------------------
    def _finish(self):
        torch.cuda.synchronize()
        self.stats["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        if self.view is not None:
            self.view.final()                 # the final map and render
        self._save_outputs(self.end_payload)
        self.snap_q.put(("done", self.version, self.last_keyframe, {}, dict(self.stats)))
        self.done_ev.wait(timeout=MAPPER_EXIT_TIMEOUT_S)   # keep shared tensors alive until the tracker has them

    def _save_outputs(self, payload):
        """Save the map and the evaluation files."""
        from slam import LazyCameraProvider
        from utils.eval_utils import eval_ate
        trainer = self.trainer
        poses = _unpack_poses(payload["poses"])
        poses_online = _unpack_poses(payload["poses_online"])
        trainer._all_frame_poses = poses
        if not self.config["Results"]["save_results"]:
            return
        save_dir = self.config["Results"]["save_dir"]
        num_frames = payload["timing"]["num_frames"]
        trainer.map_state.save_mesh(num_frames)
        cam_provider = LazyCameraProvider(trainer.cameras, trainer.total_frames, trainer._get_camera)
        try:
            trainer._eval_ate = eval_ate(cam_provider, trainer.map_state.keyframe_buffer.get_keyframes_ids(), save_dir,
                                         num_frames, monocular=trainer.monocular,
                                         plot_axes=self.config.get("Results", {}).get("plot_axes", "xy"))
        except Exception as e:  # no ground truth
            print(f"Eval: keyframe ATE skipped ({type(e).__name__}: {e})")
            trainer._eval_ate = float("nan")
        trainer._dump_icp_outputs(save_dir)
        # the poses as tracked
        with open(os.path.join(save_dir, "trajectory_all_online.json"), "w") as f:
            json.dump([dict(uid=uid, R=R.tolist(), T=T.tolist(), R_gt=R_gt.tolist(), T_gt=T_gt.tolist())
                       for uid, R, T, R_gt, T_gt in poses_online], f)
        timing = dict(payload["timing"], mapper=self.stats, peak_gpu_gb=self.stats.get("peak_gpu_gb"),
                      ate_rmse_m=trainer._eval_ate, num_triangles=int(trainer.triangles._triangle_indices.shape[0]))
        if payload.get("frame_log"):          # per-frame timing of the tracker's loop (Tracker.run)
            with open(os.path.join(save_dir, "frame_times.json"), "w") as f:
                json.dump(dict(columns=["frame", "start_s", "processing_s", "tracking_s", "waiting_s", "keyframe"],
                               keyframe_codes={"0": "none", "1": "mapping keyframe", "2": "tracking keyframe"},
                               max_fps=float(payload["timing"].get("async_cfg", {}).get("max_fps", 0) or 0) or None,
                               rows=payload["frame_log"]), f)
        timing_path = os.path.join(save_dir, "timing.json")
        with open(timing_path, "w") as f:
            json.dump(timing, f, indent=2)
        print("Timing saved to", timing_path)
