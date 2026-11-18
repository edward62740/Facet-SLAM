"""Dataset that reads the frames d435_capture.py is writing."""
import os
import sys
import time

import numpy as np

from utils.dataset import MonocularDataset

UNBOUNDED = 999999          # frame count while the camera runs
STALL_S = 20.0              # give up after this long without a new frame (s)


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def latest(root):
    """index of the newest complete frame, -1 before the first"""
    n = _read_int(os.path.join(root, "latest"))
    return -1 if n is None else n


def ended(root):
    """the number of frames once the capture has stopped, else None"""
    return _read_int(os.path.join(root, "END"))


class _Paths:
    """Frame paths by index; waits for frames not written yet."""
    EXTENSIONS = dict(rgb=(".bmp", ".png"), depth=(".pgm", ".png"))

    def __init__(self, root, sub):
        self.root, self.sub = root, sub

    def __len__(self):
        n = ended(self.root)
        return UNBOUNDED if n is None else n

    def __getitem__(self, i):
        stem = os.path.join(self.root, self.sub, "%06d" % int(i))
        waited = time.monotonic()
        while True:
            path = next((stem + ext for ext in self.EXTENSIONS[self.sub] if os.path.exists(stem + ext)), None)
            if path is not None:
                break
            n = ended(self.root)
            if n is not None and int(i) >= n:
                raise IndexError(f"frame {i}: the camera stopped after {n} frames")
            if time.monotonic() - waited > STALL_S:
                raise RuntimeError(f"no frame {i} from the camera within {STALL_S:.0f} s (see {self.root}/capture.log)")
            time.sleep(0.002)
        return path


class _Poses:
    """no ground truth: identity for every frame, as the D435i recordings"""

    def __init__(self, root):
        self.root = root

    def __len__(self):
        n = ended(self.root)
        return UNBOUNDED if n is None else n

    def __getitem__(self, i):
        return np.eye(4, dtype=np.float64)


class LiveFrames:
    """Frame 0, then always the newest frame."""

    def __init__(self, root, limit=None):
        self.root, self.limit, self.taken = root, limit, []

    def __len__(self):
        return len(self.taken)

    def __getitem__(self, key):
        return [] if isinstance(key, slice) else self.taken[key]

    def __iter__(self):
        last = -1
        while not self.limit or len(self.taken) < self.limit:
            if last == 0 and not os.path.exists(os.path.join(self.root, "GO")):
                open(os.path.join(self.root, "GO"), "w").close()      # the camera resumes with frame 1
                print("Tracking started: the camera can move", file=sys.stderr, flush=True)
            waited = time.monotonic()
            while True:
                newest = latest(self.root)
                if newest > last:
                    break
                if ended(self.root) is not None and latest(self.root) <= last:
                    return
                if time.monotonic() - waited > STALL_S:
                    raise RuntimeError(f"no new frame from the camera within {STALL_S:.0f} s (see {self.root}/capture.log)")
                time.sleep(0.001)
            last = 0 if last < 0 else newest          # the map is initialised from frame 0
            self.taken.append(last)
            yield last


class LiveDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        self.color_paths = _Paths(path, "rgb")
        self.depth_paths = _Paths(path, "depth")
        self.poses = _Poses(path)
        self.num_imgs = len(self.poses)

    def live_frames(self, limit=None):
        return LiveFrames(self.path, limit)
