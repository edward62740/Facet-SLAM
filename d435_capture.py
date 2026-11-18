"""Records a RealSense D435 / D435i to disk as a TUM-format sequence while SLAM reads it (main.py --d435).

  python d435_capture.py --out SPOOL [--width 640 --height 480 --fps 30 --max_depth 5]
  python d435_capture.py --out SPOOL --replay DIR    # feed a recording instead of a camera
  python d435_capture.py --to_png SPOOL              # compress a finished recording
"""
import argparse
import json
import os
import signal
import sys
import time

import cv2
import numpy as np

PNG = [cv2.IMWRITE_PNG_COMPRESSION, 1]
# extensions and imwrite parameters per format
FORMATS = dict(raw=(".bmp", ".pgm", []), png=(".png", ".png", PNG))


def write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


class Spool:
    def __init__(self, root, max_depth, fmt="raw"):
        self.root, self.max_depth, self.n, self.depth_scale = root, float(max_depth), 0, None
        self.ext_rgb, self.ext_depth, self.params = FORMATS[fmt]
        self.t_first = None                       # time of frame 1
        self.t_last = None                        # wall clock of the newest frame
        for sub in ("rgb", "depth"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        self.lists = {name: open(os.path.join(root, name + ".txt"), "w") for name in ("rgb", "depth", "groundtruth")}
        self.lists["rgb"].write("# ts filename\n")
        self.lists["depth"].write("# ts filename\n")
        self.lists["groundtruth"].write("# ts tx ty tz qx qy qz qw\n")

    def calibration(self, width, height, fx, fy, cx, cy, coeffs, depth_scale, source):
        """Write the camera calibration."""
        self.depth_scale = float(depth_scale)
        coeffs = [float(c) for c in coeffs] + [0.0] * 5
        with open(os.path.join(self.root, "caminfo.txt"), "w") as f:
            f.write("## camera parameters\nW H fx fy cx cy depth_scale depth_trunc dataset_type\n"
                    f"{width} {height} {fx} {fy} {cx} {cy} {self.depth_scale} {self.max_depth} tum\n")
        write_atomic(os.path.join(self.root, "calibration.json"), json.dumps(dict(
            width=int(width), height=int(height), fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            k1=coeffs[0], k2=coeffs[1], p1=coeffs[2], p2=coeffs[3], k3=coeffs[4], depth_scale=self.depth_scale,
            distorted=bool(any(abs(c) > 1e-12 for c in coeffs[:5])), max_depth=self.max_depth, source=source)))

    def put(self, bgr, depth, stamp):
        """Write one frame."""
        self.t_last = time.perf_counter()
        if self.n == 1:
            self.t_first = self.t_last
        depth = np.ascontiguousarray(depth, dtype=np.uint16)
        if self.max_depth > 0:
            depth = np.where(depth > self.max_depth * self.depth_scale, 0, depth).astype(np.uint16)
        names = {}
        for sub, image, ext in (("depth", depth, self.ext_depth), ("rgb", bgr, self.ext_rgb)):
            names[sub] = "%06d%s" % (self.n, ext)
            tmp = os.path.join(self.root, sub, ".tmp_" + names[sub])
            if not cv2.imwrite(tmp, image, self.params):
                raise OSError("could not write " + tmp)
            os.replace(tmp, os.path.join(self.root, sub, names[sub]))
        self.lists["rgb"].write("%.6f rgb/%s\n" % (stamp, names["rgb"]))
        self.lists["depth"].write("%.6f depth/%s\n" % (stamp, names["depth"]))
        self.lists["groundtruth"].write("%.6f 0 0 0 0 0 0 1\n" % stamp)
        if self.n % 30 == 0:
            for f in self.lists.values():
                f.flush()
        write_atomic(os.path.join(self.root, "latest"), str(self.n))
        self.n += 1

    def holding(self):
        """True while waiting for the tracker to start."""
        return self.n == 1 and not os.path.exists(os.path.join(self.root, "GO"))

    def close(self):
        for f in self.lists.values():
            f.close()
        write_atomic(os.path.join(self.root, "END"), str(self.n))


def realsense(args, spool, stopped):
    import pyrealsense2 as rs
    pipeline, config = rs.pipeline(), rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    try:
        device = profile.get_device()
        units = device.first_depth_sensor().get_depth_scale()            # metres per depth unit
        k = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        align = rs.align(rs.stream.color)                                # depth resampled into the colour camera
        for _ in range(args.warmup):                                     # auto-exposure settles
            pipeline.wait_for_frames(10000)
        name = f"{device.get_info(rs.camera_info.name)} {device.get_info(rs.camera_info.serial_number)}"
        spool.calibration(k.width, k.height, k.fx, k.fy, k.ppx, k.ppy, k.coeffs, round(1.0 / units), name)
        print(f"{name}: {k.width}x{k.height} at {args.fps} Hz, fx {k.fx:.2f} fy {k.fy:.2f} cx {k.ppx:.2f} cy {k.ppy:.2f}, "
              f"depth unit {units * 1000:g} mm", flush=True)
        t0 = None
        while not stopped(spool.n):
            frames = align.process(pipeline.wait_for_frames(5000))
            depth, colour = frames.get_depth_frame(), frames.get_color_frame()
            if not depth or not colour or spool.holding():
                continue
            stamp = frames.get_timestamp() / 1000.0
            t0 = stamp if t0 is None else t0
            spool.put(np.asanyarray(colour.get_data()), np.asanyarray(depth.get_data()), stamp - t0)
    finally:
        pipeline.stop()


def replay(args, spool, stopped):
    root = args.replay

    def listing(name):
        rows = [l.split() for l in open(os.path.join(root, name)) if l.strip() and not l.startswith("#")]
        return np.array([float(r[0]) for r in rows]), [os.path.join(root, r[1]) for r in rows]

    cam = [l.split() for l in open(os.path.join(root, "caminfo.txt")) if l.strip() and l.strip()[0].isdigit()][0]
    width, height, (fx, fy, cx, cy, depth_scale) = int(cam[0]), int(cam[1]), (float(v) for v in cam[2:7])
    t_rgb, rgb = listing("rgb.txt")
    t_depth, depth = listing("depth.txt")
    spool.calibration(width, height, fx, fy, cx, cy, [], depth_scale, "replay of " + os.path.abspath(root))
    print(f"replay of {root}: {width}x{height}, {len(rgb)} frames, fx {fx:.2f} fy {fy:.2f}", flush=True)
    period = float(np.median(np.diff(t_rgb)))
    start = time.perf_counter()
    for i, stamp in enumerate(t_rgb):
        while spool.holding() and not stopped(spool.n):
            time.sleep(0.005)
            start = time.perf_counter() - (t_rgb[i - 1] - t_rgb[0]) / args.replay_speed    # the clock stands still
        if stopped(spool.n):
            break
        due = (stamp - t_rgb[0]) / args.replay_speed
        late = time.perf_counter() - start - due
        if late > period / args.replay_speed:
            continue                                                     # its time has passed: dropped
        if late < 0:
            time.sleep(-late)
        j = int(np.argmin(np.abs(t_depth - stamp)))
        spool.put(cv2.imread(rgb[i], cv2.IMREAD_COLOR), cv2.imread(depth[j], cv2.IMREAD_ANYDEPTH), stamp - t_rgb[0])


def to_png(root):
    """Convert a raw recording to PNG in place."""
    if not os.path.exists(os.path.join(root, "END")):
        sys.exit("not a finished recording (no END file): " + root)
    done = 0
    for listing in ("rgb.txt", "depth.txt"):
        path = os.path.join(root, listing); out = []
        for line in open(path):
            parts = line.split()
            if line.startswith("#") or len(parts) != 2 or parts[1].endswith(".png"):
                out.append(line); continue
            src = os.path.join(root, parts[1]); dst = os.path.splitext(parts[1])[0] + ".png"
            image = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if image is None or not cv2.imwrite(os.path.join(root, dst), image, PNG):
                sys.exit("could not convert " + src)
            os.remove(src); out.append("%s %s\n" % (parts[0], dst)); done += 1
        write_atomic(path, "".join(out))
    print(f"converted {done} images of {root} to PNG")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--to_png":
        return to_png(sys.argv[2])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="spool directory (created)")
    ap.add_argument("--format", choices=sorted(FORMATS), default="raw", help="raw: BMP + 16-bit PGM, uncompressed (default); png")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_depth", type=float, default=5.0, help="metres; depth beyond it is written as 0 (0: keep all)")
    ap.add_argument("--serial", default=None, help="serial number, with several cameras attached")
    ap.add_argument("--warmup", type=int, default=30, help="frames discarded before the first one")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0: until stopped)")
    ap.add_argument("--frames", type=int, default=0, help="stop after this many frames (0: until stopped)")
    ap.add_argument("--replay", default=None, help="a recorded sequence instead of a camera")
    ap.add_argument("--replay_speed", type=float, default=1.0)
    args = ap.parse_args()
    spool = Spool(args.out, args.max_depth, args.format)
    flag = dict(stop=False)

    def on_signal(signum, frame):
        flag["stop"] = True
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    def stopped(n):
        return (flag["stop"] or (args.frames > 0 and n >= args.frames)
                or (args.seconds > 0 and spool.t_first is not None and time.perf_counter() - spool.t_first >= args.seconds)
                or os.path.exists(os.path.join(args.out, "STOP")))
    try:
        (replay if args.replay else realsense)(args, spool, stopped)
    finally:
        spool.close()
        seconds = (spool.t_last - spool.t_first) if spool.t_first is not None else 0.0    # frame 1 to the last frame
        print(f"stopped: {spool.n} frames, {(spool.n - 2) / max(seconds, 1e-9):.1f} Hz over {seconds:.1f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
