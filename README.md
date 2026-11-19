<div align="center">

# Facet-SLAM

[Edward Beng Wai Tan](https://edward62740.github.io/), [Siew-Kei Lam](https://siewkeilam.github.io/ei-research-group/index.html)



Please refer to the [website](https://edward62740.github.io/facetslam/) for visualizations or the [paper](https://arxiv.org/abs/2611.23935) for more details.
</div>


## Overview

This is a real-time triangle splatting SLAM system, which can run on Jetson AGX-class compute. 



## Installation

Install eigen3 and pcl:

```
sudo apt install -y libeigen3-dev libpcl-dev
```

Run the install script:
```bash
git clone --recursive <this repository URL> facet-slam
cd facet-slam
export CUDA_HOME=/usr/local/...     # only if nvcc on your PATH is not a 12.x toolkit
bash install.sh
```

`install.sh` does the following:

1. fetches the submodules if they are missing;
2. creates the Python environment `.venv` from `uv.lock` and builds the CUDA extensions against it (`uv sync`);
3. builds the GICP tracker module (`native/pygicp`: fast_gicp at the commit GS-ICP SLAM uses, with the three patches
   in `native/pygicp/patches`);
4. builds a static Ceres 2.2.0 and the window-optimisation module (`native/ceres_window_opt`);
5. runs the two self-tests and checks that PyTorch sees the GPU.

The first run takes a while: it downloads PyTorch (about 2 GB), compiles five extensions and builds Ceres (about ten
minutes). It ends with `Installed.` and the command for a first run.

Options:

- `bash install.sh --no-live` leaves out `pyrealsense2`, for machines that will not use a camera. This option is needed for the Realsense demo.
- `EIGEN_INCLUDE_DIR=/path/to/eigen3 bash install.sh` if Eigen is not in `/usr/include/eigen3`.

The steps can also be run by hand:

```bash
uv sync
bash native/pygicp/build.sh
bash native/ceres_window_opt/build_ceres.sh native/ceres_window_opt/ceres
bash native/ceres_window_opt/build.sh
.venv/bin/python native/ceres_window_opt/test_ceres_window_opt.py    # expect 7 PASS lines
.venv/bin/python scripts/test_pin_geometry.py                        # expect ALL OK
```

After changing a submodule, force its extension to be rebuilt, since `uv sync` may reuse a cached wheel:

```bash
uv sync --reinstall-package diff-triangle-rasterization
```

## Datasets

Please use the standard sources to download, or

```bash
bash scripts/download_tum.sh       # datasets/tum/
bash scripts/download_replica.sh   # datasets/replica/
```

ScanNet scenes are read as TUM-format sequences from `datasets/scannet/<scene>/`. So symlink it to `datasets`.

## Running

A few usage examples below.
```bash
source .venv/bin/activate
python main.py --dataset fr1_desk --max_fps 30 --eval     # TUM at the camera rate (e.g. 30FPS)
python main.py --dataset room0 --eval                     # Replica, UNCAPPED FPS
python main.py --dataset_path datasets/tum/rgbd_dataset_freiburg1_desk --config configs/rgbd/tum/fr1_desk.yaml
bash tum.sh        # table I settings
bash replica.sh    # all 8 replica scenes
```

Note on performance numbers:
You will get slightly different numbers over different runs and FPS. The variance can be higher on worse performing scenes. If you want to mitigate this, set a fixed seed via args, and ensure there is no CPU/GPU contention. You can also set the #CPU threads etc. in `slam_mp.py` and cap the FPS.


Sequence names for `--dataset`:

- TUM RGB-D: `fr1_desk fr1_desk2 fr1_xyz fr1_room fr1_360 fr1_rpy fr2_xyz fr3_office fr3_stfar fr3_stnear`
- Replica: `room0 room1 room2 office0 office1 office2 office3 office4`
- ScanNet: `scene0000_00 scene0059_00 scene0106_00 scene0169_00 scene0181_00 scene0207_00`

Main flags:

| Flag | Effect |
|---|---|
| `--max_fps N` | pace the input at a camera rate of N frames per second (default: no pacing) |
| `--eval` | after the run, score renders at the estimated poses and save example images |
| `--seed N` | random seed (default 0) |
| `--num_frames N` | stop after N frames |
| `--output_dir DIR` | where runs are written (default `output`) |
| `--dry_run` | print the merged configuration and exit |
| `--verbose` | show the log of the SLAM processes instead of the progress bar |
| `-o KEY=VALUE` | override a setting, dot notation; `+KEY=VALUE` adds a key no config defines |

### Configuration

`main.py` reads the per-sequence config under `configs/rgbd/`, then the Facet-SLAM settings on top: `configs/facet/tum.yaml`
for TUM and RealSense input, `configs/facet/replica.yaml` for Replica, `configs/facet/scannet.yaml` for ScanNet
(`--preset` chooses one explicitly). Settings that are in no config file are code defaults, listed with their meaning
in `slam_mp._gicp_cfg`.

The keyframe window optimisation is optional. It is on by default and is switched off with

```bash
python main.py --dataset fr1_desk --max_fps 30 -o Training.async.gicp.window_opt=false
```

The poses then stay as tracked, and the native module of `native/ceres_window_opt` is not needed.

### Output

A run is written to `OUTPUT_DIR/<dataset>_<sequence>/<timestamp>/`:

The summary is also printed at the end of the run. The log of the SLAM processes goes to
`OUTPUT_DIR/<name>_seed<seed>.log`.

### Live viewer

This one is for the rerun viewer. Specify the flag as needed. Please don't enable this if you are doing FPS testing, as it is quite slow.
```bash
python main.py --dataset fr1_desk --max_fps 30 --viewer both                      # web viewer on localhost
python main.py --dataset fr1_desk --max_fps 30 --viewer both --viewer_sink spawn  # native Rerun window
python main.py --dataset fr1_desk --viewer map --viewer_layout map                # the map alone, large
```


### Live input from a RealSense D435 / D435i

```bash
python main.py --d435                                   # 640x480 at 30 Hz , until Ctrl-C
python main.py --d435 --viewer both                     # with the live viewer
python main.py --d435 --d435_seconds 60 --d435_size 848x480  #specify size
```

Keep the camera still until `Tracking started` appears: the map is initialised from the first frame. Ctrl-C stops the
camera, and the run then finishes and saves as usual. Depth beyond `--d435_max_depth` (5 m) is dropped, beyond this the depth is rather noisy and unreliable.

Every frame is also recorded, to `OUTPUT_DIR/d435_live/<time>/`, uncompressed (about 2.8 GB per minute). A recording
can be compressed and processed again:

```bash
python d435_capture.py --to_png <recording>                                         # shrink it in place
python main.py --config <recording>/replay.yaml --dataset_path <recording>          # offline, every frame
python main.py --d435_replay <recording>                                            # through the live path, at its recorded rate
```
## Release roadmap
- [x] Minimal release
- [x] Clean and modularize the code
- [x] Add the jetson demo code
- [ ] v1.0 (stable)
- [ ] Improvements on initialization and densification
- [ ] Add support for other cameras
      
## Citation

Please cite the following if you use our implementation

```
@misc{tan2026facetslamrealtimetrianglesplatting,
      title={Facet-SLAM: Real-Time Triangle Splatting SLAM}, 
      author={Edward Beng Wai Tan and Siew-Kei Lam},
      year={2026},
      eprint={2611.23935},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2611.23935}, 
}
```
and also TS-SLAM

```
@inproceedings{fry2026triangle,
  title     = {Triangle Splatting {SLAM}},
  author    = {Fry, Nicholas and Dexheimer, Eric and Mazur, Kirill and Kelly, Paul H. J. and Davison, Andrew J.},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Licence and acknowledgements

Please see the various licenses contained within this code for usage. In general, only non-commercial (research) use is allowed. 

Our work builds directly on [TS-SLAM](https://github.com/nmjfry/ts-slam), so please cite theirs if you use our implementation. We also thank [GS-ICP SLAM](https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM), [Mesh Splatting](https://github.com/meshsplatting/mesh-splatting) and many others.
