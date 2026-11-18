#!/bin/bash
# Sets up Facet-SLAM: the Python environment (uv) and the two native modules.
# Needs a CUDA 12.x toolkit, uv, git, cmake, g++, curl, libeigen3-dev, libpcl-dev.
# Usage: bash install.sh [--no-live]     (--no-live: without pyrealsense2)
set -euo pipefail
cd "$(dirname "$0")"
EIGEN=${EIGEN_INCLUDE_DIR:-/usr/include/eigen3}
SYNC=()
[ "${1:-}" = "--no-live" ] && SYNC=(--no-default-groups)

fail() { echo "install.sh: $*" >&2; exit 1; }
for tool in uv git cmake g++ curl; do
    command -v "$tool" > /dev/null || fail "$tool not found"
done
[ -n "${CUDA_HOME:-}" ] && export PATH="$CUDA_HOME/bin:$PATH"
command -v nvcc > /dev/null || fail "nvcc not found: install a CUDA 12.x toolkit or set CUDA_HOME"
nvcc --version | grep -q "release 12\." || fail "need a CUDA 12.x toolkit (found: $(nvcc --version | grep release)); set CUDA_HOME"
[ -f "$EIGEN/Eigen/Core" ] || fail "Eigen not found in $EIGEN (libeigen3-dev, or set EIGEN_INCLUDE_DIR)"
ls /usr/include/pcl-* > /dev/null 2>&1 || echo "install.sh: PCL headers not found (libpcl-dev)" >&2

echo "== 1/5  submodules"
if [ -z "$(ls -A submodules/simple-knn 2> /dev/null)" ]; then
    git submodule update --init --recursive
fi

echo "== 2/5  Python environment (slow the first time)"
uv sync "${SYNC[@]}"

echo "== 3/5  GICP module"
bash native/pygicp/build.sh

echo "== 4/5  window optimisation module"
if [ ! -f native/ceres_window_opt/ceres/lib/libceres.a ]; then
    bash native/ceres_window_opt/build_ceres.sh native/ceres_window_opt/ceres "$EIGEN" "$(nproc)"
fi
bash native/ceres_window_opt/build.sh native/ceres_window_opt/ceres .venv/bin/python "$EIGEN"

echo "== 5/5  tests"
.venv/bin/python native/ceres_window_opt/test_ceres_window_opt.py
.venv/bin/python scripts/test_pin_geometry.py
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'torch does not see a GPU'; print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"

echo
echo "Installed. Try: .venv/bin/python main.py --dataset fr1_desk --max_fps 30 --eval"
