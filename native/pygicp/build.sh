#!/bin/bash
# Builds pygicp: fast_gicp (GS-ICP SLAM's fork, commit e2954b1) with the patches in patches/.
# Needs PCL, Eigen, OpenMP.   Usage: bash native/pygicp/build.sh [python]
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); PY=${1:-$D/../../.venv/bin/python}
if [ ! -d "$D/src" ]; then git clone --recursive https://github.com/Lab-of-AI-and-Robotics/fast_gicp.git "$D/src"; git -C "$D/src" checkout e2954b1; fi
cd "$D/src"; git reset -q --hard e2954b1; git apply "$D"/patches/*.patch
"$PY" setup.py build_ext --build-lib "$D/build/lib.$("$PY" -c 'import sysconfig; print(sysconfig.get_platform() + "-" + sysconfig.get_python_version().replace(".", ""))' | sed 's/cpython-//')" 2>&1 | tail -3
ls "$D"/build/lib.*/pygicp*.so
