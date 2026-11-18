#!/bin/bash
# Builds the window optimisation module against the static Ceres of build_ceres.sh.
# Usage: bash native/ceres_window_opt/build.sh [CERES_PREFIX] [PYTHON] [EIGEN_INCLUDE_DIR]
set -e
D=$(cd "$(dirname "$0")" && pwd); C=$(realpath -sm "${1:-$D/ceres}"); PY=$(realpath -sm "${2:-$D/../../.venv/bin/python}"); EIGEN=$(realpath -sm "${3:-/usr/include/eigen3}"); cd "$D"
g++ -O3 -DNDEBUG -std=c++17 -fPIC -shared ceres_window_opt.cpp -o ceres_window_opt$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))") \
  -I"$C/include" -I"$C/include/ceres/internal/miniglog" -I"$EIGEN" $($PY -c "import pybind11, sysconfig; print('-I' + pybind11.get_include(), '-I' + sysconfig.get_paths()['include'])") \
  "$C/lib/libceres.a" -lpthread
echo "built $(ls ceres_window_opt*.so)"
