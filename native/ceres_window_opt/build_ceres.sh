#!/bin/bash
# Builds a static Ceres Solver 2.2.0 (no glog, SuiteSparse, LAPACK or CUDA).
# Usage: build_ceres.sh PREFIX [EIGEN_INCLUDE_DIR] [JOBS]
set -euo pipefail
PREFIX=$(realpath -m "$1"); EIGEN=${2:-/usr/include/eigen3}; JOBS=${3:-8}
SRC=$(dirname "$PREFIX")/ceres-src; mkdir -p "$SRC"; cd "$SRC"
[ -s ceres-solver-2.2.0.tar.gz ] || curl -fsSL -o ceres-solver-2.2.0.tar.gz https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.2.0.tar.gz
sha256sum ceres-solver-2.2.0.tar.gz
rm -rf ceres-solver-2.2.0 build; tar -xzf ceres-solver-2.2.0.tar.gz; mkdir build; cd build
cmake ../ceres-solver-2.2.0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DMINIGLOG=ON -DMINIGLOG_MAX_LOG_LEVEL=-1 -DGFLAGS=OFF -DSUITESPARSE=OFF -DLAPACK=OFF -DEIGENSPARSE=ON -DUSE_CUDA=OFF -DSCHUR_SPECIALIZATIONS=ON \
  -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_DOCUMENTATION=OFF -DEigen3_DIR= -DEIGEN3_INCLUDE_DIR="$EIGEN" \
  -DCMAKE_PREFIX_PATH="$(dirname "$EIGEN")/.."
make -j"$JOBS" && make install
echo "ceres installed in $PREFIX"
