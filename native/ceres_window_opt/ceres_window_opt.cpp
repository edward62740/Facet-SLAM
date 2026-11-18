// Pose-only sliding-window optimisation of RGB-D keyframe poses, solved by Ceres: automatic differentiation, Levenberg-Marquardt, and
// the per-pair exposure eliminated by the Schur complement. No hand-written Jacobians, linearisation or elimination.
// Standard forms only:
//  - poses: unit quaternion (ceres::EigenQuaternionManifold) and translation, camera to world; keyframes outside the window
//    are constant (a local window, as ORB-SLAM's) and the first keyframe of the run fixes the gauge
//  - landmarks: points with normals held fixed in their host keyframe's frame (pose-only: no landmark parameters), so a host's correction moves
//    them rigidly
//  - geometric observation: point-to-plane between the landmark and the point the observing keyframe measured at its
//    projection, along the landmark's normal, Huber loss; correspondences fixed inside a solve, re-associated between solves
//  - photometric observation (direct methods, e.g. DSO): the landmark's grey level in its host against the observing
//    keyframe's image at its projection (ceres::BiCubicInterpolator), per-pair affine exposure as a free parameter block
//    with a weak prior, Huber loss
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/rotation.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <chrono>
#include <vector>
namespace py = pybind11;
using arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using iarr = py::array_t<long, py::array::c_style | py::array::forcecast>;
using farr = py::array_t<float, py::array::c_style | py::array::forcecast>;

// r = n_w . (T_a X_h - T_k y_k) / sigma, against the observer's measured point y_k, along the landmark's normal n_h in its
// host frame (n_w = R_a n_h)
struct PointToPlane {
  PointToPlane(const double* X, const double* n, const double* y, double sigma) : sigma_(sigma) { for (int i = 0; i < 3; ++i) { X_[i] = X[i]; n_[i] = n[i]; y_[i] = y[i]; } }
  template <typename T> bool operator()(const T* qa, const T* ta, const T* qk, const T* tk, T* r) const {
    // Eigen quaternion storage (x, y, z, w) -> Ceres rotation convention (w, x, y, z)
    const T Qa[4] = {qa[3], qa[0], qa[1], qa[2]}, Qk[4] = {qk[3], qk[0], qk[1], qk[2]};
    const T Xh[3] = {T(X_[0]), T(X_[1]), T(X_[2])}, nh[3] = {T(n_[0]), T(n_[1]), T(n_[2])}, yk[3] = {T(y_[0]), T(y_[1]), T(y_[2])};
    T Xw[3], nw[3], yw[3];
    ceres::QuaternionRotatePoint(Qa, Xh, Xw); ceres::QuaternionRotatePoint(Qa, nh, nw); ceres::QuaternionRotatePoint(Qk, yk, yw);
    r[0] = (nw[0] * (Xw[0] + ta[0] - yw[0] - tk[0]) + nw[1] * (Xw[1] + ta[1] - yw[1] - tk[1]) + nw[2] * (Xw[2] + ta[2] - yw[2] - tk[2])) / T(sigma_);
    return true;
  }
  double X_[3], n_[3], y_[3], sigma_;
};

using Grid = ceres::Grid2D<float, 1>;
using Interp = ceres::BiCubicInterpolator<Grid>;
// r = (I_k(pi(T_k^-1 T_a X_h)) - alpha c - beta) / sigma
struct Photo {
  Photo(const double* X, double c, const Interp* I, const double* K, double sigma) : c_(c), sigma_(sigma), I_(I) {
    for (int i = 0; i < 3; ++i) X_[i] = X[i]; for (int i = 0; i < 4; ++i) K_[i] = K[i]; }
  template <typename T> bool operator()(const T* qa, const T* ta, const T* qk, const T* tk, const T* ab, T* r) const {
    const T Qa[4] = {qa[3], qa[0], qa[1], qa[2]}, Qki[4] = {qk[3], -qk[0], -qk[1], -qk[2]};
    const T Xh[3] = {T(X_[0]), T(X_[1]), T(X_[2])};
    T Xw[3], Xk[3]; ceres::QuaternionRotatePoint(Qa, Xh, Xw);
    const T d[3] = {Xw[0] + ta[0] - tk[0], Xw[1] + ta[1] - tk[1], Xw[2] + ta[2] - tk[2]}; ceres::QuaternionRotatePoint(Qki, d, Xk);
    if (Xk[2] < T(1e-2)) return false;
    const T u = T(K_[0]) * Xk[0] / Xk[2] + T(K_[2]), v = T(K_[1]) * Xk[1] / Xk[2] + T(K_[3]); T I; I_->Evaluate(v, u, &I);
    r[0] = (I - ab[0] * T(c_) - ab[1]) / T(sigma_); return true;
  }
  double X_[3], c_, K_[4], sigma_; const Interp* I_;
};
struct AffinePrior {
  explicit AffinePrior(double s) : s_(s) {}
  template <typename T> bool operator()(const T* ab, T* r) const { r[0] = (ab[0] - T(1.0)) / T(s_); r[1] = ab[1] / T(s_); return true; }
  double s_;
};

// quat: N x 4 (x y z w, updated), trans: N x 3 (updated), pose_free: N; lm_x: M x 3 points in their host's frame, lm_host: M
// obs_lm, obs_kf: O; obs_y: O x 3, the observer's measured point in its frame; obs_n: O x 3, the landmark's normal in its host's
// frame; sigma: geometric noise (m)
// photo: None or dict(lm, kf, pair: P; c: P host grey levels; images: K x H x W float32; kf_image: N, row in images or -1;
//        ab: Q x 2 per-pair exposure (updated); K: fx fy cx cy; sigma; prior_sigma)
py::dict solve(arr quat, arr trans, iarr pose_free, arr lm_x, iarr lm_host, iarr obs_lm, iarr obs_kf, arr obs_y, arr obs_n, double sigma,
               int max_iters, int threads, double huber, py::object photo) {
  auto Q = quat.mutable_unchecked<2>(); auto Tt = trans.mutable_unchecked<2>(); auto pf = pose_free.unchecked<1>();
  auto X = lm_x.unchecked<2>(); auto lh = lm_host.unchecked<1>(); auto ol = obs_lm.unchecked<1>(); auto ok = obs_kf.unchecked<1>();
  auto oy = obs_y.unchecked<2>(); auto on = obs_n.unchecked<2>();
  const long N = Q.shape(0);
  ceres::Problem problem; ceres::LossFunction* loss = new ceres::HuberLoss(huber); auto* order = new ceres::ParameterBlockOrdering;
  std::vector<char> used(N, 0);
  for (long o = 0; o < ol.shape(0); ++o) {
    const long j = ol(o), k = ok(o), a = lh(j); if (a == k) continue;
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<PointToPlane, 1, 4, 3, 4, 3>(new PointToPlane(X.data(j, 0), on.data(o, 0), oy.data(o, 0), sigma)),
                             loss, Q.mutable_data(a, 0), Tt.mutable_data(a, 0), Q.mutable_data(k, 0), Tt.mutable_data(k, 0));
    used[a] = used[k] = 1;
  }
  std::vector<Grid> grids; std::vector<Interp> interps; farr images; arr ab_arr, pc_arr, K_arr; iarr pl_arr, pk_arr, pp_arr, im_arr;
  bool has_ab = false;
  if (!photo.is_none()) {
    py::dict P = photo.cast<py::dict>(); images = P["images"].cast<farr>(); ab_arr = P["ab"].cast<arr>(); pl_arr = P["lm"].cast<iarr>();
    pk_arr = P["kf"].cast<iarr>(); pp_arr = P["pair"].cast<iarr>(); im_arr = P["kf_image"].cast<iarr>(); pc_arr = P["c"].cast<arr>();
    K_arr = P["K"].cast<arr>(); const double sg = P["sigma"].cast<double>(), pr = P["prior_sigma"].cast<double>();
    const long K = images.shape(0); const int H = (int)images.shape(1), W = (int)images.shape(2); grids.reserve(K); interps.reserve(K);
    for (long i = 0; i < K; ++i) { grids.emplace_back(images.data(i), 0, H, 0, W); interps.emplace_back(grids.back()); }
    auto pl = pl_arr.unchecked<1>(); auto pk = pk_arr.unchecked<1>(); auto pp = pp_arr.unchecked<1>(); auto im = im_arr.unchecked<1>();
    auto pc = pc_arr.unchecked<1>(); auto ab = ab_arr.mutable_unchecked<2>();
    for (long o = 0; o < pl.shape(0); ++o) {
      const long j = pl(o), k = pk(o), a = lh(j); if (a == k || im(k) < 0) continue;
      problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Photo, 1, 4, 3, 4, 3, 2>(new Photo(X.data(j, 0), pc(o), &interps[im(k)], K_arr.data(0), sg)),
                               loss, Q.mutable_data(a, 0), Tt.mutable_data(a, 0), Q.mutable_data(k, 0), Tt.mutable_data(k, 0), ab.mutable_data(pp(o), 0));
      used[a] = used[k] = 1;
    }
    // each photometric residual touches one exposure block, so the exposure blocks are eliminated first (Schur complement)
    for (long q = 0; q < ab.shape(0); ++q) if (problem.HasParameterBlock(ab.mutable_data(q, 0))) {
      problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AffinePrior, 2, 2>(new AffinePrior(pr)), nullptr, ab.mutable_data(q, 0));
      order->AddElementToGroup(ab.mutable_data(q, 0), 0); has_ab = true;
    }
  }
  for (long i = 0; i < N; ++i) if (used[i]) {
    problem.SetManifold(Q.mutable_data(i, 0), new ceres::EigenQuaternionManifold);
    if (!pf(i)) { problem.SetParameterBlockConstant(Q.mutable_data(i, 0)); problem.SetParameterBlockConstant(Tt.mutable_data(i, 0)); }
    order->AddElementToGroup(Q.mutable_data(i, 0), 1); order->AddElementToGroup(Tt.mutable_data(i, 0), 1);
  }
  ceres::Solver::Options opts; opts.max_num_iterations = max_iters; opts.num_threads = threads; opts.logging_type = ceres::SILENT;
  opts.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  if (has_ab) { opts.linear_solver_type = ceres::DENSE_SCHUR; opts.linear_solver_ordering.reset(order); }
  else { opts.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY; delete order; }
  ceres::Solver::Summary sum; const auto t0 = std::chrono::steady_clock::now();
  { py::gil_scoped_release release; ceres::Solve(opts, &problem, &sum); }
  py::dict d; d["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  d["initial_cost"] = sum.initial_cost; d["final_cost"] = sum.final_cost; d["iterations"] = (int)sum.iterations.size();
  d["usable"] = sum.IsSolutionUsable(); d["residuals"] = sum.num_residuals; d["message"] = sum.message;
  d["linear_solver"] = std::string(ceres::LinearSolverTypeToString(sum.linear_solver_type_used));
  return d;
}
PYBIND11_MODULE(ceres_window_opt, m) { m.def("solve", &solve, "pose-only window optimisation of keyframe poses (see ceres_window_opt.cpp)"); }
