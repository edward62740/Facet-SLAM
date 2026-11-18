"""Known-answer checks of ceres_window_opt.solve on synthetic scenes."""
import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ceres_window_opt as window_opt
rng = np.random.default_rng(1)
def rot(w): return cv2.Rodrigues(np.asarray(w, float))[0]
def q_from_R(R):
    from math import sqrt; t = np.trace(R)
    if t > 0: s = sqrt(t + 1) * 2; return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, .25 * s])
    i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3; s = sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2; q = np.zeros(4); q[i] = .25 * s; q[j] = (R[j, i] + R[i, j]) / s; q[k] = (R[k, i] + R[i, k]) / s; q[3] = (R[k, j] - R[j, k]) / s; return q
def R_from_q(q):
    x, y, z, w = q / np.linalg.norm(q); return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
def state(T): return np.array([q_from_R(t[:3, :3]) for t in T]), T[:, :3, 3].copy()
def errs(q, t, Tt):
    e_t = [np.linalg.norm(t[i] - Tt[i, :3, 3]) for i in range(len(Tt))]; e_r = [np.degrees(np.linalg.norm(cv2.Rodrigues(R_from_q(q[i]) @ Tt[i, :3, :3].T)[0])) for i in range(len(Tt))]; return max(e_t), max(e_r)
ok_all = True
def check(name, c, msg):
    global ok_all; print(('PASS ' if c else 'FAIL ') + name + ': ' + msg, flush=True); ok_all &= bool(c)
# ---- geometric scene
N = 8; Tg = []
for i in range(N):
    T = np.eye(4); T[:3, :3] = rot([.02 * i, .05 * i - .15, .01 * i]); T[:3, 3] = [.15 * i - .5, .03 * i, .05 * i]; Tg.append(T)
Tg = np.array(Tg); pts, nor = [], []
for ax, off in [(2, 3.), (0, -1.5), (1, 1.2), (0, 1.8)]:
    p = rng.uniform(-1.5, 1.5, (800, 3)); p[:, 2] += 2.; p[:, ax] = off; n = np.zeros((800, 3)); n[:, ax] = 1.; pts.append(p); nor.append(n)
pts, nor = np.concatenate(pts), np.concatenate(nor); host = rng.integers(0, N, len(pts)); X = np.zeros((len(pts), 3)); keep = np.ones(len(pts), bool)
for j in range(len(pts)):
    Ti = np.linalg.inv(Tg[host[j]]); X[j] = Ti[:3, :3] @ pts[j] + Ti[:3, 3]; keep[j] = X[j, 2] > .3
host, X, pw, nw = host[keep], X[keep], pts[keep], nor[keep]
ol, ok, oy = [], [], []
for k in range(N):
    Ti = np.linalg.inv(Tg[k]); y = pw @ Ti[:3, :3].T + Ti[:3, 3]; j = np.flatnonzero((y[:, 2] > .3) & (host != k)); ol.append(j); ok.append(np.full(len(j), k)); oy.append(y[j])
ol, ok, oy = np.concatenate(ol), np.concatenate(ok), np.concatenate(oy)
on_ = np.einsum('nji,nj->ni', Tg[host[ol]][:, :3, :3], nw[ol])          # the plane normals in each landmark's host frame
free = np.ones(N, int); free[0] = 0; E = np.zeros(0, int)
def geo(T, iters, pf=free, huber=1.345):
    q, t = state(T); d = window_opt.solve(q, t, pf, X, host, ol, ok, oy, on_, .01, iters, 4, huber, None); return q, t, d
q, t, d = geo(Tg, 5); check('truth is a fixed point', d['initial_cost'] < 1e-12 and errs(q, t, Tg)[0] < 1e-9, f"cost {d['initial_cost']:.2e}")
Tp = Tg.copy()
for i in range(1, N): Tp[i, :3, :3] = rot(rng.normal(0, .01, 3)) @ Tg[i, :3, :3]; Tp[i, :3, 3] += rng.normal(0, .01, 3)
e0 = errs(*state(Tp), Tg); q, t, d = geo(Tp, 30); e1 = errs(q, t, Tg)
check('pose-only recovers poses', e1[0] < 1e-5 and e1[1] < 1e-3, f"translation {100*e0[0]:.2f} -> {100*e1[0]:.1e} cm, rotation {e0[1]:.3f} -> {e1[1]:.1e} deg, solver {d['linear_solver']}")
fw = np.zeros(N, int); fw[4:] = 1; q0, t0 = state(Tp); q, t, d = geo(Tp, 5, pf=fw)
check('constant keyframes stay fixed', np.array_equal(q[:4], q0[:4]) and np.array_equal(t[:4], t0[:4]) and not np.array_equal(t[4:], t0[4:]), 'poses outside the window unchanged, window poses changed')
Xw = np.einsum('nij,nj->ni', Tp[host[ol]][:, :3, :3], X[ol]) + Tp[host[ol]][:, :3, 3]; Yw = np.einsum('nij,nj->ni', Tp[ok][:, :3, :3], oy) + Tp[ok][:, :3, 3]; Nw = np.einsum('nij,nj->ni', Tp[host[ol]][:, :3, :3], on_)
want = .5 * ((((Xw - Yw) * Nw).sum(1) / .01) ** 2).sum(); q, t, d = geo(Tp, 1, huber=1e9)
check('cost is the point-to-plane sum of squares', abs(d['initial_cost'] - want) < 1e-6 * want, f"ceres {d['initial_cost']:.6g}, by hand {want:.6g}")
# ---- photometric scene
fx, fy, cx, cy, H, W = 500., 500., 320., 240., 480, 640; vv, uu = np.mgrid[0:H, 0:W]; rxx, ryy = (uu - cx) / fx, (vv - cy) / fy
def tex(x, y): return .5 + .2 * np.sin(3 * x + 1) * np.cos(2.3 * y) + .1 * np.sin(5 * x + 2 * y + .5) + .05 * np.cos(9 * y - 4 * x)
M_ = 6; Tq = []
for i in range(M_):
    T = np.eye(4); T[:3, :3] = rot([.015 * i, .03 * i - .08, .01 * i]); T[:3, 3] = [.12 * i - .3, .05 * i - .1, .03 * i]; Tq.append(T)
Tq = np.array(Tq)
def render(T, gain=1., bias=0.):
    d = np.stack([rxx, ryy, np.ones_like(rxx)], -1) @ T[:3, :3].T; s = (3. - T[2, 3]) / d[..., 2]; return (gain * tex(T[0, 3] + s * d[..., 0], T[1, 3] + s * d[..., 1]) + bias).astype(np.float32)
def scene(gain, bias):
    imgs = np.stack([render(Tq[i], gain[i], bias[i]) for i in range(M_)]); h, Xh, cs = [], [], []
    for i in range(M_):
        v_, u_ = np.mgrid[20:H - 20:18, 20:W - 20:18]; ray = np.stack([(u_.ravel() - cx) / fx, (v_.ravel() - cy) / fy, np.ones(u_.size)], 1); dw = ray @ Tq[i][:3, :3].T; z = (3. - Tq[i][2, 3]) / dw[:, 2]; Xw_ = Tq[i][:3, 3] + z[:, None] * dw
        h.append(np.full(len(ray), i)); Xh.append(ray * z[:, None]); cs.append(gain[i] * tex(Xw_[:, 0], Xw_[:, 1]) + bias[i])
    h, Xh, cs = np.concatenate(h), np.concatenate(Xh), np.concatenate(cs); pl, pk, pp, pc, pairs = [], [], [], [], []
    for k in range(M_):
        for hh in range(M_):
            if hh == k: continue
            j = np.flatnonzero(h == hh); Xw_ = Xh[j] @ Tq[hh][:3, :3].T + Tq[hh][:3, 3]; Ti = np.linalg.inv(Tq[k]); x = Xw_ @ Ti[:3, :3].T + Ti[:3, 3]; u = fx * x[:, 0] / x[:, 2] + cx; v = fy * x[:, 1] / x[:, 2] + cy; o = (u > 15) & (u < W - 16) & (v > 15) & (v < H - 16)
            if o.sum() < 30: continue
            pl.append(j[o]); pk.append(np.full(o.sum(), k)); pp.append(np.full(o.sum(), len(pairs))); pc.append(cs[j[o]]); pairs.append((k, hh))
    return dict(h=h, X=Xh, imgs=imgs, lm=np.concatenate(pl), kf=np.concatenate(pk), pair=np.concatenate(pp), c=np.concatenate(pc), pairs=pairs)
free_ = np.ones(M_, int); free_[0] = 0
def photo(S, T0, iters=40):
    q, t = state(T0); ab = np.tile([1., 0.], (len(S['pairs']), 1)); P = dict(lm=S['lm'], kf=S['kf'], pair=S['pair'], c=S['c'], images=S['imgs'], kf_image=np.arange(M_), ab=ab, K=np.array([fx, fy, cx, cy]), sigma=.05, prior_sigma=10.)
    d = window_opt.solve(q, t, free_, S['X'], S['h'], E, E, np.zeros((0, 3)), np.zeros((0, 3)), .01, iters, 4, 1.345, P); return q, t, ab, d
Tn = Tq.copy()
for i in range(1, M_): Tn[i, :3, :3] = rot(rng.normal(0, .004, 3)) @ Tq[i, :3, :3]; Tn[i, :3, 3] += rng.normal(0, .004, 3)
S = scene(np.ones(M_), np.zeros(M_)); q, t, ab, d = photo(S, Tq, iters=5); e = errs(q, t, Tq)
check('photometric: truth is a fixed point', e[0] < 2e-4 and e[1] < 5e-3, f"{S['lm'].size} observations, {len(S['pairs'])} pairs, initial cost {d['initial_cost']:.2e}, moved {100*e[0]:.1e} cm {e[1]:.1e} deg")
e0 = errs(*state(Tn), Tq); q, t, ab, d = photo(S, Tn); e = errs(q, t, Tq)
check('photometric pose-only recovers poses on the Schur solver', e[0] < e0[0] / 20 and e[1] < e0[1] / 20 and d['linear_solver'].startswith('DENSE_SCHUR'), f"translation {100*e0[0]:.2f} -> {100*e[0]:.1e} cm, rotation {e0[1]:.3f} -> {e[1]:.1e} deg, solver {d['linear_solver']}")
ga, bi = 1 + .15 * np.sin(np.arange(M_)), .04 * np.cos(np.arange(M_)); Sg = scene(ga, bi); q, t, ab, d = photo(Sg, Tn); e = errs(q, t, Tq); ta = np.array([[ga[k] / ga[h], bi[k] - ga[k] / ga[h] * bi[h]] for k, h in Sg['pairs']])
check('photometric recovers poses and per-pair exposure', e[0] < e0[0] / 10 and np.abs(ab - ta).max() < .02, f"translation -> {100*e[0]:.1e} cm, exposure error {np.abs(ab - ta).max():.1e}")
sys.exit(0 if ok_all else 1)
