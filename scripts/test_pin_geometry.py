"""Checks of the fixed-mean step (mesh_splatting/scene/pin_geometry.py)."""
import sys, os, math
sys.path.insert(0, os.getcwd())
import torch
from mesh_splatting.scene.pin_geometry import q_matrix, to_q, from_q

torch.manual_seed(0)
T, V = 200, 607                                                          # 7 vertices belong to no triangle
perm = torch.randperm(V); F = perm[:3 * T].view(T, 3)                    # each vertex in at most one triangle
X0 = torch.randn(V, 3, dtype=torch.float64); target = X0 + 0.3 * torch.randn(V, 3, dtype=torch.float64)
A = torch.rand(V, 3, dtype=torch.float64) * 5 + 0.2                      # anisotropic curvature
Q = q_matrix("cpu", torch.float64); C0 = X0[F].mean(1)
loss_fn = lambda x: 0.5 * (A[F] * (x[F] - target[F]) ** 2).sum() + 0.1 * (x[F].std(1) ** 2).sum()
make = lambda p, opt: torch.optim.Adam([p], lr=0.01, eps=1e-15) if opt == "adam" else torch.optim.SGD([p], lr=0.01)


def run(mode, opt="adam", steps=400):
    x = torch.nn.Parameter(X0.clone()); o = make(x, opt)
    for _ in range(steps):
        o.zero_grad(); loss_fn(x).backward()
        with torch.no_grad():
            if mode == "intrinsic":
                Gq = to_q(x.grad[F], Q); Gq[:, 0] = 0.0; x.grad[F] = Gq; x.data[F] = to_q(x.data[F], Q)
        o.step()
        with torch.no_grad():
            if mode == "intrinsic":
                x.data[F] = from_q(x.data[F], Q)
            x.data[F] += (C0 - x.data[F].mean(1)).unsqueeze(1)
    return x.data.clone()


def run_reparam(opt="adam", steps=400):
    """Reference: x = mu 1 + Q e."""
    B = Q[:, 1:]; e = torch.nn.Parameter(torch.einsum("kr,tkd->trd", B, X0[F] - C0.unsqueeze(1))); o = make(e, opt)
    build = lambda: X0.clone().index_put((F,), C0.unsqueeze(1) + torch.einsum("kr,trd->tkd", B, e))
    for _ in range(steps):
        o.zero_grad(); loss_fn(build()).backward(); o.step()
    return build().detach()


ok = True
def check(name, cond, msg):
    global ok; ok &= bool(cond); print("%-4s %-50s %s" % ("ok" if cond else "FAIL", name, msg))

check("Q orthonormal, first column the mean direction", torch.allclose(Q.T @ Q, torch.eye(3, dtype=torch.float64)) and torch.allclose(Q[:, 0], torch.full((3,), 1 / math.sqrt(3), dtype=torch.float64)), "")
xi, xr = run("intrinsic"), run("recentre")
for name, x in (("intrinsic", xi), ("recentre", xr)):
    check("%s: mean stays fixed" % name, (x[F].mean(1) - C0).abs().max() < 1e-12, "max drift %.1e" % (x[F].mean(1) - C0).abs().max())
    check("%s: other vertices untouched" % name, torch.equal(x[perm[3 * T:]], X0[perm[3 * T:]]), "")
d = (xi - run_reparam()).abs().max().item(); check("Adam: intrinsic == explicit x = mu 1 + Q e", d < 1e-9, "max |dx| %.1e" % d)
d = (run("intrinsic", "sgd") - run_reparam("sgd")).abs().max().item(); check("SGD: intrinsic == explicit", d < 1e-9, "max |dx| %.1e" % d)
d = (run("recentre", "sgd") - run("intrinsic", "sgd")).abs().max().item(); check("SGD: recentre == intrinsic", d < 1e-9, "max |dx| %.1e" % d)
check("Adam: recentre != intrinsic", (xi - xr).abs().max() > 1e-3, "max |dx| %.1e" % (xi - xr).abs().max())
print("\nALL OK" if ok else "\nSOME CHECKS FAILED"); sys.exit(0 if ok else 1)
