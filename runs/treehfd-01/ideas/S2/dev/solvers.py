"""Dev: compare least-squares solvers on real per-tree systems."""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator, lsmr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from smoothed_hfd import SmoothedTreeHFD, _build_system, _f32, _prepare_tree, _shift_values  # noqa: E402

from setup import analytical, real  # noqa: E402

name, alpha = sys.argv[1], float(sys.argv[2])
ntree = int(sys.argv[3]) if len(sys.argv) > 3 else 10
model, X, Xe = analytical(0) if name == "analytical" else real(name)
s = SmoothedTreeHFD(model, alpha=alpha)
X = _f32(X)
shifts = _shift_values(X, s.union_splits)
tol = 1e-10
agg = {}


def rec(k, t, itn, x, xref):
    e = np.linalg.norm(x - xref) / max(np.linalg.norm(xref), 1e-300)
    obj = np.linalg.norm(A @ x - b) ** 2
    obj_ref = np.linalg.norm(A @ xref - b) ** 2
    ng = np.linalg.norm(A.T @ (A @ x - b)) / (np.linalg.norm(A.T @ b))
    a = agg.setdefault(k, [0.0, 0, 0.0, 0.0, 0.0])
    a[0] += t
    a[1] += itn
    a[2] = max(a[2], e)
    a[3] = max(a[3], (obj - obj_ref) / obj_ref)
    a[4] = max(a[4], ng)


for t in range(0, len(s.trees), max(1, len(s.trees) // ntree)):
    P = _prepare_tree(s.trees[t], s._pairs(t, 2), s.tree_vars[t], X, shifts, s.num_shift)
    if P.const is not None:
        continue
    A, b, _ = _build_system(P, alpha)
    ncol = A.shape[1]
    # Reference: dense QR-based lstsq on the column-scaled system.
    t0 = time.time()
    d = 1.0 / np.sqrt(np.asarray(A.multiply(A).sum(0)).ravel())
    As = A @ __import__("scipy.sparse", fromlist=["diags"]).diags(d)
    t_scale = time.time() - t0
    t0 = time.time()
    G = (As.T @ As).toarray()
    t_gram = time.time() - t0
    t0 = time.time()
    R = sla.cholesky(G + 1e-10 * np.eye(ncol), lower=False)
    t_chol = time.time() - t0
    ev = np.linalg.eigvalsh(G)
    nullity = int(np.sum(ev < 1e-9 * ev.max()))
    xref = None

    t0 = time.time()
    r = lsmr(A, b, atol=tol, btol=tol, maxiter=max(2000, 20 * ncol))
    x_plain = r[0]
    if xref is None:
        xref = x_plain
    rec("plain", time.time() - t0, r[2], r[0], xref)

    t0 = time.time()
    r = lsmr(As, b, atol=tol, btol=tol, maxiter=max(2000, 20 * ncol))
    rec("colscale", time.time() - t0 + t_scale, r[2], d * r[0], xref)

    t0 = time.time()
    op = LinearOperator(A.shape, matvec=lambda z: As @ sla.solve_triangular(R, z),
                        rmatvec=lambda u: sla.solve_triangular(R, As.T @ u, trans="T"),
                        dtype=float)
    r = lsmr(op, b, atol=tol, btol=tol, maxiter=max(2000, 20 * ncol))
    x = d * sla.solve_triangular(R, r[0])
    rec("chol_prec", time.time() - t0 + t_scale + t_gram + t_chol, r[2], x, xref)

    t0 = time.time()
    G0 = (A.T @ A).toarray()
    lam = 1e-11 * G0.diagonal().max()
    R0 = sla.cholesky(G0 + lam * np.eye(ncol), lower=False)
    op0 = LinearOperator(A.shape, matvec=lambda z: A @ sla.solve_triangular(R0, z),
                         rmatvec=lambda u: sla.solve_triangular(R0, A.T @ u, trans="T"),
                         dtype=float)
    r = lsmr(op0, b, atol=tol, btol=tol, maxiter=max(2000, 20 * ncol))
    x = sla.solve_triangular(R0, r[0])
    rec("chol_unscaled", time.time() - t0, r[2], x, xref)

    t0 = time.time()
    x = sla.cho_solve((R, False), As.T @ b)
    for _ in range(2):
        x = x + sla.cho_solve((R, False), As.T @ (b - As @ x))
    rec("seminormal", time.time() - t0 + t_scale + t_gram + t_chol, 0, d * x, xref)
    print(f"tree {t}: shape {A.shape} nnz {A.nnz} gram {t_gram:.3f}s chol {t_chol:.3f}s "
          f"nullity {nullity}", flush=True)

for k, (tt, it, e, o, g) in agg.items():
    print(f"{k:>11}: time {tt:.2f}s iters {it} max rel diff vs plain {e:.2e} obj excess {o:.2e} rel normal-eq resid {g:.2e}")
