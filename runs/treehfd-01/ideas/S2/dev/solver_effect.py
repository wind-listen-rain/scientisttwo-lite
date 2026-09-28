"""Dev: isolate solver vs fallback effects at alpha = 0 (and alpha > 0)."""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import lsqr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/bench")
import smoothed_hfd as sh  # noqa: E402
from harness import orthogonality  # noqa: E402

from setup import real  # noqa: E402

name = sys.argv[1]
alphas = [float(a) for a in sys.argv[2].split(",")]
model, X, Xe = real(name)
pt, pe = model.predict(X), model.predict(Xe)
exact = sh._solve_ls
log = []


def lsqr_default(A, b, stats):
    r = lsqr(A, b)
    log.append((int(r[1]), int(r[2]), A.shape[1]))
    return r[0]


for solver in ("exact", "lsqr_default"):
    sh._solve_ls = exact if solver == "exact" else lsqr_default
    for a in alphas:
        log.clear()
        h = sh.SmoothedTreeHFD(model, alpha=a).fit(X)
        mt, it = h.predict(X)
        me, ie = h.predict(Xe)
        rin = np.mean((pt - h.eta0 - mt.sum(1) - it.sum(1)) ** 2) / np.var(pt)
        rout = np.mean((pe - h.eta0 - me.sum(1) - ie.sum(1)) ** 2) / np.var(pe)
        oi = orthogonality(pt, mt, it, h.interaction_list)
        oo = orthogonality(pe, me, ie, h.interaction_list)
        msg = f"{solver:>13} alpha={a}: resid_in={rin:.5f} resid_out={rout:.5f} ortho_in={oi:.4f} ortho_out={oo:.4f}"
        if log:
            ist = np.array([l[0] for l in log])
            msg += f"  lsqr istop counts={dict(zip(*np.unique(ist, return_counts=True)))} itn/ncol mean={np.mean([l[1] / l[2] for l in log]):.2f}"
        print(msg, flush=True)
