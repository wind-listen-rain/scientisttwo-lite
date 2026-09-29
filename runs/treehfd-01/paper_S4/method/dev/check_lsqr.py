"""Baseline TreeHFD with default vs tight lsqr tolerances: does lsqr converge? (diagnostic)"""
import sys

import numpy as np
from scipy.sparse.linalg import lsqr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/S4/lib")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/S4/dev")
import treehfd_mod.tree as tmod  # noqa: E402
from common import analytical, real  # noqa: E402
from treehfd_mod import XGBTreeHFD  # noqa: E402


def ortho(pred, main, inter, il):
    v = np.var(pred)
    worst = None
    for c, (j, k) in enumerate(il):
        if np.var(inter[:, c]) < 0.01 * v:
            continue
        for mm in (j, k):
            if np.var(main[:, mm]) > 0:
                r = abs(np.corrcoef(inter[:, c], main[:, mm])[0, 1])
                worst = r if worst is None else max(worst, r)
    return worst


info = []


def lsqr_tight(A, b):
    res = lsqr(A, b, atol=1e-14, btol=1e-14, iter_lim=100000)
    info.append((res[1], res[2]))
    return res


def lsqr_default(A, b):
    res = lsqr(A, b)
    info.append((res[1], res[2]))
    return res


name = sys.argv[1]
model, Xtr, Xte = analytical(0) if name == "analytical" else real(name)
for label, fn in (("default", lsqr_default), ("tight", lsqr_tight)):
    info.clear()
    tmod.lsqr = fn
    h = XGBTreeHFD(model)
    h.fit(Xtr, verbose=False)
    main, inter = h.predict(Xtr, verbose=False)
    p = model.predict(Xtr)
    rin = np.mean((p - h.eta0 - main.sum(1) - inter.sum(1)) ** 2) / np.var(p)
    istop = np.array([i[0] for i in info])
    iters = np.array([i[1] for i in info])
    print(f"{label}: resid_in={rin:.5f} ortho_in={ortho(p, main, inter, h.interaction_list):.4f} "
          f"istop counts={np.bincount(istop)} iters median={np.median(iters)} max={iters.max()}")
