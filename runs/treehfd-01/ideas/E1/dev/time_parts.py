"""Timing breakdown of an anchored GT-LOCO fit (diagnostic).

usage: time_parts.py <dataset|analytical> [n_trees]
Reports, summed over trees: eigendecompositions, rho = 0 path, anchored path, virtual-atom
construction, and the distribution of leave-out block sizes (incl. the real row).
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/E1/lib")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/E1/dev")
import agtloco  # noqa: E402
from common import analytical, real  # noqa: E402

name = sys.argv[1]
nt = int(sys.argv[2]) if len(sys.argv) > 2 else 100
model, Xtr, _ = analytical(0) if name == "analytical" else real(name)
tim = {"eigh": 0.0, "path": 0.0, "path_anchor": 0.0, "virtual": 0.0, "tree_fit": 0.0}
sizes = []
T = agtloco.GTLocoTree
orig = {k: getattr(T, "_" + k) for k in ("path", "path_anchor", "virtual_atoms")}
orig_eigh = np.linalg.eigh


def timed(key, f):
    def g(*a, **k):
        t0 = time.perf_counter()
        out = f(*a, **k)
        tim[key] += time.perf_counter() - t0
        return out
    return g


def va_spy(self, *a, **k):
    t0 = time.perf_counter()
    out = orig["virtual_atoms"](self, *a, **k)
    tim["virtual"] += time.perf_counter() - t0
    if out.get("nv"):
        for mem, vp, _ in out["blocks"]:
            sizes.extend([1 + int(np.sum(r >= 0)) for r in vp])
    return out


T._path = timed("path", orig["path"])
T._path_anchor = timed("path_anchor", orig["path_anchor"])
T._virtual_atoms = va_spy
np.linalg.eigh = timed("eigh", orig_eigh)
ens = agtloco.GTLocoHFD(model)
X = np.asarray(Xtr, dtype=float)
Ttr = ens._tree_predict(X)
X32 = agtloco._f32(X)
union = agtloco._union_splits(ens.xgb_table)
anchor = {"X32": X32, "shifts": agtloco._shift_values(X, X32, union), "n_shift": 2 * len(union)}
n1 = []
for t in range(nt):
    table = pd.DataFrame(ens.xgb_table[ens.xgb_table["Tree"] == t])
    tree = T(table, 2, ens.max_depth)
    t0 = time.perf_counter()
    _, _, _, n1f, m, nv = tree.fit(X, Ttr[:, t], anchor)
    tim["tree_fit"] += time.perf_counter() - t0
    n1.append(n1f)
print(f"{name}: n={len(X)} trees={nt} N1/n mean={np.mean(n1):.3f}")
print({k: round(v, 2) for k, v in tim.items()})
s = np.array(sizes)
if len(s):
    print("block sizes: mean", round(s.mean(), 2), "quantiles", np.quantile(s, [.5, .9, .99, 1]),
          "sum b^2", int(np.sum(s ** 2)), "count", len(s))
