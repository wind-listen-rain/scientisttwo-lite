"""Consistency of virtual atoms with the fixed model (diagnostic).

usage: check_atoms.py <dataset|analytical> [n_trees]
For each tree: routing error of the exact tree evaluator on X_train (vs XGBoost), number of
kept virtual atoms, and whether each atom's target equals the tree output of a training
point in the same full joint cell (the tree is constant on a joint cell, so a mismatch means
a copy binned inconsistently with the model's routing).
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/E1/lib")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/E1/dev")
import agtloco  # noqa: E402
from common import analytical, real  # noqa: E402

agtloco.KEEP_PATH = True
name = sys.argv[1]
nt = int(sys.argv[2]) if len(sys.argv) > 2 else 20
model, Xtr, Xte = analytical(0) if name == "analytical" else real(name)
ens = agtloco.GTLocoHFD(model)
X = np.asarray(Xtr, dtype=float)
Ttr = ens._tree_predict(X)
X32 = agtloco._f32(X)
union = agtloco._union_splits(ens.xgb_table)
anchor = {"X32": X32, "shifts": agtloco._shift_values(X, X32, union), "n_shift": 2 * len(union)}
print(f"n={len(X)} |S|={len(union)}")
tot_bad, tot_cmp, tot_v = 0, 0, 0
orig_fit = agtloco.GTLocoTree._virtual_atoms
store = {}


def spy(self, *a, **k):
    out = orig_fit(self, *a, **k)
    store["va"] = out
    return out


agtloco.GTLocoTree._virtual_atoms = spy
for t in range(nt):
    table = pd.DataFrame(ens.xgb_table[ens.xgb_table["Tree"] == t])
    tree = agtloco.GTLocoTree(table, 2, ens.max_depth)
    store.clear()
    tree.fit(X, Ttr[:, t], anchor)
    va = store.get("va")
    if va is None or va["nv"] == 0:
        print(f"tree {t}: no virtual atoms")
        continue
    lb = np.column_stack([np.digitize(X[:, j], tree.split_list[k])
                          for k, j in enumerate(tree.main_variables)])
    V = lb.shape[1]
    Hv = va["Hv"]
    Cv = Hv.indices.reshape(va["nu"], -1)[:, :V] - tree.off_main[:-1]
    y = Ttr[:, t] - tree.eta0
    key_r = {tuple(r): y[i] for i, r in enumerate(lb)}
    bad, cmp_ = 0, 0
    for r, yv in zip(Cv, va["yv"], strict=True):
        k = tuple(r)
        if k in key_r:
            cmp_ += 1
            bad += abs(key_r[k] - yv) > 1e-5
    sizes = [1 + int(np.sum(r >= 0)) for mem, vp, _ in va["blocks"] for r in vp]
    print(f"tree {t}: route_err={tree.route_err:.2e} nv={va["nv"]} nu={va["nu"]} merge_err={tree.merge_err:.1e} ({va['nv'] / len(X):.2f}n) "
          f"same-cell-as-real={cmp_} mismatched={bad} block size mean={np.mean(sizes):.2f} "
          f"max={max(sizes)}")
    tot_bad += bad
    tot_cmp += cmp_
    tot_v += va["nv"]
print(f"TOTAL virtual={tot_v} compared={tot_cmp} mismatched={tot_bad}")
