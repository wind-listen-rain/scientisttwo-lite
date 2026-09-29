"""R1: closed-form leave-one-cell-out (PRESS) residuals vs brute-force refits, per tree and for the full ensemble.

usage: check_bruteforce.py <dataset>      ->  raw/bruteforce__<dataset>.json
For a sample of training points i (10 uniformly random points + 6 points that are Good-Turing singletons of
tree 0) and every tree t, the tree is refitted WITHOUT point i (same tree structure, the partition and the
unseen-cell rules recomputed on the n-1 remaining points) and the held-out residual of point i is computed
for four configurations (prior/rule variant, common kappa index g):  (ridge,zero), (lattice,harmonic) x g in {2,4,6}.
Closed form = the leave-out residual used by the method (GTLocoTree.fit output).
Errors reported: relative RMS error sqrt(mean((closed-brute)^2)/mean(brute^2)) and relative error of the mean
square (the risk R). Ensemble level: residual summed over the 100 trees (this is what the ensemble step uses).
"""
import sys

import numpy as np
import pandas as pd

import gtloco
from common import SUPP, dump, real, set_config

ds = sys.argv[1]
set_config()
model, Xtr, _ = real(ds)
n = len(Xtr)
ens = gtloco.GTLocoHFD(model)
Ttr = ens._tree_predict(Xtr)
NT = ens.n_estimators
V = gtloco.variants()                       # (ridge,zero), (ridge,harmonic), (lattice,zero), (lattice,harmonic)
CONF = [(0, g) for g in (2, 4, 6)] + [(3, g) for g in (2, 4, 6)]
rng = np.random.default_rng(0)
tables = [pd.DataFrame(ens.xgb_table[ens.xgb_table["Tree"] == t]) for t in range(NT)]

# points: 6 singletons of tree 0 + 10 random points
tr0 = gtloco.GTLocoTree(tables[0], 2, ens.max_depth)
tr0.fit(Xtr, Ttr[:, 0])
lb = np.column_stack([np.digitize(Xtr[:, j], tr0.split_list[k]) for k, j in enumerate(tr0.main_variables)])
_, inv, cnt = np.unique(lb, axis=0, return_inverse=True, return_counts=True)
single0 = np.flatnonzero(cnt[inv.ravel()] == 1)
pts = np.unique(np.concatenate([rng.choice(single0, size=min(6, len(single0)), replace=False),
                                rng.choice(n, size=10, replace=False)]))
print(ds, "points", len(pts), flush=True)

closed = np.zeros((NT, len(CONF), len(pts)))
brute = np.zeros_like(closed)
is_single = np.zeros((NT, len(pts)), bool)
for t in range(NT):
    tree = gtloco.GTLocoTree(tables[t], 2, ens.max_depth)
    loo, _, _, _, _ = tree.fit(Xtr, Ttr[:, t])
    for c, (v, g) in enumerate(CONF):
        closed[t, c] = loo[v, g, pts]
    if tree.main_variables.size:
        lbt = np.column_stack([np.digitize(Xtr[:, j], tree.split_list[k]) for k, j in enumerate(tree.main_variables)])
        _, inv, cnt = np.unique(lbt, axis=0, return_inverse=True, return_counts=True)
        is_single[t] = cnt[inv.ravel()][pts] == 1
    for a, i in enumerate(pts):
        keep = np.ones(n, bool)
        keep[i] = False
        tr = gtloco.GTLocoTree(tables[t], 2, ens.max_depth)
        tr.fit(Xtr[keep], Ttr[keep, t])
        for c, (v, g) in enumerate(CONF):
            if tr.main_variables.size == 0:
                brute[t, c, a] = closed[t, c, a]     # constant tree: identical by construction
                continue
            tr.finalize(g, v)
            m, o = tr.predict(Xtr[i:i + 1])
            brute[t, c, a] = Ttr[i, t] - tree.eta0 - m.sum() - o.sum()
    if t % 10 == 0:
        print("tree", t, flush=True)


def stats(c, b):
    return {"rel_rms": float(np.sqrt(np.mean((c - b) ** 2) / np.mean(b ** 2))),
            "rel_err_risk": float(np.mean(c ** 2) / np.mean(b ** 2) - 1),
            "max_abs_diff": float(np.max(np.abs(c - b))), "n": int(c.size)}


res = {"dataset": ds, "n_train": n, "points": pts.tolist(), "configs": CONF, "trees": {}, "ensemble": {}}
for c, (v, g) in enumerate(CONF):
    name = f"{gtloco.PRIORS[V[v][0]]}/{gtloco.RULES[V[v][1]]} kappa={gtloco.KAPPAS[g]:g}"
    e = {"all_trees_all_points": stats(closed[:, c], brute[:, c]),
         "all_trees_singleton_cells": stats(closed[:, c][is_single], brute[:, c][is_single]) if is_single.any() else None,
         "all_trees_repeated_cells": stats(closed[:, c][~is_single], brute[:, c][~is_single]),
         "tree0": stats(closed[0, c], brute[0, c]), "tree50": stats(closed[50, c], brute[50, c]),
         "tree99": stats(closed[-1, c], brute[-1, c]),
         "ensemble_sum_over_trees": stats(closed[:, c].sum(0), brute[:, c].sum(0))}
    res["trees"][name] = e
dump(res, SUPP / f"raw/bruteforce__{ds}.json")
print("done", ds)
