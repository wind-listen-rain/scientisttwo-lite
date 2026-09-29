"""Diagnose where baseline out-of-sample residual comes from (per tree: seen/unseen full cells, unseen pair cells)."""
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import analytical, real  # noqa: E402
from treehfd_mod import XGBTreeHFD  # noqa: E402

name = sys.argv[1]
model, Xtr, Xte = analytical(0) if name == "analytical" else real(name)
hfd = XGBTreeHFD(model)
t0 = time.time()
hfd.fit(Xtr, verbose=False)
print("fit", time.time() - t0)
Ttr = hfd._tree_predict(Xtr)
Tte = hfd._tree_predict(Xte)
n = len(Xtr)
E_tr = np.zeros(n)
E_te = np.zeros(len(Xte))
E_te_unseen_full = np.zeros(len(Xte))
E_te_unseen_pair = np.zeros(len(Xte))
stats = []
for t, tree in enumerate(hfd.treehfd_list):
    cp = tree.cartesian_partition
    if cp.main_variables.size == 0:
        continue
    mtr, itr = tree.predict(Xtr)
    mte, ite = tree.predict(Xte)
    rtr = Ttr[:, t] - tree.eta0 - mtr.sum(1) - ite[:0].sum() - itr.sum(1)
    rte = Tte[:, t] - tree.eta0 - mte.sum(1) - ite.sum(1)
    E_tr += rtr
    E_te += rte
    # full-cell keys
    V = len(cp.main_variables)
    btr = np.stack([np.digitize(Xtr[:, j], cp.split_list[i]) for i, j in enumerate(cp.main_variables)], 1)
    bte = np.stack([np.digitize(Xte[:, j], cp.split_list[i]) for i, j in enumerate(cp.main_variables)], 1)
    keys_tr, cnt = np.unique(btr, axis=0, return_counts=True)
    N1 = int(np.sum(cnt == 1))
    seen = {tuple(k) for k in keys_tr}
    unseen_full = np.array([tuple(k) not in seen for k in bte])
    # unseen pair cells
    up = np.zeros(len(Xte), bool)
    mv = list(cp.main_variables)
    for k, (a, b) in enumerate(tree.interaction_list):
        ia, ib = mv.index(a), mv.index(b)
        s = {tuple(c) for c in btr[:, [ia, ib]]}
        up |= np.array([tuple(c) not in s for c in bte[:, [ia, ib]]])
    E_te_unseen_full += np.where(unseen_full, rte, 0)
    E_te_unseen_pair += np.where(up, rte, 0)
    m = int(cp.partition_index[-1])
    stats.append((t, V, len(tree.interaction_list), m, len(keys_tr), N1 / n, unseen_full.mean(), up.mean(),
                  np.mean(rtr ** 2), np.mean(rte[~unseen_full] ** 2) if (~unseen_full).any() else 0,
                  np.mean(rte[unseen_full] ** 2) if unseen_full.any() else 0))
S = np.array(stats)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
print("cols: t V I m ncells N1/n te_unseen_full te_unseen_pair mse_tr mse_te_seen mse_te_unseen")
print(S[:10])
print("mean over trees:", S.mean(0))
vtr = np.var(Ttr.sum(1))
vte = np.var(Tte.sum(1))
print("resid_in", np.mean(E_tr ** 2) / vtr, "resid_out", np.mean(E_te ** 2) / vte)
print("sum per-tree mse_tr / var", np.sum(S[:, 8]) / vtr, "te seen/unseen avg per-tree", S[:, 9].sum() / vte, S[:, 10].sum() / vte)
print("resid_out contribution from unseen-full parts", np.mean(E_te_unseen_full ** 2) / vte,
      "unseen-pair parts", np.mean(E_te_unseen_pair ** 2) / vte)
