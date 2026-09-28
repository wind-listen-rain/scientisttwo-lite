"""Closed-form (block Woodbury) leave-out residuals vs brute-force refits for one tree.

usage: check_loo.py <dataset> <tree index> [orphans|plain] [npick=12]
Extends S4/dev/check_loo.py to the anchored path. The brute-force refit removes point i and
rebuilds everything from X_{-i}: the partition, the union-grid medians and hence all virtual
atoms (so copies that depended on x_i disappear or move, as in an honest leave-out).
'orphans': singletons owning a count-1 pair cell or main bin; 'plain': the other singletons.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/E1/lib")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/E1/dev")
import agtloco  # noqa: E402
from common import real  # noqa: E402

agtloco.KEEP_PATH = True
name, t = sys.argv[1], int(sys.argv[2])
kind = sys.argv[3] if len(sys.argv) > 3 else "orphans"
npick = int(sys.argv[4]) if len(sys.argv) > 4 else 12
model, Xtr, _ = real(name)
ens = agtloco.GTLocoHFD(model)
X = np.asarray(Xtr, dtype=float)
Ttr = ens._tree_predict(X)
union = agtloco._union_splits(ens.xgb_table)


def anchor_of(Xs):
    X32 = agtloco._f32(Xs)
    return {"X32": X32, "shifts": agtloco._shift_values(Xs, X32, union),
            "n_shift": 2 * len(union)}


table = pd.DataFrame(ens.xgb_table[ens.xgb_table["Tree"] == t])
tree = agtloco.GTLocoTree(table, 2, ens.max_depth)
loo, _, risk, n1f, m, nv = tree.fit(X, Ttr[:, t], anchor_of(X))
VAR = agtloco.variants()
print(f"{name} tree {t}: m={m} N1/n={n1f:.3f} virtual={nv}")
lb = np.column_stack([np.digitize(X[:, j], tree.split_list[k])
                      for k, j in enumerate(tree.main_variables)])
_, inv, cnt = np.unique(lb, axis=0, return_inverse=True, return_counts=True)
single = np.flatnonzero(cnt[inv.ravel()] == 1)
orph = []
for i in single:
    own = False
    for p in tree.pairs:
        code = lb[i, p["ia"]] * tree.nb[p["ib"]] + lb[i, p["ib"]]
        own |= bool(p["cnt"][np.searchsorted(p["codes"], code)] == 1)
    for k in range(lb.shape[1]):
        own |= bool(np.sum(lb[:, k] == lb[i, k]) == 1)
    orph.append(own)
orph = np.array(orph, dtype=bool)
single = single[orph] if kind == "orphans" else single[~orph]
print(f"checked singletons ({kind}): {len(single)}")
rng = np.random.default_rng(0)
pick = rng.choice(single, size=min(npick, len(single)), replace=False)
n = len(X)
fits = {}
for i in pick:
    keep = np.ones(n, bool)
    keep[i] = False
    tr = agtloco.GTLocoTree(table, 2, ens.max_depth)
    tr.fit(X[keep], Ttr[keep, t], anchor_of(X[keep]))
    fits[i] = tr
for ri, rho in enumerate(agtloco.RHOS):
    for v, (pr, ru) in enumerate(VAR):
        for g in (0, 2, 4):
            errs = []
            for i in pick:
                tr = fits[i]
                tr.finalize(g, v, ri)
                mm, oo = tr.predict(X[i:i + 1])
                pred = tr.eta0 + mm.sum() + oo.sum()
                # the closed form uses the full-sample intercept; compare on that scale
                errs.append((Ttr[i, t] - pred - (tree.eta0 - tr.eta0), loo[v, ri, g, i]))
            errs = np.array(errs)
            b, c = np.mean(errs[:, 0] ** 2), np.mean(errs[:, 1] ** 2)
            print(f"rho={rho:<4g} {agtloco.PRIORS[pr]:7s} {agtloco.RULES[ru]:8s} "
                  f"kappa={agtloco.KAPPAS[g]:<5g} brute msq={b:.3e} closed msq={c:.3e} "
                  f"ratio={c / b:.3f} max|diff|={np.max(np.abs(errs[:, 0] - errs[:, 1])):.3e}")
