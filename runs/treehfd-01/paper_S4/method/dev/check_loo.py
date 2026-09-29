"""Check closed-form leave-out residuals against brute-force refits for one tree (diagnostic).

usage: check_loo.py <dataset> <tree index> [orphans]
With 'orphans', only singletons owning a count-1 pair cell or main bin are checked (they use
the unseen-cell rules in the leave-out).
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/S4/lib")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/S4/dev")
import gtloco  # noqa: E402
from common import real  # noqa: E402

gtloco.KEEP_PATH = True
name, t = sys.argv[1], int(sys.argv[2])
only_orphans = "orphans" in sys.argv[3:]
model, Xtr, Xte = real(name)
ens = gtloco.GTLocoHFD(model)
Ttr = ens._tree_predict(Xtr)
table = pd.DataFrame(ens.xgb_table[ens.xgb_table["Tree"] == t])
tree = gtloco.GTLocoTree(table, 2, ens.max_depth)
loo, _, risk, n1f, m = tree.fit(Xtr, Ttr[:, t])
VAR = gtloco.variants()
for v, (pr, ru) in enumerate(VAR):
    print(f"risk {gtloco.PRIORS[pr]:7s} {gtloco.RULES[ru]:8s}", np.round(risk[v], 5))
lb = np.column_stack([np.digitize(Xtr[:, j], tree.split_list[k])
                      for k, j in enumerate(tree.main_variables)])
_, inv, cnt = np.unique(lb, axis=0, return_inverse=True, return_counts=True)
single = np.flatnonzero(cnt[inv.ravel()] == 1)
if only_orphans:
    orph = []
    for i in single:
        own = False
        for p in tree.pairs:
            code = lb[i, p["ia"]] * tree.nb[p["ib"]] + lb[i, p["ib"]]
            own |= bool(p["cnt"][np.searchsorted(p["codes"], code)] == 1)
        orph.append(own)
    single = single[np.array(orph, dtype=bool)]
print("checked singletons:", len(single))
rng = np.random.default_rng(0)
pick = rng.choice(single, size=min(12, len(single)), replace=False)
n = len(Xtr)
for v, (pr, ru) in enumerate(VAR):
    for g in (0, 2, 4):
        errs = []
        for i in pick:
            keep = np.ones(n, bool)
            keep[i] = False
            tr = gtloco.GTLocoTree(table, 2, ens.max_depth)
            tr.fit(Xtr[keep], Ttr[keep, t])
            tr.finalize(g, v)
            mm, oo = tr.predict(Xtr[i:i + 1])
            pred = tr.eta0 + mm.sum() + oo.sum()
            # closed form uses the full-sample intercept; compare on that scale
            errs.append((Ttr[i, t] - pred - (tree.eta0 - tr.eta0), loo[v, g, i]))
        errs = np.array(errs)
        print(f"{gtloco.PRIORS[pr]:7s} {gtloco.RULES[ru]:8s} kappa={gtloco.KAPPAS[g]:<5g} "
              f"brute msq={np.mean(errs[:, 0] ** 2):.3e} closed msq={np.mean(errs[:, 1] ** 2):.3e} "
              f"max|diff|={np.max(np.abs(errs[:, 0] - errs[:, 1])):.3e}")
