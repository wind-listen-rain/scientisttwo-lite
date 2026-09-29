"""Round 3 diagnostic: where do the kept virtual copies land?

usage: diag_copies.py <dataset> [split=<s>] [trees=0,10,20,...]
For each tree (after the E1 fit), recomputes the kept copies exactly as
GTLocoTree._virtual_atoms does and reports, weighted by copy count:
  * pop_joint: share landing in a joint cell of tree t that holds training points (the copy's
    tree output then equals those points' outputs: it only re-weights P_n);
  * thin_main: share moving x_j into a main bin with <= 2 training points;
  * virtual mass relative to the real mass at rho = 1 (sum of weights / 1).
X_train and the model only; no held-out data.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "runs/treehfd-01/ideas/E1/lib"))
sys.path.insert(0, str(ROOT / "runs/treehfd-01/ideas/E1/dev"))
import agtloco  # noqa: E402
from common import real  # noqa: E402

opts = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
agtloco.KEEP_PATH = True
agtloco.RHOS = (0.0, 0.25)
model, Xtr, _ = real(sys.argv[1], int(opts.get("split", 0)))
X = np.asarray(Xtr, dtype=float)
n = len(X)
hfd = agtloco.GTLocoHFD(model)
hfd.fit(X)
X32 = agtloco._f32(X)
union = agtloco._union_splits(hfd.xgb_table)
shifts = agtloco._shift_values(X, X32, union)
n_shift = 2 * len(union)
trees = [int(t) for t in opts.get("trees", ",".join(map(str, range(0, 100, 10)))).split(",")]
tot = {"kept": 0, "pop": 0, "thin": 0}
for t in trees:
    tree = hfd.treehfd_list[t]
    if tree.main_variables.size == 0:
        continue
    lb = np.column_stack([np.digitize(X[:, j], bins=tree.split_list[k], right=False)
                          for k, j in enumerate(tree.main_variables)])
    joint = {tuple(r) for r in lb.tolist()}
    cnt_main = [np.bincount(lb[:, k], minlength=tree.nb[k]) for k in range(lb.shape[1])]
    kept = pop = thin = 0
    for vi, j in enumerate(tree.main_variables):
        for vals, _ in shifts.get(int(j), []):
            valid = np.flatnonzero(~np.isnan(vals))
            nbin = np.digitize(vals[valid], tree.split_list[vi], right=False)
            moved = nbin != lb[valid, vi]
            rows = valid[moved]
            if rows.size == 0:
                continue
            LBc = lb[rows].copy()
            LBc[:, vi] = nbin[moved]
            ok = np.ones(rows.size, dtype=bool)
            for p in tree.pairs:
                code = LBc[:, p["ia"]] * tree.nb[p["ib"]] + LBc[:, p["ib"]]
                pos = np.minimum(np.searchsorted(p["codes"], code), len(p["codes"]) - 1)
                ok &= p["codes"][pos] == code
            LBc = LBc[ok]
            kept += len(LBc)
            pop += sum(tuple(r) in joint for r in LBc.tolist())
            thin += int(np.sum(cnt_main[vi][LBc[:, vi]] <= 2))
    tot["kept"] += kept
    tot["pop"] += pop
    tot["thin"] += thin
    print(f"tree {t:3d}: V={len(tree.main_variables)} pairs={len(tree.pairs)} kept/n="
          f"{kept / n:.3f} mass(rho=1)={kept / (n_shift * n):.3f} pop_joint={pop / max(kept, 1):.3f} "
          f"thin_main={thin / max(kept, 1):.3f} n_virtual(fit)={tree.n_virtual}")
print(f"ALL: kept/n per tree={tot['kept'] / n / len(trees):.3f} pop_joint="
      f"{tot['pop'] / max(tot['kept'], 1):.3f} thin_main={tot['thin'] / max(tot['kept'], 1):.3f} "
      f"|S|={len(union)}")
