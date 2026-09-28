"""Dev: why NEW_PAIR_CELLS="tie" blows up out of sample (per-tree solve diagnostics)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import smoothed_hfd as sh  # noqa: E402

from setup import real  # noqa: E402

name, alpha = sys.argv[1], float(sys.argv[2])
model, X, Xe = real(name)
X, Xe = sh._f32(X), sh._f32(Xe)
h = sh.SmoothedTreeHFD(model, alpha=alpha)
shifts = sh._shift_values(X, h.union_splits)
for mode in ("fit", "tie", "drop"):
    sh.NEW_PAIR_CELLS = mode
    worst = []
    for t, tree in enumerate(h.trees):
        P = sh._prepare_tree(tree, h.tree_pairs[t], h.tree_vars[t], X, shifts, h.num_shift)
        if P.const is not None:
            continue
        A, rhs, S = sh._build_system(P, alpha)
        stats = sh._new_stats()
        coef = sh._solve_ls(A, rhs, stats)
        F = sh._solve_tree(P, alpha, sh._new_stats())
        m, it = sh._predict_tree(F, Xe)
        rec = F.eta0 + m.sum(1) + it.sum(1)
        err = np.mean((tree.output(Xe) - rec) ** 2)
        sv = np.linalg.svd(A.toarray(), compute_uv=False)
        rank_tol = sv[0] * 1e-9
        worst.append((err, t, float(np.abs(coef).max()), stats["istop"],
                      int(np.sum(sv < rank_tol)), float(sv[sv >= rank_tol].min() / sv[0])))
    worst.sort(reverse=True)
    print(f"{mode}: total test mse over trees {sum(w[0] for w in worst):.4g}")
    for w in worst[:4]:
        print(f"   tree {w[1]:3d}: test mse {w[0]:.4g} max|coef| {w[2]:.3g} istop {w[3]} "
              f"null dims {w[4]} smallest rel sv {w[5]:.3g}")
