"""Dev: per-tree system sizes (columns, full cells) for a dataset."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from smoothed_hfd import SmoothedTreeHFD, _f32, _prepare_tree, _shift_values  # noqa: E402

from setup import analytical, real  # noqa: E402

name = sys.argv[1]
model, X, Xe = analytical(0) if name == "analytical" else real(name)
s = SmoothedTreeHFD(model, alpha=0.2)
X = _f32(X)
t0 = time.time()
shifts = _shift_values(X, s.union_splits)
print("shift_s", time.time() - t0, "num_shift", s.num_shift)
ncols, nfulls, nvirt = [], [], []
t0 = time.time()
for t, tree in enumerate(s.trees):
    P = _prepare_tree(tree, s._pairs(t, 2), s.tree_vars[t], X, shifts, s.num_shift)
    if P.const is not None:
        continue
    ncols.append(int(P.nbins.sum() + sum(len(b) for b in P.pair_bins)))
    nfulls.append(P.nfull)
    nvirt.append(P.num_virtual)
    assert P.y_spread < 1e-9, P.y_spread
print("prepare_s", time.time() - t0)
for k, v in (("ncol", ncols), ("nfull", nfulls), ("nvirt", nvirt)):
    v = np.array(v)
    print(k, "mean", v.mean(), "median", np.median(v), "max", v.max())
