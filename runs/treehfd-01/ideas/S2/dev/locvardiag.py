"""Dev: per-variable local variability (harness definition) for baseline and S2 at several alphas."""
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import smoothed_hfd as sh  # noqa: E402
from treehfd_mod import XGBTreeHFD  # noqa: E402

from setup import real  # noqa: E402

name = sys.argv[1]
alphas = [float(a) for a in sys.argv[2].split(",")]
model, X, Xe = real(name)
pt = model.predict(X)
idx = [NearestNeighbors(n_neighbors=10).fit(X[:, [j]]).kneighbors(X[:, [j]], return_distance=False)
       for j in range(X.shape[1])]


def per_var(main):
    return [np.mean(np.var(main[idx[j], j], axis=1)) / np.var(main[:, j]) if np.var(main[:, j]) > 0 else np.nan
            for j in range(X.shape[1])]


h = XGBTreeHFD(model)
h.fit(X, verbose=False)
m, _ = h.predict(X, verbose=False)
print(f"{name} base : " + " ".join(f"{v:.4f}" for v in per_var(m)), "| var share:",
      " ".join(f"{np.var(m[:, j]) / np.var(pt):.3f}" for j in range(X.shape[1])))
for a in alphas:
    s = sh.SmoothedTreeHFD(model, alpha=a).fit(X)
    m, _ = s.predict(X)
    print(f"{name} a={a:<4}: " + " ".join(f"{v:.4f}" for v in per_var(m)), "| var share:",
          " ".join(f"{np.var(m[:, j]) / np.var(pt):.3f}" for j in range(X.shape[1])))
print("distinct values per variable:", [len(np.unique(X[:, j])) for j in range(X.shape[1])])
