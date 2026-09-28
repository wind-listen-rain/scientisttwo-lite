"""Dev: share of pair-cell lookups on held-out points by cell kind."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from smoothed_hfd import SmoothedTreeHFD, _f32  # noqa: E402

from setup import analytical, real  # noqa: E402

name = sys.argv[1]
alphas = [float(a) for a in sys.argv[2].split(",")]
model, X, Xe = analytical(0) if name == "analytical" else real(name)
for a in alphas:
    h = SmoothedTreeHFD(model, alpha=a).fit(X)
    for tag, Z in (("train", X), ("test", Xe)):
        Z = _f32(Z)
        counts = np.zeros(3)
        for F in h.fitted:
            if F.variables.size == 0 or not F.pairs:
                continue
            B = np.column_stack([np.searchsorted(F.split_list[vi], Z[:, j], side="right")
                                 for vi, j in enumerate(F.variables)])
            for q, (ji, ki) in enumerate(F.pair_local):
                counts += np.bincount(F.pair_kind[q][B[:, ji], B[:, ki]], minlength=3)
        frac = counts / counts.sum()
        print(f"{name} alpha={a} {tag:>5}: empirical={frac[0]:.4f} virtual-only={frac[1]:.4f} "
              f"fallback={frac[2]:.4f}", flush=True)
