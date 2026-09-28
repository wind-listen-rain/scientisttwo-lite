"""Dev: CV scores per alpha with paired standard errors (difference to the best alpha), label-free, X_train only."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import smoothed_hfd as sh  # noqa: E402

from setup import analytical, real  # noqa: E402

grid = tuple(float(a) for a in sys.argv[2].split(","))
for name in sys.argv[1].split(","):
    model, X, _ = analytical(int(name[-1])) if name.startswith("analytical") else real(name)
    s = sh.SmoothedTreeHFD(model, alpha_grid=grid)
    for one_se in (False, True):
        sh.CV_ONE_SE = one_se
        pick, cv = s._select_alpha(sh._f32(X), 2)
        if one_se:
            print(f"{name:>12}: " + " ".join(f"a={g}: {cv[str(g)]:.4f}±{s.diag['cv_se'][str(g)]:.4f}" for g in grid)
                  + f" | argmin={argmin} 1-SE={pick}", flush=True)
        else:
            argmin = pick
