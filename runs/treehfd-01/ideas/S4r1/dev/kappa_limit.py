"""Attribution check (development only): is the kappa -> 0 limit of GT-LOCO still different
from the baseline?

usage: kappa_limit.py <dataset> [...]
Fits GT-LOCO with the kappa grid replaced by (1e-4, 1e-3, 1e-2) and prints, for every
(prior, rule) variant and common kappa, resid_in / resid_out on the benchmark split. If the
numbers do not move as kappa -> 0, the prior acts only as a tie-breaker among solutions that
fit the data equally well (weakly or non-identified directions), not as shrinkage.
Held-out numbers are for attribution reporting only; nothing here feeds back into the method.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gtloco  # noqa: E402
from common import analytical, real  # noqa: E402

gtloco.KEEP_PATH = True
gtloco.ORTHO_TERM = False
gtloco.RESID_IN_CAP = None
gtloco.KAPPAS = np.array([1e-4, 1e-3, 1e-2])
gtloco.SHIFTS = np.array([0])
for name in sys.argv[1:]:
    model, Xtr, Xte = analytical(0) if name == "analytical" else real(name)
    h = gtloco.GTLocoHFD(model)
    h.fit(Xtr)
    line = []
    for v, (pr, ru) in enumerate(gtloco.variants()):
        for g, k in enumerate(gtloco.KAPPAS):
            for tree in h.treehfd_list:
                tree.finalize(g, v)
            r = []
            for X in (Xtr, Xte):
                p = model.predict(X)
                m, it = h.predict(X)
                r.append(np.mean((p - h.eta0 - m.sum(1) - it.sum(1)) ** 2) / np.var(p))
            line.append(f"{gtloco.PRIORS[pr]}/{gtloco.RULES[ru]} k={k:g}: in={r[0]:.5f} out={r[1]:.5f}")
    print(f"=== {name}\n  " + "\n  ".join(line), flush=True)
