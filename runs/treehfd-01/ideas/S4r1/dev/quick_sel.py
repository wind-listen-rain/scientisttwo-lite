"""Selection summary of the current lib/gtloco.py on benchmark setups (development only).

usage: quick_sel.py <dataset|analytical> [...]
Prints the label-free diagnostics and, for diagnosis only, held-out resid_out / ortho_out.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gtloco  # noqa: E402
from cand_table import ortho  # noqa: E402
from common import analytical, real  # noqa: E402

for name in sys.argv[1:]:
    model, Xtr, Xte = analytical(0) if name == "analytical" else real(name)
    h = gtloco.GTLocoHFD(model)
    t0 = time.time()
    h.fit(Xtr)
    tf = time.time() - t0
    d = h.diagnostics
    res = []
    for X in (Xtr, Xte):
        p = model.predict(X)
        m, it = h.predict(X)
        res += [np.mean((p - h.eta0 - m.sum(1) - it.sum(1)) ** 2) / np.var(p),
                ortho(p, m, it, h.interaction_list)]
    print(f"{name}: fit {tf:.1f}s sel={d['selection'][:3]} min_R={d['min_risk'][:3]} "
          f"cap_binding={d['cap_binding']} n_adm={d['n_admissible']} "
          f"R/var={d['risk_over_var']:.5f} Omega/var={d['omega_over_var']:.2e} "
          f"n_omega={len(d['omega_evaluated'])} K={len(h.interaction_list)} | resid_in={res[0]:.5f} "
          f"resid_out={res[2]:.5f} ortho_in={res[1]:.4f} ortho_out={res[3]:.4f}", flush=True)
