"""Wall time of a whole ensemble fit (as the runner times method.fit), diagnostic.

usage: time_fit.py <impl.py> <dataset|analytical[:rep]> [...]
"""
import importlib.machinery
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

E1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E1 / "lib"))
sys.path.insert(0, str(E1 / "dev"))
from common import analytical, real  # noqa: E402

ld = importlib.machinery.SourceFileLoader("agt", sys.argv[1])
mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("agt", ld))
ld.exec_module(mod)
for name in sys.argv[2:]:
    if name.startswith("analytical"):
        model, Xtr, _ = analytical(int(name.split(":")[1]) if ":" in name else 0)
    else:
        model, Xtr, _ = real(name)
    t0 = time.time()
    hfd = mod.GTLocoHFD(model)
    hfd.fit(np.asarray(Xtr, float))
    d = hfd.diagnostics
    print(f"{name}: fit {time.time() - t0:.1f}s selection {d['selection'][:4]} "
          f"min_risk {d['min_risk'][:4]}", flush=True)
