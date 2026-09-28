"""Dev: compare baseline TreeHFD with SmoothedTreeHFD at fixed alphas."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from smoothed_hfd import SmoothedTreeHFD, _f32  # noqa: E402
from treehfd_mod import XGBTreeHFD  # noqa: E402

from setup import analytical, real  # noqa: E402

name = sys.argv[1]
alphas = [None if a == "cv" else float(a) for a in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.0]
run_base = "--nobase" not in sys.argv
model, X, Xe = analytical(0) if name == "analytical" else real(name)
pt, pe = model.predict(X), model.predict(Xe)


def report(tag, eta0, fn, fit_s):
    out = []
    for XX, pp in ((X, pt), (Xe, pe)):
        m, it = fn(XX)
        rec = eta0 + m.sum(1) + it.sum(1)
        out.append(np.mean((pp - rec) ** 2) / np.var(pp))
    print(f"{tag:>14}: resid_in={out[0]:.5f} resid_out={out[1]:.5f} fit={fit_s:.1f}s", flush=True)


if run_base:
    t0 = time.time()
    h = XGBTreeHFD(model)
    h.fit(X, verbose=False)
    report("baseline", h.eta0, lambda Z: h.predict(Z, verbose=False), time.time() - t0)

for a in alphas:
    t0 = time.time()
    s = SmoothedTreeHFD(model, alpha=a).fit(X)
    report(f"S2 alpha={a}", s.eta0, s.predict, time.time() - t0)
    print("   ", {k: v for k, v in s.diag.items()})
