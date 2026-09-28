"""Dev: harness-equivalent real-data metrics for several configurations in-process."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/bench")
from harness import local_variability, orthogonality  # noqa: E402
from smoothed_hfd import SmoothedTreeHFD  # noqa: E402
from treehfd_mod import XGBTreeHFD  # noqa: E402

from setup import real  # noqa: E402

names = sys.argv[1].split(",")
configs = sys.argv[2].split(",")  # "base", "cv", or alpha values
extra = dict(a.split("=") for a in sys.argv[3:])
import smoothed_hfd as _sh  # noqa: E402
if "cm" in extra:
    _sh.CONSTRAINT_MEASURE = extra["cm"]
if "grid" in extra:
    _sh.ALPHA_GRID = tuple(float(a) for a in extra["grid"].split("/"))
for _k, _v in extra.items():  # generic overrides of module constants, e.g. CV_ONE_SE=0
    if _k.isupper():
        _t = type(getattr(_sh, _k))
        setattr(_sh, _k, _v not in ("0", "False") if _t is bool else _t(_v))


def ortho_detail(pred, main, inter, il):
    v = np.var(pred)
    rows = []
    for c, (j, k) in enumerate(il):
        if np.var(inter[:, c]) < 0.01 * v:
            continue
        for m in (j, k):
            if np.var(main[:, m]) > 0:
                rows.append((abs(np.corrcoef(inter[:, c], main[:, m])[0, 1]), (int(j), int(k)), int(m),
                             float(np.var(inter[:, c]) / v)))
    rows.sort(reverse=True)
    return rows[:3]


for name in names:
    model, X, Xe = real(name)
    pt, pe = model.predict(X), model.predict(Xe)
    for cfg in configs:
        t0 = time.time()
        if cfg == "base":
            h = XGBTreeHFD(model)
            h.fit(X, verbose=False)
            fn = lambda Z: h.predict(Z, verbose=False)  # noqa: E731
            eta0, il = h.eta0, h.interaction_list
        else:
            h = SmoothedTreeHFD(model, alpha_grid=_sh.ALPHA_GRID, alpha=None if cfg == "cv" else float(cfg))
            h.fit(X)
            fn, eta0, il = h.predict, h.eta0, h.interaction_list
        fit_s = time.time() - t0
        mt, it = fn(X)
        me, ie = fn(Xe)
        rin = np.mean((pt - eta0 - mt.sum(1) - it.sum(1)) ** 2) / np.var(pt)
        rout = np.mean((pe - eta0 - me.sum(1) - ie.sum(1)) ** 2) / np.var(pe)
        oin = orthogonality(pt, mt, it, il)
        oout = orthogonality(pe, me, ie, il)
        lv = local_variability(X, mt)
        f = lambda v: "None" if v is None else f"{v:.4f}"  # noqa: E731
        print(f"{name:>12} {cfg:>5}: resid_in={rin:.5f} resid_out={rout:.5f} ortho_in={f(oin)} "
              f"ortho_out={f(oout)} locvar={lv:.3g} fit={fit_s:.1f}s"
              + (f" alpha={h.alpha}" if cfg != "base" else ""), flush=True)
        if "detail" in extra:
            print("      top ortho_out:", ortho_detail(pe, me, ie, il))
