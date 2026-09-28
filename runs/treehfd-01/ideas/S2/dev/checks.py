"""Dev: correctness checks (tree traversal == model, determinism, interface)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from smoothed_hfd import SmoothedTreeHFD, _f32  # noqa: E402

from setup import analytical, real  # noqa: E402

spec = importlib.util.spec_from_file_location("m", Path(__file__).resolve().parents[1] / "method.py")
method = importlib.util.module_from_spec(spec)
spec.loader.exec_module(method)

for name in sys.argv[1].split(","):
    model, X, Xe = analytical(0) if name == "analytical" else real(name)
    h = SmoothedTreeHFD(model, alpha=0.0)
    for Z, tag in ((X, "train"), (Xe, "test")):
        tot = sum(t.output(_f32(Z)) for t in h.trees)
        d = np.abs(tot - model.predict(Z, output_margin=True)).max()
        print(f"{name} {tag}: max |sum_t T_t - margin| = {d:.2e} (scale {np.abs(tot).max():.2e})")
    s1 = method.fit(model, X)
    s2 = method.fit(model, X)
    o1, o2 = method.predict(s1, Xe), method.predict(s2, Xe)
    same = all(np.array_equal(np.asarray(o1[k]), np.asarray(o2[k])) for k in o1)
    print(f"{name}: deterministic={same} alpha={s1.alpha} shapes main={o1['main'].shape} "
          f"inter={o1['inter'].shape} inter_list={o1['inter_list'].shape} "
          f"finite={np.isfinite(o1['main']).all() and np.isfinite(o1['inter']).all()}")
    print(f"{name}: diag={s1.diag}")
