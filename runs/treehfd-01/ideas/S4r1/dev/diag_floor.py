"""DIAGNOSTIC ONLY (never used by the method): approximation floor of the main+pair family on test
points, obtained by fitting the decomposition on train U test inputs. Tells how much of resid_out is
extrapolation to unseen joint cells versus representational error of the order-2 family."""
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import gtloco  # noqa: E402
from common import analytical, real  # noqa: E402

gtloco.KEEP_PATH = True
name = sys.argv[1]
model, Xtr, Xte = analytical(0) if name == "analytical" else real(name)
Xall = np.vstack([Xtr, Xte])
hfd = gtloco.GTLocoHFD(model)
hfd.fit(Xall)
pte = model.predict(Xte)
for g in range(0, 6):
    for tree in hfd.treehfd_list:
        tree.finalize(g, 1)
    main, inter = hfd.predict(Xte)
    print(f"kappa={gtloco.KAPPAS[g]}: test resid when test cells are seen = "
          f"{np.mean((pte - hfd.eta0 - main.sum(1) - inter.sum(1)) ** 2) / np.var(pte):.5f}")
