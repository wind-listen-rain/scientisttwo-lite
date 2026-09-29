"""Which interactions drive ortho_in / ortho_out along the kappa path (diagnostic only).

usage: diag_ortho.py <dataset|analytical> [variant index] [ortho=hard|soft]
"""
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import gtloco  # noqa: E402
from common import analytical, real  # noqa: E402

gtloco.KEEP_PATH = True
opts = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
gtloco.HARD_ORTHO = opts.get("ortho", "hard") == "hard"
name = sys.argv[1]
pos = [a for a in sys.argv[2:] if "=" not in a]
var = int(pos[0]) if pos else len(gtloco.variants()) - 1
model, Xtr, Xte = analytical(0) if name == "analytical" else real(name)
hfd = gtloco.GTLocoHFD(model)
hfd.fit(Xtr)
il = hfd.interaction_list
print("variant", [(gtloco.PRIORS[a], gtloco.RULES[b]) for a, b in [gtloco.variants()[var]]],
      "n_test", len(Xte))
for g in [0, 2, 4]:
    for tree in hfd.treehfd_list:
        tree.finalize(g, var)
    print(f"== kappa={gtloco.KAPPAS[g]}")
    for split, X in (("tr", Xtr), ("te", Xte)):
        pred = model.predict(X)
        v = np.var(pred)
        main, inter = hfd.predict(X)
        for c, (j, k) in enumerate(il):
            vi = np.var(inter[:, c])
            if vi < 0.01 * v:
                continue
            cj = np.corrcoef(inter[:, c], main[:, j])[0, 1]
            ck = np.corrcoef(inter[:, c], main[:, k])[0, 1]
            print(f"  {split} ({j},{k}) var/V={vi / v:.4f} corr_j={cj:+.4f} corr_k={ck:+.4f} "
                  f"varmain_j/V={np.var(main[:, j]) / v:.4f} varmain_k/V={np.var(main[:, k]) / v:.4f}")
