"""Development check: GT-LOCO fit, selection, and (diagnostic only) actual in/out residual per candidate.

usage: dev_gtloco.py <dataset|analytical> [full] [ortho=hard|soft] [eps=<EPS_MAIN>] [rho=<LATTICE_RIDGE>]
                     [rules=zero,harmonic,nearest] [priors=ridge,lattice]
The held-out numbers are printed for diagnosis only; the method never sees them.
"""
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import gtloco  # noqa: E402
from common import analytical, real  # noqa: E402

gtloco.KEEP_PATH = True
opts = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
gtloco.HARD_ORTHO = opts.get("ortho", "hard") == "hard"
gtloco.EPS_MAIN = float(opts.get("eps", gtloco.EPS_MAIN))
gtloco.LATTICE_RIDGE = float(opts.get("rho", gtloco.LATTICE_RIDGE))
if "rules" in opts:
    gtloco.RULES = tuple(opts["rules"].split(","))
if "priors" in opts:
    gtloco.PRIORS = tuple(opts["priors"].split(","))
VAR = gtloco.variants()
name = sys.argv[1]
model, Xtr, Xte = analytical(0) if name == "analytical" else real(name)
hfd = gtloco.GTLocoHFD(model)
t0 = time.time()
hfd.fit(Xtr)
print(f"fit {time.time() - t0:.2f}s")
d = hfd.diagnostics
vtr, vte = np.var(model.predict(Xtr)), np.var(model.predict(Xte))
sel = d["selection"]
print("selection", sel[0], [gtloco.PRIORS[VAR[sel[1]][0]], gtloco.RULES[VAR[sel[1]][1]]], sel[2],
      f"R/var={sel[3] / vtr:.5f}", "N1/n mean", round(float(np.mean(d["n1_frac"])), 4),
      "m mean/max", np.mean(d["m"]), np.max(d["m"]))
print("chosen kappa quantiles", np.quantile(d["kappa_chosen"], [0, .25, .5, .75, 1]))


def ortho(pred, main, inter, il):
    v = np.var(pred)
    worst = None
    for c, (j, k) in enumerate(il):
        if np.var(inter[:, c]) < 0.01 * v:
            continue
        for mm in (j, k):
            if np.var(main[:, mm]) > 0:
                r = abs(np.corrcoef(inter[:, c], main[:, mm])[0, 1])
                worst = r if worst is None else max(worst, r)
    return -1 if worst is None else worst


def evaluate():
    out = {}
    for split, X, v in (("in", Xtr, vtr), ("out", Xte, vte)):
        main, inter = hfd.predict(X)
        rec = hfd.eta0 + main.sum(1) + inter.sum(1)
        out[split] = np.mean((model.predict(X) - rec) ** 2) / v
        out["o" + split] = ortho(model.predict(X), main, inter, hfd.interaction_list)
    return out


r = evaluate()
print(f"SELECTED resid_in={r['in']:.5f} resid_out={r['out']:.5f} ortho_in={r['oin']:.4f} "
      f"ortho_out={r['oout']:.4f}")
cands = {(c[0], c[1], c[2]): c[3] for c in d["candidates"]}
G = len(gtloco.KAPPAS)
if "full" in sys.argv[2:]:
    print("mode prior rule par  R/var  resid_in resid_out ortho_in ortho_out")
    for v, (pr, ru) in enumerate(VAR):
        lab = f"{gtloco.PRIORS[pr]:7s} {gtloco.RULES[ru]:8s}"
        for s in gtloco.SHIFTS:
            for t, tree in enumerate(hfd.treehfd_list):
                tree.finalize(int(np.clip(d["per_tree_idx"][t, v] + s, 0, G - 1)), v)
            r = evaluate()
            print(f"shift  {lab} {s:+d} {cands[('shift', v, int(s))] / vtr:.5f} "
                  f"{r['in']:.5f} {r['out']:.5f} {r['oin']:.4f} {r['oout']:.4f}")
        for g in range(G - 3):
            for tree in hfd.treehfd_list:
                tree.finalize(g, v)
            r = evaluate()
            print(f"common {lab} k={gtloco.KAPPAS[g]:g} {cands[('common', v, g)] / vtr:.5f} "
                  f"{r['in']:.5f} {r['out']:.5f} {r['oin']:.4f} {r['oout']:.4f}")
