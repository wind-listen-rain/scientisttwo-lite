"""Development check: anchored GT-LOCO fit, selection, and (diagnostic only) held-out metrics.

usage: dev_e1.py <dataset|analytical[:rep]> [split=<s>] [rhos=0,0.25,1] [anchor_priors=ridge,lattice]
                 [ablate]
With 'ablate', also reports the best candidate restricted to each rho mode (fixed common rho
0 / 0.25 / 1, per-tree rho), i.e. the fixed-rho ablation. Held-out numbers are printed for
diagnosis only; the method never sees them and they are not used for selection.
"""
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/E1/lib")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/E1/dev")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/bench")
import agtloco  # noqa: E402
from common import analytical, real  # noqa: E402
from harness import local_variability, orthogonality  # noqa: E402  (read-only import)

agtloco.KEEP_PATH = True
opts = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
if "rhos" in opts:
    agtloco.RHOS = tuple(float(r) for r in opts["rhos"].split(","))
if "anchor_priors" in opts:
    agtloco.ANCHOR_PRIORS = tuple(opts["anchor_priors"].split(","))
VAR = agtloco.variants()
name = sys.argv[1]
if name.startswith("analytical"):
    rep = int(name.split(":")[1]) if ":" in name else 0
    model, Xtr, Xte = analytical(rep)
else:
    model, Xtr, Xte = real(name, int(opts.get("split", 0)))
hfd = agtloco.GTLocoHFD(model)
t0 = time.time()
hfd.fit(Xtr)
print(f"{name} split={opts.get('split', 0)} RHOS={agtloco.RHOS} fit {time.time() - t0:.2f}s")
d = hfd.diagnostics
vtr, vte = np.var(model.predict(Xtr)), np.var(model.predict(Xte))
NR = len(agtloco.RHOS)
RMN = [f"rho={r:g}" for r in agtloco.RHOS] + ["rho=per-tree"]


def label(c):
    pr, ru = VAR[c[1]]
    return f"{c[0]:6s} {agtloco.PRIORS[pr]:7s} {agtloco.RULES[ru]:8s} {RMN[c[2]]:13s} {c[3]:+d}"


def ortho(pred, main, inter, il):
    o = orthogonality(pred, main, inter, il)
    return float("nan") if o is None else o


def evaluate():
    out = {}
    for split, X, v in (("in", Xtr, vtr), ("out", Xte, vte)):
        main, inter = hfd.predict(X)
        rec = hfd.eta0 + main.sum(1) + inter.sum(1)
        out[split] = np.mean((model.predict(X) - rec) ** 2) / v
        out["o" + split] = ortho(model.predict(X), main, inter, hfd.interaction_list)
        if split == "in":
            out["locvar"] = local_variability(Xtr, main)
    return out


def apply(c):
    mode, v, rm, par = c[:4]
    G = len(agtloco.KAPPAS)
    for t, tree in enumerate(hfd.treehfd_list):
        if rm < NR:
            r, g0 = rm, d["per_tree_idx"][t, v, rm]
        else:
            r, g0 = d["per_tree_joint"][t, v]
        g = int(np.clip(g0 + par, 0, G - 1)) if mode == "shift" else par
        tree.finalize(g, v, r)


sel = d["selection"]
print("selection", label(sel), f"R/var={d['risk_over_var']:.5f} resid_in(est)/var="
      f"{d['resid_in_over_var']:.5f}")
print("N1/n mean", round(float(np.mean(d["n1_frac"])), 4), "m mean/max", np.mean(d["m"]),
      np.max(d["m"]), "virtual/n mean", round(float(np.mean(d["n_virtual"])) / len(Xtr), 3),
      "route_err max", f"{np.max(d['route_err']):.1e}")
print("chosen kappa quantiles", np.quantile(d["kappa_chosen"], [0, .25, .5, .75, 1]))
rc = d["rho_chosen"]
print("chosen rho counts", {float(r): int(np.sum(rc == r)) for r in agtloco.RHOS})
pj = d["per_tree_joint"][:, sel[1], 0]
print("per-tree joint argmin rho counts (selected variant)",
      {float(agtloco.RHOS[r]): int(np.sum(pj == r)) for r in range(NR)})
r = evaluate()
print(f"SELECTED resid_in={r['in']:.5f} resid_out={r['out']:.5f} ortho_in={r['oin']:.4f} "
      f"ortho_out={r['oout']:.4f} locvar_in={r['locvar']:.4g}")
if "ablate" in sys.argv[2:]:
    cands = d["candidates"]
    print("rho-mode ablation: best candidate by R within each rho mode")
    for rm in range(NR + 1):
        sub = [c for c in cands if c[2] == rm]
        best = sub[int(agtloco._first_min(np.array([c[4] for c in sub])))]
        apply(best)
        r = evaluate()
        print(f"  {RMN[rm]:13s} {label(best)} R/var={best[4] / d['var_t']:.5f} "
              f"resid_in={r['in']:.5f} resid_out={r['out']:.5f} ortho_in={r['oin']:.4f} "
              f"ortho_out={r['oout']:.4f} locvar_in={r['locvar']:.4g}")
    apply(sel)
