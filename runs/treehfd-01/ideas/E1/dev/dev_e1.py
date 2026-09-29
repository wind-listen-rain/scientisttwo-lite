"""Development check: anchored GT-LOCO fit, selection, and (diagnostic only) held-out metrics.

usage: dev_e1.py <dataset|analytical[:rep]> [split=<s>] [rhos=0,0.25,1] [anchor_priors=ridge,lattice]
                 [se=<k>] [ablate] [ortho]
Reports the selection (k-SE rule, SE_RULE of lib/agtloco.py unless se=<k>) and the plain argmin
of R. With 'ablate', also reports, restricted to each rho mode (fixed common rho 0 / 0.25 / 1,
per-tree rho), the best candidate by R and the k-SE choice, i.e. the fixed-rho ablation (the
rho = 0 mode is S4). With 'ortho', reports orthogonality at a common kappa for rho = 0 / 0.25 / 1
(same kappa in every tree, so only rho changes). Held-out numbers are printed for diagnosis
only; the method never sees them and they are not used for selection.
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
if "se" in opts:
    agtloco.SE_RULE = float(opts["se"])
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
print(f"{name} split={opts.get('split', 0)} RHOS={agtloco.RHOS} SE_RULE={agtloco.SE_RULE:g} "
      f"fit {time.time() - t0:.2f}s")
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
      f"{d['resid_in_over_var']:.5f} tied={d['n_tied']}")
print("argmin   ", label(d["min_risk"]), f"R/var={d['min_risk'][4] / d['var_t']:.5f} "
      f"resid_in(est)/var={d['min_risk'][5] / d['var_t']:.5f}")
if "s4_selection" in d:   # round 3: fidelity cap reference and feasible-set size
    print("S4 ref   ", label(d["s4_selection"]), f"R/var={d['s4_selection'][4] / d['var_t']:.5f} "
          f"resid_in(est)/var={d['s4_selection'][5] / d['var_t']:.5f} "
          f"feasible={d['n_feasible']}/{len(d['candidates'])} FID_CAP={agtloco.FID_CAP}")
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


def fmt(tag, c, r):
    return (f"  {tag:13s} {label(c)} R/var={c[4] / d['var_t']:.5f} "
            f"resid_in={r['in']:.5f} resid_out={r['out']:.5f} ortho_in={r['oin']:.4f} "
            f"ortho_out={r['oout']:.4f} locvar_in={r['locvar']:.4g}")


cands = d["candidates"]
if d["min_risk"][:4] != sel[:4]:
    apply(d["min_risk"])
    print("ARGMIN" + fmt("", d["min_risk"], evaluate())[15:])
    apply(sel)
else:
    print("ARGMIN" + f" resid_in={r['in']:.5f} resid_out={r['out']:.5f} ortho_in={r['oin']:.4f} "
          f"ortho_out={r['oout']:.4f} locvar_in={r['locvar']:.4g} (same as selection)")
if "ablate" in sys.argv[2:]:
    sq = d["cand_loo"].astype(float) ** 2
    fid = np.array([c[5] for c in cands])
    k = agtloco.SE_RULE if agtloco.SE_RULE > 0 else 1.0
    print(f"rho-mode ablation: best candidate by R, and the {k:g}-SE choice, within each rho mode")
    for rm in range(NR + 1):
        idx = np.array([i for i, c in enumerate(cands) if c[2] == rm])
        best = cands[idx[int(agtloco._first_min(np.array([cands[i][4] for i in idx])))]]
        apply(best)
        print(fmt(RMN[rm], best, evaluate()))
        _, i_se, _ = agtloco._se_select(sq[idx], fid[idx], k)
        c = cands[idx[i_se]]
        apply(c)
        print(fmt(RMN[rm] + f"/se{k:g}", c, evaluate()))
    apply(sel)
if "ortho" in sys.argv[2:]:
    v = sel[1]
    print("orthogonality at a common kappa (selected variant), rho = 0 / 0.25 / 1")
    for g in (0, 2, 4, 6):
        for rm in range(NR):
            c = next(c for c in cands if c[0] == "common" and c[1] == v and c[2] == rm
                     and c[3] == g)
            apply(c)
            print(fmt(f"k={agtloco.KAPPAS[g]:g}", c, evaluate()))
    apply(sel)
