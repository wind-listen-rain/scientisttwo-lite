"""Development check: ensemble step with the one-SE fidelity rule vs plain argmin of R.

usage: compare_se.py <dataset|analytical> [splits=0,1,...] [se=1.0]
One fit per split; both selections are finalised from the same candidate set. For real data,
split 0 is the benchmark split (others are diagnostic only); for "analytical", the split is
the repetition index. Held-out numbers are for diagnosis only; the method never sees them.
"""
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/S4/lib")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/S4/dev")
import gtloco  # noqa: E402
from ablate_shift import ortho, split  # noqa: E402
from common import analytical  # noqa: E402

opts = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
SPLITS = [int(s) for s in opts.get("splits", "0").split(",")]
gtloco.KEEP_PATH = True
gtloco.SE_RULE = float(opts.get("se", 1.0))
name = sys.argv[1]
VAR = gtloco.variants()
G = len(gtloco.KAPPAS)
ratios = []
for seed in SPLITS:
    model, Xtr, Xte = analytical(seed) if name == "analytical" else split(name, seed)
    hfd = gtloco.GTLocoHFD(model)
    hfd.fit(Xtr)
    d = hfd.diagnostics
    vtr = np.var(model.predict(Xtr))

    def ev(c):
        mode, v, par = c[:3]
        for t, tree in enumerate(hfd.treehfd_list):
            g = (int(np.clip(d["per_tree_idx"][t, v] + par, 0, G - 1)) if mode == "shift"
                 else par)
            tree.finalize(g, v)
        out = {}
        for s, X in (("in", Xtr), ("out", Xte)):
            p = model.predict(X)
            main, inter = hfd.predict(X)
            out[s] = np.mean((p - hfd.eta0 - main.sum(1) - inter.sum(1)) ** 2) / np.var(p)
            out["o" + s] = ortho(p, main, inter, hfd.interaction_list)
        return out

    print(f"=== {name} split {seed}")
    res = {}
    for lab, c in (("argmin", d["min_risk"]), ("1se", d["selection"])):
        r = ev(c)
        res[lab] = r
        print(f"  {lab:7s} {c[0]} {gtloco.PRIORS[VAR[c[1]][0]]}/{gtloco.RULES[VAR[c[1]][1]]} "
              f"{c[2]:+d}  R/var={c[3] / vtr:.5f} fid/var={c[4] / vtr:.5f} | resid_in={r['in']:.5f} "
              f"resid_out={r['out']:.5f} ortho_in={r['oin']:.4f} ortho_out={r['oout']:.4f}")
    ratios.append({k: res["1se"][k] / res["argmin"][k] for k in res["1se"]})
if len(ratios) > 1:
    print("=== 1se / argmin ratios")
    for k in ratios[0]:
        r = np.array([x[k] for x in ratios])
        print(f"  {k:5s}: " + " ".join(f"{x:.3f}" for x in r)
              + f" | geo-mean {np.exp(np.nanmean(np.log(r))):.3f}")
