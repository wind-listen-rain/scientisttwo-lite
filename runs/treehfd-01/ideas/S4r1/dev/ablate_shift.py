"""Ablation (development only): does the ensemble-level kappa step pull its weight?

usage: ablate_shift.py <dataset> [splits=0,1,2,...]
For each split (0 = benchmark split, others diagnostic only) compares, on held-out points:
  * selected   : the method as submitted (ensemble step over shifts / common kappa);
  * per-tree   : ensemble risk restricted to shift 0 (per-tree argmin, variant by ensemble risk);
  * per-tree-v : per-tree argmin with the selected variant (shift 0);
and prints the ensemble risk R/Var (train only) next to held-out numbers.
"""
import sys

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "lib"))
import gtloco  # noqa: E402

DATA = str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/bench/data"
XGB_PARAMS = dict(eta=0.1, n_estimators=100, max_depth=6, n_jobs=4)


def split(name, seed):
    d = np.load(f"{DATA}/{name}.npz")
    X, y = d["X"], d["y"]
    perm = np.random.default_rng(seed).permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = perm[:cut], perm[cut:]
    model = xgb.XGBRegressor(random_state=0, **XGB_PARAMS).fit(X[tr], y[tr])
    return model, X[tr], X[te]


def ortho(pred, main, inter, il):
    v = np.var(pred)
    worst = np.nan
    for c, (j, k) in enumerate(il):
        if np.var(inter[:, c]) < 0.01 * v:
            continue
        for mm in (j, k):
            if np.var(main[:, mm]) > 0:
                r = abs(np.corrcoef(inter[:, c], main[:, mm])[0, 1])
                worst = r if np.isnan(worst) else max(worst, r)
    return worst


def main():
    opts = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
    splits = [int(s) for s in opts.get("splits", "0").split(",")]
    gtloco.KEEP_PATH = True
    name = sys.argv[1]
    rows = []
    for seed in splits:
        model, Xtr, Xte = split(name, seed)
        hfd = gtloco.GTLocoHFD(model)
        hfd.fit(Xtr)
        d = hfd.diagnostics
        G = len(gtloco.KAPPAS)
        vtr = np.var(model.predict(Xtr))
        cands = d["candidates"]
        sel = d["selection"]

        def ev(mode, v, par, hfd=hfd, d=d, G=G, Xtr=Xtr, Xte=Xte, model=model):
            for t, tree in enumerate(hfd.treehfd_list):
                g = (int(np.clip(d["per_tree_idx"][t, v] + par, 0, G - 1)) if mode == "shift"
                     else par)
                tree.finalize(g, v)
            out = {}
            for s, X in (("in", Xtr), ("out", Xte)):
                p = model.predict(X)
                mn, inter = hfd.predict(X)
                out[s] = np.mean((p - hfd.eta0 - mn.sum(1) - inter.sum(1)) ** 2) / np.var(p)
                out["o" + s] = ortho(p, mn, inter, hfd.interaction_list)
            return out

        s0 = [c for c in cands if c[0] == "shift" and c[2] == 0]
        best0 = s0[int(gtloco._first_min(np.array([c[3] for c in s0])))]
        res = {"selected": (sel, ev(*sel[:3])), "per-tree": (best0, ev(*best0[:3])),
               "per-tree-v": (None, ev("shift", sel[1], 0))}
        VAR = gtloco.variants()
        print(f"=== {name} split {seed}: N1/n mean {np.mean(d['n1_frac']):.3f}")
        for lab, (c, r) in res.items():
            desc = (f"{c[0]} {gtloco.PRIORS[VAR[c[1]][0]]}/{gtloco.RULES[VAR[c[1]][1]]} {c[2]} "
                    f"R/var={c[3] / vtr:.5f}" if c is not None else "")
            print(f"  {lab:10s} resid_in={r['in']:.5f} resid_out={r['out']:.5f} "
                  f"ortho_in={r['oin']:.4f} ortho_out={r['oout']:.4f}  {desc}")
        rows.append({k: v[1] for k, v in res.items()})

    if len(rows) > 1:
        print("=== summary: selected / per-tree ratios")
        for k in ("in", "out", "oin", "oout"):
            r = np.array([row["selected"][k] / row["per-tree"][k] for row in rows])
            print(f"  {k:5s}: " + " ".join(f"{x:.3f}" for x in r)
                  + f" | geo-mean {np.exp(np.nanmean(np.log(r))):.3f}")


if __name__ == "__main__":
    main()
