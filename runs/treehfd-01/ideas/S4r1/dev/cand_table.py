"""Candidate table of the ensemble step (development only).

usage: cand_table.py <dataset|analytical> [splits=0,1,...] [rho=<LATTICE_RIDGE>] [tag=<suffix>]
For each split seed s (0 = benchmark split; others are alternative 80/20 splits used only for
diagnosis; for "analytical", s is the repetition), fits GT-LOCO once and, for every ensemble
candidate (variant x {shift, common kappa}), records
  * label-free quantities the method can use: R, in-sample residual, the in-sample
    orthogonality term Omega (variance of each interaction explained by its two main effects).
  * held-out quantities (diagnosis only, never used by the method): resid_out, ortho_in,
    ortho_out, and Omega on test points.
Saves dev/cache/cand_<dataset>_s<seed>.npz.
"""
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE))
import gtloco  # noqa: E402
from common import DATA, XGB_PARAMS, analytical  # noqa: E402


def split(name, seed):
    if name == "analytical":
        return analytical(seed)
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


def omega(main, inter, il):
    """Sum over interactions of the variance explained by their two main effects."""
    mc = main - main.mean(0)
    tot = 0.0
    for c, (j, k) in enumerate(il):
        y = inter[:, c] - inter[:, c].mean()
        B = mc[:, [j, k]]
        coef = np.linalg.lstsq(B, y, rcond=None)[0]
        tot += np.mean((B @ coef) ** 2)
    return tot


def main():
    name = sys.argv[1]
    opts = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
    gtloco.KEEP_PATH = True
    # the table is of the unconstrained candidate set; the selection rules are compared offline
    gtloco.ORTHO_TERM = False
    gtloco.RESID_IN_CAP = None
    if "rho" in opts:
        gtloco.LATTICE_RIDGE = float(opts["rho"])
    tag = opts.get("tag", "")
    G = len(gtloco.KAPPAS)
    (HERE / "cache").mkdir(exist_ok=True)
    for seed in [int(s) for s in opts.get("splits", "0").split(",")]:
        model, Xtr, Xte = split(name, seed)
        ptr, pte = model.predict(Xtr), model.predict(Xte)
        vtr, vte = np.var(ptr), np.var(pte)
        hfd = gtloco.GTLocoHFD(model)
        t0 = time.time()
        hfd.fit(Xtr)
        tfit = time.time() - t0
        d = hfd.diagnostics
        il = hfd.interaction_list
        rows = []
        for (mode, v, par, r, f) in d["candidates"]:
            for t, tree in enumerate(hfd.treehfd_list):
                g = (int(np.clip(d["per_tree_idx"][t, v] + par, 0, G - 1)) if mode == "shift"
                     else par)
                tree.finalize(g, v)
            res = [0 if mode == "shift" else 1, v, par, r / vtr, f / vtr]
            for X, p, var in ((Xtr, ptr, vtr), (Xte, pte, vte)):
                m, it = hfd.predict(X)
                res += [np.mean((p - hfd.eta0 - m.sum(1) - it.sum(1)) ** 2) / var,
                        ortho(p, m, it, il), omega(m, it, il) / var,
                        float(np.sum(np.var(it, axis=0)) / var)]
            rows.append(res)
        # columns: mode, variant, par, R, fid | resid_in, ortho_in, omega_in, inter_in |
        #          resid_out, ortho_out, omega_out, inter_out
        tab = np.array(rows)
        sel = d["selection"]
        isel = [i for i, c in enumerate(d["candidates"]) if c[:3] == sel[:3]][0]
        np.savez(HERE / "cache" / f"cand_{name}_s{seed}{tag}.npz", tab=tab,
                 isel=isel, vtr=vtr, tfit=tfit)
        r = tab[isel]
        print(f"{name} s{seed}: fit {tfit:.1f}s sel={sel[:3]} R={r[3]:.5f} in={r[5]:.5f} "
              f"out={r[9]:.5f} oin={r[6]:.4f} oout={r[10]:.4f} Om_in={r[7]:.2e} "
              f"Om_out={r[11]:.2e} inter={r[8]:.3f}", flush=True)


if __name__ == "__main__":
    main()
