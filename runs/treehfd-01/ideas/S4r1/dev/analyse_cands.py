"""Compare ensemble-step selection rules on the candidate tables (development only).

usage: analyse_cands.py [datasets=abalone,airfoil,concrete,nutrition] [splits=1,...,7]
Reads dev/cache/cand_<dataset>_s<seed>.npz (from cand_table.py). Every rule below uses only
label-free columns (R, in-sample residual, Omega_in, interaction variance on X_train, per-point
squared leave-out residuals). The held-out columns are only used to print what each rule would
have given, as geometric-mean ratios to the current rule (argmin R).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COLS = dict(mode=0, v=1, par=2, R=3, fid=4, resid_in=5, ortho_in=6, om_in=7, inter_in=8,
            resid_out=9, ortho_out=10, om_out=11, inter_out=12)
SHOW = ("resid_in", "resid_out", "ortho_in", "ortho_out", "inter_in")


def first_min(x):
    x = np.nan_to_num(np.asarray(x, float), nan=np.inf)
    return int(np.argmax(x <= x.min() * (1 + 1e-9) + 1e-300))


def rules(tab):
    R, fid, om = (tab[:, COLS[k]] for k in ("R", "fid", "om_in"))
    shift0 = (tab[:, 0] == 0) & (tab[:, 2] == 0)
    lat = tab[:, 1] >= 2
    i0 = first_min(R)
    out = {"argminR": i0, "R+Omega": first_min(R + om)}
    # (A one-standard-error rule towards parsimony was tried here and dropped: the per-point
    # squared leave-out residuals are heavy-tailed (r / (1 - h) with h near 1), so the paired
    # standard error admitted even the kappa = 1e4 candidates.)
    for c in (1.25, 1.5):
        ok = fid <= c * fid.min()
        out[f"cap{c}"] = int(np.flatnonzero(ok)[first_min(R[ok])])
    ok = np.flatnonzero(fid <= 1.5 * fid.min())
    out["cap1.5+Omega"] = int(ok[first_min((R + om)[ok])])      # round-4 method
    out["per_tree"] = int(np.flatnonzero(shift0)[first_min(R[shift0])])
    out["lattice_only"] = int(np.flatnonzero(lat)[first_min(R[lat])])
    return out


def main():
    opts = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
    names = opts.get("datasets", "abalone,airfoil,concrete,nutrition").split(",")
    splits = [int(s) for s in opts.get("splits", "1,2,3,4,5,6,7").split(",")]
    for name in names:
        per_rule = {}
        for s in [0, *splits]:
            f = HERE / "cache" / f"cand_{name}_s{s}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            tab = z["tab"]
            sel = rules(tab)
            assert sel["argminR"] == int(z["isel"]), (name, s)
            base = tab[sel["argminR"]]
            for r, i in sel.items():
                ratio = [tab[i, COLS[k]] / base[COLS[k]] for k in SHOW]
                per_rule.setdefault(r, {})[s] = (ratio, tab[i, :3])
        print(f"=== {name}: ratio to argmin R; geo-mean over splits {splits} | split 0 (benchmark)")
        print(f"{'rule':12s} " + " ".join(f"{k:>10s}" for k in SHOW) + " | "
              + " ".join(f"{k:>10s}" for k in SHOW) + " | split-0 choice")
        for r, d in per_rule.items():
            dev = np.array([d[s][0] for s in splits if s in d])
            gm = np.exp(np.nanmean(np.log(dev), axis=0))
            s0 = d.get(0, ([np.nan] * len(SHOW), None))
            ch = s0[1]
            print(f"{r:12s} " + " ".join(f"{x:10.3f}" for x in gm) + " | "
                  + " ".join(f"{x:10.3f}" for x in s0[0]) + " | "
                  + (f"{'shift' if ch[0] == 0 else 'common'} v{int(ch[1])} {int(ch[2]):+d}"
                     if ch is not None else ""))


if __name__ == "__main__":
    main()
