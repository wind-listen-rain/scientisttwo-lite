"""LATTICE_RIDGE comparison on the diagnostic splits (development only).

usage: analyse_rho.py [datasets=...] [splits=1,...,7] [rhos=0.01,0.1,1.0]
For each rho, applies the round-4 selection (in-sample residual cap 1.5, argmin R; Omega is
left out because it only breaks near-ties) to the candidate tables of cand_table.py (rho = 0.1
is the untagged table) and prints geometric-mean ratios to rho = 0.1: the label-free R of the
selected candidate first, then the held-out numbers (diagnosis only).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COLS = dict(R=3, resid_in=5, ortho_in=6, resid_out=9, ortho_out=10, inter_in=8)


def first_min(x):
    x = np.nan_to_num(np.asarray(x, float), nan=np.inf)
    return int(np.argmax(x <= x.min() * (1 + 1e-9) + 1e-300))


def select(tab, cap=1.5):
    ok = np.flatnonzero(tab[:, 4] <= cap * tab[:, 4].min())
    return tab[ok[first_min(tab[ok, 3])]]


def main():
    opts = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
    names = opts.get("datasets", "abalone,airfoil,concrete,nutrition").split(",")
    splits = [int(s) for s in opts.get("splits", "1,2,3,4,5,6,7").split(",")]
    rhos = opts.get("rhos", "0.01,0.1,1.0").split(",")
    keys = list(COLS)
    print(f"{'dataset':10s} {'rho':>5s} " + " ".join(f"{k:>10s}" for k in keys)
          + "   (geo-mean ratio to rho=0.1; R is label-free)")
    for name in names:
        ref = {}
        for s in splits:
            ref[s] = select(np.load(HERE / "cache" / f"cand_{name}_s{s}.npz")["tab"])
        for rho in rhos:
            tag = "" if rho == "0.1" else f"_rho{rho}"
            rat = []
            for s in splits:
                f = HERE / "cache" / f"cand_{name}_s{s}{tag}.npz"
                if not f.exists():
                    continue
                r = select(np.load(f)["tab"])
                rat.append([r[COLS[k]] / ref[s][COLS[k]] for k in keys])
            gm = np.exp(np.nanmean(np.log(np.array(rat)), axis=0))
            print(f"{name:10s} {rho:>5s} " + " ".join(f"{x:10.3f}" for x in gm)
                  + f"   n={len(rat)}")


if __name__ == "__main__":
    main()
