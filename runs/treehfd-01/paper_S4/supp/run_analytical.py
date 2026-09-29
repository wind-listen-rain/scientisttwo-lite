"""Analytical benchmark case (harness protocol, rep r = 0..9): per-rep metrics of TreeHFD, GT-LOCO and the R2 variants.

usage: run_analytical.py <rep>      ->  raw/analytical__<rep>.json
Uses the harness' samples, model seeds and metric definitions (component targets are only used to SCORE,
never inside any decomposition).
"""
import sys

from common import (SUPP, SweepHFD, analytical_data, analytical_metrics, baseline_fit, dump,
                    out_from_baseline, purify, set_config)

rep = int(sys.argv[1])
model, X, Xn = analytical_data(rep)
res = {"rep": rep}
set_config()
hb, tb = baseline_fit(model, X)
ob = out_from_baseline(hb, X, Xn)
res["treehfd"] = analytical_metrics(model, X, Xn, ob)
res["purified_treehfd"] = analytical_metrics(model, X, Xn, purify(ob, X, Xn))
sw = SweepHFD(model)
sw.fit(X)


def ev(c):
    sw.apply(c)
    return analytical_metrics(model, X, Xn, sw.out(X, Xn)) | {"cand": [c[0], int(c[1]), float(c[2])]}


res["gtloco"] = ev(sw.choose())
res["lattice_common"] = ev(sw.choose(shift_cands=False))
set_config(priors=("ridge",), rules=("zero", "harmonic"))
sw = SweepHFD(model)
sw.fit(X)
res["ridge_zero"] = ev(sw.choose(variants=[0], shift_cands=False))
res["ridge_harmonic"] = ev(sw.choose(variants=[1], shift_cands=False))
dump(res, SUPP / f"raw/analytical__{rep}.json")
print("done", rep)
