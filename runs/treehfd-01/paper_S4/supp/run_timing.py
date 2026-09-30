"""R5(ii): repeat the fit-time measurement 5 times per dataset for TreeHFD and GT-LOCO (submitted method.py).

Same fit() entry points as the harness (bench/baseline_method.py and method/method.py), same model and X_train
(real(): harness split and seed). Runs sequentially in one process, alternating baseline / GT-LOCO within each repetition,
BLAS threads fixed to 4 for both. Time = wall clock of fit() only (not predict). Other jobs were not running.
usage: run_timing.py  ->  R5_timing.json, R5_timing.md
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import importlib.util
import sys
import time

import numpy as np

from common import DATASETS, ROOT, S4, SUPP, dump, real


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


base = load(ROOT / "bench/baseline_method.py", "base_m")
gt = load(S4 / "method/method.py", "gt_m")
R = 5
res = {}
order = ["concrete", "airfoil", "nutrition", "abalone", "powerplant", "parkinson", "bike", "housing", "superconduct"]
for ds in order:
    model, Xtr, _ = real(ds)
    tb, tg = [], []
    for r in range(R):
        t0 = time.perf_counter(); base.fit(model, Xtr); tb.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); gt.fit(model, Xtr); tg.append(time.perf_counter() - t0)
    ratios = [g / b for g, b in zip(tg, tb)]
    res[ds] = {"treehfd_s": tb, "gtloco_s": tg, "ratio_each_rep": ratios, "ratio_median_of_medians": float(np.median(tg) / np.median(tb)),
               "ratio_median": float(np.median(ratios)), "ratio_min": float(min(ratios)), "ratio_max": float(max(ratios))}
    print(ds, res[ds]["ratio_median"], res[ds]["ratio_min"], res[ds]["ratio_max"], flush=True)
    dump(res, SUPP / "R5_timing.json")
md = ["| dataset | TreeHFD fit_s median [min, max] | GT-LOCO fit_s median [min, max] | ratio of medians | per-rep ratio median [min, max] |", "|---|---|---|---|---|"]
for ds in DATASETS:
    r = res[ds]
    md.append(f"| {ds} | {np.median(r['treehfd_s']):.1f} [{min(r['treehfd_s']):.1f}, {max(r['treehfd_s']):.1f}] | "
              f"{np.median(r['gtloco_s']):.1f} [{min(r['gtloco_s']):.1f}, {max(r['gtloco_s']):.1f}] | {r['ratio_median_of_medians']:.2f} | "
              f"{r['ratio_median']:.2f} [{r['ratio_min']:.2f}, {r['ratio_max']:.2f}] |")
(SUPP / "R5_timing.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
