"""R4: a second analytical test case, OUTSIDE the fixed benchmark (supplementary diagnostic only).

Specified before any result was looked at:
  p = 6, X ~ N(0, Sigma), Sigma_ij = 0.6^|i-j| (AR(1)), n_train = n_test = 5000 (fresh samples per seed),
  f(x) = x1 + 0.8 x2^2 + 0.5 x3^3 + 1.5 x4 x5   (x6 does not enter f), Y = f(X) + N(0, 0.5^2),
  XGBoost with the benchmark hyper-parameters (eta 0.1, 100 trees, depth 6), seeds rep = 0..9:
  train rng 3000+rep, test rng 4000+rep, random_state = rep.
Ground-truth HFD (mains and all 15 pairs) is computed exactly by gt_poly.PolyHFD (validated on the
benchmark case, where it reproduces the paper's closed forms to 1e-13). The truth is used only to SCORE.
usage: run_new_case.py <rep>   ->  raw/newcase__<rep>.json
"""
import sys

import numpy as np
import xgboost as xgb

from common import (SUPP, SweepHFD, XGB_PARAMS, baseline_fit, dump, harness, out_from_baseline,
                    real_metrics, set_config)
from gt_poly import PolyHFD

P = 6
COV = 0.6 ** np.abs(np.subtract.outer(np.arange(P), np.arange(P)))


def mono(*pairs):
    e = [0] * P
    for j, a in pairs:
        e[j] = a
    return tuple(e)


F = {mono((0, 1)): 1.0, mono((1, 2)): 0.8, mono((2, 3)): 0.5, mono((3, 1), (4, 1)): 1.5}


def f(X):
    return X[:, 0] + 0.8 * X[:, 1] ** 2 + 0.5 * X[:, 2] ** 3 + 1.5 * X[:, 3] * X[:, 4]


def score(model, X, Xn, o, truth):
    tm, tp = truth.components(Xn)
    main, inter = o.te
    il = [tuple(q) for q in o.il]
    est = np.zeros_like(tp)
    for c, q in enumerate(il):
        est[:, truth.pairs.index(q)] = inter[:, c]
    row = {f"mse_eta{j + 1}": float(np.mean((tm[:, j] - main[:, j]) ** 2)) for j in range(P)}
    row["mse_main_mean"] = float(np.mean([row[f"mse_eta{j + 1}"] for j in range(P)]))
    k45 = truth.pairs.index((3, 4))
    row["mse_eta45"] = float(np.mean((tp[:, k45] - est[:, k45]) ** 2))
    oth = [k for k in range(len(truth.pairs)) if k != k45]
    row["mse_others_vs_truth"] = float(np.mean((tp[:, oth] - est[:, oth]) ** 2))
    row["mse_others_zero"] = float(np.mean(est[:, oth] ** 2))     # estimate^2, benchmark-style (target 0)
    row.update(real_metrics(model, X, Xn, o))
    row["var_eta45_true"] = float(np.var(tp[:, k45]))
    row["var_others_true_max"] = float(np.max(np.var(tp[:, oth], axis=0)))
    return row


if __name__ == "__main__":
    rep = int(sys.argv[1])
    truth = PolyHFD(COV, F, deg=3)
    assert truth.rank == truth.nun and truth.residual < 1e-9
    rng_tr, rng_te = np.random.default_rng(3000 + rep), np.random.default_rng(4000 + rep)
    X = rng_tr.multivariate_normal(np.zeros(P), COV, size=5000)
    y = f(X) + rng_tr.normal(0, 0.5, 5000)
    Xn = rng_te.multivariate_normal(np.zeros(P), COV, size=5000)
    model = xgb.XGBRegressor(random_state=rep, **XGB_PARAMS).fit(X, y)
    res = {"rep": rep, "xgb_r2_test": float(1 - np.mean((f(Xn) + 0 - model.predict(Xn)) ** 2) / np.var(f(Xn)))}
    set_config()
    hb, _ = baseline_fit(model, X)
    res["treehfd"] = score(model, X, Xn, out_from_baseline(hb, X, Xn), truth)
    sw = SweepHFD(model)
    sw.fit(X)
    sw.apply(sw.choose())
    res["gtloco"] = score(model, X, Xn, sw.out(X, Xn), truth)
    dump(res, SUPP / f"raw/newcase__{rep}.json")
    print("done", rep)
