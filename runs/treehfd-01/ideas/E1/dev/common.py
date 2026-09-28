"""Development helpers: rebuild the harness setups (model + split) for local diagnostics.

Copied from S4/dev/common.py. `real(name, split)` with split > 0 gives the diagnostic-only
splits of S4 round 3 (split 0 is the benchmark split).
"""
import numpy as np
import xgboost as xgb

DATA = str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/bench/data"
XGB_PARAMS = dict(eta=0.1, n_estimators=100, max_depth=6, n_jobs=4)
RHO = 0.5


def real(name, split=0):
    d = np.load(f"{DATA}/{name}.npz")
    X, y = d["X"], d["y"]
    perm = np.random.default_rng(split).permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = perm[:cut], perm[cut:]
    model = xgb.XGBRegressor(random_state=0, **XGB_PARAMS).fit(X[tr], y[tr])
    return model, X[tr], X[te]


def analytical(rep=0):
    mu, cov = np.zeros(6), np.full((6, 6), RHO)
    np.fill_diagonal(cov, 1.0)
    rng_tr, rng_te = np.random.default_rng(1000 + rep), np.random.default_rng(2000 + rep)
    X = rng_tr.multivariate_normal(mu, cov, size=5000)
    y = np.sin(2 * np.pi * X[:, 0]) + X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] + rng_tr.normal(0, 0.5, 5000)
    Xn = rng_te.multivariate_normal(mu, cov, size=5000)
    model = xgb.XGBRegressor(random_state=rep, **XGB_PARAMS).fit(X, y)
    return model, X, Xn


def resid(model, out, X):
    p = model.predict(X)
    rec = out["intercept"] + out["main"].sum(1) + out["inter"].sum(1)
    return float(np.mean((p - rec) ** 2) / np.var(p))
