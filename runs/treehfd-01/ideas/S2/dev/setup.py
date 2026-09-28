"""Dev helper: rebuild the harness's (model, X_train, X_eval) for a dataset. Testing only."""
import numpy as np
import xgboost as xgb

DATA = str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/bench/data"
XGB_PARAMS = dict(eta=0.1, n_estimators=100, max_depth=6, n_jobs=4)


def real(name):
    d = np.load(f"{DATA}/{name}.npz")
    X, y = d["X"], d["y"]
    perm = np.random.default_rng(0).permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = perm[:cut], perm[cut:]
    model = xgb.XGBRegressor(random_state=0, **XGB_PARAMS).fit(X[tr], y[tr])
    return model, X[tr], X[te]


def analytical(rep=0, dim=6, n=5000, rho=0.5):
    mu, cov = np.zeros(dim), np.full((dim, dim), rho)
    np.fill_diagonal(cov, 1.0)
    rng_tr, rng_te = np.random.default_rng(1000 + rep), np.random.default_rng(2000 + rep)
    X = rng_tr.multivariate_normal(mu, cov, size=n)
    y = np.sin(2 * np.pi * X[:, 0]) + X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] + rng_tr.normal(0, 0.5, n)
    Xn = rng_te.multivariate_normal(mu, cov, size=n)
    model = xgb.XGBRegressor(random_state=rep, **XGB_PARAMS).fit(X, y)
    return model, X, Xn


def resid(model, out_fn, X):
    p = model.predict(X)
    r = out_fn(X)
    rec = r["intercept"] + r["main"].sum(1) + r["inter"].sum(1)
    return float(np.mean((p - rec) ** 2) / np.var(p))
