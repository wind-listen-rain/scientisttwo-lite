"""Dev only: rebuild the harness's models (same settings) and cache the fitted baseline TreeHFD state.
Labels are used here only to train the XGBoost model exactly as the harness does; the method never sees them."""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from treehfd_mod import XGBTreeHFD  # noqa: E402

DATA = Path("D:/scientisttwo-lite/bench/data")
XGB_PARAMS = dict(eta=0.1, n_estimators=100, max_depth=6, n_jobs=4)


def make(name):
    if name.startswith("analytical"):
        rep = int(name[-1])
        rho, dim, n = 0.5, 6, 5000
        cov = np.full((dim, dim), rho); np.fill_diagonal(cov, 1.0)
        rng_tr, rng_te = np.random.default_rng(1000 + rep), np.random.default_rng(2000 + rep)
        X = rng_tr.multivariate_normal(np.zeros(dim), cov, size=n)
        y = np.sin(2 * np.pi * X[:, 0]) + X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] + rng_tr.normal(0, 0.5, n)
        Xn = rng_te.multivariate_normal(np.zeros(dim), cov, size=n)
        model = xgb.XGBRegressor(random_state=rep, **XGB_PARAMS).fit(X, y)
        return model, X, Xn
    d = np.load(DATA / f"{name}.npz")
    X, y = d["X"], d["y"]
    perm = np.random.default_rng(0).permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = perm[:cut], perm[cut:]
    model = xgb.XGBRegressor(random_state=0, **XGB_PARAMS).fit(X[tr], y[tr])
    return model, X[tr], X[te]


if __name__ == "__main__":
    for name in sys.argv[1:]:
        model, Xtr, Xte = make(name)
        t0 = time.time()
        hfd = XGBTreeHFD(model)
        hfd.fit(Xtr, interaction_order=2, verbose=False)
        fit_s = time.time() - t0
        m_tr, i_tr = hfd.predict(Xtr, verbose=False)
        m_te, i_te = hfd.predict(Xte, verbose=False)
        model.save_model(HERE / "cache" / f"{name}_model.json")
        with open(HERE / "cache" / f"{name}.pkl", "wb") as f:
            pickle.dump(dict(Xtr=Xtr, Xte=Xte, eta0=hfd.eta0, il=hfd.interaction_list, m_tr=m_tr, i_tr=i_tr,
                             m_te=m_te, i_te=i_te, p_tr=model.predict(Xtr), p_te=model.predict(Xte),
                             table=hfd.xgb_table, fit_s=fit_s), f)
        print(name, Xtr.shape, "pairs", hfd.interaction_list.shape, f"fit {fit_s:.1f}s", flush=True)
