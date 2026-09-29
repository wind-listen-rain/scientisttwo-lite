"""Dev only (diagnosis, label-free): sampling variability of the step-2 leak coefficient of the big pairs.
HC0 sandwich SE and row-bootstrap SE in-sample; row-bootstrap SE on the eval rows and on each held-out fold."""
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def coef(M, y):
    Z = M - M.mean(0)
    ok = Z.var(0) > 0
    b = np.zeros(M.shape[1])
    b[ok] = np.linalg.lstsq(Z[:, ok], y - y.mean(), rcond=None)[0]
    return b


def hc0(M, y):
    Z = M - M.mean(0)
    b = coef(M, y)
    e = y - y.mean() - Z @ b
    br = np.linalg.pinv(Z.T @ Z)
    return np.sqrt(np.diag(br @ ((Z * (e * e)[:, None]).T @ Z) @ br))


def boot(M, y, B=300):
    rng = np.random.default_rng(0)
    n = len(y)
    return np.std([coef(M[r], y[r]) for r in (rng.integers(0, n, n) for _ in range(B))], 0)


if __name__ == "__main__":
    K = int(sys.argv[1])
    for name in sys.argv[2:]:
        d = pickle.load(open(HERE / "cache" / f"{name}.pkl", "rb"))
        fd = pickle.load(open(HERE / "cache" / f"{name}_folds{K}.pkl", "rb"))
        il = np.asarray(d["il"]).reshape(-1, 2)
        vt = np.var(d["p_tr"])
        folds = np.arange(len(d["Xtr"])) % K
        print(f"== {name}")
        for c, (j, k) in enumerate(il):
            if np.var(d["i_tr"][:, c]) < 0.01 * vt:
                continue
            Mi, yi = d["m_tr"][:, [j, k]], d["i_tr"][:, c]
            Me, ye = d["m_te"][:, [j, k]], d["i_te"][:, c]
            print(f"  ({j},{k}) in {np.round(coef(Mi, yi), 4)} hc0 {np.round(hc0(Mi, yi), 4)} boot {np.round(boot(Mi, yi), 4)}"
                  f" | eval {np.round(coef(Me, ye), 4)} boot {np.round(boot(Me, ye), 4)}")
            for f in range(K):
                Mo, yo = fd["m_oof"][folds == f][:, [j, k]], fd["i_oof"][folds == f][:, c]
                print(f"        fold {f} ho {np.round(coef(Mo, yo), 4)} boot {np.round(boot(Mo, yo), 4)}")
