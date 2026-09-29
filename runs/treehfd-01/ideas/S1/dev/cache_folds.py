"""Dev only: TreeHFD refitted on K-1 of K row folds of X_train (same fixed model), components on the held-out fold
(out-of-fold, "oof") and on the fold's own training rows ("inf"). Label-free. Tie-breaks seeded (fresh default_rng(0)).

usage: python dev/cache_folds.py K name [name ...]
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from treehfd_mod import XGBTreeHFD  # noqa: E402


def align(il_full, il_f, inter):
    out = np.zeros((inter.shape[0], len(il_full)))
    pos = {tuple(p): c for c, p in enumerate(np.asarray(il_full).reshape(-1, 2).tolist())}
    for c, p in enumerate(np.asarray(il_f).reshape(-1, 2).tolist()):
        out[:, pos[tuple(p)]] = inter[:, c]
    return out


if __name__ == "__main__":
    K = int(sys.argv[1])
    for name in sys.argv[2:]:
        d = pickle.load(open(HERE / "cache" / f"{name}.pkl", "rb"))
        model = xgb.XGBRegressor()
        model.load_model(HERE / "cache" / f"{name}_model.json")
        X = d["Xtr"]
        n, p = X.shape
        folds = np.arange(n) % K
        il = np.asarray(d["il"]).reshape(-1, 2)
        m_oof, i_oof = np.zeros((n, p)), np.zeros((n, len(il)))
        inf, times = [], []
        for f in range(K):
            tr, ho = folds != f, folds == f
            t0 = time.time()
            h = XGBTreeHFD(model)
            h.fit(X[tr], interaction_order=2, verbose=False)
            t1 = time.time()
            m, i = h.predict(X[ho], verbose=False)
            t2 = time.time()
            m_oof[ho] = m
            i_oof[ho] = align(il, h.interaction_list, i)
            mt, it = h.train_components
            inf.append(dict(rows=np.flatnonzero(tr), eta0=h.eta0, m=mt, i=align(il, h.interaction_list, it)))
            same = h.interaction_list.shape == il.shape and np.all(h.interaction_list == il)
            times.append((t1 - t0, t2 - t1))
            print(f"{name} fold {f}: fit {t1 - t0:.1f}s predict {t2 - t1:.1f}s  same pairs {same}", flush=True)
        with open(HERE / "cache" / f"{name}_folds{K}.pkl", "wb") as fh:
            pickle.dump(dict(K=K, m_oof=m_oof, i_oof=i_oof, inf=inf, times=times), fh)
        print(f"{name}: full fit {d['fit_s']:.1f}s, folds total {sum(a + b for a, b in times):.1f}s", flush=True)
