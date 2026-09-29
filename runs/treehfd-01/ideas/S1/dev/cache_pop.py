"""Dev only (diagnosis): analytical case, baseline TreeHFD components of the cached full fit's model at 100k fresh
*inputs* from the benchmark's input distribution N(0, Sigma) (no labels, no ground-truth components), to measure the
population leak of the interactions onto their parents' main effects."""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from treehfd_mod import XGBTreeHFD  # noqa: E402

N_POP = 100_000
for name in sys.argv[1:]:
    rep = int(name[-1])
    d = pickle.load(open(HERE / "cache" / f"{name}.pkl", "rb"))
    model = xgb.XGBRegressor()
    model.load_model(HERE / "cache" / f"{name}_model.json")
    t0 = time.time()
    h = XGBTreeHFD(model)
    h.fit(d["Xtr"], interaction_order=2, verbose=False)
    m_chk, i_chk = h.train_components
    assert np.allclose(m_chk, d["m_tr"]) and np.allclose(i_chk, d["i_tr"])
    cov = np.full((6, 6), 0.5); np.fill_diagonal(cov, 1.0)
    Xp = np.random.default_rng(9000 + rep).multivariate_normal(np.zeros(6), cov, size=N_POP)
    m, i = h.predict(Xp, verbose=False)
    with open(HERE / "cache" / f"{name}_pop.pkl", "wb") as fh:
        pickle.dump(dict(Xp=Xp, m=m, i=i, p=model.predict(Xp)), fh)
    print(name, f"{time.time() - t0:.0f}s", flush=True)
