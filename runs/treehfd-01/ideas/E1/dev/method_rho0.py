"""Check (7a): method.py with RHOS = (0,) and SE_RULE = 0 (S4's argmin), which must reproduce S4
exactly."""
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/E1/lib")
import agtloco  # noqa: E402

agtloco.RHOS = (0.0,)
agtloco.SE_RULE = 0.0


def fit(model, X_train, interaction_order=2):
    hfd = agtloco.GTLocoHFD(model)
    hfd.fit(X_train, interaction_order=interaction_order)
    return hfd


def predict(state, X):
    main, inter = state.predict(X)
    inter_list = state.interaction_list if inter.shape[1] else np.empty((0, 2), int)
    return {"intercept": state.eta0, "main": main, "inter": inter, "inter_list": inter_list}
