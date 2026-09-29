"""S1: TreeHFD + ensemble-level re-orthogonalization of the interactions on the union-of-splits partition.

TreeHFD is fitted exactly as in the baseline; then, for each interaction (j, k), its additive part on the union of the
ensemble's split thresholds (cross-fitted difference-penalised least squares, see lib/s1_projection.py) is moved into
the main effects eta_j, eta_k and the intercept. The move is pointwise sum-preserving, so the reconstruction is the
baseline's at every x.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from s1_projection import apply_projection, fit_projection  # noqa: E402
from treehfd_mod import XGBTreeHFD  # noqa: E402


def fit(model, X_train, interaction_order=2):
    hfd = XGBTreeHFD(model)
    hfd.fit(X_train, interaction_order=interaction_order, verbose=False)
    state = {"hfd": hfd, "eta0": float(hfd.eta0), "proj": None, "diag": None}
    if hfd.interaction_list.shape[0] > 0:
        main, inter = hfd.train_components  # = hfd.predict(X_train), accumulated during the fit
        state["proj"], state["eta0"], state["diag"] = fit_projection(
            hfd.eta0, main, inter, hfd.interaction_list, X_train, hfd.xgb_table)
    return state


def predict(state, X):
    hfd = state["hfd"]
    main, inter = hfd.predict(X, verbose=False)
    if state["proj"] is not None:
        main, inter = apply_projection(state["proj"], main, inter, X)
    inter_list = hfd.interaction_list if inter.shape[1] else np.empty((0, 2), int)
    return {"intercept": state["eta0"], "main": main, "inter": inter, "inter_list": inter_list}
