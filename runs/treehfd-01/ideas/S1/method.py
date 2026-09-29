"""S1: TreeHFD + ensemble-level re-orthogonalization of the interactions against their parents' main effects.

TreeHFD is fitted exactly as in the baseline. Its aggregated main effects are piecewise constant on the union of the
ensemble's split thresholds. Each interaction's in-sample component along its parents' final main effects (least squares
under the empirical measure of X_train) is moved into them. The move is shrunk per pair by the empirical-Bayes factor
max(0, 1 - q / W), where W is the pair's robust Wald statistic. So only the part of the leak that is distinguishable
from row-sampling noise is transferred, and small main effects are not rescaled by noise.
The move is pointwise sum-preserving, so the reconstruction is the baseline's at every x. See lib/s1_projection.py.
Step 1 of the previous revisions (union-bin shape transfer) is still available for ablations (proj_kw={"step1": True}).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from s1_projection import apply_projection, fit_projection  # noqa: E402
from treehfd_mod import XGBTreeHFD  # noqa: E402


def fit(model, X_train, interaction_order=2, proj_kw=None):
    hfd = XGBTreeHFD(model)
    hfd.fit(X_train, interaction_order=interaction_order, verbose=False)
    state = {"hfd": hfd, "eta0": float(hfd.eta0), "proj": None, "diag": None}
    if hfd.interaction_list.shape[0] > 0:
        main, inter = hfd.train_components  # = hfd.predict(X_train), accumulated during the fit
        state["proj"], state["eta0"], state["diag"] = fit_projection(
            hfd.eta0, main, inter, hfd.interaction_list, X_train, hfd.xgb_table, **(proj_kw or {}))
    return state


def predict(state, X):
    hfd = state["hfd"]
    main, inter = hfd.predict(X, verbose=False)
    if state["proj"] is not None:
        main, inter = apply_projection(state["proj"], main, inter, X)
    inter_list = hfd.interaction_list if inter.shape[1] else np.empty((0, 2), int)
    return {"intercept": state["eta0"], "main": main, "inter": inter, "inter_list": inter_list}
