"""Attribution run gtloco_k0_ridge_harmonic: GT-LOCO in the kappa -> 0 limit (no effective shrinkage).

KAPPAS = [1e-4] (a hundredth of the method's smallest grid value), no ensemble shift or common
kappa, orthogonality term and resid_in cap off, and a single variant: prior "ridge", rule
"harmonic". What remains is the direct solver with exact orthogonality, the prior only as a
tie-breaker among (nearly) equally good fits, and the deterministic unseen-cell rule.
"""
import sys
from pathlib import Path

import numpy as np

S4R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S4R1 / "lib"))
import gtloco  # noqa: E402

gtloco.KAPPAS = np.array([1e-4])
gtloco.SHIFTS = np.array([0])
gtloco.COMMON_KAPPA = False
gtloco.ORTHO_TERM = False
gtloco.RESID_IN_CAP = None
gtloco.PRIORS = ("ridge",)
gtloco.RULES = ("harmonic",)


def fit(model, X_train, interaction_order=2):
    hfd = gtloco.GTLocoHFD(model)
    hfd.fit(X_train, interaction_order=interaction_order)
    return hfd


def predict(state, X):
    main, inter = state.predict(X)
    inter_list = state.interaction_list if inter.shape[1] else np.empty((0, 2), int)
    return {"intercept": state.eta0, "main": main, "inter": inter, "inter_list": inter_list}
