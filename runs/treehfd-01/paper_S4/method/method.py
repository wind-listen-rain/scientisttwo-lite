"""GT-LOCO TreeHFD: Good-Turing-calibrated leave-one-cell-out shrinkage of TreeHFD (see NOTES.md)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from gtloco import GTLocoHFD  # noqa: E402


def fit(model, X_train, interaction_order=2):
    hfd = GTLocoHFD(model)
    hfd.fit(X_train, interaction_order=interaction_order)
    return hfd


def predict(state, X):
    main, inter = state.predict(X)
    inter_list = state.interaction_list if inter.shape[1] else np.empty((0, 2), int)
    return {"intercept": state.eta0, "main": main, "inter": inter, "inter_list": inter_list}
