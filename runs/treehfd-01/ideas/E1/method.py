"""Anchored GT-LOCO TreeHFD (idea E1): S4's GT-LOCO with a model-anchored Gaussian prior on
virtual atoms (the fixed tree's exact outputs at neighbour-bin copies), see NOTES.md."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from agtloco import GTLocoHFD  # noqa: E402


def fit(model, X_train, interaction_order=2):
    hfd = GTLocoHFD(model)
    hfd.fit(X_train, interaction_order=interaction_order)
    return hfd


def predict(state, X):
    main, inter = state.predict(X)
    inter_list = state.interaction_list if inter.shape[1] else np.empty((0, 2), int)
    return {"intercept": state.eta0, "main": main, "inter": inter, "inter_list": inter_list}
