"""Model-anchored smoothed-measure TreeHFD (idea S2). See NOTES.md."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from smoothed_hfd import SmoothedTreeHFD  # noqa: E402


def fit(model, X_train, interaction_order=2):
    return SmoothedTreeHFD(model).fit(np.asarray(X_train), interaction_order=interaction_order)


def predict(state, X):
    main, inter = state.predict(np.asarray(X))
    return {"intercept": state.eta0, "main": main, "inter": inter,
            "inter_list": state.interaction_list}
