"""Dev: method.py with a non-default NEW_PAIR_CELLS (ablation runs through the official harness)."""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import smoothed_hfd  # noqa: E402

VARIANT = "fit"  # the "fit" ablation; the default (method.py) is "drop"
smoothed_hfd.NEW_PAIR_CELLS = VARIANT
os.environ.setdefault("S2_DIAG_PATH", str(HERE / f"diag_{VARIANT}_subset.jsonl"))


def fit(model, X_train, interaction_order=2):
    return smoothed_hfd.SmoothedTreeHFD(model).fit(np.asarray(X_train),
                                                   interaction_order=interaction_order)


def predict(state, X):
    main, inter = state.predict(np.asarray(X))
    return {"intercept": state.eta0, "main": main, "inter": inter,
            "inter_list": state.interaction_list}
