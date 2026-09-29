"""Dev only: the baseline (bench/baseline_method.py) run on S1's package copy, whose only functional difference is the
seeded tie-break (TIE_SEED). Paired reference for S1's harness runs: same components before S1's transfer."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from treehfd_mod import XGBTreeHFD  # noqa: E402


def fit(model, X_train, interaction_order=2):
    hfd = XGBTreeHFD(model)
    hfd.fit(X_train, interaction_order=interaction_order, verbose=False)
    return hfd


def predict(state, X):
    main, inter = state.predict(X, verbose=False)
    inter_list = state.interaction_list if inter.shape[1] else np.empty((0, 2), int)
    return {"intercept": state.eta0, "main": main, "inter": inter, "inter_list": inter_list}
