"""Ablation no_ensemble_level_kappa_step: per-tree kappa argmin, no ensemble correction.

The ensemble step offers no kappa shift (SHIFTS = [0]) and no common kappa, so its only
candidates are the per-tree argmin of R under each (prior, rule) variant. The variant is picked
by the summed leave-out risk among those candidates. This is the "per-tree" configuration of
round 3 (dev/ablate_shift.py). The kappa grid, priors and rules are unchanged.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import gtloco, method  # noqa: E402

gtloco.SHIFTS = np.array([0])
gtloco.COMMON_KAPPA = False

fit = method.fit
predict = method.predict
