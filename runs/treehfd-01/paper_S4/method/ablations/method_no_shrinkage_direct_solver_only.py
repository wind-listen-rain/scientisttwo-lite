"""Ablation no_shrinkage_direct_solver_only: GT-LOCO without the self-supervised shrinkage.

The kappa grid is only its smallest value (1e-2 pseudo-counts, near-unregularised) and the
ensemble step has shift 0 only, so every tree takes the near-interpolating solution of the
direct eigendecomposition solver. Kept as in method.py: vectorised construction, exact
orthogonality, deterministic unseen-cell rule, and the (prior, rule) variant choice by the
ensemble leave-out risk. At kappa = 0.01 that choice mainly picks the rule; the prior acts
only on weakly identified directions.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import gtloco, method  # noqa: E402

gtloco.KAPPAS = gtloco.KAPPAS[:1].copy()
gtloco.SHIFTS = np.array([0])

fit = method.fit
predict = method.predict
