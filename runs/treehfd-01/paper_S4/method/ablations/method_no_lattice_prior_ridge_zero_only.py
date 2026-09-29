"""Ablation no_lattice_prior_ridge_zero_only: plain ridge prior and zero unseen-cell rule only.

Removes the lattice (GMRF) prior and the harmonic-extension rule; the lattice branch never
runs. Kept as in method.py: the kappa grid, the leave-out risk R, the per-tree argmin and the
ensemble step (shifts and common kappa).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import gtloco, method  # noqa: E402

gtloco.PRIORS = ("ridge",)
gtloco.RULES = ("zero",)

fit = method.fit
predict = method.predict
