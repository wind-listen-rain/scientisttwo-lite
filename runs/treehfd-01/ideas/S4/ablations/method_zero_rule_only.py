"""Optional sub-ablation of no_lattice_prior_ridge_zero_only: zero unseen-cell rule only.

Keeps both priors (ridge and lattice), but RULES = ("zero",), so the harmonic extension is
never used for unseen pair cells. Compared with no_lattice_prior_ridge_zero_only, this
separates the unseen-cell rule from the lattice prior. Everything else is as in method.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import gtloco, method  # noqa: E402

gtloco.RULES = ("zero",)

fit = method.fit
predict = method.predict
