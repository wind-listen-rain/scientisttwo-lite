"""Ablation soft_orthogonality_rows: baseline soft orthogonality rows instead of exact ones.

HARD_ORTHO = False: the pair coefficients are no longer restricted to the null space of their
orthogonality rows. The baseline's weighted rows (p_cell / sqrt(p_bin)) are added to the
normal matrix instead, and are traded off against the fit and the shrinkage penalty. All
else is kept, including the kappa grid, priors, rules and ensemble step.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import gtloco, method  # noqa: E402

gtloco.HARD_ORTHO = False

fit = method.fit
predict = method.predict
