"""Attribution run base_exact_harmonic: baseline TreeHFD, SOLVER = "exact", RULE = "harmonic" (see _base.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _base  # noqa: E402

_base.SOLVER = "exact"
_base.RULE = "harmonic"

fit = _base.fit
predict = _base.predict
