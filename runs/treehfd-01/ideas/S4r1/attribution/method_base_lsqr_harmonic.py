"""Attribution run base_lsqr_harmonic: baseline TreeHFD, SOLVER = "lsqr", RULE = "harmonic" (see _base.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _base  # noqa: E402

_base.SOLVER = "lsqr"
_base.RULE = "harmonic"

fit = _base.fit
predict = _base.predict
