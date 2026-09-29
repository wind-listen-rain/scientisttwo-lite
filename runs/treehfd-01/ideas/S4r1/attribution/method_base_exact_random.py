"""Attribution run base_exact_random: baseline TreeHFD, SOLVER = "exact", RULE = "random" (see _base.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _base  # noqa: E402

_base.SOLVER = "exact"
_base.RULE = "random"

fit = _base.fit
predict = _base.predict
