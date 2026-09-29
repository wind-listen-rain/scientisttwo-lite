"""Shared loader for the ablation method files.

Imports `lib/gtloco.py` and the submitted `method.py` (its `fit` / `predict`, unchanged) from
the S4 working directory. An ablation sets module-level switches of `gtloco` after importing
this; `GTLocoHFD` reads them at fit time, and every harness call runs in its own process, so
the switches never leak into the submitted method.
"""
import importlib.util
import sys
from pathlib import Path

S4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S4 / "lib"))
import gtloco  # noqa: E402

_spec = importlib.util.spec_from_file_location("gtloco_method", S4 / "method.py")
method = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(method)
assert method.GTLocoHFD is gtloco.GTLocoHFD  # same module object, so the switches apply
