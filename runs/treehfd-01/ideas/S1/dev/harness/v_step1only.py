"""Dev only: ablation wrapper around S1's method.py with fit_projection arguments {'step1': True, 'close': False}."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("s1_method", Path(__file__).resolve().parents[2] / "method.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
predict = _m.predict


def fit(model, X_train, interaction_order=2):
    return _m.fit(model, X_train, interaction_order, proj_kw={'step1': True, 'close': False})
