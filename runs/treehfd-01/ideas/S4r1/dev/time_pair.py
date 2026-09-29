"""Fit-time comparison, baseline TreeHFD vs a method file, run back to back (development).

usage: time_pair.py <method.py> <dataset|analytical> [...]
Each fit runs in a fresh subprocess through the harness's own runner (same model and X_train
as the harness), baseline first, then the method, so both see the same machine load.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import analytical, real  # noqa: E402

ROOT = HERE.parents[4]
RUNNER = ROOT / "bench" / "runner.py"
BASE = ROOT / "bench" / "baseline_method.py"


def fit_s(method, model, Xtr, Xev):
    with tempfile.TemporaryDirectory() as d:
        job = Path(d)
        model.save_model(job / "model.json")
        np.save(job / "X_train.npy", Xtr)
        np.save(job / "X_eval.npy", Xev)
        (job / "config.json").write_text(json.dumps({"interaction_order": 2}))
        subprocess.run([sys.executable, str(RUNNER), str(method), str(job)], check=True,
                       capture_output=True)
        return json.loads((job / "timing.json").read_text())["fit_s"]


def main():
    method = Path(sys.argv[1]).resolve()
    for name in sys.argv[2:]:
        model, Xtr, Xte = analytical(0) if name == "analytical" else real(name)
        tb = fit_s(BASE, model, Xtr, Xte)
        tm = fit_s(method, model, Xtr, Xte)
        print(f"{name:12s} baseline {tb:7.1f}s  method {tm:7.1f}s  ratio {tm / tb:5.2f}", flush=True)


if __name__ == "__main__":
    main()
