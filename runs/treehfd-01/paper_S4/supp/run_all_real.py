"""Driver: runs every real-data job of run_real.py with a small process pool (skips finished jobs)."""
import subprocess
import sys
import time
from pathlib import Path

PY = "D:/scientisttwo-lite/.conda/python.exe"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_real import CFGS  # noqa: E402

W = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DS = ["superconduct", "housing", "bike", "powerplant", "abalone", "nutrition", "parkinson", "concrete", "airfoil"]
size = {d: i for i, d in enumerate(DS)}
jobs = [(j, d) for d in DS for j in ["core", "ridge"] + [f"cfg:{c}" for c in CFGS]]
jobs = [(j, d) for j, d in jobs if not (HERE / f"raw/{j.replace(':', '_')}__{d}.json").exists()]
running = []
(HERE / "raw/logs").mkdir(exist_ok=True)
while jobs or running:
    running = [(p, n) for p, n in running if p.poll() is None]
    while jobs and len(running) < W:
        j, d = jobs.pop(0)
        name = f"{j.replace(':', '_')}__{d}"
        p = subprocess.Popen([PY, str(HERE / "run_real.py"), j, d], cwd=HERE,
                             stdout=open(HERE / f"raw/logs/{name}.log", "w"), stderr=subprocess.STDOUT)
        running.append((p, name))
        print(time.strftime("%H:%M:%S"), "start", name, flush=True)
    time.sleep(5)
print("ALL DONE", flush=True)
