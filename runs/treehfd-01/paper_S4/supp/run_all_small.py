"""Driver for the analytical jobs: run_analytical.py (10 reps) and run_new_case.py (10 reps)."""
import subprocess
import sys
import time
from pathlib import Path

PY = "D:/scientisttwo-lite/.conda/python.exe"
HERE = Path(__file__).resolve().parent
W = int(sys.argv[1]) if len(sys.argv) > 1 else 3
jobs = [("run_analytical.py", r, f"analytical__{r}") for r in range(10)] + \
       [("run_new_case.py", r, f"newcase__{r}") for r in range(10)]
jobs = [j for j in jobs if not (HERE / f"raw/{j[2]}.json").exists()]
running = []
while jobs or running:
    running = [p for p in running if p.poll() is None]
    while jobs and len(running) < W:
        s, r, n = jobs.pop(0)
        running.append(subprocess.Popen([PY, str(HERE / s), str(r)], cwd=HERE,
                                        stdout=open(HERE / f"raw/logs/{n}.log", "w"), stderr=subprocess.STDOUT))
        print(time.strftime("%H:%M:%S"), "start", n, flush=True)
    time.sleep(5)
print("ALL DONE", flush=True)
