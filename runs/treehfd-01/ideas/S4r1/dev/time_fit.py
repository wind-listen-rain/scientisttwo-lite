"""Time a full GT-LOCO fit on one benchmark setup, with a per-stage breakdown (diagnostic).

usage: time_fit.py <dataset|analytical> [threads=<k>]
"""
import cProfile
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gtloco  # noqa: E402
from common import analytical, real  # noqa: E402

name = sys.argv[1]
opts = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
model, Xtr, Xte = analytical(0) if name == "analytical" else real(name)
if "threads" in opts:
    from threadpoolctl import threadpool_limits
    threadpool_limits(int(opts["threads"]))
hfd = gtloco.GTLocoHFD(model)
t0 = time.time()
prof = cProfile.Profile()
prof.enable()
hfd.fit(Xtr)
prof.disable()
print(f"{name}: fit {time.time() - t0:.2f}s  n={len(Xtr)}  m mean/max={sum(hfd.diagnostics['m']) / 100:.0f}/"
      f"{max(hfd.diagnostics['m'])}")
pstats.Stats(prof).sort_stats("tottime").print_stats(18)
