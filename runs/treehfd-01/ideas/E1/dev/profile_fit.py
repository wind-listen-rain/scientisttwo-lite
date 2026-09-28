"""Profile a whole anchored GT-LOCO fit (diagnostic).

usage: profile_fit.py <dataset|analytical> [n_lines]
"""
import cProfile
import pstats
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/E1/lib")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/E1/dev")
import agtloco  # noqa: E402
from common import analytical, real  # noqa: E402

name = sys.argv[1]
nl = int(sys.argv[2]) if len(sys.argv) > 2 else 25
model, Xtr, Xte = analytical(0) if name == "analytical" else real(name)
hfd = agtloco.GTLocoHFD(model)
t0 = time.time()
PROF = str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/E1/dev/fit.prof"
cProfile.run("hfd.fit(Xtr)", PROF)
print(f"fit {time.time() - t0:.2f}s  m mean {sum(hfd.diagnostics['m']) / 100:.0f} "
      f"max {max(hfd.diagnostics['m'])}")
pstats.Stats(PROF).sort_stats("tottime").print_stats(nl)
