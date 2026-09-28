"""cProfile of a whole ensemble fit of a given implementation (diagnostic).
usage: prof_fit.py <impl.py> <dataset> [n_lines]"""
import cProfile, pstats, sys, importlib.machinery, importlib.util
from pathlib import Path
import numpy as np
E1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E1 / "lib")); sys.path.insert(0, str(E1 / "dev"))
from common import analytical, real
ld = importlib.machinery.SourceFileLoader("agt", sys.argv[1])
mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("agt", ld)); ld.exec_module(mod)
name = sys.argv[2]
model, Xtr, _ = analytical(0) if name.startswith("analytical") else real(name)
X = np.asarray(Xtr, float)
cProfile.run("mod.GTLocoHFD(model).fit(X)", "prof_fit.out")
st = pstats.Stats("prof_fit.out")
st.sort_stats("tottime").print_stats(int(sys.argv[3]) if len(sys.argv) > 3 else 25)
st.sort_stats("cumtime").print_stats(15)
