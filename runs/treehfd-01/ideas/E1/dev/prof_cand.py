"""cProfile of GTLocoTree.fit of a given implementation (diagnostic).
usage: prof_cand.py <impl.py> <dataset> <first_tree> <n_trees>"""
import cProfile, pstats, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
E1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E1 / "lib")); sys.path.insert(0, str(E1 / "dev"))
from common import analytical, real
loader = importlib.machinery.SourceFileLoader("agt", sys.argv[1])
spec = importlib.util.spec_from_loader("agt", loader); mod = importlib.util.module_from_spec(spec); loader.exec_module(mod)
name, t0, nt = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
model, Xtr, _ = analytical(0) if name == "analytical" else real(name)
X = np.asarray(Xtr, float)
ens = mod.GTLocoHFD(model); Ttr = ens._tree_predict(X); X32 = mod._f32(X)
union = mod._union_splits(ens.xgb_table)
anchor = {"X32": X32, "shifts": mod._shift_values(X, X32, union), "n_shift": 2 * len(union)}
def run():
    for t in range(t0, t0 + nt):
        table = pd.DataFrame(ens.xgb_table[ens.xgb_table["Tree"] == t])
        tree = mod.GTLocoTree(table, 2, ens.max_depth); tree.fit(X, Ttr[:, t], anchor)
cProfile.run("run()", "prof.out")
pstats.Stats("prof.out").sort_stats("tottime").print_stats(18)
