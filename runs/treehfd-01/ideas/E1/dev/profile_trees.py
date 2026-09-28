"""Profile the anchored GT-LOCO per-tree fit on a few trees of a dataset (diagnostic).

usage: profile_trees.py <dataset|analytical> <n_trees>
"""
import cProfile
import pstats
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/E1/lib")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/E1/dev")
import agtloco  # noqa: E402
from common import analytical, real  # noqa: E402

name, nt = sys.argv[1], int(sys.argv[2])
model, Xtr, Xte = analytical(0) if name == "analytical" else real(name)
ens = agtloco.GTLocoHFD(model)
X = np.asarray(Xtr, dtype=float)
Ttr = ens._tree_predict(X)
t0 = time.time()
X32 = agtloco._f32(X)
union = agtloco._union_splits(ens.xgb_table)
anchor = {"X32": X32, "shifts": agtloco._shift_values(X, X32, union), "n_shift": 2 * len(union)}
print(f"shifts {time.time() - t0:.2f}s")


def run():
    for t in range(nt):
        t0 = time.time()
        tree = agtloco.GTLocoTree(pd.DataFrame(ens.xgb_table[ens.xgb_table["Tree"] == t]), 2,
                                  ens.max_depth)
        _, _, _, n1f, m, nv = tree.fit(X, Ttr[:, t], anchor)
        print(f"tree {t}: m={m} N1/n={n1f:.3f} nv={nv} {time.time() - t0:.2f}s", flush=True)


PROF = str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/E1/dev/agtloco.prof"
cProfile.run("run()", PROF)
pstats.Stats(PROF).sort_stats("tottime").print_stats(20)
