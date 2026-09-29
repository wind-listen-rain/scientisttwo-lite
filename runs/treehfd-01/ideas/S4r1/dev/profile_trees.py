"""Profile GT-LOCO per-tree fit on a few trees of a dataset (diagnostic).

usage: profile_trees.py <dataset> <n_trees>
"""
import cProfile
import pstats
import sys
import time

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import gtloco  # noqa: E402
from common import analytical, real  # noqa: E402

name, nt = sys.argv[1], int(sys.argv[2])
model, Xtr, Xte = analytical(0) if name == "analytical" else real(name)
ens = gtloco.GTLocoHFD(model)
Ttr = ens._tree_predict(Xtr)


def run():
    for t in range(nt):
        t0 = time.time()
        tree = gtloco.GTLocoTree(pd.DataFrame(ens.xgb_table[ens.xgb_table["Tree"] == t]), 2,
                                 ens.max_depth)
        _, _, _, n1f, m = tree.fit(Xtr, Ttr[:, t])
        print(f"tree {t}: m={m} N1/n={n1f:.3f} {time.time() - t0:.2f}s", flush=True)


PROF = str(__import__("pathlib").Path(__file__).resolve().parent / "gtloco.prof")
cProfile.run("run()", PROF)
pstats.Stats(PROF).sort_stats("cumulative").print_stats(25)
