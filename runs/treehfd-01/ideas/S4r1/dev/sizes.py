"""Problem sizes per tree (main bins, pair cells, pairs, lattice sizes) for a dataset (diagnostic)."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import analytical, real  # noqa: E402
from treehfd_mod import XGBTreeHFD  # noqa: E402
from treehfd_mod.cartesian_partition import CartesianTreePartition  # noqa: E402
from treehfd_mod.tree_structure import (  # noqa: E402
    extract_interactions,
    extract_tree_structure,
    extract_variable_paths,
    extract_variables,
)

name = sys.argv[1]
model, Xtr, Xte = analytical(0) if name == "analytical" else real(name)
ens = XGBTreeHFD(model)
rows = []
for t in range(ens.n_estimators):
    ts = extract_tree_structure(pd.DataFrame(ens.xgb_table[ens.xgb_table["Tree"] == t]))
    if len(ts[0]) == 0:
        continue
    paths = extract_variable_paths(ts, ens.max_depth)
    mv = extract_variables(paths)
    il = extract_interactions(paths)
    part = CartesianTreePartition(mv)
    glob = part.compute_partition_main(Xtr, ts[0], ts[2])
    nb = np.diff(part.partition_index)
    lb = glob - part.partition_index[:-1]
    mvl = list(mv)
    ncell, nlat = 0, 0
    for a, b in il:
        ia, ib = mvl.index(a), mvl.index(b)
        ncell += len(np.unique(lb[:, ia] * nb[ib] + lb[:, ib]))
        nlat = max(nlat, nb[ia] * nb[ib])
    rows.append((len(mv), len(il), int(nb.sum()), ncell, int(nb.sum()) + ncell, nlat))
R = np.array(rows)
print("n", len(Xtr), "trees", len(R))
print("cols: V I main_bins pair_cells m max_lattice")
for q in (0, 0.5, 0.9, 1.0):
    print(q, np.quantile(R, q, axis=0))
print("sum m^3 / 1e9:", np.sum(R[:, 4].astype(float) ** 3) / 1e9)
