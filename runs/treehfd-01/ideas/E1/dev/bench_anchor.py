"""Benchmark GTLocoTree._path_anchor / _virtual_atoms on captured inputs (diagnostic).

usage: bench_anchor.py <dataset> <first tree> <n trees>
Fits the given trees once (capturing the arguments of every anchored path), then re-times the
anchored path alone and reports the maximum difference of its outputs between two runs of the
current code (sanity) - used to compare implementations before switching.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

E1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E1 / "lib"))
sys.path.insert(0, str(E1 / "dev"))
import agtloco  # noqa: E402
from common import analytical, real  # noqa: E402

name, t0, nt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
model, Xtr, _ = analytical(0) if name == "analytical" else real(name)
T = agtloco.GTLocoTree
calls, tva = [], [0.0]
orig_pa, orig_va = T._path_anchor, T._virtual_atoms


def spy_pa(self, *a, **k):
    calls.append((self, a, k))
    return orig_pa(self, *a, **k)


def spy_va(self, *a, **k):
    s = time.perf_counter()
    out = orig_va(self, *a, **k)
    tva[0] += time.perf_counter() - s
    return out


T._path_anchor, T._virtual_atoms = spy_pa, spy_va
ens = agtloco.GTLocoHFD(model)
X = np.asarray(Xtr, dtype=float)
Ttr = ens._tree_predict(X)
X32 = agtloco._f32(X)
union = agtloco._union_splits(ens.xgb_table)
anchor = {"X32": X32, "shifts": agtloco._shift_values(X, X32, union), "n_shift": 2 * len(union)}
s = time.perf_counter()
for t in range(t0, t0 + nt):
    table = pd.DataFrame(ens.xgb_table[ens.xgb_table["Tree"] == t])
    tree = T(table, 2, ens.max_depth)
    tree.fit(X, Ttr[:, t], anchor)
print(f"{name} trees {t0}..{t0 + nt - 1}: fit {time.perf_counter() - s:.2f}s, "
      f"virtual atoms {tva[0]:.2f}s, {len(calls)} anchored paths")
sizes = []
for self, a, _ in calls[::2]:
    va = a[8]
    sizes += [1 + int(np.sum(r >= 0)) for _, vp, _ in va["blocks"] for r in vp]
sizes = np.array(sizes)
print(f"block sizes: count {len(sizes)} mean {sizes.mean():.2f} max {sizes.max()} "
      f"sum b^2 {int(np.sum(sizes ** 2))} padded sum b^2 "
      f"{int(sum(len(m) * (1 + vp.shape[1]) ** 2 for s_, a, _ in calls[::2] for m, vp, _ in a[8]['blocks']))}")
s = time.perf_counter()
outs = [orig_pa(self, *a, **k) for self, a, k in calls]
print(f"anchored paths (current code): {time.perf_counter() - s:.2f}s")
np.save(E1 / "dev" / "bench_ref.npy", np.concatenate([o[1].ravel() for o in outs]))
ref_file = E1 / "dev" / "bench_prev.npy"
if ref_file.exists():
    prev = np.load(ref_file)
    cur = np.concatenate([o[1].ravel() for o in outs])
    if prev.shape == cur.shape:
        print(f"max |loo - previous implementation's loo| = {np.max(np.abs(prev - cur)):.3e} "
              f"(scale {np.max(np.abs(prev)):.3e})")
