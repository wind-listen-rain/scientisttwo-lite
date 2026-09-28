"""Compare two implementations of lib/agtloco.py tree by tree (outputs and time).

usage: check_speedup.py <reference.py> <candidate.py> <dataset|analytical> [first_tree] [n_trees]
                        [end2end]
Per tree, both modules fit the same tree on the same inputs; the script reports the maximum
absolute difference of every returned array (leave-out and in-sample residuals, risks) and of
the coefficient paths, and the fit time of each. With 'end2end' it also fits the whole
ensemble with both modules and compares the selection and the predictions on X_train.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

E1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E1 / "lib"))
sys.path.insert(0, str(E1 / "dev"))
from common import analytical, real  # noqa: E402


def load(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


ref = load(sys.argv[1], "agt_ref")
cand = load(sys.argv[2], "agt_cand")
for mod in (ref, cand):
    mod.KEEP_PATH = True
name = sys.argv[3]
t_first = int(sys.argv[4]) if len(sys.argv) > 4 else 0
nt = int(sys.argv[5]) if len(sys.argv) > 5 else 5
model, Xtr, Xte = analytical(0) if name == "analytical" else real(name)
X = np.asarray(Xtr, dtype=float)


def setup(mod):
    ens = mod.GTLocoHFD(model)
    Ttr = ens._tree_predict(X)
    X32 = mod._f32(X)
    union = mod._union_splits(ens.xgb_table)
    anchor = {"X32": X32, "shifts": mod._shift_values(X, X32, union), "n_shift": 2 * len(union)}
    return ens, Ttr, anchor


setups = [setup(m) for m in (ref, cand)]
tot = [0.0, 0.0]
worst = {}
for t in range(t_first, t_first + nt):
    outs, trees = [], []
    for k, (mod, (ens, Ttr, anchor)) in enumerate(zip((ref, cand), setups)):
        table = pd.DataFrame(ens.xgb_table[ens.xgb_table["Tree"] == t])
        tree = mod.GTLocoTree(table, 2, ens.max_depth)
        s = time.perf_counter()
        outs.append(tree.fit(X, Ttr[:, t], anchor))
        tot[k] += time.perf_counter() - s
        trees.append(tree)
    diffs = {}
    for key, a, b in zip(("loo", "ins", "risk"), outs[0][:3], outs[1][:3]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        diffs[key] = float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300)) if a.size else 0.0
    if trees[0].betas.size:
        a, b = trees[0].betas, trees[1].betas
        diffs["betas"] = float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))
    same_arg = bool(np.array_equal(ref._first_min(outs[0][2], axis=2),
                                   cand._first_min(outs[1][2], axis=2)))
    for key, v in diffs.items():
        worst[key] = max(worst.get(key, 0.0), v)
    print(f"tree {t}: m={outs[0][4]} nv={outs[0][5]} rel.diff "
          + " ".join(f"{k}={v:.1e}" for k, v in diffs.items())
          + f" same per-tree argmin={same_arg}", flush=True)
print(f"{name} trees {t_first}..{t_first + nt - 1}: fit reference {tot[0]:.2f}s, "
      f"candidate {tot[1]:.2f}s (x{tot[0] / max(tot[1], 1e-9):.2f}); worst rel.diff "
      + " ".join(f"{k}={v:.1e}" for k, v in worst.items()))

if "end2end" in sys.argv[4:]:
    res = []
    for mod in (ref, cand):
        mod.KEEP_PATH = False
        hfd = mod.GTLocoHFD(model)
        s = time.perf_counter()
        hfd.fit(X)
        el = time.perf_counter() - s
        main, inter = hfd.predict(X)
        mo, io = hfd.predict(np.asarray(Xte, float))
        res.append((hfd, el, main, inter, mo, io))
        print(f"end2end {mod.__name__}: fit {el:.2f}s selection {hfd.diagnostics['selection'][:4]}",
              flush=True)
    a, b = res
    print("same selection:", a[0].diagnostics["selection"][:4] == b[0].diagnostics["selection"][:4],
          "same kappa/rho per tree:",
          bool(np.array_equal(a[0].diagnostics["kappa_chosen"], b[0].diagnostics["kappa_chosen"])
               and np.array_equal(a[0].diagnostics["rho_chosen"], b[0].diagnostics["rho_chosen"])))
    for k, lab in ((2, "main_train"), (3, "inter_train"), (4, "main_test"), (5, "inter_test")):
        print(f"  max |diff| {lab}: {np.max(np.abs(a[k] - b[k])) if a[k].size else 0.0:.2e}")
    print(f"end2end fit time: reference {a[1]:.2f}s candidate {b[1]:.2f}s")
