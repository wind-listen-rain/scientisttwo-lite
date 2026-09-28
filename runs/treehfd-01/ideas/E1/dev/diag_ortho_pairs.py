"""Which interaction sets ortho_in / ortho_out, for S4's selection (rho = 0 mode) and E1's.

usage: diag_ortho_pairs.py <dataset> <split> [...]
For every interaction: variance share Var[eta_jk] / Var[T] (the harness counts it only when
>= 1%) and |corr| with each of its two main effects, on train and test. Diagnosis only.
"""
import sys
from pathlib import Path

import numpy as np

E1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E1 / "lib"))
sys.path.insert(0, str(E1 / "dev"))
import agtloco  # noqa: E402
from common import real  # noqa: E402

agtloco.KEEP_PATH = True
name = sys.argv[1]
for split in map(int, sys.argv[2:]):
    model, Xtr, Xte = real(name, split)
    hfd = agtloco.GTLocoHFD(model)
    hfd.fit(Xtr)
    d = hfd.diagnostics
    cands = d["candidates"]
    NR = len(agtloco.RHOS)
    idx = [i for i, c in enumerate(cands) if c[2] == 0]
    s4 = cands[idx[int(agtloco._first_min(np.array([cands[i][4] for i in idx])))]]

    def apply(c):
        mode, v, rm, par = c[:4]
        for t, tree in enumerate(hfd.treehfd_list):
            r, g0 = (rm, d["per_tree_idx"][t, v, rm]) if rm < NR else d["per_tree_joint"][t, v]
            g = int(np.clip(g0 + par, 0, len(agtloco.KAPPAS) - 1)) if mode == "shift" else par
            tree.finalize(g, v, r)

    for tag, c in (("S4", s4), ("E1", d["selection"])):
        apply(c)
        print(f"== {name} split {split} {tag} {c[:4]}")
        for lab, X in (("in ", Xtr), ("out", Xte)):
            pred = model.predict(X)
            v = np.var(pred)
            main, inter = hfd.predict(X)
            rows = []
            for k, (j, l) in enumerate(hfd.interaction_list):
                share = np.var(inter[:, k]) / v
                cj = abs(np.corrcoef(inter[:, k], main[:, j])[0, 1]) if np.var(main[:, j]) > 0 else 0
                cl = abs(np.corrcoef(inter[:, k], main[:, l])[0, 1]) if np.var(main[:, l]) > 0 else 0
                rows.append((share, j, l, cj, cl))
            rows.sort(reverse=True)
            top = [f"({j},{l}) share={s:.4f} corr={max(cj, cl):.4f}{'*' if s >= 0.01 else ''}"
                   for s, j, l, cj, cl in rows[:6]]
            big = [max(cj, cl) for s, j, l, cj, cl in rows if s >= 0.01]
            print(f"  {lab} ortho={max(big) if big else float('nan'):.4f} n_big={len(big)} | "
                  + "; ".join(top))
