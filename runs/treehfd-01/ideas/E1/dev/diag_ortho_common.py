"""Threshold-robust orthogonality: S4's selection (rho = 0 mode, argmin) vs E1's selection.

usage: diag_ortho_common.py <dataset> [splits...]
Per split and for train / test: the harness ortho (max |corr| over interactions with variance
share >= 1%, own set per method), the max |corr| over the COMMON set (interactions >= 1% in S4
or in E1, the same pairs for both), the median over that set of the per-pair |corr| ratio
E1 / S4, and the summed interaction variance share. Diagnosis only.
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
splits = [int(s) for s in sys.argv[2:]] or list(range(8))
for split in splits:
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

    stats = {}
    for tag, c in (("S4", s4), ("E1", d["selection"])):
        apply(c)
        for lab, X in (("in", Xtr), ("out", Xte)):
            pred = model.predict(X)
            v = np.var(pred)
            main, inter = hfd.predict(X)
            share, corr = [], []
            for k, (j, l) in enumerate(hfd.interaction_list):
                share.append(np.var(inter[:, k]) / v)
                cs = [abs(np.corrcoef(inter[:, k], main[:, q])[0, 1]) for q in (j, l)
                      if np.var(main[:, q]) > 0 and np.var(inter[:, k]) > 0]
                corr.append(max(cs) if cs else 0.0)
            stats[tag, lab] = (np.array(share), np.array(corr))
    for lab in ("in", "out"):
        sh4, c4 = stats["S4", lab]
        sh1, c1 = stats["E1", lab]
        common = (sh4 >= 0.01) | (sh1 >= 0.01)
        own4 = c4[sh4 >= 0.01].max() if np.any(sh4 >= 0.01) else np.nan
        own1 = c1[sh1 >= 0.01].max() if np.any(sh1 >= 0.01) else np.nan
        cm4 = c4[common].max() if common.any() else np.nan
        cm1 = c1[common].max() if common.any() else np.nan
        ratio = np.median(c1[common] / np.maximum(c4[common], 1e-12)) if common.any() else np.nan
        print(f"{name} split {split} {lab:3s}: harness S4={own4:.4f} E1={own1:.4f} | common set "
              f"(n={int(common.sum())}) S4={cm4:.4f} E1={cm1:.4f} median pair ratio={ratio:.3f} | "
              f"n_big S4={int(np.sum(sh4 >= 0.01))} E1={int(np.sum(sh1 >= 0.01))} | inter share "
              f"S4={sh4.sum():.4f} E1={sh1.sum():.4f}", flush=True)
