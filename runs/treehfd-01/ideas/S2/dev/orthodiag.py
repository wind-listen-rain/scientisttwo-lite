"""Dev: where does the ortho violation come from? Within-tree vs cross-tree covariance.

For each interaction (j,k) passing the 1% variance threshold, cov(sum_t eta_jk^t, sum_t eta_m^t) on train/test
is split into the t = t' part (constrained by the per-tree orthogonality rows) and the t != t' part.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import smoothed_hfd as sh  # noqa: E402

from setup import real  # noqa: E402

name = sys.argv[1]
alphas = [float(a) for a in sys.argv[2].split(",")]
extra = dict(a.split("=") for a in sys.argv[3:])
for k, v in extra.items():
    setattr(sh, k, type(getattr(sh, k))(v))
model, X, Xe = real(name)
for a in alphas:
    h = sh.SmoothedTreeHFD(model, alpha=a).fit(X)
    for tag, Z in (("in", X), ("out", Xe)):
        Zf = sh._f32(Z)
        pred = model.predict(Z)
        v = np.var(pred)
        n = Z.shape[0]
        per_main, per_inter = {}, {}
        for t, F in enumerate(h.fitted):
            if F.variables.size == 0:
                continue
            m, it = sh._predict_tree(F, Zf)
            for vi, j in enumerate(F.variables):
                per_main.setdefault(int(j), []).append((t, m[:, vi]))
            for q, p in enumerate(F.pairs):
                per_inter.setdefault(p, []).append((t, it[:, q]))
        rows = []
        for p, lst in per_inter.items():
            I = np.sum([x for _, x in lst], axis=0)
            if np.var(I) < 0.01 * v:
                continue
            for mvar in p:
                M = np.sum([x for _, x in per_main[mvar]], axis=0)
                cov = np.mean((I - I.mean()) * (M - M.mean()))
                mt = dict(per_main[mvar])
                within = sum(np.mean((x - x.mean()) * (mt[t] - mt[t].mean())) for t, x in lst if t in mt)
                corr = cov / np.sqrt(np.var(I) * np.var(M))
                rows.append((abs(corr), p, mvar, np.var(I) / v, within / np.sqrt(np.var(I) * np.var(M)),
                             (cov - within) / np.sqrt(np.var(I) * np.var(M))))
        rows.sort(reverse=True)
        print(f"{name} alpha={a} {tag}: " + "; ".join(
            f"{p}|{m}: |corr|={c:.4f} share={s:.4f} within={w:+.4f} cross={x:+.4f}" for c, p, m, s, w, x in rows[:4]))
