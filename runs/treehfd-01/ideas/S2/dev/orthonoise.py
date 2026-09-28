"""Dev: sampling noise of ortho_out and effect of a post-hoc ensemble-level orthogonalisation.

1. Bootstrap (B=500, fixed seed) of the test set: sd of the harness ortho statistic for baseline and S2, and the
   paired difference S2 - baseline. This uses X_test only to *evaluate*, never to fit.
2. Post-hoc transfer (P_n on X_train): regress each ensemble interaction on its two ensemble main effects and move the
   fitted part into the main effects (reconstruction unchanged). Reports ortho_in/ortho_out afterwards.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/bench")
from harness import orthogonality  # noqa: E402
import smoothed_hfd as sh  # noqa: E402
from treehfd_mod import XGBTreeHFD  # noqa: E402

from setup import real  # noqa: E402



def transfer(main_tr, inter_tr, il, mains, inters):
    """Apply the P_n(X_train) regression transfer to the (main, inter) arrays in mains/inters."""
    p = main_tr.shape[1]
    scale = np.ones(p)
    betas = []
    mu = main_tr.mean(0)
    for c, (j, k) in enumerate(il):
        Z = np.column_stack((main_tr[:, j] - mu[j], main_tr[:, k] - mu[k]))
        y = inter_tr[:, c] - inter_tr[:, c].mean()
        b = np.linalg.lstsq(Z, y, rcond=None)[0]
        betas.append(b)
        scale[j] += b[0]
        scale[k] += b[1]
    outs = []
    for M, I in zip(mains, inters):
        M2, I2 = M.copy(), I.copy()
        for c, (j, k) in enumerate(il):
            I2[:, c] -= betas[c][0] * (M[:, j] - mu[j]) + betas[c][1] * (M[:, k] - mu[k])
        M2 = mu + (M - mu) * scale
        outs.append((M2, I2))
    return outs


name = sys.argv[1]
model, X, Xe = real(name)
pt, pe = model.predict(X), model.predict(Xe)
res = {}
h = XGBTreeHFD(model)
h.fit(X, verbose=False)
res["base"] = (h.predict(X, verbose=False), h.predict(Xe, verbose=False), h.interaction_list)
s = sh.SmoothedTreeHFD(model, alpha=0.35).fit(X)
res["S2"] = (s.predict(X), s.predict(Xe), s.interaction_list)
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(pe), len(pe)) for _ in range(500)]
stat = {}
for tag, ((mt, it), (me, ie), il) in res.items():
    il = [tuple(int(a) for a in q) for q in il]
    stat[tag] = np.array([orthogonality(pe[b], me[b], ie[b], il) or 0.0 for b in boots])
    (mt2, it2), (me2, ie2) = transfer(mt, it, il, [mt, me], [it, ie])
    print(f"{name} {tag:>4}: ortho_out={orthogonality(pe, me, ie, il):.4f} boot_sd={stat[tag].std():.4f} | "
          f"after transfer: ortho_in={orthogonality(pt, mt2, it2, il)} ortho_out={orthogonality(pe, me2, ie2, il)}")
d = stat["S2"] - stat["base"]
print(f"{name} S2-base ortho_out: boot mean={d.mean():+.4f} sd={d.std():.4f} "
      f"95% CI=[{np.quantile(d, .025):+.4f},{np.quantile(d, .975):+.4f}]  n_test={len(pe)}")
# Same paired bootstrap for resid_out (relative change S2 / base - 1).
err = {tag: (pe - (eta - 0) - me.sum(1) - ie.sum(1)) ** 2 for tag, ((_, _), (me, ie), _), eta in
       zip(res, res.values(), (h.eta0, s.eta0))}
r = np.array([err["S2"][b].mean() / err["base"][b].mean() - 1 for b in boots])
print(f"{name} S2/base-1 resid_out: point={err['S2'].mean() / err['base'].mean() - 1:+.3f} "
      f"95% CI=[{np.quantile(r, .025):+.3f},{np.quantile(r, .975):+.3f}]")
