"""Round 3 development tool: fit once, then score EVERY ensemble candidate.

usage: cands_r3.py <dataset|analytical[:rep]> [split=<s>] [rhos=0,0.25] [tag=<name>]
Writes dev/cache/<dataset>_s<split>_<tag>.npz with, per ensemble candidate (same order as
agtloco's candidate list): key (mode 0 = shift / 1 = common, variant, rho mode, par), risk R,
in-sample fidelity, the per-point ensemble leave-out residuals (float32), and the harness
metrics of that candidate (resid_in, resid_out, ortho_in, ortho_out, locvar_in; analytical:
the mse_* metrics and resid_out). Selection rules are then compared offline
(dev/rules_r3.py). Held-out metrics are for diagnosis only; the method never sees them.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "runs/treehfd-01/ideas/E1/lib"))
sys.path.insert(0, str(ROOT / "runs/treehfd-01/ideas/E1/dev"))
sys.path.insert(0, str(ROOT / "bench"))
import agtloco  # noqa: E402
from common import analytical, real  # noqa: E402
from harness import eta_main, eta_order2, local_variability, orthogonality  # noqa: E402

agtloco.KEEP_PATH = True
opts = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
if "rhos" in opts:
    agtloco.RHOS = tuple(float(r) for r in opts["rhos"].split(","))
tag = opts.get("tag", "r" + "_".join(f"{r:g}" for r in agtloco.RHOS))
name = sys.argv[1]
split = int(opts.get("split", 0))
is_ana = name.startswith("analytical")
if is_ana:
    split = int(name.split(":")[1]) if ":" in name else 0
    model, Xtr, Xte = analytical(split)
    name = "analytical"
else:
    model, Xtr, Xte = real(name, split)
hfd = agtloco.GTLocoHFD(model)
t0 = time.time()
hfd.fit(Xtr)
fit_s = time.time() - t0
d = hfd.diagnostics
NR = len(agtloco.RHOS)
G = len(agtloco.KAPPAS)
p_tr, p_te = model.predict(Xtr), model.predict(Xte)
v_tr, v_te = np.var(p_tr), np.var(p_te)
nn_idx = [NearestNeighbors(n_neighbors=min(10, len(Xtr))).fit(Xtr[:, [j]])
          .kneighbors(Xtr[:, [j]], return_distance=False) for j in range(Xtr.shape[1])]


def locvar(main):
    vals = []
    for j in range(main.shape[1]):
        tot = np.var(main[:, j])
        if tot <= 0:
            continue
        vals.append(np.mean(np.var(main[nn_idx[j], j], axis=1)) / tot)
    return float(np.mean(vals)) if vals else np.nan


def apply(c):
    mode, v, rm, par = c[:4]
    for t, tree in enumerate(hfd.treehfd_list):
        if rm < NR:
            r, g0 = rm, d["per_tree_idx"][t, v, rm]
        else:
            r, g0 = d["per_tree_joint"][t, v]
        g = int(np.clip(g0 + par, 0, G - 1)) if mode == "shift" else par
        tree.finalize(g, v, r)


def nz(o):
    return np.nan if o is None else o


il = hfd.interaction_list
il_t = [tuple(p) for p in il.tolist()]
cands = d["candidates"]
if is_ana:
    target = np.zeros_like(Xte)
    target[:, 0] = np.sin(2 * np.pi * Xte[:, 0]) + eta_main(Xte[:, 0])
    for j in (1, 2, 3):
        target[:, j] = eta_main(Xte[:, j])
    t12, t34 = eta_order2(Xte[:, 0], Xte[:, 1]), eta_order2(Xte[:, 2], Xte[:, 3])
    others = [c for c, p in enumerate(il_t) if p not in ((0, 1), (2, 3))]
    names = [f"mse_eta{j + 1}" for j in range(6)] + ["mse_eta12", "mse_eta34",
                                                      "mse_others", "resid_out"]
else:
    names = ["resid_in", "resid_out", "ortho_in", "ortho_out", "locvar_in"]
met = np.full((len(cands), len(names)), np.nan)
t1 = time.time()
for ci, c in enumerate(cands):
    apply(c)
    if is_ana:
        main, inter = hfd.predict(Xte)
        row = [np.mean((target[:, j] - main[:, j]) ** 2) for j in range(6)]
        for pair, tgt in (((0, 1), t12), ((2, 3), t34)):
            est = inter[:, il_t.index(pair)] if pair in il_t else 0.0
            row.append(np.mean((tgt - est) ** 2))
        row.append(np.mean(inter[:, others] ** 2) if others else 0.0)
        row.append(np.mean((p_te - hfd.eta0 - main.sum(1) - inter.sum(1)) ** 2) / v_te)
    else:
        mtr, itr = hfd.predict(Xtr)
        mte, ite = hfd.predict(Xte)
        row = [np.mean((p_tr - hfd.eta0 - mtr.sum(1) - itr.sum(1)) ** 2) / v_tr,
               np.mean((p_te - hfd.eta0 - mte.sum(1) - ite.sum(1)) ** 2) / v_te,
               nz(orthogonality(p_tr, mtr, itr, il)), nz(orthogonality(p_te, mte, ite, il)),
               locvar(mtr)]
        if ci == 0:  # the precomputed-neighbour locvar must equal the harness function
            ref = local_variability(Xtr, mtr)
            assert abs(row[4] - ref) <= 1e-12 * max(1.0, abs(ref)), (row[4], ref)
    met[ci] = row
key = np.array([(0 if c[0] == "shift" else 1, c[1], c[2], c[3]) for c in cands], dtype=int)
out = ROOT / "runs/treehfd-01/ideas/E1/dev/cache"
out.mkdir(exist_ok=True)
sel = [i for i, c in enumerate(cands) if c[:4] == d["selection"][:4]][0]
np.savez_compressed(out / f"{name}_s{split}_{tag}.npz", key=key,
                    risk=np.array([c[4] for c in cands]), fid=np.array([c[5] for c in cands]),
                    loo=d["cand_loo"], met=met, names=np.array(names), var_t=d["var_t"],
                    rhos=np.array(agtloco.RHOS), fit_s=fit_s, sel=sel,
                    n_virtual=np.array(d["n_virtual"]),
                    per_tree_joint=d["per_tree_joint"], per_tree_idx=d["per_tree_idx"])
print(f"{name} split={split} RHOS={agtloco.RHOS} fit {fit_s:.1f}s, {len(cands)} candidates "
      f"scored in {time.time() - t1:.1f}s -> {tag}")
