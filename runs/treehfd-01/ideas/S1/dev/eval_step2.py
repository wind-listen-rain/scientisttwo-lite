"""Dev only: compare S1 configurations on cached baseline fits, label-free and paired (every configuration transforms the
*same* baseline components, so tie-break jitter cannot enter the comparison).

Real data: ortho_in / ortho_out / locvar ratio (harness formulas) and a paired bootstrap of the eval rows for the
ortho_out *difference* to the baseline (mean, sd, P(S1 < baseline)).
Analytical: the same, plus population quantities on 100k fresh *inputs* (dev/cache_pop.py; no labels, no ground-truth
components): ortho_pop (harness formula) and leak_pop = sum over all pairs and parents of cov(eta_jk, eta_j)^2/var(eta_j),
divided by Var T (the interaction variance still lying along its parents' main effects), and the rescale of the
noise-variable main effects x5, x6.

usage: python dev/eval_step2.py CONFIG [CONFIG ...] -- name [name ...]     (CONFIG names: see CONFIGS)
"""
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE))
import s1_projection as S  # noqa: E402
from dev_eval import locvar, ortho  # noqa: E402

CONFIGS = {
    "prev": dict(step1=True, shrink="none"),          # previous official S1
    "s2none": dict(step1=False, shrink="none"),       # step 2 only, unshrunk
    "s2var": dict(step1=False, shrink="var_eb"),      # step 2 only, per-variable EB shrinkage
    "s2coef": dict(step1=False, shrink="coef_eb"),    # step 2 only, per-coefficient EB shrinkage
    "s2gate": dict(step1=False, shrink="pair_gate"),  # step 2 only, Wald gate z=2 (older ablation)
    "s2pair": dict(step1=False, shrink="pair_eb"),    # step 2 only, per-pair EB shrinkage (Wald)
    "s12pair": dict(step1=True, shrink="pair_eb"),    # step 1 + step 2 with per-pair EB shrinkage
    "s12var": dict(step1=True, shrink="var_eb"),      # step 1 + step 2 with per-variable EB shrinkage
    "s1only": dict(step1=True, close=False),          # step 1 only
}
N_BOOT = 500


def leak_total(main, inter, il, vt):
    Mc = main - main.mean(0)
    vm = np.mean(Mc ** 2, 0)
    tot = 0.0
    for c, (j, k) in enumerate(il):
        y = inter[:, c] - inter[:, c].mean()
        for m in (j, k):
            if vm[m] > 0:
                tot += np.mean(y * Mc[:, m]) ** 2 / vm[m]
    return tot / vt


def boot_diff(pred, mb, ib, mn, inn, il):
    rng = np.random.default_rng(0)
    n = pred.size
    d = []
    for _ in range(N_BOOT):
        r = rng.integers(0, n, n)
        a, b = ortho(pred[r], mb[r], ib[r], il), ortho(pred[r], mn[r], inn[r], il)
        if a is None or b is None:
            return None
        d.append(b - a)
    d = np.array(d)
    return d.mean(), d.std(), float(np.mean(d < 0))


def run(cfg, name):
    d = pickle.load(open(HERE / "cache" / f"{name}.pkl", "rb"))
    il = [tuple(x) for x in np.asarray(d["il"]).reshape(-1, 2)]
    kw = CONFIGS[cfg]
    proj, e0, dg = S.fit_projection(d["eta0"], d["m_tr"], d["i_tr"], d["il"], d["Xtr"], d["table"], **kw)
    m_tr, i_tr = S.apply_projection(proj, d["m_tr"], d["i_tr"], d["Xtr"])
    m_te, i_te = S.apply_projection(proj, d["m_te"], d["i_te"], d["Xte"])
    lv = locvar(d["Xtr"], m_tr) / locvar(d["Xtr"], d["m_tr"])
    oi, oo = ortho(d["p_tr"], m_tr, i_tr, il), ortho(d["p_te"], m_te, i_te, il)
    bd = boot_diff(d["p_te"], d["m_te"], d["i_te"], m_te, i_te, il)
    sc = dg.get("scale")
    row = dict(oi=oi, oo=oo, lv=lv, bd=bd, ntr=dg["n_transfer"], sc=sc)
    pp = HERE / "cache" / f"{name}_pop.pkl"
    if pp.exists():
        pp = pickle.load(open(pp, "rb"))
        m_p, i_p = S.apply_projection(proj, pp["m"], pp["i"], pp["Xp"])
        vt = np.var(pp["p"])
        row.update(op=ortho(pp["p"], m_p, i_p, il), lp=leak_total(m_p, i_p, il, vt),
                   op0=ortho(pp["p"], pp["m"], pp["i"], il), lp0=leak_total(pp["m"], pp["i"], il, vt))
    return row


def fmt(x):
    return "  -   " if x is None else f"{x:.4f}"


if __name__ == "__main__":
    cut = sys.argv.index("--")
    cfgs, names = sys.argv[1:cut], sys.argv[cut + 1:]
    for name in names:
        d = pickle.load(open(HERE / "cache" / f"{name}.pkl", "rb"))
        il = [tuple(x) for x in np.asarray(d["il"]).reshape(-1, 2)]
        print(f"== {name}: baseline ortho_in {fmt(ortho(d['p_tr'], d['m_tr'], d['i_tr'], il))} "
              f"ortho_out {fmt(ortho(d['p_te'], d['m_te'], d['i_te'], il))}", flush=True)
        for cfg in cfgs:
            r = run(cfg, name)
            bd = "" if r["bd"] is None else f"  d_out {r['bd'][0]:+.4f}+-{r['bd'][1]:.4f} P(better) {r['bd'][2]:.2f}"
            sc = "" if r["sc"] is None else f"  scale [{min(r['sc']):.3f},{max(r['sc']):.3f}]"
            pop = ""
            if "op" in r:
                pop = (f"  | pop: ortho {r['op0']:.4f}->{r['op']:.4f} leak {r['lp0']:.2e}->{r['lp']:.2e}"
                       f"  s5,s6 {r['sc'][4]:.3f},{r['sc'][5]:.3f}" if r["sc"] else "")
            print(f"  {cfg:7s} ortho_in {fmt(r['oi'])} ortho_out {fmt(r['oo'])} locvar {r['lv']:.3f}x "
                  f"step1 {r['ntr']}{sc}{bd}{pop}", flush=True)
