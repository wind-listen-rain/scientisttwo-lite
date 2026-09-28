"""Diagnostic (development only): is the ortho_out regression real or split noise?

usage: diag_ortho_out.py <dataset> [splits=0,1,2,...] [boot=500]
For each split seed s (s = 0 is the benchmark split; others are alternative 80/20 splits used
for diagnosis only), fits the XGBoost model as the harness does, then the baseline TreeHFD and
GT-LOCO, and prints resid/ortho metrics, the per-interaction breakdown of ortho_out, a test-set
bootstrap of ortho_out, and GT-LOCO with the unseen-cell rule switched (same kappas).
Held-out numbers are printed for diagnosis only; the method never sees them.
"""
import sys
import time

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/runs/treehfd-01/ideas/S4/lib")
import gtloco  # noqa: E402
from treehfd import XGBTreeHFD  # noqa: E402

DATA = str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/bench/data"
XGB_PARAMS = dict(eta=0.1, n_estimators=100, max_depth=6, n_jobs=4)
opts = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
SPLITS = [int(s) for s in opts.get("splits", "0").split(",")]
BOOT = int(opts.get("boot", 500))
gtloco.KEEP_PATH = True


def split(name, seed):
    d = np.load(f"{DATA}/{name}.npz")
    X, y = d["X"], d["y"]
    perm = np.random.default_rng(seed).permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = perm[:cut], perm[cut:]
    model = xgb.XGBRegressor(random_state=0, **XGB_PARAMS).fit(X[tr], y[tr])
    return model, X[tr], X[te]


def corr_table(pred, main, inter, il):
    """(pair, var fraction, |corr| with main j, |corr| with main k) for interactions >= 1%."""
    v = np.var(pred)
    rows = []
    for c, (j, k) in enumerate(il):
        if np.var(inter[:, c]) < 0.01 * v:
            continue
        cs = [abs(np.corrcoef(inter[:, c], main[:, mm])[0, 1]) if np.var(main[:, mm]) > 0
              else 0.0 for mm in (j, k)]
        rows.append(((int(j), int(k)), np.var(inter[:, c]) / v, cs[0], cs[1]))
    return rows


def ortho(pred, main, inter, il):
    rows = corr_table(pred, main, inter, il)
    return max((max(r[2], r[3]) for r in rows), default=np.nan)


def metrics(model, Xtr, Xte, main_fn):
    out = {}
    for s, X in (("in", Xtr), ("out", Xte)):
        p = model.predict(X)
        e0, main, inter, il = main_fn(X)
        out["resid_" + s] = np.mean((p - e0 - main.sum(1) - inter.sum(1)) ** 2) / np.var(p)
        out["ortho_" + s] = ortho(p, main, inter, il)
        out["_" + s] = (p, main, inter, il)
    return out


def boot(p, main, inter, il, rng):
    n = len(p)
    vals = np.empty(BOOT)
    for b in range(BOOT):
        i = rng.integers(0, n, n)
        vals[b] = ortho(p[i], main[i], inter[i], il)
    return vals


def fmt(m):
    return (f"resid_in={m['resid_in']:.5f} resid_out={m['resid_out']:.5f} "
            f"ortho_in={m['ortho_in']:.4f} ortho_out={m['ortho_out']:.4f}")


name = sys.argv[1]
summary = []
for seed in SPLITS:
    model, Xtr, Xte = split(name, seed)
    t0 = time.time()
    base = XGBTreeHFD(model)
    base.fit(Xtr, interaction_order=2, verbose=False)
    tb = time.time() - t0

    def base_fn(X, base=base):
        m, i = base.predict(X, verbose=False)
        return base.eta0, m, i, base.interaction_list

    t0 = time.time()
    hfd = gtloco.GTLocoHFD(model)
    hfd.fit(Xtr)
    tg = time.time() - t0

    def gt_fn(X, hfd=hfd):
        m, i = hfd.predict(X)
        return hfd.eta0, m, i, hfd.interaction_list

    mb, mg = metrics(model, Xtr, Xte, base_fn), metrics(model, Xtr, Xte, gt_fn)
    sel = hfd.diagnostics["selection"]
    VAR = gtloco.variants()
    pr, ru = VAR[sel[1]]
    print(f"=== {name} split {seed}  (fit base {tb:.1f}s, gtloco {tg:.1f}s)  selection "
          f"{sel[0]} {gtloco.PRIORS[pr]}/{gtloco.RULES[ru]} {sel[2]}")
    print(f"  baseline : {fmt(mb)}")
    print(f"  gtloco   : {fmt(mg)}")
    row = {"seed": seed, "base": mb, "gt": mg}
    # Same kappas, other rules (and other prior) to isolate the unseen-cell rule.
    kap = [t.kappa_idx for t in hfd.treehfd_list]
    for v, (pr2, ru2) in enumerate(VAR):
        if v == sel[1]:
            continue
        for t, g in zip(hfd.treehfd_list, kap, strict=True):
            t.finalize(g, v)
        mv = metrics(model, Xtr, Xte, gt_fn)
        row[f"{gtloco.PRIORS[pr2]}/{gtloco.RULES[ru2]}"] = mv
        print(f"  same kappa {gtloco.PRIORS[pr2]:7s}/{gtloco.RULES[ru2]:8s}: {fmt(mv)}")
    for t, g in zip(hfd.treehfd_list, kap, strict=True):
        t.finalize(g, sel[1])
    # Per-interaction breakdown (test set).
    for lab, m in (("baseline", mb), ("gtloco", mg)):
        p, main, inter, il = m["_out"]
        tab = corr_table(p, main, inter, il)
        print(f"  {lab} test interactions >= 1%: " + ", ".join(
            f"{a}:{b:.3f}|{c:.3f},{d:.3f}" for a, b, c, d in sorted(tab, key=lambda r: -r[1])))
    # Test-set bootstrap of ortho_out.
    rng = np.random.default_rng(12345)
    bb = boot(*mb["_out"], rng)
    rng = np.random.default_rng(12345)
    bg = boot(*mg["_out"], rng)
    print(f"  bootstrap ortho_out: baseline {np.mean(bb):.4f} +- {np.std(bb):.4f}, "
          f"gtloco {np.mean(bg):.4f} +- {np.std(bg):.4f}, P(gtloco > baseline) = "
          f"{np.mean(bg > bb):.2f}")
    summary.append(row)

if len(summary) > 1:
    print("=== summary over splits (ratio gtloco / baseline)")
    keys = ["resid_in", "resid_out", "ortho_in", "ortho_out"]
    for k in keys:
        r = np.array([s["gt"][k] / s["base"][k] for s in summary])
        print(f"  {k:9s}: " + " ".join(f"{x:.2f}" for x in r)
              + f"  | geo-mean {np.exp(np.mean(np.log(r))):.3f}")
    for lab in [k for k in summary[0] if "/" in k]:
        r = np.array([s[lab]["ortho_out"] / s["base"]["ortho_out"] if lab in s else np.nan
                      for s in summary])
        print(f"  ortho_out ratio, same kappa {lab}: " + " ".join(f"{x:.2f}" for x in r))
