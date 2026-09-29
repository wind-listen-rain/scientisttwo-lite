"""Dev only (diagnosis, label-free): which X_train-only estimate of the step-2 leak coefficient predicts the leak
measured on the eval inputs? a_in = in-sample (full fit), a_oof = out-of-fold (fold refits), a_inf = fold fits on
their own rows, a_cor = a_in + a_oof - a_inf (optimism-corrected)."""
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diag_step2 import coefs  # noqa: E402
from dev_eval import ortho  # noqa: E402


def step2(main, inter, il, coef):
    main, inter = main.copy(), inter.copy()
    s = np.ones(main.shape[1])
    for c, (j, k) in enumerate(il):
        inter[:, c] -= coef[c, 0] * main[:, j] + coef[c, 1] * main[:, k]
        s[j] += coef[c, 0]; s[k] += coef[c, 1]
    return main * s, inter


K = int(sys.argv[1])
for name in sys.argv[2:]:
    d = pickle.load(open(HERE / "cache" / f"{name}.pkl", "rb"))
    fd = pickle.load(open(HERE / "cache" / f"{name}_folds{K}.pkl", "rb"))
    il = np.asarray(d["il"]).reshape(-1, 2)
    cols = [c for c in range(len(il)) if np.var(d["i_tr"][:, c]) > 0]
    vt = np.var(d["p_tr"])
    big = [c for c in cols if np.var(d["i_tr"][:, c]) >= 0.01 * vt]
    a_in = coefs(d["m_tr"], d["i_tr"], il, cols)
    a_te = coefs(d["m_te"], d["i_te"], il, cols)
    a_oof = coefs(fd["m_oof"], fd["i_oof"], il, cols)
    a_inf = np.mean([coefs(f["m"], f["i"], il, cols) for f in fd["inf"]], 0)
    # per-fold held-out coefficients (spread = fold-to-fold variability)
    folds = np.arange(len(d["Xtr"])) % K
    a_ho = np.array([coefs(fd["m_oof"][folds == f], fd["i_oof"][folds == f], il, cols) for f in range(K)])
    a_cor = a_in + a_oof - a_inf
    mv = np.var(d["m_tr"], 0)
    print(f"== {name} (K={K})  big pairs {len(big)}/{len(cols)}")
    # error of each estimate vs eval coef, weighted by var(main) (leak variance units), big pairs and all pairs
    for lab, est in (("none", 0 * a_in), ("a_in", a_in), ("a_oof", a_oof), ("a_inf", a_inf), ("a_cor", a_cor)):
        errs = []
        for sel in (big, cols):
            w = np.array([[mv[il[c, 0]], mv[il[c, 1]]] for c in sel])
            errs.append(np.sum(w * (est[sel] - a_te[sel]) ** 2) / vt)
        m2, i2 = step2(d["m_te"], d["i_te"], il, est)
        m1, i1 = step2(d["m_tr"], d["i_tr"], il, est)
        print(f"   {lab:6s} leak err big {errs[0]:.2e} all {errs[1]:.2e}   ortho_in {ortho(d['p_tr'], m1, i1, il):.4f}"
              f"  ortho_out {ortho(d['p_te'], m2, i2, il):.4f}")
    print("   big pairs: (j,k) a_in / a_inf / a_oof [per fold] / a_te")
    for c in big:
        for s in (0, 1):
            print(f"     ({il[c,0]},{il[c,1]}) m={il[c,s]}: {a_in[c,s]:+.4f} / {a_inf[c,s]:+.4f} / {a_oof[c,s]:+.4f} "
                  f"{np.round(a_ho[:, c, s], 4)} / {a_te[c,s]:+.4f}")
