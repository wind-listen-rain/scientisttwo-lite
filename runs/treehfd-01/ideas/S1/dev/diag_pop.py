"""Dev only (diagnosis): analytical case, population leak coefficients a_pop (100k fresh inputs, no labels, no
ground-truth components) vs X_train-only estimates. Also the implied main-effect scale factors (x5, x6 = noise vars)."""
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diag_se import coef, hc0  # noqa: E402


def all_coefs(main, inter, il):
    return np.array([coef(main[:, [j, k]], inter[:, c]) for c, (j, k) in enumerate(il)])


def scale(a, il, p):
    s = np.ones(p)
    for c, (j, k) in enumerate(il):
        s[j] += a[c, 0]; s[k] += a[c, 1]
    return s


K = 2
for name in sys.argv[1:]:
    d = pickle.load(open(HERE / "cache" / f"{name}.pkl", "rb"))
    fd = pickle.load(open(HERE / "cache" / f"{name}_folds{K}.pkl", "rb"))
    pp = pickle.load(open(HERE / "cache" / f"{name}_pop.pkl", "rb"))
    il = np.asarray(d["il"]).reshape(-1, 2)
    p = d["m_tr"].shape[1]
    a_in = all_coefs(d["m_tr"], d["i_tr"], il)
    se_in = np.array([hc0(d["m_tr"][:, [j, k]], d["i_tr"][:, c]) for c, (j, k) in enumerate(il)])
    a_te = all_coefs(d["m_te"], d["i_te"], il)
    a_pop = all_coefs(pp["m"], pp["i"], il)
    a_oof = all_coefs(fd["m_oof"], fd["i_oof"], il)
    mv = np.var(pp["m"], 0)
    vt = np.var(pp["p"])
    print(f"== {name}")
    for lab, a in (("pop", a_pop), ("eval", a_te), ("in", a_in), ("oof", a_oof)):
        err = sum(mv[il[c, s]] * (a[c, s] - a_pop[c, s]) ** 2 for c in range(len(il)) for s in (0, 1)) / vt
        print(f"   {lab:5s} scale {np.round(scale(a, il, p), 3)}  leak err vs pop {err:.2e}")
    for z in (1.0, 2.0):
        a = np.where(np.abs(a_in) > z * se_in, a_in, 0.0)
        err = sum(mv[il[c, s]] * (a[c, s] - a_pop[c, s]) ** 2 for c in range(len(il)) for s in (0, 1)) / vt
        print(f"   gate{z:.0f} scale {np.round(scale(a, il, p), 3)}  leak err vs pop {err:.2e}")
    a = a_in * np.clip(1 - se_in ** 2 / np.maximum(a_in ** 2, 1e-300), 0, 1)
    err = sum(mv[il[c, s]] * (a[c, s] - a_pop[c, s]) ** 2 for c in range(len(il)) for s in (0, 1)) / vt
    print(f"   JS    scale {np.round(scale(a, il, p), 3)}  leak err vs pop {err:.2e}")
    print("   pair: a_in (se) / a_oof / a_pop / a_eval    [inter var/VarT]")
    for c, (j, k) in enumerate(il):
        print(f"     ({j},{k}) [{np.var(pp['i'][:, c]) / vt:.4f}]  " + "   ".join(
            f"m{v}: {a_in[c, s]:+.3f}({se_in[c, s]:.3f}) / {a_oof[c, s]:+.3f} / {a_pop[c, s]:+.3f} / {a_te[c, s]:+.3f}"
            for s, v in enumerate((j, k))))
