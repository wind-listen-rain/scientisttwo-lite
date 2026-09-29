"""Dev only: apply the S1 projection to cached baseline fits; label-free metrics (harness formulas) plus a paired
bootstrap of the eval rows for ortho_out. No ground-truth component is computed here.

usage: python dev/dev_eval2.py [key=value ...] name [name ...]
  keys are fit_projection arguments: m_min, se_rule, close, directions, criterion, n_folds (e.g. se_rule=0 close=0)
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE))
import s1_projection as S  # noqa: E402
from dev_eval import locvar, ortho, resid  # noqa: E402

N_BOOT = 200


def boot_ortho(pred, mb, ib, mn, inn, il, n_boot=N_BOOT):
    """Paired bootstrap of the eval rows: (mean, sd) of ortho for baseline and new, and P(new < base)."""
    rng = np.random.default_rng(0)
    n = pred.size
    ob, on = [], []
    for _ in range(n_boot):
        r = rng.integers(0, n, n)
        ob.append(ortho(pred[r], mb[r], ib[r], il))
        on.append(ortho(pred[r], mn[r], inn[r], il))
    if ob[0] is None or on[0] is None:
        return None
    ob, on = np.array(ob, float), np.array(on, float)
    return ob.mean(), ob.std(), on.mean(), on.std(), float(np.mean(on < ob))


def f(x):
    return "None" if x is None else f"{x:.4f}"


def run(name, **kw):
    d = pickle.load(open(HERE / "cache" / f"{name}.pkl", "rb"))
    il = [tuple(x) for x in np.asarray(d["il"]).reshape(-1, 2)]
    t0 = time.time()
    proj, e0, dg = S.fit_projection(d["eta0"], d["m_tr"], d["i_tr"], d["il"], d["Xtr"], d["table"], **kw)
    m_tr, i_tr = S.apply_projection(proj, d["m_tr"], d["i_tr"], d["Xtr"])
    t1 = time.time()
    m_te, i_te = S.apply_projection(proj, d["m_te"], d["i_te"], d["Xte"])
    base = (ortho(d["p_tr"], d["m_tr"], d["i_tr"], il), ortho(d["p_te"], d["m_te"], d["i_te"], il),
            locvar(d["Xtr"], d["m_tr"]))
    new = (ortho(d["p_tr"], m_tr, i_tr, il), ortho(d["p_te"], m_te, i_te, il), locvar(d["Xtr"], m_tr))
    dr = max(abs(resid(d["p_tr"], e0, m_tr, i_tr) - resid(d["p_tr"], d["eta0"], d["m_tr"], d["i_tr"])),
             abs(resid(d["p_te"], e0, m_te, i_te) - resid(d["p_te"], d["eta0"], d["m_te"], d["i_te"])))
    bt = boot_ortho(d["p_te"], d["m_te"], d["i_te"], m_te, i_te, il)
    bts = "boot -" if bt is None else f"boot {bt[0]:.4f}+-{bt[1]:.4f} -> {bt[2]:.4f}+-{bt[3]:.4f} P(better) {bt[4]:.2f}"
    sc = dg.get("scale")
    scs = "" if sc is None else f" scale [{min(sc):.3f},{max(sc):.3f}]"
    print(f"{name:12s} ortho_in {f(base[0])}->{f(new[0])}  ortho_out {f(base[1])}->{f(new[1])}  "
          f"locvar {base[2]:.4g}->{new[2]:.4g} ({new[2] / base[2]:.2f}x)  |dresid| {dr:.0e}  "
          f"proj {t1 - t0:.2f}s  transfers {dg['n_transfer']}/{dg['n_pairs']} closed {dg.get('n_closed')}{scs}  {bts}", flush=True)
    return dict(base=base, new=new, boot=bt, diag=dg)


if __name__ == "__main__":
    kw, names = {}, []
    for a in sys.argv[1:]:
        if "=" in a:
            k, v = a.split("=")
            kw[k] = float(v) if k in ("se_rule", "close_z") else (bool(int(v)) if k in ("close", "directions") else
                                                          (int(v) if k in ("m_min", "n_folds") else v))
        else:
            names.append(a)
    print("config", kw)
    for n in names:
        run(n, **kw)
