"""Dev only: apply the S1 projection to cached baseline fits and report label-free metrics (same formulas as harness)."""
import pickle, sys, time
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import s1_projection as S  # noqa: E402


def ortho(pred, main, inter, il):
    v = np.var(pred); worst = None
    for c, (j, k) in enumerate(il):
        if np.var(inter[:, c]) < 0.01 * v:
            continue
        for m in (j, k):
            if np.var(main[:, m]) > 0:
                r = abs(np.corrcoef(inter[:, c], main[:, m])[0, 1]); worst = r if worst is None else max(worst, r)
    return worst


def locvar(X, main, k=10):
    vals = []
    for j in range(X.shape[1]):
        tot = np.var(main[:, j])
        if tot <= 0: continue
        idx = NearestNeighbors(n_neighbors=k).fit(X[:, [j]]).kneighbors(X[:, [j]], return_distance=False)
        vals.append(np.mean(np.var(main[idx, j], axis=1)) / tot)
    return float(np.mean(vals))


def resid(pred, e0, m, i):
    return float(np.mean((pred - e0 - m.sum(1) - i.sum(1)) ** 2) / np.var(pred))


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
    base = (ortho(d["p_tr"], d["m_tr"], d["i_tr"], il), ortho(d["p_te"], d["m_te"], d["i_te"], il), locvar(d["Xtr"], d["m_tr"]))
    new = (ortho(d["p_tr"], m_tr, i_tr, il), ortho(d["p_te"], m_te, i_te, il), locvar(d["Xtr"], m_tr))
    dr = (resid(d["p_tr"], e0, m_tr, i_tr) - resid(d["p_tr"], d["eta0"], d["m_tr"], d["i_tr"]),
          resid(d["p_te"], e0, m_te, i_te) - resid(d["p_te"], d["eta0"], d["m_te"], d["i_te"]))
    grid = list(kw.get("tau_grid", S.TAU_GRID))
    taus = np.bincount([grid.index(t) for t in dg["tau_per_pair"]], minlength=len(grid))
    print(f"{name:12s} ortho_in {f(base[0])}->{f(new[0])}  ortho_out {f(base[1])}->{f(new[1])}  "
          f"locvar {base[2]:.3g}->{new[2]:.3g}  dresid {dr[0]:.1e},{dr[1]:.1e}  proj {t1-t0:.2f}s (base fit {d['fit_s']:.1f}s) taus {taus}", flush=True)


if __name__ == "__main__":
    import os
    kw = {}
    if os.environ.get("CRIT"): kw["criterion"] = os.environ["CRIT"]
    if os.environ.get("MMIN"): kw["m_min"] = int(os.environ["MMIN"])
    if os.environ.get("SEL"): kw["select"] = os.environ["SEL"]
    if os.environ.get("SCALE"): kw["scale"] = os.environ["SCALE"]
    for n in sys.argv[1:]:
        run(n, **kw)
