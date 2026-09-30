"""Shared helpers for the rebuttal experiments (supp/).

Only a fitted XGBoost model and its X_train are given to any decomposition; labels are used
solely to train the given model exactly as the harness does (same splits, seeds, hyper-parameters)
and are never passed to a decomposition. Nothing under bench/ or tasks/ is modified; the
harness is imported read-only for its metric functions.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(str(__import__("pathlib").Path(__file__).resolve().parents[4]))
S4 = ROOT / "runs/treehfd-01/paper_S4"
SUPP = S4 / "supp"
sys.path.insert(0, str(S4 / "method/lib"))
sys.path.insert(0, str(ROOT / "bench"))
import harness  # noqa: E402  (read-only import: metric functions and constants)
import gtloco  # noqa: E402
from treehfd import XGBTreeHFD  # noqa: E402

DATASETS = harness.FULL_DATA
XGB_PARAMS = harness.XGB_PARAMS
DEFAULTS = dict(KAPPAS=gtloco.KAPPAS.copy(), EPS_MAIN=gtloco.EPS_MAIN, LATTICE_RIDGE=gtloco.LATTICE_RIDGE,
                PRIORS=gtloco.PRIORS, RULES=gtloco.RULES)


def real(name):
    """Same model and split as harness.real_data (split seed 0, random_state=0)."""
    d = np.load(harness.DATA / f"{name}.npz")
    X, y = d["X"], d["y"]
    perm = np.random.default_rng(0).permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = perm[:cut], perm[cut:]
    model = xgb.XGBRegressor(random_state=0, **XGB_PARAMS).fit(X[tr], y[tr])
    return model, X[tr], X[te]


def analytical_data(rep):
    """Same samples and model as harness.analytical (rep-th repetition)."""
    mu, cov = np.zeros(6), np.full((6, 6), harness.RHO)
    np.fill_diagonal(cov, 1.0)
    rng_tr, rng_te = np.random.default_rng(1000 + rep), np.random.default_rng(2000 + rep)
    X = rng_tr.multivariate_normal(mu, cov, size=5000)
    y = np.sin(2 * np.pi * X[:, 0]) + X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] + rng_tr.normal(0, 0.5, 5000)
    Xn = rng_te.multivariate_normal(mu, cov, size=5000)
    model = xgb.XGBRegressor(random_state=rep, **XGB_PARAMS).fit(X, y)
    return model, X, Xn


def set_config(kappas=None, eps_main=None, lattice_ridge=None, priors=None, rules=None):
    gtloco.KAPPAS = np.asarray(DEFAULTS["KAPPAS"] if kappas is None else kappas, float)
    gtloco.EPS_MAIN = DEFAULTS["EPS_MAIN"] if eps_main is None else eps_main
    gtloco.LATTICE_RIDGE = DEFAULTS["LATTICE_RIDGE"] if lattice_ridge is None else lattice_ridge
    gtloco.PRIORS = DEFAULTS["PRIORS"] if priors is None else tuple(priors)
    gtloco.RULES = DEFAULTS["RULES"] if rules is None else tuple(rules)
    gtloco.KEEP_PATH = True   # keep the kappa path so the same fit can be re-finalised


# ----------------------------------------------------------------------------- decompositions
class Out:
    """A decomposition evaluated on train and eval inputs (arrays only)."""
    def __init__(self, eta0, il, Xtr, Xte, pred):
        self.eta0, self.il = float(eta0), np.asarray(il).reshape(-1, 2)
        self.pred = pred            # function X -> (main, inter)
        self.tr = pred(Xtr)
        self.te = pred(Xte)


def baseline_fit(model, X):
    t0 = time.perf_counter()
    h = XGBTreeHFD(model)
    h.fit(X, interaction_order=2, verbose=False)
    return h, time.perf_counter() - t0


def out_from_baseline(h, Xtr, Xte):
    il = h.interaction_list if h.interaction_list.shape[0] else np.empty((0, 2), int)
    cache = {}

    def pred(X):
        # The baseline fills unseen test cells with a *random* tie-break among the nearest cells
        # (cartesian_partition.py, np.random.default_rng()), so its predictions differ between calls.
        # Memoise per input array so that one decomposition is evaluated consistently everywhere.
        key = (X.__array_interface__["data"][0], X.shape)
        if key not in cache:
            cache[key] = h.predict(X, verbose=False)
        return cache[key]
    return Out(h.eta0, il, Xtr, Xte, pred)


class SweepHFD(gtloco.GTLocoHFD):
    """GT-LOCO with the ensemble step decoupled from the fit.

    fit() runs the per-tree solves once and stores, for every (variant, shift) and (variant,
    common kappa), the summed leave-out and in-sample residuals (the same accumulators as
    GTLocoHFD.fit, over a wide shift range). choose() then reproduces the ensemble selection for
    any shift set / candidate restriction without refitting; with the default arguments it is
    identical to GTLocoHFD.fit (checked against the official results by run_real_core.py).
    """
    WIDE = np.arange(-16, 11)

    def fit(self, X, interaction_order=2, verbose=False):
        X = np.asarray(X, float)
        n = X.shape[0]
        self.interaction_order = interaction_order
        self.depth_variable = self.max_depth
        tp = self._tree_predict(X)
        G, NV = len(gtloco.KAPPAS), len(gtloco.variants())
        V = gtloco.variants()
        self.n_train = n
        self.acc_shift = np.zeros((NV, len(self.WIDE), n))
        self.ins_shift = np.zeros((NV, len(self.WIDE), n))
        self.acc_common = np.zeros((NV, G, n))
        self.ins_common = np.zeros((len(gtloco.PRIORS), G, n))
        self.per_tree_idx = np.zeros((self.n_estimators, NV), dtype=int)
        self.treehfd_list = []
        eta0 = 0.0
        for t in range(self.n_estimators):
            table = pd.DataFrame(self.xgb_table[self.xgb_table["Tree"] == t])
            tree = gtloco.GTLocoTree(table, interaction_order, self.depth_variable)
            loo, ins, risk, _, _ = tree.fit(X, tp[:, t])
            idx = gtloco._first_min(risk, axis=1)
            self.per_tree_idx[t] = idx
            for v, (pr, _) in enumerate(V):
                g = np.clip(idx[v] + self.WIDE, 0, G - 1)
                self.acc_shift[v] += loo[v, g]
                self.ins_shift[v] += ins[pr, g]
                self.acc_common[v] += loo[v]
            self.ins_common += ins
            eta0 += tree.eta0
            self.treehfd_list.append(tree)
        self.eta0 = float(eta0)
        self.var_t = float(np.var(tp.sum(axis=1)))
        lists = [np.array(t.interaction_list, dtype=int).reshape(-1, 2) for t in self.treehfd_list]
        allp = np.concatenate(lists, axis=0) if lists else np.empty((0, 2), int)
        self.interaction_list = np.unique(allp, axis=0) if len(allp) else np.empty((0, 2), dtype=int)
        index = {tuple(p): k for k, p in enumerate(self.interaction_list.tolist())}
        self._inter_index = [np.array([index[tuple(p)] for p in t.interaction_list], dtype=int)
                             for t in self.treehfd_list]

    def candidates(self, shifts=range(-5, 4), common=True, variants=None, shift_cands=True):
        """List of (mode, variant, par, risk, in-sample fidelity) in the official tie-break order."""
        V = gtloco.variants()
        G = len(gtloco.KAPPAS)
        shifts = np.asarray(list(shifts), int)
        order = np.argsort(np.abs(shifts), kind="stable")
        pos = {int(s): i for i, s in enumerate(self.WIDE)}
        out = []
        for v, (pr, _) in enumerate(V):
            if variants is not None and v not in variants:
                continue
            for si in (order if shift_cands else []):
                s = int(shifts[si])
                out.append(("shift", v, s, float(np.mean(self.acc_shift[v, pos[s]] ** 2)),
                            float(np.mean(self.ins_shift[v, pos[s]] ** 2))))
            if common:
                for g in range(G):
                    out.append(("common", v, g, float(np.mean(self.acc_common[v, g] ** 2)),
                                float(np.mean(self.ins_common[pr, g] ** 2))))
        return out

    def choose(self, **kw):
        c = self.candidates(**kw)
        i = int(gtloco._first_min(np.array([x[3] for x in c])))
        return c[i]

    def apply(self, cand):
        """Finalise every tree with the candidate (mode, variant, par, ...)."""
        mode, v, par = cand[:3]
        G = len(gtloco.KAPPAS)
        for t, tree in enumerate(self.treehfd_list):
            g = int(np.clip(self.per_tree_idx[t, v] + par, 0, G - 1)) if mode == "shift" else int(par)
            tree.finalize(g, v)

    def out(self, Xtr, Xte):
        il = self.interaction_list if self.interaction_list.shape[0] else np.empty((0, 2), int)
        return Out(self.eta0, il, Xtr, Xte, lambda X: self.predict(X))


# ----------------------------------------------------------------------------- metrics
def recon(o, side):
    m, i = getattr(o, side)
    return o.eta0 + m.sum(1) + i.sum(1)


def real_metrics(model, Xtr, Xte, o):
    ptr, pte = model.predict(Xtr), model.predict(Xte)
    return {"resid_in": float(np.mean((ptr - recon(o, "tr")) ** 2) / np.var(ptr)),
            "resid_out": float(np.mean((pte - recon(o, "te")) ** 2) / np.var(pte)),
            "ortho_in": harness.orthogonality(ptr, o.tr[0], o.tr[1], o.il),
            "ortho_out": harness.orthogonality(pte, o.te[0], o.te[1], o.il)}


def analytical_metrics(model, Xtr, Xn, o):
    """Same definitions as harness.analytical, plus resid_in / ortho_in / ortho_out."""
    main, inter = o.te
    il = [tuple(p) for p in o.il]
    p = 6
    target = np.zeros((len(Xn), p))
    target[:, 0] = np.sin(2 * np.pi * Xn[:, 0]) + harness.eta_main(Xn[:, 0])
    for j in (1, 2, 3):
        target[:, j] = harness.eta_main(Xn[:, j])
    row = {f"mse_eta{j + 1}": float(np.mean((target[:, j] - main[:, j]) ** 2)) for j in range(p)}
    for name, pair, tgt in (("mse_eta12", (0, 1), harness.eta_order2(Xn[:, 0], Xn[:, 1])),
                            ("mse_eta34", (2, 3), harness.eta_order2(Xn[:, 2], Xn[:, 3]))):
        est = inter[:, il.index(pair)] if pair in il else np.zeros(len(Xn))
        row[name] = float(np.mean((tgt - est) ** 2))
    others = [c for c, q in enumerate(il) if q not in ((0, 1), (2, 3))]
    row["mse_others"] = float(np.mean(inter[:, others] ** 2)) if others else 0.0
    row.update(real_metrics(model, Xtr, Xn, o))
    return row


# ----------------------------------------------------------------------------- purification (R2 d)
def purify(o, Xtr, Xte, nbins=20, iters=10):
    """Label-free post-hoc purification of a decomposition, using X_train only.

    For every interaction: subtract its X_train mean, then remove the additive part in (x_j, x_k)
    (backfitting of quantile-bin means: `nbins` bins, `iters` sweeps) and add the removed parts to
    the matching main effects and the intercept; finally re-centre every main effect on X_train.
    The reconstruction is unchanged by construction. Returns a new Out.
    """
    p = Xtr.shape[1]
    edges = [np.unique(np.quantile(Xtr[:, j], np.linspace(0, 1, nbins + 1)[1:-1])) for j in range(p)]
    b_tr = np.column_stack([np.digitize(Xtr[:, j], edges[j]) for j in range(p)])
    il = o.il
    inter_tr = o.tr[1]
    consts = np.zeros(len(il))
    corr = []
    for c, (j, k) in enumerate(il):
        v = inter_tr[:, c] - inter_tr[:, c].mean()
        consts[c] = inter_tr[:, c].mean()
        nj, nk = len(edges[j]) + 1, len(edges[k]) + 1
        cj = np.maximum(np.bincount(b_tr[:, j], minlength=nj), 1)
        ck = np.maximum(np.bincount(b_tr[:, k], minlength=nk), 1)
        fj, fk = np.zeros(nj), np.zeros(nk)
        for _ in range(iters):
            fj = np.bincount(b_tr[:, j], v - fk[b_tr[:, k]], nj) / cj
            fk = np.bincount(b_tr[:, k], v - fj[b_tr[:, j]], nk) / ck
        corr.append((fj, fk))

    def apply(X):
        m0, i0 = o.pred(X)
        b = np.column_stack([np.digitize(X[:, j], edges[j]) for j in range(p)])
        m, i = m0.copy(), i0.copy()
        for c, (j, k) in enumerate(il):
            fj, fk = corr[c]
            i[:, c] = i0[:, c] - consts[c] - fj[b[:, j]] - fk[b[:, k]]
            m[:, j] += fj[b[:, j]]
            m[:, k] += fk[b[:, k]]
        return m, i

    m_tr, _ = apply(Xtr)
    mu = m_tr.mean(0)          # re-centre the main effects on X_train
    eta0 = o.eta0 + consts.sum() + mu.sum()

    def pred(X):
        m, i = apply(X)
        return m - mu, i
    return Out(eta0, il, Xtr, Xte, pred)


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


def gm(xs):
    xs = np.array([x for x in xs if x is not None and x > 0], float)
    return float(np.exp(np.mean(np.log(xs)))) if len(xs) else None
