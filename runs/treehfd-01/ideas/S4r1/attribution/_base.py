"""Baseline TreeHFD with a switchable solver and unseen-cell rule (attribution runs only).

The original `treehfd` package is used unchanged (read only). Two things can be switched,
through module-level settings that each `method_*.py` in this directory sets on import:

* SOLVER = "lsqr"  : the baseline's `scipy.sparse.linalg.lsqr` with default tolerances.
  SOLVER = "exact" : the same least-squares problem (same matrix and target, built by the
                     baseline code) solved exactly for its minimum-norm solution, which is the
                     point lsqr converges to from x0 = 0. It uses an eigendecomposition of
                     A'A, dropping eigenvalues below 1e-12 * max.
* RULE = "random"      : the baseline's prediction-time fallback for unseen pair cells: the
                         L1-nearest observed cell in bin-index space, with ties broken by
                         the larger count and then at random with an unseeded RNG.
  RULE = "nearest_det" : the same, but the last tie-break takes the smallest cell index, so
                         nothing is random.
  RULE = "zero"        : unseen pair cells take 0 (GT-LOCO's "zero" rule).
  RULE = "harmonic"    : unseen pair cells take the harmonic extension of the observed cells
                         under GT-LOCO's lattice prior (`gtloco.GTLocoTree._lattices`, same
                         geometry and LATTICE_RIDGE).

The coefficients of observed cells and of main bins are exactly the baseline's for a given
SOLVER, so a RULE switch changes only the values of unseen pair cells at prediction time.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import treehfd.tree as _tree_mod
from scipy.sparse.linalg import lsqr as _lsqr
from treehfd import XGBTreeHFD

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import gtloco  # noqa: E402

SOLVER = "lsqr"
RULE = "random"


def _exact(A, b, *args, **kwargs):
    A = A.toarray()
    s, V = np.linalg.eigh(A.T @ A)
    keep = s > 1e-12 * max(float(s[-1]), 1e-300)
    x = V[:, keep] @ ((V[:, keep].T @ (A.T @ b)) / s[keep])
    return (x,)


def _prepare_tree(tree, X):
    """Local bins, quantile positions and the unseen-cell lattice of each pair of a tree."""
    cp = tree.cartesian_partition
    mv = list(cp.main_variables)
    V = len(mv)
    if V == 0:
        tree._rule = None
        return
    n = X.shape[0]
    nb = np.array([len(s) + 1 for s in cp.split_list])
    off = cp.partition_index[:V + 1].astype(int)
    pos = []
    for idx, j in enumerate(mv):
        c = np.bincount(np.digitize(X[:, j], bins=cp.split_list[idx], right=False),
                        minlength=nb[idx])
        pos.append((np.cumsum(c) - c / 2) / n)
    pairs = []
    for k, (a, b) in enumerate(tree.interaction_list):
        ia, ib = mv.index(a), mv.index(b)
        cells = cp.cell_list[k]
        u, v = cells[:, 0] - off[ia], cells[:, 1] - off[ib]
        codes = u * nb[ib] + v
        order = np.argsort(codes)
        start = int(cp.partition_index[V + k])
        pairs.append({"ia": ia, "ib": ib, "codes": codes[order], "u": u[order],
                      "v": v[order], "cnt": cp.counts_list[k][order],
                      "coef": tree.hfd_coeffs[start + order]})
    if RULE == "harmonic" and pairs:
        ns = SimpleNamespace(pairs=pairs, pos=pos)
        gtloco.GTLocoTree._lattices(ns, n)
    for p in pairs:
        na, nbb = nb[p["ia"]], nb[p["ib"]]
        lat = np.zeros(na * nbb)
        lat[p["codes"]] = p["coef"]
        unseen = np.setdiff1d(np.arange(na * nbb), p["codes"])
        if len(unseen) and RULE == "harmonic":
            lat[unseen] = p["E"] @ p["coef"]
        elif len(unseen) and RULE == "nearest_det":
            uu, vv = unseen // nbb, unseen % nbb
            dist = np.abs(uu[:, None] - p["u"][None, :]) + np.abs(vv[:, None] - p["v"][None, :])
            # min distance, then max count, then smallest index (np.lexsort: last key first)
            key = np.lexsort((np.broadcast_to(np.arange(len(p["codes"])), dist.shape),
                              np.broadcast_to(-p["cnt"], dist.shape), dist), axis=1)[:, 0]
            lat[unseen] = p["coef"][key]
        p["lattice"] = lat
        for key in ("S", "E", "unseen"):
            p.pop(key, None)
    tree._rule = {"nb": nb, "off": off, "pairs": pairs}


class BaseTreeHFD(XGBTreeHFD):
    def fit(self, X, interaction_order=2):
        _tree_mod.lsqr = _exact if SOLVER == "exact" else _lsqr
        super().fit(X, interaction_order=interaction_order, verbose=False)
        if RULE != "random":
            for tree in self.treehfd_list:
                _prepare_tree(tree, np.asarray(X, float))

    def predict(self, X_new):
        if RULE == "random":
            return super().predict(X_new, verbose=False)
        n = X_new.shape[0]
        main = np.zeros((n, X_new.shape[1]))
        inter = np.zeros((n, self.interaction_list.shape[0]))
        index = {tuple(p): k for k, p in enumerate(np.asarray(self.interaction_list).tolist())}
        for tree in self.treehfd_list:
            r = tree._rule
            if r is None:
                continue
            cp = tree.cartesian_partition
            lb = np.column_stack([np.digitize(X_new[:, j], bins=cp.split_list[i], right=False)
                                  for i, j in enumerate(cp.main_variables)])
            main[:, cp.main_variables] += tree.hfd_coeffs[lb + r["off"][:-1]]
            for p, pair in zip(r["pairs"], tree.interaction_list, strict=True):
                inter[:, index[tuple(pair)]] += p["lattice"][lb[:, p["ia"]] * r["nb"][p["ib"]]
                                                             + lb[:, p["ib"]]]
        return main, inter


def fit(model, X_train, interaction_order=2):
    hfd = BaseTreeHFD(model)
    hfd.fit(X_train, interaction_order=interaction_order)
    return hfd


def predict(state, X):
    main, inter = state.predict(np.asarray(X, float))
    inter_list = state.interaction_list if inter.shape[1] else np.empty((0, 2), int)
    return {"intercept": state.eta0, "main": main, "inter": inter, "inter_list": inter_list}
