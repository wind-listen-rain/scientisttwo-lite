"""Model-anchored smoothed-measure TreeHFD (idea S2).

TreeHFD estimates the Hoeffding functional decomposition (HFD) of each tree
T_l by a per-tree constrained least-squares problem whose weights come from
the raw empirical measure P_n of X_train. Here P_n is replaced by the
deterministic, coordinate-wise discrete kernel smoother

    mu~ = (1 - alpha) P_n
          + alpha / |S| * sum_{j in S} sum_{s=+-1} 1/2 * P_n o Shift_{j,s}^{-1},

where Shift_{j,s} moves x_ij to the training median of the adjacent
non-empty bin (s = +-1) of the ensemble union-of-splits grid on x_j, and S is
the set of variables split on anywhere in the ensemble. A copy with no
adjacent bin (edge bin) keeps its mass on the original point, so every
training point carries total mass 1/n.

For each tree, the shifted copies fall into the tree's own Cartesian cells.
Copies that cross one of the tree's thresholds create "virtual" full cells
whose target is the tree's exact output at the copy (the fixed model is only
evaluated, never refit). By default (NEW_PAIR_CELLS = "drop") a copy that
would put one of the tree's interactions into a pair cell without empirical
mass is rejected for that tree (its mass stays on the original point), so
virtual atoms only re-anchor pair cells the data populate. Residual rows,
zero-mean rows and hierarchical orthogonality rows are all built from mu~.
At alpha = 0 the per-tree system is exactly the original TreeHFD system.

Other changes w.r.t. the original package:
  * sparse COO assembly of each per-tree system (no dense num_cells x
    partition_size arrays);
  * LSMR with tight tolerances, convergence check (istop), warm-started
    retry and a dense normal-equation fallback;
  * deterministic, geometry-aware fallback for pair cells that are unseen
    at prediction time (precomputed lookup table per interaction);
  * split thresholds and inputs are rounded to float32 exactly as XGBoost
    does, so the Cartesian partitions route points exactly like the model;
  * alpha chosen label-free by 2-fold cross-fitting on X_train, scoring the
    held-out reconstruction error against the (known) tree outputs, with a
    one-standard-error rule towards less smoothing.
"""
from __future__ import annotations

import ast
import json
import os
import time

import numpy as np
import scipy.linalg as sla
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, lsmr
from scipy.special import logit

from treehfd_mod.tree_structure import (
    extract_interactions,
    extract_variable_paths,
    extract_variables,
)

# Hyper-parameters (see NOTES.md).
ALPHA_GRID = (0.0, 0.1, 0.2, 0.35)  # candidate smoothing weights
CV_TREE_STRIDE = 5                  # alpha cross-fit uses every 5th tree
CV_SEED = 0                         # fixed permutation for the 2-fold split
LSMR_TOL = 1e-10                    # atol = btol for LSMR
LSMR_CONLIM = 1e12                  # LSMR condition-number limit
LSMR_MAXITER_FACTOR = 20            # maxiter = factor * num_columns (>= 2000)
LSMR_RETRY_FACTOR = 5               # retry with 5x more iterations
RIDGE_REL = 1e-11                   # preconditioner shift lam / max diag(A^T A)
CONSTRAINT_MEASURE = "smoothed"     # measure of mean/ortho rows ("smoothed" | "empirical")
TIE_RTOL = 1e-9                     # relative tolerance for distance ties
CV_ONE_SE = True                    # smallest alpha within one paired SE of the best CV score
# Pair cells without empirical mass that shifted copies reach:
#   "fit"  - own coefficient column (revisions 1-2);
#   "tie"  - no own column: tied to the column of the cell the prediction-time
#            fallback maps them to (virtual atoms only refine empirical cells);
#   "drop" - the copy is rejected for this tree and its mass stays on the
#            original point (revision 3 default).
NEW_PAIR_CELLS = "drop"


def _f32(a):
    """Round to float32 (as XGBoost does) and return float64."""
    return np.asarray(a, dtype=np.float32).astype(np.float64)


# ---------------------------------------------------------------------------
# Tree parsing and exact tree evaluation.
# ---------------------------------------------------------------------------
class _Tree:
    """Arrays describing one xgboost tree, indexed by node id."""

    def __init__(self, table, offset):
        nodes = table["Node"].to_numpy().astype(int)
        size = int(nodes.max()) + 1
        self.feat = np.full(size, -1, dtype=int)
        self.thr = np.zeros(size)
        self.left = np.zeros(size, dtype=int)
        self.right = np.zeros(size, dtype=int)
        self.missing = np.zeros(size, dtype=int)
        self.leaf = np.zeros(size)
        feat_str = table["Feature"].to_numpy()
        is_leaf = feat_str == "Leaf"
        inner = nodes[~is_leaf]
        self.feat[inner] = [int(f[1:]) for f in feat_str[~is_leaf]]
        self.thr[inner] = _f32(table["Split"].to_numpy()[~is_leaf])

        def child(col):
            return np.array([int(str(s).split("-")[1]) for s in
                             table[col].to_numpy()[~is_leaf]], dtype=int)

        if inner.size:
            self.left[inner] = child("Yes")
            self.right[inner] = child("No")
            self.missing[inner] = child("Missing")
        self.leaf[nodes[is_leaf]] = _f32(table["Gain"].to_numpy()[is_leaf])
        self.offset = offset  # share of base_score attributed to the tree
        # Structure tuple in the format of treehfd.tree_structure.
        self.structure = (self.feat.copy(),
                          np.vstack((self.left, self.right)),
                          self.thr.copy())

    def output(self, X, rows=None, col=None, vals=None):
        """Exact tree output (leaf value + offset).

        If rows is given, evaluates the rows X[rows] with column `col`
        replaced by `vals` (the shifted copies).
        """
        Xr = X if rows is None else X[rows]
        node = np.zeros(Xr.shape[0], dtype=int)
        active = np.nonzero(self.feat[node] >= 0)[0]
        while active.size:
            nd = node[active]
            f = self.feat[nd]
            x = Xr[active, f]
            if col is not None:
                x = np.where(f == col, vals[active], x)
            nxt = np.where(x < self.thr[nd], self.left[nd], self.right[nd])
            nan = np.isnan(x)
            if nan.any():
                nxt[nan] = self.missing[nd[nan]]
            node[active] = nxt
            active = active[self.feat[nxt] >= 0]
        return self.leaf[node] + self.offset


# ---------------------------------------------------------------------------
# Smoothed measure: union-of-splits grid shifts.
# ---------------------------------------------------------------------------
def _shift_values(X, union_splits):
    """Shifted coordinates for every variable split on in the ensemble.

    Returns a dict j -> (down_j, up_j), arrays of length n holding the
    training median of the previous / next non-empty union bin of x_ij
    (NaN if x_ij lies in the first / last non-empty bin).
    """
    shifts = {}
    for j, U in union_splits.items():
        x = X[:, j]
        u = np.searchsorted(U, x, side="right")
        occ, inv = np.unique(u, return_inverse=True)
        order = np.argsort(u, kind="stable")
        bounds = np.searchsorted(u[order], occ, side="left")
        bounds = np.append(bounds, len(x))
        rep = np.array([np.median(x[order[bounds[b]:bounds[b + 1]]])
                        for b in range(len(occ))])
        rep = _f32(rep)
        down = np.full(len(x), np.nan)
        up = np.full(len(x), np.nan)
        has_down = inv > 0
        has_up = inv < len(occ) - 1
        down[has_down] = rep[inv[has_down] - 1]
        up[has_up] = rep[inv[has_up] + 1]
        shifts[j] = (down, up)
    return shifts


def _merge_empty_bins(split, x):
    """Drop tree thresholds delimiting bins without training points.

    Same rule as the original package: an empty bin at an edge is merged
    into its neighbour, an interior empty region is split at its midpoint.
    """
    b = np.searchsorted(split, x, side="right")
    occ = np.unique(b)
    if len(occ) == len(split) + 1:
        return split
    new = []
    for lo, hi in zip(occ[:-1], occ[1:]):
        if hi == lo + 1:
            new.append(split[lo])
        else:
            new.append(0.5 * (split[lo] + split[hi - 1]))
    return np.array(new, dtype=float)


def _compress_rows(B, radix):
    """Integer id of each distinct row of the bin matrix B."""
    key = np.zeros(B.shape[0], dtype=np.int64)
    span = 1
    for c in range(B.shape[1]):
        if span * radix[c] >= 2 ** 62:
            _, key = np.unique(key, return_inverse=True)
            key = key.astype(np.int64)
            span = int(key.max()) + 1
        key = key * radix[c] + B[:, c]
        span *= int(radix[c])
    uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
    return inv.ravel(), first


# ---------------------------------------------------------------------------
# Per-tree preparation and solve.
# ---------------------------------------------------------------------------
class _Prepared:
    pass


def _prepare_tree(tree, pairs_all, variables, X, shifts, num_shift):
    """Partitions, atoms of mu~ and full-cell structure for one tree."""
    n = X.shape[0]
    P = _Prepared()
    P.variables = variables
    P.pairs = pairs_all
    P.n = n
    y0 = tree.output(X)
    if variables.size == 0:
        P.const = float(np.mean(y0))
        return P
    P.const = None
    V = len(variables)
    split_list, nbins = [], np.zeros(V, dtype=int)
    B0 = np.zeros((n, V), dtype=int)
    for vi, j in enumerate(variables):
        s = np.unique(tree.thr[tree.feat == j])
        s = _merge_empty_bins(s, X[:, j])
        split_list.append(s)
        nbins[vi] = len(s) + 1
        B0[:, vi] = np.searchsorted(s, X[:, j], side="right")
    P.split_list, P.nbins = split_list, nbins
    vpos = {int(j): vi for vi, j in enumerate(variables)}
    pair_local = [(vpos[j], vpos[k]) for (j, k) in pairs_all]
    if NEW_PAIR_CELLS == "drop":  # empirical pair cells of each interaction
        emp_keys = [np.unique(B0[:, ji] * nbins[ki] + B0[:, ki]) for ji, ki in pair_local]

    # Shifted copies crossing a tree threshold -> virtual atoms.
    moved_count = np.zeros(n, dtype=int)
    Bc_list, yc_list = [], []
    for vi, j in enumerate(variables):
        if j not in shifts:
            continue
        for vals in shifts[j]:
            valid = np.nonzero(~np.isnan(vals))[0]
            nb = np.searchsorted(split_list[vi], vals[valid], side="right")
            mv = nb != B0[valid, vi]
            rows = valid[mv]
            if rows.size == 0:
                continue
            Bc = B0[rows].copy()
            Bc[:, vi] = nb[mv]
            if NEW_PAIR_CELLS == "drop":
                ok = np.ones(rows.size, dtype=bool)
                for q, (ji, ki) in enumerate(pair_local):
                    if vi in (ji, ki):
                        ok &= np.isin(Bc[:, ji] * nbins[ki] + Bc[:, ki], emp_keys[q])
                rows, Bc = rows[ok], Bc[ok]
                if rows.size == 0:
                    continue
            Bc_list.append(Bc)
            yc_list.append(tree.output(X, rows=rows, col=j, vals=vals[rows]))
            np.add.at(moved_count, rows, 1)
    if Bc_list:
        Bc = np.vstack(Bc_list)
        yc = np.concatenate(yc_list)
    else:
        Bc = np.zeros((0, V), dtype=int)
        yc = np.zeros(0)
    m0, mc = n, Bc.shape[0]
    B = np.vstack((B0, Bc))
    y = np.concatenate((y0, yc))
    # Atom weights: w = (1 - alpha) * wa + alpha * wb.
    P.wa = np.concatenate((np.full(m0, 1.0 / n), np.zeros(mc)))
    kept = num_shift - moved_count
    P.wb = np.concatenate((kept / (num_shift * n), np.full(mc, 1.0 / (num_shift * n))))
    P.num_virtual = mc

    # Unique full cells.
    full_id, first = _compress_rows(B, nbins)
    P.full_id = full_id
    P.full_bins = B[first]
    P.nfull = len(first)
    # T is constant on a full cell; average for numerical safety.
    cnt = np.bincount(full_id, minlength=P.nfull)
    P.full_y = np.bincount(full_id, weights=y, minlength=P.nfull) / cnt
    P.y_spread = float(np.max(np.abs(y - P.full_y[full_id]))) if len(y) else 0.0

    # Pair cells of each interaction, indexed from full cells.
    P.pair_local, P.pair_id, P.pair_bins = pair_local, [], []
    for ji, ki in pair_local:
        key = P.full_bins[:, ji] * nbins[ki] + P.full_bins[:, ki]
        uniq, inv = np.unique(key, return_inverse=True)
        P.pair_id.append(inv.ravel())
        P.pair_bins.append(np.column_stack((uniq // nbins[ki], uniq % nbins[ki])))
    # Geometry for the prediction-time fallback (bin midpoints / IQR).
    P.mid = []
    for vi, j in enumerate(variables):
        x = X[:, j]
        lo, hi = float(np.min(x)), float(np.max(x))
        edges = np.concatenate(([lo], np.clip(split_list[vi], lo, hi), [hi]))
        q75, q25 = np.percentile(x, [75, 25])
        scale = q75 - q25
        if not scale > 0:
            scale = np.std(x)
        if not scale > 0:
            scale = 1.0
        P.mid.append(0.5 * (edges[:-1] + edges[1:]) / scale)
    return P


def _solve_ls(A, b, stats):
    """Minimum-norm least-squares solution of A x = b.

    LSMR (atol = btol = LSMR_TOL) right-preconditioned by the Cholesky
    factor R of A^T A + lam I (lam = RIDGE_REL * max diag). Starting from
    z = 0, LSMR on A R^{-1} returns the minimiser of ||R x||^2 =
    ||A x||^2 + lam ||x||^2 among least-squares solutions, i.e. the same
    minimum-norm solution plain LSQR/LSMR converge to (the per-tree systems
    are often rank deficient), but in a handful of iterations. istop is
    checked: on non-convergence LSMR is re-run warm-started with a larger
    iteration limit, then a dense normal-equation lstsq is used.
    """
    ncol = A.shape[1]
    maxiter = max(2000, LSMR_MAXITER_FACTOR * ncol)
    G = (A.T @ A).toarray()
    lam = RIDGE_REL * max(float(G.diagonal().max()), 1e-300)
    R = None
    for _ in range(4):
        try:
            R = sla.cholesky(G + lam * np.eye(ncol), lower=False, check_finite=False)
            break
        except np.linalg.LinAlgError:
            lam *= 100.0
    if R is None:  # no usable preconditioner: plain LSMR
        stats["no_precond"] += 1
        op, back = A, (lambda z: z)
    else:
        op = LinearOperator(
            A.shape, dtype=float,
            matvec=lambda z: A @ sla.solve_triangular(R, z, check_finite=False),
            rmatvec=lambda u: sla.solve_triangular(R, A.T @ u, trans="T",
                                                   check_finite=False))
        back = lambda z: sla.solve_triangular(R, z, check_finite=False)  # noqa: E731
    res = lsmr(op, b, atol=LSMR_TOL, btol=LSMR_TOL, conlim=LSMR_CONLIM, maxiter=maxiter)
    z, istop, itn = res[0], int(res[1]), int(res[2])
    stats["itn"].append(itn)
    stats["istop"][istop] = stats["istop"].get(istop, 0) + 1
    if istop in (3, 6, 7):  # ill-conditioned or iteration limit reached
        stats["retry"] += 1
        res = lsmr(op, b, atol=LSMR_TOL, btol=LSMR_TOL, conlim=LSMR_CONLIM,
                   maxiter=LSMR_RETRY_FACTOR * maxiter, x0=z)
        z, istop = res[0], int(res[1])
        if istop in (3, 6, 7):
            stats["direct"] += 1
            return np.linalg.lstsq(G, A.T @ b, rcond=None)[0]
    return back(z)


class _FittedTree:
    pass


def _solve_tree(P, alpha, stats):
    """Build the sparse per-tree system under mu~(alpha) and solve it."""
    F = _FittedTree()
    F.variables = P.variables
    F.pairs = P.pairs
    if P.const is not None:
        F.eta0 = P.const
        return F
    A, rhs, S = _build_system(P, alpha)
    F.eta0 = S["eta0"]
    coef = _solve_ls(A, rhs, stats)

    # Store prediction tables.
    V = len(P.variables)
    main_off = S["main_off"]
    F.split_list = P.split_list
    F.main_vals = [coef[main_off[vi]:main_off[vi + 1]] for vi in range(V)]
    F.pair_local = P.pair_local
    F.pair_tables = []
    F.pair_kind = []  # diagnostics: 0 empirical mass, 1 virtual-only, 2 no mass
    for q, (ji, ki) in enumerate(P.pair_local):
        own, has = S["pair_own"][q], S["pair_has"][q]
        stats["tied_cells"] += int(np.sum(has & ~own))
        F.pair_tables.append(_pair_table(P.pair_bins[q][own], coef[S["pair_cols"][q][own]],
                                         S["pair_W"][q][own], P.nbins[ji], P.nbins[ki],
                                         P.mid[ji], P.mid[ki], stats))
        kind = np.full((P.nbins[ji], P.nbins[ki]), 2, dtype=np.int8)
        pb = P.pair_bins[q][has]
        kind[pb[:, 0], pb[:, 1]] = np.where(S["pair_emp"][q][has] > 0, 0, 1)
        F.pair_kind.append(kind)
    F.num_virtual = P.num_virtual
    return F


def _build_system(P, alpha):
    """Sparse least-squares system (A, rhs) of one tree under mu~(alpha)."""
    n = P.n
    w = (1.0 - alpha) * P.wa + alpha * P.wb
    W_all = np.bincount(P.full_id, weights=w, minlength=P.nfull)
    keep = W_all > 0
    Wf = W_all[keep]
    Wf = Wf / Wf.sum()
    bins_f = P.full_bins[keep]
    yf = P.full_y[keep]
    nf = len(Wf)
    # Measure of the zero-mean / orthogonality rows (and of the intercept).
    W_emp = np.bincount(P.full_id, weights=P.wa, minlength=P.nfull)[keep]
    Wc = W_emp if CONSTRAINT_MEASURE == "empirical" else Wf
    eta0 = float(np.sum(Wc * yf))
    V = len(P.variables)
    nbins = P.nbins
    main_off = np.concatenate(([0], np.cumsum(nbins)))
    ncol = int(main_off[-1])

    # Pair columns. Every pair cell with positive smoothed mass owns a column
    # (NEW_PAIR_CELLS = "fit"), or only those with positive empirical mass
    # ("tie"/"drop"); the others share the column of the owning cell the
    # prediction-time fallback maps them to.
    pair_cols, pair_has, pair_own, pair_W, pair_Wc, pair_emp, pair_fid = ([] for _ in range(7))
    for q, (ji, ki) in enumerate(P.pair_local):
        fid = P.pair_id[q][keep]
        L = len(P.pair_bins[q])
        Wq = np.bincount(fid, weights=Wf, minlength=L)
        Eq = np.bincount(fid, weights=W_emp, minlength=L)
        has = Wq > 0
        own = has if NEW_PAIR_CELLS == "fit" else has & (Eq > 0)
        col = np.full(L, -1, dtype=int)
        col[own] = ncol + np.arange(int(own.sum()))
        ncol += int(own.sum())
        tied = has & ~own
        if tied.any():
            near = _nearest_cell(P.pair_bins[q][tied], P.pair_bins[q][own], Wq[own],
                                 P.mid[ji], P.mid[ki])
            col[tied] = col[own][near]
        pair_cols.append(col)
        pair_has.append(has)
        pair_own.append(own)
        pair_W.append(Wq)
        pair_Wc.append(np.bincount(fid, weights=Wc, minlength=L))
        pair_emp.append(Eq)
        pair_fid.append(fid)

    rows, cols, vals = [], [], []
    target = []
    r0 = 0
    # Residual rows: one per full cell with positive mass.
    sw = n * np.sqrt(Wf)
    nnz_row = V + len(P.pairs)
    C = np.empty((nf, nnz_row), dtype=int)
    for vi in range(V):
        C[:, vi] = main_off[vi] + bins_f[:, vi]
    for q in range(len(P.pairs)):
        C[:, V + q] = pair_cols[q][pair_fid[q]]
    rows.append(np.repeat(np.arange(nf), nnz_row))
    cols.append(C.ravel())
    vals.append(np.repeat(sw, nnz_row))
    target.append(sw * (yf - eta0))
    r0 += nf
    # Zero-mean rows (main effects, then interactions).
    mu_main = []
    for vi in range(V):
        mu = np.bincount(bins_f[:, vi], weights=Wc, minlength=nbins[vi])
        mu_main.append(mu)
        rows.append(np.full(nbins[vi], r0))
        cols.append(main_off[vi] + np.arange(nbins[vi]))
        vals.append(n * mu)
        r0 += 1
    # Tied cells repeat their column; the COO -> CSR conversion sums duplicates.
    for q in range(len(P.pairs)):
        has = pair_has[q]
        rows.append(np.full(int(has.sum()), r0))
        cols.append(pair_cols[q][has])
        vals.append(n * pair_Wc[q][has])
        r0 += 1
    target.append(np.zeros(r0 - nf))
    # Hierarchical orthogonality rows: E_mu~[eta_jk | X_j in bin] = 0.
    r_ortho = r0
    for q in range(len(P.pairs)):
        pos = pair_has[q] & (pair_Wc[q] > 0)
        c = pair_cols[q][pos]
        Wq = pair_Wc[q][pos]
        for axis in (0, 1):
            b = P.pair_bins[q][pos, axis]
            present, rank = np.unique(b, return_inverse=True)
            marg = np.bincount(rank, weights=Wq)
            rows.append(r0 + rank.ravel())
            cols.append(c)
            vals.append(n * Wq / np.sqrt(marg[rank.ravel()]))
            r0 += len(present)
    target.append(np.zeros(r0 - r_ortho))

    A = sparse.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(r0, ncol)).tocsr()
    rhs = np.concatenate(target)
    return A, rhs, {"eta0": eta0, "main_off": main_off, "pair_cols": pair_cols,
                    "pair_has": pair_has, "pair_own": pair_own, "pair_W": pair_W,
                    "pair_emp": pair_emp}


def _nearest_cell(miss, pb, Wq, mid_j, mid_k):
    """Index (into pb) of the trained pair cell nearest to each cell of miss.

    Distance between bin midpoints scaled by the training IQR of each
    variable; ties -> largest smoothed mass Wq -> lowest cell index.
    """
    cj, ck = mid_j[pb[:, 0]], mid_k[pb[:, 1]]
    d2 = ((mid_j[miss[:, 0]][:, None] - cj[None, :]) ** 2
          + (mid_k[miss[:, 1]][:, None] - ck[None, :]) ** 2)
    dmin = d2.min(axis=1, keepdims=True)
    cand = d2 <= dmin * (1.0 + TIE_RTOL) + 1e-300
    score = np.where(cand, Wq[None, :], -np.inf)
    return np.argmax(score, axis=1)  # first max -> lowest index


def _pair_table(pb, cv, Wq, nbj, nbk, mid_j, mid_k, stats):
    """Lookup table (nbj x nbk) of interaction values with fallback.

    Cells without a coefficient of their own take the value of the nearest
    trained cell (`_nearest_cell`), the same mapping the fit uses for tied
    cells.
    """
    table = np.full((nbj, nbk), np.nan)
    table[pb[:, 0], pb[:, 1]] = cv
    miss = np.argwhere(np.isnan(table))
    if len(miss):
        stats["fallback_cells"] += int(len(miss))
        table[miss[:, 0], miss[:, 1]] = cv[_nearest_cell(miss, pb, Wq, mid_j, mid_k)]
    stats["table_cells"] += int(nbj * nbk)
    return table


def _predict_tree(F, X):
    """Main-effect (n x |V|) and interaction (n x |pairs|) values of a tree."""
    n = X.shape[0]
    if F.variables.size == 0:
        return np.zeros((n, 0)), np.zeros((n, 0))
    V = len(F.variables)
    Bn = np.empty((n, V), dtype=int)
    main = np.empty((n, V))
    for vi, j in enumerate(F.variables):
        Bn[:, vi] = np.searchsorted(F.split_list[vi], X[:, j], side="right")
        main[:, vi] = F.main_vals[vi][Bn[:, vi]]
    inter = np.empty((n, len(F.pairs)))
    for q, (ji, ki) in enumerate(F.pair_local):
        inter[:, q] = F.pair_tables[q][Bn[:, ji], Bn[:, ki]]
    return main, inter


# ---------------------------------------------------------------------------
# Ensemble-level class.
# ---------------------------------------------------------------------------
class SmoothedTreeHFD:
    """TreeHFD of an xgboost regressor under the smoothed input measure."""

    def __init__(self, xgb_model, alpha_grid=ALPHA_GRID, alpha=None):
        booster = xgb_model.get_booster()
        config = json.loads(booster.save_config())
        gb = config["learner"]["gradient_booster"]
        mp = config["learner"]["learner_model_param"]
        if int(mp.get("num_class", "0")) > 1 or int(mp.get("num_target", "1")) != 1:
            raise NotImplementedError("Only single-output models are supported.")
        if int(gb["gbtree_model_param"]["num_parallel_tree"]) != 1:
            raise NotImplementedError("Only gradient boosting (num_parallel_tree=1).")
        self.max_depth = int(gb["tree_train_param"]["max_depth"])
        self.num_feature = int(mp["num_feature"])
        base = float(np.array(ast.literal_eval(mp["base_score"])).ravel()[0])
        if config["learner"]["objective"]["name"] == "binary:logistic":
            base = float(logit(base))
        table = booster.trees_to_dataframe()
        num_trees = int(table["Tree"].max()) + 1
        self.trees, self.tree_vars, self.tree_pairs = [], [], []
        for t in range(num_trees):
            tt = table[table["Tree"] == t]
            tree = _Tree(tt, base / num_trees)
            self.trees.append(tree)
            if (tree.feat >= 0).any():
                paths = extract_variable_paths(tree.structure, self.max_depth)
                self.tree_vars.append(np.asarray(extract_variables(paths), dtype=int))
                self.tree_pairs.append([tuple(int(a) for a in p)
                                        for p in extract_interactions(paths)])
            else:
                self.tree_vars.append(np.empty(0, dtype=int))
                self.tree_pairs.append([])
        # Ensemble union-of-splits grid.
        splits = table[table["Feature"] != "Leaf"]
        self.union_splits = {}
        for f, g in splits.groupby("Feature"):
            self.union_splits[int(f[1:])] = np.unique(_f32(g["Split"].to_numpy()))
        self.num_shift = 2 * len(self.union_splits)
        self.alpha_grid = tuple(alpha_grid)
        self.alpha = alpha
        self.diag = {}

    def _pairs(self, t, interaction_order):
        return self.tree_pairs[t] if interaction_order == 2 else []

    def fit(self, X, interaction_order=2):
        t_start = time.time()
        X = _f32(X)
        self.interaction_order = interaction_order
        stats = _new_stats()
        if self.alpha is None:
            self.alpha, cv_scores = self._select_alpha(X, interaction_order)
            self.diag["cv_scores"] = cv_scores
        t_cv = time.time()
        shifts = _shift_values(X, self.union_splits)
        self.fitted = []
        num_virtual = 0
        for t, tree in enumerate(self.trees):
            P = _prepare_tree(tree, self._pairs(t, interaction_order),
                              self.tree_vars[t], X, shifts, self.num_shift)
            F = _solve_tree(P, self.alpha, stats)
            num_virtual += getattr(P, "num_virtual", 0)
            self.fitted.append(F)
        self.eta0 = float(sum(F.eta0 for F in self.fitted))
        pairs = sorted({p for F in self.fitted for p in F.pairs})
        self.interaction_list = (np.array(pairs, dtype=int).reshape(-1, 2)
                                 if pairs else np.empty((0, 2), dtype=int))
        self._pair_index = {p: i for i, p in enumerate(pairs)}
        self.diag.update({
            "alpha": self.alpha, "cv_s": t_cv - t_start,
            "main_fit_s": time.time() - t_cv,
            "virtual_atoms_per_tree": num_virtual / len(self.trees),
            "lsmr_itn_mean": float(np.mean(stats["itn"])) if stats["itn"] else 0.0,
            "lsmr_itn_max": int(np.max(stats["itn"])) if stats["itn"] else 0,
            "lsmr_istop": stats["istop"], "lsmr_retry": stats["retry"],
            "direct_fallback": stats["direct"], "no_precond": stats["no_precond"],
            "fallback_cells": stats["fallback_cells"],
            "table_cells": stats["table_cells"], "tied_cells": stats["tied_cells"],
            "new_pair_cells": NEW_PAIR_CELLS,
        })
        path = os.environ.get("S2_DIAG_PATH")
        if path:
            with open(path, "a") as fh:
                fh.write(json.dumps({"n": int(X.shape[0]), "p": int(X.shape[1]),
                                     **self.diag}) + "\n")
        return self

    def _select_alpha(self, X, interaction_order):
        """2-fold cross-fit of alpha on X_train (label-free).

        Score: held-out normalised reconstruction error against the (known)
        tree outputs. With CV_ONE_SE, the smallest alpha whose score is
        within one paired standard error (per-point differences to the best
        alpha, pooled over both folds) of the best score is returned, i.e.
        the least departure from the empirical measure that is not
        significantly worse; otherwise the argmin.
        """
        grid = self.alpha_grid
        if len(grid) == 1:
            return grid[0], {}
        n = X.shape[0]
        perm = np.random.default_rng(CV_SEED).permutation(n)
        halves = (np.sort(perm[: n // 2]), np.sort(perm[n // 2:]))
        cv_trees = range(0, len(self.trees), CV_TREE_STRIDE)
        scores = np.zeros(len(grid))
        stats = _new_stats()
        point_err = []  # per held-out point normalised squared errors (grid x n_B), per fold
        for fold in range(2):
            XA, XB = X[halves[fold]], X[halves[1 - fold]]
            shiftsA = _shift_values(XA, self.union_splits)
            T_sum = np.zeros(XB.shape[0])
            R = np.zeros((len(grid), XB.shape[0]))
            for t in cv_trees:
                tree = self.trees[t]
                TB = tree.output(XB)
                T_sum += TB
                P = _prepare_tree(tree, self._pairs(t, interaction_order),
                                  self.tree_vars[t], XA, shiftsA, self.num_shift)
                for a, alpha in enumerate(grid):
                    F = _solve_tree(P, alpha, stats)
                    main, inter = _predict_tree(F, XB)
                    R[a] += TB - (F.eta0 + main.sum(1) + inter.sum(1))
            var = np.var(T_sum)
            scores += np.mean(R ** 2, axis=1) / (var if var > 0 else 1.0)
            point_err.append(R ** 2 / (var if var > 0 else 1.0))
        scores /= 2
        best = int(np.argmin(scores))  # ties -> smallest alpha
        E = np.concatenate(point_err, axis=1)
        se = np.array([np.std(E[a] - E[best], ddof=1) / np.sqrt(E.shape[1])
                       for a in range(len(grid))])
        pick = best
        if CV_ONE_SE:
            pick = next(a for a in np.argsort(grid, kind="stable")
                        if scores[a] <= scores[best] + se[a])
        self.diag["cv_se"] = {str(a): float(s) for a, s in zip(grid, se)}
        return float(grid[pick]), {str(a): float(s) for a, s in zip(grid, scores)}

    def predict(self, X):
        X = _f32(X)
        n = X.shape[0]
        main = np.zeros((n, self.num_feature))
        inter = np.zeros((n, self.interaction_list.shape[0]))
        for F in self.fitted:
            if F.variables.size == 0:
                continue
            m, it = _predict_tree(F, X)
            main[:, F.variables] += m
            if it.shape[1]:
                idx = [self._pair_index[p] for p in F.pairs]
                inter[:, idx] += it
        return main, inter


def _new_stats():
    return {"itn": [], "istop": {}, "retry": 0, "direct": 0, "no_precond": 0,
            "fallback_cells": 0, "table_cells": 0, "tied_cells": 0}
