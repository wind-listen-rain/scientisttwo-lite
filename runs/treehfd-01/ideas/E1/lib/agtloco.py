"""Anchored GT-LOCO TreeHFD (idea E1).

Starts from GT-LOCO TreeHFD (idea S4, copied verbatim below where unchanged): for each tree,
the TreeHFD least-squares problem (zero-mean rows, residual rows, exact hierarchical
orthogonality through a null-space parametrisation beta = Q gamma) is augmented with a
Tikhonov penalty (kappa / n) * beta' P beta ("ridge" or "lattice" prior), and the strength is
selected label-free by the Good-Turing leave-one-cell-out risk R.

E1 adds a second, data-dependent Gaussian prior centred on the fixed tree's own outputs at
"virtual atoms" (the model-anchored copies of idea S2). Per tree t it minimises

    (1/n) sum_i (h_i' beta - y_i)^2 + rho * sum_v w_v (h_v' beta - y_v)^2 + (kappa/n) beta' P beta

subject to the mean and hierarchical-orthogonality constraints, which stay HARD and under
the empirical measure P_n (so the estimand is TreeHFD's empirical HFD).
  * v runs over virtual atoms: each copy of x_i moves one split variable x_ij to the
    training median of the adjacent non-empty bin of the ensemble union-of-splits grid
    (float32, as XGBoost). A copy is kept for tree t only if it changes one of tree t's
    main bins and every pair cell it touches has empirical mass ("drop" rule of S2).
  * y_v = T_t(copy) - eta0 is read exactly from the fixed tree's leaves (no refit, no RNG).
  * w_v = 1 / (2 |S| n), S = variables split on anywhere, so rho is the virtual-to-real mass
    ratio. rho = 0 is exactly S4.
rho enters the normal matrix and right-hand side linearly (N + rho A, rhs + rho b), so each
(prior, rho) needs one eigendecomposition and the kappa path stays in closed form.

The leave-out of the risk R is made honest for the anchor: a Good-Turing singleton i is
removed together with every virtual row that depends on it (block Woodbury downdate):
  * its own copies;
  * copies of other points shifted into a union bin whose only member is i (their location
    is x_ij itself);
  * copies of other points touching a pair cell or main bin that empties without i ("drop"
    would reject them); that cell then becomes latent and takes the unseen-cell rule value.
The ensemble step of S4 is kept, with rho as a new candidate dimension (common rho, or the
per-tree argmin of R over (rho, kappa)); ties go to the smaller rho, then the smaller kappa.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import null_space

from treehfd_mod.cartesian_partition import CartesianTreePartition
from treehfd_mod.ensemble import XGBTreeHFD
from treehfd_mod.tree_structure import (
    extract_interactions,
    extract_tree_structure,
    extract_variable_paths,
    extract_variables,
)

# ---------------------------------------------------------------- S4 settings (unchanged)
# Strengths in pseudo-count units: lambda = kappa / n, i.e. a unit ridge weight equals the
# residual weight of kappa training points.
KAPPAS = np.array([1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 1e4])
EPS_MAIN = 1.0             # ridge weight of main-effect bins relative to interaction cells
PRIORS = ("ridge", "lattice")
LATTICE_RIDGE = 0.1        # ridge part of the lattice prior (edge weights have mean 1)
RULES = ("zero", "harmonic")
SHIFTS = np.arange(-5, 4)  # global shifts of the per-tree kappa index (ensemble step)
SE_RULE = 0.0              # k-SE rule of the ensemble step (0 = plain argmin, as S4)
KEEP_PATH = False          # development only: keep the whole path after fitting
HARD_ORTHO = True          # exact per-tree orthogonality
# ---------------------------------------------------------------- E1 settings
RHOS = (0.0, 0.25, 1.0)    # anchor strength (virtual-to-real mass ratio); must start at 0
ANCHOR_PRIORS = ("ridge", "lattice")  # priors for which rho > 0 is fitted (compute fallback)
BLOCK_CHUNK = 4e6          # max elements of a batched block temporary (memory bound)


def variants() -> list[tuple[int, int]]:
    """(prior, rule) index pairs; order encodes the tie-break preference."""
    return [(pr, ru) for pr in range(len(PRIORS)) for ru in range(len(RULES))]


def _first_min(risk: np.ndarray, axis: int = -1) -> np.ndarray:
    """Index of the first entry within a relative 1e-9 of the minimum (tie-break)."""
    risk = np.nan_to_num(risk, nan=np.inf)
    lo = np.min(risk, axis=axis, keepdims=True)
    return np.argmax(risk <= lo * (1 + 1e-9) + 1e-300, axis=axis)


def _nearest_cells(pos_a, pos_b, seen_u, seen_v, seen_cnt, query_u, query_v,
                   exclude=None):
    """Nearest observed pair cell (quantile coordinates) for each query cell."""
    du = pos_a[query_u][:, None] - pos_a[seen_u][None, :]
    dv = pos_b[query_v][:, None] - pos_b[seen_v][None, :]
    dist = np.round(du ** 2 + dv ** 2, 12)
    if exclude is not None:
        dist[np.arange(len(query_u)), exclude] = np.inf
    key = np.lexsort((np.broadcast_to(np.arange(len(seen_u)), dist.shape),
                      np.broadcast_to(-seen_cnt, dist.shape), dist), axis=1)
    return key[:, 0]


def _inv_sqrt(B: np.ndarray) -> np.ndarray:
    s, V = np.linalg.eigh(B)
    return (V / np.sqrt(np.maximum(s, 1e-300))) @ V.T


def _f32(a) -> np.ndarray:
    """Round to float32 (as XGBoost does) and return float64."""
    return np.asarray(a, dtype=np.float32).astype(np.float64)


# ------------------------------------------------------------------ virtual atoms (from S2)
class _TreeEval:
    """Exact evaluation of one xgboost tree (float32 thresholds, as XGBoost routes)."""

    def __init__(self, table: pd.DataFrame) -> None:
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

    def output(self, X, rows=None, col=None, vals=None) -> np.ndarray:
        """Leaf value of X[rows] with column `col` replaced by `vals` (the copies)."""
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
        return self.leaf[node]


def _union_splits(xgb_table: pd.DataFrame) -> dict:
    """Ensemble union-of-splits grid (float32) of every variable split on."""
    splits = xgb_table[xgb_table["Feature"] != "Leaf"]
    return {int(f[1:]): np.unique(_f32(g["Split"].to_numpy()))
            for f, g in splits.groupby("Feature")}


def _shift_values(X: np.ndarray, X32: np.ndarray, union_splits: dict) -> dict:
    """Shifted coordinates of every training point, for every variable split on.

    Returns j -> [(down, own_down), (up, own_up)]: the training median of the previous /
    next non-empty union bin of x_ij (NaN if there is none) and, when that target bin holds
    a single training point, the index of that point (else -1). Union-bin membership uses
    float32 inputs and thresholds (XGBoost routing); the median is taken on the raw inputs,
    so a copy is binned by the Cartesian partition exactly like a training point with the
    same value (the tree itself is evaluated at the float32 value).
    """
    shifts = {}
    for j, U in union_splits.items():
        x = X[:, j]
        u = np.searchsorted(U, X32[:, j], side="right")
        occ, inv, cnt = np.unique(u, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        order = np.argsort(u, kind="stable")
        bounds = np.append(np.searchsorted(u[order], occ, side="left"), len(x))
        rep = np.array([np.median(x[order[bounds[b]:bounds[b + 1]]])
                        for b in range(len(occ))])
        sole = np.where(cnt == 1, order[bounds[:-1]], -1)
        out = []
        for d in (-1, 1):
            tgt = inv + d
            ok = (tgt >= 0) & (tgt < len(occ))
            vals = np.full(len(x), np.nan)
            own = np.full(len(x), -1, dtype=int)
            vals[ok] = rep[tgt[ok]]
            own[ok] = sole[tgt[ok]]
            out.append((vals, own))
        shifts[j] = out
    return shifts


class GTLocoTree:
    """Anchored GT-LOCO decomposition of a single tree."""

    def __init__(self, tree_table: pd.DataFrame, interaction_order: int,
                 depth_variable: int) -> None:
        self.table = tree_table
        self.tree_structure = extract_tree_structure(tree_table)
        self.interaction_list: list[list[int]] = []
        main_variables = np.empty(0, dtype=int)
        if len(self.tree_structure[0]) > 0:
            paths = extract_variable_paths(self.tree_structure, depth_variable)
            main_variables = extract_variables(paths)
            if interaction_order == 2:  # noqa: PLR2004
                self.interaction_list = extract_interactions(paths)
        self.main_variables = np.asarray(main_variables, dtype=int)
        self.eta0 = 0.0
        self.betas = np.empty((0, 0, 0, 0))
        self.beta = np.empty(0)
        self.pairs: list[dict] = []
        self.split_list: list[np.ndarray] = []
        self.kappa_idx = 0
        self.variant = 0
        self.rho_idx = 0
        self.n_virtual = 0
        self.route_err = 0.0

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y_tree: np.ndarray, anchor: dict | None = None) -> tuple:
        """Fit every (prior, rho, kappa).

        Returns leave-out residuals (variant, rho, kappa, n), in-sample residuals
        (prior, rho, kappa, n), risks (variant, rho, kappa), N1/n, m, number of virtual atoms.
        """
        n = X.shape[0]
        G = len(KAPPAS)
        NV, NR = len(variants()), len(RHOS)
        self.eta0 = float(np.mean(y_tree))
        y = y_tree - self.eta0
        if self.main_variables.size == 0:
            loo = np.broadcast_to(y, (NV, NR, G, n)).astype(np.float32)
            ins = np.broadcast_to(y, (len(PRIORS), NR, G, n)).astype(np.float32)
            return loo, ins, np.mean(loo.astype(float) ** 2, axis=3), 0.0, 0, 0

        # Cartesian partitions of main effects (baseline code, incl. empty-bin removal).
        part = CartesianTreePartition(self.main_variables)
        glob = part.compute_partition_main(X, self.tree_structure[0],
                                           self.tree_structure[2])
        self.split_list = part.split_list
        off_main = part.partition_index.astype(int)
        self.off_main = off_main
        nb = np.diff(off_main)
        self.nb = nb
        lb = glob - off_main[:-1]
        V = len(self.main_variables)
        mv = list(self.main_variables)
        m_main = int(off_main[-1])
        cnt_main = np.bincount(glob.ravel(), minlength=m_main)
        # Quantile coordinate of each main bin (empirical CDF at the bin centre).
        self.pos = []
        for j in range(V):
            c = cnt_main[off_main[j]:off_main[j + 1]]
            self.pos.append((np.cumsum(c) - c / 2) / n)

        # Observed pair cells of each interaction.
        cols = [glob]
        off = m_main
        self.pairs = []
        for a, b in self.interaction_list:
            ia, ib = mv.index(a), mv.index(b)
            code = lb[:, ia] * nb[ib] + lb[:, ib]
            uniq, inv, cnt = np.unique(code, return_inverse=True, return_counts=True)
            cols.append((off + inv.ravel())[:, None])
            self.pairs.append({"ia": ia, "ib": ib, "codes": uniq, "off": off,
                               "cnt": cnt, "u": uniq // nb[ib], "v": uniq % nb[ib]})
            off += len(uniq)
        m = int(off)
        C = np.hstack(cols)
        ncomp = C.shape[1]
        cnt_col = np.bincount(C.ravel(), minlength=m)
        sizes = np.concatenate([nb, [len(p["codes"]) for p in self.pairs]]).astype(int)
        comp_of_col = np.repeat(np.arange(ncomp), sizes)

        # Normal equations, scaled by 1/n^2 (same minimiser as the baseline system).
        H = sparse.csr_matrix((np.ones(n * ncomp), C.ravel(),
                               np.arange(0, n * ncomp + 1, ncomp)), shape=(n, m))
        N = (H.T @ H).toarray() / n
        rhs = H.T @ y / n
        pc = cnt_col / n
        Mt = sparse.csr_matrix((pc, (np.arange(m), comp_of_col)), shape=(m, ncomp))
        N += (Mt @ Mt.T).toarray()
        if HARD_ORTHO:
            self._ortho_basis(n)
        else:
            self._add_soft_ortho(N, n, cnt_main, m)
            for p in self.pairs:
                p["Q"] = np.eye(len(p["codes"]))
        self.m_main = m_main
        self.red_off = m_main + np.concatenate(
            [[0], np.cumsum([p["Q"].shape[1] for p in self.pairs])]).astype(int)

        # Full joint cells, Good-Turing singletons and leave-out functionals of orphans.
        _, cell_inv, cell_cnt = np.unique(lb, axis=0, return_inverse=True,
                                          return_counts=True)
        idx1 = np.flatnonzero(cell_cnt[cell_inv.ravel()] == 1)
        if "lattice" in PRIORS or "harmonic" in RULES:
            self._lattices(n)
        orphans = self._orphan_functionals(X, lb, C, cnt_col, cnt_main, idx1, m)

        # Virtual atoms of this tree (anchor prior) and the leave-out blocks.
        va = None
        if anchor is not None and max(RHOS) > 0:
            va = self._virtual_atoms(anchor, lb, C, cnt_col, y_tree, idx1, n, m)
        self.n_virtual = 0 if va is None else va["nv"]

        loo = np.empty((NV, NR, G, n), dtype=np.float32)
        ins = np.empty((len(PRIORS), NR, G, n), dtype=np.float32)
        self.betas = np.zeros((len(PRIORS), NR, m, G))
        for pi, prior in enumerate(PRIORS):
            # beta = R gamma with R = Q P^{-1/2} block diagonal: whitened normal matrix
            # R' (N + rho A) R, eigendecomposed once per rho for the whole kappa path.
            R = self._reduction(prior)
            Nw = self._left(self._left(N, R).T, R)
            anchored = va is not None and va["nv"] > 0 and prior in ANCHOR_PRIORS
            if anchored:
                Aw = self._left(self._left(va["A"], R).T, R)
            loo_p = np.empty((len(RULES), NR, G, n), dtype=np.float32)
            for ri, rho in enumerate(RHOS):
                if ri > 0 and not anchored:
                    # No anchor for this tree / prior: identical to rho = 0.
                    self.betas[pi, ri] = self.betas[pi, 0]
                    loo_p[:, ri], ins[pi, ri] = loo_p[:, 0], ins[pi, 0]
                    continue
                M = Nw if rho == 0 else Nw + rho * Aw
                S, U = np.linalg.eigh(0.5 * (M + M.T))
                S = np.clip(S, 0.0, None)
                T = self._right(U, R, m)                     # beta = T gamma_white
                if rho == 0:
                    betas, lp, ins[pi, ri] = self._path(T, S, rhs, H, y, idx1, orphans, n)
                else:
                    betas, lp, ins[pi, ri] = self._path_anchor(
                        T, S, rhs + rho * va["b"], H, y, idx1, orphans, n, va,
                        rho * va["w"])
                self.betas[pi, ri] = betas
                loo_p[:, ri] = lp
            for vi, (pr, ru) in enumerate(variants()):
                if pr == pi:
                    loo[vi] = loo_p[ru]
        risk = np.mean(loo.astype(float) ** 2, axis=3)       # (variants, rho, G)
        return loo, ins, risk, len(idx1) / n, m, self.n_virtual

    def _path(self, T, S, rhs, H, y, idx1, orphans, n) -> tuple:
        """S4 path (rho = 0): coefficients, leave-out residuals (rule, kappa, n) and
        in-sample residuals (kappa, n) along the kappa grid."""
        q = T.T @ rhs
        D = 1.0 / (S[None, :] + KAPPAS[:, None] / n)       # (G, m_red)
        betas = T @ (D * q[None, :]).T                        # (m, G)
        resid = y[None, :] - (H @ betas).T                    # (G, n)
        loo = np.repeat(resid[None, :, :], len(RULES), axis=0)
        if len(idx1) == 0:
            return betas, loo, resid
        Z = np.asarray(H[idx1] @ T)                           # (N1, m_red)
        h = (D @ (Z ** 2).T) / n                              # (G, N1) leverages
        one_m_h = np.maximum(1.0 - h, 1e-12)
        loo[:, :, idx1] = (resid[:, idx1] / one_m_h)[None, :, :]
        rows, pts, funcs = orphans
        if len(rows) == 0:
            return betas, loo, resid
        # beta_{-i} = beta - T diag(d) Z_i' r_i / ((1 - h_i) n); an orphan's leave-out
        # prediction is a linear functional a' beta_{-i} (a depends on the rule).
        scale = (resid[:, pts] / one_m_h[:, rows]).T / n      # (n_orph, G)
        Zr = Z[rows]
        for ru, A in enumerate(funcs):
            AT = np.asarray(A @ T)                            # (n_orph, m_red)
            corr = (AT * Zr) @ D.T                            # (n_orph, G)
            loo[ru][:, pts] = (y[pts][:, None] - A @ betas + corr * scale).T
        return betas, loo, resid

    def _path_anchor(self, T, S, rhs, H, y, idx1, orphans, n, va, wv) -> tuple:
        """Anchored path (rho > 0) with block leave-outs.

        In the eigen coordinates gamma (beta = T gamma) the system matrix is
        Lambda = diag(S + kappa/n) = sum_rows w_r z_r z_r' + prior. Removing a block B of
        rows (weights W, unweighted residuals e_B at the full fit) gives the leave-out
        residuals e_- = (I - K W)^{-1} e_B with K = Z_B Lambda^{-1} Z_B', and
        gamma_- = gamma - Lambda^{-1} Z_B' W e_-. The singleton's leave-out residual is
        the entry of its real row; an orphan's is y_i - a' beta_-, a the rule functional.
        """
        G = len(KAPPAS)
        q = T.T @ rhs
        D = 1.0 / (S[None, :] + KAPPAS[:, None] / n)       # (G, m_red)
        betas = T @ (D * q[None, :]).T                        # (m, G)
        resid = y[None, :] - (H @ betas).T                    # (G, n)
        loo = np.repeat(resid[None, :, :], len(RULES), axis=0)
        if len(idx1) == 0:
            return betas, loo, resid
        mr = T.shape[1]
        Zr = np.asarray(H[idx1] @ T)                          # (N1, m_red)
        er = resid[:, idx1]                                   # (G, N1)
        Hn = va["Hv"][va["vneed"]]
        # Last row / column: the zero padding row (index -1 in the blocks).
        Zv = np.vstack([np.asarray(Hn @ T), np.zeros((1, mr))])            # (n_need+1, mr)
        ev = np.hstack([(va["yv"][va["vneed"]][:, None] - Hn @ betas).T,
                        np.zeros((G, 1))])                                  # (G, n_need+1)
        rows, pts, funcs = orphans
        orph_of = np.full(len(idx1), -1, dtype=int)
        orph_of[rows] = np.arange(len(rows))
        ATs = [np.asarray(A @ T) for A in funcs]              # (n_orph, m_red) per rule
        corr = [np.zeros((G, len(rows))) for _ in funcs]
        loo_real = np.empty((G, len(idx1)))
        for mem, vpos, vmult in va["blocks"]:
            b = 1 + vpos.shape[1]
            iu, ju = np.triu_indices(b)
            step = max(1, int(BLOCK_CHUNK // (max(len(iu), b) * mr + 1)))
            for c0 in range(0, len(mem), step):
                mm, vp = mem[c0:c0 + step], vpos[c0:c0 + step]
                # removed weights: real row 1/n, merged virtual rows wv * (dependent copies)
                W = np.hstack([np.full((len(mm), 1), 1.0 / n),
                               wv * vmult[c0:c0 + step]])                    # (Nb, b)
                Zb = np.concatenate([Zr[mm][:, None, :], Zv[vp]], axis=1)   # (Nb, b, mr)
                Eb = np.concatenate([er[:, mm][:, :, None], ev[:, vp]], axis=2)  # (G,Nb,b)
                if b == 1:   # rank-one PRESS
                    h = (D @ (Zb[:, 0, :] ** 2).T) / n                          # (G, Nb)
                    Em = Eb / np.maximum(1.0 - h, 1e-12)[:, :, None]
                else:
                    # K = Z_B Lambda^{-1} Z_B' for all kappas: one GEMM per block row r over
                    # the pairs (r, s >= r) (small blocks, broadcast products without
                    # gathered copies) or one batched product per kappa (large blocks).
                    if len(iu) <= G * b:
                        K = np.empty((G, len(mm), b, b))
                        for r in range(b):
                            P = (Zb[:, r:r + 1, :] * Zb[:, r:, :]).reshape(-1, mr) @ D.T
                            P = P.reshape(len(mm), b - r, G).transpose(2, 0, 1)
                            K[:, :, r, r:] = P
                            K[:, :, r:, r] = P
                    else:
                        Zt = Zb.transpose(0, 2, 1)
                        K = np.stack([np.matmul(Zb * D[g], Zt) for g in range(G)])
                    Em = np.linalg.solve(np.eye(b) - K * W[None, :, None, :],
                                         Eb[..., None])[..., 0]
                loo_real[:, mm] = Em[:, :, 0]
                o = orph_of[mm]
                sel = o >= 0
                if sel.any():
                    WE = Em[:, sel, :] * W[sel][None, :, :]                   # (G, No, b)
                    for ru, AT in enumerate(ATs):
                        F = (AT[o[sel]][:, None, :] * Zb[sel]).reshape(-1, mr) @ D.T
                        F = F.reshape(int(sel.sum()), b, G).transpose(2, 0, 1)
                        corr[ru][:, o[sel]] = np.sum(F * WE, axis=2)
        loo[:, :, idx1] = loo_real[None, :, :]
        for ru, A in enumerate(funcs):
            loo[ru][:, pts] = (y[pts][:, None] - A @ betas).T + corr[ru]
        return betas, loo, resid

    def _virtual_atoms(self, anchor, lb, C, cnt_col, y_tree, idx1, n, m) -> dict:
        """Kept virtual atoms of this tree, anchor matrices and leave-out blocks."""
        X32, shifts, n_shift = anchor["X32"], anchor["shifts"], anchor["n_shift"]
        ev = _TreeEval(self.table)
        off = y_tree - ev.output(X32)
        offset = float(np.median(off))        # base_score share of the tree
        self.route_err = float(np.max(np.abs(off - offset)))
        Cs, ys, org, uown = [], [], [], []
        for vi, j in enumerate(self.main_variables):
            for vals, own in shifts.get(int(j), []):
                valid = np.flatnonzero(~np.isnan(vals))
                nbin = np.digitize(vals[valid], self.split_list[vi], right=False)
                moved = nbin != lb[valid, vi]
                rows = valid[moved]
                if rows.size == 0:
                    continue
                LBc = lb[rows].copy()
                LBc[:, vi] = nbin[moved]
                ok = np.ones(rows.size, dtype=bool)
                cols = [LBc + self.off_main[:-1]]
                for p in self.pairs:     # "drop": every pair cell must have empirical mass
                    code = LBc[:, p["ia"]] * self.nb[p["ib"]] + LBc[:, p["ib"]]
                    pos = np.minimum(np.searchsorted(p["codes"], code), len(p["codes"]) - 1)
                    ok &= p["codes"][pos] == code
                    cols.append((p["off"] + pos)[:, None])
                if not ok.any():
                    continue
                rows = rows[ok]
                Cs.append(np.hstack(cols)[ok])
                org.append(rows)
                uown.append(own[rows])
                ys.append(ev.output(X32, rows=rows, col=int(j), vals=_f32(vals[rows]))
                          + offset)
        nv = int(sum(len(r) for r in org))
        if nv == 0:
            return {"nv": 0}
        Cv = np.vstack(Cs)
        ncomp = Cv.shape[1]
        yv = np.concatenate(ys) - self.eta0
        # Copies with the same cells (hence the same joint cell and the same tree output)
        # are merged into one row whose weight is w times their multiplicity.
        Cu, uinv, ucnt = np.unique(Cv, axis=0, return_inverse=True, return_counts=True)
        uinv = uinv.ravel()
        nu = len(Cu)
        yu = np.bincount(uinv, weights=yv, minlength=nu) / ucnt
        self.merge_err = float(np.max(np.abs(yv - yu[uinv])))
        Hv = sparse.csr_matrix((np.ones(nu * ncomp), Cu.ravel(),
                                np.arange(0, nu * ncomp + 1, ncomp)), shape=(nu, m))
        w = 1.0 / (n_shift * n)
        A = (Hv.T @ sparse.diags(w * ucnt) @ Hv).toarray()
        bvec = Hv.T @ (w * ucnt * yu)

        # Points whose removal deletes each copy: its origin, the sole member of the union
        # bin it was shifted into, the sole owner of every cell it touches that has a
        # single training point (the cell empties and "drop" would reject the copy).
        col_owner = np.full(m, -1, dtype=int)
        one = cnt_col[C] == 1
        col_owner[C[one]] = np.nonzero(one)[0]
        owners = np.concatenate([np.concatenate(org), np.concatenate(uown),
                                 col_owner[Cv].T.ravel()])
        vid = np.concatenate([np.arange(nv), np.arange(nv), np.tile(np.arange(nv), ncomp)])
        a_of = np.full(n, -1, dtype=int)
        a_of[idx1] = np.arange(len(idx1))
        keep = owners >= 0
        a = a_of[owners[keep]]
        vid = vid[keep]
        keep = a >= 0
        # (singleton, copy) pairs, then the number of dependent copies per merged row.
        key = np.unique(a[keep].astype(np.int64) * nv + vid[keep])
        a, vid = key // nv, key % nv
        key, mult = np.unique(a * nu + uinv[vid], return_counts=True)
        a, uid = (key // nu).astype(int), (key % nu).astype(int)
        uneed = np.unique(uid)
        upos = np.concatenate([np.searchsorted(uneed, uid), [-1]])  # -1: zero padding row
        mult = np.concatenate([mult, [0]])
        cnt = np.bincount(a, minlength=len(idx1))
        start = np.concatenate([[0], np.cumsum(cnt)])
        # Blocks are grouped by size (exact up to 8 rows incl. the real row, then rounded up
        # to a multiple of 8) and padded with zero-weight rows, which leaves the downdate
        # exact.
        size = cnt + 1
        bucket = np.where(size <= 8, size, 8 * np.ceil(size / 8)).astype(int)
        blocks = []
        for s in np.unique(bucket):
            mem = np.flatnonzero(bucket == s)
            j = np.arange(s - 1)[None, :]
            take = np.where(j < cnt[mem][:, None], start[mem][:, None] + j, len(upos) - 1)
            blocks.append((mem, upos[take], mult[take]))
        return {"nv": nv, "nu": nu, "Hv": Hv, "yv": yu, "w": w, "A": A, "b": bvec,
                "vneed": uneed, "blocks": blocks,
                "max_block": int(size.max()) if len(size) else 0}

    def _ortho_basis(self, n: int) -> None:
        """Orthogonality null space Q_k of each pair (orthonormal columns, possibly none)."""
        for p in self.pairs:
            nc = len(p["codes"])
            A = np.zeros((self.nb[p["ia"]] + self.nb[p["ib"]], nc))
            A[p["u"], np.arange(nc)] = p["cnt"] / n
            A[self.nb[p["ia"]] + p["v"], np.arange(nc)] = p["cnt"] / n
            p["Q"] = null_space(A)                 # (nc, k)

    def _reduction(self, prior: str) -> list:
        """Pair blocks R_k = Q_k P_k^{-1/2} of beta = R gamma (main bins: EPS_MAIN^{-1/2})."""
        R = []
        for p in self.pairs:
            Qk = p["Q"]
            if prior == "ridge" or Qk.shape[1] == 0:
                R.append(Qk)
            else:
                R.append(Qk @ _inv_sqrt(Qk.T @ p["S"] @ Qk))
        return R

    def _left(self, M: np.ndarray, R: list) -> np.ndarray:
        """R' M for the block-diagonal R (rows of M indexed by the full columns)."""
        out = np.empty((self.red_off[-1], M.shape[1]))
        out[:self.m_main] = M[:self.m_main] / np.sqrt(EPS_MAIN)
        for p, Rk, r0 in zip(self.pairs, R, self.red_off[:-1], strict=True):
            if Rk.shape[1]:
                out[r0:r0 + Rk.shape[1]] = Rk.T @ M[p["off"]:p["off"] + Rk.shape[0]]
        return out

    def _right(self, U: np.ndarray, R: list, m: int) -> np.ndarray:
        """R U for the block-diagonal R (rows of U indexed by the reduced columns)."""
        out = np.zeros((m, U.shape[1]))
        out[:self.m_main] = U[:self.m_main] / np.sqrt(EPS_MAIN)
        for p, Rk, r0 in zip(self.pairs, R, self.red_off[:-1], strict=True):
            if Rk.shape[1]:
                out[p["off"]:p["off"] + Rk.shape[0]] = Rk @ U[r0:r0 + Rk.shape[1]]
        return out

    def _add_soft_ortho(self, N: np.ndarray, n: int, cnt_main: np.ndarray, m: int) -> None:
        """Baseline soft orthogonality rows (weights p_cell / sqrt(p_bin)), added to N."""
        if not self.pairs:
            return
        rows, ocols, vals = [], [], []
        r0 = 0
        p_main = cnt_main / n
        for p in self.pairs:
            ccol = p["off"] + np.arange(len(p["codes"]))
            pcell = p["cnt"] / n
            for side, var in (("u", p["ia"]), ("v", p["ib"])):
                rows.append(r0 + p[side])
                ocols.append(ccol)
                vals.append(pcell / np.sqrt(p_main[self.off_main[var] + p[side]]))
                r0 += self.nb[var]
        O = sparse.csr_matrix((np.concatenate(vals),
                               (np.concatenate(rows), np.concatenate(ocols))),
                              shape=(r0, m))
        N += (O.T @ O).toarray()

    def _lattices(self, n: int) -> None:
        """Lattice GMRF of each pair, with the never-observed cells eliminated (see S4)."""
        raw = []
        for p in self.pairs:
            pa, pb = self.pos[p["ia"]], self.pos[p["ib"]]
            na, nbb = len(pa), len(pb)
            idx = np.arange(na * nbb).reshape(na, nbb)
            wb = 1.0 / np.maximum(np.diff(pb), 0.5 / n)
            wa = 1.0 / np.maximum(np.diff(pa), 0.5 / n)
            src = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
            dst = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
            w = np.concatenate([np.tile(wb, na), np.repeat(wa, nbb)])
            raw.append((na * nbb, src, dst, w))
        w_all = np.concatenate([r[3] for r in raw] + [np.ones(0)])
        scale = np.mean(w_all) if len(w_all) else 1.0
        for p, (ncell, src, dst, w) in zip(self.pairs, raw, strict=True):
            w = w / scale
            K = LATTICE_RIDGE * np.eye(ncell)
            np.add.at(K, (src, src), w)
            np.add.at(K, (dst, dst), w)
            np.add.at(K, (src, dst), -w)
            np.add.at(K, (dst, src), -w)
            obs = p["codes"]
            unseen = np.setdiff1d(np.arange(ncell), obs)
            S = K[np.ix_(obs, obs)]
            E = np.zeros((len(unseen), len(obs)))
            if len(unseen):
                E = -np.linalg.solve(K[np.ix_(unseen, unseen)], K[np.ix_(unseen, obs)])
                S = S + K[np.ix_(obs, unseen)] @ E
            p["S"], p["E"], p["unseen"] = 0.5 * (S + S.T), E, unseen

    def _unseen_weights(self, p: dict, lbu: int, lbv: int, own: int,
                        own_gone: bool) -> list:
        """Per-rule weights over the pair's observed cells for an unseen lookup (see S4)."""
        nc = len(p["codes"])
        out = []
        for rule in RULES:
            w = np.zeros(nc)
            if rule == "nearest":
                if not (own_gone and nc == 1):
                    excl = np.array([own]) if own_gone else None
                    nn = _nearest_cells(self.pos[p["ia"]], self.pos[p["ib"]], p["u"],
                                        p["v"], p["cnt"], np.array([lbu]), np.array([lbv]),
                                        exclude=excl)[0]
                    w[nn] = 1.0
            elif rule == "harmonic":
                S = p["S"]
                h_own = np.zeros(nc)
                if own_gone:
                    h_own = -S[own] / S[own, own]
                    h_own[own] = 0.0
                code = lbu * self.nb[p["ib"]] + lbv
                k = np.searchsorted(p["unseen"], code)
                if k < len(p["unseen"]) and p["unseen"][k] == code:
                    w = p["E"][k].copy()
                    if own_gone:
                        w += w[own] * h_own
                        w[own] = 0.0
                else:  # the target is the emptied own cell
                    w = h_own
            out.append(w)
        return out

    def _orphan_functionals(self, X, lb, C, cnt_col, cnt_main, idx1, m) -> tuple:
        """Leave-out prediction functionals of singletons owning a cell that empties.

        Returns (rows in idx1, point indices, [sparse (n_orph, m) matrix per rule]).
        """
        V = len(self.main_variables)
        rows = (np.flatnonzero(np.any(cnt_col[C[idx1]] == 1, axis=1)) if len(idx1)
                else np.empty(0, dtype=int))
        pts = idx1[rows]
        no = len(pts)
        if no == 0:
            return rows, pts, []
        LB = lb[pts].copy()
        for j in range(V):
            u = LB[:, j]
            last = self.nb[j] - 1
            gone = (cnt_main[self.off_main[j] + u] == 1) & (last > 0)
            if not gone.any():
                continue
            new = u.copy()
            new[gone & (u == 0)] = 1
            new[gone & (u == last)] = last - 1
            inner = gone & (u > 0) & (u < last)
            if inner.any():
                s, ui = self.split_list[j], u[inner]
                mid = 0.5 * (s[ui - 1] + s[ui])
                new[inner] = np.where(X[pts[inner], self.main_variables[j]] < mid,
                                      ui - 1, ui + 1)
            LB[:, j] = new
        o_all = np.arange(no)
        r_seen, c_seen = [np.repeat(o_all, V)], [(LB + self.off_main[:-1]).ravel()]
        ent = [([], [], []) for _ in RULES]
        for k, p in enumerate(self.pairs):
            nc = len(p["codes"])
            code = LB[:, p["ia"]] * self.nb[p["ib"]] + LB[:, p["ib"]]
            own = C[pts, V + k] - p["off"]
            own_gone = p["cnt"][own] == 1
            pos = np.minimum(np.searchsorted(p["codes"], code), nc - 1)
            seen = (p["codes"][pos] == code) & ~((pos == own) & own_gone)
            r_seen.append(o_all[seen])
            c_seen.append(p["off"] + pos[seen])
            for o in np.flatnonzero(~seen):
                ws = self._unseen_weights(p, LB[o, p["ia"]], LB[o, p["ib"]], own[o],
                                          bool(own_gone[o]))
                for ru, w in enumerate(ws):
                    nz = np.flatnonzero(w)
                    ent[ru][0].append(np.full(len(nz), o))
                    ent[ru][1].append(p["off"] + nz)
                    ent[ru][2].append(w[nz])
        r_seen, c_seen = np.concatenate(r_seen), np.concatenate(c_seen)
        funcs = [sparse.csr_matrix((np.concatenate([np.ones(len(r_seen)), *v]),
                                    (np.concatenate([r_seen, *r]), np.concatenate([c_seen, *c]))),
                                   shape=(no, m))
                 for r, c, v in ent]
        return rows, pts, funcs

    # ------------------------------------------------------------ finalise
    def finalize(self, kappa_idx: int, variant: int, rho_idx: int = 0) -> None:
        """Fix the coefficients and build the lattice of every interaction."""
        self.kappa_idx = int(kappa_idx)
        self.variant = int(variant)
        self.rho_idx = int(rho_idx)
        if self.main_variables.size == 0:
            self.beta = np.empty(0)
            return
        prior, rule = variants()[self.variant]
        self.beta = self.betas[prior, self.rho_idx][:, self.kappa_idx].copy()
        for p in self.pairs:
            na, nbb = self.nb[p["ia"]], self.nb[p["ib"]]
            lat = np.zeros(na * nbb)
            b_obs = self.beta[p["off"] + np.arange(len(p["codes"]))]
            lat[p["codes"]] = b_obs
            unseen = np.setdiff1d(np.arange(na * nbb), p["codes"])
            if len(unseen) and RULES[rule] == "nearest":
                nn = _nearest_cells(self.pos[p["ia"]], self.pos[p["ib"]], p["u"],
                                    p["v"], p["cnt"], unseen // nbb, unseen % nbb)
                lat[unseen] = b_obs[nn]
            elif len(unseen) and RULES[rule] == "harmonic":
                lat[unseen] = p["E"] @ b_obs
            p["lattice"] = lat
        if not KEEP_PATH:
            self.betas = np.empty((0, 0, 0, 0))
            self.table = None
            for p in self.pairs:
                for key in ("Q", "S", "E", "unseen"):
                    p.pop(key, None)

    # ------------------------------------------------------------- predict
    def predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X_new.shape[0]
        if self.main_variables.size == 0:
            return np.zeros((n, 0)), np.zeros((n, 0))
        lb = np.column_stack([np.digitize(X_new[:, j], bins=self.split_list[k],
                                          right=False)
                              for k, j in enumerate(self.main_variables)])
        y_main = self.beta[lb + self.off_main[:-1]]
        y_order2 = np.zeros((n, len(self.pairs)))
        for k, p in enumerate(self.pairs):
            y_order2[:, k] = p["lattice"][lb[:, p["ia"]] * self.nb[p["ib"]]
                                          + lb[:, p["ib"]]]
        return y_main, y_order2


class GTLocoHFD(XGBTreeHFD):
    """Anchored GT-LOCO TreeHFD decomposition of an xgboost regression model."""

    def fit(self, X: np.ndarray, interaction_order: int = 2,
            verbose: bool = False) -> None:
        if self.num_outputs != 1:
            error_msg = "Anchored GT-LOCO TreeHFD is implemented for single-output models."
            raise NotImplementedError(error_msg)
        if RHOS[0] != 0:
            error_msg = "RHOS must start at 0 (S4 is the rho = 0 member)."
            raise ValueError(error_msg)
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        self.interaction_order = interaction_order
        self.depth_variable = self.max_depth
        tree_predictions = self._tree_predict(X)
        # Virtual atoms: shifted copies on the ensemble union-of-splits grid (X_train only).
        X32 = _f32(X)
        union = _union_splits(self.xgb_table)
        anchor = {"X32": X32, "shifts": _shift_values(X, X32, union),
                  "n_shift": 2 * len(union)}
        G, NV, NR = len(KAPPAS), len(variants()), len(RHOS)
        RM, NS = NR + 1, len(SHIFTS)   # rho modes: common RHOS[r] (r < NR) or per-tree
        # Ensemble accumulators (summed over trees, per training point) of the leave-out
        # residuals and of the in-sample residuals, for every candidate choice.
        acc_shift = np.zeros((NV, RM, NS, n))
        acc_common = np.zeros((NV, RM, G, n))
        ins_shift = np.zeros((NV, RM, NS, n))
        ins_common = np.zeros((NV, RM, G, n))
        per_tree_idx = np.zeros((self.n_estimators, NV, NR), dtype=int)
        per_tree_joint = np.zeros((self.n_estimators, NV, 2), dtype=int)
        self.treehfd_list = []
        eta0 = 0.0
        diag = {"n1_frac": [], "m": [], "risk": [], "n_virtual": [], "route_err": []}
        for t in range(self.n_estimators):
            table = pd.DataFrame(self.xgb_table[self.xgb_table["Tree"] == t])
            tree = GTLocoTree(table, interaction_order, self.depth_variable)
            loo, ins, risk, n1f, m, nv = tree.fit(X, tree_predictions[:, t], anchor)
            # Per-tree choices: smallest risk, ties towards the smaller rho, then kappa.
            idx = _first_min(risk, axis=2)                                   # (NV, NR)
            joint = _first_min(risk.reshape(NV, NR * G), axis=1)
            per_tree_idx[t] = idx
            per_tree_joint[t] = np.column_stack(np.divmod(joint, G))
            for v, (pr, _) in enumerate(variants()):
                for rm in range(RM):
                    if rm < NR:
                        r, g0 = rm, idx[v, rm]
                    else:
                        r, g0 = per_tree_joint[t, v]
                    g_shift = np.clip(g0 + SHIFTS, 0, G - 1)
                    acc_shift[v, rm] += loo[v, r, g_shift]
                    ins_shift[v, rm] += ins[pr, r, g_shift]
                    acc_common[v, rm] += loo[v, r]
                    ins_common[v, rm] += ins[pr, r]
            eta0 += tree.eta0
            self.treehfd_list.append(tree)
            diag["n1_frac"].append(n1f)
            diag["m"].append(m)
            diag["risk"].append(risk)
            diag["n_virtual"].append(nv)
            diag["route_err"].append(tree.route_err)
        self.eta0 = float(eta0)

        # Ensemble-level candidates. Candidate order encodes the tie-break preference:
        # variant order, then rho mode (smaller common rho first, per-tree rho last), then
        # the unshifted per-tree kappa first, smaller shifts, common kappas.
        cands, loos, inss = [], [], []
        order = np.argsort(np.abs(SHIFTS), kind="stable")
        for v in range(NV):
            for rm in range(RM):
                for si in order:
                    cands.append(("shift", v, rm, int(SHIFTS[si])))
                    loos.append(acc_shift[v, rm, si])
                    inss.append(ins_shift[v, rm, si])
                for g in range(G):
                    cands.append(("common", v, rm, g))
                    loos.append(acc_common[v, rm, g])
                    inss.append(ins_common[v, rm, g])
        sq = np.array(loos) ** 2                             # (candidates, n)
        risks = np.mean(sq, axis=1)
        fidelity = np.mean(np.array(inss) ** 2, axis=1)       # in-sample MSE vs T(x)
        i_min = int(_first_min(risks))
        i_sel = i_min
        if SE_RULE > 0:
            se = np.std(sq - sq[i_min], axis=1) / np.sqrt(n) * SE_RULE
            tied = np.flatnonzero(risks <= risks[i_min] * (1 + 1e-9) + se)
            i_sel = int(tied[_first_min(fidelity[tied])])
        del sq, loos, inss
        cands = [(*c, float(r), float(f)) for c, r, f in zip(cands, risks, fidelity,
                                                            strict=True)]
        best = cands[i_sel]
        mode, variant, rm, par = best[:4]
        chosen_g, chosen_r = [], []
        for t, tree in enumerate(self.treehfd_list):
            if rm < NR:
                r, g0 = rm, per_tree_idx[t, variant, rm]
            else:
                r, g0 = per_tree_joint[t, variant]
            g = int(np.clip(g0 + par, 0, G - 1)) if mode == "shift" else par
            tree.finalize(g, variant, r)
            chosen_g.append(g)
            chosen_r.append(r)
        # Label-free trade-off report (X_train and tree outputs only): estimated
        # out-of-sample risk and in-sample residual, relative to Var[T(X_train)].
        var_t = float(np.var(tree_predictions.sum(axis=1)))
        rho_names = [f"rho={r:g}" for r in RHOS] + ["rho=per-tree"]
        diag.update({"selection": best, "selection_rho_mode": rho_names[rm],
                     "candidates": cands, "min_risk": cands[i_min],
                     "risk_over_var": best[4] / var_t, "resid_in_over_var": best[5] / var_t,
                     "kappa_chosen": KAPPAS[np.array(chosen_g)],
                     "rho_chosen": np.array(RHOS)[np.array(chosen_r)],
                     "per_tree_idx": per_tree_idx, "per_tree_joint": per_tree_joint,
                     "var_t": var_t})
        self.diagnostics = diag

        # Global interaction list and column maps.
        lists = [np.array(t.interaction_list, dtype=int).reshape(-1, 2)
                 for t in self.treehfd_list]
        allp = np.concatenate(lists, axis=0) if lists else np.empty((0, 2), int)
        self.interaction_list = (np.unique(allp, axis=0) if len(allp)
                                 else np.empty((0, 2), dtype=int))
        index = {tuple(p): k for k, p in enumerate(self.interaction_list.tolist())}
        self._inter_index = [np.array([index[tuple(p)] for p in t.interaction_list],
                                      dtype=int) for t in self.treehfd_list]

    def predict(self, X_new: np.ndarray, verbose: bool = False) -> tuple:
        X_new = np.asarray(X_new, dtype=float)
        n = X_new.shape[0]
        y_main = np.zeros((n, X_new.shape[1]))
        y_order2 = np.zeros((n, self.interaction_list.shape[0]))
        for tree, idx in zip(self.treehfd_list, self._inter_index, strict=True):
            if tree.main_variables.size == 0:
                continue
            m_t, o_t = tree.predict(X_new)
            y_main[:, tree.main_variables] += m_t
            if len(idx):
                y_order2[:, idx] += o_t
        return y_main, y_order2
