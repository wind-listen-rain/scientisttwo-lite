"""GT-LOCO TreeHFD: Good-Turing-calibrated leave-one-cell-out shrinkage of TreeHFD.

For each tree, the TreeHFD least-squares problem (zero-mean rows, residual rows weighted by
sqrt(n * n_c), hierarchical orthogonality) is augmented with a Tikhonov penalty
lambda * beta' P beta, so that cells supported by few points are shrunk.  Two priors P:
  * "ridge":   P = I on interaction cells and EPS_MAIN * I on main bins (every coefficient
               shrunk towards 0; since the data weight of a cell is proportional to its
               count, thin interaction cells are shrunk far more than populated ones, i.e.
               towards the additive explanation);
  * "lattice": on each pair (j, k), a Gaussian Markov random field on the *whole* (j, k)
               bin lattice (observed and never-observed cells): first differences between
               4-neighbour cells weighted by 1 / (distance between bin centres in quantile
               coordinates), plus a small ridge.  Unobserved cells are latent columns, which
               are eliminated in closed form (Schur complement); main bins keep the ridge.
Orthogonality is imposed exactly (coefficients live in the null space of each pair's
orthogonality rows), so shrinkage only selects among hierarchically orthogonal solutions.
The problem is solved exactly for a grid of lambdas from one eigendecomposition per prior.

The strength is selected without labels, from X_train and the tree outputs only: each
training point that is alone in its full joint cell (a Good-Turing "singleton") is left out
in closed form (PRESS, r / (1 - h)), which simulates a new point landing in an unseen joint
cell; points in repeated cells keep their in-sample residual (exact for them since the tree
is constant on a joint cell). The mean of these squared leave-out residuals is
R = (1 - N1/n) * sum_{n_c >= 2} w_c r_c^2 + (N1/n) * mean_{n_c = 1} [r_c / (1 - h_cc)]^2
with w_c = n_c / (n - N1).
Pair cells never observed in training take a deterministic value given by a rule:
"zero" (the prior mean: additive explanation) or "harmonic" (the lattice-prior posterior
mean of the latent cell given the observed cells, i.e. its harmonic extension); a third rule,
"nearest" (value of the closest observed cell in quantile coordinates), is implemented but
not a candidate by default. The leave-out simulates the rule for singleton points whose pair
cell (or main bin) empties when they are removed.
Per-tree choices are finally shifted or replaced by one ensemble-level choice (prior, rule,
strength), selected with the leave-out residuals summed over trees for each training point,
since errors of different trees are correlated.
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

# Strengths in pseudo-count units: lambda = kappa / n, i.e. a unit ridge weight equals the
# residual weight of kappa training points. The grid starts at 0.01 (near-unregularised):
# below it the closed-form leave-out of points owning a count-1 cell becomes optimistic
# (checked against brute-force refits).
KAPPAS = np.array([1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 1e4])
EPS_MAIN = 1.0             # ridge weight of main-effect bins relative to interaction cells
PRIORS = ("ridge", "lattice")
LATTICE_RIDGE = 0.1        # ridge part of the lattice prior (edge weights have mean 1)
# Unseen-cell rules in the candidate set ("nearest" is implemented but was never selected
# by the leave-out risk in development, so it is not a candidate).
RULES = ("zero", "harmonic")
SHIFTS = np.arange(-5, 4)  # global shifts of the per-tree kappa index (ensemble step)
COMMON_KAPPA = True        # ensemble step also offers one common kappa for all trees
                           # (False is used only by the ensemble-step ablation)
# Ensemble step: candidates within SE_RULE paired standard errors of the minimum risk are
# treated as tied, and the one with the best in-sample fidelity is taken. 0 = plain argmin
# (default). 1 trades about 15% of resid_in for about 5% of resid_out in development, so
# it is off.
SE_RULE = 0.0
KEEP_PATH = False          # development only: keep the whole kappa path after fitting
HARD_ORTHO = True          # exact per-tree orthogonality (False: baseline soft rows)


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
    """Nearest observed pair cell (quantile coordinates) for each query cell.

    Ties are broken by the larger cell count, then the smaller cell index, so the rule is
    deterministic. `exclude` optionally masks one observed cell per query.
    """
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


class GTLocoTree:
    """GT-LOCO decomposition of a single tree."""

    def __init__(self, tree_table: pd.DataFrame, interaction_order: int,
                 depth_variable: int) -> None:
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
        self.betas = np.empty((0, 0, 0))
        self.beta = np.empty(0)
        self.pairs: list[dict] = []
        self.split_list: list[np.ndarray] = []
        self.kappa_idx = 0
        self.variant = 0

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y_tree: np.ndarray) -> tuple:
        """Fit every (prior, kappa); return leave-out residuals (variant, kappa, n), risks."""
        n = X.shape[0]
        G = len(KAPPAS)
        NV = len(variants())
        self.eta0 = float(np.mean(y_tree))
        y = y_tree - self.eta0
        if self.main_variables.size == 0:
            loo = np.broadcast_to(y, (NV, G, n)).astype(np.float32)
            ins = np.broadcast_to(y, (len(PRIORS), G, n)).astype(np.float32)
            return loo, ins, np.mean(loo.astype(float) ** 2, axis=2), 0.0, 0

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
            # beta = Q gamma, Q = diag(I_main, Q_1, ..., Q_I), Q_k an orthonormal basis of the
            # pair cells satisfying the orthogonality rows exactly (dim 0 prunes the pair).
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

        loo = np.empty((NV, G, n), dtype=np.float32)
        ins = np.empty((len(PRIORS), G, n), dtype=np.float32)
        self.betas = np.zeros((len(PRIORS), m, G))
        for pi, prior in enumerate(PRIORS):
            # beta = R gamma with R = Q P^{-1/2} block diagonal: whitened normal matrix
            # R' N R, eigendecomposed once for the whole kappa path.
            R = self._reduction(prior)
            Nw = self._left(self._left(N, R).T, R)
            S, U = np.linalg.eigh(0.5 * (Nw + Nw.T))
            S = np.clip(S, 0.0, None)
            T = self._right(U, R, m)                     # beta = T gamma_white
            betas, loo_p, ins[pi] = self._path(T, S, rhs, H, y, idx1, orphans, n)
            self.betas[pi] = betas
            for vi, (pr, ru) in enumerate(variants()):
                if pr == pi:
                    loo[vi] = loo_p[ru]
        risk = np.mean(loo.astype(float) ** 2, axis=2)       # (variants, G)
        return loo, ins, risk, len(idx1) / n, m

    def _path(self, T, S, rhs, H, y, idx1, orphans, n) -> tuple:
        """Coefficients, leave-out residuals (rule, kappa, n) and in-sample residuals
        (kappa, n) along the kappa grid."""
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
        """Lattice GMRF of each pair, with the never-observed cells eliminated.

        On the full (j, k) bin lattice, K = L + LATTICE_RIDGE * I where L is the weighted
        first-difference Laplacian of 4-neighbour cells, with weights 1 / (distance between
        bin centres in quantile coordinates), normalised to mean 1 over the tree. With O the
        observed and U the unobserved cells, the prior on the observed cells is the Schur
        complement S = K_OO - K_OU K_UU^{-1} K_UO, and the posterior mean of the latent cells
        is the harmonic extension beta_U = E beta_O, E = -K_UU^{-1} K_UO.
        """
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
        """Per-rule weights over the pair's observed cells for an unseen lookup.

        The target cell (lbu, lbv) is unobserved, or is the point's own cell which empties
        when the point is left out (`own_gone`); the own cell is then latent as well.
        """
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
                # Conditional mean of the own cell given the other observed cells.
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
        # Main bins that vanish: the point moves to the neighbour bin, as the baseline
        # merges an empty bin with its neighbours (split at the bin midpoint).
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
            own = C[pts, V + k] - p["off"]            # the point's own observed cell
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
    def finalize(self, kappa_idx: int, variant: int) -> None:
        """Fix the coefficients and build the lattice of every interaction."""
        self.kappa_idx = int(kappa_idx)
        self.variant = int(variant)
        if self.main_variables.size == 0:
            self.beta = np.empty(0)
            return
        prior, rule = variants()[self.variant]
        self.beta = self.betas[prior][:, self.kappa_idx].copy()
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
            self.betas = np.empty((0, 0, 0))
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
    """GT-LOCO TreeHFD decomposition of an xgboost regression model."""

    def fit(self, X: np.ndarray, interaction_order: int = 2,
            verbose: bool = False) -> None:
        if self.num_outputs != 1:
            error_msg = "GT-LOCO TreeHFD is implemented for single-output models."
            raise NotImplementedError(error_msg)
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        self.interaction_order = interaction_order
        self.depth_variable = self.max_depth
        tree_predictions = self._tree_predict(X)
        G, NV = len(KAPPAS), len(variants())
        # Ensemble accumulators (summed over trees, per training point) of the leave-out
        # residuals and of the in-sample residuals, for every candidate choice.
        acc_shift = np.zeros((NV, len(SHIFTS), n))
        acc_common = np.zeros((NV, G, n))
        ins_shift = np.zeros((NV, len(SHIFTS), n))
        ins_common = np.zeros((len(PRIORS), G, n))
        per_tree_idx = np.zeros((self.n_estimators, NV), dtype=int)
        self.treehfd_list = []
        eta0 = 0.0
        diag = {"n1_frac": [], "m": [], "risk": []}
        for t in range(self.n_estimators):
            table = pd.DataFrame(self.xgb_table[self.xgb_table["Tree"] == t])
            tree = GTLocoTree(table, interaction_order, self.depth_variable)
            loo, ins, risk, n1f, m = tree.fit(X, tree_predictions[:, t])
            # Per-tree choice: smallest risk, ties towards the smaller kappa.
            idx = _first_min(risk, axis=1)
            per_tree_idx[t] = idx
            for v, (pr, _) in enumerate(variants()):
                g_shift = np.clip(idx[v] + SHIFTS, 0, G - 1)
                acc_shift[v] += loo[v, g_shift]
                acc_common[v] += loo[v]
                ins_shift[v] += ins[pr, g_shift]
            ins_common += ins
            eta0 += tree.eta0
            self.treehfd_list.append(tree)
            diag["n1_frac"].append(n1f)
            diag["m"].append(m)
            diag["risk"].append(risk)
        self.eta0 = float(eta0)

        # Ensemble-level candidates. Candidate order encodes the tie-break preference:
        # variant order, unshifted per-tree choice first, then smaller shifts / kappas.
        cands, loos, inss = [], [], []
        order = np.argsort(np.abs(SHIFTS), kind="stable")
        for v, (pr, _) in enumerate(variants()):
            for si in order:
                cands.append(("shift", v, int(SHIFTS[si])))
                loos.append(acc_shift[v, si])
                inss.append(ins_shift[v, si])
            for g in range(G if COMMON_KAPPA else 0):
                cands.append(("common", v, g))
                loos.append(acc_common[v, g])
                inss.append(ins_common[pr, g])
        sq = np.array(loos) ** 2                             # (candidates, n)
        risks = np.mean(sq, axis=1)
        fidelity = np.mean(np.array(inss) ** 2, axis=1)       # in-sample MSE vs T(x)
        i_min = int(_first_min(risks))
        # Optional k-standard-error rule towards in-sample fidelity (SE_RULE = k): among
        # candidates whose estimated risk is within k paired standard errors of the minimum,
        # take the one that best reconstructs T on X_train. With k = 0 this is the argmin of
        # R (ties towards the earlier candidate, as before).
        i_sel = i_min
        if SE_RULE > 0:
            se = np.std(sq - sq[i_min], axis=1) / np.sqrt(n) * SE_RULE
            tied = np.flatnonzero(risks <= risks[i_min] * (1 + 1e-9) + se)
            i_sel = int(tied[_first_min(fidelity[tied])])
        cands = [(*c, float(r), float(f)) for c, r, f in zip(cands, risks, fidelity,
                                                            strict=True)]
        best = cands[i_sel]
        mode, variant, par = best[:3]
        chosen = []
        for t, tree in enumerate(self.treehfd_list):
            g = (int(np.clip(per_tree_idx[t, variant] + par, 0, G - 1)) if mode == "shift"
                 else par)
            tree.finalize(g, variant)
            chosen.append(g)
        # Label-free trade-off report (X_train and tree outputs only): estimated
        # out-of-sample risk and in-sample residual, relative to Var[T(X_train)].
        var_t = float(np.var(tree_predictions.sum(axis=1)))
        diag.update({"selection": best, "candidates": cands, "min_risk": cands[i_min],
                     "risk_over_var": best[3] / var_t, "resid_in_over_var": best[4] / var_t,
                     "kappa_chosen": KAPPAS[np.array(chosen)],
                     "per_tree_kappa": KAPPAS[per_tree_idx],
                     "per_tree_idx": per_tree_idx})
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
