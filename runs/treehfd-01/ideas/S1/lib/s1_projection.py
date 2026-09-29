"""S1: ensemble-level re-orthogonalization of TreeHFD interactions on the union-of-splits partition.

For every retained pair (j, k), the aggregated TreeHFD interaction eta_jk is regressed on an additive, piecewise
constant model g_j(b_j) + g_k(b_k) under the empirical measure of X_train, where b_j indexes the bins of the union of
all split thresholds of the ensemble on x_j (merged to a minimum count). A first-difference penalty along the ordered
bins smooths g; its strength is chosen by K-fold cross-fitting on the rows of X_train (no labels involved). The fitted
part is then moved from the interaction to the main effects:
    eta_jk <- eta_jk - g_j - g_k,  eta_j <- eta_j + g_j - c_j,  eta_k <- eta_k + g_k - c_k,  eta0 <- eta0 + c_j + c_k,
which leaves intercept + sum(main) + sum(inter) unchanged at every x.
"""
import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, solve

M_MIN = 30          # minimum number of X_train rows per merged union bin
N_FOLDS = 3         # cross-fitting folds (deterministic: row index mod N_FOLDS)
RIDGE = 1e-8        # numerical ridge on the normal equations, in units of rows
# Difference-penalty grid. With SCALE = "abs", lambda = tau (in units of rows): bins holding many more than tau rows
# are barely smoothed, thin bins are pooled with their neighbours. With SCALE = "mean", lambda = tau * n / n_bins.
# tau = inf forces g_j, g_k constant, i.e. no transfer (the pair stays as TreeHFD fitted it).
TAU_GRID = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 1e4, np.inf)
SCALE = "abs"
SELECT = "pair"     # "pair": one tau per pair; "global": one tau for all pairs
CRITERION = "mse"   # held-out score, see fit_projection


def union_edges(xgb_table, X, m_min=M_MIN):
    """Union of all split thresholds of the ensemble per variable, greedily merged so each bin holds >= m_min rows."""
    edges = []
    feat = xgb_table["Feature"].to_numpy()
    split = xgb_table["Split"].to_numpy(dtype=float)
    for j in range(X.shape[1]):
        thr = np.unique(split[feat == f"f{j}"])
        thr = thr[np.isfinite(thr)]
        if thr.size == 0:
            edges.append(thr)
            continue
        cnt = np.bincount(np.digitize(X[:, j], thr), minlength=thr.size + 1)
        keep, acc, right = [], 0, int(cnt.sum())
        for i in range(thr.size):  # threshold i separates bin i from bin i + 1
            acc += cnt[i]
            right -= cnt[i]
            if acc >= m_min and right >= m_min:
                keep.append(thr[i])
                acc = 0
        edges.append(np.array(keep, dtype=float))
    return edges


def diff_penalty(n_bins):
    """D^T D for the first-difference operator D on n_bins ordered bins."""
    P = np.zeros((n_bins, n_bins))
    if n_bins > 1:
        i = np.arange(n_bins - 1)
        P[i, i] += 1.0
        P[i + 1, i + 1] += 1.0
        P[i, i + 1] -= 1.0
        P[i + 1, i] -= 1.0
    return P


class PairSystem:
    """Normal equations of min sum_i (y_i - g_j[bj_i] - g_k[bk_i])^2 + lam_j |D g_j|^2 + lam_k |D g_k|^2.

    Gauge: g_k[0] = 0 (the constant is carried by g_j; the penalty only sees differences, so this is exact).
    """

    def __init__(self, bj, bk, nbj, nbk, Pj, Pk, scale_j, scale_k):
        self.bj, self.bk, self.nbj, self.nbk = bj, bk, nbj, nbk
        self.Pj, self.Pk, self.scale_j, self.scale_k = Pj, Pk, scale_j, scale_k

    def stats(self, rows, y):
        bj, bk, nbj, nbk = self.bj[rows], self.bk[rows], self.nbj, self.nbk
        nj = np.bincount(bj, minlength=nbj).astype(float)
        nk = np.bincount(bk, minlength=nbk).astype(float)
        C = np.bincount(bj * nbk + bk, minlength=nbj * nbk).reshape(nbj, nbk).astype(float)
        rj = np.bincount(bj, weights=y[rows], minlength=nbj)
        rk = np.bincount(bk, weights=y[rows], minlength=nbk)
        return nj, nk, C, rj, rk

    def solve(self, st, tau):
        nj, nk, C, rj, rk = st
        nbj, nbk = self.nbj, self.nbk
        if np.isinf(tau):  # infinite difference penalty: constant fit only
            return np.full(nbj, rj.sum() / max(nj.sum(), 1.0)), np.zeros(nbk)
        dim = nbj + nbk
        A = np.zeros((dim, dim))
        A[:nbj, :nbj] = np.diag(nj) + tau * self.scale_j * self.Pj
        A[nbj:, nbj:] = np.diag(nk) + tau * self.scale_k * self.Pk
        A[:nbj, nbj:] = C
        A[nbj:, :nbj] = C.T
        b = np.concatenate([rj, rk])
        keep = np.r_[0:nbj, nbj + 1:dim]  # drop g_k[0] (gauge)
        A = A[np.ix_(keep, keep)]
        A[np.diag_indices_from(A)] += RIDGE
        try:
            sol = cho_solve(cho_factor(A, check_finite=False), b[keep], check_finite=False)
        except LinAlgError:
            sol = solve(A, b[keep], assume_a="sym", check_finite=False)
        return sol[:nbj], np.concatenate([[0.0], sol[nbj:]])


def _leak(inter_col, main_cols):
    """Interaction variance lying along its parents' main effects: sum_m cov(eta_jk, eta_m)^2 / var(eta_m)."""
    y = inter_col - inter_col.mean()
    out = 0.0
    for m in main_cols:
        m = m - m.mean()
        vm = np.mean(m * m)
        if vm > 0:
            out += np.mean(y * m) ** 2 / vm
    return out


def fit_projection(eta0, main_tr, inter_tr, inter_list, X, xgb_table, m_min=M_MIN, tau_grid=TAU_GRID,
                   n_folds=N_FOLDS, scale=SCALE, select=SELECT, criterion=CRITERION):
    """Fit the transfers g on X_train. Returns (state dict, new intercept, diagnostics).

    criterion (held-out, on the rows of the left-out fold):
      "leak": sum over pairs of _leak(eta_jk - g_j - g_k, [eta_j + G_j, eta_k + G_k]), where G_j is the total transfer
              into eta_j (all pairs for select="global", this pair only for select="pair");
      "mse":  squared error of the additive fit, sum (eta_jk - g_j - g_k)^2.
    """
    n, p = X.shape
    edges = union_edges(xgb_table, X, m_min)
    bins = [np.digitize(X[:, j], edges[j]) for j in range(p)]
    nbins = [e.size + 1 for e in edges]
    pens = [diff_penalty(b) for b in nbins]
    scales = [n / b if scale == "mean" else 1.0 for b in nbins]
    il = np.asarray(inter_list, dtype=int).reshape(-1, 2)
    active = [c for c, (j, k) in enumerate(il)
              if np.var(inter_tr[:, c]) > 0 and (nbins[j] > 1 or nbins[k] > 1)]
    systems = {c: PairSystem(bins[il[c, 0]], bins[il[c, 1]], nbins[il[c, 0]], nbins[il[c, 1]],
                             pens[il[c, 0]], pens[il[c, 1]], scales[il[c, 0]], scales[il[c, 1]]) for c in active}
    folds = np.arange(n) % n_folds
    G = len(tau_grid)
    score = np.zeros((G, len(active)))  # per tau and pair
    for f in range(n_folds):
        tr, ho = folds != f, folds == f
        trans = [dict() for _ in range(G)]  # held-out transfers per tau: c -> (v_j, v_k)
        for c in active:
            j, k = il[c]
            st = systems[c].stats(tr, inter_tr[:, c])
            for t, tau in enumerate(tau_grid):
                gj, gk = systems[c].solve(st, tau)
                trans[t][c] = (gj[bins[j][ho]], gk[bins[k][ho]])
        for t in range(G):
            if criterion == "leak" and select == "global":
                main_ho = main_tr[ho].copy()
                for c, (vj, vk) in trans[t].items():
                    main_ho[:, il[c, 0]] += vj
                    main_ho[:, il[c, 1]] += vk
            for a, c in enumerate(active):
                j, k = il[c]
                vj, vk = trans[t][c]
                r = inter_tr[ho, c] - vj - vk
                if criterion == "mse":
                    score[t, a] += np.sum((r - r.mean()) ** 2)
                elif select == "global":
                    score[t, a] += _leak(r, [main_ho[:, j], main_ho[:, k]])
                else:
                    score[t, a] += _leak(r, [main_tr[ho, j] + vj, main_tr[ho, k] + vk])
    # Smallest held-out score; exact ties go to the smoother (larger) tau.
    if select == "global":
        tot = score.sum(1)
        t_best = np.full(len(active), G - 1 - int(np.argmin(tot[::-1])))
    else:
        t_best = G - 1 - np.argmin(score[::-1], axis=0)
    pairs, shift = [], 0.0
    for a, c in enumerate(active):
        j, k = il[c]
        tau = tau_grid[t_best[a]]
        if np.isinf(tau):
            continue
        gj, gk = systems[c].solve(systems[c].stats(np.ones(n, bool), inter_tr[:, c]), tau)
        cj, ck = float(np.mean(gj[bins[j]])), float(np.mean(gk[bins[k]]))
        shift += cj + ck
        pairs.append(dict(col=int(c), j=int(j), k=int(k), gj=gj - cj, gk=gk - ck, cj=cj, ck=ck, tau=float(tau)))
    diag = dict(tau_per_pair=[float(tau_grid[t]) for t in t_best], score=score.sum(1).tolist(),
                n_active=len(active), n_bins=nbins)
    return dict(edges=edges, pairs=pairs), float(eta0) + shift, diag


def apply_projection(proj, main, inter, X):
    """Move g_j + g_k from each interaction column to its main effects (pointwise sum-preserving)."""
    main, inter = main.copy(), inter.copy()
    bins = {}
    for pr in proj["pairs"]:
        j, k, c = pr["j"], pr["k"], pr["col"]
        for v in (j, k):
            if v not in bins:
                bins[v] = np.digitize(X[:, v], proj["edges"][v])
        vj, vk = pr["gj"][bins[j]], pr["gk"][bins[k]]
        inter[:, c] -= vj + vk + pr["cj"] + pr["ck"]
        main[:, j] += vj
        main[:, k] += vk
    return main, inter
