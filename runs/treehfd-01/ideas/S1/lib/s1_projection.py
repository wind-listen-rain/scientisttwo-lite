"""S1: ensemble-level re-orthogonalization of TreeHFD interactions on the union-of-splits partition.

Step 1 (union-bin transfer). For every retained pair (j, k), the aggregated TreeHFD interaction eta_jk is regressed on
an additive, piecewise constant model g_j(b_j) + g_k(b_k) under the empirical measure of X_train, where b_j indexes the
bins of the union of all split thresholds of the ensemble on x_j (merged to a minimum count, about n^(1/4) bins). A
first-difference penalty along the ordered bins smooths g; its strength is chosen per pair by K-fold cross-fitting on the
rows of X_train (no labels involved) with a paired two-standard-error rule: the smoothest candidate -- including "no
transfer" -- whose held-out error is not significantly worse than the best one is kept. The fitted part is moved to the
main effects:
    eta_jk <- eta_jk - g_j - g_k,  eta_j <- eta_j + g_j - c_j,  eta_k <- eta_k + g_k - c_k,  eta0 <- eta0 + c_j + c_k.

Step 2 (closing step on the final main effects). Merged bins (and the smoothing) do not span the final main effects
eta_j, which are piecewise constant on the full, unmerged union grid. So, as the last move, each interaction's
component along its parents' final main effects is moved into them: least squares of eta_jk on (eta_j, eta_k) under the
empirical measure gives (a, b); eta_jk <- eta_jk - a eta_j - b eta_k and eta_j <- s_j eta_j with s_j = 1 + sum of the
coefficients on eta_j over its pairs. Every transfer into eta_j is along eta_j itself, so all pairs are exactly
uncorrelated with the final main effects in-sample at the same time, and the shape (hence roughness) of eta_j is
unchanged.

Both steps leave intercept + sum(main) + sum(inter) unchanged at every x.
"""
import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, solve
from scipy.stats import chi2, norm

M_MIN = None        # minimum X_train rows per merged union bin; None = max(M_FLOOR, n^M_EXP), see min_count
M_FLOOR = 30
M_EXP = 0.75        # rows per bin ~ n^(3/4), i.e. about n^(1/4) bins per variable
N_FOLDS = 3         # cross-fitting folds (deterministic: row index mod N_FOLDS)
RIDGE = 1e-8        # numerical ridge on the normal equations, in units of rows
# Difference-penalty grid, lambda = tau in units of rows: bins holding many more than tau rows are barely smoothed,
# thin bins are pooled with their neighbours. tau = inf forces g_j, g_k constant, i.e. no transfer for the pair.
TAU_GRID = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 1e4, np.inf)
SE_RULE = 2.0       # step 1: paired "z standard errors" rule (0 = plain argmin of the held-out error)
CLOSE = True        # step 2 (closing step on the final main effects)
CLOSE_Z = 0.0       # step 2: >0 moves a pair only if its leak is significant at |N(0,1)| > CLOSE_Z (ablation)
SCALE_BOUNDS = (0.5, 2.0)  # step 2 is skipped for a variable whose main effect would be rescaled outside these bounds
DIRECTIONS = False  # step 1: add the parents' main effects as unpenalised columns (ablation only, see NOTES.md)


def min_count(n, m_min=M_MIN):
    """Rows per merged bin. Default n^(3/4), i.e. about n^(1/4) bins: deliberately coarser than the MSE-optimal n^(1/3)
    bins of a histogram regression, so that step 1 only makes coarse (smooth, reliably estimated) shape corrections;
    the fine-grid part along the final main effects is left to step 2, which adds no roughness. Never below M_FLOOR."""
    return int(m_min) if m_min else max(M_FLOOR, int(np.ceil(n ** M_EXP)))


def union_edges(xgb_table, X, m_min=M_FLOOR):
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


def _sym_solve(A, b):
    A = A.copy()
    A[np.diag_indices_from(A)] += RIDGE * np.maximum(1.0, np.abs(np.diag(A)))
    try:
        return cho_solve(cho_factor(A, check_finite=False), b, check_finite=False)
    except LinAlgError:
        try:
            return solve(A, b, assume_a="sym", check_finite=False)
        except LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]


class PairSystem:
    """Normal equations of min sum_i (y_i - g_j[bj_i] - g_k[bk_i] - U_i beta)^2 + tau (|D g_j|^2 + |D g_k|^2).

    U (n, q) holds unpenalised columns (the parents' current main effects, see fit_projection); q may be 0.
    Gauge: g_k[0] = 0 (the constant is carried by g_j; the penalty only sees differences, so this is exact).
    """

    def __init__(self, bj, bk, nbj, nbk, Pj, Pk, U):
        self.bj, self.bk, self.nbj, self.nbk, self.Pj, self.Pk, self.U = bj, bk, nbj, nbk, Pj, Pk, U

    def stats(self, rows, y):
        bj, bk, nbj, nbk, U, yr = self.bj[rows], self.bk[rows], self.nbj, self.nbk, self.U[rows], y[rows]
        nj = np.bincount(bj, minlength=nbj).astype(float)
        nk = np.bincount(bk, minlength=nbk).astype(float)
        C = np.bincount(bj * nbk + bk, minlength=nbj * nbk).reshape(nbj, nbk).astype(float)
        rj = np.bincount(bj, weights=yr, minlength=nbj)
        rk = np.bincount(bk, weights=yr, minlength=nbk)
        Uj = np.zeros((nbj, U.shape[1]))
        Uk = np.zeros((nbk, U.shape[1]))
        for a in range(U.shape[1]):
            Uj[:, a] = np.bincount(bj, weights=U[:, a], minlength=nbj)
            Uk[:, a] = np.bincount(bk, weights=U[:, a], minlength=nbk)
        return nj, nk, C, rj, rk, Uj, Uk, U.T @ U, U.T @ yr

    def solve(self, st, tau):
        """Returns (g_j, g_k, beta)."""
        nj, nk, C, rj, rk, Uj, Uk, UU, Uy = st
        nbj, nbk, q = self.nbj, self.nbk, UU.shape[0]
        if np.isinf(tau):  # infinite difference penalty: g_j constant, g_k = 0, plus U beta
            A = np.block([[np.array([[nj.sum()]]), Uj.sum(0)[None, :]], [Uj.sum(0)[:, None], UU]])
            sol = _sym_solve(A, np.concatenate([[rj.sum()], Uy]))
            return np.full(nbj, sol[0]), np.zeros(nbk), sol[1:]
        dim = nbj + nbk + q
        A = np.zeros((dim, dim))
        A[:nbj, :nbj] = np.diag(nj) + tau * self.Pj
        A[nbj:nbj + nbk, nbj:nbj + nbk] = np.diag(nk) + tau * self.Pk
        A[:nbj, nbj:nbj + nbk] = C
        A[nbj:nbj + nbk, :nbj] = C.T
        A[:nbj, nbj + nbk:] = Uj
        A[nbj + nbk:, :nbj] = Uj.T
        A[nbj:nbj + nbk, nbj + nbk:] = Uk
        A[nbj + nbk:, nbj:nbj + nbk] = Uk.T
        A[nbj + nbk:, nbj + nbk:] = UU
        b = np.concatenate([rj, rk, Uy])
        keep = np.r_[0:nbj, nbj + 1:dim]  # drop g_k[0] (gauge)
        sol = _sym_solve(A[np.ix_(keep, keep)], b[keep])
        return sol[:nbj], np.concatenate([[0.0], sol[nbj:nbj + nbk - 1]]), sol[nbj + nbk - 1:]


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


def _heldout(system, y, bj, bk, mj, mk, dirs, folds, n_folds, tau_grid, criterion):
    """Cross-fitted held-out losses of one pair: per row and tau for "mse", per tau for "leak".

    dirs: for each column of system.U, 0 if it is the main effect of x_j, 1 if of x_k.
    """
    G = len(tau_grid)
    loss = np.zeros((G, y.size))
    leak = np.zeros(G)
    U = system.U
    for f in range(n_folds):
        tr, ho = folds != f, folds == f
        st = system.stats(tr, y)
        for t, tau in enumerate(tau_grid):
            gj, gk, beta = system.solve(st, tau)
            v = [gj[bj[ho]], gk[bk[ho]]]
            for a, d in enumerate(dirs):
                v[d] = v[d] + beta[a] * U[ho, a]
            r = y[ho] - v[0] - v[1]
            loss[t, ho] = (r - r.mean()) ** 2
            if criterion == "leak":
                leak[t] += _leak(r, [mj[ho] + v[0], mk[ho] + v[1]])
    return loss, leak


def _choose(loss, leak, criterion, se_rule):
    """Index of the chosen tau. Exact ties go to the smoother (larger) tau.

    With se_rule = z > 0 ("mse" only): the largest tau whose held-out error exceeds the minimum by at most z standard
    errors of the paired per-row difference (so "no transfer" is kept unless the transfer is a significant gain).
    """
    G = loss.shape[0]
    tot = loss.sum(1) if criterion == "mse" else leak
    t_min = G - 1 - int(np.argmin(tot[::-1]))
    if criterion != "mse" or se_rule <= 0:
        return t_min
    n = loss.shape[1]
    for t in range(G - 1, t_min, -1):
        d = loss[t] - loss[t_min]
        if d.sum() <= se_rule * np.sqrt(n) * d.std():
            return t
    return t_min


def _close(main, inter, il, cols, z=CLOSE_Z, bounds=SCALE_BOUNDS):
    """Step 2: per pair, least squares of eta_jk on its parents' main effects; returns (coef (K, 2), scale (p,)).

    With z > 0 (ablation; the default z = 0 moves every pair), a pair is moved only if its leak is significant: robust
    (HC0 sandwich) Wald statistic of the coefficients above the chi-square quantile with the tail probability of
    |N(0,1)| > z. A variable whose rescale factor falls outside `bounds` is excluded and the pairs are refitted without
    it, until all factors are within bounds.
    """
    p = main.shape[1]
    Mc = main - main.mean(0)
    var = np.mean(Mc ** 2, 0)
    excluded = var <= 1e-12 * max(var.max(), 1e-300)
    alpha = 2.0 * norm.sf(z)
    for _ in range(p + 1):
        coef = np.zeros((len(il), 2))
        for c in cols:
            j, k = il[c]
            use = [a for a, v in enumerate((j, k)) if not excluded[v]]
            if not use:
                continue
            Z = Mc[:, [(j, k)[a] for a in use]]
            y = inter[:, c] - inter[:, c].mean()
            beta = np.linalg.lstsq(Z, y, rcond=None)[0]
            if z > 0:
                e = y - Z @ beta
                bread = np.linalg.pinv(Z.T @ Z)
                V = bread @ ((Z * (e * e)[:, None]).T @ Z) @ bread
                try:
                    wald = float(beta @ np.linalg.solve(V, beta))
                except np.linalg.LinAlgError:
                    wald = np.inf
                if not wald > chi2.isf(alpha, len(use)):
                    continue
            coef[c, use] = beta
        scale = np.ones(p)
        for c in cols:
            j, k = il[c]
            scale[j] += coef[c, 0]
            scale[k] += coef[c, 1]
        bad = ~excluded & ((scale < bounds[0]) | (scale > bounds[1]))
        if not bad.any():
            break
        excluded |= bad
    return coef, scale


def fit_projection(eta0, main_tr, inter_tr, inter_list, X, xgb_table, m_min=M_MIN, tau_grid=TAU_GRID,
                   n_folds=N_FOLDS, criterion="mse", se_rule=SE_RULE, close=CLOSE, directions=DIRECTIONS,
                   close_z=CLOSE_Z):
    """Fit the transfers on X_train. Returns (state dict, new intercept, diagnostics).

    Step 1 model per pair: eta_jk ~ g_j(b_j) + g_k(b_k) [+ beta_j eta_j + beta_k eta_k if directions], where eta_j are
    the TreeHFD main effects (unpenalised columns). With directions, the reference "tau = inf" is the main-effect
    direction fit, so penalised bin terms are only moved when they add significant structure beyond it.
    criterion (step 1, held-out on the rows of the left-out fold):
      "mse":  squared error of the additive fit (with the paired SE rule se_rule);
      "leak": _leak(residual, [eta_j + transfer_j, eta_k + transfer_k]) (plain argmin, for ablations).
    """
    n, p = X.shape
    edges = union_edges(xgb_table, X, min_count(n, m_min))
    bins = [np.digitize(X[:, j], edges[j]) for j in range(p)]
    nbins = [e.size + 1 for e in edges]
    pens = [diff_penalty(b) for b in nbins]
    il = np.asarray(inter_list, dtype=int).reshape(-1, 2)
    nonzero = [c for c in range(il.shape[0]) if np.var(inter_tr[:, c]) > 0]
    mvar = np.var(main_tr, 0)
    folds = np.arange(n) % n_folds
    G = len(tau_grid)
    t_best = {}
    pairs, shift = [], 0.0
    for c in nonzero:
        j, k = il[c]
        if nbins[j] == 1 and nbins[k] == 1:
            continue
        dirs = [a for a, v in enumerate((j, k)) if directions and mvar[v] > 0]
        U = main_tr[:, [(j, k)[a] for a in dirs]]
        system = PairSystem(bins[j], bins[k], nbins[j], nbins[k], pens[j], pens[k], U)
        y = inter_tr[:, c]
        loss, leak = _heldout(system, y, bins[j], bins[k], main_tr[:, j], main_tr[:, k], dirs, folds, n_folds,
                              tau_grid, criterion)
        t_best[c] = _choose(loss, leak, criterion, se_rule)
        tau = tau_grid[t_best[c]]
        if np.isinf(tau):  # no bin transfer (a direction-only part is left to step 2)
            continue
        gj, gk, beta = system.solve(system.stats(np.ones(n, bool), y), tau)
        b = np.zeros(2)
        b[dirs] = beta
        cj = float(np.mean(gj[bins[j]] + b[0] * main_tr[:, j]))
        ck = float(np.mean(gk[bins[k]] + b[1] * main_tr[:, k]))
        shift += cj + ck
        pairs.append(dict(col=int(c), j=int(j), k=int(k), gj=gj, gk=gk, bj=float(b[0]), bk=float(b[1]),
                          cj=cj, ck=ck, tau=float(tau)))
    proj = dict(edges=edges, pairs=pairs, il=il, coef=None, scale=None)
    diag = dict(tau_hist=np.bincount(list(t_best.values()), minlength=G).tolist(), n_transfer=len(pairs),
                n_pairs=il.shape[0], n_bins=nbins)
    if close and nonzero:
        main1, inter1 = apply_projection(proj, main_tr, inter_tr, X)
        proj["coef"], proj["scale"] = _close(main1, inter1, il, nonzero, close_z)
        diag["n_closed"] = int(np.any(proj["coef"] != 0, axis=1).sum())
        diag["scale"] = proj["scale"].tolist()
    return proj, float(eta0) + shift, diag


def apply_projection(proj, main, inter, X):
    """Apply step 1 then step 2 (pointwise sum-preserving). main, inter: TreeHFD components at the rows of X."""
    base = main
    main, inter = main.copy(), inter.copy()
    bins = {}
    for pr in proj["pairs"]:
        j, k, c = pr["j"], pr["k"], pr["col"]
        for v in (j, k):
            if v not in bins:
                bins[v] = np.digitize(X[:, v], proj["edges"][v])
        vj = pr["gj"][bins[j]] + pr["bj"] * base[:, j] - pr["cj"]
        vk = pr["gk"][bins[k]] + pr["bk"] * base[:, k] - pr["ck"]
        inter[:, c] -= vj + vk + pr["cj"] + pr["ck"]
        main[:, j] += vj
        main[:, k] += vk
    if proj["coef"] is not None:
        coef, il = proj["coef"], proj["il"]
        for c in np.flatnonzero(np.any(coef != 0, axis=1)):
            j, k = il[c]
            inter[:, c] -= coef[c, 0] * main[:, j] + coef[c, 1] * main[:, k]
        main *= proj["scale"]
    return main, inter
