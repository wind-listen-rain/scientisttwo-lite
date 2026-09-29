"""Exact ground-truth HFD (mains + pairs) of a low-degree polynomial under a Gaussian input law.

Hoeffding functional decomposition with hierarchical orthogonality (Chastaing et al. 2012):
f = c0 + sum_j eta_j(x_j) + sum_{j<k} eta_jk(x_j, x_k), E[eta_j] = 0, E[eta_jk] = 0,
E[eta_jk g(x_j)] = 0, E[eta_jk g(x_k)] = 0 for all g.
For a polynomial f of degree <= D under a Gaussian law every component is a polynomial of degree <= D
(conditional expectations of Gaussian vectors preserve the degree), so it suffices to
  * write each component in the monomial basis (degree <= D),
  * impose f = sum of components as a polynomial identity,
  * impose the orthogonality conditions against monomials x_j^a, x_k^b (a, b <= D), with exact
    Gaussian moments (Gauss-Hermite quadrature after a Cholesky transform),
and solve the stacked linear system. The solution is unique (rank checked); the identity residual is reported.
"""
import itertools

import numpy as np
from numpy.polynomial.hermite_e import hermegauss


def _gh_moments(cov2, deg, nq=24):
    """E[x^a y^b] for a bivariate zero-mean Gaussian with covariance cov2, a, b <= deg (exact)."""
    z, w = hermegauss(nq)
    w = w / w.sum()
    L = np.linalg.cholesky(cov2)
    Z1, Z2 = np.meshgrid(z, z, indexing="ij")
    W = np.outer(w, w)
    x = L[0, 0] * Z1
    y = L[1, 0] * Z1 + L[1, 1] * Z2
    return np.array([[np.sum(W * x ** a * y ** b) for b in range(deg + 1)] for a in range(deg + 1)])


class PolyHFD:
    def __init__(self, cov, fcoef, deg=3):
        """cov: (p, p); fcoef: {exponent tuple (len p): coefficient} (degree <= deg)."""
        p = cov.shape[0]
        self.p, self.deg, self.cov = p, deg, cov
        monos2 = [(a, b) for a in range(deg + 1) for b in range(deg + 1) if a + b <= deg]
        self.pairs = list(itertools.combinations(range(p), 2))
        cols = []       # (kind, index, exponent-tuple over p variables)
        for j in range(p):
            for a in range(0, deg + 1):
                e = [0] * p
                e[j] = a
                cols.append(("main", j, tuple(e)))
        for (j, k) in self.pairs:
            for a, b in monos2:
                e = [0] * p
                e[j], e[k] = a, b
                cols.append(("pair", (j, k), tuple(e)))
        self.cols = cols
        allmon = sorted({e for e in itertools.product(range(deg + 1), repeat=p) if sum(e) <= deg})
        midx = {e: i for i, e in enumerate(allmon)}
        nun = 1 + len(cols)            # intercept + coefficients
        rows, rhs = [], []
        # (1) polynomial identity
        A = np.zeros((len(allmon), nun))
        A[midx[tuple([0] * p)], 0] = 1.0
        for c, (_, _, e) in enumerate(cols):
            A[midx[e], 1 + c] = 1.0
        rows.append(A)
        rhs.append(np.array([fcoef.get(e, 0.0) for e in allmon]))
        # (2) orthogonality; moments from 1-D / 2-D Gaussian marginals
        z, w = hermegauss(24)
        w = w / w.sum()

        def m1(j, a):
            return np.sum(w * (np.sqrt(cov[j, j]) * z) ** a)
        for j in range(p):
            r = np.zeros(nun)
            for c, (kind, idx, e) in enumerate(cols):
                if kind == "main" and idx == j:
                    r[1 + c] = m1(j, e[j])
            rows.append(r[None]); rhs.append(np.zeros(1))
        for (j, k) in self.pairs:
            M = _gh_moments(cov[np.ix_([j, k], [j, k])], 2 * deg)
            tests = [(0, 0)] + [(a, 0) for a in range(1, deg + 1)] + [(0, b) for b in range(1, deg + 1)]
            R = np.zeros((len(tests), nun))
            for c, (kind, idx, e) in enumerate(cols):
                if kind == "pair" and idx == (j, k):
                    for t, (ta, tb) in enumerate(tests):
                        R[t, 1 + c] = M[e[j] + ta, e[k] + tb]
            rows.append(R); rhs.append(np.zeros(len(tests)))
        S, y = np.vstack(rows), np.concatenate(rhs)
        self.rank = np.linalg.matrix_rank(S)
        self.nun = nun
        sol, *_ = np.linalg.lstsq(S, y, rcond=None)
        self.residual = float(np.max(np.abs(S @ sol - y)))
        self.eta0 = float(sol[0])
        self.coef = sol[1:]

    def components(self, X):
        """(main (n, p), pair (n, n_pairs)) of the exact HFD at the points X."""
        n = X.shape[0]
        main = np.zeros((n, self.p))
        pair = np.zeros((n, len(self.pairs)))
        pidx = {q: i for i, q in enumerate(self.pairs)}
        for c, (kind, idx, e) in enumerate(self.cols):
            term = self.coef[c] * np.prod([X[:, j] ** a for j, a in enumerate(e) if a], axis=0)
            if kind == "main":
                main[:, idx] += term
            else:
                pair[:, pidx[idx]] += term
        return main, pair
