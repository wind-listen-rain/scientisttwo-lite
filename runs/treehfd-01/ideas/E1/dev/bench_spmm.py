"""Micro-benchmark: incidence (0/1, fixed nnz per row) times dense (diagnostic)."""
import time, numpy as np
from scipy import sparse
rng = np.random.default_rng(0)
m, mr, ncomp = 1000, 900, 20
T = rng.standard_normal((m, mr))
for nrow in (5000, 35000):
    C = rng.integers(0, m, size=(nrow, ncomp))
    H = sparse.csr_matrix((np.ones(nrow * ncomp), C.ravel(), np.arange(0, nrow * ncomp + 1, ncomp)), shape=(nrow, m))
    t = time.perf_counter(); Z1 = np.asarray(H @ T); t1 = time.perf_counter() - t
    t = time.perf_counter()
    Z2 = np.empty((nrow, mr))
    step = 512
    for c0 in range(0, nrow, step):
        Cc = C[c0:c0 + step]
        z = T[Cc[:, 0]].copy()
        for c in range(1, ncomp):
            z += T[Cc[:, c]]
        Z2[c0:c0 + step] = z
    t2 = time.perf_counter() - t
    t = time.perf_counter(); Z3 = np.asarray((T.T @ H.T.tocsr().T.T.toarray().T.T) if False else 0); t3 = 0
    # dense GEMM with the dense incidence matrix
    t = time.perf_counter(); Hd = H.toarray(); Z4 = Hd @ T; t4 = time.perf_counter() - t
    # float32 T for comparison
    print(nrow, f"csr {t1:.3f}s gather {t2:.3f}s dense-gemm {t4:.3f}s maxdiff {np.max(np.abs(Z1 - Z2)):.1e} {np.max(np.abs(Z1 - Z4)):.1e}")
