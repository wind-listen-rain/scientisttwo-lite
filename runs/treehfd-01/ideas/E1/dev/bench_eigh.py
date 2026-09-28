"""CPU time vs wall time of numpy eigh (is LAPACK multi-threaded here?)."""
import time

import numpy as np

rng = np.random.default_rng(0)
for m in (400, 800):
    A = rng.standard_normal((m, m))
    A = A @ A.T / m
    np.linalg.eigh(A)
    w0, c0 = time.perf_counter(), time.process_time()
    for _ in range(20):
        np.linalg.eigh(A)
    w, c = (time.perf_counter() - w0) / 20, (time.process_time() - c0) / 20
    print(f"m={m}: wall {w * 1e3:.1f} ms, cpu {c * 1e3:.1f} ms, cpu/wall {c / w:.2f}")
