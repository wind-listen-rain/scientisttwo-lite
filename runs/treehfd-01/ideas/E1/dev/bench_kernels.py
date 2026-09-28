"""Micro-benchmark of the block-K kernel variants (diagnostic)."""
import time, numpy as np
rng = np.random.default_rng(0)
G, mr = 12, 800
D = 1.0 / (rng.random(mr)[None, :] * 10 + np.logspace(-2, 4, G)[:, None] / 1e4)
for b, Nb in ((2, 4000), (5, 4000), (8, 4000), (12, 2000), (24, 500)):
    Zb = rng.standard_normal((Nb, b, mr))
    out = {}
    # A: current (loop over r, elementwise products, GEMM with D.T), chunked
    def A(Zb):
        n = Zb.shape[0]; K = np.empty((G, n, b, b))
        for r in range(b):
            P = (Zb[:, r:r + 1, :] * Zb[:, r:, :]).reshape(-1, mr) @ D.T
            P = P.reshape(n, b - r, G).transpose(2, 0, 1)
            K[:, :, r, r:] = P; K[:, :, r:, r] = P
        return K
    def chunked(f, step):
        return lambda Zb: np.concatenate([f(Zb[c:c + step]) for c in range(0, len(Zb), step)], axis=1)
    # E: (Nb, b*G, mr) @ (Nb, mr, b)
    def E(Zb):
        n = Zb.shape[0]
        Zw = (Zb[:, :, None, :] * D[None, None, :, :]).reshape(n, b * G, mr)
        K = np.matmul(Zw, Zb.transpose(0, 2, 1)).reshape(n, b, G, b)
        return K.transpose(2, 0, 1, 3)
    # F: per g batched matmul
    def F(Zb):
        Zt = Zb.transpose(0, 2, 1)
        return np.stack([np.matmul(Zb * D[g], Zt) for g in range(G)])
    # H: pair products via triu indices then GEMM
    iu, ju = np.triu_indices(b)
    def Hk(Zb):
        n = Zb.shape[0]
        P = (Zb[:, iu, :] * Zb[:, ju, :]).reshape(-1, mr) @ D.T
        P = P.reshape(n, len(iu), G).transpose(2, 0, 1)
        K = np.empty((G, n, b, b)); K[:, :, iu, ju] = P; K[:, :, ju, iu] = P
        return K
    ref = A(Zb)
    for name, f in (("A", A), ("A_c", chunked(A, max(1, int(3e5 // (b * (b + 1) // 2 * mr))) or 1)),
                    ("E_c", chunked(E, max(1, int(3e5 // (b * G * mr))))), ("F_c", chunked(F, max(1, int(1e6 // (b * mr))))),
                    ("H_c", chunked(Hk, max(1, int(3e5 // (len(iu) * mr)))))):
        t = time.perf_counter(); K = f(Zb); el = time.perf_counter() - t
        out[name] = (round(el, 3), float(np.max(np.abs(K - ref)) / np.max(np.abs(ref))))
    fl = Nb * b * (b + 1) / 2 * mr * 2 * G
    print(b, Nb, f"ideal-sym GFlop {fl / 1e9:.2f}", out, flush=True)
