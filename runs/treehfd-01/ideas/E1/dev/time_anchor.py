"""Section timings inside GTLocoTree._path_anchor and fit (diagnostic, results unchanged).

usage: time_anchor.py <dataset|analytical> [n_trees] [first_tree] [chunk]
Captures the arguments of every _path_anchor call, then re-runs the sections with timers.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

E1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E1 / "lib"))
sys.path.insert(0, str(E1 / "dev"))
import agtloco  # noqa: E402
from common import analytical, real  # noqa: E402

agtloco.ANCHOR_PRIORS = ("lattice",)
name = sys.argv[1]
nt = int(sys.argv[2]) if len(sys.argv) > 2 else 20
t_first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
if len(sys.argv) > 4:
    agtloco.BLOCK_CHUNK = float(sys.argv[4])
model, Xtr, _ = analytical(0) if name == "analytical" else real(name)
T = agtloco.GTLocoTree
calls = []
orig = T._path_anchor


def spy(self, *a, **k):
    calls.append((self, a, k))
    return orig(self, *a, **k)


T._path_anchor = spy
ens = agtloco.GTLocoHFD(model)
X = np.asarray(Xtr, dtype=float)
Ttr = ens._tree_predict(X)
X32 = agtloco._f32(X)
union = agtloco._union_splits(ens.xgb_table)
anchor = {"X32": X32, "shifts": agtloco._shift_values(X, X32, union), "n_shift": 2 * len(union)}
t0 = time.perf_counter()
for t in range(t_first, t_first + nt):
    table = pd.DataFrame(ens.xgb_table[ens.xgb_table["Tree"] == t])
    tree = T(table, 2, ens.max_depth)
    tree.fit(X, Ttr[:, t], anchor)
print(f"{name}: {nt} trees fit {time.perf_counter() - t0:.2f}s, {len(calls)} anchored paths")
tim = {}


def tic(key, t0):
    tim[key] = tim.get(key, 0.0) + time.perf_counter() - t0
    return time.perf_counter()


KAPPAS = agtloco.KAPPAS
for self, (Tm, S, rhs, H, y, idx1, orphans, n, va, wv), _ in calls:
    t = time.perf_counter()
    G = len(KAPPAS)
    q = Tm.T @ rhs
    D = 1.0 / (S[None, :] + KAPPAS[:, None] / n)
    betas = Tm @ (D * q[None, :]).T
    resid = y[None, :] - (H @ betas).T
    t = tic("betas+resid", t)
    mr = Tm.shape[1]
    Zr = np.asarray(H[idx1] @ Tm)
    er = resid[:, idx1]
    t = tic("Zr", t)
    Hn = va["Hv"][va["vneed"]]
    Zv = np.vstack([np.asarray(Hn @ Tm), np.zeros((1, mr))])
    ev = np.hstack([(va["yv"][va["vneed"]][:, None] - Hn @ betas).T, np.zeros((G, 1))])
    t = tic("Zv", t)
    rows, pts, funcs = orphans
    ATs = [np.asarray(A @ Tm) for A in funcs]
    orph_of = np.full(len(idx1), -1, dtype=int)
    orph_of[rows] = np.arange(len(rows))
    t = tic("ATs", t)
    for mem, vpos, vmult in va["blocks"]:
        b = 1 + vpos.shape[1]
        tim.setdefault(f"n_b{b}", 0)
        tim[f"n_b{b}"] += len(mem)
        iu, ju = np.triu_indices(b)
        step = max(1, int(agtloco.BLOCK_CHUNK // (max(len(iu), b) * mr + 1)))
        for c0 in range(0, len(mem), step):
            mm, vp = mem[c0:c0 + step], vpos[c0:c0 + step]
            W = np.hstack([np.full((len(mm), 1), 1.0 / n), wv * vmult[c0:c0 + step]])
            Zb = np.concatenate([Zr[mm][:, None, :], Zv[vp]], axis=1)
            Eb = np.concatenate([er[:, mm][:, :, None], ev[:, vp]], axis=2)
            t = tic("gather", t)
            if b == 1:
                h = (D @ (Zb[:, 0, :] ** 2).T) / n
                Em = Eb / np.maximum(1.0 - h, 1e-12)[:, :, None]
                t = tic("b1", t)
            else:
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
                t = tic(f"K b<=8" if b <= 8 else "K b>8", t)
                Em = np.linalg.solve(np.eye(b) - K * W[None, :, None, :], Eb[..., None])[..., 0]
                t = tic("solve", t)
            o = orph_of[mm]
            sel = o >= 0
            if sel.any():
                WE = Em[:, sel, :] * W[sel][None, :, :]
                for ru, AT in enumerate(ATs):
                    F = (AT[o[sel]][:, None, :] * Zb[sel]).reshape(-1, mr) @ D.T
                    F = F.reshape(int(sel.sum()), b, G).transpose(2, 0, 1)
                    _ = np.sum(F * WE, axis=2)
            t = tic("orphan F", t)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in sorted(tim.items())})
