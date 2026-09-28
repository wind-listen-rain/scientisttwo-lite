"""评测环境指纹：判断两台机器上的官方评测结果能不能直接比较。

评测协议文件（bench/）逐字节相同，并不保证两台机器评测的是同一批数据。2026-09-28 从 Mac（arm64）换到 Windows（x86-64）
时实测发现三处会跟着平台变（详见 docs/JOURNAL.md）：
  1. 合成数据：harness 用 multivariate_normal 采样，协方差是等相关矩阵，特征值有 5 重重复，
     SVD 给出的特征基由 LAPACK 实现决定 → 同一个种子抽出不同的样本；
  2. XGBoost 模型（实测两平台一致，仍然检查）；
  3. locvar_in 用的一维 10 近邻：有并列值时选哪几个邻居由实现决定（abalone 实测差 7.7%）。
这里把三者的哈希合成一个指纹。指纹相同 → 数据、模型、近邻完全一致，结果可以直接比较；否则要在本机重建基线。
只读地复用 bench/harness.py 的常量与采样代码，不改评测协议。
用法: python tools/env_fingerprint.py  → 在标准输出打印 JSON
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import harness as H  # noqa: E402


def h(*arrays):
    d = hashlib.sha256()
    for a in arrays:
        d.update(np.ascontiguousarray(a).tobytes())
    return d.hexdigest()


def main():
    parts = {}
    mu, cov = np.zeros(H.DIM), np.full((H.DIM, H.DIM), H.RHO)
    np.fill_diagonal(cov, 1.0)
    an = hashlib.sha256()
    an_model = hashlib.sha256()
    for rep in range(10):  # 与 harness.analytical 完全相同的采样顺序
        rng_tr, rng_te = np.random.default_rng(1000 + rep), np.random.default_rng(2000 + rep)
        X = rng_tr.multivariate_normal(mu, cov, size=H.NSAMPLE)
        y = np.sin(2 * np.pi * X[:, 0]) + X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] + rng_tr.normal(0, 0.5, H.NSAMPLE)
        Xn = rng_te.multivariate_normal(mu, cov, size=H.NSAMPLE)
        an.update(h(X, y, Xn).encode())
        model = xgb.XGBRegressor(random_state=rep, **H.XGB_PARAMS).fit(X, y)
        an_model.update(h(model.predict(Xn)).encode())
    parts["analytical_data"] = an.hexdigest()
    parts["analytical_models"] = an_model.hexdigest()
    real_model, knn = hashlib.sha256(), hashlib.sha256()
    for name in H.FULL_DATA:  # 与 harness.real_data 相同的划分与模型
        d = np.load(H.DATA / f"{name}.npz")
        X, y = d["X"], d["y"]
        perm = np.random.default_rng(0).permutation(len(y))
        cut = int(0.8 * len(y))
        tr, te = perm[:cut], perm[cut:]
        model = xgb.XGBRegressor(random_state=0, **H.XGB_PARAMS).fit(X[tr], y[tr])
        real_model.update(h(model.predict(X[tr]), model.predict(X[te])).encode())
        for j in range(X.shape[1]):  # 与 harness.local_variability 相同的近邻查询
            nn = NearestNeighbors(n_neighbors=min(10, len(tr))).fit(X[tr][:, [j]])
            knn.update(h(nn.kneighbors(X[tr][:, [j]], return_distance=False)).encode())
    parts["real_models"] = real_model.hexdigest()
    parts["knn"] = knn.hexdigest()
    fp = hashlib.sha256("".join(parts[k] for k in sorted(parts)).encode()).hexdigest()[:12]
    print(json.dumps({"fp": fp, "parts": parts, "platform": f"{platform.system()}-{platform.machine()}",
                      "python": platform.python_version(), "numpy": np.__version__, "xgboost": xgb.__version__,
                      "sklearn": sklearn.__version__}, ensure_ascii=False))


if __name__ == "__main__":
    main()
