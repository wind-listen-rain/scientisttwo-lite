"""独立复核：用流水线从没见过的随机种子，把基线和 S4 做逐次配对对比。

- 合成数据：10 次重复，训练/测试种子 5000+rep / 6000+rep（官方评测用 1000+rep / 2000+rep）
- 真实数据：8 个数据集 × 5 种新划分（种子 1001..1005；官方划分是 0，S4 的工程智能体诊断时用过 1..7）
  Superconductivity 基线单次要 12 分钟，不纳入
复用 bench/harness.py 的指标函数与子进程运行器，只把种子换掉；输出 verify/independent_check.json
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import harness as H  # noqa: E402

METHODS = {"baseline": ROOT / "bench" / "baseline_method.py", "S4": ROOT / "runs" / "treehfd-01" / "ideas" / "S4" / "method.py"}
DATASETS = ["abalone", "airfoil", "concrete", "nutrition", "powerplant", "parkinson", "housing", "bike"]
SPLITS = [1001, 1002, 1003, 1004, 1005]
REPS = range(10)


def analytical_rep(method, rep):
    mu, cov = np.zeros(H.DIM), np.full((H.DIM, H.DIM), H.RHO)
    np.fill_diagonal(cov, 1.0)
    rng_tr, rng_te = np.random.default_rng(5000 + rep), np.random.default_rng(6000 + rep)
    X = rng_tr.multivariate_normal(mu, cov, size=H.NSAMPLE)
    y = np.sin(2 * np.pi * X[:, 0]) + X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] + rng_tr.normal(0, 0.5, H.NSAMPLE)
    Xn = rng_te.multivariate_normal(mu, cov, size=H.NSAMPLE)
    model = xgb.XGBRegressor(random_state=5000 + rep, **H.XGB_PARAMS).fit(X, y)
    out = H.run_method(method, model, X, Xn, 3600)
    main, inter, il = out["main_eval"], out["inter_eval"], [tuple(p) for p in out["inter_list"]]
    target = np.zeros((H.NSAMPLE, H.DIM))
    target[:, 0] = np.sin(2 * np.pi * Xn[:, 0]) + H.eta_main(Xn[:, 0])
    for j in (1, 2, 3):
        target[:, j] = H.eta_main(Xn[:, j])
    row = {f"mse_eta{j + 1}": float(np.mean((target[:, j] - main[:, j]) ** 2)) for j in range(H.DIM)}
    for name, pair, tgt in (("mse_eta12", (0, 1), H.eta_order2(Xn[:, 0], Xn[:, 1])),
                            ("mse_eta34", (2, 3), H.eta_order2(Xn[:, 2], Xn[:, 3]))):
        est = inter[:, il.index(pair)] if pair in il else np.zeros(H.NSAMPLE)
        row[name] = float(np.mean((tgt - est) ** 2))
    others = [c for c, p in enumerate(il) if p not in ((0, 1), (2, 3))]
    row["mse_others"] = float(np.mean(inter[:, others] ** 2)) if others else 0.0
    row["resid_out"] = H.resid_ratio(model.predict(Xn), out, "eval")
    return row


def real_split(method, name, seed):
    d = np.load(H.DATA / f"{name}.npz")
    X, y = d["X"], d["y"]
    perm = np.random.default_rng(seed).permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = perm[:cut], perm[cut:]
    model = xgb.XGBRegressor(random_state=seed, **H.XGB_PARAMS).fit(X[tr], y[tr])
    out = H.run_method(method, model, X[tr], X[te], 3600)
    p_tr, p_te = model.predict(X[tr]), model.predict(X[te])
    il = out["inter_list"]
    return {"resid_in": H.resid_ratio(p_tr, out, "train"), "resid_out": H.resid_ratio(p_te, out, "eval"),
            "ortho_in": H.orthogonality(p_tr, out["main_train"], out["inter_train"], il),
            "ortho_out": H.orthogonality(p_te, out["main_eval"], out["inter_eval"], il),
            "locvar_in": H.local_variability(X[tr], out["main_train"]), "fit_s": out["timing"]["fit_s"]}


def main():
    jobs = [("analytical", m, r) for m in METHODS for r in REPS] + \
           [(ds, m, s) for ds in DATASETS for s in SPLITS for m in METHODS]
    res = {}

    def work(job):
        kind, m, k = job
        r = analytical_rep(METHODS[m], k) if kind == "analytical" else real_split(METHODS[m], kind, k)
        print(f"{kind} {m} {k} done", flush=True)
        return job, r

    with ThreadPoolExecutor(max_workers=6) as pool:
        for (kind, m, k), r in pool.map(work, jobs):
            res.setdefault(kind, {}).setdefault(m, {})[str(k)] = r
    (Path(__file__).parent / "independent_check.json").write_text(json.dumps(res, indent=1))
    print("all done", flush=True)


if __name__ == "__main__":
    main()
