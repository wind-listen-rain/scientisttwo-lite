"""TreeHFD 任务的只读评测脚本（按 Bénard, NeurIPS 2025 的 Table 1 / Table 2 协议重建）。

用法:
  python bench/harness.py --method <method.py> --mode subset|full --out results.json [--timeout 3600]

子集模式: 解析案例 3 次重复 + airfoil / concrete / abalone
全量模式: 解析案例 10 次重复 + 论文 Table 2 的 9 个数据集
被测方法在子进程里运行（见 runner.py），拿不到标签和真实分量；指标全部在本进程计算。
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors

BENCH = Path(__file__).resolve().parent
DATA = BENCH / "data"
SUBSET_DATA = ["airfoil", "concrete", "abalone"]
FULL_DATA = ["abalone", "airfoil", "bike", "housing", "concrete", "nutrition", "parkinson", "powerplant", "superconduct"]
XGB_PARAMS = dict(eta=0.1, n_estimators=100, max_depth=6, n_jobs=4)  # 与 treehfd 官方示例一致
DIM, NSAMPLE, RHO = 6, 5000, 0.5


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def protocol_manifest():
    files = sorted(BENCH.glob("*.py")) + sorted(DATA.glob("*.npz"))
    return {str(f.relative_to(BENCH)): sha256(f) for f in files}


def run_method(method, model, X_train, X_eval, timeout):
    with tempfile.TemporaryDirectory() as d:
        job = Path(d)
        model.save_model(job / "model.json")
        np.save(job / "X_train.npy", X_train)
        np.save(job / "X_eval.npy", X_eval)
        (job / "config.json").write_text(json.dumps({"interaction_order": 2}))
        r = subprocess.run([sys.executable, str(BENCH / "runner.py"), str(method), str(job)],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"方法运行失败:\n{r.stderr[-3000:]}")
        out = dict(np.load(job / "out.npz"))
        out["timing"] = json.loads((job / "timing.json").read_text())
    n_tr, n_ev, p = X_train.shape[0], X_eval.shape[0], X_train.shape[1]
    k = out["inter_list"].shape[0]
    expect = {"main_train": (n_tr, p), "inter_train": (n_tr, k), "main_eval": (n_ev, p), "inter_eval": (n_ev, k)}
    for key, shape in expect.items():
        if out[key].shape != shape or not np.all(np.isfinite(out[key])):
            raise ValueError(f"{key} 形状应为 {shape}，实际 {out[key].shape}，或含非有限值")
    return out


def recon(out, split):
    return out["intercept"] + out[f"main_{split}"].sum(1) + out[f"inter_{split}"].sum(1)


def resid_ratio(pred, out, split):
    return float(np.mean((pred - recon(out, split)) ** 2) / np.var(pred))


def orthogonality(pred, main, inter, inter_list):
    """论文 B.3.1：方差≥1% V[T(X)] 的交互项与其对应主效应的最大 |相关系数|；无符合条件的交互项返回 None。"""
    v = np.var(pred)
    worst = None
    for c, (j, k) in enumerate(inter_list):
        if np.var(inter[:, c]) < 0.01 * v:
            continue
        for m in (j, k):
            if np.var(main[:, m]) > 0:
                r = abs(np.corrcoef(inter[:, c], main[:, m])[0, 1])
                worst = r if worst is None else max(worst, r)
    return None if worst is None else float(worst)


def local_variability(X, main, k=10):
    """论文 B.3.1：按单个变量取 10 近邻，分量在近邻内的方差取平均，再除以该分量的总方差，对主效应取平均。"""
    vals = []
    for j in range(X.shape[1]):
        tot = np.var(main[:, j])
        if tot <= 0:
            continue
        nn = NearestNeighbors(n_neighbors=min(k, len(X))).fit(X[:, [j]])
        idx = nn.kneighbors(X[:, [j]], return_distance=False)
        vals.append(np.mean(np.var(main[idx, j], axis=1)) / tot)
    return float(np.mean(vals)) if vals else None


def eta_main(x):
    return RHO / (1 + RHO ** 2) * (x ** 2 - 1)


def eta_order2(x, z):
    return RHO * (1 - RHO ** 2) / (1 + RHO ** 2) - RHO / (1 + RHO ** 2) * (x ** 2 + z ** 2) + x * z


def analytical(method, reps, timeout):
    mu, cov = np.zeros(DIM), np.full((DIM, DIM), RHO)
    np.fill_diagonal(cov, 1.0)
    rows = []
    for rep in range(reps):
        rng_tr, rng_te = np.random.default_rng(1000 + rep), np.random.default_rng(2000 + rep)
        X = rng_tr.multivariate_normal(mu, cov, size=NSAMPLE)
        y = np.sin(2 * np.pi * X[:, 0]) + X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] + rng_tr.normal(0, 0.5, NSAMPLE)
        Xn = rng_te.multivariate_normal(mu, cov, size=NSAMPLE)
        model = xgb.XGBRegressor(random_state=rep, **XGB_PARAMS).fit(X, y)
        out = run_method(method, model, X, Xn, timeout)
        main, inter, il = out["main_eval"], out["inter_eval"], [tuple(p) for p in out["inter_list"]]
        target = np.zeros((NSAMPLE, DIM))
        target[:, 0] = np.sin(2 * np.pi * Xn[:, 0]) + eta_main(Xn[:, 0])
        for j in (1, 2, 3):
            target[:, j] = eta_main(Xn[:, j])
        row = {f"mse_eta{j + 1}": float(np.mean((target[:, j] - main[:, j]) ** 2)) for j in range(DIM)}
        for name, pair, tgt in (("mse_eta12", (0, 1), eta_order2(Xn[:, 0], Xn[:, 1])),
                                ("mse_eta34", (2, 3), eta_order2(Xn[:, 2], Xn[:, 3]))):
            est = inter[:, il.index(pair)] if pair in il else np.zeros(NSAMPLE)  # 缺失的交互项按 0 估计计分
            row[name] = float(np.mean((tgt - est) ** 2))
        others = [c for c, p in enumerate(il) if p not in ((0, 1), (2, 3))]
        row["mse_others"] = float(np.mean(inter[:, others] ** 2)) if others else 0.0
        row["resid_out"] = resid_ratio(model.predict(Xn), out, "eval")
        row["fit_s"] = out["timing"]["fit_s"]
        rows.append(row)
        print(f"  analytical rep {rep + 1}/{reps}: eta12={row['mse_eta12']:.4f} eta1={row['mse_eta1']:.4f}", flush=True)
    keys = rows[0].keys()
    return {"mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
            "std": {k: float(np.std([r[k] for r in rows])) for k in keys}, "reps": reps}


def real_data(method, name, timeout):
    d = np.load(DATA / f"{name}.npz")
    X, y = d["X"], d["y"]
    perm = np.random.default_rng(0).permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = perm[:cut], perm[cut:]
    model = xgb.XGBRegressor(random_state=0, **XGB_PARAMS).fit(X[tr], y[tr])
    out = run_method(method, model, X[tr], X[te], timeout)
    p_tr, p_te = model.predict(X[tr]), model.predict(X[te])
    il = out["inter_list"]
    res = {
        "n": int(len(y)), "p": int(X.shape[1]),
        "xgb_r2_test": float(1 - np.mean((y[te] - p_te) ** 2) / np.var(y[te])),
        "resid_in": resid_ratio(p_tr, out, "train"),
        "resid_out": resid_ratio(p_te, out, "eval"),
        "ortho_in": orthogonality(p_tr, out["main_train"], out["inter_train"], il),
        "ortho_out": orthogonality(p_te, out["main_eval"], out["inter_eval"], il),
        "locvar_in": local_variability(X[tr], out["main_train"]),
        "fit_s": out["timing"]["fit_s"],
    }
    print(f"  {name}: resid_in={res['resid_in']:.4f} resid_out={res['resid_out']:.4f} "
          f"ortho_in={res['ortho_in']} locvar={res['locvar_in']:.4g} fit={res['fit_s']:.1f}s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--mode", choices=["subset", "full"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=3600, help="单次方法调用的超时秒数")
    a = ap.parse_args()
    method = Path(a.method).resolve()
    t0 = time.time()
    result = {"method": str(method), "method_sha256": sha256(method), "mode": a.mode,
              "protocol": protocol_manifest(), "errors": {}}
    print(f"[harness] {a.mode} 模式评测 {method}", flush=True)
    try:
        result["analytical"] = analytical(method, 3 if a.mode == "subset" else 10, a.timeout)
    except Exception as e:  # noqa: BLE001 — 失败要记录下来交给审查员，而不是中断
        result["errors"]["analytical"] = repr(e)[:3000]
    result["real"] = {}
    for name in (SUBSET_DATA if a.mode == "subset" else FULL_DATA):
        try:
            result["real"][name] = real_data(method, name, a.timeout)
        except Exception as e:  # noqa: BLE001
            result["errors"][name] = repr(e)[:3000]
    result["wall_s"] = time.time() - t0
    Path(a.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"[harness] 完成，用时 {result['wall_s']:.0f}s，错误 {list(result['errors'])}", flush=True)


if __name__ == "__main__":
    main()
