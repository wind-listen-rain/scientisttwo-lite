"""在独立子进程里运行被测方法：只读入模型与输入矩阵，写回分解结果。

用法（由 harness.py 调用）: python runner.py <method.py> <job_dir>
job_dir 内: model.json（xgboost 模型）、X_train.npy、X_eval.npy、config.json
输出:      out.npz（intercept, main_train, inter_train, main_eval, inter_eval, inter_list）、timing.json
真实分量与标签从不写入 job_dir。
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb


def load_method(path):
    spec = importlib.util.spec_from_file_location("method_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    method_path, job = sys.argv[1], Path(sys.argv[2])
    cfg = json.loads((job / "config.json").read_text())
    model = xgb.XGBRegressor()
    model.load_model(job / "model.json")
    X_train = np.load(job / "X_train.npy")
    X_eval = np.load(job / "X_eval.npy")
    method = load_method(method_path)

    t0 = time.time()
    state = method.fit(model, X_train, interaction_order=cfg.get("interaction_order", 2))
    t1 = time.time()
    tr = method.predict(state, X_train)
    ev = method.predict(state, X_eval)
    t2 = time.time()

    np.savez(job / "out.npz",
             intercept=np.float64(ev["intercept"]),
             main_train=np.asarray(tr["main"], float), inter_train=np.asarray(tr["inter"], float),
             main_eval=np.asarray(ev["main"], float), inter_eval=np.asarray(ev["inter"], float),
             inter_list=np.asarray(ev["inter_list"], int).reshape(-1, 2))
    (job / "timing.json").write_text(json.dumps({"fit_s": t1 - t0, "predict_s": t2 - t1}))


if __name__ == "__main__":
    main()
