"""下载 TreeHFD 论文 Table 2 用到的 9 个 UCI 数据集，存为 bench/data/<name>.npz（X, y, feature_names）。

论文没有给出每个数据集的特征/目标选择，这里按论文 Table 2 的 n、p 反推，
偏差记在 bench/data/SOURCES.md 里。只需运行一次。
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from ucimlrepo import fetch_ucirepo

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)
notes = ["# 数据来源与处理\n", "| 数据集 | 来源 | n | p | 目标 | 说明 |", "|---|---|---|---|---|---|"]


def save(name, X: pd.DataFrame, y, src, note=""):
    X = X.apply(pd.to_numeric, errors="coerce")
    mask = X.notna().all(axis=1) & pd.notna(y)
    X, y = X[mask], np.asarray(y, dtype=float)[mask.to_numpy()]
    np.savez(OUT / f"{name}.npz", X=X.to_numpy(float), y=y, feature_names=np.array(X.columns, dtype=str))
    notes.append(f"| {name} | {src} | {len(y)} | {X.shape[1]} | {getattr(y, 'name', 'y')} | {note} |")
    print(f"{name}: n={len(y)} p={X.shape[1]}")


def uci(i):
    d = fetch_ucirepo(id=i)
    return d.data.features.copy(), d.data.targets.copy()


X, y = uci(1)  # Abalone：性别 one-hot 去掉一列，p=9
X = pd.get_dummies(X, columns=["Sex"], drop_first=True, dtype=float)
save("abalone", X, y["Rings"], "UCI id=1", "Sex one-hot（drop_first，p=9 与论文一致）")

X, y = uci(291)  # Airfoil Self-Noise
save("airfoil", X, y.iloc[:, 0], "UCI id=291")

X, y = uci(275)  # Bike Sharing（小时级）；论文 p=8，取 8 个非冗余特征
cols = ["hr", "weekday", "workingday", "season", "weathersit", "temp", "hum", "windspeed"]
d = fetch_ucirepo(id=275)
df = d.data.original
save("bike", df[cols], df["cnt"], "UCI id=275 hour", "论文 p=8 未列特征，选 hr/weekday/workingday/season/weathersit/temp/hum/windspeed")

h = fetch_california_housing(as_frame=True)
save("housing", h.data, h.target, "sklearn California Housing")

X, y = uci(165)  # Concrete
save("concrete", X, y.iloc[:, 0], "UCI id=165")

d = fetch_ucirepo(id=887)  # NHANES 年龄预测子集：n=2278, p=7，推断为论文的 "Nutrition"
df = d.data.original
feat = ["RIAGENDR", "PAQ605", "BMXBMI", "LBXGLU", "DIQ010", "LBXGLT", "LBXIN"]
save("nutrition", df[feat], df["RIDAGEYR"], "UCI id=887 NHANES", "推断：n、p 与论文一致，目标取年龄 RIDAGEYR")

d = fetch_ucirepo(id=189)  # Parkinsons Telemonitoring：去掉 subject# 与两个 UPDRS 目标，p=19
df = d.data.original
feat = [c for c in df.columns if c not in ("subject#", "motor_UPDRS", "total_UPDRS")]
save("parkinson", df[feat], df["total_UPDRS"], "UCI id=189", "目标 total_UPDRS")

X, y = uci(294)  # Combined Cycle Power Plant
save("powerplant", X, y.iloc[:, 0], "UCI id=294")

X, y = uci(464)  # Superconductivity
save("superconduct", X, y.iloc[:, 0], "UCI id=464")

(OUT / "SOURCES.md").write_text("\n".join(notes) + "\n")
