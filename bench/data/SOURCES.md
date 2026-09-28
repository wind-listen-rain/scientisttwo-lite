# 数据来源、处理与许可

9 个数据集对应 TreeHFD 论文（Bénard, NeurIPS 2025）Table 2。论文没有写明每个数据集的特征和目标，
这里按论文给出的 n、p 反推，由 `bench/fetch_data.py` 生成，存为 `<名字>.npz`（包含 X、y、feature_names）。
仓库里直接附带生成好的 npz 文件：评测完整性校验会核对它们的 SHA-256，重新下载可能因为上游数据更新而对不上。

| 文件 | 数据集 | 来源 | n | p | 目标 | 处理 | 许可 |
|---|---|---|---|---|---|---|---|
| abalone.npz | Abalone | UCI id=1, doi:10.24432/C55C7W（Nash et al.） | 4177 | 9 | Rings | Sex one-hot，去掉一列 | CC BY 4.0 |
| airfoil.npz | Airfoil Self-Noise | UCI id=291, doi:10.24432/C5VW2C（Brooks et al.） | 1503 | 5 | scaled sound pressure | 无 | CC BY 4.0 |
| bike.npz | Bike Sharing（小时级） | UCI id=275, doi:10.24432/C5W894（Fanaee-T） | 17379 | 8 | cnt | 论文 p=8 但没列特征，这里取 hr/weekday/workingday/season/weathersit/temp/hum/windspeed | CC BY 4.0 |
| housing.npz | California Housing | scikit-learn `fetch_california_housing`（Pace & Barry 1997，1990 年美国人口普查） | 20640 | 8 | MedHouseVal | 无 | 公开数据 |
| concrete.npz | Concrete Compressive Strength | UCI id=165, doi:10.24432/C5PK67（Yeh） | 1030 | 8 | 抗压强度 | 无 | CC BY 4.0 |
| nutrition.npz | NHANES 2013-2014 年龄预测子集（推断为论文里的 "Nutrition"） | UCI id=887, doi:10.24432/C5BS66 | 2278 | 7 | RIDAGEYR | 推断：n、p 与论文一致 | CC BY 4.0 |
| parkinson.npz | Parkinsons Telemonitoring | UCI id=189, doi:10.24432/C5ZS3N（Tsanas & Little） | 5875 | 19 | total_UPDRS | 去掉 subject# 与两个 UPDRS 列 | CC BY 4.0 |
| powerplant.npz | Combined Cycle Power Plant | UCI id=294, doi:10.24432/C5002N（Tüfekci & Kaya） | 9568 | 4 | PE | 无 | CC BY 4.0 |
| superconduct.npz | Superconductivity | UCI id=464, doi:10.24432/C53P47（Hamidieh） | 21263 | 81 | critical_temp | 无 | CC BY 4.0 |

和论文 Table 2 的差异：Abalone 是 4177 行（论文写 4176），Bike 是 17379 行（论文写 17389）。
