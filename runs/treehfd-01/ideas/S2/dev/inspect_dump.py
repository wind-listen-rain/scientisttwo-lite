import ast
import json
import sys

import numpy as np
import xgboost as xgb

from setup import real

name = sys.argv[1] if len(sys.argv) > 1 else "concrete"
m, X, Xe = real(name)
b = m.get_booster()
df = b.trees_to_dataframe()
print(df.head(8).to_string())
cfg = json.loads(b.save_config())
print(cfg["learner"]["learner_model_param"])
ok = all((g["Node"].values == np.arange(len(g))).all() for _, g in df.groupby("Tree"))
print("contiguous nodes", ok)
leaf = b.predict(xgb.DMatrix(X), pred_leaf=True)
lv = df[df.Feature == "Leaf"]
mp = {(t, n): v for t, n, v in zip(lv.Tree, lv.Node, lv.Gain)}
s = np.array([[mp[(t, int(leaf[i, t]))] for t in range(leaf.shape[1])] for i in range(len(X))]).sum(1)
base = float(np.array(ast.literal_eval(cfg["learner"]["learner_model_param"]["base_score"])).ravel()[0])
print("base", base, "max diff", np.abs(s + base - m.predict(X, output_margin=True)).max())
sp = df[df.Feature != "Leaf"]
for f, g in sp.groupby("Feature"):
    print(f, len(g), g.Split.nunique())
print("float32 exact splits:", np.all(np.float32(sp.Split.values).astype(float) == sp.Split.values))
