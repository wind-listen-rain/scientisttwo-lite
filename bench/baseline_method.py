"""基线方法：原版 TreeHFD（treehfd 1.4.1，提交 5ec6cca）。

被测方法须实现同样的两个函数：
  fit(model, X_train, interaction_order=2) -> state      # 只能用模型和训练输入
  predict(state, X) -> {"intercept": float, "main": (n, p), "inter": (n, K), "inter_list": (K, 2)}
"""
import numpy as np

from treehfd import XGBTreeHFD


def fit(model, X_train, interaction_order=2):
    hfd = XGBTreeHFD(model)
    hfd.fit(X_train, interaction_order=interaction_order, verbose=False)
    return hfd


def predict(state, X):
    main, inter = state.predict(X, verbose=False)
    inter_list = state.interaction_list if inter.shape[1] else np.empty((0, 2), int)
    return {"intercept": state.eta0, "main": main, "inter": inter, "inter_list": inter_list}
