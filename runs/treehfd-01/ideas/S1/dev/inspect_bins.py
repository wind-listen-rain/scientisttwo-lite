import pickle, sys
import numpy as np
for name in sys.argv[1:]:
    d = pickle.load(open(f"cache/{name}.pkl", "rb"))
    X, t = d["Xtr"], d["table"]
    p = X.shape[1]
    out = []
    for j in range(p):
        s = np.unique(t.loc[t.Feature == f"f{j}", "Split"].to_numpy(float))
        b = np.digitize(X[:, j], s)
        cnt = np.bincount(b, minlength=len(s) + 1)
        out.append((len(s), int((cnt > 0).sum()), int((cnt >= 10).sum()), len(np.unique(X[:, j]))))
    v = np.var(d["p_tr"])
    big = [(tuple(d["il"][c]), round(np.var(d["i_tr"][:, c]) / v, 4)) for c in range(d["il"].shape[0]) if np.var(d["i_tr"][:, c]) >= 0.01 * v]
    print(name, "n", X.shape[0], "(n_thr, nonempty, >=10, n_unique):", out)
    print("   big pairs", big)
