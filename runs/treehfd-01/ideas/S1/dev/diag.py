import pickle, sys
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import s1_projection as S
from dev_eval import ortho

def lv(X, m, j):
    tot = np.var(m[:, j])
    if tot <= 0: return None
    idx = NearestNeighbors(n_neighbors=10).fit(X[:, [j]]).kneighbors(X[:, [j]], return_distance=False)
    return np.mean(np.var(m[idx, j], axis=1)) / tot

name = sys.argv[1]
d = pickle.load(open(HERE / "cache" / f"{name}.pkl", "rb"))
proj, e0 = S.fit_projection(d["eta0"], d["m_tr"], d["i_tr"], d["il"], d["Xtr"], d["table"])
m_tr, i_tr = S.apply_projection(proj, d["m_tr"], d["i_tr"], d["Xtr"])
m_te, i_te = S.apply_projection(proj, d["m_te"], d["i_te"], d["Xte"])
X = d["Xtr"]
print("nbins", [e.size + 1 for e in proj["edges"]])
for j in range(X.shape[1]):
    print(f"var {j}: sd main {np.std(d['m_tr'][:, j]):.4g}->{np.std(m_tr[:, j]):.4g} locvar {lv(X, d['m_tr'], j)} -> {lv(X, m_tr, j)}")
v = np.var(d["p_tr"]); vt = np.var(d["p_te"])
for pr in proj["pairs"]:
    c, j, k = pr["col"], pr["j"], pr["k"]
    if np.var(d["i_tr"][:, c]) < 0.005 * v: continue
    def cc(a, b): return abs(np.corrcoef(a, b)[0, 1]) if np.std(b) > 0 else 0
    print(f"pair ({j},{k}) tau {pr['tau']} var/V {np.var(d['i_tr'][:, c])/v:.3f}->{np.var(i_tr[:, c])/v:.3f} "
          f"in: {cc(d['i_tr'][:,c], d['m_tr'][:,j]):.3f},{cc(d['i_tr'][:,c], d['m_tr'][:,k]):.3f} -> {cc(i_tr[:,c], m_tr[:,j]):.3f},{cc(i_tr[:,c], m_tr[:,k]):.3f} "
          f"out: {cc(d['i_te'][:,c], d['m_te'][:,j]):.3f},{cc(d['i_te'][:,c], d['m_te'][:,k]):.3f} -> {cc(i_te[:,c], m_te[:,j]):.3f},{cc(i_te[:,c], m_te[:,k]):.3f}"
          f" cv/min {np.round(np.array(pr['cv'])/min(pr['cv']),4)}")
