"""Dev only (diagnosis, label-free): in-sample step-2 leak coefficients vs the same coefficients on the eval inputs,
per pair; which pair sets ortho_out. Uses cached baseline components."""
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import s1_projection as S  # noqa: E402


def coefs(main, inter, il, cols):
    Mc = main - main.mean(0)
    out = np.zeros((len(il), 2))
    for c in cols:
        j, k = il[c]
        Z = Mc[:, [j, k]]
        ok = Z.var(0) > 0
        y = inter[:, c] - inter[:, c].mean()
        b = np.zeros(2)
        b[ok] = np.linalg.lstsq(Z[:, ok], y, rcond=None)[0]
        out[c] = b
    return out


def corrs(pred, main, inter, il):
    v = np.var(pred)
    rows = []
    for c, (j, k) in enumerate(il):
        big = np.var(inter[:, c]) >= 0.01 * v
        for m in (j, k):
            r = abs(np.corrcoef(inter[:, c], main[:, m])[0, 1]) if np.var(main[:, m]) > 0 else 0.0
            rows.append((c, j, k, m, big, r))
    return rows


if __name__ == "__main__":
  for name in sys.argv[1:]:
      d = pickle.load(open(HERE / "cache" / f"{name}.pkl", "rb"))
      il = np.asarray(d["il"]).reshape(-1, 2)
      cols = [c for c in range(len(il)) if np.var(d["i_tr"][:, c]) > 0]
      a_in = coefs(d["m_tr"], d["i_tr"], il, cols)
      a_te = coefs(d["m_te"], d["i_te"], il, cols)
      vt = np.var(d["p_tr"])
      print(f"== {name}: n_tr {len(d['Xtr'])}  n_te {len(d['Xte'])}  pairs {len(il)}")
      print("   main var / VarT:", np.round(np.var(d["m_tr"], 0) / vt, 4))
      s_in = np.ones(il.max() + 1)
      s_te = np.ones(il.max() + 1)
      for c in cols:
          j, k = il[c]
          s_in[j] += a_in[c, 0]; s_in[k] += a_in[c, 1]
          s_te[j] += a_te[c, 0]; s_te[k] += a_te[c, 1]
      print("   scale in-sample:", np.round(s_in, 3))
      print("   scale eval     :", np.round(s_te, 3))
      x, y = a_in[cols].ravel(), a_te[cols].ravel()
      print(f"   coef corr(in, eval) {np.corrcoef(x, y)[0, 1]:.3f}  slope eval~in {np.dot(x, y) / np.dot(x, x):.3f}")
      proj, e0, dg = S.fit_projection(d["eta0"], d["m_tr"], d["i_tr"], d["il"], d["Xtr"], d["table"])
      m_te, i_te = S.apply_projection(proj, d["m_te"], d["i_te"], d["Xte"])
      rb = corrs(d["p_te"], d["m_te"], d["i_te"], il)
      rn = corrs(d["p_te"], m_te, i_te, il)
      top = sorted(range(len(rn)), key=lambda t: -rn[t][5] * rn[t][4])[:5]
      print("   top eval corrs after S1 (pair, main, big, base->S1, a_in, a_te, inter var/VarT):")
      for t in top:
          c, j, k, m, big, r = rn[t]
          side = 0 if m == j else 1
          print(f"     ({j},{k}) m={m} big={big} {rb[t][5]:.4f}->{r:.4f}  a_in {a_in[c, side]:+.4f} a_te {a_te[c, side]:+.4f}"
                f"  var {np.var(d['i_tr'][:, c]) / vt:.4f}")
