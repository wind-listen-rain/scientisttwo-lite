"""Round 3: compact 8-split ablation table (geo-mean over splits) from dev/cache.

usage: summary_r3.py > dev/logs/summary_r3.md
Rows: grid G1 = rho in {0, 0.25, 1} (rounds 1-2), G2 = rho in {0, 0.25} (round 3); rules as in
rules_r3.py. Columns: ratio to S4 (all metrics) and ratio to the baseline (resid_in, resid_out).
Diagnosis only (held-out numbers never enter the method).
"""
import numpy as np

import rules_r3 as R

ROWS = [("S4", "r0_0.25", "S4"), ("E1 round 1: G1 argmin", "r0_0.25_1", "argmin"),
        ("E1 round 2: G1 one-SE", "r0_0.25_1", "se1"), ("G1 + fidelity cap", "r0_0.25_1", "cons"),
        ("G2 argmin", "r0_0.25", "argmin"), ("G2 one-SE", "r0_0.25", "se1"),
        ("**G2 + fidelity cap (round 3)**", "r0_0.25", "cons"),
        ("G2 + fidelity cap + one-SE", "r0_0.25", "cons_se1")]
for ds in ("airfoil", "concrete", "abalone", "nutrition"):
    base = R.baseline(ds)
    z = {t: {s: dict(np.load(R.DEV / "cache" / f"{ds}_s{s}_{t}.npz")) for s in range(8)}
         for t in ("r0_0.25", "r0_0.25_1")}
    names = list(z["r0_0.25"][0]["names"])
    print(f"\n#### {ds} (8 splits; ratio to S4, then resid_in / resid_out ratio to baseline)\n")
    print("| selection | " + " | ".join(names) + " | resid_in / base | resid_out / base |")
    print("|---" * (len(names) + 3) + "|")
    for lab, t, rule in ROWS:
        cells = []
        for mi, m in enumerate(names):
            r = []
            for s in range(8):
                zz = z[t][s]
                i, i4 = R.RULES[rule](zz), R.rule_s4(zz)
                r.append(zz["met"][i, mi] / zz["met"][i4, mi])
            cells.append(f"{R.gm(r)[0]:.3f}")
        for m in ("resid_in", "resid_out"):
            mi = names.index(m)
            r = [z[t][s]["met"][R.RULES[rule](z[t][s]), mi] / base[s][m] for s in range(8)]
            cells.append(f"{R.gm(r)[0]:.3f}")
        print(f"| {lab} | " + " | ".join(cells) + " |")
print("\n#### analytical (reps 0-2, mean over reps; ratio to S4)\n")
za = {t: {s: dict(np.load(R.DEV / "cache" / f"analytical_s{s}_{t}.npz")) for s in range(3)}
      for t in ("r0_0.25", "r0_0.25_1")}
names = list(za["r0_0.25"][0]["names"])
print("| selection | " + " | ".join(names) + " |\n" + "|---" * (len(names) + 1) + "|")
for lab, t, rule in ROWS:
    v = np.mean([za[t][s]["met"][R.RULES[rule](za[t][s])] for s in range(3)], axis=0)
    v4 = np.mean([za[t][s]["met"][R.rule_s4(za[t][s])] for s in range(3)], axis=0)
    print(f"| {lab} | " + " | ".join(f"{a / b:.3f}" for a, b in zip(v, v4, strict=True)) + " |")
