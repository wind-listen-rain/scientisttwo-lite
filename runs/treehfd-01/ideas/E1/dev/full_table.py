"""Markdown table of a full-mode result: baseline, S4 and E1 on this machine (ratios to baseline).

usage: full_table.py <E1 full json>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
base = json.load(open(ROOT / "baseline/env-22b414443084/full.json"))
s4 = json.load(open(ROOT / "runs/treehfd-01/ideas/S4/full_env-22b414443084.json"))
e1 = json.load(open(sys.argv[1]))
if e1.get("errors"):
    print("ERRORS:", e1["errors"])


def r(v, b):
    if v is None or b is None:
        return "n/a"
    return f"{v / b:.3f}"


print("| dataset | resid_in S4 / E1 | resid_out S4 / E1 | ortho_in S4 / E1 | ortho_out S4 / E1 | "
      "locvar_in S4 / E1 | fit_s base → E1 (S4, E1 ratio) |")
print("|---|---|---|---|---|---|---|")
for ds, b in base["real"].items():
    a, e = s4["real"][ds], e1["real"].get(ds)
    if e is None:
        print(f"| {ds} | missing |||||| ")
        continue
    cells = [f"{r(a[k], b[k])} / **{r(e[k], b[k])}**" for k in
             ("resid_in", "resid_out", "ortho_in", "ortho_out", "locvar_in")]
    print(f"| {ds} | " + " | ".join(cells) +
          f" | {b['fit_s']:.1f} → {e['fit_s']:.1f} ({a['fit_s'] / b['fit_s']:.2f}, "
          f"**{e['fit_s'] / b['fit_s']:.2f}**) |")
print()
ba, aa, ea = (x["analytical"]["mean"] for x in (base, s4, e1))
print("| analytical (10 reps) | baseline | S4 | E1 |")
print("|---|---|---|---|")
for k in ba:
    print(f"| {k} | {ba[k]:.5g} | {aa[k]:.5g} ({aa[k] / ba[k]:.3f}) | {ea[k]:.5g} "
          f"(**{ea[k] / ba[k]:.3f}**) |")
