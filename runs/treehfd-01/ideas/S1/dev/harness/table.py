"""Dev only: side-by-side table of harness result files (subset mode). First file = reference for the % columns.
usage: python dev/harness/table.py label=file [label=file ...]"""
import json
import sys

runs = [(a.split("=", 1)[0], json.load(open(a.split("=", 1)[1]))) for a in sys.argv[1:]]
ref = runs[0][1]
print("| metric | " + " | ".join(l for l, _ in runs) + " |")
print("|---|" + "---|" * len(runs))


def cell(v, r):
    if v is None:
        return "null"
    s = f"{v:.4g}"
    if r not in (None, 0) and v is not ref:
        s += f" ({100 * (v / r - 1):+.1f}%)"
    return s


keys = list(ref["analytical"]["mean"].keys())
for k in keys:
    r = ref["analytical"]["mean"][k]
    row = [f"{r:.4g}"] + [cell(res["analytical"]["mean"][k], r) for _, res in runs[1:]]
    print(f"| analytical.{k} | " + " | ".join(row) + " |")
for ds in ref["real"]:
    for k in ("resid_in", "resid_out", "ortho_in", "ortho_out", "locvar_in", "fit_s"):
        r = ref["real"][ds][k]
        row = ["null" if r is None else f"{r:.4g}"] + [cell(res["real"][ds][k], r) for _, res in runs[1:]]
        print(f"| {ds}.{k} | " + " | ".join(row) + " |")
