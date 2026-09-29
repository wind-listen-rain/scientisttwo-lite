"""Dev only: side-by-side table of a result file vs the baseline of this machine."""
import json, sys
new = json.load(open(sys.argv[1]))
base = json.load(open(sys.argv[2] if len(sys.argv) > 2 else str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/baseline/env-22b414443084/subset.json"))
print("errors:", new["errors"], "| eval_env:", new.get("eval_env"))
if "analytical" in new:
    for k, v in new["analytical"]["mean"].items():
        b = base["analytical"]["mean"][k]
        print(f"analytical {k:11s} base {b:.5f}  S1 {v:.5f}  ratio {v / b:.3f}")
for ds, r in new["real"].items():
    b = base["real"].get(ds, {})
    cells = []
    for k in ("resid_in", "resid_out", "ortho_in", "ortho_out", "locvar_in", "fit_s"):
        x, y = b.get(k), r.get(k)
        cells.append(f"{k} {'-' if x is None else f'{x:.4g}'}->{'-' if y is None else f'{y:.4g}'}")
    print(f"{ds:12s} " + " | ".join(cells))
