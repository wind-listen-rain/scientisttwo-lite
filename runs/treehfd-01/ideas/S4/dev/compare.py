"""Print a side-by-side comparison of a result JSON against the baseline JSON."""
import json
import sys

base = json.load(open(sys.argv[2] if len(sys.argv) > 2 else
                      str(__import__("pathlib").Path(__file__).resolve().parents[5]) + "/baseline/subset.json"))
new = json.load(open(sys.argv[1]))
print("errors:", new["errors"])
if "analytical" in new:
    print(f"{'analytical':12s} {'baseline':>10s} {'new':>10s} {'ratio':>7s}")
    for k, v in base["analytical"]["mean"].items():
        w = new["analytical"]["mean"][k]
        print(f"{k:12s} {v:10.5f} {w:10.5f} {w / v:7.3f}")
for d, r in new["real"].items():
    b = base["real"].get(d)
    if b is None:
        continue
    print(f"--- {d}")
    for k in ("resid_in", "resid_out", "ortho_in", "ortho_out", "locvar_in", "fit_s"):
        v, w = b[k], r[k]
        if v is None or w is None:
            print(f"{k:12s} {v} {w}")
        else:
            print(f"{k:12s} {v:10.5f} {w:10.5f} {w / v:7.3f}")
