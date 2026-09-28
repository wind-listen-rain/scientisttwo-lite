"""Dev: print harness results (json) next to the baseline results of the same mode, with relative changes."""
import json
import sys

base = json.load(open(sys.argv[1]))
new = json.load(open(sys.argv[2]))
print("errors:", new["errors"])
for k, v in new["analytical"]["mean"].items():
    b = base["analytical"]["mean"][k]
    print(f"analytical.{k}: {b:.4g} -> {v:.4g} ({(v / b - 1) * 100:+.1f}%)  sd {new['analytical']['std'][k]:.2g}")
for name, r in new["real"].items():
    for k in ("resid_in", "resid_out", "ortho_in", "ortho_out", "locvar_in", "fit_s"):
        b, v = base["real"][name].get(k), r.get(k)
        rel = "" if b is None or v is None else f"({(v / b - 1) * 100:+.1f}%)"
        fmt = lambda x: "None" if x is None else f"{x:.4g}"  # noqa: E731
        print(f"{name}.{k}: {fmt(b)} -> {fmt(v)} {rel}")
