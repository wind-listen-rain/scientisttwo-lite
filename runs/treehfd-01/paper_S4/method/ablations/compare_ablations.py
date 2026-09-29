"""Side-by-side table: baseline, full method and ablation result JSONs (same mode and env).

usage: compare_ablations.py subset|full [label=path ...]
Defaults: the baseline and method.py results of this machine (env-22b414443084), plus every
ablations/<mode>_<name>.json. Values are printed with the ratio to the full method in brackets.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ENV = "env-22b414443084"
REAL_KEYS = ("resid_in", "resid_out", "ortho_in", "ortho_out", "locvar_in", "fit_s")


def main():
    mode = sys.argv[1]
    runs = {"baseline": ROOT / f"baseline/{ENV}/{mode}.json",
            "method": HERE.parent / f"{mode}_{ENV}.json"}
    extra = dict(a.split("=", 1) for a in sys.argv[2:])
    if extra:
        runs.update({k: Path(v) for k, v in extra.items()})
    else:
        for f in sorted(HERE.glob(f"{mode}_*.json")):
            runs[f.stem[len(mode) + 1:]] = f
    res = {k: json.loads(p.read_text()) for k, p in runs.items() if p.exists()}
    ref = res["method"]
    labels = list(res)
    for k in labels:
        if res[k].get("errors"):
            print(f"!! {k} errors: {res[k]['errors']}")
    print("columns: " + " | ".join(f"[{i}] {k}" for i, k in enumerate(labels)))

    def row(name, vals, r):
        cells = []
        for v in vals:
            if v is None:
                cells.append(f"{'null':>18s}")
            elif r:
                cells.append(f"{v:10.5f} ({v / r:5.2f})")
            else:
                cells.append(f"{v:10.5f}       ")
        print(f"  {name:11s}" + " ".join(cells))

    if "analytical" in ref:
        print("--- analytical (mean over reps)")
        for m in ref["analytical"]["mean"]:
            row(m, [res[k].get("analytical", {}).get("mean", {}).get(m) for k in labels],
                ref["analytical"]["mean"][m])
    for d in ref["real"]:
        print(f"--- {d}")
        for m in REAL_KEYS:
            row(m, [res[k]["real"].get(d, {}).get(m) for k in labels], ref["real"][d][m])


if __name__ == "__main__":
    main()
