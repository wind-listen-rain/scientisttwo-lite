"""Side-by-side table of full-mode result files against this machine's baseline (development).

usage: compare_full.py label=path [label=path ...] [--md]
Every value is printed with its ratio to the baseline (env-22b414443084/full.json).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "baseline" / "env-22b414443084" / "full.json"
REAL = ("resid_in", "resid_out", "ortho_in", "ortho_out", "locvar_in", "fit_s")


def rows(res):
    out = {}
    for k, v in res.get("analytical", {}).get("mean", {}).items():
        out[f"analytical.{k}"] = v
    for d, r in res.get("real", {}).items():
        for k in REAL:
            out[f"{d}.{k}"] = r.get(k)
    return out


def main():
    md = "--md" in sys.argv
    runs = [a.split("=", 1) for a in sys.argv[1:] if "=" in a]
    base = rows(json.loads(BASE.read_text()))
    res = {lab: json.loads(Path(p).read_text()) for lab, p in runs}
    for lab, r in res.items():
        if r.get("errors"):
            print(f"# {lab}: errors {r['errors']}")
    tabs = {lab: rows(r) for lab, r in res.items()}
    head = ["metric", "baseline"] + [lab for lab, _ in runs]
    if md:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
    else:
        print(f"{'metric':26s} {'baseline':>10s} " + " ".join(f"{h:>20s}" for h in head[2:]))
    for key, b in base.items():
        cells = []
        for lab, _ in runs:
            v = tabs[lab].get(key)
            if v is None or b is None:
                cells.append("NA" if v is None else f"{v:.4g}")
            else:
                cells.append(f"{v:.4g} ({(v / b - 1) * 100:+.1f}%)")
        bs = "NA" if b is None else f"{b:.4g}"
        if md:
            print(f"| {key} | {bs} | " + " | ".join(cells) + " |")
        else:
            print(f"{key:26s} {bs:>10s} " + " ".join(f"{c:>20s}" for c in cells))


if __name__ == "__main__":
    main()
