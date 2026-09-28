"""Compare harness result files metric by metric.

usage: compare.py <a.json> <b.json> [c.json ...]
Prints every metric of every file and the ratio of each file to the first one.
"""
import json
import sys

files = sys.argv[1:]
res = [json.load(open(f)) for f in files]
print("files:", *[f"[{k}] {f}" for k, f in enumerate(files)], sep="\n  ")
for r, f in zip(res, files, strict=True):
    if r.get("errors"):
        print("ERRORS in", f, r["errors"])
blocks = [("analytical", lambda r: r.get("analytical", {}).get("mean", {}))]
for name in res[0].get("real", {}):
    blocks.append((name, lambda r, name=name: r.get("real", {}).get(name, {})))
for title, get in blocks:
    print(f"== {title}")
    base = get(res[0])
    for k, v0 in base.items():
        if k in ("n", "p"):
            continue
        vals = [get(r).get(k) for r in res]
        cells = []
        for v in vals:
            if v is None or v0 is None:
                cells.append(f"{str(v):>12s}        ")
            else:
                ratio = v / v0 if v0 else float("nan")
                cells.append(f"{v:12.6g} ({ratio:5.3f})")
        print(f"  {k:12s}", " ".join(cells))
