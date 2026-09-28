"""Summarise check (7d): dev/logs/ablate_rho_<ds>.log against the baseline per split.

usage: summarize_ablation.py <dataset> [...]
Baseline per-split numbers are read from S4's dev/logs/diag_ortho_out_<ds>.log (the real-data
models and resid/ortho metrics are deterministic across machines; locvar is not reported
there). Prints per split: selection, then each rho mode's R-best candidate, with ratios to the
baseline, and geometric means over splits. Diagnosis only (held-out numbers).
"""
import re
import sys
from pathlib import Path

import numpy as np

DEV = Path(__file__).resolve().parent
S4LOG = DEV.parents[1] / "S4" / "dev" / "logs"
MET = ("resid_in", "resid_out", "ortho_in", "ortho_out")
num = r"([0-9.eE+-]+|nan)"


def metrics(line):
    return {k: float(re.search(k + "=" + num, line).group(1)) for k in (*MET, "locvar_in")
            if re.search(k + "=" + num, line)}


def baseline(ds):
    out, split = {}, None
    for line in (S4LOG / f"diag_ortho_out_{ds}.log").read_text(encoding="utf-8").splitlines():
        m = re.match(rf"=== {ds} split (\d+)", line)
        if m:
            split = int(m.group(1))
        elif line.startswith("  baseline :") and split is not None:
            out[split] = metrics(line)
    return out


def e1(ds):
    out, split = {}, None
    for line in (DEV / "logs" / f"ablate_rho_{ds}.log").read_text(encoding="utf-8").splitlines():
        m = re.match(rf"=== {ds} split (\d+)", line)
        if m:
            split = int(m.group(1))
            out[split] = {"modes": {}}
        elif split is None:
            continue
        elif line.startswith("selection "):
            out[split]["sel_label"] = line.split(" R/var")[0][len("selection "):].strip()
            out[split]["sel_R"] = float(re.search(r"R/var=" + num, line).group(1))
        elif line.startswith("chosen rho counts"):
            out[split]["rho_counts"] = line[len("chosen rho counts "):]
        elif line.startswith("SELECTED"):
            out[split]["sel"] = metrics(line)
        elif re.match(r"  rho=(\S+)", line):
            mode = re.match(r"  (rho=\S+)", line).group(1)
            d = metrics(line)
            d["R"] = float(re.search(r"R/var=" + num, line).group(1))
            d["label"] = line[2:].split(" R/var")[0]
            out[split]["modes"][mode] = d
    return {s: v for s, v in out.items() if "sel" in v}


def fmt(d, base):
    cells = []
    for k in MET:
        v, b = d.get(k, np.nan), base.get(k, np.nan)
        r = v / b if b and not np.isnan(b) and not np.isnan(v) else np.nan
        cells.append(f"{k}={v:.5f}({r:4.2f})")
    return " ".join(cells) + f" locvar={d.get('locvar_in', np.nan):.3g}"


for ds in sys.argv[1:]:
    base, res = baseline(ds), e1(ds)
    print(f"##### {ds}: splits {sorted(res)}")
    rows = {"selected": [], "rho=0": [], "rho=0.25": [], "rho=1": [], "rho=per-tree": []}
    for s in sorted(res):
        r, b = res[s], base.get(s, {})
        print(f"-- split {s}: selection [{r.get('sel_label')}] R/var={r.get('sel_R'):.5f} "
              f"rho counts {r.get('rho_counts')}")
        print(f"   baseline     {fmt(b, b)}")
        print(f"   selected     {fmt(r['sel'], b)}")
        rows["selected"].append((r["sel"], b))
        for mode, d in r["modes"].items():
            print(f"   {mode:12s} {fmt(d, b)} R/var={d['R']:.5f} [{d['label']}]")
            rows[mode].append((d, b))
    print(f"   geo-mean ratio to baseline over splits (n = splits where both are defined)")
    for mode, lst in rows.items():
        cells = []
        for k in MET:
            rat = [d[k] / b[k] for d, b in lst if k in d and k in b
                   and not np.isnan(d[k]) and not np.isnan(b[k]) and b[k] > 0]
            cells.append(f"{k}={np.exp(np.mean(np.log(rat))):.3f}(n={len(rat)})" if rat
                         else f"{k}=n/a")
        print(f"   {mode:12s}", " ".join(cells))
    # ratio of the selection to the rho = 0 member (= S4's selection)
    cells = []
    for k in (*MET, "locvar_in"):
        rat = [r["sel"][k] / r["modes"]["rho=0"][k] for r in res.values()
               if "rho=0" in r["modes"] and not np.isnan(r["sel"].get(k, np.nan))
               and not np.isnan(r["modes"]["rho=0"].get(k, np.nan)) and r["modes"]["rho=0"][k] > 0]
        cells.append(f"{k}={np.exp(np.mean(np.log(rat))):.3f}(n={len(rat)})" if rat else f"{k}=n/a")
    print("   selected / rho=0 (S4):", " ".join(cells))
