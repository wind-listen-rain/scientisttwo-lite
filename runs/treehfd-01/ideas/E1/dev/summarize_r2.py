"""Summarise the round-2 ablation: dev/logs/ablate_r2_<ds>.log against the baseline per split.

usage: summarize_r2.py <dataset> [...]
Rows (geo-mean over splits of the ratio to the baseline, n = splits where both are defined):
  S4          rho = 0 mode, best by R (= S4's selection)
  S4/se1      rho = 0 mode, one-SE rule (S4 with SE_RULE = 1)
  E1-argmin   plain argmin of R over all candidates (E1 round 1)
  E1-se1      one-SE rule over all candidates (E1 round 2, the submitted selection)
  rho=0.25    rho = 0.25 mode, best by R;  rho=0.25/se1, rho=1, rho=1/se1 likewise
Then E1-se1 / S4, E1-argmin / S4 and E1-se1 / E1-argmin, and the orthogonality at a common
kappa (same kappa in every tree, only rho changes) relative to rho = 0 at that kappa.
Baseline per-split numbers come from S4's dev/logs/diag_ortho_out_<ds>.log (deterministic
real-data models and metrics). Diagnosis only (held-out numbers).
"""
import re
import sys
from pathlib import Path

import numpy as np

DEV = Path(__file__).resolve().parent
S4LOG = DEV.parents[1] / "S4" / "dev" / "logs"
MET = ("resid_in", "resid_out", "ortho_in", "ortho_out", "locvar_in")
num = r"([0-9.eE+-]+|nan)"


def metrics(line):
    return {k: float(re.search(k + "=" + num, line).group(1)) for k in MET
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


def parse(ds):
    out, split = {}, None
    for line in (DEV / "logs" / f"ablate_r2_{ds}.log").read_text(encoding="utf-8").splitlines():
        m = re.match(rf"=== {ds} split (\d+)", line)
        if m:
            split = int(m.group(1))
            out[split] = {"rows": {}, "kappa": {}}
            continue
        if split is None:
            continue
        r = out[split]
        if line.startswith("selection "):
            r["sel_label"] = line.split(" R/var")[0][len("selection "):].strip()
            r["tied"] = int(re.search(r"tied=(\d+)", line).group(1))
        elif line.startswith("chosen rho counts"):
            r["rho_counts"] = line[len("chosen rho counts "):]
        elif line.startswith("chosen kappa quantiles"):
            r["kappa_q"] = line[len("chosen kappa quantiles "):]
        elif line.startswith("SELECTED"):
            r["rows"]["E1-se1"] = metrics(line)
        elif line.startswith("ARGMIN"):
            r["rows"]["E1-argmin"] = (metrics(line) if "same as selection" not in line
                                      else dict(r["rows"]["E1-se1"]))
        elif re.match(r"  rho=\S+", line):
            mode = re.match(r"  (rho=\S+)", line).group(1)
            mode = {"rho=0": "S4", "rho=0/se1": "S4/se1"}.get(mode, mode)
            r["rows"][mode] = metrics(line)
        elif re.match(r"  k=\S+", line):
            k = re.match(r"  k=(\S+)", line).group(1)
            rho = re.search(r"rho=(\S+)\s+\+", line).group(1)
            r["kappa"].setdefault(k, {})[rho] = metrics(line)
    return {s: v for s, v in out.items() if "E1-se1" in v["rows"]}


def gm(pairs):
    rat = [a / b for a, b in pairs if not np.isnan(a) and not np.isnan(b) and b > 0]
    return (np.exp(np.mean(np.log(rat))), len(rat)) if rat else (np.nan, 0)


ORDER = ("S4", "S4/se1", "E1-argmin", "E1-se1", "rho=0.25", "rho=0.25/se1", "rho=1", "rho=1/se1")
for ds in sys.argv[1:]:
    base, res = baseline(ds), parse(ds)
    print(f"##### {ds}: splits {sorted(res)}")
    for s in sorted(res):
        r = res[s]
        print(f"-- split {s}: E1-se1 [{r['sel_label']}] tied={r['tied']} rho {r['rho_counts']} "
              f"kappa q {r['kappa_q']}")
        for row in ORDER:
            if row in r["rows"]:
                d, b = r["rows"][row], base.get(s, {})
                print(f"   {row:13s} " + " ".join(
                    f"{k}={d.get(k, np.nan):.5g}({d.get(k, np.nan) / b[k] if b.get(k) else np.nan:4.2f})"
                    for k in MET[:4]) + f" locvar={d.get('locvar_in', np.nan):.4g}")
    print("   geo-mean ratio to baseline over splits:")
    for row in ORDER:
        cells = []
        for k in MET[:4]:
            g, n = gm([(res[s]["rows"][row].get(k, np.nan), base[s].get(k, np.nan))
                       for s in res if row in res[s]["rows"] and s in base])
            cells.append(f"{k}={g:.3f}(n={n})")
        print(f"   {row:13s}", " ".join(cells))
    for num_row, den_row in (("E1-se1", "S4"), ("E1-argmin", "S4"), ("E1-se1", "E1-argmin"),
                             ("S4/se1", "S4")):
        cells = []
        for k in MET:
            g, n = gm([(res[s]["rows"][num_row].get(k, np.nan), res[s]["rows"][den_row].get(k, np.nan))
                       for s in res if num_row in res[s]["rows"] and den_row in res[s]["rows"]])
            cells.append(f"{k}={g:.3f}(n={n})")
        print(f"   {num_row} / {den_row}:".ljust(26), " ".join(cells))
    print("   fixed common kappa, ratio to rho = 0 at the same kappa (geo-mean over splits):")
    kappas = sorted({k for s in res for k in res[s]["kappa"]}, key=float)
    for kap in kappas:
        cells = []
        for rho in ("0.25", "1"):
            for k in ("resid_in", "resid_out", "ortho_in", "ortho_out"):
                g, n = gm([(res[s]["kappa"][kap][rho].get(k, np.nan), res[s]["kappa"][kap]["0"].get(k, np.nan))
                           for s in res if kap in res[s]["kappa"]])
                cells.append(f"{k}={g:.3f}")
            cells.append("|")
        print(f"   kappa={kap:6s} rho=0.25: " + " ".join(cells[:5]) + " rho=1: " + " ".join(cells[5:-1]))
