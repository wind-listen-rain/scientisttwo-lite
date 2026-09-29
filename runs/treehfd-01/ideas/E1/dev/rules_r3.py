"""Round 3: compare ensemble selection rules offline on the cached candidate tables.

usage: rules_r3.py <tag> <dataset> [...]
For each rule, geo-mean over the 8 diagnostic splits of the ratio to the baseline (per-split
baseline from S4's dev/logs/diag_ortho_out_<ds>.log, as in summarize_r2.py) and of the ratio
to S4 (the rho = 0 mode's plain argmin, which is S4's selection), then the benchmark split
(split 0) alone. Rules (all label-free: they see R, the in-sample fidelity and the per-point
leave-out residuals only):
  S4        rho = 0 mode, plain argmin of R (= S4)
  argmin    plain argmin of R over all candidates (E1 round 1)
  se1       one-SE rule towards fidelity over all candidates (E1 round 2)
  cons      smallest R among the candidates whose in-sample fidelity is <= S4's
  cons_se1  one-SE rule towards fidelity within that feasible set
Diagnosis only: held-out numbers never enter the method.
"""
import sys
from pathlib import Path

import numpy as np

DEV = Path(__file__).resolve().parent
sys.path.insert(0, str(DEV.parent / "lib"))
sys.path.insert(0, str(DEV))
import re  # noqa: E402

import agtloco  # noqa: E402

S4LOG = DEV.parents[1] / "S4" / "dev" / "logs"
MET = ("resid_in", "resid_out", "ortho_in", "ortho_out", "locvar_in")


def baseline(ds):
    """Per-split baseline metrics (copied from summarize_r2.py, which runs on import)."""
    out, split = {}, None
    for line in (S4LOG / f"diag_ortho_out_{ds}.log").read_text(encoding="utf-8").splitlines():
        m = re.match(rf"=== {ds} split (\d+)", line)
        if m:
            split = int(m.group(1))
        elif line.startswith("  baseline :") and split is not None:
            out[split] = {k: float(v) for k in MET
                          for v in re.findall(k + r"=([0-9.eE+-]+|nan)", line)[:1]}
    return out


def first_min(r, idx):
    return int(idx[agtloco._first_min(r[idx])])


def rule_s4(z):
    return first_min(z["risk"], np.flatnonzero(z["key"][:, 2] == 0))


def rule_argmin(z):
    return first_min(z["risk"], np.arange(len(z["risk"])))


def se_within(z, idx, k=1.0):
    sq = z["loo"][idx].astype(float) ** 2
    _, i, _ = agtloco._se_select(sq, z["fid"][idx], k)
    return int(idx[i])


def feasible(z):
    ref = z["fid"][rule_s4(z)]
    return np.flatnonzero(z["fid"] <= ref * (1 + 1e-12))


RULES = {
    "S4": rule_s4,
    "argmin": rule_argmin,
    "se1": lambda z: se_within(z, np.arange(len(z["risk"]))),
    "cons": lambda z: first_min(z["risk"], feasible(z)),
    "cons_se1": lambda z: se_within(z, feasible(z)),
}


def gm(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return (float(np.exp(np.mean(np.log(x)))), len(x)) if len(x) else (np.nan, 0)


def describe(z, i, rhos):
    k = z["key"][i]
    rm = "per-tree" if k[2] == len(rhos) else f"{rhos[k[2]]:g}"
    pr, ru = agtloco.variants()[k[1]]
    return (f"{'shift' if k[0] == 0 else 'common'} {agtloco.PRIORS[pr]}/{agtloco.RULES[ru]} "
            f"rho={rm} {k[3]:+d}")


def main():
    tag = sys.argv[1]
    for ds in sys.argv[2:]:
        files = sorted((DEV / "cache").glob(f"{ds}_s*_{tag}.npz"))
        zs = {int(f.name.split("_s")[1].split("_")[0]): dict(np.load(f)) for f in files}
        if not zs:
            continue
        names = list(zs[min(zs)]["names"])
        rhos = zs[min(zs)]["rhos"]
        base = baseline(ds) if ds != "analytical" else {}
        print(f"=== {ds} [{tag}] splits {sorted(zs)}")
        picks = {r: {s: f(z) for s, z in zs.items()} for r, f in RULES.items()}
        head = " ".join(f"{m:>16s}" for m in names)
        if ds == "analytical":
            print(f"{'rule':9s} mean over reps, ratio to S4 (in brackets)")
            for r in RULES:
                vals = np.array([zs[s]["met"][picks[r][s]] for s in zs])
                s4 = np.array([zs[s]["met"][picks["S4"][s]] for s in zs])
                print(f"{r:9s} " + " ".join(f"{m:.5g}({m / q:.3f})" for m, q in
                                            zip(vals.mean(0), s4.mean(0), strict=True)))
            continue
        print(f"{'rule':9s} geo-mean ratio to baseline (n) | ratio to S4\n{'':9s} {head}")
        for r in RULES:
            cells, cells_s4 = [], []
            for mi, m in enumerate(names):
                rb = [zs[s]["met"][picks[r][s], mi] / base[s][m] for s in zs if m in base[s]]
                rs = [zs[s]["met"][picks[r][s], mi] / zs[s]["met"][picks["S4"][s], mi]
                      for s in zs]
                g, n = gm(rb)
                cells.append(f"{g:.3f}({n})")
                cells_s4.append(f"{gm(rs)[0]:.3f}")
            print(f"{r:9s} " + " ".join(f"{c:>16s}" for c in cells) + " | "
                  + " ".join(cells_s4))
        if 0 in zs:
            z = zs[0]
            print("benchmark split 0 (absolute):")
            for r in RULES:
                i = picks[r][0]
                print(f"  {r:9s} " + " ".join(f"{m}={v:.5g}" for m, v in
                                              zip(names, z["met"][i], strict=True))
                      + f"  R/var={z['risk'][i] / z['var_t']:.5f}  [{describe(z, i, rhos)}]")
        # chosen rho mode / kappa shift per rule
        for r in ("cons", "cons_se1", "se1"):
            print(f"  {r} picks: " + "; ".join(describe(zs[s], picks[r][s], rhos)
                                              for s in sorted(zs)))


if __name__ == "__main__":
    main()
