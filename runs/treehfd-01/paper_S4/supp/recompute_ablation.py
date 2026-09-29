"""R5(iii): recompute the ablation tables from the result files (results/*.json), no re-running.

Per-dataset ratios ablation / full GT-LOCO and ablation / TreeHFD for resid_out, resid_in, ortho_in, ortho_out,
geometric means over all datasets and excluding abalone + airfoil (robustness), and per-dataset sign statements.
"""
import json
import math
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
L = lambda n: json.load(open(R / n))
BASE, FULL = L("baseline_full.json"), L("method_full.json")
ABL = {"no_shrinkage_direct_solver_only": "No shrinkage (kappa=0.01 only)",
       "no_lattice_prior_ridge_zero_only": "No lattice prior / harmonic rule",
       "no_ensemble_level_kappa_step": "No ensemble-level kappa step",
       "soft_orthogonality_rows": "Soft orthogonality rows"}
ABLD = {k: L(f"ablation_{k}.json") for k in ABL}
DS = list(FULL["real"])
EXCL = {"abalone", "airfoil"}
METR = ["resid_out", "resid_in", "ortho_in", "ortho_out"]


def gm(xs):
    xs = [x for x in xs if x is not None]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def rt(a, b):
    return None if a is None or b is None or b == 0 else a / b


def f2(x):
    return "n/a" if x is None else f"{x:.2f}"


out = {"envs": {"base": BASE.get("eval_env"), "full": FULL.get("eval_env"),
                **{k: v.get("eval_env") for k, v in ABLD.items()}}, "tables": {}}
md = [f"eval_env: {out['envs']}", ""]
for m in METR:
    for ref_name, ref in (("TreeHFD", BASE), ("full GT-LOCO", FULL)):
        rows = ["| variant | " + " | ".join(DS) + " | gm all | gm excl. abalone, airfoil |", "|---|" + "---|" * (len(DS) + 2)]
        variants = ([("Full GT-LOCO", FULL)] if ref is BASE else []) + [(ABL[k], ABLD[k]) for k in ABL]
        tab = {}
        for label, res in variants:
            rs = {d: rt(res["real"][d][m], ref["real"][d][m]) for d in DS}
            g_all = gm(rs.values())
            g_ex = gm([v for d, v in rs.items() if d not in EXCL])
            tab[label] = {"per_dataset": rs, "gm_all": g_all, "gm_excl_abalone_airfoil": g_ex}
            rows.append(f"| {label} | " + " | ".join(f2(rs[d]) for d in DS) + f" | {f2(g_all)} | {f2(g_ex)} |")
        out["tables"][f"{m}_vs_{ref_name}"] = tab
        md += [f"### {m}, ratio to {ref_name}", ""] + rows + [""]
# analytical (means over reps) ratio to full
AM = ["mse_eta12", "mse_eta34", "mse_others", "resid_out"]
rows = ["| variant | " + " | ".join(f"{a} (ratio to full)" for a in AM) + " |", "|---|---|---|---|---|"]
for k in ABL:
    rows.append(f"| {ABL[k]} | " + " | ".join(f2(rt(ABLD[k]['analytical']['mean'][a], FULL['analytical']['mean'][a])) for a in AM) + " |")
md += ["### analytical case, ablation / full GT-LOCO (mean over 10 reps)", ""] + rows + [""]
# statement checks
notes = []
for k in ("no_lattice_prior_ridge_zero_only", "no_ensemble_level_kappa_step"):
    for d in ("concrete", "nutrition"):
        s = " ".join(f"{m}: full {FULL['real'][d][m]:.4g} -> ablation {ABLD[k]['real'][d][m]:.4g} (x{rt(ABLD[k]['real'][d][m], FULL['real'][d][m]):.2f})"
                     if FULL['real'][d][m] is not None and ABLD[k]['real'][d][m] is not None else f"{m}: n/a"
                     for m in METR)
        notes.append(f"* {ABL[k]} on {d}: {s}")
md += ["### raw numbers behind the statements on concrete and nutrition", ""] + notes + [""]
Path(__file__).with_name("ablation_recomputed.md").write_text("\n".join(md), encoding="utf-8")
json.dump(out, open(Path(__file__).with_name("ablation_recomputed.json"), "w"), indent=1)
print("\n".join(md))
