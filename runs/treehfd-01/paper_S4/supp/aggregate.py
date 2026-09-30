"""Aggregate raw/ results into tables (SUPP_TABLES.md, R2/R3/R4/R5 json) and figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H = Path(__file__).resolve().parent
RAW = H / "raw"
DS = ["abalone", "airfoil", "bike", "housing", "concrete", "nutrition", "parkinson", "powerplant", "superconduct"]
L = lambda n: json.load(open(RAW / n))
core = {d: L(f"core__{d}.json") for d in DS}
ridge = {d: L(f"ridge__{d}.json") for d in DS}
CFG = ["lat0.03", "lat0.3", "main0.3", "main3", "kappa6", "kappa24"]
cfg = {c: {d: L(f"cfg_{c}__{d}.json") for d in DS} for c in CFG}


def gm(x):
    x = [v for v in x if v is not None and v > 0]
    return float(np.exp(np.mean(np.log(x)))) if x else None


def rt(a, b):
    return None if a is None or b is None or b == 0 else a / b


def f2(x, d=3):
    return "n/a" if x is None else f"{x:.{d}g}"


md = []
out = {}

# ------------------------------------------------------------------ sanity: re-fit equals official
official = json.load(open(H.parent / "results/method_full.json"))["real"]
chk = {d: {m: (core[d]["default"][m], official[d][m]) for m in ("resid_in", "resid_out", "ortho_out")} for d in DS}
out["reproduction_check"] = chk
maxdiff = max(abs(a - b) / max(abs(b), 1e-12) for d in DS for a, b in chk[d].values() if a is not None and b is not None)
md += [f"Re-fit of the submitted GT-LOCO in this directory reproduces the official results/method_full.json: max relative difference over resid_in/resid_out/ortho_out on 9 datasets = {maxdiff:.2e}.", ""]

# ------------------------------------------------------------------ R2
REAL_V = [("TreeHFD", lambda d: core[d]["baseline"]),
          ("(d) TreeHFD + purification", lambda d: core[d]["purified_baseline"]),
          ("(a) ridge, zero rule, one global kappa", lambda d: ridge[d]["ridge_zero"]),
          ("(b) ridge, harmonic rule, one global kappa", lambda d: ridge[d]["ridge_harmonic"]),
          ("(e) lattice prior + harmonic/zero rule, one global kappa", lambda d: core[d]["lattice_common"]),
          ("ridge prior/zero rule, full ensemble selection (per-tree kappa + shift)", lambda d: ridge[d]["ridge_full_selection"]),
          ("(c) full GT-LOCO", lambda d: core[d]["default"])]
r2 = {}
for m in ("resid_out", "resid_in", "ortho_out"):
    rows = [f"**{m}, ratio to TreeHFD**", "", "| variant | " + " | ".join(DS) + " | geo-mean |", "|---|" + "---|" * (len(DS) + 1)]
    for name, fn in REAL_V[1:]:
        rs = [rt(fn(d)[m], core[d]["baseline"][m]) for d in DS]
        r2.setdefault(m, {})[name] = {"per_dataset": dict(zip(DS, rs)), "gm": gm(rs)}
        rows.append(f"| {name} | " + " | ".join("n/a" if r is None else f"{r:.2f}" for r in rs) + f" | {gm(rs):.3f} |")
    md += rows + [""]
out["R2_real"] = r2
# analytical
AV = ["treehfd", "purified_treehfd", "ridge_zero", "ridge_harmonic", "lattice_common", "gtloco"]
ana = [L(f"analytical__{r}.json") for r in range(10)]
AM = ["mse_eta12", "mse_eta34", "mse_others", "resid_in", "resid_out", "ortho_out"]
rows = ["**Analytical case, mean ± std over 10 repetitions**", "", "| variant | " + " | ".join(AM) + " |", "|---|" + "---|" * len(AM)]
r2a = {}
for v in AV:
    cells = []
    for m in AM:
        x = np.array([a[v][m] for a in ana if a[v][m] is not None], float)
        r2a.setdefault(v, {})[m] = {"mean": float(x.mean()), "std": float(x.std()), "n": len(x)}
        cells.append(f"{x.mean():.4g} ± {x.std():.2g}")
    rows.append(f"| {v} | " + " | ".join(cells) + " |")
md += rows + [""]
out["R2_analytical"] = r2a

# ------------------------------------------------------------------ R4 existing analytical: paired differences
r4a = {}
rows = ["**Existing analytical case, paired per-rep difference GT-LOCO − TreeHFD (mean ± SE over 10 reps)**", "",
        "| metric | TreeHFD mean | GT-LOCO mean | diff ± SE | reps with GT-LOCO lower |", "|---|---|---|---|---|"]
for m in ["mse_eta1", "mse_eta2", "mse_eta3", "mse_eta4", "mse_eta5", "mse_eta6", "mse_eta12", "mse_eta34", "mse_others", "resid_out", "resid_in", "ortho_out"]:
    b = np.array([a["treehfd"][m] for a in ana]); g = np.array([a["gtloco"][m] for a in ana])
    dlt = g - b
    se = dlt.std(ddof=1) / np.sqrt(len(dlt))
    r4a[m] = {"treehfd": float(b.mean()), "gtloco": float(g.mean()), "diff": float(dlt.mean()), "se": float(se), "n_lower": int((dlt < 0).sum())}
    rows.append(f"| {m} | {b.mean():.4g} | {g.mean():.4g} | {dlt.mean():.3g} ± {se:.2g} | {(dlt < 0).sum()}/10 |")
md += rows + [""]
out["R4_existing_analytical_paired"] = r4a
# new case
nc = [L(f"newcase__{r}.json") for r in range(10)]
NM = ["mse_eta1", "mse_eta2", "mse_eta3", "mse_eta4", "mse_eta5", "mse_eta6", "mse_main_mean", "mse_eta45", "mse_others_zero", "ortho_out", "resid_out", "resid_in"]
rows = ["**New case (AR(1) rho=0.6, f = x1 + 0.8 x2^2 + 0.5 x3^3 + 1.5 x4 x5), 10 seeds**", "",
        "| metric | TreeHFD | GT-LOCO | diff GT−TreeHFD ± SE | seeds GT-LOCO lower |", "|---|---|---|---|---|"]
r4n = {}
for m in NM:
    b = np.array([a["treehfd"][m] for a in nc], float); g = np.array([a["gtloco"][m] for a in nc], float)
    dlt = g - b
    se = dlt.std(ddof=1) / np.sqrt(len(dlt))
    r4n[m] = {"treehfd_mean": float(b.mean()), "treehfd_std": float(b.std()), "gtloco_mean": float(g.mean()), "gtloco_std": float(g.std()),
              "diff": float(dlt.mean()), "se": float(se), "n_lower": int((dlt < 0).sum())}
    rows.append(f"| {m} | {b.mean():.4g} ± {b.std():.2g} | {g.mean():.4g} ± {g.std():.2g} | {dlt.mean():.3g} ± {se:.2g} | {(dlt < 0).sum()}/10 |")
rows.append(f"\nTrue Var[eta_45] = {nc[0]['treehfd']['var_eta45_true']:.3f}; largest variance among the other 14 true pair components = {nc[0]['treehfd']['var_others_true_max']:.2g}; "
            f"XGB R^2 against f on test = {np.mean([a['xgb_r2_test'] for a in nc]):.3f}.")
md += rows + [""]
out["R4_new_case"] = r4n

# ------------------------------------------------------------------ R3
SETTINGS = [("default (lat 0.1, main 1, 12 kappas, shifts -5..+3)", lambda d: core[d]["default"])]
SETTINGS += [(f"lattice ridge {v}", (lambda c: lambda d: cfg[c][d]["default_sel"])(c)) for c, v in (("lat0.03", 0.03), ("lat0.3", 0.3))]
SETTINGS += [(f"main-bin ridge {v}", (lambda c: lambda d: cfg[c][d]["default_sel"])(c)) for c, v in (("main0.3", 0.3), ("main3", 3))]
SETTINGS += [(f"kappa grid {g} pts (shifts -5..+3 in index units)", (lambda c: lambda d: cfg[c][d]["default_sel"])(c)) for c, g in (("kappa6", 6), ("kappa24", 24))]
SETTINGS += [(f"kappa grid {g} pts (shifts rescaled to same kappa range)", (lambda c: lambda d: cfg[c][d]["scaled_shift_sel"])(c)) for c, g in (("kappa6", 6), ("kappa24", 24))]
SETTINGS += [(f"shift range {k}", (lambda k: lambda d: core[d]["shift_range"][k])(k)) for k in ("0..0", "-3..+1", "-8..+5")]
r3 = {}
rows = ["| setting | gm resid_out / TreeHFD | per-dataset range resid_out | gm resid_in / TreeHFD | per-dataset range resid_in | gm ortho_out / TreeHFD |", "|---|---|---|---|---|---|"]
for name, fn in SETTINGS:
    ro = [rt(fn(d)["resid_out"], core[d]["baseline"]["resid_out"]) for d in DS]
    ri = [rt(fn(d)["resid_in"], core[d]["baseline"]["resid_in"]) for d in DS]
    oo = [rt(fn(d)["ortho_out"], core[d]["baseline"]["ortho_out"]) for d in DS]
    r3[name] = {"gm_resid_out": gm(ro), "gm_resid_in": gm(ri), "gm_ortho_out": gm(oo), "resid_out": dict(zip(DS, ro)), "resid_in": dict(zip(DS, ri)), "ortho_out": dict(zip(DS, oo))}
    rows.append(f"| {name} | {gm(ro):.3f} | {min(ro):.2f}–{max(ro):.2f} | {gm(ri):.3f} | {min(ri):.2f}–{max(ri):.2f} | {gm(oo):.3f} |")
md += ["**R3 sensitivity (one change at a time; label-free; defaults were chosen during development with held-out numbers visible)**", ""] + rows + [""]
out["R3"] = r3
worst = max(r3.items(), key=lambda kv: kv[1]["gm_resid_out"])
out["R3_worst_setting_resid_out"] = worst[0]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
names = list(r3)
for a, key, ttl in ((ax[0], "gm_resid_out", "geo-mean resid_out / TreeHFD"), (ax[1], "gm_resid_in", "geo-mean resid_in / TreeHFD")):
    y = np.arange(len(names))
    a.barh(y, [r3[n][key] for n in names], color=["#1f77b4"] + ["#7f7f7f"] * (len(names) - 1))
    a.axvline(r3[names[0]][key], color="k", lw=0.8, ls="--")
    a.set_yticks(y); a.set_yticklabels([n[:38] for n in names] if a is ax[0] else [], fontsize=7); a.invert_yaxis(); a.set_title(ttl, fontsize=9)
plt.tight_layout(); plt.savefig(H / "R3_sensitivity.png", dpi=130); plt.close()

# ------------------------------------------------------------------ R5 (i) shift sweep
SH = [str(s) for s in range(-5, 4)]
r5 = {}
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
for a, m in zip(axs, ("resid_in", "resid_out", "ortho_out")):
    for d in DS:
        ys = [rt(core[d]["shift_sweep"][s][m], core[d]["baseline"][m]) for s in SH]
        if any(v is None for v in ys):
            continue
        a.plot(range(-5, 4), ys, marker="o", ms=3, lw=1, label=d)
        sel = core[d]["default"]["cand"]
        if sel[0] == "shift":
            a.plot([sel[2]], [rt(core[d]["default"][m], core[d]["baseline"][m])], "k*", ms=11)
    a.set_yscale("log"); a.axhline(1, color="grey", lw=0.6); a.set_ylabel(f"{m} / TreeHFD")
axs[0].legend(fontsize=6, ncol=3); axs[-1].set_xlabel("fixed ensemble shift of the per-tree kappa index (star = selected, when a shift candidate was selected)")
plt.tight_layout(); plt.savefig(H / "R5_shift_sweep.png", dpi=130); plt.close()
rows = ["| dataset | selected candidate | resid_out at selected | best fixed shift for resid_out (oracle) | resid_out there | shift minimising R (risk) | resid_out there | best shift for resid_in |", "|---|---|---|---|---|---|---|---|"]
gmsel, gmrisk, gmor = [], [], []
for d in DS:
    sw = core[d]["shift_sweep"]
    o = min(SH, key=lambda s: sw[s]["resid_out"])
    rk = min(SH, key=lambda s: sw[s]["risk_over_var"])
    ri = min(SH, key=lambda s: sw[s]["resid_in"])
    sel = core[d]["default"]
    b = core[d]["baseline"]
    r5[d] = {"selected": sel["cand"], "resid_out_selected": sel["resid_out"], "oracle_shift": int(o), "resid_out_oracle": sw[o]["resid_out"],
             "risk_shift": int(rk), "resid_out_risk_shift": sw[rk]["resid_out"], "resid_in_best_shift": int(ri),
             "sweep": {s: {k: sw[s][k] for k in ("resid_in", "resid_out", "ortho_out", "risk_over_var")} for s in SH}}
    gmsel.append(sel["resid_out"] / b["resid_out"]); gmrisk.append(sw[rk]["resid_out"] / b["resid_out"]); gmor.append(sw[o]["resid_out"] / b["resid_out"])
    rows.append(f"| {d} | {sel['cand'][0]} {sel['cand'][2]:g} (variant {sel['cand'][1]}) | {sel['resid_out']:.4g} | {o} | {sw[o]['resid_out']:.4g} | {rk} | {sw[rk]['resid_out']:.4g} | {ri} |")
rows += ["", f"Geo-mean resid_out / TreeHFD: selected {gm(gmsel):.3f}; fixed shift chosen by min risk {gm(gmrisk):.3f}; oracle best fixed shift (uses held-out, diagnostic only) {gm(gmor):.3f}.", ""]
per_shift = {s: gm([core[d]["shift_sweep"][s]["resid_out"] / core[d]["baseline"]["resid_out"] for d in DS]) for s in SH}
per_shift_in = {s: gm([core[d]["shift_sweep"][s]["resid_in"] / core[d]["baseline"]["resid_in"] for d in DS]) for s in SH}
rows += ["| fixed shift | " + " | ".join(SH) + " |", "|---|" + "---|" * 9,
         "| gm resid_out / TreeHFD | " + " | ".join(f"{per_shift[s]:.3f}" for s in SH) + " |",
         "| gm resid_in / TreeHFD | " + " | ".join(f"{per_shift_in[s]:.3f}" for s in SH) + " |", ""]
md += ["**R5(i) fixed-shift sweep (selected variant per dataset, no selection over shifts)**", ""] + rows
out["R5_shift"] = {"per_dataset": r5, "gm_selected": gm(gmsel), "gm_risk_shift": gm(gmrisk), "gm_oracle": gm(gmor), "gm_per_shift_resid_out": per_shift, "gm_per_shift_resid_in": per_shift_in}

(H / "SUPP_TABLES.md").write_text("\n".join(md), encoding="utf-8")
json.dump(out, open(H / "aggregate_results.json", "w"), indent=1, default=str)
print("\n".join(md))
