from common import *
import json, os
os.makedirs("tables", exist_ok=True)
out = {}
names = list(ABL)
# analytical
rows = ["| metric | TreeHFD (baseline) | GT-LOCO | ratio |", "|---|---|---|---|"]
for m in AMET + ["fit_s"]:
    b = BASE["analytical"]["mean"][m]; f = FULL["analytical"]["mean"][m]
    bs = BASE["analytical"]["std"][m]; fs = FULL["analytical"]["std"][m]
    rows.append(f"| {m} | {fmt(b)} ± {fmt(bs,2)} | {fmt(f)} ± {fmt(fs,2)} | {fr(ratio(f,b))} |")
out["analytical"] = "\n".join(rows)
# real, absolute
rows = ["| dataset | metric | TreeHFD | GT-LOCO | ratio |", "|---|---|---|---|---|"]
for d in DS:
    for m in METR:
        b = BASE["real"][d][m]; f = FULL["real"][d][m]
        rows.append(f"| {d} | {m} | {fmt(b)} | {fmt(f)} | {fr(ratio(f,b))} |")
out["real"] = "\n".join(rows)
# real, ratios
rows = ["| dataset | XGB test R² | resid_in | resid_out | ortho_in | ortho_out | locvar_in |", "|---|---|---|---|---|---|---|"]
for d in DS:
    rs = [fr(ratio(FULL["real"][d][m], BASE["real"][d][m])) for m in METR]
    rows.append(f"| {d} | {FULL['real'][d]['xgb_r2_test']:.3f} | " + " | ".join(rs) + " |")
rows.append("| **geo-mean** | | " + " | ".join(fr(gm([ratio(FULL['real'][d][m], BASE['real'][d][m]) for d in DS])) for m in METR) + " |")
out["real_ratio"] = "\n".join(rows)
# ablations (ratio to TreeHFD baseline)
def line(label, res, m):
    rs = [ratio(res["real"][d][m], BASE["real"][d][m]) for d in DS]
    return f"| {label} | " + " | ".join(fr(r) for r in rs) + f" | {fr(gm(rs))} |"
for m in ["resid_out", "resid_in", "ortho_in", "ortho_out"]:
    rows = ["| variant | " + " | ".join(DS) + " | geo-mean |", "|---|" + "---|" * (len(DS) + 1)]
    rows.append(line("**Full GT-LOCO**", FULL, m))
    for k in names: rows.append(line(ABL[k], ABLD[k], m))
    out["abl_" + m] = "\n".join(rows)
rows = ["| variant | " + " | ".join(AMET[6:]) + " |", "|---|" + "---|" * 4]
rows.append("| TreeHFD | " + " | ".join(fmt(BASE['analytical']['mean'][m]) for m in AMET[6:]) + " |")
rows.append("| **Full GT-LOCO** | " + " | ".join(fmt(FULL['analytical']['mean'][m]) for m in AMET[6:]) + " |")
for k in names: rows.append(f"| {ABL[k]} | " + " | ".join(fmt(ABLD[k]['analytical']['mean'][m]) for m in AMET[6:]) + " |")
out["abl_analytical"] = "\n".join(rows)
# compute
rows = ["| dataset | TreeHFD fit_s | GT-LOCO fit_s | ratio |", "|---|---|---|---|"]
frs = {}
b = BASE["analytical"]["mean"]["fit_s"]; f = FULL["analytical"]["mean"]["fit_s"]
rows.append(f"| analytical (mean of 10) | {b:.1f} | {f:.1f} | {f/b:.2f} |"); frs["analytical"] = f/b
for d in DS:
    b = BASE["real"][d]["fit_s"]; f = FULL["real"][d]["fit_s"]; frs[d] = f/b
    rows.append(f"| {d} | {b:.1f} | {f:.1f} | {f/b:.2f} |")
out["compute"] = "\n".join(rows)
for k, v in out.items(): open(f"tables/{k}.md", "w", encoding="utf-8").write(v + "\n")
# scalars for prose
S = {}
S["n_ds"] = len(DS)
for m in METR:
    S[f"gm_{m}"] = fr(gm([ratio(FULL['real'][d][m], BASE['real'][d][m]) for d in DS]))
rr = {d: ratio(FULL['real'][d]['resid_out'], BASE['real'][d]['resid_out']) for d in DS}
S["better_ds"] = ", ".join(f"{d} ({rr[d]:.2f}×)" for d in DS if rr[d] < 0.97)
S["same_ds"] = ", ".join(d for d in DS if 0.97 <= rr[d] <= 1.03)
S["worse_ds"] = ", ".join(d for d in DS if rr[d] > 1.03) or "none"
S["best_resid_out"] = f"{min(rr, key=rr.get)} ({rr[min(rr, key=rr.get)]:.2f}×)"
ri = {d: ratio(FULL['real'][d]['resid_in'], BASE['real'][d]['resid_in']) for d in DS}
S["worst_resid_in"] = f"{max(ri, key=ri.get)} ({ri[max(ri, key=ri.get)]:.2f}×)"
S["gm_resid_in_raw"] = fr(gm(list(ri.values())))
for d in DS:
    S[f"oo_{d}"] = fr(ratio(FULL['real'][d]['ortho_out'], BASE['real'][d]['ortho_out']))
    S[f"oi_{d}"] = fr(ratio(FULL['real'][d]['ortho_in'], BASE['real'][d]['ortho_in']))
S["worse_ortho_out"] = ", ".join(f"{d} ({S['oo_'+d]}×)" for d in DS if S['oo_'+d] != 'n/a' and float(S['oo_'+d]) > 1.03) or "none"
S["max_fit_ratio"] = f"{max(frs, key=frs.get)} ({max(frs.values()):.2f}×)"
S["min_fit_ratio"] = f"{min(frs, key=frs.get)} ({min(frs.values()):.2f}×)"
for m in ["resid_out", "mse_others", "mse_eta12", "mse_eta34"]:
    S[f"an_{m}"] = fr(ratio(FULL['analytical']['mean'][m], BASE['analytical']['mean'][m]))
    for k in names:
        S[f"an_{m}_{k}"] = fr(ratio(ABLD[k]['analytical']['mean'][m], FULL['analytical']['mean'][m]))
for k in names:
    for m in ["resid_out", "resid_in"]:
        S[f"abl_{m}_{k}"] = fr(gm([ratio(ABLD[k]['real'][d][m], FULL['real'][d][m]) for d in DS]))
S["errors"] = str(len(FULL["errors"]) + sum(len(a["errors"]) for a in ABLD.values()) + len(BASE["errors"]))
S["eval_env"] = FULL["eval_env"]
S["abal_ri"] = fr(ratio(ABLD["no_ensemble_level_kappa_step"]["real"]["abalone"]["resid_in"], FULL["real"]["abalone"]["resid_in"]))
S["conc_ri"] = fr(ratio(ABLD["no_ensemble_level_kappa_step"]["real"]["concrete"]["resid_in"], FULL["real"]["concrete"]["resid_in"]))
json.dump(S, open("tables/scalars.json", "w"), indent=1)
print(json.dumps(S, indent=1, ensure_ascii=False))
