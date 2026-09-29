"""R1: bootstrap of the held-out test rows (1000 resamples), fitted decompositions and model fixed.

Reads raw/core__<dataset>.npz (test-point predictions of the SAME fitted TreeHFD and GT-LOCO decompositions, saved by
run_real.py core). For each resample of the test rows (same row indices for both methods) it recomputes resid_out and
ortho_out (harness definitions, including the 1% variance rule for interactions) and reports paired ratios
GT-LOCO / TreeHFD with percentile 95% CIs and the fraction of resamples in which GT-LOCO is lower ("wins").
Dataset-level: geometric-mean ratio (i) over test-row resamples only (datasets independent), (ii) resampling the
datasets with replacement (point ratios), (iii) both levels.
Caveat: the fitted TreeHFD baseline uses a random tie-break for unseen test cells (one draw is used here; the spread of
the other draws is in raw/core__*.json "baseline_redraws").
"""
import numpy as np

from common import DATASETS, SUPP, dump, harness

B = 1000
SEED = 12345


def load(ds):
    z = np.load(SUPP / f"raw/core__{ds}.npz")
    out = {}
    for m in ("base", "gt"):
        rec = z[f"{m}_eta0"] + z[f"{m}_main"].sum(1) + z[f"{m}_inter"].sum(1)
        out[m] = dict(main=z[f"{m}_main"], inter=z[f"{m}_inter"], il=z[f"{m}_il"], sq=(z["p_te"] - rec) ** 2)
    return z["p_te"], out


def ortho_boot(P, main, inter, il, C):
    """Vectorised harness.orthogonality for every resample (rows of the count matrix C). NaN = undefined."""
    n = len(P)
    if inter.shape[1] == 0:
        return np.full(C.shape[0], np.nan)
    W = C / n
    mP = W @ P
    vP = W @ P ** 2 - mP ** 2
    mI = W @ inter
    vI = W @ inter ** 2 - mI ** 2
    mM = W @ main
    vM = W @ main ** 2 - mM ** 2
    worst = np.full(C.shape[0], -np.inf)
    for c, (j, k) in enumerate(il):
        valid = vI[:, c] >= 0.01 * vP
        for m in (j, k):
            cov = W @ (inter[:, c] * main[:, m]) - mI[:, c] * mM[:, m]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.abs(cov / np.sqrt(vI[:, c] * vM[:, m]))
            ok = valid & (vM[:, m] > 0) & np.isfinite(r)
            worst = np.where(ok, np.maximum(worst, r), worst)
    worst[~np.isfinite(worst)] = np.nan
    return worst


def pct(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [None, None]


def gmean_cols(M):
    with np.errstate(invalid="ignore"):
        return np.exp(np.nanmean(np.log(M), axis=1))


rng = np.random.default_rng(SEED)
res = {"B": B, "per_dataset": {}}
R_res, R_ort = {}, {}
for ds in DATASETS:
    P, o = load(ds)
    n = len(P)
    idx = rng.integers(0, n, size=(B, n))
    C = np.stack([np.bincount(i, minlength=n) for i in idx]).astype(float)
    W = C / n
    var_p = (W @ P ** 2) - (W @ P) ** 2
    rb = (W @ o["base"]["sq"]) / var_p
    rg = (W @ o["gt"]["sq"]) / var_p
    ratio_r = rg / rb
    ob = ortho_boot(P, o["base"]["main"], o["base"]["inter"], o["base"]["il"], C)
    og = ortho_boot(P, o["gt"]["main"], o["gt"]["inter"], o["gt"]["il"], C)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_o = og / ob
    ok = np.isfinite(ratio_o)
    pt = {"resid_out_base": float(np.mean(o["base"]["sq"]) / np.var(P)),
          "resid_out_gt": float(np.mean(o["gt"]["sq"]) / np.var(P)),
          "ortho_out_base": harness.orthogonality(P, o["base"]["main"], o["base"]["inter"], o["base"]["il"]),
          "ortho_out_gt": harness.orthogonality(P, o["gt"]["main"], o["gt"]["inter"], o["gt"]["il"])}
    lo, hi = pct(ratio_r)
    d = {"n_test": n, "point": pt,
         "resid_out": {"ratio_point": pt["resid_out_gt"] / pt["resid_out_base"], "ci95": [lo, hi],
                       "boot_mean": float(ratio_r.mean()), "boot_sd": float(ratio_r.std()),
                       "win_rate": float(np.mean(ratio_r < 1)), "ci_includes_1": bool(lo <= 1 <= hi)},
         "ortho_out": None}
    if pt["ortho_out_base"] is not None and pt["ortho_out_gt"] is not None and ok.sum() > 0:
        lo, hi = pct(ratio_o[ok])
        d["ortho_out"] = {"ratio_point": pt["ortho_out_gt"] / pt["ortho_out_base"], "ci95": [lo, hi],
                          "boot_median": float(np.median(ratio_o[ok])), "n_valid_resamples": int(ok.sum()),
                          "win_rate": float(np.mean(ratio_o[ok] < 1)), "ci_includes_1": bool(lo <= 1 <= hi)}
    res["per_dataset"][ds] = d
    R_res[ds], R_ort[ds] = ratio_r, np.where(ok, ratio_o, np.nan)
    print(ds, d["resid_out"]["ratio_point"], d["resid_out"]["ci95"], d["resid_out"]["win_rate"], flush=True)

dsr = list(R_res)
Mr = np.stack([R_res[d] for d in dsr], axis=1)
Mo = np.stack([R_ort[d] for d in dsr], axis=1)
pointr = np.array([res["per_dataset"][d]["resid_out"]["ratio_point"] for d in dsr])
pointo = np.array([res["per_dataset"][d]["ortho_out"]["ratio_point"] if res["per_dataset"][d]["ortho_out"] else np.nan
                   for d in dsr])
res["geo_mean"] = {"datasets": dsr}
for name, M, point in (("resid_out", Mr, pointr), ("ortho_out", Mo, pointo)):
    valid = np.isfinite(point)
    nv = int(valid.sum())
    Mv = M[:, valid]
    lp = np.log(point[valid])
    g_rows = gmean_cols(Mv)                                     # (i) test-row resampling only
    dd = rng.integers(0, nv, size=(B, nv))
    g_ds = np.exp(np.mean(lp[dd], axis=1))                      # (ii) datasets resampled, point ratios
    bb = rng.integers(0, B, size=(B, nv))
    g_both = gmean_cols(np.stack([Mv[bb[:, a], dd[:, a]] for a in range(nv)], axis=1))   # (iii) both levels
    res["geo_mean"][name] = {"point": float(np.exp(lp.mean())), "n_datasets": nv,
                             "test_rows_only": {"ci95": pct(g_rows), "win_rate": float(np.nanmean(g_rows < 1))},
                             "datasets_only": {"ci95": pct(g_ds), "win_rate": float(np.mean(g_ds < 1))},
                             "both_levels": {"ci95": pct(g_both), "win_rate": float(np.nanmean(g_both < 1))}}
    if name == "resid_out":
        keep = np.array([d not in ("abalone", "airfoil") for d in dsr]) & valid
        dd2 = rng.integers(0, int(keep.sum()), size=(B, int(keep.sum())))
        res["geo_mean"][name]["excluding_abalone_airfoil"] = {
            "point": float(np.exp(np.mean(np.log(point[keep])))),
            "test_rows_only_ci95": pct(gmean_cols(M[:, keep])),
            "datasets_only_ci95": pct(np.exp(np.mean(np.log(point[keep])[dd2], axis=1)))}
dump(res, SUPP / "R1_bootstrap.json")


def f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def verdict(x):
    if x["ci_includes_1"]:
        return "no significant gain (CI includes 1)"
    return "significant gain" if x["ci95"][1] < 1 else "significant loss"


md = ["| dataset | n_test | resid_out ratio (point) | 95% CI | GT-LOCO wins | verdict | ortho_out ratio (point) | 95% CI | wins | verdict |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for ds in DATASETS:
    d = res["per_dataset"][ds]
    r, o = d["resid_out"], d["ortho_out"]
    tail = (f"{f(o['ratio_point'])} | [{f(o['ci95'][0])}, {f(o['ci95'][1])}] | {o['win_rate']:.3f} | {verdict(o)} |" if o
            else "n/a | n/a | n/a | undefined (no interaction above 1% variance) |")
    md.append(f"| {ds} | {d['n_test']} | {f(r['ratio_point'])} | [{f(r['ci95'][0])}, {f(r['ci95'][1])}] | "
              f"{r['win_rate']:.3f} | {verdict(r)} | {tail}")
g = res["geo_mean"]
md += ["", "| geometric mean over datasets | point | CI (test rows only) | CI (datasets only) | CI (both levels) |",
       "|---|---|---|---|---|"]
for name in ("resid_out", "ortho_out"):
    x = g[name]
    md.append(f"| {name} ({x['n_datasets']} datasets) | {f(x['point'])} | "
              f"[{f(x['test_rows_only']['ci95'][0])}, {f(x['test_rows_only']['ci95'][1])}] | "
              f"[{f(x['datasets_only']['ci95'][0])}, {f(x['datasets_only']['ci95'][1])}] | "
              f"[{f(x['both_levels']['ci95'][0])}, {f(x['both_levels']['ci95'][1])}] |")
x = g["resid_out"]["excluding_abalone_airfoil"]
md.append(f"| resid_out excl. abalone, airfoil | {f(x['point'])} | [{f(x['test_rows_only_ci95'][0])}, "
          f"{f(x['test_rows_only_ci95'][1])}] | [{f(x['datasets_only_ci95'][0])}, {f(x['datasets_only_ci95'][1])}] | |")
(SUPP / "R1_bootstrap.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
