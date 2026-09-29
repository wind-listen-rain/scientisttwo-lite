# S4 ablations (GT-LOCO TreeHFD)

Each `method_<name>.py` has the harness interface of `../method.py`.
* It imports `../lib/gtloco.py` and `../method.py` through `_shared.py`, sets module-level switches of `gtloco`,
  and reuses `method.fit` / `method.predict` unchanged.
* The harness runs each method in its own process, so the switches never reach the submitted method.

| file | switches | what is removed |
|---|---|---|
| `method_no_shrinkage_direct_solver_only.py` | `KAPPAS = [1e-2]`, `SHIFTS = [0]` | The κ grid and its GT/PRESS choice, and the ensemble κ shift or common κ. Every tree takes κ = 0.01. Kept: direct solver, exact orthogonality, deterministic unseen-cell rule, and the (prior, rule) choice by ensemble leave-out risk. At κ = 0.01 that choice mainly picks the rule. The lattice prior is not strictly zero-strength: in rough directions it still acts on weakly identified cells. |
| `method_no_lattice_prior_ridge_zero_only.py` | `PRIORS = ('ridge',)`, `RULES = ('zero',)` | The lattice (GMRF) prior and the harmonic rule. The κ grid, R, per-tree argmin and ensemble step are kept. |
| `method_no_ensemble_level_kappa_step.py` | `SHIFTS = [0]`, `COMMON_KAPPA = False` | Ensemble κ shift and common κ. Each tree keeps its per-tree argmin of R. The variant is chosen by summed leave-out risk among these per-tree-argmin candidates only. This is the round-3 "per-tree" configuration of `dev/ablate_shift.py`. |
| `method_soft_orthogonality_rows.py` | `HARD_ORTHO = False` | Exact orthogonality. The baseline's soft rows are added to the normal matrix instead. All else is kept. |
| `method_zero_rule_only.py` (optional extra) | `RULES = ('zero',)` | Only the harmonic rule; both priors are kept. Suggested in the lattice ablation's description to separate the rule from the prior. |

`COMMON_KAPPA` is the only change to `lib/gtloco.py` made for the ablations.
* It is a new switch that defaults to `True`, which is exactly the previous candidate set.
* The submitted method was re-run on the subset after the change (`logs/subset_method_recheck.json`) to check it
  is unchanged.

Tools:
* `run_subset.sh`: runs the sequential subset self-tests (`subset_<name>.json`, logs in `logs/`), the method.py
  re-check, and the selection diagnostics.
* `diag_selection.py <method file>`: label-free selection diagnostics from `state.diagnostics` (selected candidate,
  chosen-κ quantiles, R/Var, resid_in/Var, per-variant minimum R). It uses the harness's model and X_train.
* `compare_ablations.py subset|full`: side-by-side table against the baseline and method.py results of this
  machine (env-22b414443084).

## Subset self-test (this machine, env-22b414443084; one run each, not an official evaluation)
All five files ran with no errors. The method.py re-check reproduces `../subset_env-22b414443084.json` exactly
on every metric except fit_s. Ratio to the full method (`compare_ablations.py subset`):

| ablation | analytical resid_out / mse_others / mse_eta12 / mse_eta34 | resid_in (air, con, aba) | resid_out (air, con, aba) | ortho_out (air, con, aba) |
|---|---|---|---|---|
| no_shrinkage_direct_solver_only | 1.09 / 1.13 / 1.02 / 1.00 | 0.82, 0.75, 0.78 | 1.03, 1.09, 1.16 | 1.32, 1.02, 0.90 |
| no_lattice_prior_ridge_zero_only | 1.40 / 1.14 / 1.36 / 1.24 | 0.87, 0.93, 0.93 | 1.32, 0.98, 1.29 | 1.08, 0.96, 1.21 |
| no_ensemble_level_kappa_step | 1.04 / 0.81 / 1.07 / 1.10 | 1.24, 2.71, 3.27 | 1.06, 1.04, 1.03 | 1.06, 0.95, 1.28 |
| soft_orthogonality_rows | 1.00 / 0.99 / 1.01 / 1.00 | 1.00, 1.12, 0.97 | 1.00, 0.99, 0.97 | 0.93, 0.97, 1.42 |
| zero_rule_only (extra) | 1.28 / 0.97 / 1.31 / 1.20 | 1.01, 1.00, 1.00 | 1.17, 1.00, 1.21 | 1.02, 0.98, 1.21 |

* ortho_in with soft rows is 0.85 / 2.29 / 1.26× (air, con, aba).
* Without the ensemble step, concrete has no interaction above the 1% threshold, so its ortho_in is null.
* Chosen-κ diagnostics are in `logs/diag_selection_subset.log`:
  * no_shrinkage: κ = 0.01 everywhere. The variant is lattice/harmonic, except concrete, where it is lattice/zero.
  * no_ensemble_step: per-tree median κ is 1 (analytical, concrete, abalone) and 0.3 (airfoil), against 0.03–0.1
    in the full method.
* Fit time:
  * soft_orthogonality_rows is 1.4–2.8× the baseline of this machine (analytical 53 s vs 19 s, concrete 16 s
    vs 6 s). Its full-mode time was not measured.
  * All other ablations are ≤ 1.1× the full method.
