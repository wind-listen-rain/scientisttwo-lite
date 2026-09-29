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
