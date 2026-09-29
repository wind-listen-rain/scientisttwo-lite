# ablation_coder (claude-opus-5-5)

## Prompt

You are one agent inside an autonomous research pipeline (a reduced re-implementation of the ScientistTwo framework). Your role is given below. Do only your role's job, then stop.

## Research problem
Improve on the human state-of-the-art method **TreeHFD** (Bénard, NeurIPS 2025, "Tree Ensemble Explainability through the Hoeffding Functional Decomposition and TreeHFD Algorithm"). TreeHFD decomposes a fitted XGBoost model T(x) into an intercept, main effects η_j(x_j) and second-order interactions η_jk(x_j, x_k) that estimate the Hoeffding functional decomposition (HFD): hierarchically orthogonal components under dependent inputs, estimated only from the training sample.

## Shared files (absolute paths)
- Original paper text: <ROOT>/tasks/treehfd_paper.txt
- Original source code (READ ONLY, installed as package `treehfd`): <ROOT>/tasks/treehfd/src/treehfd/
- Benchmark description (READ ONLY): <ROOT>/bench/BENCHMARK.md
- Benchmark harness (READ ONLY): <ROOT>/bench/harness.py
- Baseline results (this machine): <ROOT>/baseline/env-22b414443084/subset.json and <ROOT>/baseline/env-22b414443084/full.json
- Python interpreter: always call `<ROOT>/.conda/python.exe` by its absolute path (a conda env with numpy, scipy, scikit-learn,
  xgboost, pandas, matplotlib, treehfd); a bare `python` may resolve to a different environment.

## Execution environment (note added when this run moved to another machine)
- Numbers are only comparable within one evaluation environment (this machine: `22b414443084`). The benchmark files are
  identical everywhere, but the analytical-case samples and the locvar_in nearest-neighbour ties differ between machines
  (real-data models and deterministic real-data metrics do not). Official result files record their environment in the
  field "eval_env"; files without it come from an earlier machine and must not be compared number-for-number with new
  results. Methods validated earlier in this run were re-run here as `subset_env-22b414443084.json` / `full_env-22b414443084.json`
  next to their original result files; use those as references.
- Shell commands run in Git Bash on Windows. Use absolute paths with forward slashes and call `<ROOT>/.conda/python.exe` explicitly.

## Red lines (any violation makes the work invalid)
1. Never modify anything under <ROOT>/bench/ or <ROOT>/tasks/. Work only inside your own working directory.
2. The evaluation protocol is fixed: the given XGBoost model, its hyper-parameters, the datasets, the splits, the sample sizes and the metrics. A method receives only the fitted model and the model's own training inputs X_train; it must never use labels, ground-truth components, test inputs during fitting, or extra data.
3. The decomposition must explain *the given model*. Do not refit, retrain, re-seed or replace the XGBoost model, and do not average decompositions of other models trained on other data or seeds (that changes the experiment, not the method).
4. Do not game the metrics: no special-casing of datasets, no hard-coded outputs, no tuning against the analytical ground truth formulas.
5. Keep compute comparable: a method whose fit time exceeds 5x the baseline on a dataset must justify it; more than 20x is not acceptable.
6. Report honestly. If something failed or was not run, say so.

When your role asks for a JSON reply, your final message must contain exactly one JSON object inside a ```json fenced block and nothing after it.


# Role: Ablation Coding Agent
Implement each ablation below as a separate method file in <ROOT>/runs/treehfd-01/ideas/S4/ablations/method_<name>.py (same interface
as method.py; import shared code from the working directory rather than duplicating it). Do not change method.py's
behaviour. Self-test each file with the subset benchmark to make sure it runs.

Ablations:
{
 "ablations": [
  {
   "name": "no_shrinkage_direct_solver_only",
   "removes": "The GT-LOCO Tikhonov shrinkage: the ridge/lattice priors, the Good-Turing/PRESS choice of kappa, and the ensemble-level kappa shift. This is the core mechanism.",
   "how": "In lib/gtloco.py restrict KAPPAS to the single smallest value (1e-2, effectively unregularised) and set SHIFTS to [0]. Every tree then takes the near-interpolating solution from the direct eigendecomposition solver, with the vectorised construction, exact orthogonality and deterministic unseen-cell rule left as they are. Comparing against the full method gives the gain from the self-supervised shrinkage alone. Comparing against the original treehfd gives the effect of the side changes (direct solver, hard orthogonality, deterministic fallback) without shrinkage. Report resid_in, resid_out, ortho_in, ortho_out, mse_others and mse_eta12/34 on the subset and the full run, and check the chosen-kappa diagnostics."
  },
  {
   "name": "no_lattice_prior_ridge_zero_only",
   "removes": "The geometry-weighted lattice (Gaussian Markov random field) prior and the harmonic-extension unseen-cell rule. Only the plain ridge prior with the zero (additive) rule remains.",
   "how": "Set PRIORS = ('ridge',) and RULES = ('zero',) in lib/gtloco.py, keeping the kappa grid, the risk criterion R, the per-tree argmin and the ensemble step. The lattice branch then never runs. Differences from the full method show whether split-geometry smoothing and harmonic filling add anything beyond simple ridge shrinkage towards the additive explanation. Rule selection can be isolated further by keeping PRIORS but forcing RULES = ('zero',). Expect mainly ortho_out, resid_out and the unseen-cell behaviour to differ."
  },
  {
   "name": "no_ensemble_level_kappa_step",
   "removes": "The ensemble-level correction (idea step 6), which picks a global kappa shift or common kappa by the risk of the leave-out residuals summed over trees. Each tree keeps its own per-tree argmin of R.",
   "how": "Force the ensemble step to choose the candidate with shift 0 by setting SHIFTS = [0] and restricting the variant choice to the per-tree variant argmin. Keep the KAPPAS grid, priors and rules unchanged. Report resid_in, resid_out and ortho_out. The round-3 notes say this step mostly buys in-sample fidelity (resid_in about 2.5-3x lower) rather than resid_out. This ablation checks that on the benchmark split and on the full datasets, and shows how much of the resid_in cost is bounded by this step."
  },
  {
   "name": "soft_orthogonality_rows",
   "removes": "The exact hierarchical-orthogonality parameterisation, which is an added side change not in the baseline or the idea. Each pair's coefficients are restricted to the null space of its orthogonality rows.",
   "how": "Set HARD_ORTHO = False in lib/gtloco.py so the baseline's soft orthogonality rows are used, and keep everything else, including the shrinkage. Compare ortho_in and ortho_out, plus resid_in and resid_out. Any gain that disappears here is due to the exact-orthogonality side change rather than to the Good-Turing shrinkage. The notes report that soft rows are traded off against the penalty, with in-sample interaction/main correlations of 0.06-0.13 versus at most 0.012 for hard rows."
  }
 ]
}

Reply with JSON: {"status": "done|failed", "files": [str], "notes": str}


## Reply

I implemented all four ablations, plus one optional extra, as separate method files in `S4/ablations/`. Each one ran through the subset benchmark with no errors, and none of them changes how `method.py` behaves.

**How they work.** Each file loads the shared code through `ablations/_shared.py`, sets some switches in `lib/gtloco.py`, and reuses `method.fit` and `method.predict` unchanged. The harness runs every method in its own process, so the switches never reach the submitted method.

**One change to `lib/gtloco.py`.** Removing the ensemble step needed a new switch, `COMMON_KAPPA`, which defaults to True (the old behaviour). I re-ran `method.py` on the subset after adding it. Every metric except fit time matches the official result on this machine, digit for digit.

**Two interpretation choices you should know about:**
- **No ensemble step:** I read "restricting the variant choice to the per-tree variant argmin" as the round-3 "per-tree" setup. There is no κ shift and no common κ, each tree keeps its own best κ, and the (prior, rule) variant is still picked by the leave-out risk summed over trees.
- **No shrinkage:** as specified, only κ = 0.01 and shift 0 remain. The (prior, rule) variant is still chosen by the leave-out risk, and at κ = 0.01 the lattice prior still acts slightly on weakly supported cells. The chosen variant was lattice/harmonic on the analytical case, airfoil and abalone, and lattice/zero on concrete.

**Subset results** (single run each, ratio to the full method; resid_in, resid_out and ortho_out columns read airfoil / concrete / abalone):

| Ablation | analytical resid_out | resid_in | resid_out | ortho_out |
|---|---|---|---|---|
| No shrinkage | 1.09 | 0.82 / 0.75 / 0.78 | 1.03 / 1.09 / 1.16 | 1.32 / 1.02 / 0.90 |
| Ridge prior + zero rule only | 1.40 | 0.87 / 0.93 / 0.93 | 1.32 / 0.98 / 1.29 | 1.08 / 0.96 / 1.21 |
| No ensemble step | 1.04 | 1.24 / 2.71 / 3.27 | 1.06 / 1.04 / 1.03 | 1.06 / 0.95 / 1.28 |
| Soft orthogonality rows | 1.00 | 1.00 / 1.12 / 0.97 | 1.00 / 0.99 / 0.97 | 0.93 / 0.97 / 1.42 |
| Zero rule only (extra) | 1.28 | 1.01 / 1.00 / 1.00 | 1.17 / 1.00 / 1.21 | 1.02 / 0.98 / 1.21 |

- **No shrinkage:** mse_others is 1.13× on the analytical case.
- **Ridge prior + zero rule only:** mse_eta12 is 1.36× and mse_eta34 is 1.24× on the analytical case.
- **No ensemble step:** each tree's median κ rises from 0.03–0.1 to 0.3–1, and concrete's ortho_in becomes null because no interaction passes the 1% variance threshold. The in-sample cost (2.7× and 3.3×) matches the round-3 notes.
- **Soft orthogonality rows:** ortho_in is 0.85 / 2.29 / 1.26×.
- **Zero rule only:** this optional extra keeps both priors but drops the harmonic rule. Its results are close to the ridge-plus-zero ablation, which suggests the loss there comes mostly from the harmonic rule, not the lattice prior.

**Fit time.** Soft orthogonality is the slow one: 1.4–2.8× the baseline on this machine (analytical 53 s vs 19 s, concrete 16 s vs 6 s). Its full-mode time is not measured. The other ablations take at most 1.1× the full method's time.

**Not run:** I did not run the full benchmark for any ablation.

Everything is in `S4/ablations/`, with a README that has the table above.

```json
{"status": "done", "files": ["<ROOT>/runs/treehfd-01/ideas/S4/ablations/method_no_shrinkage_direct_solver_only.py", "<ROOT>/runs/treehfd-01/ideas/S4/ablations/method_no_lattice_prior_ridge_zero_only.py", "<ROOT>/runs/treehfd-01/ideas/S4/ablations/method_no_ensemble_level_kappa_step.py", "<ROOT>/runs/treehfd-01/ideas/S4/ablations/method_soft_orthogonality_rows.py", "<ROOT>/runs/treehfd-01/ideas/S4/ablations/method_zero_rule_only.py", "<ROOT>/runs/treehfd-01/ideas/S4/ablations/_shared.py", "<ROOT>/runs/treehfd-01/ideas/S4/ablations/diag_selection.py", "<ROOT>/runs/treehfd-01/ideas/S4/ablations/compare_ablations.py", "<ROOT>/runs/treehfd-01/ideas/S4/ablations/run_subset.sh", "<ROOT>/runs/treehfd-01/ideas/S4/ablations/README.md", "<ROOT>/runs/treehfd-01/ideas/S4/ablations/subset_no_shrinkage_direct_solver_only.json", "<ROOT>/runs/treehfd-01/ideas/S4/ablations/subset_no_lattice_prior_ridge_zero_only.json", "<ROOT>/runs/treehfd-01/ideas/S4/ablations/subset_no_ensemble_level_kappa_step.json", "<ROOT>/runs/treehfd-01/ideas/S4/ablations/subset_soft_orthogonality_rows.json", "<ROOT>/runs/treehfd-01/ideas/S4/ablations/subset_zero_rule_only.json", "<ROOT>/runs/treehfd-01/ideas/S4/ablations/logs/diag_selection_subset.log", "<ROOT>/runs/treehfd-01/ideas/S4/ablations/logs/subset_method_recheck.json", "<ROOT>/runs/treehfd-01/ideas/S4/lib/gtloco.py"], "notes": "Each ablation file imports lib/gtloco.py and method.py through ablations/_shared.py, sets gtloco module switches at import time and reuses method.fit / method.predict unchanged. The harness runs each method in its own process, so the switches never reach the submitted method. (1) no_shrinkage_direct_solver_only: KAPPAS=[1e-2], SHIFTS=[0]. The (prior, rule) variant is still chosen by ensemble leave-out risk as specified; at kappa=0.01 the lattice prior still acts slightly on weakly identified directions. (2) no_lattice_prior_ridge_zero_only: PRIORS=('ridge',), RULES=('zero',); the lattice branch never runs. (3) no_ensemble_level_kappa_step: SHIFTS=[0] plus a new switch gtloco.COMMON_KAPPA=False, so the only candidates are the per-tree argmin under each variant, with the variant chosen by summed leave-out risk (the round-3 'per-tree' setup of dev/ablate_shift.py). (4) soft_orthogonality_rows: HARD_ORTHO=False. Extra, optional: method_zero_rule_only.py (RULES=('zero',), both priors), the rule-only isolation suggested in ablation 2. The only edit to lib/gtloco.py is COMMON_KAPPA (default True = previous candidate set). A subset re-run of method.py after the edit reproduces subset_env-22b414443084.json exactly on all metrics except fit_s (logs/subset_method_recheck.json). Subset self-tests (this machine, one run each): all 5 files complete with errors=[]. Ratio to the full method, resid_in / resid_out / ortho_out for airfoil, concrete, abalone: no_shrinkage 0.82,0.75,0.78 / 1.03,1.09,1.16 / 1.32,1.02,0.90, analytical resid_out 1.09, mse_others 1.13. ridge_zero 0.87,0.93,0.93 / 1.32,0.98,1.29 / 1.08,0.96,1.21, analytical resid_out 1.40, eta12 1.36, eta34 1.24. no_ensemble_step 1.24,2.71,3.27 / 1.06,1.04,1.03 / 1.06,0.95,1.28, analytical resid_out 1.04; per-tree median kappa 0.3-1 vs 0.03-0.1, which confirms the round-3 finding that the step mostly buys resid_in. soft_ortho 1.00,1.12,0.97 / 1.00,0.99,0.97 / 0.93,0.97,1.42, ortho_in 0.85,2.29,1.26. zero_rule_only resid_out 1.17,1.00,1.21, analytical resid_out 1.28, so the ridge_zero loss comes mostly from the harmonic rule. Chosen-kappa and variant diagnostics are in logs/diag_selection_subset.log (no_shrinkage picks lattice/harmonic except lattice/zero on concrete). Fit time: soft_orthogonality_rows is 1.4-2.8x the baseline on this machine (analytical 53s vs 19s, concrete 16s vs 6s); its full-mode time is not measured. The other ablations are <=1.1x the full method. Not done: no full-mode runs of the ablations."}
```
