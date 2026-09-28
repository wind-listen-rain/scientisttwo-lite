# engineer (opus)

## Prompt

You are one agent inside an autonomous research pipeline (a reduced re-implementation of the ScientistTwo framework). Your role is given below. Do only your role's job, then stop.

## Research problem
Improve on the human state-of-the-art method **TreeHFD** (Bénard, NeurIPS 2025, "Tree Ensemble Explainability through the Hoeffding Functional Decomposition and TreeHFD Algorithm"). TreeHFD decomposes a fitted XGBoost model T(x) into an intercept, main effects η_j(x_j) and second-order interactions η_jk(x_j, x_k) that estimate the Hoeffding functional decomposition (HFD): hierarchically orthogonal components under dependent inputs, estimated only from the training sample.

## Shared files (absolute paths)
- Original paper text: <ROOT>/tasks/treehfd_paper.txt
- Original source code (READ ONLY, installed as package `treehfd`): <ROOT>/tasks/treehfd/src/treehfd/
- Benchmark description (READ ONLY): <ROOT>/bench/BENCHMARK.md
- Benchmark harness (READ ONLY): <ROOT>/bench/harness.py
- Baseline results: <ROOT>/baseline/subset.json and <ROOT>/baseline/full.json
- Python interpreter: always call `<ROOT>/.conda/bin/python` by its absolute path (a conda env with numpy, scipy, scikit-learn,
  xgboost, pandas, matplotlib, treehfd); a bare `python` may resolve to a different environment.

## Red lines (any violation makes the work invalid)
1. Never modify anything under <ROOT>/bench/ or <ROOT>/tasks/. Work only inside your own working directory.
2. The evaluation protocol is fixed: the given XGBoost model, its hyper-parameters, the datasets, the splits, the sample sizes and the metrics. A method receives only the fitted model and the model's own training inputs X_train; it must never use labels, ground-truth components, test inputs during fitting, or extra data.
3. The decomposition must explain *the given model*. Do not refit, retrain, re-seed or replace the XGBoost model, and do not average decompositions of other models trained on other data or seeds (that changes the experiment, not the method).
4. Do not game the metrics: no special-casing of datasets, no hard-coded outputs, no tuning against the analytical ground truth formulas.
5. Keep compute comparable: a method whose fit time exceeds 5x the baseline on a dataset must justify it; more than 20x is not acceptable.
6. Report honestly. If something failed or was not run, say so.

When your role asks for a JSON reply, your final message must contain exactly one JSON object inside a ```json fenced block and nothing after it.


# Role: Engineering Agent
Improve the implementation in <ROOT>/runs/treehfd-01/ideas/S4 following the critic's feedback below: fix bugs, tune the method's own
hyper-parameters, or adjust the implementation — without changing the core idea and within the red lines.
Self-test with `<ROOT>/.conda/bin/python <ROOT>/bench/harness.py --method method.py --mode subset --out subset_selftest.json`.
Update NOTES.md with what you changed and why.

Idea:
{
 "title": "Good-Turing-calibrated leave-one-cell-out shrinkage (GT-LOCO TreeHFD): self-supervised per-tree regularisation, a Bayesian unseen-cell rule and a direct solver",
 "addresses": [
  "No shrinkage on sparsely populated cells causes large in-sample/out-of-sample residual gaps",
  "Prediction-time fallback for unseen interaction cells is non-deterministic and uses a distance metric that ignores actual split geometry",
  "Least-squares solver convergence is never checked, risking silently under-converged tree coefficients",
  "Spurious interaction candidates are never pruned, leaking noise into components that should be exactly zero"
 ],
 "mechanism": "The in-sample to out-of-sample residual gap (concrete 0.0015 to 0.025) arises because the per-tree least squares, whose residual rows are weighted sqrt(n*n_c), interpolates the tree output on every observed full joint cell, including cells supported by a single point. Thinly supported interaction coefficients absorb the tree's order-3-and-higher content and do not transfer to test joint cells, many of which never appeared in training. GT-LOCO adds a hierarchical Gaussian prior (a Tikhonov penalty) that shrinks interaction-cell coefficients toward zero, i.e. toward the additive explanation. Because the data weight scales with n*n_c, the prior automatically shrinks sparse cells far more than well-populated ones. An optional prior adds a geometry-weighted first-difference penalty between adjacent cells on the (j,k) bin lattice, weighted by actual bin-midpoint distances from split_list rather than index distance. The penalty strength is not a tuned hyper-parameter. It is chosen per tree by an out-of-sample risk estimate built only from X_train and the tree's own outputs: closed-form leave-one-joint-cell-out residuals r_c/(1-h_cc) (PRESS) on singleton cells, mixed with in-sample error on repeated cells, weighted by the Good-Turing estimate N1/n of the probability that a new point lands in an unseen joint cell. The estimate is R(lambda) = (1 - N1/n) * sum_{n_c>=2} w_c r_c^2 + (N1/n) * mean_{n_c=1} [r_c/(1-h_cc)]^2. At lambda near 0, interaction cells supported only by singletons have h_cc near 1, so R blows up exactly where the baseline overfits. The same prior gives a deterministic, principled rule for unseen (j,k) cells at predict time: the cell takes its prior/posterior value (zero under the ridge prior, harmonic extension under the lattice prior). This replaces the unseeded random L1-index fallback. R also chooses between the two rules, because leaving out a singleton simulates meeting an unseen pair cell. Eigendecomposition-based direct solves replace lsqr, removing the unchecked-convergence issue. This is per-tree coefficient regularisation with self-supervised selection, distinct from S1 (ensemble-level re-projection) and S2 (measure smoothing).",
 "implementation_sketch": "(1) Per tree, reuse treehfd's partition code. Build the ortho, mean and residual blocks as in optimization_matrix.py, vectorised with np.unique/np.bincount instead of Python loops, and keep the joint-cell count n_c of each residual row. (2) Penalty P: 1 on interaction columns and a tiny epsilon on main-effect columns. Optionally add lattice first-difference rows weighted by 1/|bin-midpoint distance| from split_list, with the unobserved lattice cells as latent columns. (3) Whiten and eigendecompose: P^{-1/2} A'A P^{-1/2} = U S U' (numpy eigh; at most about 600 columns per tree). For a log grid of about 10 lambdas (1e-6 to 1e2 times mean(S)), plus the baseline lsqr solution as a candidate, compute beta(lambda) = P^{-1/2} U (S+lambda)^{-1} U' P^{-1/2} A'b. For residual rows only, compute h_cc = ||(S+lambda)^{-1/2} U' P^{-1/2} a_c||^2; each a_c has at most |V|+|I| nonzeros. (4) Evaluate R(lambda) with unweighted residuals r_c = y_c - recon_c and Good-Turing weights; pick the argmin, breaking ties toward smaller lambda. (5) Store beta and the chosen unseen-cell rule. At predict time use np.digitize for bins and the deterministic prior value for unseen pair cells, with no RNG. (6) Optionally choose one global multiplier on all per-tree lambdas by the same risk computed on summed per-point leave-out residuals across trees, which accounts for correlated errors between trees. (7) Log the distribution of chosen lambdas and the resulting resid_in increase for honest reporting.",
 "expected_effect_on_metrics": "resid_out should fall on all real datasets, most on concrete and abalone, whose in/out gaps are 16x and 9x. resid_in rises slightly; this is an explicit, bounded trade-off because lambda near 0 is always a candidate. ortho_out and mse_others should fall, because thin spurious interaction cells get shrunk. mse_eta12/mse_eta34 should be roughly unchanged, since the true interactions have well-populated cells with low leverage. resid_out and ortho_out become reproducible across runs because the random fallback is gone. Expected fit_s is 0.5-1.5x baseline, since vectorised construction and small eigendecompositions replace Python loops and lsqr.",
 "risks": "PRESS removes only the held-out residual row, not that point's contribution to the ortho and mean rows, so the leave-out estimate is slightly optimistic. Good-Turing on joint cells degenerates when almost all cells are singletons (high p: parkinson, superconduct); R then reduces to pure leave-one-out, which still matches test-time extrapolation. Per-tree selection ignores error correlation across trees; step 6 mitigates this. Shrinking interactions reallocates variance to main effects through the ortho rows and may slightly raise ortho_in. Gains may be modest if resid_out is dominated by genuine higher-order tree content in seen cells rather than by overfitting of thin cells. Untested.",
 "id": "S4",
 "novelty": 6,
 "closest": [
  {
   "title": "Hierarchical Shrinkage: Improving the Accuracy and Interpretability of Tree-Based Methods (Agarwal, Tan, Ronen, Singh, Yu; ICML 2022, arXiv:2202.00858)",
   "url": "https://arxiv.org/abs/2202.00858",
   "overlap": "Core mechanism match: post-hoc shrinkage of tree-derived estimates toward a coarser (ancestor/additive) explanation, with shrinkage strength governed by the number of samples supporting each node/cell — exactly the 'thin cells shrink more' logic GT-LOCO applies to interaction coefficients. Differs in target (raw node predictions vs. per-tree functional-decomposition coefficients) and in how the single strength parameter is chosen (CV/heuristic vs. this idea's Good-Turing/PRESS risk estimate)."
  },
  {
   "title": "Good-Turing Frequency Estimation (I.J. Good, 1953; Gale & Sampson, 'Good-Turing Smoothing Without Tears', 1995) together with Golub, Heath & Wahba's Generalized Cross-Validation for ridge parameters (1979)",
   "url": "https://en.wikipedia.org/wiki/Good%E2%80%93Turing_frequency_estimation",
   "overlap": "Supplies the two statistical primitives the idea fuses: N1/n as the missing-probability-mass estimate for unseen categories, and the hat-matrix PRESS/(1-h_cc) leave-one-out shortcut for selecting a ridge penalty without refitting. GT-LOCO's risk functional R(lambda) is essentially a bespoke convex combination of these two textbook estimators, applied to joint tree-leaf cells rather than the word- or class-frequency tables these methods were built for."
  }
 ]
}

Current official results versus the baseline:
| metric | baseline | S4 |
|---|---|---|
| analytical.mse_eta1 | 0.02866 | 0.02805 (-2.1%) |
| analytical.mse_eta2 | 0.01977 | 0.01973 (-0.2%) |
| analytical.mse_eta3 | 0.01633 | 0.01641 (+0.5%) |
| analytical.mse_eta4 | 0.0166 | 0.01663 (+0.2%) |
| analytical.mse_eta5 | 0.000497 | 0.0004882 (-1.8%) |
| analytical.mse_eta6 | 0.0003735 | 0.0003856 (+3.2%) |
| analytical.mse_eta12 | 0.03513 | 0.03446 (-1.9%) |
| analytical.mse_eta34 | 0.03203 | 0.03108 (-3.0%) |
| analytical.mse_others | 0.002181 | 0.001678 (-23.1%) |
| analytical.resid_out | 0.01032 | 0.008586 (-16.8%) |
| analytical.fit_s | 11.2 | 10.76 (-4.0%) |
| airfoil.resid_in | 0.009845 | 0.01235 (+25.5%) |
| airfoil.resid_out | 0.02639 | 0.02247 (-14.8%) |
| airfoil.ortho_in | 0.00967 | 0.008396 (-13.2%) |
| airfoil.ortho_out | 0.09874 | 0.1116 (+13.0%) |
| airfoil.locvar_in | 5.199e-05 | 4.567e-05 (-12.2%) |
| airfoil.fit_s | 1.82 | 1.894 (+4.1%) |
| concrete.resid_in | 0.001534 | 0.002164 (+41.1%) |
| concrete.resid_out | 0.025 | 0.02104 (-15.8%) |
| concrete.ortho_in | 0.006361 | 0.006129 (-3.6%) |
| concrete.ortho_out | 0.08218 | 0.07164 (-12.8%) |
| concrete.locvar_in | 0.01249 | 0.01322 (+5.8%) |
| concrete.fit_s | 3.266 | 3.127 (-4.3%) |
| abalone.resid_in | 0.004553 | 0.006338 (+39.2%) |
| abalone.resid_out | 0.04062 | 0.02094 (-48.5%) |
| abalone.ortho_in | 0.0268 | 0.02524 (-5.8%) |
| abalone.ortho_out | 0.01509 | 0.04118 (+172.9%) |
| abalone.locvar_in | 0.002774 | 0.002738 (-1.3%) |
| abalone.fit_s | 11.07 | 6.064 (-45.2%) |

Critic feedback:
Mechanism is real (self-supervised shrinkage clearly reduces resid_out: -16.8% analytical, -14.8% airfoil, -15.8% concrete, -48.5% abalone) and the engineering is solid — no red-line issues found (no labels/test data in lib/gtloco.py, no model retraining, no RNG, fit_s within limits, self-tests reproduce the official subset numbers to the printed digit, and NOTES.md documents brute-force LOO and lsqr-convergence checks). But per BENCHMARK.md's own bar ('an improvement must hold across datasets and metrics; trading one metric for another must be reported, not hidden'), this submission trades rather than uniformly improves, and one trade is large enough to need a fix before calling it a win:

1. abalone.ortho_out is +172.9% (0.015→0.041), and airfoil.ortho_out is +13.0% worse — the two subset datasets where the chosen variant is lattice/harmonic. Hypothesis worth testing directly: the 'harmonic' unseen-cell rule fills unobserved (j,k) cells with a smoothed extrapolation that has no constraint to be orthogonal to the main effects under the *test* distribution, whereas 'zero' contributes exactly 0 to any correlation. So harmonic can win on resid_out while quietly breaking the very hierarchical-orthogonality property TreeHFD is supposed to deliver, especially on datasets with many unseen test cells. NOTES.md's own explanation ('one interaction at ~1.5% of Var[T], one std error for n_test=836') is asserted, not verified — run a multi-split/bootstrap check on abalone before trusting it, since this is the single largest metric regression in the whole table. If confirmed as real (not noise), bias the rule choice against 'harmonic' for pairs where a large fraction of test mass would land in unseen cells, or add an orthogonality-aware term to the risk R so selection isn't blind to this failure mode.

2. resid_in rises consistently wherever resid_out falls (25–41% on the subset datasets, up to 69% on nutrition in the full run). This is disclosed, not hidden, and is the expected bias/variance trade-off — but it means the method is not a clean win, it's a redistribution. Worth explicitly bounding: is there a dataset where resid_in degrades enough to matter for downstream use of the decomposition, and is the always-more-aggressive ensemble-level κ shift (−1..−3 steps, always toward less regularization than the per-tree choice) actually calibrated, or just a fixed correction that happens to work on this dataset mix? An ablation (with/without the ensemble shift) would clarify whether that step is pulling its weight or just re-adding back some of the overfitting the per-tree step removed.

Recommend: keep the resid_out mechanism, but before promoting to 'good' (a) verify the abalone ortho_out result isn't a real, harmonic-rule-specific defect via a multi-seed check, and (b) either fix or clearly bound the ortho_out/resid_in trade-off so it can't blow up unnoticed on datasets outside this subset.

Reply with JSON: {"status": "done|failed", "changes": str}


## Reply

I kept the method's output exactly as submitted. The critic's harmonic-rule hypothesis doesn't hold up on the data, and the only resid_in fix I found trades away resid_out. The subset self-test reproduces every official metric to all printed digits, and the full-mode run matches the previous full run exactly apart from timing.

- **Abalone ortho_out (+173%) looks like split noise.** On 7 alternative 80/20 splits (used only for diagnosis) the ratio to baseline is 0.86–1.07, geo-mean 0.98. A 300-resample bootstrap of the benchmark test set gives P(GT-LOCO worse than baseline) = 0.54. The baseline's 0.016 there is its own lowest value across the 8 splits.
- **The harmonic rule isn't the cause.** At the same κ, switching to the zero rule gives *higher* ortho_out on that split (0.050 vs 0.041). Over 8 splits of 4 datasets, ortho_out geo-means are 0.89–1.00× baseline (1.11× for abalone if the outlier split is included). So I added no orthogonality penalty; the benchmark-split number stays worse and is reported as such.
- **Ensemble κ shift ablation.** Compared with per-tree κ only, the shift barely moves resid_out (0.98–1.07×) but cuts resid_in to 0.32–0.79×. Without it, resid_in would be about 4.5× baseline on abalone and 3.8× on concrete. The critic's suspicion is partly right: on nutrition and abalone the shift costs 3.5–7% resid_out and about 20% ortho_out. I kept it because it is what keeps resid_in bounded.
- **The risk estimate R ranks candidates but its level isn't calibrated.** Its ratio to held-out resid_out is about 1.0 on concrete, 1.2 on airfoil, 1.9 on nutrition and 2.2 on abalone.
- **One-standard-error rule (tested, left off).** Picking the best in-sample candidate within one SE of the minimum lowers resid_in to 0.84–0.99× but raises resid_out 1.00–1.06×. It's a pure trade, so `SE_RULE` defaults to 0, which is exactly the old selection.
- **Label-free reporting of the trade-off (the one code change).** After fitting, the method now reports its own in-sample residual and estimated risk, computed from X_train only. The in-sample residual equals the harness's resid_in. Across all runs, resid_in is 1.00–2.11× baseline, at most 0.030 of Var[T] (nutrition).

These decisions (keeping the shift, leaving the SE rule off) were informed by held-out numbers from the diagnostic splits. Everything is written up in `NOTES.md`, with logs in `dev/logs/`.

```json
{"status": "done", "changes": "The decomposition itself is unchanged: the subset self-test reproduces every official metric to all printed digits (compare.py ratio 1.000; fit_s within 1%), and the full-mode rerun (no errors) matches the previous full run exactly on every metric except fit_s (0.16-1.53x baseline). (a) The critic's abalone ortho_out check: new dev/diag_ortho_out.py compares baseline and GT-LOCO on 8 80/20 splits (split 0 = benchmark; splits 1-7 for diagnosis only), bootstraps the test set and swaps the unseen-cell rule at fixed kappa. Abalone ortho_out ratio on splits 1-7 is 0.86-1.07 (geo-mean 0.98). Bootstrap on the benchmark split: baseline 0.072+-0.044, GT-LOCO 0.069+-0.048, P(GT-LOCO > baseline) = 0.54. The harmonic hypothesis is not supported: at the same kappa the zero rule gives higher ortho_out on that split (0.050 vs 0.041). Geo-means over 8 splits: airfoil 0.89, concrete 0.96, nutrition 1.00, abalone 1.11 (0.98 without the benchmark split). So no orthogonality penalty or rule bias was added; the benchmark-split number is still reported as worse. (c) Ensemble-shift ablation (dev/ablate_shift.py, 8 splits x 4 datasets): relative to per-tree kappa only, the shift leaves resid_out at 0.98-1.07x and cuts resid_in to 0.32-0.79x (without it resid_in would be about 4.5x baseline on abalone and 3.8x on concrete), at a cost of about 20% ortho_out on abalone and nutrition. It is kept. R ranks candidates but its level is not calibrated (R / held-out resid_out is 1.0-2.2). (b) Added a k-SE rule towards in-sample fidelity (SE_RULE). With k=1, resid_in falls to 0.84-0.99x but resid_out rises to 1.00-1.06x (dev/compare_se.py), a pure trade, so the default is 0 (exactly the old argmin). Code change in lib/gtloco.py: tree.fit also returns in-sample residuals along the kappa path; each ensemble candidate records its label-free in-sample fidelity; diagnostics gain risk_over_var, resid_in_over_var (equal to the harness resid_in) and min_risk, so the resid_in cost is visible from X_train alone. Measured bound: resid_in is 1.00-2.11x baseline over the 9 full datasets and 32 split runs, at most 0.030 of Var[T]. dev/check_loo.py and dev/profile_trees.py were updated for the new signature; logs are in dev/logs/. NOTES.md has a new round-3 section plus updated hyper-parameter, trade-off and file sections. Held-out numbers from the diagnostic splits informed the keep/leave-off decisions, and this is stated in NOTES.md."}
```
