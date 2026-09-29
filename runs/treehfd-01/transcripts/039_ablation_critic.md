# ablation_critic (claude-sonnet-5-5)

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


# Role: Ablation Critic
Inspect the component breakdown of the selected method. Clean context; trace, do not recompute.

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

Official full-set results of the full method and each ablation, versus the baseline (lower is better):
| metric | baseline | full_method | w/o_no_shrinkage_direct_solver_only | w/o_no_lattice_prior_ridge_zero_only | w/o_no_ensemble_level_kappa_step | w/o_soft_orthogonality_rows |
|---|---|---|---|---|---|---|
| analytical.mse_eta1 | 0.0284 | 0.02824 (-0.6%) | 0.02829 (-0.4%) | 0.02834 (-0.2%) | 0.02846 (+0.2%) | 0.02859 (+0.7%) |
| analytical.mse_eta2 | 0.01791 | 0.01791 (-0.0%) | 0.01784 (-0.4%) | 0.01789 (-0.1%) | 0.01815 (+1.3%) | 0.01821 (+1.7%) |
| analytical.mse_eta3 | 0.01659 | 0.01646 (-0.8%) | 0.01653 (-0.4%) | 0.01651 (-0.5%) | 0.01654 (-0.3%) | 0.01681 (+1.3%) |
| analytical.mse_eta4 | 0.01618 | 0.01619 (+0.1%) | 0.01614 (-0.2%) | 0.01619 (+0.1%) | 0.01642 (+1.5%) | 0.01653 (+2.1%) |
| analytical.mse_eta5 | 0.0003006 | 0.0002894 (-3.7%) | 0.0002905 (-3.4%) | 0.0002916 (-3.0%) | 0.0002931 (-2.5%) | 0.0002911 (-3.2%) |
| analytical.mse_eta6 | 0.0003665 | 0.0003626 (-1.1%) | 0.0003654 (-0.3%) | 0.0003626 (-1.1%) | 0.0003666 (+0.0%) | 0.000361 (-1.5%) |
| analytical.mse_eta12 | 0.03118 | 0.03026 (-3.0%) | 0.03044 (-2.4%) | 0.04323 (+38.6%) | 0.03299 (+5.8%) | 0.03035 (-2.7%) |
| analytical.mse_eta34 | 0.0282 | 0.02769 (-1.8%) | 0.02766 (-1.9%) | 0.0357 (+26.6%) | 0.02996 (+6.2%) | 0.0278 (-1.4%) |
| analytical.mse_others | 0.002125 | 0.001648 (-22.4%) | 0.001896 (-10.7%) | 0.001869 (-12.0%) | 0.001388 (-34.7%) | 0.001659 (-21.9%) |
| analytical.resid_out | 0.01016 | 0.008371 (-17.6%) | 0.009126 (-10.2%) | 0.01185 (+16.6%) | 0.008556 (-15.8%) | 0.00836 (-17.7%) |
| analytical.fit_s | 17.32 | 30.88 (+78.4%) | 29.54 (+70.6%) | 16.55 (-4.4%) | 30.42 (+75.7%) | 50.83 (+193.6%) |
| abalone.resid_in | 0.004553 | 0.006338 (+39.2%) | 0.004964 (+9.0%) | 0.005898 (+29.5%) | 0.02074 (+355.4%) | 0.006167 (+35.4%) |
| abalone.resid_out | 0.04097 | 0.02094 (-48.9%) | 0.02437 (-40.5%) | 0.02702 (-34.0%) | 0.02154 (-47.4%) | 0.02029 (-50.5%) |
| abalone.ortho_in | 0.0268 | 0.02524 (-5.8%) | 0.02712 (+1.2%) | 0.02586 (-3.5%) | 0.01056 (-60.6%) | 0.03172 (+18.3%) |
| abalone.ortho_out | 0.01556 | 0.04118 (+164.6%) | 0.03701 (+137.8%) | 0.04972 (+219.5%) | 0.0525 (+237.4%) | 0.05833 (+274.8%) |
| abalone.locvar_in | 0.00256 | 0.00253 (-1.2%) | 0.002522 (-1.5%) | 0.002533 (-1.0%) | 0.002249 (-12.1%) | 0.002976 (+16.3%) |
| abalone.fit_s | 17.28 | 12.41 (-28.1%) | 13.86 (-19.7%) | 8.614 (-50.1%) | 14.68 (-15.0%) | 29.82 (+72.6%) |
| airfoil.resid_in | 0.009845 | 0.01235 (+25.5%) | 0.01012 (+2.8%) | 0.01076 (+9.3%) | 0.01526 (+55.0%) | 0.01235 (+25.5%) |
| airfoil.resid_out | 0.02691 | 0.02247 (-16.5%) | 0.02308 (-14.2%) | 0.02971 (+10.4%) | 0.02386 (-11.3%) | 0.02245 (-16.6%) |
| airfoil.ortho_in | 0.009669 | 0.008396 (-13.2%) | 0.01008 (+4.3%) | 0.007657 (-20.8%) | 0.007186 (-25.7%) | 0.007169 (-25.9%) |
| airfoil.ortho_out | 0.0969 | 0.1116 (+15.2%) | 0.1471 (+51.8%) | 0.1207 (+24.5%) | 0.1183 (+22.1%) | 0.1042 (+7.5%) |
| airfoil.locvar_in | 5.199e-05 | 4.567e-05 (-12.2%) | 4.563e-05 (-12.2%) | 5.321e-05 (+2.3%) | 4.743e-05 (-8.8%) | 4.783e-05 (-8.0%) |
| airfoil.fit_s | 3.941 | 3.911 (-0.8%) | 3.854 (-2.2%) | 2.195 (-44.3%) | 4.08 (+3.5%) | 6.483 (+64.5%) |
| bike.resid_in | 0.02134 | 0.02144 (+0.5%) | 0.02136 (+0.1%) | 0.02141 (+0.3%) | 0.02184 (+2.4%) | 0.02145 (+0.5%) |
| bike.resid_out | 0.02464 | 0.0246 (-0.2%) | 0.02465 (+0.0%) | 0.02459 (-0.2%) | 0.02461 (-0.1%) | 0.02459 (-0.2%) |
| bike.ortho_in | 0.01337 | 0.01338 (+0.0%) | 0.01336 (-0.1%) | 0.01337 (-0.0%) | 0.01345 (+0.6%) | 0.01323 (-1.1%) |
| bike.ortho_out | 0.02121 | 0.02121 (+0.0%) | 0.02121 (+0.0%) | 0.0212 (-0.1%) | 0.02124 (+0.1%) | 0.02122 (+0.1%) |
| bike.locvar_in | 0.0003446 | 0.0003258 (-5.4%) | 0.0003391 (-1.6%) | 0.0003112 (-9.7%) | 0.0002385 (-30.8%) | 0.0002792 (-19.0%) |
| bike.fit_s | 46.42 | 36.7 (-20.9%) | 38.39 (-17.3%) | 24.35 (-47.6%) | 40.41 (-13.0%) | 62.87 (+35.4%) |
| housing.resid_in | 0.01002 | 0.01112 (+10.9%) | 0.01017 (+1.5%) | 0.0107 (+6.8%) | 0.01372 (+37.0%) | 0.01123 (+12.1%) |
| housing.resid_out | 0.01544 | 0.01406 (-9.0%) | 0.01493 (-3.3%) | 0.01416 (-8.3%) | 0.01397 (-9.6%) | 0.01403 (-9.1%) |
| housing.ortho_in | 0.02798 | 0.02834 (+1.3%) | 0.02801 (+0.1%) | 0.0286 (+2.2%) | 0.02874 (+2.7%) | 0.02916 (+4.2%) |
| housing.ortho_out | 0.04883 | 0.04829 (-1.1%) | 0.04869 (-0.3%) | 0.04921 (+0.8%) | 0.04011 (-17.9%) | 0.04764 (-2.4%) |
| housing.locvar_in | 0.001796 | 0.001347 (-25.0%) | 0.001431 (-20.3%) | 0.001338 (-25.5%) | 0.00127 (-29.3%) | 0.001275 (-29.0%) |
| housing.fit_s | 99.21 | 112.5 (+13.4%) | 120.3 (+21.3%) | 69.15 (-30.3%) | 124.2 (+25.2%) | 201.1 (+102.6%) |
| concrete.resid_in | 0.001534 | 0.002164 (+41.1%) | 0.001619 (+5.6%) | 0.002013 (+31.3%) | 0.005857 (+281.8%) | 0.002422 (+57.9%) |
| concrete.resid_out | 0.02499 | 0.02104 (-15.8%) | 0.02291 (-8.3%) | 0.02072 (-17.1%) | 0.02195 (-12.2%) | 0.02086 (-16.5%) |
| concrete.ortho_in | 0.006359 | 0.006129 (-3.6%) | 0.005433 (-14.6%) | 0.007239 (+13.8%) | NA () | 0.01402 (+120.5%) |
| concrete.ortho_out | 0.08227 | 0.07164 (-12.9%) | 0.07343 (-10.8%) | 0.06886 (-16.3%) | 0.06783 (-17.6%) | 0.06973 (-15.2%) |
| concrete.locvar_in | 0.01249 | 0.01321 (+5.8%) | 0.01753 (+40.4%) | 0.008632 (-30.9%) | 0.007136 (-42.9%) | 0.01111 (-11.1%) |
| concrete.fit_s | 4.682 | 6.273 (+34.0%) | 7.003 (+49.6%) | 3.858 (-17.6%) | 7.203 (+53.8%) | 16.2 (+246.1%) |
| nutrition.resid_in | 0.01786 | 0.03009 (+68.5%) | 0.01918 (+7.4%) | 0.02514 (+40.7%) | 0.06657 (+272.8%) | 0.03004 (+68.2%) |
| nutrition.resid_out | 0.09665 | 0.07253 (-25.0%) | 0.08581 (-11.2%) | 0.07277 (-24.7%) | 0.06524 (-32.5%) | 0.07218 (-25.3%) |
| nutrition.ortho_in | 0.01622 | 0.01539 (-5.1%) | 0.0155 (-4.4%) | 0.01565 (-3.5%) | 0.0149 (-8.2%) | 0.02957 (+82.3%) |
| nutrition.ortho_out | 0.1062 | 0.08168 (-23.1%) | 0.1004 (-5.5%) | 0.08811 (-17.0%) | 0.05689 (-46.4%) | 0.08159 (-23.2%) |
| nutrition.locvar_in | 0.002818 | 0.002733 (-3.0%) | 0.002873 (+2.0%) | 0.00264 (-6.3%) | 0.002218 (-21.3%) | 0.002524 (-10.5%) |
| nutrition.fit_s | 6.56 | 11 (+67.6%) | 12.1 (+84.5%) | 6.6 (+0.6%) | 13.58 (+107.1%) | 21.84 (+232.9%) |
| parkinson.resid_in | 0.007206 | 0.007291 (+1.2%) | 0.007267 (+0.8%) | 0.007291 (+1.2%) | 0.008607 (+19.4%) | 0.007299 (+1.3%) |
| parkinson.resid_out | 0.009542 | 0.009561 (+0.2%) | 0.009515 (-0.3%) | 0.009561 (+0.2%) | 0.009813 (+2.8%) | 0.009563 (+0.2%) |
| parkinson.ortho_in | 0.08201 | 0.08204 (+0.0%) | 0.08207 (+0.1%) | 0.08204 (+0.0%) | 0.08504 (+3.7%) | 0.08274 (+0.9%) |
| parkinson.ortho_out | 0.1176 | 0.1174 (-0.2%) | 0.1173 (-0.2%) | 0.1174 (-0.2%) | 0.1221 (+3.9%) | 0.1181 (+0.4%) |
| parkinson.locvar_in | 0.00469 | 0.004348 (-7.3%) | 0.004519 (-3.7%) | 0.004348 (-7.3%) | 0.004087 (-12.9%) | 0.004384 (-6.5%) |
| parkinson.fit_s | 42.09 | 39.51 (-6.1%) | 43.63 (+3.7%) | 25.38 (-39.7%) | 44.05 (+4.7%) | 94.45 (+124.4%) |
| powerplant.resid_in | 0.0006305 | 0.0006673 (+5.8%) | 0.0006475 (+2.7%) | 0.0006678 (+5.9%) | 0.0009113 (+44.5%) | 0.0006683 (+6.0%) |
| powerplant.resid_out | 0.001104 | 0.001065 (-3.5%) | 0.001072 (-2.9%) | 0.001072 (-2.9%) | 0.001118 (+1.3%) | 0.001065 (-3.5%) |
| powerplant.ortho_in | NA | NA () | NA () | NA () | NA () | NA () |
| powerplant.ortho_out | NA | NA () | NA () | NA () | NA () | NA () |
| powerplant.locvar_in | 0.0005291 | 0.0005232 (-1.1%) | 0.0005306 (+0.3%) | 0.0005231 (-1.1%) | 0.0005161 (-2.5%) | 0.0005208 (-1.6%) |
| powerplant.fit_s | 12.1 | 29.89 (+146.9%) | 34.44 (+184.5%) | 17.88 (+47.7%) | 34.67 (+186.4%) | 45.91 (+279.3%) |
| superconduct.resid_in | 0.01009 | 0.01018 (+0.8%) | 0.01013 (+0.3%) | 0.01016 (+0.6%) | 0.01119 (+10.8%) | 0.01018 (+0.8%) |
| superconduct.resid_out | 0.009106 | 0.00909 (-0.2%) | 0.009072 (-0.4%) | 0.009096 (-0.1%) | 0.009648 (+6.0%) | 0.00909 (-0.2%) |
| superconduct.ortho_in | NA | NA () | NA () | NA () | NA () | NA () |
| superconduct.ortho_out | NA | NA () | NA () | NA () | NA () | NA () |
| superconduct.locvar_in | 0.0006625 | 0.0006501 (-1.9%) | 0.0006507 (-1.8%) | 0.0006528 (-1.5%) | 0.0006091 (-8.1%) | 0.0006459 (-2.5%) |
| superconduct.fit_s | 359 | 169.3 (-52.8%) | 187 (-47.9%) | 173.2 (-51.8%) | 187.5 (-47.8%) | 259.7 (-27.7%) |

Questions: Is the gain attributable to the proposed mechanism rather than to generic tricks (extra regularisation,
more compute, averaging)? Is any component unnecessary or harmful, so that removing it would give a simpler or better
method? Decide "good" if the breakdown is clean, otherwise "refine" and state precisely which component to remove or
change. Reply with JSON: {"decision": "good|refine", "feedback": str}


## Reply

```json
{
  "decision": "refine",
  "feedback": "The breakdown is not clean. Much of the gain comes from the solver and the deterministic unseen-cell rule, not from Good-Turing/PRESS shrinkage, and one metric family regresses.\n\n1. Attribution. The 'direct solver only' ablation (no shrinkage) already gets most of the effect. Its analytical mse_eta1-6 numbers match the full method within about 0.5%. mse_eta5 is -3.4% versus -3.7% for the full method. abalone resid_out is -40.5% versus -48.9%, and airfoil resid_out is -14.2% versus -16.5%. On these metrics the shrinkage adds a small increment. It adds more on concrete (-8.3% vs -15.8%), nutrition (-11% vs -25%) and analytical mse_others (-10.7% vs -22.4%). The abalone and concrete gains therefore come mainly from replacing lsqr and the random fallback, which the idea lists as a side-fix. They do not show that the leave-one-cell-out mechanism is what matters. The write-up should credit them accordingly, and the exact-solver step should be evaluated alone against the baseline. A run of the baseline lsqr with the deterministic fallback would separate the solver effect from the fallback effect. The ablation cannot separate them, because it bundles both.\n\n2. Lattice prior is essential and should be kept. Ridge-to-zero only makes analytical mse_eta12 +38.6%, mse_eta34 +26.6% and resid_out +16.6% versus the baseline. The zero rule for unseen cells hurts true interactions. It also degrades airfoil resid_out (+10.4%) and ortho_in on concrete. The lattice prior is what makes the method a net gain.\n\n3. Ensemble-level kappa step (step 6) is mixed, so change it rather than keep it as is. Removing it makes resid_in much worse (abalone +355%, concrete +282%, nutrition +273%) and hurts mse_eta12/34 (+6%). It also produces an NA concrete ortho_in, which looks like a numerical failure. Removing it improves analytical mse_others (-34.7% vs -22.4%), nutrition ortho_out (-46% vs -23%) and housing ortho_out (-18% vs -1%). Its objective is the summed leave-out residual, which trades away orthogonality and off-target components. Add an orthogonality term to its risk, or cap the multiplier so that resid_in stays within a bounded factor of the baseline.\n\n4. Regressions and claim mismatches in the full method:\n- ortho_out is worse on abalone (+165%) and airfoil (+15%). The idea predicted a fall.\n- resid_in is +39% on abalone, +41% on concrete and +69% on nutrition. The idea called this 'slight'.\n- fit_s is +78% on analytical, +147% on powerplant and +34% on concrete. The idea expected 0.5-1.5x. This is within the 5x limit but should be reported.\n- The full method gives no gain on bike, parkinson or superconduct resid_out (about 0%), where cells are nearly all singletons.\n\n5. The 'soft orthogonality rows' variant is helpful. Without it, ortho_in worsens on concrete (+120%), nutrition (+82%) and abalone (+18%), and fit time roughly doubles. Keep it. It is not in the idea's stated mechanism, so document it.\n\nConcrete refinements: keep the lattice prior and the orthogonality rows, retune or constrain the kappa step as above, and add the orthogonality term to the selection risk to fix the ortho_out regressions. Report the solver-only gain as a separate contribution."
}
```
