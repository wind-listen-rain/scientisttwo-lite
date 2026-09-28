# subset_critic (sonnet)

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


# Role: Subset Critic
Decide whether an idea's implementation beats the reproduced baseline on the benchmark subset. You work in a clean
context: judge only from the evidence below and the files in <ROOT>/runs/treehfd-01/ideas/S4 (read method.py, lib/, NOTES.md as needed).
Trace, do not recompute: the table was produced by the official harness.

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

Official subset results (candidate vs reproduced baseline; all metrics lower is better):
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

Decision rules:
- "bad": substantially worse than the baseline on the main metrics, broken, or violates a red line (check the code:
  retraining/re-seeding the model, using extra data, special-casing datasets, touching the protocol).
- "good": consistently better than the baseline across the analytical case and the datasets, with no hidden trade-off
  that a reader would consider a regression, and compute within limits.
- "refine": shows potential but needs engineering (bugs, tuning, numerical issues). Give concrete, actionable feedback.

Reply with JSON: {"decision": "good|refine|bad", "feedback": str}


## Reply

```json
{"decision": "refine", "feedback": "Mechanism is real (self-supervised shrinkage clearly reduces resid_out: -16.8% analytical, -14.8% airfoil, -15.8% concrete, -48.5% abalone) and the engineering is solid — no red-line issues found (no labels/test data in lib/gtloco.py, no model retraining, no RNG, fit_s within limits, self-tests reproduce the official subset numbers to the printed digit, and NOTES.md documents brute-force LOO and lsqr-convergence checks). But per BENCHMARK.md's own bar ('an improvement must hold across datasets and metrics; trading one metric for another must be reported, not hidden'), this submission trades rather than uniformly improves, and one trade is large enough to need a fix before calling it a win:\n\n1. abalone.ortho_out is +172.9% (0.015→0.041), and airfoil.ortho_out is +13.0% worse — the two subset datasets where the chosen variant is lattice/harmonic. Hypothesis worth testing directly: the 'harmonic' unseen-cell rule fills unobserved (j,k) cells with a smoothed extrapolation that has no constraint to be orthogonal to the main effects under the *test* distribution, whereas 'zero' contributes exactly 0 to any correlation. So harmonic can win on resid_out while quietly breaking the very hierarchical-orthogonality property TreeHFD is supposed to deliver, especially on datasets with many unseen test cells. NOTES.md's own explanation ('one interaction at ~1.5% of Var[T], one std error for n_test=836') is asserted, not verified — run a multi-split/bootstrap check on abalone before trusting it, since this is the single largest metric regression in the whole table. If confirmed as real (not noise), bias the rule choice against 'harmonic' for pairs where a large fraction of test mass would land in unseen cells, or add an orthogonality-aware term to the risk R so selection isn't blind to this failure mode.\n\n2. resid_in rises consistently wherever resid_out falls (25–41% on the subset datasets, up to 69% on nutrition in the full run). This is disclosed, not hidden, and is the expected bias/variance trade-off — but it means the method is not a clean win, it's a redistribution. Worth explicitly bounding: is there a dataset where resid_in degrades enough to matter for downstream use of the decomposition, and is the always-more-aggressive ensemble-level κ shift (−1..−3 steps, always toward less regularization than the per-tree choice) actually calibrated, or just a fixed correction that happens to work on this dataset mix? An ablation (with/without the ensemble shift) would clarify whether that step is pulling its weight or just re-adding back some of the overfitting the per-tree step removed.\n\nRecommend: keep the resid_out mechanism, but before promoting to 'good' (a) verify the abalone ortho_out result isn't a real, harmonic-rule-specific defect via a multi-seed check, and (b) either fix or clearly bound the ortho_out/resid_in trade-off so it can't blow up unnoticed on datasets outside this subset."}
```
