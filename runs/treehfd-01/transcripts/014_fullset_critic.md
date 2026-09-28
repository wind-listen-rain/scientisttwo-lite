# fullset_critic (sonnet)

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


# Role: Full-Set Critic
Decide whether the idea's implementation beats the original state of the art on the *full* benchmark (analytical
case with 10 repetitions and all 9 datasets). Clean context; trace, do not recompute. Files: <ROOT>/runs/treehfd-01/ideas/S4.

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

Official full-set results (candidate vs reproduced baseline; lower is better):
| metric | baseline | S4 |
|---|---|---|
| analytical.mse_eta1 | 0.02886 | 0.02868 (-0.6%) |
| analytical.mse_eta2 | 0.0184 | 0.01843 (+0.2%) |
| analytical.mse_eta3 | 0.01585 | 0.01589 (+0.3%) |
| analytical.mse_eta4 | 0.01743 | 0.01742 (-0.0%) |
| analytical.mse_eta5 | 0.0004581 | 0.0004438 (-3.1%) |
| analytical.mse_eta6 | 0.0003432 | 0.0003352 (-2.3%) |
| analytical.mse_eta12 | 0.03289 | 0.03209 (-2.4%) |
| analytical.mse_eta34 | 0.03169 | 0.03118 (-1.6%) |
| analytical.mse_others | 0.00205 | 0.001629 (-20.5%) |
| analytical.resid_out | 0.01004 | 0.008483 (-15.5%) |
| analytical.fit_s | 11.37 | 11.1 (-2.4%) |
| abalone.resid_in | 0.004553 | 0.006338 (+39.2%) |
| abalone.resid_out | 0.04125 | 0.02094 (-49.2%) |
| abalone.ortho_in | 0.0268 | 0.02524 (-5.8%) |
| abalone.ortho_out | 0.01492 | 0.04118 (+176.0%) |
| abalone.locvar_in | 0.002774 | 0.002738 (-1.3%) |
| abalone.fit_s | 11.07 | 6.126 (-44.7%) |
| airfoil.resid_in | 0.009845 | 0.01235 (+25.5%) |
| airfoil.resid_out | 0.0269 | 0.02247 (-16.4%) |
| airfoil.ortho_in | 0.00967 | 0.008396 (-13.2%) |
| airfoil.ortho_out | 0.09832 | 0.1116 (+13.5%) |
| airfoil.locvar_in | 5.199e-05 | 4.567e-05 (-12.2%) |
| airfoil.fit_s | 1.824 | 1.893 (+3.8%) |
| bike.resid_in | 0.02134 | 0.02144 (+0.5%) |
| bike.resid_out | 0.02464 | 0.0246 (-0.2%) |
| bike.ortho_in | 0.01337 | 0.01338 (+0.0%) |
| bike.ortho_out | 0.02121 | 0.02121 (+0.0%) |
| bike.locvar_in | 0.0003441 | 0.0003254 (-5.4%) |
| bike.fit_s | 31.04 | 16.99 (-45.3%) |
| housing.resid_in | 0.01002 | 0.01112 (+10.9%) |
| housing.resid_out | 0.01541 | 0.01406 (-8.8%) |
| housing.ortho_in | 0.02797 | 0.02834 (+1.3%) |
| housing.ortho_out | 0.04883 | 0.04829 (-1.1%) |
| housing.locvar_in | 0.00165 | 0.001238 (-25.0%) |
| housing.fit_s | 74.4 | 44.27 (-40.5%) |
| concrete.resid_in | 0.001534 | 0.002164 (+41.1%) |
| concrete.resid_out | 0.02501 | 0.02104 (-15.9%) |
| concrete.ortho_in | 0.006361 | 0.006129 (-3.6%) |
| concrete.ortho_out | 0.08172 | 0.07164 (-12.3%) |
| concrete.locvar_in | 0.01249 | 0.01322 (+5.8%) |
| concrete.fit_s | 3.249 | 3.121 (-4.0%) |
| nutrition.resid_in | 0.01786 | 0.03009 (+68.5%) |
| nutrition.resid_out | 0.09573 | 0.07253 (-24.2%) |
| nutrition.ortho_in | 0.01622 | 0.01539 (-5.1%) |
| nutrition.ortho_out | 0.1079 | 0.08168 (-24.3%) |
| nutrition.locvar_in | 0.00287 | 0.002797 (-2.5%) |
| nutrition.fit_s | 4.814 | 4.852 (+0.8%) |
| parkinson.resid_in | 0.007206 | 0.007291 (+1.2%) |
| parkinson.resid_out | 0.009537 | 0.009561 (+0.3%) |
| parkinson.ortho_in | 0.08201 | 0.08204 (+0.0%) |
| parkinson.ortho_out | 0.1176 | 0.1174 (-0.2%) |
| parkinson.locvar_in | 0.004694 | 0.004354 (-7.3%) |
| parkinson.fit_s | 33.06 | 17.3 (-47.7%) |
| powerplant.resid_in | 0.0006305 | 0.0006673 (+5.8%) |
| powerplant.resid_out | 0.001104 | 0.001065 (-3.5%) |
| powerplant.ortho_in | NA | NA () |
| powerplant.ortho_out | NA | NA () |
| powerplant.locvar_in | 0.0005368 | 0.000531 (-1.1%) |
| powerplant.fit_s | 7.665 | 11.65 (+52.0%) |
| superconduct.resid_in | 0.01009 | 0.01018 (+0.8%) |
| superconduct.resid_out | 0.009105 | 0.00909 (-0.2%) |
| superconduct.ortho_in | NA | NA () |
| superconduct.ortho_out | NA | NA () |
| superconduct.locvar_in | 0.0006632 | 0.0006508 (-1.9%) |
| superconduct.fit_s | 701.3 | 112.4 (-84.0%) |

Paper-reported TreeHFD numbers for reference: analytical η(1)=0.02, η(1,2)=0.04, η(3,4)=0.04; real-data residual about
1% of Var[T(x)], orthogonality about 0.05 or smaller.

Rules as for the subset critic, but "good" requires the improvement to hold on the full benchmark, not only on the
subset. Name every metric/dataset where the candidate is worse.
Reply with JSON: {"decision": "good|refine|bad", "feedback": str}


## Reply

I traced the official numbers against `full_0.json` and `baseline/full.json` — they match the table exactly, and `NOTES.md` documents the design (fixed, non-per-dataset hyperparameters; self-supervised risk from `X_train`/tree outputs only; no labels or test data used) with no red-line violations. Fit-time ratios are all ≤1.52× baseline (powerplant), well under the 5× flag, with most datasets much faster (down to −84% on superconduct).

```json
{"decision": "good", "feedback": "Verified full_0.json against the table and against baseline/full.json — all values match. The primary generalization metrics win broadly and are not gamed (fixed global hyperparameters, self-supervised selection from X_train/tree outputs only, no per-dataset tuning). Analytical: resid_out -15.5%, mse_others -20.5%, mse_eta12/34 -2.4%/-1.6%, mse_eta5/6 -3.1%/-2.3%, fit_s -2.4%, with mse_eta1/4 flat; only mse_eta2 (+0.2%) and mse_eta3 (+0.3%) are worse, both inside noise (baseline std ~0.005-0.006). Real-data resid_out improves on 6/9 datasets substantially (abalone -49.2%, nutrition -24.2%, airfoil -16.4%, concrete -15.9%, housing -8.8%, powerplant -3.5%) and is flat within noise on bike (-0.2%), parkinson (+0.3%) and superconduct (-0.2%). Every metric/dataset where the candidate is worse: resid_in rises on 8/9 real datasets (nutrition +68.5%, concrete +41.1%, abalone +39.2%, airfoil +25.5%, housing +10.9%, powerplant +5.8%, parkinson +1.2% negligible, bike +0.5% negligible; superconduct +0.8% negligible) — this is the reported, bounded shrinkage trade-off, and stays below the method's own resid_out except superconduct, where baseline shows the same pattern. ortho_out is worse on abalone (+176.0%, the largest regression in the table) and airfoil (+13.5%); it improves or is flat on concrete (-12.3%), nutrition (-24.3%), housing (-1.1%), bike (~0%) and parkinson (~0%). ortho_in is worse on housing (+1.3%, negligible). locvar_in is worse on concrete (+5.8%, minor). fit_s is worse on airfoil (+3.8%, negligible) and powerplant (+52.0%, notable but far under the 5x/20x compute red lines). NOTES.md's round-3 diagnostics (8 splits never used by the method, plus a test-set bootstrap) support that the abalone ortho_out spike is mostly split-level noise in a high-variance max-|corr| metric on 836 test points (P(candidate>baseline)=0.54 under bootstrap; ratio ~0.98 geo-mean on the other 7 splits) rather than a systematic defect of the harmonic rule, and that the resid_in rise is an explicit, monitorable cost of the shrinkage that buys the resid_out gain. On the full benchmark the candidate beats the reproduced TreeHFD baseline on the paper's central quality metric (resid_out / mse_*) across essentially every dataset and the analytical case, with honestly reported, bounded trade-offs and no red-line violations."}
```
