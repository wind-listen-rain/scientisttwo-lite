# coder (opus)

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


# Role: Idea Coding Agent
Implement the research idea below as a method in your working directory <ROOT>/runs/treehfd-01/ideas/S4.
- `method.py` already contains a working copy of the baseline; `lib/treehfd_mod/` is a modifiable copy of the original
  package (imported by method.py). Change whatever you need inside <ROOT>/runs/treehfd-01/ideas/S4.
- Keep the interface in <ROOT>/bench/BENCHMARK.md exactly.
- Self-test with the subset benchmark: `<ROOT>/.conda/bin/python <ROOT>/bench/harness.py --method method.py --mode subset --out subset_selftest.json`
  and iterate until the method runs without errors. The official evaluation is re-run by the pipeline afterwards.
- Write NOTES.md: what you implemented, every design decision and hyper-parameter, and your self-test numbers.

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
 ],
 "note": "A previous implementation attempt of this idea was interrupted midway; its partial files are in the working directory (see `git log`). Review them, then complete or redo."
}

Reply with JSON: {"status": "implemented|failed", "summary": str, "selftest": str}


## Reply

I finished the GT-LOCO method in `lib/gtloco.py`, and it runs without errors in both the subset self-test and a full-mode run (analytical case plus all nine datasets). resid_out improved or held everywhere, but resid_in rose and ortho_out got worse on abalone.

**What I kept from the interrupted attempt and what I added:**
- **Kept:** the vectorised build of the TreeHFD system, exact per-tree hierarchical orthogonality, the κ grid, the closed-form leave-out risk R with Good-Turing weights, and the ensemble-level shift/common-κ step.
- **Added the lattice prior as the idea describes it:** it covers the whole (j,k) bin lattice, with never-observed cells as latent columns eliminated in closed form. Unseen cells then get their harmonic extension at predict time, and the leave-out simulates that rule exactly. The earlier "nearest cell" rule is still in the code but is not a candidate, because R never picked it on the three subset datasets.
- **Speed:** I vectorised the leave-out step and replaced the sparse transforms with block-wise dense ones.

**Checks:**
- The closed-form leave-out is within about 5–10% of brute-force refits.
- The rewritten parts reproduce the old numbers exactly.
- The baseline's lsqr hits its iteration limit on 3 of 100 concrete trees, but tighter tolerances change nothing. So the direct solver removes a risk but is not where the gains come from.

**Subset self-test (new / baseline):**

| Task | resid_out | resid_in | ortho_out | fit time |
|---|---|---|---|---|
| analytical | 0.83× | – | – | 0.97× |
| airfoil | 0.85× | 1.25× | 1.13× (worse) | 1.05× |
| concrete | 0.84× | 1.41× | 0.87× | 0.96× |
| abalone | 0.52× | 1.39× | 2.73× (0.015 → 0.041, worse) | 0.55× |

On the analytical case mse_others is 0.77×, and the true interactions eta12/eta34 are slightly better (0.97–0.98×).

**Full mode:**
- **resid_out** improved on housing (0.91×) and nutrition (0.76×); powerplant 0.97×.
- **No change** on bike, parkinson or superconduct: every metric is within ±1%, except locvar_in, which is 0.93–0.98×. R chooses only mild shrinkage there, so their error seems to come from higher-order tree content the decomposition can't represent — the idea named this risk.
- **resid_in** rose most on nutrition (1.69×).
- **Fit time** is 0.16–1.05× baseline, except powerplant at 1.53×; all are well under the 5× limit.

**Deviations from the idea:**
- **Main-effect ridge:** I kept the weight at 1 instead of a tiny ε. R couldn't tell the values apart (under 3% difference), and a tiny ε raised in-sample orthogonality error on airfoil from 0.008 to 0.026.
- **Lattice distances:** these use quantile coordinates rather than raw split distances, because edge bins are unbounded and raw distances depend on units.
- **No lsqr-solution candidate:** the smallest κ (0.01, almost unregularised) takes the place of the baseline lsqr solution.
- **Exact orthogonality:** this is an addition not in the idea. Without it, the penalty raises in-sample interaction/main correlations to 0.06–0.13.

The abalone ortho_out comes from one small interaction whose test correlation (about 0.04) is roughly one standard error at this test size. I read it as noise in the metric, but it is still a worse number and NOTES.md reports it. The dev scripts print held-out numbers next to R, so I saw them. This round's choices were made from R or label-free in-sample checks. The interrupted attempt's saved runs compared configurations on the subset benchmark, so some inherited settings may have been influenced by held-out results.

Files are in `<ROOT>/runs/treehfd-01/ideas/S4`:
- NOTES.md
- subset_selftest.json
- full_selftest.json
- lib/gtloco.py

```json
{"status": "implemented", "summary": "GT-LOCO TreeHFD in lib/gtloco.py (method.py interface unchanged). Per tree: vectorised TreeHFD normal equations; exact hierarchical orthogonality via per-pair null-space parameterisation (addition not in the idea); Tikhonov penalty lambda = kappa/n with two priors: ridge (P=I on pair cells, EPS_MAIN=1 on main bins; deviation from the idea's tiny epsilon because R was flat within 3% and a tiny eps raised in-sample ortho on airfoil) and a lattice GMRF over the full (j,k) bin lattice (4-neighbour first differences weighted 1/quantile-distance, plus ridge 0.1) with unobserved cells as latent columns eliminated by Schur complement; one eigendecomposition per prior gives the whole 12-value kappa grid (1e-2..1e4; no lsqr). Label-free risk R = in-sample residuals for points in repeated joint cells plus closed-form PRESS r/(1-h) for Good-Turing singletons (exactly the idea's mixture), with the unseen-cell rule (zero = additive prior mean, or harmonic = lattice posterior mean) simulated for singletons whose cell empties. Per-tree argmin, then an ensemble-level shift or common kappa chosen by R on per-point leave-out residuals summed over trees. Unseen pair cells are deterministic at predict time (no RNG). Checks: closed-form leave-out within about 5-10% of brute-force refits; vectorised and block rewrites reproduce the previous implementation exactly; baseline lsqr non-convergence (3/100 trees on concrete) changes nothing, so the direct solver removes a risk rather than producing gains. Full-mode run: no errors; resid_out 0.51x-1.00x baseline on every dataset (no gain on bike, parkinson, superconduct); resid_in up to 1.39-1.41x (abalone, concrete) and 1.69x (nutrition) by design; ortho_out worse on abalone (0.015 to 0.041, about 1 SE of noise at n_test=836) and airfoil (1.13x), better on concrete (0.88x) and nutrition (0.76x); fit time 0.16-1.05x baseline except powerplant 1.53x. Held-out residuals were printed during development; this round's choices used only R or label-free in-sample quantities, but the earlier interrupted attempt compared configs on the subset benchmark and some inherited settings may reflect that. All disclosed in NOTES.md.", "selftest": "Subset (subset_selftest.json, no errors), new/baseline: analytical resid_out 0.00859 vs 0.01032 (0.83x), mse_others 0.00168 vs 0.00218 (0.77x), mse_eta12 0.98x, mse_eta34 0.97x, other main effects within 3.5%, fit 10.8s vs 11.2s; airfoil resid_out 0.0225 vs 0.0264 (0.85x), resid_in 1.25x, ortho_out 0.112 vs 0.099 (worse), fit 1.9s vs 1.8s; concrete resid_out 0.0210 vs 0.0250 (0.84x), resid_in 1.41x, ortho_out 0.072 vs 0.082, fit 3.1s vs 3.3s; abalone resid_out 0.0209 vs 0.0406 (0.52x), resid_in 1.39x, ortho_out 0.041 vs 0.015 (worse), fit 6.1s vs 11.1s. Full mode (full_selftest.json, no errors): analytical resid_out 0.845x, mse_others 0.80x; housing resid_out 0.91x; nutrition resid_out 0.76x (resid_in 1.69x); powerplant 0.97x (fit 1.53x); bike, parkinson and superconduct unchanged within 1% on all metrics except locvar_in (0.93-0.98x); superconduct fit 113s vs 701s."}
```
