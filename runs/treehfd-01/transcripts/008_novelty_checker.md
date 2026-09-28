# novelty_checker (sonnet)

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


# Role: Novelty Checker
Assess the novelty of one research idea. Use WebSearch to find the two most closely related published papers
(beyond TreeHFD itself), open them with WebFetch if needed, and compare.

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
 "id": "S4"
}

Score novelty from 1 (already published as-is) to 10 (no close prior work). Reply with JSON:
{"novelty": int, "closest": [{"title": str, "url": str, "overlap": str}], "rationale": str}


## Reply

```json
{
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
  "rationale": "Every individual ingredient here is decades-old and well published: ridge/Tikhonov shrinkage of least-squares coefficients, PRESS/GCV via the hat-matrix diagonal for penalty selection (Golub-Heath-Wahba 1979), Good-Turing missing-mass estimation (Good 1953, Gale & Sampson 1995), graph-Laplacian/fused-lasso smoothing over a 2D lattice, and sample-count-weighted shrinkage of tree-based estimates toward a coarser model (Hierarchical Shrinkage, 2022). I found no paper that combines a Good-Turing missing-mass weight with a PRESS-style in/out-of-sample risk mix to select a per-tree ridge penalty, nor any that uses this combination to give a deterministic empirical-Bayes fallback rule for unseen interaction cells in a tree-ensemble functional decomposition — that specific composition and its application to TreeHFD's per-tree least-squares step (a method published only in Oct 2025) is new simply because the base method is new. But the contribution is an assembly of known statistical tools into a well-engineered pipeline rather than a new estimator, prior, or theoretical result; a reviewer familiar with GCV/PRESS and Good-Turing smoothing would recognize each piece immediately and likely view the combination as a natural, if non-trivial, engineering step rather than a conceptual breakthrough. This places it in the middle of the scale: clearly not already published (nothing targets TreeHFD or this exact risk criterion), but not far from obvious once TreeHFD's overfitting failure mode is diagnosed, since 'shrink sparse cells with a CV-style criterion, use N1/n for the unseen-mass rate' is a fairly standard move for anyone versed in smoothing/regularization literature."
}
```
