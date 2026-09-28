# idea_generator (opus)

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


# Role: Idea Generator
Propose 2 new research ideas that improve TreeHFD by resolving the limitations below. An idea must be a *new
methodological component* (a new estimator, constraint, basis, projection, regulariser, solver formulation, ...),
not a configuration change, and must be implementable within the method interface in <ROOT>/bench/BENCHMARK.md
within the red lines. Prefer ideas that are more novel than, and clearly distinct from, the existing ones.

Limitations:
[
 {
  "title": "Prediction-time fallback for unseen interaction cells is non-deterministic and uses a distance metric that ignores actual split geometry",
  "evidence": "cartesian_partition.py:208-217 (predict_partition): when a test point's interaction cell was empty in training, the code picks the 'nearest' training cell via `cell_distance = np.sum(np.abs(cell_array_train - cell), axis=1)` (L1 distance between raw bin *indices*, not bin boundary values), then breaks ties among equal-distance candidates with `np.random.default_rng().choice(...)` — an unseeded RNG call. Since Cartesian tree partitions have non-uniform bin widths per variable, two adjacent-index bins can be far apart in feature space while two same-count-away bins on a finely-split variable are numerically closer; the L1-index heuristic conflates the two. The independent 5000-point test set in the analytical benchmark and the 20% held-out real-data splits routinely hit interaction cells absent from training.",
  "kind": "numerical",
  "why_actionable": "Both fixes only require X_train/model access: (1) fix the RNG seed (or use a deterministic tie-break, e.g. lowest cell index) to make resid_out/ortho_out reproducible across runs; (2) replace the L1 index distance with a distance computed on the actual bin boundary midpoints (already stored in split_list), which more faithfully extrapolates and should reduce resid_out and ortho_out on datasets with many interaction cells (e.g. parkinson p=19, superconduct p=81).",
  "id": "L1"
 },
 {
  "title": "Least-squares solver convergence is never checked, risking silently under-converged tree coefficients",
  "evidence": "tree.py:115 calls `self.hfd_coeffs = lsqr(constr_mat, target)[0]` and discards the rest of scipy's `lsqr` return tuple (istop, itn, r1norm, etc.), with no `damp`, no custom `iter_lim`, and no residual/convergence check. `lsqr` is an iterative Krylov solver; for deeper trees with more cells and interaction constraints (constr_ortho + constr_mean + constr_resid stacked in optimization_matrix.py:76-78), the system can become ill-conditioned or need more iterations than the default limit to converge to the target tolerance.",
  "kind": "numerical",
  "why_actionable": "Checking `istop`/`itn` and re-solving with a higher `iter_lim` (or switching to a direct sparse least-squares solve, e.g. `scipy.sparse.linalg.lsmr` with tighter tolerances, or LSQR with `x0` warm start) when convergence fails is a pure implementation fix using only the tree's own constraint matrix — no labels or extra data involved — and would tighten resid_in, which propagates to resid_out and ortho metrics across all datasets, especially higher-p ones (parkinson, superconduct) where trees have the most cells.",
  "id": "L2"
 },
 {
  "title": "No shrinkage on sparsely populated cells causes large in-sample/out-of-sample residual gaps",
  "evidence": "baseline/subset.json real-data resid_in vs resid_out ratios: concrete (n=1030) 0.00153 -> 0.0250 (~16.3x blowup), abalone (n=4177, p=9) 0.00455 -> 0.0406 (~8.9x), airfoil (n=1503) 0.00985 -> 0.0264 (~2.7x). The smallest-n dataset (concrete) shows the largest degradation. optimization_matrix.py's `constr_resid` weights each cell by `sqrt(nsample*counts)` but the loss has no penalty discouraging large-magnitude coefficients in cells with very few supporting points, so per-cell coefficient estimates for thinly populated cells are high-variance and generalize poorly.",
  "kind": "estimator",
  "why_actionable": "Adding a ridge-type penalty (or a minimum-count threshold that merges/shrinks coefficients toward a coarser neighboring cell's value) in the per-tree least-squares problem, computed purely from X_train cell counts, would reduce resid_out without needing labels, retraining, or extra data — directly targeting the metric where the baseline currently degrades most.",
  "id": "L3"
 },
 {
  "title": "Spurious interaction candidates are never pruned, leaking noise into components that should be exactly zero",
  "evidence": "baseline/subset.json analytical: `mse_others` = 0.00218 (mean, std 0.00021) — nonzero, even though ground truth for non-(1,2)/(3,4) interaction pairs is exactly 0. tree_structure.py `extract_interactions` (lines 162-183) adds *every* variable pair that co-occurs anywhere along any tree path to the candidate list, with no check on how many training points actually populate the resulting interaction cells or how much variance the fitted component carries relative to noise.",
  "kind": "estimator",
  "why_actionable": "A post-hoc pruning/shrinkage step — e.g., soft-thresholding a candidate interaction's fitted coefficients toward 0 when its cell supports are small or its estimated variance is statistically indistinguishable from the residual noise level — uses only X_train and the already-fitted per-tree coefficients, so it doesn't touch labels, the model, or other red lines, and should lower mse_others (and likely ortho metrics, since spurious interactions are a source of orthogonality violation) without degrading mse_eta12/mse_eta34 on the true interactions.",
  "id": "L4"
 },
 {
  "title": "The piecewise-constant density approximation leaves an irreducible bias at the fixed, shallow tree depth used in the benchmark",
  "evidence": "Paper Theorem 6 (lines 341-352) bounds the tree-HFD approximation error to m's HFD by `K1 * E[(m-T)^2] + K2 * Delta_{A,f}^2 * (sum_l sqrt(E[Tl^2]))^2`, and Corollary 1 (lines 353-365) only guarantees this vanishes when tree depth grows to infinity with n. The benchmark protocol fixes `max_depth=6` for all datasets and repetitions (BENCHMARK.md / harness.py XGB_PARAMS), so the `Delta_{A,f}` bias term is fixed regardless of how much data TreeHFD's own fit step uses — it cannot be reduced simply by better estimation from X_train under the current algorithm, since the Cartesian partition granularity is inherited entirely from the (fixed, unretrainable) XGBoost tree structure.",
  "kind": "theoretical",
  "why_actionable": "Since the tree ensemble itself cannot be retrained (red line 3), the only permissible lever is a post-hoc bias correction inside the HFD estimator: e.g., replace the implicit assumption that f is constant within each Cartesian cell with a local first-order correction (a weighted local-linear adjustment of the cell coefficient using the within-cell distribution of X_train), which only uses the fitted model and X_train. This directly targets the residual/orthogonality gap on datasets with more skewed or higher-dimensional inputs (e.g. abalone, parkinson, superconduct) where within-cell density variation is largest.",
  "id": "L5"
 }
]

Existing ideas (be distinct from these):
[
 {
  "id": "S1",
  "title": "Ensemble-level re-orthogonalization on the union-of-splits partition (sum-preserving, cross-fitted smoothed projection)"
 },
 {
  "id": "S2",
  "title": "Model-anchored smoothed-measure TreeHFD: neighbour-bin kernel smoothing of the input measure with exact tree targets on virtual cells"
 }
]

Reply with JSON: {"ideas": [{"title": str, "addresses": [limitation titles], "mechanism": str,
"implementation_sketch": str, "expected_effect_on_metrics": str, "risks": str}]}


## Reply

I couldn't run a grounding check: permission to write a script in the working directory was denied, and the inline Python command was blocked. Both ideas below come from reading the source (`optimization_matrix.py`, `tree.py`, `cartesian_partition.py`, `ensemble.py`), `harness.py` and the baseline numbers, and neither has been tested.

```json
{
  "ideas": [
    {
      "title": "Conditional-probability-enriched TreeHFD (CPE-HFD): a measure-adapted, piecewise-smooth basis replacing the within-cell constant-density approximation",
      "addresses": [
        "The piecewise-constant density approximation leaves an irreducible bias at the fixed, shallow tree depth used in the benchmark"
      ],
      "mechanism": "When the inputs are dependent, the exact HFD of a single tree is not piecewise constant on the tree's Cartesian partition. Conditioning the HFD equations on X_j gives eta_j(x_j) = E[T|X_j=x_j] - eta0 - sum_{u not containing j} E[eta_u(X_u)|X_j=x_j]. For a tree T = sum_l v_l prod_i 1{x_i in I_li}, every term on the right is a combination of within-bin conditional cell-probability curves pi_{k,b|j}(x_j) = P(X_k in B_{k,b} | X_j = x_j), where k ranges over the other variables of the same tree. TreeHFD implicitly treats these curves as constant inside each bin. That is the Delta_{A,f} term of Theorem 6, and it cannot shrink at max_depth=6. CPE-HFD keeps TreeHFD's least-squares-with-orthogonality formulation but enlarges each component's function class with exactly these curves. The curves are estimated once from X_train alone by a 1-D kernel smoother (on the rank scale of X_j) of the indicators 1{X_k <= t} at the tree thresholds t, which gives conditional CDFs. Main effect on bin a of x_j: eta_j = alpha_{j,a} + sum_{k,b} beta_{j,a,k,b} * delta_{k,b|j}(x_j), where delta is pi-hat minus its within-bin mean, so TreeHFD's constant part stays identifiable. Interaction (j,k) on an observed cell (a,b): theta_ab + gamma^j_ab * delta_{k,b|j}(x_j) + gamma^k_ab * delta_{j,a|k}(x_k). This lets interactions carry the univariate within-cell pieces that must cancel the main-effect curvature (in the analytical truth, eta_12 contains -rho/(1+rho^2)(x1^2+x2^2)). Hierarchical orthogonality becomes point-level moment constraints E_n[eta_jk(X) g(X_j)] = 0 for every basis function g of eta_j. These strictly refine TreeHFD's bin-level constraints, which are the special case where g is a bin indicator. The residual is minimised point-wise rather than cell-wise. Theorem 6's partition-resolution bias is thereby replaced by the estimation error of 1-D conditional-probability smoothers, which does not depend on tree depth. This differs from S2, which smooths the input measure but keeps piecewise-constant components; CPE changes the function basis itself.",
      "implementation_sketch": "(1) For each ordered variable pair (j,k), precompute conditional CDF curves F_k(t | x_j) for every threshold t that any tree uses on k. XGBoost hist reuses a shared quantile set of thresholds, so cache by (j,k,t). Estimator: Nadaraya-Watson with an Epanechnikov kernel on the rank of X_j and a fixed rule-of-thumb bandwidth h = n^(-1/5) in rank units, computed in O(n log n) with sorted cumulative sums. Store values on sorted X_train[:,j] and linearly interpolate at predict time. If X_j is discrete or tied, the curves are constant within bins, delta is 0, the columns are dropped, and the method falls back exactly to baseline. (2) Per tree, reuse treehfd.tree_structure and CartesianTreePartition for variables, bins and interaction pairs, and XGBTreeHFD._tree_predict for y_tree. (3) Build a sparse point-level design with n rows: TreeHFD's indicator columns plus enrichment columns for the other variables k in the same tree, restricted to populated (j-bin, k-bin) pairs. Drop columns whose within-bin variance is below 1e-8. (4) Add constraint rows: zero mean per component, and for each pair (j,k) and each column g of eta_j (resp. eta_k) a row sum_i g(x_ij) * A_i[cols of eta_jk]. Weight them like TreeHFD's constraints (about sqrt(n)) or enforce them exactly via KKT. (5) Solve (A'A + C'WC + mu R) beta = A'y_tree with a direct Cholesky and check the residual. Column counts are about 2-3x TreeHFD's. mu is a small ridge on enrichment coefficients only, fixed relative to the trace (e.g. 1e-3 * tr/c), for stability and never tuned on metrics. (6) At predict time, compute bins as in the baseline, evaluate curves by interpolation, and sum over trees as in ensemble.py. Handle unseen interaction cells deterministically (zero interaction, or the rule from idea 2). Group points by (full joint cell, x_j-quantile) with sufficient statistics if n is large.",
      "expected_effect_on_metrics": "Analytical mse_eta1-4 and mse_eta12/mse_eta34 should fall. Rho=0.5 dependence makes the true components smooth, and the true interactions contain univariate x^2 pieces that a depth-6 cell-constant basis cannot represent. ortho_in and ortho_out should fall, because orthogonality is enforced against within-bin variation, not only bin averages. resid_in should be at or below baseline (the function class is nested); resid_out should be neutral to slightly lower. locvar_in may move either way because components become piecewise-smooth; this trade-off must be reported. mse_eta5/6 and mse_others should be essentially unchanged. Expected fit_s is 2-4x baseline, under the 5x line if the design is built vectorised.",
      "risks": "More columns add variance on small-n data (concrete has 824 training points), so the enrichment ridge is needed. The kernel smoother has boundary bias in the tails of X_j, which affects extrapolation at held-out points. A tree with 6 variables inflates enrichment columns; cap them by keeping only partners that co-occur with j on a path, or the top partners by cell mass. Point-level rows may push fit time past 5x on superconduct (about 17k training points, p=81) unless sufficient-statistic grouping is used. Gains concentrate on continuous, correlated inputs and will be negligible on discrete or one-hot features. Untested."
    },
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
      "risks": "PRESS removes only the held-out residual row, not that point's contribution to the ortho and mean rows, so the leave-out estimate is slightly optimistic. Good-Turing on joint cells degenerates when almost all cells are singletons (high p: parkinson, superconduct); R then reduces to pure leave-one-out, which still matches test-time extrapolation. Per-tree selection ignores error correlation across trees; step 6 mitigates this. Shrinking interactions reallocates variance to main effects through the ortho rows and may slightly raise ortho_in. Gains may be modest if resid_out is dominated by genuine higher-order tree content in seen cells rather than by overfitting of thin cells. Untested."
    }
  ]
}
```
