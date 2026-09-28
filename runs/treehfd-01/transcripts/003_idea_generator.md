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
(none)

Reply with JSON: {"ideas": [{"title": str, "addresses": [limitation titles], "mechanism": str,
"implementation_sketch": str, "expected_effect_on_metrics": str, "risks": str}]}


## Reply

I've checked the full baseline numbers. Parkinson has ortho_in = 0.082, and the paper's own Theorem 3(ii) says the summed ensemble components are not exactly orthogonal. That supports the first idea. Here are the two ideas.

```json
{
  "ideas": [
    {
      "title": "Ensemble-level re-orthogonalization on the union-of-splits partition (sum-preserving, cross-fitted smoothed projection)",
      "addresses": [
        "The piecewise-constant density approximation leaves an irreducible bias at the fixed, shallow tree depth used in the benchmark",
        "Spurious interaction candidates are never pruned, leaking noise into components that should be exactly zero"
      ],
      "mechanism": "TreeHFD makes each tree's interaction orthogonal only to functions that are piecewise constant on that tree's own coarse partition of x_j. The paper's Theorem 3(ii) admits that the summed ensemble interaction eta_jk = sum_l eta_jk^l is NOT orthogonal to the summed main effects. The covariance is bounded only by Delta_{A,f}, the same discretization term that appears in the Theorem 6 error bound. The baseline shows this directly: ortho_in = 0.082 on parkinson and 0.028 on housing, even in-sample. The aggregated main effect eta_j is piecewise constant on the union U_j of all split thresholds on x_j across the ensemble. This union grid is much finer than any single depth-6 tree's grid, and we get it at no cost from the fixed model. The new step is a global, post-hoc projection. For each retained pair (j,k), solve an additive least-squares problem under the empirical measure of X_train: (g_j, g_k) = argmin E_n[(eta_jk - g_j(x_j) - g_k(x_k))^2] + lambda * R(g). Here g_j lives on the union bins U_j (merged to a minimum count), and R is a first-difference (P-spline-like) penalty along the ordered bins, so the transfer is smooth and generalises out of sample. Then set eta_jk <- eta_jk - g_j - g_k, eta_j <- eta_j + g_j and eta_k <- eta_k + g_k, and move the constants into the intercept. Hierarchical orthogonality only requires eta_jk to be orthogonal to functions of its own subsets, so the pairs can be projected independently and in any order. The move is pointwise sum-preserving, so the reconstruction (and therefore resid_in and resid_out) is unchanged at every x. The only change is how T is split among components: each interaction becomes orthogonal to the whole union-bin function space of x_j and x_k, which contains the final eta_j and eta_k. This is a finer discretization of the true HFD constraint, and it shrinks the effective Delta_{A,f} without retraining. The penalty lambda is picked label-free by K-fold cross-fitting on X_train only: fit the projection on K-1 folds and score the held-out |corr(eta_jk, eta_j)|.",
      "implementation_sketch": "method.py: (1) Run TreeHFD as in the baseline (bench/baseline_method.py logic, reusing the treehfd package) to get per-tree components. (2) From model.get_booster().trees_to_dataframe(), collect all split thresholds per feature. Build union edges U_j, then greedily merge adjacent bins that hold fewer than m_min (e.g. 10) X_train points. Store the edges. (3) Evaluate the aggregated eta_jk on X_train (n x K) and the bin indices b_j = digitize(X_train[:, j], U_j). (4) For each pair, assemble the normal equations with np.bincount: the Gram blocks are diagonal counts for (j,j) and (k,k) and a joint count table for (j,k). Add lambda * D^T D difference penalties and solve the small dense (B_j + B_k) system with a fixed gauge (centred g). Cost is O(n + B^2) per pair, so seconds even for superconduct. (5) Pick lambda from a small grid by 3-fold cross-fitting on X_train rows (deterministic fold split). (6) At predict time, compute the baseline components, then subtract g_j[digitize(x_j)] + g_k[digitize(x_k)] from each interaction column and add them to the main-effect columns. The intercept absorbs the centring constants. No RNG is used, and the added fit time is expected to be <10% of baseline.",
      "expected_effect_on_metrics": "ortho_in should drop to about 0 up to the lambda smoothing (largest gains on parkinson 0.082, housing 0.028, abalone 0.027). ortho_out should drop substantially on airfoil, concrete, nutrition and parkinson (currently 0.08-0.12), because the smoothed transfer generalises. resid_in and resid_out are unchanged by construction. Analytical: lower mse_eta12 and mse_eta34, since the true eta_jk are exactly orthogonal to all functions of their parent variables, and main-effect mass leaked into interactions returns to eta_1..eta_4. mse_others should fall, because the additive part of spurious pairs such as (1,3), created by per-tree coarse discretization under rho=0.5, is moved to main effects. locvar_in stays neutral or improves slightly thanks to the difference penalty on g_j. fit_s rises by about 1.0-1.1x.",
      "risks": "(a) Moving mass into main effects could slightly raise locvar_in on the finest union bins if lambda is too small. This is mitigated by the difference penalty and the minimum-count merging. (b) Orthogonality is enforced under the empirical training measure, so ortho_out gains depend on how well the cross-fitted lambda generalises; small-n datasets (concrete, n=824 train) may gain less. (c) If the aggregated interaction for a true pair is mostly pure interaction already, mse_eta12 and mse_eta34 may barely move; the gain is then mainly on ortho metrics. (d) The union bins of a variable split often (e.g. sin(2*pi*x1)) can be numerous, but B_j is at most a few hundred after merging, so the dense solves stay cheap. (e) This step changes neither the model nor the reconstruction, so it stays within red lines 2-3."
    },
    {
      "title": "Model-anchored smoothed-measure TreeHFD: neighbour-bin kernel smoothing of the input measure with exact tree targets on virtual cells",
      "addresses": [
        "No shrinkage on sparsely populated cells causes large in-sample/out-of-sample residual gaps",
        "Prediction-time fallback for unseen interaction cells is non-deterministic and uses a distance metric that ignores actual split geometry",
        "Least-squares solver convergence is never checked, risking silently under-converged tree coefficients"
      ],
      "mechanism": "In the HFD of a fitted model, the target function T is known everywhere; the only thing estimated from X_train is the input measure P_X. TreeHFD plugs in the raw empirical measure. So the per-tree least-squares problem (residual rows plus orthogonality and mean rows) only constrains full Cartesian cells that contain training points, and it lets pair-cell coefficients backed by one or two points absorb 3rd-order residue. That is the ~16x resid_in to resid_out blow-up on concrete, 5x on nutrition and 9x on abalone. It also forces the ad-hoc random nearest-index fallback for pair cells never seen in training. We replace P_n with a deterministic smoothed measure estimator: mu~ = (1 - alpha) * P_n + alpha * (1/|S|) * sum_{j in S} sum_{s=+/-1} (1/2) * P_n o Shift_{j,s}. Shift_{j,s} moves x_ij to the representative (training median) of the adjacent bin of the ensemble union-of-splits grid on x_j. S is the set of variables split on anywhere. This is a coordinate-wise discrete kernel smoother of the training measure. It stays local: one fine bin along one axis, so points stay near the data manifold, unlike interventional or marginal sampling. It is the same measure for every tree, so all per-tree problems target one common population surrogate. For each tree, the shifted copies fall into the tree's own Cartesian cells. When the shift crosses one of the tree's thresholds, the copy creates a 'virtual' full cell, and its target is the tree's exact output there, read from the fixed model's leaves. The virtual cells enter the residual rows with weight sqrt(n * mu~(cell)), and the orthogonality and mean rows use the smoothed pair and main marginals. Thinly populated pair cells are therefore tied to their neighbours through the true tree function. This is a model-anchored shrinkage that, unlike ridge, does not bias toward zero. Pair cells adjacent to the data now get fitted coefficients instead of a fallback. Any cell still unseen at prediction time uses a deterministic fallback: nearest trained cell by distance between bin-midpoint coordinates scaled by each variable's training IQR, ties broken by largest count then lowest index. Each per-tree system is assembled directly in scipy.sparse COO form (the baseline builds dense num_cells x partition_size arrays) and solved with LSMR under tight atol/btol. If istop signals non-convergence, it retries with a warm start and a higher iter_lim, then falls back to a sparse QR or normal-equation solve. The smoothing weight alpha is selected label-free by 2-fold cross-fitting on X_train: fit on one half and score the held-out reconstruction error against T, which is known on held-out training inputs.",
      "implementation_sketch": "method.py reimplements the per-tree fit on top of the treehfd package's tree-structure and partition helpers, without modifying the package. (1) Build the union edges U_j and bin medians r_j from X_train and the booster's split table (shared with idea 1's utilities). (2) Build the shifted copies X^(j,s) for j in S and s = +/-1, dropping rows with no neighbouring bin. Get per-tree outputs for all copies with one booster.predict(DMatrix(X^(j,s)), pred_leaf=True) per (j,s), chunked, mapped to leaf values from trees_to_dataframe. This queries the given model only; no labels, no new samples, no RNG. (3) Per tree: compute cell indices for original and shifted rows with the tree's own partition (compute_partition_main logic applied to all rows). Rows whose shift crosses none of the tree's thresholds just add weight to the original cell. Deduplicate unique full cells with summed weights via np.unique on the bin matrix. Build the residual, orthogonality and mean blocks as sparse COO with weighted np.bincount counts. (4) Solve with scipy.sparse.linalg.lsmr(atol=btol=1e-10, maxiter=large), check istop, and fall back if needed. (5) Predict: digitize; for each pair cell, look up the trained cell id through a dict keyed by the bin tuple; for misses, use the deterministic geometry-aware fallback. (6) Choose alpha from {0, 0.1, 0.2, 0.35} by 2-fold cross-fit on a deterministic split, run on a subsample of trees (e.g. every 5th tree) to bound cost. Expected fit time is about 1.5-4x baseline. Sparse assembly offsets the extra rows, which matters for superconduct (700 s baseline, where the 5x limit must be kept and reported).",
      "expected_effect_on_metrics": "resid_out should fall markedly on small and medium datasets with large in/out gaps: concrete (0.025), nutrition (0.096), abalone (0.041), airfoil (0.026). resid_in may rise slightly, because the fit no longer interpolates single-point cells; this trade-off will be reported. ortho_out should improve, since the orthogonality constraints use smoothed pair marginals that better match the test distribution, and unseen-cell fallbacks become rare and geometry-aware. Results become fully deterministic run to run. Analytical: lower resid_out and lower mse_eta12/mse_eta34 variance. mse_others may drop slightly as thin spurious-pair cells stop absorbing residue. On large-n datasets (bike, housing, superconduct) the gains are expected to be small but non-negative.",
      "risks": "(a) Under strong input dependence, shifted copies can land in low-density regions where T extrapolates poorly. This is the interventional-SHAP failure mode the paper criticises, although here it is limited to one adjacent fine bin on one axis with weight alpha. Choosing alpha by cross-fitting guards against it, but the smoothed measure is a deliberate small departure from the empirical HFD target, and this must be stated. (b) Compute: the number of unique cells per tree may grow 2-5x. Without careful sparse assembly the fit could exceed 5x baseline on superconduct (p=81); if so, restrict S per tree to the tree's own variables (already implied) and cap the copies per variable, and report timings honestly. (c) A reviewer could read evaluating the model on shifted training inputs as 'extra data'. Mitigation: the copies are a deterministic function of X_train alone and play the role of a kernel density estimate of P_X. The model is only evaluated, never refit, and no labels or test inputs are touched. (d) The alpha cross-fit adds overhead and may choose alpha = 0 on large-n datasets, which recovers the baseline plus the solver and fallback fixes."
    }
  ]
}
```
