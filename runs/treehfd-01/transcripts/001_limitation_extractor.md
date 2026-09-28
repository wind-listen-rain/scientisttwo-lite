# limitation_extractor (sonnet)

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


# Role: Limitation Extractor
Read the original paper (<ROOT>/tasks/treehfd_paper.txt), the source code (<ROOT>/tasks/treehfd/src/treehfd/) and the
baseline results (<ROOT>/baseline/subset.json). Identify concrete, *actionable* limitations of TreeHFD: theoretical
bottlenecks, estimator design choices, numerical issues, or empirical weaknesses visible in the metrics. Each limitation
must point to evidence (a section, equation, code location, or a baseline number) and say why fixing it could improve the
benchmark metrics without breaking the red lines.

Limitations already collected (do not repeat them):
(none)

Verifier feedback on what is missing (may be empty):
(none)

Reply with JSON: {"limitations": [{"title": str, "evidence": str, "kind": "theoretical|estimator|numerical|empirical", "why_actionable": str}]}
Give 3-6 new limitations.


## Reply

I've reviewed the paper (Theorems 2–7, Section 3 algorithm/loss, Limitations), the `treehfd` source (`ensemble.py`, `tree.py`, `cartesian_partition.py`, `optimization_matrix.py`, `tree_structure.py`, `validation.py`), and `baseline/subset.json`. Here are 5 concrete, evidence-backed limitations.

```json
{"limitations": [
  {"title": "Prediction-time fallback for unseen interaction cells is non-deterministic and uses a distance metric that ignores actual split geometry",
   "evidence": "cartesian_partition.py:208-217 (predict_partition): when a test point's interaction cell was empty in training, the code picks the 'nearest' training cell via `cell_distance = np.sum(np.abs(cell_array_train - cell), axis=1)` (L1 distance between raw bin *indices*, not bin boundary values), then breaks ties among equal-distance candidates with `np.random.default_rng().choice(...)` — an unseeded RNG call. Since Cartesian tree partitions have non-uniform bin widths per variable, two adjacent-index bins can be far apart in feature space while two same-count-away bins on a finely-split variable are numerically closer; the L1-index heuristic conflates the two. The independent 5000-point test set in the analytical benchmark and the 20% held-out real-data splits routinely hit interaction cells absent from training.",
   "kind": "numerical",
   "why_actionable": "Both fixes only require X_train/model access: (1) fix the RNG seed (or use a deterministic tie-break, e.g. lowest cell index) to make resid_out/ortho_out reproducible across runs; (2) replace the L1 index distance with a distance computed on the actual bin boundary midpoints (already stored in split_list), which more faithfully extrapolates and should reduce resid_out and ortho_out on datasets with many interaction cells (e.g. parkinson p=19, superconduct p=81)."},

  {"title": "Least-squares solver convergence is never checked, risking silently under-converged tree coefficients",
   "evidence": "tree.py:115 calls `self.hfd_coeffs = lsqr(constr_mat, target)[0]` and discards the rest of scipy's `lsqr` return tuple (istop, itn, r1norm, etc.), with no `damp`, no custom `iter_lim`, and no residual/convergence check. `lsqr` is an iterative Krylov solver; for deeper trees with more cells and interaction constraints (constr_ortho + constr_mean + constr_resid stacked in optimization_matrix.py:76-78), the system can become ill-conditioned or need more iterations than the default limit to converge to the target tolerance.",
   "kind": "numerical",
   "why_actionable": "Checking `istop`/`itn` and re-solving with a higher `iter_lim` (or switching to a direct sparse least-squares solve, e.g. `scipy.sparse.linalg.lsmr` with tighter tolerances, or LSQR with `x0` warm start) when convergence fails is a pure implementation fix using only the tree's own constraint matrix — no labels or extra data involved — and would tighten resid_in, which propagates to resid_out and ortho metrics across all datasets, especially higher-p ones (parkinson, superconduct) where trees have the most cells."},

  {"title": "No shrinkage on sparsely populated cells causes large in-sample/out-of-sample residual gaps",
   "evidence": "baseline/subset.json real-data resid_in vs resid_out ratios: concrete (n=1030) 0.00153 -> 0.0250 (~16.3x blowup), abalone (n=4177, p=9) 0.00455 -> 0.0406 (~8.9x), airfoil (n=1503) 0.00985 -> 0.0264 (~2.7x). The smallest-n dataset (concrete) shows the largest degradation. optimization_matrix.py's `constr_resid` weights each cell by `sqrt(nsample*counts)` but the loss has no penalty discouraging large-magnitude coefficients in cells with very few supporting points, so per-cell coefficient estimates for thinly populated cells are high-variance and generalize poorly.",
   "kind": "estimator",
   "why_actionable": "Adding a ridge-type penalty (or a minimum-count threshold that merges/shrinks coefficients toward a coarser neighboring cell's value) in the per-tree least-squares problem, computed purely from X_train cell counts, would reduce resid_out without needing labels, retraining, or extra data — directly targeting the metric where the baseline currently degrades most."},

  {"title": "Spurious interaction candidates are never pruned, leaking noise into components that should be exactly zero",
   "evidence": "baseline/subset.json analytical: `mse_others` = 0.00218 (mean, std 0.00021) — nonzero, even though ground truth for non-(1,2)/(3,4) interaction pairs is exactly 0. tree_structure.py `extract_interactions` (lines 162-183) adds *every* variable pair that co-occurs anywhere along any tree path to the candidate list, with no check on how many training points actually populate the resulting interaction cells or how much variance the fitted component carries relative to noise.",
   "kind": "estimator",
   "why_actionable": "A post-hoc pruning/shrinkage step — e.g., soft-thresholding a candidate interaction's fitted coefficients toward 0 when its cell supports are small or its estimated variance is statistically indistinguishable from the residual noise level — uses only X_train and the already-fitted per-tree coefficients, so it doesn't touch labels, the model, or other red lines, and should lower mse_others (and likely ortho metrics, since spurious interactions are a source of orthogonality violation) without degrading mse_eta12/mse_eta34 on the true interactions."},

  {"title": "The piecewise-constant density approximation leaves an irreducible bias at the fixed, shallow tree depth used in the benchmark",
   "evidence": "Paper Theorem 6 (lines 341-352) bounds the tree-HFD approximation error to m's HFD by `K1 * E[(m-T)^2] + K2 * Delta_{A,f}^2 * (sum_l sqrt(E[Tl^2]))^2`, and Corollary 1 (lines 353-365) only guarantees this vanishes when tree depth grows to infinity with n. The benchmark protocol fixes `max_depth=6` for all datasets and repetitions (BENCHMARK.md / harness.py XGB_PARAMS), so the `Delta_{A,f}` bias term is fixed regardless of how much data TreeHFD's own fit step uses — it cannot be reduced simply by better estimation from X_train under the current algorithm, since the Cartesian partition granularity is inherited entirely from the (fixed, unretrainable) XGBoost tree structure.",
   "kind": "theoretical",
   "why_actionable": "Since the tree ensemble itself cannot be retrained (red line 3), the only permissible lever is a post-hoc bias correction inside the HFD estimator: e.g., replace the implicit assumption that f is constant within each Cartesian cell with a local first-order correction (a weighted local-linear adjustment of the cell coefficient using the within-cell distribution of X_train), which only uses the fitted model and X_train. This directly targets the residual/orthogonality gap on datasets with more skewed or higher-dimensional inputs (e.g. abalone, parkinson, superconduct) where within-cell density variation is largest."}
]}
```
