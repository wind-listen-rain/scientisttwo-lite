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
 "title": "Model-anchored smoothed-measure TreeHFD: neighbour-bin kernel smoothing of the input measure with exact tree targets on virtual cells",
 "addresses": [
  "No shrinkage on sparsely populated cells causes large in-sample/out-of-sample residual gaps",
  "Prediction-time fallback for unseen interaction cells is non-deterministic and uses a distance metric that ignores actual split geometry",
  "Least-squares solver convergence is never checked, risking silently under-converged tree coefficients"
 ],
 "mechanism": "In the HFD of a fitted model, the target function T is known everywhere; the only thing estimated from X_train is the input measure P_X. TreeHFD plugs in the raw empirical measure. So the per-tree least-squares problem (residual rows plus orthogonality and mean rows) only constrains full Cartesian cells that contain training points, and it lets pair-cell coefficients backed by one or two points absorb 3rd-order residue. That is the ~16x resid_in to resid_out blow-up on concrete, 5x on nutrition and 9x on abalone. It also forces the ad-hoc random nearest-index fallback for pair cells never seen in training. We replace P_n with a deterministic smoothed measure estimator: mu~ = (1 - alpha) * P_n + alpha * (1/|S|) * sum_{j in S} sum_{s=+/-1} (1/2) * P_n o Shift_{j,s}. Shift_{j,s} moves x_ij to the representative (training median) of the adjacent bin of the ensemble union-of-splits grid on x_j. S is the set of variables split on anywhere. This is a coordinate-wise discrete kernel smoother of the training measure. It stays local: one fine bin along one axis, so points stay near the data manifold, unlike interventional or marginal sampling. It is the same measure for every tree, so all per-tree problems target one common population surrogate. For each tree, the shifted copies fall into the tree's own Cartesian cells. When the shift crosses one of the tree's thresholds, the copy creates a 'virtual' full cell, and its target is the tree's exact output there, read from the fixed model's leaves. The virtual cells enter the residual rows with weight sqrt(n * mu~(cell)), and the orthogonality and mean rows use the smoothed pair and main marginals. Thinly populated pair cells are therefore tied to their neighbours through the true tree function. This is a model-anchored shrinkage that, unlike ridge, does not bias toward zero. Pair cells adjacent to the data now get fitted coefficients instead of a fallback. Any cell still unseen at prediction time uses a deterministic fallback: nearest trained cell by distance between bin-midpoint coordinates scaled by each variable's training IQR, ties broken by largest count then lowest index. Each per-tree system is assembled directly in scipy.sparse COO form (the baseline builds dense num_cells x partition_size arrays) and solved with LSMR under tight atol/btol. If istop signals non-convergence, it retries with a warm start and a higher iter_lim, then falls back to a sparse QR or normal-equation solve. The smoothing weight alpha is selected label-free by 2-fold cross-fitting on X_train: fit on one half and score the held-out reconstruction error against T, which is known on held-out training inputs.",
 "implementation_sketch": "method.py reimplements the per-tree fit on top of the treehfd package's tree-structure and partition helpers, without modifying the package. (1) Build the union edges U_j and bin medians r_j from X_train and the booster's split table (shared with idea 1's utilities). (2) Build the shifted copies X^(j,s) for j in S and s = +/-1, dropping rows with no neighbouring bin. Get per-tree outputs for all copies with one booster.predict(DMatrix(X^(j,s)), pred_leaf=True) per (j,s), chunked, mapped to leaf values from trees_to_dataframe. This queries the given model only; no labels, no new samples, no RNG. (3) Per tree: compute cell indices for original and shifted rows with the tree's own partition (compute_partition_main logic applied to all rows). Rows whose shift crosses none of the tree's thresholds just add weight to the original cell. Deduplicate unique full cells with summed weights via np.unique on the bin matrix. Build the residual, orthogonality and mean blocks as sparse COO with weighted np.bincount counts. (4) Solve with scipy.sparse.linalg.lsmr(atol=btol=1e-10, maxiter=large), check istop, and fall back if needed. (5) Predict: digitize; for each pair cell, look up the trained cell id through a dict keyed by the bin tuple; for misses, use the deterministic geometry-aware fallback. (6) Choose alpha from {0, 0.1, 0.2, 0.35} by 2-fold cross-fit on a deterministic split, run on a subsample of trees (e.g. every 5th tree) to bound cost. Expected fit time is about 1.5-4x baseline. Sparse assembly offsets the extra rows, which matters for superconduct (700 s baseline, where the 5x limit must be kept and reported).",
 "expected_effect_on_metrics": "resid_out should fall markedly on small and medium datasets with large in/out gaps: concrete (0.025), nutrition (0.096), abalone (0.041), airfoil (0.026). resid_in may rise slightly, because the fit no longer interpolates single-point cells; this trade-off will be reported. ortho_out should improve, since the orthogonality constraints use smoothed pair marginals that better match the test distribution, and unseen-cell fallbacks become rare and geometry-aware. Results become fully deterministic run to run. Analytical: lower resid_out and lower mse_eta12/mse_eta34 variance. mse_others may drop slightly as thin spurious-pair cells stop absorbing residue. On large-n datasets (bike, housing, superconduct) the gains are expected to be small but non-negative.",
 "risks": "(a) Under strong input dependence, shifted copies can land in low-density regions where T extrapolates poorly. This is the interventional-SHAP failure mode the paper criticises, although here it is limited to one adjacent fine bin on one axis with weight alpha. Choosing alpha by cross-fitting guards against it, but the smoothed measure is a deliberate small departure from the empirical HFD target, and this must be stated. (b) Compute: the number of unique cells per tree may grow 2-5x. Without careful sparse assembly the fit could exceed 5x baseline on superconduct (p=81); if so, restrict S per tree to the tree's own variables (already implied) and cap the copies per variable, and report timings honestly. (c) A reviewer could read evaluating the model on shifted training inputs as 'extra data'. Mitigation: the copies are a deterministic function of X_train alone and play the role of a kernel density estimate of P_X. The model is only evaluated, never refit, and no labels or test inputs are touched. (d) The alpha cross-fit adds overhead and may choose alpha = 0 on large-n datasets, which recovers the baseline plus the solver and fallback fixes.",
 "id": "S2"
}

Score novelty from 1 (already published as-is) to 10 (no close prior work). Reply with JSON:
{"novelty": int, "closest": [{"title": str, "url": str, "overlap": str}], "rationale": str}


## Reply

```json
{
  "novelty": 5,
  "closest": [
    {
      "title": "Apley, D. W. and Zhu, J. (2020), \"Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models\" (Accumulated Local Effects, ALE), J. R. Statist. Soc. B",
      "url": "https://christophm.github.io/interpretable-ml-book/ale.html",
      "overlap": "ALE's entire raison d'être is the same tradeoff S2 targets: PDP/interventional averaging extrapolates into low-density regions under dependent inputs, so ALE restricts itself to small local neighborhoods (finite differences within adjacent bins/windows of the feature's own distribution) and accumulates them, staying near the data manifold rather than sampling the full marginal. S2's 'shift x_ij to the representative of the adjacent bin along one axis' construction is a discretized, model-anchored version of exactly this local-neighborhood philosophy, and the paper explicitly frames itself against the same interventional-extrapolation failure mode ALE was built to fix. The novelty here is porting that local-smoothing principle into TreeHFD's per-tree constrained least-squares system and exploiting exact tree-leaf lookups at the shifted points (something ALE has no analogue for, since it targets generic black-box finite differences, not a closed-form measure used in an orthogonality-constrained solve)."
    },
    {
      "title": "Hooker, G. (2007), \"Generalized Functional ANOVA Diagnostics for High-Dimensional Functions of Dependent Variables\", J. Comput. Graph. Statist.",
      "url": "https://www.tandfonline.com/doi/abs/10.1198/106186007X237892",
      "overlap": "This is the canonical prior solution to the same estimation problem: replace the raw empirical/independence measure with a weighting/reference measure chosen to concentrate on high-density regions, then solve a weighted hierarchical-orthogonality projection — precisely the move S2 makes by substituting mu~ = (1-alpha) P_n + alpha * (local neighbor-bin mixture) for the raw empirical P_n inside TreeHFD's per-tree LS rows. TreeHFD's own paper (Bénard 2025) cites Hooker (2007) directly as the classical weighted-fANOVA route it deliberately avoids by using the untouched empirical measure instead; S2 is essentially reopening that design choice and re-introducing a (local, bounded, cross-fit-tuned) weighted measure, which is conceptually the generalized-fANOVA move applied to a tree-structured, model-anchored setting rather than a fully novel resolution of the dependence problem."
    }
  ],
  "rationale": "The core statistical move in S2 — trade a small, controlled measure perturbation for variance reduction in thinly populated cells, while staying local to avoid the extrapolation failure that plagues interventional/marginal approaches — is not new in spirit: it is essentially Hooker's (2007) generalized/weighted functional ANOVA (already cited and explicitly contrasted against in the TreeHFD paper itself) combined with the local-neighborhood restriction popularized by ALE (Apley & Zhu, 2020) to keep queries near the data manifold. What appears genuinely new is the specific instantiation for TreeHFD: using the union-of-splits grid to define 'adjacent bin' shifts per split-variable, reading the *exact* tree-leaf value at the resulting virtual cells (exploiting that T is known everywhere, unlike generic black-box ALE/Hooker settings where the function must be queried/estimated), folding this into the existing per-tree constrained least-squares system as extra weighted rows, and selecting the blend weight alpha by label-free cross-fitting against the known model output. That combination — model-anchored virtual cells feeding a hierarchical-orthogonality LS solve — hasn't been described in the literature I could find, plausibly because TreeHFD itself is only a few months old (NeurIPS 2025 / arXiv Oct 2025) and no follow-up work extending it was found. The solver hardening (sparse COO assembly, LSMR with convergence checks and QR/normal-equation fallback) and the deterministic geometry-aware fallback for unseen cells are sound engineering fixes but are standard numerical-linear-algebra practice, not novel contributions, and don't add to the score. Net: a well-motivated, non-trivial but incremental recombination of two established ideas (weighted/generalized fANOVA + local-neighborhood/ALE-style restriction) applied to a very recent method, rather than a conceptually new mechanism."
}
```
