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
 "title": "Conditional-probability-enriched TreeHFD (CPE-HFD): a measure-adapted, piecewise-smooth basis replacing the within-cell constant-density approximation",
 "addresses": [
  "The piecewise-constant density approximation leaves an irreducible bias at the fixed, shallow tree depth used in the benchmark"
 ],
 "mechanism": "When the inputs are dependent, the exact HFD of a single tree is not piecewise constant on the tree's Cartesian partition. Conditioning the HFD equations on X_j gives eta_j(x_j) = E[T|X_j=x_j] - eta0 - sum_{u not containing j} E[eta_u(X_u)|X_j=x_j]. For a tree T = sum_l v_l prod_i 1{x_i in I_li}, every term on the right is a combination of within-bin conditional cell-probability curves pi_{k,b|j}(x_j) = P(X_k in B_{k,b} | X_j = x_j), where k ranges over the other variables of the same tree. TreeHFD implicitly treats these curves as constant inside each bin. That is the Delta_{A,f} term of Theorem 6, and it cannot shrink at max_depth=6. CPE-HFD keeps TreeHFD's least-squares-with-orthogonality formulation but enlarges each component's function class with exactly these curves. The curves are estimated once from X_train alone by a 1-D kernel smoother (on the rank scale of X_j) of the indicators 1{X_k <= t} at the tree thresholds t, which gives conditional CDFs. Main effect on bin a of x_j: eta_j = alpha_{j,a} + sum_{k,b} beta_{j,a,k,b} * delta_{k,b|j}(x_j), where delta is pi-hat minus its within-bin mean, so TreeHFD's constant part stays identifiable. Interaction (j,k) on an observed cell (a,b): theta_ab + gamma^j_ab * delta_{k,b|j}(x_j) + gamma^k_ab * delta_{j,a|k}(x_k). This lets interactions carry the univariate within-cell pieces that must cancel the main-effect curvature (in the analytical truth, eta_12 contains -rho/(1+rho^2)(x1^2+x2^2)). Hierarchical orthogonality becomes point-level moment constraints E_n[eta_jk(X) g(X_j)] = 0 for every basis function g of eta_j. These strictly refine TreeHFD's bin-level constraints, which are the special case where g is a bin indicator. The residual is minimised point-wise rather than cell-wise. Theorem 6's partition-resolution bias is thereby replaced by the estimation error of 1-D conditional-probability smoothers, which does not depend on tree depth. This differs from S2, which smooths the input measure but keeps piecewise-constant components; CPE changes the function basis itself.",
 "implementation_sketch": "(1) For each ordered variable pair (j,k), precompute conditional CDF curves F_k(t | x_j) for every threshold t that any tree uses on k. XGBoost hist reuses a shared quantile set of thresholds, so cache by (j,k,t). Estimator: Nadaraya-Watson with an Epanechnikov kernel on the rank of X_j and a fixed rule-of-thumb bandwidth h = n^(-1/5) in rank units, computed in O(n log n) with sorted cumulative sums. Store values on sorted X_train[:,j] and linearly interpolate at predict time. If X_j is discrete or tied, the curves are constant within bins, delta is 0, the columns are dropped, and the method falls back exactly to baseline. (2) Per tree, reuse treehfd.tree_structure and CartesianTreePartition for variables, bins and interaction pairs, and XGBTreeHFD._tree_predict for y_tree. (3) Build a sparse point-level design with n rows: TreeHFD's indicator columns plus enrichment columns for the other variables k in the same tree, restricted to populated (j-bin, k-bin) pairs. Drop columns whose within-bin variance is below 1e-8. (4) Add constraint rows: zero mean per component, and for each pair (j,k) and each column g of eta_j (resp. eta_k) a row sum_i g(x_ij) * A_i[cols of eta_jk]. Weight them like TreeHFD's constraints (about sqrt(n)) or enforce them exactly via KKT. (5) Solve (A'A + C'WC + mu R) beta = A'y_tree with a direct Cholesky and check the residual. Column counts are about 2-3x TreeHFD's. mu is a small ridge on enrichment coefficients only, fixed relative to the trace (e.g. 1e-3 * tr/c), for stability and never tuned on metrics. (6) At predict time, compute bins as in the baseline, evaluate curves by interpolation, and sum over trees as in ensemble.py. Handle unseen interaction cells deterministically (zero interaction, or the rule from idea 2). Group points by (full joint cell, x_j-quantile) with sufficient statistics if n is large.",
 "expected_effect_on_metrics": "Analytical mse_eta1-4 and mse_eta12/mse_eta34 should fall. Rho=0.5 dependence makes the true components smooth, and the true interactions contain univariate x^2 pieces that a depth-6 cell-constant basis cannot represent. ortho_in and ortho_out should fall, because orthogonality is enforced against within-bin variation, not only bin averages. resid_in should be at or below baseline (the function class is nested); resid_out should be neutral to slightly lower. locvar_in may move either way because components become piecewise-smooth; this trade-off must be reported. mse_eta5/6 and mse_others should be essentially unchanged. Expected fit_s is 2-4x baseline, under the 5x line if the design is built vectorised.",
 "risks": "More columns add variance on small-n data (concrete has 824 training points), so the enrichment ridge is needed. The kernel smoother has boundary bias in the tails of X_j, which affects extrapolation at held-out points. A tree with 6 variables inflates enrichment columns; cap them by keeping only partners that co-occur with j on a path, or the top partners by cell mass. Point-level rows may push fit time past 5x on superconduct (about 17k training points, p=81) unless sufficient-statistic grouping is used. Gains concentrate on continuous, correlated inputs and will be negligible on discrete or one-hot features. Untested.",
 "id": "S3"
}

Score novelty from 1 (already published as-is) to 10 (no close prior work). Reply with JSON:
{"novelty": int, "closest": [{"title": str, "url": str, "overlap": str}], "rationale": str}


## Reply

```json
{
  "novelty": 3,
  "closest": [
    {
      "title": "Generalized Functional ANOVA in Closed-Form: A Unified View of Additive Explanations (Ferrere, Bousquet, Gamboa, Loubes, 2026, arXiv:2605.18422)",
      "url": "https://arxiv.org/abs/2605.18422",
      "overlap": "Targets exactly the same problem S3 is motivated by: a piecewise-constant / bin-level approximation to the generalized (dependent-input) Hoeffding decomposition leaves an irreducible bias, and the fix is to enlarge the basis used in the least-squares fit to something smooth and hierarchically orthogonal 'by construction' rather than orthogonal only at the bin level. It explicitly benchmarks against TreeHFD and applies to tree-ensemble black boxes. The realization differs (global normalized-Legendre polynomial basis with inverse marginal-density weighting, projection-estimated densities, and a single global least-squares solve) versus S3's tree-threshold-anchored, per-pair conditional-CDF kernel-smoothed enrichment columns layered on top of TreeHFD's own Cartesian-partition design matrix with point-level moment constraints. But the core conceptual move — 'replace piecewise-constant HFD components with a richer, measure-adapted basis solved by enlarged least-squares, to kill exactly this discretization bias' — is the same move, already published and already compared against the same baseline."
    },
    {
      "title": "L2-Boosting on Generalized Hoeffding Decomposition for Dependent Variables — Application to Sensitivity Analysis (Chastaing & Gamboa, 2013, arXiv:1310.2532)",
      "url": "https://arxiv.org/abs/1310.2532",
      "overlap": "Gives a practical estimator of the Chastaing–Gamboa–Prieur generalized (dependent-input) Hoeffding decomposition via empirical Gram–Schmidt orthogonalization against a chosen basis of smooth functions and greedy L2-boosting, rather than piecewise-constant bins. This is the same recipe S3 uses at a higher level of abstraction (enforce hierarchical orthogonality against a function basis richer than bin indicators, fit by projection/least-squares), predating both TreeHFD and S3 by over a decade, though it is model-agnostic/general-basis rather than tied to a fitted tree ensemble's own split structure."
    }
  ],
  "rationale": "The idea's stated target — that TreeHFD's within-cell constant-density approximation is an irreducible source of bias under dependence at fixed tree depth (its own Theorem 6 term) — is not a novel diagnosis; it is the same premise a contemporaneous 2026 paper (Ferrere et al.) already identifies and solves with a full alternative estimator (smooth Riesz-basis least squares, hierarchically orthogonal by construction, benchmarked directly against TreeHFD). The general strategy of moving from bin-level orthogonality constraints to a smoother basis with orthogonality enforced via projection/least-squares also traces back to Chastaing & Gamboa's 2013 L2-boosting estimator of the generalized Hoeffding decomposition, itself built on the Chastaing-Gamboa-Prieur (2012) and Hooker (2007) line that TreeHFD already cites as its own ancestor. What is left as genuinely new in S3 is a specific engineering instantiation: reusing TreeHFD's own tree-partition machinery unmodified and grafting on conditional-CDF curves evaluated at the tree's own split thresholds (kernel-smoothed on the rank scale, restricted to variable pairs that actually co-occur on tree paths, with point-level moment constraints as a strict refinement of TreeHFD's bin-level constraints and an explicit fallback to the exact TreeHFD baseline on discrete features). That combination doesn't appear to exist as published work. But because the higher-level idea (richer-than-piecewise-constant basis + least squares + orthogonality-by-construction to fix exactly this dependent-input discretization bias) is already staked out, including a paper that benchmarks the same problem against the same baseline, this reads as an incremental basis-choice variant on an established estimation strategy rather than a new direction. Score reflects meaningful implementation novelty but low conceptual novelty."
}
```
