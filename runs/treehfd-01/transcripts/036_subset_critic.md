# subset_critic (claude-sonnet-5-5)

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


# Role: Subset Critic
Decide whether an idea's implementation beats the reproduced baseline on the benchmark subset. You work in a clean
context: judge only from the evidence below and the files in <ROOT>/runs/treehfd-01/ideas/S1 (read method.py, lib/, NOTES.md as needed).
Trace, do not recompute: the table was produced by the official harness.

Idea:
{
 "title": "Ensemble-level re-orthogonalization on the union-of-splits partition (sum-preserving, cross-fitted smoothed projection)",
 "addresses": [
  "The piecewise-constant density approximation leaves an irreducible bias at the fixed, shallow tree depth used in the benchmark",
  "Spurious interaction candidates are never pruned, leaking noise into components that should be exactly zero"
 ],
 "mechanism": "TreeHFD makes each tree's interaction orthogonal only to functions that are piecewise constant on that tree's own coarse partition of x_j. The paper's Theorem 3(ii) admits that the summed ensemble interaction eta_jk = sum_l eta_jk^l is NOT orthogonal to the summed main effects. The covariance is bounded only by Delta_{A,f}, the same discretization term that appears in the Theorem 6 error bound. The baseline shows this directly: ortho_in = 0.082 on parkinson and 0.028 on housing, even in-sample. The aggregated main effect eta_j is piecewise constant on the union U_j of all split thresholds on x_j across the ensemble. This union grid is much finer than any single depth-6 tree's grid, and we get it at no cost from the fixed model. The new step is a global, post-hoc projection. For each retained pair (j,k), solve an additive least-squares problem under the empirical measure of X_train: (g_j, g_k) = argmin E_n[(eta_jk - g_j(x_j) - g_k(x_k))^2] + lambda * R(g). Here g_j lives on the union bins U_j (merged to a minimum count), and R is a first-difference (P-spline-like) penalty along the ordered bins, so the transfer is smooth and generalises out of sample. Then set eta_jk <- eta_jk - g_j - g_k, eta_j <- eta_j + g_j and eta_k <- eta_k + g_k, and move the constants into the intercept. Hierarchical orthogonality only requires eta_jk to be orthogonal to functions of its own subsets, so the pairs can be projected independently and in any order. The move is pointwise sum-preserving, so the reconstruction (and therefore resid_in and resid_out) is unchanged at every x. The only change is how T is split among components: each interaction becomes orthogonal to the whole union-bin function space of x_j and x_k, which contains the final eta_j and eta_k. This is a finer discretization of the true HFD constraint, and it shrinks the effective Delta_{A,f} without retraining. The penalty lambda is picked label-free by K-fold cross-fitting on X_train only: fit the projection on K-1 folds and score the held-out |corr(eta_jk, eta_j)|.",
 "implementation_sketch": "method.py: (1) Run TreeHFD as in the baseline (bench/baseline_method.py logic, reusing the treehfd package) to get per-tree components. (2) From model.get_booster().trees_to_dataframe(), collect all split thresholds per feature. Build union edges U_j, then greedily merge adjacent bins that hold fewer than m_min (e.g. 10) X_train points. Store the edges. (3) Evaluate the aggregated eta_jk on X_train (n x K) and the bin indices b_j = digitize(X_train[:, j], U_j). (4) For each pair, assemble the normal equations with np.bincount: the Gram blocks are diagonal counts for (j,j) and (k,k) and a joint count table for (j,k). Add lambda * D^T D difference penalties and solve the small dense (B_j + B_k) system with a fixed gauge (centred g). Cost is O(n + B^2) per pair, so seconds even for superconduct. (5) Pick lambda from a small grid by 3-fold cross-fitting on X_train rows (deterministic fold split). (6) At predict time, compute the baseline components, then subtract g_j[digitize(x_j)] + g_k[digitize(x_k)] from each interaction column and add them to the main-effect columns. The intercept absorbs the centring constants. No RNG is used, and the added fit time is expected to be <10% of baseline.",
 "expected_effect_on_metrics": "ortho_in should drop to about 0 up to the lambda smoothing (largest gains on parkinson 0.082, housing 0.028, abalone 0.027). ortho_out should drop substantially on airfoil, concrete, nutrition and parkinson (currently 0.08-0.12), because the smoothed transfer generalises. resid_in and resid_out are unchanged by construction. Analytical: lower mse_eta12 and mse_eta34, since the true eta_jk are exactly orthogonal to all functions of their parent variables, and main-effect mass leaked into interactions returns to eta_1..eta_4. mse_others should fall, because the additive part of spurious pairs such as (1,3), created by per-tree coarse discretization under rho=0.5, is moved to main effects. locvar_in stays neutral or improves slightly thanks to the difference penalty on g_j. fit_s rises by about 1.0-1.1x.",
 "risks": "(a) Moving mass into main effects could slightly raise locvar_in on the finest union bins if lambda is too small. This is mitigated by the difference penalty and the minimum-count merging. (b) Orthogonality is enforced under the empirical training measure, so ortho_out gains depend on how well the cross-fitted lambda generalises; small-n datasets (concrete, n=824 train) may gain less. (c) If the aggregated interaction for a true pair is mostly pure interaction already, mse_eta12 and mse_eta34 may barely move; the gain is then mainly on ortho metrics. (d) The union bins of a variable split often (e.g. sin(2*pi*x1)) can be numerous, but B_j is at most a few hundred after merging, so the dense solves stay cheap. (e) This step changes neither the model nor the reconstruction, so it stays within red lines 2-3.",
 "id": "S1",
 "novelty": 3,
 "closest": [
  {
   "title": "Purifying Interaction Effects with the Functional ANOVA: An Efficient Algorithm for Recovering Identifiable Additive Models (Lengerich, Tan, Chang, Hooker, Caruana; AISTATS 2020)",
   "url": "https://arxiv.org/abs/1911.04974",
   "overlap": "This is an exact, sum-preserving 'purification' algorithm for piecewise-constant functions (tree ensembles / GA2Ms with interactions) that redistributes mass between interaction and main-effect terms to restore functional-ANOVA orthogonality/identifiability under the empirical (training) distribution, leaving the reconstruction unchanged. That is precisely the core mechanism proposed here (project eta_jk onto functions of x_j and x_k, subtract the projection from the interaction, add it to the mains, sum-preserving, under the empirical measure). The idea's contribution on top of this is largely implementation detail: doing the projection on the union-of-tree-split bins rather than an arbitrary Cartesian partition, and adding a P-spline-like smoothing penalty with cross-fitted lambda so the correction generalizes out-of-sample (targeting ortho_out) instead of Lengerich's exact in-sample purification."
  },
  {
   "title": "Achieving interpretable machine learning by functional decomposition of black-box models into explainable predictor effects (npj Artificial Intelligence, 2025)",
   "url": "https://arxiv.org/abs/2407.18650",
   "overlap": "Introduces 'stacked orthogonality' and an efficient post-hoc orthogonalization procedure that moves explanatory mass from higher-order interactions into main effects (and lower-order into higher, hierarchically) so that main effects capture as much signal as possible without leaking into interactions — the same hierarchical-orthogonality-restoration goal TreeHFD already targets, solved post-hoc rather than during fitting. It differs from the idea by operating on a neural-additive-model surrogate rather than directly on tree-ensemble split partitions, and it doesn't appear to add a smoothing/regularization step for out-of-sample generalization."
  }
 ]
}

Official subset results (candidate vs reproduced baseline; all metrics lower is better):
| metric | baseline | S1 |
|---|---|---|
| analytical.mse_eta1 | 0.02705 | 0.02652 (-1.9%) |
| analytical.mse_eta2 | 0.01818 | 0.01763 (-3.0%) |
| analytical.mse_eta3 | 0.01737 | 0.01577 (-9.2%) |
| analytical.mse_eta4 | 0.01769 | 0.01645 (-7.1%) |
| analytical.mse_eta5 | 0.0003277 | 0.0003328 (+1.5%) |
| analytical.mse_eta6 | 0.0005445 | 0.0005578 (+2.4%) |
| analytical.mse_eta12 | 0.03387 | 0.03288 (-2.9%) |
| analytical.mse_eta34 | 0.03168 | 0.0294 (-7.2%) |
| analytical.mse_others | 0.002148 | 0.002148 (-0.0%) |
| analytical.resid_out | 0.01022 | 0.01026 (+0.3%) |
| analytical.fit_s | 19.08 | 19.41 (+1.7%) |
| airfoil.resid_in | 0.009845 | 0.009845 (-0.0%) |
| airfoil.resid_out | 0.02695 | 0.02648 (-1.7%) |
| airfoil.ortho_in | 0.009669 | 0.009669 (+0.0%) |
| airfoil.ortho_out | 0.0989 | 0.09708 (-1.8%) |
| airfoil.locvar_in | 5.199e-05 | 5.199e-05 (-0.0%) |
| airfoil.fit_s | 3.31 | 2.988 (-9.7%) |
| concrete.resid_in | 0.001534 | 0.001534 (-0.0%) |
| concrete.resid_out | 0.02512 | 0.02506 (-0.3%) |
| concrete.ortho_in | 0.006359 | 0.006359 (-0.0%) |
| concrete.ortho_out | 0.08218 | 0.08182 (-0.4%) |
| concrete.locvar_in | 0.01249 | 0.01249 (-0.0%) |
| concrete.fit_s | 6.084 | 5.23 (-14.0%) |
| abalone.resid_in | 0.004553 | 0.004553 (-0.0%) |
| abalone.resid_out | 0.04034 | 0.04048 (+0.4%) |
| abalone.ortho_in | 0.0268 | 0.009651 (-64.0%) |
| abalone.ortho_out | 0.01586 | 0.0162 (+2.2%) |
| abalone.locvar_in | 0.00256 | 0.00256 (+0.0%) |
| abalone.fit_s | 21.95 | 17.16 (-21.8%) |

Decision rules:
- "bad": substantially worse than the baseline on the main metrics, broken, or violates a red line (check the code:
  retraining/re-seeding the model, using extra data, special-casing datasets, touching the protocol).
- "good": consistently better than the baseline across the analytical case and the datasets, with no hidden trade-off
  that a reader would consider a regression, and compute within limits.
- "refine": shows potential but needs engineering (bugs, tuning, numerical issues). Give concrete, actionable feedback.

Reply with JSON: {"decision": "good|refine|bad", "feedback": str}


## Reply

```json
{"decision":"refine","feedback":"S1 is not broken and violates no red line. It is sum-preserving, uses no labels, and does not touch the model, so resid_in is unchanged and resid_out moves by at most +0.4% (+0.3% analytical, +0.4% abalone). But it does not consistently beat the baseline, and the current default is far from the idea's claimed mechanism.\n\nEvidence:\n- On airfoil and concrete, ortho_in and locvar_in are identical to the baseline. ortho_out changes by only -1.8% and -0.4%, which NOTES.md itself calls noise. Only abalone changes: ortho_in -64%, but ortho_out +2.2%.\n- The analytical gains are small, mostly -2% to -9% on mse_eta1..4, mse_eta12 and mse_eta34. mse_eta5 and mse_eta6 regress (+1.5% and +2.4%). mse_others is exactly unchanged, so the spurious-pair purification the idea promised did not happen.\n- The expected effect, ortho_in about 0 everywhere and ortho_out substantially lower, was not delivered. The union-of-splits smoothed transfer (STEP1) is disabled by default, and NOTES.md says it showed no benefit. What runs is a significance-shrunk, restricted Lengerich-style purification. That is essentially the closest prior work the idea cites, so the novelty is weak.\n- Compute is fine: fit_s is +1.7% on analytical and lower on the real datasets, which is machine noise.\n\nActionable feedback:\n1. Find out why the exact or lightly-regularized union-bin projection did not reduce ortho_in on airfoil and concrete. In-sample orthogonality should approach 0 by construction, so a leftover of 0.0097 or 0.0064 suggests the shrinkage gate is switching the projection off. It may also be that the ortho metric is dominated by something the projection does not span. Check with an unshrunk, exact in-sample projection to confirm the metric can actually reach 0.\n2. Replace the significance gate, which zeroes out the correction on small leaks, with a smooth, label-free shrinkage such as the cross-fit lambda in the original idea. Use it to get real ortho_out gains on datasets with leaks (parkinson, nutrition, housing) while keeping abalone ortho_out from rising.\n3. Fix the eta5 and eta6 regressions and the abalone ortho_out increase. Bound the change to main effects of variables with no interaction leak, or drop the per-rep rescaling that moves x5 and x6.\n4. Re-run the subset. Accept only if the ortho metrics improve on at least two of the three real datasets and no metric regresses by more than about 1%."}
```
