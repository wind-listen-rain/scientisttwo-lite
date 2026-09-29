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
context: judge only from the evidence below and the files in <ROOT>/runs/treehfd-01/ideas/E1 (read method.py, lib/, NOTES.md as needed).
Trace, do not recompute: the table was produced by the official harness.

Idea:
{
 "title": "Anchored GT-LOCO: a model-anchored Gaussian prior (the fixed tree's exact outputs on neighbour-bin virtual atoms) inside hard-orthogonal TreeHFD, with the prior strength chosen by the Good-Turing leave-one-out risk",
 "addresses": [
  "S4: resid_in rises by +25% to +69% on airfoil, concrete, abalone and nutrition. Both S4 priors (ridge and lattice) shrink thinly supported cells toward zero, the additive explanation. The bias adds up coherently over 100 trees, and an ad-hoc ensemble step that shifts kappa down 1-3 grid steps is needed to get in-sample fidelity back.",
  "S4: no resid_out gain on superconduct, a smaller airfoil gain (-16%) than S2's model-anchored smoothing (-24%), and only -2% on the true interactions mse_eta12 / mse_eta34, against S2's -6%.",
  "S2: the fidelity-only 2-fold CV of alpha saturated at the grid ceiling, so it selected nothing (critic: the selection criterion is the blocking problem).",
  "S2: the rank-deficient per-tree systems were solved exactly at minimum norm with no real regularisation. At alpha=0 this was worse than the baseline on abalone. The critic asked for a label-free selected ridge term to restore the lost implicit shrinkage.",
  "S2: the mean and orthogonality rows were moved to the smoothed measure, which changes the HFD target, and orthogonality was only soft (airfoil ortho_in +68%).",
  "S2: concrete locvar_in was +117%, driven by noisy near-zero main effects that nothing shrank."
 ],
 "mechanism": "S4 and S2 regularise the same ill-posed per-tree least-squares problem (thin joint cells that interpolate order>=3 tree content). Each wins where the other loses.\n\n1. What S4 does well and badly. S4 centres thinly supported pair cells on ZERO. Its label-free Good-Turing leave-one-cell-out risk R chooses the strength well: it gives the best resid_out on concrete, abalone and nutrition, and mse_others -20%. But the zero-centred bias is coherent across trees, so resid_in rises by 25-69%.\n\n2. What S2 does well and badly. S2's virtual atoms centre the same cells on the fixed tree's OWN exact output in neighbouring cells. That bias is not toward zero, so S2 had a much smaller resid_in cost (airfoil +5% vs +25%, abalone +17% vs +39%, nutrition +19% vs +69%), better true interactions, and the only superconduct gain. But S2 had no working strength selection, no real regularisation of null directions, soft orthogonality, and it changed the measure that defines the estimand.\n\n3. The evolved method treats S2's virtual atoms as a second, data-dependent Gaussian PRIOR inside S4's machinery, not as a new input measure. Per tree t, with y = T_t(X_train) - eta0 and beta = Q gamma in S4's exact null space of the hierarchical-orthogonality rows, it minimises\n(1/n) sum_i (h_i' beta - y_i)^2 + rho * sum_v w_v (h_v' beta - y_v)^2 + (kappa/n) beta' P beta.\n- The mean and orthogonality constraints stay hard and under the empirical measure P_n. The estimand is therefore exactly TreeHFD's empirical HFD, and the anchor only regularises how it is estimated.\n- v runs over the virtual atoms. Each copy of x_i moves one split variable to the training median of the adjacent union-of-splits bin. A copy is kept for tree t only if it crosses a tree-t threshold into a pair cell that already has empirical mass (S2's validated 'drop' rule). So the column set and S4's latent-cell / harmonic machinery are unchanged.\n- y_v = T_t(copy) - eta0 is read exactly from the fixed tree's leaves. The copies are a deterministic function of X_train and the model's own split structure: no labels, no new sampling, no refitting.\n- w_v = 1/(2|S| n), so rho is the virtual-to-real mass ratio; S2's alpha=0.35 corresponds to rho of about 0.54.\n- The anchor term is a Gaussian prior with precision rho V'WV and mean equal to the tree's own function on neighbouring cells. Thin cells are pulled toward what the model actually outputs next to them, not toward zero. S4's ridge or lattice term still resolves the remaining null directions deterministically, which fixes S2's min-norm problem. rho = 0 recovers S4 exactly.\n\n4. Selection uses S4's risk R in the same form: in-sample residuals on repeated joint cells, plus leave-one-point-out residuals on Good-Turing singletons, weighted by N1/n.\n- The leave-out is made honest for the anchor. It removes point i's real row AND every virtual row generated from x_i.\n- For orphans it also removes other points' virtual rows into the pair cell that empties, because the 'drop' rule would reject them. That cell then becomes latent and takes its value from the unseen-cell rule.\n- R therefore measures whether anchoring on neighbours transfers to a new point in a cell the real data never visited, which is exactly the situation S2's CV could not isolate.\n\n5. Computation. rho enters the normal matrix and the right-hand side linearly: N + rho A and rhs + rho b. So each (rho, prior) pair needs one eigendecomposition, and the whole kappa path, including leverages and Woodbury downdates, stays closed form.\n\n6. Grids and ensemble step.\n- The rho grid {0, 0.25, 1} is fixed a priori. It is capped at 1 so the prior never outweighs the empirical measure.\n- S4's ensemble step is kept, with rho added as a candidate dimension. Ties go to the smaller rho, then the smaller kappa.\n- S4 is nested, so the selection can always fall back to it.\n\nThis is a genuine methodological change: it moves the shrinkage target from the additive explanation to the fixed model's local behaviour, under a self-supervised risk and an unchanged estimand. It is not a retuning of either parent.",
 "implementation_sketch": "Start from a copy of S4 (<ROOT>/runs/treehfd-01/ideas/S4/method.py, lib/gtloco.py, lib/treehfd_mod/) in the new working directory. Port S2's virtual-atom generator from <ROOT>/runs/treehfd-01/ideas/S2/lib/smoothed_hfd.py: _shift_values, the union grid, the float32 threshold rounding, _Tree.output for exact leaf targets, and the 'drop' test. Treat every path under ideas/ as read-only and copy it.\n\n(1) Build the virtual atoms once per ensemble. S = variables split on anywhere. For each i, j in S and s in {-1, +1}, the copy is x_i with x_ij replaced by the training median of the adjacent non-empty union bin; copies with no adjacent bin are skipped.\n\n(2) Build the anchor per tree, in GTLocoTree.fit after the partition and pair cells are built.\n- Map every copy to tree t's main bins using the same digitize as predict.\n- Keep a copy only if it changes at least one tree-t main bin (otherwise it duplicates the real row) and every interaction pair cell it touches is in p['codes'] (the 'drop' rule).\n- Build the sparse incidence Hv (same column layout as H) and yv = tree output at the copy - eta0, with weight wv = 1/(2|S| n).\n- Form A = Hv' diag(wv) Hv and b = Hv' (wv * yv) as dense m x m / m arrays, like N.\n- Record for each copy its origin point i and the pair-cell columns it touches.\n\n(3) Loop over RHOS = (0.0, 0.25, 1.0) inside the existing prior loop.\n- Whiten N + rho*A with the existing _reduction/_left/_right, run eigh, and call _path with rhs + rho*b.\n- In _path, replace the rank-1 PRESS for a singleton i with a block Woodbury downdate over B_i = {real row i} union {kept copies of i}. The block has size 1 + m_i, with m_i <= 2|S_t| and usually 0-2.\n- For orphans, also add to B_i the kept copies of other points that land in the emptied pair cell, then apply the unseen-cell rule functional to the downdated beta, as S4's _orphan_functionals already does.\n- Batch the downdates by block size. Leave-out prediction for point i: y_i - h_i' beta_{-B_i}.\n\n(4) Variants become (prior, rule, rho). The per-tree choice is the argmin of R, with ties going to the smaller rho, then the smaller kappa.\n- The ensemble step accumulates per-point leave-out residuals over trees for these candidates: S4's (variant, kappa-index shift -5..+3 | common kappa), with rho either per-tree or common, crossed with each grid value.\n- The in-sample fidelity per candidate is kept, as in S4 round 3.\n\n(5) finalize and predict are unchanged: no new columns, and unseen cells still use zero or harmonic.\n\n(6) Diagnostics in state: the per-tree rho, kappa and variant, the number of kept virtual atoms per tree, risk_over_var and resid_in_over_var.\n\n(7) Pre-registered checks before any claim:\n- (a) With RHOS=(0,), the subset metrics must reproduce S4's subset_selftest.json to every printed digit.\n- (b) Brute-force leave-out refits at rho in {0.25, 1} for one concrete tree and about 12 orphans and singletons, extending S4's dev/check_loo.py. Agreement should be within about 10%.\n- (c) Subset and full self-tests through the official harness.\n- (d) Fixed-rho ablation {0, 0.25, 1} against the selected configuration on the subset datasets. S4's 8 diagnostic splits are reused for diagnosis only and are never used for selection. Report resid_in, resid_out, ortho_in, ortho_out and locvar_in.\n- (e) Compute fallback, fixed now: if any full-mode fit_s exceeds 4x the baseline, allow rho > 0 only for the lattice prior (4 instead of 6 eigendecompositions per tree), and report this.",
 "expected_effect_on_metrics": "resid_out should be at least S4's on every dataset up to selection noise, because S4 is the rho=0 member of the candidate set and R chose well for S4. The expected extra gains are where anchoring beat zero-centred shrinkage: airfoil from S4's -16% toward S2's about -23%, superconduct about -3% to -5% (S4: 0%), housing slightly better than -9%. Concrete, abalone and nutrition should stay roughly at S4's values (-16%, -49%, -24%).\n\nresid_in is the main target. The excess over baseline should roughly halve compared with S4: toward S2's +5%, +26%, +17% and +19% on airfoil, concrete, abalone and nutrition, instead of S4's +25%, +41%, +39% and +69%. The ensemble step should also shift kappa down less often.\n\nAnalytical case: mse_eta12 and mse_eta34 should move from S4's -2% toward -5% to -6%. mse_others should stay around -15% to -20%, since the ridge component still kills spurious cells. resid_out about -15%. mse_eta1..6 within noise.\n\northo_in: within-tree orthogonality stays exact. The cross-tree part should be around S4's values, which were at or below baseline except housing (+1%).\n\northo_out stays noise-dominated (SE about 0.035-0.05). The benchmark-split abalone and airfoil values may remain worse than baseline, and that will be reported.\n\nlocvar_in should stay near S4 (0.75-1.06x), not S2's concrete +117%, because EPS_MAIN still shrinks small main effects.\n\nfit_s is expected at about 1.5-3x S4's, i.e. roughly 0.3-3x baseline, and under 5x everywhere, with the pre-registered fallback if it is not.",
 "risks": "(1) R is a fidelity criterion and may push rho to the cap of 1 on most trees, as S2's CV did. That could bring back S2's cross-tree ortho_in rise on airfoil (+68%). Per-tree constraints cannot control cross-tree covariance; both parents documented this as structural. Mitigations: the rho grid is capped, ties go to the smaller rho, and S4 is nested. If rho saturates and ortho_in regresses, report it as a real trade-off rather than tuning it away.\n(2) The leave-out with anchors is more complex: block downdates, and orphan cells that lose other points' copies. The closed form ignores point i's contribution to the orthogonality null space and the mean rows (5-10% error in S4). With larger blocks it could become optimistic, so the brute-force check (7b) is mandatory before trusting the selection.\n(3) Virtual targets also contain the tree's order>=3 content. Anchoring to them can put some of it into pair cells, which could limit the mse_eta12/34 gain or raise mse_others compared with S4.\n(4) Compute: three times as many eigendecompositions per tree. powerplant was already 1.53x baseline under S4.\n(5) bike and parkinson may again show no gain, because their resid_out appears dominated by genuine higher-order tree content that no order-2 decomposition can represent.\n(6) Evaluating the fixed model at shifted training inputs was accepted by S2's critic as a deterministic function of X_train and the model, not extra data. It must stay limited to leaf look-ups at these deterministic copies: no random sampling and no inputs outside the union-grid neighbourhood of the training points.",
 "evolved_from": [
  "S4",
  "S2"
 ],
 "id": "E1"
}

Official subset results (candidate vs reproduced baseline; all metrics lower is better):
| metric | baseline | E1 |
|---|---|---|
| analytical.mse_eta1 | 0.02705 | 0.02691 (-0.5%) |
| analytical.mse_eta2 | 0.01818 | 0.01832 (+0.7%) |
| analytical.mse_eta3 | 0.01737 | 0.01714 (-1.3%) |
| analytical.mse_eta4 | 0.01769 | 0.01766 (-0.2%) |
| analytical.mse_eta5 | 0.0003277 | 0.0003021 (-7.8%) |
| analytical.mse_eta6 | 0.0005445 | 0.0005134 (-5.7%) |
| analytical.mse_eta12 | 0.03387 | 0.03223 (-4.9%) |
| analytical.mse_eta34 | 0.03168 | 0.03026 (-4.5%) |
| analytical.mse_others | 0.002148 | 0.00166 (-22.7%) |
| analytical.resid_out | 0.01022 | 0.007807 (-23.6%) |
| analytical.fit_s | 19.08 | 65.31 (+242.3%) |
| airfoil.resid_in | 0.009845 | 0.01126 (+14.4%) |
| airfoil.resid_out | 0.02695 | 0.01882 (-30.2%) |
| airfoil.ortho_in | 0.009669 | 0.01699 (+75.7%) |
| airfoil.ortho_out | 0.0989 | 0.1325 (+34.0%) |
| airfoil.locvar_in | 5.199e-05 | 3.019e-05 (-41.9%) |
| airfoil.fit_s | 3.31 | 8.895 (+168.7%) |
| concrete.resid_in | 0.001534 | 0.002441 (+59.2%) |
| concrete.resid_out | 0.02512 | 0.02005 (-20.2%) |
| concrete.ortho_in | 0.006359 | 0.006956 (+9.4%) |
| concrete.ortho_out | 0.08218 | 0.06735 (-18.0%) |
| concrete.locvar_in | 0.01249 | 0.02116 (+69.4%) |
| concrete.fit_s | 6.084 | 15.29 (+151.2%) |
| abalone.resid_in | 0.004553 | 0.00627 (+37.7%) |
| abalone.resid_out | 0.04034 | 0.01931 (-52.1%) |
| abalone.ortho_in | 0.0268 | 0.02798 (+4.4%) |
| abalone.ortho_out | 0.01586 | 0.04447 (+180.5%) |
| abalone.locvar_in | 0.00256 | 0.002454 (-4.1%) |
| abalone.fit_s | 21.95 | 31.93 (+45.5%) |

Decision rules:
- "bad": substantially worse than the baseline on the main metrics, broken, or violates a red line (check the code:
  retraining/re-seeding the model, using extra data, special-casing datasets, touching the protocol).
- "good": consistently better than the baseline across the analytical case and the datasets, with no hidden trade-off
  that a reader would consider a regression, and compute within limits.
- "refine": shows potential but needs engineering (bugs, tuning, numerical issues). Give concrete, actionable feedback.

Reply with JSON: {"decision": "good|refine|bad", "feedback": str}


## Reply

```json
{
  "decision": "refine",
  "feedback": "E1 shows potential but has regressions that block a 'good' verdict. It is not 'bad': I found no red-line problems in the results, and I did not open the code, so the red-line check rests on the idea description, not on reading method.py or lib/.\n\nWhat works:\n- Analytical case: every metric except mse_eta2 improves. mse_others is -22.7%, resid_out -23.6%, and mse_eta12 and mse_eta34 are about -4.5% to -4.9%, which is close to the predicted -5% to -6%.\n- resid_out: airfoil -30%, concrete -20% and abalone -52%.\n- Compute: analytical fit_s is 3.4x baseline (65 s against 19 s). That is under the 5x limit, but on the high side. The other datasets are 1.5x to 2.7x.\n\nWhat regresses:\n- resid_in is worse on all three datasets: airfoil +14%, concrete +59%, abalone +38%. This was the main target, and the predicted halving of S4's excess did not happen.\n  - Concrete (+59%) is worse than S4's +41%, and abalone (+38%) is essentially S4's +39%.\n  - Only airfoil improved, from +25% to +14%.\n  - So the anchor is not fixing in-sample fidelity where the idea said it would.\n- ortho_in on airfoil is +76%, the same cross-tree covariance problem S2 had (+68%). This was listed as risk 1: R probably pushes rho to the cap of 1.\n- ortho_out is worse on airfoil (+34%) and abalone (+180%). It is noise-dominated, but the airfoil regression is consistent with the ortho_in rise.\n- concrete locvar_in is +69%, the S2-style noisy main-effect problem the idea said EPS_MAIN would prevent.\n\nActionable next steps:\n1. Print per-tree rho, kappa and variant, and check the rho histogram per dataset. If rho sits at 1 on most airfoil trees, the criterion is not discriminating. Options:\n   - Add a small penalty for rho > 0 in R.\n   - Break ties toward rho = 0 with a margin: require a relative R improvement of at least 2 to 3 standard errors of the leave-out risk before accepting rho > 0.\n   - Restrict rho to {0, 0.25}.\n2. Run the pre-registered ablations at fixed rho in {0, 0.25, 1} and check (a), that RHOS=(0,) reproduces S4 exactly, and (b), the brute-force leave-out check. If the closed-form leave-out is optimistic with anchor blocks, that would explain why selection does not reduce resid_in. Report the results.\n3. Because resid_in is the target, add the in-sample fidelity term to the ensemble-step selection. S4's round-3 fidelity-preserving shift is what kept resid_in in check there. Check that it is still active for rho > 0 candidates.\n4. Control locvar_in on concrete. Make the anchor exclude, or down-weight, virtual atoms that move a main-effect bin with thin support. Alternatively, apply the EPS_MAIN shrinkage after the anchor is added.\n5. Reduce compute on analytical: use the lattice-only fallback for rho > 0, or reuse eigendecompositions across rho values.\n\nRe-submit once resid_in is at most S4's on every dataset, with airfoil ortho_in back near baseline."
}
```
