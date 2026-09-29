# engineer (claude-opus-5-5)（续接会话 3b98c049-136c-4ed9-870d-cb8c5e17dba0）

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


# Role: Engineering Agent
Improve the implementation in <ROOT>/runs/treehfd-01/ideas/E1 following the critic's feedback below: fix bugs, tune the method's own
hyper-parameters, or adjust the implementation — without changing the core idea and within the red lines.
Self-test with `<ROOT>/.conda/python.exe <ROOT>/bench/harness.py --method method.py --mode subset --out subset_selftest.json`.
Update NOTES.md with what you changed and why.

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

Current official results versus the baseline:
| metric | baseline | E1 |
|---|---|---|
| analytical.mse_eta1 | 0.02705 | 0.02689 (-0.6%) |
| analytical.mse_eta2 | 0.01818 | 0.01833 (+0.8%) |
| analytical.mse_eta3 | 0.01737 | 0.01712 (-1.4%) |
| analytical.mse_eta4 | 0.01769 | 0.01765 (-0.3%) |
| analytical.mse_eta5 | 0.0003277 | 0.0003019 (-7.9%) |
| analytical.mse_eta6 | 0.0005445 | 0.0005138 (-5.6%) |
| analytical.mse_eta12 | 0.03387 | 0.0321 (-5.2%) |
| analytical.mse_eta34 | 0.03168 | 0.0303 (-4.4%) |
| analytical.mse_others | 0.002148 | 0.001602 (-25.4%) |
| analytical.resid_out | 0.01022 | 0.007682 (-24.9%) |
| analytical.fit_s | 19.08 | 65.12 (+241.3%) |
| airfoil.resid_in | 0.009845 | 0.01204 (+22.3%) |
| airfoil.resid_out | 0.02695 | 0.01895 (-29.7%) |
| airfoil.ortho_in | 0.009669 | 0.01437 (+48.6%) |
| airfoil.ortho_out | 0.0989 | 0.09392 (-5.0%) |
| airfoil.locvar_in | 5.199e-05 | 2.888e-05 (-44.5%) |
| airfoil.fit_s | 3.31 | 9.655 (+191.7%) |
| concrete.resid_in | 0.001534 | 0.002531 (+65.0%) |
| concrete.resid_out | 0.02512 | 0.01975 (-21.4%) |
| concrete.ortho_in | 0.006359 | 0.006939 (+9.1%) |
| concrete.ortho_out | 0.08218 | 0.06965 (-15.2%) |
| concrete.locvar_in | 0.01249 | 0.01848 (+47.9%) |
| concrete.fit_s | 6.084 | 13.36 (+119.6%) |
| abalone.resid_in | 0.004553 | 0.00627 (+37.7%) |
| abalone.resid_out | 0.04034 | 0.01931 (-52.1%) |
| abalone.ortho_in | 0.0268 | 0.02798 (+4.4%) |
| abalone.ortho_out | 0.01586 | 0.04447 (+180.5%) |
| abalone.locvar_in | 0.00256 | 0.002454 (-4.1%) |
| abalone.fit_s | 21.95 | 33.82 (+54.1%) |

Critic feedback:
The subset table and the author's own NOTES.md are consistent, and the pre-registered checks (a)/(b) were largely carried through (rho=0 reproduces S4 bit-for-bit; brute-force LOO checks pass within ~5-10%), so there's no evidence of a red-line violation or a rigged result. resid_out is a genuine, consistent win (airfoil -29.7%, concrete -21.4%, abalone -52.1%, analytical -24.9%, all better than S4's own resid_out gains per NOTES.md), and analytical mse_others (-25.4%) and mse_eta5/6 also improve. That part of the idea is validated.

But the method's own stated main target failed, and it says so plainly: resid_in was supposed to roughly halve versus S4 (toward S2's ~+5%/+26%/+17%) but instead came back *unchanged or worse* than S4 (NOTES.md: 'the idea's main target was not reached: resid_in is unchanged relative to S4 (0.98-1.01x over 8 splits)'). Concrete resid_in is now worse than S4 (+65.0% vs baseline here, vs S4's own ~+41%). The mechanism identified by the author is that the label-free risk R saturates rho at 1 on >=95% of trees and then drives kappa to the floor of its grid, so the anchor is *replacing* S4's shrinkage rather than adding to it as intended, spending the freed fidelity budget on resid_in instead of banking it. This is a real, diagnosed failure mode, not noise.

Second issue: abalone.ortho_out regressed sharply (+180.5% vs baseline, and NOTES.md confirms this is worse than S4's own already-bad 2.597x -> 2.805x), and concrete shows a real ortho_in/ortho_out regression versus S4 (1.6x/1.4x per the 8-split ablation) that the author flags as a genuine trade-off rather than noise. A reader would reasonably call this a regression the idea didn't disclose in its 'expected effect' section (which predicted ortho_in would stay near S4 and only ortho_out would be noise-dominated).

Third, compute is not settled for the full run even though it's within limits on the subset (all four subset fit_s ratios are under 3.5x baseline, so no red-line issue here). NOTES.md reports that in full mode, bike (7.2x), nutrition (5.2x) and powerplant (5.1x) exceed the 5x guideline, and the exact speed-up for the anchored leave-out exists but is 'not validated' and therefore not in the submitted code. This must be resolved (validated and included, or the compute fallback tightened further) before a full run can be trusted to stay within the compute red line.

Concrete next steps: (1) finish and validate the anchored-leave-out speed-up so full-mode fit_s comes back under 5x everywhere, or extend the pre-registered fallback trigger to also restrict rho's grid/mode when threatened; (2) address the rho-saturation failure mode directly rather than accepting the corner solution - e.g. try the author's own observation that rho=0.25 is 'the resid_in-friendlier choice' on concrete/abalone/nutrition and consider a risk criterion or tie-break that doesn't purely chase fidelity (since R currently ignores the resid_in cost entirely); (3) investigate and mitigate the concrete/abalone ortho regression before claiming this beats S4 - right now it trades a real fidelity gain for a real, disclosed orthogonality cost that isn't clearly net-positive. The honesty of the writeup (reporting the failed hypothesis and the ortho regression instead of hiding them) is a good sign, but the method as submitted doesn't yet deliver what the idea promised and has an open compute compliance question for full mode.

Reply with JSON: {"status": "done|failed", "changes": str}


## Reply

I addressed all three critic points. Compute is fixed and the selection change helps a little, but E1 still doesn't halve resid_in and still doesn't beat S4 on orthogonality. The subset and full self-tests both ran without errors, and every full-mode fit time is now under 5× the baseline.

- **Compute:** round 1's unvalidated speed-up is now checked and in `lib/agtloco.py`.
  - On bike's heavy trees it matches the round-1 code to within 3e-10 and picks the same per-tree choices.
  - On all 32 diagnostic fits the plain-argmin selection gives exactly round 1's results.
  - Check (7a), which requires ρ = 0 to reproduce S4, was re-run through the harness on this machine. Every metric equals S4's, including the analytical case.
  - Full-mode fit times are now 1.12–4.80× baseline (round 1: up to 7.23×). Bike went from 7.23× to 4.07×, nutrition from 5.20× to 4.35×, powerplant from 5.14× to 4.80×.
  - Powerplant has little margin. Most of its time is the four eigendecompositions per tree, and no faster exact alternative helped. If a later run tips it over 5×, the fourth eigendecomposition is the justification; it is far from 20×.
- **ρ saturation and resid_in:** before running the ablation, I committed to S4's one-standard-error rule. Among candidates within one SE of the best risk it picks the one that fits the model best on its own training inputs. Over 8 splits:

  | E1 / S4 | airfoil | concrete | abalone | nutrition |
  |---|---|---|---|---|
  | resid_in | 0.94 | 0.99 | 0.98 | 0.96 |
  | resid_out | 0.86 | 0.91 | 0.91 | 0.93 |

  - E1 is now at or below S4 on both fidelity metrics on all four datasets, but halving S4's resid_in excess is still not reached. The rule only moved resid_in by 0 to −4% compared with round 1.
  - The reason: the risk estimate prefers ρ = 1 over ρ = 0.25 by more than one SE. In 23 of 32 fits the best ρ = 0.25 candidate fits the training inputs better but falls outside the tie set.
  - I did not switch to ρ = 0.25, because that would be choosing from held-out numbers. Its results are reported for the next round.
  - In the full run, resid_out is below S4 on 7 of 9 datasets and within 0.1% on bike and housing. resid_in is below S4 on 7 of 9 but higher on concrete (1.59× vs 1.41× baseline) and powerplant (1.12× vs 1.06×).
- **Orthogonality:** most of concrete's regression comes from the metric's 1% variance cut-off, not from worse orthogonality.
  - The anchor raises concrete's interaction variance by 21%, so more pairs cross the cut-off and enter the max. Measured on the same pairs for both methods, E1/S4 is 1.04/1.05 instead of 1.49/1.42.
  - A real cost of 4–7% remains, and it sits at small κ. At fixed κ ≥ 1 the anchor lowers both resid_in and resid_out, as the idea intended; the risk estimate pushes κ to the bottom of its grid, which is the costly regime.
  - It is not mitigated. The airfoil benchmark split's ortho_in (1.76× baseline) is a real increase on that split. Concrete's locvar_in is 1.69× baseline, and analytical spurious interactions are 0.816× baseline (S4 0.776×).

NOTES.md is rewritten for round 2; round 1's notes are kept in `dev/NOTES_round1.md`.

```json
{"status": "done", "changes": "Round 2 of E1 (all in <ROOT>/runs/treehfd-01/ideas/E1; NOTES.md rewritten, round-1 notes kept in dev/NOTES_round1.md). (1) COMPUTE (critic point 1): the round-1 exact leave-out speed-up is validated and now in lib/agtloco.py: cache-sized block chunks (BLOCK_CHUNK 3e5), exact size buckets up to 32 (BLOCK_EXACT), a rank-one fast path, needed virtual rows sliced once per tree, and hashed row-unique with exact verification. dev/check_speedup.py on bike trees 60-64: relative differences <= 2.6e-10 (loo), 5.7e-14 (risk), 2.7e-12 (betas); per-tree argmins identical; 1.64x faster. On all 32 ablation fits the plain argmin equals round 1's selection. Check 7a re-run through the harness on this machine: dev/method_rho0.py (RHOS=(0,), SE_RULE=0) equals S4/subset_env-22b414443084.json on every metric incl. analytical. Official full run (idle machine, no errors, wall 1746 s): fit_s/baseline abalone 1.70, airfoil 2.10, bike 4.07 (was 7.23), housing 2.64, concrete 2.92, nutrition 4.35 (was 5.20), parkinson 2.54, powerplant 4.80 (was 5.14), superconduct 1.12, analytical 3.37. All below 5x, no further fallback; powerplant has little margin (eigh 34 of 58 s; LAPACK driver alternatives benchmarked, none faster). (2) RHO SATURATION / RESID_IN (critic point 2): pre-committed S4's one-SE rule toward label-free in-sample fidelity (SE_RULE=1, k fixed before running, no other k tried; new helper _se_select; diagnostics min_risk and n_tied; SE_RULE=0 restores round 1 exactly). 8-split ablation (dev/ablate_r2.sh, dev/summarize_r2.py): E1/S4 resid_in 0.939/0.990/0.984/0.957 and resid_out 0.862/0.909/0.912/0.930 (airfoil/concrete/abalone/nutrition), so E1 is now at or below S4 on both fidelity metrics on all four. The one-SE rule itself moves resid_in only 0 to -4% vs round 1 (resid_out +0 to +2%). The idea's target (halve S4's resid_in excess) is still NOT reached. Measured reason: R prefers rho=1 over rho=0.25 by more than one paired SE (in 23/32 fits the best rho=0.25 candidate fits better in-sample yet is outside the tie set). rho=0.25 (resid_in-friendlier, e.g. abalone 1.19x vs 1.37x baseline) is reported but deliberately not adopted, since that would be choosing from held-out numbers. Full benchmark split vs S4: resid_out below S4 on 7/9 datasets and within 0.1% on bike and housing; resid_in below S4 on 7/9, above on concrete (1.59x vs 1.41x) and powerplant (1.12x vs 1.06x). Subset: airfoil resid_in 1.144x / resid_out 0.698x baseline, concrete 1.592x / 0.798x, abalone unchanged 1.377x / 0.479x, analytical resid_out 0.764x. (3) ORTHOGONALITY (critic point 3), investigated, not mitigated: at fixed common kappa, rho=1 lowers resid_in and resid_out when kappa>=1 (as the idea intended) but raises resid_in, and ortho_in on concrete (1.44-1.53x), when kappa<=0.1; R drives kappa to its floor. Recomputing ortho on the same pair set for both methods (dev/diag_ortho_common.py) shows concrete's harness regression vs S4 (ortho_in 1.49x, ortho_out 1.42x) is mostly the 1% variance threshold: the anchor raises interaction variance 21%, so more pairs enter the max (e.g. pair (0,7) share 0.83%->1.30% with nearly equal |corr|); on the same pairs E1/S4 is 1.04/1.05. A residual +4-7% real cost remains on concrete/abalone/nutrition. No claim that E1 beats S4 on orthogonality. Other disclosed costs: airfoil benchmark-split ortho_in 1.76x baseline (a real per-pair increase on that split; 0.81x S4 over 8 splits), concrete locvar_in 1.69x baseline (1.17x S4 over 8 splits), analytical mse_others 0.816x (S4 0.776x). Also corrected a transcription error in round 1's concrete ablation row."}
```
