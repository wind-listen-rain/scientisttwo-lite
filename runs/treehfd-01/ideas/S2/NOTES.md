# S2 — Model-anchored smoothed-measure TreeHFD

## What was implemented
`method.py` → `lib/smoothed_hfd.py` (`SmoothedTreeHFD`). It is a self-contained re-implementation of the per-tree
TreeHFD fit. From the package copy (`lib/treehfd_mod`, unmodified) it reuses only the tree-path helpers
(`extract_variable_paths`, `extract_variables`, `extract_interactions`), so the per-tree variable set and
interaction list are exactly the baseline's (depth_variable = max_depth). Interface is unchanged:
`fit(model, X_train, interaction_order=2)` / `predict(state, X)`.

1. **Smoothed input measure** (the core idea). P_n is replaced by
   `mu~ = (1-α) P_n + α/|S| Σ_{j∈S} Σ_{s=±1} ½ P_n∘Shift_{j,s}^{-1}`.
   * S = variables split on anywhere in the ensemble. The union-of-splits grid U_j = all (float32) thresholds on x_j over all trees.
   * Shift_{j,s} moves x_ij to the **training median of the adjacent non-empty union bin** (s=±1), rounded to float32.
     A copy with no adjacent bin (first/last non-empty bin) keeps its mass on the original point, so every training
     point carries exactly 1/n total mass and mu~ is a probability measure.
   * The copies are a deterministic function of X_train. No RNG, labels, test inputs or new samples are used.
2. **Per tree:** Cartesian partition from X_train (same empty-bin merging rule as the package). Each shifted copy is
   dropped into the tree's own partition. If the shift crosses one of the tree's thresholds on x_j, the copy is a
   *virtual atom*: its target is the tree's **exact output at the copy**, obtained by traversing the fixed tree (checked:
   Σ_t T_t(x) equals XGBoost's margin to 4e-5 on a scale of 80, i.e. float32 accumulation). Copies that do not cross a
   threshold add their weight to the original point's atom. Atoms are deduplicated into full cells with summed mu~ weights.
   *Revision 3 (`NEW_PAIR_CELLS = "drop"`, default):* a copy whose shift would put any of the tree's interactions
   into a pair cell (b_j, b_k) **without empirical mass** is rejected for that tree. Its mass stays on the original
   point, exactly like a copy that crosses no threshold. So virtual atoms only re-weight and re-anchor pair cells the
   data already populate. Pair-cell columns = pair cells with empirical mass, the same column set as α = 0. Revisions 1–2
   (`"fit"`) gave every pair cell reached by a copy its own column. See "Revision 3".
3. **Least-squares rows, all under mu~** (at α=0 this is the original system exactly, with the same n-scaling):
   residual rows `n·sqrt(mu~(cell))·(Σ coefs − (T − η0))`, zero-mean rows with the smoothed main/pair marginals, and
   hierarchical-orthogonality rows `n·mu~(c)/sqrt(mu~_j(b))`. The intercept per tree is η0 = E_mu~[T_t].
   Assembled directly as scipy.sparse COO. No dense `num_cells × partition_size` arrays.
4. **Solver with convergence check.** LSMR with atol = btol = 1e-10 and conlim = 1e12. The per-tree systems are frequently
   **rank deficient** (13–250 null directions per tree on concrete), and the baseline's LSQR implicitly returns the
   minimum-norm solution. Plain LSMR at 1e-10 needed ~6000 iterations/tree at α>0 (45 s on concrete; it would take
   about 3700 s on superconduct). So LSMR is **right-preconditioned by the Cholesky factor R of AᵀA + λI**
   (λ = 1e-11·max diag(AᵀA)). Starting from z=0, LSMR on A R⁻¹ returns the minimiser of ‖Rx‖² = ‖Ax‖² + λ‖x‖² among
   LS solutions, i.e. the same min-norm solution. It agrees with converged plain LSMR to ~1e-6 relative on concrete α=0,
   in 2–3 iterations/tree. istop is checked: on 3/6/7 (ill-conditioned / iteration limit) LSMR is re-run warm-started
   with 5× the iteration limit, then a dense `lstsq` on the normal equations is used. If Cholesky fails, λ×100 (up to
   3 times), then un-preconditioned LSMR is used. In the runs whose diagnostics I inspected (analytical rep 0, airfoil,
   concrete, abalone, nutrition), all 100 trees converged at the first attempt with istop=2, in at most 3 iterations,
   with no retries or direct fallbacks. I did not inspect superconduct's solver diagnostics.
   *No column scaling* is used: it changes which null-space solution is returned (60–70% different coefficients).
5. **Deterministic, geometry-aware fallback.** For each interaction a full lookup table over (b_j, b_k) is precomputed.
   Cells that own a column get their coefficient (under "drop": the cells with empirical mass). Others get the
   coefficient of the nearest trained cell under the
   Euclidean distance between bin midpoints (bins clipped to the training range), each coordinate divided by its
   training IQR (std, then 1, if the IQR is 0). Ties go to the largest smoothed mass, then the lowest cell index.
   Prediction is `np.searchsorted` + table lookup (vectorised; no random tie-breaking). The mapping is in
   `_nearest_cell`, which the "tie" ablation also uses.
6. **float32 exactness.** Thresholds parsed from `trees_to_dataframe` are rounded to float32, as are the inputs, so
   partitions route points exactly as XGBoost does.
7. **α selection, label-free.** 2-fold cross-fit on X_train (fixed permutation, seed 0). For each fold, fit on one half
   (medians and partitions from that half) for every α in the grid, predict the other half, and score
   `mean((Σ_t T_t − Σ_t recon_t)²)/Var(Σ_t T_t)` over every 5th tree (20 trees). T_t is known on held-out training inputs
   because the model is only evaluated. One `_prepare_tree` per (fold, tree) is shared by all α, since atom weights are
   linear in α.
   *Revision 2:* α is picked with the **one-standard-error rule**: the smallest α whose CV score is within one paired
   SE of the best score. The SE comes from per-point differences to the best α, pooled over both folds. Before, the
   argmin was used. The rule prefers the smallest departure from P_n that is not significantly worse. On every
   dataset tested it still picks 0.35 (see "Revision 2").

## Hyper-parameters (all fixed a priori, none dataset-specific)
| name | value | note |
|---|---|---|
| α grid | {0, 0.1, 0.2, 0.35} | as in the idea |
| CV | 2 folds, seed 0, every 5th tree | as in the idea |
| CV decision rule | one-SE rule (paired per-point SE) | revision 2; before: argmin |
| kernel | ±1 adjacent non-empty union-grid bin, training median, all j∈S equally weighted | as in the idea |
| LSMR atol=btol | 1e-10, conlim 1e12, maxiter max(2000, 20·ncol), retry ×5 | |
| preconditioner shift λ | 1e-11 · max diag(AᵀA) | |
| constraint measure | smoothed (mu~) | an "empirical" switch exists for ablation only |
| fallback | IQR-scaled bin-midpoint L2, ties: mass then index | |
| `NEW_PAIR_CELLS` | "drop" | revision 3; before: "fit". "tie" exists for ablation only (unstable, see Revision 3) |

## Self-test (subset harness, `subset_selftest.json`) vs baseline (`baseline/subset.json`)
Revision 3 (default `NEW_PAIR_CELLS = "drop"`) ran without errors (wall 20 s). The revision-2 configuration ("fit")
was also re-run through the official harness via `dev/method_variant.py` (`dev/subset_fit.json`). It matches the
pipeline's `subset_1.json` bit for bit, so the refactor did not change the "fit" path.

The cross-fit chose α = 0.35 on all subset datasets and on all three analytical reps (`dev/cvse.py`, one-SE rule).
CV scores (held-out normalised reconstruction error, α = 0 / 0.1 / 0.2 / 0.35; ± = paired SE of the difference to
α = 0.35):
* revision 3 ("drop"):
  * airfoil 0.0720±.0030 / 0.0570±.0008 / 0.0534±.0004 / 0.0501
  * concrete 0.1485±.0112 / 0.1022±.0024 / 0.0923±.0011 / 0.0853
  * abalone 0.1432±.0058 / 0.1156±.0014 / 0.1088±.0007 / 0.1030
  * analytical reps 0/1/2 at α = 0.35: 0.0412 / 0.0345 / 0.0321
* revision 2 ("fit"):
  * airfoil 0.0720±.0039 / 0.0448±.0009 / 0.0412±.0004 / 0.0381
  * concrete 0.1485±.0113 / 0.0974±.0023 / 0.0880±.0010 / 0.0811
  * abalone 0.1432±.0087 / 0.1093±.0014 / 0.1017±.0006 / 0.0955
  * nutrition 0.610 / 0.392 / 0.351 / 0.321 (revision 1, no SE)
  * analytical reps 0/1/2 at α = 0.35: 0.0404 / 0.0344 / 0.0308

The method is deterministic: outputs were bit-identical across runs on analytical, concrete, airfoil, abalone and nutrition
(`dev/checks.py`, revision 2; the "drop" rule adds no randomness).

Analytical (3 reps; lower is better; official harness):

| metric | baseline | S2 rev. 3 "drop" (mean ± std) | S2 rev. 2 "fit" |
|---|---|---|---|
| mse_eta1 | 0.02866 | 0.02805 ± 0.00095 | 0.02808 |
| mse_eta2 | 0.01977 | 0.01956 ± 0.0051 | 0.01959 |
| mse_eta3 | 0.01633 | 0.01641 ± 0.0036 | 0.01643 |
| mse_eta4 | 0.01660 | 0.01671 ± 0.0046 | 0.01675 |
| mse_eta5 | 0.00050 | 0.00048 ± 0.00010 | 0.00048 |
| mse_eta6 | 0.00037 | 0.00038 ± 0.00030 | 0.00039 |
| mse_eta12 | 0.03513 | **0.03265** ± 0.0019 (−7.1%) | 0.03273 |
| mse_eta34 | 0.03203 | **0.03014** ± 0.0015 (−5.9%) | 0.03035 |
| mse_others | 0.00218 | **0.00199** ± 0.0002 (−8.7%) | 0.00201 |
| resid_out | 0.01032 | **0.00874** ± 0.00029 (−15.3%) | 0.00870 |
| fit_s | 11.2 | 3.0 | 3.0 |

Real data (official harness; rev. 2 = "fit", rev. 3 = "drop", the default):

| dataset | resid_in | resid_out | ortho_in | ortho_out | locvar_in | fit_s |
|---|---|---|---|---|---|---|
| airfoil baseline | 0.00985 | 0.02639 | 0.0097 | 0.0987 | 5.20e-5 | 1.8 |
| airfoil rev. 2 | 0.01112 | 0.01930 | 0.0273 | 0.1364 | 2.70e-5 | 1.0 |
| airfoil **rev. 3** | 0.01035 (+5%) | **0.02041** (−23%) | 0.0162 (+68%) | 0.1296 (+31%) | **3.32e-5** (−36%) | 0.9 |
| concrete baseline | 0.00153 | 0.02500 | 0.0064 | 0.0822 | 0.0125 | 3.3 |
| concrete rev. 2 | 0.00207 | 0.02076 | 0.0113 | 0.0783 | 0.0225 | 2.0 |
| concrete **rev. 3** | 0.00194 (+26%) | **0.02136** (−15%) | **0.0057** (−11%) | **0.0605** (−26%) | 0.0271 (+117%) | 1.8 |
| abalone baseline | 0.00455 | 0.04062 | 0.0268 | 0.0151 | 0.00277 | 11.1 |
| abalone rev. 2 | 0.00541 | 0.01993 | 0.0293 | 0.0423 | 0.00259 | 3.3 |
| abalone **rev. 3** | 0.00532 (+17%) | **0.02201** (−46%) | 0.0288 (+7%) | 0.0425 (+181%) | **0.00255** (−8%) | 3.1 |

## Additional dev evaluations (in-process replica of the harness metrics; `dev/evalreal.py`)
Same split, model and metric code as the harness (the metric functions are imported from `bench/harness.py`). These
are **not** official harness runs. Baseline numbers are from `baseline/full.json`. Some fit times were measured while
another dev job was running, so they are upper bounds. α = 0.35 was chosen on every dataset.

| dataset | resid_in base → S2 | resid_out base → S2 | ortho_in base → S2 | ortho_out base → S2 | locvar base → S2 | fit_s base → S2 |
|---|---|---|---|---|---|---|
| nutrition | 0.01786 → 0.02159 | 0.09573 → **0.07571** | 0.0162 → 0.0176 | 0.1079 → **0.1032** | 0.00287 → 0.00286 | 4.8 → 2.1 |
| powerplant | 0.00063 → 0.00066 | 0.00110 → **0.00106** | None | None | 0.000537 → 0.000529 | 7.7 → 2.5 |
| parkinson | 0.00721 → 0.00750 | 0.00954 → **0.00943** | 0.0820 → 0.0874 | 0.1176 → 0.1237 | 0.00469 → 0.00447 | 33.1 → 13.6 |
| bike | 0.02134 → 0.02168 | 0.02464 → 0.02461 | 0.0134 → 0.0134 | 0.0212 → **0.0206** | 0.000344 → 0.000342 | 31.0 → 18.1 |
| housing | 0.01002 → 0.01026 | 0.01541 → **0.01439** | 0.0280 → 0.0279 | 0.0488 → **0.0475** | 0.00165 → **0.00129** | 74.4 → 15.4 |
| superconduct | 0.01009 → **0.00966** | 0.00910 → **0.00864** | None | None | 0.000663 → 0.000649 | 701 → 160 |

## Ablations (dev, in-process)
* **α = 0 (solver + fallback + float32 fixes only):**
  * airfoil: resid_out 0.02615, ortho_out 0.1063.
  * concrete: resid_out 0.02437.
  * abalone: resid_out **0.0742**, ortho_in 0.0401, ortho_out **0.0972**, i.e. *worse* than the baseline.
  * Cause (`dev/solver_effect.py`): the baseline's `lsqr` (atol=btol=1e-6, iter_lim=2·ncol) hits its iteration
    limit (istop=7) on 39/100 abalone trees. That early stopping acts as implicit ridge-type shrinkage. Replacing only the
    solver with the default `lsqr` in my α=0 pipeline gives back resid_out 0.0412 and ortho_out 0.0174, so the
    fallback change is not the cause.
  * At α=0.35 the two solvers agree: exact resid_out 0.01993 / ortho_out 0.0423 vs lsqr-default 0.01989 / 0.0415.
    Smoothing makes the problem well-posed. Relative to the *exactly solved* TreeHFD problem, smoothing improves
    every metric on abalone.
* **Constraint rows under P_n instead of mu~** (residual rows still under mu~): essentially identical results.
  * abalone ortho_out 0.0423 vs 0.0423; airfoil 0.1350 vs 0.1364; concrete ortho_in 0.0065 vs 0.0113.
  * So the constraint measure is *not* the cause of the ortho degradation. Default kept as specified (smoothed).
* **Larger α (grid extended to 0.5/0.7/0.85; NOT adopted):**
  * CV would then pick α=0.85. resid_out improves slightly (concrete 0.0204, abalone 0.0182, airfoil 0.0182).
  * But resid_in and ortho degrade sharply (concrete ortho_out 0.129, ortho_in 0.040). The capped grid from the idea is kept.
* **Where held-out points land** (`dev/cellkinds.py`, share of (point, tree, pair) lookups on test points):
  * Cells with empirical mass: 99.8–99.9% for both α=0 and α=0.35.
  * Fallback lookups were rare to begin with (0.11–0.22%). Smoothing turns most of them into fitted virtual cells
    (airfoil 0.22% → 0%, abalone 0.11% → 0.05%, concrete 0.22% → 0.08%).
  * Hence the resid_out gain comes mainly from **re-anchoring the coefficients of thinly populated empirical cells**
    through the virtual full cells, not from the unseen-cell fallback.
* **Solver timings** (`dev/solvers.py`, 10 concrete trees at α=0.2):
  * plain LSMR: 4.2 s, 62k iterations.
  * column-scaled LSMR: 1.3 s (but a different null-space solution).
  * Cholesky-preconditioned LSMR: 0.07 s, 20 iterations, same objective and min-norm solution.
  * superconduct: preconditioned 0.7 s/tree, of which forming AᵀA is ~0.25–1.3 s for up to 5.9M nnz;
    plain LSMR would take ~37 s/tree.

## Official full-mode harness (revision 3, `full_selftest.json`, vs `baseline/full.json`)
The revision-3 run (default "drop") finished with no errors in 275 s wall time and overwrote the revision-2
`full_selftest.json`. The revision-2 ("fit") numbers are kept in brackets from the earlier run.
Each cell is baseline → rev. 3 [rev. 2].

| dataset | resid_in | resid_out | ortho_in | ortho_out | locvar_in | fit_s |
|---|---|---|---|---|---|---|
| abalone | .00455 → .00533 [.00541] | .0413 → **.0220** [.0199] | .0268 → .0288 [.0293] | .0149 → .0425 [.0423] | .00277 → **.00256** [.00259] | 11.1 → 3.2 |
| airfoil | .00985 → .01035 [.01112] | .0269 → **.0204** [.0193] | .0097 → .0162 [.0273] | .0983 → .1296 [.1364] | 5.2e-5 → **3.3e-5** [2.7e-5] | 1.8 → 0.9 |
| bike | .02134 → .02163 [.02168] | .02464 → .02459 [.02461] | .01337 → .01346 [.01335] | .0212 → .0230 [.0206] | 3.44e-4 → 3.47e-4 [3.42e-4] | 31.0 → 18.0 |
| housing | .01002 → .01025 [.01026] | .0154 → **.0145** [.0144] | .02797 → .02792 [.02794] | .0488 → **.0479** [.0475] | .00165 → **.00128** [.00129] | 74.4 → 14.9 |
| concrete | .00153 → .00194 [.00207] | .0250 → **.0214** [.0208] | .0064 → **.0057** [.0113] | .0817 → **.0605** [.0783] | .0125 → .0271 [.0225] | 3.2 → 1.8 |
| nutrition | .0179 → .0213 [.0216] | .0957 → **.0760** [.0757] | .0162 → .0173 [.0176] | .1079 → **.1018** [.1032] | .00287 → .00296 [.00286] | 4.8 → 2.1 |
| parkinson | .00721 → .00743 [.00750] | .00954 → .00948 [.00943] | .0820 → **.0797** [.0874] | .1176 → **.1155** [.1237] | .00469 → **.00450** [.00447] | 33.1 → 12.9 |
| powerplant | .00063 → .00066 [.00066] | .00110 → .00109 [.00106] | None | None | 5.37e-4 → 5.29e-4 [5.29e-4] | 7.7 → 2.4 |
| superconduct | .01009 → **.00965** [.00966] | .00911 → **.00861** [.00864] | None | None | 6.63e-4 → 6.49e-4 [6.49e-4] | 701 → 159 |

Changes from rev. 2 to rev. 3 on the six non-subset datasets:
* parkinson's ortho regression turns into a small gain: ortho_in 0.0874 → 0.0797, ortho_out 0.1237 → 0.1155, both
  below the baseline.
* nutrition's ortho improves a little.
* **bike ortho_out gets worse**: 0.0206 → 0.0230, above the baseline's 0.0212 (+8.6%). ortho_in is flat.
* Everything else moves by ≤ 1% relative.

Analytical, 10 reps (baseline → rev. 3 [rev. 2]):
* Main effects: mse_eta1 .02886 → .02869 [.02871], eta2 .01840 → .01836 [.01838], eta3 .01585 → .01589 [.01591],
  eta4 .01743 → .01743 [.01745], eta5 .000458 → .000439, eta6 .000343 → .000336 (all within noise).
* Interactions: mse_eta12 .03289 → **.03094** [.03103] (−5.9%), mse_eta34 .03169 → **.02982** [.02974] (−5.9%),
  mse_others .00205 → **.00190** [.00191] (−7.5%).
* resid_out .01004 → **.00879** [.00870] (−12.4%). fit_s 11.4 → 3.2.

## Revision 2: response to the critic (ortho trade-off, α saturation)
The critic asked for a multi-objective, orthogonality-aware α criterion, or for down-weighting virtual atoms in the
orthogonality/mean rows. I first isolated where the orthogonality loss comes from. All numbers below are from the
in-process replica (`dev/`), except the self-test.

**1. The ortho violation is cross-tree, not per-tree constraint slack** (`dev/orthodiag.py`).
For the interaction/main pair that attains the max, cov(Σ_t η_jk^t, Σ_t η_m^t) was split into a t = t' part and a
t ≠ t' part. Both are reported as correlation contributions.

| dataset, α, split | max pair | \|corr\| | within-tree | cross-tree |
|---|---|---|---|---|
| airfoil 0 in | (0,1)\|1 | 0.0097 | −0.0000 | −0.0097 |
| airfoil 0.35 in | (0,1)\|1 | 0.0273 | +0.0000 | +0.0273 |
| airfoil 0.35 out | (3,4)\|4 | 0.1364 | −0.0037 | −0.1326 |
| concrete 0 in | (1,7)\|1 | 0.0064 | −0.0000 | −0.0064 |
| concrete 0.35 in | (1,7)\|1 | 0.0113 | +0.0010 | +0.0103 |
| concrete 0.35 out | (0,4)\|0 | 0.0783 | −0.0047 | −0.0735 |
| abalone 0 in | (3,4)\|3 | 0.0401 | −0.0000 | +0.0401 |
| abalone 0.35 in | (3,4)\|3 | 0.0293 | +0.0000 | +0.0293 |
| abalone 0.35 out | (4,6)\|4 | 0.0423 | +0.0019 | +0.0404 |

* On training points the within-tree part is ≤ 0.001 in every case. At α = 0.35 it is ≤ 0.005 on test points.
  (At α = 0 on abalone it reaches 0.03 on test points, from the exact but under-determined solve.)
* The per-tree orthogonality rows only make η_jk^t orthogonal to functions of *tree t's own* bins of x_j. Main effects
  of the other trees are constant on other, finer bins. Their covariance with η_jk^t is not in any per-tree system,
  for the baseline and for S2 alike.
* This is why moving the constraint rows between P_n and μ̃ changed nothing (revision-1 ablation). Re-weighting virtual
  atoms in those rows, the critic's second suggestion, acts on the within-tree part only, and that part is already ≈ 0.
  So I did not pursue it.

**2. The ortho change is a step at α = 0⁺, then flat. A multi-objective α cannot trade it off**
(`dev/evalreal.py`, fixed α; each cell is resid_in / resid_out / ortho_in / ortho_out):

| α | airfoil | concrete | abalone |
|---|---|---|---|
| baseline (official) | .00985 / .02639 / .0097 / .0987 | .00153 / .02500 / .0064 / .0822 | .00455 / .04062 / .0268 / .0151 |
| 0 | .00985 / .02615 / .0097 / .1063 | .00153 / .02437 / .0064 / .0757 | .00455 / .07423 / .0401 / .0972 |
| 0.05 | .01005 / .02081 / .0241 / .1330 | .00165 / .02226 / .0044 / .1521 | .00475 / .02521 / .0279 / .0407 |
| 0.1 | .01024 / .02046 / .0248 / .1337 | .00173 / .02167 / .0041 / .1505 | .00489 / .02302 / .0284 / .0411 |
| 0.2 | .01060 / .01991 / .0259 / .1349 | .00188 / .02111 / .0035 / .0665 | .00512 / .02119 / .0289 / .0417 |
| 0.35 | .01112 / .01930 / .0273 / .1364 | .00207 / .02076 / .0113 / .0783 | .00541 / .01993 / .0293 / .0423 |

* Any α > 0 pins the null directions of the rank-deficient per-tree systems to the tree's values on virtual cells,
  instead of the minimum-norm choice. That discrete change moves the cross-tree covariance. The size of α then hardly
  matters for ortho: airfoil ortho_in is 0.024–0.027 for every α in [0.05, 0.35]. Concrete's max jumps between
  interactions near the 1% inclusion threshold.
* resid_in grows roughly linearly with α. resid_out falls monotonically.
* An ortho penalty in the CV score can therefore only switch between α = 0 and α ≈ 0.35. Which one wins depends on an
  arbitrary λ. On abalone α = 0 is worse than both the baseline and α = 0.35 on every metric. I did not implement it.
* Instead the decision rule became the one-SE rule (§7). It is still a fidelity criterion, but it only moves away
  from P_n when the held-out gain is significant. It picks 0.35 everywhere because the gain from 0.2 to 0.35 is
  ≥ 6 paired SEs on every dataset (see the self-test section). **The saturation is a real, significant fidelity gain,
  not an artefact of an under-powered criterion.** Values above 0.35 (revision 1: CV would pick 0.85) degrade ortho,
  so the cap from the idea is kept.

**3. The ortho_out differences are within test-sample noise; the resid_out gains are not** (`dev/orthonoise.py`).
Paired bootstrap of the test set: B = 500, seed 0, X_test used only to evaluate.

| dataset (n_test) | sd of ortho_out (base / S2) | S2 − base ortho_out, 95% CI | S2/base − 1 resid_out, 95% CI |
|---|---|---|---|
| airfoil (301) | 0.048 / 0.056 | [−0.067, +0.156] | −28% [−39%, −17%] |
| concrete (206) | 0.062 / 0.057 | [−0.086, +0.117] | −17% [−29%, −2.5%] |
| abalone (836) | 0.041 / 0.044 | [−0.090, +0.080] | −51% [−59%, −42%] |

* The +38% / +180% ortho_out changes in the results table are within one bootstrap SD. The baseline itself moves by
  about ±0.001 between runs because of its random fallback (0.0975 / 0.0985 on airfoil in two runs here).
* ortho_in is computed on the training sample and is deterministic. Its increases are real but small in absolute
  size: airfoil +0.018, concrete +0.005, abalone +0.0025.

**4. Two ortho fixes were tried and rejected (not in method.py):**
* *Ensemble-level orthogonality rows.* This was a second pass that adds to each tree's system the rows
  E_μ̃[η_jk^t · g_m(X_m)] = 0, with g_m the standardised ensemble main effect from pass 1. It **blew up**: resid_out
  was 2.7e6 on airfoil at α = 0, and 16.5 on concrete and 8.9e3 on abalone at α = 0.35. resid_in was unchanged.
  The per-tree systems have exact null directions (redistribution among interactions). A soft row that touches them
  is satisfied with arbitrarily large coefficients, which cancel on training cells but not elsewhere. Making the rows
  safe would need a projection onto each tree's row space. Given the next result, that was not worth the cost.
* *Post-hoc ensemble orthogonalisation.* Each ensemble interaction was regressed on its two ensemble main effects
  (P_n on X_train), and the fitted part moved into the main effects. The reconstruction is exactly unchanged. This
  sets ortho_in to 0 **by construction**, but ortho_out did **not** improve:
  * S2: airfoil 0.136 → 0.156, concrete 0.078 → 0.101, abalone 0.042 → 0.041.
  * Baseline: airfoil 0.099 → 0.092, concrete 0.082 → 0.075, abalone 0.016 → 0.024.
  So removing the in-sample cross-tree correlation does not carry over to test points. Shipping it would only make
  ortho_in uninformative, which would be close to gaming the metric. Rejected.

**5. The concrete locvar_in increase comes from near-zero components** (`dev/locvardiag.py`, harness definition per
variable).

| variable | x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7 |
|---|---|---|---|---|---|---|---|---|
| share of Var T (base) | 0.227 | 0.044 | 0.001 | 0.083 | 0.026 | 0.003 | 0.019 | 0.355 |
| locvar base | .0021 | .0044 | .0173 | .0027 | .0019 | .0676 | .0037 | .0002 |
| locvar S2 (0.35) | .0019 | .0037 | .0340 | .0029 | .0024 | .1268 | .0077 | .0001 |

* The harness averages the per-variable ratios without weights, so x2 and x5 dominate the +80%. Their main effects
  carry 0.1% and 0.2–0.3% of Var T.
* The large components (x0, x1, x7) get smoother. A variance-share-weighted average (my hand computation from the
  rounded values above, not a harness metric) is about 0.0017 for both methods.

## Revision 3: virtual atoms only refine empirically populated pair cells (critic's suggestion)
**Hypothesis (critic).** Part of the ortho regression comes from the new pair-cell *columns* that virtual atoms
create. (Earlier revisions only changed constraint rows, or applied post-hoc ensemble fixes.) `dev/cellkinds.py`
had already shown that the resid_out gain comes from re-anchoring thinly populated *empirical* cells. So the
suggestion was: do not create a column for a pair cell with zero empirical mass.

**Implementation** (`NEW_PAIR_CELLS` in `lib/smoothed_hfd.py`; α = 0 is unchanged in every mode):
* `"tie"`: all virtual atoms are kept, but a pair cell with zero empirical mass has no column of its own. In the
  residual, zero-mean and orthogonality rows its term uses the column of the cell that the prediction-time fallback
  maps it to (the shared `_nearest_cell`). Fit and prediction therefore use the same parametrisation.
* `"drop"`: a copy is rejected for tree t if its shift would put any of tree t's interactions into a pair cell
  without empirical mass (in tree t's partition). Its mass stays on the original point. Pair columns are then exactly
  the empirical pair cells, as at α = 0, and unseen cells go through the deterministic fallback as before.

**1. "tie" is numerically unusable** (`dev/evalreal.py`, α = 0.35; `dev/tiediag.py`).
* resid_out explodes out of sample while resid_in stays normal: concrete 4.2e8, abalone 3.0e9 (resid_in
  0.0022 / 0.0055). Airfoil did not blow up: resid_out 0.0200, ortho_in 0.0168, ortho_out 0.1177.
* LSMR converges (istop = 2), so this is not a solver failure. Tying one column into zero-mean and ortho rows of
  several bins turns the per-tree system's *exact* null directions (neutralised by the min-norm solve) into
  *near*-null ones. On concrete the smallest relative singular value falls from ~1e-4 ("fit"/"drop") to 1e-7–1e-9,
  and only 2–20 exact null dimensions remain (vs 50–160).
* The small, mutually inconsistent virtual-row targets excite those directions: coefficients reach 9e7 on tree 15.
  They cancel on training cells but not on test points.
* Fixing this would need a truncation/ridge parameter, i.e. a new tuning knob. I did not pursue it.

**2. "drop" works and is the new default.** Official subset harness (tables above) and `dev/` diagnostics:
* It keeps 81% / 75% / 92% of the virtual atoms per tree on airfoil / concrete / abalone (4014 vs 4955, 1311 vs 1757,
  5798 vs 6335). All 100 solves per dataset converge with istop = 2 in ≤ 3 iterations.
* **The resid_out gain mostly survives.** Share of the rev.-2 gain over the baseline that is kept: airfoil 84%,
  concrete 86%, abalone 90%, analytical 98%. It is still −23% / −15% / −46% / −15% vs the baseline.
* **The ortho regression shrinks on airfoil and concrete**:
  * airfoil ortho_in 0.0273 → 0.0162 (baseline 0.0097); ortho_out 0.1364 → 0.1296 (baseline 0.0987).
  * concrete ortho_in 0.0113 → **0.0057** and ortho_out 0.0783 → **0.0605**, both now better than the baseline
    (0.0064 / 0.0822).
  * abalone does not move: 0.0293 → 0.0288 (in) and 0.0423 → 0.0425 (out). Baseline: 0.0268 / 0.0151.
* resid_in improves slightly vs rev. 2 on all three datasets (fewer atoms off the data), but stays above the baseline.
* Analytical metrics are equal or slightly better than rev. 2 (mse_eta12 −7.1%, mse_eta34 −5.9%, mse_others −8.7% vs
  the baseline). The only exception is resid_out, 0.00874 vs 0.00870.
* Within- vs cross-tree split of the max ortho pair (`dev/orthodiag.py`, α = 0.35, "drop"; compare with the
  revision-2 table):

  | dataset, split | max pair | \|corr\| | within-tree | cross-tree | rev. 2 cross-tree |
  |---|---|---|---|---|---|
  | airfoil in | (2,4)\|2 | 0.0162 | −0.0001 | +0.0164 | +0.0273 |
  | airfoil out | (3,4)\|4 | 0.1296 | −0.0036 | −0.1260 | −0.1326 |
  | concrete in | (1,7)\|1 | 0.0057 | +0.0004 | +0.0053 | +0.0103 |
  | concrete out | (1,7)\|1 | 0.0605 | −0.0021 | −0.0584 | −0.0735 (max was (0,4)\|0) |
  | abalone in | (3,4)\|3 | 0.0288 | −0.0000 | +0.0288 | +0.0293 |
  | abalone out | (4,6)\|4 | 0.0425 | +0.0019 | +0.0406 | +0.0404 |

* Test-point lookups (`dev/cellkinds.py`): fallback shares are 0.22% / 0.22% / 0.11%, the α = 0 levels. No
  virtual-only cells remain.
* α sweep ("drop", `dev/evalreal.py`; each cell is resid_in / resid_out / ortho_in / ortho_out). The ortho values
  are again nearly flat in α. The CV rule still picks 0.35 everywhere (`dev/cvse.py`, self-test section).

  | α | airfoil | concrete | abalone |
  |---|---|---|---|
  | 0.1 | .01000 / .02194 / .0175 / .1277 | .00169 / .02250 / .0045 / .0713 | .00486 / .02633 / .0282 / .0427 |
  | 0.2 | .01014 / .02125 / .0170 / .1287 | .00180 / .02184 / .0041 / .0666 | .00506 / .02384 / .0286 / .0425 |
  | 0.35 | .01035 / .02041 / .0162 / .1296 | .00194 / .02136 / .0057 / .0605 | .00532 / .02201 / .0288 / .0425 |

* locvar_in on concrete worsens further, 0.0225 → 0.0271 (baseline 0.0125). `dev/locvardiag.py` at α = 0.35: again
  x2 (0.0173 → 0.0647) and x5 (0.0676 → 0.1321), main effects with 0.1% / 0.2% of Var T. The large components move
  by ≤ 0.0004: x0 / x1 / x3 / x7 = .0023 / .0040 / .0031 / .0002 (rev. 2: .0019 / .0037 / .0029 / .0001; baseline
  .0021 / .0044 / .0027 / .0002).

**3. Reading.**
* The critic's hypothesis is **partly confirmed**. On airfoil and concrete, about 40–50% of the in-sample cross-tree
  coupling came from the new pair-cell columns. Removing them fixes concrete's ortho completely and helps airfoil,
  at the cost of 2–16% of the resid_out gain.
* On abalone the ortho gap to the baseline has a different source: the exact min-norm per-tree solve versus the
  baseline's early-stopped LSQR (revision-1 ablation: at α = 0 the exact solve gives ortho_out 0.097). Smoothing
  already reduces this. The remainder is cross-tree covariance, which no per-tree system (baseline or S2) controls.
  I treat it, and the ortho_out gap on airfoil, as a **structural limit** of the idea. I do not treat them as something
  still to tune.

**4. Departures from the idea as written (stated explicitly).**
* The smoothed measure is no longer identical for all trees. Each tree rejects the copies that would create empty
  pair cells in its own partition, so μ̃_t moves a tree-specific part of α's mass back to the originating points.
* The idea's claim that "pair cells adjacent to the data get fitted coefficients instead of a fallback" is given up.
  Unseen cells use the deterministic fallback again (0.1–0.2% of test lookups).
* The core is kept: a kernel-smoothed measure with exact tree targets on virtual cells, label-free CV of α, the
  solver and the deterministic fallback.
* The default was switched after looking at the subset datasets, which are also the reported ones. It is a single
  binary structural switch that the critic proposed a priori. It was not tuned per dataset. The official full-mode
  run (section above) checks it on the six other datasets: parkinson and nutrition improve, bike's ortho_out gets
  worse (0.0206 → 0.0230), and the rest are flat.

## Findings / honest caveats (revision 3 default, official harness)
* **Clear gains:**
  * resid_out drops on every dataset: −24% airfoil, −15% concrete, −47% abalone, −21% nutrition, −6% housing,
    −5% superconduct; about equal or slightly better on bike, powerplant and parkinson.
  * All analytical metrics improve slightly or stay within noise (10 reps): mse_eta12 −6%, mse_eta34 −6%,
    mse_others −7.5%, resid_out −12%.
  * Fit is 1.7–5x **faster** than the baseline everywhere, including CV (superconduct 159 s vs 701 s).
    The 5x budget is not an issue.
  * ortho is now better than the baseline on concrete (in −11%, out −26%) and parkinson (in −3%, out −2%). ortho_out
    is also better on nutrition and housing.
* **Trade-offs (not hidden):**
  * resid_in rises on every dataset except superconduct (e.g. concrete 0.0015 → 0.0019, nutrition 0.0179 → 0.0213,
    abalone 0.0046 → 0.0053). This is expected: the fit no longer interpolates single-point cells.
  * ortho remains worse on airfoil (in 0.0097 → 0.0162, out 0.098 → 0.130) and abalone (in 0.0268 → 0.0288,
    out 0.015 → 0.042), and slightly worse on nutrition (in 0.0162 → 0.0173) and bike (in +0.6%, out
    0.0212 → 0.0230). The idea's prediction that
    ortho_out would improve therefore **holds on 4 of 7 datasets that report it, not in general**.
    * Revision 3 removed the part of the regression caused by new pair-cell columns. That part was about half of the
      in-sample cross-tree coupling on airfoil and concrete, and parkinson's whole regression.
    * The rest is cross-tree covariance, which per-tree systems (baseline or S2) do not control. On abalone it comes
      mainly from solving the per-tree systems exactly instead of the baseline's early-stopped LSQR. I consider it a
      structural limit of the idea (Revision 3, §3).
    * On airfoil the out-of-sample maximum is interaction (3,4), with a variance share of 1.0% (threshold artefact risk).
    * Revision 2's bootstrap (`dev/orthonoise.py`, a dev script, not part of the harness) put the subset ortho_out
      differences within test-sample noise. It is not relied on for the conclusions here.
  * locvar_in is worse on concrete (0.0125 → 0.0271) and slightly worse on bike and nutrition (≤ 3%); better
    elsewhere. The concrete increase comes from main effects with ≤ 0.3% of Var T (Revision 2 §5, Revision 3).
  * "drop" gives back 2–16% of revision 2's resid_out gain in exchange for the ortho improvements above.
* The smoothed measure is a **deliberate departure from the empirical HFD target** (risk (a) of the idea). The CV
  criterion only scores fidelity, not orthogonality. With the one-SE rule it still picks the largest grid value (0.35)
  everywhere, because the fidelity gain is highly significant.
* The "solver convergence" fix alone (α=0) is **not** an improvement. On abalone the exactly-converged min-norm
  TreeHFD solution generalises worse than the baseline's early-stopped LSQR. The fix only pays off together with smoothing.
* Evaluating the fixed model on shifted training inputs (virtual cells) is a deterministic function of X_train
  (a kernel-smoothed measure). No labels, test inputs, refitting or re-seeding are involved.
* The official full-mode harness was run in revisions 2 and 3 (table above). It agrees with the in-process replica.
  In revision 3 the subset harness was also run on the revision-2 configuration (`dev/subset_fit.json`), so the
  "fit" vs "drop" comparison uses official harness numbers for both.

## Files
* `method.py`: benchmark entry point. `lib/smoothed_hfd.py`: implementation. `lib/treehfd_mod/`: untouched package copy (only path helpers used).
* `dev/`: development scripts. They rebuild the harness's (model, X_train, X_eval) exactly as `bench/harness.py` does,
  which needs y to *train XGBoost*. The method itself never sees y. `compare.py`, `evalreal.py`, `solvers.py`,
  `solver_effect.py`, `cellkinds.py`, `checks.py` and `sizes.py` produced the numbers quoted here.
  Revision 2 added:
  * `orthodiag.py`: within- vs cross-tree split.
  * `orthonoise.py`: bootstrap CIs, plus the rejected post-hoc transfer, which lives only in that script.
  * `locvardiag.py`: per-variable locvar.
  * `cvse.py`: CV scores, paired SEs and the one-SE choice.
  * `compare_full.py`: harness JSON vs baseline JSON.
  * `evalreal.py` now accepts `NAME=value` overrides of module constants.
  The rejected coupling-row prototype was removed from `lib/`.
  Revision 3 added:
  * `tiediag.py`: per-tree conditioning / coefficient size for "fit" / "tie" / "drop".
  * `method_variant.py`: `method.py` with `NEW_PAIR_CELLS = "fit"`, so the revision-2 configuration can be run
    through the official harness.
  * `subset_fit.json`: its official subset result, identical to `subset_1.json`.
  * `diag_fit_subset.jsonl`: per-fit diagnostics of that run.
