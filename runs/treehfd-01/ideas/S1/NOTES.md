# S1 — Ensemble-level re-orthogonalization on the union-of-splits partition

## This revision in one paragraph (response to the critic)
The previous version projected each interaction onto penalised additive functions of *merged* union bins (30 rows)
and chose the penalty per pair by a plain cross-fitted argmin. That made concrete's ortho_in worse, because the merged-bin
space does not contain the final main effects. It also roughened main effects (airfoil +23%, concrete +12%, superconduct
2.5x) and did nothing in the analytical case. Three changes:
1. **Step 2, a closing projection on the final main effects.** Each interaction's component along its parents' final
   main effects is moved into them. This is a rescaling of eta_j, so it is sum-preserving and does not change the
   roughness (the shape) of eta_j. ortho_in is now exactly 0 on every dataset, and it also produces all of the analytical
   gains.
2. **Coarser bins and a stricter selection rule for step 1.** Bins now hold n^(3/4) rows (about n^(1/4) bins) instead
   of 30 rows, and a paired two-standard-error rule picks the smoothest penalty, including "no transfer", whose held-out
   error is not significantly worse than the best one. locvar_in is now 0.97x-1.05x the baseline on all 12 dev
   datasets (it was up to 2.5x).
3. **Uncertainty for ortho_out** via a paired bootstrap of the eval rows (dev only).

Self-test against the baseline:
- Analytical: mse_eta1..4 are 3-11% lower, mse_eta12 is 2.9% lower and mse_eta34 is 7.4% lower.
- **Trade-off:** mse_eta5 is 16% higher and mse_eta6 19% higher (see Honest assessment). mse_others is unchanged.
- Real data: ortho_in is 0 on all three datasets. ortho_out is -7% on airfoil, -9% on concrete and +44% on abalone. All
  three changes are within the bootstrap noise. locvar_in is between -3% and +4%. resid is unchanged by construction.

## What is implemented
`method.py` = baseline TreeHFD (copy in `lib/treehfd_mod/`, algorithm unchanged) + a post-hoc, pointwise
sum-preserving re-orthogonalization (`lib/s1_projection.py`), fitted on X_train only (no labels, no RNG).

0. **TreeHFD fit** exactly as in the baseline (`XGBTreeHFD.fit`). The only change to the package copy is bookkeeping:
   the training-point components are accumulated during the fit (`hfd.train_components`). This is bit-identical to
   `hfd.predict(X_train)` and saves a second pass over the ensemble.
1. **Step 1 — union-bin transfer (the idea's projection).**
   - For each variable j, all split thresholds of the ensemble on x_j are collected. They are merged greedily
     left-to-right until each bin holds `>= max(30, n^(3/4))` X_train rows.
   - For each interaction (j,k), solve `min_g sum_i (eta_jk - g_j(b_j) - g_k(b_k))^2 + tau (|D g_j|^2 + |D g_k|^2)`,
     where D is the first difference along the ordered bins. The normal equations are assembled with `np.bincount`, and
     the gauge is fixed exactly by `g_k[0] = 0`.
   - tau comes from the grid {0, 1, 3, 10, 30, 100, 300, 1e3, 1e4, inf} rows, where `inf` means no transfer. It is chosen
     per pair by 3-fold cross-fitting (`row % 3`) of the held-out squared error, computed row by row.
   - **Paired two-standard-error rule:** keep the largest tau whose summed held-out error exceeds the minimum by at most
     2 x sqrt(n) x sd(per-row difference). A transfer therefore needs a significant held-out gain, and among
     statistically equivalent fits the smoothest one wins.
   - The fitted g is moved: `eta_jk -= g_j + g_k`, `eta_j += g_j - c_j`, `eta_k += g_k - c_k`,
     `eta0 += c_j + c_k`.
2. **Step 2 — closing projection on the final main effects (new).**
   - After step 1, fit `eta_jk ~ a_c eta_j + b_c eta_k` for each pair c = (j,k) by least squares on centred columns
     under the empirical measure. Then set `eta_jk -= a_c eta_j + b_c eta_k` and `eta_j *= s_j`, with
     `s_j = 1 + sum_c a_c` over the pairs containing j.
   - Every transfer into eta_j is along eta_j itself. So after the move, each interaction is exactly uncorrelated
     in-sample with the *final* eta_j and eta_k, for all pairs at once.
   - The locvar of eta_j (normalised by its own variance) is unchanged by a rescaling.
   - Guard: a variable whose factor s_j would fall outside [0.5, 2] is excluded and the pairs are refitted without it.
     Across all 12 dev datasets the factors stay within [0.85, 1.37], so the guard never triggered.
   - At predict time, the stored coefficients are applied to the step-1 main effects at the new points.

Both steps are pointwise sum-preserving. On every dev dataset, the normalised resid_in / resid_out change by
<= 2e-9 (floating point).

## Hyper-parameters
| name | value | role |
|---|---|---|
| `M_MIN` / `M_EXP` / `M_FLOOR` | None / 0.75 / 30 | rows per merged bin = max(30, ceil(n^0.75)): 154 (concrete) ... 1490 (superconduct) |
| `N_FOLDS` | 3 | cross-fitting folds (`row % 3`) |
| `TAU_GRID` | 0, 1, 3, 10, 30, 100, 300, 1e3, 1e4, inf | first-difference penalty in rows; inf = no transfer |
| `SE_RULE` | 2.0 | step 1: paired z-standard-error rule (0 = plain argmin) |
| `CLOSE` | True | step 2 on/off |
| `CLOSE_Z` | 0.0 | step 2 significance gate (robust Wald test at |N(0,1)| > z); 0 = move every pair (ablation below) |
| `SCALE_BOUNDS` | (0.5, 2.0) | guard on the step-2 rescale factors |
| `DIRECTIONS` | False | ablation: parents' main effects as extra unpenalised columns in step 1 |
| `RIDGE` | 1e-8 | relative numerical ridge on the normal equations |

`fit_projection(...)` accepts all of these for ablations. `criterion="leak"`, the idea's held-out |corr| criterion
(squared-covariance form), is still available. The previous options `select="global"` and `scale="mean"` were removed;
their results are in the git history of this file.

## Changes in this revision and why

### 1. concrete ortho_in +117% — root cause and fix
- **Same measure.** The harness measures ortho_in as the Pearson correlation on the X_train rows between each
  interaction (variance >= 1% of Var T) and its parents' *final* main effects. That is the same empirical measure the
  projection uses.
- **Different function space.** The final eta_j is piecewise constant on the full, unmerged union grid. The merged
  bins do not span it, and the difference penalty leaves a residual. With 30-row bins on n = 824, the step-1 transfer
  created a new correlation with the final eta_j.
- **Step 1 alone can increase the leak.** The ablation table shows "step 1 only" with the new coarse bins raising
  parkinson's ortho_in from 0.082 to 0.103.
- **Why step 2, not a per-pair fallback.** A per-pair fallback to the baseline (the critic's suggestion) keeps the
  baseline's leak. Step 2 removes the leak exactly. It is a special case of the idea's projection: eta_j is a function
  on the union grid, which is exactly the space the idea wanted the interaction to be orthogonal to. The idea claimed
  the projection space "contains the final eta_j"; step 2 makes that claim true.
- **Result.** ortho_in is below 2e-16 on every dataset, so ortho_in is no longer an informative metric for this method.

### 2. locvar_in regression
- **Diagnosis (superconduct, with n^(2/3) bins and the 2-SE rule).** Main effects there hold 1e-5 to 4e-3 of Var T
  each, and every variable sits in about 72 pairs. The step-1 transfers into x12 have a bin-to-bin roughness index
  (`sum (dg)^2 / (var(g) * n_bins)`) of about 0.4-1.5, which is close to white noise. The cross-fit still judged most of
  them significant (n = 17010).
- **Interpretation.** The additive leak TreeHFD leaves in its aggregated interactions is itself step-shaped: it comes
  from the mismatch between each tree's grid and the union grid. Moving it faithfully roughens small main effects, and
  locvar normalises by each component's own variance. The trade-off between exact HFD orthogonality and smooth main
  effects is inherent to the idea.
- **Fix.** Step 1 now makes only coarse, reliably estimated shape corrections: about n^(1/4) bins, plus the 2-SE rule.
  The fine-grid part along the main effects is left to step 2, which adds no roughness.
- **Why n^(3/4) rows per bin.** This is deliberately coarser than the MSE-optimal n^(1/3) bins of a histogram
  regression (i.e. n^(2/3) rows). I chose it by comparing label-free locvar on the benchmark's own datasets, including
  the subset datasets; no separate development data exists.
  - superconduct: 661-row bins (n^(2/3)) gave 1.24x; 1500 rows gave 1.05x; 4000 rows gave 0.98x.
  - z = 3 or 4.5 in the SE rule changed superconduct only slightly (1.20x, 1.16x), so the bin size is the effective lever.
- **Tried and rejected:** adding the parents' main effects as unpenalised columns in step 1 (`directions=True`). The
  bin terms then capture only the rough residual structure beyond the main-effect direction, and locvar rose (concrete
  1.11x with the final bins; airfoil 2.49x and concrete 1.75x with 30-row bins).
- **Not tried:** a second-difference penalty. Its tau -> inf limit is a linear trend in bin index, not "no transfer",
  so "no transfer" would need a separate candidate. The coarse bins already brought locvar to about 1.0x.
- **The critic's "joint leak + roughness criterion"** is played by the SE rule: it takes the smoothest candidate that is
  not significantly worse.

### 3. Cross-fit selection
- Selection is still per pair, on the held-out squared error of the additive fit. That error is an unbiased estimate of
  the distance to the population additive projection (by Pythagoras).
- The idea's held-out |corr| criterion is biased toward "no transfer". The held-out rows were part of TreeHFD's own
  fit, so tau = inf is scored in-sample, while every finite tau is scored honestly. Correlations on n/3 rows also carry
  noise of about 1/sqrt(n/3).
- Both alternatives the critic asked about are in the ablation table: the leak criterion and the unpenalised/argmin
  choice (`se_rule=0`, which allows tau = 0).
- In the analytical case step 1 still transfers 0 of 15 pairs: no shape correction beyond the main-effect directions is
  estimable. The analytical leak lies along the main-effect directions: step 2 removes 0.2-0.7% of var(eta_12) and
  var(eta_34), with coefficients 0.02-0.07. That is the part that the previous version's cross-fit judged "not estimable"
  on ~120 bins, but which is well estimated with 2 parameters.

### 4. ortho_out with uncertainty
`dev/dev_eval2.py` runs a paired bootstrap of the eval rows (200 resamples). It reports mean ± sd of ortho_out for the
baseline and for S1, and the fraction of resamples where S1 is lower ("P(better)"). A bootstrapped max|corr| is biased
upward, so compare the paired columns, not the level.

## Dev results, final configuration (`dev/final_config_dev.txt`; cached harness models, label-free metrics)
| dataset | ortho_in | ortho_out | bootstrap ortho_out (base -> S1), P(better) | locvar_in | step-1 transfers | step-2 factors |
|---|---|---|---|---|---|---|
| airfoil | .0097 -> 0 | .0985 -> .0923 | .151±.040 -> .151±.043, 0.52 | 1.04x | 2/10 | [0.99, 1.01] |
| concrete | .0064 -> 0 | .0817 -> .0751 | .152±.064 -> .154±.067, 0.61 | 0.97x | 1/28 | [0.99, 1.23] |
| abalone | .0268 -> 0 | .0159 -> .0242 | .071±.039 -> .069±.037, 0.57 | 1.00x | 6/36 | [0.97, 1.07] |
| analytical rep 0 | .0712 -> 0 | .0797 -> .0450 | .085±.025 -> .051±.018, 0.84 | 1.00x | 0/15 | [1.02, 1.09] |
| analytical rep 1 | .0598 -> 0 | .1090 -> .0475 | .117±.020 -> .062±.016, 1.00 | 1.00x | 0/15 | [1.02, 1.10] |
| analytical rep 2 | .0680 -> 0 | .0634 -> .0553 | .068±.022 -> .072±.016, 0.41 | 1.00x | 0/15 | [1.02, 1.10] |
| nutrition | .0162 -> 0 | .1101 -> .1055 | .165±.055 -> .167±.056, 0.51 | 1.00x | 0/21 | [1.00, 1.02] |
| parkinson | .0820 -> 0 | .1176 -> .0981 | .128±.023 -> .109±.040, 0.62 | 0.98x | 71/157 | [0.89, 1.21] |
| housing | .0280 -> 0 | .0488 -> .0386 | .054±.018 -> .046±.016, 0.82 | 1.00x | 13/28 | [0.91, 1.06] |
| bike | .0134 -> 0 | .0212 -> .0212 | .043±.018 -> .039±.016, 0.69 | 1.00x | 4/28 | [1.00, 1.02] |
| powerplant | null | null | - | 1.00x | 0/6 | [1.00, 1.02] |
| superconduct | null | null | - | 1.05x (old: 2.5x) | 2829/2927 | [0.85, 1.37] |

Projection time: 0.02-0.8 s on the small datasets and 35 s on superconduct (about 8% of its 421 s TreeHFD fit).

## Ablations (`dev/ablation.txt`; cells = ortho_in / ortho_out / locvar ratio)
| config | airfoil | concrete | abalone | analytical0 | analytical1 | analytical2 | nutrition | parkinson | housing | bike |
|---|---|---|---|---|---|---|---|---|---|---|
| previous (m=30, argmin, no step 2) | .0098/.0985/1.23x | .0138/.0813/1.12x | .0099/.0220/1.02x | .0712/.0797/1.00x | .0598/.1090/0.98x | .0680/.0634/1.00x | .0162/.1101/1.00x | .0322/.0824/1.09x | .0023/.0413/1.00x | .0011/.0212/1.16x |
| step 1 only | .0119/.1000/1.04x | .0064/.0817/0.97x | .0269/.0160/1.00x | .0712/.0797/1.00x | .0598/.1090/1.00x | .0680/.0634/1.00x | .0162/.1101/1.00x | .1026/.1370/0.98x | .0108/.0493/1.00x | .0042/.0212/1.00x |
| step 2 only | 0/.0923/1.00x | 0/.0751/1.00x | 0/.0242/1.00x | 0/.0450/1.00x | 0/.0475/1.00x | 0/.0553/1.00x | 0/.1055/1.00x | 0/.0990/1.00x | 0/.0385/1.00x | 0/.0212/1.00x |
| **final** | 0/.0923/1.04x | 0/.0751/0.97x | 0/.0242/1.00x | 0/.0450/1.00x | 0/.0475/1.00x | 0/.0553/1.00x | 0/.1055/1.00x | 0/.0981/0.98x | 0/.0386/1.00x | 0/.0212/1.00x |
| step 2 Wald-gated (z=2) | .0119/.1000/1.04x | .0064/.0817/0.97x | .0097/.0242/1.00x | 0/.0450/1.00x | .0383/.0669/1.00x | 0/.0553/1.00x | .0162/.1101/1.00x | .0140/.0981/0.98x | .0108/.0493/1.00x | .0042/.0212/1.00x |
| step 1 + directions | 0/.0923/1.07x | 0/.0752/1.11x | 0/.0242/0.99x | 0/.0450/1.00x | 0/.0475/1.00x | 0/.0553/1.00x | 0/.1055/1.00x | 0/.0978/0.94x | 0/.0384/1.00x | 0/.0212/0.99x |
| step 1 argmin (no SE rule) | 0/.0923/1.07x | 0/.0751/0.99x | 0/.0234/1.03x | 0/.0450/1.02x | 0/.0475/0.97x | 0/.0553/0.99x | 0/.1055/1.00x | 0/.0992/0.89x | 0/.0399/1.01x | 0/.0212/1.00x |
| bins n^(2/3) | 0/.0968/1.06x | 0/.0751/0.96x | 0/.0242/1.01x | 0/.0450/1.00x | 0/.0475/1.00x | 0/.0553/1.00x | 0/.1055/1.00x | 0/.0983/0.91x | 0/.0389/1.00x | 0/.0212/1.00x |

The first row reproduces the previous official S1 numbers exactly, which confirms that the refactoring preserved
behaviour. `criterion="leak"` with the final bins gave airfoil/concrete/abalone 0/.0923/1.09x, 0/.0751/0.96x and
0/.0227/1.00x.

**Choice of the step-2 gate (`CLOSE_Z = 0`, ungated), made before running the harness.**
- **Case for gating.** Without a gate, step 2 also rescales the near-zero main effects of the noise variables x5 and x6
  in the analytical case by 1.05-1.10. That comes from many small leaks in spurious pairs. The factors were visible
  label-free in dev, and I expected mse_eta5/6 to rise.
- **Case against gating.** The robust Wald gate (z = 2) prevents that rescale, but it also leaves real, systematic leaks
  in place. Analytical rep 1's true pair (0,1) has coefficients of about 0.02/0.05 in all three reps and still fails the
  per-rep test. The gate also makes airfoil's ortho_in and ortho_out worse.
- **Decision.** Step 2 has at most 2 parameters per pair and no roughness cost. TreeHFD itself imposes its orthogonality
  constraints exactly under the empirical measure. So I kept step 2 ungated. The gate was not selected by looking at
  ground-truth mse values.

## Self-test (subset, this machine `22b414443084`)
`subset_selftest.json` / `.log` (final code, 130 s wall, no errors) vs `baseline/env-22b414443084/subset.json`:

| analytical (mean ± sd over 3 reps) | baseline | S1 | change |
|---|---|---|---|
| mse_eta1 | 0.02705 ± 0.00082 | 0.02628 ± 0.00068 | -2.8% |
| mse_eta2 | 0.01818 ± 0.0025 | 0.01754 ± 0.0013 | -3.5% |
| mse_eta3 | 0.01737 ± 0.0031 | 0.01551 ± 0.0033 | -10.7% |
| mse_eta4 | 0.01769 ± 0.0045 | 0.01632 ± 0.0047 | -7.7% |
| mse_eta5 | 0.0003277 ± 0.00012 | 0.0003793 ± 0.00014 | **+15.7%** |
| mse_eta6 | 0.0005445 ± 0.00017 | 0.0006485 ± 0.0002 | **+19.1%** |
| mse_eta12 | 0.03387 ± 0.00038 | 0.03288 ± 0.0010 | -2.9% |
| mse_eta34 | 0.03168 ± 0.0039 | 0.02934 ± 0.0024 | -7.4% |
| mse_others | 0.002148 | 0.002146 | -0.1% |
| resid_out | 0.01022 | 0.01026 | +0.3% |
| fit_s | 19.1 | 19.1 | -0.1% |

Per rep (harness log, baseline -> S1):
- mse_eta12: 0.0344 -> 0.0328, 0.0335 -> 0.0317, 0.0337 -> 0.0342. It improves in 2 of 3 reps; in rep 3 it is +1.5%.
- mse_eta1: 0.0281 -> 0.0272, 0.0270 -> 0.0256, 0.0261 -> 0.0261.

The baseline's eval-time tie-break jitter is about ±1% on these values (see Unchanged baseline behaviour).

| dataset | resid_in | resid_out | ortho_in | ortho_out | locvar_in | fit_s |
|---|---|---|---|---|---|---|
| airfoil | .009845 -> .009845 | .02695 -> .02649 | .00967 -> 3e-17 | .0989 -> .0918 (-7%) | 5.20e-5 -> 5.40e-5 (+3.9%) | 3.3 -> 3.0 |
| concrete | .001534 -> .001534 | .02512 -> .02514 | .00636 -> 2e-16 | .0822 -> .0751 (-9%) | .01249 -> .01206 (-3.4%) | 6.1 -> 5.2 |
| abalone | .004553 -> .004553 | .04034 -> .04106 | .0268 -> 1e-16 | .0159 -> .0228 (+44%) | .00256 -> .00256 (-0.1%) | 21.9 -> 17.4 |

## Honest assessment
- **Reconstruction is untouched.** resid_in is identical. resid_out differs only by the baseline's own eval-time
  tie-break jitter; the transfers change the normalised residuals by <= 2e-9.
- **ortho_in is 0 by construction**, so it no longer discriminates. The informative metric is ortho_out.
- **ortho_out is clearly better only where the eval set is large enough to tell:**
  - analytical reps 0 and 1: 0.080 -> 0.045 and 0.109 -> 0.048. Rep 2 is 0.063 -> 0.055, but the bootstrap is
    undecided (P = 0.41).
  - housing: 0.049 -> 0.039 (P = 0.82).
  - parkinson: 0.118 -> 0.098 (P = 0.62).
- **On the small datasets, ortho_out is inside the test-sample noise:** bootstrap sd 0.04-0.065, with P(better) of
  0.52-0.61. airfoil (-7%) and concrete (-9%) moved the right way. abalone's point estimate got worse
  (0.016 -> 0.023), but its bootstrap sd is 0.039 and P(better) = 0.57. I cannot claim a consistent ortho_out gain on
  small datasets, and nobody can measure one there.
- **Analytical mse:**
  - Gains: mse_eta1..4 (-3% to -11%), mse_eta12 (-2.9%) and mse_eta34 (-7.4%). mse_eta12 improves in only 2 of 3 reps.
  - Losses: mse_eta5 +16% and mse_eta6 +19% (+5e-5 and +1e-4 absolute). The cause is the step-2 rescale of the
    noise-only main effects, which is correct for the HFD of the *fitted model* T but moves away from the
    ground-truth 0.
  - mse_others is unchanged: the spurious interactions lose only about 0.1% of their variance.
- **locvar_in is 0.97x-1.05x everywhere** (superconduct 1.05x, previously 2.5x), with small increases on airfoil (+4%)
  and superconduct (+5%).
- **Where the gains come from.** Step 1, the union-bin part that is the idea's original mechanism, now contributes
  little that the label-free metrics can see. "Step 2 only" gives the same ortho numbers with locvar exactly 1.00x, and
  in the analytical case step 1 makes no transfer at all, so every analytical change comes from step 2. I kept step 1
  because it is the idea's core; it still changes main-effect shapes on parkinson, housing and superconduct. A reviewer
  may reasonably prefer the simpler "step 2 only" variant.
- **Novelty.** Step 2 is a restricted purification (Lengerich et al., 2020) along the estimated main-effect directions,
  so this revision strengthens, rather than weakens, the novelty concern already flagged in the idea.
- **Not run by me:** the full mode. The numbers for bike, housing, nutrition, parkinson, powerplant and superconduct are
  dev numbers from cached harness models.

## Unchanged baseline behaviour
For eval points that fall in unseen interaction cells, TreeHFD's `predict_partition` breaks ties with an unseeded RNG.
As a result, eval components, resid_out and the analytical mse values jitter slightly from run to run (about 1%), in
both the baseline and S1. The S1 transfers themselves are deterministic and exactly sum-preserving.

## Files
- `method.py` — interface (`fit`, `predict`).
- `lib/s1_projection.py` — union bins, penalised additive projection with cross-fitting and the SE rule (step 1),
  closing projection (step 2), transfer.
- `lib/treehfd_mod/` — copy of the baseline package; only bookkeeping added (`train_bins`, `train_components`).
- `dev/` — development scripts only, not used by the method:
  - `cache_baseline.py` rebuilds the harness models and caches baseline fits. Labels are used only to train the XGBoost
    model exactly as the harness does.
  - `dev_eval2.py` computes label-free metrics plus the paired bootstrap of ortho_out.
  - `ablation.py` produces the ablation table (`ablation.txt`).
  - `dev_eval.py` holds the older metric helpers.
  - `compare.py` compares result files.
  - `final_config_dev.txt` holds the dev numbers of the final configuration.
  - `cache/` holds the cached fits (large; not for version control).
- `subset_selftest.json` / `.log` — self-test of the final code.
