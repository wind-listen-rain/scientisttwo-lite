# S1 — Ensemble-level re-orthogonalization of TreeHFD interactions (sum-preserving, significance-shrunk)

## This revision in one paragraph (response to the critic)
The critic flagged three regressions: abalone ortho_out +53%, mse_eta5/6 +16-19%, and airfoil locvar +4%. It also said
step 1 contributed nothing. Diagnosis:
- abalone's ortho_out change and the eta5/6 rescale both come from step 2 moving in-sample leaks that are not
  distinguishable from row-sampling noise.
- The row cross-fit of the previous version could not see this, because the held-out rows were part of TreeHFD's own fit.
- airfoil's locvar regression came from step 1.

Changes:
1. **Empirical-Bayes shrinkage of step 2.** Each pair's transfer is multiplied by `w = max(0, 1 - q/W)`, where W is the
   pair's robust Wald statistic.
2. **Step 1 is off by default.** It stays available as an ablation.
3. **Deterministic tie-break.** The tie-break of TreeHFD's `predict_partition` is seeded, so harness runs can be paired
   against a seeded baseline and every difference can be attributed.

Paired self-test against the seeded baseline:
- resid is identical everywhere.
- mse_eta5 is +1.5% and mse_eta6 +2.4% (was +15.7% and +19.1%).
- abalone ortho_out is +0.0011 (was +0.0082). airfoil locvar is +0.0% (was +3.9%).
- mse_eta1..4 are -1.9% to -9.2%, mse_eta12 -2.8% and mse_eta34 -7.0%.
- **Price:** ortho_in is no longer exactly 0. It is at most the baseline's everywhere and -64% on abalone.
- **Also:** the airfoil and concrete ortho_out "wins" of the previous version are gone. They were within eval-sample
  noise; details below.

## What is implemented (default)
`method.py` = baseline TreeHFD (package copy in `lib/treehfd_mod/`, algorithm unchanged) + a post-hoc, pointwise
sum-preserving transfer (`lib/s1_projection.py`), fitted on X_train only (no labels, no RNG).

- **TreeHFD fit.** Exactly as in the baseline.
  - The package copy has only two changes. (a) Bookkeeping: `hfd.train_components` is accumulated during the fit; it is
    identical to `predict(X_train)`.
  - (b) `TIE_SEED = 0` in `cartesian_partition.py`. Where the baseline breaks ties between equally close training cells
    with an unseeded `default_rng()`, a fresh `default_rng(0)` is used per call. This makes predictions deterministic
    and independent of call order. The seed was set before any run and is not tuned.
- **Step 2, the leak transfer.** For each interaction (j, k) with nonzero variance:
  1. Least squares of the centred `eta_jk` on the centred `(eta_j, eta_k)` on the X_train rows gives the in-sample leak
     `beta = (a, b)`.
  2. Its HC0 sandwich covariance is `V = sum_i infl_i infl_i^T`, with `infl_i = (Z^T Z)^-1 z_i e_i` the influence of
     row i. The Wald statistic is `W = beta^T V^-1 beta`.
  3. **Shrinkage.** `beta <- max(0, 1 - q/W) * beta`, where q = 2, or 1 if a parent's main effect is identically 0.
     This is the empirical-Bayes posterior mean of beta under a prior `N(0, tau^2 V)`, with tau^2 estimated by moments
     from E[W] = q (1 + tau^2). There is no tuned constant.
  4. **Transfer.** `eta_jk -= a eta_j + b eta_k` and `eta_j *= s_j`, with `s_j = 1 + sum of the shrunk coefficients
     on eta_j`.
  - The aggregated eta_j is piecewise constant on the union of the ensemble's split thresholds on x_j. So this is the
    projection of the interaction onto the direction that the final union-grid main effect spans.
  - It is a rescale, so the normalised locvar of every main effect is unchanged (1.000x on every dataset).
  - Guard, unchanged: a variable whose factor would leave [0.5, 2] is excluded and its pairs are refitted. The guard
    never triggered on the dev datasets.
- **Step 1** (union-bin shape transfer, previous revisions) runs only with `proj_kw={"step1": True}`; see the ablations.

The transfer is pointwise sum-preserving. In the paired harness runs, resid_in and resid_out equal the seeded
baseline's to all printed digits.

## Diagnosis (critic items 1 and 2)
Dev scripts, all label-free: `dev/diag_step2.py`, `diag_se.py`, `diag_oof.py`, `diag_pop.py`, `cache_folds.py`,
`cache_pop.py`.

### abalone ortho_out (+53% in the previous version)
- **Not step 1, and not lambda or bin counts.** Step-2-only gives the identical 0.0242.
- **Which pair drives it.** The max is set by pair (3,4) (variance 1.6% of Var T). Its eval |corr| with eta_3 went from
  0.0055 to 0.0242.
- **The evidence on X_train says the leak is real:**
  - In-sample coefficient +0.055, HC0 se 0.019 (z ≈ 3).
  - Refitting TreeHFD on each half of X_train and scoring the other half gives held-out coefficients +0.006 and +0.143.
- **The eval set disagrees within its noise.** The eval-row coefficient is -0.015 with bootstrap se 0.052, which is
  1.3 combined se away from the in-sample value.
- **The coefficients are tail-driven.** The 1% most influential rows carry 17-27% of the total |influence| on these
  coefficients on abalone (in-sample and eval), and 9-19% in the analytical case.
- **Paired bootstrap of the eval rows.** For the ortho_out difference: -0.0019 ± 0.0126, P(S1 better) = 0.57. So the
  +53% was eval-sample noise around a real in-sample leak, not a systematic generalisation failure. No X_train-only rule
  can predict this eval realisation.
- **What EB shrinkage does here.** This pair has W = 8.5, so it moves w = 0.77 of the leak, and leaves the
  insignificant small ones. abalone
  ortho_out is now 0.0162 vs 0.0151 (paired; bootstrap diff +0.0002 ± 0.0071).

### mse_eta5 / mse_eta6 (+16% / +19% in the previous version)
- **Cause.** Step 2 rescaled the noise-variable main effects by s = 1.05-1.10. Each factor is the sum of about 5 leak
  coefficients from immaterial pairs (variance 1e-4 of Var T), each with se 0.02-0.07. Each coefficient is
  individually insignificant.
- **Population check.** I ran the analytical cached fits on 100k fresh *inputs* from the benchmark's input distribution
  (`dev/cache_pop.py`; no labels, no ground-truth components). The population leak factors of x5/x6 are
  0.99/0.94, 1.13/1.01 and 1.07/1.02 (reps 0-2). The in-sample factors exceed them in 5 of 6 cases (mean +0.06), so the
  leak into noise variables is partly an in-sample artefact.
- **The large pairs behave well.** The in-sample HC0 se are calibrated against the population: over the 12 coefficients
  of the true pairs, RMS((in - pop)/se) = 0.96, with mean +0.28 se, i.e. no strong systematic optimism. That is what justifies shrinking by these
  se.

### Why not a "cross-fitted factor"
- **The old row cross-fit is not honest.** It kept TreeHFD's components fixed, so the held-out rows had been used by
  TreeHFD's own fit.
- **The honest version was tried and rejected.** I refitted TreeHFD on 2 folds of X_train (same model), scored out of
  fold, and used the out-of-fold coefficient. It costs 1.6-1.8x the baseline fit on top, and it predicted the population
  leak *worse* than the in-sample coefficient: mean population-leak error 3.1e-4 vs 2.3e-4 over the 3 reps
  (`dev/diag_oof.py`, `dev/diag_pop.py`).

### Shrinkage variants compared (label-free; `dev/eval_step2.py`, before any harness run of them)
Analytical population leak `sum cov(eta_jk, eta_j)^2 / var(eta_j) / Var T` after the transfer. Baseline:
1.65e-3 / 1.30e-3 / 3.00e-4.

| step 2 variant | rep 0 | rep 1 | rep 2 | x5,x6 factors (reps 0/1/2) | abalone ortho_in / out |
|---|---|---|---|---|---|
| unshrunk (previous) | 1.06e-4 | 6.1e-5 | **5.48e-4 (worse than baseline)** | 1.05,1.09 / 1.10,1.08 / 1.09,1.10 | 0 / .0242 |
| per-variable EB on s_j | 2.38e-4 | 3.01e-4 | 2.86e-4 | 1.00,1.00 / 1.03,1.05 / 1.03,1.07 | .0520 / .0418 |
| per-coefficient EB | 1.97e-4 | 3.13e-4 | 2.81e-4 | 1.04,1.02 / 1.02,1.04 / 1.02,1.06 | - |
| Wald gate z=2 (older ablation) | 9.8e-5 | 5.31e-4 | 5.46e-4 | 1.00,1.00 all reps | - |
| **per-pair EB (chosen)** | 1.75e-4 | 2.54e-4 | 2.83e-4 | 1.01,1.00 / 0.98,1.01 / 1.02,1.04 | .0097 / .0173 |

- **Why per-pair EB.** It is the only variant that lowers the population leak in all three reps. Its mean leak equals
  the unshrunk one (2.4e-4), and it keeps the noise-variable factors near 1.
- **Order of decisions.** I chose it from these label-free numbers, then ran the harness.
- **Disclosure.** The population inputs are simulated from the analytical case's known input distribution. They are
  used only for this dev diagnosis, never by the method.

## Paired harness comparison (critic items 3 and 4), subset mode, this machine
- **How the pairing works.** `dev/harness/baseline_seeded.py` is the baseline method run on the seeded package copy. Its
  components before S1's transfer are identical to S1's, so every difference below is caused by the transfer alone.
- **Ablations.** They are wrappers `dev/harness/v_*.py` around `method.py`, run through the unmodified harness. The
  outputs are `dev/harness/*.json|log`, and the official run is `subset_selftest.json`.
- **Sequencing.** All six runs were sequential, with nothing else running.

| metric | seeded baseline | **S1 final** (step 2, EB) | step 2 unshrunk only | step 1 only | step 1 + step 2 EB | previous S1 (step 1 + unshrunk) |
|---|---|---|---|---|---|---|
| mse_eta1 | 0.02705 | 0.02652 (-1.9%) | 0.02628 (-2.8%) | 0.02705 (0%) | 0.02652 (-1.9%) | 0.02628 (-2.8%) |
| mse_eta2 | 0.01818 | 0.01763 (-3.0%) | 0.01754 (-3.5%) | 0.01818 (0%) | 0.01763 (-3.0%) | 0.01754 (-3.5%) |
| mse_eta3 | 0.01737 | 0.01577 (-9.2%) | 0.01551 (-10.7%) | 0.01737 (0%) | 0.01577 (-9.2%) | 0.01551 (-10.7%) |
| mse_eta4 | 0.01769 | 0.01645 (-7.1%) | 0.01632 (-7.7%) | 0.01769 (0%) | 0.01645 (-7.1%) | 0.01632 (-7.7%) |
| mse_eta5 | 0.0003277 | 0.0003328 (**+1.5%**) | 0.0003793 (+15.7%) | 0.0003277 (0%) | 0.0003328 (+1.5%) | 0.0003793 (+15.7%) |
| mse_eta6 | 0.0005445 | 0.0005578 (**+2.4%**) | 0.0006485 (+19.1%) | 0.0005445 (0%) | 0.0005578 (+2.4%) | 0.0006485 (+19.1%) |
| mse_eta12 | 0.03382 | 0.03288 (-2.8%) | 0.03277 (-3.1%) | 0.03382 (0%) | 0.03288 (-2.8%) | 0.03277 (-3.1%) |
| mse_eta34 | 0.03162 | 0.02940 (-7.0%) | 0.02930 (-7.3%) | 0.03162 (0%) | 0.02940 (-7.0%) | 0.02930 (-7.3%) |
| mse_others | 0.002148 | 0.002148 (0%) | 0.002149 (0%) | 0.002148 (0%) | 0.002148 (0%) | 0.002149 (0%) |
| resid_out (analytical) | 0.01026 | 0.01026 (0%) | 0.01026 (0%) | 0.01026 (0%) | 0.01026 (0%) | 0.01026 (0%) |
| fit_s (analytical) | 19.4 | 19.7 (+1.5%) | 19.5 | 19.3 | - | 19.8 |
| airfoil ortho_in | 0.00967 | 0.00967 (0%) | 5e-17 | 0.01191 (+23%) | 0.01191 | 3e-17 |
| airfoil ortho_out | 0.09708 | 0.09708 (0%) | 0.0909 (-6.4%) | 0.09922 (+2.2%) | 0.0992 | 0.09086 (-6.4%) |
| airfoil locvar_in | 5.199e-5 | 5.199e-5 (0%) | 0% | +3.9% | +3.9% | +3.9% |
| concrete ortho_in | 0.00636 | 0.00636 (0%) | 2e-16 | 0.00636 | 0.00636 | 2e-16 |
| concrete ortho_out | 0.08182 | 0.08182 (0%) | 0.07521 (-8.1%) | 0.08182 | 0.08182 | 0.07521 (-8.1%) |
| concrete locvar_in | 0.01249 | 0.01249 (0%) | 0% | -3.4% | -3.4% | -3.4% |
| abalone ortho_in | 0.0268 | 0.00965 (-64%) | 7e-16 | 0.02687 | 0.00969 | 1e-16 |
| abalone ortho_out | 0.01506 | 0.01620 (+7.6%) | 0.0232 (+54%) | 0.01518 (+0.8%) | 0.0162 | 0.02322 (+54%) |
| abalone locvar_in | 0.00256 | 0.00256 (0%) | 0% | -0.1% | -0.1% | -0.1% |
| resid_in / resid_out (3 datasets) | - | identical | identical | identical | identical | identical |
| fit_s airfoil / concrete / abalone | 3.1 / 5.5 / 17.0 | 3.2 / 5.4 / 17.4 | 3.1 / 5.3 / 17.2 | 3.1 / 5.5 / 17.4 | 3.2 / 5.6 / 17.9 | 3.2 / 5.6 / 17.5 |

Per rep (harness log, seeded baseline -> S1 final):
- mse_eta12: 0.0342 -> 0.0327, 0.0337 -> 0.0323, 0.0336 -> 0.0337. That is -4.4%, -4.2% and +0.3%; the previous
  version had +1.5% in rep 3.
- mse_eta1: 0.0281 -> 0.0274, 0.0270 -> 0.0262, 0.0261 -> 0.0260.

**Attribution (critic item 3).**
- **resid.** With ties fixed, every variant's resid_in and resid_out equal the baseline's. The previous abalone
  resid_out +1.5% and airfoil resid_out -1.1% were tie-break jitter. The unseeded official baseline itself differs from
  the seeded one by -0.4% to +1.8% in resid_out and by up to 5% in ortho_out.
- **locvar.** Every locvar change is caused by step 1. Step 2 is a rescale and leaves locvar exactly unchanged.

**Ablation verdict (critic item 4).**
- Step 1 alone does nothing on the analytical case (0 of 15 pairs transferred), makes airfoil worse (ortho_in +23%,
  ortho_out +2%, locvar +3.9%) and makes concrete's locvar 3.4% better.
- On top of EB step 2 it changes nothing measurable except those locvar moves.
- On the full-mode dev data, step 1 alone raises parkinson's ortho_in and ortho_out (.082 -> .103, .118 -> .137).
- **Step 1 is therefore dropped from the default.** The code remains for ablations.

## Official self-test vs the official (unseeded) baseline `baseline/env-22b414443084/subset.json`
`subset_selftest.json` / `.log` (final code, 134 s wall, no errors). These numbers include the baseline's own
tie-break jitter; see the paired table for attribution.

| metric | baseline | S1 | change |
|---|---|---|---|
| mse_eta1 / eta2 / eta3 / eta4 | .02705 / .01818 / .01737 / .01769 | .02652 / .01763 / .01577 / .01645 | -1.9% / -3.0% / -9.2% / -7.1% |
| mse_eta5 / eta6 | .0003277 / .0005445 | .0003328 / .0005578 | +1.5% / +2.4% (sd over reps: ±36% / ±31%) |
| mse_eta12 / eta34 | .03387 / .03168 | .03288 / .02940 | -2.9% / -7.2% |
| mse_others / resid_out | .002148 / .01022 | .002148 / .01026 | 0% / +0.3% (tie jitter) |
| fit_s analytical | 19.1 | 19.7 | +3% |
| airfoil ortho_in / ortho_out / locvar / resid_out | .00967 / .0989 / 5.20e-5 / .02695 | .00967 / .0971 / 5.20e-5 / .02648 | 0 / -1.8% / 0 / -1.7% (jitter) |
| concrete ortho_in / ortho_out / locvar / resid_out | .00636 / .0822 / .01249 / .02512 | .00636 / .0818 / .01249 / .02506 | 0 / -0.4% / 0 / -0.3% (jitter) |
| abalone ortho_in / ortho_out / locvar / resid_out | .0268 / .01586 / .00256 / .04034 | .00965 / .01620 / .00256 / .04048 | -64% / +2.2% / 0 / +0.4% (jitter) |
| fit_s airfoil / concrete / abalone | 3.3 / 6.1 / 22.0 | 3.2 / 5.4 / 17.4 | within machine noise |

## Dev results on the full-mode datasets (cached harness models; label-free; paired; `dev/ablation.txt`)
| dataset | ortho_in base -> S1 | ortho_out base -> S1 | paired bootstrap d(ortho_out), P(S1 better) | locvar | step-2 factors |
|---|---|---|---|---|---|
| nutrition | .0162 -> .0162 | .1101 -> .1101 | 0 (nothing significant) | 1.000x | [1.000, 1.000] |
| parkinson | .0820 -> .0140 | .1176 -> .0937 | -.020 ± .039, 0.67 | 1.000x | [0.883, 1.169] |
| housing | .0280 -> .0104 | .0488 -> .0488 | -.002 ± .005, 0.55 | 1.000x | [0.999, 1.041] |
| bike | .0134 -> .0134 | .0212 -> .0212 | -.002 ± .007, 0.56 | 1.000x | [0.996, 1.038] |
| powerplant | null | null | - | 1.000x | [0.999, 1.005] |
| analytical reps 0/1/2 (5000 eval pts) | .071/.060/.068 -> .013/.017/.016 | .080/.109/.063 -> .032/.062/.045 | P = 0.93 / 1.00 / 0.58 | 1.000x | [0.98, 1.06] |

- **superconduct.** Not re-run; its cache is gone. It has no interaction at or above 1% of Var T, so ortho is null. locvar
  is exactly 1.000x, because step 2 is a rescale. On a synthetic problem of superconduct's size (n = 17010, 2927 pairs),
  step 2 takes 4.8 s, about 1% of its 421 s TreeHFD fit. Dropping step 1 also removes the previous 35 s.

## Hyper-parameters
| name | value | role |
|---|---|---|
| `SHRINK` | "pair_eb" | step 2 shrinkage: "none" (previous revision), "var_eb", "coef_eb", "pair_gate" (ablations) |
| `STEP1` | False | union-bin shape transfer (previous revisions' step 1) |
| `SCALE_BOUNDS` | (0.5, 2.0) | guard on the step-2 rescale factors (never triggered on the dev datasets) |
| `TIE_SEED` (package copy) | 0 | deterministic tie-break in `predict_partition` |
| `CLOSE_Z` | 2.0 | only for `shrink="pair_gate"` (ablation) |
| step 1 only: `M_EXP`, `M_FLOOR`, `N_FOLDS`, `TAU_GRID`, `SE_RULE`, `DIRECTIONS`, `RIDGE` | unchanged | see the previous revision's notes in the git history of this file |

The per-pair EB factor has no free constant: q is the number of coefficients. Nothing was tuned on ground-truth mse
values. The harness mse numbers above were produced after the configuration was fixed.

## Honest assessment
- **Regressions the critic listed:**
  - mse_eta5/6: +1.5% / +2.4%, down from +16% / +19%. Not zero. The per-rep factors are x5 1.01 / 0.98 / 1.02 and
    x6 1.00 / 1.01 / 1.04, so rep 2's x6 still moves by 4%. Both changes are far inside the across-rep sd (±31-36%).
  - abalone ortho_out: +0.0011 paired (+7.6% vs the seeded baseline, +2.2% vs the official one), down from +0.0082.
    The paired bootstrap of the eval rows says it is noise (+0.0002 ± 0.0071).
  - airfoil locvar: 0%.
  - abalone resid_out: shown to be tie jitter.
- **What was given up:**
  - ortho_in is no longer 0. Pairs whose in-sample leak is statistically indistinguishable from noise keep it:
    airfoil, concrete, nutrition and bike are unchanged from the baseline, and the other datasets fall 60-85%.
  - The airfoil and concrete ortho_out "gains" (-7% / -8%) disappear. Their paired bootstrap had shown them to be noise
    (P(better) 0.54 / 0.64; differences 0.7 and 0.35 bootstrap sd), because those datasets have no significant leak to
    move.
  - On the subset's three real datasets S1 now changes only abalone. On the full-mode dev data it helps parkinson
    (ortho_out -20%) and does nothing on housing (the unshrunk version's -21% on housing, P = 0.84, is gone).
- **Analytical case: the gains are kept but slightly smaller than unshrunk.** mse_eta1..4 are -2% to -9%, mse_eta12
  -2.8% (2 of 3 reps better, 1 neutral) and mse_eta34 -7.0%. ortho_out on the 5000-point eval sets improves in all three
  reps, and the population leak decreases in all three reps (the unshrunk version worsened rep 2).
- **mse_others is unchanged.** The spurious pairs lose essentially no variance; they are not purified. That part of the
  idea's expected effect did not materialise.
- **Novelty.** The default method is now a significance-shrunk, restricted purification (Lengerich et al., 2020) along
  the estimated main-effect directions. The union-of-splits shape transfer (the idea's original step) is an ablation
  that showed no benefit. The novelty concern stands.
- **Fit time.** About +1-3% on the analytical case, and within machine noise on the real datasets.
- **Not run by me:** full mode. The full-mode numbers above are label-free dev numbers on cached harness models.

## Files
- `method.py` — interface (`fit`, `predict`); `fit(..., proj_kw=...)` is used only by the ablation wrappers.
- `lib/s1_projection.py` — step 2 (`_close`, shrinkage options), step 1 (union bins, penalised additive projection,
  cross-fit), `apply_projection`.
- `lib/treehfd_mod/` — copy of the baseline package: bookkeeping (`train_bins`, `train_components`) and `TIE_SEED`.
- `dev/` — development only, not used by the method:
  - `harness/` — seeded baseline and ablation wrappers, `run_all.sh`, their harness results, `table.py`.
  - `eval_step2.py` — paired label-free comparison of configurations; population metrics for the analytical case.
    Its output is `ablation.txt`.
  - Diagnosis scripts:
    - `diag_step2.py` — in-sample vs eval coefficients.
    - `diag_se.py` — HC0 and bootstrap se.
    - `diag_oof.py` — fold-refit estimates.
    - `diag_pop.py` — population leak.
  - Caching scripts:
    - `cache_baseline.py` — rebuilds the harness models and caches baseline fits. Labels are used only to train XGBoost
      exactly as the harness does.
    - `cache_folds.py` — TreeHFD refits on folds of X_train.
    - `cache_pop.py` — analytical components at 100k fresh inputs.
  - Earlier revisions: `dev_eval.py` and `dev_eval2.py` (metric helpers) and `ablation_prev_revision.txt` /
    `final_config_dev_prev_revision.txt`.
  - `cache/` — cached fits; large, not for version control.
- `subset_selftest.json` / `.log` — official self-test of this revision.
