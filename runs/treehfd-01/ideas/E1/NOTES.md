# E1 — Anchored GT-LOCO TreeHFD (model-anchored Gaussian prior inside hard-orthogonal TreeHFD)

## Status (round 2)
**Implemented. The subset and full self-tests run without errors through the official harness on this machine
(`22b414443084`), and every full-mode fit time is now below 5× the baseline** (largest: powerplant 4.80×; bike
went from 7.2× to 4.07×). Round-1 notes, unchanged, are in `dev/NOTES_round1.md`.

What round 2 did, one item per point of the critic:
1. **Compute.** The exact leave-out speed-up from round 1 is now validated and in `lib/agtloco.py`.
   * Outputs are unchanged: per-tree relative differences are ≤ 3e-10, the per-tree argmins are identical, and the
     plain-argmin selection reproduces round 1 on all 32 diagnostic fits.
   * Full-mode fit_s is 1.12–4.80× baseline (round 1: 1.25–7.23×). No further fallback was needed.
2. **ρ saturation / resid_in.** I committed in advance to S4's one-standard-error rule towards in-sample fidelity
   (`SE_RULE = 1`, the conventional k; no other k was run), ran the 8-split ablation once, and report it as it came
   out.
   * **E1 is now at or below S4 on both fidelity metrics on all four diagnosed datasets** (geo-mean over 8 splits):
     resid_in 0.94–0.99× S4 and resid_out 0.86–0.93× S4. The round-1 argmin had resid_in 0.98–1.01× S4.
   * **The idea's target (halve S4's resid_in excess) is still not reached.** The rule moves resid_in by only
     −4…0% relative to round 1.
   * **The reason is measured, not guessed.** R prefers ρ = 1 over ρ = 0.25 by more than one paired SE. In 23 of
     32 fits, the best ρ = 0.25 candidate has a lower in-sample residual than the one-SE choice, so it would have been
     picked had it been tied. The saturation is therefore not a noisy corner that a tie-break can undo.
3. **Orthogonality regression.** Investigated with two diagnostics (fixed κ, and a common pair set).
   * **Concrete's large ortho regression is mostly a threshold artefact of the metric.**
     * The anchor raises concrete's total interaction variance by 21%, so more pairs cross the harness's 1% variance
       threshold (train: 10 vs 8 pair-splits; test: 13 vs 8).
     * Measured on the same pairs for both methods, E1 / S4 is 1.04 (ortho_in) and 1.05 (ortho_out) instead of
       1.49 / 1.42.
   * The remaining real cost (+4–7% on concrete, abalone and nutrition) sits at small κ. At a fixed common κ, the
     anchor raises ortho_in only when κ ≤ 0.1, and there only on concrete (1.44–1.53×).
   * **Not mitigated in the estimator.** The only lever inside the idea would be a selection that knowingly gives up
     more than one SE of estimated held-out risk (larger κ or smaller ρ). I did not adopt one from held-out
     numbers, so I **do not claim that E1 beats S4 on orthogonality**.

## Round 2 in detail

### 1. Compute: speed-up validated and included
* **Changes** (exact; the results do not depend on them):
  * block temporaries are cache-sized (`BLOCK_CHUNK` 4e6 → 3e5 elements);
  * leave-out blocks are batched by exact size up to 32 rows (`BLOCK_EXACT`; was 8, and padding to a multiple of 8
    had added 54% to Σb²);
  * size-1 blocks skip the gathers;
  * the needed virtual rows (`Hn`, `yn`) are sliced once per tree instead of once per ρ;
  * the merged-copy rows are grouped with a verified 64-bit hash (`_unique_rows`) instead of `np.unique(axis=0)`.
* **Output checks.**
  * `dev/check_speedup.py <round-1 code> <new code> bike 60 5` fits bike trees 60–64 with both codes. Worst relative
    differences: leave-out residuals 2.6e-10 (they are stored in float32), in-sample residuals 8.6e-11, risks
    5.7e-14, coefficient paths 2.7e-12. The per-tree argmins are identical, and the new code is 1.64× faster on these
    heavy trees.
  * On all 32 fits of the 8-split ablation, the plain argmin of the new code gives the same printed resid_in,
    resid_out, ortho_in, ortho_out and locvar_in as round 1's selection.
  * Check (7a) was re-run through the harness on this machine: `dev/method_rho0.py` (RHOS = (0,), SE_RULE = 0)
    equals `S4/subset_env-22b414443084.json` on every metric, including the analytical case
    (`dev/subset_rho0_env-22b414443084.json`, `dev/logs/subset_rho0_env.log`).
* **Where bike's time went** (`dev/time_parts.py bike`, 100 trees):
  * per-tree fits 351 s → 194 s;
  * anchored leave-out 263 s → 120 s;
  * virtual atoms 31.6 s → 20.7 s;
  * eigendecompositions unchanged (23–25 s).
  * Logs: `dev/logs/time_parts_bike.log` (round 1) and `time_parts_bike_cand1.log` (round 2).
* **Powerplant is bound by the eigendecompositions.**
  * `dev/prof_fit.py`: `eigh` takes 34 of 58 s. That is 4 per tree (ridge ρ = 0, lattice ρ ∈ {0, 0.25, 1}) at
    dimension ≈ 900, about 85 ms each.
  * I benchmarked alternatives at m = 900, single-threaded: `np.linalg.eigh` 94 ms, SciPy `evd` 89 ms, `evr`
    114 ms. None is a usable gain.
  * The block-K kernel already runs at ≈ 16 GFlop/s (`dev/bench_kernels.py`). CSR is the fastest incidence product
    (`dev/bench_spmm.py`).
  * Dropping the ridge prior's eigendecomposition would break the nesting of S4: ridge is chosen in 10 of the 32
    ρ = 0 (S4) fits. So I kept it.
* **Full-mode fit_s / baseline, official harness on an idle machine** (round 1 → round 2):

  | dataset | analytical | abalone | airfoil | bike | housing | concrete | nutrition | parkinson | powerplant | superconduct |
  |---|---|---|---|---|---|---|---|---|---|---|
  | round 1 | 3.47 | 1.90 | 2.44 | **7.23** | 2.77 | 3.15 | **5.20** | 3.00 | **5.14** | 1.25 |
  | round 2 | 3.37 | 1.70 | 2.10 | 4.07 | 2.64 | 2.92 | 4.35 | 2.54 | **4.80** | 1.12 |

  * Powerplant is below 5× with little margin; run-to-run variation on this hybrid-core CPU is about ±10%.
  * If a later run puts it above 5×, the justification is the fourth eigendecomposition per tree (the anchored
    ρ = 0.25 path). It is far from the 20× limit.

### 2. Selection: the one-SE rule (pre-committed, reported as it came out)
* **Rule.**
  * Among the ensemble candidates whose R is within one paired SE of the minimum, take the one with the smallest
    in-sample residual to T(X_train). This is label-free: it uses X_train and the model only.
  * This is S4's `SE_RULE`, which S4 tested and left off because in S4 it was a pure resid_in / resid_out trade.
  * In E1 the idea's main target is resid_in, so the rule is aligned with the idea. k = 1 was fixed before any run;
    no other k was run.
* **Ablation.** `dev/ablate_r2.sh` → `dev/logs/ablate_r2_<ds>.log`; summary `dev/summarize_r2.py` →
  `dev/logs/ablate_r2_summary.log`. S4's 8 diagnostic splits are used for diagnosis only; split 0 is the benchmark
  split. Geo-mean ratio to the baseline over the 8 splits (ortho: n splits where defined):

  | dataset | selection | resid_in | resid_out | ortho_in | ortho_out |
  |---|---|---|---|---|---|
  | airfoil | S4 (ρ = 0 mode, argmin) | 1.262 | 0.858 | 0.959 | 0.889 |
  | | E1 round 1 (argmin) | 1.232 | 0.733 | 0.777 | 1.003 |
  | | **E1 round 2 (one-SE)** | **1.185** | **0.739** | 0.777 | 1.047 |
  | | ρ = 0.25 mode, argmin (diagnostic) | 1.228 | 0.808 | 0.774 | 0.979 |
  | concrete | S4 | 1.548 | 0.833 | 0.938 (7) | 0.962 (7) |
  | | E1 round 1 | 1.559 | 0.751 | 1.500 (6) | 1.301 |
  | | **E1 round 2** | **1.533** | **0.757** | 1.393 (7) | 1.320 |
  | | ρ = 0.25 mode | 1.437 | 0.782 | 1.258 (7) | 1.286 |
  | abalone | S4 | 1.390 | 0.536 | 0.716 | 1.107 |
  | | E1 round 1 | 1.369 | 0.488 | 0.744 | 1.054 |
  | | **E1 round 2** | **1.368** | **0.489** | 0.744 | 1.054 |
  | | ρ = 0.25 mode | 1.193 | 0.536 | 0.771 | 1.086 |
  | nutrition | S4 | 1.456 | 0.811 | 1.010 | 1.001 |
  | | E1 round 1 | 1.445 | 0.741 | 1.053 | 1.047 |
  | | **E1 round 2** | **1.393** | **0.754** | 1.052 | 1.047 |
  | | ρ = 0.25 mode | 1.285 | 0.797 | 1.033 | 1.040 |

  * Correction: round 1's NOTES printed the concrete "selected" row as 1.556 / 0.750 / 1.348 / 1.302. Its own
    summary log says 1.559 / 0.751 / 1.500 / 1.301, which is the row above.
* **E1 round 2 / S4** (geo-mean of per-split ratios):

  | | resid_in | resid_out | ortho_in | ortho_out | locvar_in |
  |---|---|---|---|---|---|
  | airfoil | 0.939 | 0.862 | 0.810 | 1.177 | 0.658 |
  | concrete | 0.990 | 0.909 | 1.486 | 1.418 | 1.171 |
  | abalone | 0.984 | 0.912 | 1.039 | 0.952 | 1.112 |
  | nutrition | 0.957 | 0.930 | 1.042 | 1.047 | 1.053 |

  * One-SE / argmin within E1: resid_in 0.962 / 0.983 / 0.999 / 0.964, resid_out 1.009 / 1.009 / 1.000 / 1.017.
    Orthogonality is unchanged except airfoil ortho_out 1.044.
  * For comparison, the same rule inside S4: resid_in 0.84–0.99, resid_out 1.00–1.06.
* **Why the rule does so little in E1.**
  * The tie sets are small (3–24 candidates) and always inside the ρ = 1 / per-tree modes, so the rule only moves κ
    within ρ = 1.
  * In 23 of 32 fits, the best ρ = 0.25 candidate has a lower in-sample residual than the one-SE choice, yet it was
    not picked (abalone 8/8, nutrition 7/8, concrete 5/8, airfoil 3/8). Hence it lies outside the tie set: R's
    preference for ρ = 1 over ρ = 0.25 exceeds one paired SE.
  * Held-out resid_out agrees with R: ρ = 1 is better in 31 of 32 fits (round 1).
  * The ρ = 0.25 mode would give the resid_in the idea hoped for, but only by giving up estimated held-out risk
    beyond its noise level. Choosing it, or dropping ρ = 1 from the pre-registered grid, would be a choice made from
    these held-out numbers, so I did not do it. It is reported for the next round.

### 3. Orthogonality: where the regression comes from
* **Fixed κ** (`dev_e1.py ... ortho`). A common κ is used in every tree, the selected variant is kept, and only ρ
  changes. Ratio ρ = 1 / ρ = 0 at the same κ, geo-mean over 8 splits:

  | κ | airfoil resid_in / resid_out / ortho_in | concrete | abalone | nutrition |
  |---|---|---|---|---|
  | 0.01 | 1.12 / 0.81 / 0.75 | 1.43 / 0.83 / **1.53** | 1.26 / 0.78 / 0.87 | 1.30 / 0.81 / 1.04 |
  | 0.1 | 1.06 / 0.85 / 0.84 | 1.31 / 0.88 / **1.44** | 1.13 / 0.88 / 0.94 | 1.16 / 0.88 / 1.04 |
  | 1 | 0.93 / 0.86 / 1.11 | 1.08 / 0.93 / 1.11 | 0.98 / 0.93 / 0.91 | 1.01 / 0.93 / 1.02 |
  | 10 | 0.82 / 0.79 / 1.04 | 0.97 / 0.95 / n/a | 0.93 / 0.91 / n/a | 0.94 / 0.93 / n/a |

  * At κ ≥ 1 the anchor does what the idea intended. It is a better shrinkage centre than zero: it lowers resid_in
    and resid_out together, with little ortho effect.
  * At small κ it acts as extra data competing with the real rows. That raises resid_in and, on concrete,
    ortho_in.
  * R drives κ to the floor at ρ = 1, so the selected configuration sits in the costly regime.
* **Common pair set** (`dev/diag_ortho_common.py`, summary `dev/logs/diag_ortho_common_summary.log`). The harness's
  ortho is a max over the interactions whose variance is ≥ 1% of Var[T], each method with its own set. The table
  also recomputes it over the same pairs for both methods (pairs ≥ 1% in either). E1 / S4, geo-mean over 8 splits:

  | | harness ortho_in | same pairs | harness ortho_out | same pairs | pair-splits ≥ 1% S4 → E1 (train / test) | interaction variance E1 / S4 |
  |---|---|---|---|---|---|---|
  | airfoil | 0.810 | 0.810 | 1.177 | 1.072 | 24 → 24 / 24 → 27 | 1.02 |
  | concrete | **1.486** | **1.043** | **1.418** | **1.046** | 8 → 10 / 8 → 13 | 1.21 |
  | abalone | 1.039 | 1.039 | 0.952 | 0.952 | 19 → 19 / 19 → 21 | 1.08 |
  | nutrition | 1.042 | 1.042 | 1.047 | 1.047 | 48 → 48 / 52 → 50 | 1.07 |

  * Concrete example (`dev/logs/diag_ortho_pairs_concrete.log`): in splits 1 and 6, pair (0,7) has variance share
    0.83% / 0.73% under S4 (not counted) and 1.30% / 1.02% under E1 (counted). Its |corr| with the main effects is
    nearly the same in both: train 0.027 → 0.026 and 0.025 → 0.023; test 0.080 → 0.084 and 0.189 → 0.204.
  * The same happens on the airfoil benchmark split's ortho_out: pair (3,4) has share 0.87% → 1.02%, and its |corr|
    is 0.148 under S4 vs 0.133 under E1.
  * The airfoil benchmark split's **ortho_in** increase is real, not a threshold effect: the same 3 pairs give
    0.0170 vs 0.0084. Over 8 splits, airfoil ortho_in is 0.81× S4.
  * The abalone benchmark split's ortho_out (2.86× baseline) is S4's known split noise (S4 2.65×). Over 8 splits
    E1 / S4 = 0.95.

## Self-test results (this machine, `22b414443084`)
### Subset (`subset_selftest.json`): ratio to `baseline/env-22b414443084/subset.json`
| | S4 (env file) | E1 round 1 | **E1 round 2** |
|---|---|---|---|
| analytical resid_out | 0.782 | 0.751 | **0.764** |
| analytical mse_eta12 / eta34 | 0.946 / 0.959 | 0.948 / 0.956 | **0.951 / 0.955** |
| analytical mse_others | 0.745 | 0.746 | **0.773** |
| analytical mse_eta1..6 | 0.94–1.00 | 0.92–1.01 | **0.92–1.01** |
| analytical fit_s | 1.51 | 3.49 | **3.46** |
| airfoil resid_in / resid_out | 1.255 / 0.834 | 1.223 / 0.703 | **1.144 / 0.698** |
| airfoil ortho_in / ortho_out | 0.868 / 1.128 | 1.486 / 0.950 | **1.757 / 1.340** |
| airfoil locvar_in / fit_s | 0.878 / 1.25 | 0.555 / 2.98 | **0.581 / 2.61** |
| concrete resid_in / resid_out | 1.411 / 0.838 | 1.650 / 0.786 | **1.592 / 0.798** |
| concrete ortho_in / ortho_out | 0.964 / 0.872 | 1.091 / 0.848 | **1.094 / 0.820** |
| concrete locvar_in / fit_s | 1.058 / 1.14 | 1.479 / 2.37 | **1.694 / 2.28** |
| abalone resid_in / resid_out | 1.392 / 0.519 | 1.377 / 0.479 | **1.377 / 0.479** |
| abalone ortho_in / ortho_out | 0.942 / 2.597 | 1.044 / 2.805 | **1.044 / 2.805** |
| abalone locvar_in / fit_s | 0.988 / 0.65 | 0.959 / 1.58 | **0.959 / 1.43** |

Abalone selects the same candidate as in round 1 (a tie set of 6, whose best in-sample member is the argmin).
Round 1's subset file is kept as `dev/subset_selftest_round1.json`.

### Full mode (`full_selftest.json`, no errors, wall 1,746 s): ratio to `baseline/env-22b414443084/full.json`
| dataset | resid_in S4 / E1 | resid_out S4 / E1 | ortho_in S4 / E1 | ortho_out S4 / E1 | locvar_in S4 / E1 | fit_s base → E1 (S4, E1 ratio) |
|---|---|---|---|---|---|---|
| abalone | 1.392 / **1.377** | 0.511 / **0.471** | 0.942 / **1.044** | 2.646 / **2.857** | 0.988 / **0.959** | 17.3 → 29.3 (0.72, **1.70**) |
| airfoil | 1.255 / **1.144** | 0.835 / **0.699** | 0.868 / **1.757** | 1.152 / **1.368** | 0.878 / **0.581** | 3.9 → 8.3 (0.99, **2.10**) |
| bike | 1.005 / **1.002** | 0.998 / **0.999** | 1.000 / **0.999** | 1.000 / **1.000** | 0.946 / **0.975** | 46.4 → 189.0 (0.79, **4.07**) |
| housing | 1.109 / **1.057** | 0.910 / **0.911** | 1.013 / **1.002** | 0.989 / **0.956** | 0.750 / **0.743** | 99.2 → 262.2 (1.13, **2.64**) |
| concrete | 1.411 / **1.592** | 0.842 / **0.802** | 0.964 / **1.094** | 0.871 / **0.819** | 1.058 / **1.694** | 4.7 → 13.7 (1.34, **2.92**) |
| nutrition | 1.685 / **1.392** | 0.750 / **0.692** | 0.949 / **1.060** | 0.769 / **0.921** | 0.970 / **1.022** | 6.6 → 28.5 (1.68, **4.35**) |
| parkinson | 1.012 / **1.005** | 1.002 / **0.990** | 1.000 / **1.001** | 0.998 / **0.997** | 0.927 / **0.941** | 42.1 → 106.7 (0.94, **2.54**) |
| powerplant | 1.058 / **1.121** | 0.965 / **0.938** | n/a / **n/a** | n/a / **n/a** | 0.989 / **0.981** | 12.1 → 58.1 (2.47, **4.80**) |
| superconduct | 1.008 / **1.004** | 0.998 / **0.996** | n/a / **n/a** | n/a / **n/a** | 0.981 / **0.982** | 359.0 → 403.2 (0.47, **1.12**) |

| analytical (10 reps) | baseline | S4 | E1 round 1 | **E1 round 2** |
|---|---|---|---|---|
| mse_eta1..4 | | 0.992–1.001 | 0.998–1.005 | **0.999–1.005** |
| mse_eta5 / eta6 | 0.000301 / 0.000367 | 0.963 / 0.989 | 0.950 / 0.978 | **0.951 / 0.976** |
| mse_eta12 / eta34 | 0.03118 / 0.02821 | 0.970 / 0.982 | 0.959 / 0.973 | **0.960 / 0.972** |
| mse_others | 0.002125 | 0.776 | 0.801 | **0.816** |
| resid_out | 0.01016 | 0.824 | 0.793 | **0.800** |
| fit_s | 17.3 s | 1.78 | 3.47 | **3.37** |

* **resid_out.** Below S4 on 7 of 9 datasets: abalone, airfoil, concrete, nutrition, parkinson, powerplant and
  superconduct. On bike (0.999 vs 0.998) and housing (0.911 vs 0.910) it is 0.1% above S4, i.e. equal within noise.
* **resid_in.** Below S4 on 7 of 9 datasets; nutrition goes from 1.69× to 1.39× baseline. It is higher than S4 on
  concrete (1.59× vs 1.41×) and powerplant (1.12× vs 1.06×).
  * Relative to E1 round 1, it is lower or equal everywhere except powerplant (equal) and abalone (the same
    selection).
* **Analytical.** mse_others is slightly worse than round 1 and S4 (0.816 vs 0.801 / 0.776). The true interactions
  are about 1% better than S4.

## What the method does
Per tree t: y = T_t(X_train) − η0_t. β is parameterised in S4's exact null space of the hierarchical-orthogonality
rows (β = Q γ per pair). The method minimises

  (1/n) Σ_i (h_i'β − y_i)² + ρ Σ_v w_v (h_v'β − y_v)² + (κ/n) β'Pβ + [S4's zero-mean rows]

* **Unchanged from S4 / TreeHFD.** Orthogonality is hard, and both it and the zero-mean conditions are under the
  empirical measure P_n. So the estimand is TreeHFD's empirical HFD; the anchor only regularises how it is
  estimated.
  * Precision about "hard": as in S4 and the baseline, the zero-mean conditions are least-squares rows with the
    baseline's weights, not exact constraints. Only orthogonality is exact. The idea's text calls both "hard"; the
    code is S4's.
  * P is S4's `ridge` or `lattice` prior. Unseen pair cells use S4's `zero` or `harmonic` rule.
* **Virtual atoms (ported from S2, `_shift_values`, `_TreeEval`, `_union_splits`):**
  * S = variables split on anywhere in the ensemble. U_j = union of all (float32) thresholds on x_j.
  * For every training point i, j ∈ S and s = ±1, the copy is x_i with x_ij replaced by the training median of the
    adjacent non-empty union bin. Union-bin membership uses float32 inputs and thresholds, as XGBoost routes.
  * The median is taken on the raw inputs, so the Cartesian partition bins a copy exactly like a training point
    with the same value.
  * Copies with no adjacent non-empty bin are skipped.
* **Per tree, a copy is kept** only if:
  * it moves one of tree t's main bins, using the same `np.digitize` as predict (otherwise it duplicates the real
    row); and
  * every pair cell it touches has empirical mass (S2's "drop" rule).

  So the column set and all of S4's latent-cell / harmonic machinery are unchanged.
* **Target and weight.**
  * The target is y_v = T_t(copy) − η0_t, read from the fixed tree's leaves (`_TreeEval`, float32 thresholds).
  * `route_err` is the tree evaluator's maximum deviation from the package's per-tree predictions on X_train. It is
    ≤ 7.6e-6 in all 32 diagnostic fits (float32 leaf accumulation).
  * w_v = 1/(2|S| n), so ρ is the virtual-to-real mass ratio.
  * The copies are a deterministic function of X_train and the model's split structure: no RNG, labels, test inputs
    or refitting.
* **Merging.** Copies whose cells are identical fall in the same joint cell of tree t, so they have the same tree
  output (checked: `merge_err` ≤ 1.3e-15). They are merged into one row with weight w × multiplicity; this is exact.
  Round 2: the row grouping uses a 64-bit hash of the cell indices (`_unique_rows`), verified exactly against the
  rows (falls back to `np.unique(axis=0)` on any collision); only the order of the merged rows changes.
* **Anchor matrices.** A = Hv' diag(w·mult) Hv and b = Hv' (w·mult·y_v), dense like N. ρ enters linearly (N + ρA,
  rhs + ρb). Each (prior, ρ) therefore needs one eigendecomposition of the whitened R'(N + ρA)R, and the whole κ path
  stays closed form. ρ = 0 is S4, bit for bit (check 7a).
* **Risk R (S4's Good-Turing leave-one-cell-out risk), made honest for the anchor.**
  * Points in repeated joint cells keep their in-sample residual.
  * A Good-Turing singleton i is left out with a **block Woodbury downdate**. In eigen coordinates Λ = diag(S + κ/n):
    * e₋ = (I − K W)⁻¹ e_B, with K = Z_B Λ⁻¹ Z_B';
    * γ₋ = γ − Λ⁻¹ Z_B' W e₋.
  * The block B_i contains:
    1. the real row i;
    2. every kept copy generated from x_i;
    3. copies of other points shifted into a union bin whose **only** member is i (their location is x_ij itself;
       an addition beyond the idea, for honesty);
    4. copies of other points that touch a main bin or pair cell whose only training point is i ("drop" would reject
       them once i is gone).
  * For **orphans**, the emptied cell then becomes latent. The leave-out prediction is S4's unseen-cell rule
    functional applied to the downdated β₋ (`_orphan_functionals`, unchanged).
  * Blocks are grouped by size and batched: exact sizes up to 32 (`BLOCK_EXACT`, round 2; was 8), then rounded up to
    a multiple of 8 and padded with zero-weight rows, which leaves the downdate exact. Chunks are cache-sized
    (`BLOCK_CHUNK` = 3e5 elements, round 2; was 4e6), and size-1 blocks skip the gathers (plain rank-one PRESS).
* **Selection.**
  * The variants are (prior, rule) × ρ. The per-tree choice is the argmin of R over (ρ, κ), with ties going to the
    smaller ρ, then the smaller κ.
  * S4's ensemble step is kept: leave-out residuals are summed over trees per point, and the in-sample fidelity is
    kept per candidate.
  * ρ is added as a candidate dimension with four modes: common ρ = 0, common ρ = 0.25, common ρ = 1, or the
    per-tree joint argmin over (ρ, κ). Each mode is crossed with S4's candidates (per-tree κ index shifted by
    −5..+3, or a common κ).
  * The candidate order encodes the tie-break: variant, then ρ mode (smaller common ρ first, per-tree last), then
    |shift|.
  * S4 is nested (the ρ = 0 mode), so the selection can always fall back to it.
  * **Round 2: one-SE rule (`SE_RULE = 1`, S4's existing code path, now `_se_select`).** Among the ensemble
    candidates whose R is within one paired standard error of the minimum (SE of the per-point difference of squared
    ensemble leave-out residuals to the argmin), the one with the smallest in-sample residual to T(X_train) is taken;
    remaining ties follow the candidate order. With `SE_RULE = 0` the plain argmin of round 1 is recovered exactly.
* **finalize / predict** are S4's: there are no new columns, and unseen cells use zero or harmonic.
* **Diagnostics** (`state.diagnostics`):
  * per tree: chosen κ and ρ (`kappa_chosen`, `rho_chosen`), number of kept virtual atoms (`n_virtual`), `route_err`,
    and the per-tree argmins (`per_tree_idx`, `per_tree_joint`);
  * the selected candidate and its ρ mode, the plain argmin (`min_risk`) and the size of the one-SE tie set
    (`n_tied`);
  * `risk_over_var` and `resid_in_over_var` (both label-free, / Var[T(X_train)]);
  * the full candidate table.


## Hyper-parameters (all fixed a priori; nothing is per-dataset)
| name | value | source |
|---|---|---|
| RHOS | (0, 0.25, 1) | idea: fixed a priori, capped at 1 so the virtual mass never exceeds the empirical mass |
| ANCHOR_PRIORS | ("lattice",) | pre-registered compute fallback (7e), applied in round 1. ridge keeps its ρ = 0 member |
| SE_RULE | **1 (round 2)** | one-SE rule towards in-sample fidelity (S4's code path); k fixed before any run |
| w_v | 1/(2\|S\| n) per copy | idea |
| shift kernel | ±1 adjacent non-empty union-grid bin, training median | S2 |
| copy acceptance | moves a tree-t main bin, all touched pair cells populated ("drop") | idea / S2 |
| BLOCK_CHUNK, BLOCK_EXACT | 3e5 elements, 32 rows (round 2) | speed only; results unchanged |
| KAPPAS, PRIORS, RULES, EPS_MAIN, LATTICE_RIDGE, SHIFTS, HARD_ORTHO | as S4 (1e-2…1e4; ridge, lattice; zero, harmonic; 1; 0.1; −5..+3; True) | S4, unchanged |

## Deviations from the idea (and why)
1. **Compute fallback (7e), applied in round 1: ρ > 0 is fitted for the lattice prior only.**
   * The trigger was pre-registered ("any full-mode fit_s > 4× baseline"). With both priors anchored, powerplant's
     per-tree fits alone took 7.2× baseline.
   * Details are in `dev/NOTES_round1.md`. Round 2 needed no further fallback.
2. **Ensemble selection (round 2): one-SE rule instead of the plain argmin.**
   * The idea's tie-break ("ties go to the smaller ρ, then the smaller κ") is kept for exact ties, through the
     candidate order.
   * Statistical ties (within one paired SE of R) now go to the best in-sample fidelity.
   * `SE_RULE = 0` restores round 1 exactly.
3. **Leave-out block item 3** (copies of other points shifted into a union bin whose only member is i) is an extra
   removal not listed in the idea. The union-bin median of a singleton bin is x_ij itself, so these copies do not
   exist without i.
4. **Identical copies are merged** (exact). This is an implementation detail.

## Pre-registered checks
* **(a) ρ = 0 reproduces S4.** `dev/method_rho0.py` (RHOS = (0,), SE_RULE = 0) through the subset harness **on
  this machine** equals `S4/subset_env-22b414443084.json` on every metric, including the analytical case.
  * Round 1 had also checked it on the old machine (`dev/subset_rho0.json`) and on 32 dataset-splits.
* **(b) Closed-form block leave-out vs brute-force refits** (round 1, `dev/check_loo.py`, `dev/logs/check_loo_*`):
  mean-square(closed) / mean-square(brute) is 0.83–1.10. That matches S4's own closed-form error, with no optimism
  that grows with ρ. Round 2's code gives the same leave-out residuals to 3e-10, so these results still hold.
* **Virtual-atom consistency** (round 1, `dev/check_atoms.py`): 0 mismatches in 5,368 comparisons; route error
  ≤ 4.4e-6.
* **(c) Subset and full self-tests:** above; no errors.
* **(d) ρ ablation over the 8 diagnostic splits:** above (round 2 adds the one-SE rows and the fixed-κ table). Round
  1's version is in `dev/logs/ablate_rho_*.log`.
* **(e) Compute fallback:** applied in round 1. In round 2 every full-mode fit_s is below 5×.

## Trade-offs and caveats (reported, not hidden)
* **The idea's main target is still not reached.** The idea expected the resid_in excess over baseline to roughly
  halve relative to S4. Round 2 gives 0.94–0.99× S4 over 8 splits.
  * On the benchmark split it is below S4 on 7 of 9 datasets. It is above S4 on concrete (1.59× vs 1.41× baseline)
    and powerplant (1.12× vs 1.06×).
  * The mechanism is identified: R prefers ρ = 1 by more than its SE, and ρ = 1 comes with κ at the floor of its
    grid, where the anchor competes with the real rows. The fixed-κ table shows the anchor lowers resid_in only
    when κ ≥ 1.
* **resid_out is the real gain.** It is 7–14% below S4 over 8 splits on the four diagnosed datasets. It is below S4
  on 7 of 9 full-mode datasets and within 0.1% of S4 on bike and housing. The analytical case is 0.80× baseline
  (S4 0.82×).
* **Orthogonality.**
  * Within-tree orthogonality is exact.
  * On the harness metric, E1 is worse than S4 on concrete (1.49× / 1.42× over 8 splits). About 90% of that is the
    1% variance threshold: the same pairs give 1.04× / 1.05×.
  * A real +4–7% remains on concrete, abalone and nutrition. Airfoil is 0.81× S4 on ortho_in and 1.07× on
    ortho_out (same pairs).
  * On the benchmark split, airfoil ortho_in is 1.76× baseline (S4 0.87×). That is a real per-pair increase on this
    split (0.0170 vs 0.0084 on the same 3 pairs).
* **locvar_in.** Concrete is 1.69× baseline on the benchmark split (S4 1.06×) and 1.17× S4 over 8 splits: the
  anchored thin cells give noisier main effects there. Airfoil is smoother (0.58×; 0.66× S4 over 8 splits).
* **Analytical spurious interactions.** mse_others is 0.816× baseline, slightly worse than S4 (0.776×) and round 1
  (0.801×). The anchor gives interactions a little more variance, the same effect as the threshold crossings on
  concrete.
* **Compute.**
  * 1.12–4.80× baseline in full mode; powerplant is closest to 5×.
  * The cost is inherent to the design: two extra eigendecompositions per tree plus the block leave-out.
  * Timings are single-threaded BLAS (`OPENBLAS_NUM_THREADS=1`) on a hybrid-core i9-14900HX, with ±10% run-to-run
    variation. Both official runs of round 2 were made on an otherwise idle machine.
* **Held-out numbers.** No choice in round 2 was made from held-out numbers:
  * the speed-up is output-preserving;
  * `SE_RULE = 1` was committed before its ablation was run (only airfoil split 0 had been seen when the ablation
    started, and it did not change the plan);
  * the ρ = 0.25 alternative was deliberately **not** adopted.

## Files
* `method.py`: benchmark interface (unchanged). `lib/agtloco.py`: the method. `lib/treehfd_mod/`: unmodified
  package copy.
* `subset_selftest.json` / `.log`, `full_selftest.json` / `.log`: round-2 harness outputs.
* `dev/`:
  * **Selection and ablation:** `dev_e1.py` (options `se=`, `ablate`, `ortho`), `ablate_r2.sh`,
    `summarize_r2.py`, `diag_ortho_pairs.py`, `diag_ortho_common.py`, `summarize_ortho_common.py`.
  * **Round-1 ablation:** `ablate_rho.sh`, `summarize_ablation.py`.
  * **Checks:** `check_speedup.py` (implementation vs implementation, per tree and end to end), `check_loo.py`,
    `check_atoms.py`, `method_rho0.py`, `subset_rho0.json`, `subset_rho0_env-22b414443084.json`.
  * **Compute:**
    * `time_fit.py`, `time_parts.py` (option `impl=`), `time_anchor.py`, `prof_fit.py`, `prof_cand.py`,
      `profile_fit.py`, `profile_trees.py`;
    * micro-benchmarks: `bench_kernels.py`, `bench_spmm.py`, `bench_eigh.py`, `bench_anchor.py`, `bench_prev.npy`.
  * **Helpers:** `common.py`, `compare.py`, `run_log.py`, `full_table.py`.
  * **Round-1 state:** `NOTES_round1.md`, `agtloco_before_speedup.py.txt` (round-1 code),
    `subset_selftest_round1.json`, `full_run1_unoptimised.json`.
  * `logs/`: every log cited above. The round-2 logs are `ablate_r2_*`, `diag_ortho_*`, `subset_rho0_env.log`,
    `time_parts_bike_cand1.log` and `full_table_round2.md`.
