# S1 — Ensemble-level re-orthogonalization on the union-of-splits partition

## What is implemented
`method.py` = baseline TreeHFD (copy in `lib/treehfd_mod/`, algorithm unchanged) + a post-hoc, sum-preserving
projection step (`lib/s1_projection.py`).

1. **TreeHFD fit** exactly as the baseline (`XGBTreeHFD.fit`). The only change to the package copy is bookkeeping:
   `TreeHFD.fit` keeps its training cells and `XGBTreeHFD.fit` sums `hfd_coeffs[cells]` into
   `hfd.train_components` (main, inter) on X_train, then frees the cells. This is bit-identical to
   `hfd.predict(X_train)` (max abs difference 0.0 on analytical rep 0, airfoil, abalone and concrete) and saves the
   second pass over the ensemble, which alone cost about 0.25x the fit time.
2. **Union bins.** For each variable j, all split thresholds of the ensemble on x_j (`xgb_table`, `Feature == f"f{j}"`)
   are collected and sorted; `np.digitize(x, edges)` gives the bin (same `x < split` convention as xgboost / TreeHFD).
   Adjacent bins are merged greedily left-to-right until each holds `>= M_MIN = 30` X_train rows. The last bin is also
   kept `>= M_MIN`, so no bin is empty.
3. **Additive projection per interaction (j,k).** Solve
   `min_g  sum_i (eta_jk(x_i) - g_j[b_j(i)] - g_k[b_k(i)])^2 + lam (|D g_j|^2 + |D g_k|^2)`
   under the empirical measure of X_train (D = first difference along the ordered bins). The normal equations are
   assembled with `np.bincount`: diagonal count blocks, the joint count table for the cross block, and bin sums of
   eta_jk. The gauge is fixed exactly by `g_k[0] = 0` (the penalty only sees differences), a 1e-8 ridge is added for
   numerical safety, and the dense `(B_j + B_k - 1)` system is solved by Cholesky. At `lam = 0` the corrected
   interaction is orthogonal, in-sample, to every function of the merged bins of x_j and of x_k. Checked: within-bin
   sums of the corrected interaction are about 1e-9 of its original L1 norm on abalone and parkinson.
4. **Transfer** (pointwise sum-preserving):
   `eta_jk <- eta_jk - g_j - g_k`, `eta_j <- eta_j + g_j - c_j`, `eta_k <- eta_k + g_k - c_k`,
   `eta0 <- eta0 + c_j + c_k`, with `c = mean_{X_train}(g)`. The main effects stay centred and the reconstruction is
   the baseline's at every x (checked: max |change| <= 1e-9 relative on train and eval points). Pairs are independent,
   because hierarchical orthogonality of eta_jk only concerns functions of x_j and x_k, so the order does not matter.
   At predict time, new points are binned with the stored edges and the stored `g` vectors are applied. Points outside
   the training range fall into the end bins.
5. **Choice of the penalty (label-free cross-fitting on X_train rows only).** 3 folds, deterministic (`row % 3`);
   no RNG anywhere. Grid `tau in {0, 1, 3, 10, 30, 100, 300, 1e3, 1e4, inf}`, `lam = tau` in units of rows: a bin
   holding many more than tau rows is barely smoothed, while thin bins are pooled with their neighbours. `tau = inf`
   means g is constant, i.e. the pair is left exactly as TreeHFD fitted it. For every pair, the tau with the smallest
   **held-out squared error of the additive fit** `sum_ho (eta_jk - g_j - g_k)^2` (fitted on the other two folds) is
   kept; exact ties go to the larger tau. The pair is then refitted on all rows with that tau.

## Hyper-parameters
| name | value | role |
|---|---|---|
| `M_MIN` | 30 | min X_train rows per merged union bin |
| `N_FOLDS` | 3 | cross-fitting folds (`row % 3`) |
| `TAU_GRID` | 0, 1, 3, 10, 30, 100, 300, 1e3, 1e4, inf | first-difference penalty, in rows |
| `SCALE` | `"abs"` | lam = tau (alternative `"mean"`: lam = tau * n / n_bins) |
| `SELECT` | `"pair"` | one tau per interaction (alternative `"global"`) |
| `CRITERION` | `"mse"` | held-out squared error (alternative `"leak"`, see below) |
| `RIDGE` | 1e-8 | numerical ridge on the normal equations |

All alternatives can still be selected through `fit_projection(...)` arguments for ablations.

## Design decisions and deviations from the idea (with reasons)
All dev numbers below come from `dev/dev_eval.py` on cached baseline fits of the harness models, using label-free
metrics only (resid, ortho, locvar), written as "baseline -> S1". No ground-truth component was used for any choice.

**(a) Held-out criterion: squared error instead of the idea's held-out |corr(eta_jk, eta_j)|.** I implemented the
idea's criterion as `criterion="leak"`: the held-out sum over pairs of cov(eta_jk', eta_m')^2 / var(eta_m'), which is
the squared-covariance form of the correlation, weighted by interaction variance. It can run per pair or with one
global tau. It turned out to be biased toward "no transfer":
- The held-out rows were part of TreeHFD's own fit, so the untouched decomposition (tau = inf) is scored in-sample,
  while every finite tau is scored honestly.
- Correlations estimated on n/3 rows carry sampling noise of about 1/sqrt(n/3), which is as large as the signal.

With one global tau, the leak criterion left airfoil, concrete, abalone, nutrition and housing unchanged.

The squared-error criterion is unbiased. By Pythagoras, E||eta_jk - g||^2 = const + ||P eta_jk - g||^2, where P is the
population projection onto {f(x_j) + f(x_k)}. Minimising it therefore minimises the L2 error of the estimated
transfer, which is exactly the distance of the corrected interaction to the HFD-orthogonal subspace. Each pair is a
separate estimation problem, so tau is chosen per pair.

Dev comparison with m_min = 10 (ortho_in / ortho_out / locvar_in):

| dataset | baseline | mse, per pair | leak, per pair | leak, global | fixed tau = 0 |
|---|---|---|---|---|---|
| airfoil | .0097/.0985/5.2e-5 | .0214/.0985/4.4e-4 | .0214/.0985/4.4e-4 | unchanged | .0001/.0932/6.6e-4 |
| concrete | .0064/.0817/.0125 | .0044/.0591/.0291 | .0055/.0745/.0270 | unchanged | .0015/.0734/.0385 |
| abalone | .0268/.0159/.00256 | .0099/.0254/.00267 | .0099/.0334/.00254 | unchanged | .0003/.0469/.00603 |
| nutrition | .0162/.1101/.00282 | unchanged | unchanged | unchanged | .0086/.1388/.00452 |
| analytical rep 0 | .0712/.0797/.00532 | unchanged | .0287/.0391/.00538 | .0287/.0340/.00515 | .0000/.0862/.0188 |
| parkinson | .0820/.1176/.00469 | .0585/.0819/.0067 | .0135/.0784/.00513 | .0569/.0823/.0104 | .0000/.0962/.0148 |
| housing | .0280/.0488/.00180 | .0024/.0416/.00179 | .0148/.0498/.00179 | unchanged | .0000/.0478/.00193 |
| bike | .0134/.0212/.000345 | .0011/.0212/.000399 | .0127/.0212/.000346 | .0096/.0212/.000341 | .0000/.0214/.000415 |

`tau = 0` (exact in-sample projection) confirms the idea's in-sample claim: ortho_in drops to about 0. However, it
does not generalise: ortho_out is worse on abalone, nutrition and analytical, and main effects become 1.1-13x rougher.

**(b) Penalty in units of rows (`SCALE="abs"`), not in units of the mean bin count.** With `lam = tau * n / n_bins`, a
variable with few bins (e.g. a binary one) gets its single difference shrunk as hard as a fine variable. Parkinson
(0,1) involves such a variable, and every tau >= 10 then left parkinson unchanged. With `lam = tau` rows, only bins
that are thin relative to tau are pooled, which is the noise-driven smoothing the idea intends. Elsewhere the two
scalings gave similar numbers.

**(c) `tau = inf` is in the grid**, so cross-fitting can decline the transfer for a pair, and it often does. In the
analytical case only 3 of 45 pairs got a transfer (all in rep 1, all at tau = 1e4).

**(d) `M_MIN = 30` instead of the idea's example value 10.** This trades resolution against roughness. Dev results
(mse, per pair; ortho_in / ortho_out / locvar_in):

| dataset | m_min 5 | m_min 10 | m_min 30 (final) |
|---|---|---|---|
| airfoil | .0225/.0985/3.9e-3 | .0214/.0985/4.4e-4 | .0098/.0985/6.4e-5 |
| concrete | .0060/.0594/.0346 | .0044/.0591/.0291 | .0138/.0813/.0140 |
| abalone | .0099/.0256/.00268 | .0099/.0254/.00267 | .0099/.0220/.00260 |
| parkinson | .0585/.0820/.0067 | .0585/.0819/.0067 | .0322/.0824/.00509 |
| housing | .0024/.0416/.00179 | .0024/.0416/.00179 | .0023/.0413/.00179 |
| bike | .0011/.0212/.000488 | .0011/.0212/.000399 | .0011/.0212/.000399 |

m_min = 10 makes locvar 8x worse on airfoil and 2.3x worse on concrete. m_min = 30 keeps the increase within
1.1-1.2x and is better on parkinson. It makes concrete's ortho_in worse: with n = 824, bins of at least 30 rows are
coarser than some individual trees' grids, so the transfer can undo part of the per-tree orthogonality. Caveat: this
value was chosen by looking at label-free metrics of the benchmark's own datasets, including the subset datasets,
because no separate development data exists.

**(e) Other choices.** Folds are deterministic (`row % 3`): X_train is already a random permutation in the harness,
and the method uses no RNG. The gauge is fixed exactly by `g_k[0] = 0`. Pairs with zero variance, or where both
variables have a single bin, are skipped.

**Unchanged baseline behaviour.** For eval points that fall in unseen interaction cells, TreeHFD's
`predict_partition` breaks ties with an unseeded RNG. As a result, eval components, resid_out and the analytical mse
values jitter slightly from run to run, in both the baseline and S1. The S1 transfer itself is deterministic and
exactly sum-preserving.

## Self-test (subset, this machine `22b414443084`)
`subset_selftest.json` (final code; 135 s wall, no errors), compared with `baseline/env-22b414443084/subset.json`:

| analytical (mean of 3 reps) | baseline | S1 |
|---|---|---|
| mse_eta1 | 0.02705 | 0.02687 |
| mse_eta2 | 0.01818 | 0.01804 |
| mse_eta3 | 0.01737 | 0.01737 |
| mse_eta4 | 0.01769 | 0.01769 |
| mse_eta5 | 0.00033 | 0.00033 |
| mse_eta6 | 0.00054 | 0.00055 |
| mse_eta12 | 0.03387 | 0.03347 |
| mse_eta34 | 0.03168 | 0.03162 |
| mse_others | 0.00215 | 0.00215 |
| resid_out | 0.01022 | 0.01023 |
| fit_s | 19.1 | 19.9 |

The analytical changes (at most about 1%) are at the level of the eval-time tie-break jitter. Reps 0 and 2 have no
transfer at all, and rep 1 has 3 heavily smoothed ones.

| dataset | resid_in | resid_out | ortho_in | ortho_out | locvar_in | fit_s |
|---|---|---|---|---|---|---|
| airfoil | .009845 -> .009845 | .02695 -> .02648 | .00967 -> .00979 | .0989 -> .0980 | 5.20e-5 -> 6.38e-5 | 3.3 -> 3.2 |
| concrete | .001534 -> .001534 | .02512 -> .02510 | .00636 -> .01383 | .0822 -> .0813 | .01249 -> .01399 | 6.1 -> 5.4 |
| abalone | .004553 -> .004553 | .04034 -> .04106 | .02680 -> .00987 | .01586 -> .02069 | .00256 -> .00260 | 21.9 -> 17.7 |

Dev check on the full-mode datasets (cached fits, `dev/final_config_dev.txt`; not an official run):

| dataset | ortho_in | ortho_out | locvar_in |
|---|---|---|---|
| nutrition | unchanged (20 of 21 pairs at tau = inf) | .1101 -> .1101 | unchanged |
| parkinson | .0820 -> .0322 | .1176 -> .0824 | .00469 -> .00509 |
| housing | .0280 -> .0023 | .0488 -> .0413 | .00180 -> .00179 |
| bike | .0134 -> .0011 | .0212 -> .0212 | .000345 -> .000399 |
| superconduct | null -> null | null -> null | .000663 -> .00167 |

The projection step takes 0.03-1.8 s on these datasets and 54 s on superconduct (2927 pairs, about 13% of its 421 s
TreeHFD fit on this machine). The harness `fit_s` should therefore be about 1.0-1.15x the baseline. The full mode was
not run by me.

## Honest assessment
- **Reconstruction is untouched.** resid_in is identical, and resid_out differs only by the baseline's tie-break
  jitter (checked: the transfer changes the reconstruction by <= 1e-9 relative).
- **In-sample orthogonality improves clearly where a real additive leak exists:** abalone 0.027 -> 0.010,
  parkinson 0.082 -> 0.032, housing 0.028 -> 0.002, bike 0.013 -> 0.001. **It gets worse on concrete**
  (0.0064 -> 0.0138), for the reason given in (d). It is unchanged on airfoil and nutrition.
- **Out-of-sample orthogonality** improves on parkinson (0.118 -> 0.082) and housing (0.049 -> 0.041). On airfoil,
  concrete, nutrition and abalone, ortho_out sits at the test-sample noise floor: the standard deviation of a
  correlation is about 1/sqrt(n_test), i.e. 0.058 / 0.070 / 0.047 / 0.035. The small changes there (abalone
  0.016 -> 0.021 is worse) cannot be told apart from noise.
- **Roughness is the main cost.** locvar_in rises 1.1-1.2x on airfoil, concrete, parkinson and bike, and 2.5x on
  superconduct. On superconduct the rise is concentrated in main effects with negligible variance (1e-5 to 1e-4 of
  Var T). There, many small transfers (about 70 pairs per variable) make these components relatively rough, and the
  metric normalises by each component's own variance. The method moves about a quarter of superconduct's interaction
  variance into main effects (interaction share 0.044 -> 0.032).
- **The analytical case is essentially unchanged.** The additive part left in the TreeHFD interactions there is small
  in L2: in-sample corr(eta_12, eta_1) is about 0.07, i.e. about 0.5% of var(eta_12). Cross-fitting judges it not
  estimable better than leaving it alone, so the idea's expected gains on mse_eta12 / mse_eta34 / mse_others did not
  materialise. The leak criterion would have moved it (analytical ortho_out 0.080 -> 0.034 in dev), but by the
  Pythagoras argument the mse metrics could change by at most about 0.5% of var(eta_jk) either way.
- **Novelty caveat (already flagged in the idea):** the core mechanism is the purification of Lengerich et al. (2020).
  The additions here are the union-of-splits bins and the cross-fitted smoothing.

## Files
- `method.py` — interface (`fit`, `predict`).
- `lib/s1_projection.py` — union bins, penalised additive projection, cross-fitting, transfer.
- `lib/treehfd_mod/` — copy of the baseline package; only bookkeeping added (`train_bins`, `train_components`).
- `dev/` — development scripts only, not used by the method:
  - `cache_baseline.py` rebuilds the harness models and caches baseline fits. Labels are used only to train the
    XGBoost model exactly as the harness does.
  - `dev_eval.py` applies the projection to cached fits and reports label-free metrics (resid, ortho, locvar).
  - `compare.py` compares result files.
  - `final_config_dev.txt` holds the dev numbers of the final configuration.
- `subset_selftest.json` / `.log` — self-test.
