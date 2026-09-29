# S1 — Ensemble-level re-orthogonalization on the union-of-splits partition

## What is implemented
`method.py` = baseline TreeHFD (unchanged copy in `lib/treehfd_mod/`) + a post-hoc, sum-preserving projection step
(`lib/s1_projection.py`).

1. **TreeHFD fit** exactly as the baseline (`XGBTreeHFD.fit`), then the aggregated components are evaluated on X_train
   (`hfd.predict(X_train)`, deterministic on training points).
2. **Union bins.** For each variable j, all split thresholds of the ensemble on x_j (`xgb_table`, `Feature == f"f{j}"`)
   are collected and sorted; `np.digitize(x, edges)` gives the bin (same `x < split` convention as xgboost / TreeHFD).
   Adjacent bins are merged greedily left-to-right until each holds `>= M_MIN = 30` X_train rows (the last bin is kept
   `>= M_MIN` too), so no bin is empty.
3. **Additive projection per interaction (j,k).** Solve
   `min_g  sum_i (eta_jk(x_i) - g_j[b_j(i)] - g_k[b_k(i)])^2 + lam (|D g_j|^2 + |D g_k|^2)`
   under the empirical measure of X_train (D = first difference along the ordered bins). The normal equations are
   assembled with `np.bincount` (diagonal count blocks, joint count table for the cross block, bin sums of eta_jk),
   the gauge is fixed exactly by `g_k[0] = 0` (penalty only sees differences), a 1e-8 ridge is added for numerical
   safety, and the dense `(B_j + B_k - 1)` system is solved by Cholesky. At `lam = 0` the corrected interaction is
   orthogonal, in-sample, to every function of the merged bins of x_j and of x_k (checked: within-bin sums of the
   corrected interaction are ~1e-9 of its original L1 norm on abalone and parkinson).
4. **Transfer** (pointwise sum-preserving):
   `eta_jk <- eta_jk - g_j - g_k`, `eta_j <- eta_j + g_j - c_j`, `eta_k <- eta_k + g_k - c_k`,
   `eta0 <- eta0 + c_j + c_k`, with `c = mean_{X_train}(g)`, so the main effects stay centred and the reconstruction is
   the baseline's at every x (checked: max |change| ~1e-9 relative on train and eval points, resid_in/out unchanged).
   Pairs are independent (hierarchical orthogonality of eta_jk only concerns functions of x_j and x_k), so the order
   does not matter. At predict time, new points are binned with the stored edges and the stored `g` vectors are
   applied (points outside the training range fall into the end bins).
5. **Choice of the penalty (label-free cross-fitting on X_train rows only).** 3 folds, deterministic (`row % 3`),
   no RNG anywhere. Grid `tau in {0, 1, 3, 10, 30, 100, 300, 1e3, 1e4, inf}`, `lam = tau` in units of rows
   (a bin holding >> tau rows is barely smoothed, thin bins are pooled with neighbours). `tau = inf` means g constant,
   i.e. the pair is left exactly as TreeHFD fitted it. For every pair the tau with the smallest **held-out squared
   error of the additive fit** `sum_ho (eta_jk - g_j - g_k)^2` (fitted on the other two folds) is kept; exact ties go to
   the larger tau. The pair is then refitted on all rows with that tau.

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

All alternatives are still selectable in `fit_projection(...)` for ablations.

## Design decisions and deviations from the idea (with reasons)
PLACEHOLDER_DECISIONS

## Self-test (subset, this machine `22b414443084`)
PLACEHOLDER_SELFTEST

## Honest assessment
PLACEHOLDER_ASSESSMENT

## Files
- `method.py` — interface (`fit`, `predict`).
- `lib/s1_projection.py` — union bins, penalised additive projection, cross-fitting, transfer.
- `lib/treehfd_mod/` — unmodified copy of the baseline package.
- `dev/` — development scripts only (not used by the method): `cache_baseline.py` rebuilds the harness models and
  caches baseline fits (labels only to train the XGBoost model exactly as the harness does), `dev_eval.py` applies the
  projection to cached fits and reports label-free metrics (resid, ortho, locvar), `compare.py` compares result files,
  `final_config_dev.txt` the dev numbers of the final configuration.
- `subset_selftest.json` / `.log` — self-test.
