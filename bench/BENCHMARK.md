# TreeHFD benchmark (fixed protocol)

Reconstructed from Bénard (NeurIPS 2025), Table 1 (analytical case) and Table 2 / Appendix B.3.1 (real data).

## Method interface
A method is a Python file defining:

```python
def fit(model, X_train, interaction_order=2):   # model: fitted xgboost.XGBRegressor; X_train: its training inputs
    return state
def predict(state, X):
    return {"intercept": float, "main": (n, p) array, "inter": (n, K) array, "inter_list": (K, 2) int array}
```
The harness runs the method in a separate process that never sees labels or ground truth. The reconstruction is
`intercept + main.sum(1) + inter.sum(1)` and should approximate the model prediction T(x).
Baseline implementation: `bench/baseline_method.py` (original TreeHFD).

Self-test: `<ROOT>/.conda/bin/python <ROOT>/bench/harness.py --method method.py --mode subset --out subset_selftest.json` (~1-3 min). A single fit call slower than 1 hour is a timeout failure.

## Analytical case (paper Table 1)
X ~ N(0, Σ), p = 6, unit variances, all correlations ρ = 0.5; Y = sin(2πX1) + X1·X2 + X3·X4 + N(0, 0.5²); n = 5000 train,
5000 independent test points; XGBoost eta=0.1, 100 trees, max_depth=6. Ground-truth HFD components are analytical.
Metrics (lower is better, mean over repetitions): `mse_eta1..6` (main effects), `mse_eta12`, `mse_eta34` (true interactions),
`mse_others` (spurious interactions, target 0), `resid_out` (normalised residual on test points), `fit_s`.
Subset mode: 3 repetitions. Full mode: 10 repetitions (as in the paper).

## Real data (paper Table 2): 80/20 split, same XGBoost settings
Per dataset (lower is better):
- `resid_in` / `resid_out`: MSE between the reconstruction and T(x), divided by Var[T(x)], on train / held-out points.
- `ortho_in` / `ortho_out`: max |corr| between each interaction with variance ≥ 1% of Var[T(x)] and its two main effects
  (hierarchical orthogonality; `null` when no interaction passes the 1% threshold).
- `locvar_in`: local variability — for each variable, variance of its main effect over the 10 nearest neighbours in that
  variable, averaged over points, normalised by the component variance, averaged over main effects.
- `fit_s`: fit time in seconds. `xgb_r2_test` is informational only.
Subset mode: airfoil, concrete, abalone. Full mode: abalone, airfoil, bike, housing, concrete, nutrition, parkinson,
powerplant, superconduct.

An improvement must hold across datasets and metrics; trading one metric for another must be reported, not hidden.
