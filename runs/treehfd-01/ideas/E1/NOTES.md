# E1 — Anchored GT-LOCO TreeHFD (model-anchored Gaussian prior inside hard-orthogonal TreeHFD)

## Status
**Implemented, and the subset and full self-tests run without errors** through the official harness on this
machine (`22b414443084`). **Unfinished: compute.** In full mode, bike (7.2×), nutrition (5.2×) and powerplant (5.1×)
exceed the 5× fit-time guideline. The rule says > 5× must be justified and > 20× is unacceptable; none is near 20×. An
exact speed-up of the anchored leave-out was written but **not validated**, so it is not in the submitted code (see
"Compute: unfinished").

This round completed an interrupted earlier attempt: code in `lib/agtloco.py`, checks (7a)/(7b) done on the old
machine. In this round I:
* re-validated the code;
* fixed two stale dev scripts;
* applied the **pre-registered compute fallback (7e)**, because its trigger was met;
* ran checks (7a)–(7e) here.

Code layout:
* `method.py` is the benchmark interface (unchanged) and calls `lib/agtloco.py` (`GTLocoHFD`, `GTLocoTree`).
* `lib/treehfd_mod/` is the unmodified package copy; its tree parsing and Cartesian main-effect partition are
  reused.
* Everything that is not new is S4's code, copied verbatim.

**Main outcome, in one paragraph.** The anchor gives a consistent held-out fidelity gain over S4:
* resid_out is 9–15% lower than S4 on all four diagnosed datasets, as a geo-mean over 8 splits;
* on the benchmark split, airfoil falls from 0.83× to 0.70× baseline;
* on the analytical case, resid_out falls from 0.78× to 0.75×.

The idea's **main target was not reached: resid_in is unchanged relative to S4** (0.98–1.01× over 8 splits), not
halved. The label-free risk R pushes ρ to the cap (ρ = 1 on ≥ 95% of trees in every fit) and κ to the bottom of its
grid. There is also a **real ortho regression on concrete** (ortho_in 1.6×, ortho_out 1.4× vs S4, 8-split geo-mean).
Details and all numbers are below.

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
  * Blocks are grouped by size and batched: exact sizes up to 8, then rounded up to a multiple of 8 and padded with
    zero-weight rows, which leaves the downdate exact.
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
* **finalize / predict** are S4's: there are no new columns, and unseen cells use zero or harmonic.
* **Diagnostics** (`state.diagnostics`):
  * per tree: chosen κ and ρ (`kappa_chosen`, `rho_chosen`), number of kept virtual atoms (`n_virtual`), `route_err`,
    and the per-tree argmins (`per_tree_idx`, `per_tree_joint`);
  * the selected candidate and its ρ mode;
  * `risk_over_var` and `resid_in_over_var` (both label-free, / Var[T(X_train)]);
  * the full candidate table.

## Hyper-parameters (all fixed a priori; nothing is per-dataset)
| name | value | source |
|---|---|---|
| RHOS | (0, 0.25, 1) | idea: fixed a priori, capped at 1 so the virtual mass never exceeds the empirical mass |
| ANCHOR_PRIORS | ("lattice",) | **pre-registered compute fallback (7e), applied** (see below). ridge keeps its ρ = 0 member |
| w_v | 1/(2\|S\| n) per copy | idea |
| shift kernel | ±1 adjacent non-empty union-grid bin, training median | S2 |
| copy acceptance | moves a tree-t main bin, all touched pair cells populated ("drop") | idea / S2 |
| BLOCK_CHUNK | 4e6 elements | memory bound of the batched block temporaries (no effect on results) |
| KAPPAS, PRIORS, RULES, EPS_MAIN, LATTICE_RIDGE, SHIFTS, SE_RULE, HARD_ORTHO | as S4 (1e-2…1e4; ridge, lattice; zero, harmonic; 1; 0.1; −5..+3; 0; True) | S4, unchanged |

## Deviations from the idea (and why)
1. **Compute fallback (7e) applied: ρ > 0 is fitted for the lattice prior only.**
   * The idea fixed the trigger in advance: "any full-mode fit_s > 4× baseline".
   * With both priors anchored, the per-tree fits alone took 87.2 s on powerplant. That is **7.2×** the baseline's
     12.1 s full-mode fit_s on this machine (`dev/time_parts.py powerplant`). The harness fit_s can only be larger.
   * Airfoil was 4.3× (14.1 s vs 3.3 s).
   * With the fallback there are 4 instead of 6 eigendecompositions per tree, and powerplant's per-tree fits took
     58.8 s (4.9×).
   * The eigendecompositions are ≥ 60% of the time, one per (prior, ρ) of dimension m_red ≈ 900 on powerplant. I
     found no exact way to share them across ρ: the κ path needs the spectrum of N + ρA, and (N + ρA + λI) is a
     two-parameter family.
   * I benchmarked the alternatives single-threaded at m = 900: LAPACK drivers evd 83 ms, evr 109 ms, ev 459 ms,
     float32 eigh 120 ms, tridiagonal reduction alone 46 ms. None is a usable win.
2. **Leave-out block item 3** (copies of other points shifted into a union bin whose only member is i) is an extra
   removal not listed in the idea. The union-bin median of a singleton bin is x_ij itself, so these copies do not
   exist without i.
3. **Identical copies are merged** (exact, see above). This is an implementation detail, not a modelling change.

## Pre-registered checks
* **(a) ρ = 0 reproduces S4.**
  * `dev/method_rho0.py` (RHOS = (0,)) through the subset harness on the old machine: every metric equals S4's
    `subset_selftest.json` to every digit (`dev/subset_rho0.json`).
  * On this machine, the ρ = 0 mode of the ablation equals S4's logged results (resid_in, resid_out, ortho_in,
    ortho_out) as strings on **all 32 dataset-splits** (4 datasets × 8 splits of `S4/dev/logs/diag_ortho_out_*.log`).
  * It also equals `S4/subset_env-22b414443084.json` on the benchmark split (airfoil 0.01235 / 0.02247, concrete
    0.00216 / 0.02104, abalone 0.00634 / 0.02094).
  *   * I did not re-run `dev/method_rho0.py` through the harness on this machine (analytical case included) for lack of time.
* **(b) Closed-form block leave-out vs brute-force refits.** `dev/check_loo.py` refits the tree on X₋ᵢ and rebuilds
  **everything** from X₋ᵢ: the partition, the union-bin medians, hence all virtual atoms. It prints
  mean-square(closed) / mean-square(brute):

  | tree, singletons | ρ = 0 (S4's approximation) | ρ = 0.25 | ρ = 1 |
  |---|---|---|---|
  | concrete tree 10, 12 orphans (old machine, both priors) | 0.965–1.009 | 0.976–1.018 | 0.967–1.018 |
  | concrete tree 10, 12 plain singletons | 0.982–1.002 | 0.981–1.002 | 0.978–1.001 |
  | airfoil tree 3, 12 orphans (lattice rows) | 0.849–1.063 | 0.838–1.094 | 0.832–1.098 |
  | airfoil tree 3, 12 plain singletons (lattice rows) | 0.999–1.025 | 0.999–1.023 | 0.997–1.013 |

  * The anchored closed form is as accurate as S4's closed form at ρ = 0. On airfoil orphans both are off by up to
    15–17% for `zero`/`ridge`; that is S4's known approximation (point i's role in the mean / orthogonality rows is
    ignored), not an effect of the anchor.
  * For the variant that is actually selected (lattice/harmonic), the ratios are 0.95–1.06 at ρ = 0 and 0.97–1.10 at
    ρ = 1. There is no optimism that would favour large ρ.
  * The brute-force refits themselves confirm the direction R reports on plain singletons: at κ = 0.01 the
    mean-square leave-out error is 0.0204 at ρ = 0, 0.0184 at ρ = 0.25 and 0.0157 at ρ = 1.
  * Logs: `dev/logs/check_loo_*.log`.
* **Virtual-atom consistency** (`dev/check_atoms.py`, concrete 8 trees and airfoil 5 trees):
  * 0 mismatches between a virtual target and the tree output of a real training point in the same joint cell
    (5,368 comparisons);
  * route error ≤ 4.4e-6;
  * 1.5–3.4 kept copies per training point per tree.
* **(c) Subset and full self-tests:** below.
* **(d) Fixed-ρ ablation** on S4's 8 diagnostic splits: below. Held-out numbers are for diagnosis only and were
  **not** used for any choice.
* **(e) Compute fallback:** applied (deviation 1). Resulting fit times are below.

## Self-test results (this machine, `22b414443084`)
### Subset (`subset_selftest.json`), ratios to the baseline `baseline/env-22b414443084/subset.json`
| | baseline | S4 (env file) | E1 | E1 / S4 |
|---|---|---|---|---|
| analytical resid_out | 0.01022 | 0.00800 (0.78) | 0.00768 (**0.75**) | 0.96 |
| analytical mse_eta12 / eta34 | 0.03387 / 0.03168 | 0.946 / 0.959 | 0.948 / 0.956 | 1.00 / 1.00 |
| analytical mse_others | 0.00215 | 0.745 | 0.746 | 1.00 |
| analytical mse_eta1..6 | | 0.94–1.00 | 0.92–1.01 | |
| analytical fit_s | 19.1 | 28.9 (1.51) | 66.6 (**3.49**) | |
| airfoil resid_in / resid_out | 0.00985 / 0.02695 | 1.255 / 0.834 | 1.223 / **0.703** | 0.97 / 0.84 |
| airfoil ortho_in / ortho_out | 0.0097 / 0.0989 | 0.868 / 1.128 | **1.486** / 0.950 | 1.71 / 0.84 |
| airfoil locvar_in / fit_s | 5.2e-5 / 3.3 s | 0.878 / 1.25 | 0.555 / 2.98 | |
| concrete resid_in / resid_out | 0.00153 / 0.02512 | 1.411 / 0.838 | **1.650** / 0.786 | 1.17 / 0.94 |
| concrete ortho_in / ortho_out | 0.0064 / 0.0822 | 0.964 / 0.872 | 1.091 / 0.848 | 1.13 / 0.97 |
| concrete locvar_in / fit_s | 0.0125 / 6.1 s | 1.058 / 1.14 | **1.479** / 2.37 | |
| abalone resid_in / resid_out | 0.00455 / 0.04034 | 1.392 / 0.519 | 1.377 / **0.479** | 0.99 / 0.92 |
| abalone ortho_in / ortho_out | 0.0268 / 0.0159 | 0.942 / 2.597 | 1.044 / **2.805** | 1.11 / 1.08 |
| abalone locvar_in / fit_s | 0.00256 / 21.9 s | 0.988 / 0.65 | 0.959 / 1.58 | |

### Full mode (`full_selftest.json`), ratios to `baseline/env-22b414443084/full.json`
| dataset | resid_in S4 / E1 | resid_out S4 / E1 | ortho_in S4 / E1 | ortho_out S4 / E1 | locvar_in S4 / E1 | fit_s base → E1 (S4, E1 ratio) |
|---|---|---|---|---|---|---|
| abalone | 1.392 / **1.377** | 0.511 / **0.471** | 0.942 / **1.044** | 2.646 / **2.857** | 0.988 / **0.959** | 17.3 → 32.8 (0.72, **1.90**) |
| airfoil | 1.255 / **1.223** | 0.835 / **0.704** | 0.868 / **1.486** | 1.152 / **0.969** | 0.878 / **0.555** | 3.9 → 9.6 (0.99, **2.44**) |
| bike | 1.005 / **1.005** | 0.998 / **0.998** | 1.000 / **1.000** | 1.000 / **1.000** | 0.946 / **0.946** | 46.4 → 335.4 (0.79, **7.23**) |
| housing | 1.109 / **1.125** | 0.910 / **0.900** | 1.013 / **1.011** | 0.989 / **0.973** | 0.750 / **0.721** | 99.2 → 274.4 (1.13, **2.77**) |
| concrete | 1.411 / **1.650** | 0.842 / **0.790** | 0.964 / **1.091** | 0.871 / **0.847** | 1.058 / **1.479** | 4.7 → 14.8 (1.34, **3.15**) |
| nutrition | 1.685 / **1.521** | 0.750 / **0.675** | 0.949 / **1.044** | 0.769 / **0.828** | 0.970 / **1.009** | 6.6 → 34.1 (1.68, **5.20**) |
| parkinson | 1.012 / **1.012** | 1.002 / **1.002** | 1.000 / **1.000** | 0.998 / **0.998** | 0.927 / **0.927** | 42.1 → 126.5 (0.94, **3.00**) |
| powerplant | 1.058 / **1.122** | 0.965 / **0.938** | n/a / **n/a** | n/a / **n/a** | 0.989 / **0.982** | 12.1 → 62.2 (2.47, **5.14**) |
| superconduct | 1.008 / **1.017** | 0.998 / **1.003** | n/a / **n/a** | n/a / **n/a** | 0.981 / **0.980** | 359.0 → 448.4 (0.47, **1.25**) |

| analytical (10 reps) | baseline | S4 | E1 |
|---|---|---|---|
| mse_eta1 | 0.028398 | 0.028238 (0.994) | 0.028405 (**1.000**) |
| mse_eta2 | 0.017909 | 0.017908 (1.000) | 0.017948 (**1.002**) |
| mse_eta3 | 0.016593 | 0.016465 (0.992) | 0.016567 (**0.998**) |
| mse_eta4 | 0.016181 | 0.016189 (1.001) | 0.016259 (**1.005**) |
| mse_eta5 | 0.00030061 | 0.0002894 (0.963) | 0.00028571 (**0.950**) |
| mse_eta6 | 0.00036652 | 0.0003626 (0.989) | 0.00035862 (**0.978**) |
| mse_eta12 | 0.03118 | 0.030258 (0.970) | 0.029905 (**0.959**) |
| mse_eta34 | 0.028205 | 0.027686 (0.982) | 0.027431 (**0.973**) |
| mse_others | 0.0021245 | 0.0016481 (0.776) | 0.0017024 (**0.801**) |
| resid_out | 0.01016 | 0.008371 (0.824) | 0.0080545 (**0.793**) |
| fit_s | 17.316 | 30.883 (1.784) | 59.998 (**3.465**) |

Official harness, full mode: **no errors**, wall time 2,008 s (`full_selftest.json` / `.log`; a copy is in
`dev/full_run1_unoptimised.json`). Other processes (dev ablations, profiling) ran during part of this run, so
fit_s may be inflated by roughly 10–20%. Summary:
* **resid_out vs S4:** better on abalone, airfoil, housing, concrete, nutrition and powerplant. Identical on bike and
  parkinson: the selection there reproduces S4's metrics to the printed digits. Superconduct is +0.5% (1.003×
  baseline vs S4 0.998×).
* **Analytical (10 reps):**
  * resid_out 0.79× baseline (S4 0.82×);
  * mse_eta12 / mse_eta34 0.959 / 0.973 (S4 0.970 / 0.982), i.e. a ~1% extra gain, not the hoped-for 5–6%;
  * mse_others 0.80× (S4 0.78×, slightly worse);
  * mse_eta1..6 within 0.95–1.005.
* **resid_in vs S4:** better on nutrition (1.52× vs 1.69× baseline), airfoil and abalone (slightly). Worse on
  concrete (1.65× vs 1.41×), powerplant (1.12× vs 1.06×), housing and superconduct (+1%).
* **Fit time: THREE DATASETS EXCEED 5× BASELINE:**
  * bike 7.2× (335 s vs 46 s);
  * nutrition 5.2×;
  * powerplant 5.1×.
  All others are 1.25–3.5×. See "Compute: unfinished" below.

## (d) Fixed-ρ ablation over S4's 8 diagnostic splits (`dev/ablate_rho.sh`, `dev/summarize_ablation.py`)
Split s is a `default_rng(s)` 80/20 permutation with the XGBoost model refit under the benchmark settings; split 0 is
the benchmark split. Splits 1–7 are used **only for diagnosis**, as in S4. Within each ρ mode, the candidate with the
smallest R is taken (the ρ = 0 mode is exactly S4's selection). Geo-mean ratio to the baseline over 8 splits (ortho
only over splits where an interaction passes the 1% threshold, n in brackets). Logs: `dev/logs/ablate_rho_*.log`;
summary: `dev/logs/ablate_rho_summary.log`.

| dataset | ρ mode | resid_in | resid_out | ortho_in | ortho_out |
|---|---|---|---|---|---|
| airfoil | ρ = 0 (S4) | 1.262 | 0.858 | 0.959 | 0.889 |
| | ρ = 0.25 | 1.228 | 0.808 | 0.774 | 0.979 |
| | ρ = 1 | 1.232 | 0.733 | 0.777 | 1.003 |
| | **selected** | 1.232 | **0.733** | 0.777 | 1.003 |
| concrete | ρ = 0 (S4) | 1.548 | 0.833 | 0.938 (7) | 0.962 (7) |
| | ρ = 0.25 | 1.437 | 0.782 | 1.258 (7) | 1.286 |
| | ρ = 1 | 1.563 | 0.750 | 1.500 (6) | 1.301 |
| | **selected** | 1.556 | **0.750** | 1.348 (6) | 1.302 |
| abalone | ρ = 0 (S4) | 1.390 | 0.536 | 0.716 | 1.107 |
| | ρ = 0.25 | 1.193 | 0.536 | 0.771 | 1.086 |
| | ρ = 1 | 1.369 | 0.488 | 0.744 | 1.054 |
| | **selected** | 1.369 | **0.488** | 0.744 | 1.054 |
| nutrition | ρ = 0 (S4) | 1.456 | 0.811 | 1.010 | 1.001 |
| | ρ = 0.25 | 1.285 | 0.797 | 1.033 | 1.040 |
| | ρ = 1 | 1.446 | 0.740 | 1.054 | 1.045 |
| | **selected** | 1.444 | **0.742** | 1.053 | 1.048 |

Selected / S4 (ρ = 0 member), geo-mean over the 8 splits:

| | resid_in | resid_out | ortho_in | ortho_out | locvar_in |
|---|---|---|---|---|---|
| airfoil | 0.976 | **0.854** | 0.810 | 1.127 | 0.651 |
| concrete | 1.007 | **0.901** | **1.632** (6) | **1.394** (7) | 1.168 |
| abalone | 0.985 | **0.912** | 1.039 | 0.952 | 1.112 |
| nutrition | 0.992 | **0.914** | 1.043 | 1.046 | 1.050 |

What the selection does, on all 32 fits:
* **ρ saturates.** ρ = 1 is chosen on ≥ 95 of 100 trees in every fit (100/100 in 25 of the 32). The ensemble mode is
  "common ρ = 1" or "per-tree ρ", which are then nearly identical.
* **κ moves to the floor of its grid.** The chosen per-tree κ has median 0.01 in most fits, and the ensemble shift is
  −1…−5; on abalone it is often a common κ = 0.01. The anchor **replaces** the κ shrinkage instead of adding to it.
  The selection sits at a corner of the pre-registered (ρ, κ) grid. I did not widen either grid:
  * ρ ≤ 1 is the idea's principled cap;
  * κ < 0.01 is where S4 found the closed-form leave-out becoming optimistic.
* **R's ordering of ρ agrees with held-out residuals.** R/Var falls strictly in ρ (0 > 0.25 > 1) in all 32 fits.
  Held-out resid_out at ρ = 1 is below ρ = 0 in 31 of 32 (the exception is one concrete split). For example,
  airfoil split 0 goes 0.0225 → 0.0210 → 0.0190 at ρ = 0 → 0.25 → 1.
* **ρ = 0.25 would have been the resid_in-friendlier choice.** It has lower resid_in than both ρ = 0 and ρ = 1 on
  concrete, abalone and nutrition (e.g. abalone 1.19× vs 1.37× baseline) and keeps most of the resid_out gain on
  airfoil and nutrition. But R is a fidelity criterion and prefers ρ = 1. I did **not** change the selection rule to
  favour it: that would be tuning against held-out numbers.

## Trade-offs and caveats (reported, not hidden)
* **The main target, resid_in, was not reached.** The idea expected the resid_in excess over the baseline to roughly
  halve relative to S4. Instead it is unchanged (selected / S4 = 0.98–1.01 over 8 splits).
  * On the concrete benchmark split it is worse than S4 (1.65× vs 1.41× baseline).
  * Mechanism: R takes the anchor's out-of-sample gain and spends the freed-up bias budget on an even smaller κ. With
    ρ = 1 the virtual rows carry as much weight as the real rows, and they also pull thin cells away from the
    real-point values.
* **resid_out is the real gain.** It is 9–15% better than S4 on four datasets over 8 splits, and 0.70× baseline on
  the airfoil benchmark split (S4 0.83×; S2 ≈ 0.76× there). It is 0.75× on the analytical case (S4 0.78×).
* **The true interactions do not improve.** mse_eta12 / mse_eta34 are the same as S4 (0.95–0.96× baseline), not the
  hoped-for −5…−6% beyond S4. mse_others is unchanged (0.75×): the lattice and ridge still suppress spurious cells.
* **Orthogonality.**
  * Within-tree orthogonality stays exact.
  * Cross-tree orthogonality is worse on **concrete** (ortho_in 1.63×, ortho_out 1.39× vs S4 over 8 splits). This
    is risk (1) of the idea, materialised on concrete rather than airfoil.
  * On airfoil, ortho_in is better than S4 over 8 splits (0.81×), but **worse on the benchmark split** (0.0144 vs
    0.0084, 1.49× baseline).
  * Abalone and nutrition are within ±5% of S4.
  * The benchmark-split abalone ortho_out stays far above the baseline (2.8×, S4 2.6×). S4 showed this is split
    noise (8-split geo-mean ≈ 1.05–1.1).
* **locvar_in.** Concrete is 1.48× baseline on the benchmark split (S4 1.06×; 1.17× vs S4 over 8 splits). The
  anchored thin cells give noisier main effects there, like S2's concrete, though smaller. Airfoil is smoother
  (0.56×).
* **Compute.**
  * With the fallback, fit_s is 1.6–3.5× baseline on the subset (analytical 3.49×, the largest) and 1.25–7.2× in full mode (bike 7.2×, nutrition 5.2×, powerplant 5.1× exceed 5×)
    in full mode.
  * The cost is inherent to the design: 2 extra eigendecompositions of dimension m_red per tree, plus the block
    downdates.
  * All timings are on a hybrid-core CPU (i9-14900HX) with single-threaded BLAS (`OPENBLAS_NUM_THREADS=1`, set by
    the pipeline). Run-to-run variation of ±10–20% is plausible.
* **Held-out numbers seen during development.** The dev scripts print held-out residuals next to R. No choice in this
  round was made from them. The only change this round, the fallback, was triggered by a pre-registered compute rule.

## Compute: unfinished (state at the end of this round)
* **Where the time goes.**
  * On bike, the block leave-out takes 263 of 351 s in `dev/time_parts.py bike` (`dev/logs/time_parts_bike.log`).
    Bike has ~5,000 Good-Turing singletons per tree, and their blocks have mean size 7.6 (max 257), versus the
    "usually 0–2" copies the idea assumed.
  * K = Z_B Λ⁻¹ Z_B' costs ~b² × m_red × 12 per block.
  * On nutrition, the leave-out and the eigendecompositions take about 13 s each. On powerplant, the 4
    eigendecompositions per tree take ~35 s.
* **Justification for > 5×.** The extra cost is the honest block leave-out that the selection needs, plus one more
  eigendecomposition per ρ. Bike and parkinson pay it without any metric change (their selection ends up equal to
  S4's).
* **Exact speed-ups found but not validated end to end.** The submitted `lib/agtloco.py` does NOT contain them;
  the patched file is `dev/agtloco_speedup_untested.py.txt`:
  * cache-sized chunks: BLOCK_CHUNK 4e6 → 3e5. Measured 1.6× on the bike leave-out (8.7 → 5.3 s on 3 heavy trees),
    with outputs equal to 1.8e-15;
  * exact size buckets up to 32 instead of 8: padding currently adds 54% to Σb²;
  * a hashed row-unique in `_virtual_atoms`: np.unique(axis=0) is 40% of the 0.3 s per tree there;
  * a rank-one fast path.

  The comparison run of that patch was interrupted by the session end, so its equality of outputs and its end-to-end
  timing are unverified.
* **Not done.** A second harness run with the speed-ups. `dev/method_rho0.py` through the harness on this machine
  (check 7a is covered by the 32 dataset-splits instead).

## Files
* `method.py`: benchmark interface.
* `lib/agtloco.py`: the method.
* `lib/treehfd_mod/`: unmodified package copy.
* `subset_selftest.json` / `.log`, `full_selftest.json` / `.log`: harness outputs.
* `dev/`:
  * `dev_e1.py`: selection diagnostics and the ρ-mode ablation (`ablate`).
  * `ablate_rho.sh`, `summarize_ablation.py`: check (7d).
  * `check_loo.py`: check (7b). `check_atoms.py`: virtual-atom consistency (fixed this round for merged rows).
  * `method_rho0.py`, `subset_rho0.json`: check (7a).
  * `time_parts.py` (option `anchor_priors=`, added this round), `time_anchor.py`, `profile_fit.py`,
    `profile_trees.py`, `bench_eigh.py`: compute diagnostics.
  * `common.py`, `compare.py`, `run_log.py`, `full_table.py`: helpers.
  * `bench_anchor.py`, `time_anchor.py`: leave-out benchmarks.
  * `agtloco_before_speedup.py.txt`: identical to the submitted `lib/agtloco.py`.
  * `agtloco_speedup_untested.py.txt`: the unvalidated speed-up.
  * `logs/`: check and ablation logs.
