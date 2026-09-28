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

