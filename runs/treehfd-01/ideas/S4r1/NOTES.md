# S4 — GT-LOCO TreeHFD (Good-Turing-calibrated leave-one-cell-out shrinkage)

## Status
Implemented. The subset self-test and a full-mode run (analytical case with 10 reps plus all 9 datasets) both
complete with no errors. Round 2 finished and partly redid an earlier interrupted attempt (commit `ca6d096`).
Round 3 verified the ortho_out regression, ablated the ensemble step and added label-free trade-off reporting;
its decomposition was unchanged.
**Round 4 (this directory, `S4r1`; section below):**
* changes the ensemble κ step: it adds an orthogonality term to its risk and caps the in-sample residual;
* adds the attribution runs the critic asked for (`attribution/`), which correct the earlier attribution;
* re-runs the full self-test on this machine (`full_selftest.json`, env 22b414443084): no errors.
  Compared with the official round-3 run, only abalone and nutrition change; every other metric except fit_s is
  identical.
* **Unfinished:**
  * idle-machine fit times for bike, parkinson and superconduct (the session ended during that run);
  * the κ → 0 check output was not saved to a log;
  * the Ω evaluation count was not checked on four of the nine datasets.

Code: `method.py` (interface unchanged) → `lib/gtloco.py` (`GTLocoHFD`, `GTLocoTree`). It reuses the baseline's
tree parsing and Cartesian main-effect partition from `lib/treehfd_mod/` without changes. Development and
diagnostic scripts are in `dev/`; round-4 logs are in `dev/logs/`. The round-3 logs stayed in the
original `S4/` directory and were not copied here.

## Round 4 (S4r1): response to the critic
### What changed in the method, and why
Both changes are in the ensemble step (`GTLocoHFD.fit`). Per-tree fitting, priors, rules, κ grid and exact
orthogonality are unchanged.
1. **Orthogonality term in the ensemble risk** (`ORTHO_TERM = True`; critic points 3 and 4). A candidate's risk is
   now R + Ω, where Ω is measured on X_train (label-free). For each interaction η_jk, Ω adds the variance of η_jk
   explained by the linear span of its two main effects (η_j, η_k). Summed over interactions, this is the
   variance a hierarchically orthogonal explanation would have to move from interactions into main effects.
   It has the same units as R, so there is no weight to tune.
   * The argmin is exact but cheap. Since Ω ≥ 0, candidates are visited in order of increasing R, and Ω is
     computed until R alone exceeds the best R + Ω found. It took 1–4 evaluations on the five real datasets
     checked (abalone, airfoil, concrete, nutrition, housing; `dev/quick_sel.py`) and 16 on analytical rep 0, at
     about 0.3 s each there. The count on bike, parkinson, powerplant and superconduct was not checked.
   * New code: `GTLocoTree.components_train` and `GTLocoHFD._omega`.
2. **resid_in cap** (`RESID_IN_CAP = 1.5`; critic points 3 and 4). A candidate is admissible only if its in-sample
   residual, mean((T − recon)²) on X_train, is ≤ 1.5 × the smallest over all candidates.
   * The smallest is the least-regularised fit (κ = 0.01). On the benchmark split its resid_in is 1.01–1.04×
     the baseline's on airfoil, concrete, nutrition and abalone (`dev/cache/cand_*_s0.npz`).
   * So the in-sample fidelity lost to shrinkage is bounded a priori, at about 1.5× baseline resid_in.
   * 1.5 was fixed before the full run, as a round number between "no cap" and the 1.25 that the diagnostic
     splits show costs 2–6% resid_out. It was not tuned on the benchmark split.
3. New diagnostics in `state.diagnostics`: `omega_over_var` (Ω of the choice / Var[T(X_train)]),
   `omega_evaluated`, `n_admissible` and `cap_binding`. `min_risk` is still the unconstrained argmin of R.
4. `KEEP_PATH` additionally keeps the per-candidate squared leave-out residuals (development only).

Not changed, and checked:
* **LATTICE_RIDGE ∈ {0.01, 0.1, 1}.** On diagnostic splits 1–7 of four datasets, every metric of the selected
  candidate moves by ≤ 3%, with no consistent direction, and R agrees (`dev/logs/analyse_rho.log`). It stays
  at 0.1.
* **A one-SE rule towards parsimony** (most shrinkage within one SE of the minimum R) was tried and dropped.
  The per-point leave-out residuals r/(1−h) are heavy-tailed, so the paired SE admitted even κ = 1e4.

### Evidence for the ensemble-step changes (diagnostic splits only; `dev/logs/analyse_cands.log`)
`dev/cand_table.py` fits GT-LOCO once per split and evaluates every ensemble candidate.
* Splits 1–7 are alternative 80/20 splits with a refitted XGBoost model. They are used for development only;
  split 0 is the benchmark split.
* Rules are compared by geometric-mean ratio to the round-3 rule (argmin R).

| rule (splits 1–7) | abalone in / out / o_in / o_out | airfoil | concrete | nutrition |
|---|---|---|---|---|
| R + Ω | 1.000 / 1.000 / 1.000 / 1.000 | all 1.000 | all 1.000 | all 1.000 |
| cap 1.5 | 0.947 / 1.020 / 1.045 / 0.997 | all 1.000 | 0.905 / 1.021 / 1.003 / 1.003 | 0.918 / 1.011 / 0.991 / 0.972 |
| cap 1.5 + Ω (**adopted**) | 0.947 / 1.020 / 1.045 / 0.997 | all 1.000 | 0.905 / 1.021 / 1.003 / 1.003 | 0.918 / 1.011 / 0.991 / 0.972 |
| cap 1.25 | 0.880 / 1.055 / 1.011 / 1.003 | 0.929 / 1.020 / 1.008 / 1.016 | 0.749 / 1.055 / 0.966 / 0.994 | 0.822 / 1.045 / 0.989 / 0.976 |
| per-tree only (no ensemble step) | 3.17 / 0.958 / 0.703 / 0.781 | 1.27 / 1.011 / 0.956 / 0.969 | 2.37 / 0.990 / 0.892 / 1.021 | 2.48 / 0.942 / 0.984 / 0.842 |

(in = resid_in, out = resid_out, o_in / o_out = ortho_in / ortho_out.)

What this shows:
* **The orthogonality term is real but tiny.**
  * On real data Ω/Var[T] is 2e-5 to 2e-4 (analytical ≈ 1e-3), while R/Var[T] is 0.013–0.14.
  * In-sample max |corr| is 0.006–0.08, so the variance explained by the main effects is ≈ corr² · Var(η_jk).
  * As a result, Ω only breaks near-ties of R. It changed no choice on the 28 diagnostic splits.
  * On the benchmark split it changes one: abalone moves from lattice/harmonic to ridge/harmonic, with the same
    shift −3. The two R values differ by less than Ω.
  * Consequence: abalone resid_out 0.0209 → 0.0219 and resid_in 0.00634 → 0.00596. ortho_in (0.0252) and
    ortho_out (0.041 → 0.042) are essentially unchanged.
  * It is kept because the critic asked for it, it is principled (HFD requires the interactions to be orthogonal
    to their main effects), and it is neutral on the diagnostic splits. It does **not** fix ortho_out.
* **The cap is a pure trade, as expected.** It gives 5–10% lower resid_in for 1–2% higher resid_out on three of
  the four datasets, and it never binds on airfoil.
  * On the benchmark split it binds only on nutrition: shift −2 → −3, resid_in 0.0301 → 0.0236 (+69% → +32%
    vs baseline), resid_out 0.0725 → 0.0775 (−25.0% → −19.8%), ortho_out 0.0817 → 0.0802.
* **ortho_out cannot be fixed from the ensemble step with a label-free criterion.**
  * More shrinkage (per-tree only) gives better ortho_out on the diagnostic splits: abalone 0.78×, nutrition
    0.84×.
  * On the abalone benchmark split, however, the same configuration is 1.28× worse (0.0525 vs 0.041).
  * The test-set noise floor of a single correlation is 1/√n_test: abalone 0.035 (836 points), airfoil 0.058
    (301). The baseline's 0.016 on abalone is its lowest over 8 splits (range 0.016–0.114, round 3). The
    method's 0.042 is inside that noise band.
  * None of the rules above tracks ortho_out on split 0 better than on the other splits. ortho_out on abalone
    and airfoil is therefore reported as worse, not "fixed".
* **The NA concrete ortho_in in the no-ensemble-step ablation is not a numerical failure.** With per-tree κ
  only, no interaction reaches the metric's 1%-of-Var[T] threshold, so the harness returns null by
  definition.

### Attribution (critic point 1; `attribution/README.md` and `attribution/full_table.md`)
Each full-mode run changes one thing relative to the baseline, or to the previous row. resid_out change vs
baseline:

| | analytical | abalone | airfoil | concrete | nutrition | housing |
|---|---|---|---|---|---|---|
| deterministic nearest fallback (lsqr) | +0.1% | −0.4% | −0.3% | +0.1% | −0.3% | 0.0% |
| harmonic fill of unseen cells (lsqr coefficients) | +0.6% | +0.6% | −3.2% | −2.6% | +1.3% | +0.6% |
| exact solve of the baseline system (**solver alone**) | +0.2% | **+81.4%** | −0.3% | −0.1% | +0.6% | −0.2% |
| GT-LOCO, κ = 1e-4, ridge / harmonic | −1.2% | +12.9% | −3.2% | −2.7% | +1.3% | +0.6% |
| GT-LOCO, κ = 1e-4, lattice / harmonic | −2.3% | −2.0% | **−13.1%** | −0.6% | +3.5% | +0.8% |
| κ = 0.01 fixed (the earlier "no_shrinkage" ablation) | −10.2% | −40.5% | −14.2% | −8.3% | −11.2% | −3.3% |
| S4 (round 3, official) | −17.6% | −48.9% | −16.5% | −15.8% | −25.0% | −9.0% |
| S4r1 (this round, self-test) | −17.6% | −46.5% | −16.5% | −15.8% | −19.8% | −9.0% |

How to credit the gains:
* **The solver and the deterministic fallback are not where the gain comes from.**
  * Determinism changes nothing measurable.
  * The exact solver changes nothing on eight datasets and makes abalone much worse: lsqr's early stop was
    implicitly regularising weakly identified directions there, and the exact solution loses that.
  * Direct solver + exact orthogonality + deterministic harmonic rule, without effective shrinkage (κ = 1e-4),
    stays at baseline level.
  * The one exception is airfoil, where the lattice prior as a pure tie-breaker gives −13%.
* **The earlier "no_shrinkage_direct_solver_only" ablation (κ = 0.01) was mislabelled.**
  * κ = 0.01 pseudo-counts already strongly shrinks weakly identified directions.
  * On the benchmark split, lattice/harmonic abalone resid_out is 0.0401 at κ = 1e-4, 0.0300 at 1e-3 and
    0.0244 at 1e-2 (`dev/kappa_limit.py`). Its output was printed in the session and not saved; re-run
    the script to regenerate it.
  * The critic's attribution was based on that ablation. The runs above do not support it.
* **The gain comes from the prior's shrinkage.** On top of a fixed κ = 0.01, the GT/PRESS-selected per-tree κ
  plus the ensemble step add roughly the other half on analytical, concrete, nutrition and housing, and less
  on abalone and airfoil:
  * analytical resid_out −10% → −18%, mse_others −11% → −22%;
  * concrete −8% → −16%, nutrition −11% → −20/−25%, housing −3% → −9%, abalone −40% → −47/−49%.
* This is the mechanism the idea describes. The "exact-solver" contribution is ≈ 0 and should not be credited
  as a separate gain.

### Critic point 5: the "soft orthogonality rows" column is inverted
The submitted method imposes hierarchical orthogonality **exactly**: interaction coefficients live in the null
space of each pair's orthogonality rows (`HARD_ORTHO = True`). The ablation column "w/o_soft_orthogonality_rows"
is the method **with the baseline's soft rows instead** (`HARD_ORTHO = False`), not the method without them.
* That column shows that exact orthogonality is what keeps ortho_in down: soft rows give concrete ortho_in
  +120% and nutrition +82%. It also shows that soft rows double the fit time.
* The critic's "keep it" therefore means keeping exact orthogonality, which is done.
* It is an engineering addition that is not in the idea's stated mechanism (see "Deviations from the idea").

### Critic point 4: regressions, updated for this round (full self-test vs baseline, this machine)
* **resid_in:** abalone +31% (was +39%), airfoil +26%, concrete +41%, nutrition +32% (was +69%), housing
  +11%; +0.5–6% elsewhere. The cap now guarantees resid_in ≤ 1.5 × the least-regularised fit. Concrete's
  +41% is within the cap because its least-regularised fit is itself 1.02× the baseline's.
* **ortho_out:** worse on abalone (0.0156 → 0.0422, +171%) and airfoil (+15%); see above for why this is at the
  test-set noise level and not fixable by the ensemble step. Better on concrete (−13%) and nutrition (−25%).
* **No gain on bike, parkinson or superconduct** (resid_out within ±0.2%). These three models reconstruct well
  already (resid_out 0.009–0.025). The shrinkage R selects there is mild (median κ ≈ 0.03–0.1), and
  none of the attribution rows moves them by more than 1.3%. Their residual is order-≥3 content of the trees
  in seen cells, which no order-2 decomposition can represent. This is the "modest gains" risk the idea
  named.
* **fit_s:** see the timing table below. The pipeline runs with `OPENBLAS_NUM_THREADS=1`, and two dense
  eigendecompositions per tree (one per prior, m ≈ 1000 on the analytical case and powerplant) take ≈ 57% of
  the fit.
  * Dropping the ridge prior would save ≈ 35% (half of the eigendecompositions and block products). It was
    not done, because the ridge prior is the idea's primary prior and is chosen on abalone, parkinson and
    powerplant.
  * The fit_s values inside `full_selftest.json` are **not** usable: up to nine harness processes were
    running at the same time.

### Full self-test, round 4 (`full_selftest.json`; this machine, 22b414443084; ratio to `baseline/env-22b414443084/full.json`)
The harness does not write the "eval_env" field; the pipeline adds it to official files. No errors.

| dataset | resid_in | resid_out | ortho_in | ortho_out | locvar_in |
|---|---|---|---|---|---|
| abalone | 0.005955 (1.31×) | 0.02192 (0.54×) | 0.02526 (0.94×) | 0.04216 (2.71×, worse) | 0.002533 (0.99×) |
| airfoil | 0.01235 (1.25×) | 0.02247 (0.84×) | 0.008396 (0.87×) | 0.1116 (1.15×, worse) | 4.567e-05 (0.88×) |
| bike | 0.02144 (1.00×) | 0.0246 (1.00×) | 0.01338 (1.00×) | 0.02121 (1.00×) | 0.0003258 (0.95×) |
| housing | 0.01112 (1.11×) | 0.01406 (0.91×) | 0.02834 (1.01×) | 0.04829 (0.99×) | 0.001347 (0.75×) |
| concrete | 0.002164 (1.41×) | 0.02104 (0.84×) | 0.006129 (0.96×) | 0.07164 (0.87×) | 0.01321 (1.06×) |
| nutrition | 0.0236 (1.32×) | 0.07752 (0.80×) | 0.01496 (0.92×) | 0.0802 (0.76×) | 0.002812 (1.00×) |
| parkinson | 0.007291 (1.01×) | 0.009561 (1.00×) | 0.08204 (1.00×) | 0.1174 (1.00×) | 0.004348 (0.93×) |
| powerplant | 0.0006673 (1.06×) | 0.001065 (0.96×) | n/a | n/a | 0.0005232 (0.99×) |
| superconduct | 0.01018 (1.01×) | 0.00909 (1.00×) | n/a | n/a | 0.0006501 (0.98×) |

Analytical (10 reps), identical to the official round-3 run:
* resid_out 0.824×, mse_others 0.776×;
* mse_eta12 0.970×, mse_eta34 0.982×;
* mse_eta1..6 0.963–1.001×.

Only abalone (the Ω near-tie) and nutrition (the cap) differ from the official round-3 run
(`../S4/full_env-22b414443084.json`).

### Fit time on an idle machine (`dev/time_pair.py`, `dev/logs/time_pair_idle.log`)
The baseline and the method are timed back to back, each fit in a fresh process through the harness's
`runner.py`, with the same model and X_train. One run each.

| dataset | baseline fit_s | GT-LOCO (round 4) fit_s | ratio |
|---|---|---|---|
| analytical (rep 0) | 19.7 | 34.8 | 1.77 |
| abalone | 17.4 | 13.3 | 0.77 |
| airfoil | 3.1 | 3.7 | 1.19 |
| concrete | 5.7 | 6.8 | 1.19 |
| nutrition | 8.1 | 10.9 | 1.35 |
| powerplant | 13.3 | 31.6 | **2.37** |
| housing | 115.8 | 114.1 | 0.99 |
| bike, parkinson, superconduct | – | – | **not measured** |

**Unfinished:** the timing run was cut off by the end of the session before bike, parkinson and superconduct.
In the official round-3 run, which had the same per-tree code but no Ω or cap, they took 0.79×, 0.94× and
0.47× the baseline. Round 4 adds Ω evaluations, measured at about 0.3 s each on the analytical case; their
cost on superconduct, which has many interactions, was not measured. All measured ratios are within the 5×
limit, and only powerplant and the analytical case exceed 1.5×.

## Round 3: response to the critic
The critic asked for three things:
* (a) check with several splits or seeds whether the abalone ortho_out regression (+173%) is real and specific
  to the harmonic rule;
* (b) fix or bound the ortho_out / resid_in trade-off;
* (c) ablate the ensemble-level κ shift.

Protocol for all checks below:
* "Split s" is an 80/20 permutation with `default_rng(s)`, and the XGBoost model is refit with the benchmark
  settings. Split 0 is the benchmark split; splits 1–7 exist **only for diagnosis**. The method never sees them.
* Held-out numbers were used to decide *whether to change* the method; they are reported here in full.

### Code changes (outputs unchanged)
1. `GTLocoTree.fit` now also returns the in-sample residuals along the κ path.
   * The ensemble step accumulates them per training point, as it already did for the leave-out residuals.
   * Every ensemble candidate therefore gets a label-free in-sample fidelity, mean((T − recon)²) on X_train,
     stored as the 5th field of `diagnostics["candidates"]`.
   * New diagnostics: `risk_over_var` (R of the chosen candidate), `resid_in_over_var` (its in-sample residual)
     and `min_risk`. The first two are divided by Var[T(X_train)].
   * `resid_in_over_var` is exactly the harness's `resid_in`: concrete gives 0.00216 in both. The in-sample cost
     of the shrinkage is thus reported by the method itself, from X_train and the model only, on any new dataset.
2. New option `SE_RULE`: a k-standard-error rule towards in-sample fidelity in the ensemble step. It was tested
   with k = 1 and left **off** (k = 0 is exactly the previous argmin); see below.
3. Verification:
   * `dev/compare.py subset_selftest.json subset_0.json` gives ratio 1.000 on every metric, and fit_s is
     within 1%.
   * `full_selftest.json` was re-run on an idle machine: no errors, same metrics as the round-2 full run, and
     fit_s from 0.16× (superconduct) to 1.53× (powerplant) baseline.
4. New dev scripts: `diag_ortho_out.py` (baseline vs GT-LOCO over splits, test-set bootstrap, rule swap at fixed
   κ), `ablate_shift.py` and `compare_se.py`. `check_loo.py` and `profile_trees.py` were updated for the new
   return signature.
   * Caveat: the first `compare_se.py` runs imported an earlier `ablate_shift.py` that had no `__main__` guard.
     So `dev/logs/compare_se1_*.log` begin with a copy of the ablation run under `SE_RULE = 1`. The comparison
     itself follows.

### (a) abalone ortho_out is split noise, not a harmonic-rule defect
Ratio of ortho_out, GT-LOCO / baseline, per split (`dev/logs/diag_ortho_out_*.log`):

| dataset | split 0 (bench) | splits 1–7 | geo-mean (all 8) | geo-mean (splits 1–7) |
|---|---|---|---|---|
| abalone | **2.61** | 0.96 1.07 0.94 1.04 0.99 0.86 1.01 | 1.11 | **0.98** |
| airfoil | 1.14 | 0.64 1.15 0.95 0.97 0.92 0.90 0.62 | 0.89 | – |
| concrete | 0.87 | 1.20 n/a 1.05 0.65 0.98 1.05 1.05 | 0.96 (7 splits) | – |
| nutrition | 0.71 | 1.17 1.02 0.95 0.99 1.09 1.07 1.06 | 1.00 | – |

(n/a: no interaction reached the 1% variance threshold.)

* **Test-set bootstrap on the benchmark split** (300 resamples of the 836 abalone test points):
  * ortho_out is 0.072 ± 0.044 for the baseline and 0.069 ± 0.048 for GT-LOCO, with P(GT-LOCO > baseline) = 0.54.
    Bootstrap means exceed the point estimates because a max of |corr| is biased upward under resampling.
  * The spread of the metric (sd ≈ 0.04) is larger than the observed gap (0.041 − 0.016).
  * The interactions that pass the threshold carry 1.3–3% of Var[T]. One correlation at n = 836 has a standard
    error of about 0.035.
  * The baseline's 0.016 is its own lowest value over the 8 splits (range 0.016–0.114).
* **The harmonic hypothesis is not supported.** With the same per-tree κ and only the rule swapped:
  * On the benchmark split the `zero` rule gives *higher* ortho_out: ridge/zero 0.050 and lattice/zero 0.050,
    versus lattice/harmonic 0.041.
  * On abalone splits 1–7, the zero rule at the same κ has a ratio to baseline of 1.00 (geo-mean).
  * On airfoil split 0, lattice/zero gives 0.113 and lattice/harmonic 0.112.
  * All GT-LOCO variants sit near 0.04–0.05 on this split, so the gap does not come from the unseen-cell rule.
* **Decision:** no orthogonality penalty or rule bias was added, because it would target a defect that 32
  split-level comparisons do not show. The benchmark-split number stays as it is and is reported as a worse
  number.

### (c) Ablation of the ensemble κ step
Ratio of the selected configuration to the per-tree argmin only (shift 0, variant by ensemble risk). Geo-mean over
8 splits (`dev/logs/ablate_shift_*.log`):

| dataset | resid_in | resid_out | ortho_in | ortho_out |
|---|---|---|---|---|
| abalone | **0.32** | 1.035 | 1.52 | 1.20 |
| airfoil | 0.79 | 0.98 | 1.06 | 1.02 |
| concrete | **0.42** | 1.00 | 1.12 (3 splits) | 1.00 (4 splits) |
| nutrition | **0.41** | 1.07 | 1.02 | 1.22 |

What the ensemble step does:
* **It does not buy resid_out** (0.98–1.07). **It buys in-sample fidelity.** Without it, resid_in would be
  2.5–3× higher: per-tree-only resid_in is 0.018–0.024 on abalone (about 4.5× baseline) and 0.0059 on concrete
  (3.8× baseline).
* **The critic's worry is partly confirmed.** On nutrition and abalone, the shift costs 3.5–7% resid_out and
  about 20% ortho_out relative to the per-tree choice. R prefers the shift more strongly than held-out data
  support.
* **R is a relative criterion, not a calibrated level.** R/Var divided by held-out resid_out is ≈ 1.0 on
  concrete, 1.2 on airfoil, 1.9 on nutrition and 2.2 on abalone. Read `risk_over_var` accordingly.
* **The step is kept.** Dropping it would trade a ≤ 7% resid_out gain for a 2.5–3× resid_in loss. It is the part
  of the method that keeps the resid_in cost bounded.

### (b) Trying to reduce the resid_in cost: one-SE rule (tested, not adopted)
With `SE_RULE = 1`, the ensemble step takes the best in-sample candidate among those whose R is within one paired
standard error of the minimum. Ratio to the argmin, geo-mean over 8 splits (`dev/logs/compare_se1_*.log`):

| | concrete | airfoil | nutrition | abalone |
|---|---|---|---|---|
| resid_in | 0.85 | 0.99 | 0.84 | 0.89 |
| resid_out | 1.025 | 1.00 | 1.053 | 1.060 |
| ortho_out | 0.995 | 1.06 | 0.965 | 0.994 |

This is a pure trade that dominates nothing, and it gives back part of the idea's main gain (resid_out), so
it is off. This is a "do not change" decision; it was made with held-out numbers from the diagnostic splits.

### (b) How large the resid_in cost is, and how to see it
* Over the 9 full-mode datasets and 8 splits each of abalone, airfoil, concrete and nutrition, resid_in divided
  by the baseline's ranges from 1.00 to 2.11. The worst case is concrete split 4: 0.0041 vs 0.0020.
* The largest absolute value is 0.030 of Var[T] (nutrition, benchmark split; R² of the model = 0.29).
* resid_in stays below the method's own resid_out in every run except superconduct. There it is 0.0102 vs
  0.0091, and the baseline shows the same pattern (0.0101 vs 0.0091).
* It rises only where resid_out falls. Where R selects mild shrinkage (bike, parkinson, superconduct,
  powerplant), resid_in stays within 1.00–1.06× baseline.
* A user can read `diagnostics["resid_in_over_var"]` next to `risk_over_var` after fitting, without labels or
  test data, so the cost cannot grow unnoticed on a dataset outside this benchmark.

## What the method does
For every tree t (100 per model), with y = tree output minus its mean on X_train:

1. **Same TreeHFD least-squares problem, built vectorised.** The baseline partition code builds the main bins.
   Pair cells are the observed (bin_j, bin_k) cells, found with `np.unique`. The normal matrix is
   H'H/n + M M', where H is the n×m point-to-cell incidence matrix and M holds the zero-mean rows. This has the
   same minimiser as the baseline rows weighted sqrt(n·n_c), scaled by 1/n².
2. **Exact hierarchical orthogonality.** Each pair's coefficients are parameterised in the null space Q_k of its
   discretised orthogonality rows, so that E[η_jk | bin of x_j] = E[η_jk | bin of x_k] = 0 holds exactly. The
   baseline instead uses soft rows. A pair whose null space is empty is pruned automatically (η_jk ≡ 0).
   Reason: with a penalty, soft rows get traded off against the shrinkage. In the diagnostic
   `dev/diag_ortho.py abalone ortho=soft`, the soft version at κ = 1 reaches in-sample interaction/main
   correlations of 0.06–0.13, versus ≤ 0.012 with hard constraints.
3. **Tikhonov penalty λ·β'Pβ, where λ = κ/n (κ in pseudo-count units).** There are two priors:
   * `ridge`: P = I on pair cells and EPS_MAIN·I on main bins. The data weight of a cell is proportional to its
     count, so thinly supported cells are shrunk much more than populated ones.
   * `lattice`: a Gaussian Markov random field on the **whole** (j,k) bin lattice, including never-observed cells
     as latent columns. It penalises first differences between 4-neighbour cells with weights 1/(distance
     between bin centres). Weights are normalised to mean 1 per tree, plus LATTICE_RIDGE·I. The latent cells
     are eliminated in closed form: the prior on observed cells is the Schur complement
     S = K_OO − K_OU K_UU⁻¹ K_UO, and the posterior mean of the latent cells is the harmonic extension
     β_U = E β_O with E = −K_UU⁻¹ K_UO. Main bins keep the ridge.
4. **Direct solver (no lsqr).** With β = R γ and R = diag(EPS_MAIN^{-1/2} I, Q_k P_k^{-1/2}), one symmetric
   eigendecomposition R'NR = U S U' per prior gives β(κ) for the whole κ grid in closed form. R is applied
   block-wise with dense BLAS.
5. **Self-supervised risk R(κ, prior, rule)**, computed from X_train and the tree outputs only:
   * points in repeated full joint cells keep their in-sample residual;
   * Good-Turing singletons (points alone in their full joint cell) use the closed-form PRESS leave-out
     r/(1−h), with leverages h = Z diag(1/(S+κ/n)) Z'/n.
   * The mean of the squared values is exactly
     R = (1−N1/n)·Σ_{n_c≥2} w_c r_c² + (N1/n)·mean_{n_c=1}[r_c/(1−h_cc)]², with w_c = n_c/(n−N1).
   * "Orphan" singletons own a pair cell or main bin that empties when they are removed. For them the
     leave-out prediction is the unseen-cell **rule** applied to the rank-one-downdated β₋ᵢ: the cell becomes
     latent. A main bin that empties is merged into its neighbour at the bin midpoint, as the baseline does.
     This is written as a sparse linear functional per orphan and fully vectorised.
6. **Unseen pair cells at predict time are deterministic (no RNG).** Two rules:
   * `zero`: the prior mean, i.e. the additive explanation.
   * `harmonic`: the posterior mean of the latent cell given the observed cells.
   For an orphan's own emptied cell c, the harmonic leave-out value is the conditional mean
   −S_c,O' β_O' / S_cc. For another unseen cell u it is E_u β with β_c replaced by that value (tower property).
7. **Per-tree choice, then an ensemble-level correction (idea step 6).** Each tree picks the argmin of R over
   κ for each (prior, rule) variant. Ties go to the smaller κ and the earlier variant. Then one ensemble-level
   candidate is chosen by the risk of the leave-out residuals **summed over trees per training point**, which
   accounts for correlated errors between trees. The candidates are (variant, global shift of every tree's κ
   index by −5..+3) or (variant, common κ for all trees). The per-tree κ and chosen variant are stored in
   `state.diagnostics` (idea step 7). Since round 3, each candidate also carries its in-sample fidelity, and
   the chosen candidate's R and in-sample residual (both / Var[T(X_train)]) are reported there.
   **Round 4:** only candidates whose in-sample residual is ≤ RESID_IN_CAP × the smallest one are admissible.
   Among them, the choice is the exact argmin of R + Ω, where Ω is the in-sample variance of each interaction
   explained by its two main effects, summed over interactions (see the round-4 section).
8. **Prediction** uses `np.digitize` on the stored splits and looks up the main-bin coefficients and the pair
   lattice. Unseen pair cells were filled at finalisation by the chosen rule.

## Hyper-parameters and design decisions (all fixed; nothing is per-dataset)
| name | value | reason |
|---|---|---|
| KAPPAS | 1e-2 … 1e4 (12 values, log spaced) | Pseudo-count units. The smallest value is near-unregularised; it replaces the idea's "baseline lsqr solution as a candidate" (see deviations). Below 0.01 the closed-form leave-out of orphans becomes optimistic compared with brute-force refits. |
| PRIORS | ridge, lattice | From the idea. |
| RULES | zero, harmonic | From the idea. A third rule, `nearest` (closest observed cell in quantile coordinates, deterministic tie-break), is implemented but not a candidate: in a 6-variant development run R never selected it on airfoil, concrete or abalone. |
| EPS_MAIN | 1.0 | Ridge weight on main bins. The idea says "tiny ε"; see deviations. |
| LATTICE_RIDGE | 0.1 | Ridge part of the lattice prior, relative to mean edge weight 1. Inherited from the earlier attempt; not tuned in this round. |
| SHIFTS | −5..+3 | Ensemble-level shifts of the per-tree κ index. |
| SE_RULE | 0 (off) | k-SE rule towards in-sample fidelity in the ensemble step. k = 1 was tested in round 3: a pure resid_in / resid_out trade, so it is off. |
| HARD_ORTHO | True | See step 2. |
| ORTHO_TERM | True | Round 4: the ensemble risk is R + Ω (orthogonality violation in variance units; no weight). |
| RESID_IN_CAP | 1.5 | Round 4: admissible candidates have in-sample residual ≤ 1.5 × the smallest. Fixed a priori; it is a trade of about 5–10% resid_in for 1–2% resid_out on the diagnostic splits. |
| lattice geometry | bin centres in quantile coordinates (empirical CDF at the bin centre) | Deviation from "bin-midpoint distances from split_list": the edge bins are unbounded and raw distances depend on feature units. Quantile coordinates are unit-free and defined for every bin. |

### Deviations from the idea (and why)
* **EPS_MAIN = 1, not a tiny ε.** R cannot tell the values apart: R/Var[T] differs by < 3% between
  ε ∈ {1e-3, 0.1, 1} on airfoil, concrete and abalone. The tie was broken by an in-sample, label-free quantity:
  with a tiny ε, shrinking interactions pushes their content into main effects (a risk the idea itself flags).
  On airfoil this raised ortho_in from 0.008 to 0.026. Held-out residuals were printed in the same dev runs and
  were almost identical across ε (airfoil resid_out 0.02257 / 0.02255 / 0.02247).
* **No explicit "baseline lsqr solution" candidate.** κ = 0.01 plays that role. The solutions also differ
  because orthogonality is exact here, so the lsqr solution is not in the same model family.
* **Exact orthogonality (step 2)** is an addition not in the idea. The earlier attempt introduced it and this
  round kept it (reason in step 2).
* **Leave-out approximation.** PRESS removes the point's residual row but not its contribution to the mean rows
  or the orthogonality null space. `dev/check_loo.py` compares against brute-force refits of one tree
  (concrete, tree 5, 12 orphan singletons). The closed form is within about 5–10% in mean square for all
  variants: slightly optimistic under the ridge prior and slightly pessimistic under the lattice prior. This
  is the risk the idea anticipated.

## Validation / checks done
* `dev/check_loo.py concrete 5 orphans`: closed form against brute-force leave-one-out refits (numbers above).
* The vectorised orphan construction and the block-wise R'NR transform reproduce the previous loop and sparse
  implementation exactly (concrete: identical R/Var = 0.02053 and resid_out = 0.02104).
* `dev/check_lsqr.py`: the baseline's lsqr with default tolerances stops at the iteration limit on 3 of 100
  concrete trees (0/100 on airfoil). Tightening tolerances leaves resid_in and ortho_in unchanged to 3
  significant digits. So baseline lsqr convergence is not a material issue on these data. The direct solver
  removes the risk but is **not** where the gains come from.
* Determinism: there is no RNG anywhere. The full-mode run reproduced the final subset run's airfoil,
  concrete and abalone metrics to all printed digits.

## Self-test results (subset, `subset_selftest.json`), ratio = new / baseline
*Rounds 2–3, earlier machine; the file is in `../S4/`, not here. For this machine and round, see the round-4
section.*

| | baseline | GT-LOCO | ratio |
|---|---|---|---|
| analytical mse_eta1 | 0.02866 | 0.02805 | 0.98 |
| analytical mse_eta12 | 0.03513 | 0.03446 | 0.98 |
| analytical mse_eta34 | 0.03203 | 0.03108 | 0.97 |
| analytical mse_others | 0.00218 | 0.00168 | **0.77** |
| analytical resid_out | 0.01032 | 0.00859 | **0.83** |
| analytical fit_s | 11.2 | 10.8 | 0.97 |
| airfoil resid_in / resid_out | 0.00985 / 0.02639 | 0.01235 / 0.02247 | 1.25 / **0.85** |
| airfoil ortho_in / ortho_out | 0.0097 / 0.0987 | 0.0084 / 0.1116 | 0.87 / 1.13 |
| concrete resid_in / resid_out | 0.00153 / 0.02500 | 0.00216 / 0.02104 | 1.41 / **0.84** |
| concrete ortho_in / ortho_out | 0.0064 / 0.0822 | 0.0061 / 0.0716 | 0.96 / 0.87 |
| abalone resid_in / resid_out | 0.00455 / 0.04062 | 0.00634 / 0.02094 | 1.39 / **0.52** |
| abalone ortho_in / ortho_out | 0.0268 / 0.0151 | 0.0252 / 0.0412 | 0.94 / **2.73 (worse)** |
| fit_s airfoil / concrete / abalone | 1.8 / 3.3 / 11.1 | 1.9 / 3.1 / 6.1 | 1.05 / 0.96 / 0.55 |

mse_eta2..6 are within ±3.5% of baseline. locvar_in is within 0.88–1.06×.

## Full-mode run, rounds 2–3 (earlier machine; the official evaluation is re-run by the pipeline)
*Numbers from the earlier machine, kept for history. The round-4 `full_selftest.json` in this directory is
summarised in the round-4 section.*

| dataset | resid_in | resid_out | ortho_in | ortho_out | locvar_in | fit_s (base → new) |
|---|---|---|---|---|---|---|
| analytical (10 reps) | – | 0.845× | – | – | – | 11.4 → 11.2 |
| abalone | 1.39× | **0.51×** | 0.94× | **2.76× (worse)** | 0.99× | 11.1 → 6.2 |
| airfoil | 1.25× | 0.84× | 0.87× | 1.14× (worse) | 0.88× | 1.8 → 1.9 |
| bike | 1.00× | 1.00× | 1.00× | 1.00× | 0.95× | 31.0 → 17.1 |
| housing | 1.11× | 0.91× | 1.01× | 0.99× | 0.75× | 74.4 → 44.7 |
| concrete | 1.41× | 0.84× | 0.96× | 0.88× | 1.06× | 3.2 → 3.2 |
| nutrition | **1.69×** | **0.76×** | 0.95× | 0.76× | 0.98× | 4.8 → 4.9 |
| parkinson | 1.01× | 1.00× | 1.00× | 1.00× | 0.93× | 33.1 → 17.4 |
| powerplant | 1.06× | 0.97× | n/a | n/a | 0.99× | 7.7 → 11.7 (1.53×) |
| superconduct | 1.01× | 1.00× | n/a | n/a | 0.98× | 701 → 113 |

Analytical (10 reps): mse_others 0.80×, mse_eta12 0.98×, mse_eta34 0.98×, and mse_eta1..6 between 0.97× and 1.003×.

### Selected configuration per dataset (from `dev/dev_gtloco.py`; label-free, visible in `state.diagnostics`)
| dataset | variant (prior, rule) | ensemble step | chosen κ quantiles (0,25,50,75,100%) | mean N1/n |
|---|---|---|---|---|
| analytical rep 0 | lattice, harmonic | shift −1 | 0.01, 0.1, 0.1, 0.3, 1 | 0.32 |
| airfoil | lattice, harmonic | shift −1 | 0.01, 0.03, 0.1, 0.3, 1000 | 0.30 |
| concrete | lattice, harmonic | shift −2 | 0.01, 0.03, 0.1, 0.1, 0.3 | 0.35 |
| abalone | lattice, harmonic | shift −3 | 0.01, 0.03, 0.03, 0.1, 1 | 0.16 |
| bike | lattice, harmonic | shift −2 | 0.01, 0.03, 0.1, 0.1, 0.3 | 0.36 |
| housing | lattice, harmonic | shift −2 | 0.01, 0.1, 0.1, 0.3, 1 | 0.42 |
| nutrition | lattice, zero | shift −2 | 0.01, 0.03, 0.1, 0.3, 1 | 0.37 |
| parkinson | ridge, zero | shift −3 | 0.01, 0.03, 0.03, 0.1, 1 | 0.52 |
| powerplant | ridge, harmonic | shift −2 | 0.01, 0.03, 0.1, 0.1, 0.3 | 0.11 |
| superconduct | lattice, zero | shift −3 | 0.01, 0.01, 0.03, 0.03, 0.3 | 0.15 |

The ensemble step always shifts the per-tree κ **down** by 1 to 3 grid steps (a factor of about 3 to 30). Per-tree
selection over-regularises because the shrinkage bias adds up coherently across trees, while leave-out noise
partly cancels.

### Where the resid_out gain comes from (development diagnostics on held-out points, never used by the method)
Common κ for all trees, compared with the baseline:
* **abalone**: baseline 0.0406. ridge/zero at κ = 0.01 (almost no shrinkage) gives 0.0320. lattice/harmonic at
  κ = 0.01 gives 0.0244. Selected: 0.0209.
* **airfoil**: baseline 0.0264. ridge/zero at κ = 0.01 gives 0.0297 (worse than baseline). lattice/harmonic at
  κ = 0.01 gives 0.0231. Selected: 0.0225.
* **concrete**: baseline 0.0250. ridge/zero at κ = 0.01 gives 0.0233. lattice/harmonic at κ = 0.01 gives 0.0230.
  Selected: 0.0210.

So the gain has three parts in dataset-dependent proportions:
* (a) the exact, hard-orthogonal solve;
* (b) the harmonic unseen-cell rule, which matters most on airfoil and abalone;
* (c) the leave-out-selected shrinkage, which matters most on concrete and adds about 10–15% on abalone.

**Superseded in round 4** (`attribution/README.md`). "κ = 0.01 (almost no shrinkage)" was wrong: κ = 0.01
already shrinks weakly identified directions strongly. At κ = 1e-4, lattice/harmonic abalone gives 0.0401,
i.e. baseline level. So (a) is ≈ 0, and negative on abalone for an exact solve of the baseline system.
(b) is ≈ 0 on unshrunk coefficients, except the lattice tie-break on airfoil (−13%). Almost all of the gain
is (c), shrinkage by the prior.

## Trade-offs and caveats (reported, not hidden)
* **resid_in rises** by design. It rises wherever resid_out falls, which is the in-sample vs out-of-sample
  trade-off the idea describes.
  * Round 3: up to 1.4× on concrete and abalone, and 1.69× on nutrition, the dataset with the weakest model
    (R² = 0.29).
  * Round 4 caps it at 1.5× the least-regularised fit: nutrition is now 1.32×, abalone 1.31×, concrete 1.41×
    and airfoil 1.26× baseline. The cost is nutrition resid_out −19.8% instead of −25.0%.
* **ortho_out is worse on the benchmark split of abalone (0.015 → 0.041; 0.042 in round 4) and airfoil
  (0.099 → 0.112)**. It is
  better on concrete (0.88×) and nutrition (0.76×). Round 3 checked this across 8 splits and with a test-set
  bootstrap (section above):
  * abalone splits 1–7: geo-mean 0.98×;
  * airfoil, all 8 splits: 0.89×;
  * bootstrap on the benchmark split: P(GT-LOCO > baseline) = 0.54;
  * it is not caused by the harmonic rule: the zero rule at the same κ is worse on that split.

  So it is sampling noise in a max-|corr| metric on 836 test points, but it remains a worse number on the
  benchmark split.
* **The ensemble κ step trades a little resid_out and ortho for a lot of resid_in**: relative to per-tree-only,
  resid_out is 0.98–1.07× and resid_in 0.32–0.79×. See round 3 (c).
  * Round 4 adds the orthogonality term Ω to its risk. Ω is 1–2 orders of magnitude smaller than R, so it
    only breaks near-ties; on the benchmark split it moves abalone resid_out from −48.9% to −46.5%.
  * Round 4 also adds the resid_in cap (above).
* **No gain on bike, parkinson or superconduct** (all metrics within ±1%, apart from locvar). R selects mild
  shrinkage there (median κ ≈ 0.03–0.1). Their resid_out appears to be dominated by genuine higher-order tree
  content in seen cells, which an order-2 decomposition cannot represent. This is the "modest gains" risk the
  idea named.
* **Fit time.** On the earlier machine it was 0.16–1.05× baseline, except powerplant (1.53×). On this machine,
  where BLAS is single-threaded, see the round-4 timing table. Everything is under the 5× limit.
* **Design decisions and held-out numbers.**
  * The dev scripts print held-out residuals next to R, so these numbers were seen during development.
  * In this round, every configuration choice (dropping `nearest`, EPS_MAIN) was made from R or from in-sample,
    label-free quantities.
  * The earlier interrupted attempt compared configurations on the subset benchmark (`dev/subset_v1.json`,
    `subset_v2_hard.json` and `subset_v3_eps1.json`, all produced with its older code). Its choices of
    KAPPAS, LATTICE_RIDGE and HARD_ORTHO may have been informed by those runs.
  * No choice depends on the dataset, and none uses labels, analytical ground truth or test inputs.
* `dev/diag_floor.py` was left by the earlier attempt. It fits on train ∪ test **for diagnosis only**. It was
  not run in this round and is not used by the method.

## Files
* `method.py`: benchmark interface (`fit`, `predict`).
* `lib/gtloco.py`: the method.
* `lib/treehfd_mod/`: unmodified copy of the baseline package (partition code reused).
* `subset_selftest.json` (rounds 2–3, in `../S4/`).
* `dev/`:
  * `dev_gtloco.py`: selection diagnostics and the full candidate path.
  * `check_loo.py`: closed-form vs brute-force leave-out check.
  * `check_lsqr.py`: baseline lsqr convergence check.
  * `diag_ortho.py`, `sizes.py`, `profile_trees.py`, `compare.py`: other diagnostics.
  * Round 3: `diag_ortho_out.py` (multi-split baseline vs GT-LOCO, bootstrap, rule swap), `ablate_shift.py`
    (ensemble step ablation) and `compare_se.py` (k-SE rule vs argmin). Their logs are in `../S4/dev/logs/`.
  * Round 4:
    * `cand_table.py`: per-split table of every ensemble candidate, cached in `cache/`, which is not versioned.
    * `analyse_cands.py`, `analyse_rho.py`: selection rules and LATTICE_RIDGE on the diagnostic splits.
    * `kappa_limit.py`: the κ → 0 check.
    * `quick_sel.py`: selection summary per dataset.
    * `time_fit.py`, `time_pair.py`: profiling and idle-machine timing.
    * `compare_full.py`: table vs this machine's baseline.
    * Logs are in `logs/`.
* `attribution/` (round 4): one-change-at-a-time full-mode runs (baseline solver and fallback switches,
  GT-LOCO at κ → 0), with results, `full_table.md` and `README.md`.
* `full_selftest.json`, `full_selftest.log`: round-4 full self-test on this machine.
