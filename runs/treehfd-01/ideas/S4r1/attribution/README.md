# Attribution runs (round 4, S4r1): where does GT-LOCO's gain come from?

The critic asked whether the gains come from GT-LOCO's side-fixes: the exact solver and the deterministic
unseen-cell rule. The ablation "no_shrinkage_direct_solver_only" (κ = 0.01) seemed to show that they did. To
separate the two, each run below changes **one** thing relative to the baseline, or to the previous row. All
are full-mode harness runs on this machine (env-22b414443084), compared with `baseline/env-22b414443084/full.json`.
Full table: `full_table.md` (all metrics; generated with `dev/compare_full.py --md`).

| column | file | what it is |
|---|---|---|
| det_nearest | `method_base_lsqr_nearest_det.py` | Baseline lsqr solution, baseline L1-nearest fallback. The random last tie-break is replaced by the smallest cell index. |
| harmonic_fill | `method_base_lsqr_harmonic.py` | Baseline lsqr solution. Unseen pair cells take the harmonic extension under GT-LOCO's lattice prior (same geometry and ridge). |
| zero_fill | `method_base_lsqr_zero.py` | Baseline lsqr solution. Unseen pair cells take 0. |
| exact | `method_base_exact_random.py` | The baseline's own least-squares system, solved exactly for its minimum-norm solution, which is what lsqr converges to from 0. Uses eigh of A'A. Baseline random fallback. **This is the solver change alone.** |
| exact+harm | `method_base_exact_harmonic.py` | exact + harmonic_fill. |
| k0_ridge | `method_gtloco_k0_ridge_harmonic.py` | GT-LOCO machinery with κ = 1e-4, 100× below the method's smallest grid value, so there is no effective shrinkage. Direct solver, exact orthogonality, ridge prior, harmonic rule. |
| k0_lattice | `method_gtloco_k0_lattice_harmonic.py` | Same as k0_ridge with the lattice prior. The prior acts only as a tie-breaker among fits of (nearly) equal data fit. |
| k001 | `../../S4/ablations/no_shrinkage_direct_solver_only_full.json` | The earlier "no shrinkage" ablation: κ = 0.01 for every tree, variant chosen by R. |
| S4 | `../../S4/full_env-22b414443084.json` | GT-LOCO as officially evaluated in round 3. |
| S4r1 | `../full_selftest.json` | This round: ensemble step with orthogonality term and resid_in cap. |

`_base.py` wraps the unmodified `treehfd` package, which is read only. SOLVER swaps `treehfd.tree.lsqr` in
memory; RULE changes only the values of unseen pair cells at prediction time. The lsqr runs reproduce the
baseline's in-sample components exactly (max abs difference 0 on concrete).

## Key numbers (value and change vs baseline)
| metric | baseline | det_nearest | harmonic_fill | zero_fill | exact | k0_ridge | k0_lattice | k001 | S4 | S4r1 |
|---|---|---|---|---|---|---|---|---|---|---|
| analytical resid_out | 0.01016 | +0.1% | +0.6% | +24.7% | +0.2% | −1.2% | −2.3% | −10.2% | −17.6% | −17.6% |
| analytical mse_others | 0.002125 | 0.0% | +0.6% | −2.2% | +0.1% | −0.7% | −1.8% | −10.7% | −22.4% | −22.4% |
| analytical mse_eta12 | 0.03118 | −0.1% | +2.2% | +37.6% | +0.1% | +1.8% | −0.4% | −2.4% | −3.0% | −3.0% |
| abalone resid_out | 0.04097 | −0.4% | +0.6% | +12.2% | **+81.4%** | +12.9% | −2.0% | −40.5% | −48.9% | −46.5% |
| airfoil resid_out | 0.02691 | −0.3% | −3.2% | +10.4% | −0.3% | −3.2% | **−13.1%** | −14.2% | −16.5% | −16.5% |
| concrete resid_out | 0.02499 | +0.1% | −2.6% | −2.8% | −0.1% | −2.7% | −0.6% | −8.3% | −15.8% | −15.8% |
| nutrition resid_out | 0.09665 | −0.3% | +1.3% | −1.4% | +0.6% | +1.3% | +3.5% | −11.2% | −25.0% | −19.8% |
| housing resid_out | 0.01544 | 0.0% | +0.6% | −0.5% | −0.2% | +0.6% | +0.8% | −3.3% | −9.0% | −9.0% |
| powerplant resid_out | 0.001104 | +0.1% | −0.8% | +0.1% | −0.2% | −0.8% | −0.8% | −2.9% | −3.5% | −3.5% |
| bike / parkinson / superconduct resid_out | – | ≤ ±0.1% | ≤ ±0.3% | ≤ ±0.8% | ≤ ±1.3% | ≤ ±0.7% | ≤ ±0.3% | ≤ ±0.4% | ≤ ±0.2% | ≤ ±0.2% |

## Conclusions
1. **The random fallback costs nothing.** Making it deterministic (det_nearest) moves every metric by ≤ 0.4%,
   except ortho_out on nutrition (+6%), which is a max-|corr| on 1,300 test points. Determinism is a
   reproducibility fix, not a source of gain.
2. **The harmonic rule, applied to the baseline's coefficients, gives about 0** (−3.2% to +1.3% resid_out). The
   zero rule hurts: analytical +25%, abalone +12%, mse_eta12 +38%.
3. **The exact solver alone gives nothing, and hurts abalone badly.** exact = lsqr to ±0.6% on eight datasets.
   On abalone the exact minimum-norm solution has resid_out +81%, ortho_in +49% and ortho_out ×5.9. The
   baseline's lsqr stops early there, and the early stop acts as an implicit regulariser of weakly identified
   directions. An exact solver removes that regularisation, so something else has to supply it.
4. **GT-LOCO without effective shrinkage (κ → 0) is at baseline level.** This holds with the direct solver,
   exact orthogonality and the harmonic rule:
   * k0_ridge: −3% to +13% resid_out;
   * k0_lattice: −2% to +3.5% resid_out, with one exception: airfoil −13%. There the lattice prior, used only as
     a tie-breaker between equally good fits, picks smooth values for weakly identified cells, and the harmonic
     rule extends them.
5. **The earlier "no_shrinkage_direct_solver_only" ablation was mislabelled.** κ = 0.01 pseudo-counts is already
   effective shrinkage of weakly identified directions. On the benchmark split (`dev/kappa_limit.py`), for
   lattice/harmonic with a common κ:
   * abalone resid_out: 0.0401 (κ = 1e-4), 0.0300 (κ = 1e-3), 0.0244 (κ = 0.01);
   * nutrition: 0.1000, 0.0963, 0.0868;
   * concrete: 0.0248, 0.0245, 0.0230;
   * airfoil is flat (0.0234, 0.0233, 0.0231).

   Its gains are shrinkage gains, not solver gains.
6. **Credit, from these rows:**
   * the side-fixes (solver, determinism, rule on unshrunk coefficients) contribute ≈ 0, and abalone is worse;
   * the lattice prior as a tie-breaker contributes airfoil −13%;
   * shrinkage by the prior, from κ → 0 to the GT/PRESS-selected per-tree κ plus the ensemble step, gives the
     rest: analytical −2% → −18%, mse_others −2% → −22%, abalone −2% → −47/−49%, concrete −1% → −16%,
     nutrition +3.5% → −20/−25%, housing +1% → −9%.
   * Within the shrinkage, a fixed κ = 0.01 gets roughly half of the resid_out gain on analytical, concrete and
     nutrition, and half of the mse_others gain. The leave-out-selected κ gets the rest.
   * The critic's reading, that most of the effect comes from the solver and the deterministic unseen-cell
     rule, rested on the κ = 0.01 ablation. The runs above do not support that reading.

fit_s in these runs is **not** comparable: up to nine harness processes ran at the same time, and "exact"
solves a dense eigenproblem of A'A on purpose. For timing, see `../dev/logs/time_pair_idle.log`.
