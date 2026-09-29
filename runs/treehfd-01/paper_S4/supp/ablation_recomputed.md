eval_env: {'base': '22b414443084', 'full': '22b414443084', 'no_shrinkage_direct_solver_only': '22b414443084', 'no_lattice_prior_ridge_zero_only': '22b414443084', 'no_ensemble_level_kappa_step': '22b414443084', 'soft_orthogonality_rows': '22b414443084'}

### resid_out, ratio to TreeHFD

| variant | abalone | airfoil | bike | housing | concrete | nutrition | parkinson | powerplant | superconduct | gm all | gm excl. abalone, airfoil |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Full GT-LOCO | 0.51 | 0.84 | 1.00 | 0.91 | 0.84 | 0.75 | 1.00 | 0.96 | 1.00 | 0.85 | 0.92 |
| No shrinkage (kappa=0.01 only) | 0.59 | 0.86 | 1.00 | 0.97 | 0.92 | 0.89 | 1.00 | 0.97 | 1.00 | 0.90 | 0.96 |
| No lattice prior / harmonic rule | 0.66 | 1.10 | 1.00 | 0.92 | 0.83 | 0.75 | 1.00 | 0.97 | 1.00 | 0.90 | 0.92 |
| No ensemble-level kappa step | 0.53 | 0.89 | 1.00 | 0.90 | 0.88 | 0.68 | 1.03 | 1.01 | 1.06 | 0.87 | 0.93 |
| Soft orthogonality rows | 0.50 | 0.83 | 1.00 | 0.91 | 0.83 | 0.75 | 1.00 | 0.96 | 1.00 | 0.85 | 0.92 |

### resid_out, ratio to full GT-LOCO

| variant | abalone | airfoil | bike | housing | concrete | nutrition | parkinson | powerplant | superconduct | gm all | gm excl. abalone, airfoil |
|---|---|---|---|---|---|---|---|---|---|---|---|
| No shrinkage (kappa=0.01 only) | 1.16 | 1.03 | 1.00 | 1.06 | 1.09 | 1.18 | 1.00 | 1.01 | 1.00 | 1.06 | 1.05 |
| No lattice prior / harmonic rule | 1.29 | 1.32 | 1.00 | 1.01 | 0.98 | 1.00 | 1.00 | 1.01 | 1.00 | 1.06 | 1.00 |
| No ensemble-level kappa step | 1.03 | 1.06 | 1.00 | 0.99 | 1.04 | 0.90 | 1.03 | 1.05 | 1.06 | 1.02 | 1.01 |
| Soft orthogonality rows | 0.97 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 |

### resid_in, ratio to TreeHFD

| variant | abalone | airfoil | bike | housing | concrete | nutrition | parkinson | powerplant | superconduct | gm all | gm excl. abalone, airfoil |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Full GT-LOCO | 1.39 | 1.25 | 1.00 | 1.11 | 1.41 | 1.68 | 1.01 | 1.06 | 1.01 | 1.20 | 1.16 |
| No shrinkage (kappa=0.01 only) | 1.09 | 1.03 | 1.00 | 1.01 | 1.06 | 1.07 | 1.01 | 1.03 | 1.00 | 1.03 | 1.03 |
| No lattice prior / harmonic rule | 1.30 | 1.09 | 1.00 | 1.07 | 1.31 | 1.41 | 1.01 | 1.06 | 1.01 | 1.13 | 1.11 |
| No ensemble-level kappa step | 4.55 | 1.55 | 1.02 | 1.37 | 3.82 | 3.73 | 1.19 | 1.45 | 1.11 | 1.86 | 1.68 |
| Soft orthogonality rows | 1.35 | 1.25 | 1.00 | 1.12 | 1.58 | 1.68 | 1.01 | 1.06 | 1.01 | 1.21 | 1.18 |

### resid_in, ratio to full GT-LOCO

| variant | abalone | airfoil | bike | housing | concrete | nutrition | parkinson | powerplant | superconduct | gm all | gm excl. abalone, airfoil |
|---|---|---|---|---|---|---|---|---|---|---|---|
| No shrinkage (kappa=0.01 only) | 0.78 | 0.82 | 1.00 | 0.91 | 0.75 | 0.64 | 1.00 | 0.97 | 0.99 | 0.86 | 0.88 |
| No lattice prior / harmonic rule | 0.93 | 0.87 | 1.00 | 0.96 | 0.93 | 0.84 | 1.00 | 1.00 | 1.00 | 0.95 | 0.96 |
| No ensemble-level kappa step | 3.27 | 1.24 | 1.02 | 1.23 | 2.71 | 2.21 | 1.18 | 1.37 | 1.10 | 1.56 | 1.45 |
| Soft orthogonality rows | 0.97 | 1.00 | 1.00 | 1.01 | 1.12 | 1.00 | 1.00 | 1.00 | 1.00 | 1.01 | 1.02 |

### ortho_in, ratio to TreeHFD

| variant | abalone | airfoil | bike | housing | concrete | nutrition | parkinson | powerplant | superconduct | gm all | gm excl. abalone, airfoil |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Full GT-LOCO | 0.94 | 0.87 | 1.00 | 1.01 | 0.96 | 0.95 | 1.00 | n/a | n/a | 0.96 | 0.98 |
| No shrinkage (kappa=0.01 only) | 1.01 | 1.04 | 1.00 | 1.00 | 0.85 | 0.96 | 1.00 | n/a | n/a | 0.98 | 0.96 |
| No lattice prior / harmonic rule | 0.96 | 0.79 | 1.00 | 1.02 | 1.14 | 0.96 | 1.00 | n/a | n/a | 0.98 | 1.02 |
| No ensemble-level kappa step | 0.39 | 0.74 | 1.01 | 1.03 | n/a | 0.92 | 1.04 | n/a | n/a | 0.81 | 1.00 |
| Soft orthogonality rows | 1.18 | 0.74 | 0.99 | 1.04 | 2.20 | 1.82 | 1.01 | n/a | n/a | 1.20 | 1.33 |

### ortho_in, ratio to full GT-LOCO

| variant | abalone | airfoil | bike | housing | concrete | nutrition | parkinson | powerplant | superconduct | gm all | gm excl. abalone, airfoil |
|---|---|---|---|---|---|---|---|---|---|---|---|
| No shrinkage (kappa=0.01 only) | 1.07 | 1.20 | 1.00 | 0.99 | 0.89 | 1.01 | 1.00 | n/a | n/a | 1.02 | 0.98 |
| No lattice prior / harmonic rule | 1.02 | 0.91 | 1.00 | 1.01 | 1.18 | 1.02 | 1.00 | n/a | n/a | 1.02 | 1.04 |
| No ensemble-level kappa step | 0.42 | 0.86 | 1.01 | 1.01 | n/a | 0.97 | 1.04 | n/a | n/a | 0.85 | 1.01 |
| Soft orthogonality rows | 1.26 | 0.85 | 0.99 | 1.03 | 2.29 | 1.92 | 1.01 | n/a | n/a | 1.25 | 1.35 |

### ortho_out, ratio to TreeHFD

| variant | abalone | airfoil | bike | housing | concrete | nutrition | parkinson | powerplant | superconduct | gm all | gm excl. abalone, airfoil |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Full GT-LOCO | 2.65 | 1.15 | 1.00 | 0.99 | 0.87 | 0.77 | 1.00 | n/a | n/a | 1.11 | 0.92 |
| No shrinkage (kappa=0.01 only) | 2.38 | 1.52 | 1.00 | 1.00 | 0.89 | 0.94 | 1.00 | n/a | n/a | 1.17 | 0.97 |
| No lattice prior / harmonic rule | 3.19 | 1.25 | 1.00 | 1.01 | 0.84 | 0.83 | 1.00 | n/a | n/a | 1.16 | 0.93 |
| No ensemble-level kappa step | 3.37 | 1.22 | 1.00 | 0.82 | 0.82 | 0.54 | 1.04 | n/a | n/a | 1.06 | 0.82 |
| Soft orthogonality rows | 3.75 | 1.08 | 1.00 | 0.98 | 0.85 | 0.77 | 1.00 | n/a | n/a | 1.14 | 0.91 |

### ortho_out, ratio to full GT-LOCO

| variant | abalone | airfoil | bike | housing | concrete | nutrition | parkinson | powerplant | superconduct | gm all | gm excl. abalone, airfoil |
|---|---|---|---|---|---|---|---|---|---|---|---|
| No shrinkage (kappa=0.01 only) | 0.90 | 1.32 | 1.00 | 1.01 | 1.02 | 1.23 | 1.00 | n/a | n/a | 1.06 | 1.05 |
| No lattice prior / harmonic rule | 1.21 | 1.08 | 1.00 | 1.02 | 0.96 | 1.08 | 1.00 | n/a | n/a | 1.05 | 1.01 |
| No ensemble-level kappa step | 1.28 | 1.06 | 1.00 | 0.83 | 0.95 | 0.70 | 1.04 | n/a | n/a | 0.96 | 0.89 |
| Soft orthogonality rows | 1.42 | 0.93 | 1.00 | 0.99 | 0.97 | 1.00 | 1.01 | n/a | n/a | 1.04 | 0.99 |

### analytical case, ablation / full GT-LOCO (mean over 10 reps)

| variant | mse_eta12 (ratio to full) | mse_eta34 (ratio to full) | mse_others (ratio to full) | resid_out (ratio to full) |
|---|---|---|---|---|
| No shrinkage (kappa=0.01 only) | 1.01 | 1.00 | 1.15 | 1.09 |
| No lattice prior / harmonic rule | 1.43 | 1.29 | 1.13 | 1.42 |
| No ensemble-level kappa step | 1.09 | 1.08 | 0.84 | 1.02 |
| Soft orthogonality rows | 1.00 | 1.00 | 1.01 | 1.00 |

### raw numbers behind the statements on concrete and nutrition

* No lattice prior / harmonic rule on concrete: resid_out: full 0.02104 -> ablation 0.02072 (x0.98) resid_in: full 0.002164 -> ablation 0.002013 (x0.93) ortho_in: full 0.006129 -> ablation 0.007239 (x1.18) ortho_out: full 0.07164 -> ablation 0.06886 (x0.96)
* No lattice prior / harmonic rule on nutrition: resid_out: full 0.07253 -> ablation 0.07277 (x1.00) resid_in: full 0.03009 -> ablation 0.02514 (x0.84) ortho_in: full 0.01539 -> ablation 0.01565 (x1.02) ortho_out: full 0.08168 -> ablation 0.08811 (x1.08)
* No ensemble-level kappa step on concrete: resid_out: full 0.02104 -> ablation 0.02195 (x1.04) resid_in: full 0.002164 -> ablation 0.005857 (x2.71) ortho_in: n/a ortho_out: full 0.07164 -> ablation 0.06783 (x0.95)
* No ensemble-level kappa step on nutrition: resid_out: full 0.07253 -> ablation 0.06524 (x0.90) resid_in: full 0.03009 -> ablation 0.06657 (x2.21) ortho_in: full 0.01539 -> ablation 0.0149 (x0.97) ortho_out: full 0.08168 -> ablation 0.05689 (x0.70)
