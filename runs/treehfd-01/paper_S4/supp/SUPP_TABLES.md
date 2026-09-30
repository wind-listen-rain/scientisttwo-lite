Re-fit of the submitted GT-LOCO in this directory reproduces the official results/method_full.json: max relative difference over resid_in/resid_out/ortho_out on 9 datasets = 0.00e+00.

**resid_out, ratio to TreeHFD**

| variant | abalone | airfoil | bike | housing | concrete | nutrition | parkinson | powerplant | superconduct | geo-mean |
|---|---|---|---|---|---|---|---|---|---|---|
| (d) TreeHFD + purification | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| (a) ridge, zero rule, one global kappa | 0.69 | 1.12 | 1.00 | 0.92 | 0.86 | 0.74 | 1.00 | 0.97 | 1.00 | 0.912 |
| (b) ridge, harmonic rule, one global kappa | 0.57 | 0.97 | 1.00 | 0.91 | 0.85 | 0.73 | 0.99 | 0.97 | 1.00 | 0.875 |
| (e) lattice prior + harmonic/zero rule, one global kappa | 0.57 | 0.86 | 1.00 | 0.91 | 0.84 | 0.78 | 1.00 | 0.96 | 1.00 | 0.866 |
| ridge prior/zero rule, full ensemble selection (per-tree kappa + shift) | 0.55 | 0.97 | 1.00 | 0.92 | 0.82 | 0.76 | 1.00 | 0.96 | 1.00 | 0.871 |
| (c) full GT-LOCO | 0.52 | 0.84 | 1.00 | 0.91 | 0.84 | 0.75 | 1.00 | 0.96 | 1.00 | 0.855 |

**resid_in, ratio to TreeHFD**

| variant | abalone | airfoil | bike | housing | concrete | nutrition | parkinson | powerplant | superconduct | geo-mean |
|---|---|---|---|---|---|---|---|---|---|---|
| (d) TreeHFD + purification | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| (a) ridge, zero rule, one global kappa | 1.26 | 1.02 | 1.00 | 1.06 | 1.17 | 1.48 | 1.01 | 1.07 | 1.00 | 1.110 |
| (b) ridge, harmonic rule, one global kappa | 1.26 | 1.02 | 1.00 | 1.06 | 1.17 | 1.48 | 1.01 | 1.07 | 1.00 | 1.110 |
| (e) lattice prior + harmonic/zero rule, one global kappa | 1.26 | 1.22 | 1.00 | 1.15 | 1.47 | 1.46 | 1.01 | 1.07 | 1.00 | 1.171 |
| ridge prior/zero rule, full ensemble selection (per-tree kappa + shift) | 1.31 | 1.08 | 1.00 | 1.07 | 1.33 | 1.41 | 1.01 | 1.06 | 1.01 | 1.132 |
| (c) full GT-LOCO | 1.39 | 1.25 | 1.00 | 1.11 | 1.41 | 1.68 | 1.01 | 1.06 | 1.01 | 1.196 |

**ortho_out, ratio to TreeHFD**

| variant | abalone | airfoil | bike | housing | concrete | nutrition | parkinson | powerplant | superconduct | geo-mean |
|---|---|---|---|---|---|---|---|---|---|---|
| (d) TreeHFD + purification | 1.08 | 1.07 | 1.01 | 0.82 | 1.06 | 0.91 | 0.84 | n/a | n/a | 0.964 |
| (a) ridge, zero rule, one global kappa | 2.99 | 1.11 | 1.00 | 1.04 | 0.83 | 0.74 | 1.00 | n/a | n/a | 1.114 |
| (b) ridge, harmonic rule, one global kappa | 2.58 | 1.10 | 1.00 | 1.05 | 0.82 | 0.68 | 1.00 | n/a | n/a | 1.076 |
| (e) lattice prior + harmonic/zero rule, one global kappa | 2.58 | 1.08 | 1.00 | 1.00 | 0.87 | 0.77 | 1.00 | n/a | n/a | 1.093 |
| ridge prior/zero rule, full ensemble selection (per-tree kappa + shift) | 2.68 | 1.20 | 1.00 | 1.01 | 0.83 | 0.80 | 1.00 | n/a | n/a | 1.116 |
| (c) full GT-LOCO | 2.62 | 1.13 | 1.00 | 0.99 | 0.88 | 0.74 | 1.00 | n/a | n/a | 1.095 |

**Analytical case, mean ± std over 10 repetitions**

| variant | mse_eta12 | mse_eta34 | mse_others | resid_in | resid_out | ortho_out |
|---|---|---|---|---|---|---|
| treehfd | 0.03114 ± 0.0031 | 0.0281 ± 0.0039 | 0.002124 ± 0.00028 | 0.00259 ± 0.00026 | 0.01018 ± 0.0012 | 0.06456 ± 0.022 |
| purified_treehfd | 0.03169 ± 0.0034 | 0.02834 ± 0.0034 | 0.002127 ± 0.00028 | 0.00259 ± 0.00026 | 0.01018 ± 0.0012 | 0.04522 ± 0.017 |
| ridge_zero | 0.04354 ± 0.0073 | 0.03583 ± 0.0059 | 0.001868 ± 0.00025 | 0.002702 ± 0.00027 | 0.01187 ± 0.0017 | 0.07422 ± 0.024 |
| ridge_harmonic | 0.03214 ± 0.0033 | 0.02892 ± 0.004 | 0.001931 ± 0.00026 | 0.002683 ± 0.00029 | 0.009238 ± 0.001 | 0.06633 ± 0.024 |
| lattice_common | 0.03022 ± 0.0027 | 0.02761 ± 0.0038 | 0.001632 ± 0.00025 | 0.003119 ± 0.00035 | 0.008334 ± 0.00096 | 0.06618 ± 0.024 |
| gtloco | 0.03026 ± 0.0028 | 0.02769 ± 0.0038 | 0.001648 ± 0.00028 | 0.003102 ± 0.00035 | 0.008371 ± 0.00091 | 0.06604 ± 0.024 |

**Existing analytical case, paired per-rep difference GT-LOCO − TreeHFD (mean ± SE over 10 reps)**

| metric | TreeHFD mean | GT-LOCO mean | diff ± SE | reps with GT-LOCO lower |
|---|---|---|---|---|
| mse_eta1 | 0.0284 | 0.02824 | -0.000161 ± 5.4e-05 | 9/10 |
| mse_eta2 | 0.01791 | 0.01791 | -9.58e-07 ± 6.3e-05 | 5/10 |
| mse_eta3 | 0.01659 | 0.01646 | -0.000128 ± 8.7e-05 | 7/10 |
| mse_eta4 | 0.01618 | 0.01619 | 8.46e-06 ± 4.5e-05 | 4/10 |
| mse_eta5 | 0.0003006 | 0.0002894 | -1.12e-05 ± 7.7e-06 | 6/10 |
| mse_eta6 | 0.0003665 | 0.0003626 | -3.92e-06 ± 5.4e-06 | 6/10 |
| mse_eta12 | 0.03114 | 0.03026 | -0.000878 ± 0.00028 | 9/10 |
| mse_eta34 | 0.0281 | 0.02769 | -0.000409 ± 0.00036 | 9/10 |
| mse_others | 0.002124 | 0.001648 | -0.000476 ± 3.1e-05 | 10/10 |
| resid_out | 0.01018 | 0.008371 | -0.00181 ± 0.00019 | 10/10 |
| resid_in | 0.00259 | 0.003102 | 0.000512 ± 6e-05 | 0/10 |
| ortho_out | 0.06456 | 0.06604 | 0.00148 ± 0.0011 | 4/10 |

**New case (AR(1) rho=0.6, f = x1 + 0.8 x2^2 + 0.5 x3^3 + 1.5 x4 x5), 10 seeds**

| metric | TreeHFD | GT-LOCO | diff GT−TreeHFD ± SE | seeds GT-LOCO lower |
|---|---|---|---|---|
| mse_eta1 | 0.009418 ± 0.0023 | 0.009264 ± 0.0022 | -0.000154 ± 0.00013 | 7/10 |
| mse_eta2 | 0.05381 ± 0.013 | 0.05413 ± 0.013 | 0.000316 ± 0.00046 | 4/10 |
| mse_eta3 | 0.1797 ± 0.06 | 0.1809 ± 0.06 | 0.00118 ± 0.00045 | 3/10 |
| mse_eta4 | 0.0359 ± 0.012 | 0.0363 ± 0.012 | 0.000397 ± 0.00035 | 4/10 |
| mse_eta5 | 0.03214 ± 0.0075 | 0.03211 ± 0.0074 | -3.73e-05 ± 0.00013 | 7/10 |
| mse_eta6 | 0.000399 ± 0.00019 | 0.0003616 ± 0.00014 | -3.75e-05 ± 3.6e-05 | 5/10 |
| mse_main_mean | 0.0519 ± 0.01 | 0.05217 ± 0.011 | 0.000278 ± 0.00018 | 4/10 |
| mse_eta45 | 0.05328 ± 0.0052 | 0.0593 ± 0.0092 | 0.00602 ± 0.0023 | 0/10 |
| mse_others_zero | 0.01588 ± 0.0029 | 0.01085 ± 0.003 | -0.00503 ± 0.00083 | 10/10 |
| ortho_out | 0.07915 ± 0.04 | 0.07958 ± 0.04 | 0.000422 ± 0.00057 | 3/10 |
| resid_out | 0.01385 ± 0.0021 | 0.01171 ± 0.0013 | -0.00213 ± 0.00057 | 10/10 |
| resid_in | 0.001945 ± 0.00021 | 0.003006 ± 0.00098 | 0.00106 ± 0.00029 | 0/10 |

True Var[eta_45] = 0.713; largest variance among the other 14 true pair components = 2.5e-28; XGB R^2 against f on test = 0.949.

**R3 sensitivity (one change at a time; label-free; defaults were chosen during development with held-out numbers visible)**

| setting | gm resid_out / TreeHFD | per-dataset range resid_out | gm resid_in / TreeHFD | per-dataset range resid_in | gm ortho_out / TreeHFD |
|---|---|---|---|---|---|
| default (lat 0.1, main 1, 12 kappas, shifts -5..+3) | 0.855 | 0.52–1.00 | 1.196 | 1.00–1.68 | 1.095 |
| lattice ridge 0.03 | 0.855 | 0.52–1.00 | 1.189 | 1.00–1.67 | 1.083 |
| lattice ridge 0.3 | 0.863 | 0.57–1.00 | 1.187 | 1.00–1.63 | 1.107 |
| main-bin ridge 0.3 | 0.862 | 0.56–1.00 | 1.208 | 1.00–1.69 | 1.107 |
| main-bin ridge 3 | 0.854 | 0.51–1.00 | 1.176 | 1.00–1.54 | 1.083 |
| kappa grid 6 pts (shifts -5..+3 in index units) | 0.861 | 0.53–1.00 | 1.214 | 1.00–1.53 | 1.116 |
| kappa grid 24 pts (shifts -5..+3 in index units) | 0.852 | 0.52–1.00 | 1.202 | 1.00–1.57 | 1.098 |
| kappa grid 6 pts (shifts rescaled to same kappa range) | 0.861 | 0.53–1.00 | 1.214 | 1.00–1.53 | 1.116 |
| kappa grid 24 pts (shifts rescaled to same kappa range) | 0.855 | 0.53–1.00 | 1.185 | 1.00–1.57 | 1.092 |
| shift range 0..0 | 0.871 | 0.57–1.00 | 1.202 | 1.00–1.55 | 1.109 |
| shift range -3..+1 | 0.855 | 0.52–1.00 | 1.196 | 1.00–1.68 | 1.095 |
| shift range -8..+5 | 0.855 | 0.52–1.00 | 1.196 | 1.00–1.68 | 1.095 |

**R5(i) fixed-shift sweep (selected variant per dataset, no selection over shifts)**

| dataset | selected candidate | resid_out at selected | best fixed shift for resid_out (oracle) | resid_out there | shift minimising R (risk) | resid_out there | best shift for resid_in |
|---|---|---|---|---|---|---|---|
| abalone | shift -3 (variant 3) | 0.02094 | -1 | 0.01971 | -3 | 0.02094 | -5 |
| airfoil | shift -1 (variant 3) | 0.02247 | -2 | 0.02239 | -1 | 0.02247 | -5 |
| bike | shift -2 (variant 3) | 0.0246 | -1 | 0.02459 | -2 | 0.0246 | -5 |
| housing | shift -2 (variant 3) | 0.01406 | -1 | 0.01383 | -2 | 0.01406 | -5 |
| concrete | shift -2 (variant 3) | 0.02104 | -1 | 0.02091 | -2 | 0.02104 | -5 |
| nutrition | shift -2 (variant 2) | 0.07253 | 0 | 0.06736 | -2 | 0.07253 | -5 |
| parkinson | shift -3 (variant 0) | 0.009561 | -5 | 0.009529 | -3 | 0.009561 | -5 |
| powerplant | shift -2 (variant 1) | 0.001065 | -2 | 0.001065 | -2 | 0.001065 | -5 |
| superconduct | shift -3 (variant 2) | 0.00909 | -4 | 0.009068 | -3 | 0.00909 | -5 |

Geo-mean resid_out / TreeHFD: selected 0.855; fixed shift chosen by min risk 0.855; oracle best fixed shift (uses held-out, diagnostic only) 0.840.

| fixed shift | -5 | -4 | -3 | -2 | -1 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|---|---|
| gm resid_out / TreeHFD | 0.902 | 0.890 | 0.869 | 0.851 | 0.845 | 0.876 | 1.012 | 1.432 | 2.538 |
| gm resid_in / TreeHFD | 1.034 | 1.051 | 1.105 | 1.221 | 1.442 | 1.860 | 2.629 | 4.190 | 7.546 |
