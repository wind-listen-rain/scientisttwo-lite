
#### airfoil (8 splits; ratio to S4, then resid_in / resid_out ratio to baseline)

| selection | resid_in | resid_out | ortho_in | ortho_out | locvar_in | resid_in / base | resid_out / base |
|---|---|---|---|---|---|---|---|
| S4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.262 | 0.858 |
| E1 round 1: G1 argmin | 0.976 | 0.854 | 0.811 | 1.127 | 0.651 | 1.232 | 0.733 |
| E1 round 2: G1 one-SE | 0.939 | 0.862 | 0.810 | 1.177 | 0.658 | 1.185 | 0.740 |
| G1 + fidelity cap | 0.966 | 0.870 | 0.805 | 1.130 | 0.648 | 1.220 | 0.746 |
| G2 argmin | 0.973 | 0.941 | 0.806 | 1.100 | 0.651 | 1.228 | 0.808 |
| G2 one-SE | 0.926 | 0.948 | 0.851 | 1.316 | 0.650 | 1.168 | 0.814 |
| **G2 + fidelity cap (round 3)** | 0.973 | 0.941 | 0.806 | 1.100 | 0.651 | 1.228 | 0.808 |
| G2 + fidelity cap + one-SE | 0.926 | 0.948 | 0.851 | 1.316 | 0.650 | 1.168 | 0.814 |

#### concrete (8 splits; ratio to S4, then resid_in / resid_out ratio to baseline)

| selection | resid_in | resid_out | ortho_in | ortho_out | locvar_in | resid_in / base | resid_out / base |
|---|---|---|---|---|---|---|---|
| S4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.548 | 0.833 |
| E1 round 1: G1 argmin | 1.008 | 0.901 | 1.631 | 1.395 | 1.168 | 1.559 | 0.751 |
| E1 round 2: G1 one-SE | 0.990 | 0.909 | 1.483 | 1.418 | 1.171 | 1.533 | 0.757 |
| G1 + fidelity cap | 0.869 | 0.934 | 1.310 | 1.403 | 1.139 | 1.345 | 0.778 |
| G2 argmin | 0.929 | 0.939 | 1.337 | 1.382 | 1.139 | 1.437 | 0.782 |
| G2 one-SE | 0.812 | 0.980 | 1.317 | 1.236 | 1.125 | 1.257 | 0.816 |
| **G2 + fidelity cap (round 3)** | 0.831 | 0.960 | 1.309 | 1.405 | 1.136 | 1.287 | 0.800 |
| G2 + fidelity cap + one-SE | 0.808 | 0.982 | 1.060 | 0.983 | 1.119 | 1.251 | 0.817 |

#### abalone (8 splits; ratio to S4, then resid_in / resid_out ratio to baseline)

| selection | resid_in | resid_out | ortho_in | ortho_out | locvar_in | resid_in / base | resid_out / base |
|---|---|---|---|---|---|---|---|
| S4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.390 | 0.536 |
| E1 round 1: G1 argmin | 0.985 | 0.912 | 1.039 | 0.952 | 1.112 | 1.369 | 0.488 |
| E1 round 2: G1 one-SE | 0.985 | 0.912 | 1.039 | 0.952 | 1.112 | 1.368 | 0.489 |
| G1 + fidelity cap | 0.925 | 0.956 | 1.060 | 0.969 | 1.109 | 1.285 | 0.512 |
| G2 argmin | 0.858 | 1.001 | 1.077 | 0.981 | 1.110 | 1.193 | 0.536 |
| G2 one-SE | 0.854 | 1.005 | 1.081 | 0.980 | 1.109 | 1.187 | 0.539 |
| **G2 + fidelity cap (round 3)** | 0.858 | 1.001 | 1.077 | 0.981 | 1.110 | 1.193 | 0.536 |
| G2 + fidelity cap + one-SE | 0.854 | 1.005 | 1.081 | 0.980 | 1.109 | 1.187 | 0.539 |

#### nutrition (8 splits; ratio to S4, then resid_in / resid_out ratio to baseline)

| selection | resid_in | resid_out | ortho_in | ortho_out | locvar_in | resid_in / base | resid_out / base |
|---|---|---|---|---|---|---|---|
| S4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.456 | 0.811 |
| E1 round 1: G1 argmin | 0.992 | 0.914 | 1.042 | 1.046 | 1.050 | 1.445 | 0.741 |
| E1 round 2: G1 one-SE | 0.957 | 0.930 | 1.042 | 1.047 | 1.053 | 1.393 | 0.754 |
| G1 + fidelity cap | 0.929 | 0.947 | 1.023 | 1.037 | 1.043 | 1.354 | 0.768 |
| G2 argmin | 0.882 | 0.984 | 1.023 | 1.040 | 1.054 | 1.284 | 0.798 |
| G2 one-SE | 0.843 | 1.003 | 1.022 | 1.065 | 1.058 | 1.227 | 0.813 |
| **G2 + fidelity cap (round 3)** | 0.863 | 0.994 | 1.014 | 1.024 | 1.044 | 1.257 | 0.806 |
| G2 + fidelity cap + one-SE | 0.835 | 1.011 | 1.014 | 1.047 | 1.047 | 1.216 | 0.820 |

#### analytical (reps 0-2, mean over reps; ratio to S4)

| selection | mse_eta1 | mse_eta2 | mse_eta3 | mse_eta4 | mse_eta5 | mse_eta6 | mse_eta12 | mse_eta34 | mse_others | resid_out |
|---|---|---|---|---|---|---|---|---|---|---|
| S4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| E1 round 1: G1 argmin | 1.005 | 1.005 | 1.003 | 1.002 | 0.982 | 0.976 | 1.002 | 0.997 | 1.001 | 0.961 |
| E1 round 2: G1 one-SE | 1.006 | 1.004 | 1.004 | 1.003 | 0.982 | 0.975 | 1.006 | 0.996 | 1.037 | 0.977 |
| G1 + fidelity cap | 1.006 | 1.004 | 1.004 | 1.003 | 0.980 | 0.975 | 1.005 | 0.996 | 1.028 | 0.973 |
| G2 argmin | 1.003 | 1.002 | 1.001 | 1.001 | 0.996 | 0.991 | 1.005 | 0.999 | 1.019 | 0.996 |
| G2 one-SE | 1.004 | 1.001 | 1.001 | 1.001 | 0.995 | 0.992 | 1.004 | 0.999 | 1.029 | 1.004 |
| **G2 + fidelity cap (round 3)** | 1.004 | 1.001 | 1.001 | 1.001 | 0.995 | 0.992 | 1.010 | 0.998 | 1.040 | 1.008 |
| G2 + fidelity cap + one-SE | 1.004 | 1.001 | 1.001 | 1.001 | 0.993 | 0.994 | 1.010 | 0.999 | 1.050 | 1.016 |
