| dataset | n_test | resid_out ratio (point) | 95% CI | GT-LOCO wins | verdict | ortho_out ratio (point) | 95% CI | wins | verdict |
|---|---|---|---|---|---|---|---|---|---|
| abalone | 836 | 0.524 | [0.422, 0.653] | 1.000 | significant gain | 2.619 | [0.233, 2.965] | 0.479 | no significant gain (CI includes 1) |
| airfoil | 301 | 0.844 | [0.705, 0.983] | 0.991 | significant gain | 1.130 | [0.827, 1.231] | 0.325 | no significant gain (CI includes 1) |
| bike | 3476 | 0.998 | [0.995, 1.001] | 0.901 | no significant gain (CI includes 1) | 1.000 | [0.978, 1.012] | 0.615 | no significant gain (CI includes 1) |
| housing | 4128 | 0.910 | [0.858, 0.967] | 0.998 | significant gain | 0.989 | [0.850, 1.153] | 0.574 | no significant gain (CI includes 1) |
| concrete | 206 | 0.839 | [0.758, 0.925] | 1.000 | significant gain | 0.876 | [0.235, 1.288] | 0.641 | no significant gain (CI includes 1) |
| nutrition | 456 | 0.753 | [0.671, 0.855] | 1.000 | significant gain | 0.739 | [0.618, 1.318] | 0.802 | no significant gain (CI includes 1) |
| parkinson | 1175 | 1.002 | [0.976, 1.035] | 0.461 | no significant gain (CI includes 1) | 0.998 | [0.981, 1.006] | 0.716 | no significant gain (CI includes 1) |
| powerplant | 1914 | 0.964 | [0.938, 0.989] | 0.997 | significant gain | n/a | n/a | n/a | undefined (no interaction above 1% variance) |
| superconduct | 4253 | 0.998 | [0.986, 1.010] | 0.587 | no significant gain (CI includes 1) | n/a | n/a | n/a | undefined (no interaction above 1% variance) |

| geometric mean over datasets | point | CI (test rows only) | CI (datasets only) | CI (both levels) |
|---|---|---|---|---|
| resid_out (9 datasets) | 0.855 | [0.825, 0.888] | [0.734, 0.957] | [0.734, 0.960] |
| ortho_out (7 datasets) | 1.095 | [0.736, 1.173] | [0.870, 1.482] | [0.709, 1.203] |
| resid_out excl. abalone, airfoil | 0.919 | [0.897, 0.943] | [0.852, 0.984] | |