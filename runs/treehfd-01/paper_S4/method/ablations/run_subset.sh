#!/bin/sh
# Subset self-tests of every ablation (sequential, so fit_s is not skewed), a re-run of
# method.py to confirm the COMMON_KAPPA switch leaves it unchanged, then selection diagnostics.
PY=D:/scientisttwo-lite/.conda/python.exe
H=D:/scientisttwo-lite/bench/harness.py
A=D:/scientisttwo-lite/runs/treehfd-01/ideas/S4/ablations
cd "$A"
for n in no_shrinkage_direct_solver_only no_lattice_prior_ridge_zero_only no_ensemble_level_kappa_step soft_orthogonality_rows zero_rule_only; do
  $PY $H --method $A/method_$n.py --mode subset --out $A/subset_$n.json > $A/logs/subset_$n.log 2>&1
done
$PY $H --method $A/../method.py --mode subset --out $A/logs/subset_method_recheck.json > $A/logs/subset_method_recheck.log 2>&1
for f in ../method.py method_no_shrinkage_direct_solver_only.py method_no_lattice_prior_ridge_zero_only.py method_no_ensemble_level_kappa_step.py method_soft_orthogonality_rows.py method_zero_rule_only.py; do
  $PY diag_selection.py $f
done > logs/diag_selection_subset.log 2>&1
echo ALL_DONE >> logs/diag_selection_subset.log
