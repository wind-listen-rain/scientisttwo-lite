#!/usr/bin/env bash
# Round 2: one-SE selection vs plain argmin vs each rho mode, and fixed-kappa orthogonality,
# on S4's diagnostic splits (split 0 = benchmark split).
# usage: ablate_r2.sh <dataset> [splits...]   -> dev/logs/ablate_r2_<dataset>.log
# Held-out numbers are for diagnosis only; the method never sees them.
PY=D:/scientisttwo-lite/.conda/python.exe
DEV=D:/scientisttwo-lite/runs/treehfd-01/ideas/E1/dev
ds=$1
shift
splits=${*:-0 1 2 3 4 5 6 7}
log=$DEV/logs/ablate_r2_$ds.log
: > "$log"
for s in $splits; do
  echo "=== $ds split $s" >> "$log"
  "$PY" "$DEV/dev_e1.py" "$ds" "split=$s" ablate ortho >> "$log" 2>&1
done
echo "done" >> "$log"
