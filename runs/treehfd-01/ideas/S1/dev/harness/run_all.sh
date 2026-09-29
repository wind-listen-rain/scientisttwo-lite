#!/bin/sh
# Dev only: official self-test, then the paired (seeded-baseline) reference and the ablations, sequentially.
PY=D:/scientisttwo-lite/.conda/python.exe
H=D:/scientisttwo-lite/bench/harness.py
S1=D:/scientisttwo-lite/runs/treehfd-01/ideas/S1
cd $S1
$PY $H --method method.py --mode subset --out subset_selftest.json > subset_selftest.log 2>&1
for v in baseline_seeded v_step1only v_step2none v_prev v_step12eb; do
  $PY $H --method dev/harness/$v.py --mode subset --out dev/harness/$v.json > dev/harness/$v.log 2>&1
done
echo ALLDONE > dev/harness/done.flag
