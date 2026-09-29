#!/usr/bin/env bash
# Round 3: score every ensemble candidate for each (dataset, split, rho grid), 8 in parallel.
# usage: run_cands_r3.sh <rhos> <tag> <dataset> [...]   (analytical -> reps 0..2)
# -> dev/cache/*.npz, log dev/logs/cands_r3_<tag>.log
PY=D:/scientisttwo-lite/.conda/python.exe
DEV=D:/scientisttwo-lite/runs/treehfd-01/ideas/E1/dev
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
rhos=$1; tag=$2; shift 2
jobs=()
for ds in "$@"; do
  if [ "$ds" = analytical ]; then
    for r in 0 1 2; do jobs+=("analytical:$r split=0"); done
  else
    for s in 0 1 2 3 4 5 6 7; do jobs+=("$ds split=$s"); done
  fi
done
printf '%s\n' "${jobs[@]}" | xargs -P 8 -I{} sh -c "$PY $DEV/cands_r3.py {} rhos=$rhos tag=$tag" \
  >> "$DEV/logs/cands_r3_$tag.log" 2>&1
echo "done" >> "$DEV/logs/cands_r3_$tag.log"
