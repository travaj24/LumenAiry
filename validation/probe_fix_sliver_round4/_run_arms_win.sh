#!/bin/sh
# ROUND 4 -- run the decision table on one build's kernels, in order.
# Usage: sh _run_arms_win.sh HASWELL PRESCOTT ...
cd /c/tmp/lum_sliver4 || exit 1
for k in "$@"; do
  echo "=== win $k ==="
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_CORETYPE="$k" python -u \
    validation/probe_fix_sliver_round4/p4_decisions.py \
    > "validation/probe_fix_sliver_round4/_p4_win_$k.log" 2>&1
  echo "   exit $?"
done
