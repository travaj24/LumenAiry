#!/bin/sh
# ROUND 4 -- the PINNED rungs of the thread ladder on Windows: every requested
# coretype in $@ crossed with threads {1, 2, 4}.  The unpinned rung is run
# separately (_run_unpinned_*.sh) because on this 24-thread box an unpinned
# OpenBLAS is >30x slower on these small eigenproblems than a pinned one.
cd /c/tmp/lum_sliver4 || exit 1
for k in "$@"; do
  for t in 1 2 4; do
    echo "=== win $k t$t ==="
    OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t \
    OPENBLAS_CORETYPE="$k" python -u \
      validation/probe_fix_sliver_round4/p4_decisions.py \
      > "validation/probe_fix_sliver_round4/logs/p4_win_${k}_t${t}.log" 2>&1
    echo "   exit $?"
  done
done
echo LADDER_WIN_DONE
