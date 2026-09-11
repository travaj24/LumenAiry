#!/bin/sh
# ROUND 4 -- run a list of "<coretype>:<threads>" arms CONCURRENTLY on
# Windows.  Each pinned arm is a separate process using at most <threads>
# BLAS threads, so the arms are independent and the wall time is set by the
# box's free cores rather than by their number.  Usage:
#   sh _run_ladder_par_win.sh NEHALEM:1 NEHALEM:2 SANDYBRIDGE:4 ...
cd /c/tmp/lum_sliver4 || exit 1
for a in "$@"; do
  k=${a%%:*}; t=${a##*:}
  (
    OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t \
    OPENBLAS_CORETYPE="$k" python -u \
      validation/probe_fix_sliver_round4/p4_decisions.py \
      > "validation/probe_fix_sliver_round4/logs/p4_win_${k}_t${t}.log" 2>&1
    echo "win $k t$t exit $?"
  ) &
done
wait
echo LADDER_PAR_WIN_DONE
