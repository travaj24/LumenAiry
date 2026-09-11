#!/bin/sh
# ROUND 4 -- the UNPINNED rung, Windows.  Usage:
#   sh _run_unpinned_win.sh <kernel> [case-prefix ...]
#
# PMM_PROBE_UNPINNED=1 makes g_fixtures REMOVE all three thread variables
# before numpy loads libopenblas (they are read once, at load).  On this
# 24-thread box that means OpenBLAS takes 24 threads on eigenproblems a few
# hundred wide, which is far slower than pinned, so this rung is run on a
# declared SUBSET of the table -- see the docstring of p4_decisions.main.
cd /c/tmp/lum_sliver4 || exit 1
k="$1"; shift
echo "=== win $k tUNPINNED $* ==="
PMM_PROBE_UNPINNED=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_CORETYPE="$k" python -u \
  validation/probe_fix_sliver_round4/p4_decisions.py "$@" \
  > "validation/probe_fix_sliver_round4/logs/p4_win_${k}_tUNPINNED.log" 2>&1
echo "   exit $?"
