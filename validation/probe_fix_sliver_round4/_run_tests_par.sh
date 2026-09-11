#!/bin/sh
# ROUND 4 -- the eight sliver test files under several OpenBLAS kernels AT
# ONCE.  Each arm pins one BLAS thread, so the arms are independent and the
# wall time is set by the box's free cores rather than by their number.
#   sh _run_tests_par.sh win HASWELL PRESCOTT NEHALEM SANDYBRIDGE
BUILD="$1"; shift
if [ "$BUILD" = "wsl" ]; then
  ROOT=/mnt/c/tmp/lum_sliver4; PY=$HOME/lumvenv/bin/python
else
  ROOT=/c/tmp/lum_sliver4; PY=python
fi
cd "$ROOT" || exit 1
FILES="tests/unit/test_fix_pmmstack_sliver_walls.py
tests/unit/test_fix_pmmstack_sliver_walls_round2.py
tests/unit/test_fix_pmmstack_sliver_round3.py
tests/unit/test_fix_pmmstack_sliver_round4.py
tests/unit/test_verify_pmmstack_sliver_walls.py
tests/unit/test_verify_pmmstack_sliver_round2.py
tests/unit/test_verify_pmmstack_sliver_round3.py
tests/unit/test_m1_conditioning_guard.py"
for k in "$@"; do
  (
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_CORETYPE="$k" "$PY" -m pytest $FILES \
      -q --no-header -p no:randomly -p no:cacheprovider --tb=line -rf \
      > "validation/probe_fix_sliver_round4/logs/tests_${BUILD}_${k}.log" 2>&1
    echo "$BUILD $k exit $?"
  ) &
done
wait
echo "TESTS_PAR_${BUILD}_DONE"
