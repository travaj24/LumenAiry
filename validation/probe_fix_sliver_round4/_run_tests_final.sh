#!/bin/sh
# ROUND 4 -- the FINAL sweep: the eight sliver test files under one build's
# OpenBLAS kernels, with per-test durations captured.
#
#   sh _run_tests_final.sh win HASWELL ZEN PRESCOTT SANDYBRIDGE NEHALEM
#   sh _run_tests_final.sh wsl HASWELL ZEN PRESCOTT SANDYBRIDGE NEHALEM
BUILD="$1"
shift
if [ "$BUILD" = "wsl" ]; then
  ROOT=/mnt/c/tmp/lum_sliver4
  PY=$HOME/lumvenv/bin/python
else
  ROOT=/c/tmp/lum_sliver4
  PY=python
fi
cd "$ROOT" || exit 1
mkdir -p validation/probe_fix_sliver_round4/logs
FILES="tests/unit/test_fix_pmmstack_sliver_walls.py
tests/unit/test_fix_pmmstack_sliver_walls_round2.py
tests/unit/test_fix_pmmstack_sliver_round3.py
tests/unit/test_fix_pmmstack_sliver_round4.py
tests/unit/test_verify_pmmstack_sliver_walls.py
tests/unit/test_verify_pmmstack_sliver_round2.py
tests/unit/test_verify_pmmstack_sliver_round3.py
tests/unit/test_m1_conditioning_guard.py"
for k in "$@"; do
  echo "=== $BUILD $k ==="
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_CORETYPE="$k" "$PY" -m pytest $FILES \
    -q --no-header -p no:randomly -p no:cacheprovider --tb=line -rf \
    --durations=0 \
    > "validation/probe_fix_sliver_round4/logs/tests_${BUILD}_${k}.log" 2>&1
  echo "exit $? -> $(grep -E 'passed|failed' validation/probe_fix_sliver_round4/logs/tests_${BUILD}_${k}.log | tail -1)"
done
echo "FINAL_TESTS_${BUILD}_DONE"
