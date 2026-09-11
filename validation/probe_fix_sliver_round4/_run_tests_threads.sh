#!/bin/sh
# ROUND 4 -- the eight sliver test files down the THREAD ladder, at one
# OpenBLAS kernel.
#
#   sh _run_tests_threads.sh win HASWELL 1 2 4 24
#   sh _run_tests_threads.sh wsl HASWELL 1 2 4 24
#
# 24 is this box's UNPINNED thread count (threadpoolctl reports 24 with all
# three variables removed, on both builds), so "t24" IS the unpinned arm for
# files that pin with os.environ.setdefault.  CI's unpinned lane resolves to
# 4, which is why 4 is on the ladder as well.
BUILD="$1"; KERN="$2"; shift 2
if [ "$BUILD" = "wsl" ]; then
  ROOT=/mnt/c/tmp/lum_sliver4; PY=$HOME/lumvenv/bin/python
else
  ROOT=/c/tmp/lum_sliver4; PY=python
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
for t in "$@"; do
  echo "=== $BUILD $KERN t$t ==="
  OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t \
  OPENBLAS_CORETYPE="$KERN" "$PY" -m pytest $FILES \
    -q --no-header -p no:randomly -p no:cacheprovider --tb=line -rf \
    > "validation/probe_fix_sliver_round4/logs/tests_${BUILD}_${KERN}_t${t}.log" 2>&1
  echo "exit $? -> $(tail -1 validation/probe_fix_sliver_round4/logs/tests_${BUILD}_${KERN}_t${t}.log)"
done
echo "TESTS_${BUILD}_${KERN}_THREADS_DONE"
