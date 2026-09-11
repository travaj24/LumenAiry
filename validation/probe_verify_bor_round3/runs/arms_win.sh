#!/usr/bin/env bash
# Windows py3.14 arms.  Every command carries the three thread variables and
# OPENBLAS_CORETYPE on the COMMAND LINE, and the probe reads the LOADED kernel
# back out of threadpoolctl into its JSON.  PYTHONPATH pins the tree.
set -u
HERE="C:/tmp/lum_vbor3"
PROBE="$1"; shift
ARGS="$*"
cd "$HERE" || exit 1
for CT in HASWELL SANDYBRIDGE KATMAI; do
  for TH in 1 4; do
    if [ "$TH" = "4" ] && [ "$CT" != "HASWELL" ]; then continue; fi
    LOG="validation/probe_verify_bor_round3/runs/$(basename "$PROBE" .py)_win_${CT}_t${TH}.log"
    echo "=== win $CT t$TH -> $LOG"
    OMP_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH MKL_NUM_THREADS=$TH \
      OPENBLAS_CORETYPE=$CT PYTHONPATH="C:\\tmp\\lum_vbor3" \
      LUM_EXPECT_ROOT="C:/tmp/lum_vbor3" \
      python "validation/probe_verify_bor_round3/$PROBE" $ARGS > "$LOG" 2>&1
    echo "   exit $?  $(grep -c WROTE "$LOG") json"
  done
done
