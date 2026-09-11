#!/usr/bin/env bash
# WSL py3.12 arms, driven from the Windows side.  Same rule: the thread
# variables and OPENBLAS_CORETYPE go on the command line, PYTHONPATH pins the
# tree, and the probe reads the LOADED kernel back out of threadpoolctl.
set -u
HERE="C:/tmp/lum_vbor3"
PROBE="$1"; shift
ARGS="$*"
cd "$HERE" || exit 1
for CT in HASWELL SANDYBRIDGE KATMAI; do
  for TH in 1 4; do
    if [ "$TH" = "4" ] && [ "$CT" != "HASWELL" ]; then continue; fi
    LOG="validation/probe_verify_bor_round3/runs/$(basename "$PROBE" .py)_wsl_${CT}_t${TH}.log"
    echo "=== wsl $CT t$TH -> $LOG"
    wsl -e bash -lc "cd /mnt/c/tmp/lum_vbor3 && OMP_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_CORETYPE=$CT PYTHONPATH=/mnt/c/tmp/lum_vbor3 LUM_EXPECT_ROOT=/mnt/c/tmp/lum_vbor3 ~/lumvenv/bin/python validation/probe_verify_bor_round3/$PROBE $ARGS" > "$LOG" 2>&1
    echo "   exit $?  $(grep -c WROTE "$LOG") json"
  done
done
