#!/bin/sh
# ROUND 4 -- the CI-EQUIVALENT UNPINNED arm, WSL.  Usage:
#   sh _run_ci_emul_wsl.sh HASWELL PRESCOTT ...
#
# PMM_PROBE_UNPINNED=1 removes all three thread variables before numpy loads
# libopenblas, and PMM_PROBE_AFFINITY=4 narrows the process to four CPUs
# first.  An unpinned OpenBLAS sizes its pool from the CPUs it can SEE, so
# this is CI's configuration exactly: Linux, four cores, no pin -- measured
# back as num_threads = 4 -- rather than this box's twenty-four, which is an
# extreme neither CI nor any user runs and which does not finish.
cd /mnt/c/tmp/lum_sliver4 || exit 1
for k in "$@"; do
  (
    PMM_PROBE_UNPINNED=1 PMM_PROBE_AFFINITY=4 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_CORETYPE="$k" ~/lumvenv/bin/python -u \
      validation/probe_fix_sliver_round4/p4_decisions.py \
      > "validation/probe_fix_sliver_round4/logs/p4_wsl_${k}_tUNPINNED.log" 2>&1
    echo "wsl $k tUNPINNED exit $?"
  ) &
done
wait
echo CI_EMUL_WSL_DONE
