#!/bin/bash
# ROUND 2, D1 per arm: the loss-ladder DECISIONS (ladder A and the GAIN ladder
# C), PRE and POST, across the WSL kernel ladder.  --fast skips ladder B,
# whose 39 healthy-lossy rows are measured once at full length on Haswell/t1.
cd /mnt/c/tmp/lum_bor2 || exit 1
for K in HASWELL NEHALEM PRESCOTT SANDYBRIDGE; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_CORETYPE=$K \
    PYTHONPATH=/mnt/c/tmp/lum_bor2 ~/lumvenv/bin/python -u validation/probe_fix_bor_round2/r2_loss_ladder.py --fast \
    > validation/probe_fix_bor_round2/runs/r2fast_POST_wsl_${K}_t1.log 2>&1
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_CORETYPE=$K \
    PYTHONPATH=/mnt/c/tmp/lum_bor2_pre ~/lumvenv/bin/python -u validation/probe_fix_bor_round2/r2_loss_ladder.py --fast \
    > validation/probe_fix_bor_round2/runs/r2fast_PRE_wsl_${K}_t1.log 2>&1
  echo "$K POST: $(grep 'A refused on' validation/probe_fix_bor_round2/runs/r2fast_POST_wsl_${K}_t1.log) | $(grep 'C refused on' validation/probe_fix_bor_round2/runs/r2fast_POST_wsl_${K}_t1.log)"
  echo "$K PRE : $(grep 'A refused on' validation/probe_fix_bor_round2/runs/r2fast_PRE_wsl_${K}_t1.log) | $(grep 'C refused on' validation/probe_fix_bor_round2/runs/r2fast_PRE_wsl_${K}_t1.log)"
done
