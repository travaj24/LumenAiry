#!/bin/sh
cd /mnt/c/tmp/lum_sliver4 || exit 1
for k in "$@"; do
  echo "=== wsl $k ==="
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_CORETYPE="$k" ~/lumvenv/bin/python -u \
    validation/probe_fix_sliver_round4/p4_decisions.py \
    > "validation/probe_fix_sliver_round4/_p4_wsl_$k.log" 2>&1
  echo "   exit $?"
done
