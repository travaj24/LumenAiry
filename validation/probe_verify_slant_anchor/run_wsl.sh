#!/bin/sh
# WSL arm driver.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P="$1"; shift
for T in /mnt/c/tmp/lum_vslant /mnt/c/tmp/lum_vslant_pre /mnt/c/tmp/lum_vslant_rev; do
  echo "=== WSL ARM $T -- $P ==="
  LUM_ARM_TREE="$T" ~/lumvenv/bin/python "/mnt/c/tmp/lum_vslant/validation/probe_verify_slant_anchor/$P" "$@" 2>&1 | tail -70
  echo "--- exit $? ---"
done
