#!/bin/sh
# Windows arm driver: run one probe on all three trees, thread caps pinned.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P="$1"; shift
for T in C:/tmp/lum_vslant C:/tmp/lum_vslant_pre C:/tmp/lum_vslant_rev; do
  echo "=== WIN ARM $T -- $P ==="
  LUM_ARM_TREE="$T" python "C:/tmp/lum_vslant/validation/probe_verify_slant_anchor/$P" "$@" 2>&1 | tail -70
  echo "--- exit $? ---"
done
