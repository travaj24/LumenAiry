#!/bin/bash
# Kernel ladder for the M2 T3-1 red.  Usage: ladder_t31.sh <nodeid-or-k-expr> <outdir>
cd /c/tmp/lum_reds
K="$1"; OUT="$2"; mkdir -p "$OUT"
for CT in HASWELL NEHALEM KATMAI SANDYBRIDGE; do
  for NT in 1 4; do
    tag="${CT}_t${NT}"
    OPENBLAS_CORETYPE=$CT OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT \
      PYTHONPATH=/c/tmp/lum_reds timeout 1800 python -m pytest $K --capture=sys -p no:randomly -q \
      > "$OUT/$tag.log" 2>&1
    echo "$tag: $(grep -oE '[0-9]+ (passed|failed)|[0-9]+ failed|no tests ran|error' "$OUT/$tag.log" | tr '\n' ' ')"
  done
done
