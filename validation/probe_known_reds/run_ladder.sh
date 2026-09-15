#!/bin/bash
# Kernel ladder over a pytest selection.
#   run_ladder.sh <outdir> <pytest args...>
# OPENBLAS_CORETYPE in {HASWELL, NEHALEM, KATMAI, SANDYBRIDGE} x threads {1, 4}.
# ZEN aliases Haswell on this box; SKYLAKEX crashes and is not used.
cd /c/tmp/lum_reds || exit 1
OUT="$1"; shift
mkdir -p "$OUT"
for CT in HASWELL NEHALEM KATMAI SANDYBRIDGE; do
  for NT in 1 4; do
    tag="${CT}_t${NT}"
    OPENBLAS_CORETYPE=$CT OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT \
      MKL_NUM_THREADS=$NT PYTHONPATH=/c/tmp/lum_reds \
      timeout 3600 python -m pytest "$@" --capture=sys -p no:randomly -q \
      > "$OUT/$tag.log" 2>&1
    echo "$tag: $(tail -4 "$OUT/$tag.log" | grep -oE '[0-9]+ (passed|failed|skipped|error)[a-z]*' | tr '\n' ' ')"
  done
done
