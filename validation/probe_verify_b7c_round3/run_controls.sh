#!/bin/sh
# VERIFY-WP-B7c round 3 -- the converged reference, the two ladders and the
# lattice A/B (claims 5 and 6).
set -e
cd "$(dirname "$0")"
tag="${1:-win}"
py="${V3_PY:-python}"
for m in converged gridladder windowladder ab fold; do
  echo "=== $m ==="
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH="${V3_ROOT:-/c/tmp/lum_vmb3}" "$py" v3control.py "$m" \
    "control_${m}_${tag}.json" > "log_control_${m}_${tag}.txt" 2>&1
  echo "done $m"
done
