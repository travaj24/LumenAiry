#!/bin/sh
# VERIFY-WP-B7c round 3 -- one shard of the ladder population.
# Usage: sh run_ladders.sh <tag> <optic> [optic ...]
set -e
cd "$(dirname "$0")"
tag="$1"; shift
for o in "$@"; do
  echo "=== $o ==="
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=/c/tmp/lum_vmb3 python v3ladder.py "$o" "ladder_${o}_${tag}" \
    > "log_ladder_${o}_${tag}.txt" 2>&1
  echo "done $o"
done
