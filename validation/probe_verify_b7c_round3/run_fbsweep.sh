#!/bin/sh
set -e
cd "$(dirname "$0")"
tag="$1"; shift
for o in "$@"; do
  echo "=== $o ==="
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=/c/tmp/lum_vmb3 python v3fbsweep.py "$o" "fbsweep_${o}_${tag}.json" \
    > "log_fbsweep_${o}_${tag}.txt" 2>&1
  echo "done $o"
done
