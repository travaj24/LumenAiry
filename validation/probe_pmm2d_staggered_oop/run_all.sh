#!/usr/bin/env bash
# Run every probe script.  Usage:  bash validation/probe_pmm2d_staggered_oop/run_all.sh [m1 m2 ...]
set -u
W=/c/tmp/lum_aniso_oop
D=$W/validation/probe_pmm2d_staggered_oop
export PYTHONPATH=$W OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$D/logs"
for s in "$@"; do
  f=$(ls "$D"/${s}_*.py 2>/dev/null | head -n 1)
  [ -z "$f" ] && { echo "no script for $s"; continue; }
  echo "=== $s -> $f"
  python "$f" > "$D/logs/$s.log" 2>&1
  echo "   exit=$?  ($(wc -l < "$D/logs/$s.log") lines)"
done
