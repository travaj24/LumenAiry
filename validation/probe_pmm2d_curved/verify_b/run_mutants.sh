#!/bin/bash
# usage: run_mutants.sh <testfile-relative-path> <tag>   (runs every mutant)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for m in $(python /c/tmp/lum_vcurved_b/validation/probe_pmm2d_curved/verify_b/v8_mutants.py list); do
  d=/c/tmp/vcurved_b_mut/$m
  cp /c/tmp/lum_vcurved_b/$1 $d/$1 2>/dev/null
  (cd $d && PYTHONPATH=C:/tmp/vcurved_b_mut/$m python -m pytest $1 --capture=sys -p no:randomly -q -n 6 -p no:cacheprovider 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed" | sed "s/^/[$m] /")
done
echo MUTANTS_DONE $2
