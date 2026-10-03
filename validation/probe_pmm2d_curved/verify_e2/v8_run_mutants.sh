#!/bin/bash
# run the E2 unit file (and this verifier's decision file if present) under each mutant
cd /c/tmp/lum_vcurved_e2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="C:/tmp/lum_vcurved_e2;C:/tmp/lum_vcurved_e2/validation/probe_pmm2d_curved/verify_e2"
FILES=${FILES:-tests/unit/test_pmm2d_staggered_curved_e2.py}
run() { V8_MUT=$1 timeout 1700 python -m pytest $FILES -p v8_mut_plugin --capture=sys -p no:randomly -q -p no:cacheprovider 2>&1 | grep -E "passed|failed|FAILED|ERROR" > validation/probe_pmm2d_curved/verify_e2/logs/v8_pytest_${TAG:-e2}_$1.txt; }
for grp in "h_no_measure h_no_offdiag e_no_factor" "sqrt_off tang_missed maps_ignored" "no_ride ride_below no_qmatch"; do
  for m in $grp; do run $m & done; wait
done
echo DONE
