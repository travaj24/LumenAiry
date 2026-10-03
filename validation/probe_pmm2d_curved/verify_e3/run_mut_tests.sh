#!/bin/bash
# run the E3 unit file in a mutant scratch tree: bash run_mut_tests.sh TREE
T=$1
cd /c/tmp/$T
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/$T
timeout 3000 python -m pytest tests/unit/test_pmm2d_staggered_curved_e3.py --capture=sys -p no:randomly -q -p no:cacheprovider 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed" > /c/tmp/lum_vcurved_e3/validation/probe_pmm2d_curved/verify_e3/logs/mut_$T.txt
