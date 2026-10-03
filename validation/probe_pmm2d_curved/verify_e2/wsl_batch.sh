#!/bin/bash
# second build (WSL Ubuntu, ~/lumvenv, BLAS pinned): the lead verifier's kernel probes + test files
cd /mnt/c/tmp/lum_vcurved_e2/validation/probe_pmm2d_curved/verify_e2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_vcurved_e2
PY=~/lumvenv/bin/python
mkdir -p logs
$PY v2_brute.py sinx_siny 4 20 > logs/wsl_v2_sxsy.txt 2>&1 &
$PY v2b_tangent_ladder.py sinx_siny 4 > logs/wsl_v2b_sxsy.txt 2>&1 &
$PY v2d_graze_sweep.py > logs/wsl_v2d.txt 2>&1 &
wait
$PY v2c_circle_pairs.py > logs/wsl_v2c.txt 2>&1 &
$PY v8_mutations.py builder 5 > logs/wsl_v8_builder5.txt 2>&1 &
(cd /mnt/c/tmp/lum_vcurved_e2 && $PY -m pytest tests/unit/test_verify_pmm2d_curved_e2.py tests/unit/test_pmm2d_staggered_curved_e2.py --capture=sys -p no:randomly -q -p no:cacheprovider > validation/probe_pmm2d_curved/verify_e2/logs/wsl_pytest_e2.txt 2>&1) &
wait
echo WSL_BATCH_DONE
