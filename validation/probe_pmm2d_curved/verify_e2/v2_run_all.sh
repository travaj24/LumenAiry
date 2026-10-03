#!/bin/bash
# the brute-force oracle over every case (3 at a time)
cd /c/tmp/lum_vcurved_e2/validation/probe_pmm2d_curved/verify_e2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_vcurved_e2
r() { timeout 1750 python -u v2_brute.py $1 $2 $3 > logs/v2f_$1_M$2_n$3.txt 2>&1; }
r circ_sin 4 20 & r tangent 4 20 & r graze3 4 20 & wait
r circ_sin 4 28 & r graze6 4 20 & r tan_x 4 20 & wait
r sin_circ 4 20 & r circ_sin_tau 4 20 & r sinx_siny 5 24 & wait
echo ALLDONE
