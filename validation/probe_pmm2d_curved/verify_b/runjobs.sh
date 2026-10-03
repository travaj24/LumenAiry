#!/bin/bash
# usage: runjobs.sh <script.py> <jobsfile> <parallel> ; each line = args
cd /c/tmp/lum_vcurved_b/validation/probe_pmm2d_curved/verify_b
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_vcurved_b
cat "$2" | xargs -P "$3" -L 1 python "$1"
echo ALL_DONE "$2"
