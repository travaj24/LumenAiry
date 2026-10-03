#!/bin/bash
# runjobs.sh <jobsfile> <parallel> : each line = args of v4_incident.py
cd /c/tmp/lum_vcurved_c/validation/probe_pmm2d_curved/verify_c
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_vcurved_c
mkdir -p logs
cat "$1" | xargs -P "$2" -I{} bash -c 'a="{}"; n=$(echo $a | tr " " "_"); timeout 7000 python /c/tmp/lum_vcurved_c/validation/probe_pmm2d_curved/verify_c/'"${3:-v4_incident.py}"' $a > logs/$n.log 2>&1; echo "done $a rc=$?"'
