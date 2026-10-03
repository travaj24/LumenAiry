#!/bin/bash
# runs one job line per python process, at most 3 at a time
cd /c/tmp/lum_vcurved_e2/validation/probe_pmm2d_curved/verify_e2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_vcurved_e2
cat "$1" | xargs -P 3 -I{} bash -c 'n=$(echo "{}" | tr " " "_"); timeout 1800 python {} > logs/$n.log 2>&1; echo "done rc=$? {}"'
