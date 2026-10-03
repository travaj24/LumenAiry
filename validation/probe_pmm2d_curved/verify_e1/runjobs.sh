#!/bin/sh
# usage: sh runjobs.sh JOBFILE NPAR   (each line: "script.py args...")
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_vcurved_e1
grep -v '^#' "$1" | grep -v '^$' | xargs -P "$2" -I{} sh -c 'timeout 1750 python {} > "logs/$(echo {} | tr " /" "__").log" 2>&1 || echo "FAIL {}"' >> "$1.log"
echo DONE >> "$1.log"
