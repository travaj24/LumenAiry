#!/bin/bash
# runs one c_ladders job per line of $1, $2 at a time, BLAS pinned
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved_c
cat "$1" | xargs -P "$2" -I{} bash -c 'a="{}"; python c_ladders.py $a > "log_$(echo $a | tr " " "_").txt" 2>&1'
