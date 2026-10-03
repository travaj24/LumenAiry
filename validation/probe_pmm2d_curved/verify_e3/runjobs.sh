#!/bin/bash
# run a job list with N parallel workers, single-threaded BLAS + XLA (Windows build)
cd /c/tmp/lum_vcurved_e3/validation/probe_pmm2d_curved/verify_e3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_vcurved_e3
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
mkdir -p logs
cat "$1" | xargs -P "$2" -I{} bash -c 'j="{}"; log=logs/$(echo "$j" | sed "s/\.py//; s/ /_/g")_win.log; timeout ${3:-1780} python $j > "$log" 2>&1; echo "done $j rc=$?"'
