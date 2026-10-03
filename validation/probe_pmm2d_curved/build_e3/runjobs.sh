#!/bin/bash
# usage: runjobs.sh jobs.txt NPAR  -- one log per job: <job words joined by _>.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH:-C:/tmp/lum_curved_e3}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
PY=${PY:-python}
cat "$1" | xargs -P "$2" -I{} bash -c 'j="{}"; log=$(echo "$j" | sed "s/\.py//; s/ /_/g").log; timeout 1790 '"$PY"' $j > "$log" 2>&1; echo "done $j rc=$?"'
