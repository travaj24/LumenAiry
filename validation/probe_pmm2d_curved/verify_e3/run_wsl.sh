#!/bin/bash
# WSL second build (jax 0.10, CPython 3.12): run a job list with N parallel
# workers; probes dump <name>_wsl.json next to themselves (the _ve3 BUILD tag).
#   bash run_wsl.sh JOBFILE N [TIMEOUT_S]
V=/mnt/c/tmp/lum_vcurved_e3/validation/probe_pmm2d_curved/verify_e3
cd "$V"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/c/tmp/lum_vcurved_e3 LUM_TREE=/mnt/c/tmp/lum_vcurved_e3
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
mkdir -p logs
TO=${3:-1780}
export TO
cat "$1" | xargs -P "$2" -I{} bash -c 'j="{}"; log=logs/$(echo "$j" | sed "s/\.py//; s/ /_/g; s#/#_#g")_wsl.log; timeout $TO ~/lumvenv/bin/python $j > "$log" 2>&1; echo "done $j rc=$?"'
