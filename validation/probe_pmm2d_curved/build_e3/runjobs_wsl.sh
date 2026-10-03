#!/bin/bash
# WSL second build: outputs go to wsl/ (the probes dump next to themselves, so
# each job runs on a COPY of the probe directory)
cd /mnt/c/tmp/lum_curved_e3/validation/probe_pmm2d_curved/build_e3/wsl
cp ../_e3common.py ../f3_grads.py ../f4_events.py ../f5_degenerate.py ../f6_jit.py ../f7_stripe.py ../f2_parity.py .
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_curved_e3 LUM_TREE=/mnt/c/tmp/lum_curved_e3
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
cat ../$1 | xargs -P "$2" -I{} bash -c 'j="{}"; log=$(echo "$j" | sed "s/\.py//; s/ /_/g").log; timeout 1790 ~/lumvenv/bin/python $j > "$log" 2>&1; echo "done $j rc=$?"'
