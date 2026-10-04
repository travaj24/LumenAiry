#!/bin/bash
# PRE vs POST bytes on Windows.  Usage: bash run_b.sh [b_bytes] [b2_bytes_jax]
cd /c/tmp/lum_symgrad_verify/validation/probe_jax_symgrad_verify
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for p in "$@"; do
  LUM_TREE=C:/tmp/lum_symgrad_verify VTAG=post python $p.py > ${p}_post_win.log 2>&1
  LUM_TREE=C:/tmp/lum_symgrad_verify_base VTAG=pre python $p.py > ${p}_pre_win.log 2>&1
done
