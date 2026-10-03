#!/bin/sh
# the verifier's mutation matrix against the Phase D unit tests (+ the
# Phase B verifier file for the D-1 tolerance); one log per kind
cd /c/tmp/lum_vcurved_d
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="C:/tmp/lum_vcurved_d;C:/tmp/lum_vcurved_d/validation/probe_pmm2d_curved/verify_d"
OUT=validation/probe_pmm2d_curved/verify_d/mut_logs
mkdir -p $OUT
for k in $*; do
  VD_MUT=$k python -m pytest -p vd_mutplugin --capture=sys -p no:randomly -q -n 3 \
    tests/unit/test_pmm2d_staggered_curved_d.py tests/unit/test_verify_pmm2d_curved_b.py tests/unit/test_verify_pmm2d_curved_d.py \
    -p no:cacheprovider > $OUT/$k.txt 2>&1
  echo "$k: $(tail -1 $OUT/$k.txt)" >> $OUT/summary.txt
done
