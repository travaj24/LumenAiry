#!/bin/bash
# run a job list (one "ARGS..." line per job: SCRIPT ARGS) with N parallel
# workers; every job pinned to this tree, BLAS threads 1 on the command line
LIST=$1; N=${2:-3}; PY=${PY:-python}
cd /c/tmp/lum_curved_e3b/validation/probe_pmm2d_curved/build_e3r2
grep -v '^#' "$LIST" | grep -v '^$' | xargs -P "$N" -I{} bash -c \
  "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved_e3b $PY {} 2>&1 | grep -v Warning | tail -6"
