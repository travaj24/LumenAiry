#!/bin/bash
# WSL build: run a job list (SCRIPT ARGS per line) with N workers, pinned to
# the tree (LUM_TREE / PYTHONPATH), BLAS threads 1 on the command line.
# usage (from Windows): wsl bash /mnt/c/.../run_wsl.sh LIST N [TREE]
LIST=$1; N=${2:-3}; TREE=${3:-/mnt/c/tmp/lum_curved_e3b}
cd /mnt/c/tmp/lum_curved_e3b/validation/probe_pmm2d_curved/build_e3r2
grep -v '^#' "$LIST" | grep -v '^$' | xargs -P "$N" -I{} bash -c \
  "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LUM_TREE=$TREE PYTHONPATH=$TREE:. E3R2_TAG=\$E3R2_TAG R2_NAME=\$R2_NAME ~/lumvenv/bin/python {} 2>&1 | grep -v Warning | tail -6"
