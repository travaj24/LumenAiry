#!/bin/bash
# WSL build: run probe scripts against TREE (a /mnt/c path) with tag TAG,
# SERIALLY (jax 0.10.2 under WSL core-dumped under parallel load before).
# usage: wsl -e bash /mnt/c/tmp/lum_symgrad/validation/probe_jax_symgrad/run_wsl.sh TREE TAG SCRIPT [SCRIPT ...]
TREE=$1; TAG=$2; shift 2
cd "$(dirname "$0")"
for s in "$@"; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LUM_TREE="$TREE" \
    PYTHONPATH="$TREE:." SG_TAG="$TAG" ~/lumvenv/bin/python "$s" 2>&1 | grep -v Warn
done
