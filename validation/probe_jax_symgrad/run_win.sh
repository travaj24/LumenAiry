#!/bin/bash
# Windows build (git bash): run probe scripts against TREE with tag TAG.
# usage: bash run_win.sh TREE TAG SCRIPT [SCRIPT ...]
# (PRE tree = `git archive 5ea82b44 lumenairy | tar -x -C DIR`.)
TREE=$1; TAG=$2; shift 2
cd "$(dirname "$0")"
for s in "$@"; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LUM_TREE="$TREE" \
    PYTHONPATH="$TREE;." SG_TAG="$TAG" python "$s" 2>&1 | grep -v Warn
done
