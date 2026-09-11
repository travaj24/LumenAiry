#!/usr/bin/env bash
# ONE ARM of the round-3 matrix on the WSL build (py3.12), invoked as
#   wsl -e bash -lc '/mnt/c/tmp/lum_bor3/validation/probe_fix_bor_round3/runs/arm_wsl.sh HASWELL 1'
set -u
CT="${1:-HASWELL}"
TH="${2:-1}"
ROOT=/mnt/c/tmp/lum_bor3
OUT="$ROOT/validation/probe_fix_bor_round3/runs"
mkdir -p "$OUT"
TAG="wsl_${CT}_t${TH}"
export OMP_NUM_THREADS="$TH" OPENBLAS_NUM_THREADS="$TH" MKL_NUM_THREADS="$TH"
export OPENBLAS_CORETYPE="$CT" PYTHONPATH="$ROOT"
cd "$ROOT" || exit 1
PY=~/lumvenv/bin/python
$PY -c "
import lumenairy, threadpoolctl, sys
print('lumenairy.__file__ =', lumenairy.__file__)
print('python =', sys.version.split()[0])
for d in threadpoolctl.threadpool_info():
    if d.get('user_api') == 'blas':
        print('LOADED kernel =', d.get('architecture'), 'threads =', d.get('num_threads'))
" 2>&1 | tee "$OUT/pin_$TAG.log"
FILES=$(cat "$OUT/files.txt" | tr "
" " ")
$PY -m pytest $FILES -q -p no:randomly 2>&1 | tee "$OUT/set_$TAG.log" | tail -5
