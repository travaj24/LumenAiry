#!/usr/bin/env bash
# ONE ARM of the round-3 matrix on the WINDOWS build (py3.14).
#   usage: arm_win.sh <CORETYPE> <THREADS>
# Every command carries the three thread variables explicitly and reads the
# LOADED kernel back from threadpoolctl rather than trusting the request.
set -u
CT="${1:-HASWELL}"
TH="${2:-1}"
ROOT=/c/tmp/lum_bor3
OUT="$ROOT/validation/probe_fix_bor_round3/runs"
mkdir -p "$OUT"
TAG="win_${CT}_t${TH}"
export OMP_NUM_THREADS="$TH" OPENBLAS_NUM_THREADS="$TH" MKL_NUM_THREADS="$TH"
export OPENBLAS_CORETYPE="$CT" PYTHONPATH="$ROOT"
cd "$ROOT" || exit 1
python -c "
import lumenairy, threadpoolctl, sys
print('lumenairy.__file__ =', lumenairy.__file__)
print('python =', sys.version.split()[0])
for d in threadpoolctl.threadpool_info():
    if d.get('user_api') == 'blas':
        print('LOADED kernel =', d.get('architecture'), 'threads =', d.get('num_threads'))
" 2>&1 | tee "$OUT/pin_$TAG.log"
FILES=$(cat "$OUT/files.txt" | tr "
" " ")
python -m pytest $FILES -q -p no:randomly 2>&1 | tee "$OUT/set_$TAG.log" | tail -5
