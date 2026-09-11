#!/usr/bin/env bash
# TASK F -- kernel x thread ladder driver (Windows py3.14).
# usage: vf_ladder_win.sh <script.py> <pre|post> <treepath>
set -u
S="$1"; B="$2"; T="$3"
cd /c/tmp/lum_vbor/validation/probe_verify_bor_guards || exit 1
for CT in HASWELL NEHALEM PRESCOTT SANDYBRIDGE; do
  for NT in 1 4; do
    echo "=== win $CT t$NT $B ==="
    OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT \
      OPENBLAS_CORETYPE=$CT PYTHONPATH="$T" \
      python "$S" "$B" "win_${CT}_t${NT}" 2>&1 | tail -8
  done
done
