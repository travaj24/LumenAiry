#!/usr/bin/env bash
# TASK F -- kernel x thread ladder driver (WSL py3.12), run INSIDE one wsl call.
# usage (from git bash):  vf_ladder_wsl.sh <script.py> <pre|post> <wsl-treepath> ["extra"]
set -u
S="$1"; B="$2"; T="$3"; EX="${4:-}"
wsl -e bash -lc "
cd $T/validation/probe_verify_bor_guards || exit 1
for CT in HASWELL NEHALEM PRESCOTT SANDYBRIDGE; do
  for NT in 1 4; do
    echo \"=== wsl \$CT t\$NT $B ===\"
    OMP_NUM_THREADS=\$NT OPENBLAS_NUM_THREADS=\$NT MKL_NUM_THREADS=\$NT \
      OPENBLAS_CORETYPE=\$CT PYTHONPATH=$T \
      ~/lumvenv/bin/python $S $B wsl_\${CT}_t\${NT} $EX 2>&1 | tail -10
  done
done
"
