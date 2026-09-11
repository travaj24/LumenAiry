#!/bin/bash
# The verification's own DECISION file across the full kernel x thread x build
# ladder.  Every command carries the three threading env vars explicitly.
set -u
F=tests/unit/test_verify_bor_guards_round2.py
OUT=/c/tmp/lum_vbor2/validation/probe_verify_bor_round2/runs
for C in HASWELL NEHALEM KATMAI SANDYBRIDGE; do
  ( cd /c/tmp/lum_vbor2 && OPENBLAS_CORETYPE=$C OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_vbor2 \
    python -m pytest -p no:randomly -q -rX $F ) > "$OUT/new_win_${C}_t1.log" 2>&1
  wsl -e bash -lc "cd /mnt/c/tmp/lum_vbor2 && OPENBLAS_CORETYPE=$C \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=/mnt/c/tmp/lum_vbor2 ~/lumvenv/bin/python -m pytest \
    -p no:randomly -q -rX $F" > "$OUT/new_wsl_${C}_t1.log" 2>&1
done
( cd /c/tmp/lum_vbor2 && OPENBLAS_CORETYPE=HASWELL OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=C:/tmp/lum_vbor2 \
  python -m pytest -p no:randomly -q -rX $F ) > "$OUT/new_win_HASWELL_t4.log" 2>&1
wsl -e bash -lc "cd /mnt/c/tmp/lum_vbor2 && OPENBLAS_CORETYPE=HASWELL \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONPATH=/mnt/c/tmp/lum_vbor2 ~/lumvenv/bin/python -m pytest -p no:randomly \
  -q -rX $F" > "$OUT/new_wsl_HASWELL_t4.log" 2>&1
echo "NEWFILE MATRIX DONE"
