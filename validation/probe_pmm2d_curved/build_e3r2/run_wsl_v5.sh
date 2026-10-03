#!/bin/bash
# WSL build: the verifier's V-E3-3 event probe on the round-2 tree, with
# faulthandler (its first WSL run dumped core after the seventh case)
cd /mnt/c/tmp/lum_curved_e3b/validation/probe_pmm2d_curved/verify_e3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LUM_TREE=/mnt/c/tmp/lum_curved_e3b VE3_TAG=r2 PYTHONPATH=/mnt/c/tmp/lum_curved_e3b:.
for k in 1; do
  timeout 1700 ~/lumvenv/bin/python -X faulthandler v5_events.py 3 > ../build_e3r2/logs/v5_r2_wsl_run$k.log 2>&1
  echo "run $k rc=$?"
  grep -v Warn ../build_e3r2/logs/v5_r2_wsl_run$k.log | grep "NaN at\|CHECK\|Fatal\|Segmentation\|File \"" | cut -c1-90 | head -14
done
