#!/bin/bash
# the verifier's suite run (Windows): half A then half B, -n 8
cd /c/tmp/lum_vcurved_e3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_vcurved_e3
V=validation/probe_pmm2d_curved/verify_e3
for h in A B; do
  timeout 1780 python -m pytest $(cat $V/suite_$h.txt) --capture=sys -p no:randomly -q -n 8 -p no:cacheprovider > $V/logs/suite_${h}_win.log 2>&1
  echo "suite $h rc=$?" >> $V/logs/suite_rc_win.txt
done
