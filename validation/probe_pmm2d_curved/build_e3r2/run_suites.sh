#!/bin/bash
# round-2 suite run (Windows): half A then half B, -n 8 (lists: suite_A/B.txt,
# the verifier's plus the two eig-VJP audits)
cd /c/tmp/lum_curved_e3b
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved_e3b
V=validation/probe_pmm2d_curved/build_e3r2
for h in A B; do
  timeout 1780 python -m pytest $(cat $V/suite_$h.txt) --capture=sys -p no:randomly -q -n 8 -p no:cacheprovider > $V/logs/suite_${h}_win.log 2>&1
  echo "suite $h rc=$?" >> $V/logs/suite_rc_win.txt
done
