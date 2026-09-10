#!/bin/sh
# Runs the remaining probes in order once M4 has written its JSON.
# m6 must run LAST: it reads results/m4b_staircase_ladder.json.
cd /c/tmp/lum_slantp || exit 1
export PYTHONPATH=/c/tmp/lum_slantp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=validation/probe_pmm2d_staggered_slant
until [ -f "$D/results/m4_pillar.json" ]; do sleep 15; done
for s in m4b_staircase_ladder m5_census_cascade m7_slant_x_aniso m6_cost; do
  echo "RUNNING $s"
  python -u "$D/$s.py" > "$D/logs/$s.log" 2>&1 || echo "FAILED $s"
  echo "DONE $s"
done
echo ALL_DONE
