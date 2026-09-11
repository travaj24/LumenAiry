#!/bin/bash
# ROUND 2 test matrix, WSL py3.12.  Every arm pins the three thread
# variables ON THE COMMAND LINE and records the LOADED kernel (read back from
# threadpoolctl, never inferred from the request) plus lumenairy.__file__
# before pytest runs.  The five arms run CONCURRENTLY -- each pytest process is
# pinned to ONE BLAS thread, so they do not contend for the same cores, and the
# arms are independent by construction.
cd /mnt/c/tmp/lum_bor2 || exit 1
FILES=$(tr '\n' ' ' < validation/probe_fix_bor_round2/runs/files.txt)
arm() {  # $1 = requested coretype, $2 = threads, $3 = label
  local LOG="validation/probe_fix_bor_round2/runs/matrix_wsl_${3}.log"
  OPENBLAS_CORETYPE="$1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=$2 \
    MKL_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_bor2 ~/lumvenv/bin/python -c "
import lumenairy, threadpoolctl
print('lumenairy.__file__ =', lumenairy.__file__)
for d in threadpoolctl.threadpool_info():
    print('  loaded', d.get('internal_api'), d.get('architecture'), d.get('num_threads'))
" > "$LOG" 2>&1
  OPENBLAS_CORETYPE="$1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=$2 \
    MKL_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_bor2 \
    ~/lumvenv/bin/python -m pytest $FILES -q -p no:randomly --no-header >> "$LOG" 2>&1
  echo "=== wsl $3 -> $(tail -1 "$LOG")"
}
arm HASWELL     1 HASWELL_t1     &
arm NEHALEM     1 NEHALEM_t1     &
arm PRESCOTT    1 PRESCOTT_t1    &
arm SANDYBRIDGE 1 SANDYBRIDGE_t1 &
arm HASWELL     4 HASWELL_t4     &
wait
