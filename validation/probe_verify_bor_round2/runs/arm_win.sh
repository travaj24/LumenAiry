#!/bin/bash
# One Windows arm of the verification matrix.  $1 = requested OPENBLAS_CORETYPE,
# $2 = thread count.  Every threading env var is on the COMMAND LINE.
set -u
CORE="$1"; THR="$2"
OUT="/c/tmp/lum_vbor2/validation/probe_verify_bor_round2/runs/win_${CORE}_t${THR}.log"
cd /c/tmp/lum_vbor2 || exit 9
{
  echo "=== ARM win / requested=${CORE} / threads=${THR} ==="
  OPENBLAS_CORETYPE="$CORE" OMP_NUM_THREADS="$THR" OPENBLAS_NUM_THREADS="$THR" \
  MKL_NUM_THREADS="$THR" PYTHONPATH=C:/tmp/lum_vbor2 python -c "
import lumenairy, numpy, scipy, sys, threadpoolctl
print('lumenairy.__file__ =', lumenairy.__file__)
print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'py', sys.version.split()[0])
for d in threadpoolctl.threadpool_info():
    print('LOADED', d.get('internal_api'), d.get('version'), 'arch=', d.get('architecture'), 'threads=', d.get('num_threads'))
"
  OPENBLAS_CORETYPE="$CORE" OMP_NUM_THREADS="$THR" OPENBLAS_NUM_THREADS="$THR" \
  MKL_NUM_THREADS="$THR" PYTHONPATH=C:/tmp/lum_vbor2 python -m pytest -p no:randomly -q \
    $(cat /c/tmp/lum_vbor2/validation/probe_verify_bor_round2/runs/files.txt | tr '\n' ' ') 2>&1
  echo "=== EXIT $? ==="
} > "$OUT" 2>&1
