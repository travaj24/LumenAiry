#!/bin/bash
set -u
CORE="$1"; THR="$2"
OUT="/c/tmp/lum_vbor2/validation/probe_verify_bor_round2/runs/wsl_${CORE}_t${THR}.log"
FILES=$(cat /c/tmp/lum_vbor2/validation/probe_verify_bor_round2/runs/files.txt | tr '\n' ' ')
wsl -e bash -lc "cd /mnt/c/tmp/lum_vbor2 && { echo '=== ARM wsl / requested=${CORE} / threads=${THR} ==='; OPENBLAS_CORETYPE=${CORE} OMP_NUM_THREADS=${THR} OPENBLAS_NUM_THREADS=${THR} MKL_NUM_THREADS=${THR} PYTHONPATH=/mnt/c/tmp/lum_vbor2 ~/lumvenv/bin/python -c \"
import lumenairy, numpy, scipy, sys, threadpoolctl
print('lumenairy.__file__ =', lumenairy.__file__)
print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'py', sys.version.split()[0])
for d in threadpoolctl.threadpool_info():
    print('LOADED', d.get('internal_api'), d.get('version'), 'arch=', d.get('architecture'), 'threads=', d.get('num_threads'))
\"; OPENBLAS_CORETYPE=${CORE} OMP_NUM_THREADS=${THR} OPENBLAS_NUM_THREADS=${THR} MKL_NUM_THREADS=${THR} PYTHONPATH=/mnt/c/tmp/lum_vbor2 ~/lumvenv/bin/python -m pytest -p no:randomly -q ${FILES} 2>&1; echo \"=== EXIT \$? ===\"; }" > "$OUT" 2>&1
