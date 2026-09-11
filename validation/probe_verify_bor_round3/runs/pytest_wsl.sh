#!/usr/bin/env bash
# The 21-file BOR/EME set, WSL py3.12, one arm per OPENBLAS_CORETYPE.
set -u
cd /c/tmp/lum_vbor3 || exit 1
FILES=$(cat validation/probe_verify_bor_round3/runs/files.txt | tr '\n' ' ')
for CT in HASWELL SANDYBRIDGE KATMAI; do
  LOG="validation/probe_verify_bor_round3/runs/set_wsl_${CT}_t1.log"
  wsl -e bash -lc "cd /mnt/c/tmp/lum_vbor3 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_CORETYPE=$CT PYTHONPATH=/mnt/c/tmp/lum_vbor3 ~/lumvenv/bin/python -c \"
import lumenairy, threadpoolctl, sys, numpy, scipy
print('TREE', lumenairy.__file__, lumenairy.__version__)
print('PY', sys.version.split()[0], 'np', numpy.__version__, 'sp', scipy.__version__)
for d in threadpoolctl.threadpool_info():
    if d.get('user_api') == 'blas':
        print('LOADED', d.get('architecture'), 'threads', d.get('num_threads'))
\" && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_CORETYPE=$CT PYTHONPATH=/mnt/c/tmp/lum_vbor3 ~/lumvenv/bin/python -m pytest $FILES -q -p no:randomly" > "$LOG" 2>&1
  echo "wsl $CT exit=$? :: $(tail -3 "$LOG" | tr '\n' ' ')"
done
