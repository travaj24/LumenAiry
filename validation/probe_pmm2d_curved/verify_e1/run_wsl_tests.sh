#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /mnt/c/tmp/lum_vcurved_e1
PYTHONPATH=/mnt/c/tmp/lum_vcurved_e1 ~/lumvenv/bin/python -c "import lumenairy;print(lumenairy.__file__)"
PYTHONPATH=/mnt/c/tmp/lum_vcurved_e1 ~/lumvenv/bin/python -m pytest tests/unit/test_pmm2d_staggered_curved_e1.py tests/unit/test_verify_pmm2d_curved_e1.py --capture=sys -p no:randomly -q -p no:cacheprovider 2>&1 | tail -6
