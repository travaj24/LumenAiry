#!/bin/bash
# WSL build: the verifier's byte-identity set, pre vs post (same file)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/lumvenv/bin/python
V=/mnt/c/tmp/lum_vcurved_e1/validation/probe_pmm2d_curved/verify_e1
cd /mnt/c/tmp/vce1_pre_eae4 && PYTHONPATH=/mnt/c/tmp/vce1_pre_eae4 $PY $V/v1_bytes.py /mnt/c/tmp/vce1_pre_eae4 wslpre 2>&1 | tail -1
cd /mnt/c/tmp/lum_vcurved_e1 && PYTHONPATH=/mnt/c/tmp/lum_vcurved_e1 $PY $V/v1_bytes.py /mnt/c/tmp/lum_vcurved_e1 wslpost 2>&1 | tail -1
