#!/bin/bash
# second build (WSL): the cheap probes + byte identity PRE vs POST on this build
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LUMENAIRY_DISABLE_JAX=1
PY=~/lumvenv/bin/python
cd /mnt/c/tmp/lum_vcurved_b/validation/probe_pmm2d_curved/verify_b
export PYTHONPATH=/mnt/c/tmp/lum_vcurved_b
#$PY v1_geometry.py > wsl_v1.log 2>&1
#$PY v2_quadrature.py moments > wsl_v2.log 2>&1
$PY v5_identity_kink.py identity > wsl_v5i.log 2>&1
$PY v5_identity_kink.py corner0 > wsl_v5c.log 2>&1
(cd /mnt/c/tmp/vcurved_b_pre && PYTHONPATH=/mnt/c/tmp/vcurved_b_pre $PY /mnt/c/tmp/lum_vcurved_b/validation/probe_pmm2d_curved/verify_b/v4_bytes.py /mnt/c/tmp/vcurved_b_pre pre_wsl) > wsl_v4.log 2>&1
$PY v4_bytes.py /mnt/c/tmp/lum_vcurved_b post_wsl >> wsl_v4.log 2>&1
echo WSL_PROBES_DONE
