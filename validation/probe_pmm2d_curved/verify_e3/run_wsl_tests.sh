#!/bin/bash
# WSL second build: the E3 file, the verifier's file, curved A-D and the JAX
# gates, -n 6
cd /mnt/c/tmp/lum_vcurved_e3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_vcurved_e3
timeout 1780 ~/lumvenv/bin/python -m pytest tests/unit/test_pmm2d_staggered_curved_e3.py tests/unit/test_verify_pmm2d_curved_e3.py tests/unit/test_pmm2d_staggered_curved_a.py tests/unit/test_pmm2d_staggered_curved_b.py tests/unit/test_pmm2d_staggered_curved_c.py tests/unit/test_pmm2d_staggered_curved_d.py tests/unit/test_v5_20_2_pmm_jones_2d_jax.py tests/unit/test_v5_14_2_jax_stacks.py tests/unit/test_backend_disable_jax.py tests/unit/test_audit_w3_pmm_jax_guards.py --capture=sys -p no:randomly -q -n 6 -p no:cacheprovider > validation/probe_pmm2d_curved/verify_e3/logs/suite_wsl.log 2>&1
echo "rc=$?" >> validation/probe_pmm2d_curved/verify_e3/logs/suite_wsl.log
