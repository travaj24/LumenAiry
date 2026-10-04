#!/bin/bash
# Verifier probes on WSL (serial).  Usage: wsl -e bash run_wsl.sh [probes...]
cd /mnt/c/tmp/lum_symgrad_verify/validation/probe_jax_symgrad_verify
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/lumvenv/bin/python
HEAD=/mnt/c/tmp/lum_symgrad_verify
BASE=/mnt/c/tmp/lum_symgrad_verify_base
for p in "$@"; do
  case $p in
    b_bytes)
      PYTHONPATH=$HEAD LUM_TREE=$HEAD VTAG=post $PY b_bytes.py > b_post_wsl.log 2>&1
      PYTHONPATH=$BASE LUM_TREE=$BASE VTAG=pre $PY b_bytes.py > b_pre_wsl.log 2>&1 ;;
    b2_bytes_jax)
      PYTHONPATH=$HEAD LUM_TREE=$HEAD VTAG=post $PY b2_bytes_jax.py > b2_post_wsl.log 2>&1
      PYTHONPATH=$BASE LUM_TREE=$BASE VTAG=pre $PY b2_bytes_jax.py > b2_pre_wsl.log 2>&1 ;;
    v5_hess_tree)
      PYTHONPATH=$HEAD LUM_TREE=$HEAD VTAG=post $PY v5_hess_tree.py > v5_post_wsl.log 2>&1
      PYTHONPATH=$BASE LUM_TREE=$BASE VTAG=pre $PY v5_hess_tree.py > v5_pre_wsl.log 2>&1 ;;
    tests)
      cd $HEAD
      PYTHONPATH=$HEAD $PY -c "import lumenairy; assert lumenairy.__file__.startswith('$HEAD'), lumenairy.__file__"
      PYTHONPATH=$HEAD $PY -m pytest tests/unit/test_verify_jax_symmetric_point_gradients.py tests/unit/test_jax_symmetric_point_gradients.py "tests/unit/test_pmm2d_staggered_curved_e3.py::test_e3r2_rcwa_jax_symmetry_breaking_gradient_at_a_symmetric_cell" "tests/unit/test_pmm2d_staggered_curved_e3.py::test_e3r2_pmm1d_jax_angle_gradient_at_normal_incidence" -p no:cacheprovider -q --no-header -rfEX --durations=15 > validation/probe_jax_symgrad_verify/tests_wsl.log 2>&1
      cd - > /dev/null ;;
    d2_pre)
      PYTHONPATH=$BASE LUM_TREE=$BASE VTAG=pre $PY d2_jones2d_li.py > d2_pre_wsl.log 2>&1 ;;
    *)
      PYTHONPATH=$HEAD LUM_TREE=$HEAD $PY $p.py > ${p}_wsl.log 2>&1 ;;
  esac
  echo "done $p"
done
