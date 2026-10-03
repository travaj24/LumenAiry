#!/bin/bash
# WSL: re-run the two tests of the -n 4 sweep that failed (the near-symmetric
# gate after its fail-before bar fix; the hybrid pmm_jones_2d file whose worker
# crashed inside XLA compile), serially
cd /mnt/c/tmp/lum_curved_e3b
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_curved_e3b
~/lumvenv/bin/python -X faulthandler -m pytest "tests/unit/test_pmm2d_staggered_curved_e3.py::test_e3r2_near_symmetric_cells_are_inside_the_rule" tests/unit/test_v5_20_2_pmm_jones_2d_jax.py --capture=sys -p no:randomly -q -p no:cacheprovider 2>&1 | tail -n 4
