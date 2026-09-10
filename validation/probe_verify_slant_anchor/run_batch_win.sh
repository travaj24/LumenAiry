#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=C:/tmp/lum_vslant/validation/probe_verify_slant_anchor
run() { echo "=== WIN $2 -- $1 ==="; LUM_ARM_TREE="$2" python -u "$D/$1"; echo "--- done $1 $2 ---"; }
run t3_v1_jax.py       C:/tmp/lum_vslant
run t3_v1_jax.py       C:/tmp/lum_vslant_pre
run t3_v1_jax.py       C:/tmp/lum_vslant_rev
run t5_durability.py   C:/tmp/lum_vslant
run t2_v2_sign.py      C:/tmp/lum_vslant_rev
run t4_o2_census.py    C:/tmp/lum_vslant
run t4_o2_census.py    C:/tmp/lum_vslant_rev
