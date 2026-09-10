#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=/mnt/c/tmp/lum_vslant/validation/probe_verify_slant_anchor
PY=~/lumvenv/bin/python
run() { echo "=== WSL $2 -- $1 ==="; LUM_ARM_TREE="$2" $PY -u "$D/$1"; echo "--- done $1 $2 ---"; }
run t3_v1_jax.py       /mnt/c/tmp/lum_vslant
run t3_v1_jax.py       /mnt/c/tmp/lum_vslant_pre
run t3_v1_jax.py       /mnt/c/tmp/lum_vslant_rev
run t5_durability.py   /mnt/c/tmp/lum_vslant
run t2_v2_sign.py      /mnt/c/tmp/lum_vslant
run t4_o2_census.py    /mnt/c/tmp/lum_vslant
run t4_o2_census.py    /mnt/c/tmp/lum_vslant_rev
