#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=/mnt/c/tmp/lum_vslant/validation/probe_verify_slant_anchor
PY=~/lumvenv/bin/python
L=$D/results/batch_wsl.log
while ! grep -q -- "--- done t4_o2_census.py /mnt/c/tmp/lum_vslant_rev ---" "$L"; do sleep 10; done
run() { echo "=== WSL $2 -- $1 ==="; LUM_ARM_TREE="$2" $PY -u "$D/$1"; echo "--- done $1 $2 ---"; }
run t2_v2_sign.py      /mnt/c/tmp/lum_vslant_rev
run t5b_durability_rest.py /mnt/c/tmp/lum_vslant
