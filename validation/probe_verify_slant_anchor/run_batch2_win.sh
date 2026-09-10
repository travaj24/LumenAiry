#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=C:/tmp/lum_vslant/validation/probe_verify_slant_anchor
L=$D/results/batch_win.log
while ! grep -q -- "--- done t4_o2_census.py C:/tmp/lum_vslant_rev ---" "$L"; do sleep 10; done
run() { echo "=== WIN $2 -- $1 ==="; LUM_ARM_TREE="$2" python -u "$D/$1"; echo "--- done $1 $2 ---"; }
run t2_v2_sign.py      C:/tmp/lum_vslant_rev
run t5b_durability_rest.py C:/tmp/lum_vslant
