#!/bin/bash
# VERIFY-WP-C3 CLAIM 9b -- the K1 BUILD SPREAD sweep, WSL side.
# One design-121 N=1024 collins run per (OPENBLAS_CORETYPE, thread count).
set -u
TREE=/mnt/c/tmp/vc3_head
OUT=/mnt/c/tmp/lum_vc3/validation/probe_verify_c3
PY=~/lumvenv/bin/python
D121=/mnt/d/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics
N=${N:-1024}

run () {   # $1 = tag, $2 = coretype ('' = unset), $3 = threads
  tag=$1; ct=$2; th=$3
  if [ -n "$ct" ]; then export OPENBLAS_CORETYPE="$ct"; else unset OPENBLAS_CORETYPE; fi
  cd "$TREE" && OMP_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th MKL_NUM_THREADS=$th \
    D121_ROOT=$D121 PYTHONPATH=$TREE \
    $PY $OUT/v_d121.py $TREE $OUT/v_d121_wsl_n${N}_${tag}.json $N '' collins \
    2>&1 | grep -E '^\[v_d121\]|^    ' | sed "s/^/[$tag] /"
}

for ct in DEFAULT SkylakeX Haswell Zen Nehalem SandyBridge Prescott generic Core2 Barcelona; do
  c="$ct"; [ "$ct" = "DEFAULT" ] && c=""
  run "ct_${ct}" "$c" 1
done
for th in 4 8; do
  run "th_${th}" "" "$th"
done
echo "SWEEP DONE"
