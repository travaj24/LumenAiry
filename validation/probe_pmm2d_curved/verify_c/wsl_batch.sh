#!/bin/bash
# second-build batch: WSL Ubuntu, ~/lumvenv, BLAS pinned, lumenairy from the worktree
cd /mnt/c/tmp/lum_vcurved_c/validation/probe_pmm2d_curved/verify_c
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_vcurved_c
PY=~/lumvenv/bin/python
mkdir -p logs
run() { n=$1; shift; $PY "$@" > logs/wsl_$n.txt 2>&1; }
(LUM_TREE=/mnt/c/tmp/vcc_pre PYTHONPATH=/mnt/c/tmp/vcc_pre $PY v1_bytes.py pre_wsl > logs/wsl_v1pre.txt 2>&1) &
run v1post v1_bytes.py post_wsl see &
run v2 v2_merge.py &
run v3 v3_struct.py &
run v5 v5_primitives.py geom &
run v8 v8_viewer.py &
wait
run v5b v5b_ellipse_layout.py &
run v11 v11_departures.py &
run v4f5 v4_incident.py film circle 5 0 0 &
run v4f6 v4_incident.py film circle 6 25 40 &
run v4s5 v4_incident.py spacer circle 5 &
run v4s6 v4_incident.py spacer stretch 6 &
wait
run v4r5 v4_incident.py recip 5 &
run v6a v6_twolayer.py annulus 3 &
run v10 v10_macro.py 4 &
(LUM_TREE=/mnt/c/tmp/vcc_pre PYTHONPATH=/mnt/c/tmp/vcc_pre $PY v3b_unmapped_floor.py > logs/wsl_v3b_pre.txt 2>&1) &
run v3b v3b_unmapped_floor.py &
wait
echo WSL_BATCH_DONE
