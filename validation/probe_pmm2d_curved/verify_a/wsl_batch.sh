#!/bin/bash
# second-build batch: WSL Ubuntu, ~/lumvenv, BLAS pinned, lumenairy from the worktree
cd /mnt/c/tmp/lum_vcurved_a/validation/probe_pmm2d_curved/verify_a
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_vcurved_a
PY=~/lumvenv/bin/python
run() { n=$1; shift; $PY "$@" > wsl_log_$n.txt 2>&1; }
run v2_ident v2_operators.py ident &
run v2_sep v2_operators.py sep &
run v3_film v3_formulation.py film 4 8 &
run v3_stripe v3_formulation.py stripe 4 10 &
run v3_proj v3_formulation.py proj &
run v4_cap v4_stretch.py cap &
run v4_nq6 v4_stretch.py nq 6 &
run v5_split v5_traps.py split &
run v5_hgram v5_traps.py hgram &
run v5_absorb v5_traps.py absorb &
run v5_mag v5_traps.py mag &
run v7_trap v7_walls.py trap &
wait
run v4_l_sine008 v4_stretch.py ladder sine0.08 8 &
run v4_l_asym v4_stretch.py ladder harm_asym 8 &
run v4_l_film v4_stretch.py ladder film_asym 8 &
run v7_film v7_walls.py film &
run v8_vac v8_stack.py vacuum &
run v8_refuse v8_stack.py refuse &
run v11_nodes v11_circle.py nodes &
wait
echo WSL_BATCH_DONE
