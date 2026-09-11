#!/bin/bash
# The verification's probe matrix.  Every command carries OMP_NUM_THREADS,
# OPENBLAS_NUM_THREADS and MKL_NUM_THREADS explicitly, and every probe prints
# lumenairy.__file__ and the LOADED OpenBLAS kernel into its own JSON.
set -u
POST=/c/tmp/lum_vbor2
PRE=/c/tmp/lum_vbor2_pre
D=validation/probe_verify_bor_round2

win () {  # $1 tree  $2 coretype  $3 threads  $4.. script+args
  local tree=$1 core=$2 thr=$3; shift 3
  ( cd "$tree" && OPENBLAS_CORETYPE=$core OMP_NUM_THREADS=$thr \
    OPENBLAS_NUM_THREADS=$thr MKL_NUM_THREADS=$thr \
    PYTHONPATH="$(cygpath -w $tree)" python -u "$@" )
}
wsl_ () {  # $1 tree(wsl path)  $2 coretype  $3 threads  $4.. script+args
  local tree=$1 core=$2 thr=$3; shift 3
  wsl -e bash -lc "cd $tree && OPENBLAS_CORETYPE=$core OMP_NUM_THREADS=$thr \
    OPENBLAS_NUM_THREADS=$thr MKL_NUM_THREADS=$thr PYTHONPATH=$tree \
    ~/lumvenv/bin/python -u $*"
}

for C in HASWELL NEHALEM KATMAI SANDYBRIDGE; do
  win $POST $C 1 $D/v2_eme_units.py --tag POST_win_${C}_t1
  win $PRE  $C 1 $D/v2_eme_units.py --tag PRE_win_${C}_t1
  wsl_ /mnt/c/tmp/lum_vbor2 $C 1 $D/v2_eme_units.py --tag POST_wsl_${C}_t1
  wsl_ /mnt/c/tmp/lum_vbor2_pre $C 1 $D/v2_eme_units.py --tag PRE_wsl_${C}_t1
done
win  $POST HASWELL 4 $D/v2_eme_units.py --tag POST_win_HASWELL_t4
wsl_ /mnt/c/tmp/lum_vbor2 HASWELL 4 $D/v2_eme_units.py --tag POST_wsl_HASWELL_t4

wsl_ /mnt/c/tmp/lum_vbor2     HASWELL 1 $D/v7_identity.py --tag POST_wsl_Haswell_t1
wsl_ /mnt/c/tmp/lum_vbor2_pre HASWELL 1 $D/v7_identity.py --tag PRE_wsl_Haswell_t1
wsl_ /mnt/c/tmp/lum_vbor2     HASWELL 1 $D/v3_ladders.py --fast --tag POST_wsl_Haswell_t1
wsl_ /mnt/c/tmp/lum_vbor2_pre HASWELL 1 $D/v3_ladders.py --fast --tag PRE_wsl_Haswell_t1
win  $POST KATMAI 1      $D/v3_ladders.py --fast --tag POST_win_Katmai_t1
win  $PRE  KATMAI 1      $D/v3_ladders.py --fast --tag PRE_win_Katmai_t1
win  $POST SANDYBRIDGE 1 $D/v3_ladders.py --fast --tag POST_win_Sandybridge_t1
win  $POST NEHALEM 1     $D/v3_ladders.py --fast --tag POST_win_Nehalem_t1
win  $POST HASWELL 4     $D/v3_ladders.py --fast --tag POST_win_Haswell_t4
wsl_ /mnt/c/tmp/lum_vbor2 HASWELL 1 $D/v1_census.py --tag wsl_Haswell_t1
win  $POST KATMAI 1      $D/v4_cutoff.py --tag win_Katmai_t1
wsl_ /mnt/c/tmp/lum_vbor2 HASWELL 1 $D/v4_cutoff.py --tag wsl_Haswell_t1
echo "PROBE MATRIX DONE"
