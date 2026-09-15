#!/bin/bash
# The item-D kernel ladder, final pass.
#
# Writes to validation/probe_known_reds/ladder_final/, a directory no earlier
# run of this session knows about: two earlier ladder loops of this session
# outlived the shells that launched them and kept writing into ladder_waveD/
# and ladder_itemD/, which is why those two directories' logs are interleaved
# and must not be read.  Killing them was not permitted, so they are simply
# routed around; they add load and nothing else.
#
# Scope is the two selections whose claims are arm-dependent:
#   T31 -- the single T3-1 test, the one that classified by BLAS build
#   C78 -- the c7 / c8 halo files, whose restored stimulus must hold on
#          every kernel, plus the gbd budget pins
cd /c/tmp/lum_reds || exit 1
OUT=/c/tmp/lum_reds/validation/probe_known_reds/ladder_final
mkdir -p "$OUT"

T31="tests/unit/test_pmm_m2_window_contract.py::test_halfwidth_2_moves_the_answer_only_inside_the_mortar_band"
C78="tests/unit/test_niche_c7_ray_density_halo_check.py tests/unit/test_niche_c8_inverse_support_bound.py tests/unit/test_wave5_gbd_dense_mem_budget.py"

for CT in HASWELL NEHALEM KATMAI SANDYBRIDGE; do
  for NT in 1 4; do
    tag="${CT}_t${NT}"
    for name in T31 C78; do
      sel=$([ "$name" = T31 ] && echo "$T31" || echo "$C78")
      OPENBLAS_CORETYPE=$CT OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT \
        MKL_NUM_THREADS=$NT PYTHONPATH=/c/tmp/lum_reds \
        timeout 5400 python -u -m pytest $sel --capture=sys -p no:randomly -q -rs \
        > "$OUT/${name}_${tag}.log" 2>&1
      echo "${name} ${tag}: $(grep -oE '[0-9]+ (passed|failed|skipped)' "$OUT/${name}_${tag}.log" | tr '\n' ' ')"
    done
  done
done
echo LADDER_FINAL_DONE
