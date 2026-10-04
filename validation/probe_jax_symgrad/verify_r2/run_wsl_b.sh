#!/bin/bash
cd /mnt/c/tmp/lum_symgrad_verify2/validation/probe_jax_symgrad/verify_r2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/lumvenv/bin/python
$PY -u p2_families.py berreman_two_iso_lossy_normal berreman_two_iso_lossy_oblique pmm_jones_1d_lossy_aniso rcwa_jones_2d_cross_lossy rcwa_jones_2d_cross_li_uniaxial rcwa_efficiency_2d_cross_te_tm rcwastack_two_patterned_cross rcwastack_keep_symmetry pmm_jones_1d_accidental_0p2 pmm_efficiency_1d_lossy_te_tm pmmstack_shared_3layer_lossy pmmstack_perlayer_3layer_lossy hybrid_two_patterned_traced_layout 2>&1 | grep "^{" | cut -c1-260
mv p2_families_all_wsl.json p2_families_set_wsl.json
$PY p4_bytes.py post 2>&1 | tail -1
LUMROOT=/mnt/c/tmp/lum_symgrad_verify2_base $PY p4_bytes.py pre 2>&1 | tail -1
$PY p4_compare.py wsl | head -40
$PY p10_testsizes.py 2>&1 | grep "^{"
