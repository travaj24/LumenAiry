#!/bin/bash
# VERIFY-WP-B12: the WSL (py3.12) half, in sequence.
set -u
R=/mnt/c/tmp/lum_vb12
P=$R/validation/probe_verify_b12
PY=$HOME/lumvenv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$R" || exit 1

echo "=== V6 margins ==="
PYTHONPATH="$R" $PY -u "$P/probe_v6_margins.py" 2>&1

echo "=== V4 consumers ==="
PYTHONPATH="$R" $PY -u "$P/probe_v4_consumers.py" 2>&1

echo "=== mutation matrix ==="
for M in "" identity state_only sign_plus conic_only biconic_drop \
         field_frame_drop always_flat opd_sign no_sec n_exit_one; do
  printf '=== MUTATION: %s ===\n' "${M:-baseline}"
  VB12_MUTATION="$M" PYTHONPATH="$R:$P" $PY -m pytest \
    tests/unit/test_audit2609_b12_fga_reference_plane.py \
    tests/unit/test_verify_b12_fga_reference_plane.py \
    -p vb12_mutate -p no:randomly --capture=sys -q --no-header -rf 2>&1 \
    | grep -E "^FAILED|ImportError|passed|failed|error|no tests ran"
done

echo "=== the ten pinned files ==="
PYTHONPATH="$R" $PY -X faulthandler -m pytest \
  tests/unit/test_fga.py tests/unit/test_fga_h4_h5.py \
  tests/unit/test_g1_gate_generality.py \
  tests/unit/test_niche_audit_w9_dispatch2.py \
  tests/unit/test_niche_p8_capstone.py tests/unit/test_niche_p7_seidel_gate.py \
  tests/unit/test_audit2609_a4_fga_s10.py \
  tests/unit/test_audit2609_b7b_caustic_routing.py \
  tests/unit/test_audit2609_b12_fga_reference_plane.py \
  tests/unit/test_verify_b12_fga_reference_plane.py \
  -p no:randomly --capture=sys -q --no-header -rf 2>&1 | tail -8

echo "=== the raytrace / differential consumers ==="
PYTHONPATH="$R" $PY -X faulthandler -m pytest \
  tests/unit/test_analytic_ray_transfer.py \
  tests/unit/test_gbd_feature_complete.py \
  tests/unit/test_audit2609_a1_exit_vertex.py \
  tests/unit/test_raytrace.py tests/unit/test_audit_raytrace.py \
  tests/unit/test_audit_w3_raytrace_parity.py \
  tests/unit/test_audit_w5_raytrace_bundles.py \
  tests/unit/test_audit_w6_raytrace.py \
  tests/unit/test_niche_audit_w3_raytrace_sources.py \
  tests/unit/test_audit2609_a1_raytrace.py \
  tests/unit/test_audit2609_b9_raytrace_perf.py \
  tests/unit/test_v5_4_1_raytrace_mirror_backward_ray.py \
  tests/unit/test_v5_4_6_wave8_raytrace.py \
  -p no:randomly --capture=sys -q --no-header -rf 2>&1 | tail -8

echo "ALLDONE-WSL"
