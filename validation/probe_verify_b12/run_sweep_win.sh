#!/bin/bash
# VERIFY-WP-B12: the census / walker / dispatcher-pin / public-API /
# doc-consistency / history sweep plus the except budget, and the raytrace /
# differential consumers, on Windows.
set -u
cd /c/tmp/lum_vb12 || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="C:/tmp/lum_vb12"

echo "=== raytrace / differential consumers ==="
python -X faulthandler -m pytest \
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

echo "=== census / walker / dispatcher-pin / public-API / doc / history sweep ==="
python -X faulthandler -m pytest \
  tests/unit/test_audit2609_a17_history_lint.py \
  tests/unit/test_audit2609_a17_history_relocation.py \
  tests/unit/test_audit2609_a22_history_fingerprint_tool.py \
  tests/unit/test_audit2609_a23_census_mechanism.py \
  tests/unit/test_audit_except_budget.py \
  tests/unit/test_eme_census_determinacy.py \
  tests/unit/test_public_api.py \
  tests/unit/test_v4_14_0_dispatcher_pin_apply_lens.py \
  tests/unit/test_v4_14_0_dispatcher_pin_hfpi.py \
  tests/unit/test_v4_14_0_dispatcher_pin_welford_mirror.py \
  tests/unit/test_v4_14_1_dispatcher_pin_cache_clears.py \
  tests/unit/test_v4_14_2_dispatcher_pin_cache_locks.py \
  tests/unit/test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py \
  tests/unit/test_v4_15_3_dispatcher_pin_2d_scalar_field.py \
  tests/unit/test_v4_15_dispatcher_pin_validate_grid_params.py \
  tests/unit/test_v4_16_0_walker_all_symmetry.py \
  tests/unit/test_v4_16_0_walker_dy_threading.py \
  tests/unit/test_v4_16_0_walker_sentinel_propagation.py \
  tests/unit/test_v4_16_0_walker_xp_of_dispatch.py \
  tests/unit/test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py \
  tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py \
  tests/unit/test_v5_2_3_walker_changelog_content.py \
  tests/unit/test_v5_2_walker_changelog_changeset.py \
  tests/unit/test_v5_2_walker_pep562_forwarding.py \
  tests/unit/test_v5_2_walker_sentinel_reduce.py \
  tests/unit/test_v5_2_walker_shell_vs_canonical.py \
  tests/unit/test_v5_3_2_walker_source_line_citation.py \
  tests/unit/test_v5_3_walker_changelog_self_citation.py \
  tests/unit/test_v5_4_1_walker_scope_the_workaround.py \
  tests/unit/test_v5_4_7_walker_v20_cross_backend_parity.py \
  -p no:randomly --capture=sys -q --no-header -rf 2>&1 | tail -12

echo "ALLDONE-SWEEP"
