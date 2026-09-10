# Run transcripts

Assembled by `make_runs_md.sh` from the raw logs (`*.log` is gitignored).

## The 22 named files, Windows

`tests/unit/test_pmm*.py tests/unit/test_fix_pmm*.py tests/unit/test_verify_pmm*.py tests/unit/test_rcwa*.py`,
`OMP` = `OPENBLAS` = `MKL` = 1, `-p no:randomly`:

```
    return fn(*args, **kwargs)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================ slowest 15 durations =============================
232.74s call     tests/unit/test_pmm2d_lossless_closure_two_sided.py::test_solver_warning_matches_the_predicate_on_every_arm
65.95s call     tests/unit/test_pmm2d_staggered_oop.py::test_g5_no_fourier_floor_two_sided
63.56s call     tests/unit/test_rcwa.py::test_analytic_energy_and_clean_convergence[te]
63.49s call     tests/unit/test_fix_pmm2d_mortar_round2.py::test_fail_before_the_sliver_the_guard_refuses_is_measurably_wrong
62.30s call     tests/unit/test_rcwa.py::test_analytic_energy_and_clean_convergence[tm]
46.07s call     tests/unit/test_pmm_m3_efficiency.py::test_t34_guard_fires_on_every_silent_wrong_cell_of_this_build
38.42s call     tests/unit/test_pmm2d_staggered_mortar.py::test_taper_agrees_with_the_hybrid_staircase_at_the_same_slices
37.92s call     tests/unit/test_pmm2d_staggered_mortar.py::test_g3_h_row_v1_v2_swap_is_load_bearing
37.60s call     tests/unit/test_verify_pmm2d_perlayer_slant.py::test_slanted_split_across_non_conforming_grids_matches_the_1d_oracle[28.0-22.0]
37.25s call     tests/unit/test_pmm_m2_window_contract.py::test_min_feature_is_the_accuracy_lever_on_the_per_layer_path_too
35.85s call     tests/unit/test_verify_pmm2d_perlayer_slant.py::test_slanted_per_layer_on_coinciding_grids_is_bit_exact_vs_shared
35.20s call     tests/unit/test_verify_pmm2d_perlayer_slant.py::test_slanted_split_across_non_conforming_grids_matches_the_1d_oracle[15.0-0.0]
34.60s call     tests/unit/test_pmm2d_staggered_mortar.py::test_g5_stripe_pair_per_order_vs_the_exact_1d_oracle
29.80s call     tests/unit/test_pmm2d_staggered_nonuniform.py::test_n3_two_exact_wall_representations_agree_inside_the_triangle_bar
28.96s call     tests/unit/test_pmm2d_staggered_oop_corner_convergence.py::test_staggered_oop_corner_ladder_steps_shrink_and_start_above_the_floor
582 passed, 50 warnings in 2245.06s (0:37:25)
```

## The fix's gate, both builds

```
--- WIN
......................                                                   [100%]
============================= slowest 6 durations =============================
15.12s call     tests/unit/test_fix_slant_anchor_v1_v2_o2.py::test_o2_the_measured_cures_still_solve
12.50s call     tests/unit/test_fix_slant_anchor_v1_v2_o2.py::test_v2_the_walks_add_over_two_sheared_layers
12.03s call     tests/unit/test_fix_slant_anchor_v1_v2_o2.py::test_v2_the_transmission_converges_to_the_lab_referenced_staircase
11.75s call     tests/unit/test_fix_slant_anchor_v1_v2_o2.py::test_v2_the_cross_engine_arm_agrees_with_the_independent_pure_engine
7.55s call     tests/unit/test_fix_slant_anchor_v1_v2_o2.py::test_v1_the_vertical_control_still_solves_and_matches_numpy
3.41s call     tests/unit/test_fix_slant_anchor_v1_v2_o2.py::test_v1_the_vertical_traced_solve_still_differentiates
22 passed in 78.43s (0:01:18)
--- WSL
......................                                                   [100%]
============================= slowest 6 durations ==============================
15.09s call     tests/unit/test_fix_slant_anchor_v1_v2_o2.py::test_o2_the_measured_cures_still_solve
12.73s call     tests/unit/test_fix_slant_anchor_v1_v2_o2.py::test_v2_the_transmission_converges_to_the_lab_referenced_staircase
11.77s call     tests/unit/test_fix_slant_anchor_v1_v2_o2.py::test_v2_the_walks_add_over_two_sheared_layers
10.25s call     tests/unit/test_fix_slant_anchor_v1_v2_o2.py::test_v2_the_cross_engine_arm_agrees_with_the_independent_pure_engine
9.92s call     tests/unit/test_fix_slant_anchor_v1_v2_o2.py::test_v1_the_vertical_control_still_solves_and_matches_numpy
6.54s call     tests/unit/test_fix_slant_anchor_v1_v2_o2.py::test_v1_the_vertical_traced_solve_still_differentiates
22 passed in 78.83s (0:01:18)
```

## This verification's gate, both builds

```
--- WIN, tests/unit/test_verify_slant_anchor_v1_v2_o2.py
........                                                                 [100%]
============================= slowest 8 durations =============================
9.21s call     tests/unit/test_verify_slant_anchor_v1_v2_o2.py::test_verify_v1_no_eighth_route_carries_a_shear_into_the_jnp_twin
7.07s call     tests/unit/test_verify_slant_anchor_v1_v2_o2.py::test_verify_o2_the_two_populations_separate_on_an_independent_family
4.41s call     tests/unit/test_verify_slant_anchor_v1_v2_o2.py::test_verify_v2_a_second_sheared_layer_sits_at_the_ACCUMULATED_walk
3.93s call     tests/unit/test_verify_slant_anchor_v1_v2_o2.py::test_verify_o2_the_refusal_is_the_same_with_the_census_armed_or_not
0.17s call     tests/unit/test_verify_slant_anchor_v1_v2_o2.py::test_verify_v2_the_anchor_stays_unimodular_through_a_wood_anomaly
0.11s call     tests/unit/test_verify_slant_anchor_v1_v2_o2.py::test_verify_v2_the_default_factorization_exposes_no_transmitted_field
0.07s call     tests/unit/test_verify_slant_anchor_v1_v2_o2.py::test_verify_v2_a_net_zero_walk_is_the_exact_identity_through_a_solve
0.01s call     tests/unit/test_verify_slant_anchor_v1_v2_o2.py::test_verify_v2_a_sheared_stack_exposes_no_other_frame_bearing_surface
8 passed in 25.30s
--- WSL, tests/unit/test_verify_slant_anchor_v1_v2_o2.py
........                                                                 [100%]
============================= slowest 8 durations ==============================
6.93s call     tests/unit/test_verify_slant_anchor_v1_v2_o2.py::test_verify_v1_no_eighth_route_carries_a_shear_into_the_jnp_twin
6.31s call     tests/unit/test_verify_slant_anchor_v1_v2_o2.py::test_verify_o2_the_two_populations_separate_on_an_independent_family
3.96s call     tests/unit/test_verify_slant_anchor_v1_v2_o2.py::test_verify_o2_the_refusal_is_the_same_with_the_census_armed_or_not
3.85s call     tests/unit/test_verify_slant_anchor_v1_v2_o2.py::test_verify_v2_a_second_sheared_layer_sits_at_the_ACCUMULATED_walk
0.12s call     tests/unit/test_verify_slant_anchor_v1_v2_o2.py::test_verify_v2_the_default_factorization_exposes_no_transmitted_field
0.09s call     tests/unit/test_verify_slant_anchor_v1_v2_o2.py::test_verify_v2_the_anchor_stays_unimodular_through_a_wood_anomaly
0.05s call     tests/unit/test_verify_slant_anchor_v1_v2_o2.py::test_verify_v2_a_net_zero_walk_is_the_exact_identity_through_a_solve
0.01s call     tests/unit/test_verify_slant_anchor_v1_v2_o2.py::test_verify_v2_a_sheared_stack_exposes_no_other_frame_bearing_surface
8 passed in 22.00s
```

## ruff (WSL)

```
$ ruff check lumenairy/ tests/ validation/probe_verify_slant_anchor/
All checks passed!

$ ruff --version
ruff 0.15.16
```
