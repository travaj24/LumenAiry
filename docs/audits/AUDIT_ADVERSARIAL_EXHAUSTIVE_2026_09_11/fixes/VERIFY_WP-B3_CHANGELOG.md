# VERIFY-B3 changelog text (verifier's additions to WP-B3)

### Fixed -- `normalisation='auto'` declines the photometric branch instead of raising when a stop has no leg in front of it (VERIFY-B3, audit K13)

WP-B3's `'auto'` asked two of the three questions the photometric estimator needs. It checked that
`z_output` gave the walk a plane to close on and that every leg was free space, but not that every
re-emission had an incoming leg to be scaled by. `_reemission_measure` multiplies each re-emitted path by
`r_in`, the geometric length of the leg that reached the surface, and a diffracting surface sitting on the
plane the paths were last emitted from makes that zero for every path -- so `'auto'` resolved to
`'physical'` and the call died inside a private helper with
`ValueError: apply_aperture_diffraction: normalisation='physical' needs the geometric length of the leg
that ended at this surface ...`, a message naming a function the caller never invoked and offering
`init_paths_from_field` as the remedy. Two ordinary layouts reach it: a flat apertured surface at
`object_distance = 0`, and a stop placed at a surface (two surfaces at zero thickness). Both work again
and return the legacy sum with a warning. `lumenairy/propagators/hfpi.py:1141`
(`_walk_zero_length_reemission`, which reads the axial gaps from the prescription before the walk starts),
`:1688` (the diffractor list resolved above the estimator decision so both can use it), `:1713`
(`'auto'`'s third condition), `:1718` (forcing `'physical'` there still refuses -- there is no factor to
apply -- but now with the `CONVENTIONS.md` section 2 prefix, the offending surface index and four real
remedies). The `'auto'` fallback is byte-identical to `normalisation='legacy'` spelled out. Every array
this work package proved byte-identical stays byte-identical (60 comparisons re-run against both
`81d5b586` and `284daccc`). Tests:
`tests/unit/test_audit2609_b3_propagator_kernels.py::TestK13PrescriptionWalkOutputPlane::test_a_zero_length_re_emission_does_not_get_a_photometric_default`.

### Fixed -- the walk's "NOT photometric" warning names the condition that actually failed (VERIFY-B3, audit K13)

Three independent conditions send `propagate_hfpi_through_prescription` down the legacy branch and a
caller can fail more than one at once, but every legacy walk got the same sentence: that the bundle was
binned at the last surface because no output plane was given, ending in "Pass `z_output=<the plane you
want the field on>`". A caller who had passed `z_output` through a powered prescription was told to pass
`z_output`; a caller behind a mirror was told the walk had not propagated to a separate plane when it
had. The warning is now assembled from the answers the estimator was resolved on and states the reasons
that applied -- a missing output plane, an element with power (with the 4879x measurement), a
zero-length re-emission (with the surface index) -- or, for an explicit `normalisation='legacy'`, that it
was asked for. `lumenairy/propagators/hfpi.py:1888`. The `NOT photometric` phrase every existing matcher
uses is unchanged, and the returned field is untouched. **Migration note:** this warning's text moved in
5.47.0 and moves again here; a caller filtering on the pre-5.47 wording (`since v5.46`, `Pass
normalisation='physical' if your surface list ends`) or on 5.47.0's (`Pass z_output=<the plane you want
the field on>`) should match on `NOT photometric` instead. Test:
`...::TestK13PrescriptionWalkOutputPlane::test_the_legacy_warning_names_the_condition_that_failed`.

### Changed -- the chirp-Z resampler's odd-`N` origin handling and its per-axis faithful-zone test are pinned (VERIFY-B3, audit K6)

`resample_field(method='chirpz')` shipped correct on odd and non-square grids and was not pinned there:
dropping the half-pixel `off_in` shift that the `ifftshift` origin requires, or replacing the `N_in // 2`
frequency-bin centre with `N_in / 2`, left all 33 of this work package's tests green while costing a
relative L2 of 1.04 to 1.10 -- a 100 %-class error -- on every odd grid, and exactly zero on even ones,
which is what every K6 fixture used. Ignoring `_warn_mft_output_window`'s new per-axis `N_out_y` likewise
changed nothing any test could see, although it is the parameter the non-square extent-preserving default
was given it for. No library change: six new cases now measure the chirp-Z leg against an explicit
Dirichlet-kernel double sum written for non-square grids (65x65, 45x63, 33x21, 17x17; bar 1e-12, measured
6.0e-15 to 1.8e-14) and pin the faithful-zone warning to the axis it is about (a 44x63 input at
dx 1 -> 0.7 um must warn on y alone because y rounds up to 44.1 um against a 44 um period; a 45x63 input
must stay silent because y rounds down to 44.8 um and x lands exactly on its 63 um period). Tests:
`...::TestK6ChirpZResampler::test_the_chirpz_leg_places_an_odd_grids_origin`,
`...::TestK6ChirpZResampler::test_the_faithful_zone_warning_sizes_each_axis_separately`.
