### Fixed -- FGA: the frozen-Gaussian swarm's image-side leg starts at the exit-vertex plane, not on the last surface

`lumenairy.raytrace.ray_transfer_jacobian` and
`lumenairy.raytrace.ray_transfer_jacobian_analytic` return the base ray's state
and Jacobian ON the last surface (`z = sag(rho)`) -- the same plane
`lumenairy.raytrace.trace` stops on and `TraceResult.image_rays` reports.  Four
sites in `lumenairy/propagators/fga.py` added the image-side free-space leg
`z_image` to that state as if it were on the last surface's VERTEX plane, which
is what `output_plane_distance` is measured from: `_fga_core`, `_fga_coarse`'s
coarse trace, `_fga_coarse`'s vector path and `_caustic_zone`.  Every beamlet
therefore carried a spurious optical path of `n_exit * sag(rho) * sec(theta)`
and a spurious transverse offset of `sag(rho) * u` -- a defocus-plus-spherical
term growing quadratically with pupil radius, measured at **7.79 waves of
optical path and 0.66 um of height at the rim** of an N-SF11 R = +/-1.6 mm
biconvex at 633 nm, 6.77 waves on an N-BAF10 biconvex at 1.064 um, 5.36 waves on
a bent N-LAK22 singlet at 1.55 um, 1.06 waves on a weakly curved N-SF6 meniscus,
and **exactly zero on a flat last surface**.

The two primitives now take an explicit output reference plane,
`reference='surface'` (the default, unchanged and the convention the rest of
`lumenairy.raytrace` uses) or `reference='exit_vertex'`, and the four FGA sites
ask for the latter.  The projection is one shared implementation
(`differential._project_to_exit_vertex_plane`) used by the finite-difference
backend, the numba analytic kernel, the `_AdrtDual` NumPy path and the JAX path;
it takes its sag and sag gradient from the package's general surface kernels, so
it is exact for conic, even-aspheric, biconic, freeform and field-frame
surfaces, resolves the exit-medium index through
`raytrace.exit_vertex.resolve_exit_index`, carries the propagation-direction
sign through a mirror, and projects the 4x4 Jacobian as well as the state (the
transfer distance depends on where the ray lands, so the composite derivative
carries the two extra blocks; pinned against an independent finite difference).

Measured against a brute-force Rayleigh-Sommerfeld oracle built on an exact
conic raytrace (`validation/probe_wp_b12/`, both builds, identical):

| prescription | plane | before | after |
|---|---|---|---|
| N-SF11 biconvex R = +/-1.6 mm, 633 nm | exit vertex | 0.0737 | **0.9995** |
| the same | focus, 926.0 um | 0.1252 | **0.9998** |
| N-BAF10 biconvex R = +/-2.10 mm, 1.064 um | focus | 0.0953 | **0.9998** |
| N-LAK22 bent singlet, 1.55 um | focus | 0.1798 | **0.9997** |
| N-SF6 converging meniscus, 850 nm | focus | 0.5306 | **0.9997** |
| N-LASF9 plano-convex, flat last surface | both | 0.9979 / 0.9982 | **bit-identical** |

`_caustic_zone`'s estimate moves with it: the near edge of the zone was pulled
in by about the last surface's sag (0.41 % of the focal distance on the N-SF11
biconvex, 0.50 % on the bent singlet) and now reproduces an independent
evaluation of the same estimator to the last printed digit.

The universal dispatcher's caustic route is **unchanged** -- a single-valued
field inside the sag-screen aberration envelope still takes `'phase_screen'` --
but its justification is not: with the reference plane repaired, `'fga'` is the
more accurate member there (0.9998 against 0.9991 on the fixture 5.47.0 routed
on) and the screen is roughly thirty times cheaper, so the branch is now a cost
choice and its comment says so with the numbers.  Whether to move it back is a
maintainer decision, and the measurement it turns on is recorded in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B12_REPORT.md`.

**Migration.**  **Every field returned by `apply_real_lens_fga`,
`apply_real_lens_fga_vector`, `apply_real_lens_universal(method='fga')` and
`apply_real_lens_auto` when it dispatches to FGA changes on any prescription
whose LAST surface is curved**, and so does every `_caustic_zone` estimate on
such a prescription (and therefore, at planes near the edge of a caustic zone, a
routing decision that depended on it).  The fields move from wrong to right --
0.07-0.53 fidelity to 0.999-class against an independent diffraction oracle --
so there is no reason to want the old answers back, and there is no keyword that
restores them: the previous behaviour is reachable only from the parent commit.
A prescription whose last surface is FLAT is unaffected, bit for bit, proved
archive-to-archive.  Callers who pinned FGA output digests on a curved-last-
surface prescription must re-record them.  `apply_real_lens_gbd` and
`apply_prescription_persurface_to_beamlets` are untouched: they keep the
`'surface'` default and their own in-line vertex correction.

Pinned by `tests/unit/test_audit2609_b12_fga_reference_plane.py` (14 tests, one
`slow`); the eight files that pinned the previous numbers are restated against
the oracle with derived bars.
