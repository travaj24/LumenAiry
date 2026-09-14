# WP-B7b changelog text

Release 5.47.0, after WP-B7 and VERIFY-B7.  The two behaviour changes WP-B7
measured and escalated, plus the measured envelope of the uniform fold
completion.  Full evidence in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B7b_REPORT.md`.

---

### Changed (DEFAULT MOVE) -- at a caustic, a single-valued field inside the aberration envelope now takes `'phase_screen'`, not `'fga'`

`apply_real_lens_universal(method='auto')`'s caustic gate preferred
`apply_real_lens_fga` for ANY field near a geometric caustic.  That is right for
a MULTI-VALUED field -- several local directions cross the region and only a
phase-space swarm transports them independently -- and measurably wrong for a
single-valued one.

MEASURED against a brute-force Rayleigh-Sommerfeld oracle built on an exact
conic raytrace (WP-B7b fixture: N-SF11 biconvex R = +/-1.6 mm, t = 0.60 mm,
0.30 mm aperture, lambda = 633 nm, N = 256, dx = 1.4 um, w = 80 um, read at the
traced best focus 925.9 um past the exit vertex; oracle converged to 1.1e-06
relative L2 in its ray quadrature and 1.2e-14 in its azimuthal one):

| member | fidelity | intensity-rms spot | EE(3 um) |
|---|---|---|---|
| oracle | 1 | 1.523 um | 0.8585 |
| `phase_screen` | **0.9991** | 1.540 um | 0.8551 |
| `fga` | **0.1251** | 7.241 um | 0.0698 |

and `'phase_screen'` is the closer member at every NA of a 0.048 .. 0.260 sweep
on that singlet (its rms spot error grows 0.016 -> 0.112 um with NA -- the
thin-screen obliquity ceiling is real -- against `'fga'` 13.363 -> 4.243 um).
It is not a sampling deficit: over fifteen sampling settings `'fga'` CONVERGES
in `n_p` (0.1412 / 0.1450 / 0.1462 at 21 / 41 / 61) and is inert in `dq_step`
to four digits.  WP-B7 measured the same on an N-BK7 f = 1.2 mm NA 0.145
singlet at 1.0 um: 0.3234 (0.3826 at the best of the same fifteen settings)
against 0.9965.

The H2 aberration gate keeps the other half of the decision: a prescription
whose sag-screen estimate is OVER `aberration_threshold` still never reaches the
thin screen, so it keeps `'fga'` at a caustic.  That class is the 2026-07-19
displaced / Debye-oracle regime where the analytic model is 58-123 % wrong (the
G1 matrix designs read 20 .. 2893 rad against the 2.0 rad budget) and it is
outside what the oracle above covers (its whole NA ladder reads
0.003 .. 0.231 rad).

**Migration.**  The calls whose answer changes are
`apply_real_lens_universal(..., method='auto')` (and `apply_real_lens_auto` is
NOT affected -- it is the older GBD/FGA 2-way subset and has no `phase_screen`
member) at an `output_plane_distance` inside the caustic zone, when BOTH: the
field is single-valued (`_tilt_dispersion` at or below `multivalued_threshold`,
or `multivalued=False` passed), AND the prescription is inside the sag-screen
aberration envelope (`_sag_screen_aberration_rad` at or below
`aberration_threshold`).  Those calls now return `apply_real_lens` + exact
angular spectrum where they returned `apply_real_lens_fga`.

How much the answer moves, on the measured fixture: the returned field's
intensity-rms spot goes **7.241 um -> 1.540 um** (oracle 1.523 um), its fidelity
against the oracle **0.1251 -> 0.9991**, and EE(3 um) **0.0698 -> 0.8551**
(oracle 0.8585); the two returned fields overlap each other at fidelity 0.1247,
so this is a different answer, not a refinement of one.
Across the affected part of the NA ladder (0.126 / 0.160 / 0.202 / 0.260) the
rms spot error against the oracle falls from +6.770 / +5.718 / +4.929 / +4.243
um to -0.000 / +0.018 / +0.058 / +0.112 um.  Wall clock on that fixture falls
from 7.3 s to 0.7 s.

To get the old route back, force the member: `method='fga'`.  `caustic_pad_dof`
only narrows the zone -- inside it the route is unchanged and outside it the
plane leaves the caustic branch for `'traced'` -- so it is not a way back
(measured by VERIFY-B7b at the near edge, the midpoint and the far edge of the
unpadded zone).

* `lumenairy/propagators/fga.py` (`_universal_route`'s caustic branch;
  `apply_real_lens_universal`'s member map and split-step note).
* Tests: `tests/unit/test_audit2609_b7b_caustic_routing.py`
  (`..._single_valued_field_at_a_caustic_routes_to_phase_screen`,
  `..._multivalued_field_at_a_caustic_still_routes_to_fga`,
  `..._aberrated_caustic_keeps_fga_two_sided`,
  `..._phase_screen_is_the_closer_member_at_the_caustic`);
  the restated expectations in `tests/unit/test_audit2609_a4_fga_s10.py`.

### Changed (DEFAULT MOVE) -- FGA's analytic-Jacobian predicate is now the analytic primitive's own domain

`_pick_ray_transfer` gated on `_is_all_conic`, a hand-kept whitelist that had
drifted from `ray_transfer_jacobian_analytic` in BOTH directions:

* it excluded `aspheric_coeffs`, although WP-B9 gave that primitive even-aspheric
  support -- so an aspheric prescription traced the 9-ray finite-difference
  bundle whatever `exact_jacobian` said, `exact_jacobian=True` included
  (silently ignored);
* it did not check `field_decenter` / `field_tilt` / `field_sag_callable`, which
  the primitive DOES reject -- so a field-decentred conic reached the analytic
  path and raised `NotImplementedError` at call time instead of falling back
  (a latent bug, reproduced against the parent commit).

The predicate is now `_analytic_jacobian_applies`, mirroring the primitive's
guard term for term (`_is_all_conic` stays as a back-compat alias).  This is
what lets FGA dispatch on a predicate where `gbd.jacobian='auto'` dispatches on
the primitive's own `NotImplementedError`.

MEASURED on the A4 singlet with `aspheric_coeffs {4: 4.0e3}`, 4001 rays: the two
primitives' base-ray exit states agree to 3.5e-16 relative in height, 2.8e-16 in
slope and 1.3e-17 m in OPL -- they trace the same base ray -- while their
JACOBIANS differ by 2.4e-09 relative, which is the finite-difference central
truncation at the shipped steps (the step ladder reads 2.43e-07 / 2.42e-09 /
1.18e-10 at `h_pos` = 1e-5 / 1e-6 / 1e-7 and turns up to 8.5e-10 at 1e-8 as
round-off takes over).  The analytic side is the exact one.

**Migration.**  The calls whose answer changes are
`apply_real_lens_fga` / `apply_real_lens_fga_vector` /
`apply_real_lens_universal(method='fga')` on a rotationally-symmetric
prescription carrying an even-aspheric departure, at `coarse_stride=1`, with
`exact_jacobian` left at its `None` default or set to `True`.  Those calls
switch from the finite-difference 9-ray bundle to the exact analytic single-ray
Jacobian: the differential transfer stops carrying the 2.4e-09 relative FD
truncation quoted above, and the trace count drops 9N -> N (36009 -> 4001 rays
on that measurement; 8924 -> 7484 bytes per FGA lattice point -- a fixed
1440 B saving, so the chunk sizer fits 1.19x more lattice points at a fixed
`mem_budget_mb` on that grid and swarm; the ratio depends on `n_p` and reads
1.003x on VERIFY-B7b's 128^2 configuration).  A
field-decentred / tilted conic changes from raising `NotImplementedError` to
completing on the FD primitive.  An all-conic prescription is byte-identical
(proved: `apply_real_lens_fga` on the A4 conic singlet returns a byte-identical
field against the parent commit).

To get the old behaviour back on an aspheric prescription, pass
`exact_jacobian=False`.

* `lumenairy/propagators/fga.py` (`_analytic_jacobian_applies` + the
  `_is_all_conic` alias; `_pick_ray_transfer`; the memory-model call site; the
  `exact_jacobian` parameter docs).
* Tests: the restated assertions in `tests/unit/test_fga_h4_h5.py`
  (`test_h4_exact_jacobian_default_analytic_for_conic`);
  `tests/unit/test_audit2609_b7b_caustic_routing.py`
  (`..._predicate_is_the_analytic_primitive_s_own_domain`,
  `..._field_decentred_conic_falls_back_instead_of_raising`,
  `..._aspheric_swap_costs_nothing_and_removes_the_fd_truncation`).

### Added -- `apply_real_lens_traced_uniform` reports and warns when the fold's `zeta` is extrapolated

No default moves and no field changes (proved byte-identical against the parent
commit on three cases, including the one that now warns).

The uniform fold completion fits `kappa` in `zeta(r) = kappa (r_c - r)` on the
band of radii reached by BOTH coalescing branches -- the only radii where the
eikonal difference defining `zeta` exists -- and then evaluates `zeta` across a
fit band `uniform_fit_halfwidth` wide and a dark fill 20 Airy lengths deep.
When the two-branch band is much narrower than the fit band, the two CFU
coefficients and the tail they continue are an EXTRAPOLATION of the fold normal
form rather than a fit to it, and nothing said so.

MEASURED against a brute-force Rayleigh-Sommerfeld oracle on three singlets,
indexed by `zeta_extrapolation = uniform_fit_halfwidth / zeta_band`:

| zeta_extrapolation | 0.16 | 1.9 | 2.3 | 3.4 | 5.4 | 9.8 | 453.6 |
|---|---|---|---|---|---|---|---|
| fidelity | 0.944 | 0.957 | 0.933 | 0.969 | 0.972 | 0.962 | 0.930 |
| power / oracle | 0.947 | 0.975 | 0.971 | 1.030 | 1.049 | 1.125 | 1.228 |
| multibranch fidelity | 0.882 | 0.805 | 0.856 | 0.853 | 0.884 | 0.887 | 0.832 |

The SHAPE is reliable and is the better one at every plane measured (it beats
the plain multibranch it would fall back to, by 0.06-0.15 of fidelity, so
falling back would be a regression, and at the widest-band plane it also beats
`amplitude_model='ray_density'`, 0.9435 vs 0.9395); the ABSOLUTE ENERGY the dark
tail carries degrades with the extrapolation, within -5.3 % / +4.9 % up to ~5
and +12.5 % / +22.8 % from ~10 up.  `_trace_meridional_fold` now returns the
two-branch band width, `apply_real_lens_traced_uniform` reports `zeta_band` /
`zeta_extrapolation` in its diagnostics, and above
`_ZETA_EXTRAPOLATION_MAX = 8.0` (in the measured gap) it emits a
`RuntimeWarning` naming the measured envelope and the alternatives.  The
function's docstring carries the table, and the grid gate that decides whether
any of it runs at all: on a fast singlet at its marginal focus the fold's Airy
layer is 2.82 um wide, so the same optic and plane COMPLETES at dx = 2.10 um
(and scores 0.9435, the best of four members) and FALLS BACK at dx = 4.00 um.

* `lumenairy/elements/_lens_traced_uniform.py` (`_ZETA_EXTRAPOLATION_MAX`;
  `band` in `_trace_meridional_fold`; the diagnostics and the warning;
  the accuracy-envelope docstring section).
* Tests: `tests/unit/test_audit2609_b7b_caustic_routing.py`
  (`..._fold_band_is_the_two_branch_band`,
  `..._uniform_warns_only_when_zeta_is_extrapolated`, two-sided and with the
  field's byte-identity under the warning asserted).
