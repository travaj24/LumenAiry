# LumenAiry Coding Conventions

v4.16.1 (audit AUDIT_V4_16_0_DEEP item 23): codify the de-facto
conventions used throughout the library so new APIs land in the
right place without each author having to re-derive the contract.

This document is short by design.  Where a convention is enforced by
a regression / walker test, the test path is given inline.

---

## 1. Factory-verb naming -- ``create_*`` vs ``make_*``

LumenAiry uses two factory verbs with distinct, non-overlapping
semantics:

* ``create_*`` -- returns a FIELD or a SOURCE.  Output is either a
  2-D complex ``numpy.ndarray`` representing a sampled scalar /
  vectorial field, or a :class:`lumenairy.Source` dataclass
  wrapping one.  These helpers always require an ``N`` / ``dx`` /
  ``wavelength`` triple (the field sampling grid).

  Examples (alphabetical):
  ``create_annular_beam``,
  ``create_annular_incoherent_source``,
  ``create_bessel_beam``,
  ``create_circular_polarized``,
  ``create_diffractive_lens``,
  ``create_elliptical_polarized``,
  ``create_fiber_mode``,
  ``create_fresnel_zone_plate``,
  ``create_gaussian_beam``,
  ``create_gaussian_schell_source``,
  ``create_hermite_gauss``,
  ``create_kinoform``,
  ``create_laguerre_gauss``,
  ``create_led_source``,
  ``create_linear_polarized``,
  ``create_microlens_array``,
  ``create_multi_field_sources``,
  ``create_periodic_phase_mask``,
  ``create_point_source``,
  ``create_schell_model_source``,
  ``create_tilted_plane_wave``,
  ``create_top_hat_beam``,
  ``create_zoom_configs``.

* ``make_*`` -- returns a non-field aggregate: a prescription
  ``dict``, a :class:`RayBundle`, a ``BSDFModel``, or a JAX ray
  state.  These helpers do NOT take an ``N`` / ``dx`` / ``wavelength``
  grid; the description is intrinsically geometric or symbolic.

  Examples (alphabetical):
  ``make_biconic``,
  ``make_bsdf``,
  ``make_cylindrical``,
  ``make_doublet``,
  ``make_fan``,
  ``make_grid``,
  ``make_jax_ray_state``,
  ``make_lg_aberration_merit_jax``,
  ``make_off_axis_parabola``,
  ``make_ray``,
  ``make_ring``,
  ``make_rings``,
  ``make_singlet``.

**Rule for new APIs**: pick the verb by the return type.  If a new
factory returns a field, use ``create_``.  If it returns a
prescription / bundle / non-field object, use ``make_``.

This contract is checked by
``tests/unit/test_v4_16_1_agent_d.py::test_factory_verb_naming_contract``.

---

## 2. Error message prefix -- ``f"{fn_name}: ..."``

When a public function raises, the message should begin with the
function name and a colon::

    raise ValueError(
        f"compute_psf: pupil field must be 2-D complex; "
        f"got shape {E.shape} with dtype {E.dtype}.  "
        f"Pass ``input_kind='psf'`` if you already have a PSF intensity."
    )

Rationale: when a user pastes an error message into a search /
bug-report, the function name should be the first token they see.
Adoption is partial (~40% as of v4.16.0); ``_validation.py`` is the
gold-standard reference (100% prefixed + actionable copy-paste
snippets).

## 3. RNG kwarg name -- ``rng``

New APIs that consume randomness should accept a ``rng`` keyword
argument, accepting any of:

* ``None`` (default) -- use the module-level / global numpy random
  state.
* ``int`` -- seed.  The function wraps the integer in a
  :class:`numpy.random.Generator`.
* :class:`numpy.random.Generator` -- used directly.
* :class:`lumenairy.backend.RandomState` -- cross-backend wrapper
  (NumPy / CuPy / JAX); used directly.

Avoid the legacy names ``seed`` / ``random_state`` / ``random_seed``
in NEW code.  The library has 4-way drift on this kwarg historically
(8 sites use ``seed``, 5 use ``rng``, 1 each ``random_state`` /
``random_seed``); ``rng`` is the canonical choice going forward.

## 4. Wavelength kwarg name -- ``wavelength``

Always ``wavelength`` (full word).  Do NOT use ``lam`` / ``wl`` /
``wvl`` / ``lambda_`` / ``lam0``.  Units: meters, vacuum.

This is checked by the v4.13.0 dispatcher-pin tests.

## 5. Spatial sampling kwargs -- ``dx`` and ``dy``

``dx`` is the x-axis pixel pitch in meters.  ``dy`` is the y-axis
pixel pitch in meters, defaulting to ``dx`` when omitted
(square-grid convention).

Helpers that internally support anamorphic / per-axis pitch should
take ``dy=None`` and propagate ``dy = dy if dy is not None else dx``
explicitly; the v4.16.0 ``dy``-threading walker pins the propagation
at every dispatcher.

## 6. Units

* Lengths: meters (m).
* Wavelengths: meters (vacuum).
* Angles: radians.  When a sibling ``angle_deg`` is offered, the two
  must be mutually exclusive (see v4.14.3 polarization-family
  conflict-resolution work).
* Phases: radians.
* Indices of refraction: dimensionless complex ``n + 1j*kappa``.

## 7. Sign conventions

**v5.2 one-stop table** (audit AUDIT_V5_1_0 deferred doc item):

| Quantity                              | Convention                            | Source                                                |
|---------------------------------------|---------------------------------------|-------------------------------------------------------|
| Time                                  | ``exp(-i omega t)``                   | standard physics                                      |
| Forward propagation                   | ``exp(+i k z)``                       | matches time convention                               |
| Wave-side mirror radius               | ``R > 0`` -> concave (focusing)       | Welford signed-R; raytrace consistency since v4.10    |
| Refraction-side surface radius        | ``R > 0`` -> convex (center of curvature behind surface, +z side) | Optiland / Zemax / Welford raytrace |
| OPD sign                              | ``OPD > 0`` -> wavefront LEADS reference (phase advance) | ``opd_fan_data`` / ``plot_opd_summary``     |
| Lens phase                            | ``phi = -k n_substrate * sag``        | thin / real lens kernels                              |
| Reflective phase pickup               | ``+pi`` on normal-incidence mirror    | matches ``exp(+i k z)``                               |
| Aperture transmission                 | ``E_out = E_in * t``, ``t in [0, 1]`` | no phase added by clear aperture                      |
| Decenter axis convention              | ``decenter=(dx, dy)`` moves the SURFACE | (v5.2: ``frame='surface'`` opt-in; default = field-frame, v5.1 behavior) |
| Surface ``tilt`` key (2-tuple, radians) | ``tilt = (t0, t1)`` **IS the linear sag ramp** ``t0*(x - dcx) + t1*(y - dcy)`` added to the surface's z-departure, with the correspondingly tilted normal (``dfdx += t0``, ``dfdy += t1``).  As a right-hand rotation pair that is ``theta_x = t1`` and ``theta_y = -t0`` (a +x rotation ramps in y; a +y rotation ramps in -x).  There is **no** ``theta_z`` on this key.  Every consumer reads it this way -- the ``surface_frame=False`` ramp, the ``surface_frame=True`` rigid-body branch, the traced / GBD ray models and the lumenairy-free geometric oracle -- so flipping ``surface_frame`` no longer re-points the element (audit L2 / L19) | ``raytrace/surface.py::_field_frame_sag_and_grad`` (canonical), ``elements/_lens_real.py`` (field-frame ramp; surface-frame ``theta_x = t1``, ``theta_y = -t0``), ``elements/_lens_real.py::_disp_surface_z_grad``, ``validation/oracles/geom_spot_decenter_oracle.py``; cross-model pin ``tests/unit/test_niche_p9_decenter_tilt.py`` |
| Field normalisation and power         | ``sum(abs(E)**2) * dx * dy`` **IS** the optical power -- the propagators are Parseval-unitary and carry no impedance factor -- so any element that crosses an index step must apply the POWER transmittance ``T = (n2 cos theta_t) / (n1 cos theta_i) * abs(t)**2``, not ``abs(t)**2`` (audit L13) | ``elements/_lens_real.py`` ``fresnel=True`` block; closed form ``T = 4 n1 n2 / (n1 + n2)**2`` at normal incidence |
| Richards--Wolf ``pupil`` indexing     | indexed by the physical exit-pupil (aperture) coordinate, **NOT** the projected ray direction; the two differ by a point inversion -- invisible for a 180-degree-symmetric pupil, immediately wrong for coma, tilt, a decentred sub-aperture, a segmented aperture or a metasurface pupil.  The ``Returns`` block states ``E_z``'s symmetry and sign (audit K10 / K16) | ``propagators/vector_diffraction.py::richards_wolf_focus`` |
| Coordinate-break tilt (rigid frame)   | ``tilt_x_deg`` / ``tilt_y_deg`` / ``tilt_z_deg`` are Zemax ``Tilt About X/Y/Z``, defined by the **LOCAL-TO-WORLD** rotation: right-hand ``R_math(+theta)`` composed in **intrinsic X->Y->Z** order (``PARM 6 = 0``), so ``r_world = world_R @ r_local + world_origin`` and a ``+90 deg`` ``tilt_x`` puts the new local ``+z`` at world ``-y``.  A coordinate break is a PASSIVE frame change, so the matrix applied to RAY coordinates is the **TRANSPOSE** of that (``local = world_R.T @ world``) — getting those two confused is what kept `intersection.py` / `differential.py` / `ui/model.py` inverted from 3.7.1 to v5.29 (audit W3-1, 2026-07-25; OpticStudio KB KA-01638 "Rotation Matrix and Tilt About X/Y/Z"). ``decenter_x_m`` / ``decenter_y_m`` move the new frame's ORIGIN by ``+d`` (in the old frame for ``PARM 6 = 0``, the new frame for ``= 1``) | ``raytrace/world.py::_apply_coord_break`` (canonical), ``raytrace/intersection.py::_apply_coord_break`` (its transpose) |
| Polarization convention               | Jones / Stokes follow the IEEE / right-hand-rule circular convention: ``S3 = -2 Im(Ex conj(Ey))``, so ``'right'`` = ``(1, +i)/sqrt(2)`` with ``S3 = +1``. Born & Wolf §1.4.2 uses the exact NEGATIVE (``s3 = +2 Im(Ex conj(Ey))``) and calls that state left-circular; textbook retarder Mueller matrices match with the ``(S1,S2)<->S3`` block sign flipped (measured to 4.4e-16; audit E-M13, 2026-07-25) | ``polarization.py`` module docstring, "Circular-polarization sign convention" |
| Waveplate slow-axis phase             | ``exp(+i * retardance)`` (slow axis picks up POSITIVE relative phase under ``exp(-i omega t)`` / ``exp(+i k n z)``); matches the rigorous solver family (``berreman_jones_1d`` / ``rcwa_jones_1d`` transmission Jones) so solver-derived Jones drop into ``JonesField`` pipelines WITHOUT conjugation. A QWP with fast axis at +45 deg on x-pol yields ``S3 = -1`` ('left'). The v5.4.6 P3-22 ``exp(-i * retardance)`` "DECOUPLED" note predates the Berreman/RCWA retarder Jones and is SUPERSEDED by this cross-family alignment. | ``polarization.py::apply_waveplate`` (v5.17.0 audit P2-15) |
| Refractive index                      | ``n_complex = n + 1j * kappa`` with ``kappa > 0`` for ABSORPTION (passive media) | ``glass.py`` registry                                 |
| Grating/coating polarization aliases  | ``s`` == ``te`` (E perpendicular to the plane of incidence); ``p`` == ``tm`` (E in the plane). The grating solvers (``rcwa_*``, ``thin_grating``) speak ``te``/``tm`` and the ``coatings`` TMM speaks ``s``/``p``; both aliases are accepted everywhere (case-insensitive) | ``rcwa.py::_normalize_pol``, ``coatings.py`` |

* Time: ``exp(-i omega t)`` -- standard physics convention.
* Forward propagation: ``exp(+i k z)``.
* OPD sign: positive OPD means a phase advance (wavefront leads the
  reference sphere) -- see ``opd_fan_data`` / ``plot_opd_summary``
  documentation.
* **Power.** ``sum(abs(E)**2) * dx * dy`` IS the optical power: the
  propagators are Parseval-unitary and carry no impedance factor, so any
  element that crosses an index step must apply the POWER transmittance
  ``T = (n2 * cos(theta_t)) / (n1 * cos(theta_i)) * abs(t)**2``, not
  ``abs(t)**2``.  This is the convention ``apply_real_lens(fresnel=True)``
  implements (audit finding **L13**): before v5.46 it applied ``abs(t)**2``,
  which telescopes to the right answer only for an element that both starts
  and ends in air at normal incidence.  Measured transmittance of a single
  AIR -> N-BK7 face: 0.632344 (``abs(t)**2``) against **0.958057** (the
  power transmittance, and the closed form ``4 n1 n2 / (n1 + n2)**2`` to
  1.1e-16).
* **Surface tilt.** ``tilt = (t0, t1)`` on a prescription surface is the sag
  ramp ``t0*x + t1*y``, i.e. the rotation pair ``theta_x = t1``,
  ``theta_y = -t0``.  The wave and ray models share this ONE reading (audit
  **L2** / **L19**); a document or docstring that describes ``tilt`` as
  ``(theta_x, theta_y, theta_z)`` is describing the coordinate-break keys
  ``tilt_x_deg`` / ``tilt_y_deg`` / ``tilt_z_deg``, which are a different
  key with a different (rigid-frame) meaning -- see the row above.

The table above is the canonical one-stop summary; each entry is
documented at the call site in the module that owns the convention.
A future audit that finds a per-call-site contradiction with this
table should treat the table as the source-of-truth and fix the
call site (or, if the call-site is correct, refresh the table with
the citation of the corrected call-site).

### 7.1 Grating-solver incidence angle and Jones basis

The rigorous grating family (``rcwa_*``, ``pmm_*``, Berreman) does NOT
share a single incidence-angle keyword.  The conventions, and the
reasons they differ, are:

* **1-D entries** (``rcwa_jones_1d`` / ``rcwa_efficiency_1d`` /
  ``berreman_jones_1d`` and their ``*_vs_wavelength`` / ``*_segments``
  siblings) take ``angle`` (the classical mount, radians) and ALSO
  accept ``theta`` as an ALIAS.  ``theta`` IS ``angle`` -- the same
  number in the plane of periodicity (azimuth ``phi = 0``, planar
  mount).  When BOTH are supplied ``theta`` takes precedence (the
  RCWA family routes every entry through ``_resolve_incidence`` so
  this is uniform and intentional; it is NOT the ``angle`` /
  ``angle_deg`` case of section 6, which stays mutually exclusive).
  The 1-D Jones entries have no conical ``phi`` (Berreman does).
* **2-D entries** (``rcwa_jones_2d`` / ``rcwa_efficiency_2d`` /
  ``pmm_jones_2d`` / ``pmm_efficiency_2d*``) take the conical pair
  ``theta`` (polar) and ``phi`` (azimuth).  They do NOT accept
  ``angle`` -- passing ``angle=`` raises ``TypeError`` (unexpected
  keyword).  Use ``theta`` for the polar angle.
* **Jones basis.** EVERY solver -- 1-D and 2-D -- returns the
  zeroth-order Jones matrix in the lab ``(x, y)`` CARTESIAN basis:
  columns = response to incident ``E_x`` / ``E_y``, index 0 = ``x``,
  index 1 = ``y``.  At ``phi = 0`` the ``x`` column is the ``p``
  (``tm``) channel and the ``y`` column the ``s`` (``te``) channel,
  **up to the sign of the p unit vector**: measured against an
  independent analytic TMM on a uniform slab at 0 deg / 30 deg /
  60 deg, ``J[0, 0] = -r_p`` (exactly ``-1.000000`` in ratio at every
  angle) and ``J[1, 1] = +r_s``.  That sign is the standard Fresnel
  ``p`` convention and not an error -- at normal incidence the lab
  basis correctly gives ``J_xx = J_yy`` (isotropy), which a literal
  ``te``/``tm`` matrix would not -- but a consumer who reads the
  ``x`` column as ``r_p`` picks up a sign.  PMM and RCWA agree with
  each other to 1e-15 on both entries.  For conical incidence
  (``phi != 0``) the plane of incidence is rotated away from
  ``x``/``y`` entirely and the ``te``/``tm`` matrix is the lab one
  conjugated by that rotation.

## 8. Top-level re-exports

Public symbols re-exported from ``lumenairy/__init__.py`` are
organised into tiers (Tier 1 -- core; Tier 5 -- advanced /
specialised).  Every symbol in ``lumenairy/__init__.py`` must be
mirrored in the module-level ``__all__``; the
``__all__``-symmetry walker
(``test_v4_16_0_walker_all_symmetry``) enforces this contract on
every commit.

## 9. Sentinel pattern

Default-overriding sentinels follow the v4.14.1 pattern:

* Module-level singleton with ``__slots__ = ()``.
* Custom ``__repr__`` for debug clarity.
* Pickle-safe (registered with ``copyreg`` when relevant).
* Sentinel comparison via ``is`` only; never ``==``.

Examples: ``_ZeroApertureMaskSentinel``, ``_AngleUnsetSentinel``,
``_NoDefaultSentinel``.

## 10. Optional dependency lazy import

Optional dependencies (``jax``, ``cupy``, ``numba``, ``numexpr``,
``h5py``, ``zarr``, ``filelock``, ``pyfftw``, ``astropy``,
``refractiveindex``, ``pymoo``) follow this contract:

* Top-of-module ``_FOO_AVAILABLE = importlib.util.find_spec(...) is
  not None``.
* No module-level ``import foo``; defer to first call.
* On first call, lazy import + cache.
* On call without the dep installed, raise ``ImportError`` with a
  ``pip install`` hint citing both the bare package and the
  ``lumenairy[group]`` extras name.

See ``lumenairy/glass.py`` (refractiveindex) and
``lumenairy/optimize/multi_objective.py`` (pymoo) for canonical
examples.

``threadpoolctl`` is **not** on this list: as of v5.46 it is a HARD
dependency (``pyproject.toml`` ``dependencies``).  It is what makes
``set_blas_threads`` / ``rcwa_blas_threads`` and the ``@_with_blas_limit``
wrapper on every public RCWA entry point actually apply a cap; without it
they warn and are inert, which on an oversubscribed many-core box is not a
micro-optimisation (measured: a 1-D TM RCWA solve at ``n_orders=81`` takes
18.2 s unpinned against 0.13 s at one thread -- 140x; ``inv()`` of a
163x163 complex matrix 2.29 s against 0.0057 s -- 400x).  The code still
degrades to a no-op if the import fails, so the section 10 absence
semantics are preserved.

## 10.1 Process-global knobs are scoped with ``override``

Every registered process-global knob has a getter / setter pair AND is
reachable through ``lumenairy.override(**knobs)``, a context manager that
restores the previous values in reverse order on every exit path::

    with lumenairy.override(fft_threads=1, pyfftw_planner='FFTW_ESTIMATE'):
        ...

**Rule for new knobs**: register the knob beside its setter with
``lumenairy._knobs.register_knob(name, getter=..., setter=..., doc=...)``.
The getter must be cheap and side-effect-free.  Registration happens at
import time of the module that OWNS the knob, so the registry
(``lumenairy._knobs.knobs()``) grows as a session proceeds.  ``override`` is process-global, not thread-local,
and validates every name BEFORE the first setter runs, so a typo cannot
leave half a block applied.  ``tests/conftest.py`` snapshots and restores
every registered knob per test (``LUMEN_TEST_KNOB_LEAK_STRICT=1`` fails the
leaking test instead).

---

## 11. ``formulation='fff_nv'`` is entry-point-specific (D13)

The ``fff_nv`` (Fast Fourier Factorization with normal vectors) formulation
token names a DIFFERENT algorithm on each 2-D entry point.  All three are the
correct curved-wall factorization for their solver, but the shared token
invites wrong cross-entry assumptions -- so pick the entry point by what the
cell needs, not by the token:

| Entry point | ``fff_nv`` means | Crossed (both-axes) cells |
| --- | --- | --- |
| ``rcwa_efficiency_2d`` | Popov--Neviere normal-vector FFF projector | supported |
| ``rcwa_jones_2d`` | Li-2003 successive full-tensor ``L2 L1`` factorization | supported |
| ``pmm_jones_2d`` | separable-only Popov--Neviere reduction | **raises** (separable cells only) |

E.g. ``fff_nv`` on a crossed (x- AND y-patterned) cell works on
``rcwa_jones_2d`` but RAISES on ``pmm_jones_2d``.  Each function's own
docstring states its scope; this table is the cross-reference.

**Validated scope (v5.46, audit H3).**  ``rcwa_jones_2d(formulation='fff_nv')``
now emits a validated-scope warning on a CURVED cell -- the Jones counterpart of
``rcwa_efficiency_2d``'s refusal -- and takes ``allow_nonseparable_nv=True`` to
acknowledge it.  The normal-vector field the entry builds is the separable
axis-aligned one; on a curved or non-separable boundary it is not the Popov--
Neviere field the formulation is derived for, and it manufactured form
birefringence on a symmetric cell.  The notice reaches the out-of-plane
(conical) path as well.

**Picking a formulation is an observable-dependent choice (audit G6).**
``pmm_jones_2d`` gained ``formulation='auto'`` (``'fff_nv'`` on a separable
in-plane cell, ``'laurent'`` otherwise; ``'laurent'`` on the JAX path, where
``'fff_nv'`` does not exist).  The DEFAULT stays ``'laurent'`` so the documented
exact reduction of a scalar cell to
``pmm_efficiency_2d_cell(formulation='laurent')`` does not move; ``'auto'`` is
opt-in and is the recommended setting for new code.  The docstring's accuracy
ordering is the REFLECTION Jones ordering: on the TRANSMISSION retardance of a
form-birefringent stripe ``'li'`` is the most accurate of the three at
``n_orders >= 9``.  Pick the rule for the observable you are designing against.

---

## Changelog of this file

* v4.16.1 (audit AUDIT_V4_16_0_DEEP item 23): initial draft.  Author:
  Andrew Traverso -- Agent D.
* v5.21 delta-audit (D13): added section 11 (``fff_nv`` is
  entry-point-specific).
* v5.46 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, G4): section 7.1's
  Jones-basis bullet restated -- EVERY solver, 1-D and 2-D, returns the
  zeroth-order Jones in the lab ``(x, y)`` Cartesian basis with **index 0 =
  x (= p / tm at phi = 0), index 1 = y (= s / te at phi = 0)**, and
  ``J[0, 0] = -r_p`` (the standard Fresnel p convention, measured exactly
  ``-1.000000`` in ratio against an analytic TMM at 0 / 30 / 60 deg).  The
  previous wording ("the 1-D solvers return ``te``/``tm``") invited a
  te-first reading that transposes a retardance sign.  Confirmed here
  against ``rcwa_jones_1d`` vs ``PMM2DStackHybrid`` on a form-birefringent
  grating (``J[0,0]`` <-> ``Jxx``, ``J[1,1]`` <-> ``Jyy``), which is
  WP-A13's request.
* v5.46 (audit L13): section 7 states the POWER convention --
  ``sum(abs(E)**2)*dx*dy`` IS the power, so an index step needs
  ``T = (n2 cos theta_t)/(n1 cos theta_i) abs(t)**2``.
* v5.46 (audit L2 / L19): the section 7 tilt row now states the
  **implemented** convention (``tilt = (t0, t1)`` IS the sag ramp
  ``t0*x + t1*y``, i.e. ``theta_x = t1``, ``theta_y = -t0``) and names its
  five consumers.  The previous row claimed a
  ``(theta_x, theta_y, theta_z)`` triple, which no consumer of this key has
  ever implemented; that spelling belongs to the coordinate-break keys
  ``tilt_x_deg`` / ``tilt_y_deg`` / ``tilt_z_deg`` one row down.  No stored
  ``tilt`` value needs migrating.
* v5.46 (audit K10 / K16): section 7 gained the Richards--Wolf
  ``pupil``-coordinate row.
* v5.46 (audit H3 / G6): section 11 records ``rcwa_jones_2d``'s
  ``fff_nv`` validated-scope warning and ``allow_nonseparable_nv``, and
  ``pmm_jones_2d``'s new ``formulation='auto'`` with the
  observable-dependent accuracy ordering.
* v5.46 (audit H5, V4): section 10 records that ``threadpoolctl`` is now a
  HARD dependency, and new section 10.1 records the
  ``lumenairy.override(...)`` knob registry that replaces bare ``set_*``
  calls (audit V6: 53 ``set_*`` verbs, 0 context managers, 0 resets).
