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
| Tilt axis convention                  | ``tilt=(theta_x, theta_y, theta_z)`` radians, right-hand rotation around +x, +y, +z respectively | (v5.2 surface-frame branch) |
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
* **Jones basis.** The 2-D solvers return the zeroth-order Jones
  matrix in the lab ``(x, y)`` CARTESIAN basis (columns = response to
  incident ``E_x`` / ``E_y``).  The 1-D solvers return ``te``/``tm``
  (``s``/``p``).  The two bases coincide -- up to the ``tm`` <-> ``x``,
  ``te`` <-> ``y`` identification -- ONLY at ``phi = 0``.  For conical
  incidence (``phi != 0``) the plane of incidence is rotated, so
  ``te``/``tm`` no longer align with ``x``/``y`` and the matrices are
  related by that rotation.

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

---

## Changelog of this file

* v4.16.1 (audit AUDIT_V4_16_0_DEEP item 23): initial draft.  Author:
  Andrew Traverso -- Agent D.
* v5.21 delta-audit (D13): added section 11 (``fff_nv`` is
  entry-point-specific).
