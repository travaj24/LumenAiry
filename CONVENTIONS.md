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

* Time: ``exp(-i omega t)`` -- standard physics convention.
* Forward propagation: ``exp(+i k z)``.
* OPD sign: positive OPD means a phase advance (wavefront leads the
  reference sphere) -- see ``opd_fan_data`` / ``plot_opd_summary``
  documentation.

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

## Changelog of this file

* v4.16.1 (audit AUDIT_V4_16_0_DEEP item 23): initial draft.  Author:
  Andrew Traverso -- Agent D.
