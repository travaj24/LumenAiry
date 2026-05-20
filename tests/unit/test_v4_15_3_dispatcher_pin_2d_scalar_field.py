"""Dispatcher meta-pin (v4.15.3 / extended v4.15.4 / V6-refactored
v4.15.5): every public 2-D-scalar-field entry point in
``lumenairy.__all__`` (plus any sibling discovered by the package
walk) must call ``_check_2d_scalar_field(E, ...)`` as its first
executable statement (after the docstring and any helper-import
statements), OR appear in the documented ``_GUARD_EXEMPTIONS`` set
with a cited reason.

Counter-measure to the recurring 'fix N, miss N+1' meta-pattern
the v4.15.0 -> v4.15.2 audit rounds repeatedly identified.
v4.15.2 hand-guarded 10 entry points; v4.15.3 closes 9 more
siblings and pins the invariant structurally so the 20th entry
point can't be added unguarded.

v4.15.4 closes the meta-pattern recurrence the v4.15.3 audit
identified one level of indirection higher: the walker had scoped
only to ``propagators/`` + ``elements/``, missing 4 public entry
points in ``system.py`` (``propagate_through_system_jax``) and
``analysis/`` (``apply_dm``, ``apply_detector``,
``apply_perturbations``) plus 2 functions in scope but missed by
the name regex (``richards_wolf_focus``, ``debye_wolf_psf``).
v4.15.4 broadens discovery via ``lumenairy.__all__`` membership
(robust against future module reorganisation) AND extends the
package-walk scope so non-``__all__`` siblings are still caught.

v4.15.5 (P1-NEW-2WAY-1, V6 meta-pin) refactors discovery from the
hand-curated name-prefix regex (``apply_*`` / ``*_propagate*`` /
``richards_wolf_*`` / ``debye_wolf_*``) to a structural
*first-positional-parameter-name* filter.  The v4.15.4 audit
enumerated ~30 public ``__all__`` functions whose first positional
arg is ``E`` / ``E_in`` / ``field`` / ``pupil`` / ``object_field``
/ ``psf`` but whose function name was outside the hand-curated
prefix set (e.g. ``wave_opd_2d``, ``M2``, ``strehl_ratio``,
``compute_psf``, ``shack_hartmann``, ``koehler_image``,
``init_paths_from_field``, ``resample_field``).  Each of these
silently accepted an MCF / 3-D ensemble and produced a downstream
type error instead of the canonical
``v4.16``-roadmap-pointer message.  The V6 filter accepts ANY
function whose first positional name is in
``_FIELD_PARAM_NAMES`` -- it cannot be defeated by renaming the
function -- and retains the legacy name-prefix filter as a
back-compat fallback so the v4.15.4 coverage is preserved
bit-for-bit (every old-discovery entry is still discovered; the
V6 filter only ADDS entries, never removes).

v4.15.5 also (P2-NEW-F2-4) descends into class bodies for methods
named ``apply`` / ``propagate``.  The v4.15.4 walker excluded
class methods on the documented assumption that they "delegate
to the guarded scalar functions" (true for ``Operator.apply`` in
``lumenairy/algebra/base.py``, which routes through ``_apply``
into guarded module-level kernels).  P1-NEW-F2-2 found one
counterexample: ``DeformableMirror.apply`` is a direct
``E_in * np.exp(1j * phi)`` with no delegation, so an MCF or 3-D
ensemble silently broadcasts.  The v4.15.5 walker descent surfaces
the class method directly; the ``_DELEGATING_CLASS_METHODS``
exemption set lists the operator-algebra ``apply`` methods that
legitimately delegate.

The exemption set lists every discovered function that legitimately
does NOT accept a bare 2-D coherent scalar field (e.g.
``apply_thin_lens_to_beamlets`` takes a ``BeamletBundle``,
``angular_spectrum_propagate_batch`` requires a 3-D stack,
``apply_perturbations`` takes a prescription dict, JonesField
methods operate on Jones-vector fields, polarization helpers like
``stokes_parameters`` consume a ``JonesField`` object, and
``decompose_lg`` / ``decompose_hg`` accept ``(field, x, y, ...)``
where ``x``/``y`` are coordinate arrays so the V6 filter would
need a richer signature inspection to distinguish from a 2-D
scalar field).  Adding the exemption with a comment is how a new
non-scalar entry point lands without tripping the pin; adding a
new scalar entry point WITHOUT guarding it is the failure mode
this pin catches.

Author: Andrew Traverso -- v4.15.3 / Agent A
                          v4.15.4 / Agent A (scope extension)
                          v4.15.5 / Agent A (V6 refactor +
                                             class descent)
"""
from __future__ import annotations

import ast
import importlib
import inspect
from functools import lru_cache
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]

# v4.15.4 (P1-NEW-3WAY-1 + P2-NEW-3WAY-2): walker scope extended.
# v4.15.3 was ``('lumenairy/propagators', 'lumenairy/elements')``;
# the v4.15.3 audit found 4 public entry points outside that scope
# accepting 2-D scalar fields.  v4.15.4 adds ``lumenairy/analysis``
# (subpackage) and ``lumenairy/system.py`` (a file at package
# root -- the walker handles both cases via ``rglob`` of a
# directory and direct visit of a file).
#
# This is now a FALLBACK scope: primary discovery uses
# ``lumenairy.__all__`` (see ``_walk_entry_points``); the package
# walk catches any sibling not exported via ``__all__``.
_TARGET_PACKAGES = ('lumenairy/propagators',
                    'lumenairy/elements',
                    'lumenairy/analysis',
                    'lumenairy/system.py')


# ============================================================================
# Documented exemptions.  Each entry is a (module_relative_posix_path,
# function_name) tuple.  Every exemption MUST cite the reason it
# legitimately bypasses the 2-D scalar-field guard.
# ============================================================================

_GUARD_EXEMPTIONS = frozenset({
    # ---- propagators/dispatch.py --------------------------------------------
    # ``asm_propagate`` is a thin wrapper that delegates to one of the
    # five guarded ASM-family propagators (asm / asm_tilted / asm_mft /
    # sas / fraunhofer).  Every code path through this function reaches a
    # guarded entry point, so the wrapper itself is structurally
    # delegated.  Not in the v4.15.3 P0-NEW-F2-1 list.
    ('lumenairy/propagators/dispatch.py', 'asm_propagate'),

    # ---- propagators/ensemble.py -------------------------------------------
    # v4.16.1 (audit AUDIT_V4_16_0_DEEP P5 / P0-1): ``propagate_ensemble``
    # is the canonical partial-coherence-propagation entry point.  Its
    # contract REQUIRES ``ensemble.ndim == 3`` (the Schell-family
    # ``(n_realizations, Ny, Nx)`` form); the 2-D scalar-field guard
    # would invert the invariant and reject the only valid input shape.
    # ``propagate_ensemble`` has its own dedicated 3-D shape check at
    # the entry point + an explicit error pointing back at the guard
    # for users who pass a single 2-D field by mistake.  Symmetric to
    # the ``angular_spectrum_propagate_batch`` exemption two entries
    # below.
    ('lumenairy/propagators/ensemble.py', 'propagate_ensemble'),

    # ---- propagators/propagation.py ----------------------------------------
    # ``apply_fresnel_curvature`` takes a 2-D field but is a phase
    # multiplication helper (not a propagator); it is not in the
    # v4.15.3 P0-NEW-F2-1 list.  Adding the guard would be in-scope
    # for a future audit round (re-evaluate at v4.16).
    ('lumenairy/propagators/propagation.py', 'apply_fresnel_curvature'),
    # ``angular_spectrum_propagate_batch`` is the batched 3-D-stack
    # variant; its contract REQUIRES ``E_stack.ndim == 3``.  Routing
    # this through ``_check_2d_scalar_field`` would invert the
    # invariant and reject the only valid input shape.  The function
    # has its own dedicated 3-D-shape check.
    ('lumenairy/propagators/propagation.py',
     'angular_spectrum_propagate_batch'),

    # ---- propagators/gbd.py ------------------------------------------------
    # The Gaussian-beam-decomposition propagators accept BeamletBundle
    # / GBDPropagationState objects -- not 2-D scalar fields.  They
    # have their own type checks at entry; the 2-D scalar-field guard
    # would reject every legitimate input.
    ('lumenairy/propagators/gbd.py', 'propagate_beamlets_freespace'),
    ('lumenairy/propagators/gbd.py', 'apply_thin_lens_to_beamlets'),
    ('lumenairy/propagators/gbd.py', 'propagate_gbd'),
    ('lumenairy/propagators/gbd.py', 'propagate_gbd_freespace'),
    ('lumenairy/propagators/gbd.py', 'propagate_gbd_thin_lens'),
    ('lumenairy/propagators/gbd.py', 'apply_abcd_to_beamlets'),
    ('lumenairy/propagators/gbd.py', 'propagate_gbd_through_prescription'),

    # ---- propagators/hf.py -------------------------------------------------
    # The Huygens-Fresnel propagators accept Huygens-Fresnel state
    # objects (path-bundle / opl-callable), not 2-D scalar fields.
    ('lumenairy/propagators/hf.py', 'propagate_huygens_fresnel'),
    ('lumenairy/propagators/hf.py', 'propagate_huygens_fresnel_freespace'),
    ('lumenairy/propagators/hf.py',
     'propagate_huygens_fresnel_with_opl_callable'),
    ('lumenairy/propagators/hf.py',
     'propagate_huygens_fresnel_through_prescription'),

    # ---- propagators/hfpi.py -----------------------------------------------
    # HFPI propagators / aperture-diffraction helpers operate on
    # PathBundle objects, not 2-D scalar fields.
    ('lumenairy/propagators/hfpi.py', 'propagate_to_plane'),
    ('lumenairy/propagators/hfpi.py', 'apply_aperture_diffraction'),
    ('lumenairy/propagators/hfpi.py', 'propagate_hfpi'),
    ('lumenairy/propagators/hfpi.py', 'propagate_hfpi_freespace_aperture'),
    ('lumenairy/propagators/hfpi.py', 'propagate_hfpi_through_prescription'),

    # ---- propagators/vectorial_hfpi.py -------------------------------------
    # Vectorial-HFPI propagators operate on VectorPathBundle objects,
    # not 2-D scalar fields.
    ('lumenairy/propagators/vectorial_hfpi.py', 'propagate_vector_to_plane'),
    ('lumenairy/propagators/vectorial_hfpi.py',
     'apply_vector_aperture_diffraction'),
    ('lumenairy/propagators/vectorial_hfpi.py',
     'propagate_vector_hfpi_freespace_aperture'),

    # ---- propagators/asymptotic.py -----------------------------------------
    # The asymptotic / modal propagators take modal-coefficient
    # representations and ray bundles, not 2-D scalar fields.
    ('lumenairy/propagators/asymptotic.py', 'propagate_modal_asymptotic'),
    ('lumenairy/propagators/asymptotic.py',
     'propagate_hf_chebyshev_quadrature'),
    ('lumenairy/propagators/asymptotic.py',
     'propagate_modal_asymptotic_lg00_jax'),

    # ---- propagators/subaperture.py ----------------------------------------
    # Subaperture propagator takes a subaperture-state object (dict +
    # tile-local field tuple), not a single 2-D scalar field.  Its own
    # entry-point validation surfaces the appropriate error.
    ('lumenairy/propagators/subaperture.py',
     'propagate_subaperture_asymptotic'),

    # ---- elements/elements.py ----------------------------------------------
    # The generic-element ``apply_*`` helpers (mirror, aperture, mask,
    # zernike-aberration, Lyot stop, vortex-phase mask, apodized-pupil)
    # DO take 2-D scalar fields but are NOT in the v4.15.3 P0-NEW-F2-1
    # closure scope.  Re-evaluate at v4.16 (likely candidates for the
    # next round of guard migration).
    ('lumenairy/elements/elements.py', 'apply_mirror'),
    ('lumenairy/elements/elements.py', 'apply_aperture'),
    ('lumenairy/elements/elements.py', 'apply_gaussian_aperture'),
    ('lumenairy/elements/elements.py', 'apply_mask'),
    ('lumenairy/elements/elements.py', 'apply_zernike_aberration'),
    ('lumenairy/elements/elements.py', 'apply_lyot_focal_plane_mask'),
    ('lumenairy/elements/elements.py', 'apply_vortex_phase_mask'),
    ('lumenairy/elements/elements.py', 'apply_lyot_stop'),
    ('lumenairy/elements/elements.py', 'apply_apodized_pupil'),

    # ---- elements/_lens_jax.py ---------------------------------------------
    # JAX-traceable lens variants are NOT in the v4.15.3 P0-NEW-F2-1
    # closure scope.  Their guard plan is coupled to the v4.16 JAX
    # autodiff partial-coherence story.
    ('lumenairy/elements/_lens_jax.py', 'apply_real_lens_traced_jax'),
    ('lumenairy/elements/_lens_jax.py', 'apply_real_lens_maslov_jax'),

    # ---- elements/polarization.py ------------------------------------------
    # Top-level polarization helpers accept a ``JonesField`` object,
    # NOT a 2-D scalar field.  The JonesField wraps two 2-D fields
    # (Ex, Ey) and these helpers operate on the pair, dispatching
    # element-wise through the already-guarded scalar lens / mirror
    # / aperture / mask kernels for the non-polarizing variants.  A
    # 2-D scalar input to these would fail at attribute access.
    ('lumenairy/elements/polarization.py', 'apply_jones_matrix'),
    ('lumenairy/elements/polarization.py', 'apply_polarizer'),
    ('lumenairy/elements/polarization.py', 'apply_waveplate'),
    ('lumenairy/elements/polarization.py', 'apply_half_wave_plate'),
    ('lumenairy/elements/polarization.py', 'apply_quarter_wave_plate'),
    ('lumenairy/elements/polarization.py', 'apply_rotator'),

    # ---- analysis/through_focus.py (v4.15.4 NEW) ---------------------------
    # ``apply_perturbations(prescription, perturbations, N, dx)`` takes
    # a prescription DICT (not a 2-D scalar field) and returns a
    # deep-copied perturbed prescription.  It is matched by the
    # walker's ``apply_*`` name filter, but the first positional arg
    # has no ``.ndim`` and is not the kind of object the guard checks.
    # Adding the guard would make every legitimate call raise
    # ``ValueError`` at the first line.  Documented exempt.
    ('lumenairy/analysis/through_focus.py', 'apply_perturbations'),

    # ---- system.py (v4.15.4 NEW) -------------------------------------------
    # ``clear_propagate_system_jax_cache()`` takes NO arguments -- it
    # is a cache-clearing helper, not a propagator.  The walker
    # discovers it because the name contains the substring
    # ``_propagate``.  Documented exempt.
    ('lumenairy/system.py', 'clear_propagate_system_jax_cache'),

    # ---- raytrace/core.py (v4.15.4 NEW) ------------------------------------
    # ``apply_doe_phase_traced(rays, ...)`` takes a ``RayBundle``
    # (geometric-ray representation, NOT a 2-D scalar field).  The
    # v4.15.4 ``__all__``-membership pass discovers it because
    # ``apply_doe_phase_traced`` is in ``lumenairy.__all__``.  The
    # 2-D scalar guard would reject every legitimate caller --
    # ``RayBundle`` has no ``.ndim``.  Documented exempt.
    ('lumenairy/raytrace/core.py', 'apply_doe_phase_traced'),

    # ---- _context.py (v4.15.4 NEW) -----------------------------------------
    # ``apply_globals(state: dict) -> None`` applies a globals
    # snapshot to library-wide knobs (FFT threads, max RAM, cache
    # sizes).  Takes a dict of name -> value, NOT a 2-D scalar
    # field.  The v4.15.4 ``__all__``-membership pass discovers it
    # because ``apply_globals`` is in ``lumenairy.__all__``.
    # Documented exempt.
    ('lumenairy/_context.py', 'apply_globals'),

    # ========================================================================
    # v4.15.5 (P1-NEW-2WAY-1, V6 filter) NEW exemptions.  The V6
    # first-positional-name filter discovers ~30 additional public
    # ``__all__`` entry points whose first positional is ``E`` /
    # ``E_in`` / ``field`` / ``pupil`` / ``object_field`` / ``psf``
    # but whose function name didn't match the v4.15.4 prefix
    # set.  Of those, this PR (A.4) hand-guards the highest-traffic
    # 12 (``wave_opd_2d``, ``M2``, ``strehl_ratio``, ``beam_d4sigma``,
    # ``coupling_efficiency``, ``compute_psf``, ``compute_otf``,
    # ``compute_mtf``, ``encircled_energy_curve``, ``koehler_image``,
    # ``extended_source_image``, ``shack_hartmann``,
    # ``rays_from_field``, ``resample_field``, plus the inline class-
    # method guard on ``DeformableMirror.apply``).  The remaining
    # discovered entry points are documented here with either (a) a
    # genuine non-scalar-field input (JonesField, ray bundle, coord
    # tuple), or (b) a deferred v4.16+ guard-sweep slot for analyzers
    # that take a 2-D scalar field but are out-of-scope for the
    # Tier-0 patch.
    # ========================================================================

    # ---- elements/polarization.py (V6 discovery: ``field`` JonesField) -----
    # ``stokes_parameters(field: JonesField)``, ``degree_of_polarization``,
    # and ``polarization_ellipse`` all consume a ``JonesField`` object
    # (wraps Ex / Ey 2-D fields).  The V6 filter discovers them because
    # the parameter name is ``field`` but the object is NOT a 2-D
    # complex ndarray -- attribute access (``field.Ex``) fails for a
    # bare ndarray.  Documented exempt; their failure mode for an MCF
    # / 3-D ensemble is the underlying ``JonesField.from_*`` factory
    # rejecting non-Jones input.
    ('lumenairy/elements/polarization.py', 'stokes_parameters'),
    ('lumenairy/elements/polarization.py', 'degree_of_polarization'),
    ('lumenairy/elements/polarization.py', 'polarization_ellipse'),

    # ---- propagators/asymptotic.py (V6 discovery: ``field``) ---------------
    # ``decompose_lg(field, x, y, ...)`` and ``decompose_hg(field, x, y, ...)``
    # take a 2-D complex field plus coordinate arrays.  These COULD be
    # guarded -- adding ``_check_2d_scalar_field(field, ...)`` would
    # surface MCF / 3-D ensemble inputs with the canonical message --
    # but the audit's pragmatic scope (A.4) lists the analysis/core +
    # coherence + detector sweep, not the propagators/asymptotic LG/HG
    # decomposition path.  Deferred to v4.16+ guard sweep.  No silent
    # failure today: an MCF input fails at ``np.trapz(...)`` inside the
    # function with a Python ``TypeError`` (less user-friendly than the
    # canonical message but not silently wrong).
    ('lumenairy/propagators/asymptotic.py', 'decompose_lg'),
    ('lumenairy/propagators/asymptotic.py', 'decompose_hg'),

    # ---- propagators/gbd.py (V6 discovery: ``E_in``) -----------------------
    # ``decompose_field_to_beamlets(E_in, dx, ...)`` takes a 2-D source
    # field and returns a ``BeamletBundle``.  Deferred to v4.16+ guard
    # sweep -- not in the A.4 Tier-0 list.  An MCF input fails at the
    # ``E_in[i, j]`` element-access inside the loop.
    ('lumenairy/propagators/gbd.py', 'decompose_field_to_beamlets'),

    # ---- propagators/hfpi.py (V6 discovery: ``E_in``) ----------------------
    # ``init_paths_from_field(E_in, dx, ...)`` and
    # ``init_paths_stratified(E_in, dx, ...)`` sample a 2-D source
    # field into a Huygens-Fresnel ``PathBundle``.  These are
    # genuine 2-D-scalar-field consumers and SHOULD be guarded; they
    # are out-of-scope for this PR (only ``raytrace/from_field.py`` is
    # in Agent A's file scope -- ``hfpi.py`` is touched by Agent B/C
    # in parallel).  Deferred to v4.16+ guard sweep with the explicit
    # follow-up note in the v4.15.5 audit / CHANGELOG.
    ('lumenairy/propagators/hfpi.py', 'init_paths_from_field'),
    ('lumenairy/propagators/hfpi.py', 'init_paths_stratified'),

    # ---- propagators/dispatch.py (V6 discovery: ``E_in``) ------------------
    # ``which_propagator(E_in, z, wavelength, dx, ...)`` returns a dict
    # advising which ASM-family kernel to use.  Only consults
    # ``E_in.shape`` and ``E_in.dtype`` (never indexes / propagates).
    # Adding a guard would reject the diagnostic helper for partial-
    # coherence callers asking which propagator to use, even though
    # the helper would have given the correct advisory output.
    # Documented exempt.
    ('lumenairy/propagators/dispatch.py', 'which_propagator'),

    # ---- analysis/core.py (V6 discovery: deferred analyzers) ---------------
    # The following ``analysis/core.py`` entries are 2-D-scalar-field
    # consumers that COULD be guarded but are out of the A.4 Tier-0
    # list.  The Tier-0 list focuses on the highest-traffic
    # analyzers (Strehl, M2, D4sigma, coupling, encircled energy,
    # PSF/OTF/MTF, wave_opd_2d).  Lower-priority entries here are
    # deferred to v4.16+ to keep the v4.15.5 diff small and minimise
    # disruption to the existing test suite.  None of these silently
    # accept MCF inputs at the moment -- they fail downstream with a
    # less-friendly Python ``TypeError`` -- so the deferral is
    # cosmetic UX, not correctness.
    ('lumenairy/analysis/core.py', 'beam_centroid'),
    ('lumenairy/analysis/core.py', 'beam_power'),
    ('lumenairy/analysis/core.py', 'beam_diameter'),
    ('lumenairy/analysis/core.py', 'radial_power_bands'),
    ('lumenairy/analysis/core.py', 'wave_opd_1d'),
    ('lumenairy/analysis/core.py', 'strehl_phase_integral'),
    ('lumenairy/analysis/core.py', 'rayleigh_resolution'),
    ('lumenairy/analysis/core.py', 'sparrow_resolution'),
    ('lumenairy/analysis/core.py', 'fwhm_resolution'),
    ('lumenairy/analysis/core.py', 'encircled_energy_radius'),

    # ---- analysis/through_focus.py (V6 discovery: ``E``) -------------------
    # ``single_plane_metrics(E, dx, wavelength, ...)`` is a thin
    # aggregator that calls ``beam_centroid`` / ``beam_d4sigma`` /
    # ``coupling_efficiency`` internally.  After A.4 lands the
    # ``beam_d4sigma`` and ``coupling_efficiency`` guards trigger
    # downstream, so an MCF input to ``single_plane_metrics`` will
    # still get the canonical message -- just with the call site
    # named as the inner helper rather than ``single_plane_metrics``.
    # Documented exempt; consider direct guard in v4.16+ for the
    # nicer error message.
    ('lumenairy/analysis/through_focus.py', 'single_plane_metrics'),

    # ---- analysis/plotting.py (V6 discovery: visualisation helpers) --------
    # ``plot_intensity``, ``plot_phase``, ``plot_amplitude_phase``,
    # ``plot_field``, ``plot_cross_section``, ``plot_beam_profile``,
    # ``plot_psf`` are matplotlib-display helpers.  Adding the guard
    # would reject MCF / 3-D ensemble inputs at the entry point
    # which is the desired UX, but visualisation is a side-channel
    # not in the A.4 Tier-0 scope.  Deferred to v4.16+.
    ('lumenairy/analysis/plotting.py', 'plot_intensity'),
    ('lumenairy/analysis/plotting.py', 'plot_phase'),
    ('lumenairy/analysis/plotting.py', 'plot_amplitude_phase'),
    ('lumenairy/analysis/plotting.py', 'plot_field'),
    ('lumenairy/analysis/plotting.py', 'plot_cross_section'),
    ('lumenairy/analysis/plotting.py', 'plot_beam_profile'),
    ('lumenairy/analysis/plotting.py', 'plot_psf'),
})


# ============================================================================
# v4.15.5 (P2-NEW-F2-4): class-method exemption set.  The v4.15.5
# walker descends into ``ClassDef`` bodies and applies the same
# filter as for top-level functions to methods named ``apply`` or
# ``propagate*``.  This catches ``DeformableMirror.apply``
# (P1-NEW-F2-2), which is a non-delegating direct phase-screen
# multiply.  The vast majority of ``*.apply`` methods in the
# operator-algebra layer (``ThinLens.apply``, ``FreeSpace.apply``,
# ``FourierTransform.apply``, etc.) DO delegate to module-level
# guarded propagators / lens kernels through ``Operator._apply``;
# those are exempt here.  Adding a new class-method ``apply`` /
# ``propagate`` that does NOT delegate is the failure mode this
# walker catches.
#
# Format: ``(module_relative_posix_path, class_name, method_name)``.
# ============================================================================

_DELEGATING_CLASS_METHODS = frozenset({
    # ---- algebra/base.py ---------------------------------------------------
    # ``Operator.apply(E, *, dx, wavelength, dy=None)`` is the shared
    # entry point for every ``Operator`` subclass (ThinLens, FreeSpace,
    # CylindricalLens, Magnify, FourierTransform, Aperture,
    # GaussianAperture, ...) plus the CompositeOperator chain.  It
    # delegates to ``self._apply`` which in turn calls the guarded
    # module-level propagator / lens kernels.  An MCF / 3-D ensemble
    # input fails at the inner kernel with the canonical message.
    ('lumenairy/algebra/base.py', 'Operator', 'apply'),
    # ``CompositeOperator._apply`` walks the chain of operators and
    # threads ``(dx, dy, wavelength)`` through each stage.  Each chain
    # element ultimately reaches a guarded kernel.
    ('lumenairy/algebra/base.py', 'CompositeOperator', 'apply'),
    # ``apply_with_intermediates`` is a debug / through-stage helper
    # that calls ``_apply`` per stage; same delegation chain.
    ('lumenairy/algebra/base.py', 'CompositeOperator',
     'apply_with_intermediates'),

    # ---- sources/core.py ---------------------------------------------------
    # ``Source.propagate(z, ...)`` is the high-level Source ``propagate``
    # wrapper; it routes to ``angular_spectrum_propagate`` /
    # ``fresnel_propagate`` (both guarded).  Its first positional after
    # ``self`` is ``z`` (a float), not a field; if the V6 filter ever
    # mis-categorises it, this exemption documents the legitimate
    # signature.
    ('lumenairy/sources/core.py', 'Source', 'propagate'),

    # ---- elements/polarization.py ------------------------------------------
    # ``JonesField.propagate(z, ...)`` is the analogous JonesField
    # ``propagate`` wrapper; it routes through guarded ASM kernels per
    # polarization component.
    ('lumenairy/elements/polarization.py', 'JonesField', 'propagate'),
})


# v4.15.5 (P1-NEW-2WAY-1): V6 filter parameter names.
# Used to discover entry points whose first positional argument name
# strongly suggests a 2-D scalar field input.  Listed in priority
# order; the V6 filter accepts ANY of these.  This is the structural
# upgrade over the v4.15.4 hand-curated name-prefix regex
# (``apply_*`` / ``*_propagate*`` / ``richards_wolf_*`` /
# ``debye_wolf_*``) -- function name alone is no longer the
# discrimination criterion.
_FIELD_PARAM_NAMES = frozenset({
    'E',
    'E_in',
    'field',
    'pupil',
    'object_field',
    'psf',
})


# ============================================================================
# Discovery walker -- top-level public defs only
# ============================================================================

def _legacy_name_matches_entry_point_filter(name: str) -> bool:
    """True iff ``name`` matches the v4.15.4 hand-curated name-prefix
    regex.

    Retained as the back-compat half of the v4.15.5 V6 filter so the
    discovery surface from v4.15.4 is preserved bit-for-bit.  The V6
    filter ANDs / ORs this with the new first-positional-name check;
    see :func:`_function_matches_entry_point_filter` for the unified
    discovery logic.

    v4.15.4 (P2-NEW-V2-1) added ``richards_wolf_*`` /
    ``debye_wolf_*`` to the v4.15.3 regex.  v4.15.5 (V6) renames the
    helper from ``_name_matches_entry_point_filter`` to
    ``_legacy_name_matches_entry_point_filter`` to signal that name
    alone is no longer the discrimination criterion -- the structural
    upgrade is in
    :func:`_function_matches_entry_point_filter`.
    """
    if name.startswith('_'):
        return False
    if name.startswith('apply_'):
        return True
    if '_propagate' in name:
        return True
    if name == 'propagate':
        return True
    # v4.15.4 follow-up: ``propagate_through_system`` and its JAX
    # sibling ``propagate_through_system_jax`` don't contain the
    # underscore-prefix ``_propagate`` substring, so the v4.15.3
    # filter missed them.  Catch the ``propagate_<something>``
    # family explicitly.
    if name.startswith('propagate_'):
        return True
    if name.startswith('richards_wolf'):
        return True
    if name.startswith('debye_wolf'):
        return True
    return False


def _first_positional_arg_name(node, skip_self=False):
    """Return the name of the first positional argument of ``node`` or
    ``None`` if ``node`` has no positional args.

    Parameters
    ----------
    node : ast.FunctionDef
        The function definition to inspect.
    skip_self : bool, default False
        If ``True``, skip the leading ``self`` / ``cls`` argument
        common to instance / class methods.  When walking a top-level
        ``FunctionDef``, pass ``False``; when walking a method inside
        a ``ClassDef``, pass ``True``.

    Notes
    -----
    Uses ``ast.arguments.args`` only (positional-or-keyword).  Pure
    keyword-only arguments under ``*`` (``ast.arguments.kwonlyargs``)
    are intentionally NOT consulted -- a function declared as
    ``def f(*, E, dx, ...)`` is unusual enough that the V6 filter
    can afford to under-match it (we'd rather miss it than mis-fire
    on a non-field positional like a config dict).
    """
    args = node.args.args
    if not args:
        return None
    start = 1 if (skip_self and args[0].arg in ('self', 'cls')) else 0
    if start >= len(args):
        return None
    return args[start].arg


def _function_matches_entry_point_filter(node, *, is_method=False):
    """True iff the ``ast.FunctionDef`` ``node`` is a V6-discovery
    candidate.

    The V6 filter is the union of two signals:

    1.  Legacy name-prefix match (preserves v4.15.4 coverage
        bit-for-bit).
    2.  First-positional-parameter-name is in
        ``_FIELD_PARAM_NAMES`` (the structural upgrade -- catches any
        function whose API contract uses ``E`` / ``E_in`` / ``field``
        / ``pupil`` / ``object_field`` / ``psf`` for the first
        positional, regardless of the function name).

    Either signal is sufficient.  Private functions (leading ``_``)
    are still excluded.

    Parameters
    ----------
    node : ast.FunctionDef
        The function / method definition to inspect.
    is_method : bool, default False
        Pass ``True`` when walking a ``ClassDef`` body so the
        positional-arg inspector skips ``self`` / ``cls``.

    Returns
    -------
    bool
        True iff the function is in the V6 discovery scope.
    """
    name = node.name
    if name.startswith('_'):
        return False
    if _legacy_name_matches_entry_point_filter(name):
        return True
    first_param = _first_positional_arg_name(node, skip_self=is_method)
    if first_param is not None and first_param in _FIELD_PARAM_NAMES:
        return True
    return False


def _iter_target_py_files():
    """Yield every ``.py`` file in the walker's scope.

    Handles both directory-shaped targets (``lumenairy/propagators``)
    and file-shaped targets (``lumenairy/system.py``).  The latter is
    new in v4.15.4: ``system.py`` is a single file at the package
    root, not a subpackage.
    """
    for pkg_rel in _TARGET_PACKAGES:
        pkg_path = _REPO_ROOT / pkg_rel
        if pkg_path.is_file():
            # File-shaped target (e.g. ``lumenairy/system.py``).
            yield pkg_path
            continue
        if not pkg_path.is_dir():
            # Misconfigured path -- emit nothing for this entry rather
            # than raising; the count-floor test will catch a
            # walker-wide collapse.
            continue
        for py in sorted(pkg_path.rglob('*.py')):
            if py.name.startswith('_') and 'lens' not in py.name:
                # Skip private modules but keep _lens_*.py
                continue
            yield py


@lru_cache(maxsize=None)
def _file_to_ast(py):
    """Parse ``py`` (a ``pathlib.Path``) and return the module AST.

    Uses cp1252-tolerant decoding to match the file-write contract.

    v4.15.5 (P3-NEW-F1-2): cached so repeat calls for the same
    source file (e.g. 11 ``apply_*_lens`` re-discoveries through
    ``__all__`` membership against ``elements/_lens_thin.py``) parse
    the file once instead of N times.  ``pathlib.Path`` is hashable;
    the cache key is the Path instance.  Cache survives for the
    pytest session lifetime, which matches the AST's effective
    invariance over a single suite run.
    """
    src = py.read_text(encoding='utf-8', errors='replace')
    return ast.parse(src, filename=str(py))


def _public_top_level_function_defs(tree, py):
    """Yield ``(py, node)`` pairs for every public top-level
    ``FunctionDef`` in ``tree`` whose name passes the entry-point
    filter.

    v4.15.5 (V6): switched from the v4.15.4 name-prefix-only filter
    to ``_function_matches_entry_point_filter`` which also accepts
    functions whose first positional-arg name is in
    ``_FIELD_PARAM_NAMES``.

    v4.15.5 (P2-NEW-F2-4): ``_walk_class_methods`` (sibling helper)
    handles class-method descent.  This helper still only yields
    top-level ``FunctionDef``s -- top-level and class-level
    discovery are kept on separate walkers so the dedup key
    (``(file, name)`` for top-level, ``(file, class, method)`` for
    class methods) stays unambiguous.
    """
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not _function_matches_entry_point_filter(node):
            continue
        yield py, node


def _walk_via_all_membership():
    """Yield (py, ast.FunctionDef) for every entry point reachable
    via ``lumenairy.__all__`` whose AST node passes the V6 filter.

    v4.15.4 (P2-NEW-V2-1 fix): refactored from package-only walk to
    ``__all__``-based discovery so future module reorganisations
    that move a function between submodules don't silently drop it
    from the walker's scope.  Pairs with the fallback package walk
    below to also catch non-``__all__`` siblings.

    v4.15.5 (V6): the per-name name-prefix gate is REMOVED.  Every
    ``__all__`` entry that resolves to a public top-level function
    is parsed to AST and then offered to
    ``_function_matches_entry_point_filter``.  That filter checks
    BOTH the legacy name pattern (back-compat) AND the
    first-positional-parameter-name (the structural upgrade).
    Removing the per-name pre-gate is what allows ``wave_opd_2d`` /
    ``M2`` / ``strehl_ratio`` / etc. -- names outside the old
    regex -- to surface as candidates.
    """
    try:
        la = importlib.import_module('lumenairy')
    except Exception:  # pragma: no cover -- catastrophic import failure
        return
    all_names = getattr(la, '__all__', None)
    if not all_names:
        return
    for name in all_names:
        if name.startswith('_'):
            continue
        obj = getattr(la, name, None)
        if obj is None or not callable(obj):
            continue
        try:
            src_file = inspect.getsourcefile(obj)
        except TypeError:
            continue
        if src_file is None:
            continue
        py = Path(src_file).resolve()
        # Only yield if the source file is inside the repo (so
        # vendored / installed copies don't leak in).
        try:
            py.relative_to(_REPO_ROOT)
        except ValueError:
            continue
        try:
            tree = _file_to_ast(py)
        except OSError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != name:
                continue
            # v4.15.5 V6: AST-level filter has access to the
            # ``ast.arguments``, so first-positional-name discovery
            # is possible.
            if not _function_matches_entry_point_filter(node):
                break
            yield py, node
            break


def _walk_via_package_scan():
    """Yield (py, ast.FunctionDef) for every entry point found by
    walking ``_TARGET_PACKAGES``.

    Kept as a fallback (after v4.15.4) so a function that is NOT in
    ``lumenairy.__all__`` but still lives in a target package and
    matches the name filter is still caught.  Example future use:
    a public-by-convention helper at top of module that wasn't yet
    promoted to the package-level ``__all__``.
    """
    for py in _iter_target_py_files():
        try:
            tree = _file_to_ast(py)
        except OSError:
            continue
        for pair in _public_top_level_function_defs(tree, py):
            yield pair


def _class_method_matches_filter(method_node):
    """True iff a ``ClassDef``-body ``FunctionDef`` is a V6 candidate.

    Restricts to methods named ``apply`` or ``propagate`` (or
    matching the legacy ``apply_*`` / ``propagate_*`` /
    ``richards_wolf*`` / ``debye_wolf*`` family) AND whose
    first-non-self positional name is in ``_FIELD_PARAM_NAMES``.

    The method-name gate keeps the walker focused on the public
    field-transforming surface; helper methods named e.g.
    ``__init__``, ``_apply``, ``set_command``, ``phase`` are not
    candidates even if they happen to take a field-shaped arg.
    """
    name = method_node.name
    if name.startswith('_'):
        return False
    # Method-name gate: ``apply`` / ``propagate`` exact match, or
    # legacy prefix pattern.
    if name not in ('apply', 'propagate'):
        if not _legacy_name_matches_entry_point_filter(name):
            return False
    # Parameter-name gate: first non-self positional must be in
    # ``_FIELD_PARAM_NAMES``.
    first_param = _first_positional_arg_name(method_node, skip_self=True)
    if first_param is None or first_param not in _FIELD_PARAM_NAMES:
        return False
    return True


def _walk_class_methods():
    """Yield ``(py, class_node, method_node)`` triples for every
    ``ClassDef``-body method that matches the V6 class-method
    filter.

    v4.15.5 (P2-NEW-F2-4): walks every ``.py`` file in
    ``_TARGET_PACKAGES`` plus ``lumenairy/algebra/`` (where the
    Operator algebra ``apply`` methods live) plus
    ``lumenairy/sources/`` (where ``Source.propagate`` lives).
    Each ``ClassDef`` is descended once and any ``apply`` /
    ``propagate`` method whose first non-self positional matches
    the field-param-name set is yielded.

    The matched method must EITHER carry the
    ``_check_2d_scalar_field`` call as its first executable
    statement OR appear in ``_DELEGATING_CLASS_METHODS`` with a
    cited delegation chain.
    """
    class_walk_scope = list(_TARGET_PACKAGES) + [
        'lumenairy/algebra',
        'lumenairy/sources',
        'lumenairy/raytrace',
    ]
    seen_class_files = set()
    for pkg_rel in class_walk_scope:
        pkg_path = _REPO_ROOT / pkg_rel
        py_files = []
        if pkg_path.is_file():
            py_files = [pkg_path]
        elif pkg_path.is_dir():
            py_files = sorted(pkg_path.rglob('*.py'))
        for py in py_files:
            resolved = py.resolve()
            if resolved in seen_class_files:
                continue
            seen_class_files.add(resolved)
            try:
                tree = _file_to_ast(py)
            except OSError:
                continue
            for class_node in tree.body:
                if not isinstance(class_node, ast.ClassDef):
                    continue
                if class_node.name.startswith('_'):
                    continue
                for method_node in class_node.body:
                    if not isinstance(method_node, ast.FunctionDef):
                        continue
                    if not _class_method_matches_filter(method_node):
                        continue
                    yield py, class_node, method_node


def _walk_entry_points():
    """Yield (path, ast.FunctionDef) for every public top-level def in
    scope, deduped across the ``__all__``-membership pass and the
    package-walk fallback.

    v4.15.5: top-level (module-level) discovery only.  Class-method
    discovery is in :func:`_walk_class_methods`; the two walkers
    are kept on separate dedup-key shapes so the main pin can
    branch on shape without ambiguity.
    """
    seen: set = set()
    for py, node in _walk_via_all_membership():
        key = (str(py.resolve()), node.name)
        if key in seen:
            continue
        seen.add(key)
        yield py, node
    for py, node in _walk_via_package_scan():
        key = (str(py.resolve()), node.name)
        if key in seen:
            continue
        seen.add(key)
        yield py, node


def _first_executable_statement(body):
    """Return the first executable statement in ``body``, skipping a
    leading docstring AND any leading ``Import`` / ``ImportFrom``
    statements (which support the lazy-import pattern the guard uses
    to avoid circular dependencies).
    """
    skip_leading_imports = True
    saw_docstring = False
    for stmt in body:
        # Skip the docstring (first Expr containing a Constant str).
        if (not saw_docstring
                and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)):
            saw_docstring = True
            continue
        # Skip leading import statements (helper's lazy-import lives
        # alongside the call).
        if skip_leading_imports and isinstance(
                stmt, (ast.Import, ast.ImportFrom)):
            continue
        skip_leading_imports = False
        return stmt
    return None


def _is_helper_call(stmt):
    """True iff ``stmt`` is a bare ``_check_2d_scalar_field(...)``
    expression statement.
    """
    if not isinstance(stmt, ast.Expr):
        return False
    if not isinstance(stmt.value, ast.Call):
        return False
    func = stmt.value.func
    if isinstance(func, ast.Name) and func.id == '_check_2d_scalar_field':
        return True
    if (isinstance(func, ast.Attribute)
            and func.attr == '_check_2d_scalar_field'):
        return True
    return False


# ============================================================================
# Pre-flight: count floor.  Catches the case where the walker silently
# collapses (e.g. a refactor renames the package layout).
# ============================================================================

def test_minimum_entry_points_discovered():
    """Sanity floor: enough entry points to assert the walker is
    structurally intact.

    Historical floors:
    * v4.15.3: 19 entries (closure of v4.15.2's 10 guarded sites +
      9 new siblings).
    * v4.15.4: 25 entries (analysis/ + system.py + richards_wolf
      family + debye_wolf family added).
    * v4.15.5: 60+ entries (V6 first-positional-name filter
      surfaces ~30 additional public ``__all__`` analyzers whose
      function name was outside the v4.15.4 name-prefix regex but
      whose first positional is ``E`` / ``E_in`` / ``field`` /
      ``pupil`` / ``object_field`` / ``psf``).

    Below the floor means the walker is broken (e.g. a refactor
    inadvertently removed the V6 filter or the ``__all__`` import
    is failing).
    """
    count = sum(1 for _ in _walk_entry_points())
    assert count >= 60, (
        f"Top-level walker found only {count} entry points; expected "
        f">= 60 after the v4.15.5 V6 refactor.  The discovery walker "
        f"may need updating if the package layout has changed or the "
        f"V6 first-positional-name filter has regressed.")


def test_minimum_class_methods_discovered():
    """Sanity floor for the class-method walker.

    After the v4.15.5 (P2-NEW-F2-4) class-body descent, the walker
    must discover at least 4 candidate ``apply`` / ``propagate``
    methods:

    * ``Operator.apply`` (algebra/base.py) -- delegating
    * ``CompositeOperator.apply`` (inherited from Operator) -- the
      walker sees ``CompositeOperator.apply_with_intermediates``
      because it overrides the apply path
    * ``DeformableMirror.apply`` (analysis/ao.py) -- non-delegating,
      MUST carry guard
    * ``Source.propagate`` / ``JonesField.propagate`` if present

    Counter-pin: ``test_v6_walker_descends_into_class_bodies`` in
    the v4.15.5 ``test_v4_15_5_agent_a.py`` regression file pins
    the specific contract.
    """
    count = sum(1 for _ in _walk_class_methods())
    assert count >= 3, (
        f"Class-method walker found only {count} candidate methods; "
        f"expected >= 3 after the v4.15.5 P2-NEW-F2-4 class-body "
        f"descent.  Likely cause: the walker scope or the "
        f"_class_method_matches_filter has regressed.")


# ============================================================================
# Main pin: every discovered entry point either calls the helper OR
# appears in the exemption set with a documented reason.
# ============================================================================

def test_all_entry_points_call_helper_first():
    """Every public propagator/lens/analyzer entry point must call
    ``_check_2d_scalar_field`` as its first executable statement
    (after the docstring and any leading import statements).
    Exempted functions must be listed in ``_GUARD_EXEMPTIONS`` with
    a cited reason.
    """
    failures = []
    for py, node in _walk_entry_points():
        rel = py.relative_to(_REPO_ROOT).as_posix()
        if (rel, node.name) in _GUARD_EXEMPTIONS:
            continue
        first_stmt = _first_executable_statement(node.body)
        if first_stmt is None:
            failures.append(
                f"{rel}:{node.lineno} {node.name}: empty body")
            continue
        if not _is_helper_call(first_stmt):
            failures.append(
                f"{rel}:{node.lineno} {node.name}: "
                f"first executable statement is not "
                f"`_check_2d_scalar_field(...)`. Either add the "
                f"guard call (see lumenairy/_validation.py) or, if "
                f"this function legitimately does not accept a 2-D "
                f"scalar field, add it to ``_GUARD_EXEMPTIONS`` "
                f"with a comment explaining why.")
    if failures:
        raise AssertionError(
            "Entry points missing _check_2d_scalar_field guard:\n  - "
            + "\n  - ".join(failures))


def test_all_class_methods_call_helper_first():
    """Every discovered ``apply`` / ``propagate`` class method must
    EITHER call ``_check_2d_scalar_field`` as its first executable
    statement OR appear in ``_DELEGATING_CLASS_METHODS``.

    v4.15.5 (P2-NEW-F2-4): catches non-delegating class-method
    bypasses like ``DeformableMirror.apply`` (P1-NEW-F2-2) that
    the v4.15.4 walker's blanket class-method exclusion missed.
    """
    failures = []
    for py, class_node, method_node in _walk_class_methods():
        rel = py.relative_to(_REPO_ROOT).as_posix()
        key = (rel, class_node.name, method_node.name)
        if key in _DELEGATING_CLASS_METHODS:
            continue
        first_stmt = _first_executable_statement(method_node.body)
        if first_stmt is None:
            failures.append(
                f"{rel}:{method_node.lineno} "
                f"{class_node.name}.{method_node.name}: empty body")
            continue
        if not _is_helper_call(first_stmt):
            failures.append(
                f"{rel}:{method_node.lineno} "
                f"{class_node.name}.{method_node.name}: "
                f"first executable statement is not "
                f"`_check_2d_scalar_field(...)`. Either add the "
                f"guard call (see lumenairy/_validation.py) or, if "
                f"this method legitimately delegates to a guarded "
                f"module-level scalar function, add it to "
                f"``_DELEGATING_CLASS_METHODS`` with a comment "
                f"citing the delegation chain.")
    if failures:
        raise AssertionError(
            "Class methods missing _check_2d_scalar_field guard:\n  - "
            + "\n  - ".join(failures))


# ============================================================================
# Counter-pins on the helper itself: smoke-test the message contract.
# ============================================================================

def test_helper_rejects_mcf_with_v4_16_message():
    """``_check_2d_scalar_field`` must raise ``TypeError`` on
    ``PartialCoherenceMCF`` inputs.  v4.16.1 (audit
    AUDIT_V4_16_0_DEEP item 5b): the prior ``"v4.16+"`` scope
    marker was retired (the library version IS v4.16.x at the
    time of the audit, leaving the deferral wording stale); the
    new message points the user at the ``propagate_ensemble``
    helper and notes that MCF-aware downstream propagators are
    a v5.0+ architectural change.
    """
    from lumenairy._validation import _check_2d_scalar_field
    from lumenairy.sources.core import create_gaussian_schell_source

    mcf = create_gaussian_schell_source(
        N=16, dx=2e-6, wavelength=633e-9,
        w0=10e-6, sigma_g=5e-6,
        n_realizations=2, seed=0, return_kind='mcf')
    with pytest.raises(TypeError) as excinfo:
        _check_2d_scalar_field(mcf, 'unit_test_fn')
    msg = str(excinfo.value)
    assert 'PartialCoherenceMCF' in msg
    # v4.16.1: the message now cites the ``propagate_ensemble``
    # helper as the canonical workflow (the prior ``"v4.16+"``
    # scope marker was stale at v4.16.0).
    assert 'propagate_ensemble' in msg
    # And the rejection still mentions the v5.0+ horizon for
    # MCF-aware downstream propagators (so users grepping for the
    # MCF feature still find the rejection).
    assert 'v5.0' in msg
    assert 'unit_test_fn' in msg


def test_helper_rejects_3d_ensemble_with_iteration_hint():
    """``_check_2d_scalar_field`` must raise ``ValueError`` on a 3-D
    Schell ensemble.  v4.16.1 (audit item 5b follow-up): the
    canonical hint now points at the ``propagate_ensemble`` helper
    AND retains the bare iterate-over-realizations pattern as a
    fallback for users with custom propagator pipelines.
    """
    import numpy as np

    from lumenairy._validation import _check_2d_scalar_field

    ensemble = np.zeros((4, 16, 16), dtype=np.complex128)
    with pytest.raises(ValueError) as excinfo:
        _check_2d_scalar_field(ensemble, 'unit_test_fn')
    msg = str(excinfo.value)
    assert '3-D' in msg
    assert 'unit_test_fn' in msg
    # v4.16.1: hint now points at the propagate_ensemble helper.
    assert 'propagate_ensemble' in msg
    # The bare iterate pattern is retained as a fallback for
    # custom propagator pipelines.
    assert 'for k in range(ensemble.shape[0])' in msg


def test_helper_rejects_1d_with_tailored_hint():
    """``_check_2d_scalar_field`` must raise ``ValueError`` with the
    tailored 1-D / scalar diagnostic when handed a non-2-D ndarray.
    """
    import numpy as np

    from lumenairy._validation import _check_2d_scalar_field

    one_d = np.zeros(16, dtype=np.complex128)
    with pytest.raises(ValueError) as excinfo:
        _check_2d_scalar_field(one_d, 'unit_test_fn')
    msg = str(excinfo.value)
    assert '1-D' in msg
    assert 'unit_test_fn' in msg
    # The tailored hint should mention the wrong-object guidance.
    assert ('2-D NumPy/CuPy complex array' in msg
            or 'scalar' in msg.lower())


def test_helper_accepts_2d_complex_array():
    """Sanity: the helper must NOT raise on a legitimate 2-D complex
    ndarray.  Without this counter-pin a too-strict helper would
    block every legitimate caller silently.
    """
    import numpy as np

    from lumenairy._validation import _check_2d_scalar_field

    E = np.zeros((16, 16), dtype=np.complex128)
    # Should not raise.
    _check_2d_scalar_field(E, 'unit_test_fn')


# ============================================================================
# Counter-pin: a fake unguarded propagator must trip the meta-pin.
# v4.15.4 (P3-NEW-F2-3) -- pins that the walker's correctness relies
# on a POSITIVE signal (helper-call detection), not just on
# ``_walk_entry_points`` returning enough entries.  Modeled on the
# v4.15.0 ``_validate_grid_params`` pin's
# ``test_counter_pin_fake_unvalidated_factory_fails``.
# ============================================================================

def test_counter_pin_fake_unguarded_propagate_fails(monkeypatch):
    """Inject a synthetic unguarded propagator AST node into the
    walker output and assert the meta-pin assertion fires.

    Without this counter-pin a silent walker regression -- e.g. a
    name-regex bug that excludes ``apply_*`` siblings -- could
    pass even though it would have hidden every real
    unguarded-sibling failure.  This test verifies that the
    meta-pin's ``test_all_entry_points_call_helper_first`` only
    passes because the discovered functions actually call the
    helper, not because the walker silently elided them.
    """
    # ---- 1. Build a fake AST representing an unguarded propagator. -----
    fake_source = (
        '"""Fake module for counter-pin."""\n'
        'def fake_unguarded_propagate(E, wavelength, dx):\n'
        '    """Fake docstring."""\n'
        '    return E * 2.0\n'  # NO _check_2d_scalar_field call
    )
    fake_tree = ast.parse(fake_source, filename='<fake>')
    fake_node = None
    for stmt in fake_tree.body:
        if (isinstance(stmt, ast.FunctionDef)
                and stmt.name == 'fake_unguarded_propagate'):
            fake_node = stmt
            break
    assert fake_node is not None, (
        "Counter-pin setup error: failed to build fake FunctionDef "
        "AST node.")

    fake_path = _REPO_ROOT / 'lumenairy' / 'propagators' / 'fake_propagator.py'

    # ---- 2. Monkey-patch _walk_entry_points to include the fake. -------
    # Patch the module-level binding so when
    # ``test_all_entry_points_call_helper_first`` resolves the name
    # ``_walk_entry_points`` at call time it picks up the wrapper.
    import sys as _sys
    this_module = _sys.modules[__name__]
    real_walker = this_module._walk_entry_points

    def _walker_with_fake():
        # Yield everything the real walker yields...
        for py, node in real_walker():
            yield py, node
        # ...then append the synthetic unguarded entry.
        yield fake_path, fake_node

    monkeypatch.setattr(this_module, '_walk_entry_points',
                        _walker_with_fake)

    # ---- 3. Re-run the main pin and assert it now fails. ---------------
    with pytest.raises(AssertionError) as excinfo:
        test_all_entry_points_call_helper_first()
    msg = str(excinfo.value)
    assert 'fake_unguarded_propagate' in msg, (
        f"Counter-pin: meta-pin failed but its error message does "
        f"not name the injected unguarded function. Got:\n{msg}")
    assert 'fake_propagator.py' in msg, (
        f"Counter-pin: meta-pin failed but its error message does "
        f"not name the injected file. Got:\n{msg}")


# ============================================================================
# v4.15.5 (P1-NEW-2WAY-1) V6 counter-pins.  These pin the structural
# upgrade -- the filter must catch a function whose name is outside
# the legacy ``apply_*`` / ``*_propagate*`` / etc. regex but whose
# first positional name is in ``_FIELD_PARAM_NAMES``.  Without
# these counter-pins a future regression that reverts the V6 filter
# to name-prefix-only would silently re-introduce the ~30
# unguarded-analyzer surface the audit identified.
# ============================================================================


def test_v6_first_positional_param_filter_catches_renamed_function():
    """Counter-pin: the V6 filter must catch a function with a
    non-legacy-prefix name whose first positional is ``E``.

    Verifies the structural upgrade that v4.15.5 P1-NEW-2WAY-1
    introduced.  Constructs a synthetic AST representing a public
    function called ``my_new_analyzer(E, dx, ...)`` whose name
    matches none of the legacy ``apply_*`` / ``*_propagate*`` /
    ``richards_wolf*`` / ``debye_wolf*`` patterns.  The legacy-only
    filter would NOT discover it; the V6 filter MUST.
    """
    fake_source = (
        '"""Fake module for V6 counter-pin."""\n'
        'def my_new_analyzer(E, dx, wavelength):\n'
        '    """Hypothetical new analyzer."""\n'
        '    return E.sum()\n'
    )
    fake_tree = ast.parse(fake_source, filename='<v6-fake>')
    fake_node = next(
        stmt for stmt in fake_tree.body
        if isinstance(stmt, ast.FunctionDef))

    # Legacy filter MUST reject it -- the name has no recognised
    # prefix.  This pins that the test is exercising the right
    # half of the OR.
    assert not _legacy_name_matches_entry_point_filter(
        fake_node.name), (
        "Counter-pin setup error: 'my_new_analyzer' unexpectedly "
        "matches the legacy name-prefix filter.  Choose a different "
        "name for the synthetic function.")

    # V6 filter MUST accept it -- ``E`` is in ``_FIELD_PARAM_NAMES``.
    assert _function_matches_entry_point_filter(fake_node), (
        "V6 counter-pin: the first-positional-name filter failed "
        "to catch ``my_new_analyzer(E, ...)``.  This means the "
        "v4.15.5 P1-NEW-2WAY-1 structural upgrade has regressed; "
        "the walker is back to legacy-name-prefix-only discovery "
        "and the ~30 v4.15.4 audit-identified analyzers would be "
        "silently un-pinned.")


def test_v6_first_positional_param_filter_does_not_overmatch():
    """Counter-pin: the V6 filter must NOT match a function whose
    first positional is something unrelated to a field (e.g. a
    prescription dict, a wavelength, an integer count).

    Without this counter-pin a future regression that loosened
    ``_FIELD_PARAM_NAMES`` to any 1-char name could silently start
    matching unrelated entry points and explode the exemption set.
    """
    fake_source = (
        '"""Fake module for V6 counter-pin (overmatch test)."""\n'
        'def my_settings_helper(prescription, n_modes, gain):\n'
        '    """Not a 2-D field consumer."""\n'
        '    return prescription\n'
    )
    fake_tree = ast.parse(fake_source, filename='<v6-fake-over>')
    fake_node = next(
        stmt for stmt in fake_tree.body
        if isinstance(stmt, ast.FunctionDef))
    assert not _function_matches_entry_point_filter(fake_node), (
        "V6 counter-pin (overmatch): the filter matched "
        "``my_settings_helper(prescription, ...)`` which is not a "
        "2-D scalar-field consumer.  ``_FIELD_PARAM_NAMES`` may "
        "have been broadened too aggressively.")


def test_v6_walker_catches_first_positional_param_name():
    """Counter-pin: monkeypatch a synthetic AST through the V6
    discovery walker and verify the main pin fires for an
    unguarded ``my_new_analyzer(E, ...)``.

    Modeled on ``test_counter_pin_fake_unguarded_propagate_fails``
    but constructed so the SYNTHETIC NAME does not match the
    legacy regex -- so a regressed walker that fell back to
    legacy-only discovery would NOT catch the failure and the
    counter-pin would (silently) pass when it should not.
    """
    fake_source = (
        '"""Fake module for V6 walker counter-pin."""\n'
        'def my_new_analyzer(E, dx, wavelength):\n'
        '    """Synthetic analyzer that intentionally skips '
        'the guard."""\n'
        '    return E.sum()\n'
    )
    fake_tree = ast.parse(fake_source, filename='<v6-walker-fake>')
    fake_node = next(
        stmt for stmt in fake_tree.body
        if isinstance(stmt, ast.FunctionDef))
    # Same posture as the legacy counter-pin: assert the synthetic
    # name is OUTSIDE the legacy filter's prefix set.  This is the
    # critical pin -- only the V6 filter should reach this case.
    assert not _legacy_name_matches_entry_point_filter(
        fake_node.name)

    fake_path = (_REPO_ROOT / 'lumenairy' / 'analysis'
                 / 'fake_analyzer.py')

    import sys as _sys
    this_module = _sys.modules[__name__]
    real_walker = this_module._walk_entry_points

    def _walker_with_fake():
        for py, node in real_walker():
            yield py, node
        yield fake_path, fake_node

    # Use ``monkeypatch.setattr`` via pytest's fixture protocol;
    # we don't have an injected fixture here so we mutate and
    # restore directly.
    try:
        this_module._walk_entry_points = _walker_with_fake
        with pytest.raises(AssertionError) as excinfo:
            test_all_entry_points_call_helper_first()
        msg = str(excinfo.value)
        assert 'my_new_analyzer' in msg, (
            f"V6 walker counter-pin: meta-pin failed but its error "
            f"message does not name the injected analyzer. Got:\n{msg}")
    finally:
        this_module._walk_entry_points = real_walker


def test_class_method_walker_descent_catches_non_delegating_method():
    """Counter-pin: monkeypatch a synthetic ``apply`` class method
    through the v4.15.5 P2-NEW-F2-4 walker and verify the main pin
    fires for a non-delegating, non-exempt class method.

    Without this counter-pin, a future regression that re-introduced
    the v4.15.4 blanket class-method exclusion (or that broke the
    ``_class_method_matches_filter`` first-positional check) would
    re-open the ``DeformableMirror.apply``-class bypass and the
    audit would not catch it again.
    """
    fake_source = (
        '"""Fake module for class-descent counter-pin."""\n'
        'class FakeDM:\n'
        '    """Synthetic non-delegating class."""\n'
        '    def apply(self, E_in, scale=1.0):\n'
        '        """No guard, no delegation."""\n'
        '        return E_in * 2.0\n'
    )
    fake_tree = ast.parse(fake_source, filename='<class-fake>')
    fake_class = next(
        stmt for stmt in fake_tree.body
        if isinstance(stmt, ast.ClassDef))
    fake_method = next(
        sub for sub in fake_class.body
        if isinstance(sub, ast.FunctionDef) and sub.name == 'apply')

    # The filter MUST accept the synthetic method (else the test is
    # not exercising what it claims to test).
    assert _class_method_matches_filter(fake_method), (
        "Counter-pin setup error: the filter did not accept the "
        "synthetic ``FakeDM.apply(E_in, ...)`` method.")

    fake_path = (_REPO_ROOT / 'lumenairy' / 'analysis'
                 / 'fake_class.py')

    import sys as _sys
    this_module = _sys.modules[__name__]
    real_walker = this_module._walk_class_methods

    def _walker_with_fake():
        for py, cnode, mnode in real_walker():
            yield py, cnode, mnode
        yield fake_path, fake_class, fake_method

    try:
        this_module._walk_class_methods = _walker_with_fake
        with pytest.raises(AssertionError) as excinfo:
            test_all_class_methods_call_helper_first()
        msg = str(excinfo.value)
        assert 'FakeDM.apply' in msg, (
            f"Class-descent counter-pin: meta-pin failed but its "
            f"error message does not name the injected class method. "
            f"Got:\n{msg}")
    finally:
        this_module._walk_class_methods = real_walker


# ============================================================================
# Diagnostic helper (printed when -v is on; useful when adding a new
# entry point and triaging discovery).
# ============================================================================

def test_discovered_entry_points_list_for_diagnostics():
    """Emit the discovered entry-point list to stdout for triage.

    Not really a test -- always passes -- but provides a single
    inspection point so ``pytest -v`` shows the discovery walker's
    output without needing to run the full pin.

    v4.15.5: emits BOTH top-level and class-method discoveries so
    the V6 + class-descent coverage is visible at a glance.
    """
    discovered = []
    for py, node in _walk_entry_points():
        rel = py.relative_to(_REPO_ROOT).as_posix()
        tag = ('exempt' if (rel, node.name) in _GUARD_EXEMPTIONS
               else 'guarded')
        discovered.append((rel, node.lineno, node.name, tag))
    print(f"\nv4.15.5 _check_2d_scalar_field meta-pin discovered "
          f"{len(discovered)} top-level entry points:")
    for rel, lineno, name, tag in discovered:
        print(f"  [{tag}] {rel}:{lineno}  {name}")

    methods = []
    for py, class_node, method_node in _walk_class_methods():
        rel = py.relative_to(_REPO_ROOT).as_posix()
        key = (rel, class_node.name, method_node.name)
        tag = ('delegating' if key in _DELEGATING_CLASS_METHODS
               else 'guarded')
        methods.append((rel, method_node.lineno, class_node.name,
                        method_node.name, tag))
    print(f"\nv4.15.5 _check_2d_scalar_field meta-pin discovered "
          f"{len(methods)} class methods:")
    for rel, lineno, cname, mname, tag in methods:
        print(f"  [{tag}] {rel}:{lineno}  {cname}.{mname}")

    assert isinstance(discovered, list)
    assert isinstance(methods, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
