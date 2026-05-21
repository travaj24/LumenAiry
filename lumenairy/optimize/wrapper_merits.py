"""
lumenairy.optimize.wrapper_merits -- merit-term wrappers + meshgrid cache.

v5.1.0 split (Agent E): extracted from ``lumenairy/optimize/core.py``.
Hosts the three "wrapper" merit terms that sweep an inner sub-merit
across multiple wavelengths / field angles / tolerance perturbations
(:class:`MultiWavelengthMerit`, :class:`MultiFieldMerit`,
:class:`ToleranceAwareMerit`) plus the shared module-level meshgrid /
aperture-mask cache (:data:`_WRAPPER_MERIT_CACHE`).

Cache layout
------------
``_WRAPPER_MERIT_CACHE`` is a thread-safe LRU keyed on
``(Ny, Nx, dx, aperture_key, dtype_str)``.  The payload includes the
coordinate meshgrid, the aperture boolean mask, and the wavelength-
independent ``2*pi * Y`` / ``2*pi * X`` factors used for per-field
tilt-phase construction.  See ``_get_wrapper_merit_cache`` for the
detailed contract.

Lookup contract for ``system_abcd`` / ``through_focus_scan`` etc.
-----------------------------------------------------------------
Pre-v5.1.0, every name used in the wrapper-merit bodies lived in
``lumenairy/optimize/core.py`` and was imported at the top of that
file via ``from ..raytrace import system_abcd`` etc.  Tests that
monkey-patch ``lumenairy.optimize.core.system_abcd`` (using
:func:`unittest.mock.patch`) target THAT binding -- not the original
``lumenairy.raytrace.system_abcd``.  After the v5.1.0 split, the
wrapper-merit class bodies still need to honour those patches, so
each call site reads the function via :mod:`lumenairy.optimize.core`
(``_core.system_abcd(...)``) via a lazy module-attribute lookup.  This
preserves the v4.15.3 mock.patch test contract bit-for-bit.

The lazy lookup happens inside method bodies, not at module-import
time, so the circular dependency between ``core.py`` (which re-exports
this module's classes) and ``wrapper_merits.py`` (which calls back
into ``core``) is harmless -- by the time any ``.evaluate(...)`` runs,
``core.py`` has fully populated its namespace.
"""

from __future__ import annotations

import threading
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .context import (
    EvaluationContext,
    MeritTerm,
    _FAILED_SCAN_STREHL_SENTINEL_OBJ,
    _INVALID_FL_SENTINEL_OBJ,
    _ZERO_APERTURE_MASK,
)


# =========================================================================
# Wrapper-merit meshgrid cache (v4.14.0 perf)
# =========================================================================
#
# MultiWavelengthMerit, MultiFieldMerit, and ToleranceAwareMerit each
# rebuild np.indices / meshgrid arrays + aperture mask on every per-
# wavelength / per-field / per-trial leg.  For a 5 wavelengths x 5
# fields x 40 FD evals run at N=512 that is up to 1000 N x N meshgrid
# builds per optimisation step, none of which depend on the parameter
# vector being differenced.
#
# This module-level LRU cache memoises the wavelength/field/trial-
# invariant payload keyed on (Ny, Nx, dx, aperture_hash, dtype_str).
# Per-leg cost reduces to a single np.exp(1j * sin_a * cached_k0_Y) *
# cached_aperture_mask (MultiFieldMerit) or a single .copy() (the
# other two) plus the apply_real_lens call.
#
# Eval-count pin: the counter _WRAPPER_MERIT_MESHGRID_BUILDS records
# every actual meshgrid build (NOT cache hits) so tests can assert
# exactly one build per (N, dx, aperture) signature per optimisation
# run.

_WRAPPER_MERIT_CACHE: 'OrderedDict[tuple, dict]' = OrderedDict()
_WRAPPER_MERIT_CACHE_SIZE = 32
_WRAPPER_MERIT_MESHGRID_BUILDS = 0
# v4.14.1 (P2-1): guard concurrent get / move_to_end / __setitem__ /
# popitem(last=False) on _WRAPPER_MERIT_CACHE.  Follows the
# _ASM_CACHE_LOCK precedent in propagators/propagation.py.  Without
# this two threads racing through _get_wrapper_merit_cache could see a
# torn OrderedDict.
_WRAPPER_MERIT_CACHE_LOCK = threading.Lock()


def _wrapper_merit_aperture_key(aperture: Any) -> tuple:
    """Build a hashable key fragment representing the aperture state.

    Three branches:
      - ``None``  -> ``('none',)``.
      - ndarray   -> ``('arr', shape, dtype, content_hash)``.
        ``hash(np.ascontiguousarray(a).tobytes())`` captures content
        cheaply (a single ~N^2 byte scan; for N=512^2 complex128 that
        is ~4 MB which hashes in <1 ms).
      - scalar    -> ``('scalar', float)`` covering the common case of
        a single aperture_diameter taken from ``prescription``.
    """
    if aperture is None:
        return ('none',)
    if isinstance(aperture, np.ndarray):
        arr = np.ascontiguousarray(aperture)
        return ('arr', arr.shape, str(arr.dtype),
                hash(arr.tobytes()))
    # Scalar aperture: a Python int/float/np.floating.  Forced to a
    # plain float so np.float64(1.0) and 1.0 share the same cache key.
    return ('scalar', float(aperture))


def _get_wrapper_merit_cache(
    N: int, dx: float, aperture: Any, dtype: Any,
) -> Dict[str, Any]:
    """Return the cached (Y, X, mask, k0_Y_factor) payload for these
    grid + aperture parameters.

    The payload is a dict with keys ``'X'``, ``'Y'``, ``'mask'``,
    ``'Y_factor'`` where:

    - ``X``, ``Y``: the meshgrid coordinate arrays (shape (N, N),
      dtype float64).
    - ``mask``: boolean aperture mask (or ``None`` when ``aperture``
      is ``None``-or-zero).
    - ``Y_factor``: the wavelength-independent ``2*pi * Y / 1`` such
      that the per-wavelength tilt phase is
      ``(Y_factor / wavelength) * sin(theta_y)``.  Cached so the
      MultiFieldMerit per-leg work is one np.exp + one multiply.
    - ``r_squared``: ``X*X + Y*Y`` (cached for callers that need to
      build their own custom aperture masks against the same grid).

    The cache is LRU-bounded at 32 entries (``_WRAPPER_MERIT_CACHE_SIZE``).
    A meshgrid build increments ``_WRAPPER_MERIT_MESHGRID_BUILDS``;
    cache hits do NOT increment.  Use this counter in tests to pin
    the invariance contract.
    """
    global _WRAPPER_MERIT_MESHGRID_BUILDS

    Ny = Nx = int(N)
    dx_f = float(dx)
    # dtype may arrive as numpy dtype object, dtype string, or
    # np.complex128 type.  ``str(np.dtype(x))`` normalises to a
    # canonical short name ('complex128', 'complex64', ...).
    try:
        _dtype_obj = np.dtype(dtype)
        dtype_str = str(_dtype_obj)
    except TypeError:
        _dtype_obj = np.complex128
        dtype_str = str(dtype)
    ap_key = _wrapper_merit_aperture_key(aperture)
    key = (Ny, Nx, dx_f, ap_key, dtype_str)

    with _WRAPPER_MERIT_CACHE_LOCK:
        entry = _WRAPPER_MERIT_CACHE.get(key)
        if entry is not None:
            # LRU bookkeeping: refresh recency.
            _WRAPPER_MERIT_CACHE.move_to_end(key)
            return entry

    # Cache miss: build the grid + mask + Y-factor once and store.
    # The build itself is pure-CPU numpy and re-entrant; only the
    # OrderedDict get/move_to_end/set/popitem operations need the lock.
    _WRAPPER_MERIT_MESHGRID_BUILDS += 1
    Y_idx, X_idx = np.indices((Ny, Nx))
    X = (X_idx - Nx / 2) * dx_f
    Y = (Y_idx - Ny / 2) * dx_f
    r_squared = X * X + Y * Y

    if isinstance(aperture, np.ndarray):
        # Custom user-supplied aperture array; assume boolean-coercible.
        mask = np.asarray(aperture, dtype=bool)
        if mask.shape != (Ny, Nx):
            raise ValueError(
                f"_get_wrapper_merit_cache: aperture array shape "
                f"{mask.shape} != grid shape ({Ny}, {Nx})")
    elif aperture is None:
        # "No aperture specified" -- callers treat mask=None as
        # "full grid, no clipping."
        mask = None
    else:
        # Scalar aperture_diameter.
        ap_diam = float(aperture)
        if ap_diam > 0:
            mask = r_squared <= (ap_diam / 2.0) ** 2
        else:
            # v4.14.1 (P1-NEW-1): aperture explicitly <= 0 means
            # "block all light."  Distinct from the ``None`` branch
            # above (which means "no aperture specified, use full
            # grid").  Pre-v4.14.0 a scalar 0 produced an all-False
            # boolean mask; v4.14.0 erroneously collapsed it to None
            # and the downstream callers then treated the deliberate
            # zero as "no aperture -> full plane wave," flipping the
            # semantics.  Use a sentinel so callers can detect this
            # case via ``is`` and zero their fields explicitly.
            mask = _ZERO_APERTURE_MASK

    # Wavelength-independent Y-tilt factor: 2*pi * Y.  Per-leg the
    # tilt phase is (Y_factor / wavelength) * sin(theta_y) plus the
    # analogous X term.  Materialised so the per-leg cost is a
    # single multiply.
    Y_factor = (2.0 * np.pi) * Y
    X_factor = (2.0 * np.pi) * X

    # Cached np.ones array for ToleranceAwareMerit's per-trial
    # source field.  Stored once per (N, dtype); per-trial just
    # .copy() this and feed apply_real_lens.  apply_real_lens
    # never writes its input, but downstream merit code paths may
    # so the .copy() at call site preserves correctness.  Uses the
    # ``_dtype_obj`` computed at the head of the function.
    E_ones = np.ones((Ny, Nx), dtype=_dtype_obj)

    entry = {
        'X': X,
        'Y': Y,
        'mask': mask,
        'Y_factor': Y_factor,
        'X_factor': X_factor,
        'r_squared': r_squared,
        'E_ones': E_ones,
    }
    with _WRAPPER_MERIT_CACHE_LOCK:
        _WRAPPER_MERIT_CACHE[key] = entry
        while len(_WRAPPER_MERIT_CACHE) > _WRAPPER_MERIT_CACHE_SIZE:
            _WRAPPER_MERIT_CACHE.popitem(last=False)
    return entry


def _clear_wrapper_merit_cache() -> None:
    """Drop the wrapper-merit meshgrid cache and reset the build counter.

    v4.14.1 (P2-3): invoked from
    :func:`lumenairy.propagators.propagation.clear_asm_caches`.  Pre-
    v4.16 this was a lazy import inside ``clear_asm_caches``; v4.16
    routes the call through the central cache-clearer registry (see
    ``_cache_registry.py``).  Either way the reverse-direction
    dependency keeps optimize/core free of propagation-layer side-
    effects at import time while still leaving both caches pristine
    on a single ``clear_asm_caches()`` call.  Also callable directly
    from tests.
    """
    global _WRAPPER_MERIT_MESHGRID_BUILDS
    with _WRAPPER_MERIT_CACHE_LOCK:
        _WRAPPER_MERIT_CACHE.clear()
    _WRAPPER_MERIT_MESHGRID_BUILDS = 0


# v4.16.0 (ROADMAP #15): register the wrapper-merit clearer with the
# central registry at module-import time.  ``clear_asm_caches`` now
# walks the registry rather than enumerating clear calls by hand.
# Late-binding closure preserves ``mock.patch.object`` test semantic.
try:
    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    import sys as _sys
    _this_mod = _sys.modules[__name__]
    _register_cache_clearer(
        'wrapper_merit_meshgrid',
        lambda: getattr(_this_mod, '_clear_wrapper_merit_cache')(),
    )
except ImportError:
    pass


# =========================================================================
# Multi-wavelength support
# =========================================================================

# v4.16.2 (audit P1-NEW-F1-3): one-cycle FutureWarning latch for the
# MultiWavelengthMerit SUM->AVG transition introduced in v4.16.1.  The
# new AVG semantics are CORRECT (match the docstring and the sibling
# MultiFieldMerit / ToleranceAwareMerit classes which already divide by
# their loop length), but existing user weight-calibrations tuned
# against the pre-v4.16.1 SUM behaviour silently see a 3x drop on a
# 3-wavelength configuration.  Emit a one-shot FutureWarning the first
# time any MultiWavelengthMerit.evaluate runs with >1 wavelength so
# users notice the change and can re-scale weights if needed.  Latched
# at module level so optimisation loops (which call evaluate() many
# times per process) don't flood the warning channel.
_MULTIWL_AVG_WARNED = False


class MultiWavelengthMerit(MeritTerm):
    """Evaluate a sub-merit at multiple wavelengths and average.

    Populates ``ctx.efls_per_wavelength`` with per-wavelength EFLs
    (computed geometrically, cheap).  The sub-merit is evaluated at
    each wavelength and the results are averaged (weight * total /
    n_wavelengths).  v4.16.1 closes the SUM-vs-AVG discrepancy at
    the return: pre-v4.16.1 this class summed rather than averaged,
    silently tripling the chromatic merit contribution for a
    3-wavelength configuration vs a 1-wavelength one.  Matches the
    sibling :class:`MultiFieldMerit` and :class:`ToleranceAwareMerit`
    averaging shape.

    .. warning::
        The off-wavelength wave-leg propagation in this merit's
        ``evaluate`` always calls :func:`apply_real_lens` directly,
        irrespective of the ``wave_propagator`` selected on the
        enclosing :func:`design_optimize` call.  For high-NA designs
        optimised with ``wave_propagator='gbd'`` (or any non-real-lens
        backend) the off-nominal-wavelength penalty therefore exercises
        a different physical model than the on-axis wave leg.  A
        runtime warning fires from :func:`design_optimize` when this
        mismatch is detected.  Threading the propagator through the
        sub-merit is a v4.14+ feature -- see audit P2 #14.

    Parameters
    ----------
    wavelengths : sequence of float
        Wavelengths [m] to evaluate at.
    sub_merit : MeritTerm
        Merit term to evaluate at each wavelength.  Its ``evaluate``
        receives a modified ``ctx`` with the corresponding wavelength.
    weight : float
    """

    name = 'MultiWavelength'

    def __init__(self, wavelengths: Sequence[float],
                 sub_merit: MeritTerm, weight: float = 1.0) -> None:
        self.wavelengths = [float(w) for w in wavelengths]
        self.sub_merit = sub_merit
        self.weight = float(weight)
        self.needs_wave = sub_merit.needs_wave

    def evaluate(self, ctx: Any) -> float:
        # 4.10: re-evaluate the wave leg at each wavelength.  Pre-4.10
        # only EFL/BFL changed per-wavelength while E_exit, opd_map,
        # strehl_best, rms_radius_best were copied unchanged from ctx,
        # so wrapping StrehlMerit / RMSWavefrontMerit / MatchTargetOPDMerit
        # in MultiWavelengthMerit just averaged the same single-wavelength
        # number N times -- the chromatic-aberration penalty was a
        # no-op.  Now: for each wavelength, propagate the same input
        # field through apply_real_lens at that wavelength, run a
        # quick through-focus scan for Strehl, build an OPD map, and
        # populate the sub-context with these per-wavelength wave
        # quantities before delegating to the sub-merit.
        #
        # v5.1.0 split: each call into ``surfaces_from_prescription`` /
        # ``system_abcd`` / ``through_focus_scan`` / ``find_best_focus``
        # / ``diffraction_limited_peak`` / ``wave_opd_2d`` /
        # ``apply_real_lens`` / ``get_default_complex_dtype`` looks up
        # the name on :mod:`lumenairy.optimize.core` so that historical
        # ``mock.patch('lumenairy.optimize.core.X', ...)`` tests still
        # patch the binding actually used by this class body.
        from . import core as _core
        efls = []
        per_wl_strehl = []
        per_wl_rms = []
        total = 0.0
        for wl in self.wavelengths:
            surfs = _core.surfaces_from_prescription(ctx.prescription)
            try:
                _, efl, bfl, _ = _core.system_abcd(surfs, wl)
            except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                    np.linalg.LinAlgError, IndexError, TypeError):
                # Degenerate ABCD at this wavelength; sentinel-large
                # EFL/BFL nudges the wave leg toward the fallback
                # branch downstream.  v4.15.3 (P1-NEW-F1-3): wire the
                # ``_INVALID_FL_SENTINEL_OBJ`` singleton so a
                # downstream consumer in this scope can perform an
                # ``is``-identity check; ``float()`` of the singleton
                # still returns ``1e9`` for the existing magnitude-
                # based ``ctx_is_valid`` path.  The sentinel does NOT
                # escape into ``sub_ctx`` (the ``float(efl)``/
                # ``float(bfl)`` cast at sub-context construction
                # restores the scalar for downstream merits that
                # weren't migrated).
                efl = bfl = _INVALID_FL_SENTINEL_OBJ
            efls.append(float(efl))
            sub_E_exit = ctx.E_exit
            sub_opd = ctx.opd_map
            sub_strehl = ctx.strehl_best
            sub_rms = ctx.rms_radius_best
            sub_z = ctx.z_best
            # v4.15.3 (P1-NEW-F1-3): the sentinel form of ``bfl``
            # is not a numpy-friendly scalar -- guard the
            # ``np.isfinite``/``abs`` checks with an identity test
            # so the per-wavelength wave leg is skipped when the
            # ABCD extraction collapsed at this wavelength.
            if self.sub_merit.needs_wave and ctx.E_exit is not None and \
               bfl is not _INVALID_FL_SENTINEL_OBJ and \
               np.isfinite(bfl) and abs(bfl) < 10:
                try:
                    # Build a plane-wave input on ctx's grid and push
                    # through the prescription at this wavelength.
                    # 4.11.1: keyword call (apply_real_lens is keyword-only
                    # after E_in since 4.7); honour precision knob via
                    # get_default_complex_dtype; explicit is-None check on
                    # aperture_diameter so a deliberate 0 isn't shadowed
                    # by the grid-arbitrary fallback.
                    # v4.14.0: reuse the cached aperture mask /
                    # coordinate grids (see _get_wrapper_merit_cache);
                    # rebuilding np.indices on every per-wavelength
                    # leg was the dominant cost for 5+ wavelengths *
                    # 2N+1 FD evals per outer iteration.
                    N_pix = ctx.N
                    dx_pix = ctx.dx
                    ap = ctx.prescription.get('aperture_diameter')
                    if ap is None:
                        ap = 0.4 * N_pix * dx_pix
                    _cdtype = _core.get_default_complex_dtype()
                    _cache = _get_wrapper_merit_cache(
                        N_pix, dx_pix, ap, _cdtype)
                    mask = _cache['mask']
                    # Three branches:
                    #   mask is None             -> "no aperture
                    #     specified" (ap was None above, but the
                    #     0.4*N*dx fallback rules that out here).
                    #     Treat as full grid for completeness.
                    #   mask is _ZERO_APERTURE_MASK -> deliberate
                    #     aperture_diameter <= 0; block all light
                    #     (v4.14.1 P1-NEW-1 fix; v4.14.0 erroneously
                    #     mapped this to "full grid plane wave").
                    #   mask is an ndarray       -> circular boolean
                    #     mask for the requested aperture diameter.
                    if mask is _ZERO_APERTURE_MASK:
                        E_in_wl = np.zeros((N_pix, N_pix), dtype=_cdtype)
                    elif mask is None:
                        E_in_wl = np.ones((N_pix, N_pix), dtype=_cdtype)
                    else:
                        E_in_wl = mask.astype(_cdtype)
                    E_exit_wl = _core.apply_real_lens(
                        E_in_wl,
                        prescription=ctx.prescription,
                        wavelength=wl,
                        dx=dx_pix)
                    sub_E_exit = E_exit_wl
                    half = max(abs(bfl) / 20.0, 1e-3)
                    z_vals = np.linspace(bfl - half, bfl + half, 7)
                    ideal_pk = _core.diffraction_limited_peak(
                        E_exit_wl, wl, bfl, dx_pix)
                    scan = _core.through_focus_scan(
                        E_exit_wl, dx_pix, wl, z_vals,
                        ideal_peak=ideal_pk, verbose=False)
                    z_b, sb = _core.find_best_focus(scan, 'strehl')
                    sub_z = float(z_b)
                    sub_strehl = float(sb)
                    # 4.11.1: nanargmax so a single NaN per-z slice does
                    # not steal the argmax position.
                    if np.any(np.isfinite(scan.strehl)):
                        i_b = int(np.nanargmax(scan.strehl))
                        sub_rms = float(scan.rms_radius[i_b])
                    _, _, sub_opd = _core.wave_opd_2d(
                        E_exit_wl, dx_pix, wl, aperture=ap,
                        focal_length=bfl, f_ref=bfl)
                except (TypeError, ValueError, RuntimeError) as exc:
                    # 4.11.1: was a bare ``except Exception: pass`` which
                    # silently swallowed call-signature mistakes (the
                    # 4.10 positional-call regression hid here for the
                    # entire v4.10 series).  Warn so the failure is
                    # visible without aborting the optimizer.
                    warnings.warn(
                        f"MultiWavelengthMerit: per-wavelength wave-leg "
                        f"propagation failed at wl={wl:.3e} m "
                        f"({type(exc).__name__}: {exc}); falling back "
                        f"to the parent context's wave-leg values.",
                        RuntimeWarning, stacklevel=2)
            per_wl_strehl.append(sub_strehl)
            per_wl_rms.append(sub_rms)
            # v4.13.2 (C-P1-2): thread ctx.x so JaxMeritTerm sub-
            # merits with build_args reach the analytic-gradient
            # path instead of legacy fn(ctx) -> FD.
            sub_ctx = EvaluationContext(
                prescription=ctx.prescription, wavelength=wl,
                N=ctx.N, dx=ctx.dx, efl=float(efl), bfl=float(bfl),
                seidel=ctx.seidel, E_exit=sub_E_exit,
                opd_map=sub_opd, strehl_best=sub_strehl,
                rms_radius_best=sub_rms, z_best=sub_z,
                x=getattr(ctx, 'x', None))
            total = total + self.sub_merit.evaluate(sub_ctx)
        ctx.efls_per_wavelength = np.array(efls)
        ctx.strehls_per_wavelength = np.array(per_wl_strehl)
        ctx.rms_per_wavelength = np.array(per_wl_rms)
        # v4.16.1 (AUDIT_V4_16_0_DEEP P1-DEEP-1-1): SUM -> AVG.
        # The docstring documents this class as "average" of the
        # sub-merit across wavelengths, and BOTH sibling classes
        # ``MultiFieldMerit`` and ``ToleranceAwareMerit`` divide by
        # ``len(...)`` at their return.  Pre-v4.16.1 this method
        # silently summed the per-wavelength contributions, so adding
        # a 3rd wavelength tripled the chromatic merit's weight
        # contribution relative to a 1-wavelength configuration.
        # Fix: divide by ``max(len(self.wavelengths), 1)`` to match
        # the documented behaviour and sibling-class averaging shape.
        #
        # v4.16.2 (audit P1-NEW-F1-3): emit a one-cycle FutureWarning
        # the first time evaluate() runs with >1 wavelength so users
        # tuning weight calibrations against the pre-v4.16.1 SUM
        # behaviour notice the silent 1/N drop in the merit
        # contribution.  Latched via the module-level
        # ``_MULTIWL_AVG_WARNED`` flag so optimisation loops don't
        # flood the warning channel.  Single-wavelength configurations
        # (len == 1) are unaffected by the SUM->AVG transition and
        # don't trigger the warning.
        # v5.1.0 split (Agent E): the canonical latch lives on this
        # module, but historical reset-fixtures toggle
        # ``lumenairy.optimize.core._MULTIWL_AVG_WARNED`` at the
        # re-exported alias.  Honour both bindings -- read the alias
        # first (so a fixture reset on ``core`` re-fires the warning)
        # and write back to both on emission.
        global _MULTIWL_AVG_WARNED
        import sys as _sys
        _core_mod = _sys.modules.get('lumenairy.optimize.core')
        if _core_mod is not None:
            _latched = _core_mod.__dict__.get(
                '_MULTIWL_AVG_WARNED', None)
        else:
            _latched = None
        if _latched is None:
            _latched = _MULTIWL_AVG_WARNED
        if not _latched and len(self.wavelengths) > 1:
            _MULTIWL_AVG_WARNED = True
            if _core_mod is not None:
                _core_mod.__dict__['_MULTIWL_AVG_WARNED'] = True
            warnings.warn(
                "MultiWavelengthMerit changed from SUM to AVG semantics "
                "in v4.16.1: the per-wavelength sub-merit contributions "
                "are now divided by len(wavelengths) (matches the "
                "docstring and the sibling MultiFieldMerit / "
                "ToleranceAwareMerit classes).  This is the CORRECT new "
                "behaviour; weight calibrations tuned against the "
                "pre-v4.16.1 SUM behaviour may need to be re-scaled by "
                "len(wavelengths).  Silence this notice via "
                "``warnings.filterwarnings('ignore', "
                "category=FutureWarning, "
                "module='lumenairy.optimize.core')``.",
                FutureWarning, stacklevel=2,
            )
        return self.weight * total / max(len(self.wavelengths), 1)


# =========================================================================
# Multi-field support (off-axis)
# =========================================================================

class MultiFieldMerit(MeritTerm):
    """Evaluate a sub-merit at multiple field angles.

    At each field angle a tilted plane wave is built, propagated
    through the lens, and the sub-merit is evaluated on the
    resulting wave field.

    .. warning::
        The off-field wave-leg propagation always calls
        :func:`apply_real_lens` directly, irrespective of the
        ``wave_propagator`` setting on the enclosing
        :func:`design_optimize` call.  See audit P2 #14 for context;
        a runtime warning fires from :func:`design_optimize` when the
        propagator mismatch is detected.

    Parameters
    ----------
    field_angles : sequence of float OR sequence of (theta_x, theta_y)
        Field angles in radians (half-angle from optical axis).
        ``0`` = on-axis.  A scalar entry is interpreted as a pure
        Y-axis tilt (preserved for back-compatibility); a
        ``(theta_x, theta_y)`` tuple is interpreted as an off-axis
        plane wave with independent X and Y tilts.  The scalar form
        emits a one-shot :class:`DeprecationWarning`.
    sub_merit : MeritTerm
        Wave-based merit term to evaluate at each field.
    weight : float
    """

    name = 'MultiField'
    # Class-level flag so the deprecation warning fires exactly once
    # per process (not once per instance, not once per evaluate()).
    _scalar_warning_issued = False

    def __init__(self, field_angles: Sequence[Any],
                 sub_merit: MeritTerm, weight: float = 1.0) -> None:
        # v4.13.2 (C-P0-2): accept EITHER scalars (back-compat:
        # Y-axis tilt) OR (theta_x, theta_y) tuples.  Detect the
        # form per-entry so a mixed list still works.
        normalised: List[Tuple[float, float]] = []
        had_scalar = False
        for a in field_angles:
            if isinstance(a, (tuple, list)) and len(a) == 2:
                tx, ty = a
                normalised.append((float(tx), float(ty)))
            else:
                had_scalar = True
                normalised.append((0.0, float(a)))
        if had_scalar and not MultiFieldMerit._scalar_warning_issued:
            warnings.warn(
                "MultiFieldMerit: scalar ``field_angles`` entries are "
                "interpreted as Y-axis tilt only.  Pass "
                "(theta_x, theta_y) tuples to control both axes; the "
                "scalar form will keep working but is deprecated.",
                DeprecationWarning, stacklevel=2)
            MultiFieldMerit._scalar_warning_issued = True
        self.field_angles = normalised
        self.sub_merit = sub_merit
        self.weight = float(weight)
        self.needs_wave = True

    def evaluate(self, ctx: Any) -> float:
        from . import core as _core
        total = 0.0
        # v4.14.0: aperture mask + coordinate grids + the
        # wavelength-independent k0*Y / k0*X factors are invariant
        # across field angles and FD-eval perturbations.  Cache them
        # module-level keyed on (N, dx, aperture, dtype).  Per-leg
        # cost reduces to a single np.exp + np.where over the cached
        # mask + tilt phase; meshgrid_build_count drops from
        # n_fields * 2N_FD to 1 per optimisation run.
        Ny, Nx = ctx.N, ctx.N
        ap_diam = ctx.prescription.get('aperture_diameter')
        if ap_diam is None:
            ap_diam = 0.4 * Nx * ctx.dx
        _cdtype = _core.get_default_complex_dtype()
        _cache = _get_wrapper_merit_cache(
            ctx.N, ctx.dx, float(ap_diam), _cdtype)
        # Wavelength-independent factors: per-field the tilt phase
        # is sin(theta_x) * (k0_X_factor / wavelength) +
        # sin(theta_y) * (k0_Y_factor / wavelength).  Pre-fold the
        # 1/wavelength into the wavelength-dependent multiplier
        # below.  Note ``ctx.wavelength`` IS invariant across
        # MultiFieldMerit's loop (the field sweep is the loop axis),
        # so we form k_X/k_Y just once.
        _wl = float(ctx.wavelength)
        k_X = _cache['X_factor'] / _wl
        k_Y = _cache['Y_factor'] / _wl
        aperture_mask = _cache['mask']
        for theta_x, theta_y in self.field_angles:
            # Build tilted plane wave clipped to the lens aperture so
            # the propagated intensity reflects the lens's actual
            # acceptance.  Pre-4.10 the unclipped grid-filling plane
            # wave fed every grid pixel through apply_real_lens, then
            # Strehl was computed against a "grid-filling" reference
            # which artificially lowered the value and biased the
            # optimizer toward apertures larger than designed.
            # v4.13.2 (C-P0-2): generic off-axis tilt with both X and
            # Y components.  Pre-fix the X term was silently dropped.
            tilt_phase = np.sin(theta_x) * k_X + np.sin(theta_y) * k_Y
            # 4.11.1: honour precision knob (was hard-coded complex128
            # which silently demoted precision='single' configs).
            # v4.14.1 (P1-NEW-1): three branches -- None means "no
            # aperture specified, full grid"; _ZERO_APERTURE_MASK
            # means "aperture explicitly zero, block all light";
            # ndarray means "circular boolean mask."  Pre-v4.14.0 the
            # zero-diameter case was an all-False ndarray (correctly
            # zeroing the field); v4.14.0 collapsed it into the None
            # branch (full-grid plane wave), flipping the semantics.
            if aperture_mask is _ZERO_APERTURE_MASK:
                E_tilted = np.zeros((Ny, Nx), dtype=_cdtype)
            elif aperture_mask is None:
                E_tilted = np.exp(1j * tilt_phase).astype(_cdtype)
            else:
                E_tilted = np.where(aperture_mask, np.exp(1j * tilt_phase),
                                     0.0).astype(_cdtype)
            E_exit = _core.apply_real_lens(
                E_tilted, prescription=ctx.prescription, wavelength=ctx.wavelength, dx=ctx.dx)
            # Build sub-context.  v4.13.2 (C-P1-2): thread ctx.x so
            # JaxMeritTerm(build_args=...) sub-merits route through
            # the analytic-gradient path instead of falling back to
            # legacy fn(ctx) (which would silently degrade analytic
            # gradients to FD).
            sub_ctx = EvaluationContext(
                prescription=ctx.prescription,
                wavelength=ctx.wavelength, N=ctx.N, dx=ctx.dx,
                efl=ctx.efl, bfl=ctx.bfl, seidel=ctx.seidel,
                E_exit=E_exit, x=getattr(ctx, 'x', None))
            # Through-focus for this field
            if np.isfinite(ctx.bfl) and abs(ctx.bfl) < 10:
                half = max(abs(ctx.bfl) / 20.0, 1e-3)
                z_values = np.linspace(ctx.bfl - half, ctx.bfl + half, 21)
                try:
                    ideal = _core.diffraction_limited_peak(
                        E_exit, ctx.wavelength, ctx.bfl, ctx.dx)
                    scan = _core.through_focus_scan(
                        E_exit, ctx.dx, ctx.wavelength, z_values,
                        ideal_peak=ideal, verbose=False)
                    z_best, strehl_best = _core.find_best_focus(scan, 'strehl')
                    sub_ctx.strehl_best = float(strehl_best)
                    # 4.11.1: nanargmax so a single NaN slice doesn't
                    # steal the argmax.
                    if np.any(np.isfinite(scan.strehl)):
                        i_best = int(np.nanargmax(scan.strehl))
                        sub_ctx.rms_radius_best = float(
                            scan.rms_radius[i_best])
                except (ValueError, RuntimeError, ZeroDivisionError,
                        KeyError, np.linalg.LinAlgError, IndexError,
                        AttributeError, TypeError):
                    # Field-leg through-focus scan failed; zero
                    # Strehl is a safe sentinel (the optimizer treats
                    # it as a very-bad design).  v4.15.3
                    # (P1-NEW-F1-3): wire the
                    # ``_FAILED_SCAN_STREHL_SENTINEL_OBJ`` singleton.
                    # ``float()`` of the singleton still returns 0.0
                    # so the ``max(0.0, min_strehl - ctx.strehl_best)``
                    # arithmetic at the immediate consumer
                    # (``StrehlMerit.evaluate``) continues to work
                    # via the explicit ``float()`` coercion added at
                    # that site.
                    sub_ctx.strehl_best = _FAILED_SCAN_STREHL_SENTINEL_OBJ
            # OPD map if needed
            ap = ctx.prescription.get('aperture_diameter')
            if ap and hasattr(self.sub_merit, 'needs_wave') and self.sub_merit.needs_wave:
                try:
                    _, _, opd = _core.wave_opd_2d(
                        E_exit, ctx.dx, ctx.wavelength,
                        aperture=ap, focal_length=ctx.bfl, f_ref=ctx.bfl)
                    sub_ctx.opd_map = opd
                except (ValueError, RuntimeError, ZeroDivisionError,
                        np.linalg.LinAlgError, IndexError, AttributeError,
                        TypeError):
                    # OPD-map extraction failed (aperture mismatch /
                    # singular least-squares fit); leave None so
                    # downstream Zernike merits return 0 contribution.
                    sub_ctx.opd_map = None
            total = total + self.sub_merit.evaluate(sub_ctx)
        return self.weight * total / max(len(self.field_angles), 1)


# =========================================================================
# Tolerance-aware merit
# =========================================================================

class ToleranceAwareMerit(MeritTerm):
    """Optimise the MEAN of a sub-merit across a set of random
    perturbations.

    Instead of optimising the *nominal* Strehl / wavefront, this
    optimises the *average* over a Monte-Carlo perturbation set.
    Produces designs that are robust to manufacturing tolerances
    rather than fragile at the nominal but excellent on paper.

    .. warning::
        The perturbed wave-leg propagation always calls
        :func:`apply_real_lens` directly, irrespective of the
        ``wave_propagator`` setting on the enclosing
        :func:`design_optimize` call.  See audit P2 #14; a runtime
        warning fires from :func:`design_optimize` when the
        propagator mismatch is detected.

    Parameters
    ----------
    sub_merit : MeritTerm
        The merit evaluated at each perturbation (typically
        ``StrehlMerit`` or ``RMSWavefrontMerit``).
    perturbation_spec : list of dict
        Same format as for :func:`monte_carlo_tolerancing`:
        ``[{'surface_index': i, 'decenter_std': ..., 'tilt_std': ...,
            'form_error_rms': ...}]``
    n_trials : int
        Number of random perturbation draws per evaluation.
    seed : int
        Base seed for reproducibility.
    weight : float
    """

    name = 'ToleranceAware'

    def __init__(self, sub_merit: MeritTerm,
                 perturbation_spec: Sequence[Dict[str, Any]],
                 n_trials: int = 5, seed: int = 42,
                 weight: float = 1.0) -> None:
        self.sub_merit = sub_merit
        self.perturbation_spec = list(perturbation_spec)
        self.n_trials = int(n_trials)
        self.seed = int(seed)
        self.weight = float(weight)
        self.needs_wave = sub_merit.needs_wave

    def evaluate(self, ctx: Any) -> float:
        from . import core as _core
        from ..analysis.through_focus import apply_perturbations, Perturbation

        total = 0.0
        for t in range(self.n_trials):
            rng = np.random.default_rng(self.seed + t)
            perts = []
            for spec_idx, spec in enumerate(self.perturbation_spec):
                d_std = spec.get('decenter_std', 0.0)
                t_std = spec.get('tilt_std', 0.0)
                f_rms = spec.get('form_error_rms', 0.0)
                # Deterministic form-error seed: tying it directly to
                # the trial index + surface index means two runs with
                # the same ``self.seed`` produce identical form-error
                # realisations regardless of the global RNG state.
                # Mask to 31 bits to match the Perturbation API.
                fe_seed = ((self.seed + t) * 1_000_003
                           + spec['surface_index']
                           + spec_idx * 17) & 0x7FFFFFFF
                perts.append(Perturbation(
                    surface_index=spec['surface_index'],
                    decenter=(rng.normal(0, d_std) if d_std > 0 else 0.0,
                              rng.normal(0, d_std) if d_std > 0 else 0.0),
                    tilt=(rng.normal(0, t_std) if t_std > 0 else 0.0,
                          rng.normal(0, t_std) if t_std > 0 else 0.0),
                    form_error_rms=f_rms,
                    random_seed=fe_seed,
                    name=f'tol_trial_{t}_s{spec["surface_index"]}'))
            pres_pert = apply_perturbations(
                ctx.prescription, perts, N=ctx.N, dx=ctx.dx)

            # Per-trial ABCD: the perturbed prescription generally has a
            # different EFL/BFL from the nominal, and scanning around
            # the nominal BFL misses the actual best focus (giving an
            # artificially low Strehl that drags the optimizer away).
            try:
                surfs_p = _core.surfaces_from_prescription(pres_pert)
                _, efl_p, bfl_p, _ = _core.system_abcd(surfs_p, ctx.wavelength)
                efl_p = float(efl_p) if np.isfinite(efl_p) else ctx.efl
                bfl_p = float(bfl_p) if np.isfinite(bfl_p) else ctx.bfl
            except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                    np.linalg.LinAlgError, IndexError, TypeError):
                # Perturbed ABCD failed -- fall back to nominal
                # focus, which will under-estimate the Strehl drop
                # but is a stable sentinel.  v4.15.3 (P1-NEW-F1-3):
                # this branch is INTENTIONALLY left unwired.  The
                # fallback is a tuple-pattern ``(efl_p, bfl_p)`` not
                # a single scalar, and wrapping a tuple in a single
                # sentinel singleton would break downstream
                # ``sub_ctx.efl=efl_p``/``sub_ctx.bfl=bfl_p`` usage
                # (the consumer expects two floats, not a tuple).
                # v4.15.4 (AUDIT_V4_15_3 P2-NEW-F1-B option a) deleted
                # the defunct ``_PerturbedABCDFallbackSentinel`` class
                # outright; see the audit closure block in the v4.15.4
                # release notes.
                efl_p, bfl_p = ctx.efl, ctx.bfl

            # Re-run wave propagation for this perturbation
            # 4.10.2: honour the runtime DEFAULT_COMPLEX_DTYPE so
            # precision='single' actually halves the memory / FFT cost
            # of merit-leg propagation.  Pre-4.10.2 the hard-coded
            # complex128 silently negated the precision='single' knob.
            # v4.14.0: route through the shared wrapper-merit cache
            # so the (Ny, Nx, dx) grid-build invariants are
            # established once per design_optimize run.  The cache
            # also memoises the np.ones source array against
            # mutation by apply_real_lens (which never writes its
            # input), so per-trial we just .copy() the cached
            # template.
            _cdtype = _core.get_default_complex_dtype()
            _ap = ctx.prescription.get('aperture_diameter')
            # v4.14.2 (P1-NEW-1): the perturbed prescription preserves
            # ``aperture_diameter`` from the nominal, so a nominal-zero
            # aperture flows through to the per-trial wave-leg unchanged.
            # ``apply_perturbations`` itself does NOT call
            # ``validate_prescription``, so the validation-time rejection
            # of ``aperture_diameter <= 0`` cannot be relied upon to gate
            # this code path.  Honour the ``_ZERO_APERTURE_MASK`` sentinel
            # placed in ``_cache['mask']`` by ``_get_wrapper_merit_cache``
            # so a deliberate-zero aperture produces a zero E_in rather
            # than the cached full-ones template (which would otherwise
            # propagate a grid-filling plane wave through ``apply_real_lens``
            # and silently mis-score the perturbed trial).  Matches the
            # canonical branch at ``MultiWavelengthMerit.evaluate`` and
            # ``MultiFieldMerit.evaluate``.
            _cache = _get_wrapper_merit_cache(
                ctx.N, ctx.dx, _ap, _cdtype)
            if _cache['mask'] is _ZERO_APERTURE_MASK:
                E_in = np.zeros((ctx.N, ctx.N), dtype=_cdtype)
            else:
                E_in = _cache['E_ones'].copy()
            E_exit = _core.apply_real_lens(
                E_in, prescription=pres_pert, wavelength=ctx.wavelength, dx=ctx.dx)
            # v4.13.2 (C-P1-2): thread ctx.x so JaxMeritTerm sub-
            # merits with build_args reach the analytic-gradient
            # path instead of legacy fn(ctx) -> FD.
            sub_ctx = EvaluationContext(
                prescription=pres_pert, wavelength=ctx.wavelength,
                N=ctx.N, dx=ctx.dx, efl=efl_p, bfl=bfl_p,
                x=getattr(ctx, 'x', None))
            # Through-focus scan around the PERTURBED BFL, not the
            # nominal BFL.
            if np.isfinite(bfl_p) and abs(bfl_p) < 10:
                half = max(abs(bfl_p) / 20.0, 1e-3)
                z_values = np.linspace(bfl_p - half, bfl_p + half, 11)
                try:
                    ideal = _core.diffraction_limited_peak(
                        E_exit, ctx.wavelength, bfl_p, ctx.dx)
                    scan = _core.through_focus_scan(
                        E_exit, ctx.dx, ctx.wavelength, z_values,
                        ideal_peak=ideal, verbose=False)
                    z_best, strehl_best = _core.find_best_focus(scan, 'strehl')
                    sub_ctx.strehl_best = float(strehl_best)
                except (ValueError, RuntimeError, ZeroDivisionError,
                        KeyError, np.linalg.LinAlgError, IndexError,
                        AttributeError, TypeError):
                    # Tolerancing trial through-focus failed; treat
                    # this perturbation as worst-case (Strehl=0).
                    # v4.15.3 (P1-NEW-F1-3): wire the
                    # ``_FAILED_SCAN_STREHL_SENTINEL_OBJ`` singleton.
                    # Sibling branch to the MultiFieldMerit one above;
                    # same ``float()``-coercion contract at the consumer.
                    sub_ctx.strehl_best = _FAILED_SCAN_STREHL_SENTINEL_OBJ
            total = total + self.sub_merit.evaluate(sub_ctx)
        return self.weight * total / max(self.n_trials, 1)
