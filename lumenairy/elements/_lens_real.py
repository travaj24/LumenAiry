"""
lumenairy.elements._lens_real -- analytic split-step real-lens propagator.

Models a multi-surface refractive lens prescription as a sequence of
per-surface phase screens with angular-spectrum (or
Huygens-Fresnel / Rayleigh-Sommerfeld / Scalable-ASM) propagation
through the glass between them.  Captures exact surface sag (including
high-order spherical aberration), diffraction during in-glass
propagation, thickness effects, and compound lenses (doublets,
triplets, etc.).

Extracted from ``lenses.py`` in v3.5.5 to reduce that module's bloat.
``apply_real_lens`` is re-exported from
:mod:`lumenairy.elements.lenses` so existing imports continue to work.

Author: Andrew Traverso
"""

from __future__ import annotations

import importlib.util as _importlib_util
import os as _os
import threading as _threading
import time as _time
import warnings as _warnings
import weakref as _weakref
from typing import Any, Dict, Optional

import numpy as np

# Optional CuPy backend (lazy).  The availability probe, the first-use import
# and the isinstance test live in ONE place for the whole library
# (``backend/_optional.py``; audit 2026-09-11 TESTS-ARCH P2-9 measured five
# hand-copied implementations of each).
from ..backend._optional import (
    CUPY_AVAILABLE,
    ensure_cupy as _ensure_cupy,
    is_cupy_array as _optional_is_cupy_array,
)

cp = None  # this module's alias for the cupy module; see _ensure_cupy_loaded


def _ensure_cupy_loaded():
    """Load CuPy on first use; return True iff it is available.

    Keeps this module's ``cp`` alias populated because the GPU branches here
    read the module-level name directly (``xp = cp if _is_cupy_array(E) else
    np``); the import itself and its cache live in
    :mod:`lumenairy.backend._optional`.
    """
    global cp
    if cp is None:
        cp = _ensure_cupy()
    return cp is not None


def _is_cupy_array(x):
    """Reliable CuPy array check -- ``isinstance`` against the real CuPy type.

    ``hasattr(x, 'device')`` is not a usable duck-type test: NumPy 2.x exposes
    ``ndarray.device`` as part of the Python Array API, so every NumPy array
    would be routed into the (unusable without CUDA) CuPy branch.
    """
    if not CUPY_AVAILABLE:
        # Local short-circuit, not a delegation: the CuPy-absent answer has to
        # stay one global read on a path the band loop takes per surface.
        return False
    if not _optional_is_cupy_array(x):
        return False
    _ensure_cupy_loaded()   # a True answer implies ``cp`` is live -- bind it
    return True


# Optional numexpr fused-expression backend (lazy).
NUMEXPR_AVAILABLE = _importlib_util.find_spec('numexpr') is not None
_ne = None


def _ensure_numexpr_loaded():
    global _ne
    if _ne is None and NUMEXPR_AVAILABLE:
        import numexpr as _n
        _ne = _n
    return _ne is not None


def _drop_numexpr_out_retention():
    """Drop numexpr's per-thread reference to the last ``out=`` array.

    ``numexpr.evaluate`` is
    implemented as ``validate`` + ``re_evaluate``, and ``validate`` parks the
    whole kwargs dict -- ``out`` INCLUDED -- in
    ``numexpr.necompiler._numexpr_last`` so the replay has something to read.
    That reference is thread-local and lives until the next ``evaluate`` on
    the same thread, which on the traced route means "until the chain ends":
    ``apply_real_lens`` is the last numexpr caller in the element, so the
    field it returns stays reachable through numexpr long after the caller
    has ``del``'d its only name for it.

    MEASURED before this drain (weakref taken at the ``del`` line in
    ``apply_real_lens_traced``): ``E_analytic`` -- a full-grid complex128,
    4.295 GB at ``n_fine = 16384`` -- was STILL ALIVE at the element's
    return, at the fine leg's return AND at the end of the chain, on the
    ray_density + remap + lattice route design 121 ships.  Eleven other
    ``del`` sites freed correctly; this one did not, and the frame census
    could not see it because the census sums ``f_locals`` and the NAME is
    gone.

    Called immediately after every ``out=`` evaluate here, so the retention
    never outlives the statement that created it.  Safe by construction:
    ``evaluate`` is ``validate`` + ``re_evaluate`` and consumes the record
    inside its own call, so clearing it afterwards cannot disturb a
    computation in flight, and the library never calls ``re_evaluate``
    itself (a later user call to it now raises instead of silently replaying
    into a buffer this library owns -- which is the correct outcome, since
    the "previous evaluate" it would replay is an internal phase screen).

    ``.clear()`` and not ``del _numexpr_last.l``: since numexpr 2.11 the
    record is a ``ContextDict`` whose payload lives in a ``contextvars``
    ContextVar, so dropping the thread-local ATTRIBUTE leaves the array
    reachable through the context (MEASURED: the weakref probe still read
    STILL ALIVE, with the referrer still the same 4-key kwargs dict).
    ``.clear()`` empties the ContextVar, and it is also correct for the
    plain dict older numexpr used.  Best-effort: a numexpr whose internals
    move again simply leaves the reference where it was.
    """
    try:
        from numexpr import necompiler as _nc
        rec = getattr(_nc._numexpr_last, 'l', None)
        if rec is not None:
            rec.clear()
    except (ImportError, AttributeError, TypeError):  # pragma: no cover - numexpr detail
        pass


# Minimum field size at which the numexpr phase-screen path beats the straight
# numpy multiply: the expression-compile + thread-dispatch overhead is fixed
# This is the ONLY live copy of the constant, and it sits next to its
# three readers below.  The propagators keep their own
# ``asm._NE_MIN_SIZE``, deliberately separate.
_NUMEXPR_MIN_SIZE = 1 << 20  # 1 Mi elements (~1024 x 1024)


# Helpers shared with lenses.py / lenses_maslov.py.
from .lenses import (
    _warn_if_aperture_exceeds_grid,
    surface_sag_biconic,
    surface_sag_general,
)

# Private alias used inside the function body (matches lenses.py convention).
_surface_sag_general = surface_sag_general

from ..glass import get_glass_index, get_glass_index_complex
from ..progress import call_progress
from ..propagators.propagation import angular_spectrum_propagate

# Configuration objects (audit 2026-09-11 TESTS-ARCH section 14 item 13).
# ``lens_config`` is a LEAF -- it imports nothing from lumenairy at module
# scope -- so this edge is one-way and adds no import cost.
from .lens_config import (
    LensConfig,
    LensGeometry,
    LensNumerics,
    LensResources,
    _wants_config,
    resolve_entry_point_kwargs as _resolve_lens_config,
)

_VALID_WAVE_PROPAGATORS = ('asm', 'sas', 'fresnel', 'rayleigh_sommerfeld', 'rs')


# ---------------------------------------------------------------------------
# Opt-in geometry (sag / coordinate) precision.
# ---------------------------------------------------------------------------
# The float64 coordinate lineage (x/y meshgrids -> h_sq -> sag -> opd) is the
# dtype-INDEPENDENT memory core of the real-lens propagators: it does NOT
# shrink with a complex64 field.  Downcasting it to float32 halves that core
# (the reclaim that lets N=32768 fit a 137 GB box) but drops the surface-
# departure precision to ~1e-7 relative, which over a ~9 mm aperture is a
# sub-nm..nm OPD error.  ACCURACY-RISKY: validate with
# ``lens_sag_float32_opd_error`` before trusting a float32-sag result.  Shipped
# default None -> float64 (byte-identical to prior releases).
_LENS_SAG_DTYPE = None   # None -> float64


def set_lens_sag_dtype(dtype: Any) -> None:
    """Set the process-wide geometry (sag/coordinate) dtype for the real-lens
    propagators.  ``np.float32`` halves the float64 coordinate/sag/opd core
    (enabling larger grids) at an accuracy cost -- validate first with
    :func:`lens_sag_float32_opd_error`.  ``np.float64`` (or ``None``) restores
    the byte-identical default."""
    global _LENS_SAG_DTYPE
    if dtype is None:
        _LENS_SAG_DTYPE = None
        return
    d = np.dtype(dtype)
    if d not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(
            "set_lens_sag_dtype: dtype must be float32 or float64, "
            f"got {dtype!r}.")
    _LENS_SAG_DTYPE = None if d == np.dtype(np.float64) else np.float32


def get_lens_sag_dtype() -> Any:
    """Return the process-wide geometry dtype (``np.float32`` when set, else
    ``np.float64`` = the default)."""
    return np.float32 if _LENS_SAG_DTYPE is np.float32 else np.float64


# Restorable through ``lumenairy.override(lens_sag_dtype=...)`` and through the
# suite's autouse snapshot/restore fixture (audit 2026-09-11 TESTS-ARCH P2-5:
# 53 ``set_`` verbs, 0 context-manager forms, 0 resets, in a suite that runs
# serially).  The getter is a single global read with no side effect, and
# ``set_lens_sag_dtype(get_lens_sag_dtype())`` is an exact no-op round trip
# (``np.float64`` maps back to the ``None`` sentinel).
from .._knobs import register_knob as _register_knob  # noqa: E402

_register_knob(
    'lens_sag_dtype',
    getter=get_lens_sag_dtype,
    setter=set_lens_sag_dtype,
    doc="Geometry (sag/coordinate) dtype for the real-lens propagators: "
        "np.float64 (default, byte-identical to prior releases) or np.float32 "
        "(halves the dtype-independent memory core at ~1e-7 relative surface "
        "departure -- validate with lens_sag_float32_opd_error first).")


def _resolve_sag_real(sag_dtype: Any,
                      fn_name: str = 'apply_real_lens') -> Any:
    """Resolve the effective REAL geometry dtype: explicit kwarg wins, else
    the process global, else float64.

    An unrecognised dtype is REFUSED, with the same rule
    :func:`set_lens_sag_dtype` and ``LensResources.sag_dtype`` enforce.  This
    one setting has three spellings -- the per-call keyword, the process knob
    and the config field -- and they have to agree about what is legal, or the
    per-call one is a place a setting can be discarded in silence: resolving
    ``sag_dtype=np.float16`` to float64 without a word gives a caller who
    asked for half-precision geometry the default and no way to find out.
    """
    d = sag_dtype if sag_dtype is not None else _LENS_SAG_DTYPE
    if d is None:
        return np.float64
    try:
        dt = np.dtype(d)
    except TypeError:
        raise ValueError(
            f"{fn_name}: sag_dtype={sag_dtype!r} is not a dtype.  Pass None "
            f"(float64, the default), np.float64 or np.float32.") from None
    if dt == np.dtype(np.float32):
        return np.float32
    if dt == np.dtype(np.float64):
        return np.float64
    raise ValueError(
        f"{fn_name}: sag_dtype={sag_dtype!r} must be float32 or float64 (the "
        f"geometry lineage is real).  Pass None or np.float64 for the "
        f"byte-identical default, np.float32 to halve the geometry core.")


# Row-band (chunked) lens mode auto-default (v5.17.0).  The banded path is
# BYTE-IDENTICAL to the whole-grid path (wall clock: neutral to +9 % at
# N >= 1024, +28 % at N = 512 for an explicit band -- see the
# ``sag_chunk_rows`` docstring), so it is ON
# by default for grids large enough to benefit; below the threshold the
# whole-grid path runs exactly as before (band-loop overhead isn't worth it
# on small grids).  ``sag_chunk_rows=None`` -> auto; an explicit int > 0
# forces that band size; ``0`` forces the whole-grid path.
_SAG_CHUNK_AUTO_MIN_N = 4096
_SAG_CHUNK_AUTO_MIN_ROWS = 256


def _resolve_sag_chunk_rows(sag_chunk_rows: Optional[int], n_rows: int) -> Optional[int]:
    """Resolve the effective row-band size: ``None`` -> auto
    (``max(256, N // 16)`` when ``N >= 4096``, else whole-grid); ``0`` (or
    negative) -> whole-grid; a positive int -> that band size."""
    if sag_chunk_rows is None:
        if n_rows >= _SAG_CHUNK_AUTO_MIN_N:
            return max(_SAG_CHUNK_AUTO_MIN_ROWS, n_rows // 16)
        return None
    return int(sag_chunk_rows) if int(sag_chunk_rows) > 0 else None


def lens_sag_float32_opd_error(prescription: Dict[str, Any],
                               wavelength: float,
                               *,
                               aperture: Optional[float] = None,
                               n_samples: int = 4096,
                               field_check_n: int = 512,
                               field_check_dx: Optional[float] = None,
                               max_field_rel_error: float = 1e-3,
                               on_partial_aperture: str = 'warn'
                               ) -> Dict[str, Any]:
    """Estimate the error incurred by float32 sag precision for one
    prescription, so a caller can decide whether
    ``set_lens_sag_dtype(np.float32)`` (or ``sag_dtype=np.float32``) is safe.

    Two independent checks:

    1. **Radial OPD scan** (1-D, cheap): the summed per-surface refraction
       OPD ``sum (n2 - n1) * sag(r)`` in float32 vs float64, reported in
       waves.
    2. **Field-level A/B** (grid ``field_check_n`` / ``field_check_dx``): a
       full ``apply_real_lens`` run in float32 vs float64 geometry,
       reporting the max relative exit-field error.  This catches what the
       OPD scan systematically UNDER-reports: the exit-field phase error
       scales with the TOTAL sag depth (``k0 * OPD * eps_f32``), so a deep
       singlet can show a negligible waves-level OPD delta yet a >1e-3
       field error.

    IMPORTANT: the field-level error is CONFIG-DEPENDENT -- the f32 phase
    perturbation interferes through the in-glass diffraction, so its
    magnitude depends on dx / grid fill / beam extent, not just the
    prescription.  The DEFAULT coarse check (auto dx, N=512) is a
    gross-failure screen only; for production sign-off pass your actual
    pixel pitch via ``field_check_dx=`` (and a representative
    ``field_check_n``) so the A/B reproduces your sampling regime.

    ``ok`` requires BOTH: OPD peak < lambda/50 AND field error <
    ``max_field_rel_error``.

    .. warning::
       ``ok`` is a MEASUREMENT OVER THE CHECKED WINDOW, not a bound.  The
       pitch and the WINDOW are two different things, and passing a
       production ``field_check_dx`` fixes only the first.

       **The production-grid adjudication (2026-08-17, design 121).**  Check 1
       already IS a production-grid result: the radial OPD scan runs from the
       axis to the CLEAR-APERTURE EDGE at ``n_samples`` points and costs
       nothing, so ``max_opd_error_waves`` is full-aperture and grid-free.
       Check 2 is the one with a window, and at a production pitch the window
       that covers a production aperture IS the production grid -- design
       121's groups are 20.4 to 31.8 mm across at ``dx = 0.9028 um``, i.e.
       ``field_check_n`` of 32768.  Running that A/B needs two such grids and
       is exactly the run the float32 lever was supposed to make affordable,
       so a true production-grid field check is NOT achievable there.

       What is achievable is knowing when the proxy is one, which is why
       ``aperture_cover`` and ``field_check_n_for_full_aperture`` are
       returned and why ``on_partial_aperture`` warns by default.  The proxy
       does NOT bound the production case: it under-reads, and steeply,
       because the sag -- and with it the float32 rounding error -- grows
       toward the pupil edge that a small window never reaches.  MEASURED on
       design 121 group S25-S27 at the production pitch, ``field_check_n``
       walked up (window / cover of the 20.4 mm aperture / field error)::

            512   0.462 mm   2.3 %   1.1221e-06
           1024   0.925 mm   4.5 %   5.3847e-06
           2048   1.849 mm   9.1 %   2.9377e-05
           4096   3.698 mm  18.1 %   1.2196e-04

       Three doublings and 109x, still climbing at ~4.6x per doubling with
       82 % of the pupil unseen.  A reading taken at 512 is not evidence
       about 32768.

       **WHAT COVER >= 1 BUYS.**  Once the window reaches the clear aperture
       the field reading CONVERGES: measured on a biconvex N-SSK2 singlet
       (R = +19.6 / -27.4 mm, 4 mm aperture, dx = 1.953 um), the L-inf field
       error moved 7.34e-06 -> 2.03e-05 -> 1.5544e-04 across covers of 0.25,
       0.50 and 1.00 -- 21x -- and then 1.5544e-04 -> 1.5564e-04 (+0.1 %)
       going from cover 1.00 to 2.00.  So ``field_check_covers_aperture`` is
       not a stylistic preference: below it the reading is still moving by
       decades, at and above it the reading has stopped moving.  That is the
       sufficiency criterion this guard enforces.

       **AND WHAT ``field_rel_error_estimate`` IS -- AND IS NOT.**  It is
       ``2*pi*max_opd_error_waves``: the size of the phase perturbation the
       full-aperture radial scan already measures, so it is grid-free and
       sees the whole pupil.  It is an ORDER-OF-MAGNITUDE ESTIMATE, **not an
       upper bound**, and the difference was measured rather than assumed.
       The tempting derivation -- phase screens are unimodular, propagations
       are unitary, therefore the field difference is bounded by the OPD
       difference -- FAILS in practice, because the field path also rounds
       the COORDINATE arrays and evaluates sag out to the grid CORNER at
       ``sqrt(2)`` times the window half-extent, neither of which the radial
       scan to the aperture edge sees.  On the fixture above the estimate
       reads 1.2189e-04 against a full-cover measurement of 1.5544e-04
       (L-inf) and 1.7622e-04 (L2): it UNDER-reads by 1.3-1.4x.  Use it as
       what it is -- a grid-free reading that lands within about 2x of the
       production one, where a 512-point proxy under-reads by 100x or more --
       and never as a certificate.

       On design 121's S25-S27 it reads ``2*pi*7.7376e-04 = 4.862e-03``, i.e.
       ~4.9x above the 1e-3 ``max_field_rel_error`` gate, while the 512-point
       proxy reports 1.1e-06 and passes.  Judged on OPD instead, that group's
       worst float32 sag error is 7.74e-04 waves against the
       ``tangent_facet`` screen's own 0.0032-wave residual, so float32 sag is
       not the limiting error in that design even though it does not clear
       this function's field gate.  Both statements are true; quote the one
       your bar is written against.

    Parameters
    ----------
    prescription : dict
        Same surface/aperture prescription apply_real_lens consumes.
    wavelength : float
        Metres.
    aperture : float, optional
        Clear-aperture diameter [m].  Defaults to
        ``prescription['aperture_diameter']``.
    n_samples : int, default 4096
        Radial samples from axis to edge.
    field_check_n : int, default 512
        Grid size for the field-level A/B (0 skips it).
    max_field_rel_error : float, default 1e-3
        Field-error gate for ``ok``.
    on_partial_aperture : {'warn', 'error', 'silent'}, default 'warn'
        What to do when the field check is a PROXY rather than a
        production-grid measurement -- either because its window does not
        cover the clear aperture, or because ``field_check_dx`` was left at
        ``None`` and the pitch was therefore chosen here (to make the aperture
        span 80 % of the window) rather than by the caller.  The pitch
        condition is the one that fires on a default call: with an auto pitch
        the cover is 1.25 by construction, so a cover-only test could never
        warn, and the shipped default ``field_check_n=512`` IS such a proxy on
        any real lens -- reading its ``ok`` as a production sign-off is the
        mistake this exists to stop.  Pass ``field_check_dx=<your production
        pitch>`` to turn the proxy into a measurement, or ``'silent'`` for the
        gross-failure screen the default arguments describe.

    Returns
    -------
    dict
        ``{'max_opd_error_waves', 'rms_opd_error_waves', 'max_opd_error_nm',
        'max_field_rel_error', 'field_rel_error_estimate', 'aperture_m',
        'field_check_n', 'field_check_dx', 'field_check_window_m',
        'aperture_cover', 'field_check_n_for_full_aperture',
        'field_check_covers_aperture', 'ok'}``.

        ``field_rel_error_estimate`` is the grid-free full-aperture
        ESTIMATE ``2*pi*max_opd_error_waves`` discussed in the warning above
        -- an order-of-magnitude reference, measured to under-read the
        production field error by ~1.4x, NOT an upper bound;
        ``aperture_cover`` is the check window divided by the clear aperture,
        and a value below 1 means ``max_field_rel_error`` (and therefore
        ``ok``) saw only part of the pupil.
    """
    ap = aperture if aperture is not None else prescription.get('aperture_diameter')
    if not ap:
        raise ValueError(
            "lens_sag_float32_opd_error: need an aperture -- pass aperture= or "
            "set prescription['aperture_diameter'].")
    r = np.linspace(0.0, float(ap) / 2.0, int(n_samples))

    def _opd(real: Any) -> np.ndarray:
        h_sq = (r.astype(real)) ** 2
        opd = np.zeros_like(h_sq)
        for surf in prescription['surfaces']:
            R = surf['radius']
            kc = surf.get('conic', 0.0)
            asph = surf.get('aspheric_coeffs')
            n1 = get_glass_index(surf['glass_before'], wavelength)
            n2 = get_glass_index(surf['glass_after'], wavelength)
            sag = _surface_sag_general(h_sq, R, kc, asph)
            opd = opd + (n2 - n1) * np.where(np.isnan(sag), 0.0, sag)
        return opd.astype(np.float64)

    d = np.abs(_opd(np.float32) - _opd(np.float64))
    max_waves = float(d.max() / wavelength)

    # The GRID-FREE full-aperture ESTIMATE of the field error: the size of
    # the phase perturbation the radial scan measures over the WHOLE clear
    # aperture.  Unlike ``max_field_rel_error`` it does not depend on the
    # check window.
    #
    # It is NOT a bound, and that was MEASURED, not assumed.  The tempting
    # argument (unimodular screens between unitary propagations, so the field
    # difference cannot exceed the phase difference) fails because the field
    # path also rounds the coordinate arrays and evaluates sag out to the
    # grid CORNER, sqrt(2) beyond the window half-extent -- neither of which
    # a radial scan to the aperture edge sees.  Fixture measurement
    # 2026-08-17 (biconvex N-SSK2 R=+19.6/-27.4, 4 mm aperture, dx 1.953 um,
    # full cover): estimate 1.2189e-04 against 1.5544e-04 L-inf and
    # 1.7622e-04 L2, i.e. it under-reads by 1.3-1.4x.  Pinned two-sided by
    # tests/unit/test_sag_float32_production_window.py.
    field_bound = float(2.0 * np.pi * max_waves)

    field_rel = 0.0
    n_fc = int(field_check_n)
    if n_fc > 0:
        # A/B with the beam filling the aperture.  Default dx sized so the
        # aperture spans ~80% of the grid; pass field_check_dx= to
        # reproduce the production sampling regime instead.
        dx_fc = (float(field_check_dx) if field_check_dx
                 else float(ap) / (0.8 * n_fc))
        xs = (np.arange(n_fc) - n_fc / 2) * dx_fc
        Xf, Yf = np.meshgrid(xs, xs)
        w_beam = float(ap) / 3.0
        E_fc = np.exp(-(Xf**2 + Yf**2) / w_beam**2).astype(np.complex64)
        E64 = apply_real_lens(E_fc.copy(), prescription=prescription,
                              wavelength=wavelength, dx=dx_fc,
                              sag_dtype=np.float64)
        E32 = apply_real_lens(E_fc.copy(), prescription=prescription,
                              wavelength=wavelength, dx=dx_fc,
                              sag_dtype=np.float32)
        m = float(np.abs(E64).max())
        if m > 0:
            field_rel = float(np.abs(E32 - E64).max() / m)
    else:
        dx_fc = float(field_check_dx) if field_check_dx else float('nan')

    # Is the field A/B a MEASUREMENT of the production case or a PROXY for
    # it?  The pitch is the caller's; the WINDOW is n * dx, and the sag (and
    # so the float32 rounding error) is largest at the pupil edge a short
    # window never reaches.  Reported either way; warned about by default.
    window = (n_fc * dx_fc) if n_fc > 0 else 0.0
    cover = (window / float(ap)) if n_fc > 0 else 0.0
    n_full = int(np.ceil(float(ap) / dx_fc)) if n_fc > 0 else 0
    covers = bool(n_fc > 0 and cover >= 1.0)
    # WHY THE PITCH MATTERS AS WELL AS THE COVER.  With
    # ``field_check_dx=None`` the pitch is CHOSEN so the aperture spans
    # 80 % of the window, so ``cover`` is 1.25 by construction and the
    # cover test alone can never fire on a default call -- while the
    # docstring said it warns by default, and the same docstring's
    # "the field-level error is CONFIG-DEPENDENT" paragraph says the PITCH
    # is what the error depends on.  An auto-chosen pitch is by
    # construction not the caller's production sampling (4.88 um here
    # against a 0.9 um production grid), so it makes the reading a proxy
    # exactly as a short window does.  Both conditions raise the same
    # guard, with the message naming which one fired.
    auto_pitch = bool(n_fc > 0 and not field_check_dx)
    if n_fc > 0 and (not covers or auto_pitch):
        from ..propagators.carrier import _guard_dispose
        # Validate by VALUE, never by identity: a policy string built at
        # runtime (os.environ, a config file, an f-string) is not the interned
        # literal, and refusing such a value while naming it valid is exactly
        # the defect this campaign's audit found at
        # _check_screen_obliquity_support (fixed in cbef685).
        if str(on_partial_aperture) not in ('warn', 'error', 'silent'):
            raise ValueError(
                f"lens_sag_float32_opd_error: on_partial_aperture must be one "
                f"of ('warn', 'error', 'silent'), got "
                f"{on_partial_aperture!r}.")
        _why = ("its PITCH was chosen automatically"
                if auto_pitch and covers else
                "its WINDOW is shorter than the pupil"
                if covers is False and not auto_pitch else
                "its PITCH was chosen automatically AND its WINDOW is "
                "shorter than the pupil")
        _guard_dispose(
            str(on_partial_aperture),
            f"lens_sag_float32_opd_error: the field-level A/B ran on a "
            f"{window * 1e3:.4f} mm window ({n_fc} x "
            f"{dx_fc * 1e6:.4f} um) against a {float(ap) * 1e3:.4f} mm clear "
            f"aperture -- it saw {cover * 100:.1f} % of the pupil DIAMETER "
            f"and {_why}, so 'max_field_rel_error' ({field_rel:.4e}) and "
            f"'ok' are a PROXY and not a production-grid measurement.  Pass "
            f"field_check_dx=<your production pitch> to make it a "
            f"measurement.  The float32 sag error "
            f"grows toward the pupil edge, so this reading UNDER-states the "
            f"production one -- measured on design 121 it climbed 109x over "
            f"three window doublings and was still rising at 18 % cover.  "
            f"Either pass field_check_n >= {n_full} (which is the production "
            f"grid, and may not be affordable), or judge the run on the "
            f"grid-free full-aperture ESTIMATE 'field_rel_error_estimate' "
            f"= 2*pi*max_opd_error_waves = {field_bound:.4e}, which sees the "
            f"whole pupil this window cannot (it lands within ~1.4x of a "
            f"full-cover measurement, where a short window under-reads by "
            f"100x or more).  Pass on_partial_aperture='silent' "
            f"if the proxy is what you wanted.",
            exc=ValueError, stacklevel=3)

    return {
        'max_opd_error_waves': max_waves,
        'rms_opd_error_waves': float(np.sqrt(np.mean(d ** 2)) / wavelength),
        'max_opd_error_nm': float(d.max() * 1e9),
        'max_field_rel_error': field_rel,
        'field_rel_error_estimate': field_bound,
        'aperture_m': float(ap),
        'field_check_n': n_fc,
        'field_check_dx': float(dx_fc) if n_fc > 0 else None,
        'field_check_window_m': float(window),
        'aperture_cover': float(cover),
        'field_check_n_for_full_aperture': n_full,
        'field_check_covers_aperture': covers,
        'ok': bool(max_waves < 0.02
                   and field_rel < float(max_field_rel_error)),
    }


# ---------------------------------------------------------------------------
# Opt-in ``surface_model='displaced'`` -- ray-angle-aware refraction OPD.
# ---------------------------------------------------------------------------
# The default per-surface phase screen imprints the paraxial thin-element OPD
# ``(n2 - n1) * sag(r)`` at the transverse coordinate ``r``.  That form is
# blind to the INCOMING RAY ANGLE: it assumes every ray strikes each surface
# travelling parallel to the axis.  For the SECOND (and later) surface of a
# real lens the beam is already converging, so the ray hits obliquely -- and
# the paraxial OPD is orientation-INVARIANT (on a plano-convex singlet the
# curved-first and flat-first prescriptions imprint the identical
# ``(n_glass - 1) * |sag(r)|`` map, which is why the analytic model cannot
# distinguish the textbook 4x spherical-aberration penalty between the two
# orientations -- dual-oracle 43 vs 128 um, analytic 60.4/60.9).
#
# The eikonal-correct piston OPD of a locally-planar refracting facet crossed
# by a ray at angle ``alpha_in`` (to the z-axis) that refracts to ``alpha_out``
# is, back-projecting the transmitted local plane wave to the vertex plane,
#
#     OPD_i(r) = (n2 * cos(alpha_out) - n1 * cos(alpha_in)) * sag_i(r)         (1)
#
# (screen convention ``exp(-i k0 OPD)``; at normal incidence
# ``alpha_in = alpha_out = 0`` this reduces to the paraxial ``(n2 - n1) sag``).
# The cosines carry the incoming-ray-angle physics the paraxial screen drops,
# and they BREAK the plano-convex orientation symmetry (the air-side and
# glass-side ray bends differ), so the model splits the two orientations.
#
# ``alpha_in`` / ``alpha_out`` are sourced from a self-contained meridional ray
# fan traced through the actual conic/aspheric prescription (geometric optics,
# wave-model-independent), tabulated vs the ray's crossing height at each
# surface, and interpolated onto the grid radius.  Validated against BOTH
# campaign oracles (Zemax POP and the grid-free Debye/Huygens integral) at
# Nyquist-compliant sampling: f/5 biconvex r2m 64.5 vs 64.98 (0.7%),
# plano-convex 42.2/127.0 vs 43.2/127.6 (split ratio 0.333 vs 0.339), with the
# EE50/EE80 profiles matched too.  See ``docs/audit_real_lens_
# displaced_2026_07_19.md`` (H2(a) + G2).
#
# INPUT CONGRUENCE (G2 Task 1): the fan is launched along the input congruence
# selected by the ``conjugate`` argument -- COLLIMATED by default (axial fan,
# exact for a collimated input and byte-identical to the pre-G2 fan), or along
# a scalar conjugate ``R_in`` (marginal slope ``h/R_in``), an 'auto' carrier
# fit of ``E_in``, or an explicit wavefront.  The wave field carries the input
# curvature in its own phase; ``conjugate`` only sets the per-surface obliquity
# incidence.  This lifts the pre-G2 "collimated only" restriction -- a
# diverging/converging source now sees the true second-surface incidence.
#
# SAMPLING: like ``apply_real_lens_traced`` (hammer finding H3), the exit
# converging wavefront must be Nyquist-sampled -- ``dx <= lambda / (2 NA_exit)``
# -- or the r2m ALIASES low (the ~40 um "plateau" the 2026-07-18 audit reported
# for this class was a dx=6 um undersampling artefact, not a model floor;
# traced itself reads 40.9 um at dx=6 um and 64.8 um at dx<=3 um).


def _build_displaced_cos_luts(surfaces, thicknesses, wavelength, r_max,
                              n_fan=257, carrier_slope=None):
    """Trace a meridional ray fan through the rotationally-symmetric
    conic/aspheric ``surfaces`` and return, per surface, the LUT
    ``(crossing_height, cos_alpha_in, cos_alpha_out)`` used by the
    ``surface_model='displaced'`` refraction OPD (equation (1) above).

    Pure geometric ray trace (vectorised Newton surface intersection + vector
    Snell); independent of the wave model.  ``cos_alpha_*`` are cosines of the
    ray angle to the z-axis (unit-direction z-component) just before / after
    each surface.  Rays that miss or TIR are dropped from that surface's LUT.

    ``carrier_slope`` (G2 Task 1) generalises the launch congruence.  When
    ``None`` the fan is launched COLLIMATED (axial, ``dz=1, dy=0``) --
    byte-identical to the pre-G2 collimated fan and exact for a collimated
    input.  When a callable ``heights -> g``, each entrance ray at height ``h``
    is launched along the input congruence with marginal slope
    ``g = dW/dy(0, h)`` (the carrier wavefront gradient): the unit launch
    direction is ``(dz, dy) = (1, g)/sqrt(1+g^2)`` -- the eikonal ray normal
    to the input wavefront.  For a scalar conjugate ``s`` this is ``g = h/s``
    (``sin(alpha_in) ~ h/s`` paraxially), so the SECOND (and later) surfaces
    see the true converging/diverging incidence and the obliquity OPD (1)
    reflects the actual illumination, not an assumed collimated pupil.
    """
    n_surf = len(surfaces)
    heights = np.linspace(r_max / n_fan, r_max, int(n_fan))
    idx = []
    for s in surfaces:
        n1 = float(get_glass_index(s['glass_before'], wavelength))
        n2 = float(get_glass_index(s['glass_after'], wavelength))
        idx.append((n1, n2))

    # Ray state: position (pz, py), unit dir (dz, dy).  Collimated by default;
    # otherwise launched along the input congruence (carrier normal).
    nf = heights.size
    pz = np.zeros(nf)
    py = heights.astype(np.float64).copy()
    if carrier_slope is None:
        dz = np.ones(nf)
        dy = np.zeros(nf)
    else:
        g = np.asarray(carrier_slope(heights), dtype=np.float64).reshape(nf)
        g = np.where(np.isfinite(g), g, 0.0)
        nrm = np.sqrt(1.0 + g * g)
        dz = 1.0 / nrm
        dy = g / nrm
    alive = np.ones(nf, dtype=bool)
    z_v = 0.0
    luts = []
    for i, s in enumerate(surfaces):
        R = s['radius']
        kc = s.get('conic', 0.0) or 0.0
        asph = s.get('aspheric_coeffs')
        n1, n2 = idx[i]
        flat = (R == 0) or (not np.isfinite(R))

        # ---- intersect ray with z = z_v + sag(|y|) -------------------------
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (z_v - pz) / dz                     # vertex-plane start
        if flat:
            pz = pz + t * dz
            py = py + t * dy
            nrm_z = np.ones(nf)
            nrm_y = np.zeros(nf)
        else:
            # Newton on g(t) = pz + t dz - z_v - sag(|py + t dy|).
            for _ in range(24):
                y = py + t * dy
                r = np.abs(y)
                sag = _surface_sag_general(r * r, R, kc, asph)
                sag = np.where(np.isnan(sag), 0.0, sag)
                e = np.maximum(1e-9, 1e-6 * r)
                sp = _surface_sag_general((r + e) ** 2, R, kc, asph)
                sm = _surface_sag_general((r - e) ** 2, R, kc, asph)
                sp = np.where(np.isnan(sp), 0.0, sp)
                sm = np.where(np.isnan(sm), 0.0, sm)
                sagp = (sp - sm) / (2.0 * e)         # dsag/dr
                g = pz + t * dz - z_v - sag
                dgdt = dz - sagp * np.sign(y) * dy
                dgdt = np.where(np.abs(dgdt) < 1e-30, 1e-30, dgdt)
                # Byte-identical early exit (not a tolerance): once ``t`` reaches a
                # BITWISE fixed point, every remaining sweep reproduces it exactly, so
                # the loop can stop without changing a single output bit.  The
                # intersection residual is exactly 0 after 2 sweeps on every fixture
                # measured, and each sweep costs three ``_surface_sag_general``
                # evaluations over the whole fan -- ~10x of the geometric traces.
                _t_new = t - g / dgdt
                _t_done = np.array_equal(_t_new, t, equal_nan=True)
                t = _t_new
                del _t_new
                if _t_done:
                    break
            pz = pz + t * dz
            py = py + t * dy
            y = py
            r = np.abs(y)
            e = np.maximum(1e-9, 1e-6 * r)
            sp = _surface_sag_general((r + e) ** 2, R, kc, asph)
            sm = _surface_sag_general((r - e) ** 2, R, kc, asph)
            sp = np.where(np.isnan(sp), 0.0, sp)
            sm = np.where(np.isnan(sm), 0.0, sm)
            sagp = (sp - sm) / (2.0 * e)
            # Surface normal of z = z_v + sag(|y|): grad(z - sag) = (1, -sagp).
            nz = np.ones(nf)
            ny = -sagp * np.sign(y)
            nn = np.hypot(nz, ny)
            nrm_z = nz / nn
            nrm_y = ny / nn

        cos_in_z = dz.copy()                         # ray angle to z (incoming)
        cos_i = dz * nrm_z + dy * nrm_y              # AOI cosine (to normal)
        eta = n1 / n2
        sin2t = eta * eta * (1.0 - cos_i * cos_i)
        alive = alive & np.isfinite(py) & (sin2t <= 1.0)
        cos_t = np.sqrt(np.maximum(1.0 - sin2t, 0.0))
        ndz = eta * dz + (cos_t - eta * cos_i) * nrm_z
        ndy = eta * dy + (cos_t - eta * cos_i) * nrm_y
        nn2 = np.hypot(ndz, ndy)
        nn2 = np.where(nn2 == 0.0, 1.0, nn2)
        dz = ndz / nn2
        dy = ndy / nn2
        cos_out_z = dz.copy()                        # ray angle to z (outgoing)

        h_cross = np.abs(py)
        m = alive & np.isfinite(cos_in_z) & np.isfinite(cos_out_z)
        hh = h_cross[m]
        ci = cos_in_z[m]
        co = cos_out_z[m]
        order = np.argsort(hh)
        luts.append((hh[order], ci[order], co[order]))

        if i < n_surf - 1:
            z_v += thicknesses[i]
    return luts


def _displaced_opd(sag, r, lut, n1r, n2r):
    """Ray-angle-aware refraction OPD (equation (1)): interpolate the fan
    cosines onto grid radius ``r`` and return
    ``(n2 cos_alpha_out - n1 cos_alpha_in) * sag``.  ``lut`` is one
    ``(h, cos_in, cos_out)`` tuple from :func:`_build_displaced_cos_luts`."""
    h_lut, cin_lut, cout_lut = lut
    if h_lut.size == 0:
        # No rays survived (fully vignetted / TIR); fall back to paraxial.
        return (n2r - n1r) * sag
    cos_in = np.interp(r, h_lut, cin_lut, left=cin_lut[0], right=cin_lut[-1])
    cos_out = np.interp(r, h_lut, cout_lut, left=cout_lut[0], right=cout_lut[-1])
    return (n2r * cos_out - n1r * cos_in) * sag


def _displaced_carrier_slope_fn(conjugate, E_in, wavelength, dx, dy, Nx, Ny):
    """Return a callable ``heights -> g`` giving the meridional launch slope
    ``g = dW/dy`` at ``(x=0, y=height)`` for the input CONGRUENCE described by
    ``conjugate`` (G2 Task 1), mirroring ``apply_real_lens_traced``'s
    ``_compute_carrier`` vocabulary:

    * ``None`` / ``+-inf`` -> collimated: returns ``None`` (the fan launches
      axially -- byte-identical to the pre-G2 collimated fan).
    * ``float`` signed conjugate ``s`` (m) -> ``g(h) = h / s``
      (``s > 0`` diverging source in front of the lens, ``s < 0`` converging).
    * ``'auto'`` -> a low-order polynomial carrier fit of ``E_in`` (reuses
      ``_compute_carrier``); the meridional slope is ``dW/dy(0, h)``.
    * ``ndarray`` -> an explicit wavefront ``W`` (m, field-shaped); the slope
      is the interpolated ``dW/dy`` along the central column.

    The screen is input-independent GIVEN the conjugate, so the ``None`` /
    scalar paths are cached (see :func:`_get_displaced_cos_luts`); the
    ``'auto'`` / ``ndarray`` paths depend on ``E_in`` and are not cached.
    """
    if conjugate is None:
        return None
    if isinstance(conjugate, (int, float)) and not isinstance(conjugate, bool):
        s = float(conjugate)
        if not np.isfinite(s):
            return None                     # +-inf == collimated
        if s == 0.0:
            raise ValueError(
                "apply_real_lens: surface_model='displaced' conjugate distance "
                "must be non-zero (0 is the source's own focus).")

        def _scalar_slope(h):
            return np.asarray(h, dtype=np.float64) / s

        return _scalar_slope

    # 'auto' / ndarray: reuse the traced carrier machinery along the meridian.
    from ._lens_traced import _compute_carrier
    xax = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    yax = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    Xg, Yg = np.meshgrid(xax, yax)
    # ``dy`` is forwarded: this entry point supports anamorphic grids and
    # the ndarray-carrier branch differentiates and samples per axis
    # (VERIFY-A3 OI-10).  Bit-identical when dy == dx.
    _, grad_fn, _ = _compute_carrier(conjugate, E_in, wavelength, dx, Xg,
                                     Yg, dy=dy)

    def _carrier_slope(h):
        h = np.asarray(h, dtype=np.float64)
        _, Mv = grad_fn(np.zeros_like(h), h)
        return np.asarray(Mv, dtype=np.float64)

    return _carrier_slope


# ---------------------------------------------------------------------------
# Displaced cosine-LUT cache (G2 Task 1) -- the screen is field-independent
# GIVEN the (prescription, conjugate), so a design/optimisation loop varying
# only the field reuses the meridional-fan trace.  Bounded (FIFO-evicted) +
# registered with the central cache registry (G1 cache-audit conventions);
# only the field-INDEPENDENT congruences (collimated / scalar conjugate) are
# cached -- 'auto'/ndarray carriers depend on E_in and always rebuild.
# ---------------------------------------------------------------------------
_DISPLACED_LUT_CACHE: Dict[Any, Any] = {}
_DISPLACED_LUT_CACHE_MAX = 8
_DISPLACED_LUT_CACHE_LOCK = _threading.Lock()


def clear_displaced_lut_cache() -> None:
    """Drop every cached ``surface_model='displaced'`` cosine LUT.

    Forces the next displaced call to re-trace the meridional fan.  Registered
    with the central cache registry, so :func:`lumenairy.clear_asm_caches`
    drains it too."""
    with _DISPLACED_LUT_CACHE_LOCK:
        _DISPLACED_LUT_CACHE.clear()


try:
    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    _register_cache_clearer('displaced_cos_luts', clear_displaced_lut_cache)
except ImportError:
    pass


def _glass_key_value(name, wavelength):
    """The cache-key entry for a glass: its RESOLVED real index, not its name.

    ``GLASS_REGISTRY`` is a documented, mutable user extension point, so a name
    is not a stable identifier: re-pointing an entry leaves the key unchanged
    and the cache returns cosines traced against the OLD glass (measured 1 % of
    peak amplitude on the public API).  ``get_glass_index`` is memoised by
    ``_GLASS_VALUE_CACHE`` and invalidated on a registry write, so this costs a
    dict lookup and is strictly more correct -- two names with the same index
    at this wavelength then legitimately share an entry.  Falls back to the
    name only if the glass cannot be resolved at all (the caller will raise
    later with a better message than a KeyError from here)."""
    try:
        return float(get_glass_index(name, wavelength))
    except Exception:
        return ('unresolved', str(name))


def _sag_callable_fingerprint(cb, r_max):
    """VALUE fingerprint of a freeform ``sag_callable`` for a cache key.

    Object identity does not imply value equality for a MUTABLE callable, and
    the cos-grid cache is sold for exactly the workload that mutates one (a
    design-iteration loop re-using a multi-second trace).  Probing the callable
    on a FIXED stencil spanning the traced extent turns a state change into a
    cache MISS instead of a stale hit -- measured 164 % of peak field error
    before, for microseconds of probe against a ~3.9 s trace.

    Returned alongside (never instead of) the callable object itself, so the
    entry still holds a reference (no GC, hence no ``id`` reuse) and two
    distinct callables still miss."""
    if cb is None:
        return None
    r = float(r_max)
    if not np.isfinite(r) or r <= 0.0:
        r = 1.0
    t = np.linspace(-1.0, 1.0, 8) * r
    z8 = np.zeros(8)
    xs = np.concatenate([t, z8, 0.5 * t])
    ys = np.concatenate([z8, t, 0.5 * t])
    try:
        v = np.asarray(cb(xs, ys), dtype=np.float64).ravel()
    except Exception:
        # Unprobeable callable (e.g. it rejects vector input): fall back to
        # "never cacheable" rather than to a fingerprint we cannot trust.
        return 'unprobeable'
    v = np.where(np.isfinite(v), v, 0.0)
    return np.ascontiguousarray(v).tobytes()


def _displaced_geom_key(surfaces, thicknesses, wavelength, r_max, conjugate):
    """Hashable identity of the FIELD-INDEPENDENT displaced fan (surfaces +
    thicknesses + wavelength + fan extent + scalar conjugate).  Only the
    collimated (``conjugate is None``) and scalar-conjugate congruences are
    cacheable; returns ``None`` for the field-dependent 'auto'/ndarray cases.
    Glasses enter by RESOLVED INDEX, not by registry name -- see
    :func:`_glass_key_value`."""
    if not (conjugate is None
            or (isinstance(conjugate, (int, float))
                and not isinstance(conjugate, bool))):
        return None
    surf_key = tuple((
        float(s.get('radius', np.inf))
        if np.isfinite(s.get('radius', np.inf)) else np.inf,
        float(s.get('conic', 0.0) or 0.0),
        (tuple(sorted((int(p), float(a))
                      for p, a in s['aspheric_coeffs'].items()))
         if s.get('aspheric_coeffs') else None),
        _glass_key_value(s.get('glass_before'), wavelength),
        _glass_key_value(s.get('glass_after'), wavelength))
        for s in surfaces)
    conj_key = None if conjugate is None else float(conjugate)
    return (surf_key, tuple(float(t) for t in thicknesses),
            float(wavelength), float(r_max), conj_key)


def _get_displaced_cos_luts(surfaces, thicknesses, wavelength, r_max,
                            conjugate, carrier_slope):
    """Build (or fetch from the bounded cache) the per-surface displaced cosine
    LUTs for the given congruence.  Field-independent congruences are cached;
    'auto'/ndarray always rebuild."""
    key = _displaced_geom_key(surfaces, thicknesses, wavelength, r_max,
                              conjugate)
    if key is not None:
        with _DISPLACED_LUT_CACHE_LOCK:
            hit = _DISPLACED_LUT_CACHE.get(key)
        if hit is not None:
            return hit
    luts = _build_displaced_cos_luts(
        surfaces, thicknesses, wavelength, r_max, carrier_slope=carrier_slope)
    if key is not None:
        with _DISPLACED_LUT_CACHE_LOCK:
            if len(_DISPLACED_LUT_CACHE) >= _DISPLACED_LUT_CACHE_MAX:
                _DISPLACED_LUT_CACHE.pop(next(iter(_DISPLACED_LUT_CACHE)))
            _DISPLACED_LUT_CACHE[key] = luts
    return luts


# ---------------------------------------------------------------------------
# P3 (niche N2) -- pointwise 2-D obliquity for the displaced screen.
#
# The meridional cosine LUT above assumes rotational symmetry (a 1-D fan indexed
# by crossing radius).  Decentered / tilted / freeform elements break that
# symmetry, so P3 adds a 2-D generalisation: trace a 2-D ray GRID launched along
# the input congruence through the actual (possibly asymmetric) surfaces and
# interpolate the per-surface z-axis ray cosines ``(cos_alpha_in,
# cos_alpha_out)`` onto the field grid at each ray's CROSSING position.  The
# obliquity OPD is the SAME equation (1) --
# ``(n2 cos_alpha_out - n1 cos_alpha_in) * sag`` -- so on a rotationally-
# symmetric element the 2-D path reproduces the meridional LUT (validated to
# <0.1%, the convention-bug killer).  Decenter enters as ``sag(x - dx, y - dy)``;
# small-angle tilt as a rotated normal frame (linear sag ramp ``tx*x + ty*y``
# plus the correspondingly-tilted surface normal); freeform via a per-surface
# ``sag_callable(x, y)`` hook.  Auto-selected for asymmetric elements (the LUT
# stays the fast path for symmetric ones).  See
# docs/audit_real_lens_displaced_2026_07_19.md (P3 / N2).
# ---------------------------------------------------------------------------

def _displaced_carrier_dir_fn(conjugate, E_in, wavelength, dx, dy, Nx, Ny):
    """Return a callable ``(x0, y0) -> (gx, gy)`` giving the 2-D launch slopes
    (transverse gradient of the carrier eikonal ``W``) of the input CONGRUENCE
    for the pointwise 2-D obliquity trace.

    The 2-D analogue of :func:`_displaced_carrier_slope_fn` (which returns only
    the meridional slope ``dW/dy`` on the central column).  ``None`` for a
    collimated congruence (axial launch, byte-consistent with the meridional
    ``carrier_slope=None`` fan)."""
    if conjugate is None:
        return None
    if isinstance(conjugate, (int, float)) and not isinstance(conjugate, bool):
        s = float(conjugate)
        if not np.isfinite(s):
            return None
        if s == 0.0:
            raise ValueError(
                "apply_real_lens: surface_model='displaced' conjugate distance "
                "must be non-zero (0 is the source's own focus).")

        def _scalar_dir(x0, y0):
            return (np.asarray(x0, dtype=np.float64) / s,
                    np.asarray(y0, dtype=np.float64) / s)

        return _scalar_dir

    from ._lens_traced import _compute_carrier
    xax = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    yax = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    Xg, Yg = np.meshgrid(xax, yax)
    # ``dy`` is forwarded: this entry point supports anamorphic grids and
    # the ndarray-carrier branch differentiates and samples per axis
    # (VERIFY-A3 OI-10).  Bit-identical when dy == dx.
    _, grad_fn, _ = _compute_carrier(conjugate, E_in, wavelength, dx, Xg,
                                     Yg, dy=dy)

    def _auto_dir(x0, y0):
        L, M = grad_fn(np.asarray(x0, dtype=np.float64),
                       np.asarray(y0, dtype=np.float64))
        return np.asarray(L, dtype=np.float64), np.asarray(M, dtype=np.float64)

    return _auto_dir


def _disp_surface_z_grad(surf, x, y):
    """Surface z-departure ``f(x, y)`` [m] from the vertex plane and its
    transverse gradient ``(df/dx, df/dy)`` on the FIELD-frame coordinates
    ``(x, y)`` [m], honouring per-surface ``decenter=(dx, dy)`` (evaluate at
    ``x - dx, y - dy``), small-angle ``tilt=(tx, ty)`` (a linear ramp
    ``tx*(x-dx) + ty*(y-dy)`` plus the tilted normal), and a freeform
    ``sag_callable(xs, ys)`` hook.  ``f`` is NaN where the conic is undefined
    (the caller masks those rays).  Gradients by central finite difference
    (matching the meridional fan's convention).  Used by the pointwise 2-D
    obliquity trace only; wave-model-independent geometry."""
    dec = surf.get('decenter') or (0.0, 0.0)
    tl = surf.get('tilt') or (0.0, 0.0)
    dcx, dcy = float(dec[0]), float(dec[1])
    tx, ty = float(tl[0]), float(tl[1])
    xs = np.asarray(x, dtype=np.float64) - dcx
    ys = np.asarray(y, dtype=np.float64) - dcy
    cb = surf.get('sag_callable')
    if cb is not None:
        # Freeform hook -- FD gradient at a fixed sub-micron step (freeform
        # callables are assumed smooth at this scale).
        step = 1.0e-7
        f = np.asarray(cb(xs, ys), dtype=np.float64)
        dfdx = (np.asarray(cb(xs + step, ys), dtype=np.float64)
                - np.asarray(cb(xs - step, ys), dtype=np.float64)) / (2.0 * step)
        dfdy = (np.asarray(cb(xs, ys + step), dtype=np.float64)
                - np.asarray(cb(xs, ys - step), dtype=np.float64)) / (2.0 * step)
    else:
        R = surf['radius']
        kc = surf.get('conic', 0.0) or 0.0
        asph = surf.get('aspheric_coeffs')
        flat = (R == 0) or (not np.isfinite(R))
        r2 = xs * xs + ys * ys
        r = np.sqrt(r2)
        if flat:
            f = np.zeros_like(r)
            dfdx = np.zeros_like(r)
            dfdy = np.zeros_like(r)
        else:
            f = _surface_sag_general(r2, R, kc, asph)
            e = np.maximum(1e-9, 1e-6 * r)
            sp = _surface_sag_general((r + e) ** 2, R, kc, asph)
            sm = _surface_sag_general((r - e) ** 2, R, kc, asph)
            sp = np.where(np.isnan(sp), 0.0, sp)
            sm = np.where(np.isnan(sm), 0.0, sm)
            sagp = (sp - sm) / (2.0 * e)                    # d(sag)/dr
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_r = np.where(r > 0.0, 1.0 / r, 0.0)
            dfdx = sagp * xs * inv_r
            dfdy = sagp * ys * inv_r
    if tx != 0.0 or ty != 0.0:
        f = f + tx * xs + ty * ys
        dfdx = dfdx + tx
        dfdy = dfdy + ty
    return f, dfdx, dfdy


def _build_displaced_cos_grid(surfaces, thicknesses, wavelength, r_max,
                              Nx, Ny, dx, dy, dir_fn=None, n_launch=257,
                              n_coarse=384, interp_method='structured'):
    """Pointwise 2-D generalisation of :func:`_build_displaced_cos_luts` for
    decentered / tilted / freeform (callable-sag) elements.

    Trace a 2-D ray grid (regular square of half-extent ``r_max``, launched
    along the input congruence ``dir_fn``) through the actual surfaces and
    return, per surface, ``(cos_in_field, cos_out_field)`` -- the z-components
    of the ray direction just before / after refraction, interpolated onto the
    FIELD grid at each ray's crossing position.  These are the SAME cosines the
    meridional LUT stores, so equation (1) OPD ``(n2 cos_out - n1 cos_in) *
    sag`` on a rotationally-symmetric element reproduces the LUT path.
    Vectorised Newton intersection + vector Snell; wave-model-independent.

    ``interp_method`` (roadmap B5) selects how the per-surface cosines are
    resampled from the traced ray crossings onto the field grid:

    * ``'structured'`` (default) -- the launch fan is a STRUCTURED grid, so the
      smooth launch->crossing map is inverted by a few Newton steps (evaluating
      the crossing grids + their gradients with ``map_coordinates``) and the
      cos grids are then sampled at the inverted launch coordinates, again with
      ``map_coordinates``.  O(N^2) direct -- no triangulation build.
    * ``'delaunay'`` -- the legacy (pre-R1) scattered ``LinearNDInterpolator``
      (QHull Delaunay) path, byte-identical to v5.27.0, retained as the oracle
      the structured backend is validated against.
    """
    from scipy.interpolate import (
        LinearNDInterpolator,
        NearestNDInterpolator,
        RegularGridInterpolator,
    )
    from scipy.ndimage import distance_transform_edt, map_coordinates

    if interp_method not in ('structured', 'delaunay'):
        raise ValueError(
            "_build_displaced_cos_grid: interp_method must be 'structured' "
            f"or 'delaunay' (got {interp_method!r}).")
    nl = int(n_launch)
    ax = np.linspace(-r_max, r_max, nl)
    LX, LY = np.meshgrid(ax, ax)
    # B5: launch the FULL square ray grid (not just the inscribed disk) so the
    # ray fan stays a STRUCTURED grid -- the structured interp inverts the
    # smooth launch->crossing map with map_coordinates instead of triangulating
    # a scattered point cloud.  ``disk_flat`` selects the pupil for the legacy
    # Delaunay path; the extra corner rays are traced per-ray-independently, so
    # the disk subset is byte-identical to the former disk-only launch.
    disk_flat = ((LX * LX + LY * LY) <= (r_max * 1.0000001) ** 2).ravel()
    x0 = LX.ravel().astype(np.float64)
    y0 = LY.ravel().astype(np.float64)
    n = x0.size
    if dir_fn is None:
        gx = np.zeros(n)
        gy = np.zeros(n)
    else:
        gx, gy = dir_fn(x0, y0)
        gx = np.where(np.isfinite(gx), gx, 0.0).astype(np.float64).reshape(n)
        gy = np.where(np.isfinite(gy), gy, 0.0).astype(np.float64).reshape(n)
    nrm = np.sqrt(1.0 + gx * gx + gy * gy)
    dxr = gx / nrm
    dyr = gy / nrm
    dzr = 1.0 / nrm
    px = x0.copy()
    py = y0.copy()
    pz = np.zeros(n)
    alive = np.ones(n, dtype=bool)

    xax = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    yax = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    # The per-surface obliquity cosines vary smoothly across the aperture, so
    # the crossing->grid interpolation is done on a COARSE regular grid
    # (bounded resolution) and bilinearly upsampled to the full field grid --
    # decoupling the cost from N.  ``n_coarse`` is the sample count across the
    # TRACED APERTURE, not across the window: the coarse grid spans the whole
    # field extent (it must, so the upsample has no extrapolation), so a padded
    # grid spreads a fixed count ever more thinly over the pupil and the
    # accuracy falls LINEARLY with the pad factor (measured 24x worse over a 16x
    # pad, which is a numerical artefact of the padding rather than of anything
    # physical).  Scaling the count by the pad factor keeps the pitch inside the
    # pupil fixed at the unpadded value; ``min(Nx, ...)`` still caps it at the
    # field grid itself, where the upsample is a no-op.
    _pad_x = ((Nx * dx) / (2.0 * r_max)
              if (r_max and np.isfinite(r_max) and r_max > 0.0) else 1.0)
    _pad_y = ((Ny * dy) / (2.0 * r_max)
              if (r_max and np.isfinite(r_max) and r_max > 0.0) else 1.0)
    _ncx = min(Nx, max(n_coarse, int(np.ceil(n_coarse * max(_pad_x, 1.0)))))
    _ncy = min(Ny, max(n_coarse, int(np.ceil(n_coarse * max(_pad_y, 1.0)))))
    xcoarse = np.linspace(xax[0], xax[-1], _ncx)
    ycoarse = np.linspace(yax[0], yax[-1], _ncy)
    Xc, Yc = np.meshgrid(xcoarse, ycoarse)
    Xg, Yg = np.meshgrid(xax, yax)
    # launch-grid physical spacing -> fractional-index scale for map_coordinates
    _du = (2.0 * r_max) / (nl - 1) if nl > 1 else 1.0

    idx = []
    for s in surfaces:
        idx.append((float(get_glass_index(s['glass_before'], wavelength)),
                    float(get_glass_index(s['glass_after'], wavelength))))

    def _upsample_coarse(ci_c, co_c):
        """Bilinearly upsample the coarse cos maps to the full field grid
        (shared by both interp backends).  A no-op when the coarse grid IS the
        field grid (small N)."""
        if _ncx == Nx and _ncy == Ny:
            return ci_c, co_c
        rgi_i = RegularGridInterpolator(
            (ycoarse, xcoarse), ci_c, method='linear',
            bounds_error=False, fill_value=None)
        rgi_o = RegularGridInterpolator(
            (ycoarse, xcoarse), co_c, method='linear',
            bounds_error=False, fill_value=None)
        pq = np.stack([Yg.ravel(), Xg.ravel()], axis=-1)
        ci = rgi_i(pq).reshape(Xg.shape)
        co = rgi_o(pq).reshape(Xg.shape)
        return ci, co

    def _interp2_delaunay(pts, cin, cout):
        """LEGACY (pre-R1) scattered Delaunay interpolation of BOTH cosines onto
        the coarse grid.  ONE ``LinearNDInterpolator`` (2-column value array) is
        queried on the coarse grid; out-of-hull NaNs are nearest-filled; the
        coarse maps are upsampled to the field grid.  Byte-identical to the
        v5.27.0 path -- retained as the structured backend's validation oracle
        and reachable via ``interp_method='delaunay'``."""
        if pts.shape[0] < 4:
            ci = float(np.mean(cin)) if cin.size else 1.0
            co = float(np.mean(cout)) if cout.size else 1.0
            return (np.full(Xg.shape, ci), np.full(Xg.shape, co))
        vals = np.column_stack([cin, cout])
        q = LinearNDInterpolator(pts, vals)(Xc, Yc)
        ci_c = np.ascontiguousarray(q[..., 0])
        co_c = np.ascontiguousarray(q[..., 1])
        nan = np.isnan(ci_c)
        if bool(nan.any()):
            qn = NearestNDInterpolator(pts, vals)(Xc[nan], Yc[nan])
            ci_c[nan] = qn[:, 0]
            co_c[nan] = qn[:, 1]
        return _upsample_coarse(ci_c, co_c)

    def _interp2_structured(PX, PY, CIN, COUT, VALID):
        """B5 STRUCTURED-grid interpolation (no triangulation build).

        The launch grid is regular, so ``(PX, PY)`` -- the ray crossing
        positions as functions of launch coordinate ``(u, v)`` -- is a smooth
        curvilinear grid.  Invert it (Newton, evaluating ``PX/PY`` and their
        gradients with ``map_coordinates``) to find, for each coarse field
        point, the launch coordinate whose ray crosses there; then sample the
        cos grids at that launch coordinate with ``map_coordinates``.  Dead /
        TIR launch cells are filled by a structured nearest (EDT) fill first so
        the map is finite everywhere; field points outside the ray-crossing
        coverage clamp to the pupil edge (~the Delaunay nearest-fill)."""
        invalid = ~VALID
        if bool(invalid.any()):
            if not bool(VALID.any()):
                return _upsample_coarse(np.ones(Xc.shape), np.ones(Xc.shape))
            fi = tuple(distance_transform_edt(
                invalid, return_distances=False, return_indices=True))
            PX = PX[fi]
            PY = PY[fi]
            CIN = CIN[fi]
            COUT = COUT[fi]
        dPX_dv, dPX_du = np.gradient(PX, _du, _du)
        dPY_dv, dPY_du = np.gradient(PY, _du, _du)
        Xt = Xc.ravel()
        Yt = Yc.ravel()
        u = Xt.copy()
        v = Yt.copy()
        for _ in range(8):
            crd = np.stack([(v + r_max) / _du, (u + r_max) / _du])
            rx = Xt - map_coordinates(PX, crd, order=1, mode='nearest')
            ry = Yt - map_coordinates(PY, crd, order=1, mode='nearest')
            a = map_coordinates(dPX_du, crd, order=1, mode='nearest')
            b = map_coordinates(dPX_dv, crd, order=1, mode='nearest')
            c = map_coordinates(dPY_du, crd, order=1, mode='nearest')
            d = map_coordinates(dPY_dv, crd, order=1, mode='nearest')
            det = a * d - b * c
            det = np.where(np.abs(det) < 1e-30, 1e-30, det)
            u = np.clip(u + (d * rx - b * ry) / det, -r_max, r_max)
            v = np.clip(v + (-c * rx + a * ry) / det, -r_max, r_max)
        crd = np.stack([(v + r_max) / _du, (u + r_max) / _du])
        ci_c = np.ascontiguousarray(
            map_coordinates(CIN, crd, order=1, mode='nearest').reshape(Xc.shape))
        co_c = np.ascontiguousarray(
            map_coordinates(COUT, crd, order=1, mode='nearest').reshape(Xc.shape))
        return _upsample_coarse(ci_c, co_c)

    z_v = 0.0
    cos_grids = []
    n_surf = len(surfaces)
    for i, s in enumerate(surfaces):
        n1, n2 = idx[i]
        R = s['radius']
        flat = ((R == 0) or (not np.isfinite(R))) and (
            s.get('sag_callable') is None
            and (s.get('tilt') or (0.0, 0.0)) == (0.0, 0.0))
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (z_v - pz) / dzr
        if flat:
            px = px + t * dxr
            py = py + t * dyr
            pz = pz + t * dzr
            nxc = np.zeros(n)
            nyc = np.zeros(n)
            nzc = np.ones(n)
        else:
            for _ in range(24):
                xq = px + t * dxr
                yq = py + t * dyr
                f, dfdx, dfdy = _disp_surface_z_grad(s, xq, yq)
                f = np.where(np.isnan(f), 0.0, f)
                dfdx = np.where(np.isnan(dfdx), 0.0, dfdx)
                dfdy = np.where(np.isnan(dfdy), 0.0, dfdy)
                g = pz + t * dzr - z_v - f
                dgdt = dzr - (dfdx * dxr + dfdy * dyr)
                dgdt = np.where(np.abs(dgdt) < 1e-30, 1e-30, dgdt)
                # Byte-identical early exit (not a tolerance): once ``t`` reaches a
                # BITWISE fixed point, every remaining sweep reproduces it exactly, so
                # the loop can stop without changing a single output bit.  The
                # intersection residual is exactly 0 after 2 sweeps on every fixture
                # measured, and each sweep costs three ``_surface_sag_general``
                # evaluations over the whole fan -- ~10x of the geometric traces.
                _t_new = t - g / dgdt
                _t_done = np.array_equal(_t_new, t, equal_nan=True)
                t = _t_new
                del _t_new
                if _t_done:
                    break
            px = px + t * dxr
            py = py + t * dyr
            pz = pz + t * dzr
            f, dfdx, dfdy = _disp_surface_z_grad(s, px, py)
            nzc = np.ones(n)
            nxc = -dfdx
            nyc = -dfdy
            nn = np.sqrt(nxc * nxc + nyc * nyc + nzc * nzc)
            nxc = nxc / nn
            nyc = nyc / nn
            nzc = nzc / nn
            alive = alive & np.isfinite(f)

        cos_in_z = dzr.copy()
        cos_i = dxr * nxc + dyr * nyc + dzr * nzc
        eta = n1 / n2
        sin2t = eta * eta * (1.0 - cos_i * cos_i)
        alive = (alive & np.isfinite(px) & np.isfinite(py)
                 & np.isfinite(cos_i) & (sin2t <= 1.0))
        cos_t = np.sqrt(np.maximum(1.0 - sin2t, 0.0))
        ndx = eta * dxr + (cos_t - eta * cos_i) * nxc
        ndy = eta * dyr + (cos_t - eta * cos_i) * nyc
        ndz = eta * dzr + (cos_t - eta * cos_i) * nzc
        nn2 = np.sqrt(ndx * ndx + ndy * ndy + ndz * ndz)
        nn2 = np.where(nn2 == 0.0, 1.0, nn2)
        dxr = ndx / nn2
        dyr = ndy / nn2
        dzr = ndz / nn2
        cos_out_z = dzr.copy()

        if interp_method == 'delaunay':
            m = alive & disk_flat
            pts = np.column_stack([px[m], py[m]])
            ci, co = _interp2_delaunay(pts, cos_in_z[m], cos_out_z[m])
        else:
            valid = (alive & np.isfinite(px) & np.isfinite(py)
                     & np.isfinite(cos_in_z) & np.isfinite(cos_out_z))
            ci, co = _interp2_structured(
                px.reshape(nl, nl), py.reshape(nl, nl),
                cos_in_z.reshape(nl, nl), cos_out_z.reshape(nl, nl),
                valid.reshape(nl, nl))
        cos_grids.append((ci, co))

        if i < n_surf - 1:
            z_v += thicknesses[i]
    return cos_grids


# ---------------------------------------------------------------------------
# Pointwise cos-grid cache (roadmap B1) -- the per-surface 2-D obliquity
# cos-grid is FIELD-INDEPENDENT given (prescription surfaces + thicknesses +
# wavelength + fan extent + scalar/collimated conjugate + grid dx,N), so a
# decentered-design iteration loop that only moves the field re-uses the
# ~3.9 s Delaunay/structured trace instead of rebuilding it every call (K3).
#
# A cos-grid PAIR is ~2 * N^2 * 8 B = 16 MB @ N=1024, ~1 GB @ N=8192, so this
# is an N^2-scale cache: it MUST be byte-budgeted + OPT-IN per the Section 0
# contract.  It ships on the shared :class:`ByteBudgetedLRU` with
# ``max_bytes=0`` (DISABLED -- stores nothing) so the default is off; the
# caller enables it for a design loop with
# :func:`set_pointwise_cos_grid_cache_budget`.  The instance auto-enrolls in
# the byte-budgeted registry, so ``clear_asm_caches()`` drains it and
# ``cache_report()`` shows its footprint.  Only the FIELD-INDEPENDENT
# congruences (collimated / scalar conjugate) are cacheable; the 'auto' /
# ndarray carriers depend on E_in and always rebuild (key is None).
# ---------------------------------------------------------------------------
from ..cache import ByteBudgetedLRU as _ByteBudgetedLRU  # noqa: E402

_DISPLACED_COS_GRID_CACHE = _ByteBudgetedLRU(
    'displaced_cos_grid', max_bytes=0)      # OPT-IN: off by default


def set_pointwise_cos_grid_cache_budget(mb):
    """Enable / size the OPT-IN pointwise cos-grid cache (roadmap B1).

    The pointwise 2-D obliquity path (``surface_model='displaced'`` with
    ``displaced_obliquity='pointwise'`` on a decentered / tilted / freeform
    element) traces a 2-D ray grid to build the per-surface obliquity
    cos-grid.  That grid is FIELD-INDEPENDENT given the prescription +
    conjugate + wavelength + grid, so a design loop that only moves the field
    can re-use it.  This cache is an **N^2-scale** cache (a cos-grid pair is
    ~16 MB at N=1024, ~1 GB at N=8192) and therefore ships **off by default**;
    call this to opt in for a design loop.

    Parameters
    ----------
    mb : float or None
        * ``0`` -- DISABLE (the default state) and clear the cache.
        * ``None`` -- enable, bounded only by the collective global cache
          ceiling (``LUMENAIRY_CACHE_BUDGET_MB`` / ``set_cache_budget``).
        * ``> 0`` -- enable with a fixed local ceiling of ``mb`` megabytes
          (still also bounded by the collective global ceiling).

    Notes
    -----
    LRU eviction (byte-budgeted): once the retained bytes exceed the budget
    the least-recently-used entry is dropped, so a loop that re-uses one
    design keeps its hot entry.  ``cache_report()`` shows the live footprint;
    ``clear_asm_caches()`` / :func:`clear_pointwise_cos_grid_cache` release it.
    """
    if mb is None:
        _DISPLACED_COS_GRID_CACHE.set_budget(None)
        return
    mb = float(mb)
    if mb < 0:
        raise ValueError(
            "set_pointwise_cos_grid_cache_budget: mb must be >= 0 or None "
            f"(got {mb!r}); 0 disables, None binds to the global budget.")
    _DISPLACED_COS_GRID_CACHE.set_budget(int(mb * 1024 * 1024))


def get_pointwise_cos_grid_cache_budget():
    """Return the pointwise cos-grid cache's LOCAL byte ceiling.

    ``0`` -> disabled (the default); ``None`` -> bound only by the collective
    global budget; a positive int -> the local ceiling in bytes.  Mirrors
    :func:`set_pointwise_cos_grid_cache_budget` (which takes megabytes)."""
    return _DISPLACED_COS_GRID_CACHE.max_bytes


def _get_pointwise_cos_grid_cache_budget_bytes():
    """The cos-grid cache's local byte ceiling, for the knob registry.

    Registered instead of :func:`get_pointwise_cos_grid_cache_budget` +
    :func:`set_pointwise_cos_grid_cache_budget` because those two are not a
    round trip: the setter takes MEGABYTES and the getter returns BYTES, so
    ``set(get())`` would inflate the budget by 2**20 every time the test
    fixture restored it.  Bytes in, bytes out, no conversion, exact -- and
    ``None`` (bound only by the collective global budget) survives as ``None``.
    """
    return _DISPLACED_COS_GRID_CACHE.max_bytes


def _set_pointwise_cos_grid_cache_budget_bytes(max_bytes):
    """Apply a byte ceiling produced by
    :func:`_get_pointwise_cos_grid_cache_budget_bytes`."""
    _DISPLACED_COS_GRID_CACHE.set_budget(max_bytes)


_register_knob(
    'pointwise_cos_grid_cache_budget',
    getter=_get_pointwise_cos_grid_cache_budget_bytes,
    setter=_set_pointwise_cos_grid_cache_budget_bytes,
    doc="Local byte ceiling of the OPT-IN pointwise cos-grid cache: 0 "
        "(default) disables it, None binds it to the collective global cache "
        "budget, a positive int caps it.  Set in MEGABYTES through the public "
        "set_pointwise_cos_grid_cache_budget().")


def clear_pointwise_cos_grid_cache():
    """Drop every cached pointwise cos-grid, releasing its retained bytes.

    Registered (via the shared byte-budgeted registry) so
    :func:`lumenairy.clear_asm_caches` drains it too; does NOT change the
    enabled/budget state (a subsequent call re-populates it)."""
    _DISPLACED_COS_GRID_CACHE.clear()


def _displaced_cos_grid_key(surfaces, thicknesses, wavelength, r_max,
                            conjugate, Nx, Ny, dx, dy, n_launch, n_coarse,
                            interp_method):
    """Hashable COMPLETE key for the field-independent pointwise cos-grid
    (roadmap Section 0 -- prescription + conjugate + wavelength + grid dx,N,
    plus the fan/interp determinants).  Returns ``None`` for the
    field-DEPENDENT 'auto' / ndarray congruences (they depend on E_in and are
    never cached).  A freeform ``sag_callable`` is keyed by object identity
    (held in the key so it cannot be GC'd out from under the entry) AND by a
    VALUE fingerprint (:func:`_sag_callable_fingerprint`), so a fresh callable
    still misses and a MUTATED one misses too instead of returning a stale
    grid.  Glasses enter by RESOLVED INDEX rather than registry name (see
    :func:`_glass_key_value`)."""
    if not (conjugate is None
            or (isinstance(conjugate, (int, float))
                and not isinstance(conjugate, bool))):
        return None
    if conjugate is not None and not np.isfinite(float(conjugate)):
        conjugate = None                       # +-inf == collimated
    surf_key = tuple((
        float(s.get('radius', np.inf))
        if np.isfinite(s.get('radius', np.inf)) else np.inf,
        float(s.get('conic', 0.0) or 0.0),
        (tuple(sorted((int(p), float(a))
                      for p, a in s['aspheric_coeffs'].items()))
         if s.get('aspheric_coeffs') else None),
        tuple(float(v) for v in (s.get('decenter') or (0.0, 0.0))),
        tuple(float(v) for v in (s.get('tilt') or (0.0, 0.0))),
        s.get('sag_callable'),                 # by identity (held -> no GC)
        _sag_callable_fingerprint(s.get('sag_callable'), r_max),
        _glass_key_value(s.get('glass_before'), wavelength),
        _glass_key_value(s.get('glass_after'), wavelength))
        for s in surfaces)
    if any(sk[6] == 'unprobeable' for sk in surf_key):
        # A callable the fingerprint could not evaluate has no VALUE identity,
        # so identity keying would be the stale-hit hazard again: refuse to
        # cache rather than risk it.
        return None
    conj_key = None if conjugate is None else float(conjugate)
    return (surf_key, tuple(float(t) for t in thicknesses),
            float(wavelength), float(r_max), conj_key,
            int(Nx), int(Ny), float(dx), float(dy),
            int(n_launch), int(n_coarse), str(interp_method))


def _get_displaced_cos_grid(surfaces, thicknesses, wavelength, r_max,
                            Nx, Ny, dx, dy, dir_fn, conjugate,
                            n_launch=257, n_coarse=384,
                            interp_method='structured'):
    """Build (or fetch from the opt-in byte-budgeted cache) the per-surface
    pointwise cos-grid.  Field-independent congruences hit the cache when it is
    enabled; everything else (and the disabled default) rebuilds.  Pure
    memoization -- a cache hit returns the SAME arrays the cold trace produced,
    so downstream output is byte-identical."""
    key = _displaced_cos_grid_key(
        surfaces, thicknesses, wavelength, r_max, conjugate, Nx, Ny, dx, dy,
        n_launch, n_coarse, interp_method)
    if key is not None:
        hit = _DISPLACED_COS_GRID_CACHE.get(key)
        if hit is not None:
            return hit
    grids = _build_displaced_cos_grid(
        surfaces, thicknesses, wavelength, r_max, Nx, Ny, dx, dy,
        dir_fn=dir_fn, n_launch=n_launch, n_coarse=n_coarse,
        interp_method=interp_method)
    if key is not None:
        _DISPLACED_COS_GRID_CACHE.put(key, grids)
    return grids


def _element_is_asymmetric(surfaces):
    """True when any surface carries a non-zero decenter / tilt or a freeform
    ``sag_callable`` hook -- i.e. the meridional (rotationally-symmetric) fan is
    no longer valid and the pointwise 2-D obliquity path is required."""
    for s in surfaces or []:
        if not isinstance(s, dict):
            continue
        dec = s.get('decenter') or (0.0, 0.0)
        tl = s.get('tilt') or (0.0, 0.0)
        if tuple(float(v) for v in dec) != (0.0, 0.0):
            return True
        if tuple(float(v) for v in tl) != (0.0, 0.0):
            return True
        if s.get('sag_callable') is not None:
            return True
    return False


def _routes_to_displaced_remap_2d(surface_model, displaced_mode,
                                  displaced_obliquity, surfaces):
    """True when this keyword set routes to the 2-D transverse-walk remap
    (:func:`_apply_displaced_remap_2d`).

    The 2-D remap takes an ASYMMETRIC element under
    ``surface_model='displaced'``, either explicitly (``displaced_mode=
    'remap'``) or as the default for that element (``'screen'`` + the ``auto``
    obliquity, which an explicit ``displaced_obliquity='pointwise'`` overrides
    in favour of the single-plane screen).  One predicate so the guard that
    refuses a discarded ``displaced_n_side`` and the dispatch that consumes it
    cannot drift apart."""
    return (surface_model == 'displaced'
            and _element_is_asymmetric(surfaces)
            and (displaced_mode == 'remap'
                 or (displaced_mode == 'screen'
                     and displaced_obliquity == 'auto')))


_VALID_DISPLACED_OBLIQUITY = ('auto', 'meridional', 'pointwise')


def _resolve_displaced_obliquity(displaced_obliquity, surfaces):
    """Resolve the ``displaced_obliquity`` selector to the concrete path used:
    ``'meridional'`` (the fast 1-D cosine LUT) or ``'pointwise'`` (the 2-D ray
    grid).  ``'auto'`` (default) picks pointwise for asymmetric elements and
    keeps the byte-identical meridional LUT for symmetric ones."""
    if displaced_obliquity == 'pointwise':
        return 'pointwise'
    if displaced_obliquity == 'meridional':
        return 'meridional'
    if displaced_obliquity == 'auto':
        return ('pointwise' if _element_is_asymmetric(surfaces)
                else 'meridional')
    raise ValueError(
        f"apply_real_lens: unknown displaced_obliquity "
        f"{displaced_obliquity!r}.  Valid choices: "
        f"{sorted(_VALID_DISPLACED_OBLIQUITY)}.")


# ---------------------------------------------------------------------------
# Extreme-conjugate displaced sub-models (P2 / niche N1) -- opt-in experimental
# ``displaced_mode`` variants of ``surface_model='displaced'``.  The default
# ``'screen'`` (the per-surface obliquity screen + in-glass ASM) is unchanged
# and byte-identical.  These candidates were built + measured against the
# congruence-fixed diffraction oracle (validation/oracles/debye_oracle_v3.py):
#   * 'remap' -- the exit-plane geometric-transfer remap (candidate a);
#   * 'split' -- entrance/exit screens + reduced-distance (t/n) air propagation
#                per gap (candidate b).
# KEY MEASURED RESULT (docs/audit_real_lens_displaced_2026_07_19.md, P2): the
# default 'screen' is ALREADY within ~4-8% of the diffraction-faithful oracle on
# every extreme case (M5 real 0.96x, M5 virtual 1.00x, M1 doublet 0.92x); the
# prior "~0.50x floor" was measured against the GEOMETRIC ray-density spot, which
# over-estimates the true wave spot by ~2x near these reconvergence caustics, and
# was compounded by grid truncation (the large beams were run at < 2.4 w0
# half-width).  'remap' and 'split' MATCH 'screen' to within a few percent; they
# are exposed as documented experimental peers, not a default change.
# ---------------------------------------------------------------------------
_VALID_DISPLACED_MODES = ('screen', 'remap', 'split')


def _displaced_eikonal_fn(conjugate, E_in, wavelength, dx, dy, Nx, Ny):
    """Return a callable ``heights -> W_in`` giving the entrance-plane carrier
    eikonal ``W(0, h)`` [m] for ``displaced_mode='remap'``.

    The exit phase of the geometric-transfer remap must be referenced to the
    INPUT wavefront (cf. hammer H6): the total exit OPL is the entrance eikonal
    ``W_in(h)`` plus the per-segment lens OPL.  Mirrors the ``conjugate``
    vocabulary of :func:`_displaced_carrier_slope_fn`:

    * ``None`` / ``+-inf`` -> collimated: returns ``None`` (``W_in = 0``);
    * ``float`` signed conjugate ``s`` -> ``W_in(h) = h**2 / (2 s)`` (paraxial
      spherical carrier, consistent with the ``h/s`` launch slope);
    * ``'auto'`` / ``ndarray`` -> the carrier eikonal from ``_compute_carrier``
      evaluated along the meridian.
    """
    if conjugate is None:
        return None
    if isinstance(conjugate, (int, float)) and not isinstance(conjugate, bool):
        s = float(conjugate)
        if not np.isfinite(s):
            return None

        def _scalar_eik(h):
            h = np.asarray(h, dtype=np.float64)
            return h * h / (2.0 * s)

        return _scalar_eik

    from ._lens_traced import _compute_carrier
    xax = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    yax = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    Xg, Yg = np.meshgrid(xax, yax)
    _, _, w_fn = _compute_carrier(conjugate, E_in, wavelength, dx, Xg, Yg,
                                  dy=dy)

    def _carrier_eik(h):
        h = np.asarray(h, dtype=np.float64)
        return np.asarray(w_fn(np.zeros_like(h), h), dtype=np.float64)

    return _carrier_eik


def _displaced_carrier_dir_eik_fn(conjugate, E_in, wavelength, dx, dy, Nx, Ny):
    """Return ``(dir_fn, eik_fn)`` for the P10 (niche N11) 2-D transverse-walk
    remap: ``dir_fn(x0, y0) -> (gx, gy)`` the 2-D launch slopes (transverse
    gradient of the carrier eikonal ``W``) and ``eik_fn(x0, y0) -> W_in`` the
    entrance-plane carrier eikonal [m] (referenced into the ray OPL, cf. hammer
    H6 -- omitting it collapses a diverging-input trace onto the collimated focal
    plane).  The 2-D off-axis analogue of the meridional
    :func:`_displaced_carrier_slope_fn` (dir) + :func:`_displaced_eikonal_fn`
    (eik), built together so the ``'auto'``/ndarray carrier is fit only once.
    Mirrors the ``conjugate`` vocabulary:

    * ``None`` / ``+-inf`` -> collimated: ``(None, None)`` (axial launch, W=0);
    * ``float`` signed conjugate ``s`` -> ``gx = x0/s, gy = y0/s`` and
      ``W_in = (x0^2 + y0^2) / (2 s)`` (paraxial spherical carrier);
    * ``'auto'`` / ndarray -> the carrier gradient + eikonal from
      ``_compute_carrier``.
    """
    if conjugate is None:
        return None, None
    if isinstance(conjugate, (int, float)) and not isinstance(conjugate, bool):
        s = float(conjugate)
        if not np.isfinite(s):
            return None, None
        if s == 0.0:
            raise ValueError(
                "apply_real_lens: surface_model='displaced' conjugate distance "
                "must be non-zero (0 is the source's own focus).")

        def _dir(x0, y0):
            return (np.asarray(x0, dtype=np.float64) / s,
                    np.asarray(y0, dtype=np.float64) / s)

        def _eik(x0, y0):
            x0 = np.asarray(x0, dtype=np.float64)
            y0 = np.asarray(y0, dtype=np.float64)
            return (x0 * x0 + y0 * y0) / (2.0 * s)

        return _dir, _eik

    from ._lens_traced import _compute_carrier
    xax = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    yax = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    Xg, Yg = np.meshgrid(xax, yax)
    _, grad_fn, w_fn = _compute_carrier(conjugate, E_in, wavelength, dx, Xg,
                                        Yg, dy=dy)

    def _dir(x0, y0):
        L, M = grad_fn(np.asarray(x0, dtype=np.float64),
                       np.asarray(y0, dtype=np.float64))
        return np.asarray(L, dtype=np.float64), np.asarray(M, dtype=np.float64)

    def _eik(x0, y0):
        return np.asarray(w_fn(np.asarray(x0, dtype=np.float64),
                               np.asarray(y0, dtype=np.float64)),
                          dtype=np.float64)

    return _dir, _eik


def _build_displaced_ray_map(surfaces, thicknesses, wavelength, r_max,
                             n_fan=1025, carrier_slope=None, eikonal_fn=None):
    """Trace a meridional fan along the input congruence through the element and
    return the ENTRANCE->EXIT geometric ray map for ``displaced_mode='remap'``.

    Returns ``(h_in, h_out, opl)`` float64 arrays over the rays that survive to
    the exit vertex plane: ``h_in`` the entrance height, ``h_out`` the ray
    height at ``z = sum(thicknesses)`` (the same exit vertex plane the screen
    loop ends on), and ``opl`` the total optical path (``eikonal_fn`` entrance
    eikonal + per-segment ``n * path``), so the exit phase is ``k0 * opl``.
    Pure geometric trace (vectorised Newton intersection + vector Snell),
    identical geometry to :func:`_build_displaced_cos_luts`; wave-independent.
    """
    heights = np.linspace(r_max / n_fan, r_max, int(n_fan))
    idx = []
    for s in surfaces:
        n1 = float(get_glass_index(s['glass_before'], wavelength))
        n2 = float(get_glass_index(s['glass_after'], wavelength))
        idx.append((n1, n2))
    nf = heights.size
    pz = np.zeros(nf)
    py = heights.astype(np.float64).copy()
    if carrier_slope is None:
        dz = np.ones(nf)
        dy = np.zeros(nf)
    else:
        g = np.asarray(carrier_slope(heights), dtype=np.float64).reshape(nf)
        g = np.where(np.isfinite(g), g, 0.0)
        nrm = np.sqrt(1.0 + g * g)
        dz = 1.0 / nrm
        dy = g / nrm
    if eikonal_fn is None:
        opl = np.zeros(nf)
    else:
        opl = np.asarray(eikonal_fn(heights), dtype=np.float64).reshape(nf).copy()
        opl = np.where(np.isfinite(opl), opl, 0.0)
    alive = np.ones(nf, dtype=bool)
    z_v = 0.0
    for i, s in enumerate(surfaces):
        R = s['radius']
        kc = s.get('conic', 0.0) or 0.0
        asph = s.get('aspheric_coeffs')
        n1, n2 = idx[i]
        flat = (R == 0) or (not np.isfinite(R))
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (z_v - pz) / dz
        if flat:
            pz = pz + t * dz
            py = py + t * dy
            nrm_z = np.ones(nf)
            nrm_y = np.zeros(nf)
        else:
            for _ in range(24):
                y = py + t * dy
                r = np.abs(y)
                sag = _surface_sag_general(r * r, R, kc, asph)
                sag = np.where(np.isnan(sag), 0.0, sag)
                e = np.maximum(1e-9, 1e-6 * r)
                sp = _surface_sag_general((r + e) ** 2, R, kc, asph)
                sm = _surface_sag_general((r - e) ** 2, R, kc, asph)
                sp = np.where(np.isnan(sp), 0.0, sp)
                sm = np.where(np.isnan(sm), 0.0, sm)
                sagp = (sp - sm) / (2.0 * e)
                gg = pz + t * dz - z_v - sag
                dgdt = dz - sagp * np.sign(y) * dy
                dgdt = np.where(np.abs(dgdt) < 1e-30, 1e-30, dgdt)
                # Byte-identical early exit (not a tolerance): once ``t`` reaches a
                # BITWISE fixed point, every remaining sweep reproduces it exactly, so
                # the loop can stop without changing a single output bit.  The
                # intersection residual is exactly 0 after 2 sweeps on every fixture
                # measured, and each sweep costs three ``_surface_sag_general``
                # evaluations over the whole fan -- ~10x of the geometric traces.
                _t_new = t - gg / dgdt
                _t_done = np.array_equal(_t_new, t, equal_nan=True)
                t = _t_new
                del _t_new
                if _t_done:
                    break
            pz = pz + t * dz
            py = py + t * dy
            y = py
            r = np.abs(y)
            e = np.maximum(1e-9, 1e-6 * r)
            sp = _surface_sag_general((r + e) ** 2, R, kc, asph)
            sm = _surface_sag_general((r - e) ** 2, R, kc, asph)
            sp = np.where(np.isnan(sp), 0.0, sp)
            sm = np.where(np.isnan(sm), 0.0, sm)
            sagp = (sp - sm) / (2.0 * e)
            nz = np.ones(nf)
            ny = -sagp * np.sign(y)
            nn = np.hypot(nz, ny)
            nrm_z = nz / nn
            nrm_y = ny / nn
        # OPL segment: the unit-direction parametric step ``t`` equals the
        # geometric path length; add it in the medium BEFORE this surface.
        opl = opl + n1 * t
        cos_i = dz * nrm_z + dy * nrm_y
        eta = n1 / n2
        sin2t = eta * eta * (1.0 - cos_i * cos_i)
        alive = alive & np.isfinite(py) & (sin2t <= 1.0)
        cos_t = np.sqrt(np.maximum(1.0 - sin2t, 0.0))
        ndz = eta * dz + (cos_t - eta * cos_i) * nrm_z
        ndy = eta * dy + (cos_t - eta * cos_i) * nrm_y
        nn2 = np.hypot(ndz, ndy)
        nn2 = np.where(nn2 == 0.0, 1.0, nn2)
        dz = ndz / nn2
        dy = ndy / nn2
        if i < len(surfaces) - 1:
            z_v += thicknesses[i]
    z_exit = float(sum(thicknesses))
    with np.errstate(divide='ignore', invalid='ignore'):
        t_f = (z_exit - pz) / dz
    # The referencing leg from the last surface's sag back to the exit vertex
    # plane is travelled in ``surfaces[-1]['glass_after']``, which is not
    # necessarily air.  ``t_f`` is of order the last surface's sag, so
    # hard-coding n = 1 costs ``(n_after - 1) * |sag_last| / cos`` -- measured
    # 15.8 waves on an immersed-exit singlet.  ``idx[-1][1]`` is already
    # resolved above.
    opl = opl + idx[-1][1] * t_f
    h_out = py + t_f * dy
    m = alive & np.isfinite(h_out) & np.isfinite(opl) & np.isfinite(heights)
    return heights[m], h_out[m], opl[m]


def _residual_input_field(E_in, W_conj, wavelength):
    """``E_in`` demodulated by the congruence the geometric remap traced,
    ``F = E_in * exp(-i k0 W_conj)``.

    The remaps rebuild the exit phase from the RAY eikonal, which already
    carries the entrance eikonal of the ``conjugate=`` congruence.  Whatever
    phase the caller's field carries BEYOND that congruence -- an upstream
    element's residual wavefront, a tilt, aberration, or (with the default
    ``conjugate=None``) the whole of its curvature -- is not in the trace.
    Sampling ``np.abs(E_in)`` instead discards it: measured identical
    output (4.7e-16) for a flat and a 35-wave-defocused input, and a
    150 mm diverging source focusing at the COLLIMATED 21 mm instead of
    25 mm.

    ``F`` is smooth wherever the input matches the congruence to within a
    fraction of a wave per pixel, which is exactly the regime the remap is
    valid in, so it can be resampled by the same bilinear interpolation the
    amplitude used -- and it carries ``|E_in|`` in its modulus, so one complex
    resample replaces the old real one at no extra interpolation.  With
    ``W_conj = 0`` and a real non-negative ``E_in`` (the collimated,
    phase-free case) ``F`` is that amplitude exactly, so the legacy path is
    reproduced bit for bit."""
    E = np.asarray(E_in)
    if W_conj is None:
        return E.astype(np.complex128, copy=False) if np.iscomplexobj(E) \
            else E.astype(np.complex128)
    k0 = 2.0 * np.pi / wavelength
    W = np.asarray(W_conj, dtype=np.float64)
    W = np.where(np.isfinite(W), W, 0.0)
    return E * np.exp(-1j * k0 * W)


def _apply_displaced_remap(E_in, h_in, h_out, wavelength, dx, dy, opl,
                           eikonal_fn=None):
    """Candidate (a) exit-plane remap: turn the element into a geometric
    transfer ``h_in -> h_out`` with an energy-conserving amplitude Jacobian plus
    the exit-pupil-referenced eikonal OPD.

    Captures the transverse ray walk THROUGH the element (``h_out != h_in``)
    that a single fixed-plane screen cannot.  The input amplitude envelope
    ``|E_in|`` is warped from the entrance radius ``h_in`` to the exit radius
    ``h_out``; the exit phase is the ray eikonal ``k0 * opl`` (which carries the
    entrance-plane carrier eikonal) PLUS the input field's residual phase
    against that congruence, transported along the same rays (see
    :func:`_residual_input_field`).  Energy conservation:
    ``|E_out|^2 r_out dr_out = |E_in|^2 h_in dh_in``.  Rotationally symmetric
    (meridional-fan) model.  ``eikonal_fn(h) -> W_conj`` is the congruence the
    fan was launched along (``None`` = collimated); pass the SAME callable the
    ray map was built with.  Returns the exit-vertex-plane field (same
    reference as the default screen path)."""
    from scipy.ndimage import map_coordinates
    Ny, Nx = E_in.shape
    k0 = 2.0 * np.pi / wavelength
    order = np.argsort(h_out)
    ho = np.asarray(h_out)[order]
    hi = np.asarray(h_in)[order]
    op = np.asarray(opl)[order]
    keep = np.concatenate(([True], np.diff(ho) > 0))   # strictly increasing
    ho, hi, op = ho[keep], hi[keep], op[keep]
    if ho.size < 2:
        return np.zeros_like(E_in, dtype=np.complex128)
    x = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    y = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    r_out = np.sqrt(X * X + Y * Y)
    rc = np.clip(r_out, ho[0], ho[-1])
    hin_of = np.interp(rc, ho, hi)
    opl_of = np.interp(rc, ho, op)
    mp_fan = np.gradient(ho, hi)                        # dh_out / dh_in
    mp = np.interp(rc, ho, mp_fan)
    mp = np.where(mp <= 1e-12, 1e-12, mp)
    jac = np.sqrt(np.clip(hin_of, 0.0, None)
                  / (np.clip(rc, 1e-15, None) * mp))
    scale = np.where(r_out > 1e-15, hin_of / np.clip(r_out, 1e-15, None), 1.0)
    cx = (X * scale) / dx + Nx / 2.0
    cy = (Y * scale) / dy + Ny / 2.0
    # Resample the input field DEMODULATED by the traced congruence, so the
    # phase the caller's field carries beyond that congruence rides along with
    # the amplitude instead of being discarded.  ``|F| == |E_in|``, so this is
    # an amplitude resample plus the residual phase, not an extra pass.
    _W_in = None
    if eikonal_fn is not None:
        # ``r_out`` IS the input grid's radius here (the demodulation happens
        # on the input grid, before the resample).
        _W_in = np.asarray(eikonal_fn(r_out), dtype=np.float64)
    F = _residual_input_field(E_in, _W_in, wavelength)
    amp = (map_coordinates(F.real, [cy, cx], order=1,
                           mode='constant', cval=0.0)
           + 1j * map_coordinates(F.imag, [cy, cx], order=1,
                                  mode='constant', cval=0.0))
    E_out = amp * jac * np.exp(1j * k0 * (opl_of - float(op[0])))
    E_out = np.where(r_out <= ho[-1], E_out, 0.0)
    out_dtype = E_in.dtype if np.iscomplexobj(E_in) else np.complex128
    return E_out.astype(out_dtype)


# ---------------------------------------------------------------------------
# P10 (niche N11) -- 2-D transverse-walk remap for decentered / tilted /
# freeform elements.
#
# The P3 pointwise obliquity SCREEN imprints the refraction OPD
# ``(n2 cos_out - n1 cos_in) * sag(x-dx, y-dy)`` at the STRAIGHT-THROUGH grid
# position, so it captures the coma flare DIRECTION (centroid + skewness) but
# CANNOT represent the TRANSVERSE ray walk between a thick element's two surfaces
# -- the induced-coma spot therefore NARROWED ~0.91x where the geometric-spot
# oracle and ZOS both BROADEN ~1.02-1.03x (the plan N2 open finding).  N11
# generalises the P2 rotationally-symmetric exit-plane remap
# (:func:`_apply_displaced_remap`) to the full OFF-AXIS 2-D case: launch a 2-D
# (non-meridional) congruence fan against the decentered/tilted/freeform surface,
# build the exit ray map ``(x_out, y_out)(x_in, y_in)`` carrying the transverse
# walk, and remap the input amplitude envelope to the exit pupil with the
# energy-conserving 2-D Jacobian ``1/sqrt(|det d(x_out,y_out)/d(x_in,y_in)|)``
# plus the exit-pupil-referenced eikonal OPD.  This restores the walk-off the
# single-plane screen drops, so the analytic decentered spot BROADENS correctly.
# Honest metric = the RMS second-moment radius + common-mode-subtracted coma RMS
# (the decentered EE80 is diffraction-diluted, as P9 found for GBD): the on-axis
# RMS ~21 um MATCHES the GBD reference, the RMS broadens ~1.02 @1 mm (grid-robust),
# and the coma RMS matches the geom oracle within ~10% -- while the single-plane
# screen SHRINKS.  It is the DEFAULT for asymmetric elements (auto obliquity)
# and is also selectable via ``displaced_mode='remap'``; explicit
# ``displaced_obliquity='pointwise'`` keeps the single-plane screen (the
# documented walk-off-limited peer).  See docs/audit_real_lens_displaced_
# 2026_07_19.md (P10 / N11).
# ---------------------------------------------------------------------------

#: The 2-D transverse-walk remap's DEFAULT launch-lattice side, overridable
#: per call with ``apply_real_lens(displaced_n_side=...)``.
#:
#: Everything the exit field knows comes from ``n_side**2`` launched rays, so
#: the LAUNCH pitch -- not ``dx`` -- sets the transverse resolution of the
#: result: at 257 the pitch is 39 um across a 10 mm aperture whatever the field
#: sampling, and input structure finer than that (a hard stop edge, an
#: obscuration, an upstream DOE, speckle) is smoothed to the lattice.
#: :func:`_warn_if_remap_lattice_smooths` says so out loud whenever that is
#: true of the call in hand, and names the keyword that buys the resolution
#: back -- which no constant can, the bar being the CALLER's field pitch.
#:
#: Measured on the decentered f/5 singlet of
#: ``tests/unit/test_niche_p10_transverse_walk_remap.py`` (10 mm aperture, 5 mm
#: of N-BK7-like glass, 0.5-0.6 mm decenter), 2026-09-13, this build.  The
#: accuracy reference is a RAY-EXACT oracle: the same fan, but with each field
#: point's launch coordinate found by Newton on the TRUE trace instead of on
#: the lattice, so the oracle has no lattice at all.  Peak-relative exit-field
#: error over the illuminated core, with the trace cost and the field-grid
#: interpolation cost it feeds:
#:
#: ======  =========  =========  =========  =========  ==================
#: n_side  pitch um   |E| rms    phase rad  trace s    structured interp s
#: ======  =========  =========  =========  =========  ==================
#:    181      55.6    8.7e-04    4.7e-02      0.064   0.17 / 0.62 / 2.45
#:    257      39.1    4.3e-04    2.4e-02      0.152   0.17 / 0.64 / 2.36
#:    513      19.5    1.1e-04    5.9e-03      1.055   0.21 / 0.64 / 2.66
#:   1025       9.8    2.7e-05    1.4e-03      4.661   0.41 / 0.84 / 2.73
#: ======  =========  =========  =========  =========  ==================
#:
#: (interp at N = 512 / 1024 / 2048.)  The error is second order in the launch
#: pitch -- a clean 4x per doubling -- and the trace is ``n_side**2``, so the
#: choice is a cost/accuracy point.  257 is where the two halves of the call
#: balance: it is the largest lattice whose TRACE still costs less than the
#: interpolation it feeds on every grid measured, and it halves the remap's
#: interpolation error for that.
#:
#: What the raise does NOT buy, stated because it is easy to assume: on a
#: SMOOTH input the model's own observables were already converged at 181.  On
#: the p10 decentered singlet at N = 1280 the image-plane centroid, RMS radius
#: and EE80 move by 8e-06, 1.4e-04 and 5e-04 relative between 181 and 2049.
#: What the lattice does govern is input STRUCTURE: contrast transfer collapses
#: onto one curve in launch-samples-per-period (measured 0.94 / 0.92 at 7.2
#: samples, 0.76 / 0.70 at 3.6 and 3.1, 0.57 / 0.60 at 2.2 across four
#: lattices), so the resolved period scales with the pitch and a caller with a
#: structured pupil raises the lattice through the keyword.
#:
#: The lattice is FREE TO MOVE, which is the other half of the choice.  The
#: image-plane mirror residual of a +d / -d decenter pair -- an EXACT symmetry
#: of the physics, so any residual is the model's own artefact -- reads 2.9e-14
#: to 8.5e-12 across 181, 257, 512, 513, 1025 and 2049 with either
#: interpolation backend, the top of that range being the densest lattice's
#: longer reduction rather than anything discrete.  What decides it is the
#: symmetric input window cut in :func:`_apply_displaced_remap_2d`, not the
#: backend; see ``docs/history/lumenairy.elements._lens_real.md`` for the
#: measurement that separated the two.
_DISP_REMAP_2D_N_SIDE = 257

#: Smallest launch lattice the 2-D remap can be asked for: ``np.gradient``
#: needs three rows to have one central-difference interior row, and the
#: bilinear inversion needs a cell on each side of the lattice centre.  This is
#: a structural floor, not a quality bar -- the quality bar is the launch pitch
#: against the field pitch, which :func:`_warn_if_remap_lattice_smooths`
#: reports on every call where it binds.
_DISP_REMAP_2D_MIN_N_SIDE = 3

#: Sweep cap for the structured inversion's Newton loop.  Points still moving
#: at the cap keep whatever coordinate they reached and are then judged by the
#: residual bar like any other, so the cap bounds cost, not correctness.
_DISP_REMAP_2D_INV_MAX_ITERS = 32

#: Convergence bar for the structured inversion's Newton loop, as a fraction of
#: the LOCAL exit cell: a field point retires once its residual
#: ``|P(u, v) - (x, y)|`` is below it, and retired points leave the sweep.
#:
#: Derived, not tuned.  A residual ``r`` leaves the launch coordinate off by
#: ``r / |dP/du|``, which costs the interpolated output ``|df/du| r / |dP/du|``
#: = ``|df/du| r dstep / cell``; the bilinear interpolation it feeds already
#: costs ``~ dstep**2 |d2f/du2| / 8``.  The inversion is therefore negligible
#: while ``r / cell << dstep |f''| / (8 |f'|)``.  Measured for the transported
#: OPL over the illuminated pupil of the p10 singlet, that right-hand side is
#: 8.7e-03 / 6.2e-03 / 3.1e-03 at n_side 181 / 257 / 513 -- three decades above
#: this bar at every lattice.  On the same fixture at N = 512 the residual
#: falls 2.4e-06 -> 1.0e-11 -> 6.2e-16 m over three sweeps against a 3.9e-11 m
#: bar, so 97 % of the grid retires after two.
#:
#: It cannot be a BITWISE fixed point of the launch coordinate, which is what
#: the surface-intersection Newtons in this module use: the Jacobian here is a
#: central difference of the lattice while the residual is of the bilinear
#: interpolant, so the two disagree at O(dstep**2) and the iteration
#: limit-cycles in its last bits instead of landing (measured on the p10
#: fixture at N = 512: 95 % of the grid was still moving at sweep 32).
_DISP_REMAP_2D_INV_TOL_FRAC = 1.0e-6

#: Coverage bar for the structured inversion, as a fraction of the LOCAL exit
#: cell.  A field point the ray map reaches retires at the convergence bar
#: above; a point it does not reach either clips out of the launch rectangle
#: (and then fails the aperture cut, the fan being 3 % wider than the aperture)
#: or stalls a fraction of a cell away.  1e-3 sits three decades above the
#: convergence bar every reached point crosses and three below the smallest
#: genuine miss, so it is a gap, not a tuned number.
_DISP_REMAP_2D_INV_MISS_FRAC = 1.0e-3

#: Backends for the 2-D remap's scattered-exit -> field-grid step.
_VALID_DISP_REMAP_INTERP = ('structured', 'delaunay')


def _warn_if_remap_lattice_smooths(r_max, dx, dy, n_side):
    """Warn when the 2-D remap's launch lattice is coarser than the field grid.

    The remap is a geometric transfer: everything the exit field knows comes
    from ``n_side**2`` launched rays, so input structure finer than the launch
    pitch is not propagated, it is SMOOTHED AWAY.  Measured: a ripple at
    2.2 launch samples per period comes back at 0.51 of its input
    contrast where the (field-grid) screen path resolves it at 1.26.

    The lattice is a per-call choice (``displaced_n_side=``), so the message
    quotes both pitches and the lattice that would clear the bar -- raising it
    costs ``n_side**2`` in the trace and nothing in stability, the structured
    inversion having removed the resolution/reflection-stability trade the
    scattered backend imposed.
    """
    try:
        h = min(float(dx), float(dy))
        r = float(r_max)
    except (TypeError, ValueError):
        return
    if not (np.isfinite(h) and h > 0.0 and np.isfinite(r) and r > 0.0):
        return
    pitch = 2.0 * r / max(int(n_side) - 1, 1)
    if pitch <= 2.0 * h:
        return
    n_clear = int(np.ceil(r / h)) + 1
    import warnings
    warnings.warn(
        f"apply_real_lens: surface_model='displaced' is routing this "
        f"asymmetric element to the 2-D transverse-walk remap, which rebuilds "
        f"the exit field from a {int(n_side)}x{int(n_side)} launch lattice -- "
        f"a {pitch * 1e6:.2f} um pitch across the {2 * r * 1e3:.3f} mm traced "
        f"aperture, against a {h * 1e6:.2f} um field pitch.  Input structure "
        f"finer than the LAUNCH pitch (a hard stop edge, an obscuration, an "
        f"upstream DOE, speckle) is smoothed to that lattice, and the remap "
        f"carries no in-glass diffraction at all.  Pass "
        f"displaced_n_side={n_clear} to resolve the field pitch (the trace "
        f"costs n_side**2), displaced_obliquity='pointwise' for the "
        f"single-plane obliquity screen, which lives on the field grid, or "
        f"apply_real_lens_traced for a per-pixel ray-traced OPL.",
        RuntimeWarning, _WARN_STACKLEVEL)


def _normalise_displaced_n_side(n_side, fn_name='apply_real_lens'):
    """Validate a public ``displaced_n_side`` into an ``int``, or ``None``.

    ``None`` means "the module default" (:data:`_DISP_REMAP_2D_N_SIDE`).  A
    float that is not an exact integer is refused rather than truncated: the
    lattice is a ray count, and silently turning 512.5 into 512 is the
    discarded-setting class this family's guards exist to close."""
    if n_side is None:
        return None
    if isinstance(n_side, bool) or not isinstance(n_side, (int, np.integer,
                                                          float, np.floating)):
        raise ValueError(
            f"{fn_name}: displaced_n_side must be an integer ray count (the "
            f"side of the 2-D remap's square launch lattice, in rays) or None "
            f"for the default {_DISP_REMAP_2D_N_SIDE}; got "
            f"{type(n_side).__name__}.")
    v = float(n_side)
    if not np.isfinite(v) or v != int(v):
        raise ValueError(
            f"{fn_name}: displaced_n_side={n_side!r} is not an exact integer "
            f"ray count.  Pass the side of the square launch lattice, in rays "
            f"(the trace costs displaced_n_side**2), or None for the default "
            f"{_DISP_REMAP_2D_N_SIDE}.")
    v = int(v)
    if v < _DISP_REMAP_2D_MIN_N_SIDE:
        raise ValueError(
            f"{fn_name}: displaced_n_side={n_side!r} is below the "
            f"{_DISP_REMAP_2D_MIN_N_SIDE}-ray structural minimum (the "
            f"finite-difference Jacobian needs one central-difference interior "
            f"row and the inversion needs a bilinear cell either side of it).  "
            f"The launch pitch is 2*r_aperture/(displaced_n_side-1); the call "
            f"warns whenever it is coarser than twice the field pitch.")
    return v


def _build_displaced_ray_map_2d(surfaces, thicknesses, wavelength, r_max,
                                n_side=None, dir_fn=None, eik_fn=None,
                                r_fan_factor=1.03):
    """Pointwise 2-D generalisation of :func:`_build_displaced_ray_map` (the P2
    remap) for decentered / tilted / freeform elements (niche N11 / P10).

    ``n_side=None`` (the default) uses ``_DISP_REMAP_2D_N_SIDE``; see that
    constant for why it is a fixed 181 and what the caller warns about.

    Launch a REGULAR square ray grid (side ``n_side``, spanning
    ``+-r_fan_factor*r_max`` so the illuminated aperture disk has interior
    neighbours for the finite-difference Jacobian) along the input congruence
    ``dir_fn`` and trace it through the actual (possibly asymmetric) surfaces --
    honouring per-surface decenter / tilt / freeform ``sag_callable`` via the
    SHARED :func:`_disp_surface_z_grad` geometry (identical convention to the
    pointwise screen, the traced / GBD ray models, and the lumenairy-free
    ``geom_spot_decenter_oracle``).  Accumulates the total optical path
    ``opl = eik_fn(x0, y0) + sum n_segment * path`` (entrance eikonal + per-
    segment geometric path length, exactly as the 1-D remap does), so the exit
    phase is ``k0 * opl``.

    Returns ``(X0, Y0, XO, YO, OPL, ALIVE, dstep)`` -- the regular launch grid,
    the scattered exit map, the OPL, the alive mask (all shape
    ``(n_side, n_side)``) and the scalar launch step -- so the caller can take
    the forward Jacobian ``det d(x_out,y_out)/d(x_in,y_in)`` by finite difference
    on the regular launch grid.  Pure geometric trace; wave-model-independent.
    """
    r_fan = float(r_max) * float(r_fan_factor)
    if n_side is None:
        n_side = _DISP_REMAP_2D_N_SIDE
    ax = np.linspace(-r_fan, r_fan, int(n_side))
    dstep = float(ax[1] - ax[0])
    LX, LY = np.meshgrid(ax, ax)
    x0 = LX.ravel().astype(np.float64)
    y0 = LY.ravel().astype(np.float64)
    n = x0.size
    idx = [(float(get_glass_index(s['glass_before'], wavelength)),
            float(get_glass_index(s['glass_after'], wavelength)))
           for s in surfaces]
    if dir_fn is None:
        gx = np.zeros(n)
        gy = np.zeros(n)
    else:
        gx, gy = dir_fn(x0, y0)
        gx = np.where(np.isfinite(gx), gx, 0.0).astype(np.float64).reshape(n)
        gy = np.where(np.isfinite(gy), gy, 0.0).astype(np.float64).reshape(n)
    nrm = np.sqrt(1.0 + gx * gx + gy * gy)
    dxr = gx / nrm
    dyr = gy / nrm
    dzr = 1.0 / nrm
    px = x0.copy()
    py = y0.copy()
    pz = np.zeros(n)
    if eik_fn is None:
        opl = np.zeros(n)
    else:
        opl = np.asarray(eik_fn(x0, y0), dtype=np.float64).reshape(n).copy()
        opl = np.where(np.isfinite(opl), opl, 0.0)
    alive = np.ones(n, dtype=bool)
    z_v = 0.0
    n_surf = len(surfaces)
    for i, s in enumerate(surfaces):
        n1, n2 = idx[i]
        R = s['radius']
        flat = ((R == 0) or (not np.isfinite(R))) and (
            s.get('sag_callable') is None
            and (s.get('tilt') or (0.0, 0.0)) == (0.0, 0.0))
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (z_v - pz) / dzr
        if flat:
            px = px + t * dxr
            py = py + t * dyr
            pz = pz + t * dzr
            nxc = np.zeros(n)
            nyc = np.zeros(n)
            nzc = np.ones(n)
        else:
            for _ in range(24):
                xq = px + t * dxr
                yq = py + t * dyr
                f, dfdx, dfdy = _disp_surface_z_grad(s, xq, yq)
                f = np.where(np.isnan(f), 0.0, f)
                dfdx = np.where(np.isnan(dfdx), 0.0, dfdx)
                dfdy = np.where(np.isnan(dfdy), 0.0, dfdy)
                g = pz + t * dzr - z_v - f
                dgdt = dzr - (dfdx * dxr + dfdy * dyr)
                dgdt = np.where(np.abs(dgdt) < 1e-30, 1e-30, dgdt)
                # Byte-identical early exit (not a tolerance): once ``t`` reaches a
                # BITWISE fixed point, every remaining sweep reproduces it exactly, so
                # the loop can stop without changing a single output bit.  The
                # intersection residual is exactly 0 after 2 sweeps on every fixture
                # measured, and each sweep costs three ``_surface_sag_general``
                # evaluations over the whole fan -- ~10x of the geometric traces.
                _t_new = t - g / dgdt
                _t_done = np.array_equal(_t_new, t, equal_nan=True)
                t = _t_new
                del _t_new
                if _t_done:
                    break
            px = px + t * dxr
            py = py + t * dyr
            pz = pz + t * dzr
            f, dfdx, dfdy = _disp_surface_z_grad(s, px, py)
            nzc = np.ones(n)
            nxc = -dfdx
            nyc = -dfdy
            nn = np.sqrt(nxc * nxc + nyc * nyc + nzc * nzc)
            nxc = nxc / nn
            nyc = nyc / nn
            nzc = nzc / nn
            alive = alive & np.isfinite(f)
        # OPL segment: the unit-direction parametric step ``t`` is the geometric
        # path length in the medium BEFORE this surface (n1).
        opl = opl + n1 * t
        cos_i = dxr * nxc + dyr * nyc + dzr * nzc
        eta = n1 / n2
        sin2t = eta * eta * (1.0 - cos_i * cos_i)
        alive = (alive & np.isfinite(px) & np.isfinite(py)
                 & np.isfinite(cos_i) & (sin2t <= 1.0))
        cos_t = np.sqrt(np.maximum(1.0 - sin2t, 0.0))
        ndx = eta * dxr + (cos_t - eta * cos_i) * nxc
        ndy = eta * dyr + (cos_t - eta * cos_i) * nyc
        ndz = eta * dzr + (cos_t - eta * cos_i) * nzc
        nn2 = np.sqrt(ndx * ndx + ndy * ndy + ndz * ndz)
        nn2 = np.where(nn2 == 0.0, 1.0, nn2)
        dxr = ndx / nn2
        dyr = ndy / nn2
        dzr = ndz / nn2
        if i < n_surf - 1:
            z_v += thicknesses[i]
    z_exit = float(sum(thicknesses))
    with np.errstate(divide='ignore', invalid='ignore'):
        t_f = (z_exit - pz) / dzr
    # Exit referencing leg in ``surfaces[-1]['glass_after']``, not air -- see
    # the 1-D twin :func:`_build_displaced_ray_map` for the measurement.
    opl = opl + idx[-1][1] * t_f
    x_out = px + t_f * dxr
    y_out = py + t_f * dyr
    alive = alive & np.isfinite(x_out) & np.isfinite(y_out) & np.isfinite(opl)
    shp = (int(n_side), int(n_side))
    # ``r_ap`` = the true aperture radius: the fan was launched 3% wider so the
    # aperture-edge rays have interior Jacobian neighbours, but rays whose
    # ENTRANCE height exceeds r_ap are outside the pupil and must not contribute
    # amplitude (the exit-plane remap bypasses the per-surface stop mask, so the
    # aperture is enforced here on the entrance footprint -- mirroring the 1-D
    # remap fan, which stops exactly at r_max).
    return (x0.reshape(shp), y0.reshape(shp), x_out.reshape(shp),
            y_out.reshape(shp), opl.reshape(shp), alive.reshape(shp), dstep,
            float(r_max))


def _remap2d_interp_delaunay(XO, YO, amp_out, OPL, m, Xg, Yg):
    """Scattered exit points -> field grid by QHull Delaunay barycentric
    interpolation, the SCATTERED backend of :func:`_apply_displaced_remap_2d`.

    ONE ``LinearNDInterpolator`` (3-column value array) does the barycentric
    interpolation for the transported complex amplitude AND the OPL in one
    pass: the triangulation and the per-query-point weights depend only on
    ``pts``, and each column is a separate ``sum(weight_k * value_k)``
    reduction, so this reproduces three single-column interps bit for bit while
    building the Delaunay once and walking the full-grid query once (measured
    ~1.6x).  ``amp_grid`` / ``opl_grid`` are strided VIEWS into the single
    ``(Ny, Nx, 3)`` result -- no per-column dense copy -- so the peak footprint
    is one 3-wide grid.  The transported quantity is the COMPLEX residual
    field, carried as its real and imaginary parts (both smooth wherever the
    remap is valid, unlike a wrapped phase), so it takes two of the three
    columns.  Outside the hull every column comes back NaN: the amplitude is
    set to 0 and the OPL is nearest-filled.

    Retained as the structured backend's independent oracle: it approximates
    the same map by a different O(h**2) rule (barycentric over the exit
    triangulation, against bilinear in launch space), so the two converging to
    one answer is a check neither could give alone.  Its own limits are why it
    is not the default -- the hull stops at the outermost retained exit point,
    so a truncated pupil comes back with a ring of exactly-zero pixels inside
    the illuminated region, and the triangulation of a near-degenerate exit set
    resolves its cells arbitrarily."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    pts = np.column_stack([XO[m].ravel(), YO[m].ravel()])
    _opl_flat = OPL[m].ravel()
    _a_flat = amp_out[m].ravel()
    _q = LinearNDInterpolator(
        pts, np.column_stack([_a_flat.real, _a_flat.imag, _opl_flat]))(Xg, Yg)
    amp_grid = _q[..., 0] + 1j * _q[..., 1]
    opl_grid = _q[..., 2]
    nan = np.isnan(opl_grid)
    if bool(nan.any()):
        opl_grid[nan] = NearestNDInterpolator(pts, _opl_flat)(
            Xg[nan], Yg[nan])
        amp_grid[nan] = 0.0
    return amp_grid, opl_grid


def _remap2d_affine_seed(XOf, YOf, dstep, u0, v0, Xt, Yt):
    """Newton seed for :func:`_remap2d_interp_structured`: the exit map's own
    GLOBAL AFFINE part, inverted.

    ``P(u, v) ~ M (u, v) + b`` in the least-squares sense over the whole launch
    lattice.  ``M`` is dominated by the element's magnification, so seeding at
    ``M^-1 ((x, y) - b)`` starts one squaring of the Newton error closer than
    seeding at the target itself -- measured on the p10 decentered singlet, a
    2.4e-06 m seed residual against 5.5e-05 m, which is one whole sweep of the
    grid.  The normal equations are accumulated as moments (nine sums over the
    lattice) rather than assembled as a design matrix, so the cost is one pass
    and no ``n_side**2 x 3`` allocation.

    Falls back to the target itself when ``M`` is singular or non-finite -- a
    fold that collapses the map onto a line -- which the Newton then handles
    exactly as it did before."""
    n_v, n_u = XOf.shape
    uu = u0 + np.arange(n_u, dtype=np.float64) * dstep
    vv = v0 + np.arange(n_v, dtype=np.float64) * dstep
    U, V = np.meshgrid(uu, vv)
    n = float(U.size)
    su, sv = float(U.sum()), float(V.sum())
    suu, svv, suv = float((U * U).sum()), float((V * V).sum()), float((U * V).sum())
    G = np.array([[suu, suv, su], [suv, svv, sv], [su, sv, n]])
    rhs = np.array([
        [float((U * XOf).sum()), float((U * YOf).sum())],
        [float((V * XOf).sum()), float((V * YOf).sum())],
        [float(XOf.sum()), float(YOf.sum())]])
    try:
        coef = np.linalg.solve(G, rhs)
        Minv = np.linalg.inv(coef[:2, :].T)
    except np.linalg.LinAlgError:
        return Xt.copy(), Yt.copy()
    if not bool(np.all(np.isfinite(Minv))):
        return Xt.copy(), Yt.copy()
    px = Xt - coef[2, 0]
    py = Yt - coef[2, 1]
    return (Minv[0, 0] * px + Minv[0, 1] * py,
            Minv[1, 0] * px + Minv[1, 1] * py)


def _remap2d_interp_structured(XOf, YOf, amp_src, opl_src, box_mask,
                               u0, v0, dstep, r_ap, Xg, Yg):
    """Launch->exit map INVERTED on its own structured launch grid, the
    default backend of :func:`_apply_displaced_remap_2d`.

    The fan is a REGULAR square lattice, so ``(XOf, YOf)`` -- the exit position
    as a function of the launch coordinate ``(u, v)`` -- is a smooth
    curvilinear grid, not a scattered point cloud.  For each field point this
    solves ``P(u, v) = (x, y)`` by Newton, reading ``P`` and its lattice
    gradients through ``map_coordinates``, and then reads the transported
    amplitude and OPL at the launch coordinate that comes back -- so the
    interpolation stencil is always the regular launch quad.

    Three things follow from inverting the map rather than triangulating its
    image, and together they are why this is the default:

    * no combinatorial choice.  Every step is a smooth function of the traced
      data, so a 1-ULP perturbation of the launch grid (which is all that
      separates a +d from a -d decenter) stays a 1-ULP perturbation of the
      output.  A Delaunay backend must instead choose a diagonal for each
      near-degenerate exit quad, and that choice is not reflection-stable.
    * no resolution ceiling.  A denser lattice cannot create a sliver cell, so
      accuracy improves with ``n_side`` instead of trading against symmetry.
    * the aperture is cut on the inverted LAUNCH coordinate
      (``u**2 + v**2 <= r_ap**2``), i.e. on the entrance footprint at sub-cell
      resolution, instead of at the convex hull of the retained exit points.

    ``box_mask`` marks the launch points that carry pupil amplitude; only the
    field points inside their exit bounding box (plus one exit cell, so an
    interpolated interior point cannot fall outside it) are inverted.  That is
    exact -- the scattered path's hull is empty out there too -- and it is what
    keeps a heavily padded grid cheap.

    A field point the lattice does not reach keeps amplitude 0: Newton either
    walks out of the launch rectangle, whereupon the clipped coordinate fails
    the aperture cut because the fan is launched 3 % wider than the aperture,
    or it stalls at a residual the ``_DISP_REMAP_2D_INV_TOL_FRAC`` bar
    refuses."""
    from scipy.ndimage import map_coordinates
    n_v, n_u = XOf.shape
    amp_grid = np.zeros(Xg.shape, dtype=np.complex128)
    opl_grid = np.zeros(Xg.shape, dtype=np.float64)
    # Lattice gradients of the exit map w.r.t. the launch coordinate
    # (axis 0 = y_in, axis 1 = x_in) -- the Newton Jacobian, and the local
    # exit-cell size the inversion residual is scored against.
    dXO_dv, dXO_du = np.gradient(XOf, dstep, dstep)
    dYO_dv, dYO_du = np.gradient(YOf, dstep, dstep)
    # The longer edge of the exit cell one launch cell maps to: the scale the
    # inversion residual is meaningful against, and the margin the search box
    # needs.  A lattice quantity, so it is interpolated in ONE pass rather than
    # rebuilt from four interpolated derivatives.
    cell_lat = np.maximum(np.hypot(dXO_du, dYO_du),
                          np.hypot(dXO_dv, dYO_dv)) * dstep
    _cell_max = float(cell_lat.max())
    _bx = XOf[box_mask]
    _by = YOf[box_mask]
    box = ((Xg >= _bx.min() - _cell_max) & (Xg <= _bx.max() + _cell_max)
           & (Yg >= _by.min() - _cell_max) & (Yg <= _by.max() + _cell_max))
    if not bool(box.any()):
        return amp_grid, opl_grid
    Xt = Xg[box]
    Yt = Yg[box]
    u_hi = u0 + (n_u - 1) * dstep
    v_hi = v0 + (n_v - 1) * dstep
    # Seed from the exit map's own GLOBAL AFFINE part (a least-squares fit over
    # the launch lattice, inverted in closed form).  The map is a
    # magnification plus a transverse walk, so the affine part is most of it
    # and inverting it first costs one 3x3 solve and removes a whole Newton
    # sweep: the seed residual runs 2.5e-06 m against 5.5e-05 m for the
    # identity seed on the p10 singlet, which is one squaring of the error.
    u, v = _remap2d_affine_seed(XOf, YOf, dstep, u0, v0, Xt, Yt)
    np.clip(u, u0, u_hi, out=u)
    np.clip(v, v0, v_hi, out=v)
    crd = np.stack([(v - v0) / dstep, (u - u0) / dstep])
    # The local exit cell, read once at the seed: it is a SCALE for the two
    # bars, and the seed is already a fraction of a cell from the answer, so
    # re-reading it every sweep would buy a few percent of a quantity that is
    # three decades from either bar.
    cell = map_coordinates(cell_lat, crd, order=1, mode='nearest')
    tol = _DISP_REMAP_2D_INV_TOL_FRAC * cell
    # Sweep only the points not yet converged.  A converged point's further
    # sweeps are pure cost, and the RESIDUAL is what decides -- see
    # _DISP_REMAP_2D_INV_TOL_FRAC for why this loop cannot stop on a bitwise
    # fixed point the way the surface-intersection Newtons do.
    resid = np.empty(Xt.size)
    act = np.arange(Xt.size)
    ua = u[act]
    va = v[act]
    for _ in range(_DISP_REMAP_2D_INV_MAX_ITERS):
        rx = Xt[act] - map_coordinates(XOf, crd, order=1, mode='nearest')
        ry = Yt[act] - map_coordinates(YOf, crd, order=1, mode='nearest')
        r = np.hypot(rx, ry)
        resid[act] = r
        run = r > tol[act]
        if not bool(run.any()):
            act = act[:0]
            break
        if not bool(run.all()):
            act = act[run]
            ua = ua[run]
            va = va[run]
            rx = rx[run]
            ry = ry[run]
            crd = np.stack([(va - v0) / dstep, (ua - u0) / dstep])
        a = map_coordinates(dXO_du, crd, order=1, mode='nearest')
        b = map_coordinates(dXO_dv, crd, order=1, mode='nearest')
        c = map_coordinates(dYO_du, crd, order=1, mode='nearest')
        d = map_coordinates(dYO_dv, crd, order=1, mode='nearest')
        det = a * d - b * c
        det = np.where(np.abs(det) < 1e-30, 1e-30, det)
        ua = np.clip(ua + (d * rx - b * ry) / det, u0, u_hi)
        va = np.clip(va + (-c * rx + a * ry) / det, v0, v_hi)
        u[act] = ua
        v[act] = va
        crd = np.stack([(va - v0) / dstep, (ua - u0) / dstep])
    if act.size:
        # The cap, not the bar, stopped the loop: re-score the stragglers at
        # the coordinates they actually reached.
        resid[act] = np.hypot(
            Xt[act] - map_coordinates(XOf, crd, order=1, mode='nearest'),
            Yt[act] - map_coordinates(YOf, crd, order=1, mode='nearest'))
    ok = resid <= _DISP_REMAP_2D_INV_MISS_FRAC * cell
    ok &= (u * u + v * v) <= (r_ap * (1.0 + 1e-9)) ** 2
    crd = np.stack([(v - v0) / dstep, (u - u0) / dstep])
    _ar = map_coordinates(amp_src.real, crd, order=1, mode='nearest')
    _ai = map_coordinates(amp_src.imag, crd, order=1, mode='nearest')
    _op = map_coordinates(opl_src, crd, order=1, mode='nearest')
    amp_grid[box] = np.where(ok, _ar + 1j * _ai, 0.0)
    opl_grid[box] = np.where(ok, _op, 0.0)
    return amp_grid, opl_grid


def _apply_displaced_remap_2d(E_in, ray_map_2d, wavelength, dx, dy,
                              eik_fn=None, interp_method='structured'):
    """P10 / niche N11 -- energy-conserving 2-D transverse-walk remap for a
    decentered / tilted / freeform element.

    Generalises the P2 rotationally-symmetric exit-plane remap
    (:func:`_apply_displaced_remap`) to the full OFF-AXIS 2-D case: the element
    becomes a geometric transfer ``(x_in, y_in) -> (x_out, y_out)`` carrying the
    TRANSVERSE ray walk between the thick element's surfaces (the walk-off the
    single-plane pointwise screen drops, which made the induced-coma spot narrow
    where it must broaden).  The input amplitude envelope ``|E_in|`` sampled at
    each launched ray's ENTRANCE position is transported to its EXIT position
    with the energy-conserving 2-D Jacobian factor
    ``1/sqrt(|det d(x_out,y_out)/d(x_in,y_in)|)`` (so
    ``|E_out|^2 dA_out = |E_in|^2 dA_in``), and the exit phase is the ray eikonal
    ``k0 * OPL`` (which carries the entrance-plane carrier eikonal) PLUS the
    input field's residual phase against that congruence, transported along the
    same rays (see :func:`_residual_input_field`).  Amplitude and OPL reach the
    field grid SEPARATELY (phase-safe: the eikonal is smooth even where the
    amplitude is warped), then combine.  ``eik_fn(x, y) -> W_conj`` is the
    congruence the fan was launched along (``None`` = collimated); pass the
    SAME callable the ray map was built with.  Returns the exit-vertex-plane
    field (same reference plane as the default screen path).

    ``interp_method`` selects how the traced exit map reaches the field grid:

    * ``'structured'`` (default) -- invert the launch->exit map on its own
      regular launch lattice (:func:`_remap2d_interp_structured`).
    * ``'delaunay'`` -- the scattered QHull backend
      (:func:`_remap2d_interp_delaunay`), retained as that inversion's oracle.
    """
    from scipy.ndimage import distance_transform_edt, map_coordinates
    if interp_method not in _VALID_DISP_REMAP_INTERP:
        raise ValueError(
            f"_apply_displaced_remap_2d: interp_method must be one of "
            f"{list(_VALID_DISP_REMAP_INTERP)} (got {interp_method!r}).")
    X0, Y0, XO, YO, OPL, ALIVE, dstep, r_ap = ray_map_2d
    Ny, Nx = E_in.shape
    k0 = 2.0 * np.pi / wavelength
    # Input field at each ray's entrance position (bilinear), DEMODULATED by
    # the traced congruence so the residual input phase rides along with the
    # amplitude instead of being discarded.  ``|F| == |E_in|``, so the modulus
    # of this sample is exactly the amplitude the legacy code took.
    _Wg = None
    if eik_fn is not None:
        _ax = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
        _ay = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
        _Xg0, _Yg0 = np.meshgrid(_ax, _ay)
        _Wg = np.asarray(eik_fn(_Xg0, _Yg0), dtype=np.float64)
        del _Xg0, _Yg0
    _F = _residual_input_field(E_in, _Wg, wavelength)
    cx = X0.ravel() / dx + Nx / 2.0
    cy = Y0.ravel() / dy + Ny / 2.0
    amp_in = (map_coordinates(_F.real, [cy, cx], order=1,
                              mode='constant', cval=0.0)
              + 1j * map_coordinates(_F.imag, [cy, cx], order=1,
                                     mode='constant', cval=0.0)
              ).reshape(X0.shape)
    del _F
    # The carried envelope is the input field over the largest CENTRED window
    # the caller's grid holds: |x| <= x[-1] = (Nx/2 - 1) dx, and the same in y.
    #
    # The field axis ``(arange(N) - N/2) * d`` reaches one whole sample further
    # on the -x side than on +x, so a ray launched between x[-1] and x[-1] + dx
    # would carry NOTHING (the sample is off the grid) while its mirror
    # between x[0] - dx and x[0] carries the full envelope.  That is a rim of
    # pupil amplitude decided by a half-pixel of grid convention, and whether
    # the launch lattice has a ray in that one-pixel band is arbitrary: on the
    # p10 decentered singlet at N = 512 it does at n_side 512, 1025 and 2049
    # and does not at 181, 257 and 513.  That, and not the choice of
    # interpolation backend, is what ties the launch lattice to the mirror
    # symmetry: without the window the image-plane mirror residual reads
    # 6.3e-03 at 512 against 7.9e-14 at 181 on BOTH backends.  Cutting the
    # envelope at the symmetric window costs the outermost input row and
    # column and makes the sampled envelope mirror-exact to 4.4e-16 at every
    # lattice.
    _win = ((np.abs(X0) <= (Nx / 2.0 - 1.0) * dx)
            & (np.abs(Y0) <= (Ny / 2.0 - 1.0) * dy))
    amp_in = np.where(_win, amp_in, 0.0)
    # Forward Jacobian det d(x_out,y_out)/d(x_in,y_in) on the regular launch grid
    # (physical spacing).  Fill any dead-ray (TIR / miss) exit position by
    # nearest-alive FIRST so a dead ray does not poison a live neighbour's
    # central-difference derivative.
    XOf = XO.copy()
    YOf = YO.copy()
    dead = ~ALIVE
    if bool(dead.any()) and bool(ALIVE.any()):
        from scipy.interpolate import NearestNDInterpolator
        pa = np.column_stack([X0[ALIVE], Y0[ALIVE]])
        XOf[dead] = NearestNDInterpolator(pa, XO[ALIVE])(X0[dead], Y0[dead])
        YOf[dead] = NearestNDInterpolator(pa, YO[ALIVE])(X0[dead], Y0[dead])
    dXO_dy, dXO_dx = np.gradient(XOf, dstep, dstep)   # axis0=y_in, axis1=x_in
    dYO_dy, dYO_dx = np.gradient(YOf, dstep, dstep)
    det = dXO_dx * dYO_dy - dXO_dy * dYO_dx
    jac_amp = 1.0 / np.sqrt(np.maximum(np.abs(det), 1e-30))
    amp_out = amp_in * jac_amp
    # Enforce the aperture on the ENTRANCE footprint: the 3%-wider fan only
    # supplies Jacobian neighbours; rays launched outside r_ap carry no pupil
    # amplitude (else a ``stop_index`` prescription -- whose field is not
    # pre-apertured -- would leak the beyond-aperture ring).  The scattered
    # backend applies it by dropping those rays before it triangulates; the
    # structured backend applies the same disk to the INVERTED launch
    # coordinate, so it cuts the pupil edge at sub-launch-pitch resolution
    # while the rays just outside still serve as interpolation neighbours.
    in_ap = (X0 * X0 + Y0 * Y0) <= (r_ap * (1.0 + 1e-9)) ** 2
    m = ALIVE & in_ap & np.isfinite(amp_out) & (np.abs(amp_in) > 0.0)
    if int(m.sum()) < 4:
        return np.zeros_like(E_in, dtype=np.complex128)
    x = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    y = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    Xg, Yg = np.meshgrid(x, y)
    if interp_method == 'delaunay':
        amp_grid, opl_grid = _remap2d_interp_delaunay(
            XO, YO, amp_out, OPL, m, Xg, Yg)
    else:
        # Interpolation sources on the launch lattice.  A dead (TIR / miss)
        # ray carries no amplitude, so it enters as an exact 0 rather than as
        # an invented value; its OPL is nearest-filled by the structured (EDT)
        # fill so a NaN cannot reach a live neighbour's bilinear stencil, and
        # nothing reads it (its amplitude is 0).
        _good = ALIVE & np.isfinite(amp_out) & np.isfinite(OPL)
        amp_src = np.where(_good, amp_out, 0.0).astype(np.complex128)
        opl_src = np.asarray(OPL, dtype=np.float64)
        if not bool(_good.all()):
            _fi = tuple(distance_transform_edt(
                ~_good, return_distances=False, return_indices=True))
            opl_src = opl_src[_fi]
        amp_grid, opl_grid = _remap2d_interp_structured(
            XOf, YOf, amp_src, opl_src, m,
            float(X0[0, 0]), float(Y0[0, 0]), float(dstep), float(r_ap),
            Xg, Yg)
    opl_ref = float(np.median(OPL[m]))
    E_out = amp_grid * np.exp(1j * k0 * (opl_grid - opl_ref))
    out_dtype = E_in.dtype if np.iscomplexobj(E_in) else np.complex128
    return np.asarray(E_out, dtype=out_dtype)


def _split_step_fan_opl(surfaces, thicknesses, wavelength, h_fan):
    """OPL at the EXIT VERTEX PLANE of the split-step model itself, for a
    collimated meridional fan launched at heights ``h_fan``.

    This is a thin-screen RAY model of what :func:`apply_real_lens` actually
    does, not an idealisation of it:

    * at each surface the ray meets the screen on that surface's VERTEX PLANE
      (no axial motion -- the screen is thin), its OPL changes by
      ``-(n2 - n1) * sag_i(x)`` (the screen is ``exp(-i k0 OPD)`` under
      ``phase = exp(+i k0 OPL)``) and its transverse optical momentum is
      kicked by ``-grad OPD_i = -(n2 - n1) * grad sag_i(x)``, which is exactly
      the deflection a phase screen imparts;
    * across each gap the ASM carries a component of transverse momentum
      ``p`` a distance ``t`` at ``exp(i k0 pz t)`` with
      ``pz = sqrt(n**2 - |p|**2)``, and stationary phase puts the wavepacket at
      ``x + (p/pz) t`` with a transported phase of ``k0 n**2 t / pz`` -- i.e.
      the geometric ``n t / cos(theta)``.  So the gap leg is a straight ray
      through the glass, including the slab obliquity the in-glass ASM already
      supplies exactly.

    Returns ``(x_exit, opl)``: the model ray's transverse position on the exit
    vertex plane (``z = sum(thicknesses)``, where the last screen sits) and its
    OPL there.

    WHY THIS AND NOT ``sum (n2-n1) sag_i(h)``.  That expression evaluates every
    surface at the SAME entrance height and adds no propagation at all, so the
    residual taken against a real ray trace is dominated by the in-glass
    obliquity ``n t theta**2 / 2`` -- which the split-step model ALREADY has,
    from its ASM legs.  Fitting that and imprinting it double-counts it:
    measured 337 nm of "correction" on a plano-convex whose true model residual
    is 0.85 nm.
    """
    n_surf = len(surfaces)
    x = np.asarray(h_fan, dtype=np.float64).copy()
    p = np.zeros_like(x)                     # transverse optical momentum
    opl = np.zeros_like(x)
    for i, s_i in enumerate(surfaces):
        R_i = s_i['radius']
        kc_i = s_i.get('conic', 0.0)
        asph_i = s_i.get('aspheric_coeffs')
        R_y_i = s_i.get('radius_y')
        n1_i = float(get_glass_index(s_i['glass_before'], wavelength))
        n2_i = float(get_glass_index(s_i['glass_after'], wavelength))
        dn = n2_i - n1_i

        def _sag_at(xq, _R=R_i, _k=kc_i, _a=asph_i, _Ry=R_y_i, _s=s_i):
            xq = np.asarray(xq, dtype=np.float64)
            if _Ry is not None:
                v = surface_sag_biconic(
                    xq, np.zeros_like(xq), R_x=_R, R_y=_Ry, conic_x=_k,
                    conic_y=_s.get('conic_y'), aspheric_coeffs=_a,
                    aspheric_coeffs_y=_s.get('aspheric_coeffs_y'))
            else:
                v = _surface_sag_general(xq * xq, _R, _k, _a)
            return np.where(np.isnan(v), 0.0, v)

        sag_x = _sag_at(x)
        # d(sag)/dx by central difference on the same evaluator the screen
        # uses, so the kick is the gradient of the screen that is applied and
        # not of an analytic idealisation of it.
        e = np.maximum(1e-9, 1e-6 * np.abs(x))
        grad = (_sag_at(x + e) - _sag_at(x - e)) / (2.0 * e)
        opl = opl - dn * sag_x
        p = p - dn * grad
        if i < n_surf - 1:
            t_i = float(thicknesses[i])
            pz_sq = n2_i * n2_i - p * p
            pz = np.sqrt(np.maximum(pz_sq, 1e-12))
            opl = opl + (n2_i * n2_i) * t_i / pz
            x = x + (p / pz) * t_i
    return x, opl


def _propagate_through_glass(E: Any, thickness: float, wavelength: float,
                             n_medium_r: float, n_medium_kappa: float,
                             dx: float, dy: float, bandlimit: bool,
                             wave_propagator: Optional[str], absorption: bool,
                             k0: float, xp: Any,
                             stream_transfer_function: bool = False) -> Any:
    """Propagate ``E`` a distance ``thickness`` through a medium of real index
    ``n_medium_r`` (+ optional bulk absorption via ``n_medium_kappa``),
    dispatching on ``wave_propagator``.

    Extracted verbatim from the per-surface loop so the whole-grid path and the
    row-band (``sag_chunk_rows``) path share one glass-propagation
    implementation.  Returns the propagated field."""
    lam_medium = wavelength / n_medium_r
    if wave_propagator == 'sas':
        from ..propagators.propagation import (
            resample_field,
            scalable_angular_spectrum_propagate,
        )
        # SAS takes a SINGLE pitch and assumes a square grid, while the rest of
        # this function threads ``dy`` correctly.  An anamorphic gap would be
        # propagated as if it were square -- wrong physics on the y axis, with
        # no diagnostic -- so refuse, exactly as ``propagate_through_system``'s
        # own sas branch does (``_require_square_pitch``).
        if abs(float(dy) - float(dx)) > abs(float(dx)) * 1e-9:
            raise ValueError(
                f"apply_real_lens: wave_propagator='sas' assumes a square "
                f"grid pitch, but this call is anamorphic (dx={dx:.6g} m, "
                f"dy={dy:.6g} m) and "
                f"scalable_angular_spectrum_propagate takes only one pitch, "
                f"so the in-glass gap would be propagated as if dy == dx.  "
                f"Use wave_propagator='asm' (or 'rayleigh_sommerfeld'), which "
                f"thread the y-pitch correctly, or resample to an isotropic "
                f"grid first.")
        E, dx_new, _ = scalable_angular_spectrum_propagate(
            E, thickness, lam_medium, dx)
        if abs(dx_new - dx) > dx * 1e-6:
            # K6: the band-limited (chirp-Z) interpolant has unit MTF at
            # every frequency the grid represents, but its reconstruction
            # is PERIODIC with period ``N_in*dx_new`` per axis, so it is
            # chosen only while the lens grid's window fits inside one
            # period.  Past that it returns replicas of the field instead
            # of the zeros the spline pads with -- measured P_out/P_in
            # 1.378837 at ``dx_new/dx = 0.7727`` and 9.000535 (a 3x3
            # tiling) at 0.3091, against the spline's 0.950689 and
            # 0.999999.  The 1e-9 slack is ``_warn_mft_output_window``'s
            # own tolerance, so chirp-Z is taken on exactly the windows
            # it would not warn about, and ``min`` picks the binding axis
            # because ``resample_field`` reads one input pitch for both.
            #
            # This gap runs in GLASS: ``lam_medium = wavelength/n`` makes
            # ``dx_new`` smaller by ``n`` than the same gap in air, so the
            # window test fails far more often here than in the
            # free-space chain.  On the WP-A15a covering-array doublet
            # (N = 64, dx = 112.5 um, lambda = 632.8 nm) both gaps sit at
            # ``dx_new/dx`` = 4.2e-3 and 1.1e-3 -- deep in the spline's
            # half -- and the crossover for that grid is a 2.14 m
            # thickness.
            E, _ = resample_field(
                E, dx_new, dx, N_out=E.shape[-1],
                method=('chirpz'
                        if (E.shape[-1] * dx
                            <= min(E.shape[-2], E.shape[-1]) * dx_new
                            * (1.0 + 1e-9))
                        else 'spline'))
    elif wave_propagator == 'fresnel':
        from ..propagators.propagation import fresnel_propagate, resample_field
        E, dx_new, _ = fresnel_propagate(E, thickness, lam_medium, dx, dy=dy)
        if abs(dx_new - dx) > dx * 1e-6:
            # K6, same window-vs-period rule and the same in-glass bias
            # as the ``'sas'`` branch above: chirp-Z (unit MTF) while the
            # lens grid's window fits inside one reconstruction period,
            # the spline where it would return replicas.  Here
            # ``dx_new = lam_medium*thickness/(N*dx)``, so a thin gap in
            # a dense glass lands far inside the spline's half.
            E, _ = resample_field(
                E, dx_new, dx, N_out=E.shape[-1],
                method=('chirpz'
                        if (E.shape[-1] * dx
                            <= min(E.shape[-2], E.shape[-1]) * dx_new
                            * (1.0 + 1e-9))
                        else 'spline'))
    elif wave_propagator in ('rayleigh_sommerfeld', 'rs'):
        from ..propagators.propagation import rayleigh_sommerfeld_propagate
        E = rayleigh_sommerfeld_propagate(
            E, thickness, lam_medium, dx, dy=dy, bandlimit=bandlimit)
    elif wave_propagator == 'asm':
        E = angular_spectrum_propagate(
            E, thickness, lam_medium, dx, dy=dy, bandlimit=bandlimit,
            stream_transfer_function=stream_transfer_function)
    else:
        raise ValueError(
            f"apply_real_lens: unknown wave_propagator {wave_propagator!r}.  "
            f"Supported: 'asm', 'sas', 'fresnel', 'rayleigh_sommerfeld' "
            f"(alias 'rs').")
    if absorption and n_medium_kappa != 0.0:
        E = E * xp.exp(-k0 * n_medium_kappa * thickness)
    return E


def _screen_exp(opd: Any, k0: float, xp: Any) -> Any:
    """``exp(-1j * k0 * opd)`` built without the complex temporaries.

    ``xp.exp(-1j * k0 * opd)`` materialises a full COMPLEX grid for
    ``(-1j*k0) * opd`` and a second one for its exponential -- four
    float-grid-equivalents of transient for a unit-modulus screen, and the
    single hottest primitive in the element (112 ns/element at N = 2048).
    numpy's complex ``exp`` of a pure-imaginary argument IS ``cos + i sin``
    (its real factor is ``exp(0) == 1.0`` exactly), so writing ``cos`` and
    ``sin`` straight into the two strided views of ONE preallocated complex
    array is the same arithmetic in the same order: measured BIT-IDENTICAL
    (max|d| = 0.0) at 1.23x the speed and -17 % of the peak at N = 2048.

    The imaginary part of the argument is ``(-k0) * opd`` in both forms, so the
    two sines see bitwise-identical inputs (verified: the arguments differ by
    exactly 0 at both geometry dtypes).

    RESTRICTED TO float64 GEOMETRY, and that is not conservatism.  For a
    float32 ``opd`` the two forms are NOT the same arithmetic: numpy's
    ``complex64`` exponential carries more than float32 through its own
    sine/cosine, while ``np.cos(float32_arg, out=<float32 view>)`` does not, so
    the results diverge by ~8.4e-08 of unit modulus -- about one float32 ULP,
    but enough to break the byte-identity ``PreparedAnalyticLens`` is pinned
    against under ``set_lens_sag_dtype(np.float32)`` (measured 1.2e-07 of peak
    field).  A float32 screen is half the size anyway, so the saving this
    exists for is not on that path.  CuPy likewise keeps ``xp.exp`` (its
    elementwise kernels are already fused, so there is nothing to save)."""
    if xp is not np or np.asarray(opd).dtype != np.float64:
        return xp.exp(-1j * k0 * opd)
    arg = np.multiply(opd, -k0)
    ph = np.empty(arg.shape, dtype=np.complex128)
    np.cos(arg, out=ph.real)
    np.sin(arg, out=ph.imag)
    return ph


def _absorb_local_path(E: Any, sag: Any, kap_face: float, k0: float,
                       xp: Any) -> Any:
    """Multiply ``E`` by the per-surface half of the LOCAL-glass-path bulk
    absorption, ``exp(-k0 * kap_face * sag)``.

    :func:`_propagate_through_glass` attenuates a gap by its AXIAL thickness,
    ``exp(-k0 kappa t)``, but the glass a pixel actually crosses between
    surfaces ``i`` and ``i+1`` is ``t_i + sag_{i+1} - sag_i``.  Factorising
    that exponential puts ``exp(+k0 kappa_i sag_i)`` on surface ``i`` and
    ``exp(-k0 kappa_i sag_{i+1})`` on surface ``i+1``, so each surface can
    apply ONE local factor with the sag it has already built -- no second sag
    grid, no halo, and the product over the element telescopes back to the
    true local path.  ``kap_face`` is that surface's combined coefficient
    ``kappa_before - kappa_after`` (either term dropped where there is no gap
    on that side); it is exactly zero on the default path, on every
    non-absorbing glass, and for a flat face, so the factor is only built
    where it changes something.

    A 6 mm-centre / 5.2 mm-edge N-BK7 biconvex recovers ~13 % of its
    absorption apodisation from this term; the axial factor itself was already
    exact to 9 digits.  NaN sag (outside the conic domain) contributes a
    neutral 1.0, matching how the OPD screen zeroes the same pixels."""
    a = xp.exp(-k0 * kap_face * sag)
    if bool(xp.any(xp.isnan(a))):
        a = xp.where(xp.isnan(a), xp.ones((), dtype=a.dtype), a)
    _rdt = E.real.dtype
    if a.dtype != _rdt:
        a = a.astype(_rdt)
    return E * a


def _check_apply_real_lens_kwarg_combination(
    *,
    wave_propagator: str,
    slant_correction: bool,
    seidel_correction: bool,
    seidel_poly_order: int,
    prescription: dict,
) -> None:
    """Validate the apply_real_lens kwarg combination space.

    The 4.7 polish pass surfaced several silent-failure regimes when
    mutually-incompatible kwargs are passed.  This helper raises a
    ``ValueError`` with a precise message instead.

    Checks performed:

    * ``wave_propagator`` is one of ``'asm'``, ``'sas'``, ``'fresnel'``,
      ``'rayleigh_sommerfeld'`` (alias ``'rs'``).
    * ``slant_correction=True`` is rejected for ``wave_propagator``
      values other than ``'asm'`` or ``'rs'``.  The Fresnel and SAS
      paths internally resample / change pitch in ways that interact
      badly with the per-surface slant OPD.
    * ``seidel_correction=True`` requires at least 2 surfaces in the
      prescription (single-surface systems have no Seidel sum to
      apply).
    * ``seidel_poly_order`` must be a positive integer.  Order > 12 is
      rejected as the radial polynomial conditioning degrades.
    """
    if wave_propagator not in _VALID_WAVE_PROPAGATORS:
        raise ValueError(
            f"apply_real_lens: unknown wave_propagator "
            f"{wave_propagator!r}.  Valid choices: "
            f"{sorted(set(_VALID_WAVE_PROPAGATORS))}."
        )
    if slant_correction and wave_propagator not in ('asm', 'rs',
                                                    'rayleigh_sommerfeld'):
        raise ValueError(
            f"apply_real_lens: slant_correction=True is incompatible "
            f"with wave_propagator={wave_propagator!r}.  Use 'asm' or "
            f"'rayleigh_sommerfeld' instead, or drop "
            f"slant_correction.")
    if seidel_correction:
        try:
            n_surf = len(prescription.get('surfaces', []))
        except (AttributeError, TypeError):
            # prescription may not be a dict, or surfaces may be
            # non-len-able; treat as no surfaces and let the
            # length check below raise.
            n_surf = 0
        if n_surf < 2:
            raise ValueError(
                f"apply_real_lens: seidel_correction=True requires a "
                f"prescription with at least 2 surfaces; got "
                f"{n_surf}.")
    if not isinstance(seidel_poly_order, int) or seidel_poly_order <= 0:
        raise ValueError(
            f"apply_real_lens: seidel_poly_order must be a positive "
            f"integer; got {seidel_poly_order!r}.")
    if seidel_poly_order > 12:
        raise ValueError(
            f"apply_real_lens: seidel_poly_order={seidel_poly_order} "
            f"is too large; radial-polynomial fit conditioning "
            f"degrades above 12.")
    if slant_correction and seidel_correction:
        raise ValueError(
            "apply_real_lens: slant_correction=True and "
            "seidel_correction=True are mutually exclusive.  The Seidel "
            "block's model reference is built from the screen the split-step "
            "actually applies, and the two flags replace the SAME per-surface "
            "coefficient, so stacking them double-counts the facet obliquity "
            "(measured 173.5 -> 1488.6 nm rms exit OPD on an 8 mm cemented "
            "doublet with both on).  Pick one, or use "
            "apply_real_lens_traced for a per-pixel ray-traced OPL.")
    _check_no_silent_fold_drop(prescription, fn_name='apply_real_lens')


def _normalise_stop_index(stop_index: Any, n_surfaces: int,
                          fn_name: str = 'apply_real_lens') -> Optional[int]:
    """Resolve ``prescription['stop_index']`` to a valid surface index.

    Returns ``None`` unchanged, normalises a negative index the way Python
    indexing does (``-1`` -> the last surface) and raises a precise
    ``ValueError`` for anything still outside ``[0, n_surfaces)``.

    The range check is not cosmetic.  ``apply_real_lens`` skips the ENTRANCE
    aperture whenever ``stop_index is not None`` and applies the stop only at
    the surface whose loop index equals it, so an out-of-range value matches no
    surface and removes every aperture mask from the call -- measured 1.000 of
    the input power transmitted where the 3 mm stop should pass 0.269, with no
    warning.  ``stop_index=-1`` -- the natural Python spelling for "the last
    surface" -- was in that class before normalisation.

    Shared by :func:`apply_real_lens` and :func:`prepare_real_lens` so the two
    entry points read the key the same way (the latter then refuses a
    mid-train stop outright, which is a separate, documented limitation)."""
    if stop_index is None:
        return None
    if isinstance(stop_index, bool) or not isinstance(
            stop_index, (int, np.integer)):
        raise ValueError(
            f"{fn_name}: prescription['stop_index'] must be an integer "
            f"surface index or None; got {stop_index!r}.")
    idx = int(stop_index)
    if idx < 0:
        idx += int(n_surfaces)
    if not (0 <= idx < int(n_surfaces)):
        raise ValueError(
            f"{fn_name}: prescription['stop_index']={stop_index!r} is out of "
            f"range for a prescription with {n_surfaces} surface(s); it must "
            f"select a surface in [0, {int(n_surfaces) - 1}] (negative "
            f"indices count from the end).  An out-of-range stop matches no "
            f"surface AND suppresses the entrance aperture, i.e. it silently "
            f"removes ALL aperture clipping from the call.")
    return idx


def _check_no_silent_fold_drop(prescription: dict,
                                fn_name: str = 'apply_real_lens') -> None:
    """Raise a precise ValueError if ``prescription`` contains fold
    mirrors that the refractive-only ``apply_real_lens*`` family would
    silently drop.

    A prescription loaded from a .zmx file that contains a fold mirror
    carries it in the ``elements`` list (full element sequence) but
    NOT in the ``surfaces`` list (refracting-surface-only, the only
    thing the apply_real_lens* family iterates over).  Running them on
    the bare prescription propagates the wave along the *unfolded
    equivalent* axis -- this is scalar-physics-correct on-axis when
    every mirror is flat, but silently drops the mirror's curvature
    phase (if any) and the world-frame axis change.

    The caller picks one of two escape hatches:
      (a) Acknowledge the unfolded-equivalent treatment by setting
          ``prescription['allow_unfolded_equivalent'] = True``.
      (b) Use :func:`lumenairy.io.split_prescription_at_mirrors` to
          walk the wave segment-by-segment, applying :func:`apply_mirror`
          at each fold.
    """
    elements = prescription.get('elements')
    if elements is None:
        return
    mirror_count = sum(1 for el in elements
                       if el.get('element_type') == 'mirror')
    if mirror_count == 0:
        return
    if prescription.get('allow_unfolded_equivalent', False):
        return
    raise ValueError(
        f"{fn_name}: prescription has {mirror_count} mirror "
        f"element(s) but {fn_name} only walks refracting surfaces.  "
        f"Running this prescription as-is would silently propagate "
        f"the unfolded-equivalent path and skip the mirror's focusing "
        f"phase (if curved) and world-frame axis change.  Two ways "
        f"to proceed:\n"
        f"  (a) Acknowledge the unfolded-equivalent treatment by "
        f"setting prescription['allow_unfolded_equivalent'] = True.  "
        f"Correct for scalar on-axis fields when every mirror is flat; "
        f"otherwise lossy or wrong.\n"
        f"  (b) Use lumenairy.io.split_prescription_at_mirrors(rx) "
        f"to split the prescription at each fold, then alternate "
        f"{fn_name} (each segment) with apply_mirror (each fold).  "
        f"See Guide-Folded-Designs section 'Wave-optics through a "
        f"fold'.")


def _mirror_surface_indices(surfaces) -> list:
    """Indices of the MIRROR entries in a ``surfaces`` list.

    A mirror is spelled either ``is_mirror=True`` or
    ``glass_after='MIRROR'`` (case-insensitive); both are accepted
    everywhere in the library, so both are recognised here."""
    out = []
    for i, s in enumerate(surfaces or []):
        if not isinstance(s, dict):
            continue
        _ga = s.get('glass_after')
        if bool(s.get('is_mirror', False)) or (
                isinstance(_ga, str) and _ga.upper() == 'MIRROR'):
            out.append(i)
    return out


def _unfold_mirror_surfaces(prescription: dict,
                            fn_name: str = 'apply_real_lens') -> dict:
    """Resolve a mirror that sits directly in ``prescription['surfaces']``.

    ``_check_no_silent_fold_drop`` above inspects the ``elements`` list (what
    ``load_zemax_zmx`` populates) and offers ``allow_unfolded_equivalent`` as
    the documented escape hatch.  A HAND-BUILT prescription that puts the
    mirror straight into ``surfaces`` reaches neither that check nor that key,
    and the refractive walk would treat the mirror as a refractor -- wrong
    sign, wrong focusing phase -- so it has to be caught separately.  It used
    to be caught with an unconditional refusal that did not mention the key at
    all, i.e. the documented option did not work on this spelling of the same
    physics (``ui/waveoptics_dock.py`` had to build the unfolded prescription
    itself before it could set the flag).  The two guards now read the SAME
    key.

    UNFOLDING SEMANTICS, stated precisely.  With the flag set, every mirror
    surface is replaced IN PLACE by an index-neutral FLAT
    (``radius=inf``, ``glass_after := glass_before``, no conic / aspheric /
    biconic / freeform / ``sag_callable`` / ``form_error`` / decenter / tilt),
    keeping its ``clear_aperture`` and ``semi_diameter``.  Replacing rather
    than deleting is what keeps the model honest: the surface count, every gap
    in ``thicknesses`` and both reference planes (the input field sits on
    surface 0's vertex plane, the output on the last surface's) are unchanged,
    and a flat fold imprints exactly ``(n - n) * sag = 0``, so for a scalar
    on-axis field through a FLAT fold this is exact.  What it drops is what the
    flag's own message has always said it drops: a curved mirror's focusing
    phase, a tilted/decentred mirror's world-frame axis change, and the
    vignetting geometry of the folded arm.  Those are named in the warning.

    Returns the prescription to use -- ``prescription`` itself when it holds no
    mirror surface (no copy, no behaviour change), otherwise a shallow copy
    carrying the substituted ``surfaces`` list.
    """
    surfaces = prescription.get('surfaces') or []
    idx = _mirror_surface_indices(surfaces)
    if not idx:
        return prescription
    if not prescription.get('allow_unfolded_equivalent', False):
        raise ValueError(
            f"{fn_name}: prescription has {len(idx)} mirror surface(s) at "
            f"indices {idx} -- {fn_name} only walks REFRACTING surfaces.  "
            f"Running this prescription as-is would treat the mirror as a "
            f"refractor (wrong sign / wrong focusing phase) and propagate "
            f"along the unfolded-equivalent axis.  Two ways to proceed:\n"
            f"  (a) Acknowledge the unfolded-equivalent treatment by setting "
            f"prescription['allow_unfolded_equivalent'] = True -- the same "
            f"key {fn_name} already honours for an 'elements'-borne fold.  "
            f"Each mirror surface then becomes an index-neutral FLAT at its "
            f"own vertex plane, so every gap and both reference planes are "
            f"unchanged; exact for a scalar on-axis field through a FLAT "
            f"fold, and it drops a curved mirror's focusing phase and the "
            f"world-frame axis change otherwise.\n"
            f"  (b) Use lumenairy.io.split_prescription_at_mirrors(rx) to "
            f"split the prescription at each fold, then alternate {fn_name} "
            f"(each segment) with apply_mirror (each fold).  See "
            f"Guide-Folded-Designs section 'Wave-optics through a fold'.")
    _NEUTRALISED = ('conic', 'aspheric_coeffs', 'aspheric_coeffs_y',
                    'radius_y', 'conic_y', 'freeform_type', 'freeform_coeffs',
                    'sag_callable', 'form_error', 'decenter', 'tilt')
    new_surfaces = list(surfaces)
    curved, shifted = [], []
    for i in idx:
        s = dict(surfaces[i])
        _R = s.get('radius')
        if _R is not None and np.isfinite(_R):
            curved.append(i)
        if any(s.get(_k) is not None for _k in
               ('conic', 'aspheric_coeffs', 'freeform_type', 'sag_callable')):
            if i not in curved:
                curved.append(i)
        for _k in ('decenter', 'tilt'):
            _v = s.get(_k)
            if _v is not None and tuple(float(q) for q in _v) != (0.0, 0.0):
                shifted.append(i)
                break
        for _k in _NEUTRALISED:
            s.pop(_k, None)
        s['radius'] = np.inf
        s['glass_after'] = s.get('glass_before')
        s['is_mirror'] = False
        new_surfaces[i] = s
    import warnings as _warnings
    _warnings.warn(
        f"{fn_name}: prescription['allow_unfolded_equivalent'] is set, so the "
        f"{len(idx)} mirror surface(s) at indices {idx} are being walked as "
        f"the UNFOLDED EQUIVALENT: each becomes an index-neutral flat at its "
        f"own vertex plane (every gap and both reference planes unchanged), "
        f"which is exact for a scalar on-axis field through a FLAT fold."
        + (f"  DROPPED: the focusing phase of the CURVED mirror(s) at "
           f"{curved}." if curved else "")
        + (f"  DROPPED: the world-frame axis change of the "
           f"decentred/tilted mirror(s) at {sorted(set(shifted))}."
           if shifted else "")
        + "  Use lumenairy.io.split_prescription_at_mirrors(rx) with "
          "apply_mirror at each fold to carry them.",
        RuntimeWarning, _WARN_STACKLEVEL)
    out = dict(prescription)
    out['surfaces'] = new_surfaces
    return out


# ---------------------------------------------------------------------------
# SCREEN OBLIQUITY -- the closed-form angular correction to the sag screen.
# ---------------------------------------------------------------------------
# The default screen imprints ``(n2 - n1) * sag(x, y)`` on the surface's VERTEX
# PLANE.  The angular-spectrum steps between screens carry the angular optical
# path of the GAPS exactly (a plane-parallel plate is machine-exact at every
# tilt -- BUILD_ANGLE_AWARE_LENS_2026_08_11 S2), so the only angle-blind piece
# left is the sag screen itself: it applies the same OPD however obliquely the
# ray crosses the sag.  That is the ``~ sag * theta**2`` bound this function's
# own docstring has always quoted, and it is the ONLY term corrected here.
#
# THE AXIAL-TRANSLATION IDENTITY (derived in
# ``docs/audits/BUILD_SCREEN_OBLIQUITY_2026_08_11.md`` S2; exact, no expansion).
# Take a PLANE facet with unit normal ``nu``, media ``n1 -> n2``, sitting a
# height ``s`` above the vertex plane, between fixed reference planes.  Moving
# it down onto the vertex plane leaves the exit ray direction unchanged (a
# plane refracts identically wherever it sits along its own normal) and changes
# the EXIT-REFERENCED eikonal by exactly
#
#     Lam(facet at height s) = Lam(facet at height 0) + s * (pz1 - pz2)        (2)
#
# with ``pz1 = n1 cos(alpha_in)`` and ``pz2 = n2 cos(alpha_out)`` the AXIAL
# components of the optical momentum before / after refraction, both measured
# to the Z-AXIS (not to the facet normal).  Proof: split the "facet at height
# s" system into [n1 slab of thickness s] + [facet at height 0 over the
# remaining thickness] + [remove an n2 slab of thickness s]; the middle system
# is translation-invariant so its total eikonal minus ``p_out . x_out`` is
# constant, and the two slabs contribute
# ``n1 s / cos a1 - p_in s tan a1 = s n1 cos a1`` and ``-s n2 cos a2``.
#
# So the eikonal-exact screen OPD (convention ``exp(-i k0 OPD)``) is
#
#     OPD_i(x, y) = (pz2 - pz1) * sag_i(x, y)                                  (3)
#
# -- equation (1) of the ``surface_model='displaced'`` block above, now derived
# rather than back-projected, and EXACT for a locally planar facet.  ``pz2``
# comes from exact vector Snell at the local facet normal
# ``nu = (-grad sag, 1)/sqrt(1 + |grad sag|^2)``.
#
# WHAT IS APPLIED HERE is (3) MINUS its carrier-free value, so the correction
#
#     dOPD_i = [ (pz2 - pz1)|_{p0 + q} - (pz2 - pz1)|_{p0} ] * sag_i           (4)
#
# * is EXACTLY zero for a plane-parallel plate (``sag == 0``) at every tilt;
# * is EXACTLY zero for a carrier-free call (``q == 0``) -- the byte-null;
# * leaves the model's documented NORMAL-INCIDENCE accuracy ceiling untouched
#   (that is the ``slant_correction`` / ``surface_model='displaced'`` axis);
# * to leading order equals ``sag * (n2-n1)/(2 n1 n2) * (|p0+q|^2 - |p0|^2)``,
#   i.e. ``(n-1) sag theta^2 / 2n`` for a collimated air-side surface -- the
#   docstring's ``sag * theta**2`` bound with its exact prefactor.
#
# ``q`` is the carrier's local transverse momentum (its eikonal gradient, i.e.
# the direction cosines) and ``p0`` is the carrier-free momentum the screen
# model itself accumulates, ``-sum_{j<i} (n2-n1) grad sag_j``, evaluated at the
# same field point.  Both are closed form: NO ray trace, NO map, NO cache.
#
# R1 -- THE ANGLE-BLIND MOMENTUM KICK, AND THE DRIFT IT IS SEEN THROUGH
# (v5.35.0; derived and measured in
# ``docs/audits/BUILD_R1_WIRING_2026_08_12.md`` S1).
#
# Equation (4) fixes the screen's OPD VALUE.  The screen also has to DEFLECT:
# it kicks the field by ``-grad OPD = -(n2 - n1) grad sag``, while the exact
# tangent facet kicks by ``-dz grad sag`` with ``dz = pz2 - pz1`` the SAME
# exact vector-Snell quantity equation (3) is built from.  The kick is
# therefore wrong by ``-Lam grad sag``, ``Lam = (n2 - n1) - dz``, and that
# error is angle-dependent because ``dz`` is.
#
# Writing the exit-plane model error as ``D = dLam - p . dx`` (the eikonal
# difference plus the landing error carried back at the exit momentum, which
# is exactly how the exact-ray oracle scores it) splits the defect into an OPD
# channel and a DEFLECTION channel.  Measured on design 121 group 5, 3 mm
# pupil, 54.9 mrad: the deflection channel alone is 0.0125 w, and the OPD
# channel carries 0.0814 w of the corrected screen's 0.0870 w residual.  The
# OPD channel is the deflection defect seen through the ray DRIFT: the
# carrier-free screen error
#
#     E_i(x) = [ (n2 - n1) - dz(p0_i) ] * sag_i(x)                          (5)
#
# (whose gradient IS the angle-blind kick error above) is sampled where the
# ray actually crosses surface i, and the carrier moves that crossing by
#
#     U_i = sum_{j<i} t_j [ (p0 + q)/pz_a - p0/pz_b ]_j                     (6)
#
# -- the transverse drift the carrier adds over the gaps BEFORE surface i.
# A carrier-free error sampled at a carrier-shifted point is an ANGULAR error,
# and it is the term that bounded equation (4) at 2.9x on the fastest
# elements.  Cancelling it costs one more screen term
#
#     dOPD_R1,i(x) = - U_i . grad E_i(x)                                    (7)
#
# which is IDENTICALLY ZERO without a carrier (``U == 0``), identically zero
# for a plate (``sag == 0`` so ``E == 0``), and needs no ray trace: ``U``
# accumulates on the grid beside ``p0``.  ``p0`` is read at the CARRIER-FREE
# ray's own position when the drift is advanced (``p0 - (U . grad) p0``) --
# the element re-images its own drift, worth 14 % of the term on design 121
# group 5.
#
# MEASURED (exact-ray oracle, common-mode controlled at the exit plane;
# docs/audits/BUILD_R1_WIRING_2026_08_12.md S2): with (7), design 121 group 5
# goes 0.25848 -> 0.01905 waves rms (13.6x, against 2.9x for (4) alone), and
# the single-facet gains of S3.2 are unchanged (a lone facet has no gap in
# front of it, so ``U == 0`` and (7) is exactly zero there).  What is left is
# the deflection channel proper -- ``sag grad dz`` acting through the gaps
# AFTER the surface -- which is NOT the gradient of any scalar and so cannot
# be carried by a screen of the form ``f(x, y) sag(x, y)``; it is 0.0125 w on
# that element and it is what the guard's residual budget accounts for.

#: The legal ``screen_obliquity`` values, for MESSAGES only.
#: ``_check_screen_obliquity_support`` validates by hand and deliberately does
#: NOT test membership here: ``1 in ('auto', True, False)`` is True (``1 ==
#: True``), so a ``screen_obliquity=1`` would masquerade as ``True``.  The tuple
#: was dead; it is kept as the single place the accepted set is spelled and is
#: now read by the error message, so the two cannot drift apart.
_VALID_SCREEN_OBLIQUITY = ('auto', True, False)
_VALID_SCREEN_OBLIQUITY_POLICY = ('warn', 'error', 'silent')
# Documented tolerance for the guard: lambda/20 of piston-and-tilt-free
# wavefront error.  Below it the screen's angle-blindness is inside the
# analytic model's own normal-incidence ceiling on every element measured in
# the campaign; above it the traced path is the shipped answer.
_SCREEN_OBLIQUITY_TOL_WAVES = 0.05
# With the correction applied (equation 4 AND the R1 term, equation 7), the
# leftover is the DEFLECTION CHANNEL PROPER -- ``sag grad dz`` acting through
# the gaps after the surface, which is not the gradient of any scalar and so
# cannot be carried by a screen at all.  Measured ratios (residual /
# uncorrected) across the campaign's powered cases, re-derived exactly from
# ``_screen_obl_d121.json`` rather than from its printed 5-decimal digits
# (FIX_FINAL_WAVE_2026_08_13 S4.2):
#
#   design 121 group 5   r = 1 / 2 / 3 mm   0.040260 / 0.043894 / 0.047966
#   design 121 group 4   r = 1 / 2 / 3 mm   0.037474 / 0.050153 / 0.055224
#   design 121 group 2   r = 1 / 2 / 3 mm   0.041997 / 0.035100 / 0.029664
#   design 121 group 3   r = 1 / 2 / 3 mm   0.001773 / 0.001930 / 0.002033
#   single spherical surfaces, 10-100 mrad, N-BK7 / N-SF11, R = +-25/50 mm
#                                           0.0012 - 0.0064
#   plates (groups 0, 1)                    exactly 0 (sag == 0, so E == 0)
#
# so the WORST is 0.055224 (group 4 at 3 mm) and the worst that is materially
# large is 0.047966 (group 5 at 3 mm -- the binding case; group 4's absolute
# error is 0.00048896 waves, 102x inside the tolerance, which is why its ratio
# is not the one to design to).  0.10 therefore keeps 1.81x over the worst
# ratio measured anywhere and 2.09x over the binding case.  (Pre-R1 this was
# 0.40, from a worst of 0.351 on the same group.)
#
# THE CONSTANT IS BOUNDED ON BOTH SIDES, which is what makes it a choice rather
# than a one-sided margin.  The guard fires on ``estimate * FRAC > TOL``, so
# each shipped fixture's estimate names a FRAC at which its disposition flips:
#
#   _out_of_envelope_case  estimate 1.00880 w  -> must fire:  FRAC > 0.0496
#   design 121 group 5     estimate 0.23910 w  -> must not:   FRAC < 0.2091
#   _steep_case            estimate 0.12860 w  -> must not:   FRAC < 0.3888
#
# Every disposition the suite pins is therefore unchanged for any FRAC in
# (0.0496, 0.2091), a 4.2x-wide window, and 0.10 sits essentially at its
# geometric centre (sqrt(0.0496 * 0.2091) = 0.1018).  Raising it to 0.15 would
# buy 2.72x over the worst measured ratio and cost margin the other way (the
# design-of-record false alarm would sit 1.39x away instead of 2.09x); it was
# derived and NOT taken, because no value inside the window addresses the
# actual open item, which is:
#
# WHAT IS NOT BOUNDED.  Every case above is a ROTATIONALLY SYMMETRIC surface.
# The leftover has NOT been measured on a decentred / tilted / biconic /
# freeform element, and that -- not the size of the margin -- is the reason to
# prefer :func:`apply_real_lens_traced` on one.  What IS settled is that the
# number is not an arithmetic accident: the whole ladder reproduces to every
# printed digit, and `_screen_obl_d121.json` / `_screen_obl_sphere.json`
# reproduce with zero numeric difference, across Windows/MKL/py3.14/numpy
# 2.4.4 and WSL/OpenBLAS/py3.12/numpy 2.4.6 at 1, 2 and 8 threads (6
# configurations).
_SCREEN_OBLIQUITY_RESIDUAL_FRAC = 0.10
# Floor on ``pz**2`` inside the drift step, so a marginally-propagating pixel
# cannot divide by zero before its ``ok`` mask zeroes it.
_SCREEN_DRIFT_MIN_PZ_SQ = 1e-12


def _carrier_is_geometric(carrier) -> bool:
    """True when this carrier states DIRECTION COSINES (so the consumer has to
    multiply by ``n1`` to get the transverse OPTICAL momentum), False when it
    already states the optical momentum.

    * :class:`~._lens_traced.TiltedCarrier` -- the traced module launches UNIT
      rays along its ``(L, M)``: direction cosines.
    * a signed scalar conjugate -- ``W = sign(s)(sqrt(x^2+y^2+s^2) - |s|)`` is
      a geometric distance, so ``grad W = sin alpha``: a direction cosine.
    * ``'auto'`` -- fits ``angle(E[:, 1:] conj(E[:, :-1])) / (k0 dx)``, and the
      field's phase is ``k0 * S`` with ``S`` the OPTICAL path, so the reading
      IS ``p_x``.
    * an explicit wavefront ndarray -- documented as "reference phase =
      k0 * W", so ``grad W`` is likewise optical.

    Scaling the last two by ``n1`` double-counts it -- x1.5168 in N-BK7, which
    made the "corrected" screen worse than the uncorrected one at 100 mrad
    (0.0971 vs 0.0784 waves).  Harmless in air (n1 = 1), which is why every
    shipped fixture missed it.  ONE function so the whole-grid field and the
    row-banded evaluators cannot drift apart (they did: the banded arm kept the
    n1 and broke the byte-identity the banded path is pinned on)."""
    from ._lens_traced import TiltedCarrier
    return (isinstance(carrier, TiltedCarrier)
            or (isinstance(carrier, (int, float, np.floating, np.integer))
                and not isinstance(carrier, bool)))


def _screen_obliquity_angle_field(carrier, E_in, wavelength, dx, dy, Nx, Ny,
                                  n_medium=1.0):
    """Transverse OPTICAL MOMENTUM ``(qx, qy) = n1 * (L, M)`` for the input
    congruence ``carrier``, in the medium the carrier propagates in.

    THE UNITS ARE THE WHOLE POINT (VERIFY_ARCHITECTURE P1-1).  A carrier's
    ``(L, M)`` are DIRECTION COSINES of a UNIT ray vector -- ``L^2 + M^2 +
    N^2 = 1``.  Its consumer :func:`_facet_axial_momenta` closes the momentum
    triangle on the OPTICAL momentum ``p = n * d``, i.e. ``pz = sqrt(n1^2 -
    |p_t|^2)`` and ``|p_t| < n1`` for a propagating ray.  Those are not the
    same vector unless ``n1 == 1``.  Feeding a bare direction cosine in is a
    silent factor-``n1`` error in the transverse momentum, and it is silent
    precisely because every prescription the campaign shipped starts in air,
    where the two coincide.

    The companion accumulator ``_obl_p0*`` in :func:`apply_real_lens` IS a
    true optical momentum (it accumulates ``-(n2 - n1) * grad sag``), so the
    two terms that get added together were in different units.

    Measured on an immersed R = 19.6 mm N-SSK2 singlet at 54.9 mrad,
    exit-plane rms waves against an exact vector-Snell trace:

    .. code-block:: text

        first medium   blind      shipped q=L    correct q=n1*L
        air            0.010922   0.000033       0.000033     (n1 = 1: same)
        N-BK7          0.002765   0.001238       0.000006     2.2x -> 474x
        N-SF57         0.006510   0.003704       0.000012     1.8x -> 548x

    ``n_medium`` is the index of ``surfaces[0]['glass_before']``: the
    transverse optical momentum is conserved across the stack (the facet
    kicks are what ``_obl_p0*`` accumulates), so this is measured once at
    the medium the carrier is actually defined in and carried forward.  It
    multiplies ONLY the two geometric congruences -- see the branch comment in
    the body; ``'auto'`` and an explicit wavefront ndarray already deliver
    optical momentum and are passed through unscaled.

    Uses the traced path's own carrier vocabulary
    (:func:`~._lens_traced._compute_carrier`): a :class:`TiltedCarrier`, a
    signed scalar conjugate, ``'auto'`` (a fit of ``E_in``), or an explicit
    wavefront ndarray.  A congruence whose direction cosines are CONSTANT over
    the grid (a collimated tilt) collapses to two floats, so the correction
    costs no full-grid momentum arrays in the common case."""
    from ._lens_traced import TiltedCarrier, _compute_carrier
    n1 = float(n_medium)
    # Only the two GEOMETRIC congruences need the n1 -- see
    # :func:`_carrier_is_geometric`, which the row-banded evaluators share.
    _q_scale = n1 if _carrier_is_geometric(carrier) else 1.0
    if (isinstance(carrier, TiltedCarrier)
            and not np.isfinite(float(carrier.R))):
        # A collimated tilt has constant direction cosines everywhere, so take
        # them analytically -- ``_compute_carrier`` would build three full-grid
        # float64 arrays (~1.6 GB at N = 8192) to return two numbers.
        return _q_scale * float(carrier.L), _q_scale * float(carrier.M)
    xax = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    yax = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    Xg, Yg = np.meshgrid(xax, yax)
    _W, grad_fn, _w = _compute_carrier(carrier, E_in, wavelength, dx, Xg, Yg,
                                       dy=dy)
    L, M = grad_fn(Xg, Yg)
    L = np.asarray(L, dtype=np.float64) * _q_scale
    M = np.asarray(M, dtype=np.float64) * _q_scale
    if L.ndim and float(np.ptp(L)) == 0.0 and float(np.ptp(M)) == 0.0:
        return float(L.flat[0]), float(M.flat[0])
    return L, M


def _facet_axial_momenta(px, py, gx, gy, n1, n2, xp, inv=None):
    """``(pz2 - pz1, ok)`` -- the change in the AXIAL optical-momentum
    component across exact vector refraction at the local facet whose unit
    normal is ``nu = (-grad sag, 1) / sqrt(1 + |grad sag|**2)``, plus a mask
    that is False where the ray is evanescent in ``n1`` or totally internally
    reflected at the facet (there the correction is dropped rather than
    clamped -- a clamped cosine is a wrong OPD, and the shipped screen is the
    safe neutral).  ``inv = nu_z`` may be passed in when the caller is
    evaluating both arms on the same facet."""
    if inv is None:
        inv = 1.0 / xp.sqrt(1.0 + gx * gx + gy * gy)
    p_sq = px * px + py * py
    ok_in = p_sq < n1 * n1
    pz1 = xp.sqrt(xp.maximum(n1 * n1 - p_sq, 0.0))
    a_dot = (-gx * px - gy * py + pz1) * inv        # (n1 d_in) . nu
    b_sq = n2 * n2 - n1 * n1 + a_dot * a_dot
    ok = ok_in & (b_sq > 0.0)
    b = xp.sqrt(xp.maximum(b_sq, 0.0))
    return (b - a_dot) * inv, ok


def _screen_obliquity_delta(sag, gx, gy, p0x, p0y, qx, qy, n1, n2, xp):
    """Equation (4): the ANGULAR part of the exact thin-facet screen OPD.

    Zero wherever ``sag`` is zero, wherever the carrier momentum is zero, and
    wherever either arm's refraction is non-propagating."""
    inv = 1.0 / xp.sqrt(1.0 + gx * gx + gy * gy)     # nu_z, shared by both arms
    dz_a, ok_a = _facet_axial_momenta(p0x + qx, p0y + qy, gx, gy, n1, n2, xp,
                                      inv)
    dz_b, ok_b = _facet_axial_momenta(p0x, p0y, gx, gy, n1, n2, xp, inv)
    d = (dz_a - dz_b) * sag
    ok = ok_a & ok_b
    # The all-propagating case is the overwhelmingly common one; test it with
    # one reduction rather than paying a full-grid select every surface.
    if bool(xp.all(ok)) and bool(xp.all(xp.isfinite(d))):
        return d
    return xp.where(ok & xp.isfinite(d), d, xp.zeros((), dtype=d.dtype))


def _screen_coeff_error(sag, gx, gy, p0x, p0y, n1, n2, xp):
    """Equation (5): ``E = [(n2 - n1) - dz(p0)] * sag`` -- the CARRIER-FREE
    error of the shipped screen's own coefficient, in metres of OPD.

    Its GRADIENT is the screen's angle-blind deflection error: the shipped
    screen kicks the field by ``-(n2 - n1) grad sag`` where the exact tangent
    facet kicks by ``-dz grad sag``, so ``-grad E`` is (to the order in which
    ``dz`` varies slowly across the sag) the transverse momentum the screen
    fails to impart.  R1 is that error carried over the carrier's own ray
    drift; see the module-level derivation.

    Carrier-free by construction -- it does NOT read ``q`` -- which is why the
    R1 term it feeds vanishes identically when the drift does."""
    dz_b, ok = _facet_axial_momenta(p0x, p0y, gx, gy, n1, n2, xp)
    e = ((n2 - n1) - dz_b) * sag
    if bool(xp.all(ok)) and bool(xp.all(xp.isfinite(e))):
        return e
    return xp.where(ok & xp.isfinite(e), e, xp.zeros((), dtype=e.dtype))


def _screen_drift_step(p0x, p0y, pbx, pby, qx, qy, t, n_gap, xp):
    """Equation (6), one gap: the transverse displacement a homogeneous gap
    ADDS to the carrier's ray relative to the carrier-free ray,
    ``t * (p_a/pz_a - p_b/pz_b)``.

    ``(p0x, p0y)`` is the screen model's own accumulated transverse momentum
    at the field point and ``(pbx, pby)`` the same quantity at the
    CARRIER-FREE ray's own position (``p0`` shifted back by the drift so far);
    they differ only once a drift exists, and that feedback -- the element
    re-imaging its own drift -- is worth 14 % of the term on design 121 group
    5.  ``q`` is the carrier's transverse optical momentum, in the same units
    (see :func:`_screen_obliquity_angle_field`).

    Pixels where either arm is evanescent in the gap take a ZERO step rather
    than a clamped one, matching :func:`_screen_obliquity_delta`: a clamped
    cosine is a wrong drift, and no drift is the safe neutral."""
    n_sq = n_gap * n_gap
    pax, pay = p0x + qx, p0y + qy
    s_a = pax * pax + pay * pay
    s_b = pbx * pbx + pby * pby
    ok = (s_a < n_sq) & (s_b < n_sq)
    pza = xp.sqrt(xp.maximum(n_sq - s_a, _SCREEN_DRIFT_MIN_PZ_SQ))
    pzb = xp.sqrt(xp.maximum(n_sq - s_b, _SCREEN_DRIFT_MIN_PZ_SQ))
    zero = xp.zeros((), dtype=xp.asarray(pza).dtype)
    dux = xp.where(ok, t * (pax / pza - pbx / pzb), zero)
    duy = xp.where(ok, t * (pay / pza - pby / pzb), zero)
    return dux, duy


def _screen_drift_opd(sag, gx, gy, p0x, p0y, n1, n2, ux, uy, dx, dy, xp):
    """Equation (7): ``-U . grad E`` -- the R1 screen term.

    Zero wherever the sag is zero (a plate has no coefficient error to carry),
    zero wherever the drift is zero (no carrier, or the first surface, which
    has no gap in front of it), and zero wherever either arm's refraction is
    non-propagating."""
    e_err = _screen_coeff_error(sag, gx, gy, p0x, p0y, n1, n2, xp)
    ey, ex = xp.gradient(e_err, dy, dx)
    d = -(ux * ex + uy * ey)
    if bool(xp.all(xp.isfinite(d))):
        return d
    return xp.where(xp.isfinite(d), d, xp.zeros((), dtype=d.dtype))


def _screen_drift_opd_rows(sag_h, gx_h, gy_h, p0x_h, p0y_h, n1, n2, ux_b, uy_b,
                           dx, dy, xp, lo, hi):
    """Row-banded :func:`_screen_drift_opd` (v5.35.3).

    Equation (7) is the GRADIENT of the coefficient error, so a band cannot be
    evaluated from the band alone: ``e_err`` has to exist one row EITHER SIDE
    of the band for ``xp.gradient``'s central differences to match the
    whole-grid stencil.  The caller therefore passes ``sag_h`` / ``gx_h`` /
    ``gy_h`` / ``p0x_h`` / ``p0y_h`` already widened to rows
    ``[max(0, r0-1) : min(Ny, r1+1)]``, and ``lo:hi`` selects the band inside
    that halo.  Rows 0 and ``Ny-1`` keep their natural one-sided stencil
    because the halo is clipped at the true array edge -- exactly what the
    whole-grid ``xp.gradient`` does there.

    ``ux_b`` / ``uy_b`` are the drift at the BAND rows only (the multiply is
    pointwise).  Byte-identical to :func:`_screen_drift_opd` restricted to the
    band: the interior reduction is elementwise-equivalent (``where`` over an
    all-true mask is the identity), so a defect in one band cannot change
    another band's arithmetic."""
    e_err = _screen_coeff_error(sag_h, gx_h, gy_h, p0x_h, p0y_h, n1, n2, xp)
    ey, ex = xp.gradient(e_err, dy, dx)
    d = -(ux_b * ex[lo:hi] + uy_b * ey[lo:hi])
    if bool(xp.all(xp.isfinite(d))):
        return d
    return xp.where(xp.isfinite(d), d, xp.zeros((), dtype=d.dtype))


def _screen_obliquity_row_evaluator(carrier, dx, dy, Nx, Ny, n_medium=1.0):
    """A ``rows(r0, r1) -> (qx_band, qy_band)`` callable for the carrier
    momentum field, or ``None`` when this carrier has no closed form that can
    be evaluated a row-band at a time (v5.35.3).

    :func:`_screen_obliquity_angle_field` builds two full-grid ``(Ny, Nx)``
    float64 momentum arrays for a non-collimated carrier -- 2 grids that the
    row-banded sag path otherwise never needs (+17 GB at N = 32768).  A
    :class:`~._lens_traced.TiltedCarrier` is ANALYTIC in ``(x, y)``
    (:func:`~._lens_traced._tilted_carrier_parts` is pointwise: no reduction,
    no grid lookup, no finite difference), so its rows can be evaluated on
    demand from the same axis vectors.

    The returned band is byte-identical to the corresponding row slice of the
    whole-grid field: the y axis is rebuilt as ``arange(r0, r1) - Ny/2``, which
    is exactly ``(arange(Ny) - Ny/2)[r0:r1]`` in IEEE terms, and the SAME
    ``_tilted_carrier_parts`` -> ``asarray(float64) * n1`` chain runs on it.

    Returns ``None`` for the collimated (``R = inf``) tilt -- the caller's
    two-float fast path already costs nothing -- and for the ndarray / 'auto'
    / scalar-conjugate carriers, whose ``_compute_carrier`` set-up is itself
    whole-grid; those keep the materialised field and are simply sliced."""
    from ._lens_traced import TiltedCarrier, _tilted_carrier_parts
    if not isinstance(carrier, TiltedCarrier):
        return None
    if not np.isfinite(float(carrier.R)):
        return None
    # A TiltedCarrier always states direction cosines, so the n1 always
    # applies here -- but take it from the shared predicate anyway, so this
    # arm cannot drift from the whole-grid field the way it did when the
    # 'auto' / ndarray scaling was fixed in one place only.
    n1 = float(n_medium) if _carrier_is_geometric(carrier) else 1.0
    xax = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx

    def rows(r0, r1):
        yax = (np.arange(r0, r1, dtype=np.float64) - Ny / 2) * dy
        Xg, Yg = np.meshgrid(xax, yax)
        _W, L, M = _tilted_carrier_parts(carrier, Xg, Yg)
        return (np.asarray(L, dtype=np.float64) * n1,
                np.asarray(M, dtype=np.float64) * n1)

    return rows


def _screen_obliquity_rows_any(carrier, E_in, wavelength, dx, dy, Nx, Ny,
                               n_medium=1.0):
    """A ``rows(r0, r1) -> (qx_band, qy_band)`` for ANY non-collimated carrier
    (v5.40), or ``None`` when the caller's two-float fast path already applies.

    :func:`_screen_obliquity_row_evaluator` is the narrow version of this: it
    bands only the :class:`~._lens_traced.TiltedCarrier`, and it declines the
    ``'auto'`` / scalar-conjugate / ndarray congruences on the grounds that
    "``_compute_carrier``'s set-up is itself whole-grid".  That is true of the
    SET-UP and false of the EVALUATION, and the distinction is worth 7 float64
    grids on the route that matters:

    .. code-block:: text

        _screen_obliquity_angle_field, non-collimated carrier
          Xg, Yg = meshgrid(...)                          2 grids
          _compute_carrier -> W_full                      1 grid   DISCARDED
          L, M = grad_fn(Xg, Yg)                          2 grids
          asarray(L, float64) * n1                        2 grids
                                                          ------
          7 live to deliver 2, and at N = 32768 that is 60 GB to deliver 17

    Only the fit itself (for ``'auto'``: a global least-squares over the bright
    support) is irreducibly whole-grid.  Once its coefficients exist,
    ``grad_fn`` is POINTWISE -- a polynomial in ``(x, y)`` for ``'auto'``, a
    closed-form sphere for a scalar conjugate, an index lookup for an ndarray
    -- so a band can be evaluated on demand from the same axis vectors.

    **Byte-identity.**  ``np.meshgrid(xax, yax[r0:r1])`` is exactly the
    ``[r0:r1]`` slice of ``np.meshgrid(xax, yax)`` (the same IEEE values in
    the same order), ``grad_fn`` is pointwise, and the same
    ``asarray(., float64) * n1`` chain runs on the result.  So each band is
    bit-for-bit the corresponding rows of the whole-grid field.  The one
    thing a band CANNOT reproduce is the whole-grid collapse-to-two-floats
    that :func:`_screen_obliquity_angle_field` performs when ``ptp`` is zero
    on both components -- and that collapse is observable (a Python float
    seed and a float64 array of the same value promote differently under
    NEP 50), so the caller reproduces it as a band-wise reduction rather than
    dropping it.

    ``W_full`` is never built: ``need_W=False`` plus zero-copy
    ``np.broadcast_to`` coordinate views mean the whole-grid coordinate stack
    never allocates either.
    """
    rows = _screen_obliquity_row_evaluator(carrier, dx, dy, Nx, Ny,
                                           n_medium=n_medium)
    if rows is not None:
        return rows
    from ._lens_traced import TiltedCarrier, _compute_carrier
    if (isinstance(carrier, TiltedCarrier)
            and not np.isfinite(float(carrier.R))):
        return None                     # collimated: two floats, already free
    # Same scaling rule as the whole-grid field: ONLY the geometric
    # congruences.  This band must be byte-identical to the corresponding row
    # slice of that field, so the two cannot use different rules.
    n1 = float(n_medium) if _carrier_is_geometric(carrier) else 1.0
    xax = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    yax = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    # Zero-copy stand-ins.  With ``need_W=False`` these are read for their
    # SHAPE (and, on the ndarray branch, compared against it); nothing indexes
    # them, so no grid is materialised.
    _Xb = np.broadcast_to(xax[None, :], (Ny, Nx))
    _Yb = np.broadcast_to(yax[:, None], (Ny, Nx))
    try:
        _W, grad_fn, _w = _compute_carrier(
            carrier, E_in, wavelength, dx, _Xb, _Yb, need_W=False, dy=dy)
    except (TypeError, ValueError):
        # An unrecognised congruence, or one whose set-up genuinely needs a
        # writable coordinate grid: fall back to the whole-grid field.
        return None

    def rows(r0, r1):
        Xg, Yg = np.meshgrid(xax, yax[r0:r1])
        L, M = grad_fn(Xg, Yg)
        return (np.asarray(L, dtype=np.float64) * n1,
                np.asarray(M, dtype=np.float64) * n1)

    return rows


def _screen_obliquity_pupil_radius(prescription, Nx, Ny, dx, dy):
    """The disc the guard scores its wavefront estimate over: the declared
    aperture, else the widest per-surface semi-diameter, else the grid's
    inscribed radius."""
    ap = prescription.get('aperture_diameter')
    if ap:
        return float(ap) / 2.0
    semis = [s.get('semi_diameter') for s in (prescription.get('surfaces') or [])
             if isinstance(s, dict) and s.get('semi_diameter')]
    if semis:
        return max(float(v) for v in semis)
    return 0.5 * min(Nx * dx, Ny * dy)


def _screen_obliquity_rms_waves(field, X, Y, r_pupil, wavelength, xp):
    """Piston-and-tilt-free rms of ``field`` [m] over the pupil disc, in waves.

    Solved through the 3x3 normal equations on scaled coordinates rather than
    a least-squares factorisation of an ``(N**2, 3)`` design matrix, so the
    estimator costs three grid reductions instead of a dense solve."""
    m = (X * X + Y * Y) <= r_pupil * r_pupil
    n = float(xp.count_nonzero(m))
    if n < 4.0 or r_pupil <= 0.0:
        return 0.0
    u = xp.where(m, X / r_pupil, 0.0)
    v = xp.where(m, Y / r_pupil, 0.0)
    f = xp.where(m, field, 0.0)
    basis = (xp.where(m, xp.ones((), dtype=u.dtype), 0.0), u, v)
    A = np.array([[float(xp.sum(bi * bj)) for bj in basis] for bi in basis])
    b = np.array([float(xp.sum(bi * f)) for bi in basis])
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = np.zeros(3)
    res = f - (c[0] * basis[0] + c[1] * u + c[2] * v)
    return float(np.sqrt(float(xp.sum(res * res)) / n)) / float(wavelength)


# ---------------------------------------------------------------------------
# ROUTE 3 -- the per-pixel TANGENT-FACET screen (surface_model='tangent_facet')
# ---------------------------------------------------------------------------
# The 'thin' screen imprints ``(n2 - n1) * sag`` on the VERTEX plane: one facet
# coefficient for the whole pupil, blind both to the local ray angle and to the
# local facet tilt.  ``carrier=`` repairs its ANGULAR part as a difference
# against its own zero-angle value (equations 4 and 7 above).  This model
# replaces the approximation instead of correcting it, and therefore needs no
# carrier: a steep facet is angle-wrong even at normal arrival.
#
# Write the surface, near a pixel, as the plane facet tangent to it where THAT
# PIXEL'S RAY meets it.  For a plane facet at height ``s`` above the vertex
# plane, with both sides referenced to that plane, the screen a wave model must
# imprint is EXACTLY
#
#     OPD = dz * s ,      dz = pz2 - pz1 ,                                 (T1)
#     pz1 = sqrt(n1^2 - |p|^2),   pz2 from exact vector Snell at
#     nu = (-grad sag, 1) / sqrt(1 + |grad sag|^2).
#
# (T1) is not an expansion.  Scored against ``S_in - S_out`` built from exact
# ray algebra for a tilted plane facet under a plane wave, it is right to 1e-16
# RELATIVE at every slope (to 0.24), index ratio (including n1 > n2) and skew
# tried.  It is the axial-translation identity of
# BUILD_SCREEN_OBLIQUITY_2026_08_11 S2.2, used as the WHOLE screen rather than
# differenced against its own zero-angle value -- which is what lets it carry
# the normal-incidence steep-facet error too.
#
# Two things a real surface does that a plane facet does not:
#
#   1. THE RAY MEETS THE SURFACE, NOT THE VERTEX PLANE.  The pixel's ray rises
#      to the facet along ``p/pz1``, so the tangent plane belongs at
#      ``x_h = x + w``, ``w = s p / pz1``.  Taking the facet there and
#      extrapolating that PLANE back to the pixel's own coordinate cancels at
#      FIRST order in w -- which is exactly why (T1) is exact for a plane --
#      and leaves the curvature of the surface across the traverse:
#
#          + s (w . grad dz)  -  (dz/2) w^T (grad grad sag) w              (T2)
#
#   2. THE RAY IS DISPLACED BY ITS WALK.  It re-crosses the vertex plane at
#      ``x + W``, ``W = s (p/pz1 - p_out/pz2)``, so the exit eikonal has to be
#      referenced back to the pixel.  The FIRST order of that referencing is
#      already inside (T1); the second order is
#
#          - (1/2) W . (W . grad) p_out                                    (T3)
#
#      Carried to third order this term moves design 121 group 5 by 0.25 %
#      (0.0028620 -> 0.0028547 waves rms), so the series is truncated here on
#      a measurement rather than on taste.
#
# Every gradient above is a gradient of a GRID field, so the model needs no
# per-pixel Newton, no analytic sag derivative and no ray trace: it works for a
# conic, an asphere, a biconic, a Q-freeform, a form-error map and a
# ``sag_callable`` alike.  (A per-pixel fixed-point intersection was built and
# measured as the alternative; it reads 0.0028621 waves on design 121 group 5
# against this form's 0.0032381, i.e. 13 % better for a per-surface Newton and
# a sag source restriction.  The grid form ships.)
#
# WHAT IT DOES NOT DO.  The walk W cannot be REPRESENTED by a vertex-plane
# screen -- only referenced away -- and (T3) is the second order of that
# referencing, not a fix.  On design 121 group 5's exit face (slope 0.244,
# n 1.80 -> 1.00) the walk reaches 140 um across a 3 mm pupil and the tail of
# that series is what holds the group at 0.0032 waves rms.  Closing it is the
# transverse-REMAP axis (``surface_model='displaced'``), not this one.
_TANGENT_FACET_MIN_PZ_SQ = 1e-12

# ---------------------------------------------------------------------------
# THE ROW-BAND HALO, DERIVED (v5.37, BUILD_TF_BANDED_2026_08_16)
# ---------------------------------------------------------------------------
# ``BUILD_TANGENT_FACET`` S4 refused the band rather than approximate it, and
# named the obstruction correctly: the model differentiates a gradient TWICE.
# Writing the dependency out, backwards from what a band must produce:
#
#   route 3 ('tangent_facet')
#     the ACCUMULATOR needs ``grad(opd)`` at rows [r0, r1)
#       -> ``opd`` at [r0-1, r1+1)
#       -> the screen's own five gradients (of dz, gx, gy, ox, oy) there
#       -> dz / ox / oy -- and hence ``p`` -- at [r0-2, r1+2)
#       -> ``grad sag`` at [r0-2, r1+2)
#       -> ``sag`` at [r0-3, r1+3)
#     so: SAG HALO 3 ROWS, ACCUMULATOR HALO 2 ROWS.
#
#   the remap rung ('tangent_facet_remap')
#     the accumulator is ``p_out`` in CLOSED FORM (R3) rather than the gradient
#     of the screen, so the deepest chain is the Hessian (R4)/(R5):
#       screen at [r0, r1) -> grad grad sag there -> grad sag at [r0-1, r1+1)
#       -> ``sag`` at [r0-2, r1+2), and ``p`` at [r0, r1) -- NO halo at all.
#     so: SAG HALO 2 ROWS, ACCUMULATOR HALO 0 ROWS.  The rung that costs more
#     memory needs the NARROWER halo, which is not a coincidence: (R3) is
#     exactly the step that replaces a differenced kick with a closed-form one.
#
#   the gap transport (both rungs)
#     one gradient of the accumulator -> ACCUMULATOR HALO 1 ROW.
#
# The accumulator is written into a FRESH destination grid, never in place --
# which is what the whole-grid path does too (``_tf_px - _tk_x`` rebinds), so
# it is the faithful mirror AND it makes the band-boundary staleness hazard
# (``BUILD_OBL_BANDED_HALO`` S3.2a's deferred write) structurally absent rather
# than handled.
#
# WHAT IS *NOT* BANDED, AND WHY IT IS A REFUSAL RATHER THAN AN OMISSION.
# The remap rung's second half -- ``_tangent_facet_remap_apply`` -- stays
# whole-grid.  Its halo is DYNAMIC and its steps are GLOBALLY coupled, and both
# halves of that were measured rather than asserted:
#
#   1. THE WALK IS THE HALO, AND IT IS A LENGTH, NOT A ROW COUNT.  The field is
#      read at ``x + W``, and ``max|W|`` is a physical quantity -- 93.0 / 67.3 /
#      31.7 um on the three faces of a design-121-like SSK2/SF57 doublet -- so
#      the halo in ROWS grows as the grid refines: 15 / 27 / 50 rows at
#      dx = 8 / 4 / 2 um on that fixture, against a 256-row auto band.  Nothing
#      about it resembles the fixed 1-3 rows every other banded block uses, and
#      it is not knowable before the walk for THIS call has been computed.
#   2. ``scipy.ndimage.spline_filter`` (and ``map_coordinates``' own prefilter)
#      is a recursive IIR whose output at one pixel depends on every pixel of
#      the column, decaying like ``(-0.268)^k`` for order 3.  Measured on a
#      32-row band: a slab halo of 1 / 4 / 16 rows differs from the whole-grid
#      filter by 1.1e0 / 1.7e-2 / 2.4e-9, and only at a 64-ROW halo does the
#      difference underflow to exactly zero.  So byte-identity here is a
#      data-dependent numerical accident that costs a halo TWICE the band --
#      i.e. exactly the regime where banding buys nothing.
#   3. the demodulating eikonal is a ``|E|^2``-weighted least-squares fit whose
#      six moments are ``np.sum`` over the whole grid.  numpy's pairwise
#      summation is not re-associable in general: band-summing a 512x512 weight
#      differs from the whole-grid sum at a 1-row band (5.8e-11) and happens to
#      agree at 32 and 256.  "Happens to agree" is not a byte-identity argument.
#   4. ``float(np.min(det))`` -- the fold guard -- is a whole-grid reduction BY
#      CONSTRUCTION: it must refuse the CALL, and a band that has not yet seen
#      the folding row would run and return a field.
#
# So the apply half is REFUSED from the band rather than approximated into it,
# and the refusal is PRICED per call (``_tf_price_walk_halo``) against the band
# height actually in use and reported through ``progress``.  A silently-wrong
# band is worse than an expensive right one -- the standard
# ``BUILD_TANGENT_FACET`` S4 set when it refused the whole model.
#: Accumulator halo (rows) the banded route-3 screen needs.
_TF_MOM_HALO_ROWS = 2
#: Sag halo (rows) the banded route-3 screen needs.  It is the accumulator's
#: halo plus ONE, and written that way because that is the derivation: the sag
#: sits exactly one gradient level below the momentum in the chain above.
_TF_SAG_HALO_ROWS = _TF_MOM_HALO_ROWS + 1
#: Sag halo (rows) the banded remap SCREEN needs (the Hessian level).
_TF_REMAP_SAG_HALO_ROWS = 2
#: Accumulator halo (rows) the gap transport needs, both rungs.
_TF_GAP_HALO_ROWS = 1


def _tf_sl(v, lo, hi):
    """``v`` restricted to rows ``[lo:hi)`` -- scalars pass through unchanged.

    The tangent-facet accumulator is a pair of PYTHON FLOATS until the first
    powered surface promotes it, and every banded expression has to read the
    same object the whole-grid expression reads: under NEP 50 a float32 array
    of zeros and a Python ``0.0`` are NOT the same operand (the mistake
    ``BUILD_OBL_BANDED_HALO`` S3.2b records, worth 5e-6 of field at
    ``sag_dtype='float32'``)."""
    return v[lo:hi] if getattr(v, 'ndim', 0) else v


def _tf_rows_grad(arr, a0, a1, n_rows, dy, dx, xp):
    """``xp.gradient(arr, dy, dx)`` on the rows whose stencil is the one the
    WHOLE-GRID call would use.

    ``arr`` holds rows ``[a0:a1)`` of an ``n_rows``-row grid.  ``np.gradient``
    uses a central difference in the interior and a one-sided difference at the
    array's own first / last row, and its interior stencil does not know how
    tall the array is -- so a slab reproduces the whole-grid gradient EXACTLY
    on every row except a slab edge that is not also a grid edge.  Those two
    rows are dropped here rather than trusted; the caller sizes its halo so the
    rows it needs survive.  Returns ``(gy, gx, b0, b1)`` with ``[b0:b1)`` the
    absolute row range of the result."""
    gy, gx = xp.gradient(arr, dy, dx)
    lo = 1 if a0 > 0 else 0
    hi = (a1 - a0) - (1 if a1 < n_rows else 0)
    return gy[lo:hi], gx[lo:hi], a0 + lo, a0 + hi


def _tangent_facet_screen_rows(sag, gx, gy, px, py, n1, n2, dx, dy, xp,
                               lo, hi):
    """(T1) + (T2) + (T3) for rows ``[lo:hi)`` of the SLAB handed in.

    Every array argument spans the same slab of rows; the five internal
    gradients are taken over that slab and read back only at ``[lo:hi)``, so
    the caller must hand in one row of margin at each end that is not a true
    grid edge.  Element for element this is the whole-grid expression --
    ``_tangent_facet_screen`` below is literally this call with
    ``lo, hi = 0, Ny`` -- which is what makes the banded screen BYTE-identical
    rather than merely close."""
    inv = 1.0 / xp.sqrt(1.0 + gx * gx + gy * gy)
    p_sq = px * px + py * py
    ok = _tf_sl(p_sq, lo, hi) < n1 * n1
    pz1 = xp.sqrt(xp.maximum(n1 * n1 - p_sq, _TANGENT_FACET_MIN_PZ_SQ))
    a_dot = (-gx * px - gy * py + pz1) * inv
    b_sq = n2 * n2 - n1 * n1 + a_dot * a_dot
    ok = ok & (_tf_sl(b_sq, lo, hi) > 0.0)
    dz = (xp.sqrt(xp.maximum(b_sq, 0.0)) - a_dot) * inv
    del inv, a_dot, b_sq, p_sq
    sag_b = sag[lo:hi]
    dz_b = dz[lo:hi]
    opd = dz_b * sag_b
    # ---- (T2): the facet taken where the ray meets the surface ------------
    pz1_b = _tf_sl(pz1, lo, hi)
    wx = sag_b * (_tf_sl(px, lo, hi) / pz1_b)
    wy = sag_b * (_tf_sl(py, lo, hi) / pz1_b)
    _y, _x = xp.gradient(dz, dy, dx)
    opd += sag_b * (wx * _x[lo:hi] + wy * _y[lo:hi])
    del _x, _y
    _y, _x = xp.gradient(gx, dy, dx)             # (d/dy, d/dx) of d(sag)/dx
    opd -= 0.5 * dz_b * wx * (wx * _x[lo:hi] + wy * _y[lo:hi])
    del _x, _y
    _y, _x = xp.gradient(gy, dy, dx)
    opd -= 0.5 * dz_b * wy * (wx * _x[lo:hi] + wy * _y[lo:hi])
    del _x, _y
    # ---- (T3): the second order of the walk's referencing -----------------
    pz2_b = pz1_b + dz_b
    ox = px - dz * gx
    oy = py - dz * gy
    ux = wx - sag_b * (_tf_sl(ox, lo, hi) / pz2_b)
    uy = wy - sag_b * (_tf_sl(oy, lo, hi) / pz2_b)
    del wx, wy, pz1, pz1_b, pz2_b, dz, dz_b
    _y, _x = xp.gradient(ox, dy, dx)
    opd -= 0.5 * ux * (ux * _x[lo:hi] + uy * _y[lo:hi])
    del _x, _y, ox
    _y, _x = xp.gradient(oy, dy, dx)
    opd -= 0.5 * uy * (ux * _x[lo:hi] + uy * _y[lo:hi])
    del _x, _y, oy, ux, uy
    return opd, ok


def _tangent_facet_screen(sag, gx, gy, px, py, n1, n2, dx, dy, xp):
    """The route-3 screen at one surface: (T1) + (T2) + (T3), in metres of OPD.

    ``(px, py)`` is the FIELD's own transverse optical momentum at the pixel --
    the gradient of everything the model has imprinted so far, plus the carrier
    -- and ``(gx, gy)`` is ``grad sag`` on the same grid.  Returns ``(opd, ok)``
    with ``ok`` False wherever the ray is evanescent in ``n1`` or totally
    internally reflected at the facet; there the caller keeps the 'thin' screen,
    because a clamped cosine is a wrong OPD and the thin screen is the safe
    neutral (the same convention equation (4) uses)."""
    return _tangent_facet_screen_rows(sag, gx, gy, px, py, n1, n2, dx, dy, xp,
                                      0, sag.shape[0])


def _tangent_facet_transport_rows(px, py, t, n_gap, dx, dy, xp, lo, hi):
    """:func:`_tangent_facet_transport` for rows ``[lo:hi)`` of the slab.

    One gradient level, so the slab needs one row of margin at each end that is
    not a true grid edge."""
    if not getattr(px, 'ndim', 0) and not getattr(py, 'ndim', 0):
        return px, py
    px_b = px[lo:hi]
    py_b = py[lo:hi]
    pz = xp.sqrt(xp.maximum(n_gap * n_gap - px_b * px_b - py_b * py_b,
                            _TANGENT_FACET_MIN_PZ_SQ))
    wx = t * (px_b / pz)
    wy = t * (py_b / pz)
    del pz
    _y, _x = xp.gradient(px, dy, dx)
    nx = px_b - (wx * _x[lo:hi] + wy * _y[lo:hi])
    del _x, _y
    _y, _x = xp.gradient(py, dy, dx)
    ny = py_b - (wx * _x[lo:hi] + wy * _y[lo:hi])
    return nx, ny


def _tangent_facet_transport(px, py, t, n_gap, dx, dy, xp):
    """Carry the momentum accumulator across a gap.

    The field that arrives at pixel ``x`` came from pixel ``x - w``,
    ``w = t p / pz``, so the accumulator has to be RESAMPLED, not left alone:
    a grid-local accumulation reads the momentum of the wrong ray and is worth
    0.0335 waves rms on design 121 group 5 against 0.0032 for the transported
    one.  One Taylor term is enough -- ``p`` is very nearly linear in ``x``
    across a lens pupil, so the remainder is second order in the gap walk
    against a THIRD derivative of the sag; emulated on the group-5 fixture it
    moves the answer from 0.0028621 to 0.0028958 waves rms.

    Scalars pass through unchanged (a leading plate, and every gap before the
    first powered surface, where the accumulator is still the carrier's two
    floats)."""
    if not getattr(px, 'ndim', 0) and not getattr(py, 'ndim', 0):
        return px, py
    return _tangent_facet_transport_rows(px, py, t, n_gap, dx, dy, xp,
                                         0, px.shape[0])


# ---------------------------------------------------------------------------
# THE REMAP RUNG -- surface_model='tangent_facet_remap'
# ---------------------------------------------------------------------------
# Route 3 (above) leaves ONE residual and names it exactly: the TRANSVERSE WALK.
# The pixel's ray re-crosses the vertex plane at ``x + W`` rather than at ``x``,
# and a screen that lives on that plane can only REFERENCE that displacement
# away -- (T3) is the second order of the referencing series -- never REPRESENT
# it.  On design 121 group 5 the tail of that series holds the group at 0.0032
# waves rms against a 0.001 bar.
#
# This model represents it.  The element becomes SCREEN + COORDINATE REMAP: the
# field is resampled to the walked positions, so the exit eikonal is evaluated
# where the ray actually is.  Three consequences, and each is a simplification
# rather than another term:
#
# 1. THE OPD COLLAPSES TO ONE LINE, AND IT IS EXACT.  Along a ray the eikonal
#    grows as ``dS = p . dx + pz dz``.  Rising from ``(x, 0)`` to the hit point
#    ``(x + s q, s)``, ``q = p/pz1``, costs ``p . (s q) + pz1 s = s n1^2/pz1``;
#    descending to ``(x + W, 0)`` costs ``-s n2^2/pz2``.  So
#
#        S_out(x + W) = S_in(x) + s (n1^2/pz1 - n2^2/pz2),
#
#    and the screen a wave model must imprint at the pixel, since the value
#    imprinted THERE is the value carried to ``x + W``, is exactly
#
#        OPD = s (n2^2/pz2 - n1^2/pz1) ,                                   (R1)
#        W   = s (p/pz1 - p_out/pz2) ,                                     (R2)
#
#    with ``s`` the facet height AT THE HIT POINT and ``p_out`` from exact
#    vector Snell at the facet normal THERE.  (T1)+(T2)+(T3) is what (R1)
#    becomes after ``S_out`` is Taylor-referenced back to ``x`` and truncated;
#    the identity ``OPD_R1 = (T1) + (T2) - p_out . W`` holds term for term, so
#    the remap does not add a term -- it removes a truncation.
#
# 2. IT IS A LAGRANGIAN MODEL, WHICH NO SCREEN COULD BE.  BUILD_TANGENT_FACET
#    S0.1 measured the obstruction: a screen's kick is the gradient of its own
#    value, so the exact facet kick is unreachable and the prize arm's 0.000372
#    is 25x below what any screen can do.  A remap escapes that, because the
#    kick is the gradient of the COMPOSITE: with ``A = I + dW/dx``,
#
#        grad_x [ S_in - OPD ]  =  A^T p_out ,                             (R3)
#
#    identically -- the screen supplies ``A^T p_out`` and the coordinate change
#    divides ``A^T`` back out, leaving the field's momentum at the new pixel
#    equal to the EXACT refracted momentum.  Verified as an equality of two
#    independently computed grids (converging as h^2, the grid-gradient order),
#    not as a tolerance.
#
# 3. THE HIT POINT IS A FIXED POINT, NOT A TAYLOR SERIES.  ``s`` solves
#    ``s = sag(x + s q)``.  With ``a = grad sag . q`` and
#    ``b = q^T (grad grad sag) q`` the second-order solution is
#
#        s = sag/(1-a) + (b/2) [sag/(1-a)]^2 / (1-a) ,                     (R4)
#        grad sag |_hit = grad sag + (grad grad sag) (s q) ,               (R5)
#
#    which is EXACT for a plane facet (b = 0) -- so the plane-facet identity is
#    machine-exact with no oracle, measured at 5.95e-14 relative worst case over
#    the same 27 cells route 3 used.  (R5) is not optional: dropping it costs
#    6500x on group 5 (1.66e-04 against 2.56e-08 waves rms).
#
# MEASURED (bundle arm, design 121, 3 mm pupil, waves rms against exact rays):
#
#     group    route 3      REMAP       facet arm (the old prize)
#     g2      0.0000046   1.12e-10        0.0000196
#     g3      0.0000033   5.03e-11        0.0000159
#     g4      0.0000005   4.17e-12        0.0000004
#     g5      0.0032381   2.56e-08        0.0003724
#
# The remaining 2.56e-08 on g5 is the (R4) truncation and nothing else: pushing
# the fixed point to convergence with a per-pixel Newton reads 5.67e-12.  That
# Newton needs an analytic sag callable at off-grid points, which is the sag
# source restriction route 3's S1.6 refused, so the grid form ships here too --
# at 39000x under the acceptance bar there is nothing to buy.
#
# CAUSTIC SAFETY.  A remap is a ray map and must be SINGLE-VALUED: the field at
# one output pixel has to come from one input pixel.  That is exactly
# ``det A > 0``.  The guard refuses on a non-positive or near-zero determinant
# and on a non-convergent inversion, and it never degrades: a folded map is not
# approximated, it is declined.  Design 121's interior is comfortably clear
# (det in [0.927, 1.021] on group 5 at 3 mm), which is the design contract for
# this family; the guard is the proof rather than a hope.
#
# THE FIELD RESAMPLING.  The pull-back ``x(u)`` solves ``x + W(x) = u`` by fixed
# point (``W`` is a contraction exactly while the map is unfolded, so the
# iteration's convergence and the fold guard are the same statement), and the
# field is sampled there with ``scipy.ndimage.map_coordinates`` -- the same
# high-order resampler ``_apply_displaced_remap_2d`` and ``_lens_imap`` use --
# carrying the energy-conserving amplitude Jacobian ``1/sqrt(det A)`` derived
# below.  The field is DEMODULATED first by an analytic quadratic eikonal fitted
# to its own momentum, because a lens-interior field oscillates at a few pixels
# per fringe and interpolating that directly is what would eat the model's gain;
# the demodulation is a similarity transform (multiply, resample, divide), so it
# cannot change the physics, only the interpolation error.
_TF_REMAP_MIN_DET = 1.0e-4
#: Hard ceiling on the pull-back fixed-point sweeps.  It exists to stop a
#: DIVERGING iteration, not to truncate a converging one, so the loop also
#: refuses the moment the residual stops shrinking (see
#: ``_TF_REMAP_PROGRESS_FRAC``) -- which catches real divergence in a
#: handful of sweeps instead of all 64, and lets a slow but genuine
#: contraction finish.  Normal operation is 8-12 sweeps; a heavily padded
#: grid at fine sampling has been measured needing ~90.
_TF_REMAP_MAX_ITERS = 256
#: The residual must shrink by at least this factor per sweep to count as
#: progress.  A contraction maps the residual by its own rate, so anything
#: at or above 1 is a stalled or period-2 iteration and no number of extra
#: sweeps will help; 0.999 leaves room for a rate that is merely very close
#: to 1 while still catching the oscillating case in two sweeps.
_TF_REMAP_PROGRESS_FRAC = 0.999
#: Floor on ``1 - grad sag . q``, the rate at which the ray closes on the facet.
#: It vanishes only when the ray runs ALONG the facet -- the grazing limit the
#: ``ok`` mask declines anyway -- so this exists to keep the arithmetic finite
#: until that mask is applied, not to approximate anything.
_TF_REMAP_MIN_CLOSING = 1.0e-12
#: Pull-back convergence bar, in PIXELS.  A residual of 1e-9 px moves the
#: sampled phase by ``k0 |p| * 1e-9 dx`` <= 2 pi * 1e-9 * (dx/lambda) radians --
#: below 1e-6 rad for any grid coarser than 100 lambda, i.e. far below the
#: interpolation error it sits inside.
_TF_REMAP_PULLBACK_TOL_PX = 1.0e-9
#: Amplitude fraction of the peak below which a pixel is treated as DARK and is
#: excluded from the fold / pull-back guards.  ``det(I + dW/dx)`` and the walk
#: itself are evaluated over every pixel of the grid, including the padding
#: outside the clear aperture where the entrance aperture has already zeroed the
#: field and where ``sag`` and ``grad sag`` grow without bound -- so a converging
#: beam's perfectly ordinary 8x pad would otherwise REFUSE on dark corner
#: pixels while the illuminated pupil sits at ``det = 0.9986``, three orders
#: inside the bar -- and the message would say "change model" where the
#: actual remedy is "shrink the grid".  Scoring the guards over the
#: support is what keeps that misdiagnosis out.  1e-6 of peak
#: AMPLITUDE is 1e-12 of peak intensity: a fold there cannot move the answer,
#: and for a hard-apertured pupil the threshold is exactly the aperture because
#: the field outside it is identically zero.
_TF_REMAP_SUPPORT_FRAC = 1.0e-6
_VALID_REMAP_ORDERS = (1, 3, 5)


def _tangent_facet_remap_screen(sag, gx, gy, hxx, hxy, hyx, hyy,
                                px, py, n1, n2, xp):
    """The remap rung's screen at one surface: (R1)-(R5), in metres of OPD.

    Returns ``(opd, wx, wy, pox, poy, ok)`` -- the imprinted OPD, the transverse
    walk ``W`` the field is then remapped by, the EXACT refracted momentum at
    the walked pixel, and the propagating mask.  Where ``ok`` is False (the ray
    is evanescent in ``n1``, or the facet totally internally reflects) the
    caller keeps the 'thin' screen and the walk is zero: a clamped cosine is a
    wrong OPD and a clamped walk is a wrong position, and the thin screen at the
    pixel's own coordinate is the safe neutral (the convention route 3 and
    equation (4) both use).

    ``(hxx, hxy, hyx, hyy)`` is ``grad grad sag`` on the grid, taken by the
    caller so the two Hessian rows are computed once and shared."""
    p_sq = px * px + py * py
    ok = p_sq < n1 * n1
    pz1 = xp.sqrt(xp.maximum(n1 * n1 - p_sq, _TANGENT_FACET_MIN_PZ_SQ))
    del p_sq
    qx = px / pz1
    qy = py / pz1
    # ---- (R4): the hit point as a fixed point of s = sag(x + s q) ---------
    a_lin = gx * qx + gy * qy
    one_ma = xp.maximum(1.0 - a_lin, _TF_REMAP_MIN_CLOSING)
    ok = ok & (1.0 - a_lin > 0.0)
    del a_lin
    s1 = sag / one_ma
    b_qq = qx * (qx * hxx + qy * hxy) + qy * (qx * hyx + qy * hyy)
    s_hit = s1 + 0.5 * b_qq * s1 * s1 / one_ma
    del s1, b_qq, one_ma
    # ---- (R5): the facet normal AT the hit point --------------------------
    wx = s_hit * qx
    wy = s_hit * qy
    ghx = gx + (wx * hxx + wy * hxy)
    ghy = gy + (wx * hyx + wy * hyy)
    # ---- exact vector Snell at that facet ---------------------------------
    inv = 1.0 / xp.sqrt(1.0 + ghx * ghx + ghy * ghy)
    a_dot = (-ghx * px - ghy * py + pz1) * inv
    b_sq = n2 * n2 - n1 * n1 + a_dot * a_dot
    ok = ok & (b_sq > 0.0)
    dz = (xp.sqrt(xp.maximum(b_sq, 0.0)) - a_dot) * inv
    del inv, a_dot, b_sq
    pox = px - dz * ghx
    poy = py - dz * ghy
    del ghx, ghy
    pz2 = pz1 + dz
    del dz
    # A near-grazing EXIT (pz2 -> 0) makes both (R1) and (R2) diverge.  It is
    # declined pixel-wise, on the same "safe neutral" convention as TIR above,
    # rather than left to the fold guard: the guard would refuse the WHOLE call
    # over one grazing pixel, and a thin screen at that pixel's own coordinate
    # is a defensible answer where a divergent walk is not.
    ok = ok & (pz2 > 0.0)
    pz2 = xp.maximum(pz2, _TANGENT_FACET_MIN_PZ_SQ)
    # ---- (R1) the screen, (R2) the walk -----------------------------------
    opd = s_hit * (n2 * n2 / pz2 - n1 * n1 / pz1)
    wx = wx - s_hit * (pox / pz2)
    wy = wy - s_hit * (poy / pz2)
    del pz1, pz2, qx, qy, s_hit
    return opd, wx, wy, pox, poy, ok


def _tf_remap_quadratic_eikonal(px, py, weight, x_ax, y_ax, xp):
    """Fit ``Phi`` with ``grad Phi ~ (px, py)`` -- the analytic quadratic
    eikonal used to demodulate the field before it is resampled.

    ``Phi = c0 x + d0 y + (c1/2) x^2 + c2 x y + (d2/2) y^2``, so its gradient is
    the general LINEAR momentum field; the cross term is shared between the two
    components, which is the curl-free constraint an eikonal gradient satisfies
    and is what makes the fit a phase rather than a pair of ramps.  Weighted by
    ``|E|^2`` so the phase is fitted where the energy is.

    Exact whenever the momentum is linear in ``x`` -- which is the same
    condition ``_tangent_facet_transport``'s one-term gap transport is exact
    under, and very nearly true across a lens pupil.  Returns the five
    coefficients, or ``None`` if the normal equations are singular (an empty or
    degenerate weight), in which case the caller resamples undemodulated: the
    demodulation is a similarity transform, so losing it costs interpolation
    accuracy and nothing else."""
    w = weight
    sw = float(xp.sum(w))
    if not np.isfinite(sw) or sw <= 0.0:
        return None
    wx_ = w * x_ax[None, :]
    wy_ = w * y_ax[:, None]
    m1 = sw
    mx = float(xp.sum(wx_))
    my = float(xp.sum(wy_))
    mxx = float(xp.sum(wx_ * x_ax[None, :]))
    mxy = float(xp.sum(wx_ * y_ax[:, None]))
    myy = float(xp.sum(wy_ * y_ax[:, None]))
    bpx = float(xp.sum(w * px))
    bpy = float(xp.sum(w * py))
    bxpx = float(xp.sum(wx_ * px))
    bypx = float(xp.sum(wy_ * px))
    bxpy = float(xp.sum(wx_ * py))
    bypy = float(xp.sum(wy_ * py))
    del wx_, wy_
    # unknowns (c0, c1, c2, d0, d2); rows are d/d(unknown) of the weighted
    # residual sum over BOTH momentum components.
    m = np.array([
        [m1,  mx,  my,  0.0, 0.0],
        [mx,  mxx, mxy, 0.0, 0.0],
        [my,  mxy, myy + mxx, mx, mxy],
        [0.0, 0.0, mx,  m1,  my],
        [0.0, 0.0, mxy, my,  myy],
    ], dtype=np.float64)
    rhs = np.array([bpx, bxpx, bypx + bxpy, bpy, bypy], dtype=np.float64)
    try:
        c = np.linalg.solve(m, rhs)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(c)):
        return None
    return tuple(float(v) for v in c)


def _tf_remap_phi(coef, x, y):
    """``Phi`` from :func:`_tf_remap_quadratic_eikonal`'s coefficients, at
    arbitrary (broadcastable) coordinates -- ANALYTIC, so the remodulation at
    the pulled-back sample point costs no second interpolation and adds no
    second interpolation error."""
    c0, c1, c2, d0, d2 = coef
    return (c0 * x + d0 * y
            + (0.5 * c1) * x * x + c2 * x * y + (0.5 * d2) * y * y)


def _tangent_facet_remap_apply(E, wx, wy, pox, poy, dx, dy, k0, order, xp,
                               surface_index):
    """Apply the coordinate remap: resample ``E`` (and the momentum
    accumulator) from the pixel grid to the walked positions.

    THE AMPLITUDE JACOBIAN, DERIVED.  The remap is a coordinate transform
    ``u = M(x) = x + W(x)`` of a field, so it must move ENERGY, not values:
    ``|E_out(u)|^2 d^2u = |E_in(x)|^2 d^2x``.  With
    ``d^2u = |det A| d^2x``, ``A = dM/dx = I + dW/dx``, that is

        |E_out(u)| = |E_in(x)| / sqrt(|det A(x)|) ,

    the reciprocal square root of the FORWARD determinant evaluated at the
    SOURCE point -- the same factor and the same evaluation point as
    ``_apply_displaced_remap_2d``'s ``1/sqrt(|det d(x_out,y_out)/d(x_in,y_in)|)``.
    It is a derivation, not a normalisation: nothing here is fitted or rescaled
    to make the power come out, and the power is measured afterwards as a
    consequence.

    THE FOLD GUARD.  ``det A > 0`` is exactly the statement that ``M`` is
    single-valued and orientation-preserving -- that the output pixel has ONE
    source.  Where it is not, the field there is a sum over two or more
    branches, which no pull-back can represent, so this REFUSES rather than
    silently returning whichever branch the iteration happened to land on.
    ``_TF_REMAP_MIN_DET`` puts the bar a little above zero because
    ``1/sqrt(det)`` is an amplitude GAIN of 100x there: below that the map has
    compressed the pupil past what the grid can carry, and the answer would be
    resolution-limited even where it is single-valued.

    Momentum is not a density and carries no Jacobian: the accumulator is
    resampled with the same coordinates and no factor."""
    if xp is not np:                                   # pragma: no cover
        raise NotImplementedError(
            "apply_real_lens: surface_model='tangent_facet_remap' has no GPU "
            "path (the pull-back uses scipy.ndimage).")
    from scipy.ndimage import map_coordinates, spline_filter
    ny, nx = E.shape
    # ---- A = I + dW/dx, and its determinant -------------------------------
    _wx_y, _wx_x = np.gradient(wx, dy, dx)
    _wy_y, _wy_x = np.gradient(wy, dy, dx)
    a11 = 1.0 + _wx_x
    a12 = _wx_y
    a21 = _wy_x
    a22 = 1.0 + _wy_y
    del _wx_x, _wx_y, _wy_x, _wy_y
    det = a11 * a22 - a12 * a21
    # ---- the illuminated support: what the guards are allowed to score ------
    # Outside it the field is zero (or 1e-12 of the peak intensity) and the sag
    # that drives the walk is unbounded, so a fold there is unobservable.  See
    # ``_TF_REMAP_SUPPORT_FRAC``.
    _a = np.abs(E)
    _amax = float(_a.max()) if _a.size else 0.0
    supp = (_a > _TF_REMAP_SUPPORT_FRAC * _amax) if _amax > 0.0 else None
    del _a
    if supp is None or not bool(supp.any()):
        supp = np.ones(E.shape, dtype=bool)
    d_min_grid = float(np.min(det))
    d_min = float(np.min(det[supp]))
    if not np.isfinite(d_min) or d_min <= _TF_REMAP_MIN_DET:
        raise ValueError(
            f"apply_real_lens: surface_model='tangent_facet_remap' REFUSES at "
            f"surface {surface_index}: the transverse-walk map folds.  "
            f"min det(I + dW/dx) = {d_min:.6g} <= {_TF_REMAP_MIN_DET:g} over "
            f"the ILLUMINATED support ({int(supp.sum())} of {E.size} pixels; "
            f"whole-grid min including dark padding is {d_min_grid:.6g}), so "
            f"the map is not single-valued (or is compressed past a 100x "
            f"amplitude gain) and the field at some exit pixel is a sum over "
            f"two or more ray branches that a resampling cannot represent.  "
            f"This model is for caustic-free element interiors; use "
            f"surface_model='tangent_facet' (which references the walk away "
            f"instead of representing it), apply_real_lens_traced, or "
            f"apply_real_lens_maslov (caustic-safe) for this prescription.")
    # ---- the field's own momentum after the screen: A^T p_out (R3) --------
    # Used ONLY to fit the demodulating eikonal, so it is built and dropped
    # here rather than accumulated.
    ps_x = a11 * pox + a21 * poy
    ps_y = a12 * pox + a22 * poy
    del a11, a12, a21, a22
    x_ax = (np.arange(nx, dtype=np.float64) - nx // 2) * dx
    y_ax = (np.arange(ny, dtype=np.float64) - ny // 2) * dy
    w_amp = np.abs(E)
    w_amp *= w_amp
    coef = _tf_remap_quadratic_eikonal(ps_x, ps_y, w_amp, x_ax, y_ax, np)
    del ps_x, ps_y, w_amp
    # ---- the pull-back x(u): fixed point of x = u - W(x) -------------------
    # W is a contraction exactly while the map is unfolded, so this iteration
    # converging and the guard above passing are the same statement; a
    # non-convergence is therefore also a refusal rather than a truncation.
    # NB the walk is NOT clamped outside the support.  Zeroing it there makes
    # ``W`` discontinuous at the support edge, and the fixed point of
    # ``x = u - W(x)`` then oscillates with period 2 for every pixel within one
    # walk of that edge (measured: a 2x-padded grid that converged before
    # stalled at a 0.303 px step).  The walk stays continuous; only the
    # CONVERGENCE TEST is restricted to the support, below.
    sx = wx / dx
    sy = wy / dy
    if order > 1:
        sx = spline_filter(sx, order=order, output=np.float64)
        sy = spline_filter(sy, order=order, output=np.float64)
    iu = np.arange(nx, dtype=np.float64)[None, :] + np.zeros((ny, 1))
    iv = np.arange(ny, dtype=np.float64)[:, None] + np.zeros((1, nx))
    ix = iu.copy()
    iy = iv.copy()
    _pf = (order == 1)
    ok_conv = False
    step = float('inf')
    _prev_step = float('inf')
    _n_it = 0
    for _n_it in range(1, _TF_REMAP_MAX_ITERS + 1):
        crd = np.stack([iy.ravel(), ix.ravel()])
        nix = iu - map_coordinates(sx, crd, order=order, mode='nearest',
                                   prefilter=_pf).reshape(ny, nx)
        niy = iv - map_coordinates(sy, crd, order=order, mode='nearest',
                                   prefilter=_pf).reshape(ny, nx)
        del crd
        # Convergence is scored over the SUPPORT, for the same reason the fold
        # guard is: a dark padding pixel whose walk never settles cannot move
        # the answer, and would otherwise abort the whole call.
        step = max(float(np.max(np.abs((nix - ix)[supp]))),
                   float(np.max(np.abs((niy - iy)[supp]))))
        ix, iy = nix, niy
        if not np.isfinite(step):
            break
        if step < _TF_REMAP_PULLBACK_TOL_PX:
            ok_conv = True
            break
        # Stop as soon as the iteration stops contracting: a stalled or
        # period-2 residual will not improve with more sweeps, and
        # refusing here reports it in 2-3 sweeps rather than 256.
        if _n_it > 1 and not (step < _TF_REMAP_PROGRESS_FRAC * _prev_step):
            break
        _prev_step = step
    del sx, sy, iu, iv
    if not ok_conv:
        raise ValueError(
            f"apply_real_lens: surface_model='tangent_facet_remap' REFUSES at "
            f"surface {surface_index}: the pull-back x + W(x) = u did not "
            f"converge in {_n_it} of {_TF_REMAP_MAX_ITERS} iterations (last "
            f"step {step:.3g} px over the illuminated support, against a "
            f"{_TF_REMAP_PULLBACK_TOL_PX:g} px bar).  The walk map is not "
            f"invertible on this grid.  Same remedies as the fold refusal "
            f"above.")
    # ---- resample -----------------------------------------------------------
    # DEMODULATE by the analytic quadratic eikonal first: a lens-interior field
    # runs at a few pixels per fringe, and a spline through that is where the
    # model's accuracy would go.  Phi is analytic, so the remodulation at the
    # pulled-back point is exact and costs no second interpolation.
    x_src = x_ax[0] + ix * dx
    y_src = y_ax[0] + iy * dy
    # Inside the support ``det > _TF_REMAP_MIN_DET`` by the guard above, so the
    # clamp is a no-op there and ``jac`` is bit-identical.  Outside it a folded
    # (negative) det would make ``sqrt`` NaN and poison the resample, so the
    # model declines to represent the field there and hands back a zero -- the
    # region carries below 1e-12 of the peak intensity by construction.
    _bad_det = ~(det > _TF_REMAP_MIN_DET)
    if bool(_bad_det.any()):
        jac = np.where(_bad_det,
                       0.0,
                       1.0 / np.sqrt(np.where(_bad_det, 1.0, det)))
    else:
        jac = 1.0 / np.sqrt(det)
    del det, _bad_det
    # SIGN.  The library's field is ``A exp(+i k0 S)`` with ``p = grad S`` --
    # measured, not assumed: a plane wave ``exp(+i k0 p x)`` propagated through
    # the library's own ASM moves its centroid by ``+p z`` (+200.3 um against a
    # +200.0 um geometric prediction at p = 0.05, z = 4 mm), and the accumulator
    # step ``p -= grad OPD`` matches ``S -= OPD`` under exactly that sign.  So
    # the demodulation is ``exp(-i k0 Phi)`` and the remodulation ``exp(+i k0
    # Phi)``; getting this backwards DOUBLES the fringe rate instead of
    # flattening it, and was caught by the power it destroyed (0.944 against
    # 0.9999 of the input, on the biconvex fixture at 4 um sampling).
    f = E * jac if coef is None else (
        E * (jac * np.exp(-1j * k0 * _tf_remap_phi(
            coef, x_ax[None, :], y_ax[:, None]))))
    del jac
    crd = np.stack([iy.ravel(), ix.ravel()])
    del ix, iy
    e_out = map_coordinates(f, crd, order=order, mode='constant',
                            cval=0.0).reshape(ny, nx)
    del f
    if coef is not None:
        e_out *= np.exp(1j * k0 * _tf_remap_phi(coef, x_src, y_src))
    del x_src, y_src
    if e_out.dtype != E.dtype:
        e_out = e_out.astype(E.dtype)
    # the momentum accumulator rides the same coordinates, with NO Jacobian
    px_out = map_coordinates(pox, crd, order=order, mode='nearest'
                             ).reshape(ny, nx)
    py_out = map_coordinates(poy, crd, order=order, mode='nearest'
                             ).reshape(ny, nx)
    return e_out, px_out, py_out


def _check_screen_obliquity_support(*, carrier, screen_obliquity,
                                    on_screen_obliquity, surface_model,
                                    displaced_mode):
    """Validate the screen-obliquity kwarg combination.  Returns True when the
    correction is to be APPLIED."""
    if on_screen_obliquity not in _VALID_SCREEN_OBLIQUITY_POLICY:
        raise ValueError(
            f"apply_real_lens: on_screen_obliquity must be "
            f"{list(_VALID_SCREEN_OBLIQUITY_POLICY)}, got "
            f"{on_screen_obliquity!r}.")
    # Identity for the booleans (so 1/0 do NOT masquerade as True/False), but
    # EQUALITY for the string: a caller-built 'auto' (os.environ, config file,
    # f-string) is not the interned literal, so `is` rejected the documented
    # value.  Tests only ever passed the literal, which is interned and hid it.
    if not (screen_obliquity is True or screen_obliquity is False
            or screen_obliquity == 'auto'):
        raise ValueError(
            f"apply_real_lens: screen_obliquity must be one of "
            f"{list(_VALID_SCREEN_OBLIQUITY)}, got {screen_obliquity!r}.")
    if carrier is None:
        if screen_obliquity is True:
            raise ValueError(
                "apply_real_lens: screen_obliquity=True needs carrier= -- the "
                "correction is the DIFFERENCE between the screen OPD at the "
                "carrier's local ray angle and at normal incidence, so with "
                "no carrier there is no angle and the correction is "
                "identically zero.  Pass carrier=TiltedCarrier(...) (or a "
                "signed conjugate distance / 'auto' / an explicit wavefront), "
                "or drop screen_obliquity.")
        return False
    if surface_model in _TANGENT_FACET_MODELS:
        # Route 3 SUPERSEDES equations (4) and (7) rather than composing with
        # them: it imprints the exact tangent-facet OPD at the field's own
        # local ray angle, so the angular DIFFERENCE those equations add is
        # already inside it and adding them again double-counts.  ``carrier=``
        # is still honoured -- it seeds the momentum accumulator -- so this is
        # a refusal of the correction, not of the carrier.
        if screen_obliquity is True:
            raise ValueError(
                "apply_real_lens: screen_obliquity=True is not supported with "
                f"surface_model={surface_model!r}.  That model already imprints "
                "the exact tangent-facet OPD at the local ray angle, so the "
                "angular correction (equations 4 and 7) is inside it and "
                "adding it again double-counts.  Drop screen_obliquity, or "
                "use the default surface_model='thin' to get the correction "
                "instead of the model.")
        return False
    if surface_model != 'thin':
        raise ValueError(
            f"apply_real_lens: carrier= is only supported with the default "
            f"surface_model='thin' screen; got surface_model="
            f"{surface_model!r} (displaced_mode={displaced_mode!r}).  The "
            f"'displaced' path is ALREADY angle-aware -- it launches its "
            f"obliquity fan along conjugate= and modifies the same per-surface "
            f"sag OPD with true ray cosines -- so applying the screen-"
            f"obliquity correction on top would double-count it.  Use "
            f"conjugate= there instead.")
    return screen_obliquity is not False


#: ``stacklevel`` every warning raised inside :func:`_apply_real_lens_impl`
#: uses, so they keep pointing at the USER'S call rather than at library code.
#: The v5.40 split put ``apply_real_lens`` (which owns the accumulator-store
#: ``with`` block) between the caller and the body, and a warning that names
#: ``return _apply_real_lens_impl(...)`` as its origin is useless -- it tells
#: you where the library called itself.  3 = the impl's own frame, the
#: wrapper's, then the caller's.
_WARN_STACKLEVEL = 3


_VALID_ACCUMULATOR_STORE = ('ram', 'memmap')


class _AccumulatorStore:
    """Allocator for the PERSISTENT full-grid accumulators of the angle-true
    screen paths (v5.40).

    Everything else ``apply_real_lens`` allocates at full-grid size is a
    TRANSIENT -- the band loop frees it inside one iteration -- and is
    therefore already as small as banding can make it.  What banding cannot
    remove is the state that must be simultaneously live for the WHOLE grid
    while the loop walks it:

    * the tangent-facet momentum accumulator ``(_tf_px, _tf_py)`` and the
      fresh destination pair each surface writes into (4 float64 grids);
    * the remap rung's walk components ``(_rm_wx_g, _rm_wy_g)`` (2 more);
    * the screen-obliquity momentum pair ``_obl_p0*``, the drift pair
      ``_obl_u*``, the materialised carrier momentum ``_obl_q*`` and the
      guard's ``_obl_total``.

    Each of those is written ONCE PER SURFACE, band by band, in increasing
    row order, and read back band by band on the next surface.  That is the
    textbook out-of-core access pattern, and it is what ``'memmap'`` exploits:
    the accumulator lives in a file in ``scratch_dir`` and the OS pages in the
    band under the cursor.  At ``N = 32768`` one such grid is 8.59 GB, so the
    tangent-facet route-3 set alone is 34.4 GB of resident pages that the run
    otherwise has to hold.

    THE CONTRACT IS BIT-IDENTITY, and it is structural rather than hoped for:
    the store changes only WHERE an accumulator's bytes live.  Every
    expression that reads or writes one is unchanged, operates on the same
    dtype and the same C-contiguous layout, and sees a plain ``np.ndarray``
    (``np.asarray`` of the mapping -- a BASE-CLASS view; see :meth:`_make`),
    so no ufunc can take a different branch on the array's TYPE.

    Parameters
    ----------
    mode : {'ram', 'memmap'}
        ``'ram'`` (the default) is ``xp.empty`` / ``xp.zeros`` verbatim -- the
        store is then a pure pass-through and allocates no files at all.
    scratch_dir : str or None
        Directory the ``'memmap'`` backing files are created in.  ``None``
        creates (and removes) a private temporary directory.  A caller-supplied
        directory is NOT removed; only the files this store made in it are.

    The array namespace is supplied later, by :meth:`bind`, because the public
    entry point opens the store before it has resolved the backend.
    ``'memmap'`` is REFUSED for anything but NumPy -- a CuPy / JAX accumulator
    has no host mapping to spill to, and silently falling back to RAM would
    make the preflight's memmap credit a lie.

    Notes
    -----
    **The store releases each mapping when its VIEW dies, not only at
    :meth:`close`.**  The tangent-facet path allocates a fresh destination
    pair per surface and a fresh pair per gap transport and rebinds the
    accumulator to it, so the previous pair is garbage immediately.  Without
    per-view reaping the scratch directory would grow to every accumulator the
    call ever made -- twelve mappings, ~103 GB at ``N = 32768`` on a
    three-surface group -- instead of the four or so that are live.

    **Windows.**  Two hazards, both found by testing rather than by reading.
    A mapped file cannot be unlinked while the mapping is open, so the reaper
    drops its reference and closes the ``mmap`` explicitly before unlinking.
    And the unlink is not always COMPLETE when it returns: the entry lingers
    in a pending-delete state and an immediately following ``rmdir`` fails
    with "directory not empty" on a directory ``listdir`` already reports as
    empty.  Both are retried with backoff and WARN if they exhaust it, rather
    than leaving silent litter in the scratch directory.

    ``close`` is idempotent and runs from the public entry point's ``with``,
    so an exception mid-run cleans up exactly as a normal return does.  Any
    view still held after ``close`` is invalid -- which is why the accumulator
    names are dropped before it runs.
    """

    __slots__ = ('mode', 'xp', '_dir', '_own_dir', '_entries', '_n', '_closed')

    def __init__(self, mode='ram', scratch_dir=None):
        if mode not in _VALID_ACCUMULATOR_STORE:
            raise ValueError(
                f"apply_real_lens: accumulator_store must be one of "
                f"{_VALID_ACCUMULATOR_STORE}, got {mode!r}.")
        self.mode = mode
        self.xp = np
        self._dir = scratch_dir
        self._own_dir = False
        self._entries = []
        self._n = 0
        self._closed = False

    def bind(self, xp):
        """Pin the array namespace, once the caller has resolved the backend."""
        if self.mode == 'memmap' and xp is not np:
            raise ValueError(
                "apply_real_lens: accumulator_store='memmap' requires the "
                "NumPy backend -- a device (CuPy / JAX) accumulator has no "
                "host mapping to spill to.  Use accumulator_store='ram'.")
        self.xp = xp
        return self

    # -- allocation ------------------------------------------------------
    @property
    def active(self):
        return self.mode == 'memmap' and not self._closed

    def empty(self, shape, dtype):
        """An UNINITIALISED accumulator -- the caller fills every row."""
        if not self.active:
            return self.xp.empty(shape, dtype=dtype)
        return self._make(shape, dtype, fill=None)

    def zeros(self, shape, dtype):
        if not self.active:
            return self.xp.zeros(shape, dtype=dtype)
        return self._make(shape, dtype, fill=0)

    def full(self, shape, value, dtype):
        if not self.active:
            return self.xp.full(shape, value, dtype=dtype)
        return self._make(shape, dtype, fill=value)

    def adopt(self, arr):
        """Move an already-materialised full-grid accumulator into the store.

        Copies ``arr``'s bytes into a mapping and returns the mapped view; the
        caller drops its reference to ``arr``.  Used where the accumulator is
        SEEDED by a helper that returns an ordinary array (the carrier's
        momentum field).  Scalars and non-arrays pass straight through."""
        if not self.active or getattr(arr, 'ndim', 0) == 0:
            return arr
        out = self._make(arr.shape, arr.dtype, fill=None)
        out[...] = arr
        return out

    def astype(self, arr, dtype):
        """``arr.astype(dtype)`` for an accumulator, keeping it in the store."""
        if not self.active or getattr(arr, 'ndim', 0) == 0:
            return arr.astype(dtype)
        out = self._make(arr.shape, dtype, fill=None)
        out[...] = arr
        return out

    # -- internals -------------------------------------------------------
    def _ensure_dir(self):
        if self._dir is None:
            import tempfile
            self._dir = tempfile.mkdtemp(prefix='lumenairy_accum_')
            self._own_dir = True
        elif not _os.path.isdir(self._dir):
            _os.makedirs(self._dir, exist_ok=True)
        return self._dir

    def _make(self, shape, dtype, fill):
        d = self._ensure_dir()
        self._n += 1
        path = _os.path.join(
            d, f"accum_{_os.getpid()}_{id(self):x}_{self._n:03d}.dat")
        mm = np.memmap(path, dtype=dtype, mode='w+', shape=tuple(shape))
        if fill is not None:
            mm[...] = fill
        slot = len(self._entries)
        self._entries.append((path, mm))
        # Hand out a BASE-CLASS view.  ``np.asarray`` drops the ``np.memmap``
        # subclass while sharing the mapped buffer, so every downstream ufunc
        # sees exactly the object type the in-RAM path gives it and no result
        # can inherit a subclass wrapper.
        view = np.asarray(mm)
        # REAP EACH MAPPING WHEN ITS VIEW DIES, not only at close().  The
        # tangent-facet path allocates a FRESH destination pair per surface
        # and a fresh pair per gap transport, and rebinds the accumulator to
        # it; the previous pair becomes garbage immediately.  Without this the
        # scratch directory would grow to every accumulator the call ever
        # made -- 12 files, ~103 GB, on a three-surface group at N = 32768 --
        # instead of the four or so that are live.  The finalizer holds no
        # reference to ``view``, so it cannot itself keep the mapping alive.
        # ``atexit=False``: the ``with`` block in :func:`apply_real_lens`
        # guarantees ``close()`` on every exit, so the only thing an
        # interpreter-shutdown reap could add is a teardown-order hazard
        # (``_os`` already torn down when the finalizer runs).
        _f = _weakref.finalize(view, self._reap, slot)
        _f.atexit = False
        return view

    def _reap(self, slot):
        """Close and unlink one mapping whose view has been collected."""
        if self._closed or slot >= len(self._entries):
            return
        entry = self._entries[slot]
        if entry is None:
            return
        self._entries[slot] = None
        self._drop(*entry)

    # -- teardown --------------------------------------------------------
    @staticmethod
    def _drop(path, mm):
        """Close one mapping and unlink its file."""
        try:
            base = getattr(mm, '_mmap', None)
            del mm
            if base is not None:
                base.close()
        except (BufferError, ValueError, OSError):       # pragma: no cover
            pass
        for _attempt in range(5):
            try:
                _os.remove(path)
                return
            except FileNotFoundError:
                return
            except OSError:
                # Windows refuses the unlink while any mapping survives.
                _time.sleep(0.01 * (_attempt + 1))
        # Reported, not hidden: a scratch file this size is not something to
        # leave behind quietly.
        _warnings.warn(                                  # pragma: no cover
            f"apply_real_lens: could not remove the accumulator scratch "
            f"file {path!r}; remove it manually.", RuntimeWarning,
            stacklevel=2)

    def close(self):
        """Close every mapping and unlink its file.  Idempotent."""
        if self._closed:
            return
        self._closed = True
        entries, self._entries = self._entries, []
        for entry in entries:
            if entry is not None:
                self._drop(*entry)
        if self._own_dir and self._dir:
            # WINDOWS: unlinking a file whose last handle has just closed is
            # not always complete by the time the call returns -- the entry
            # lingers in a pending-delete state and the immediately following
            # ``rmdir`` fails with "directory not empty" on a directory that
            # ``listdir`` already reports as empty.  How long it lasts tracks
            # what else the box is doing (measured: never on an idle box after
            # this retry, reproducibly under a concurrent 16 GB job), so it is
            # retried and then REPORTED rather than left as silent litter in
            # %TEMP%.  The FILES are already gone by this point either way --
            # only the empty directory is at stake.
            for _attempt in range(10):
                try:
                    _os.rmdir(self._dir)
                    break
                except OSError:
                    _time.sleep(0.02 * (_attempt + 1))
            else:                                        # pragma: no cover
                _warnings.warn(
                    f"apply_real_lens: could not remove the accumulator "
                    f"scratch directory {self._dir!r}; remove it manually.",
                    RuntimeWarning, stacklevel=2)
            self._dir = None
            self._own_dir = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


_VALID_SURFACE_MODELS = ('thin', 'displaced', 'tangent_facet',
                         'tangent_facet_remap')
#: The two route-3 family members.  They share the momentum accumulator, the
#: gap transport, the whole-grid-only restriction and every refusal; they differ
#: only in whether the transverse walk is REFERENCED away (``'tangent_facet'``,
#: (T1)+(T2)+(T3)) or REPRESENTED (``'tangent_facet_remap'``, (R1)+(R2)).
_TANGENT_FACET_MODELS = ('tangent_facet', 'tangent_facet_remap')


def _check_displaced_support(*, surface_model, slant_correction, fresnel,
                             seidel_correction, absorption, surface_frame,
                             use_gpu, wave_propagator, prescription,
                             conjugate=None, E_shape=None,
                             displaced_mode='screen',
                             displaced_obliquity='auto',
                             displaced_n_side=None,
                             remap_order=3):
    """Validate ``surface_model`` and, for ``'displaced'``, that the requested
    feature set + prescription are within the ray-angle-aware refraction OPD's
    supported envelope.  Raises ``ValueError`` / ``NotImplementedError`` with a
    precise message instead of silently producing a wrong field."""
    if surface_model not in _VALID_SURFACE_MODELS:
        raise ValueError(
            f"apply_real_lens: unknown surface_model {surface_model!r}.  "
            f"Valid choices: {sorted(_VALID_SURFACE_MODELS)}.")
    if displaced_mode not in _VALID_DISPLACED_MODES:
        raise ValueError(
            f"apply_real_lens: unknown displaced_mode {displaced_mode!r}.  "
            f"Valid choices: {sorted(_VALID_DISPLACED_MODES)}.")
    if displaced_obliquity not in _VALID_DISPLACED_OBLIQUITY:
        raise ValueError(
            f"apply_real_lens: unknown displaced_obliquity "
            f"{displaced_obliquity!r}.  Valid choices: "
            f"{sorted(_VALID_DISPLACED_OBLIQUITY)}.")
    displaced_n_side = _normalise_displaced_n_side(displaced_n_side)
    if displaced_n_side is not None and not _routes_to_displaced_remap_2d(
            surface_model, displaced_mode, displaced_obliquity,
            prescription.get('surfaces') or []):
        raise ValueError(
            f"apply_real_lens: displaced_n_side={displaced_n_side!r} sets the "
            f"launch lattice of the 2-D transverse-walk remap, and this call "
            f"does not run it (surface_model={surface_model!r}, "
            f"displaced_mode={displaced_mode!r}, displaced_obliquity="
            f"{displaced_obliquity!r}, element "
            f"{'asymmetric' if _element_is_asymmetric(prescription.get('surfaces') or []) else 'rotationally symmetric'}"
            f").  The 2-D remap runs for a decentered / tilted / sag_callable "
            f"element under surface_model='displaced' with displaced_mode="
            f"'remap', or with the default 'screen' and "
            f"displaced_obliquity='auto'.  Drop displaced_n_side or route the "
            f"call to that model.")
    if surface_model in _TANGENT_FACET_MODELS:
        if remap_order not in _VALID_REMAP_ORDERS:
            raise ValueError(
                f"apply_real_lens: remap_order must be one of "
                f"{list(_VALID_REMAP_ORDERS)} (the spline orders "
                f"scipy.ndimage.map_coordinates offers that this build has "
                f"measured); got {remap_order!r}.")
        if surface_model == 'tangent_facet' and remap_order != 3:
            raise ValueError(
                f"apply_real_lens: remap_order= is only meaningful with "
                f"surface_model='tangent_facet_remap' (it is the interpolation "
                f"order of the transverse-walk resampling); got remap_order="
                f"{remap_order!r} with surface_model='tangent_facet', which "
                f"imprints a screen and resamples nothing.  Drop remap_order "
                f"or pass surface_model='tangent_facet_remap'.")
        # Route 3's three keywords are the 'displaced' path's, and they mean
        # nothing here: this model takes its ray angle from the field's own
        # accumulated momentum (seeded by carrier=), not from a launched fan.
        for _name, _val, _default in (('conjugate', conjugate, None),
                                      ('displaced_mode', displaced_mode,
                                       'screen'),
                                      ('displaced_obliquity',
                                       displaced_obliquity, 'auto')):
            if _val != _default:
                raise ValueError(
                    f"apply_real_lens: {_name}= is only meaningful with "
                    f"surface_model='displaced'; got {_name}={_val!r} with "
                    f"surface_model={surface_model!r}, which reads its ray "
                    f"angle from the field's own accumulated momentum "
                    f"(seed it with carrier=).")
        # A second angle-aware screen for the same job would double-count.
        if slant_correction:
            raise ValueError(
                "apply_real_lens: slant_correction=True is not supported with "
                f"surface_model={surface_model!r}.  Both replace the paraxial "
                "facet coefficient (n2-n1) with a refraction-aware one, so "
                "stacking them double-counts the same physics.  Drop "
                "slant_correction.")
        if surface_frame:
            raise NotImplementedError(
                "apply_real_lens: surface_frame=True is not supported with "
                f"surface_model={surface_model!r} (the model's momentum "
                "accumulator is defined on the FIELD grid, and the "
                "surface-frame path re-expresses the sag on a per-surface "
                "frame).  Drop surface_frame.")
        if use_gpu:
            raise NotImplementedError(
                "apply_real_lens: use_gpu=True is not supported with "
                f"surface_model={surface_model!r} -- the model is xp-generic "
                "but no CuPy run has been measured against the ray oracle, and "
                "an unmeasured accuracy claim is worse than a refusal.  (The "
                "remap rung is additionally scipy-bound: its pull-back uses "
                "scipy.ndimage.map_coordinates.)")
        if wave_propagator not in (None, 'asm', 'rayleigh_sommerfeld'):
            raise NotImplementedError(
                f"apply_real_lens: surface_model={surface_model!r} needs an "
                "exact angular-spectrum gap (the model's own momentum "
                "bookkeeping assumes it); got wave_propagator="
                f"{wave_propagator!r}.")
        return
    if remap_order != 3:
        raise ValueError(
            f"apply_real_lens: remap_order= is only meaningful with "
            f"surface_model='tangent_facet_remap' (it is the interpolation "
            f"order of the transverse-walk resampling); got remap_order="
            f"{remap_order!r} with surface_model={surface_model!r}.  Drop "
            f"remap_order or pass surface_model='tangent_facet_remap'.")
    if surface_model == 'thin':
        if conjugate is not None:
            raise ValueError(
                "apply_real_lens: conjugate= is only meaningful with "
                "surface_model='displaced' (it sets the input congruence for "
                "the obliquity fan).  The default 'thin' screen has no "
                "ray-angle fan; drop conjugate= or pass "
                "surface_model='displaced'.")
        if displaced_mode != 'screen':
            raise ValueError(
                "apply_real_lens: displaced_mode= is only meaningful with "
                "surface_model='displaced' (it selects the extreme-conjugate "
                f"displaced sub-model).  Got displaced_mode={displaced_mode!r} "
                "with surface_model='thin'; drop displaced_mode or pass "
                "surface_model='displaced'.")
        if displaced_obliquity != 'auto':
            raise ValueError(
                "apply_real_lens: displaced_obliquity= is only meaningful with "
                "surface_model='displaced' (it selects the meridional-LUT vs "
                "pointwise-2D obliquity path).  Got displaced_obliquity="
                f"{displaced_obliquity!r} with surface_model='thin'; drop it "
                "or pass surface_model='displaced'.")
        return
    _obliq = _resolve_displaced_obliquity(
        displaced_obliquity, prescription.get('surfaces') or [])
    # ``displaced`` conjugate vocabulary: None (collimated), a signed scalar
    # conjugate distance (m), 'auto', or an explicit wavefront ndarray.
    if conjugate is not None:
        if isinstance(conjugate, str):
            if conjugate != 'auto':
                raise ValueError(
                    f"apply_real_lens: surface_model='displaced' conjugate "
                    f"string must be 'auto', got {conjugate!r}.")
        elif isinstance(conjugate, np.ndarray):
            if E_shape is not None and conjugate.shape != E_shape:
                raise ValueError(
                    f"apply_real_lens: conjugate wavefront ndarray shape "
                    f"{conjugate.shape} != field shape {E_shape}.")
        elif isinstance(conjugate, (int, float)) and not isinstance(
                conjugate, bool):
            if float(conjugate) == 0.0:
                raise ValueError(
                    "apply_real_lens: surface_model='displaced' conjugate "
                    "distance must be non-zero.")
        else:
            raise ValueError(
                f"apply_real_lens: surface_model='displaced' conjugate must be "
                f"None, a signed scalar distance, 'auto', or a wavefront "
                f"ndarray, got {type(conjugate).__name__}.")
    # ``displaced`` is a self-contained ray-angle-aware OPD; it is mutually
    # exclusive with the other per-surface OPD / amplitude modifiers (they
    # would double-count or contradict the traced-fan incidence angles).
    _incompat = [name for name, on in (
        ('slant_correction', slant_correction),
        ('fresnel', fresnel),
        ('seidel_correction', seidel_correction),
        ('absorption', absorption),
        ('surface_frame', surface_frame),
        ('use_gpu', use_gpu),
    ) if on]
    if _incompat:
        raise ValueError(
            f"apply_real_lens: surface_model='displaced' is incompatible with "
            f"{_incompat}.  The displaced model supplies its own ray-angle "
            f"refraction OPD; drop those flags or use surface_model='thin'.")
    if wave_propagator not in ('asm', None):
        raise ValueError(
            f"apply_real_lens: surface_model='displaced' requires the ASM "
            f"in-glass propagator (got wave_propagator={wave_propagator!r}).")
    # The meridional cosine fan supports only rotationally-symmetric plain
    # conic / aspheric refracting surfaces; the pointwise 2-D path (P3 / N2)
    # additionally supports per-surface decenter / tilt / freeform sag_callable.
    _pointwise = (_obliq == 'pointwise')
    surfaces = prescription.get('surfaces') or []
    _asym = _element_is_asymmetric(surfaces)
    # P10 (N11): decentered / tilted / freeform elements get the 2-D
    # transverse-walk remap -- the DEFAULT (auto obliquity) and also selectable
    # via displaced_mode='remap'.  ``'split'`` (candidate b) has no 2-D
    # generalisation, so it is rejected for an asymmetric element.
    if _asym and displaced_mode == 'split':
        raise ValueError(
            "apply_real_lens: displaced_mode='split' is a rotationally-"
            "symmetric extreme-conjugate sub-model with no 2-D transverse-walk "
            "generalisation; it is incompatible with a decentered / tilted / "
            "freeform element.  Use displaced_mode='remap' (the 2-D walk-off "
            "remap), 'screen' (the pointwise obliquity screen), or "
            "apply_real_lens_traced / apply_real_lens_gbd.")
    for i, s in enumerate(surfaces):
        if not isinstance(s, dict):
            continue
        if bool(s.get('is_mirror', False)) or (
                isinstance(s.get('glass_after'), str)
                and s['glass_after'].upper() == 'MIRROR'):
            raise NotImplementedError(
                f"apply_real_lens: surface_model='displaced' does not support "
                f"mirror surface {i}; use the per-segment folded pattern.")
        # radius_y (biconic), the analytic freeform_type dispatch, and
        # form_error maps are unsupported on BOTH displaced paths.  The
        # pointwise path takes freeform via the callable ``sag_callable`` hook
        # instead of ``freeform_type``.
        for _k in ('radius_y', 'freeform_type', 'form_error'):
            if s.get(_k) is not None:
                raise NotImplementedError(
                    f"apply_real_lens: surface_model='displaced' does not "
                    f"support surfaces[{i}].{_k} (conic / aspheric surfaces, "
                    f"plus per-surface decenter / tilt / sag_callable on the "
                    f"pointwise path); use surface_model='thin' or "
                    f"apply_real_lens_traced.")
        if s.get('sag_callable') is not None and not _pointwise:
            raise NotImplementedError(
                f"apply_real_lens: surfaces[{i}].sag_callable requires the "
                f"pointwise 2-D obliquity path; it is auto-selected for "
                f"asymmetric elements, or force it with "
                f"displaced_obliquity='pointwise'.")
        if s.get('sag_callable') is not None and not callable(s['sag_callable']):
            raise TypeError(
                f"apply_real_lens: surfaces[{i}].sag_callable must be callable "
                f"(xs, ys) -> sag [m], got {type(s['sag_callable']).__name__}.")
        for _k in ('decenter', 'tilt'):
            _v = s.get(_k)
            if _v is not None and tuple(_v) != (0.0, 0.0) and not _pointwise:
                raise NotImplementedError(
                    f"apply_real_lens: surface_model='displaced' with the "
                    f"meridional (rotationally-symmetric) fan does not support "
                    f"surfaces[{i}].{_k}={_v}.  Use displaced_obliquity="
                    f"'auto'/'pointwise' (the 2-D obliquity path), "
                    f"surface_model='thin', or apply_real_lens_traced.")


def apply_real_lens(
    E_in: np.ndarray,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    bandlimit: bool = True,
    fresnel: bool = False,
    slant_correction: bool = False,
    absorption: bool = False,
    seidel_correction: bool = False,
    seidel_poly_order: int = 6,
    progress: Optional[Any] = None,
    use_gpu: bool = False,
    wave_propagator: Optional[str] = None,
    surface_frame: bool = False,
    sag_dtype: Optional[Any] = None,
    sag_chunk_rows: Optional[int] = None,
    surface_model: str = 'thin',
    conjugate: Any = None,
    displaced_mode: str = 'screen',
    displaced_obliquity: str = 'auto',
    displaced_n_side: Optional[int] = None,
    remap_order: int = 3,
    carrier: Any = None,
    screen_obliquity: Any = 'auto',
    on_screen_obliquity: str = 'warn',
    accumulator_store: str = 'ram',
    scratch_dir: Optional[str] = None,
    stream_transfer_function: bool = False,
    geometry: Optional['LensGeometry'] = None,
    numerics: Optional['LensNumerics'] = None,
    resources: Optional['LensResources'] = None,
    config: Optional['LensConfig'] = None,
) -> np.ndarray:
    """
    Propagate a field through a real lens defined by a surface prescription.

    See Also
    --------
    apply_real_lens_traced :
        Per-pixel ray-traced OPL + wave-optics amplitude envelope.
        3-10x slower, but achieves sub-nm OPD on cemented doublets and
        other multi-surface curved-interface systems where this function
        hits its uniform-glass-slab accuracy ceiling.
    apply_real_lens_maslov :
        Phase-space Maslov propagator via a Chebyshev polynomial fit
        of the canonical map.  Caustic-safe; pair with
        ``apply_real_lens_maslov_jax`` for differentiable design
        optimisation loops.

    Quick decision guide
    --------------------
    * Default / fast wave model -> ``apply_real_lens`` (this function).
    * Sub-nm OPD on cemented doublets / multi-surface curved interfaces
      -> ``apply_real_lens_traced``.
    * Inside a JAX-autodiff design optimisation, or near a caustic
      -> ``apply_real_lens_maslov`` / ``apply_real_lens_maslov_jax``.

    Description
    -----------
    Models the lens as a sequence of refracting phase screens (one per
    surface) with angular-spectrum propagation through the glass between
    them.  Captures exact surface sag (spherical aberration and higher
    orders), diffraction during in-glass propagation, thickness effects, and
    compound lenses (doublets, triplets, etc.).

    The default behaviour uses the **paraxial** thin-element OPD
    ``(n2-n1)*sag`` for the per-surface phase screen: the
    angular-spectrum propagation between surfaces already carries the
    GAPS' obliquity exactly, so the only angle-blind piece left is the
    screen's own coefficient.

    ``slant_correction=True`` replaces that coefficient with the
    eikonal-exact axial-translation identity for a COLLIMATED input,
    ``(n2*cos(theta_i - theta_t) - n1)*sag`` -- both momenta referenced to
    the Z-AXIS, which is what a facet displaced along z requires.  On a
    single refracting face it is 290x-4000x closer to an exact
    vector-Snell trace than the paraxial screen (rms OPD 0.0007 vs
    2.99 nm at R = 100 mm, 0.42 vs 120 nm at R = 20 mm).  End to end it
    helps most where ONE powered face dominates and the input really is
    collimated -- measured exit-OPD rms against an independent ray
    oracle: plano-convex curved-first 0.848 -> 0.037 nm (23x),
    flat-first 1.18 -> 0.017 nm (69x), a parabolic asphere 7.76 -> 0.053
    nm (147x).  On a thick element whose LATER surfaces see a strongly
    converging bundle the collimated assumption is what limits it
    (biconvex R = +-60 1.83 -> 1.64 nm, a fast meniscus R = 20/25
    unchanged at ~0.9 um); for those, take the ray angle from ``carrier=``
    (``screen_obliquity``) or ``surface_model='displaced'`` /
    ``'tangent_facet'``, which implement the same identity at the TRUE
    local ray angle, or use :func:`apply_real_lens_traced`.

    Oblique validity boundary
    -------------------------
    Each surface is modelled as a **normal-projected thin phase
    screen**: ``sag(x, y)`` is the axial (z) surface departure and the
    OPD ``(n2-n1)*sag`` is imprinted on a single axial plane, with exact
    homogeneous angular-spectrum propagation between surfaces.  A thin
    screen collapses the finite ray traverse through the sag onto one
    plane, so the residual OPD error per surface scales as the leading
    obliquity term ``~ sag * theta**2``, where ``theta`` is the local
    ray angle at that surface.  The bound is therefore
    **design-dependent**: it grows with fast (high-NA) surfaces, large
    sag, and off-axis fields, and shrinks toward the axis and for slow
    surfaces.  In **symmetric relays** the even-order (``theta**2``)
    errors of conjugate surfaces partially cancel, so such designs reach
    much sharper OPD agreement than the per-surface bound alone would
    predict -- do not generalise that sharpness to asymmetric systems.
    When ``sag * theta**2`` is not negligible against the target OPD
    tolerance, use :func:`apply_real_lens_traced` (per-pixel ray-traced
    OPL) or ``slant_correction=True`` (partial obliquity correction).

    v5.35.0 makes that bound BOTH correctable and measurable when the caller
    can state the input congruence: pass ``carrier=`` and the exact angular
    part of the thin-facet screen OPD is applied in closed form
    (``screen_obliquity``), while the same expression is read as an error
    estimator that warns when the angle-blindness exceeds lambda/20
    (``on_screen_obliquity``).  Note what is NOT in that bound: the
    angular-spectrum steps between the screens carry the GAPS' angular
    optical path EXACTLY -- a plane-parallel plate is machine-exact at every
    tilt -- so the obliquity piston/tilt of the glass thicknesses is not
    missing and must not be added
    (``docs/audits/BUILD_ANGLE_AWARE_LENS_2026_08_11.md``).

    Optional opt-in features add further physical realism:

    * ``fresnel=True`` -- multiply by the s/p-averaged POWER
      transmittance ``T = (n2 cos theta_t)/(n1 cos theta_i) *
      0.5(|t_s|^2 + |t_p|^2)`` at each surface, using the local angle of
      incidence derived from the surface normal.  The impedance factor
      ``(n2 cos theta_t)/(n1 cos theta_i)`` is required because this
      library's ``sum |E|^2 dx dy`` IS the power (the ASM legs are
      Parseval-unitary and the in-glass propagation adds no impedance
      term), so a bare ``|t|^2`` would under-read a single air->glass
      face by 34 %.  With it, each uncoated air-glass interface costs the
      documented ~4.2 % at normal incidence, a bare cemented N-BK7/N-SF11
      interface 0.64 %, and an air->glass->air element the product
      (8.2 % for two faces) as before.  Works with complex refractive
      indices in the weakly-absorbing sense: the Fresnel coefficients use
      the complex indices but the refraction ANGLE comes from the real
      parts (``sin^2 theta_t = (n1r/n2r)^2 sin^2 theta_i``), and
      ``theta_i`` is the AOI of an AXIAL ray at the local facet normal --
      so a strongly converging bundle's second surface is given the wrong
      incidence, the same normal-incidence ceiling the OPD screen has.
    * ``slant_correction=True`` -- replace the paraxial OPD
      ``(n2-n1)*sag`` with the eikonal-exact axial-translation identity
      ``(n2*cos(theta_i - theta_t) - n1)*sag`` (equation (3) of the
      SCREEN OBLIQUITY derivation in this module), exact for a locally
      planar facet under COLLIMATED illumination.
    * ``absorption=True`` -- apply bulk attenuation
      ``exp(-2*pi*kappa*thickness/wavelength)`` between surfaces using the
      imaginary part of the in-medium index from
      :func:`get_glass_index_complex`.

    Per-surface realism additions (set in the prescription dict, all
    optional and backward-compatible):

    * ``"clear_aperture"`` -- float, mechanical clear aperture diameter at
      this surface [m].  Field outside is zeroed (vignetting).
    * ``"decenter"`` -- ``(dx, dy)`` lateral offset of this surface [m].
    * ``"tilt"`` -- ``(tx, ty)`` small-angle surface tilt [rad].  Adds a
      linear sag ramp ``tx*x + ty*y`` to the surface (field-frame default;
      see ``surface_frame`` for the rigid-body alternative).
    * ``"form_error"`` -- 2D ndarray (same shape as the field) of additive
      sag perturbation [m].  Use to inject measured figure error or
      synthetic Zernike form error.  It is a **FIELD-frame** map: unlike
      ``decenter`` / ``tilt`` it is added after the surface-frame coordinates
      are consumed, so it is neither shifted nor rotated with the surface and
      lands on the field grid pixel for pixel.  The shape must match the field
      exactly (a mismatch raises).

    Field-frame vs surface-frame decenter / tilt (v5.2+)
    ----------------------------------------------------
    The default ``surface_frame=False`` honours ``decenter`` / ``tilt``
    in the **field frame**: the field's ``(x, y)`` grid is shifted by
    ``decenter`` (axis-symmetric sag still evaluated at the shifted
    radius) and the tilt is approximated as a linear sag ramp
    ``tx*x + ty*y``.  This is the v3.x -> v5.1 contract, kept as the
    default so existing callers see no numerical change.

    ``surface_frame=True`` (v5.2+) instead treats the surface as a RIGID
    BODY, matching the Optiland / Zemax treatment of a tilted /
    displaced asphere.  The field's ``(x, y)`` grid is mapped to
    surface-frame coordinates via the inverse rigid-body transform: a
    translation by ``-decenter`` followed by an inverse rotation
    ``R^T`` with ``R = Rx(theta_x) @ Ry(theta_y)`` (full rotation
    matrix, no small-angle linearisation), giving the surface-frame
    FOOTPRINT ``(x_s, y_s)``.  The phase is then
    ``-k0 * (n2 - n1) * z_f`` with the rotated surface's FIELD-frame
    height

    .. code-block:: text

        z_f = R_zx*x_s + R_zy*y_s + R_zz*g(x_s, y_s)
        R_z. = (-cos(theta_x) sin(theta_y), sin(theta_x),
                 cos(theta_x) cos(theta_y))

    -- NOT the bare surface-frame sag ``g(x_s, y_s)``.  Evaluating ``g``
    at the rotated footprint and discarding ``z_s`` drops
    ``R_zx*x_s + R_zy*y_s``, which to first order is the entire tilt
    ramp: a rotation RE-EXPRESSES the ramp, it does not delete it.

    ACCURACY, measured against the exact rigid-body geometry (a sphere
    rotated about its vertex is a sphere with the rotated centre, so the
    field-frame height is closed form).  R = 50 mm over a +-2 mm pupil,
    piston removed: the surface-frame branch reads 1.68 / 10.26 /
    80.1 nm at 1 / 5 / 20 mrad and the default field-frame ramp 1.64 /
    9.10 / 53.4 nm.  Both are under 0.13 waves, and the two are within
    a factor 1.5 of each other -- so ``surface_frame=True`` is NOT the
    more accurate branch for a simple tilt, it is the one that means
    "rigid body" rather than "sheared surface".  Use it when that is
    the geometry you have (off-axis aspheres, decentered parabolas,
    a mount that rotates the part); use the default when the surface is
    specified as a figure plus a wedge.  Both branches read the ``tilt``
    key the same way, so flipping the flag no longer re-points the
    element.

    The prescription dict may also specify ``"stop_index"`` (int) to apply
    the global ``"aperture_diameter"`` at a specific surface (the aperture
    stop) rather than at the entrance.

    Parameters
    ----------
    E_in : ndarray (complex, N x N)
        Input electric field.
    prescription : dict
        Required keys:

        ``"surfaces"`` : list of dict
            Each surface dict contains:

            - ``"radius"`` : float -- SIGNED radius of curvature [m]
              (``inf`` = flat).  Sign convention (v5.30, audit E-M12):
              **R > 0 puts the centre of curvature on the transmission
              (downstream) side**, i.e. the surface is convex toward the
              input -- identical to the ``R1`` / ``R2`` convention of
              :func:`lumenairy.elements.apply_spherical_lens`, and the
              same sign the library's ``surface_sag_general`` /
              ``conic_sag`` helpers use (sag > 0 off-axis for R > 0).
              Consequences: a converging plano-convex singlet is
              ``radius=+R`` then ``inf``; a converging biconvex is
              ``+R`` then ``-R``; the LAST surface of a converging
              element has ``radius < 0``.  Verified by measurement --
              ``system_abcd_prescription`` reports
              ``EFL = +97.07056596 mm`` for a 3 mm N-BK7 plano-convex
              with ``radius=[+50 mm, inf]`` at 632.8 nm, matching the
              lensmaker value ``R/(n-1) = 97.07056596 mm`` to 10
              digits, and ``-50 mm`` flips it to ``-97.07 mm``
              (diverging).
            - ``"conic"`` : float -- conic constant (0 = sphere)
            - ``"aspheric_coeffs"`` : dict or None -- {4: A4, 6: A6, ...}
            - ``"glass_before"`` : str -- glass name before this surface
            - ``"glass_after"``  : str -- glass name after this surface
            - ``"clear_aperture"`` : float, optional -- per-surface aperture [m]
            - ``"decenter"`` : (dx, dy), optional -- lateral offset [m]
            - ``"tilt"`` : (tx, ty), optional -- small-angle tilt [rad]
            - ``"form_error"`` : ndarray, optional -- additive sag map [m]

        ``"thicknesses"`` : list of float
            Center spacing [m] between consecutive surfaces.

        Optional keys:

        ``"aperture_diameter"`` : float -- clear aperture [m] (entrance, or
            applied at ``stop_index`` if provided).
        ``"stop_index"`` : int -- index of the surface that holds the
            aperture stop.
        ``"allow_unfolded_equivalent"`` : bool, default False -- acknowledge
            the UNFOLDED-EQUIVALENT treatment of a folded design.  This
            function walks refracting surfaces only, so a prescription that
            carries a fold mirror -- either as a ``'mirror'`` entry in
            ``"elements"`` (what ``load_zemax_zmx`` emits) or as a surface
            with ``is_mirror=True`` / ``glass_after='MIRROR'`` -- is REFUSED
            by default.  Setting this key to ``True`` accepts the unfolded
            walk instead: a mirror surface becomes an index-neutral FLAT at
            its own vertex plane, so every gap and both reference planes stay
            put and a scalar on-axis field through a FLAT fold is exact, while
            a curved mirror's focusing phase and a tilted/decentred mirror's
            world-frame axis change are DROPPED (the call warns, naming
            which).  The exact alternative is
            ``lumenairy.io.split_prescription_at_mirrors`` plus
            :func:`~lumenairy.elements.apply_mirror` at each fold.  Both
            spellings of a mirror read this one key.
        ``"name"`` : str -- human-readable label.

    wavelength : float
        Free-space wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx`` (square pixels).
        Anamorphic / non-square grids are supported throughout the
        per-surface phase-screen + in-glass ASM pipeline.
    bandlimit : bool
        Apply band-limiting in ASM propagation steps (default True).
    fresnel : bool
        Apply the s/p-averaged POWER transmittance
        ``T = (n2 cos theta_t)/(n1 cos theta_i) * 0.5(|t_s|^2 + |t_p|^2)`` at
        each surface (see the "Optional opt-in features" section above for the
        convention and the measured per-interface numbers).

        KNOWN APPROXIMATION.  ``theta_i`` is the angle between the local
        surface normal and the Z AXIS -- the AOI of an axial ray -- not the AOI
        of whatever bundle the field actually carries, so the second surface of
        a strongly converging element is given the wrong incidence.  No path in
        this function has BOTH a true local AOI and Fresnel: the angle-true
        models (``surface_model='displaced'`` / ``'tangent_facet'``) refuse
        ``fresnel``, and ``slant_correction`` -- which shares this same axial
        ``cos_ti`` -- refuses ``'displaced'``.  Route polarised or
        high-NA throughput work through the Jones pipeline or
        :func:`apply_real_lens_traced` instead.
    slant_correction : bool, default False
        Use the eikonal-exact thin-facet OPD
        ``(n2*cos(theta_i - theta_t) - n1)*sag`` -- the axial-translation
        identity (3), with both optical momenta referenced to the Z-AXIS,
        which is what a facet sitting a height ``sag`` above the vertex
        plane requires.  ``theta_i`` is the local facet tilt and
        ``theta_t`` its Snell refraction, both taken from the SURFACE
        NORMAL of an AXIAL ray, so the expression is exact for a
        collimated input and degrades as the bundle acquires its own
        angle.  Off by default because it is an approximation of a
        different kind from the paraxial screen, not a strict superset:
        use ``carrier=`` (``screen_obliquity``),
        ``surface_model='displaced'`` / ``'tangent_facet'`` or
        :func:`apply_real_lens_traced` when the input is not collimated.
        Measured gains are in the module docstring above and in
        ``validation/real_lens_opd``.

        Mutually exclusive with ``seidel_correction=True`` (the Seidel
        reference is built from the PARAXIAL screen, so stacking the two
        double-counts the obliquity); that combination raises.
    absorption : bool
        Apply bulk attenuation through each glass region using the
        extinction coefficient from :func:`get_glass_index_complex`.
    seidel_correction : bool, default False
        Add a radially-symmetric high-order OPD correction at the exit
        pupil, derived from a 41-ray geometric fan across the clear
        aperture.  What is fitted is the difference between (a) that
        fan's OPL on the EXIT VERTEX PLANE and (b) THIS MODEL'S OWN exit
        OPL on the same fan -- a thin-screen ray walk through the very
        screens this function applies and the glass gaps its ASM legs
        propagate -- so the residual is the split-step model's own
        error and nothing else.  The fit starts at ``rho**4`` (a
        ``rho**2`` term is defocus, not a high-order residual), it is
        HELD CONSTANT beyond the largest radius the fan actually lands
        at rather than extrapolated, and it is skipped entirely unless
        the fitted part exceeds 5 nm rms.  Cost: 41 traced rays, one
        least-squares fit and one 2-D phase multiplication.

        Measured exit-plane OPD rms against an independent
        closed-form-intersection + vector-Snell ray oracle, correction
        off -> on: an 8 mm cemented doublet 173.6 -> 1.05 nm (165x),
        the same doublet at 4 mm 10.9 -> 2.6 nm, a meniscus 5.21 ->
        0.03 nm, a four-surface air-spaced doublet 29.8 -> 0.04 nm, an
        f/2 singlet 402.6 -> 1.50 nm (268x).  On a well-corrected
        singlet (model residual 0.85 nm) the gate SKIPS and the call is
        bit-identical to leaving the flag off.

        LIMITS.  The fan is COLLIMATED and ON AXIS, so the screen is a
        radial function: it is not valid for a non-collimated or
        off-axis input (measured on a fast biconvex it still helps at
        20-50 mrad of input tilt, but by progressively less).  It also
        ignores per-surface ``decenter`` / ``tilt`` / ``form_error``,
        which the fan does not see.  Mutually exclusive with
        ``slant_correction`` (both replace the same per-surface
        coefficient).  For a per-pixel ray-traced OPL with none of
        these restrictions use :func:`apply_real_lens_traced`.

        It also assumes the exit field FILLS the pupil the fit is
        normalised to.  On a fast, thick element it does not: the
        transverse walk is inward, so the outer pupil carries only the
        diffractive tail of the geometric field and neither the ray
        trace nor this model's own eikonal describes it (measured on an
        f/2 singlet at converged sampling: |E| falls 20x between
        rho = 0.85 and rho = 0.93 and the model-vs-wave difference over
        the full pupil is 3.7 um, against 1.5 nm over rho <= 0.85).
        Whatever the radial screen puts on that annulus -- and it must
        put something -- scatters its ~5 % of the energy out of the
        core: on that fixture the exit wavefront over rho <= 0.85
        improves 268x while the focal PEAK drops 37 %.  Judge this
        option on a filled pupil, or use
        :func:`apply_real_lens_traced`, which has no radial screen.

        SAMPLING.  Judging this option (or any exit-OPD measurement on
        a hard-apertured prescription) needs ``dx`` around
        ``1.45 * aperture_diameter / 2048`` -- NOT the
        ``0.3 * wavelength / NA`` a carrier-Nyquist rule gives: on a
        coarser grid the aperture edge aliases through the in-glass ASM
        and the exit phase reads hundreds of nm of error that is
        entirely the grid (measured on a meniscus: 558 nm at N = 512,
        converging to 0.03 nm by N = 2048 over the same window).
    seidel_poly_order : int, default 6
        Highest even power of the radial polynomial fit used for the
        Seidel correction.  Order 4 is classical spherical-aberration
        (``a*r^4``); the default 6 adds the 6th-order spherical term and
        8 adds the 8th; higher is rarely beneficial because the fit is
        limited by the 1-D sampling rather than by the polynomial basis.
        Must be a positive int and is capped at 12 (validated).
    surface_frame : bool, default False
        v5.2+ opt-in.  When ``False`` (default), the per-surface
        ``"decenter"`` / ``"tilt"`` keys are honoured in the **field
        frame**: sag is evaluated on the field's ``(x, y)`` grid shifted
        by ``decenter`` and a linear sag ramp ``tx*x + ty*y`` is added
        for tilt.  This is the v3.x -> v5.1 contract and is preserved
        bit-for-bit when the flag is left at its default.

        When ``True``, the per-surface ``"decenter"`` / ``"tilt"`` are
        applied as a rigid-body transformation of the surface itself
        (Optiland / Zemax style): the field grid is mapped through the
        inverse transform to the surface-frame FOOTPRINT, and the phase
        imprints the rotated surface's FIELD-frame HEIGHT ``z_f``, not
        the bare surface-frame sag.  Use for off-axis aspheres /
        decentered parabolas where the sag's curvature must rotate with
        the surface.  It is not a strictly more accurate branch for a
        simple tilt -- measured against the exact rotated sphere, the
        two branches sit within a factor 1.5 of each other and both
        under 0.13 waves out to 20 mrad -- it is the one that means
        "rigid body" rather than "figure plus wedge".  Both branches
        read ``tilt`` the same way, so flipping the flag does not
        re-point the element.  See the "Field-frame vs surface-frame
        decenter / tilt" docstring section above for the physics and
        the measured numbers.
    sag_dtype : {None, np.float32, np.float64}, default None
        v5.17.0 opt-in geometry (coordinate/sag/OPD) dtype.  ``None``
        (default) resolves to the process-wide
        :func:`set_lens_sag_dtype` value, which defaults to float64 --
        byte-identical to prior releases.  ``np.float32`` halves the
        float64 coordinate/sag/opd core (enabling larger grids) but is
        ACCURACY-RISKY: the exit-field error scales with the total sag
        depth and is config-dependent (dx / grid fill), so validate
        the prescription with :func:`lens_sag_float32_opd_error` at
        your production sampling before trusting a float32-sag result.
    sag_chunk_rows : int or None, default None
        v5.17.0 row-band (chunked) sag / phase-screen evaluation.
        ``None`` -> AUTO: row-banded (``max(256, N // 16)`` rows per
        band) when ``N >= 4096``, whole-grid below.  ``0`` forces the
        whole-grid path; a positive int forces that band size.  The
        banded path is BYTE-IDENTICAL to the whole-grid path (every
        banded op is pointwise, same numexpr complex128-internal
        phase screen), while the full-grid coordinate / sag / OPD
        transients never materialise (~tens of GB reclaimed at
        N=32768).

        WALL CLOCK: banding is NOT free on a small grid.  Measured on a
        three-surface element (best of 3, numexpr absent), an explicit
        ``sag_chunk_rows=256`` against the whole-grid path: **+28 % at
        N = 512** (145.0 -> 186.0 ms), +5 % at N = 1024 (1149 -> 1211 ms)
        and +9 % at N = 2048 (4010 -> 4387 ms).  The memory payoff is the
        real one and it is large: the tracemalloc peak drops from 16.1 to
        6.4-6.8 float64 grids (2.4x).  The AUTO default only bands at
        N >= 4096, where the transients dominate, so the shipped default
        pays none of that -- but a caller who forces a band on a small
        grid is buying memory with time, not getting both.

        Surfaces outside the narrow
        chunk-eligible case (decenter / tilt / form error / biconic /
        freeform / clear_aperture / stop surface / fresnel / slant /
        surface-frame, or a non-NumPy backend) fall through to the
        whole-grid path per surface.
        v5.35.3 let ``carrier=`` (the angle-true screen) into the band with a
        1-/2-row halo; v5.37 let the ``'tangent_facet'`` family in with a
        3-/2-row one.  The one piece that stays whole-grid is the remap rung's
        PULL-BACK, whose halo is the transverse walk itself; it is priced per
        call and reported through ``progress`` rather than approximated.
    surface_model : {'thin', 'displaced', 'tangent_facet', \
'tangent_facet_remap'}, default 'thin'
        v5.25.1 opt-in refraction-OPD model (hammer audit H2(a)).

        ``'thin'`` (default): the paraxial thin-element screen
        ``(n2 - n1) * sag(r)`` (optionally slant-corrected), byte-identical
        to prior releases.

        ``'displaced'``: the ray-angle-aware refraction OPD
        ``(n2 cos(alpha_out) - n1 cos(alpha_in)) * sag(r)``, where
        ``alpha_in`` / ``alpha_out`` are the TRUE ray angles to the z-axis
        (before / after each surface) sourced from a meridional ray fan
        traced through the actual conic/aspheric prescription along the input
        congruence (see ``conjugate``).  This restores the incoming-ray-angle
        obliquity the paraxial screen drops -- most importantly it is NO
        LONGER orientation-invariant on a plano-convex singlet (the paraxial
        screen imprints the identical map for both orientations), so it
        reproduces the textbook ~4x spherical-aberration split between the
        curved-first and flat-first orientations.  Validated against both
        hammer-campaign oracles at Nyquist-compliant sampling (f/5 biconvex
        r2m 64.5 vs 64.98 um, 0.7%; plano-convex 42/127 vs 43/128 um;
        EE50/EE80 matched).  NB: the exit converging wavefront must be
        Nyquist-sampled (``dx <= lambda / (2 NA_exit)``, cf. finding H3 for
        ``apply_real_lens_traced``) or the windowed r2m aliases LOW -- the
        ~40 um analytic "plateau" the 2026-07-18 audit reported was a
        dx=6 um undersampling artefact (traced reads the same 40.9 um there
        and 64.8 um at dx<=3 um), not a model floor.  The default meridional
        obliquity path supports rotationally-symmetric plain conic / aspheric
        surfaces; per-surface decenter / tilt / freeform (``sag_callable``) are
        supported via the 2-D transverse-walk remap (P10 / N11, the DEFAULT for
        such elements) or the pointwise 2-D obliquity screen (see
        ``displaced_obliquity`` / ``displaced_mode``).  Biconic ``radius_y`` / analytic
        ``freeform_type`` / ``form_error`` / mirror / GPU / non-ASM / fresnel /
        slant / seidel / absorption still raise; use ``apply_real_lens_traced``
        outside that envelope.

        ``'tangent_facet'`` (route 3, opt-in): the PER-PIXEL tangent-facet
        screen.  Each pixel's OPD is the exact axial-translation identity
        ``(pz2 - pz1) * sag`` at the facet tangent to the surface where THAT
        pixel's ray meets it, evaluated at the ray angle the FIELD ITSELF
        carries -- a momentum accumulator seeded by ``carrier=`` (or by zero),
        advanced by the gradient of every screen imprinted so far, and
        resampled across each gap.  Two second-order terms follow from taking
        the facet at the hit point and from referencing the ray's transverse
        walk back to the pixel; the full derivation, the term ladder and the
        oracle numbers are above ``_tangent_facet_screen``.

        Unlike ``carrier=`` on the ``'thin'`` screen -- which corrects only the
        ANGULAR part, as a difference against the model's own zero-angle value
        -- this REPLACES the paraxial facet coefficient, so it also repairs the
        steep-facet error at NORMAL incidence, where the correction is
        identically zero by construction.  Measured against the shipped exact
        ray tracer on an R = 12.6 mm N-SSK2 biconvex at 0 mrad: blind and
        ``carrier``-corrected both 0.00141 waves rms, ``'tangent_facet'``
        0.00008 (17.6x).  At 100 mrad on an R = 19.6 mm singlet: blind 0.00423,
        ``carrier`` 0.00050, ``'tangent_facet'`` 0.00017.

        ``carrier=`` is accepted and is what supplies the arrival angle;
        ``screen_obliquity=True`` and ``slant_correction=True`` are REFUSED as
        double-counts, and the ``screen_obliquity`` accuracy guard is silent
        (it estimates the size of a correction this model does not make).
        ``surface_frame`` / ``use_gpu`` / non-ASM propagators raise -- not
        because they cannot work but because they have not been measured.

        COST.  v5.37 ROW-BANDS this path, with the halo that its
        differentiate-a-gradient-twice structure requires: 3 rows of sag and
        2 of the persistent momentum accumulator, derived above ``_tf_sl`` and
        pinned byte-identical in ``tests/unit/test_tf_banded_halo.py``.  It
        therefore follows the same ``sag_chunk_rows`` AUTO convention every
        other screen follows.  Warmed ``tracemalloc`` peak, in float64 grids of
        ``8*N*N`` bytes, as extras over the paraxial no-carrier call AT THE SAME
        BANDING (2026-08-16, biconvex singlet, band = ``max(256, N//16)``):
        **+4.06 grids banded at N = 4096 against +17.74 whole-grid** (+4.11 /
        +17.36 at N = 2048), and the old +4-grid carrier surcharge is gone --
        a collimated carrier now costs nothing and a finite-radius one +3.56.
        At N = 32768 one grid is 8.59 GB, so the term drops from +152 GB to
        +35 GB.  The banded arm is BYTE-IDENTICAL to the shipped 5.36 whole-grid
        output (960-arm two-tree comparison) and is at wall-clock parity with
        it (0.95x at N = 4096, measured interleaved on a shared box).

        ``'tangent_facet_remap'`` (the REMAP rung, opt-in): the same
        tangent-facet physics with the transverse walk REPRESENTED instead of
        referenced away.  ``'tangent_facet'`` imprints its screen on the vertex
        plane and Taylor-references the ray's displaced re-crossing back to the
        pixel; this one imprints the screen and then RESAMPLES the field to the
        walked positions, so the element is a screen PLUS a coordinate remap and
        the exit eikonal is evaluated where the ray actually is.  The OPD
        collapses to the exact one-line path difference
        ``sag_hit * (n2^2/pz2 - n1^2/pz1)`` and the walk is
        ``sag_hit * (p/pz1 - p_out/pz2)``; the full derivation, the term ladder,
        the amplitude-Jacobian derivation and the fold guard are above
        ``_tangent_facet_remap_screen``.

        WHY IT IS NOT JUST ANOTHER TERM.  A vertex-plane screen's kick is the
        gradient of its own value, which is why no screen can carry the exact
        facet kick.  A remap's kick is the gradient of the COMPOSITE, and that
        equals the exact refracted momentum identically -- verified at 1.5e-15
        to 4.6e-14 relative on a plane facet, where the model has no truncation
        left.  Against exact rays on design 121's four powered groups at a 3 mm
        pupil (waves rms): ``'tangent_facet'`` 0.0000046 / 0.0000033 /
        0.0000005 / 0.0032381 for g2/g3/g4/g5, this model 1.12e-10 / 5.03e-11 /
        4.17e-12 / 2.56e-08.

        CAUSTIC SAFETY.  A remap is a ray map, so it must be single-valued.
        ``det(I + dW/dx) > 0`` is exactly that statement, and a non-positive
        (or near-zero, or non-invertible-on-this-grid) determinant RAISES rather
        than silently returning one branch of a fold.  Element interiors of this
        family are caustic-free by design contract -- design 121 group 5 runs at
        ``det`` in [0.927, 1.021] -- and the guard is the proof.  Near a genuine
        caustic use ``apply_real_lens_maslov``.

        Everything ``'tangent_facet'`` refuses, this refuses too, for the same
        reasons; it is additionally scipy-bound (the pull-back uses
        ``scipy.ndimage.map_coordinates``) and its resampling order is
        ``remap_order``.  COST: v5.37 bands the SCREEN half (a 2-row sag halo,
        and NO accumulator halo -- (R3) hands it ``p_out`` in closed form), and
        REFUSES to band the pull-back: its halo is the WALK, a length rather
        than a row count, and three of its steps are globally coupled.  That
        refusal is priced per call against the band actually in use and
        reported through ``progress``.  Warmed extras over the paraxial
        no-carrier call at the same banding: **+13.62 grids banded at N = 4096
        against +23.74 whole-grid** (+13.24 / +23.36 at N = 2048).  See the
        derivation above ``_tf_sl``, ``remap_order``, and the module note.
    remap_order : int, default 3
        Spline order of the transverse-walk resampling, for
        ``surface_model='tangent_facet_remap'`` only (any other model raises if
        it is not the default).  One of 1 / 3 / 5 -- the orders
        ``scipy.ndimage.map_coordinates`` offers that this build has measured.
        3 is the library's standing high-order resampling choice (the same one
        ``_apply_displaced_remap_2d`` and ``_lens_imap`` use).  Measured
        2026-08-16 on the R = 12.6 mm biconvex at 4 um sampling, N = 1536, as a
        fraction of the peak amplitude: ``|o3 - o1|`` 3.69e-04 and
        ``|o5 - o3|`` 1.37e-04, for 9.2 / 15.5 / 26.4 s of wall clock.  The
        order-5 gap is only 2.7x below the order-3 one on that fixture because
        4 um is close to the exit wavefront's Nyquist there -- the resampling
        converges with the SAMPLING, not with the order alone, which is the
        same statement the model's own grid-gradient scan makes.
    conjugate : {None, float, 'auto', ndarray}, default None
        G2 Task 1 -- the INPUT CONGRUENCE for the ``surface_model='displaced'``
        obliquity fan (same vocabulary as ``apply_real_lens_traced``'s
        ``carrier``).  Only used when ``surface_model='displaced'`` (else it
        must be ``None`` or a ``ValueError`` is raised).

        * ``None`` (default) -- COLLIMATED input.  The fan launches axially;
          byte-identical to the pre-G2 collimated fan and exact for a
          collimated beam.
        * ``float`` -- a signed on-axis conjugate distance ``R_in`` (m):
          ``R_in > 0`` a diverging source in front of the lens, ``R_in < 0``
          converging.  The fan is launched with marginal slope ``h / R_in`` so
          the second (and later) surfaces see the true incidence; the OPD (1)
          then reflects the actual converging/diverging illumination.
        * ``'auto'`` -- fit a low-order polynomial carrier from ``E_in``
          (reuses ``_compute_carrier``) and launch the fan along its meridional
          slope.  For a single divergent source of unknown conjugate.
        * ``ndarray`` -- an explicit input wavefront ``W`` (m, field-shaped).

        ON THE SCREEN PATHS (``displaced_mode='screen'`` with a symmetric
        element, or ``displaced_obliquity='pointwise'``) the wave field itself
        carries the input curvature in its phase and ``conjugate`` ONLY informs
        the per-surface obliquity cosines: it adds no reference phase and does
        not modify ``E_in``.

        ON THE REMAP PATHS (``displaced_mode='remap'``, and the DEFAULT routing
        for a decentered / tilted / ``sag_callable`` element) the exit phase is
        rebuilt from the ray eikonal, which carries this congruence's entrance
        eikonal.  The input field is therefore DEMODULATED by ``W_conj`` before
        it is resampled and the residual ``angle(E_in) - k0 W_conj`` is
        transported along the traced rays and re-applied -- so an upstream
        element's wavefront, a tilt or an aberration reaches the exit plane
        instead of being replaced by the idealised congruence.  The residual is
        carried as a COMPLEX field through a bilinear resample, so it must be
        smooth on the grid: the closer ``conjugate`` is to the field's actual
        congruence, the better the transport.  (Before this was fixed a flat
        and a 35-wave-defocused input produced identical output, and a 150 mm
        diverging source focused at the collimated 21 mm instead of 25 mm.)

        The screen is field-independent given the conjugate, so the ``None`` /
        scalar paths are cached (bounded + registered; ``'auto'`` / ndarray
        rebuild).  Envelope + measured accuracy: see
        ``docs/audit_real_lens_displaced_2026_07_19.md`` (G2 section).
    displaced_mode : {'screen', 'remap', 'split'}, default 'screen'
        P2 (niche N1) opt-in EXPERIMENTAL sub-model of
        ``surface_model='displaced'`` for extreme finite conjugates.  Only used
        when ``surface_model='displaced'`` (else it must be ``'screen'`` or a
        ``ValueError`` is raised).

        * ``'screen'`` (default) -- the per-surface obliquity screen + in-glass
          ASM (the G1/G2 displaced model).  BYTE-IDENTICAL to prior releases for
          a SYMMETRIC element.  For a decentered / tilted / freeform element the
          default routes to the 2-D transverse-walk remap (see below /
          ``displaced_obliquity``).
        * ``'remap'`` -- the exit-plane geometric-transfer remap: the element
          becomes a coordinate map (the traced ray map) with an energy-conserving
          amplitude Jacobian plus the exit-pupil-referenced eikonal OPD, so the
          transverse ray walk THROUGH the element is captured explicitly.  P2
          (N1) rotationally-symmetric 1-D form for symmetric elements; P10 (N11)
          full 2-D form for decentered / tilted / freeform elements (which the
          DEFAULT ``'screen'`` also selects for such elements).
        * ``'split'`` -- entrance/exit obliquity screens with the internal gap
          propagated as the REDUCED distance ``t / n`` in air (P2 candidate b);
          rotationally-symmetric only -- rejected for an asymmetric element.

        **Measured (P2, congruence-fixed diffraction oracle
        ``validation/oracles/debye_oracle_v3.py`` + ZOS POP):** for the extreme
        CONJUGATE (symmetric) cases the DEFAULT ``'screen'`` is already within
        ~4-8% of the diffraction-faithful oracle (M5 real 0.96x, virtual 1.00x,
        M1 doublet 0.92x, M6 0.98x) and ``'remap'`` / ``'split'`` match it to a
        few percent (the prior "~0.50x floor" was a geometric-spot artefact, not
        a model floor).  **For a DECENTERED element (P10 / N11)** the 2-D
        ``'remap'`` restores the transverse walk-off the single-plane screen
        drops, so the induced-coma spot BROADENS correctly instead of narrowing:
        measured by the RMS second-moment radius + common-mode coma RMS (the
        honest metric -- the EE80 is diffraction-diluted, as for GBD), the on-axis
        RMS ~21 um matches the GBD reference, the RMS broadens ~1.02 @1 mm
        (grid-robust, sign-mirror exact), and the coma RMS matches the geom oracle
        within ~10% -- where the single-plane screen SHRINKS (RMS 0.956).  It is
        the DEFAULT for asymmetric elements.  See
        ``docs/audit_real_lens_displaced_2026_07_19.md`` (P2 + P10 sections) for
        the full measured tables + routing story.
    displaced_obliquity : {'auto', 'meridional', 'pointwise'}, default 'auto'
        P3 (niche N2) selector for the ``surface_model='displaced'`` obliquity
        path.  Only meaningful when ``surface_model='displaced'`` (else it must
        be ``'auto'`` or a ``ValueError`` is raised).

        * ``'auto'`` (default) -- the fast MERIDIONAL cosine LUT for
          rotationally-symmetric elements (byte-identical to prior releases); for
          a decentered / tilted / freeform element it routes to the 2-D
          TRANSVERSE-WALK REMAP (P10 / N11), which carries the walk-off the
          single-plane screen drops so the induced-coma spot broadens correctly.
        * ``'meridional'`` -- force the 1-D radial LUT (raises on an asymmetric
          element it cannot represent).
        * ``'pointwise'`` -- force the 2-D obliquity SCREEN: a 2-D ray grid
          launched along the input congruence is traced through the actual
          (possibly decentered / tilted / freeform) surfaces and its per-surface
          z-axis cosines are imprinted on the field grid.  On a symmetric element
          it reproduces the meridional LUT to <0.1% (the convention-bug killer).
          This is a single-plane phase SCREEN -- it captures the coma DIRECTION
          but NOT the walk-off spot growth (see the note below); it is retained
          as a documented peer, and the DEFAULT ('auto') routes to the remap.

        Per-surface asymmetry is set in the surface dict: ``decenter=(dx, dy)``
        [m] evaluates the sag at ``(x-dx, y-dy)``; ``tilt=(tx, ty)`` [rad] adds
        the small-angle field-frame linear ramp ``tx*x + ty*y`` and the
        correspondingly tilted normal (the deflection magnitude matches an
        independent rigid-rotation ray trace to <0.5%; opposite sign is the
        differing 'positive tilt' definition); ``sag_callable(xs, ys) -> sag``
        [m] supplies a freeform surface departure (used in BOTH the ray trace and
        the OPD imprint).  Validated (P3): decenter centroid shift within ~2.5% of
        ZOS, tilt within 0.3%, the coma flare DIRECTION (skewness sign), and
        +d/-d PSF mirror all exact.  COMA SPOT GROWTH (P10 / N11): the DEFAULT
        2-D remap BROADENS the decentered spot with the correct MAGNITUDE -- by the
        RMS + common-mode coma-RMS metric (the honest gate; the EE80 is
        diffraction-diluted, as for GBD) the coma RMS matches the geometric oracle
        within ~10% (RMS ratio ~1.02 @1 mm, on-axis RMS 21 um = the GBD reference,
        grid-robust) -- this closes the P3 open finding.  The single-plane
        ``'pointwise'`` SCREEN, by contrast, imprints the OPD at the
        straight-through position and CANNOT represent the transverse ray walk, so
        it NARROWS (RMS 0.956) where truth broadens -- it is
        the documented walk-off-limited peer.  See
        ``docs/audit_real_lens_displaced_2026_07_19.md`` (P3 screen limit + P10
        remap fix).
    displaced_n_side : int or None, default None
        Side of the SQUARE LAUNCH LATTICE the 2-D transverse-walk remap traces,
        in RAYS; ``None`` uses the module default (257).  The remap is a
        geometric transfer, so this -- not ``dx`` -- sets the transverse
        resolution of its output: the launch pitch is
        ``2 * r_aperture / (displaced_n_side - 1)`` across the traced aperture,
        the fan itself is launched 3 % wider so the edge rays have interior
        Jacobian neighbours, and input structure finer than that pitch (a hard
        stop edge, an obscuration, an upstream DOE, speckle) is SMOOTHED to the
        lattice.  The call warns, naming both pitches and the lattice that
        would clear the bar, whenever the launch pitch is coarser than twice
        the field pitch.

        Cost is ``displaced_n_side**2`` rays traced through the prescription;
        accuracy is second order in the launch pitch (measured 4x per
        doubling, see :data:`_DISP_REMAP_2D_N_SIDE`), with no upper limit
        imposed by the model -- the exit map is inverted on its own regular
        launch grid, so a denser lattice cannot produce the degenerate cells
        that made a scattered triangulation lose reflection symmetry.

        Only meaningful when the call actually routes to the 2-D remap (an
        asymmetric element under ``surface_model='displaced'`` with
        ``displaced_mode='remap'``, or the default ``'screen'`` +
        ``displaced_obliquity='auto'``); anything else raises rather than
        discard the setting.
    carrier : TiltedCarrier / float / 'auto' / ndarray / None, default None
        The INPUT CONGRUENCE, in the same vocabulary
        :func:`apply_real_lens_traced` takes: a
        :class:`~lumenairy.TiltedCarrier`, a signed on-axis conjugate distance
        [m], ``'auto'`` (a low-order fit of ``E_in``'s own phase), or an
        explicit wavefront array [m].  Supplying it (v5.35.0) does two things
        and nothing else:

        1. enables the **screen-obliquity correction** -- the closed-form
           angular part of the exact thin-facet screen OPD (see
           ``screen_obliquity``), and
        2. enables the **screen-obliquity accuracy guard** (see
           ``on_screen_obliquity``), which fires even with the correction
           switched off.

        The wave field still carries its own phase; ``carrier`` only states
        the local ray angle at which the sag screens are crossed.  Only
        supported with the default ``surface_model='thin'`` -- the
        ``'displaced'`` path is already angle-aware through ``conjugate=``,
        and stacking the two would double-count.  With ``carrier=None``
        (the default) this function's output is bit-unchanged.
    screen_obliquity : {'auto', True, False}, default 'auto'
        Whether to APPLY the screen-obliquity correction.  ``'auto'`` applies
        it whenever a ``carrier`` is supplied; ``False`` computes the guard's
        estimate but leaves the screens alone; ``True`` requires a
        ``carrier``.

        Each surface is modelled as a thin screen on its VERTEX plane, so the
        shipped ``(n2-n1)*sag`` OPD is the OPD of a ray crossing the sag at
        NORMAL incidence.  The exact eikonal cost of a locally planar facet
        sitting a height ``sag`` above that plane is ``(n2 cos(alpha_out) -
        n1 cos(alpha_in)) * sag`` with the angles taken to the Z-AXIS (the
        axial-translation identity, derived in the module comment); the
        correction applied here is that MINUS its normal-incidence value, so
        it is identically zero for a plane-parallel plate at every tilt,
        identically zero without a carrier, and leaves whichever screen you
        selected (paraxial / ``slant_correction`` / ``'displaced'``) as the
        zero-angle behaviour.  It is closed form -- per-surface sag gradients
        and the carrier's own direction cosines, no ray trace, no fit and no
        cache -- but it is not free: it adds a sag gradient and ~20 full-grid
        float operations per POWERED surface (flat faces are skipped by a
        single reduction).

        COST, re-measured (N-SSK2 biconvex, ``sag_chunk_rows=0`` on both
        arms, every path warmed at the same N first,
        ``on_screen_obliquity='silent'``; peak in units of one float64
        ``N**2`` grid):

        .. code-block:: text

            N      thin        + collimated carrier   + finite-R carrier
            512    0.209 s     3.88x, +11.13 grids    3.40x, +13.13
            1024   0.407 s     5.09x, +11.13          6.14x, +13.13
            2048   3.084 s     2.38x, +11.13          2.43x, +13.13

        Read the TREND, not the individual numbers (the N = 1024 thin
        baseline is an FFT-size outlier): a fixed O(N^2) per-surface
        addition measured against an O(N^2 log N) baseline must FALL with
        N, which is what these do.  An earlier revision of this docstring
        quoted 2.2x / 2.9x / 3.6x -- a RISING trend, which cannot be right
        for this shape of work -- and '+3 float geometry grids', where the
        peak surcharge measures +11.13 (+12.13 under the default policy,
        which also builds ``_obl_total``).  The correction also routes the
        surface loop to the whole-grid path for its gradient halo when the
        band cannot carry one.
    on_screen_obliquity : {'warn', 'error', 'silent'}, default 'warn'
        Policy for the accuracy guard.  With a ``carrier`` supplied, the same
        closed form is read as an ERROR ESTIMATOR: the piston-and-tilt-free
        rms of the summed correction over the pupil IS the wavefront error
        the angle-blind screens carry at those ray angles.  When it exceeds
        the documented tolerance (0.05 waves = lambda/20; and with the
        correction applied, when 10% of it still does -- the budgeted
        next-order residual, ``_SCREEN_OBLIQUITY_RESIDUAL_FRAC``, which the
        R1 entrance-curvature term took from 0.40 to 0.10) the guard emits a
        ``RuntimeWarning`` naming the
        number and recommending :func:`apply_real_lens_traced`.  ``'error'``
        raises instead; ``'silent'`` suppresses.  Carrier-free calls are
        always silent -- there is no angle to estimate against.

    accumulator_store : {'ram', 'memmap'}, default 'ram'
        Where the PERSISTENT full-grid accumulators live (v5.40).  Everything
        else this function allocates at full-grid size is a transient that
        ``sag_chunk_rows`` already keeps to one band; what banding cannot
        remove is the state that must be simultaneously live across the whole
        grid while the band loop walks it -- the tangent-facet momentum pair
        and the fresh destination pair each surface writes into, the remap
        rung's walk components, and the screen-obliquity momentum / drift /
        carrier-momentum pairs and guard accumulator.

        ``'memmap'`` backs each of those with an ``np.memmap`` in
        ``scratch_dir``, so the OS holds only the bands under the cursor.
        The accumulators are written once per surface in increasing row order
        and read back band by band on the next surface, which is the access
        pattern this trades RAM for.  At ``N = 32768`` one accumulator is
        8.59 GB, so the tangent-facet route-3 set alone is 34.4 GB.

        **Byte-identical to** ``'ram'``: the store changes only where an
        accumulator's bytes live.  Every expression that touches one is
        unchanged and sees a plain ``np.ndarray`` view of the mapping, so no
        ufunc can dispatch differently.  Files are removed on EVERY exit,
        including an exception raised mid-prescription.

        NumPy backend only -- ``'memmap'`` with ``use_gpu=True`` or a device
        array raises, rather than silently falling back to RAM.

    scratch_dir : str or None, default None
        Directory for the ``accumulator_store='memmap'`` backing files.
        ``None`` creates and removes a private temporary directory; a
        directory you supply is left in place (only the files this call made
        in it are removed).  Put it on a fast local disk: it carries one
        sequential pass per accumulator per surface.  Ignored when
        ``accumulator_store='ram'``.

    stream_transfer_function : bool, default False
        Opt-in ASM memory trim (v5.40), forwarded to
        :func:`~lumenairy.propagators.asm.angular_spectrum_propagate` for the
        in-glass propagation.  Generates the transfer function one row band at
        a time during the frequency-domain multiply, in place on the spectrum,
        instead of materialising the full ``H`` grid plus a second full grid
        for the product -- two complex full-grid arrays, 17.2 GB at
        ``N = 32768`` / complex64.  Byte-identical; the cost is that the
        streamed ``H`` is never cached, which is free above the H cache's
        2 GB per-entry cap (``N >= 16384`` at complex64, where H is not
        cacheable anyway) and a real repeat cost below it.  ``wave_propagator=
        'asm'`` and the NumPy backend only; inert elsewhere.

    Returns
    -------
    E_out : ndarray (complex, N x N)

    Notes
    -----
    With ``slant_correction=False`` and all other optional features off,
    the function reduces to the original paraxial-OPD, lossless,
    perfectly-aligned, single-aperture model and is bit-for-bit backward
    compatible with prescriptions that omit the new keys.

    GPU usage (3.1.10+)
    -------------------
    Pass ``use_gpu=True`` or a CuPy array as ``E_in`` to run the whole
    phase-screen + in-glass ASM pipeline on GPU.  Default is ``False``
    to preserve the existing CPU path bit-for-bit.  When enabled:

    * ``E_in`` is promoted to the device via ``cp.asarray`` (or kept
      as-is if already a CuPy array).
    * All meshgrids, sag arrays, and per-surface phase screens are
      built natively on the device using the CuPy namespace.
    * Internal ``angular_spectrum_propagate`` calls auto-detect the
      CuPy input and use the library's existing cuFFT-backed ASM.
    * The numexpr fused-phase-screen path is skipped on GPU (numexpr
      is CPU-only); CuPy's native elementwise kernels are used
      instead.
    * The return value is a CuPy array when ``use_gpu=True``.  Use
      ``cp.asnumpy(E_out)`` to pull it back to the host when needed.

    The returned array type follows ``use_gpu``: host -> host, device
    -> device.  Mixed-dtype callers (e.g. a complex64 host array
    promoted to the device) remain in their starting precision.

    Configuration objects
    ---------------------
    geometry, numerics, resources, config : optional
        :class:`~lumenairy.LensGeometry` / :class:`~lumenairy.LensNumerics` /
        :class:`~lumenairy.LensResources`, or the
        :class:`~lumenairy.LensConfig` that holds all three, as an alternative
        to spelling the settings out as keywords.  Purely ADDITIVE: every
        keyword above still works with the same default, and a call that
        passes none of the four runs exactly the code it ran before.  A set
        field and a keyword for the SAME setting must agree or the call
        raises; a set field this function has no parameter for also raises
        (``config.narrowed_to('apply_real_lens')`` drops those deliberately).
        See ``docs/lens_configuration.md``.

    All arguments past ``E_in`` are keyword-only (4.7+).  The
    parameter name is ``prescription`` -- the 4.6 alias
    ``lens_prescription`` was removed in 4.7.
    """
    # The defensive input guard runs FIRST --
    # before the accumulator-store context -- so the user gets a clear
    # error rather than a downstream failure, and so the v4.15.3
    # dispatcher pin (every entry point calls the guard as its first
    # executable statement) holds.  The v5.40 _AccumulatorStore wrapper
    # must not displace it.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens', input_kind='field')
    # Config objects, if any, are merged into the keywords and the call is
    # re-entered with them -- so the configured path is the SAME code as the
    # equivalent keyword call, by construction rather than by review.  Four
    # ``is not None`` tests when nothing is configured; nothing else changes.
    if _wants_config(geometry, numerics, resources, config):
        return apply_real_lens(E_in, **_resolve_lens_config(
            apply_real_lens, locals(), geometry=geometry, numerics=numerics,
            resources=resources, config=config))
    with _AccumulatorStore(accumulator_store, scratch_dir) as _store:
        return _apply_real_lens_impl(
            E_in,
            prescription=prescription,
            wavelength=wavelength,
            dx=dx,
            dy=dy,
            bandlimit=bandlimit,
            fresnel=fresnel,
            slant_correction=slant_correction,
            absorption=absorption,
            seidel_correction=seidel_correction,
            seidel_poly_order=seidel_poly_order,
            progress=progress,
            use_gpu=use_gpu,
            wave_propagator=wave_propagator,
            surface_frame=surface_frame,
            sag_dtype=sag_dtype,
            sag_chunk_rows=sag_chunk_rows,
            surface_model=surface_model,
            conjugate=conjugate,
            displaced_mode=displaced_mode,
            displaced_obliquity=displaced_obliquity,
            displaced_n_side=displaced_n_side,
            remap_order=remap_order,
            carrier=carrier,
            screen_obliquity=screen_obliquity,
            on_screen_obliquity=on_screen_obliquity,
            accumulator_store=accumulator_store,
            scratch_dir=scratch_dir,
            stream_transfer_function=stream_transfer_function,
            _accum_store=_store,
        )


def _apply_real_lens_impl(
    E_in: np.ndarray,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    bandlimit: bool = True,
    fresnel: bool = False,
    slant_correction: bool = False,
    absorption: bool = False,
    seidel_correction: bool = False,
    seidel_poly_order: int = 6,
    progress: Optional[Any] = None,
    use_gpu: bool = False,
    wave_propagator: Optional[str] = None,
    surface_frame: bool = False,
    sag_dtype: Optional[Any] = None,
    sag_chunk_rows: Optional[int] = None,
    surface_model: str = 'thin',
    conjugate: Any = None,
    displaced_mode: str = 'screen',
    displaced_obliquity: str = 'auto',
    displaced_n_side: Optional[int] = None,
    remap_order: int = 3,
    carrier: Any = None,
    screen_obliquity: Any = 'auto',
    on_screen_obliquity: str = 'warn',
    accumulator_store: str = 'ram',
    scratch_dir: Optional[str] = None,
    stream_transfer_function: bool = False,
    _accum_store: Optional['_AccumulatorStore'] = None,
) -> np.ndarray:
    """The body of :func:`apply_real_lens`.

    Split out (v5.40) so the public entry point can own the
    accumulator-store context: the store's scratch files must be released
    on EVERY exit, including an exception raised mid-prescription, and a
    ``with`` block around the call is the only place that can guarantee it
    without wrapping two thousand lines in a ``try``.  Every argument is
    forwarded verbatim; ``_accum_store`` is private and always supplied.
    """
    # The v4.15.3 input guard lives in ``apply_real_lens`` (the sole
    # caller), as its first executable statement, per the dispatcher
    # pin; this impl deliberately carries no second copy so the W4
    # input-kind census stays at one wired site per entry point.

    # When ``wave_propagator``
    # is left at the default ``None``, resolve via the library-wide
    # default set by ``set_default_wave_propagator(...)``.  Explicit
    # values bypass the resolver.
    if wave_propagator is None:
        from ..propagators.propagation import get_default_wave_propagator
        wave_propagator = get_default_wave_propagator()
    # Same for ``dy``.
    # ``None -> get_default_dy() -> dx`` chain.
    if dy is None:
        from ..propagators.propagation import get_default_dy
        dy = get_default_dy()
        if dy is None:
            dy = dx

    # Mirror-in-``surfaces``: refuse, or -- when the caller has set
    # ``allow_unfolded_equivalent``, the SAME key the ``elements``-borne fold
    # guard below honours -- substitute the unfolded-equivalent prescription
    # (each mirror an index-neutral flat at its own vertex plane) and warn.
    # Done BEFORE the model guards so they see the surfaces that will actually
    # be walked; a mirror-free prescription is returned unchanged, so the
    # default path is untouched.
    prescription = _unfold_mirror_surfaces(prescription, 'apply_real_lens')

    _check_apply_real_lens_kwarg_combination(
        wave_propagator=wave_propagator,
        slant_correction=slant_correction,
        seidel_correction=seidel_correction,
        seidel_poly_order=seidel_poly_order,
        prescription=prescription,
    )
    _check_displaced_support(
        surface_model=surface_model,
        slant_correction=slant_correction,
        fresnel=fresnel,
        seidel_correction=seidel_correction,
        absorption=absorption,
        surface_frame=surface_frame,
        use_gpu=use_gpu,
        wave_propagator=wave_propagator,
        prescription=prescription,
        conjugate=conjugate,
        E_shape=np.shape(E_in),
        displaced_mode=displaced_mode,
        displaced_obliquity=displaced_obliquity,
        displaced_n_side=displaced_n_side,
        remap_order=remap_order,
    )
    # The launch lattice the 2-D remap will trace: the validated per-call
    # override, else the module default.  Resolved once, so the smoothing
    # warning and the trace cannot disagree about it.
    _disp_n_side = _normalise_displaced_n_side(displaced_n_side)
    if _disp_n_side is None:
        _disp_n_side = _DISP_REMAP_2D_N_SIDE
    # The screen-obliquity correction + its accuracy guard.  Reached ONLY
    # through the ``carrier=`` keyword, so a call that does not pass one is
    # structurally bit-unchanged (BUILD_SCREEN_OBLIQUITY_2026_08_11 S6).
    _obl_apply = _check_screen_obliquity_support(
        carrier=carrier,
        screen_obliquity=screen_obliquity,
        on_screen_obliquity=on_screen_obliquity,
        surface_model=surface_model,
        displaced_mode=displaced_mode,
    )
    # v5.37: the tangent-facet family SUPERSEDES equations (4) and (7) --
    # ``_check_screen_obliquity_support`` already returns False for it, and the
    # guard block at the end of the call is already gated on ``not _tf_active``
    # -- so under those models the whole obliquity block was computing a
    # correction nobody added and accumulating a momentum field nobody read.
    # Gating it off here is a DEAD-CODE removal, not a behaviour change: with
    # ``_obl_apply`` False and ``_obl_total`` None the block's only remaining
    # effect was on ``_obl_p0*``, which no surviving reader touches.  It is
    # worth up to four full grids (the carrier field for a finite-radius
    # carrier, plus the accumulator pair), and the byte-identity of the change
    # is pinned across the adversarial matrix.
    _obl_active = (carrier is not None
                   and surface_model not in _TANGENT_FACET_MODELS)

    # v4.13.0 audit P1-A: the mirror-in-``surfaces`` guard.  The shared
    # ``_check_no_silent_fold_drop`` only inspects the prescription's
    # ``elements`` list (what ``load_zemax_zmx`` populates); a hand-built
    # prescription that puts a mirror directly into ``surfaces`` (via
    # ``is_mirror=True`` or ``glass_after='MIRROR'``) and omits the
    # ``elements`` key slips past it, and the refractive walk would treat the
    # mirror as a refractor with the wrong sign.  Both guards now read the
    # SAME ``allow_unfolded_equivalent`` key and are applied together at the
    # top of this function (``_unfold_mirror_surfaces``), which is why nothing
    # is left to do here -- the surfaces below are already either mirror-free
    # or the acknowledged unfolded equivalent.

    # Pre-flight grid vs prescription-aperture check.  If any surface's
    # semi-aperture exceeds the simulation grid, ASM will silently
    # truncate the field at the grid edge and lose energy that the real
    # hardware would have transmitted.  Issue a UserWarning once per
    # call site (Python's default warning filter dedups by source line).
    try:
        # Anamorphic-safe: pass BOTH axes.  ``shape[0]`` is Ny, so pairing it
        # with ``dx`` describes a semi-extent that exists on neither axis of a
        # non-square or ``dy != dx`` grid -- both of which this function
        # supports throughout.  ``dy`` is resolved a few lines below for the
        # main body; resolve it here the same way.
        _shape = np.shape(E_in)
        _dy_chk = dx if dy is None else dy
        _warn_if_aperture_exceeds_grid(
            prescription, int(_shape[1]), dx, source='apply_real_lens',
            stacklevel=_WARN_STACKLEVEL + 1,
            N_y=int(_shape[0]), dy=_dy_chk)
    except (KeyError, ValueError, TypeError, AttributeError, IndexError):
        # Aperture-check failure is informational only.
        pass

    # Select the array namespace: numpy by default; cupy if the caller
    # opted in via ``use_gpu=True`` OR passed in a cupy array.
    if use_gpu or _is_cupy_array(E_in):
        if not CUPY_AVAILABLE:
            raise ImportError(
                "use_gpu=True (or CuPy input) requires the 'cupy' package.  "
                "Install cupy-cuda12x (NVIDIA, matching your CUDA version) "
                "or cupy-rocm-6-1 (AMD ROCm); or call with use_gpu=False to "
                "stay on the CPU path.")
        # Trigger the lazy import; _is_cupy_array(E_in) above only
        # ensured cp was loaded if E_in was already CuPy, but the
        # use_gpu=True + numpy-input path does not pass through that
        # branch.  Loading explicitly here makes ``xp = cp`` safe.
        if cp is None:
            _ensure_cupy_loaded()
        xp = cp
    else:
        xp = np

    # The PERSISTENT full-grid accumulators are allocated through this store
    # (v5.40).  ``accumulator_store='ram'`` -- the default -- makes it a pure
    # pass-through to ``xp.empty`` / ``xp.zeros``, so nothing about the default
    # path moves; ``'memmap'`` spills them to ``scratch_dir``.  The public
    # entry point owns the ``with``; this is only where the backend is pinned.
    _accum = (_accum_store if _accum_store is not None
              else _AccumulatorStore(accumulator_store, scratch_dir))
    _accum.bind(xp)

    if dy is None:
        dy = dx

    surfaces = prescription['surfaces']
    thicknesses = prescription['thicknesses']
    aperture = prescription.get('aperture_diameter')
    stop_index = prescription.get('stop_index')

    # Input validation, not an invariant: an ``assert`` here is stripped under
    # ``python -O``, where a short ``thicknesses`` then surfaces as a bare
    # IndexError from inside the loop and an over-long one is accepted
    # silently.  ``prepare_real_lens`` already raises properly for the same
    # condition; CONVENTIONS SS2 requires the ``f"{fn_name}: ..."`` form.
    if len(thicknesses) != len(surfaces) - 1:
        raise ValueError(
            f"apply_real_lens: prescription needs "
            f"{len(surfaces) - 1} thickness(es) for {len(surfaces)} "
            f"surface(s) (the gap AFTER every surface but the last); got "
            f"{len(thicknesses)}.")

    stop_index = _normalise_stop_index(stop_index, len(surfaces),
                                       fn_name='apply_real_lens')

    Ny, Nx = E_in.shape
    k0 = 2 * np.pi / wavelength

    # surface_model='displaced': precompute the per-surface ray-angle cosine
    # LUTs from a collimated meridional fan (see the module-level derivation).
    # Bounded by the clear aperture (or the widest per-surface semi-diameter,
    # or the grid half-width), so the fan spans the illuminated pupil.
    _displaced = (surface_model == 'displaced')
    # v5.36.0 route 3: the per-pixel tangent-facet screen.  v5.37 ROW-BANDS it:
    # the model differentiates a gradient twice (grad grad sag, and grad p_out
    # where p_out itself carries grad sag), so the band needs a 3-row halo on
    # the sag AND a 2-row one on the persistent accumulator -- derived above
    # ``_tf_sl``, byte-identical, and pinned in test_tf_banded_halo.py.
    _tf_active = (surface_model in _TANGENT_FACET_MODELS)
    # v5.37 the REMAP rung: the walk is REPRESENTED (screen + coordinate remap)
    # rather than referenced away.  Shares every gate above with route 3 -- the
    # accumulator, the gap transport, the guard silence -- and differs only
    # inside the two blocks below.  Its SCREEN half bands on a NARROWER halo
    # than route 3's (2 rows of sag, none of the accumulator, because (R3)
    # gives the kick in closed form); its pull-back does not band at all.  See
    # the derivations above ``_tf_sl`` and ``_tangent_facet_remap_screen``.
    _tf_remap = (surface_model == 'tangent_facet_remap')
    # P10 (N11): decentered / tilted / freeform (asymmetric) elements route to
    # the 2-D transverse-walk remap -- the DEFAULT (auto obliquity) and also
    # selectable via displaced_mode='remap'; an explicit
    # displaced_obliquity='pointwise' keeps the P3 single-plane obliquity SCREEN
    # (the documented walk-off-limited peer).
    _disp_asym = _displaced and _element_is_asymmetric(surfaces)
    _disp_2d_remap = _routes_to_displaced_remap_2d(
        surface_model, displaced_mode, displaced_obliquity, surfaces)
    _remap_mode = (_displaced and displaced_mode == 'remap'
                   and not _disp_asym)             # 1-D symmetric remap (P2)
    _split_mode = _displaced and displaced_mode == 'split'
    _disp_luts = None
    _disp_ray_map = None
    _disp_ray_map_2d = None
    _disp_cos_grid = None
    _disp_pointwise = False
    # The congruence callable the remap fan was launched along, kept so the
    # apply step can demodulate ``E_in`` by the SAME wavefront (L3 -- the input
    # phase beyond the congruence is transported, not discarded).
    _disp_eik_fn = None
    if _displaced:
        # The pointwise obliquity SCREEN fires only for an EXPLICIT
        # displaced_obliquity='pointwise' (or the symmetric convention gate); the
        # asymmetric DEFAULT ('auto') routes to the 2-D remap below instead.
        _disp_pointwise = (
            (_resolve_displaced_obliquity(displaced_obliquity, surfaces)
             == 'pointwise') and not _disp_2d_remap)
        _r_max = None
        if aperture is not None:
            _r_max = float(aperture) / 2.0
        else:
            _semis = [s.get('semi_diameter') for s in surfaces
                      if s.get('semi_diameter')]
            _r_max = (max(float(v) for v in _semis) if _semis
                      else 0.5 * max(Nx * dx, Ny * dy))
        # G2 Task 1: launch the meridional fan along the INPUT CONGRUENCE
        # (conjugate=None collimated -> byte-identical; scalar R_in; 'auto';
        # explicit wavefront ndarray) so the per-surface obliquity cosines
        # reflect the true converging/diverging incidence.
        _disp_slope = _displaced_carrier_slope_fn(
            conjugate, E_in, wavelength, dx, dy, Nx, Ny)
        if _disp_2d_remap:
            # P10 (N11): 2-D transverse-walk remap for the asymmetric element.
            # Launch a full 2-D congruence fan against the decentered / tilted /
            # freeform surfaces and build the entrance->exit ray map + OPL; the
            # energy-conserving 2-D-Jacobian amplitude warp (applied after the
            # entrance aperture) restores the coma-broadening walk-off the
            # single-plane screen drops.
            _dir2, _eik2 = _displaced_carrier_dir_eik_fn(
                conjugate, E_in, wavelength, dx, dy, Nx, Ny)
            _disp_eik_fn = _eik2
            _warn_if_remap_lattice_smooths(_r_max, dx, dy, _disp_n_side)
            _disp_ray_map_2d = _build_displaced_ray_map_2d(
                surfaces, thicknesses, wavelength, _r_max,
                n_side=_disp_n_side,
                dir_fn=_dir2, eik_fn=_eik2)
        elif _disp_pointwise:
            # P3 (N2): 2-D pointwise obliquity SCREEN for decenter / tilt /
            # freeform (explicit displaced_obliquity='pointwise', or the
            # symmetric convention gate).  Trace a 2-D ray grid along the input
            # congruence and imprint the per-surface z-axis cosines on the field
            # grid.
            _disp_dir = _displaced_carrier_dir_fn(
                conjugate, E_in, wavelength, dx, dy, Nx, Ny)
            # B1: fetch from (or populate) the opt-in byte-budgeted cos-grid
            # cache (default off -> a plain rebuild); B5: the trace uses the
            # structured-grid interpolation by default.
            _disp_cos_grid = _get_displaced_cos_grid(
                surfaces, thicknesses, wavelength, _r_max, Nx, Ny, dx, dy,
                dir_fn=_disp_dir, conjugate=conjugate)
        elif _remap_mode:
            # P2 candidate (a): trace the full entrance->exit ray map + OPL
            # (with the entrance eikonal) for the geometric-transfer remap
            # applied just below (after the entrance aperture).
            _disp_eik = _displaced_eikonal_fn(
                conjugate, E_in, wavelength, dx, dy, Nx, Ny)
            _disp_eik_fn = _disp_eik
            _disp_ray_map = _build_displaced_ray_map(
                surfaces, thicknesses, wavelength, _r_max,
                carrier_slope=_disp_slope, eikonal_fn=_disp_eik)
        else:
            # 'screen' (default) + 'split' share the per-surface cosine LUTs.
            _disp_luts = _get_displaced_cos_luts(
                surfaces, thicknesses, wavelength, _r_max, conjugate,
                _disp_slope)

    # Geometry dtype: float64 (default, byte-identical) or float32 (opt-in via
    # sag_dtype= / set_lens_sag_dtype), which halves the coordinate/sag/opd
    # float64 core.  The .astype pins the dtype across NumPy casting-rule
    # versions; for float64 it is a no-op, so the default path is unchanged.
    _sag_real = _resolve_sag_real(sag_dtype)
    x = ((xp.arange(Nx, dtype=_sag_real) - Nx / 2) * dx).astype(_sag_real, copy=False)
    y = ((xp.arange(Ny, dtype=_sag_real) - Ny / 2) * dy).astype(_sag_real, copy=False)
    # Row-band (chunked) mode defers the full X/Y/h_sq_axis meshgrids: the
    # banded phase screens compute ``x2[j] + y2[i]`` per band (element-
    # identical to a slice of ``X**2 + Y**2``), so the three full-grid
    # float arrays (~26 GB at N=32768) never allocate unless a surface
    # falls through to the whole-grid path (decenter/tilt/slant/stop/...)
    # or the Seidel block needs them -- ``_ensure_full_grids`` builds them
    # on first such use.
    # v5.17.0: sag_chunk_rows=None resolves to AUTO (banded when N >= 4096;
    # byte-identical + wall-clock neutral, far leaner).  Pass 0 to force the
    # whole-grid path.
    sag_chunk_rows = _resolve_sag_chunk_rows(sag_chunk_rows, Ny)
    _chunk_grids = (sag_chunk_rows is not None and int(sag_chunk_rows) > 0
                    and xp is np)
    if _chunk_grids:
        X = Y = h_sq_axis = None
        _x_sq = x ** 2
        _y_sq = y ** 2
    else:
        X, Y = xp.meshgrid(x, y)
        h_sq_axis = X ** 2 + Y ** 2  # axis-centered distance, used for stop aperture

    def _ensure_full_grids():
        nonlocal X, Y, h_sq_axis
        if X is None:
            X, Y = xp.meshgrid(x, y)
            h_sq_axis = X ** 2 + Y ** 2
        return X, Y, h_sq_axis

    # ---- screen obliquity: the carrier's momentum field + the accumulators.
    # ``_obl_q*`` are the carrier's direction cosines (floats for a collimated
    # tilt -- the common case -- so no full-grid momentum arrays are needed);
    # ``_obl_p0*`` is the carrier-free momentum the screen model itself
    # accumulates; ``_obl_total`` is the summed correction the guard scores.
    # ``_obl_u*`` is the carrier-induced ray DRIFT (equation 6) the R1 term
    # reads; it stays a plain float 0.0 until a gap actually moves the ray, so
    # a leading plate costs nothing and a zero-angle carrier never allocates.
    _obl_qx = _obl_qy = 0.0
    _obl_p0x = _obl_p0y = 0.0
    _obl_ux = _obl_uy = 0.0
    _obl_drift_live = False
    _obl_q_zero = True
    _obl_total = None
    _obl_n_first = 1.0
    # v5.35.3: the row-band evaluator for the carrier momentum field.  Not
    # None => ``_obl_qx`` / ``_obl_qy`` are NOT materialised; bands come from
    # the closed form and ``_obl_q_whole()`` builds the full field only if a
    # surface actually falls through to the whole-grid path.
    _obl_q_rows_fn = None
    if _obl_active:
        # The carrier's direction cosines become a transverse OPTICAL
        # momentum in the FIRST medium -- the units _facet_axial_momenta and
        # the ``_obl_p0*`` accumulator both work in.  Identity for a
        # prescription starting in air; a factor n1 for an immersed one.
        _obl_n_first = float(get_glass_index(surfaces[0]['glass_before'],
                                             wavelength)) if surfaces else 1.0
        if _chunk_grids:
            _obl_q_rows_fn = _screen_obliquity_row_evaluator(
                carrier, dx, dy, Nx, Ny, n_medium=_obl_n_first)
        if _obl_q_rows_fn is None:
            _obl_qx, _obl_qy = _screen_obliquity_angle_field(
                carrier, E_in, wavelength, dx, dy, Nx, Ny,
                n_medium=_obl_n_first)
            if xp is not np:
                _obl_qx = xp.asarray(_obl_qx)
                _obl_qy = xp.asarray(_obl_qy)
            # A non-collimated congruence returns a full GRID here, and it is
            # read on every surface, so it is persistent state: adopt it into
            # the store (a collimated tilt returns two floats and passes
            # straight through).
            _obl_qx = _accum.adopt(_obl_qx)
            _obl_qy = _accum.adopt(_obl_qy)
            # A zero-angle carrier has no drift to accumulate, so R1 is skipped
            # STRUCTURALLY rather than by cancellation -- the byte-null of
            # ``test_zero_angle_carrier_is_byte_identical`` is not a tolerance.
            _obl_q_zero = (bool(xp.all(_obl_qx == 0.0))
                           and bool(xp.all(_obl_qy == 0.0)))
        else:
            # Same reduction, band-wise, short-circuiting on the first
            # non-zero band -- a real tilt exits after one band, and only a
            # genuinely zero field pays the full scan.
            _obl_q_zero = True
            _cr_q = int(sag_chunk_rows)
            _qb_x = _qb_y = None
            for _qr0 in range(0, Ny, _cr_q):
                _qb_x, _qb_y = _obl_q_rows_fn(_qr0, min(Ny, _qr0 + _cr_q))
                if not (bool(xp.all(_qb_x == 0.0))
                        and bool(xp.all(_qb_y == 0.0))):
                    _obl_q_zero = False
                    break
            del _qb_x, _qb_y
        if on_screen_obliquity != 'silent' and not _tf_active:
            # only the guard reads the accumulated correction field
            _obl_total = _accum.zeros((Ny, Nx), _sag_real)

    # ---- route 3: the tangent-facet model's momentum accumulator ----------
    # ``(_tf_px, _tf_py)`` is the FIELD's own transverse optical momentum: the
    # carrier's, minus the gradient of everything imprinted so far, resampled
    # across each gap.  It stays a pair of plain floats until the first POWERED
    # surface makes it a field, so a leading plate and every carrier-free call
    # allocate nothing here.
    _tf_px = _tf_py = 0.0
    if _tf_active and carrier is not None:
        _tf_n_first = float(get_glass_index(surfaces[0]['glass_before'],
                                            wavelength)) if surfaces else 1.0
        # v5.40: fill the seed BAND-WISE when the route is banded.  The
        # whole-grid helper holds 7 float64 grids to deliver 2 (see
        # ``_screen_obliquity_rows_any``), and on the tangent-facet route with
        # a non-collimated carrier that set-up -- not the accumulators, not the
        # screen -- is what sets the call's peak.  Writing the seed straight
        # into the accumulator store also means the FIRST surface already reads
        # from the store rather than only the destination it writes.
        _tf_seed_rows = (
            _screen_obliquity_rows_any(
                carrier, E_in, wavelength, dx, dy, Nx, Ny,
                n_medium=_tf_n_first)
            if _chunk_grids else None)
        if _tf_seed_rows is not None:
            _cr_s = int(sag_chunk_rows)
            _sx_g = _sy_g = None
            _seed_const = True
            _seed_0 = None
            for _sr0 in range(0, Ny, _cr_s):
                _sr1 = min(Ny, _sr0 + _cr_s)
                _sbx, _sby = _tf_seed_rows(_sr0, _sr1)
                if _sx_g is None:
                    _sx_g = _accum.empty((Ny, Nx), _sbx.dtype)
                    _sy_g = _accum.empty((Ny, Nx), _sby.dtype)
                    _seed_0 = (_sbx.flat[0], _sby.flat[0])
                if _seed_const:
                    # The whole-grid helper collapses a CONSTANT momentum field
                    # to two Python floats, and that is observable: a float
                    # seed and a float64 array of the same value promote
                    # differently under NEP 50 (BUILD_TF_BANDED S2.1).  Same
                    # decision, taken band-wise -- ``ptp == 0`` on both
                    # components is exactly "every element equals element 0".
                    _seed_const = (bool(np.all(_sbx == _seed_0[0]))
                                   and bool(np.all(_sby == _seed_0[1])))
                _sx_g[_sr0:_sr1] = _sbx
                _sy_g[_sr0:_sr1] = _sby
                del _sbx, _sby
            if _seed_const:
                _tf_px, _tf_py = float(_seed_0[0]), float(_seed_0[1])
            else:
                _tf_px, _tf_py = _sx_g, _sy_g
            _sx_g = _sy_g = None
        else:
            _tf_px, _tf_py = _screen_obliquity_angle_field(
                carrier, E_in, wavelength, dx, dy, Nx, Ny,
                n_medium=_tf_n_first)
            if xp is not np:
                _tf_px = xp.asarray(_tf_px)
                _tf_py = xp.asarray(_tf_py)
            # Adopt the seed so the first surface already reads from the store
            # (a collimated carrier collapses to two floats and adopt() passes
            # it through untouched).
            _tf_px = _accum.adopt(_tf_px)
            _tf_py = _accum.adopt(_tf_py)

    # ---- screen obliquity: the row-band (halo) machinery (v5.35.3) --------
    # Everything below is a band-wise restatement of the whole-grid obliquity
    # block; the arithmetic per element is the SAME expression, so the banded
    # result is byte-identical (tests/unit/test_obl_banded_halo.py).
    #
    # Two pieces of per-surface bookkeeping keep that claim true:
    #
    # * ``_obl_p0_src`` -- ``p0`` AS OF THE START of the surface.  Every band
    #   must read the same source the whole-grid block reads, and that is not
    #   automatic: while ``p0`` is still the scalar seed ``0.0``, promoting it
    #   to a full grid at band 0 would make bands 1..n read a float32 ARRAY of
    #   zeros where band 0 (and the whole grid) read a PYTHON float.  Under
    #   NEP 50 that changes the momentum-triangle arithmetic in
    #   ``_facet_axial_momenta`` from float64 to float32 -- measured 5e-6 of
    #   field error at ``sag_dtype='float32'``, invisible at the float64
    #   default.  So a scalar source is written into a FRESH destination.
    # * ``_obl_p0_pending`` -- when the source IS a grid the write is in place,
    #   but held back ONE band: the next band's R1 halo still has to read row
    #   ``r0-1`` at its pre-surface value.
    _obl_p0_src = None
    _obl_p0_dst = None
    _obl_p0_pending = None

    def _obl_band_of(v, r0, r1):
        """``v`` restricted to rows ``[r0:r1)`` -- scalars pass through."""
        return v[r0:r1] if getattr(v, 'ndim', 0) else v

    def _obl_q_bands(r0, r1):
        """``(qx, qy)`` for rows ``[r0:r1)``."""
        if _obl_q_rows_fn is not None:
            return _obl_q_rows_fn(r0, r1)
        return (_obl_band_of(_obl_qx, r0, r1), _obl_band_of(_obl_qy, r0, r1))

    def _obl_q_whole():
        """``(qx, qy)`` on the whole grid, materialising the row-evaluated
        field the first time a surface falls through to the whole-grid path."""
        nonlocal _obl_qx, _obl_qy, _obl_q_rows_fn
        if _obl_q_rows_fn is not None:
            _obl_qx, _obl_qy = _screen_obliquity_angle_field(
                carrier, E_in, wavelength, dx, dy, Nx, Ny,
                n_medium=_obl_n_first)
            _obl_qx = _accum.adopt(_obl_qx)
            _obl_qy = _accum.adopt(_obl_qy)
            _obl_q_rows_fn = None
        return _obl_qx, _obl_qy

    def _obl_accum_band(acc, band, r0, r1):
        """``acc = acc + band`` over rows ``[r0:r1)``, promoting the scalar
        seed to full-grid STORAGE exactly once (the accumulators are genuinely
        full-grid state -- only the transients are banded).  The promotion
        fills with the OLD scalar, which is what the rows this loop has not
        reached yet still hold."""
        if getattr(acc, 'ndim', 0):
            new = acc[r0:r1] + band
            if new.dtype != acc.dtype:
                acc = acc.astype(new.dtype)
            acc[r0:r1] = new
            return acc
        if getattr(band, 'ndim', 0) == 0:
            return acc + band                  # all-scalar: unchanged
        new = acc + band
        acc = _accum.full((Ny, Nx), acc, new.dtype)
        acc[r0:r1] = new
        return acc

    def _obl_flush_p0():
        """Commit the deferred in-place ``_obl_p0*`` band write."""
        nonlocal _obl_p0_pending
        if _obl_p0_pending is not None:
            _pr0, _pr1, _px, _py = _obl_p0_pending
            _obl_p0x[_pr0:_pr1] = _px
            _obl_p0y[_pr0:_pr1] = _py
            _obl_p0_pending = None

    def _obl_begin_surface():
        """Pin the momentum source every band of this surface reads."""
        nonlocal _obl_p0_src, _obl_p0_dst, _obl_p0_pending
        _obl_p0_src = (_obl_p0x, _obl_p0y)
        _obl_p0_dst = None
        _obl_p0_pending = None

    def _obl_end_surface():
        """Commit the surface's momentum accumulation."""
        nonlocal _obl_p0x, _obl_p0y, _obl_p0_src, _obl_p0_dst
        _obl_flush_p0()
        if _obl_p0_dst is not None:
            _obl_p0x, _obl_p0y = _obl_p0_dst
            _obl_p0_dst = None
        _obl_p0_src = None

    def _obl_halo_rows():
        """Sag-halo width the banded obliquity block needs: 1 row for the sag
        gradient itself, 2 when the R1 drift term is live (it takes a SECOND
        gradient, of ``e_err``, which is built from the first)."""
        return 2 if (_obl_apply and _obl_drift_live) else 1

    def _band_any_sag(R, kc, asph, cr):
        """``bool(xp.any(sag))`` for a plain conic+aspheric surface, band-wise.

        The whole-grid block is skipped entirely for a FLAT face (a plate, a
        cemented plano, a stop), and that skip is observable -- it is what
        keeps ``_obl_p0*`` a pair of floats through a leading plate.  Reproduce
        the reduction without materialising the grid; a powered surface
        short-circuits on the first band."""
        for _r0 in range(0, Ny, cr):
            _r1 = min(Ny, _r0 + cr)
            _s = _surface_sag_general(
                _x_sq[None, :] + _y_sq[_r0:_r1, None], R, kc, asph)
            if bool(xp.any(_s)):
                return True
        return False

    def _obl_band_delta(sag_h, _h0, _h1, r0, r1, n1r, n2r, grad_h=None):
        """The whole-grid obliquity block for rows ``[r0:r1)``.

        ``sag_h`` is the surface sag on the halo rows ``[_h0:_h1)``;
        ``grad_h`` is ``xp.gradient(sag_h, dy, dx)`` when the caller already
        took it on the SAME array (the slant path does) -- it is recomputed on
        the NaN-zeroed copy when the conic domain edge put NaNs in the band,
        exactly as the whole-grid block does.

        Updates ``_obl_total`` and stages the ``_obl_p0*`` accumulation;
        returns the correction to add to this band's ``opd``, or None when the
        correction is estimator-only (``screen_obliquity=False``)."""
        nonlocal _obl_p0x, _obl_p0y, _obl_p0_src, _obl_p0_dst, _obl_p0_pending
        _src_x, _src_y = _obl_p0_src
        _lo = r0 - _h0
        _hi = _lo + (r1 - r0)
        _ok_h = sag_h
        if bool(xp.any(xp.isnan(sag_h))):
            _ok_h = xp.where(xp.isnan(sag_h), 0.0, sag_h)
            grad_h = None
        if grad_h is None:
            _gy_h, _gx_h = xp.gradient(_ok_h, dy, dx)
        else:
            _gy_h, _gx_h = grad_h
        _gx_b = _gx_h[_lo:_hi]
        _gy_b = _gy_h[_lo:_hi]
        _qx_b, _qy_b = _obl_q_bands(r0, r1)
        _d = _screen_obliquity_delta(
            _ok_h[_lo:_hi], _gx_b, _gy_b,
            _obl_band_of(_src_x, r0, r1), _obl_band_of(_src_y, r0, r1),
            _qx_b, _qy_b, n1r, n2r, xp)
        if _obl_total is not None:
            _obl_total[r0:r1] += _d
        _out = None
        if _obl_apply:
            _out = _d
            if _obl_drift_live:
                # R1 (equation 7) on a 1-row halo INSIDE the sag halo: e_err
                # is pointwise but its gradient is not, so the band has to see
                # one row either side.  ``p0`` is read at its PRE-surface value
                # -- guaranteed by the pinned source + deferred write below.
                _e0 = max(0, r0 - 1)
                _e1 = min(Ny, r1 + 1)
                _el = _e0 - _h0
                _eh = _el + (_e1 - _e0)
                _out = _out + _screen_drift_opd_rows(
                    _ok_h[_el:_eh], _gx_h[_el:_eh], _gy_h[_el:_eh],
                    _obl_band_of(_src_x, _e0, _e1),
                    _obl_band_of(_src_y, _e0, _e1),
                    n1r, n2r,
                    _obl_band_of(_obl_ux, r0, r1),
                    _obl_band_of(_obl_uy, r0, r1),
                    dx, dy, xp, r0 - _e0, r0 - _e0 + (r1 - r0))
        # the screen model's OWN carrier-free momentum, accumulated at this
        # field point for the next surface's local ray angle
        _new_x = _obl_band_of(_src_x, r0, r1) - (n2r - n1r) * _gx_b
        _new_y = _obl_band_of(_src_y, r0, r1) - (n2r - n1r) * _gy_b
        if getattr(_src_x, 'ndim', 0) == 0:
            # scalar source -> fresh destination, so every band still reads
            # the scalar (see the _obl_p0_src note above).
            if _obl_p0_dst is None:
                _obl_p0_dst = (_accum.empty((Ny, Nx), _new_x.dtype),
                               _accum.empty((Ny, Nx), _new_y.dtype))
            _obl_p0_dst[0][r0:r1] = _new_x
            _obl_p0_dst[1][r0:r1] = _new_y
        else:
            _obl_flush_p0()
            if _new_x.dtype != _obl_p0x.dtype:
                _obl_p0x = _accum.astype(_obl_p0x, _new_x.dtype)
                _obl_p0y = _accum.astype(_obl_p0y, _new_y.dtype)
                _obl_p0_src = (_obl_p0x, _obl_p0y)
            _obl_p0_pending = (r0, r1, _new_x, _new_y)
        return _out

    def _obl_gap_advance(i_surf, n2r):
        """Advance the carrier's ray drift across the gap behind surface
        ``i_surf`` (equation 6).

    Factored out of the whole-grid surface body so the two row-banded
    paths reach it too: a banded path that ``continue``s past it cannot
    carry ``carrier=`` at all."""
        nonlocal _obl_ux, _obl_uy, _obl_drift_live
        # The gap the CARRIER drifts through is always the physical one here.
        # (There used to be a ``if _split_mode: t/n, n=1`` arm for the
        # 'displaced' split factorisation's reduced distance.  It was
        # unreachable: this closure runs only when ``_obl_active``, which needs
        # ``carrier is not None``, and ``_check_screen_obliquity_support``
        # raises for a carrier with any ``surface_model != 'thin'`` -- while
        # ``_split_mode`` requires ``surface_model == 'displaced'``.)
        _t_gap = float(thicknesses[i_surf])
        _n_gap = n2r
        _band_gap = _chunk_grids and (
            getattr(_obl_p0x, 'ndim', 0) or _obl_q_rows_fn is not None
            or getattr(_obl_qx, 'ndim', 0))
        if _band_gap:
            _cr = int(sag_chunk_rows)
            # Pin the drift source: ``_obl_accum_band`` may promote the scalar
            # seed to a grid at band 0, and the ``_pbx`` read below must keep
            # seeing the SAME object every band (same reason as _obl_p0_src).
            _ux_src, _uy_src = _obl_ux, _obl_uy
            for r0 in range(0, Ny, _cr):
                r1 = min(Ny, r0 + _cr)
                _p0x_b = _obl_band_of(_obl_p0x, r0, r1)
                _p0y_b = _obl_band_of(_obl_p0y, r0, r1)
                _pbx, _pby = _p0x_b, _p0y_b
                if _obl_drift_live and getattr(_obl_p0x, 'ndim', 0):
                    # the carrier-free ray is at ``x - U``, and the element
                    # re-images its own drift -- read p0 there, not here.  The
                    # gradient takes the same 1-row halo (clipped at the true
                    # edges, so rows 0 / Ny-1 keep the one-sided stencil).
                    _e0 = max(0, r0 - 1)
                    _e1 = min(Ny, r1 + 1)
                    _el = r0 - _e0
                    _eh = _el + (r1 - r0)
                    _ux_b = _obl_band_of(_ux_src, r0, r1)
                    _uy_b = _obl_band_of(_uy_src, r0, r1)
                    _gp_y, _gp_x = xp.gradient(_obl_p0x[_e0:_e1], dy, dx)
                    _pbx = _p0x_b - (_ux_b * _gp_x[_el:_eh]
                                     + _uy_b * _gp_y[_el:_eh])
                    _gp_y, _gp_x = xp.gradient(_obl_p0y[_e0:_e1], dy, dx)
                    _pby = _p0y_b - (_ux_b * _gp_x[_el:_eh]
                                     + _uy_b * _gp_y[_el:_eh])
                    del _gp_y, _gp_x
                _qx_b, _qy_b = _obl_q_bands(r0, r1)
                _du_x, _du_y = _screen_drift_step(
                    _p0x_b, _p0y_b, _pbx, _pby, _qx_b, _qy_b,
                    _t_gap, _n_gap, xp)
                _obl_ux = _obl_accum_band(_obl_ux, _du_x, r0, r1)
                _obl_uy = _obl_accum_band(_obl_uy, _du_y, r0, r1)
                del _pbx, _pby, _du_x, _du_y
        else:
            _qx, _qy = _obl_q_whole()
            _pbx, _pby = _obl_p0x, _obl_p0y
            if _obl_drift_live and getattr(_obl_p0x, 'ndim', 0):
                _gp_y, _gp_x = xp.gradient(_obl_p0x, dy, dx)
                _pbx = _obl_p0x - (_obl_ux * _gp_x + _obl_uy * _gp_y)
                _gp_y, _gp_x = xp.gradient(_obl_p0y, dy, dx)
                _pby = _obl_p0y - (_obl_ux * _gp_x + _obl_uy * _gp_y)
                del _gp_y, _gp_x
            _du_x, _du_y = _screen_drift_step(
                _obl_p0x, _obl_p0y, _pbx, _pby, _qx, _qy,
                _t_gap, _n_gap, xp)
            _obl_ux = _obl_ux + _du_x
            _obl_uy = _obl_uy + _du_y
            del _pbx, _pby, _du_x, _du_y
        if not _obl_q_zero and _t_gap != 0.0:
            _obl_drift_live = True

    # ---- the tangent-facet family: the row-band (halo) machinery (v5.37) ---
    # A band-wise restatement of the whole-grid route-3 / remap screen block.
    # The halo widths are DERIVED above ``_tf_sl``; the arithmetic per element
    # is the same expression evaluated on the same operands, which is what
    # makes the banded field byte-identical (tests/unit/test_tf_banded_halo.py).
    #
    # ``_tf_src`` pins the accumulator every band of a surface reads, and every
    # band writes into a FRESH destination -- the same rebinding the whole-grid
    # path does, so a band can never read a row this surface has already
    # rewritten and the scalar seed can never promote mid-loop.
    _tf_src = None
    _tf_dst = None
    #: full-grid walk components, filled band-wise for the remap rung
    _rm_wx_g = _rm_wy_g = None

    def _tf_halo_rows():
        """Sag halo the banded tangent-facet screen needs at this surface."""
        return _TF_REMAP_SAG_HALO_ROWS if _tf_remap else _TF_SAG_HALO_ROWS

    def _tf_begin_surface():
        nonlocal _tf_src, _tf_dst, _rm_wx_g, _rm_wy_g
        _tf_src = (_tf_px, _tf_py)
        _tf_dst = None
        _rm_wx_g = _rm_wy_g = None

    def _tf_store(new_x, new_y, r0, r1):
        """Stage this band's accumulator rows into the surface's FRESH
        destination (allocated on the first band, at the band's own dtype --
        which is the dtype the whole-grid expression produces, because it is
        the same expression on the same operands)."""
        nonlocal _tf_dst
        if _tf_dst is None:
            _tf_dst = (_accum.empty((Ny, Nx), new_x.dtype),
                       _accum.empty((Ny, Nx), new_y.dtype))
        _tf_dst[0][r0:r1] = new_x
        _tf_dst[1][r0:r1] = new_y

    def _tf_end_surface():
        nonlocal _tf_px, _tf_py, _tf_src, _tf_dst
        if _tf_dst is not None:
            _tf_px, _tf_py = _tf_dst
        _tf_src = None
        _tf_dst = None

    def _tf_band_screen(sag_h, _h0, _h1, r0, r1, n1r, n2r, thin_fn):
        """The whole-grid tangent-facet block, for rows ``[r0:r1)``.

        ``sag_h`` is the RAW sag on halo rows ``[_h0:_h1)`` (halo width
        ``_tf_halo_rows()``, clipped at the true grid edges).  ``thin_fn(a0,
        a1)`` returns the model-independent screen OPD on absolute rows
        ``[a0:a1)`` and is called ONLY when a non-propagating pixel forces the
        thin-screen fallback -- the whole-grid path computes that screen
        unconditionally, but it reads it only through the same ``xp.where``, so
        deferring it changes no bit.

        Returns the band's OPD; stages the accumulator (and, for the remap
        rung, the walk) into the surface's destination grids."""
        nonlocal _rm_wx_g, _rm_wy_g
        _src_x, _src_y = _tf_src
        _s_h = sag_h
        if bool(xp.any(xp.isnan(_s_h))):
            _s_h = xp.where(xp.isnan(_s_h), 0.0, _s_h)
        _gy_h, _gx_h, _g0, _g1 = _tf_rows_grad(_s_h, _h0, _h1, Ny, dy, dx, xp)
        # The rows the screen is EVALUATED on.  The remap rung's accumulator is
        # ``p_out`` in closed form (R3), so it needs the band and nothing more;
        # route 3's is minus the gradient of the screen, so it needs one row of
        # margin either side -- clipped at the true grid edges, where the
        # whole-grid gradient is one-sided anyway.  A halo clipped at row 0
        # makes the naturally-valid range WIDER than this, which is why the
        # target is stated rather than inferred: writing rows outside the band
        # would be harmless (they carry the same values) but it would let a
        # band silently depend on its neighbour's arithmetic.
        _t0 = max(0, r0 - (0 if _tf_remap else 1))
        _t1 = min(Ny, r1 + (0 if _tf_remap else 1))
        if _tf_remap:
            # (R4)/(R5): the Hessian is one more gradient level and is taken
            # ONCE, shared between the hit-point fixed point and the facet
            # normal there -- exactly as the whole-grid block does.
            _hxy, _hxx, _q0, _q1 = _tf_rows_grad(
                _gx_h, _g0, _g1, Ny, dy, dx, xp)
            _hyy, _hyx, _, _ = _tf_rows_grad(_gy_h, _g0, _g1, Ny, dy, dx, xp)
            _hs = slice(_t0 - _q0, _t1 - _q0)
            _gs = slice(_t0 - _g0, _t1 - _g0)
            (_opd_e, _wx, _wy, _ox, _oy,
             _ok) = _tangent_facet_remap_screen(
                _s_h[_t0 - _h0:_t1 - _h0], _gx_h[_gs], _gy_h[_gs],
                _hxx[_hs], _hxy[_hs], _hyx[_hs], _hyy[_hs],
                _tf_sl(_src_x, _t0, _t1), _tf_sl(_src_y, _t0, _t1),
                n1r, n2r, xp)
            del _hxx, _hxy, _hyx, _hyy
        else:
            _opd_e, _ok = _tangent_facet_screen_rows(
                _s_h[_g0 - _h0:_g1 - _h0], _gx_h, _gy_h,
                _tf_sl(_src_x, _g0, _g1), _tf_sl(_src_y, _g0, _g1),
                n1r, n2r, dx, dy, xp, _t0 - _g0, _t1 - _g0)
            _wx = _wy = _ox = _oy = None
        del _gy_h, _gx_h, _s_h
        # The whole-grid path takes the ``all(ok)`` reduction over the WHOLE
        # grid and skips the ``where`` when it holds; a band takes it over its
        # own rows.  The two agree element for element because ``xp.where`` on
        # an all-True mask returns the left operand's values exactly, and the
        # result DTYPE is the left operand's too: ``_tf_opd``'s dtype is
        # ``result_type(sag, p)`` and the thin screen's is ``sag``'s alone (the
        # index difference is a weak Python float), so the tangent-facet screen
        # is never the narrower of the two.  Pinned by the float32-geometry arm
        # of the byte-identity matrix.
        _all_ok = (bool(xp.all(_ok))
                   and bool(xp.all(xp.isfinite(_opd_e))))
        if not _all_ok:
            _keep = _ok & xp.isfinite(_opd_e)
            _opd_e = xp.where(_keep, _opd_e, thin_fn(_t0, _t1))
            if _tf_remap:
                _wx = xp.where(_keep, _wx, 0.0)
                _wy = xp.where(_keep, _wy, 0.0)
                _ox = xp.where(_keep, _ox, _tf_sl(_src_x, _t0, _t1))
                _oy = xp.where(_keep, _oy, _tf_sl(_src_y, _t0, _t1))
            del _keep
        del _ok
        # ORDERING, not a guard -- the whole-grid block hoists the NaN-sentinel
        # zeroing above the accumulator gradient for the reason stated there
        # (a sentinel read into a PERSISTENT accumulator travels to every later
        # surface), and the band keeps that order.
        if bool(xp.any(xp.isnan(_opd_e))):
            _opd_e = xp.where(xp.isnan(_opd_e), 0.0, _opd_e)
        if _tf_remap:
            if _rm_wx_g is None:
                _rm_wx_g = _accum.empty((Ny, Nx), _wx.dtype)
                _rm_wy_g = _accum.empty((Ny, Nx), _wy.dtype)
            _rm_wx_g[r0:r1] = _wx
            _rm_wy_g[r0:r1] = _wy
            _tf_store(_ox, _oy, r0, r1)
            return _opd_e
        _ky, _kx, _b0, _b1 = _tf_rows_grad(_opd_e, _t0, _t1, Ny, dy, dx, xp)
        _tf_store(_tf_sl(_src_x, _b0, _b1) - _kx,
                  _tf_sl(_src_y, _b0, _b1) - _ky, _b0, _b1)
        del _ky, _kx
        return _opd_e[r0 - _t0:r0 - _t0 + (r1 - r0)]

    def _tf_gap_transport(i_surf, n_gap):
        """Resample the momentum accumulator across the gap behind surface
        ``i_surf`` -- banded with a 1-row halo when ``sag_chunk_rows`` is live,
        whole-grid otherwise, byte-identical either way."""
        nonlocal _tf_px, _tf_py
        _t = float(thicknesses[i_surf])
        if not _chunk_grids or not getattr(_tf_px, 'ndim', 0):
            _tf_px, _tf_py = _tangent_facet_transport(
                _tf_px, _tf_py, _t, n_gap, dx, dy, xp)
            return
        _cr = int(sag_chunk_rows)
        _sx, _sy = _tf_px, _tf_py
        _nx_g = _ny_g = None
        for r0 in range(0, Ny, _cr):
            r1 = min(Ny, r0 + _cr)
            _a0 = max(0, r0 - _TF_GAP_HALO_ROWS)
            _a1 = min(Ny, r1 + _TF_GAP_HALO_ROWS)
            _nx, _ny = _tangent_facet_transport_rows(
                _sx[_a0:_a1], _sy[_a0:_a1], _t, n_gap, dx, dy, xp,
                r0 - _a0, r0 - _a0 + (r1 - r0))
            if _nx_g is None:
                _nx_g = _accum.empty((Ny, Nx), _nx.dtype)
                _ny_g = _accum.empty((Ny, Nx), _ny.dtype)
            _nx_g[r0:r1] = _nx
            _ny_g[r0:r1] = _ny
            del _nx, _ny
        _tf_px, _tf_py = _nx_g, _ny_g

    def _tf_price_walk_halo(i_surf, cr):
        """PRICE the halo the remap's pull-back would need, and record the
        refusal.

        The screen half of the remap rung bands with a 2-row halo; its second
        half -- resample the FIELD at ``x + W`` -- does not, and this is where
        that is stated in numbers rather than asserted.  The pull-back reads
        the field ``ceil(max|W| / dy)`` rows away plus the spline stencil, so
        the halo is DYNAMIC in a way a fixed 1-3 rows is not, and it grows as
        the grid refines at a fixed aperture (``max|W|`` is a LENGTH).
        Independently of its width, three further steps are globally coupled
        (the ``spline_filter`` IIR, the whole-grid least-squares moments and
        the ``min(det)`` fold reduction) -- see the derivation above ``_tf_sl``
        for the measurements.  Reported through ``progress`` rather than
        warned, so it costs nothing when nobody is listening: the two
        whole-grid reductions below are not even taken without a callback."""
        if progress is None:
            return
        _wmax = max(float(xp.max(xp.abs(_rm_wx_g))),
                    float(xp.max(xp.abs(_rm_wy_g))))
        _need = int(np.ceil(_wmax / dy)) + (int(remap_order) + 1) // 2 + 1
        call_progress(
            progress, 'apply_real_lens',
            i_surf / max(len(surfaces), 1),
            f"surface {i_surf}: tangent_facet_remap pull-back NOT banded -- "
            f"max|W| = {_wmax * 1e6:.3f} um = {_need} halo rows against a "
            f"{cr}-row band ({100.0 * _need / max(cr, 1):.1f} % of the band), "
            f"and the resampling is globally coupled on top of that "
            f"(spline_filter IIR -- measured to need a halo TWICE the band "
            f"before it underflows to identical -- whole-grid least-squares "
            f"moments, min(det) fold reduction).  The SCREEN half IS banded "
            f"({_TF_REMAP_SAG_HALO_ROWS}-row sag halo); the apply half runs "
            f"whole-grid.")

    # Preserve the caller's complex dtype (complex128 or complex64).
    # The numexpr ``out=E`` path below evaluates the phase screen
    # expression in complex128 internally and casts to E.dtype at
    # the final store, which is the documented mitigation that keeps
    # the per-surface OPD accurate even for large ``k0 * opd``
    # arguments regardless of storage precision.  The numpy fallback
    # restores the original dtype explicitly at the end of each
    # surface.  On the GPU path, numexpr isn't available so we use
    # the plain multiplication fallback throughout.
    if xp is cp:
        # Ensure E is a device array of appropriate complex dtype
        if not _is_cupy_array(E_in):
            E = cp.asarray(E_in)
        else:
            E = E_in.copy()
        if not cp.iscomplexobj(E):
            from ..propagators.propagation import DEFAULT_COMPLEX_DTYPE
            E = E.astype(DEFAULT_COMPLEX_DTYPE)
    else:
        if np.iscomplexobj(E_in):
            E = E_in.copy()
        else:
            from ..propagators.propagation import DEFAULT_COMPLEX_DTYPE
            E = E_in.astype(DEFAULT_COMPLEX_DTYPE)

    # Entrance aperture (only if no explicit stop surface specified)
    # Dtype-aware zero to preserve complex64
    # E (the ``0.0 + 0.0j`` literal silently upcast to complex128).
    if aperture is not None and stop_index is None:
        if _chunk_grids:
            _r_ap_sq = (aperture / 2) ** 2
            _cr = int(sag_chunk_rows)
            for _r0 in range(0, Ny, _cr):
                _r1 = min(Ny, _r0 + _cr)
                _h_b = _x_sq[None, :] + _y_sq[_r0:_r1, None]
                E[_r0:_r1] = xp.where(_h_b <= _r_ap_sq, E[_r0:_r1],
                                      xp.zeros((), dtype=E.dtype))
        else:
            # ``E`` is this function's private copy of the input, so the
            # mask is applied IN PLACE: ``xp.where`` would allocate a
            # fresh full complex grid for a result that differs from
            # ``E`` only on the zeroed pixels.  Same values, one grid
            # less.
            E[h_sq_axis > (aperture / 2) ** 2] = 0

    # P2 candidate (a): exit-plane geometric-transfer remap.  Replaces the
    # per-surface screen loop entirely -- warp the (apertured) input envelope
    # through the traced ray map h_in -> h_out with the energy Jacobian and the
    # exit-pupil-referenced eikonal OPD, then return the exit-vertex-plane field.
    if _remap_mode:
        _h_in_map, _h_out_map, _opl_map = _disp_ray_map
        E = _apply_displaced_remap(
            E, _h_in_map, _h_out_map, wavelength, dx, dy, _opl_map,
            eikonal_fn=_disp_eik_fn)
        call_progress(progress, 'apply_real_lens', 1.0, 'done')
        return E

    # P10 (N11): 2-D transverse-walk remap for a decentered / tilted / freeform
    # element.  Same early-return structure as the 1-D remap -- warp the
    # (apertured) input envelope through the full 2-D exit ray map with the
    # energy-conserving 2-D Jacobian + exit-pupil-referenced eikonal OPD, then
    # return the exit-vertex-plane field.  This carries the transverse ray walk
    # the single-plane pointwise screen drops, so the induced-coma spot broadens
    # correctly.
    if _disp_2d_remap:
        E = _apply_displaced_remap_2d(
            E, _disp_ray_map_2d, wavelength, dx, dy, eik_fn=_disp_eik_fn)
        call_progress(progress, 'apply_real_lens', 1.0, 'done')
        return E

    # Resolve glass names once.  Use complex form so we can recover kappa for
    # absorption while still having the real part for geometry/Snell.
    resolved = []
    for surf in surfaces:
        if absorption or fresnel:
            n1c = get_glass_index_complex(surf['glass_before'], wavelength)
            n2c = get_glass_index_complex(surf['glass_after'], wavelength)
        else:
            n1c = complex(get_glass_index(surf['glass_before'], wavelength), 0.0)
            n2c = complex(get_glass_index(surf['glass_after'], wavelength), 0.0)
        resolved.append((n1c, n2c))


    n_surf = len(surfaces)
    for i, surf in enumerate(surfaces):
        call_progress(progress, 'apply_real_lens',
                      i / max(n_surf, 1),
                      f'surface {i + 1}/{n_surf}')
        R = surf['radius']
        kc = surf.get('conic', 0.0)
        asph = surf.get('aspheric_coeffs')
        # Optional anamorphic fields (backward-compatible -- present
        # only on biconic / cylindrical / toroidal surfaces).
        R_y = surf.get('radius_y')
        kc_y = surf.get('conic_y')
        asph_y = surf.get('aspheric_coeffs_y')
        n1c, n2c = resolved[i]
        n1r, n2r = n1c.real, n2c.real
        # This surface's share of the LOCAL-glass-path absorption (see
        # ``_absorb_local_path``).  The first surface has no gap in front of it
        # and the last none behind it, so those halves are dropped -- which is
        # what keeps "no attenuation after the last surface" exact.
        _kap_face = 0.0
        if absorption:
            _kap_face = ((n1c.imag if i > 0 else 0.0)
                         - (n2c.imag if i < n_surf - 1 else 0.0))

        # ---- Opt-in row-band (chunked) phase screen -------------------
        # When ``sag_chunk_rows`` is set AND the surface is the plain conic+
        # aspheric case (no decenter / tilt / form-error / slant / fresnel /
        # surface-frame / biconic / freeform / clear-aperture, and not the
        # stop surface), compute the per-surface sag/OPD and apply the phase
        # screen in row-bands so the full-grid float64 sag + OPD transients
        # never materialise -- only a (chunk_rows x Nx) band at a time.  The
        # sag/OPD are pointwise and the phase screen uses the SAME numexpr
        # (complex128-internal) path the whole grid uses, so this is
        # byte-identical (test_chunked_sag_byte_identical).  Any deviation
        # from the narrow case falls through to the whole-grid path below.
        #
        # ``carrier=`` (the angle-true screen) does NOT disqualify the band
        # -- the obliquity block's two gradients are taken on a 1-/2-row
        # halo, so its per-band arithmetic is the whole-grid arithmetic
        # element for element (test_obl_banded_halo.py).
        _narrow_chunk = (
            sag_chunk_rows is not None and int(sag_chunk_rows) > 0
            and xp is np and not slant_correction and not fresnel
            and not surface_frame and not _displaced
            and (surf.get('decenter') or (0.0, 0.0)) == (0.0, 0.0)
            and (surf.get('tilt') or (0.0, 0.0)) == (0.0, 0.0)
            and surf.get('form_error') is None
            and surf.get('radius_y') is None
            # ANY freeform_type falls through to the whole-grid path --
            # Q-bfs / Q-con so their departure IS computed there, and the
            # non-Q types (zernike / xy_polynomial / chebyshev) so the
            # whole-grid path's "freeform departure is NOT included"
            # RuntimeWarning keeps firing on the (default) banded path.
            and surf.get('freeform_type') is None
            and surf.get('clear_aperture') is None
            and not (stop_index is not None and i == stop_index
                     and aperture is not None)
        )
        if _narrow_chunk:
            cr = int(sag_chunk_rows)
            _use_ne = (NUMEXPR_AVAILABLE and E.size >= _NUMEXPR_MIN_SIZE
                       and _ensure_numexpr_loaded())
            # The whole-grid block is gated on ``bool(xp.any(sag))``; evaluate
            # that reduction band-wise so a FLAT face still skips (and still
            # leaves ``_obl_p0*`` a pair of floats) without a full-grid sag.
            _obl_here = _obl_active and _band_any_sag(R, kc, asph, cr)
            # v5.37: the tangent-facet family bands here too.  ``_obl_active``
            # is False under those models (they supersede equations 4 and 7),
            # so the two ``_here`` flags are mutually exclusive by construction.
            _tf_here = _tf_active and _band_any_sag(R, kc, asph, cr)
            _hw = 0
            if _obl_here:
                _hw = _obl_halo_rows()
            elif _tf_here:
                _hw = _tf_halo_rows()
                _tf_begin_surface()
            if _obl_here:
                _obl_begin_surface()
            for r0 in range(0, Ny, cr):
                r1 = min(Ny, r0 + cr)
                if _obl_here or _tf_here:
                    # sag on a halo: the obliquity gradients need one row
                    # either side (two when the R1 drift term is live); the
                    # tangent-facet screen needs three (two for the remap
                    # rung) -- see the derivation above ``_tf_sl``.
                    _h0 = max(0, r0 - _hw)
                    _h1 = min(Ny, r1 + _hw)
                    _lo = r0 - _h0
                    sag_h = _surface_sag_general(
                        _x_sq[None, :] + _y_sq[_h0:_h1, None], R, kc, asph)
                    sag_b = sag_h[_lo:_lo + (r1 - r0)]
                else:
                    _h_b = (_x_sq[None, :] + _y_sq[r0:r1, None]
                            if h_sq_axis is None else h_sq_axis[r0:r1])
                    sag_b = _surface_sag_general(_h_b, R, kc, asph)
                opd_b = (n2r - n1r) * sag_b
                if _tf_here:
                    opd_b = _tf_band_screen(
                        sag_h, _h0, _h1, r0, r1, n1r, n2r,
                        lambda a0, a1, _s=sag_h, _o=_h0:
                            (n2r - n1r) * _s[a0 - _o:a1 - _o])
                    del sag_h
                if _obl_here:
                    _d_b = _obl_band_delta(sag_h, _h0, _h1, r0, r1, n1r, n2r)
                    if _d_b is not None:
                        opd_b = opd_b + _d_b
                    del _d_b, sag_h
                if bool(np.any(np.isnan(opd_b))):
                    opd_b = np.where(np.isnan(opd_b), 0.0, opd_b)
                # Flat-band early-out (see the whole-grid copy): a zero
                # OPD is a unit screen, so skipping it cannot change a
                # bit.
                if not bool(np.any(opd_b)):
                    pass
                elif _use_ne:
                    Eb = E[r0:r1]
                    _ne.evaluate('Eb * exp(-1j * k0 * opd_b)',
                                 local_dict={'Eb': Eb, 'k0': k0, 'opd_b': opd_b},
                                 out=Eb)
                    _drop_numexpr_out_retention()
                else:
                    ph = _screen_exp(opd_b, k0, np)
                    if ph.dtype != E.dtype:
                        ph = ph.astype(E.dtype)
                    E[r0:r1] *= ph
                if _kap_face != 0.0:
                    E[r0:r1] = _absorb_local_path(
                        E[r0:r1], sag_b, _kap_face, k0, xp)
                del sag_b, opd_b
            if _obl_here:
                _obl_end_surface()
            if _tf_here:
                _tf_end_surface()
                if _tf_remap:
                    # The screen half banded; the pull-back does not (globally
                    # coupled, and its halo is the walk).  Priced + printed.
                    _tf_price_walk_halo(i, cr)
                    E, _tf_px, _tf_py = _tangent_facet_remap_apply(
                        E, _rm_wx_g, _rm_wy_g, _tf_px, _tf_py, dx, dy, k0,
                        int(remap_order), xp, i)
                _rm_wx_g = _rm_wy_g = None
            if i < len(surfaces) - 1 and _obl_active and _obl_apply:
                _obl_gap_advance(i, n2r)
            if i < len(surfaces) - 1 and _tf_active:
                _tf_gap_transport(i, n2r)
            if i < len(surfaces) - 1:
                E = _propagate_through_glass(
                    E, thicknesses[i], wavelength, n2r, n2c.imag,
                    dx, dy, bandlimit, wave_propagator, absorption, k0, xp,
                    stream_transfer_function)
            continue

        # ---- Opt-in row-band (chunked) slant/fresnel phase screen -----
        # v5.17.x: the ``_narrow_chunk`` sibling for the PLAIN conic+aspheric
        # surface WITH slant_correction and/or fresnel on (and, optionally,
        # a per-surface clear_aperture and/or the aperture stop).  Evaluates
        # the ENTIRE refraction pipeline (per-band sag -> local normal ->
        # cos_ti / cos_tt -> refraction OPD -> phase screen -> fresnel
        # amplitude -> TIR mask -> clear_aperture / stop mask) in row-bands,
        # building ``sag`` PER BAND from the axis vectors so the full-grid
        # meshgrids (``_ensure_full_grids``, ~26 GB at N=32768) AND the
        # full-grid ``sag`` (~43 GB float64 transient) NEVER materialise --
        # only a (chunk_rows x Nx) band is live at once.  ``_ensure_full_grids``
        # is never reached on this path (we ``continue`` before it, exactly as
        # ``_narrow_chunk`` does).  The y-sag gradient is taken on a 1-row halo
        # so np.gradient's central differences match the whole-grid result
        # bit-for-bit, and the numexpr phase-screen decision reuses the SAME
        # whole-E.size gate as the whole-grid path, so the banded output is
        # BYTE-IDENTICAL to the whole-grid refraction block below
        # (test_slant_chunk_byte_identical).  Only plain surfaces qualify;
        # decenter / tilt / form-error / biconic / freeform / surface-frame
        # slant/fresnel surfaces fall through to the whole-grid path (their
        # full grids are unavoidable anyway).
        # ``carrier=`` does not disqualify this band either -- the
        # obliquity block rides the SAME halo (widened to 2 rows when the R1
        # drift term is live) and the sag gradient is shared with the
        # refraction pipeline when no NaN sentinel forces a rebuild.
        _slant_narrow_chunk = (
            sag_chunk_rows is not None and int(sag_chunk_rows) > 0
            and xp is np and (slant_correction or fresnel)
            and not surface_frame
            and (surf.get('decenter') or (0.0, 0.0)) == (0.0, 0.0)
            and (surf.get('tilt') or (0.0, 0.0)) == (0.0, 0.0)
            and surf.get('form_error') is None
            and surf.get('radius_y') is None
            and surf.get('freeform_type') is None
        )
        if _slant_narrow_chunk:
            cr = int(sag_chunk_rows)
            clear_ap = surf.get('clear_aperture')
            _is_stop = (stop_index is not None and i == stop_index
                        and aperture is not None)
            # The whole-grid fresnel amplitude REBINDS E to
            # result_type(E.dtype, geometry-real) (complex64 -> complex128
            # for the default float64 geometry).  Reproduce that promotion by
            # routing the fresnel / TIR / aperture band writes into a promoted
            # output array and rebinding E to it after the loop; with no
            # fresnel (or E already wide enough) the output IS E and the writes
            # land in place.  The phase screen always writes the pre-fresnel
            # dtype into E first, so the promotion happens at exactly the same
            # pipeline step as the whole grid.
            if fresnel:
                _out_dtype = xp.result_type(E.dtype, _sag_real)
                E_out = (E if _out_dtype == E.dtype
                         else xp.empty(E.shape, dtype=_out_dtype))
            else:
                E_out = E
            _refr_clamped = False
            _obl_here = _obl_active and _band_any_sag(R, kc, asph, cr)
            _tf_here = _tf_active and _band_any_sag(R, kc, asph, cr)
            _hw = 1
            if _obl_here:
                _hw = max(1, _obl_halo_rows())
            elif _tf_here:
                _hw = max(1, _tf_halo_rows())
                _tf_begin_surface()
            if _obl_here:
                _obl_begin_surface()
            for r0 in range(0, Ny, cr):
                r1 = min(Ny, r0 + cr)
                # 1-row halo so central-difference gradients on the band match
                # the whole-grid np.gradient result exactly; the true array
                # edges (rows 0 and Ny-1) keep their one-sided stencil in the
                # first / last band.  ``sag_halo`` built from the axis vectors
                # is byte-identical to slicing the full-grid sag
                # (_surface_sag_general is pointwise in h_sq).  v5.35.3: the
                # halo widens to 2 rows when the R1 drift term is live (it
                # differentiates a quantity that is itself a gradient); the
                # band's OWN gradient rows are unchanged either way, since
                # np.gradient's interior stencil does not know how far the
                # array extends.
                _h0 = max(0, r0 - _hw)
                _h1 = min(Ny, r1 + _hw)
                h_sq_halo = _x_sq[None, :] + _y_sq[_h0:_h1, None]
                sag_halo = _surface_sag_general(h_sq_halo, R, kc, asph)
                _lo = r0 - _h0
                _hi = _lo + (r1 - r0)
                _dsag_dy_h, _dsag_dx_h = xp.gradient(sag_halo, dy, dx)
                dsag_dy_b = _dsag_dy_h[_lo:_hi]
                dsag_dx_b = _dsag_dx_h[_lo:_hi]
                grad_sq = dsag_dx_b ** 2 + dsag_dy_b ** 2
                one_plus_g = 1.0 + grad_sq
                cos_ti = 1.0 / xp.sqrt(one_plus_g)
                sin2_ti = grad_sq / one_plus_g
                sin2_tt = (n1r / n2r) ** 2 * sin2_ti
                cos_tt = xp.sqrt(xp.maximum(1.0 - sin2_tt, 0.0))
                if (bool(xp.any(cos_ti < 1e-3))
                        or bool(xp.any(cos_tt < 1e-3))):
                    _refr_clamped = True
                cos_ti_safe = xp.maximum(cos_ti, 1e-3)
                cos_tt_safe = xp.maximum(cos_tt, 1e-3)
                sag_b = sag_halo[_lo:_hi]
                if slant_correction:
                    # The AXIAL-TRANSLATION IDENTITY, equation (3) of the
                    # SCREEN OBLIQUITY derivation above: a facet sitting a
                    # height ``sag`` over the vertex plane contributes
                    # ``(pz2 - pz1) * sag`` with BOTH momenta referenced to
                    # the Z-AXIS, not to the facet normal.  For the collimated
                    # (axial) input this screen assumes, pz1 = n1 and the
                    # refracted ray leaves at ``theta_i - theta_t`` to z, so
                    # pz2 = n2 cos(theta_i - theta_t), expanded through
                    # cos(a-b) = cos a cos b + sin a sin b on the cosines the
                    # refraction pipeline has already materialised.  Keep
                    # byte-identical to the whole-grid copy below.
                    opd = (n2r * (cos_ti_safe * cos_tt_safe
                                  + xp.sqrt(sin2_ti * sin2_tt))
                           - n1r) * sag_b
                else:
                    opd = (n2r - n1r) * sag_b
                if _tf_here:
                    # v5.37: route 3 / the remap rung REPLACE the screen above.
                    # It survives only as the thin-screen fallback on a
                    # non-propagating pixel, which is why it is handed over as a
                    # callable evaluated on the EXTENDED rows the accumulator
                    # gradient needs.  ``slant_correction`` is REFUSED with
                    # these models (both replace the same coefficient and
                    # stacking them double-counts), so this path is reached
                    # only through ``fresnel=True`` and the fallback screen is
                    # always the paraxial one.
                    opd = _tf_band_screen(
                        sag_halo, _h0, _h1, r0, r1, n1r, n2r,
                        lambda a0, a1, _s=sag_halo, _o=_h0:
                            (n2r - n1r) * _s[a0 - _o:a1 - _o])
                if _obl_here:
                    # v5.35.3: equation (4) (+ R1) on this band.  The sag
                    # gradient is handed over rather than retaken -- the
                    # obliquity block differentiates the SAME halo array
                    # unless a NaN sentinel forces the zeroed rebuild, which
                    # is exactly the whole-grid rule.
                    _d_b = _obl_band_delta(
                        sag_halo, _h0, _h1, r0, r1, n1r, n2r,
                        grad_h=(_dsag_dy_h, _dsag_dx_h))
                    if _d_b is not None:
                        opd = opd + _d_b
                    del _d_b
                if bool(xp.any(xp.isnan(opd))):
                    opd = xp.where(xp.isnan(opd), 0.0, opd)
                if (xp is np and NUMEXPR_AVAILABLE
                        and E.size >= _NUMEXPR_MIN_SIZE
                        and _ensure_numexpr_loaded()):
                    # Same whole-E.size numexpr gate as the whole-grid path so
                    # both paths make the identical numexpr-vs-numpy choice
                    # (numexpr differs from numpy exp in the last bit, so a
                    # per-band size gate would break byte-identity at the
                    # threshold).  numexpr is element-wise, so evaluating the
                    # band slice equals evaluating the whole grid bit-for-bit.
                    _Eb = E[r0:r1]
                    _ne.evaluate(
                        '_Eb * exp(-1j * k0 * _opd)',
                        local_dict={'_Eb': _Eb, 'k0': k0, '_opd': opd},
                        out=_Eb,
                    )
                    _drop_numexpr_out_retention()
                else:
                    ph = _screen_exp(opd, k0, xp)
                    if ph.dtype != E.dtype:
                        ph = ph.astype(E.dtype)
                    E[r0:r1] *= ph
                if _kap_face != 0.0:
                    E[r0:r1] = _absorb_local_path(
                        E[r0:r1], sag_b, _kap_face, k0, xp)
                # Fresnel amplitude transmission.  ``E[r0:r1] * sqrt(T_eff)``
                # promotes the band to result_type(E.dtype, geometry-real)
                # (complex64 -> complex128 for the default float64 geometry),
                # matching the whole-grid ``E = E * sqrt(T_eff)`` rebinding.
                if fresnel:
                    denom_s = n1c * cos_ti_safe + n2c * cos_tt_safe
                    denom_p = n2c * cos_ti_safe + n1c * cos_tt_safe
                    t_s = 2.0 * n1c * cos_ti_safe / denom_s
                    t_p = 2.0 * n1c * cos_ti_safe / denom_p
                    # POWER transmittance, not |t|**2 -- see the whole-grid
                    # copy below for the convention argument.  Keep the two
                    # expressions byte-identical.
                    T_eff = (0.5 * (xp.abs(t_s) ** 2 + xp.abs(t_p) ** 2)
                             * (n2r * cos_tt_safe) / (n1r * cos_ti_safe))
                    _band = E[r0:r1] * xp.sqrt(T_eff)
                else:
                    _band = E[r0:r1]
                # TIR mask (dtype-aware zero at the band's post-fresnel dtype,
                # mirroring the whole grid's ``xp.zeros((), dtype=E.dtype)``).
                # This path always has slant or fresnel on, so it always runs.
                _band = xp.where(sin2_tt < 1.0, _band,
                                 xp.zeros((), dtype=_band.dtype))
                # Per-surface clear aperture (vignetting) and aperture stop,
                # applied PER BAND (the whole-grid path applies these after
                # refraction via full-grid h_sq / h_sq_axis; decenter is
                # excluded from this path so both use the centred per-band
                # h_sq = x2 + y2, byte-identical to h_sq_axis[r0:r1]).
                if clear_ap is not None or _is_stop:
                    _h_b = _x_sq[None, :] + _y_sq[r0:r1, None]
                    if clear_ap is not None:
                        _band = xp.where(_h_b <= (clear_ap / 2) ** 2, _band,
                                         xp.zeros((), dtype=_band.dtype))
                    if _is_stop:
                        _band = xp.where(_h_b <= (aperture / 2) ** 2, _band,
                                         xp.zeros((), dtype=_band.dtype))
                E_out[r0:r1] = _band
            E = E_out
            if _obl_here:
                _obl_end_surface()
            if _tf_here:
                _tf_end_surface()
                if _tf_remap:
                    # LAST in the surface block, exactly as the whole-grid path
                    # runs it: the vignetting masks above already acted at the
                    # pixel's own incoming coordinate.
                    _tf_price_walk_halo(i, cr)
                    E, _tf_px, _tf_py = _tangent_facet_remap_apply(
                        E, _rm_wx_g, _rm_wy_g, _tf_px, _tf_py, dx, dy, k0,
                        int(remap_order), xp, i)
                _rm_wx_g = _rm_wy_g = None
            if _refr_clamped:
                import warnings
                warnings.warn(
                    "apply_real_lens: clamping near-grazing-incidence "
                    "rays at cos(theta) < 1e-3 floor.  Steep asphere or "
                    "tilted bundle exceeds the surface's physical AOI "
                    "limit; OPD on clamped pixels is artificially "
                    "capped and may differ from the true ray path by "
                    "kilo-radians.  Reduce input tilt or check the "
                    "surface profile.",
                    RuntimeWarning, _WARN_STACKLEVEL,
                )
            if i < len(surfaces) - 1 and _obl_active and _obl_apply:
                _obl_gap_advance(i, n2r)
            if i < len(surfaces) - 1 and _tf_active:
                _tf_gap_transport(i, n2r)
            if i < len(surfaces) - 1:
                E = _propagate_through_glass(
                    E, thicknesses[i], wavelength, n2r, n2c.imag,
                    dx, dy, bandlimit, wave_propagator, absorption, k0, xp,
                    stream_transfer_function)
            continue

        # Whole-grid path from here on -- build the deferred meshgrids on
        # first use (no-op when they already exist).
        X, Y, h_sq_axis = _ensure_full_grids()

        # ---- Decenter --------------------------------------------------
        # When ``surface_frame=True``
        # the decenter+tilt pair is applied as a rigid-body transform
        # of the surface itself (Optiland / Zemax convention) instead
        # of as a field-frame coordinate shift + linear sag ramp.  The
        # forward map (surface frame -> field frame) is
        # ``(x_f, y_f, 0) = Rx(tx) @ Ry(ty) @ (x_s, y_s, z_s) +
        # (dcx, dcy, 0)``; we invert it on the field-plane grid to get
        # ``(x_s, y_s)`` at which sag is evaluated.  Uses the full
        # rotation matrix (no small-angle linearisation) so arbitrary
        # tilts are correct.  Falls back to the field-frame branch
        # below when ``surface_frame=False`` (the default), preserving
        # v5.1 numerics bit-for-bit.
        decenter = surf.get('decenter') or (0.0, 0.0)
        tilt_sf = surf.get('tilt') or (0.0, 0.0)
        _sf_active = surface_frame and (
            decenter[0] != 0.0 or decenter[1] != 0.0
            or tilt_sf[0] != 0.0 or tilt_sf[1] != 0.0
        )
        if _sf_active:
            # Inverse rigid-body transform of the field-plane grid into
            # the surface frame.  R = Rx(theta_x) @ Ry(theta_y); the
            # inverse applied to (x - dcx, y - dcy, 0) gives the
            # surface-frame FOOTPRINT
            #   x_s = cy*dx_local + sx*sy*dy_local
            #   y_s = cx*dy_local
            # at which the sag is evaluated (the thin-element
            # approximation: the surface-frame footprint of the
            # field-plane normal, the same simplification the
            # field-frame branch makes when it skips the perpendicular-
            # foot solve).  The rotated surface's own FIELD-frame height
            # is restored below -- it is where the tilt lives.
            #
            # AXIS CONVENTION: the rotation angles come from the same
            # ``tilt`` key the field-frame branch, ``_disp_surface_z_grad``
            # and ``raytrace``'s ``field_tilt`` all read, where
            # ``tilt = (t0, t1)`` IS the linear sag ramp ``t0*x + t1*y``.
            # That ramp is the right-hand rotation pair
            # ``theta_x = t1`` (a +x rotation ramps in y) and
            # ``theta_y = -t0`` (a +y rotation ramps in -x), so the two
            # branches deflect the beam about the SAME axis and the wave
            # model agrees with the ray models on what a tilt means.
            _sf_thx = float(tilt_sf[1])
            _sf_thy = -float(tilt_sf[0])
            cx_f = np.cos(_sf_thx)
            sx_f = np.sin(_sf_thx)
            cy_f = np.cos(_sf_thy)
            sy_f = np.sin(_sf_thy)
            _dx_local = X - decenter[0]
            _dy_local = Y - decenter[1]
            Xs = cy_f * _dx_local + sx_f * sy_f * _dy_local
            Ys = cx_f * _dy_local
            h_sq = Xs ** 2 + Ys ** 2
            del _dx_local, _dy_local
        elif decenter[0] == 0.0 and decenter[1] == 0.0:
            # Alias the axis-centered grids.  Downstream code only reads
            # Xs/Ys/h_sq and creates new arrays when combining them
            # (e.g. ``sag + tilt[0]*Xs``), so aliasing is safe.  Saves
            # three float64 N x N allocations per surface (~24 GB at
            # N=32768).
            Xs = X
            Ys = Y
            h_sq = h_sq_axis
        else:
            Xs = X - decenter[0]
            Ys = Y - decenter[1]
            h_sq = Xs ** 2 + Ys ** 2

        # ---- Base sag (conic + asphere; biconic if radius_y given) ----
        # 4.11.2: warn when a freeform surface is encountered.
        # ``apply_real_lens`` only computes conic+aspheric+biconic sag
        # at the phase-screen step; it does NOT call
        # ``surface_sag_freeform`` for xy_polynomial / zernike /
        # chebyshev (those remain silently dropped pending a separate
        # fix), so the warning continues to fire for them.
        #
        # Forbes Q-bfs / Q-con sag is a 2-D scalar phase contribution
        # exactly analogous to the (forthcoming) xy-polynomial /
        # Zernike / Chebyshev wave-optics paths, so for
        # ``freeform_type in ('q_bfs', 'q_con')`` the freeform
        # departure is computed here and ADDED to the base conic sag.
        # The dispatch goes through ``surface_sag_freeform`` so it
        # honours the v4.15.1 P1-F1-1 radial clip + P1-F1-2 required-
        # ``r_max`` guards.  Other freeform types still warn-and-skip
        # for now (Agent F scope).
        ft = surf.get('freeform_type')
        if ft in ('q_bfs', 'q_con'):
            # Build a minimal surface dict for the dispatcher.  Use the
            # decentered (Xs, Ys) grid so the freeform departure
            # rides on the same local-coordinate frame as the rest of
            # the per-surface OPD.  surface_sag_freeform internally
            # adds its OWN base conic sag (radius/conic from the
            # dict), so we'd double-count if we added the dispatcher
            # result to ``sag`` below.  Instead, REPLACE the base sag
            # with the dispatcher result -- the dispatcher returns
            # the full ``z_bfs(r) + departure`` (Q-bfs) or
            # ``z_conic(r) + departure`` (Q-con) per its docstring.
            #
            # NB: ``surface_sag_freeform`` only honours rotationally
            # symmetric base conics, not biconic; combining Q-bfs /
            # Q-con with a biconic radius_y is an unsupported edge
            # case so we keep the original biconic sag and warn
            # instead.
            if R_y is not None:
                import warnings
                warnings.warn(
                    f"apply_real_lens: surface {i} combines "
                    f"freeform_type={ft!r} with biconic radius_y; "
                    "the freeform departure is dropped from this "
                    "wave-optics path.  Use apply_real_lens_traced "
                    "for biconic + Forbes Q.",
                    RuntimeWarning, _WARN_STACKLEVEL,
                )
                sag = surface_sag_biconic(
                    Xs, Ys, R_x=R, R_y=R_y,
                    conic_x=kc, conic_y=kc_y,
                    aspheric_coeffs=asph,
                    aspheric_coeffs_y=asph_y)
            else:
                from .freeform import surface_sag_freeform
                # The surf dict already carries the q_bfs_coeffs /
                # q_con_coeffs / r_max / norm_x / norm_y / radius /
                # conic keys; pass it through directly.
                sag = surface_sag_freeform(Xs, Ys, surf)
        else:
            if ft is not None:
                import warnings
                warnings.warn(
                    f"apply_real_lens: surface {i} has freeform_type="
                    f"{ft!r}; the freeform departure "
                    "is NOT included in the per-surface OPD by this "
                    "thin-element wave-optics path.  Use "
                    "apply_real_lens_traced (or apply_real_lens_maslov) "
                    "for a raytraced OPD that honours freeform_type.",
                    RuntimeWarning, _WARN_STACKLEVEL,
                )
            if R_y is not None:
                sag = surface_sag_biconic(
                    Xs, Ys, R_x=R, R_y=R_y,
                    conic_x=kc, conic_y=kc_y,
                    aspheric_coeffs=asph,
                    aspheric_coeffs_y=asph_y)
            elif _disp_pointwise and surf.get('sag_callable') is not None:
                # P3 (N2): freeform sag hook -- the callable returns the full
                # surface departure [m] at the (decentered) surface-frame
                # coordinates (Xs, Ys).  The pointwise obliquity trace used the
                # SAME callable for its ray intersection + normals, so the
                # obliquity OPD (n2 cos_out - n1 cos_in) * sag is self-consistent.
                sag = np.asarray(surf['sag_callable'](Xs, Ys), dtype=_sag_real)
            else:
                sag = _surface_sag_general(h_sq, R, kc, asph)

        # ---- Tilt -----------------------------------------------------
        # FIELD-frame branch (default): the tilt is the linear sag ramp
        # ``t0*x + t1*y`` -- the v3.x contract, and the same reading
        # ``raytrace``'s ``field_tilt`` and ``_disp_surface_z_grad`` use.
        #
        # SURFACE-frame branch: the rotated (Xs, Ys) above is only the
        # FOOTPRINT.  The quantity the screen must imprint is the rotated
        # surface's height in the FIELD frame,
        #     z_f = (R @ (x_s, y_s, g(x_s, y_s)))_z
        #         = R_zx*x_s + R_zy*y_s + R_zz*g(x_s, y_s),
        # with R_z. = (-cx*sy, sx, cx*cy) for R = Rx(theta_x) @ Ry(theta_y).
        # Evaluating ``g`` at the rotated footprint and DISCARDING z_s drops
        # ``R_zx*x_s + R_zy*y_s`` -- to first order the whole ramp, which is
        # the term that deviates the beam: a rigid rotation RE-EXPRESSES the
        # ramp, it does not delete it.  Without this a tilted flat face was a
        # literal no-op (0 mrad deviation where a thin prism gives (n-1)*theta)
        # and a tilted R = 50 mm sphere lost 8.15 waves of OPD at 5 mrad over
        # a +-2 mm pupil.
        tilt = surf.get('tilt') or (0.0, 0.0)
        _tilted = (tilt[0] != 0.0 or tilt[1] != 0.0)
        if _sf_active:
            if _tilted:
                sag = ((-cx_f * sy_f) * Xs + sx_f * Ys
                       + (cx_f * cy_f) * sag)
        elif _tilted:
            sag = sag + tilt[0] * Xs + tilt[1] * Ys

        # ---- Form error map -------------------------------------------
        # ``form_error`` is a FIELD-FRAME map: it is added AFTER (Xs, Ys) have
        # been consumed, so it is neither shifted by ``decenter`` nor rotated by
        # ``surface_frame`` -- it lands on the field grid exactly as supplied.
        # Shape is validated here because numpy broadcasting is happy to accept
        # an obviously-wrong figure map: an ``(Nx,)`` row broadcasts silently
        # down every row of the grid, and a mismatched 2-D map dies with a raw
        # broadcast error (or, for ``(Ny, Nx, 1)``, survives the lens and dies
        # inside the ASM) where every other input to this function gets a
        # precise ``apply_real_lens: ...`` message.
        form_err = surf.get('form_error')
        if form_err is not None:
            _fe_shape = tuple(np.shape(form_err))
            if _fe_shape != (Ny, Nx):
                raise ValueError(
                    f"apply_real_lens: surfaces[{i}]['form_error'] has shape "
                    f"{_fe_shape}, but it must be a 2-D map with the SAME "
                    f"shape as the field, ({Ny}, {Nx}).  It is an additive sag "
                    f"perturbation [m] in the FIELD frame (not shifted by "
                    f"decenter nor rotated by surface_frame), so it is sampled "
                    f"on the field grid pixel for pixel; a 1-D array would be "
                    f"broadcast across every row and a mismatched 2-D one "
                    f"cannot be aligned.")
            if not np.issubdtype(np.asarray(form_err).dtype, np.number):
                raise ValueError(
                    f"apply_real_lens: surfaces[{i}]['form_error'] has dtype "
                    f"{np.asarray(form_err).dtype}, but it must be a real "
                    f"numeric sag perturbation [m].")
            if np.iscomplexobj(form_err):
                raise ValueError(
                    f"apply_real_lens: surfaces[{i}]['form_error'] is complex; "
                    f"it must be a REAL additive sag perturbation [m].  Pass "
                    f"an amplitude/phase screen through a separate element if "
                    f"that is what was meant.")
            sag = sag + form_err

        # ---- Local surface normal -> angles of incidence/refraction ---
        # Needed only for fresnel and/or slant_correction.  When on,
        # the legacy code keeps cos_ti / cos_tt / sin2_tt / etc. all
        # alive simultaneously (~6 N x N float64 arrays), which dwarfs
        # the field memory at N >= 4096.  3.2.14: drop intermediates
        # immediately and free the grad components after grad_sq is
        # built.  At N=8192 this cuts peak refraction-step memory
        # from ~5 GB to ~1.5 GB without changing the math.
        if fresnel or slant_correction:
            # Pass dy for the y-axis spacing: dx for both gives the
            # wrong surface-normal direction on an anamorphic grid
            # (dx != dy).  np.gradient takes the spacing in the same
            # order as the array axes (y, x).
            dsag_dy, dsag_dx = xp.gradient(sag, dy, dx)
            grad_sq = dsag_dx ** 2 + dsag_dy ** 2
            # Free the gradient components -- only grad_sq is needed
            # for the rest of the refraction pipeline.
            del dsag_dx, dsag_dy
            # cos_ti / sin2_ti share `grad_sq + 1.0`; build the safe
            # versions directly to avoid two extra full-grid arrays.
            one_plus_g = 1.0 + grad_sq
            cos_ti = 1.0 / xp.sqrt(one_plus_g)
            sin2_ti = grad_sq / one_plus_g
            del one_plus_g
            del grad_sq
            sin2_tt = (n1r / n2r) ** 2 * sin2_ti
            cos_tt = xp.sqrt(xp.maximum(1.0 - sin2_tt, 0.0))
            # Warn the FIRST time per call a real ray's cosine is
            # clamped.  The 1e-3 floor (~89.94 deg) acts on physical
            # (non-TIR) rays before the TIR mask fires for steep
            # aspheres or strongly tilted bundles.  The corrected *cos
            # OPD form cannot diverge, so the clamp is harmless there;
            # the warning is kept for the Fresnel-coefficient legs,
            # which still divide by cos.
            if bool(xp.any(cos_ti < 1e-3)) or bool(xp.any(cos_tt < 1e-3)):
                import warnings
                warnings.warn(
                    "apply_real_lens: clamping near-grazing-incidence "
                    "rays at cos(theta) < 1e-3 floor.  Steep asphere or "
                    "tilted bundle exceeds the surface's physical AOI "
                    "limit; OPD on clamped pixels is artificially "
                    "capped and may differ from the true ray path by "
                    "kilo-radians.  Reduce input tilt or check the "
                    "surface profile.",
                    RuntimeWarning, _WARN_STACKLEVEL,
                )
            cos_ti_safe = xp.maximum(cos_ti, 1e-3)
            cos_tt_safe = xp.maximum(cos_tt, 1e-3)
            # cos_ti / cos_tt are no longer needed -- only the _safe
            # versions and sin2_tt (for TIR mask) survive.
            del cos_ti, cos_tt

        # ---- Refraction OPD (thin-element phase screen) ---------------
        # Note: a BPM-style "interface sub-slicing" mode was
        # prototyped (see git history) but does not deliver the
        # accuracy improvement it promises on sharp air-glass
        # interfaces: simple single-reference-medium BPM requires
        # sub-wavelength axial slabs, which for realistic
        # interface thicknesses (~100 um) means 1000s of slabs --
        # too slow.  Sub-wavelength slabs are needed because the
        # BPM approximation (reference-medium Fresnel kernel + local
        # phase correction) breaks down for step-discontinuous
        # media.  Users needing better than thin-element accuracy
        # should use ``apply_real_lens_traced`` which bypasses this
        # limitation entirely by ray-tracing each pixel.
        if _displaced and _disp_pointwise:
            # P3 (N2): pointwise 2-D obliquity.  The per-surface z-axis ray
            # cosines were traced on a 2-D ray grid (honouring decenter / tilt /
            # freeform sag_callable) and interpolated onto THIS field grid at the
            # ray crossing positions, so the obliquity OPD is the SAME equation
            # (1) evaluated per point instead of via the rotationally-symmetric
            # radial LUT.  ``sag`` already carries the decenter shift, the
            # small-angle tilt ramp, and any freeform sag_callable departure.
            _cin, _cout = _disp_cos_grid[i]
            opd = (n2r * _cout - n1r * _cin) * sag
        elif _displaced:
            # Ray-angle-aware refraction OPD
            # (n2 cos_alpha_out - n1 cos_alpha_in) * sag, cosines of the
            # TRUE ray angle to the z-axis from the collimated meridional
            # fan (interpolated onto the grid radius r).  Carries the
            # incoming-ray-angle physics the paraxial/slant screens drop and
            # splits plano-convex orientation.  See the module-level
            # derivation + oracle evidence.
            r_grid = xp.sqrt(h_sq)
            opd = _displaced_opd(sag, r_grid, _disp_luts[i], n1r, n2r)
        elif slant_correction:
            # The axial-translation identity, equation (3) of the SCREEN
            # OBLIQUITY derivation above: ``(pz2 - pz1) * sag`` with both
            # momenta referenced to the Z-AXIS.  See the banded-copy comment
            # above for the full derivation; keep byte-identical.
            opd = (n2r * (cos_ti_safe * cos_tt_safe
                          + xp.sqrt(sin2_ti * sin2_tt)) - n1r) * sag
        else:
            opd = (n2r - n1r) * sag
        # ---- Route 3: the per-pixel tangent-facet screen (v5.36.0) -----
        # REPLACES the paraxial screen above rather than correcting it: the
        # OPD is the exact axial-translation identity at the facet tangent to
        # the surface where THIS PIXEL'S ray meets it, evaluated at the ray
        # angle the field itself carries.  Derivation + the measured term
        # ladder are above ``_tangent_facet_screen``.
        # The remap rung's walk fields live from the screen block to the remap
        # block below; a flat face (or a non-remap model) leaves this False and
        # the field is never resampled, which is what keeps a plate exact.
        _rm_pending = False
        _rm_wx = _rm_wy = None
        # ONE reduction over ``sag``, shared by the tangent-facet block,
        # the obliquity block and the flat-face early-out below.
        _sag_any = bool(xp.any(sag))
        if _tf_active and _sag_any:
            # A FLAT face -- a plate, a cemented plano, a stop -- has no facet
            # to tilt and no height to translate, so the identity collapses to
            # the paraxial screen exactly and one reduction skips the block
            # (which is also what keeps a plate machine-exact at any tilt, and
            # what keeps the accumulator a pair of floats through one).
            _tf_sag = sag
            if bool(xp.any(xp.isnan(sag))):
                _tf_sag = xp.where(xp.isnan(sag), 0.0, sag)
            _tg_y, _tg_x = xp.gradient(_tf_sag, dy, dx)
            if _tf_remap:
                # ---- the REMAP rung: (R1) the screen, (R2) the walk -------
                # The Hessian is taken ONCE and shared between the hit-point
                # fixed point (R4) and the facet normal there (R5); route 3's
                # (T2b) takes the same two gradients, so the surcharge of this
                # model over that one is the walk fields, not the derivatives.
                _rm_gxy, _rm_gxx = xp.gradient(_tg_x, dy, dx)
                _rm_gyy, _rm_gyx = xp.gradient(_tg_y, dy, dx)
                (_tf_opd, _rm_wx, _rm_wy, _rm_ox, _rm_oy,
                 _tf_ok) = _tangent_facet_remap_screen(
                    _tf_sag, _tg_x, _tg_y, _rm_gxx, _rm_gxy, _rm_gyx,
                    _rm_gyy, _tf_px, _tf_py, n1r, n2r, xp)
                del _rm_gxx, _rm_gxy, _rm_gyx, _rm_gyy
            else:
                _tf_opd, _tf_ok = _tangent_facet_screen(
                    _tf_sag, _tg_x, _tg_y, _tf_px, _tf_py, n1r, n2r,
                    dx, dy, xp)
            # Keep the paraxial screen wherever the facet refraction is not
            # propagating (evanescent input, or TIR at the facet): a clamped
            # cosine is a wrong OPD and the thin screen is the safe neutral.
            _tf_all_ok = (bool(xp.all(_tf_ok))
                          and bool(xp.all(xp.isfinite(_tf_opd))))
            if not _tf_all_ok:
                _tf_keep = _tf_ok & xp.isfinite(_tf_opd)
                _tf_opd = xp.where(_tf_keep, _tf_opd, opd)
                if _tf_remap:
                    # A non-propagating pixel keeps the thin screen AT ITS OWN
                    # COORDINATE: a clamped walk is a wrong POSITION, which is
                    # strictly worse than a wrong phase because it also moves
                    # energy.  Zeroing the walk there is the same "safe neutral"
                    # choice the OPD fallback makes, and it keeps the map's
                    # Jacobian finite so the fold guard scores the real walk
                    # rather than a clamp artefact.
                    _rm_wx = xp.where(_tf_keep, _rm_wx, 0.0)
                    _rm_wy = xp.where(_tf_keep, _rm_wy, 0.0)
                    _rm_ox = xp.where(_tf_keep, _rm_ox, _tf_px)
                    _rm_oy = xp.where(_tf_keep, _rm_oy, _tf_py)
                del _tf_keep
            opd = _tf_opd
            # ORDERING, not a guard: the NaN-sentinel zeroing the OPD receives
            # a few lines below is hoisted to HERE, because the accumulator is
            # persistent.  The OPD's own sentinel costs one annulus for one
            # surface; a sentinel read into the accumulator would travel to
            # every later surface and to the whole exit field.  Route 3's own
            # terms are all proportional to the (already zeroed) ``_tf_sag``,
            # so the only way ``opd`` can be NaN here is the thin-screen
            # fallback firing on a pixel that is BOTH non-propagating and
            # outside the conic domain -- rare, unexercised by any fixture in
            # this build, and therefore ordered around rather than claimed as
            # a fixed bug.  The downstream zeroing is left in place and simply
            # becomes a no-op for this model.
            if bool(xp.any(xp.isnan(opd))):
                opd = xp.where(xp.isnan(opd), 0.0, opd)
            if _tf_remap:
                # (R3): the remap rung's kick is the gradient of the COMPOSITE,
                # not of the screen alone, and that composite gradient is
                # IDENTICALLY the exact refracted momentum -- the screen puts in
                # ``A^T p_out`` and the coordinate change divides ``A^T`` back
                # out.  So the accumulator is ``p_out`` in closed form rather
                # than differenced off the grid, and it rides the same
                # resampling as the field below.  Pinned as an EQUALITY of two
                # independently computed grids by
                # ``test_..._is_the_exact_refracted_momentum_for_a_plane_facet``
                # (1.5e-15 to 4.6e-14 relative, where the model has no
                # truncation left) -- which is precisely what route 3, whose
                # kick is the gradient of its own screen, could not have.
                _tf_px, _tf_py = _rm_ox, _rm_oy
                _rm_pending = True
                del _rm_ox, _rm_oy
            else:
                # Route 3: the field's momentum after this screen IS minus the
                # gradient of what the screen imprinted -- not the exact facet
                # kick.  (The two differ by -sag grad dz; the tangent-facet RAY
                # arm uses the exact kick with this OPD and is therefore NOT a
                # wave model, since a screen's kick is its own value's
                # gradient.  That inconsistency is what separates route 3 from
                # that arm, and what the remap rung above escapes.)
                _tk_y, _tk_x = xp.gradient(opd, dy, dx)
                _tf_px = _tf_px - _tk_x
                _tf_py = _tf_py - _tk_y
                del _tk_x, _tk_y
            del _tg_x, _tg_y, _tf_sag, _tf_ok, _tf_opd
        # ---- Screen obliquity (v5.35.0) -------------------------------
        # The angular part of the exact thin-facet screen OPD, equation (4)
        # of the module-level derivation.  Added ON TOP of whichever screen
        # the caller selected, because it is a DIFFERENCE against that
        # screen's own normal-incidence value: the paraxial / slant /
        # displaced choice sets the zero-angle behaviour and this sets how
        # it changes with the carrier's local ray angle.  Zero for a plane
        # plate, zero for a zero carrier.
        if _obl_active and _sag_any:
            # (a FLAT surface -- a plate face, a cemented plano, a stop -- has
            # nothing to correct and nothing to deflect, so one reduction skips
            # the whole block including the gradient.)
            _sag_ok = sag
            if bool(xp.any(xp.isnan(sag))):
                _sag_ok = xp.where(xp.isnan(sag), 0.0, sag)
            _og_y, _og_x = xp.gradient(_sag_ok, dy, dx)
            # v5.35.3: materialise the carrier momentum field if the banded
            # row-evaluator was in use and THIS surface fell through here.
            _q_wx, _q_wy = _obl_q_whole()
            _d_obl = _screen_obliquity_delta(
                _sag_ok, _og_x, _og_y, _obl_p0x, _obl_p0y,
                _q_wx, _q_wy, n1r, n2r, xp)
            if _obl_total is not None:
                # The ESTIMATOR scores equation (4) alone, which is the size
                # of the defect the blind screen carries (measured 7.5 % low
                # against the exact-ray truth on design 121 group 5).  R1 is a
                # SECOND correction to the SAME defect and partially cancels
                # the first, so adding its magnitude in would double-count:
                # scoring the sum reads 0.395 waves against a 0.258-wave truth.
                _obl_total += _d_obl
            if _obl_apply:
                if _obl_drift_live:
                    # R1 (equation 7): the angle-blind kick error carried over
                    # the drift the carrier has accumulated getting here.
                    # Skipped entirely at zero drift -- surface 0 always, and
                    # every surface for a zero-angle carrier -- so it can
                    # neither cost nor perturb those calls.
                    _d_obl = _d_obl + _screen_drift_opd(
                        _sag_ok, _og_x, _og_y, _obl_p0x, _obl_p0y, n1r, n2r,
                        _obl_ux, _obl_uy, dx, dy, xp)
                opd = opd + _d_obl
            # the screen model's OWN carrier-free momentum, accumulated at
            # this field point for the next surface's local ray angle
            _obl_p0x = _obl_p0x - (n2r - n1r) * _og_x
            _obl_p0y = _obl_p0y - (n2r - n1r) * _og_y
            del _og_x, _og_y, _sag_ok, _d_obl
        # 4.11.2: mask the NaN sentinel returned by surface_sag_general
        # for points outside the conic domain (norm >= 0.9999, i.e. where
        # the surface is not defined for hyperbolic / oblate conics) and
        # propagated through the slant/fresnel gradient pipeline.  Without
        # this, ``exp(-i k0 NaN) = NaN`` poisons the entire downstream
        # ASM step; the NaN >= comparisons used by the near-grazing TIR
        # guard return False, so the warning never fires for those
        # pixels.  We zero the OPD on undefined-surface pixels here; the
        # caller's clear_aperture / aperture mask should already be zeroing
        # the field on the same pixels, so a 0-OPD phase screen is a safe
        # neutral.  Tracks both slant-corrected and paraxial OPD branches.
        if bool(xp.any(xp.isnan(opd))):
            opd = xp.where(xp.isnan(opd), 0.0, opd)
        # ---- FLAT-FACE EARLY-OUT ---------------------------------
        # A plano face -- a plate, a cemented plano, a window, the stop
        # -- has ``sag == 0`` everywhere, hence ``opd == 0``, hence a
        # screen of ``exp(0) == 1 + 0j``; multiplying a finite field by
        # that returns it unchanged bit for bit.  The tangent-facet and
        # obliquity blocks have always skipped on this same reduction;
        # the default screen did not, and paid a full complex ``exp``
        # plus a full complex multiply (566 ms and 134 MB of temporaries
        # at N = 2048) for an identity.  Every other per-surface step --
        # Fresnel, the TIR mask, the apertures -- still runs: a flat face
        # refracts and vignettes even though it imprints no OPD.
        if not _sag_any:
            pass
        elif (xp is np and NUMEXPR_AVAILABLE
                and E.size >= _NUMEXPR_MIN_SIZE
                and _ensure_numexpr_loaded()):
            # Fused multiply + complex exp in one threaded, chunked pass
            # -- avoids the three complex128 N x N temporaries that
            # ``E * np.exp(-1j * k0 * opd)`` otherwise materialises.
            # With ``out=E``, numexpr evaluates the expression at
            # complex128 internal precision and casts only at the
            # final store, so complex64 E gets a double-precision
            # phase accumulation + single-precision storage.
            # CPU only -- numexpr has no GPU backend.
            _ne.evaluate(
                'E * exp(-1j * k0 * opd)',
                local_dict={'E': E, 'k0': k0, 'opd': opd},
                out=E,
            )
            # ...and drop numexpr's own reference to E, which otherwise
            # outlives every ``del`` the caller writes (D2).
            _drop_numexpr_out_retention()
        else:
            # Fallback for GPU (xp is cp) or small CPU arrays: compute
            # exp() in the array backend's precision, then cast back
            # to E's dtype so we don't silently upcast a complex64
            # field to complex128.  CuPy's exp is fused at kernel
            # level so the "three temporaries" concern doesn't apply
            # the same way on device.
            phase_exp = _screen_exp(opd, k0, xp)
            if phase_exp.dtype != E.dtype:
                phase_exp = phase_exp.astype(E.dtype)
            # In place: ``E`` is private here, and ``E = E * ph``
            # allocates a third full complex grid to hold a result that
            # overwrites its own left operand (1.33x on its own).
            E *= phase_exp

        # ---- Local-glass-path bulk absorption -------------------------
        if _kap_face != 0.0:
            E = _absorb_local_path(E, sag, _kap_face, k0, xp)

        # ---- Fresnel amplitude transmission ---------------------------
        if fresnel:
            # Average the INTENSITY coefficients for unpolarised
            # scalar throughput, not the amplitude coefficients.  At
            # Brewster's angle (or any high AOI), t_s and t_p have
            # different phases; their amplitude sum can cancel where
            # sqrt(0.5*(|t_s|^2+|t_p|^2)) correctly captures the
            # incoherent average power.  ``0.5*(t_s+t_p)`` matches only
            # 45-deg linear polarisation at low AOI.  For polarised
            # inputs route through the Jones pipeline.
            #
            # ``t_s`` / ``t_p`` are AMPLITUDE coefficients.  This library
            # treats ``sum |E|**2 dx dy`` as POWER everywhere (the ASM legs
            # are Parseval-unitary and ``_propagate_through_glass`` adds no
            # impedance factor), so crossing an index step needs the POWER
            # transmittance
            #     T = (n2 cos theta_t) / (n1 cos theta_i) * |t|**2,
            # whose extra factor is the ratio of the axial Poynting fluxes on
            # the two sides.  Without it a single AIR->N-BK7 face transmits
            # |t|**2 = 0.6323 where the physical value is 0.9581; an
            # air->glass->air element happens to be right anyway because the
            # n2/n1 factors telescope to n_last/n_first = 1, which is why the
            # error survived -- anything ENDING in glass, any bare cemented
            # interface, and the cos ratio at finite NA did not.
            denom_s = n1c * cos_ti_safe + n2c * cos_tt_safe
            denom_p = n2c * cos_ti_safe + n1c * cos_tt_safe
            t_s = 2.0 * n1c * cos_ti_safe / denom_s
            t_p = 2.0 * n1c * cos_ti_safe / denom_p
            T_eff = (0.5 * (xp.abs(t_s) ** 2 + xp.abs(t_p) ** 2)
                     * (n2r * cos_tt_safe) / (n1r * cos_ti_safe))
            E = E * xp.sqrt(T_eff)

        # ---- TIR mask -----------------------------------------------
        # Suppress regions that went into total internal reflection.
        # This must fire whenever ``sin2_tt`` was computed -- i.e. for
        # both ``fresnel=True`` and ``slant_correction=True`` paths,
        # since the slant OPD divides by ``cos_tt_safe`` which is
        # ill-defined where ``sin2_tt > 1``.  Running it only inside the
        # Fresnel block leaves slant_correction=True + fresnel=False
        # callers with unphysical residual field amplitude in TIR
        # regions.
        if fresnel or slant_correction:
            # In-place mask instead of a fresh full complex grid.  The
            # sense is ``not (sin2_tt < 1.0)`` rather than
            # ``sin2_tt >= 1.0`` so a NaN still lands on the zeroed side,
            # exactly where ``xp.where`` put it.
            _tir = sin2_tt < 1.0
            xp.logical_not(_tir, out=_tir)
            E[_tir] = 0
            del _tir

        # ---- Per-surface clear aperture (vignetting) ------------------
        clear_ap = surf.get('clear_aperture')
        if clear_ap is not None:
            # In-place mask (NaN-safe sense, as for the TIR mask above).
            _cm = h_sq <= (clear_ap / 2) ** 2
            xp.logical_not(_cm, out=_cm)
            E[_cm] = 0
            del _cm

        # ---- Aperture stop applied at this surface --------------------
        if stop_index is not None and i == stop_index and aperture is not None:
            # Respect the stop surface's decenter/displacement if any:
            # ``h_sq_axis`` is centred at the optical axis, so a
            # decentred stop modelled there clips the wrong region of
            # the beam.  The surface dict's key is ``decenter`` and its
            # value is a ``(dx, dy)`` tuple -- mirror line 520 above.
            # Dtype-aware zero for both branches.
            _dec = surf.get('decenter') or (0.0, 0.0)
            xc_stop = float(_dec[0])
            yc_stop = float(_dec[1])
            if xc_stop == 0.0 and yc_stop == 0.0:
                _sm = h_sq_axis <= (aperture / 2) ** 2
            else:
                _sm = ((X - xc_stop) ** 2 + (Y - yc_stop) ** 2
                       <= (aperture / 2) ** 2)
            xp.logical_not(_sm, out=_sm)
            E[_sm] = 0
            del _sm

        # ---- The REMAP rung: carry the field to the walked positions ---
        # LAST in the surface block, so the vignetting masks above still act at
        # the pixel's own (incoming) vertex-plane coordinate exactly as they do
        # for every other model -- the aperture is a property of the surface,
        # and the walk is what happens between the vertex plane and back.  Runs
        # at the LAST surface too: the exit face carries the largest walk of the
        # element (measured 54.9 um on design 121 group 5 at a 3 mm pupil,
        # against 18.2 and 2.2 um on the two entrance faces), so stopping one
        # surface short would drop the dominant term.
        if _rm_pending:
            E, _tf_px, _tf_py = _tangent_facet_remap_apply(
                E, _rm_wx, _rm_wy, _tf_px, _tf_py, dx, dy, k0,
                int(remap_order), xp, i)
        del _rm_wx, _rm_wy
        # ---- Propagate through glass to the next surface --------------
        # Dispatch (asm / sas / fresnel / rayleigh_sommerfeld) + bulk
        # absorption is factored into ``_propagate_through_glass`` so the
        # row-band (chunked) path above reuses the identical implementation.
        if i < len(surfaces) - 1 and _obl_active and _obl_apply:
            # ---- Advance the carrier's ray drift across this gap (eq. 6) --
            # Only when the correction is being APPLIED: the guard's estimator
            # scores equation (4) alone, so a ``screen_obliquity=False`` call
            # never reads the drift and must not pay for it.
            # Runs for EVERY gap, powered surface or not: a plate face has no
            # coefficient error of its own but the gap behind it still moves
            # the carrier's ray, and a later powered surface reads that drift.
            # The gap is always the PHYSICAL one: a carrier is refused with
            # every non-'thin' surface_model, so the 'displaced' split
            # factorisation's reduced distance cannot reach this step.
            # v5.35.3: the body is ``_obl_gap_advance`` so the two row-banded
            # surface paths reach the identical step (banded internally when
            # sag_chunk_rows is live; byte-identical either way).
            _obl_gap_advance(i, n2r)
        if i < len(surfaces) - 1 and _tf_active:
            # ---- Route 3: resample the momentum accumulator across the gap.
            # Runs for EVERY gap, powered face or not; a scalar accumulator
            # (no carrier yet, or a leading plate) passes through untouched.
            # ``'split'`` is a 'displaced'-only mode, so the physical gap is
            # always the one the model propagates.
            # v5.37: routed through ``_tf_gap_transport`` so a prescription
            # that MIXES banded and whole-grid surfaces reaches the identical
            # step from either side (banded internally when sag_chunk_rows is
            # live, byte-identical either way).
            _tf_gap_transport(i, n2r)
        if i < len(surfaces) - 1:
            if _split_mode:
                # P2 candidate (b): the internal gap is propagated as the
                # REDUCED distance ``t / n`` in air (lambda) rather than the
                # physical ``t`` at ``lambda / n`` -- the paraxial-equivalent
                # thin-lens factorisation (entrance screen + t/n + exit screen).
                E = _propagate_through_glass(
                    E, thicknesses[i] / n2r, wavelength, 1.0, 0.0,
                    dx, dy, bandlimit, wave_propagator, False, k0, xp,
                    stream_transfer_function)
            else:
                E = _propagate_through_glass(
                    E, thicknesses[i], wavelength, n2r, n2c.imag,
                    dx, dy, bandlimit, wave_propagator, absorption, k0, xp,
                    stream_transfer_function)

    # v5.35.3: the screen-obliquity momentum / drift / carrier accumulators are
    # per-surface state and are DEAD once the loop ends.  Drop them here so the
    # guard's own pupil temporaries (and the Seidel block's) do not stack on
    # top of up to six persistent full-grid arrays -- a pure lifetime change,
    # no arithmetic reads them again.  ``_obl_total`` is the one the guard
    # scores, so it survives.
    # Route 3's accumulator is dead here too, and unlike the obliquity ones it
    # exists without a carrier, so it is released outside that guard.
    _tf_px = _tf_py = None
    if _obl_active:
        _obl_p0x = _obl_p0y = _obl_ux = _obl_uy = None
        _obl_qx = _obl_qy = None

    # ----- Seidel correction ------------------------------------------
    # A ray-trace-derived radial phase screen applied at the exit pupil,
    # carrying the high-order residual the split-step model itself does not
    # reproduce.  Three things have to be true of it, and all three were not:
    #
    #   (1) BOTH OPLs must be read on the EXIT VERTEX PLANE.  ``trace`` leaves
    #       rays at the last surface's sag, so ``image_rays.opd`` is short by
    #       ``n_exit * sag_last(rho) / N`` -- exactly 0 on a plano rear face
    #       (which is what every fixture had) and -17.05 um at h = 3.6 mm on a
    #       cemented doublet's R = -291 mm rear.  Fixed by
    #       ``TraceResult.at_exit_vertex()``.
    #
    #   (2) The MODEL reference must be the model's own exit OPL, not
    #       ``sum (n2-n1) sag_i(h)``.  That sum evaluates every surface at the
    #       same entrance height with no propagation, so differencing it
    #       against a real trace produces the in-glass obliquity term
    #       ``n t theta**2 / 2`` -- which the split-step's ASM legs ALREADY
    #       carry exactly.  Fitting it imprints it a second time (measured
    #       337 nm against a 341 nm prediction on a plano-convex whose true
    #       model residual is 0.85 nm).  ``_split_step_fan_opl`` replaces it
    #       with a thin-screen ray model of what the split step does.
    #
    #   (3) The fit must start at rho**4.  A basis containing rho**2 IS a
    #       defocus term, so every reference-frame mismatch above was absorbed
    #       as focus and imprinted: 420 nm at the rim on a plano-convex,
    #       -5.5 um on a doublet, moving the doublet's best focus by 2.5 % and
    #       costing 27 % of its peak intensity.  A "Seidel-style high-order
    #       residual" starts at rho**4 by definition.
    #
    # With all three, the 5 nm gate finally does what it was written for: it
    # SKIPS the well-corrected singlets whose residual is already sub-nm, and
    # fires only where there is a real high-order residual to carry.
    # The fan is collimated and on-axis, so the correction is radially
    # symmetric and applied per pixel via r = sqrt(x^2 + y^2).
    if seidel_correction and aperture is not None:
        # Local imports to avoid circular dep at module load
        from ..raytrace import (
            _make_bundle as _rt_make_bundle,
            surfaces_from_prescription as _rt_surfaces_from_prescription,
            trace as _rt_trace,
        )
        r_pupil = 0.5 * aperture
        n_fan = 41
        # The fan spans the FULL clear aperture (0.999 rather than 0.9 of it).
        # The screen below is imprinted out to rho = 1, so every ray height the
        # fit does not cover is a radius the polynomial has to EXTRAPOLATE
        # into -- and the transverse walk through the element is inward, so a
        # fast element already lands well short of wherever the fan is
        # launched.  Measured on an f/2 singlet with the 0.9 launch: the fan
        # landed at rho = 0.808 while the screen was applied to rho = 1.0, so a
        # third of the pupil AREA (a fifth of its energy) got an extrapolated
        # correction that swung through three waves and cost 37 % of the focal
        # peak.  Launching to the rim shrinks that band; the clamp below
        # removes what is left of it.
        h_fan = np.linspace(-0.999 * r_pupil, 0.999 * r_pupil, n_fan)
        z_arr = np.zeros_like(h_fan)
        fan = _rt_make_bundle(
            x=h_fan, y=z_arr, L=z_arr, M=z_arr,
            wavelength=wavelength)
        surfs_fan = _rt_surfaces_from_prescription(prescription)
        res_fan = _rt_trace(fan, surfs_fan, wavelength)
        # (1) The ray OPL must be read on the exit VERTEX plane, not at the
        # last surface's sag where ``trace`` leaves it.
        final_fan = res_fan.at_exit_vertex()
        alive_fan = final_fan.alive
        if alive_fan.sum() >= 5:
            opl_ray = final_fan.opd[alive_fan]
            x_ray = final_fan.x[alive_fan]
            n_exit_fan = float(get_glass_index(
                surfaces[-1]['glass_after'], wavelength))
            p_ray = n_exit_fan * final_fan.L[alive_fan]
            h_alive = h_fan[alive_fan]
            # (2) The MODEL's own exit-vertex-plane OPL on the SAME fan --
            # a thin-screen ray walk through the screens this function
            # applies and the glass gaps its ASM legs propagate.
            x_model, opl_model = _split_step_fan_opl(
                surfaces, thicknesses, wavelength, h_alive)
            # Both OPLs now live on the same PLANE, but not yet at the same
            # POINT: a ray and its model counterpart launched from the same
            # entrance height land up to 12.7 um apart on the exit plane of
            # an 8 mm doublet.  The screen multiplies the field at a fixed
            # exit COORDINATE, so the ray OPL is carried the short way from
            # where the ray landed to where the model's ray landed, at the
            # exit momentum ``p = n_exit * L``:
            #     OPL_ray(x_model) = OPL_ray(x_ray) + p . (x_model - x_ray)
            # -- the standard eikonal transfer, exact to second order in a
            # displacement this small, and the same construction the exact-ray
            # oracle scores this model with.  Without it the residual is read
            # across a landing offset and comes out 114 nm where the true one
            # is 173 nm, so the correction under-corrects by a third and the
            # exit wavefront gets WORSE (measured 0.60x).
            # SIGN: the screens are ``exp(-i k0 OPD)`` under
            # ``phase = exp(+i k0 OPL)``, so the model's OPL is already the
            # NEGATIVE of the deposited OPD -- ``_split_step_fan_opl`` returns
            # an OPL directly and the two quantities are differenced as-is.
            opl_ray_at_model = opl_ray + p_ray * (x_model - x_ray)
            i_ax = int(np.argmin(np.abs(h_alive)))
            correction = ((opl_ray_at_model - opl_ray_at_model[i_ax])
                          - (opl_model - opl_model[i_ax]))
            # The screen multiplies the field at the EXIT-PLANE coordinate, so
            # the pupil coordinate of the fit is where the model ray LANDS,
            # not where it entered.
            rho = x_model / r_pupil
            # (3) rho**4 and up.  rho**2 is defocus, not a Seidel high-order
            # residual, and including it lets any residual reference mismatch
            # be absorbed as a focus shift and imprinted on the field.
            max_order = max(4, int(seidel_poly_order))
            even_powers = np.arange(4, max_order + 2, 2)
            A = np.column_stack([rho ** p for p in even_powers])
            coeffs, *_ = np.linalg.lstsq(A, correction, rcond=None)
            # Suppress fitting noise: if the RMS correction across the
            # fan is already well below typical simulation residual,
            # skip application to avoid injecting polynomial-fit
            # artefacts into otherwise-clean fields.  The gate is ~5 nm:
            # after the C-LR-1 sign fix the typical residual is a
            # few-nm range, so a 50 nm gate silently skips most real
            # corrections, while ~5 nm is well below the Marechal
            # lambda/14 ~ 35 nm at visible wavelengths and still above
            # the lstsq numerical noise floor for a 6th-order
            # even-polynomial fit on ~50 fan samples.
            # Score the gate on what will actually be IMPRINTED (the fitted
            # rho**4+ part), not on the raw residual: the two were the same
            # quantity only because the old rho**2-up basis could fit anything.
            corr_rms = float(np.sqrt(np.mean((A @ coeffs) ** 2)))
            if corr_rms > 5e-9:  # > 5 nm RMS to be worth applying
                # h_sq_axis is in the array backend (xp) already;
                # coeffs came from a CPU lstsq so scalar-broadcast
                # them into xp.  Final phase screen multiplies E on
                # the target device.
                X, Y, h_sq_axis = _ensure_full_grids()
                rho_map_sq = h_sq_axis / (r_pupil ** 2)
                corr_map = xp.zeros_like(rho_map_sq)
                for _p, c in zip(even_powers, coeffs):
                    corr_map = corr_map + float(c) * rho_map_sq ** (_p // 2)
                # (4) CLAMP, never EXTRAPOLATE.  The fit lives on the radii the
                # model's rays actually LAND at, and the transverse walk
                # through the element is inward, so even a rim-launched fan
                # lands short of the pupil edge (measured 0.87 of it on an f/2
                # singlet, 0.78 on an 8 mm cemented doublet).  Continuing a
                # rho**4 + rho**6 polynomial past its own data is where the
                # coefficients stop meaning anything: on the f/2 the shipped
                # screen ran +1926.6 nm at the last fitted radius and +25.4 nm
                # at the rim -- three waves of pure extrapolation over a third
                # of the pupil AREA carrying a fifth of its energy, which cost
                # 37 % of the focal peak while the wavefront INSIDE the fit
                # improved 215x.  Holding the last fitted value instead is a
                # PISTON over that band (unobservable) and turned the same
                # measurement into +37 %.  Outside the clear aperture the map
                # stays 0 exactly as before.
                _rho_fit = float(np.max(np.abs(rho)))
                if _rho_fit < 1.0:
                    _corr_edge = float(sum(
                        float(c) * _rho_fit ** int(_p)
                        for _p, c in zip(even_powers, coeffs)))
                    corr_map = xp.where(rho_map_sq <= _rho_fit ** 2,
                                        corr_map, _corr_edge)
                corr_map = xp.where(rho_map_sq <= 1.0, corr_map, 0.0)
                E = E * xp.exp(+1j * k0 * corr_map)

    # ---- THE SCREEN-OBLIQUITY ACCURACY GUARD (v5.35.0) ----------------
    # The same closed form, read as an ERROR ESTIMATOR: the piston-and-
    # tilt-free rms of the summed correction over the pupil is exactly the
    # wavefront error the angle-blind screen carries at this carrier angle
    # (the prescription's sag x the carrier's angle, per surface).  Silent
    # for carrier-free calls (nothing to estimate) and for small-angle
    # calls (the estimate falls under the documented tolerance).
    # Route 3 has no equation-(4) accumulator to score: its whole OPD IS the
    # angle-true screen, so the estimator (which measures the SIZE of the
    # correction the thin screen needs) has nothing to read and would be
    # meaningless.  The guard is therefore silent for surface_model=
    # 'tangent_facet' -- stated in the docstring rather than faked.
    if _obl_active and on_screen_obliquity != 'silent' and not _tf_active:
        X, Y, h_sq_axis = _ensure_full_grids()
        _r_pup = _screen_obliquity_pupil_radius(prescription, Nx, Ny, dx, dy)
        _est = _screen_obliquity_rms_waves(
            _obl_total, X, Y, _r_pup, wavelength, xp)
        _budget = _est * (_SCREEN_OBLIQUITY_RESIDUAL_FRAC if _obl_apply
                          else 1.0)
        if _budget > _SCREEN_OBLIQUITY_TOL_WAVES:
            _how = ('applied (the sag-obliquity term AND the R1 drift term), '
                    'but its own next-order residual (the DEFLECTION channel '
                    'proper, which is not the gradient of any scalar and so '
                    'no screen can carry) is budgeted at %.1f%% of that'
                    % (100.0 * _SCREEN_OBLIQUITY_RESIDUAL_FRAC)
                    if _obl_apply else
                    'NOT applied (screen_obliquity=False), so the whole term '
                    'is in your wavefront')
            _msg = (
                f"apply_real_lens: this prescription's per-surface sag "
                f"screens are angle-blind by an estimated {_est:.4f} waves "
                f"rms (piston/tilt-free, over a {_r_pup * 1e3:.3f} mm pupil) "
                f"at the supplied carrier's local ray angles, which exceeds "
                f"the {_SCREEN_OBLIQUITY_TOL_WAVES:g}-wave tolerance; the "
                f"closed-form correction is {_how}, leaving "
                f"~{_budget:.4f} waves.  A thin screen collapses the finite "
                f"ray traverse through the sag onto one plane, so this grows "
                f"as sag * theta**2 with fast surfaces and large field "
                f"angles.  Use apply_real_lens_traced (per-pixel ray-traced "
                f"OPL, carrier-aware) if that is outside your OPD budget, or "
                f"pass on_screen_obliquity='silent' to acknowledge.")
            if on_screen_obliquity == 'error':
                raise ValueError(_msg)
            import warnings
            warnings.warn(_msg, RuntimeWarning, _WARN_STACKLEVEL)

    call_progress(progress, 'apply_real_lens', 1.0, 'done')
    return E


class PreparedAnalyticLens:
    """An analytic (split-step) lens with its input-independent per-surface
    phase screens precomputed (A-P1).

    Built by :func:`prepare_real_lens`.  Each per-surface OPD screen
    ``exp(-i k0 (n2-n1) sag(h))`` and the entrance-aperture mask depend only on
    ``(prescription, wavelength, dx, dy, N)``, not on the input field, yet
    :func:`apply_real_lens` recomputes them (sag + OPD + ``exp``) on every
    call.  This caches them once; each call is then the FFT propagation legs
    (whose ASM transfer functions are already cached inside
    ``angular_spectrum_propagate``) plus one complex multiply per surface.
    Biggest effect on many-surface prescriptions and optimizer / tolerancing
    loops.  Mirrors the ``PreparedRCWA2D`` / ``PreparedTracedLens`` precedent.

    Supports only the DEFAULT propagation path -- NumPy backend, plain conic +
    aspheric refractive surfaces.  The factory raises ``NotImplementedError``
    for decentred / tilted / freeform / biconic / stop / mirror surfaces or the
    slant / fresnel / absorption / seidel / surface-frame / GPU modes; use
    :func:`apply_real_lens` directly for those.

    A prepared lens FREEZES the settings that were live when it was
    prepared (audit E-H3).  ``wave_propagator``, ``sag_dtype`` and ``_dy``
    hold the values :func:`prepare_real_lens` resolved from the
    process-wide defaults (:func:`set_default_wave_propagator` /
    :func:`set_lens_sag_dtype` / :func:`set_default_dy`), so a prepared
    object keeps reproducing the field it was built for even if a global
    default is flipped afterwards -- rebuild it to pick up new settings.
    Hard-coding ASM / ``dy = dx`` / float64 geometry here instead would
    diverge from :func:`apply_real_lens` by 49.6 on a singlet, with no
    diagnostic, after a ``set_default_wave_propagator('fresnel')``.

    BYTE-IDENTITY WITH :func:`apply_real_lens`, and its one exception.  On a
    build WITHOUT numexpr this class reproduces :func:`apply_real_lens` bit for
    bit at complex128 and complex64 (verified).  WITH numexpr installed,
    :func:`apply_real_lens` routes its phase screen through numexpr once
    ``E.size >= 2**20`` (N >= 1024 square); numexpr evaluates at complex128
    internally and narrows only at the ``out=`` store, while this class always
    takes the cast-then-multiply route (``sc = screen.astype(E.dtype);
    E = E * sc``).  For a complex64 field the two then differ by about one
    float32 ULP -- emulated exactly at max relative 1.05e-07, rms 2.4e-08.
    complex128 is unaffected, and so is any grid below the numexpr gate.  The
    difference is environment-dependent (whether numexpr is importable), which
    is why it is stated here rather than left to be discovered.
    """

    __slots__ = ('_screens', '_entrance_mask', '_gap', '_N', '_dx', '_dy',
                 '_bandlimit', 'wave_propagator', 'sag_dtype')

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __call__(self, E_in: np.ndarray) -> np.ndarray:
        """Apply the prepared analytic lens to ``E_in`` (shape must be N x N)."""
        E_in = np.asarray(E_in)
        if E_in.shape != (self._N, self._N):
            raise ValueError(
                f"PreparedAnalyticLens: E_in shape {E_in.shape} != prepared "
                f"grid ({self._N}, {self._N}).")
        # Match apply_real_lens ingestion exactly.
        if np.iscomplexobj(E_in):
            E = E_in.copy()
        else:
            from ..propagators.propagation import DEFAULT_COMPLEX_DTYPE
            E = E_in.astype(DEFAULT_COMPLEX_DTYPE)
        if self._entrance_mask is not None:
            E = np.where(self._entrance_mask, E, E.dtype.type(0))
        n_surf = len(self._screens)
        for i, screen in enumerate(self._screens):
            sc = screen if screen.dtype == E.dtype else screen.astype(E.dtype)
            E = E * sc
            if i < n_surf - 1:
                thick, lam_med = self._gap[i]
                # Dispatch on the propagator FROZEN at prepare time via
                # the same helper apply_real_lens uses, so the two agree
                # for every propagator.  ``lam_med`` is already the
                # in-medium wavelength, so the helper's ``wavelength /
                # n_medium_r`` reduces to it with ``n_medium_r=1.0``;
                # absorption is off here (the factory rejects it), which
                # makes the ``kappa`` / ``k0`` args inert.
                E = _propagate_through_glass(
                    E, thick, lam_med, 1.0, 0.0, self._dx, self._dy,
                    self._bandlimit, self.wave_propagator, False, 0.0, np)
        return E


def prepare_real_lens(
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    N: int,
    dy: Optional[float] = None,
    bandlimit: bool = True,
    wave_propagator: Optional[str] = None,
    sag_dtype: Optional[Any] = None,
) -> PreparedAnalyticLens:
    """Precompute the input-independent screens of an analytic lens (A-P1).

    Returns a :class:`PreparedAnalyticLens` whose per-surface phase screens and
    entrance-aperture mask are cached, so every subsequent ``prepared(E_in)``
    costs only the FFT legs + one complex multiply per surface (the sag / OPD /
    ``exp`` recompute that :func:`apply_real_lens` does per call is paid once).

    Only the plain-conic-aspheric path is supported; see
    :class:`PreparedAnalyticLens` for the unsupported cases (which raise here).

    **A prepared object freezes the settings that were live when it was
    prepared** (v5.29.1; audit E-H3).  ``wave_propagator``, ``dy`` and the
    geometry ``sag_dtype`` are resolved HERE against the process-wide defaults
    (:func:`set_default_wave_propagator` / :func:`set_default_dy` /
    :func:`set_lens_sag_dtype`) unless passed explicitly, and the resolved
    values are stored on the returned object (``prepared.wave_propagator`` /
    ``prepared.sag_dtype``).  Flipping a global afterwards therefore leaves the
    prepared lens unchanged -- and its output equal to
    :func:`apply_real_lens` called with the PREPARE-time settings; rebuild it
    to adopt new defaults.
    """
    # Read the fold key exactly as apply_real_lens does, so the two entry
    # points diagnose (and accept) a mirror-in-``surfaces`` identically.
    prescription = _unfold_mirror_surfaces(prescription, 'prepare_real_lens')
    surfaces = prescription['surfaces']
    thicknesses = prescription['thicknesses']
    aperture = prescription.get('aperture_diameter')
    stop_index = prescription.get('stop_index')
    if len(thicknesses) != len(surfaces) - 1:
        raise ValueError(
            f"prepare_real_lens: need {len(surfaces) - 1} thicknesses for "
            f"{len(surfaces)} surfaces, got {len(thicknesses)}.")
    # Read the key exactly as apply_real_lens does (so a malformed or
    # out-of-range stop is diagnosed identically at both entry points), then
    # refuse the well-formed-but-unsupported case.
    stop_index = _normalise_stop_index(stop_index, len(surfaces),
                                       fn_name='prepare_real_lens')
    if stop_index is not None:
        raise NotImplementedError(
            "prepare_real_lens: a decentred / mid-train stop (stop_index) is "
            "not supported; call apply_real_lens directly.")
    for i, surf in enumerate(surfaces):
        for _k in ('decenter', 'tilt'):
            _v = surf.get(_k)
            if _v is not None and tuple(_v) != (0.0, 0.0):
                raise NotImplementedError(
                    f"prepare_real_lens: surfaces[{i}].{_k}={_v} is not "
                    f"supported; call apply_real_lens directly.")
        for _k in ('form_error', 'radius_y', 'freeform_type', 'clear_aperture'):
            if surf.get(_k) is not None:
                raise NotImplementedError(
                    f"prepare_real_lens: surfaces[{i}].{_k} is not supported; "
                    f"call apply_real_lens directly.")
        # (A mirror surface can no longer be seen here: it is either refused
        # or replaced by its index-neutral unfolded equivalent above, by the
        # same ``allow_unfolded_equivalent`` key apply_real_lens honours.)

    N = int(N)
    # Resolve the process-wide defaults AT PREPARE TIME (explicit kwargs
    # win, exactly as in apply_real_lens) and freeze the resolved values
    # on the returned object.  See
    # ``docs/history/lumenairy.elements._lens_real.md``
    # for what hard-coding them here cost.
    if wave_propagator is None:
        from ..propagators.propagation import get_default_wave_propagator
        wave_propagator = get_default_wave_propagator()
    if wave_propagator not in _VALID_WAVE_PROPAGATORS:
        raise ValueError(
            f"prepare_real_lens: unknown wave_propagator "
            f"{wave_propagator!r}.  Valid choices: "
            f"{sorted(set(_VALID_WAVE_PROPAGATORS))}.")
    if dy is None:
        from ..propagators.propagation import get_default_dy
        dy = get_default_dy()
        if dy is None:
            dy = dx
    _sag_real = _resolve_sag_real(sag_dtype, 'prepare_real_lens')
    k0 = 2.0 * np.pi / wavelength
    # Grid -- matches apply_real_lens exactly (same dtype pin, float division,
    # meshgrid(x, y)).
    x = ((np.arange(N, dtype=_sag_real) - N / 2) * dx).astype(_sag_real,
                                                              copy=False)
    y = ((np.arange(N, dtype=_sag_real) - N / 2) * dy).astype(_sag_real,
                                                              copy=False)
    X, Y = np.meshgrid(x, y)
    h_sq_axis = X ** 2 + Y ** 2

    entrance_mask = None
    if aperture is not None:          # stop_index is None here (rejected above)
        entrance_mask = h_sq_axis <= (aperture / 2) ** 2

    screens = []
    gap = []
    n_surf = len(surfaces)
    for i, surf in enumerate(surfaces):
        R = surf['radius']
        kc = surf.get('conic', 0.0)
        asph = surf.get('aspheric_coeffs')
        # PYTHON floats, matching apply_real_lens (which reads ``n1c.real`` off
        # a ``complex(...)``).  This matters for ``sag_dtype=np.float32``: a
        # numpy-scalar index would promote ``(n2r - n1r) * sag`` back to
        # float64 under NEP 50 weak-scalar rules, so the prepared screen would
        # be computed at higher precision than the apply_real_lens screen it
        # must reproduce (measured 5.1e-6 field divergence before this cast).
        n1r = float(get_glass_index(surf['glass_before'], wavelength))
        n2r = float(get_glass_index(surf['glass_after'], wavelength))
        sag = _surface_sag_general(h_sq_axis, R, kc, asph)
        opd = (n2r - n1r) * sag
        if bool(np.any(np.isnan(opd))):
            opd = np.where(np.isnan(opd), 0.0, opd)
        screens.append(np.exp(-1j * k0 * opd))    # complex128 screen
        if i < n_surf - 1:
            gap.append((thicknesses[i], wavelength / n2r))  # z, in-medium lambda

    return PreparedAnalyticLens(
        _screens=screens, _entrance_mask=entrance_mask, _gap=gap, _N=N,
        _dx=dx, _dy=dy, _bandlimit=bandlimit,
        wave_propagator=wave_propagator, sag_dtype=_sag_real)


__all__ = ['apply_real_lens', 'prepare_real_lens', 'PreparedAnalyticLens']
