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
import threading as _threading
from typing import Any, Dict, Optional

import numpy as np

# Optional CuPy backend (lazy).
CUPY_AVAILABLE = _importlib_util.find_spec('cupy') is not None
cp = None  # populated by _ensure_cupy_loaded() on first use


def _ensure_cupy_loaded():
    global cp
    if cp is None and CUPY_AVAILABLE:
        import cupy as _c
        cp = _c
    return cp is not None


def _is_cupy_array(x):
    if not CUPY_AVAILABLE:
        return False
    if cp is None and not _ensure_cupy_loaded():
        return False
    return isinstance(x, cp.ndarray)


# Optional numexpr fused-expression backend (lazy).
NUMEXPR_AVAILABLE = _importlib_util.find_spec('numexpr') is not None
_ne = None


def _ensure_numexpr_loaded():
    global _ne
    if _ne is None and NUMEXPR_AVAILABLE:
        import numexpr as _n
        _ne = _n
    return _ne is not None


_NUMEXPR_MIN_SIZE = 1 << 20  # see lenses.py for rationale; sync if changed


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


def _resolve_sag_real(sag_dtype: Any) -> Any:
    """Resolve the effective REAL geometry dtype: explicit kwarg wins, else
    the process global, else float64."""
    d = sag_dtype if sag_dtype is not None else _LENS_SAG_DTYPE
    if d is not None and np.dtype(d) == np.dtype(np.float32):
        return np.float32
    return np.float64


# Row-band (chunked) lens mode auto-default (v5.17.0).  The banded path is
# BYTE-IDENTICAL to the whole-grid path and wall-clock neutral, so it is ON
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
                               max_field_rel_error: float = 1e-3
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

    Returns
    -------
    dict
        ``{'max_opd_error_waves', 'rms_opd_error_waves', 'max_opd_error_nm',
        'max_field_rel_error', 'aperture_m', 'ok'}``.
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

    return {
        'max_opd_error_waves': max_waves,
        'rms_opd_error_waves': float(np.sqrt(np.mean(d ** 2)) / wavelength),
        'max_opd_error_nm': float(d.max() * 1e9),
        'max_field_rel_error': field_rel,
        'aperture_m': float(ap),
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
                t = t - g / dgdt
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
    _, grad_fn, _ = _compute_carrier(conjugate, E_in, wavelength, dx, Xg, Yg)

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


def _displaced_geom_key(surfaces, thicknesses, wavelength, r_max, conjugate):
    """Hashable identity of the FIELD-INDEPENDENT displaced fan (surfaces +
    thicknesses + wavelength + fan extent + scalar conjugate).  Only the
    collimated (``conjugate is None``) and scalar-conjugate congruences are
    cacheable; returns ``None`` for the field-dependent 'auto'/ndarray cases."""
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
        str(s.get('glass_before')), str(s.get('glass_after')))
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
    _, grad_fn, _ = _compute_carrier(conjugate, E_in, wavelength, dx, Xg, Yg)

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
                              n_coarse=384):
    """Pointwise 2-D generalisation of :func:`_build_displaced_cos_luts` for
    decentered / tilted / freeform (callable-sag) elements.

    Trace a 2-D ray grid (regular disk of radius ``r_max``, launched along the
    input congruence ``dir_fn``) through the actual surfaces and return, per
    surface, ``(cos_in_field, cos_out_field)`` -- the z-components of the ray
    direction just before / after refraction, interpolated onto the FIELD grid
    at each ray's crossing position.  These are the SAME cosines the meridional
    LUT stores, so equation (1) OPD ``(n2 cos_out - n1 cos_in) * sag`` on a
    rotationally-symmetric element reproduces the LUT path.  Vectorised Newton
    intersection + vector Snell; wave-model-independent.
    """
    from scipy.interpolate import (
        LinearNDInterpolator,
        NearestNDInterpolator,
        RegularGridInterpolator,
    )
    ax = np.linspace(-r_max, r_max, int(n_launch))
    LX, LY = np.meshgrid(ax, ax)
    disk = (LX * LX + LY * LY) <= (r_max * 1.0000001) ** 2
    x0 = LX[disk].astype(np.float64)
    y0 = LY[disk].astype(np.float64)
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
    # the scattered->grid interpolation is done on a COARSE regular grid
    # (bounded resolution) and bilinearly upsampled to the full field grid --
    # decoupling the cost from N (a full-N scattered query is ~10 us/point).
    # ``n_coarse`` samples over the field extent resolve the aperture-scale cos
    # variation to well under the model's obliquity tolerance.
    _ncx = min(Nx, n_coarse)
    _ncy = min(Ny, n_coarse)
    xcoarse = np.linspace(xax[0], xax[-1], _ncx)
    ycoarse = np.linspace(yax[0], yax[-1], _ncy)
    Xc, Yc = np.meshgrid(xcoarse, ycoarse)
    Xg, Yg = np.meshgrid(xax, yax)

    idx = []
    for s in surfaces:
        idx.append((float(get_glass_index(s['glass_before'], wavelength)),
                    float(get_glass_index(s['glass_after'], wavelength))))

    def _interp2(pts, cin, cout):
        """Interpolate BOTH cosines from the scattered ray crossings onto the
        field grid.  ONE Delaunay (LinearNDInterpolator with a 2-column value
        array) is queried on the COARSE grid; the out-of-hull NaN points are
        nearest-filled; then the coarse cos maps are bilinearly upsampled to the
        full field grid.  Both cosines share the triangulation and the coarse
        query, so the cost is set by ``n_coarse`` (not N)."""
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
                t = t - g / dgdt
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

        m = alive
        pts = np.column_stack([px[m], py[m]])
        ci, co = _interp2(pts, cos_in_z[m], cos_out_z[m])
        cos_grids.append((ci, co))

        if i < n_surf - 1:
            z_v += thicknesses[i]
    return cos_grids


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
    _, _, w_fn = _compute_carrier(conjugate, E_in, wavelength, dx, Xg, Yg)

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
    _, grad_fn, w_fn = _compute_carrier(conjugate, E_in, wavelength, dx, Xg, Yg)

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
                t = t - gg / dgdt
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
    opl = opl + 1.0 * t_f                       # exit gap is air (n = 1)
    h_out = py + t_f * dy
    m = alive & np.isfinite(h_out) & np.isfinite(opl) & np.isfinite(heights)
    return heights[m], h_out[m], opl[m]


def _apply_displaced_remap(E_in, h_in, h_out, wavelength, dx, dy, opl):
    """Candidate (a) exit-plane remap: turn the element into a geometric
    transfer ``h_in -> h_out`` with an energy-conserving amplitude Jacobian plus
    the exit-pupil-referenced eikonal OPD.

    Captures the transverse ray walk THROUGH the element (``h_out != h_in``)
    that a single fixed-plane screen cannot.  The input amplitude envelope
    ``|E_in|`` is warped from the entrance radius ``h_in`` to the exit radius
    ``h_out``; the exit phase is rebuilt from the ray eikonal ``k0 * opl`` (which
    carries the entrance-plane carrier eikonal).  Energy conservation:
    ``|E_out|^2 r_out dr_out = |E_in|^2 h_in dh_in``.  Rotationally symmetric
    (meridional-fan) model; assumes the input phase matches the specified
    congruence.  Returns the exit-vertex-plane field (same reference as the
    default screen path)."""
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
    amp = map_coordinates(np.abs(E_in), [cy, cx], order=1,
                          mode='constant', cval=0.0)
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

def _build_displaced_ray_map_2d(surfaces, thicknesses, wavelength, r_max,
                                n_side=181, dir_fn=None, eik_fn=None,
                                r_fan_factor=1.03):
    """Pointwise 2-D generalisation of :func:`_build_displaced_ray_map` (the P2
    remap) for decentered / tilted / freeform elements (niche N11 / P10).

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
                t = t - g / dgdt
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
    opl = opl + 1.0 * t_f                       # exit gap is air (n = 1)
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


def _apply_displaced_remap_2d(E_in, ray_map_2d, wavelength, dx, dy):
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
    ``k0 * OPL`` (which carries the entrance-plane carrier eikonal).  Amplitude
    and OPL are interpolated SEPARATELY from the scattered exit points onto the
    field grid (phase-safe: the eikonal is smooth even where the amplitude is
    warped), then combined.  Returns the exit-vertex-plane field (same reference
    plane as the default screen path)."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.ndimage import map_coordinates
    X0, Y0, XO, YO, OPL, ALIVE, dstep, r_ap = ray_map_2d
    Ny, Nx = E_in.shape
    k0 = 2.0 * np.pi / wavelength
    # Input amplitude envelope at each ray's entrance position (bilinear).
    cx = X0.ravel() / dx + Nx / 2.0
    cy = Y0.ravel() / dy + Ny / 2.0
    amp_in = map_coordinates(np.abs(E_in), [cy, cx], order=1,
                             mode='constant', cval=0.0).reshape(X0.shape)
    # Forward Jacobian det d(x_out,y_out)/d(x_in,y_in) on the regular launch grid
    # (physical spacing).  Fill any dead-ray (TIR / miss) exit position by
    # nearest-alive FIRST so a dead ray does not poison a live neighbour's
    # central-difference derivative.
    XOf = XO.copy()
    YOf = YO.copy()
    dead = ~ALIVE
    if bool(dead.any()) and bool(ALIVE.any()):
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
    # pre-apertured -- would leak the beyond-aperture ring).
    in_ap = (X0 * X0 + Y0 * Y0) <= (r_ap * (1.0 + 1e-9)) ** 2
    m = ALIVE & in_ap & np.isfinite(amp_out) & (amp_in > 0.0)
    if int(m.sum()) < 4:
        return np.zeros_like(E_in, dtype=np.complex128)
    pts = np.column_stack([XO[m].ravel(), YO[m].ravel()])
    x = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
    y = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
    Xg, Yg = np.meshgrid(x, y)
    # K3 (N15 perf): the amplitude and OPL remaps share ONE Delaunay
    # triangulation of ``pts`` -- a single LinearNDInterpolator with a 2-column
    # value array does the barycentric interpolation for BOTH quantities in one
    # pass.  The triangulation and the per-query-point barycentric weights depend
    # only on ``pts`` (identical for both columns), and each column is a separate
    # ``sum(weight_k * value_k)`` reduction, so this reproduces the two former
    # single-column interps BIT-FOR-BIT while building the Delaunay once and
    # walking the full-grid query once (measured ~1.6x on this 2-D remap).
    # ``amp_grid`` / ``opl_grid`` are strided VIEWS into the single (Ny, Nx, 2)
    # result -- no per-column dense copy -- so the peak footprint is the same one
    # 2-wide grid the two separate (Ny, Nx) grids used before, not more.  Outside
    # the hull both columns come back NaN: the amplitude is set to 0 (matching the
    # former ``fill_value=0.0``) and the OPL is nearest-filled, exactly as before.
    _opl_flat = OPL[m].ravel()
    _q = LinearNDInterpolator(
        pts, np.column_stack([amp_out[m].ravel(), _opl_flat]))(Xg, Yg)
    amp_grid = _q[..., 0]
    opl_grid = _q[..., 1]
    nan = np.isnan(opl_grid)
    if bool(nan.any()):
        opl_grid[nan] = NearestNDInterpolator(pts, _opl_flat)(
            Xg[nan], Yg[nan])
        amp_grid[nan] = 0.0
    opl_ref = float(np.median(OPL[m]))
    E_out = amp_grid * np.exp(1j * k0 * (opl_grid - opl_ref))
    out_dtype = E_in.dtype if np.iscomplexobj(E_in) else np.complex128
    return np.asarray(E_out, dtype=out_dtype)


def _propagate_through_glass(E: Any, thickness: float, wavelength: float,
                             n_medium_r: float, n_medium_kappa: float,
                             dx: float, dy: float, bandlimit: bool,
                             wave_propagator: Optional[str], absorption: bool,
                             k0: float, xp: Any) -> Any:
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
        E, dx_new, _ = scalable_angular_spectrum_propagate(
            E, thickness, lam_medium, dx)
        if abs(dx_new - dx) > dx * 1e-6:
            E, _ = resample_field(E, dx_new, dx, N_out=E.shape[-1])
    elif wave_propagator == 'fresnel':
        from ..propagators.propagation import fresnel_propagate, resample_field
        E, dx_new, _ = fresnel_propagate(E, thickness, lam_medium, dx, dy=dy)
        if abs(dx_new - dx) > dx * 1e-6:
            E, _ = resample_field(E, dx_new, dx, N_out=E.shape[-1])
    elif wave_propagator in ('rayleigh_sommerfeld', 'rs'):
        from ..propagators.propagation import rayleigh_sommerfeld_propagate
        E = rayleigh_sommerfeld_propagate(
            E, thickness, lam_medium, dx, dy=dy, bandlimit=bandlimit)
    elif wave_propagator == 'asm':
        E = angular_spectrum_propagate(
            E, thickness, lam_medium, dx, dy=dy, bandlimit=bandlimit)
    else:
        raise ValueError(
            f"apply_real_lens: unknown wave_propagator {wave_propagator!r}.  "
            f"Supported: 'asm', 'sas', 'fresnel', 'rayleigh_sommerfeld' "
            f"(alias 'rs').")
    if absorption and n_medium_kappa != 0.0:
        E = E * xp.exp(-k0 * n_medium_kappa * thickness)
    return E


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
    _check_no_silent_fold_drop(prescription, fn_name='apply_real_lens')


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


_VALID_SURFACE_MODELS = ('thin', 'displaced')


def _check_displaced_support(*, surface_model, slant_correction, fresnel,
                             seidel_correction, absorption, surface_frame,
                             use_gpu, wave_propagator, prescription,
                             conjugate=None, E_shape=None,
                             displaced_mode='screen',
                             displaced_obliquity='auto'):
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
    ``(n2-n1)*sag`` for the per-surface phase screen.  Empirically this
    gives equally good or better OPD agreement with a geometric ray
    trace as the slant-corrected formula, because the angular-spectrum
    propagation between surfaces already encodes most of the obliquity
    physics.  Pass ``slant_correction=True`` to use the generalised
    ``(n2*cos(theta_t) - n1*cos(theta_i))*sag`` formula -- helpful in
    a few specific geometries (asymmetric meniscus, very steep
    asphere) but not a universal improvement.

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

    Optional opt-in features add further physical realism:

    * ``fresnel=True`` -- multiply by s/p-averaged Fresnel amplitude
      transmission at each surface using local angle of incidence derived
      from the surface normal.  Captures wavelength/index-dependent
      throughput (~4% loss per uncoated air-glass interface) and works
      naturally with complex refractive indices.
    * ``slant_correction=True`` -- replace the paraxial OPD
      ``(n2-n1)*sag`` with the generalized thin-element OPD
      ``(n2*cos(theta_t) - n1*cos(theta_i))*sag``, which is accurate at
      larger angles of incidence (faster lenses, off-axis input).
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
      synthetic Zernike form error.

    Field-frame vs surface-frame decenter / tilt (v5.2+)
    ----------------------------------------------------
    The default ``surface_frame=False`` honours ``decenter`` / ``tilt``
    in the **field frame**: the field's ``(x, y)`` grid is shifted by
    ``decenter`` (axis-symmetric sag still evaluated at the shifted
    radius) and the tilt is approximated as a linear sag ramp
    ``tx*x + ty*y``.  This is the v3.x -> v5.1 contract, kept as the
    default so existing callers see no numerical change.

    ``surface_frame=True`` (v5.2+) instead evaluates each surface's sag
    in its own rigid-body-transformed local frame, matching the Optiland
    / Zemax treatment of a tilted / displaced asphere.  The field's
    ``(x, y)`` grid is mapped to surface-frame coordinates via the
    inverse rigid-body transform: a translation by ``-decenter`` followed
    by an inverse rotation ``R^T = Ry(-ty) @ Rx(-tx)`` (full rotation
    matrix, no small-angle linearisation), then the sag is evaluated at
    the resulting ``(x_s, y_s)``.  Phase contribution is the same
    ``-k0 * (n2 - n1) * sag(x_s, y_s)`` thin-element formula as the
    field-frame branch; only the coordinate at which sag is evaluated
    changes.

    Use ``surface_frame=True`` for off-axis aspheres, decentered
    parabolas, and any system where the surface's own coordinate frame
    differs meaningfully from the field's grid frame.  Use the default
    field-frame branch for the small-tilt / small-decenter alignment-
    tolerance regime where the linearised sag ramp is the textbook
    physics.

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

            - ``"radius"`` : float -- radius of curvature [m] (inf = flat)
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
        Apply Fresnel amplitude transmission at each surface.
    slant_correction : bool, default False
        Use the generalised thin-element OPD with local angle of
        incidence: ``(n2*cos(theta_t) - n1*cos(theta_i))*sag``.  Off
        by default because the simple paraxial formula
        ``(n2-n1)*sag`` typically gives equal or better agreement
        with geometric ray-traced OPD (see
        ``validation/real_lens_opd``).
    absorption : bool
        Apply bulk attenuation through each glass region using the
        extinction coefficient from :func:`get_glass_index_complex`.
    seidel_correction : bool, default False
        Add a "Seidel-style" radially-symmetric OPD correction at the
        exit pupil derived from a 1-D geometric ray-trace fan.  A
        polynomial is fit to the difference between the geometric
        ray OPL and the analytic thin-element OPL (``(n2-n1)*sag``),
        then applied as a radial phase screen on the way out.
        Captures ~3-5x improvement on cemented doublets at essentially
        no extra cost (~41 rays traced, one polynomial fit, one 2-D
        phase multiplication).  **Off by default** because: (a) well-
        corrected singlets already achieve sub-30 nm residual against
        the geometric ray trace via the thin-element model alone, and
        (b) the Seidel correction can inject polynomial-fit artefacts
        of order 100 nm on such systems (the analytic formula doesn't
        model the Fresnel ASM contribution exactly).  A 50 nm RMS
        correction-amplitude threshold is applied internally to skip
        the correction when the thin-element model is already good
        enough.  Recommended: turn on for ``AC254*``-class cemented
        doublets and similar multi-surface curved-interface systems;
        leave off for plano-convex singlets and similar well-behaved
        cases, or use :func:`apply_real_lens_traced` for uniformly
        high accuracy.
    seidel_poly_order : int, default 8
        Highest even power of the radial polynomial fit used for the
        Seidel correction.  Order 4 is classical spherical-aberration
        (``a*r^4``); order 8 includes 6th and 8th-order spherical
        terms; higher is rarely beneficial because the fit is limited
        by the 1-D sampling rather than by the polynomial basis.
    surface_frame : bool, default False
        v5.2+ opt-in.  When ``False`` (default), the per-surface
        ``"decenter"`` / ``"tilt"`` keys are honoured in the **field
        frame**: sag is evaluated on the field's ``(x, y)`` grid shifted
        by ``decenter`` and a linear sag ramp ``tx*x + ty*y`` is added
        for tilt.  This is the v3.x -> v5.1 contract and is preserved
        bit-for-bit when the flag is left at its default.

        When ``True``, the per-surface ``"decenter"`` / ``"tilt"`` are
        applied as a rigid-body transformation of the surface itself
        (Optiland / Zemax style).  The field's ``(x, y)`` grid is
        mapped to surface-frame coordinates via the inverse rigid-body
        transform (``-decenter`` then ``R^T = Ry(-ty) @ Rx(-tx)``
        with the full rotation matrix, no small-angle linearisation)
        and the sag is evaluated at the resulting ``(x_s, y_s)``.
        Use for off-axis aspheres / decentered parabolas where the
        sag's curvature must rotate with the surface, not just acquire
        a linear ramp.  See the "Field-frame vs surface-frame
        decenter / tilt" docstring section above for the physics.
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
        phase screen) and wall-clock neutral, while the full-grid
        coordinate / sag / OPD transients never materialise (~tens of
        GB reclaimed at N=32768).  Surfaces outside the narrow
        chunk-eligible case (decenter / tilt / form error / biconic /
        freeform / clear_aperture / stop surface / fresnel / slant /
        surface-frame, or a non-NumPy backend) fall through to the
        whole-grid path per surface.
    surface_model : {'thin', 'displaced'}, default 'thin'
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

        The wave field itself (``E_in``) already carries the input curvature in
        its phase -- ``conjugate`` ONLY informs the per-surface obliquity
        cosines; it adds no reference phase and does not modify ``E_in``.  The
        screen is field-independent given the conjugate, so the ``None`` /
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

    All arguments past ``E_in`` are keyword-only (4.7+).  The
    parameter name is ``prescription`` -- the 4.6 alias
    ``lens_prescription`` was removed in 4.7.
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper (replaces the v4.15.2 inline
    # guard).  Runs FIRST so the user gets a clear, actionable error
    # rather than a downstream AttributeError or silent wrong-axis
    # broadcast.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens')

    # v5.1.0 (default-knob resolver rollout): when ``wave_propagator``
    # is left at the default ``None``, resolve via the library-wide
    # default set by ``set_default_wave_propagator(...)``.  Explicit
    # values bypass the resolver.
    if wave_propagator is None:
        from ..propagators.propagation import get_default_wave_propagator
        wave_propagator = get_default_wave_propagator()
    # v5.1.0 (default-knob resolver rollout): same for ``dy``.
    # ``None -> get_default_dy() -> dx`` chain.
    if dy is None:
        from ..propagators.propagation import get_default_dy
        dy = get_default_dy()
        if dy is None:
            dy = dx

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
    )

    # v4.13.0 audit P1-A: explicit mirror-in-surfaces guard.  The
    # shared ``_check_no_silent_fold_drop`` only inspects the
    # prescription's ``elements`` list (the full element sequence,
    # populated by ``load_zemax_zmx``); a hand-built prescription that
    # puts a mirror directly into ``surfaces`` (via ``is_mirror=True``
    # or ``glass_after='MIRROR'``) and omits the ``elements`` key
    # slips past the shared check, and ``apply_real_lens`` would
    # silently treat the mirror as a refractor with the wrong sign.
    # The v4.13.0 L4a sweep ported this guard to the 4 sibling
    # ``apply_real_lens_*`` variants (``_traced``, ``_traced_jax``,
    # ``_maslov``, ``_maslov_jax``) but missed the parent itself;
    # this guard closes the audit P1-A gap.  Fail loudly with a
    # mirror-specific message before any sag / refraction maths
    # touches the field.
    _surfaces_list = prescription.get('surfaces') or []
    _mirror_surf_idx = []
    for _i, _s in enumerate(_surfaces_list):
        if not isinstance(_s, dict):
            continue
        _gl_after = _s.get('glass_after')
        _is_mirror = bool(_s.get('is_mirror', False)) or (
            isinstance(_gl_after, str)
            and _gl_after.upper() == 'MIRROR'
        )
        if _is_mirror:
            _mirror_surf_idx.append(_i)
    if _mirror_surf_idx:
        raise ValueError(
            f"apply_real_lens: prescription has "
            f"{len(_mirror_surf_idx)} mirror surface(s) at "
            f"indices {_mirror_surf_idx} -- apply_real_lens only "
            f"walks refracting surfaces.  Running this prescription "
            f"as-is would silently treat the mirror as a refractor "
            f"(wrong sign / wrong focusing phase) and propagate "
            f"along the unfolded-equivalent axis.  Use the "
            f"per-segment pattern for folded designs: call "
            f"lumenairy.io.split_prescription_at_mirrors(rx) to "
            f"split the prescription at each fold, then alternate "
            f"apply_real_lens (each segment) with apply_mirror "
            f"(each fold).  See Guide-Folded-Designs section "
            f"'Wave-optics through a fold'.")

    # Pre-flight grid vs prescription-aperture check.  If any surface's
    # semi-aperture exceeds the simulation grid, ASM will silently
    # truncate the field at the grid edge and lose energy that the real
    # hardware would have transmitted.  Issue a UserWarning once per
    # call site (Python's default warning filter dedups by source line).
    try:
        N_grid = int(np.shape(E_in)[0])
        _warn_if_aperture_exceeds_grid(
            prescription, N_grid, dx, source='apply_real_lens')
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

    if dy is None:
        dy = dx

    surfaces = prescription['surfaces']
    thicknesses = prescription['thicknesses']
    aperture = prescription.get('aperture_diameter')
    stop_index = prescription.get('stop_index')

    assert len(thicknesses) == len(surfaces) - 1, (
        f"Need {len(surfaces) - 1} thicknesses for {len(surfaces)} surfaces, "
        f"got {len(thicknesses)}"
    )

    Ny, Nx = E_in.shape
    k0 = 2 * np.pi / wavelength

    # surface_model='displaced': precompute the per-surface ray-angle cosine
    # LUTs from a collimated meridional fan (see the module-level derivation).
    # Bounded by the clear aperture (or the widest per-surface semi-diameter,
    # or the grid half-width), so the fan spans the illuminated pupil.
    _displaced = (surface_model == 'displaced')
    # P10 (N11): decentered / tilted / freeform (asymmetric) elements route to
    # the 2-D transverse-walk remap -- the DEFAULT (auto obliquity) and also
    # selectable via displaced_mode='remap'; an explicit
    # displaced_obliquity='pointwise' keeps the P3 single-plane obliquity SCREEN
    # (the documented walk-off-limited peer).
    _disp_asym = _displaced and _element_is_asymmetric(surfaces)
    _disp_2d_remap = _disp_asym and (
        displaced_mode == 'remap'
        or (displaced_mode == 'screen' and displaced_obliquity == 'auto'))
    _remap_mode = (_displaced and displaced_mode == 'remap'
                   and not _disp_asym)             # 1-D symmetric remap (P2)
    _split_mode = _displaced and displaced_mode == 'split'
    _disp_luts = None
    _disp_ray_map = None
    _disp_ray_map_2d = None
    _disp_cos_grid = None
    _disp_pointwise = False
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
            _disp_ray_map_2d = _build_displaced_ray_map_2d(
                surfaces, thicknesses, wavelength, _r_max,
                dir_fn=_dir2, eik_fn=_eik2)
        elif _disp_pointwise:
            # P3 (N2): 2-D pointwise obliquity SCREEN for decenter / tilt /
            # freeform (explicit displaced_obliquity='pointwise', or the
            # symmetric convention gate).  Trace a 2-D ray grid along the input
            # congruence and imprint the per-surface z-axis cosines on the field
            # grid.
            _disp_dir = _displaced_carrier_dir_fn(
                conjugate, E_in, wavelength, dx, dy, Nx, Ny)
            _disp_cos_grid = _build_displaced_cos_grid(
                surfaces, thicknesses, wavelength, _r_max, Nx, Ny, dx, dy,
                dir_fn=_disp_dir)
        elif _remap_mode:
            # P2 candidate (a): trace the full entrance->exit ray map + OPL
            # (with the entrance eikonal) for the geometric-transfer remap
            # applied just below (after the entrance aperture).
            _disp_eik = _displaced_eikonal_fn(
                conjugate, E_in, wavelength, dx, dy, Nx, Ny)
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
    # v4.13.2 (audit C-P1-4): dtype-aware zero to preserve complex64
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
            E = xp.where(h_sq_axis <= (aperture / 2) ** 2, E,
                         xp.zeros((), dtype=E.dtype))

    # P2 candidate (a): exit-plane geometric-transfer remap.  Replaces the
    # per-surface screen loop entirely -- warp the (apertured) input envelope
    # through the traced ray map h_in -> h_out with the energy Jacobian and the
    # exit-pupil-referenced eikonal OPD, then return the exit-vertex-plane field.
    if _remap_mode:
        _h_in_map, _h_out_map, _opl_map = _disp_ray_map
        E = _apply_displaced_remap(
            E, _h_in_map, _h_out_map, wavelength, dx, dy, _opl_map)
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
            E, _disp_ray_map_2d, wavelength, dx, dy)
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
        _narrow_chunk = (
            sag_chunk_rows is not None and int(sag_chunk_rows) > 0
            and xp is np and not slant_correction and not fresnel
            and not surface_frame and not _displaced
            and (surf.get('decenter') or (0.0, 0.0)) == (0.0, 0.0)
            and (surf.get('tilt') or (0.0, 0.0)) == (0.0, 0.0)
            and surf.get('form_error') is None
            and surf.get('radius_y') is None
            # v5.17.1 (audit P2-04): ANY freeform_type falls through to the
            # whole-grid path -- Q-bfs / Q-con so their departure IS
            # computed there, and the non-Q types (zernike / xy_polynomial
            # / chebyshev) so the whole-grid path's "freeform departure is
            # NOT included" RuntimeWarning keeps firing on the (default)
            # banded path.  Pre-fix the band loop silently dropped the
            # departure for non-Q types with no diagnostic.  Outputs are
            # unchanged (the departure was dropped on both paths).
            and surf.get('freeform_type') is None
            and surf.get('clear_aperture') is None
            and not (stop_index is not None and i == stop_index
                     and aperture is not None)
        )
        if _narrow_chunk:
            cr = int(sag_chunk_rows)
            _use_ne = (NUMEXPR_AVAILABLE and E.size >= _NUMEXPR_MIN_SIZE
                       and _ensure_numexpr_loaded())
            for r0 in range(0, Ny, cr):
                r1 = min(Ny, r0 + cr)
                _h_b = (_x_sq[None, :] + _y_sq[r0:r1, None]
                        if h_sq_axis is None else h_sq_axis[r0:r1])
                sag_b = _surface_sag_general(_h_b, R, kc, asph)
                opd_b = (n2r - n1r) * sag_b
                if bool(np.any(np.isnan(opd_b))):
                    opd_b = np.where(np.isnan(opd_b), 0.0, opd_b)
                if _use_ne:
                    Eb = E[r0:r1]
                    _ne.evaluate('Eb * exp(-1j * k0 * opd_b)',
                                 local_dict={'Eb': Eb, 'k0': k0, 'opd_b': opd_b},
                                 out=Eb)
                else:
                    ph = np.exp(-1j * k0 * opd_b)
                    if ph.dtype != E.dtype:
                        ph = ph.astype(E.dtype)
                    E[r0:r1] = E[r0:r1] * ph
                del sag_b, opd_b
            if i < len(surfaces) - 1:
                E = _propagate_through_glass(
                    E, thicknesses[i], wavelength, n2r, n2c.imag,
                    dx, dy, bandlimit, wave_propagator, absorption, k0, xp)
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
            for r0 in range(0, Ny, cr):
                r1 = min(Ny, r0 + cr)
                # 1-row halo so central-difference gradients on the band match
                # the whole-grid np.gradient result exactly; the true array
                # edges (rows 0 and Ny-1) keep their one-sided stencil in the
                # first / last band.  ``sag_halo`` built from the axis vectors
                # is byte-identical to slicing the full-grid sag
                # (_surface_sag_general is pointwise in h_sq).
                _h0 = max(0, r0 - 1)
                _h1 = min(Ny, r1 + 1)
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
                    # v5.25.0 (hammer audit H1): the wavefront OPD of a
                    # locally-tilted refracting facet is
                    # (n2*cos_tt - n1*cos_ti) * sag -- COSINES IN THE
                    # NUMERATOR (the plane-parallel-plate result).  The
                    # historical ``n*sag/cos`` form is the geometric ray
                    # path-length through a slab, NOT the wavefront OPD;
                    # it sign-flips the leading obliquity (spherical-
                    # aberration) term, and on a symmetric biconvex the
                    # wrong-signed corrections cancelled the pupil SA
                    # entirely (dual-oracle f/5 case: 3.6 um "perfect"
                    # spot vs the true 65 um).  Keep byte-identical to
                    # the whole-grid copy below.
                    opd = (n2r * cos_tt_safe
                           - n1r * cos_ti_safe) * sag_b
                else:
                    opd = (n2r - n1r) * sag_b
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
                else:
                    ph = xp.exp(-1j * k0 * opd)
                    if ph.dtype != E.dtype:
                        ph = ph.astype(E.dtype)
                    E[r0:r1] = E[r0:r1] * ph
                # Fresnel amplitude transmission.  ``E[r0:r1] * sqrt(T_eff)``
                # promotes the band to result_type(E.dtype, geometry-real)
                # (complex64 -> complex128 for the default float64 geometry),
                # matching the whole-grid ``E = E * sqrt(T_eff)`` rebinding.
                if fresnel:
                    denom_s = n1c * cos_ti_safe + n2c * cos_tt_safe
                    denom_p = n2c * cos_ti_safe + n1c * cos_tt_safe
                    t_s = 2.0 * n1c * cos_ti_safe / denom_s
                    t_p = 2.0 * n1c * cos_ti_safe / denom_p
                    T_eff = 0.5 * (xp.abs(t_s) ** 2 + xp.abs(t_p) ** 2)
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
                    RuntimeWarning, stacklevel=2,
                )
            if i < len(surfaces) - 1:
                E = _propagate_through_glass(
                    E, thicknesses[i], wavelength, n2r, n2c.imag,
                    dx, dy, bandlimit, wave_propagator, absorption, k0, xp)
            continue

        # Whole-grid path from here on -- build the deferred meshgrids on
        # first use (no-op when they already exist).
        X, Y, h_sq_axis = _ensure_full_grids()

        # ---- Decenter --------------------------------------------------
        # v5.2 (ROADMAP v5.1 off-axis conic in surface frame;
        # AUDIT_V5_1_0 deferred feature): when ``surface_frame=True``
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
            # the surface frame.  R = Rx(tx) @ Ry(ty); the inverse
            # applied to (x - dcx, y - dcy, 0) gives:
            #   x_s = cy*dx_local + sx*sy*dy_local
            #   y_s = cx*dy_local
            # (z_s is discarded -- the thin-element approximation
            # evaluates sag at the surface-frame footprint of the
            # field-plane normal, the same simplification the
            # field-frame branch makes when it skips the perpendicular-
            # foot solve.)
            tx_f = float(tilt_sf[0])
            ty_f = float(tilt_sf[1])
            cx_f = np.cos(tx_f)
            sx_f = np.sin(tx_f)
            cy_f = np.cos(ty_f)
            sy_f = np.sin(ty_f)
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
        # v4.15.1 (P3-NEW-A): Forbes Q-bfs / Q-con sag is a 2-D scalar
        # phase contribution exactly analogous to the (forthcoming)
        # xy-polynomial / Zernike / Chebyshev wave-optics paths and is
        # explicitly delegated to this module for closure -- so for
        # ``freeform_type in ('q_bfs', 'q_con')`` we compute the
        # freeform departure here and ADD it to the base conic sag.
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
                    RuntimeWarning, stacklevel=2,
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
                    RuntimeWarning, stacklevel=2,
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

        # ---- Tilt (small-angle linear ramp added to sag) --------------
        # v5.2 (ROADMAP v5.1 off-axis conic in surface frame;
        # AUDIT_V5_1_0 deferred feature): in the surface-frame branch
        # the tilt is already encoded in the rotated (Xs, Ys), so the
        # linear sag ramp is suppressed here to avoid double-counting.
        # The field-frame branch (default) keeps the historical linear
        # ramp -- this is the v3.x -> v5.1 contract.
        tilt = surf.get('tilt') or (0.0, 0.0)
        if (tilt[0] != 0.0 or tilt[1] != 0.0) and not _sf_active:
            sag = sag + tilt[0] * Xs + tilt[1] * Ys

        # ---- Form error map -------------------------------------------
        form_err = surf.get('form_error')
        if form_err is not None:
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
            # 4.10: pass dy for the y-axis spacing -- pre-4.10 used dx
            # for both, which gave the wrong surface-normal direction on
            # anamorphic grids (dx != dy).  np.gradient takes the spacing
            # in the same order as the array axes (y, x).
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
            # 4.10: warn the FIRST time per call we clamp a real ray's
            # cosine.  The 1e-3 floor (≈89.94°) was previously silent;
            # for steep aspheres or strongly tilted bundles it acts on
            # physical (non-TIR) rays before the TIR mask fires, and
            # the historical /cos slant OPD blew up ~1000x per
            # clamped pixel (round-2 audit M-LR).  v5.25.0 (H1): the
            # corrected *cos form cannot diverge, so the clamp is now
            # harmless for the OPD -- the warning is kept for the
            # Fresnel-coefficient legs, which still divide by cos.
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
                    RuntimeWarning, stacklevel=2,
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
            # v5.25.1 (hammer H2(a)): ray-angle-aware refraction OPD
            # (n2 cos_alpha_out - n1 cos_alpha_in) * sag, cosines of the
            # TRUE ray angle to the z-axis from the collimated meridional
            # fan (interpolated onto the grid radius r).  Carries the
            # incoming-ray-angle physics the paraxial/slant screens drop and
            # splits plano-convex orientation.  See the module-level
            # derivation + oracle evidence.
            r_grid = xp.sqrt(h_sq)
            opd = _displaced_opd(sag, r_grid, _disp_luts[i], n1r, n2r)
        elif slant_correction:
            # v5.25.0 (hammer audit H1): cosines in the NUMERATOR -- the
            # wavefront OPD of a tilted refracting facet, not the ray
            # slab path-length.  See the banded-copy comment above for
            # the full derivation + oracle evidence; keep byte-identical.
            opd = (n2r * cos_tt_safe - n1r * cos_ti_safe) * sag
        else:
            opd = (n2r - n1r) * sag
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
        if (xp is np and NUMEXPR_AVAILABLE
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
        else:
            # Fallback for GPU (xp is cp) or small CPU arrays: compute
            # exp() in the array backend's precision, then cast back
            # to E's dtype so we don't silently upcast a complex64
            # field to complex128.  CuPy's exp is fused at kernel
            # level so the "three temporaries" concern doesn't apply
            # the same way on device.
            phase_exp = xp.exp(-1j * k0 * opd)
            if phase_exp.dtype != E.dtype:
                phase_exp = phase_exp.astype(E.dtype)
            E = E * phase_exp

        # ---- Fresnel amplitude transmission ---------------------------
        if fresnel:
            # 4.10: average the INTENSITY coefficients for unpolarised
            # scalar throughput, not the amplitude coefficients.  At
            # Brewster's angle (or any high AOI), t_s and t_p have
            # different phases; their amplitude sum can cancel where
            # sqrt(0.5*(|t_s|^2+|t_p|^2)) correctly captures the
            # incoherent average power.  Pre-4.10 used 0.5*(t_s+t_p)
            # which only matches 45-deg linear polarisation at low AOI.
            # For polarised inputs route through the Jones pipeline.
            denom_s = n1c * cos_ti_safe + n2c * cos_tt_safe
            denom_p = n2c * cos_ti_safe + n1c * cos_tt_safe
            t_s = 2.0 * n1c * cos_ti_safe / denom_s
            t_p = 2.0 * n1c * cos_ti_safe / denom_p
            T_eff = 0.5 * (xp.abs(t_s) ** 2 + xp.abs(t_p) ** 2)
            E = E * xp.sqrt(T_eff)

        # ---- TIR mask (audit #3.5: was inside `if fresnel:` pre-4.9) --
        # Suppress regions that went into total internal reflection.
        # This must fire whenever ``sin2_tt`` was computed -- i.e. for
        # both ``fresnel=True`` and ``slant_correction=True`` paths,
        # since the slant OPD divides by ``cos_tt_safe`` which is
        # ill-defined where ``sin2_tt > 1``.  Pre-4.9 only ran this
        # inside the Fresnel block, leaving slant_correction=True +
        # fresnel=False users with unphysical residual field amplitude
        # in TIR regions.
        if fresnel or slant_correction:
            # v4.13.2 (audit C-P1-4): dtype-aware zero.
            E = xp.where(sin2_tt < 1.0, E, xp.zeros((), dtype=E.dtype))

        # ---- Per-surface clear aperture (vignetting) ------------------
        clear_ap = surf.get('clear_aperture')
        if clear_ap is not None:
            # v4.13.2 (audit C-P1-4): dtype-aware zero.
            E = xp.where(h_sq <= (clear_ap / 2) ** 2, E,
                         xp.zeros((), dtype=E.dtype))

        # ---- Aperture stop applied at this surface --------------------
        if stop_index is not None and i == stop_index and aperture is not None:
            # 4.10.2: respect the stop surface's decenter/displacement
            # if any.  Pre-4.10.2 always used h_sq_axis (centred at the
            # optical axis), so a decentered stop was modelled at the
            # wrong location and clipped the wrong region of the beam.
            # 4.11.1: the 4.10.2 patch used ``getattr(surf, ...)`` on a
            # dict (always returns the default 0.0), and looked up the
            # wrong keys (``decenter_x_m`` / ``decenter_y_m``).  The
            # surface dict's actual key is ``decenter`` and the value
            # is a ``(dx, dy)`` tuple -- mirror line 520 above.
            # v4.13.2 (audit C-P1-4): dtype-aware zero for both branches.
            _dec = surf.get('decenter') or (0.0, 0.0)
            xc_stop = float(_dec[0])
            yc_stop = float(_dec[1])
            if xc_stop == 0.0 and yc_stop == 0.0:
                E = xp.where(h_sq_axis <= (aperture / 2) ** 2,
                             E, xp.zeros((), dtype=E.dtype))
            else:
                h_sq_stop = (X - xc_stop) ** 2 + (Y - yc_stop) ** 2
                E = xp.where(h_sq_stop <= (aperture / 2) ** 2,
                             E, xp.zeros((), dtype=E.dtype))

        # ---- Propagate through glass to the next surface --------------
        # Dispatch (asm / sas / fresnel / rayleigh_sommerfeld) + bulk
        # absorption is factored into ``_propagate_through_glass`` so the
        # row-band (chunked) path above reuses the identical implementation.
        if i < len(surfaces) - 1:
            if _split_mode:
                # P2 candidate (b): the internal gap is propagated as the
                # REDUCED distance ``t / n`` in air (lambda) rather than the
                # physical ``t`` at ``lambda / n`` -- the paraxial-equivalent
                # thin-lens factorisation (entrance screen + t/n + exit screen).
                E = _propagate_through_glass(
                    E, thicknesses[i] / n2r, wavelength, 1.0, 0.0,
                    dx, dy, bandlimit, wave_propagator, False, k0, xp)
            else:
                E = _propagate_through_glass(
                    E, thicknesses[i], wavelength, n2r, n2c.imag,
                    dx, dy, bandlimit, wave_propagator, absorption, k0, xp)

    # ----- Seidel correction ------------------------------------------
    # Apply a ray-trace-derived radial phase correction that captures
    # the residual OPD the thin-element model misses.  This is a
    # generalised "Seidel"-style correction: we ray-trace a 1-D fan,
    # take the difference between the geometric OPL and the analytic
    # thin-element OPL at each height, fit a radial even polynomial,
    # and apply that as an additional phase screen at the exit pupil.
    #
    # Captures all orders of spherical aberration up to
    # ``seidel_poly_order``, plus any residual caused by the uniform-
    # slab approximation at each interface.  For rotationally
    # symmetric on-axis collimated input the correction is radially
    # symmetric and applied per pixel via r = sqrt(x^2 + y^2).
    if seidel_correction and aperture is not None:
        # Local imports to avoid circular dep at module load
        from ..raytrace import (
            _make_bundle as _rt_make_bundle,
        )
        from ..raytrace import (
            surfaces_from_prescription as _rt_surfaces_from_prescription,
        )
        from ..raytrace import (
            trace as _rt_trace,
        )
        r_pupil = 0.5 * aperture
        n_fan = 41
        h_fan = np.linspace(-0.9 * r_pupil, 0.9 * r_pupil, n_fan)
        z_arr = np.zeros_like(h_fan)
        fan = _rt_make_bundle(
            x=h_fan, y=z_arr, L=z_arr, M=z_arr,
            wavelength=wavelength)
        surfs_fan = _rt_surfaces_from_prescription(prescription)
        res_fan = _rt_trace(fan, surfs_fan, wavelength)
        final_fan = res_fan.image_rays
        alive_fan = final_fan.alive
        if alive_fan.sum() >= 5:
            opl_ray = final_fan.opd[alive_fan]
            h_alive = h_fan[alive_fan]
            # Analytic height-dependent OPD deposited by the phase-
            # screens above: sum over surfaces of (n2-n1)*sag_i(h).
            # With the sign convention ``phase_screen = exp(-i*k*opd)``
            # this DECREASES the wave's OPL at edges for a positive
            # lens -- the wave's height-dependent OPL is therefore
            # ``-sum (n2-n1)*sag``.  The full wave output includes
            # further Fresnel ASM contributions we don't try to model
            # analytically.
            opl_analytic = np.zeros_like(h_alive)
            for surf_i in surfaces:
                R_i = surf_i['radius']
                kc_i = surf_i.get('conic', 0.0)
                asph_i = surf_i.get('aspheric_coeffs')
                R_y_i = surf_i.get('radius_y')
                n1r_i = get_glass_index(surf_i['glass_before'], wavelength)
                n2r_i = get_glass_index(surf_i['glass_after'], wavelength)
                if R_y_i is not None:
                    sag_fan_i = surface_sag_biconic(
                        h_alive, np.zeros_like(h_alive),
                        R_x=R_i, R_y=R_y_i,
                        conic_x=kc_i,
                        conic_y=surf_i.get('conic_y'),
                        aspheric_coeffs=asph_i,
                        aspheric_coeffs_y=surf_i.get('aspheric_coeffs_y'))
                else:
                    sag_fan_i = _surface_sag_general(
                        h_alive * h_alive, R_i, kc_i, asph_i)
                opl_analytic = opl_analytic + (
                    (n2r_i - n1r_i) * sag_fan_i)
            i_ax = int(np.argmin(np.abs(h_alive)))
            delta_ray = opl_ray - opl_ray[i_ax]
            # 4.11.2: REVERT THE v4.10 "C-LR-1 fix".
            # The pre-v4.10 negation here was correct.  Walking the
            # physics under the library's exp(-i*omega*t) convention
            # (forward kernel exp(+i*kz*z), phase = exp(+i*k0*OPL)):
            #   - The thin-element phase screen above is
            #     exp(-i*k0*(n2-n1)*sag) per surface.  Under
            #     phase = exp(+i*k0*OPL), this screen deposits
            #     OPL_screen = -(n2-n1)*sag at each height -- NEGATIVE
            #     at the rim of a positive lens (the wave "sees" less
            #     optical path through the thinner glass at the rim).
            #   - The geometric ray-trace delta_ray = opl_ray - opl_ray
            #     [axis] is also negative at the rim of a positive lens
            #     (rim has shorter optical path through the lens than
            #     the thicker axis).
            #   - For a paraxial lens both quantities agree at first
            #     order, leaving only the desired high-order residual.
            # The v4.10 patch dropped the negation based on the audit's
            # incorrect sign reasoning; the resulting ``correction`` was
            # approximately +2*(n-1)*sag at the rim (millimetres for a
            # 100 mm BK7 singlet), which tripled the lens's analytic OPD
            # and crashed the effective focal length by a factor of ~3.
            # v4.11.1's 50nm-->5nm gate did not help because the bogus
            # correction was mm-scale.  Restored original sign here so
            # ``correction`` is the small high-order residual the
            # polynomial fit is meant to capture, not a duplicate of
            # the lens OPD.  Round-3 audit (AUDIT_ROUND3_2026_05_16.md,
            # CRIT-1) flagged this; verified by hand against a BK7
            # plano-convex test case.
            opl_wave_rel = -(opl_analytic - opl_analytic[i_ax])
            correction = delta_ray - opl_wave_rel
            # Fit even-power polynomial in normalised pupil coord.
            rho = h_alive / r_pupil
            max_order = max(2, int(seidel_poly_order))
            even_powers = np.arange(2, max_order + 2, 2)
            A = np.column_stack([rho ** p for p in even_powers])
            coeffs, *_ = np.linalg.lstsq(A, correction, rcond=None)
            # Suppress fitting noise: if the RMS correction across the
            # fan is already well below typical simulation residual,
            # skip application to avoid injecting polynomial-fit
            # artefacts into otherwise-clean fields.
            # 4.11.1: after the C-LR-1 sign fix in 4.10 the typical
            # residual collapsed to a few-nm range, so 50 nm silently
            # skipped most real corrections.  Drop the gate to ~5 nm
            # (well below the Marechal lambda/14 ~ 35 nm at visible
            # wavelengths so meaningful corrections are still applied,
            # while ~5 nm remains above the lstsq numerical noise floor
            # for a 6th-order even-polynomial fit on ~50 fan samples).
            corr_rms = float(np.sqrt(np.mean(correction ** 2)))
            if corr_rms > 5e-9:  # > 5 nm RMS to be worth applying
                # h_sq_axis is in the array backend (xp) already;
                # coeffs came from a CPU lstsq so scalar-broadcast
                # them into xp.  Final phase screen multiplies E on
                # the target device.
                X, Y, h_sq_axis = _ensure_full_grids()
                rho_map_sq = h_sq_axis / (r_pupil ** 2)
                corr_map = xp.zeros_like(rho_map_sq)
                for p, c in zip(even_powers, coeffs):
                    corr_map = corr_map + float(c) * rho_map_sq ** (p // 2)
                corr_map = xp.where(rho_map_sq <= 1.0, corr_map, 0.0)
                E = E * xp.exp(+1j * k0 * corr_map)

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

    Supports only the DEFAULT propagation path -- NumPy backend, ASM
    propagator, plain conic + aspheric refractive surfaces.  The factory
    raises ``NotImplementedError`` for decentred / tilted / freeform / biconic
    / stop / mirror surfaces or the slant / fresnel / absorption / seidel /
    surface-frame / GPU / non-ASM modes; use :func:`apply_real_lens` directly
    for those.
    """

    __slots__ = ('_screens', '_entrance_mask', '_gap', '_N', '_dx', '_dy',
                 '_bandlimit')

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
                E = angular_spectrum_propagate(
                    E, thick, lam_med, self._dx, dy=self._dy,
                    bandlimit=self._bandlimit)
        return E


def prepare_real_lens(
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    N: int,
    dy: Optional[float] = None,
    bandlimit: bool = True,
) -> PreparedAnalyticLens:
    """Precompute the input-independent screens of an analytic lens (A-P1).

    Returns a :class:`PreparedAnalyticLens` whose per-surface phase screens and
    entrance-aperture mask are cached, so every subsequent ``prepared(E_in)``
    costs only the FFT legs + one complex multiply per surface (the sag / OPD /
    ``exp`` recompute that :func:`apply_real_lens` does per call is paid once).

    Only the default ASM / plain-conic-aspheric path is supported; see
    :class:`PreparedAnalyticLens` for the unsupported cases (which raise here).
    """
    surfaces = prescription['surfaces']
    thicknesses = prescription['thicknesses']
    aperture = prescription.get('aperture_diameter')
    stop_index = prescription.get('stop_index')
    if len(thicknesses) != len(surfaces) - 1:
        raise ValueError(
            f"prepare_real_lens: need {len(surfaces) - 1} thicknesses for "
            f"{len(surfaces)} surfaces, got {len(thicknesses)}.")
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
        if surf.get('is_mirror') or str(surf.get('glass_after', '')).upper() == 'MIRROR':
            raise NotImplementedError(
                f"prepare_real_lens: surfaces[{i}] is a mirror; call "
                f"apply_real_lens directly.")

    N = int(N)
    if dy is None:
        dy = dx
    k0 = 2.0 * np.pi / wavelength
    # Grid -- matches apply_real_lens exactly (float division, meshgrid(x, y)).
    x = (np.arange(N, dtype=np.float64) - N / 2) * dx
    y = (np.arange(N, dtype=np.float64) - N / 2) * dy
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
        n1r = get_glass_index(surf['glass_before'], wavelength)
        n2r = get_glass_index(surf['glass_after'], wavelength)
        sag = _surface_sag_general(h_sq_axis, R, kc, asph)
        opd = (n2r - n1r) * sag
        if bool(np.any(np.isnan(opd))):
            opd = np.where(np.isnan(opd), 0.0, opd)
        screens.append(np.exp(-1j * k0 * opd))    # complex128 screen
        if i < n_surf - 1:
            gap.append((thicknesses[i], wavelength / n2r))  # z, in-medium lambda

    return PreparedAnalyticLens(
        _screens=screens, _entrance_mask=entrance_mask, _gap=gap, _N=N,
        _dx=dx, _dy=dy, _bandlimit=bandlimit)


__all__ = ['apply_real_lens', 'prepare_real_lens', 'PreparedAnalyticLens']
