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
    ``n2*sag/cos(theta_t) - n1*sag/cos(theta_i)`` formula -- helpful in
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
      ``n2*sag/cos(theta_t) - n1*sag/cos(theta_i)``, which is accurate at
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
        incidence: ``n2*sag/cos(theta_t) - n1*sag/cos(theta_i)``.  Off
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
            and not surface_frame
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

        # ---- Row-band (chunked) slant/fresnel refraction -------------
        # v5.17.x: evaluate the ENTIRE refraction pipeline (local normal
        # -> cos_ti/cos_tt -> refraction OPD -> phase screen -> fresnel
        # amplitude -> TIR mask) in row-bands when slant_correction /
        # fresnel is on AND row-banding is requested, so only a
        # (chunk_rows x Nx) float64 slice is live at once instead of the
        # ~6 full-grid transients the whole-grid block below builds at
        # once (~51 GB at N=32768, which blocks 32k multi-emitter runs).
        # The y-gradient is taken on a 1-row-halo band so np.gradient's
        # central differences match the whole-grid result bit-for-bit,
        # and the numexpr phase-screen decision reuses the SAME whole-
        # E.size gate as the whole-grid path (line ~1250) so the numexpr-
        # vs-numpy choice -- and hence the last-bit result -- is identical
        # on both paths.  The fresnel amplitude multiply promotes E's
        # dtype exactly as the whole-grid ``E = E * sqrt(T_eff)`` rebinding
        # does (a float64-geometry T_eff turns a complex64 field into
        # complex128); we reproduce that by routing the fresnel/TIR band
        # writes into a promoted-dtype output array and rebinding E to it
        # after the loop.  Byte-identical to the whole-grid block below
        # (test_slant_chunk_byte_identical).
        _slant_banded = ((slant_correction or fresnel)
                         and sag_chunk_rows is not None
                         and int(sag_chunk_rows) > 0 and xp is np)
        _refr_clamped = [False]

        def _refract_band(r0, r1, e_out):
            if fresnel or slant_correction:
                # 1-row halo so central-difference gradients on the band
                # match the whole-grid np.gradient result exactly; the
                # true array edges (rows 0 and Ny-1) keep their one-sided
                # stencil in the first / last band.
                _h0 = max(0, r0 - 1)
                _h1 = min(Ny, r1 + 1)
                _dsag_dy_h, _dsag_dx_h = xp.gradient(sag[_h0:_h1], dy, dx)
                _lo = r0 - _h0
                _hi = _lo + (r1 - r0)
                dsag_dx = _dsag_dx_h[_lo:_hi]
                dsag_dy = _dsag_dy_h[_lo:_hi]
                grad_sq = dsag_dx ** 2 + dsag_dy ** 2
                one_plus_g = 1.0 + grad_sq
                cos_ti = 1.0 / xp.sqrt(one_plus_g)
                sin2_ti = grad_sq / one_plus_g
                sin2_tt = (n1r / n2r) ** 2 * sin2_ti
                cos_tt = xp.sqrt(xp.maximum(1.0 - sin2_tt, 0.0))
                if (bool(xp.any(cos_ti < 1e-3))
                        or bool(xp.any(cos_tt < 1e-3))):
                    _refr_clamped[0] = True
                cos_ti_safe = xp.maximum(cos_ti, 1e-3)
                cos_tt_safe = xp.maximum(cos_tt, 1e-3)
            sag_b = sag[r0:r1]
            if slant_correction:
                opd = n2r * sag_b / cos_tt_safe - n1r * sag_b / cos_ti_safe
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
                # Writes the phase screen IN PLACE at E's dtype (complex64
                # store / complex128 internal), matching the whole grid.
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
            # matching the whole-grid ``E = E * sqrt(T_eff)`` rebinding;
            # the promoted band is stored into ``e_out`` (allocated at that
            # dtype by the caller).
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
            if fresnel or slant_correction:
                _band = xp.where(sin2_tt < 1.0, _band,
                                 xp.zeros((), dtype=_band.dtype))
            e_out[r0:r1] = _band

        if _slant_banded:
            cr = int(sag_chunk_rows)
            # The whole-grid fresnel amplitude REBINDS E to
            # result_type(E.dtype, geometry-real) (complex64 -> complex128
            # for the default float64 geometry).  Reproduce that promotion
            # by routing the fresnel/TIR band writes into a promoted output
            # array and rebinding E to it after the loop.  With no fresnel
            # (or when E is already wide enough) the output IS E and the
            # writes land in place.  The phase screen always writes the
            # pre-fresnel dtype into E first, so the promotion happens at
            # exactly the same pipeline step as the whole grid.
            if fresnel:
                _out_dtype = xp.result_type(E.dtype, _sag_real)
                E_out = (E if _out_dtype == E.dtype
                         else xp.empty(E.shape, dtype=_out_dtype))
            else:
                E_out = E
            for r0 in range(0, Ny, cr):
                _refract_band(r0, min(Ny, r0 + cr), E_out)
            E = E_out
            if _refr_clamped[0]:
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

        if not _slant_banded:
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
                # the resulting OPL = n * sag / cos_tt_safe blows up by
                # ~1000× per clamped pixel.  See round-2 audit M-LR.
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
            if slant_correction:
                opd = n2r * sag / cos_tt_safe - n1r * sag / cos_ti_safe
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
