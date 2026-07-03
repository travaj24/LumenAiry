"""
Lens and focusing element phase screens.

This module provides phase-screen models for refractive focusing elements:
thin lenses (paraxial through aplanatic), thick singlets (spherical and
aspheric), multi-surface real lenses with split-step propagation through
glass, cylindrical lenses, axicons, and GRIN rod lenses.

All functions follow the exp(-i*omega*t) time convention and use SI meters
for spatial quantities.

Backends
--------
Most functions use NumPy by default.  ``apply_thin_lens``,
``apply_spherical_lens``, ``apply_aspheric_lens``, and
``apply_real_lens`` accept a *use_gpu* flag (or auto-dispatch when
the input is a CuPy array) and run entirely on the GPU when CuPy
is installed.  Per-surface FFT propagation, phase-screen multiply,
and aperture clipping all use the array-API ``xp.*`` namespace, so
NumPy and CuPy arrays follow the same code path.  At N=32768 with
a 16-surface prescription, the GPU path is typically 5-10x faster
than the threaded pyFFTW CPU path (NVIDIA RTX 4090 vs Ryzen 7950X).

``apply_real_lens_traced`` accepts ``use_gpu=True`` for its
Newton-inversion polynomial-fit stage; the inner ray trace and
amplitude leg can also be GPU-accelerated independently via
``amp_use_gpu=True``.

Author: Andrew Traverso
"""

from __future__ import annotations

# GPU backend ----------------------------------------------------------------
# Lazy: only check availability via find_spec, defer the actual import
# to first use.  Saves ~200 ms at lumenairy import time on machines
# with CuPy installed.
import importlib.util as _importlib_util
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

CUPY_AVAILABLE = _importlib_util.find_spec('cupy') is not None
cp = None  # populated by _ensure_cupy_loaded() on first use


def _ensure_cupy_loaded():
    global cp
    if cp is None and CUPY_AVAILABLE:
        import cupy as _c
        cp = _c
    return cp is not None


# Optional fused-expression backend ------------------------------------------
# numexpr evaluates array expressions in chunked, multi-threaded passes
# without materialising full N x N intermediates.  Used by apply_real_lens
# to fuse the ``E * exp(-1j*k0*opd)`` phase-screen multiply, which at
# N=32768 otherwise allocates three 17 GB complex128 temporaries.
NUMEXPR_AVAILABLE = _importlib_util.find_spec('numexpr') is not None
_ne = None  # populated by _ensure_numexpr_loaded() on first use


def _ensure_numexpr_loaded():
    global _ne
    if _ne is None and NUMEXPR_AVAILABLE:
        import numexpr as _n
        _ne = _n
    return _ne is not None

# Optional Numba JIT, LAZILY imported on first kernel use (audit P2-D: the eager
# ``import numba`` cost ~1.8 s of ``import lumenairy`` cold start).  Used by the
# fused polynomial-aspheric loop ``_aspheric_sag_accum_numba`` (3.2.14), which has
# a pure-NumPy fallback -- so numba is pulled in only when a caller actually hits
# that fast path AND numba is installed.  ``find_spec`` checks availability
# WITHOUT importing numba.
import importlib.util as _ilu

_NUMBA_AVAILABLE = _ilu.find_spec("numba") is not None
_numba = None                         # populated by _load_numba() on first use
_njit = None
_prange = None
_NUMBA_KERNELS: dict = {}             # kernel-name -> compiled fn (or None)


def _load_numba():
    """Import numba + njit/prange on first use; cache the handles.  Returns True
    iff numba is importable (False -> callers take the pure-NumPy fallback)."""
    global _numba, _njit, _prange
    if _numba is not None:
        return True
    if not _NUMBA_AVAILABLE:
        return False
    import numba as _nb
    from numba import njit as _nj
    from numba import prange as _pr
    _numba, _njit, _prange = _nb, _nj, _pr
    return True


# Default Newton iteration cap for the entrance->exit map inversion.
# 8 reaches sub-nm OPD residual on typical refractive lenses thanks
# to the per-pixel `active`-mask early-exit that drops converged
# pixels.  Bump to 12 explicitly via apply_real_lens_traced(
# newton_max_iters=12) for cemented multi-element systems with steep
# marginal-ray angles where the FD spline+Newton needs more passes.
# (Audit perf #5: was 12, dropped to 8 in v3.5.5; saves ~25% of the
# Newton stage which is ~30% of apply_real_lens_traced runtime.)
_NEWTON_MAX_ITERS = 8

# Minimum field size at which the numexpr phase-screen path beats the
# straight numpy multiply (overhead of expression compile + thread
# dispatch is fixed; benefit scales with array size).
_NUMEXPR_MIN_SIZE = 1 << 20  # 1 Mi elements (~1024 x 1024)




def _is_cupy_array(x):
    """
    Reliable CuPy array check.  ``hasattr(x, 'device')`` used to be a
    duck-type test for a CuPy device array but broke in NumPy 2.x
    (``ndarray`` now exposes ``.device`` via the Array API standard),
    causing every NumPy array to get routed into the CuPy branch.
    """
    if not CUPY_AVAILABLE:
        return False
    if cp is None and not _ensure_cupy_loaded():
        return False
    return isinstance(x, cp.ndarray)



# ---------------------------------------------------------------------------
# Helper: general conic + aspheric surface sag
# ---------------------------------------------------------------------------

def _get_aspheric_sag_accum_numba():
    """Compile (once, on first call) and return the fused aspheric-sag numba
    kernel, or ``None`` if numba is unavailable.  Lazy so ``import lumenairy``
    never pays the numba import / compile cost (audit P2-D)."""
    if "aspheric_sag" in _NUMBA_KERNELS:
        return _NUMBA_KERNELS["aspheric_sag"]
    if not _load_numba():
        _NUMBA_KERNELS["aspheric_sag"] = None
        return None

    @_njit(cache=True, parallel=True, fastmath=True)
    def _aspheric_sag_accum_numba(h_sq, sag, powers, coeffs):
        """In-place accumulate sum_i coeff_i * h_sq**(power_i // 2) onto
        ``sag``.  Single fused pass over h_sq, no temporary arrays.

        ``h_sq`` and ``sag`` must be contiguous float64 arrays of the
        same shape.  ``powers`` is int32, ``coeffs`` is float64; both
        1-D and same length.
        """
        flat_h = h_sq.ravel()
        flat_s = sag.ravel()
        n = flat_h.size
        n_terms = powers.size
        for i in _prange(n):
            v = flat_h[i]
            acc = 0.0
            for j in range(n_terms):
                p = powers[j] // 2
                # h_sq^p via repeated squaring keeps the inner loop
                # branch-free (Numba unrolls small fixed-power loops).
                hp = 1.0
                for _ in range(p):
                    hp *= v
                acc += coeffs[j] * hp
            flat_s[i] += acc

    _NUMBA_KERNELS["aspheric_sag"] = _aspheric_sag_accum_numba
    return _aspheric_sag_accum_numba


def surface_sag_general(
    h_sq: np.ndarray,
    R: float,
    conic: float = 0.0,
    aspheric_coeffs: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    Compute surface sag for a general conic + even-aspheric surface.

    This function is used by both the lens and mirror modules, so it is
    exported at module level (no leading underscore).

    Parameters
    ----------
    h_sq : ndarray
        Squared radial distance from the optical axis, x**2 + y**2  [m**2].
    R : float
        Radius of curvature [m].  Use ``float('inf')`` or ``np.inf`` for a
        flat surface.
    conic : float, optional
        Conic constant (default 0 = sphere, -1 = paraboloid, < -1 =
        hyperboloid, -1 < k < 0 = prolate ellipsoid, > 0 = oblate ellipsoid).
    aspheric_coeffs : dict or None, optional
        Even polynomial aspheric coefficients ``{power: coeff}``, e.g.
        ``{4: A4, 6: A6, 8: A8, 10: A10}``.  Each term contributes
        ``coeff * h_sq**(power // 2)`` to the sag.

    Returns
    -------
    sag : ndarray
        Signed surface sag (positive when R > 0).
    """
    # Array-API polymorphic: detect cupy vs numpy from the input.
    # This keeps the helper usable from both the CPU and GPU paths of
    # apply_real_lens without duplicating code.  Arithmetic
    # broadcasting (np.where, np.sqrt on cupy arrays) silently
    # converts to host, so we dispatch explicitly.
    xp = cp if _is_cupy_array(h_sq) else np
    sag = xp.zeros_like(h_sq)

    if R is not None and not np.isinf(R):
        # Conic sag: h^2 / (R * (1 + sqrt(1 - (1+k)*h^2/R^2)))
        # 4.10: outside the conic domain (norm >= 0.9999) the surface
        # is not defined.  Pre-4.10 silently returned 0 sag there,
        # which produced an apparently-flat ring at the surface edge
        # for hyperbolic / oblate conics extending past the geometric
        # rim.  Return NaN instead so downstream consumers either mask
        # those pixels (via an aperture mask) or see the failure.
        norm = (1 + conic) * h_sq / R**2
        valid = norm < 0.9999
        denom_arg = xp.where(valid, 1 - norm, 0.01)
        conic_sag = xp.where(
            valid,
            h_sq / (R * (1 + xp.sqrt(denom_arg))),
            xp.nan,
        )
        sag = conic_sag

    if aspheric_coeffs:
        # 3.2.14: fused single-pass numba kernel when available.
        # Skips the per-term temporary array allocation that the
        # legacy NumPy fallback required (5 aspheric coeffs at N=4096
        # is ~640 MB of transient memory in that path).  CuPy stays
        # on the legacy path because numba targets host arrays.
        _sag_kernel = (_get_aspheric_sag_accum_numba()
                       if xp is np and _NUMBA_AVAILABLE else None)
        if (_sag_kernel is not None
                and h_sq.dtype == np.float64
                and sag.dtype == np.float64):
            powers_arr = np.fromiter(
                (int(p) for p in aspheric_coeffs.keys()), dtype=np.int32)
            coeffs_arr = np.fromiter(
                (float(c) for c in aspheric_coeffs.values()),
                dtype=np.float64)
            # In-place accumulate; sag is contiguous from xp.zeros_like
            # above so .ravel() inside the kernel is a view.
            _sag_kernel(
                np.ascontiguousarray(h_sq), sag, powers_arr, coeffs_arr)
        else:
            for power, coeff in aspheric_coeffs.items():
                sag = sag + coeff * h_sq ** (power // 2)

    return sag


# Keep the private alias so internal callers (apply_real_lens) can use either
# name without changing semantics.
_surface_sag_general = surface_sag_general


def surface_sag_biconic(
    X: np.ndarray,
    Y: np.ndarray,
    R_x: float,
    R_y: Optional[float] = None,
    conic_x: float = 0.0,
    conic_y: Optional[float] = None,
    aspheric_coeffs: Optional[Dict[int, float]] = None,
    aspheric_coeffs_y: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Biconic / cylindrical / toroidal surface sag.

    Generalises :func:`surface_sag_general` to surfaces that have
    different curvatures and conics along the x and y axes.  Covers:

    * **Biconic** (Zemax "Biconic"): independent R_x, R_y, K_x, K_y.
      Each axis contributes its own conic sag:

          z(x,y) = C_x*x² / (1 + sqrt(1 - (1+K_x)*C_x²*x²))
                 + C_y*y² / (1 + sqrt(1 - (1+K_y)*C_y²*y²))

      where C_x = 1/R_x, C_y = 1/R_y.
    * **Cylindrical**: pass ``R_y = inf`` (focusing in x only) or
      ``R_x = inf`` (focusing in y only).
    * **Toroidal** (approx., Zemax "Toroidal"): pass R_x (rotation-axis
      radius) and R_y (cross-section radius).
    * **Rotationally symmetric**: if ``R_y is None`` the function
      reduces to :func:`surface_sag_general` via h² = x² + y².

    Aspheric coefficients may be given per-axis for fully general
    anamorphic surfaces; ``aspheric_coeffs`` (x-axis) and
    ``aspheric_coeffs_y`` (y-axis) are separate dicts of
    ``{power: coeff}`` contributing ``coeff * h² ** (power // 2)``
    along each axis.

    Parameters
    ----------
    X, Y : ndarray
        Surface-local coordinates [m] (after any decenter/tilt).
        ``X`` and ``Y`` must have the same shape; meshgrid indexing is
        up to the caller.
    R_x : float
        Radius of curvature along x-axis [m] (``inf`` = flat in x).
    R_y : float, optional
        Radius of curvature along y-axis [m].  If ``None``, the surface
        is treated as rotationally symmetric with R = R_x (legacy).
    conic_x : float, default 0
        Conic constant along x.
    conic_y : float, optional
        Conic constant along y.  Defaults to ``conic_x`` if not given.
    aspheric_coeffs : dict or None
        Even-aspheric coefficients along x, ``{power: coeff}``.
    aspheric_coeffs_y : dict or None
        Even-aspheric coefficients along y.  If ``None`` and
        ``aspheric_coeffs`` is given, the x coefficients are reused for
        y (isotropic asphere).

    Returns
    -------
    sag : ndarray
        Signed surface sag, same shape as ``X`` / ``Y``.
    """
    # Detect array backend and use cupy ops when X/Y are device arrays.
    xp = cp if _is_cupy_array(X) else np
    X = xp.asarray(X)
    Y = xp.asarray(Y)

    if R_y is None:
        # Reduce to the rotationally-symmetric formula for backward
        # compatibility.
        h_sq = X ** 2 + Y ** 2
        return surface_sag_general(h_sq, R_x, conic_x, aspheric_coeffs)

    if conic_y is None:
        conic_y = conic_x

    def _axis_sag(h_sq, R, K, asph):
        s = xp.zeros_like(h_sq)
        if R is not None and not np.isinf(R):
            norm = (1 + K) * h_sq / R ** 2
            valid = norm < 0.9999
            denom_arg = xp.where(valid, 1 - norm, 0.01)
            # v5.4.6 (audit F-19): outside the conic domain
            # (norm >= 0.9999) the surface is not defined.  Return NaN
            # (not a silent 0.0 flat ring) so callers detect 'no real
            # surface', matching surface_sag_general.
            s = xp.where(
                valid,
                h_sq / (R * (1 + xp.sqrt(denom_arg))),
                xp.nan,
            )
        if asph:
            for power, coeff in asph.items():
                s = s + coeff * h_sq ** (power // 2)
        return s

    sag_x = _axis_sag(X ** 2, R_x, conic_x, aspheric_coeffs)
    sag_y = _axis_sag(Y ** 2, R_y, conic_y,
                      aspheric_coeffs_y if aspheric_coeffs_y is not None
                      else aspheric_coeffs)
    return sag_x + sag_y


# ---------------------------------------------------------------------------
# Grid-vs-aperture safety check
# ---------------------------------------------------------------------------

def _collect_semi_diameters(prescription):
    """Return [(label, semi_diameter_m)] for every surface in the
    prescription that has a ``semi_diameter`` set.

    Looks at both ``prescription['elements']`` (Zemax-loaded path,
    where ``semi_diameter`` is the per-surface CLAP) and
    ``prescription['surfaces']`` (builder-style).  The top-level
    ``aperture_diameter`` (system-wide clear aperture) is also
    included if present.  Surfaces without a finite ``semi_diameter``
    are skipped.
    """
    out = []
    seen_labels = set()
    elements = prescription.get('elements')
    if isinstance(elements, list):
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            sd = elem.get('semi_diameter')
            if sd is None or not np.isfinite(float(sd)):
                continue
            surf_num = elem.get('surf_num', '?')
            comment = (elem.get('comment') or elem.get('name') or '').strip()
            label = f"surf {surf_num}"
            if comment:
                label = f"{label} '{comment}'"
            if label not in seen_labels:
                out.append((label, float(sd)))
                seen_labels.add(label)
    surfaces = prescription.get('surfaces')
    if isinstance(surfaces, list):
        for i, surf in enumerate(surfaces):
            if not isinstance(surf, dict):
                continue
            sd = surf.get('semi_diameter')
            if sd is None or not np.isfinite(float(sd)):
                continue
            label = f"surfaces[{i}]"
            if label not in seen_labels:
                out.append((label, float(sd)))
                seen_labels.add(label)
    ap = prescription.get('aperture_diameter')
    if ap is not None and np.isfinite(float(ap)):
        out.append(('system aperture_diameter', 0.5 * float(ap)))
    return out


def check_grid_vs_apertures(
    prescription: Dict[str, Any],
    N: int,
    dx: float,
    *,
    safety_factor: float = 1.0,
) -> List[Tuple[str, float, float, float]]:
    """Identify every prescription surface whose semi-aperture exceeds
    the simulation grid's half-extent.

    Use this as a pre-flight check before a full ASM-through-lens run.
    A surface whose ``semi_diameter`` is larger than ``safety_factor *
    N * dx / 2`` will silently truncate the field's outer rim during
    propagation, dropping any energy the real hardware would have
    transmitted past the grid edge.  This is a sim-infrastructure
    artifact, not a physical clipping by the lens itself, and will
    show up in centroid measurements as a uniform inward bias.

    Parameters
    ----------
    prescription : dict
        lumenairy prescription dict (loaded or builder-built).
    N : int
        Simulation grid size (assumed square).
    dx : float
        Grid pitch [m].
    safety_factor : float, optional
        Margin between the grid semi and the largest surface
        semi-aperture.  Pass 0.95 to flag surfaces whose semi exceeds
        95% of the grid half-extent (recommended for clean Gaussian
        wing containment); pass 1.0 (default) to flag only surfaces
        that exceed the grid outright.

    Returns
    -------
    issues : list of (label, semi_aperture_m, grid_semi_m, gap_m)
        One tuple per offending surface.  Empty if every aperture fits.

    See Also
    --------
    apply_real_lens, apply_real_lens_traced, apply_real_lens_maslov
        These functions automatically run this check on entry and emit
        a UserWarning if any aperture exceeds the grid.
    """
    if N <= 0 or dx <= 0:
        raise ValueError(f"N and dx must be positive, got N={N}, dx={dx}")
    grid_semi = 0.5 * float(N) * float(dx)
    threshold = float(safety_factor) * grid_semi
    issues = []
    for label, sd in _collect_semi_diameters(prescription):
        if sd > threshold:
            issues.append((label, sd, grid_semi, sd - grid_semi))
    return issues


def recommend_grid_for_prescription(
    prescription: Dict[str, Any],
    wavelength: float,
    *,
    source_waist: Optional[float] = None,
    doe_orders_max: Optional[Tuple[float, float]] = None,
    doe_period: Optional[Union[float, Tuple[float, float]]] = None,
    doe_to_destination_distance: Optional[float] = None,
    margin_ratio: float = 1.05,
    samples_per_wavelength: float = 4.0,
    samples_per_source_waist: float = 6.0,
    round_n_to_power_of_two: bool = True,
) -> Dict[str, Any]:
    """Recommend simulation grid parameters ``(N, dx)`` for an ASM run
    through a sequential prescription.

    The recommendation answers two coupled questions before a long
    simulation run:

    * **How wide does the grid need to be?**  Determined by the largest
      ``semi_diameter`` in the prescription, optionally extended for
      DOE diffraction-order spread (corner-order chief rays at the
      destination plane sit at ``m * lambda * z_propagation / period``
      off-axis and must fit inside ``N * dx / 2``).

    * **How fine does the pixel pitch need to be?**  Bounded above by
      the wavelength's effective Nyquist (``lambda /
      samples_per_wavelength``) and, if a source size is supplied, by
      the source-waist resolution (``source_waist /
      samples_per_source_waist``).

    The two requirements together pin ``N`` and ``dx`` (with ``N``
    rounded up to the nearest power of two for FFT efficiency by
    default) such that ``N * dx / 2 >= margin_ratio * required_extent``
    and ``dx <= min(dx_constraints)``.

    Parameters
    ----------
    prescription : dict
        lumenairy prescription dict.  Must include either ``elements``
        with ``semi_diameter`` fields or a top-level ``aperture_diameter``;
        without one of those the function raises ``ValueError``.
    wavelength : float
        Vacuum wavelength [m].  Sets the absolute Nyquist bound on
        ``dx``.
    source_waist : float, optional
        Source 1/e Gaussian waist [m].  When supplied, ``dx`` is also
        bounded above by ``source_waist / samples_per_source_waist`` so
        the source itself is well-resolved on the grid.
    doe_orders_max : (int, int) or (float, float), optional
        Maximum diffraction-order index along (x, y) the simulation
        will need to retain (e.g. ``(5.5, 5.5)`` for a 12 x 12 Dammann
        splitter's diagonal corner).  Must be supplied together with
        ``doe_period`` and ``doe_to_destination_distance`` for the
        DOE-corner-order spread to be added to the required grid
        extent.
    doe_period : float or (float, float), optional
        Grating period [m].  Scalar = isotropic; tuple = (x, y).
    doe_to_destination_distance : float, optional
        Free-space propagation distance from the DOE plane to the
        destination plane that the recommendation is sizing for [m].
    margin_ratio : float, optional
        Multiplicative safety margin on the required grid semi
        (default 1.05 = 5%).
    samples_per_wavelength : float, optional
        Minimum samples per wavelength along each grid axis (default
        4.0).  Nyquist is 2.0; 4.0 leaves comfortable headroom for the
        ASM kernel's exp(i*k*z*sqrt(...)) phase ramp.
    samples_per_source_waist : float, optional
        Minimum samples across one source waist (default 6.0).  Below
        ~4 the Gaussian source is poorly resolved.
    round_n_to_power_of_two : bool, optional
        If True (default), round ``N`` up to the nearest power of two.
        FFT-based propagators run several times faster on power-of-two
        grids than on arbitrary sizes.

    Returns
    -------
    dict
        Recommendation with at least these keys:

          ``N`` -- recommended grid size (int)
          ``dx`` -- recommended pixel pitch [m]
          ``grid_semi_m`` -- ``N * dx / 2``
          ``required_extent_m`` -- before margin
          ``limiting_aperture_m`` -- the surface that set the grid extent
          ``limiting_aperture_label`` -- human-readable label of that surface
          ``doe_extra_extent_m`` -- additional extent demanded by the DOE
          ``dx_constraints`` -- dict of every dx upper bound considered
          ``dx_limiting_constraint`` -- name of the binding constraint
          ``samples_per_wavelength`` -- effective post-rounding value

    Raises
    ------
    ValueError
        If the prescription has no resolvable aperture, or if any of
        ``doe_*`` are partially supplied (must be all-or-none).

    See Also
    --------
    check_grid_vs_apertures
        The post-flight companion that flags whether a chosen ``(N,
        dx)`` actually contains every prescription aperture.
    """
    if wavelength <= 0:
        raise ValueError(f"wavelength must be > 0, got {wavelength}")
    if margin_ratio < 1.0:
        raise ValueError(
            f"margin_ratio must be >= 1.0 (got {margin_ratio}); use 1.0 "
            "for the bare-minimum grid that just contains every aperture")

    # DOE arguments are all-or-none
    doe_args = (doe_orders_max, doe_period, doe_to_destination_distance)
    if any(a is not None for a in doe_args) and not all(
            a is not None for a in doe_args):
        raise ValueError(
            "doe_orders_max, doe_period, and doe_to_destination_distance "
            "must all be supplied together (or all be None)")

    # 1) Find the largest aperture in the prescription
    max_semi = 0.0
    aperture_label = ''
    elements = prescription.get('elements')
    if isinstance(elements, list):
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            sd = elem.get('semi_diameter')
            if sd is None or not np.isfinite(float(sd)):
                continue
            sd = float(sd)
            if sd > max_semi:
                max_semi = sd
                lbl_parts = []
                if elem.get('comment'):
                    lbl_parts.append(str(elem['comment']))
                if elem.get('surf_num') is not None:
                    lbl_parts.append(f"surf {elem['surf_num']}")
                aperture_label = (
                    ' / '.join(lbl_parts) if lbl_parts
                    else 'unknown element')
    surfaces = prescription.get('surfaces')
    if isinstance(surfaces, list):
        for i, surf in enumerate(surfaces):
            if not isinstance(surf, dict):
                continue
            sd = surf.get('semi_diameter')
            if sd is None or not np.isfinite(float(sd)):
                continue
            sd = float(sd)
            if sd > max_semi:
                max_semi = sd
                aperture_label = f"surfaces[{i}]"
    sys_ap = prescription.get('aperture_diameter')
    if sys_ap is not None and np.isfinite(float(sys_ap)):
        if 0.5 * float(sys_ap) > max_semi:
            max_semi = 0.5 * float(sys_ap)
            aperture_label = 'system aperture_diameter'

    if max_semi <= 0.0:
        raise ValueError(
            "prescription has no resolvable aperture: pass a prescription "
            "whose elements / surfaces carry 'semi_diameter' or whose "
            "top-level 'aperture_diameter' is finite")

    # 2) DOE-order extent extension (if supplied)
    doe_extra = 0.0
    if doe_orders_max is not None:
        m_x, m_y = doe_orders_max
        if isinstance(doe_period, (tuple, list)):
            period_x, period_y = (float(doe_period[0]),
                                   float(doe_period[1]))
        else:
            period_x = period_y = float(doe_period)
        if period_x <= 0 or period_y <= 0:
            raise ValueError(f"doe_period must be > 0, got {doe_period}")
        z_doe = float(doe_to_destination_distance)
        # Diagonal corner order's chief-ray displacement at the
        # destination plane (paraxial direction-cosine approximation).
        theta_x = float(m_x) * wavelength / period_x
        theta_y = float(m_y) * wavelength / period_y
        doe_extra = abs(z_doe) * float(np.hypot(theta_x, theta_y))

    required_extent = max_semi + doe_extra
    required_grid_semi = margin_ratio * required_extent

    # 3) dx upper bounds
    dx_constraints = {
        'wavelength_nyquist': float(wavelength) / float(samples_per_wavelength),
    }
    if source_waist is not None and float(source_waist) > 0:
        dx_constraints['source_waist'] = (
            float(source_waist) / float(samples_per_source_waist))
    dx_max = min(dx_constraints.values())
    dx_limiting = min(dx_constraints, key=dx_constraints.get)

    # 4) Pick (N, dx)
    n_min = int(np.ceil(2.0 * required_grid_semi / dx_max))
    if round_n_to_power_of_two:
        n = int(2 ** int(np.ceil(np.log2(max(n_min, 1)))))
    else:
        n = int(n_min)
    # Recompute dx to land grid_semi exactly at required_grid_semi.
    dx = 2.0 * required_grid_semi / n

    return {
        'N': n,
        'dx': float(dx),
        'grid_semi_m': float(n * dx / 2.0),
        'required_extent_m': float(required_extent),
        'limiting_aperture_m': float(max_semi),
        'limiting_aperture_label': aperture_label,
        'doe_extra_extent_m': float(doe_extra),
        'dx_constraints': dx_constraints,
        'dx_limiting_constraint': dx_limiting,
        'samples_per_wavelength': float(wavelength) / float(dx),
        'margin_ratio': float(margin_ratio),
    }


def _warn_if_aperture_exceeds_grid(prescription, N, dx, *,
                                    source='apply_real_lens',
                                    safety_factor=1.0,
                                    stacklevel=3):
    """Emit a UserWarning if any prescription aperture exceeds the
    simulation grid.  Called at the top of ``apply_real_lens``,
    ``apply_real_lens_traced``, and ``apply_real_lens_maslov``.

    Python's default warning filter dedups by ``(module, lineno)`` so
    repeated calls from the same site only warn once.
    """
    try:
        issues = check_grid_vs_apertures(
            prescription, N, dx, safety_factor=safety_factor)
    except (KeyError, ValueError, TypeError, AttributeError):
        # Aperture check is best-effort; a malformed prescription
        # shouldn't block the warning path entirely (the
        # propagator's own validators will raise downstream).
        return
    if not issues:
        return
    grid_semi = 0.5 * N * dx
    issues_sorted = sorted(issues, key=lambda r: -r[1])
    biggest_label, biggest_sd, _, biggest_gap = issues_sorted[0]
    body = ", ".join(
        f"{lab}={sd*1e3:.2f}mm"
        for lab, sd, _, _ in issues_sorted[:5]
    )
    if len(issues_sorted) > 5:
        body = body + f", ... (+{len(issues_sorted)-5} more)"
    msg = (
        f"{source}: {len(issues)} prescription aperture(s) exceed the "
        f"simulation grid (N={N}, dx={dx*1e6:.3f} um, "
        f"semi={grid_semi*1e3:.3f} mm). "
        f"Largest is {biggest_label} with semi_diameter="
        f"{biggest_sd*1e3:.3f} mm "
        f"({biggest_gap*1e3:+.3f} mm beyond the grid); the field will "
        f"be truncated at the grid edge during propagation, "
        f"silently dropping energy the real lens would have "
        f"transmitted. Consider increasing N or dx so "
        f"N*dx/2 >= max(semi_diameter). "
        f"Affected surfaces: {body}."
    )
    warnings.warn(msg, UserWarning, stacklevel=stacklevel)



# ---------------------------------------------------------------------------
# Shared Chebyshev helpers used by the Maslov and asymptotic propagators.
#
# v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
# The three Chebyshev Vandermonde helpers that used to live here
# (originally inlined in v3.2.2, kept here in v3.5.5 because
# propagators/asymptotic.py imports them) have moved to
# ``lumenairy/_math/chebyshev.py``.  The underscore-prefixed aliases
# below preserve every existing internal call site in this module --
# and every external import of the form
# ``from lumenairy.elements.lenses import _chebyshev_vandermonde`` --
# without forcing those callers to update their import paths.
# ---------------------------------------------------------------------------

from .._math.chebyshev import (  # noqa: F401 -- back-compat alias re-export (v5.2)
    chebyshev_derivative_vandermonde as _chebyshev_derivative_vandermonde,  # noqa: F401
)
from .._math.chebyshev import (
    chebyshev_second_derivative_vandermonde as _chebyshev_second_derivative_vandermonde,  # noqa: F401
)
from .._math.chebyshev import (
    chebyshev_vandermonde as _chebyshev_vandermonde,  # noqa: F401
)


def _multi_indices_total_degree(n_vars: int, max_order: int):
    """Enumerate multi-indices k with sum(k) <= max_order, as list of tuples."""
    out = []
    def recurse(prefix, remaining, depth):
        if depth == n_vars:
            out.append(tuple(prefix))
            return
        for k in range(remaining + 1):
            recurse(prefix + [k], remaining - k, depth + 1)
    recurse([], max_order, 0)
    return out


def _evaluate_polynomial_4d(coeffs: np.ndarray,
                              multi_indices,
                              u1: np.ndarray, u2: np.ndarray,
                              u3: np.ndarray, u4: np.ndarray,
                              max_order: int) -> np.ndarray:
    """
    Evaluate a 4-variable Chebyshev tensor-product polynomial in
    total-degree subspace at arbitrary (u1, u2, u3, u4) samples.

    Parameters
    ----------
    coeffs : ndarray, shape (M,)
        Polynomial coefficients in the same order as ``multi_indices``.
    multi_indices : list of 4-tuples
        Multi-indices (k1, k2, k3, k4) enumerating the basis.
    u1, u2, u3, u4 : ndarrays with identical shape
        Evaluation points in [-1, 1]^4.
    max_order : int
        Maximum individual index (== total-degree cap in this call).

    Returns
    -------
    value : ndarray with the broadcast shape of (u1, ..., u4)
    """
    # 3.5.6: vectorised over basis terms (no Python loop).  T_i has
    # shape (max_k+1, *u_shape); T_i[K_i] has shape (M, *u_shape);
    # product is (M, *u_shape); contracting over basis axis with
    # coeffs gives the result.  ~3x faster than the previous loop on
    # M=70 basis terms x 1024-pt u arrays.
    T1 = _chebyshev_vandermonde(u1, max_order)
    T2 = _chebyshev_vandermonde(u2, max_order)
    T3 = _chebyshev_vandermonde(u3, max_order)
    T4 = _chebyshev_vandermonde(u4, max_order)
    K = np.asarray(multi_indices, dtype=np.int64)
    K1, K2, K3, K4 = K[:, 0], K[:, 1], K[:, 2], K[:, 3]
    basis = T1[K1] * T2[K2] * T3[K3] * T4[K4]
    return np.tensordot(np.asarray(coeffs, dtype=np.float64),
                         basis, axes=([0], [0]))


def _evaluate_polynomial_4d_and_grad34(coeffs: np.ndarray,
                                         multi_indices,
                                         u1, u2, u3, u4,
                                         max_order: int
                                         ) -> Tuple[np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray]:
    """
    Evaluate the 4-variable polynomial at (u1, u2, u3, u4) and also its
    partial derivatives d/du3 and d/du4 (used for the Jacobian w.r.t.
    the v2 coordinates).

    Returns
    -------
    f, df_du3, df_du4
    """
    # 4.14.0 (Tier-2 perf, audit group): vectorised over basis terms,
    # mirroring the sibling ``_evaluate_polynomial_4d``.  The previous
    # version walked the M-term basis in a Python loop computing a
    # fused multiply-add over the evaluation grid; that loop overhead
    # is the dominant cost when this helper is hot inside the
    # asymptotic propagator's Newton iterations.
    #
    # The (M, *u_shape) basis tensors trade memory for three
    # tensordots; measured speedup vs the scalar loop at M=70 is
    # ~4.5x at an 8x8 grid, ~3.5x at 16x16, ~1.7x at 32x32 (and
    # marginally slower above ~48x48 where the (M, *u_shape)
    # intermediates spill L2 cache).  Typical
    # ``apply_real_lens_maslov`` callers run with n_v2=32, so the
    # 32x32 regime is the operational point; the cache-bound large-
    # grid regime is not exercised in production.
    T1 = _chebyshev_vandermonde(u1, max_order)
    T2 = _chebyshev_vandermonde(u2, max_order)
    T3 = _chebyshev_vandermonde(u3, max_order)
    T4 = _chebyshev_vandermonde(u4, max_order)
    dT3 = _chebyshev_derivative_vandermonde(u3, max_order)
    dT4 = _chebyshev_derivative_vandermonde(u4, max_order)
    K = np.asarray(multi_indices, dtype=np.int64)
    K1, K2, K3, K4 = K[:, 0], K[:, 1], K[:, 2], K[:, 3]
    # T12 has shape (M, *u_shape); reused for all three outputs.
    T12 = T1[K1] * T2[K2]
    c_arr = np.asarray(coeffs, dtype=np.float64)
    basis_f = T12 * T3[K3] * T4[K4]
    basis_d3 = T12 * dT3[K3] * T4[K4]
    basis_d4 = T12 * T3[K3] * dT4[K4]
    f = np.tensordot(c_arr, basis_f, axes=([0], [0]))
    df3 = np.tensordot(c_arr, basis_d3, axes=([0], [0]))
    df4 = np.tensordot(c_arr, basis_d4, axes=([0], [0]))
    return f, df3, df4


# ---------------------------------------------------------------------------
# Data normalisation helpers
# ---------------------------------------------------------------------------

def _fit_normaliser(v: np.ndarray, pad: float = 0.05):
    """Return (center, half_range) such that (v - center)/half_range sits
    in [-(1-pad), (1-pad)].

    pad leaves a narrow margin so that mild extrapolation by the
    propagator is still bounded.
    """
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    center = 0.5 * (vmin + vmax)
    half = 0.5 * (vmax - vmin) * (1.0 + pad)
    if half == 0.0:
        half = 1.0
    return center, half


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Maslov propagator -- moved to lumenairy.elements.lenses_maslov in v3.5.5
# to reduce lenses.py bloat.  Re-exported here for backwards-compatible
# `from lumenairy.elements.lenses import apply_real_lens_maslov` imports.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# JAX-traceable real-lens propagators moved to
# lumenairy.elements._lens_jax in v3.5.5.  Re-exported here so existing
# `from lumenairy.elements.lenses import apply_real_lens_traced_jax` /
# `apply_real_lens_maslov_jax` imports continue to work.
# ---------------------------------------------------------------------------
from ._lens_jax import (  # noqa: E402, F401
    apply_real_lens_maslov_jax,
    apply_real_lens_traced_jax,
)

# ---------------------------------------------------------------------------
# Analytic split-step real-lens propagator moved to
# lumenairy.elements._lens_real in v3.5.5.  Re-exported here so
# existing `from lumenairy.elements.lenses import apply_real_lens`
# imports continue to work.
# ---------------------------------------------------------------------------
from ._lens_real import (  # noqa: E402, F401
    apply_real_lens,
    get_lens_sag_dtype,
    lens_sag_float32_opd_error,
    set_lens_sag_dtype,
)

# ---------------------------------------------------------------------------
# Thin-lens / single-element phase screens were moved to
# lumenairy.elements._lens_thin in v3.5.5.  Re-exported here so existing
# `from lumenairy.elements.lenses import apply_thin_lens` etc. continue
# to work.
# ---------------------------------------------------------------------------
from ._lens_thin import (  # noqa: E402, F401
    apply_aspheric_lens,
    apply_axicon,
    apply_cylindrical_lens,
    apply_grin_lens,
    apply_spherical_lens,
    apply_thin_lens,
)

# ---------------------------------------------------------------------------
# Per-pixel ray-traced apply_real_lens variant moved to
# lumenairy.elements._lens_traced in v3.5.5.  Re-exported here so
# existing
#   from lumenairy.elements.lenses import apply_real_lens_traced
#   from lumenairy.elements.lenses import close_worker_pool
# imports continue to work.
# ---------------------------------------------------------------------------
from ._lens_traced import (  # noqa: E402, F401
    PreparedTracedLens,
    apply_real_lens_traced,
    apply_real_lens_traced_multi,
    close_worker_pool,
    get_lens_parallel_amp,
    prepare_real_lens_traced,
    set_lens_parallel_amp,
)
from .lenses_maslov import apply_real_lens_maslov  # noqa: F401, E402
