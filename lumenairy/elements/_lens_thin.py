"""
lumenairy.elements._lens_thin -- thin-lens / single-element phase screens.

Six small entry points that model an optical element as a single
thin phase mask multiplied onto the input field.  Extracted from
``lenses.py`` in v3.5.5 to reduce that module's bloat.  All names
re-exported from :mod:`lumenairy.elements.lenses` for
backwards-compatible imports.

Functions
---------
apply_thin_lens / apply_spherical_lens / apply_aspheric_lens
    Plano + bi-curved single-element refractive lenses.
apply_cylindrical_lens
    Single-axis focusing element.
apply_axicon
    Conical phase element (Bessel-beam generator).
apply_grin_lens
    Gradient-index rod lens (paraxial).

All functions accept a ``use_gpu=False`` flag and dispatch to CuPy
when CuPy is installed.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np

from ..glass import get_glass_index  # 4.10: was missing, broke apply_axicon

# CuPy is lazy-loaded; this module accesses it via the lenses module's
# lazy slot so a single load is shared across the package.
from . import lenses as _lenses_module
from .lenses import (
    CUPY_AVAILABLE,
)


def _is_cupy_array(x):
    return _lenses_module._is_cupy_array(x)


# Module-level cp alias.  Updated whenever _lenses_module's cp is loaded
# (it points at None until first GPU call, then the actual cupy
# module).  We sync via a property-style accessor below.

def __getattr__(name):
    """PEP 562 module-level __getattr__: route ``cp`` to the lenses
    module's lazy slot.  Triggers when callers do
    ``from ._lens_thin import cp`` -- the in-function references inside
    each apply_* below resolve via this fallback if `cp` isn't yet a
    module global.
    """
    if name == 'cp':
        if _lenses_module.cp is None:
            _lenses_module._ensure_cupy_loaded()
        return _lenses_module.cp
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


# ---------------------------------------------------------------------------
# Thin lens models
# ---------------------------------------------------------------------------

def apply_thin_lens(
    E_in: np.ndarray,
    *,
    f: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    xc: float = 0,
    yc: float = 0,
    use_gpu: bool = False,
    lens_model: str = 'paraxial',
    conjugates: Optional[object] = None,
) -> np.ndarray:
    """
    Apply a thin-lens phase to an optical field.

    Parameters
    ----------
    E_in : ndarray (complex), shape (Ny, Nx)
        Input electric field.
    f : float
        Focal length [m].  Positive = converging, negative = diverging.
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    xc, yc : float
        Center of the lens [m] (for decentered lenses).
    use_gpu : bool
        If True and CuPy is available, run on the GPU.
    lens_model : str
        Phase model.  One of:

        ``'paraxial'``
            Quadratic approximation: phi = -k/(2f) * r**2.
            Valid for r/f < ~0.1 (half-angle < ~6 deg).
        ``'nonparaxial'``
            Exact stigmatic spherical wavefront for INFINITE conjugates:
            phi = -sign(f) * k * (sqrt(f**2 + r**2) - |f|).
            For f > 0 this equals the historical k*(f - sqrt(f**2+r**2))
            byte-for-byte; for f < 0 the historical form had the WRONG
            SIGN (it converged exactly like its +|f| twin -- thin-lens
            audit 2026-07-18, bug 1).  Exact for collimated->focus or
            focus->collimated; at FINITE conjugates it over-corrects
            (use ``'stigmatic'``).
        ``'aplanatic'``
            Stigmatic spherical phase restricted to the Abbe sine-
            condition domain r < |f| (unit phase outside).  The
            historical profile -k*f*(1 - sqrt(1 - r**2/f**2)) had the
            WRONG quartic sign (-k r**4/8f**3 where a converging sphere
            needs +k r**4/8f**3), so it DOUBLED paraxial's spherical
            aberration instead of removing it (thin-lens audit
            2026-07-18, bug 2; measured 9.1 um focus vs paraxial 7.7 um
            vs correct 4.27 um at NA~0.1).  Its ray mapping was
            sin(theta) = tan(asin(r/f)) -- neither the sine condition
            (whose pure-phase-screen profile is the PARAXIAL quadratic)
            nor the tangent/stigmatic condition (the spherical phase).
            NOTE: the Abbe part of a real aplanat is the sqrt(cos theta)
            PUPIL APODIZATION -- an amplitude factor a pure phase mask
            must not apply; enforce apertures/apodization separately.
        ``'stigmatic'``
            Conjugate-matched EXACT ideal element (thin-lens audit
            2026-07-18, change 1): phi = k * (S(R_out) - S(R_in)) with
            S(R) = sign(R) * (sqrt(r**2 + R**2) - |R|) the exact
            spherical-wave phase (R > 0 diverging) and, by default,
            1/R_out = 1/R_in - 1/f.  This is aberration-free under the
            EXACT (ASM) propagator at ANY conjugates -- pass the
            incoming wavefront radius via ``conjugates``.  With
            R_in = inf (collimated input, the default) it reduces
            exactly to ``'nonparaxial'``.
        ``'local_only'``
            **Deprecated since v5.29.1 (audit E-H7) -- and its pre-v5.29.1
            docstring described the OPPOSITE of what it does.**  It is a
            decentered quadratic *plus* the cancelling linear ramp
            ``-k/f * (xc*X + yc*Y)``, which algebraically collapses to an
            ORIGIN-centred parabola plus the constant piston
            ``-k/(2f)(xc^2 + yc^2)``::

                -k/(2f)[(X-xc)^2 + (Y-yc)^2] - k/f (xc X + yc Y)
                  == -k/(2f)(X^2 + Y^2) - k/(2f)(xc^2 + yc^2)

            So its local phase gradient at the lenslet centre is ``-k*xc/f``
            (NOT zero): the sub-beam IS steered, by ``-xc/f``, i.e. straight
            onto the optical axis -- measured -20.0 mrad for
            ``xc = 100 um, f = 5 mm``, landing the spot at ``x = 0``.  It is
            bit-identical to ``lens_model='paraxial'`` with ``xc = yc = 0``
            up to that piston (measured ``|ratio|`` spread 7.8e-16,
            ``arg`` spread 7.1e-14 rad).

            The NO-STEER model the old text promised is the plain decentered
            ``'paraxial'`` lens (``paraxial`` with ``xc``/``yc`` set), whose
            gradient at ``(xc, yc)`` is exactly zero -- that is what a
            micro-lens array wants.  This value is kept (unchanged behaviour)
            for back-compatibility only; prefer ``'paraxial'`` with
            ``xc = yc = 0`` if you actually want an axis-centred lens.
    conjugates : float or (float, float), optional
        Only used by ``lens_model='stigmatic'``.  Either the incoming
        wavefront radius of curvature ``R_in`` alone (signed, metres,
        R > 0 diverging, ``np.inf`` = collimated; ``R_out`` is then
        derived from the lens equation 1/R_out = 1/R_in - 1/f) or an
        explicit ``(R_in, R_out)`` pair (which overrides ``f`` for the
        phase -- useful for pure curvature converters).  Defaults to
        ``R_in = inf``.

    Returns
    -------
    E_out : ndarray (complex), same shape as *E_in*

    Notes
    -----
    All arguments past ``E_in`` are keyword-only (since 4.7).  This
    makes the call order non-load-bearing and prevents typos that
    silently swap ``wavelength`` and ``dx`` (both ~1e-6).

    Scope of that guarantee (v5.30, audit E-M9).  The 4.7 keyword-only
    conversion covered the **eight** ``apply_*_lens`` entry points
    (:func:`apply_thin_lens`, :func:`apply_spherical_lens`,
    :func:`apply_aspheric_lens`, :func:`apply_cylindrical_lens`,
    :func:`apply_grin_lens`,
    :func:`lumenairy.elements.apply_real_lens`,
    :func:`lumenairy.elements.apply_real_lens_traced`,
    :func:`lumenairy.elements.apply_real_lens_maslov`) -- **not the
    whole library**.  Several element / grating entry points still take
    positional floats of similar magnitude, so the swap footgun is live
    there and the call order IS load-bearing:

    * :func:`apply_axicon` -- ``(E_in, alpha, n_axicon, wavelength, dx,
      dy)``
    * :func:`lumenairy.elements.apply_mirror` -- ``(E_in, wavelength,
      dx, radius, conic, aperture_diameter, xc, yc, dy)``
    * :func:`lumenairy.elements.apply_aperture` -- ``(E_in, dx, shape,
      params, xc, yc, dy)``
    * :func:`lumenairy.elements.apply_zernike_aberration` --
      ``(E_in, dx, coefficients, aperture_radius, dy)``
    * :func:`lumenairy.elements.thin_grating_efficiency_1d` -- an
      all-positional ``(period, n_ridge, n_groove, n_substrate,
      n_superstrate, depth, ...)`` float list

    These are deliberately NOT converted (it would break the public
    API without a deprecation cycle); pass every argument past the
    field by keyword and the footgun disappears at the call site.
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper (replaces the v4.15.2 inline
    # guard).  Runs FIRST so the user gets a clear, actionable error
    # rather than a downstream AttributeError or silent wrong-axis
    # broadcast.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_thin_lens')

    # Determine array library.  PEP 562 ``__getattr__`` cannot
    # resolve bare ``cp`` inside a function body (LEGB rules skip
    # module-level __getattr__), so we go through the lenses-module
    # lazy slot explicitly.  Same pattern as apply_cylindrical_lens /
    # apply_grin_lens / apply_axicon.
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if _lenses_module.cp is None:
            _lenses_module._ensure_cupy_loaded()
        _cp = _lenses_module.cp
        xp = _cp
        if not _is_cupy_array(E_in):
            E_in = _cp.asarray(E_in)
    else:
        xp = np

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx

    k = 2 * np.pi / wavelength

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    r_sq = (X - xc) ** 2 + (Y - yc) ** 2

    if lens_model == 'paraxial':
        lens_phase = xp.exp(-1j * k / (2 * f) * r_sq)

    elif lens_model == 'nonparaxial':
        # v5.25.0 (thin-lens audit 2026-07-18, bug 1): the historical
        # ``exp(1j*k*(f - sqrt(f**2 + r_sq)))`` expands, for f < 0, to a
        # CONVERGING quadratic -k r**2/(2|f|) -- a diverging lens that
        # focused identically to its +|f| twin (measured: f = -30 mm
        # produced the same z = +30 mm focus and peak as f = +30 mm).
        # The sign-safe stigmatic sphere is
        # ``phi = -sign(f) * k * (sqrt(r**2 + f**2) - |f|)``, which is
        # byte-identical to the historical form for f > 0 (IEEE negation
        # of an exact subtraction) and correctly DIVERGES for f < 0.
        lens_phase = xp.exp(
            -1j * np.sign(f) * k * (xp.sqrt(f ** 2 + r_sq) - abs(f)))

    elif lens_model == 'aplanatic':
        # 4.10: replacing the outside-aperture region with 0+0j inside
        # the PHASE-MASK array silently clipped the amplitude there.
        # Use 1+0j (unit phase) outside the aplanatic domain so the
        # multiplier leaves the field unchanged in the rim annulus;
        # the lens aperture itself should be enforced via a separate
        # aperture mask, not via the phase mask.
        #
        # v5.25.0 (thin-lens audit 2026-07-18, bug 2): the historical
        # profile ``-k*f*(1 - sqrt(1 - r**2/f**2))`` expands to
        # ``-k r**2/2f - k r**4/8f**3`` -- the WRONG quartic sign (a
        # converging sphere needs ``+k r**4/8f**3``), so it carried 2x
        # paraxial's spherical-aberration error in the SAME direction
        # and focused WORSE than paraxial (9.1 um vs 7.7 um vs correct
        # 4.27 um at NA~0.1).  Its implied ray mapping,
        # sin(theta) = tan(asin(r/f)), is neither the Abbe sine
        # condition (whose pure-phase-screen profile is the PARAXIAL
        # quadratic) nor the stigmatic tangent condition.  The corrected
        # phase is the exact stigmatic sphere restricted to the
        # sine-condition domain r < |f|; the sqrt(cos theta) pupil
        # APODIZATION that distinguishes a true aplanat is an AMPLITUDE
        # factor a pure phase mask must not apply (see docstring).
        r_over_f_sq = r_sq / f ** 2
        valid = r_over_f_sq < 1.0
        phase = np.sign(f) * k * (xp.sqrt(f ** 2 + r_sq) - abs(f))
        # v4.14.1 (audit P2-6): dtype-aware unit-phase sentinel so the
        # ``xp.where`` doesn't pin lens_phase to complex128 via the
        # ``1.0 + 0.0j`` complex128 literal (matches v4.13.2 canonical
        # pattern for the ``0.0 + 0.0j`` sweep).
        lens_phase_valid = xp.exp(-1j * phase)
        lens_phase = xp.where(
            valid, lens_phase_valid,
            xp.ones((), dtype=lens_phase_valid.dtype)
        )

    elif lens_model == 'stigmatic':
        # v5.25.0 (thin-lens audit 2026-07-18, change 1): the
        # conjugate-matched EXACT ideal element.  A stigmatic element
        # mapping incoming curvature R_in to outgoing R_out applies
        # ``phi = k * (S(R_out) - S(R_in))`` with the exact signed
        # spherical-wave phase ``S(R) = sign(R)*(sqrt(r**2+R**2)-|R|)``
        # (R > 0 diverging; S(inf) = 0).  Quadratic part = the lens
        # equation -k r**2/2f; quartic part = -k r**4/8 *
        # (1/R_out**3 - 1/R_in**3) -- the term the 'paraxial' model
        # omits entirely and 'nonparaxial' only gets right at infinite
        # conjugates.  Proven on the 121 six-group chain: exact-ASM +
        # stigmatic images at the analytic waist with EE(6um) = 99.9%
        # and no pedestal, where paraxial x exact-ASM left +11.9 rad of
        # fictitious spherical aberration.
        if conjugates is None:
            R_in = np.inf
            R_out_given = None
        elif np.isscalar(conjugates):
            R_in = float(conjugates)
            R_out_given = None
        else:
            _pair = tuple(conjugates)
            if len(_pair) == 1:
                R_in, R_out_given = float(_pair[0]), None
            elif len(_pair) == 2:
                R_in, R_out_given = float(_pair[0]), float(_pair[1])
            else:
                raise ValueError(
                    "conjugates must be R_in or (R_in, R_out); got "
                    f"{conjugates!r}")
        if R_in == 0 or (R_out_given is not None and R_out_given == 0):
            raise ValueError(
                "stigmatic conjugates must be nonzero (R = 0 is a point "
                "ON the element); got "
                f"R_in={R_in!r}, R_out={R_out_given!r}")
        if R_out_given is not None:
            R_out = R_out_given
        else:
            # Lens equation on signed curvatures: 1/R_out = 1/R_in - 1/f.
            inv_out = (0.0 if np.isinf(R_in) else 1.0 / R_in) - 1.0 / f
            R_out = np.inf if inv_out == 0.0 else 1.0 / inv_out

        def _sphere_phase(R):
            # Exact signed spherical-wave phase S(R); 0 for R = +/-inf.
            if np.isinf(R):
                return None                      # contributes nothing
            return np.sign(R) * k * (xp.sqrt(r_sq + R ** 2) - abs(R))

        S_out = _sphere_phase(R_out)
        S_in = _sphere_phase(R_in)
        if S_out is None and S_in is None:
            lens_phase = xp.ones_like(E_in)
        elif S_in is None:
            lens_phase = xp.exp(1j * S_out)
        elif S_out is None:
            lens_phase = xp.exp(-1j * S_in)
        else:
            lens_phase = xp.exp(1j * (S_out - S_in))

    elif lens_model == 'local_only':
        # DEPRECATED (v5.29.1, audit E-H7).  The comment that used to sit here
        # -- "the standard decentered quadratic minus the linear tilt that
        # would otherwise steer the beam" -- was backwards, as was the
        # docstring: the sum below expands to an ORIGIN-centred parabola plus a
        # constant piston, so the local gradient at (xc, yc) is -k*xc/f and the
        # sub-beam is steered ONTO THE AXIS.  The zero-gradient (no-steer)
        # model is the plain decentered 'paraxial' lens.  Behaviour is
        # deliberately unchanged (0 callers in-repo, but it is a public enum
        # value -- deprecate, don't break); see the docstring for the algebra
        # and the measured -20.0 mrad steer.
        decentered_phase = -k / (2 * f) * r_sq
        tilt_cancel = -k / f * (xc * X + yc * Y)
        lens_phase = xp.exp(1j * (decentered_phase + tilt_cancel))

    else:
        raise ValueError(
            f"Unknown lens_model: {lens_model!r}. "
            f"Choose from 'paraxial', 'nonparaxial', 'aplanatic', "
            f"'stigmatic', 'local_only'."
        )

    if conjugates is not None and lens_model != 'stigmatic':
        raise ValueError(
            f"conjugates= is only meaningful for lens_model='stigmatic' "
            f"(got lens_model={lens_model!r}); a silently-ignored "
            f"conjugates would look like a working stigmatic element.")

    # v4.13.2 (audit C-P1-5): coerce the phase mask to E_in's dtype so
    # a complex64 input stays complex64.  ``xp.exp(1j * <float64
    # phase>)`` produces complex128 regardless of E_in.dtype; without
    # this cast the multiply silently upcasts E to complex128.
    # Mirrors the v4.13.0 L6 apply_mirror dtype guard.
    if lens_phase.dtype != E_in.dtype:
        lens_phase = lens_phase.astype(E_in.dtype)

    return E_in * lens_phase


# ---------------------------------------------------------------------------
# Thick spherical singlet
# ---------------------------------------------------------------------------

def apply_spherical_lens(
    E_in: np.ndarray,
    *,
    R1: float,
    R2: float,
    d: float,
    n_lens: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    aperture_diameter: Optional[float] = None,
    xc: float = 0,
    yc: float = 0,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Apply a THIN-ELEMENT phase screen for a singlet with spherical surfaces.

    Imprints the sag-projection OPD ``-(n-1)*(sag1(h) - sag2(h))`` of a
    two-surface singlet onto a SINGLE plane.  The sag is the full spherical
    sag (not its paraxial ``h**2/2R`` expansion), so the screen carries the
    quartic and every higher even order of the surface departure -- but the
    model is the paraxial thin-element screen, NOT an exact ray trace
    through the element: it never follows a ray inside the glass, it never
    reads the centre thickness ``d``, and it cannot distinguish the two
    orientations of a plano-convex singlet.  See "Validity boundary" below
    for the measured error, and use :func:`apply_real_lens` /
    :func:`apply_real_lens_traced` when real-lens accuracy is required.

    See Also
    --------
    lumenairy.elements.apply_real_lens :
        Per-SURFACE thin screens with exact angular-spectrum propagation
        through the glass between them (so thickness and surface order do
        act), plus ``surface_model='displaced'`` for the ray-angle
        obliquity term this single screen drops.
    lumenairy.elements.apply_real_lens_traced :
        Per-pixel ray-traced OPL + wave-optics amplitude envelope; the
        reference when the OPD tolerance is tighter than the bound below.

    Parameters
    ----------
    E_in : ndarray (complex), shape (Ny, Nx)
        Input electric field.
    R1 : float
        Radius of curvature of the front surface [m].
        Positive = center of curvature on the transmission side (convex
        toward input).  ``np.inf`` for a flat surface.
    R2 : float
        Radius of curvature of the back surface [m].
        Negative = center of curvature on the input side (convex toward
        output).  Example: biconvex lens has R1 > 0, R2 < 0.
    d : float
        Center thickness [m].  ACCEPTED FOR SIGNATURE COMPATIBILITY BUT NOT
        USED BY THIS MODEL: the single-plane screen
        ``-(n-1)*(sag1 - sag2)`` has no ``d`` term, so ``d=1e-9`` and
        ``d=1.0`` return BIT-IDENTICAL fields (measured; adversarial audit
        2026-07-25, finding E-C2 -- pinned in
        ``tests/unit/test_niche_audit_ec_thin_lens_claims.py``).  It stays
        required so the call site records the physical element and the
        signature matches :func:`apply_aspheric_lens`; if the thickness
        must actually act on the field, use :func:`apply_real_lens`.
    n_lens : float
        Refractive index of the lens material.
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    aperture_diameter : float or None
        Clear aperture diameter [m].  If None the aperture is set by the
        surface radii of curvature (``h < 0.9999 * min(|R1|, |R2|)``).

        A sphere of radius ``R`` exists only for ``h < |R|``.  Requesting
        an ``aperture_diameter`` larger than ``2 * min(|R1|, |R2|)``
        therefore asks for a surface that is not there, and those pixels
        come back **NaN** (v5.30, audit E-M8 -- they used to come back
        with a clamped, finite sag of ``0.99 R`` and ``|E| = 1``, a
        transmission through a nonexistent surface).  NaN is the
        library-wide out-of-domain convention: it matches
        :func:`apply_aspheric_lens`,
        :func:`lumenairy.elements.lenses.surface_sag_general` and
        :func:`lumenairy.raytrace.conic_sag`, and it makes the
        contradiction loud instead of silent.  Keep the aperture inside
        the surface domain, or mask the NaNs yourself.
    xc, yc : float
        Lens center [m].
    use_gpu : bool
        Use GPU if available.

    Returns
    -------
    E_out : ndarray (complex), same shape as *E_in*

    Notes
    -----
    Geometry.  The thickness profile of the physical element is
    ``t(h) = d - sag1(h) + sag2(h)`` (the glass spans ``z`` in
    ``[sag1(h), d + sag2(h)]``) where each signed sag is
    ``sag(h) = R - sign(R) * sqrt(R**2 - h**2)``.  Only the ``h``-varying
    part of that profile enters the screen; the constant ``d`` piston does
    not (see the ``d`` parameter).

    The phase this function imprints is exactly

        delta_phi(h) = -k * (n - 1) * (sag1(h) - sag2(h))

    i.e. it is an EXACT evaluation of the sag-projection OPD, and reduces
    to ``-k/(2f) * h**2`` in the paraxial limit with
    ``1/f = (n-1) * (1/R1 - 1/R2)`` (lensmaker's equation).  Exactness of
    that formula is not exactness of the LENS -- see below.

    Validity boundary
    -----------------
    This is a **normal-projected thin phase screen**: ``sag(x, y)`` is the
    axial (z) surface departure and the OPD is imprinted on a single axial
    plane.  It is the same per-surface formula :func:`apply_real_lens`
    uses (see that function's "Oblique validity boundary" section for the
    general statement), with the extra simplification that BOTH surfaces
    are collapsed onto ONE plane with no propagation between them.  A thin
    screen collapses the finite ray traverse through the sag onto that
    plane, so the residual OPD error scales as the leading obliquity term
    ``~ sag * theta**2``, where ``theta`` is the local ray angle.  The
    bound is therefore **design-dependent**: it grows with fast (high-NA)
    surfaces, large sag and off-axis fields, and shrinks toward the axis
    and for slow surfaces.

    Measured against an exact meridional Snell + eikonal ray trace
    (R = 50 mm N-BK7 singlet, n = 1.51509 at 632.8 nm, collimated on-axis
    input, every wavefront referenced to its own best-fit exact sphere;
    adversarial audit 2026-07-25, finding E-C2):

    * Magnitude.  Screen-vs-trace PV error 0.011 waves at f/16, 0.18 at
      f/8, 3.88 at f/3.9, and 21.7 waves at f/2.0 -- a good wavefront
      model for slow elements only.
    * Orientation blindness.  The screen sees ``sag1 - sag2`` only, so
      flipping a plano-convex singlet end-for-end moves its output by 1.7%
      (7.82 -> 7.95 waves PV) where the true aberration moves 4.0x
      (3.94 -> 15.67 waves PV).  Orientation, thickness and internal
      propagation studies therefore need :func:`apply_real_lens`
      (per-surface screens + in-glass propagation;
      ``surface_model='displaced'`` restores the obliquity term and the
      ~4x orientation split) or :func:`apply_real_lens_traced` (per-pixel
      ray-traced OPL).
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper -- siblings missed by the
    # v4.15.2 closure now share the same first-line guard.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_spherical_lens')

    # See apply_thin_lens for the ``_lenses_module.cp`` rationale.
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if _lenses_module.cp is None:
            _lenses_module._ensure_cupy_loaded()
        _cp = _lenses_module.cp
        xp = _cp
        if not _is_cupy_array(E_in):
            E_in = _cp.asarray(E_in)
    else:
        xp = np

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx

    k = 2 * np.pi / wavelength

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    h_sq = (X - xc) ** 2 + (Y - yc) ** 2

    def _surface_sag(h_sq, R):
        """Signed spherical sag: positive for convex (R > 0).

        NaN outside the sphere's domain (``h_sq / R**2 >= 0.9999``) --
        the family convention (v5.30, audit E-M8).
        """
        if R is None or np.isinf(R):
            return xp.zeros_like(h_sq)
        # v5.30 (audit E-M8): a sphere of radius R simply does not exist
        # beyond ``h = |R|``, so return NaN there and let the aperture mask
        # zero those pixels -- exactly what the aspheric sibling
        # ``apply_aspheric_lens._aspheric_sag`` and the canonical
        # ``lenses.surface_sag_general`` / ``raytrace.conic_sag`` helpers do.
        # This function used to CLAMP ``h_sq`` to ``0.9999 R**2``, which
        # saturates the sag at a finite ``0.99 R`` and therefore imprinted a
        # unit-magnitude phase screen for a surface that is not there.  With
        # ``aperture_diameter`` larger than ``2|R|`` those pixels survived the
        # aperture: measured at R = 10 mm, aperture_diameter = 28 mm, N = 256,
        # dx = 120 um, 20916 out-of-domain pixels left the function with
        # |E| = 1.000000 and a bogus sag of 9.9 mm, while the aspheric sibling
        # on the SAME geometry returned NaN at exactly those 20916 pixels.
        # The ``aperture_diameter=None`` branch below already zeroes
        # ``h_sq >= 0.9999 * min(R1**2, R2**2)``, so this changes nothing
        # there (``where`` selects the zero, not the NaN).
        norm = h_sq / R ** 2
        valid = norm < 0.9999
        h_sq_safe = xp.where(valid, h_sq, 0.0)
        sag = R - np.sign(R) * xp.sqrt(R ** 2 - h_sq_safe)
        return xp.where(valid, sag, xp.nan)

    sag1 = _surface_sag(h_sq, R1)
    sag2 = _surface_sag(h_sq, R2)

    phase = -k * (n_lens - 1) * (sag1 - sag2)
    lens_field = xp.exp(1j * phase)

    # Clear aperture
    # v4.13.2 (audit C-P1-4): dtype-aware zero so a complex64 E_in
    # stays complex64 (was silently upcasting to complex128 via the
    # ``0.0 + 0.0j`` literal).  Matches the apply_aperture / apply_mirror
    # template added in v4.13.1 P3 #21.
    if aperture_diameter is not None:
        lens_field = xp.where(
            h_sq <= (aperture_diameter / 2) ** 2, lens_field,
            xp.zeros((), dtype=lens_field.dtype)
        )
    else:
        max_h_sq = np.inf
        if not np.isinf(R1):
            max_h_sq = min(max_h_sq, R1 ** 2)
        if not np.isinf(R2):
            max_h_sq = min(max_h_sq, R2 ** 2)
        if max_h_sq < np.inf:
            lens_field = xp.where(
                h_sq < max_h_sq * 0.9999, lens_field,
                xp.zeros((), dtype=lens_field.dtype)
            )

    # v4.13.2 (audit C-P1-5): coerce lens_field to E_in.dtype so
    # complex64 inputs stay complex64.
    if lens_field.dtype != E_in.dtype:
        lens_field = lens_field.astype(E_in.dtype)

    return E_in * lens_field


# ---------------------------------------------------------------------------
# Thick aspheric singlet (conic + even polynomial)
# ---------------------------------------------------------------------------

def apply_aspheric_lens(
    E_in: np.ndarray,
    *,
    R1: float,
    R2: float,
    d: float,
    n_lens: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    k1: float = 0,
    k2: float = 0,
    A1: Optional[Dict[int, float]] = None,
    A2: Optional[Dict[int, float]] = None,
    aperture_diameter: Optional[float] = None,
    xc: float = 0,
    yc: float = 0,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Apply a THIN-ELEMENT phase screen for a conic / aspheric singlet.

    The conic-plus-polynomial twin of :func:`apply_spherical_lens` and the
    same model: the sag-projection OPD ``-(n-1)*(sag1(h) - sag2(h))``
    imprinted on a SINGLE plane.  The sag evaluation is exact for the
    prescribed surfaces, but the element model is the paraxial thin-element
    screen -- no ray is followed inside the glass, the centre thickness
    ``d`` is not read, and the two orientations of a plano-convex singlet
    are indistinguishable.  Read :func:`apply_spherical_lens`'s "Validity
    boundary" section (it applies verbatim) and "SA-nulling conics" below
    before using this function to design an asphere; for real-lens accuracy
    use :func:`apply_real_lens` / :func:`apply_real_lens_traced`.

    Each surface follows the standard aspheric sag equation:

        sag(h) = h**2 / (R * (1 + sqrt(1 - (1+k)*h**2/R**2)))
                 + A4*h**4 + A6*h**6 + A8*h**8 + A10*h**10

    Parameters
    ----------
    E_in : ndarray (complex), shape (Ny, Nx)
        Input electric field.
    R1, R2 : float
        Radii of curvature [m] (same sign convention as
        :func:`apply_spherical_lens`).
    d : float
        Center thickness [m].  ACCEPTED FOR SIGNATURE COMPATIBILITY BUT NOT
        USED BY THIS MODEL -- ``d=1e-9`` and ``d=1.0`` return BIT-IDENTICAL
        fields (measured; adversarial audit 2026-07-25, finding E-C2).  See
        the same parameter on :func:`apply_spherical_lens`.
    n_lens : float
        Refractive index at the operating wavelength.
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    k1, k2 : float
        Conic constants for surfaces 1 and 2 (0 = sphere, -1 = paraboloid).
    A1, A2 : dict or None
        Even aspheric polynomial coefficients for each surface.
        Keys are the powers of h: ``{4: A4, 6: A6, 8: A8, 10: A10}``.
    aperture_diameter : float or None
        Clear aperture diameter [m].
    xc, yc : float
        Lens center [m].
    use_gpu : bool
        Use GPU if available.

    Returns
    -------
    E_out : ndarray (complex), same shape as *E_in*

    Notes
    -----
    With ``k1=k2=0`` and ``A1=A2=None`` this reduces to
    :func:`apply_spherical_lens`, whose "Validity boundary" section applies
    unchanged here (single-plane normal-projected screen, ``d`` unused,
    orientation-blind, residual OPD error ``~ sag * theta**2``).

    SA-nulling conics: the real lens vs this screen
    -----------------------------------------------
    Two DIFFERENT conics are involved and this docstring used to conflate
    them: it prescribed the ``-n**2`` conic on the CURVED FIRST surface of a
    plano-convex lens as a third-order-SA null for collimated input.  That
    guidance is RETRACTED -- it names the wrong surface for a real lens AND
    the wrong value for this screen (adversarial audit 2026-07-25, finding
    E-C1).  Measured with an exact meridional Snell + eikonal trace
    (R = 50 mm, N-BK7 n = 1.51509 at 632.8 nm, semi-aperture 12.5 mm =
    f/3.9, collimated on-axis input, wavefront referenced to its best-fit
    exact sphere):

    * REAL LENS.  The ``k = -n**2`` hyperboloid belongs on the EXIT surface
      of a FLAT-FIRST plano-convex singlet, where it is exactly stigmatic
      for collimated on-axis input: measured PV 0.000000 waves (it is the
      Cartesian oval for an infinite object conjugate, eccentricity ``n``).
      Placing that same conic on the curved FIRST surface gives 10.38 waves
      PV against 3.94 for a plain sphere in that orientation -- 2.6x WORSE,
      not corrected.
    * THIS SCREEN.  Within the thin-screen model the SA-minimising conic on
      a curved-first singlet is ``k = -1 - (n_lens - 1)**2`` (= -1.2653 for
      N-BK7), where the screen's sphere-referenced PV collapses to ~0
      (401-point scan of k over [-2, -0.5]: argmin -1.265, PV 0.002 waves).
      That value is a property of the SCREEN, not of the lens: fed to the
      exact trace the same conic leaves 4.34 waves PV, and the exact
      trace's own curved-first optimum sits near k1 = -0.58.

    The two disagree because the screen is the normal (z) sag PROJECTION
    collapsed onto one plane: it nulls the sag-difference OPD rather than
    the traversed optical path, so its stationary point is displaced from
    the true one and does not move with surface order or thickness.  Use
    this function to imprint a PRESCRIBED asphere and to study the screen's
    own OPD; use :func:`apply_real_lens` (``surface_model='displaced'``) or
    :func:`apply_real_lens_traced` to design or verify an SA-nulled asphere.
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper -- siblings missed by the
    # v4.15.2 closure now share the same first-line guard.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_aspheric_lens')

    # See apply_thin_lens for the ``_lenses_module.cp`` rationale.
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if _lenses_module.cp is None:
            _lenses_module._ensure_cupy_loaded()
        _cp = _lenses_module.cp
        xp = _cp
        if not _is_cupy_array(E_in):
            E_in = _cp.asarray(E_in)
    else:
        xp = np

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx

    kw = 2 * np.pi / wavelength  # wavenumber (avoid shadowing conic k)

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    h_sq = (X - xc) ** 2 + (Y - yc) ** 2

    def _aspheric_sag(h_sq, R, k_conic, A_coeffs):
        """Signed aspheric sag for one surface."""
        if R is None or np.isinf(R):
            sag = xp.zeros_like(h_sq)
            if A_coeffs:
                for power, coeff in A_coeffs.items():
                    sag = sag + coeff * h_sq ** (power // 2)
            return sag

        R_abs = abs(R)
        norm_h_sq = h_sq / R_abs ** 2
        denom_arg = 1 - (1 + k_conic) * norm_h_sq
        # 4.10: clamp invalid (outside-conic-domain) pixels to NaN so a
        # downstream aperture mask explicitly zeros them, rather than
        # silently extrapolating a 1e-12 floor that produced
        # near-singular sag (1e6 m for typical optics) outside the
        # surface domain.
        # v5.30 (audit E-L8): gate on ``norm < 0.9999`` like every sibling
        # (``lenses.surface_sag_general``, ``lenses._conic_sag_xp``,
        # ``elements.apply_mirror``'s inline sag, ``raytrace.conic_sag``)
        # rather than on ``denom_arg > 0`` (i.e. ``norm < 1.0``).  The two
        # differ on the thin shell ``0.9999 <= norm < 1.0``, where the conic
        # denominator ``1 + sqrt(denom_arg)`` is within 1e-2 of its vertical
        # tangent and the sag is numerically meaningless; the siblings NaN it.
        # Measured (R = 10 mm sphere, N = 2048, dx = 10 um): 352 pixels land
        # in that shell and used to leave this function finite (|E| = 1.000000)
        # while ``surface_sag_general`` returned NaN for all 352.  The
        # ``aperture_diameter=None`` branch below already uses the matching
        # ``h_sq < max_h_sq * 0.9999`` cut, so the two halves of this function
        # now agree on where the surface stops existing.
        valid = (1 + k_conic) * norm_h_sq < 0.9999
        denom_arg_safe = xp.where(valid, denom_arg, 1.0)
        sag_unsigned = h_sq / (R_abs * (1 + xp.sqrt(denom_arg_safe)))
        sag = np.sign(R) * sag_unsigned
        sag = xp.where(valid, sag, xp.nan)

        if A_coeffs:
            for power, coeff in A_coeffs.items():
                sag = sag + coeff * h_sq ** (power // 2)

        return sag

    sag1 = _aspheric_sag(h_sq, R1, k1, A1)
    sag2 = _aspheric_sag(h_sq, R2, k2, A2)

    phase = -kw * (n_lens - 1) * (sag1 - sag2)
    lens_field = xp.exp(1j * phase)

    # Apply aperture
    # v4.13.2 (audit C-P1-4): dtype-aware zero, see apply_spherical_lens
    # above for rationale.
    if aperture_diameter is not None:
        lens_field = xp.where(
            h_sq <= (aperture_diameter / 2) ** 2, lens_field,
            xp.zeros((), dtype=lens_field.dtype)
        )
    else:
        max_h_sq = np.inf
        if R1 is not None and not np.isinf(R1):
            if (1 + k1) > 0:
                max_h_sq = min(max_h_sq, R1 ** 2 / (1 + k1))
        if R2 is not None and not np.isinf(R2):
            if (1 + k2) > 0:
                max_h_sq = min(max_h_sq, R2 ** 2 / (1 + k2))
        if max_h_sq < np.inf:
            lens_field = xp.where(
                h_sq < max_h_sq * 0.9999, lens_field,
                xp.zeros((), dtype=lens_field.dtype)
            )

    # v4.13.2 (audit C-P1-5): coerce lens_field to E_in.dtype so
    # complex64 inputs stay complex64.
    if lens_field.dtype != E_in.dtype:
        lens_field = lens_field.astype(E_in.dtype)

    return E_in * lens_field




# ---------------------------------------------------------------------------

def apply_cylindrical_lens(
    E_in: np.ndarray,
    *,
    f: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    axis: str = 'x',
    xc: float = 0,
    yc: float = 0,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Apply a cylindrical thin-lens phase (focusing in one axis only).

    Parameters
    ----------
    E_in : ndarray (complex, N x N)
        Input electric field.
    f : float
        Focal length [m].  Positive = converging.
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    axis : ``'x'`` or ``'y'``
        Focusing axis.  ``'x'`` applies phi = -k/(2f) * (x - xc)**2;
        ``'y'`` applies phi = -k/(2f) * (y - yc)**2.
    xc, yc : float
        Lens center [m].
    use_gpu : bool
        If True and CuPy is available, run on the GPU.  Added in
        v4.13.2 (audit C-P1-6) so the module-docstring claim that
        "all functions accept use_gpu=False" is now true.

    Returns
    -------
    E_out : ndarray (complex, N x N)

    Notes
    -----
    Produces a line focus (orthogonal to the focusing axis) instead of a
    point focus.
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper (replaces the v4.15.2 inline
    # guard).
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_cylindrical_lens')

    # v4.13.2 (audit C-P1-6): dispatch through CuPy when use_gpu=True
    # or E_in is already a CuPy array.  Resolve ``cp`` via the
    # _lenses_module lazy slot rather than a bare global (which is
    # not bound in this module's namespace).  Pre-fix the three
    # sibling functions had no use_gpu path at all.
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if _lenses_module.cp is None:
            _lenses_module._ensure_cupy_loaded()
        _cp = _lenses_module.cp
        xp = _cp
        if not _is_cupy_array(E_in):
            E_in = _cp.asarray(E_in)
    else:
        xp = np

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx
    k = 2 * np.pi / wavelength

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy

    if axis == 'x':
        phase_1d = -k / (2 * f) * (x - xc) ** 2
        phase = phase_1d[None, :]
    elif axis == 'y':
        phase_1d = -k / (2 * f) * (y - yc) ** 2
        phase = phase_1d[:, None]
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    # v4.13.2 (audit C-P1-5): cast the phase mask to E_in.dtype so a
    # complex64 input stays complex64 (xp.exp(1j*phase) returns
    # complex128 from float64 phase regardless of E_in.dtype).
    phase_exp = xp.exp(1j * phase)
    if phase_exp.dtype != E_in.dtype:
        phase_exp = phase_exp.astype(E_in.dtype)

    return E_in * phase_exp


# ---------------------------------------------------------------------------
# GRIN lens
# ---------------------------------------------------------------------------

def apply_grin_lens(
    E_in: np.ndarray,
    *,
    n0: float,
    g: float,
    d: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    xc: float = 0,
    yc: float = 0,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Apply a gradient-index (GRIN) rod lens phase (thin approximation).

    Models a GRIN rod with parabolic index profile:

        n(r) = n0 * (1 - g**2 / 2 * r**2)

    Parameters
    ----------
    E_in : ndarray (complex, N x N)
        Input electric field.
    n0 : float
        On-axis refractive index.
    g : float
        Gradient constant [1/m] (also called sqrt(A)).
        Pitch P = 2 pi / g.
    d : float
        Rod length (thickness) [m].
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    xc, yc : float
        GRIN lens center [m].
    use_gpu : bool
        If True and CuPy is available, run on the GPU.  Added in
        v4.13.2 (audit C-P1-6) to honour the module-docstring claim.

    Returns
    -------
    E_out : ndarray (complex, N x N)

    Notes
    -----
    The quadratic OPD through the rod gives an effective focal length

        f = 1 / (n0 * g**2 * d)      (thin approximation, g*d << 1)

    For longer rods the exact result is ``f = 1 / (n0 * g * sin(g*d))``.
    Quarter-pitch (g*d = pi/2) collimates a point source at the front face;
    half-pitch (g*d = pi) reimages 1:1 inverted.
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper -- siblings missed by the
    # v4.15.2 closure now share the same first-line guard.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_grin_lens')

    # v4.13.2 (audit C-P1-6): CuPy dispatch (was previously numpy-only).
    # See apply_cylindrical_lens above for the _lenses_module.cp
    # resolution rationale.
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if _lenses_module.cp is None:
            _lenses_module._ensure_cupy_loaded()
        _cp = _lenses_module.cp
        xp = _cp
        if not _is_cupy_array(E_in):
            E_in = _cp.asarray(E_in)
    else:
        xp = np

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx
    k = 2 * np.pi / wavelength

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    r_sq = (X - xc) ** 2 + (Y - yc) ** 2

    phase = -k * n0 * (g ** 2 / 2) * d * r_sq
    # v4.13.2 (audit C-P1-5): cast phase mask to E_in.dtype so
    # complex64 inputs stay complex64.
    phase_exp = xp.exp(1j * phase)
    if phase_exp.dtype != E_in.dtype:
        phase_exp = phase_exp.astype(E_in.dtype)
    return E_in * phase_exp


# ---------------------------------------------------------------------------
# Axicon
# ---------------------------------------------------------------------------

def apply_axicon(
    E_in: np.ndarray,
    alpha: float,
    n_axicon: Union[float, str],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    xc: float = 0,
    yc: float = 0,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Apply an axicon (conical lens) phase to generate a Bessel-like beam.

    Parameters
    ----------
    E_in : ndarray (complex, N x N)
        Input electric field.
    alpha : float
        Physical half-angle of the cone [radians].
        Typical range: 0.5--5 degrees (0.009--0.087 rad).
    n_axicon : float or str
        Refractive index of the axicon material.  If a string is passed it
        is resolved via :func:`get_glass_index`.
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    xc, yc : float
        Axicon center [m].
    use_gpu : bool
        If True and CuPy is available, run on the GPU.  Added in
        v4.13.2 (audit C-P1-6) to honour the module-docstring claim.

    Returns
    -------
    E_out : ndarray (complex, N x N)

    Notes
    -----
    The axicon imparts a phase linear in radial distance:

        phi(r) = -k * (n - 1) * alpha * r

    A collimated input beam produces a non-diffracting Bessel-beam region
    extending over ``z_max ~ w0 / ((n - 1) * alpha)`` where *w0* is the
    input beam radius.

    Warning
    -------
    Unlike the ``apply_*_lens`` family, this function's arguments are
    **positional-or-keyword** (the 4.7 keyword-only conversion covered
    the eight lens entry points only -- see the "Scope of that
    guarantee" note in :func:`apply_thin_lens`).  ``alpha``,
    ``wavelength``, ``dx`` and ``dy`` are all small floats, so a
    transposed positional call binds silently and produces a plausible
    but wrong cone.  Pass them by keyword.
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper -- siblings missed by the
    # v4.15.2 closure now share the same first-line guard.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_axicon')

    # v4.13.2 (audit C-P1-6): CuPy dispatch.  See
    # apply_cylindrical_lens above for the _lenses_module.cp
    # resolution rationale.
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if _lenses_module.cp is None:
            _lenses_module._ensure_cupy_loaded()
        _cp = _lenses_module.cp
        xp = _cp
        if not _is_cupy_array(E_in):
            E_in = _cp.asarray(E_in)
    else:
        xp = np

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx
    k = 2 * np.pi / wavelength

    if isinstance(n_axicon, str):
        n = get_glass_index(n_axicon, wavelength)
    else:
        n = float(n_axicon)

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    r = xp.sqrt((X - xc) ** 2 + (Y - yc) ** 2)

    phase = -k * (n - 1) * alpha * r
    # v4.13.2 (audit C-P1-5): cast phase mask to E_in.dtype so
    # complex64 inputs stay complex64.
    phase_exp = xp.exp(1j * phase)
    if phase_exp.dtype != E_in.dtype:
        phase_exp = phase_exp.astype(E_in.dtype)
    return E_in * phase_exp




__all__ = [
    'apply_thin_lens',
    'apply_spherical_lens',
    'apply_aspheric_lens',
    'apply_cylindrical_lens',
    'apply_axicon',
    'apply_grin_lens',
]
