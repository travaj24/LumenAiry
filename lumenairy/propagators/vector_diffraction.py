"""
High-NA vector diffraction via the Richards-Wolf integral.

For objective lenses with NA > 0.5 the scalar-wave approximation
breaks down: the polarization state of the focused field varies
across the focal spot (longitudinal E_z component becomes
significant).  The Richards-Wolf integral computes the exact
vectorial electric field (Ex, Ey, Ez) near the focus of an
aplanatic objective by integrating plane-wave contributions over
the solid angle of the exit pupil.

This module provides:

* :func:`richards_wolf_focus` — compute (Ex, Ey, Ez) in a volume near
  focus from a scalar or Jones-vector pupil function.
* :func:`debye_wolf_psf` — convenience wrapper returning the
  intensity PSF ``|Ex|^2 + |Ey|^2 + |Ez|^2``.

References
----------
Richards, B. & Wolf, E. (1959), Proc. R. Soc. A 253, 358-379.

Author: Andrew Traverso
"""
from __future__ import annotations

import numpy as np


def richards_wolf_focus(pupil, wavelength, NA, f, dx_pupil,
                        N_focal=None, dx_focal=None,
                        z_planes=None, polarization='x'):
    """Compute the Richards-Wolf vectorial focal field.

    Parameters
    ----------
    pupil : ndarray, complex, shape (Ny, Nx)
        Scalar or x-polarised pupil function.  Amplitude = apodisation,
        phase = aberrations.  Should be zero outside the exit pupil.
    wavelength : float
        Vacuum wavelength [m].
    NA : float
        Numerical aperture of the objective.
    f : float
        Focal length [m].
    dx_pupil : float
        Pupil-plane grid spacing [m].
    N_focal : int, optional
        Focal-plane grid dimension.  Defaults to ``pupil.shape[0]``.

        .. warning::
           ``N_focal`` sets the FFT length, so ``N_focal < pupil.shape[0]``
           centre-CROPS the pupil and therefore truncates the APERTURE
           (effective NA ~ ``(N_focal*dx_pupil/2)/f``), not just the focal
           sampling -- a ``RuntimeWarning`` is emitted when the crop discards
           non-zero pupil content (S9-VD2).  Because the FFT-natural focal
           window ``N_focal * dx_focal = wavelength*f/dx_pupil`` does not
           depend on ``N_focal``, cropping gains no field of view: keep
           ``N_focal >= pupil.shape[0]`` and crop or bin the RETURNED focal
           field instead.
    dx_focal : float, optional
        Focal-plane grid spacing [m].  Default: ``wavelength / (4 * NA)``
        (Nyquist for the Airy pattern).
    z_planes : array-like of float, optional
        Axial positions [m] relative to focus at which to evaluate.
        Default: ``[0]`` (in-focus plane only).
    polarization : str or tuple
        ``'x'`` = x-polarised input (default), ``'y'`` = y-polarised,
        ``'circular'`` = right-circular.  Or a ``(px, py)`` Jones
        vector (complex) for arbitrary polarisation.

    Returns
    -------
    Ex, Ey, Ez : ndarray
        Complex field components at the focal plane(s).  Shape
        ``(N_z, N_focal, N_focal)`` if ``z_planes`` has multiple
        entries, or ``(N_focal, N_focal)`` for a single z.
    x_focal, y_focal : ndarray
        1-D focal coordinate arrays [m].
    """
    # v4.15.4 (P2-NEW-V2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  v4.15.3's walker visited
    # this file but the name regex (``apply_*`` / ``*_propagate*`` /
    # ``propagate``) did NOT match ``richards_wolf_focus``; v4.15.4
    # broadens the discovery via ``__all__`` so this site is now in
    # scope of the meta-pin.  Runs FIRST -- before the lazy import
    # of ``get_default_complex_dtype`` and the ``np.asarray(pupil)``
    # cast -- so PartialCoherenceMCF / 3-D ensemble inputs get the
    # canonical v4.16-roadmap message rather than a downstream
    # ``np.asarray`` cast error.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(pupil, 'richards_wolf_focus',
                           input_kind='pupil')

    # 4.11.2: honour precision='single' via the global default complex
    # dtype.  Pre-4.11.2 the input was always promoted to complex128,
    # silently negating ``set_default_complex_dtype(np.complex64)``.
    from .propagation import get_default_complex_dtype as _gdct
    pupil = np.asarray(pupil, dtype=_gdct())
    Np = pupil.shape[0]
    k = 2 * np.pi / wavelength
    # VD-1: an immersion NA (>= 1, e.g. oil 1.4) was silently clamped to the
    # 89.2 deg air cone, computing wrong physics with no diagnostic.  Reject
    # it -- immersion focusing needs an n_immersion parameter (not yet
    # supported): theta_max = arcsin(NA / n_immersion).
    if not (np.isfinite(NA) and 0.0 < NA < 1.0):
        raise ValueError(
            f"richards_wolf_focus: NA must be in (0, 1) for an air objective; "
            f"got {NA}.  Immersion NA (>= 1) is not yet supported (needs an "
            f"n_immersion parameter for theta_max = arcsin(NA / n_immersion)).")
    theta_max = np.arcsin(NA)

    if N_focal is None:
        N_focal = Np

    # 4.10: the underlying FFT fixes the focal-plane pitch to
    # dx_focal_fft = λ·f / (N_focal·dx_pupil)
    # regardless of any user-supplied dx_focal -- the latter only
    # relabels the axes.  Compute the actual FFT pitch and warn loudly
    # if the user passed a different value (was a silent mismatch
    # pre-4.10).  A true free-pitch focal plane requires a chirp-z
    # backend; see fresnel_propagate_mft / Bluestein.
    dx_focal_fft = wavelength * f / (N_focal * dx_pupil)
    if dx_focal is not None:
        if abs(dx_focal - dx_focal_fft) > dx_focal_fft * 1e-6:
            import warnings
            warnings.warn(
                f"richards_wolf_focus: caller-supplied dx_focal="
                f"{dx_focal:.3e} m differs from the FFT-natural pitch "
                f"{dx_focal_fft:.3e} m (= wavelength*f / (N_focal*dx_pupil)).  "
                f"The FFT only resolves the natural pitch; the caller-"
                f"supplied value is ignored.  For arbitrary focal pitch, "
                f"use a Bluestein-backed propagator.",
                RuntimeWarning, stacklevel=2,
            )
    dx_focal = dx_focal_fft

    if z_planes is None:
        z_planes = np.array([0.0])
    else:
        z_planes = np.atleast_1d(np.asarray(z_planes, dtype=np.float64))

    # Focal coordinates (at the actual FFT-natural pitch).
    # v5.4.6 (audit F-34): removed the dead focal polar coordinates
    # (rho_f = sqrt(Xf**2+Yf**2) and phi_f = arctan2(Yf, Xf)).  They were
    # computed from an Xf/Yf meshgrid and immediately discarded; the FFT-
    # based Richards-Wolf evaluation here never references focal-plane
    # polar coords.  Only x_f/y_f are needed (they are returned).
    x_f = (np.arange(N_focal) - N_focal / 2) * dx_focal
    y_f = (np.arange(N_focal) - N_focal / 2) * dx_focal

    # Pupil coordinates -> angular mapping
    x_p = (np.arange(Np) - Np / 2) * dx_pupil
    y_p = (np.arange(Np) - Np / 2) * dx_pupil
    Xp, Yp = np.meshgrid(x_p, y_p)
    rho_p = np.sqrt(Xp ** 2 + Yp ** 2)
    # Map pupil radius to convergence angle: sin(theta) = rho / f
    # 4.11.1: build the rim mask from the UNCLIPPED sin_theta_raw so
    # the geometric pupil is honoured.  Pre-4.11.1 clipped to
    # sin(theta_max) *before* the mask was built, making the mask
    # ``sin_theta <= sin(theta_max)`` identically True for every grid
    # pixel and silently extending the exit pupil to the whole array
    # (Richards-Wolf rim mask was effectively unenforced).
    sin_theta_raw = rho_p / f
    in_pupil = sin_theta_raw <= np.sin(theta_max)
    sin_theta = np.clip(sin_theta_raw, 0, np.sin(theta_max))
    theta = np.arcsin(sin_theta)
    cos_theta = np.cos(theta)
    phi_p = np.arctan2(Yp, Xp)

    # Aplanatic apodisation factor: sqrt(cos(theta)) is the energy-
    # conservation factor.  However, when we sample the pupil on a
    # Cartesian (x_p, y_p) grid and use an FFT, we are implicitly using
    # dx_p·dy_p as the area element.  The Richards-Wolf integral is
    # over solid angle dΩ = sin θ dθ dφ; the Cartesian-to-angular
    # Jacobian factor (∂(x_p,y_p)/∂Ω) is f²·cos θ.  Reconciling these,
    # the pupil weight under an FFT representation is
    #   sqrt(cos θ) (aplanatic) · 1/cos θ (Jacobian) = 1/sqrt(cos θ).
    # Pre-4.10 used only the aplanatic factor, so the effective
    # apodisation was cos^(3/2) θ instead of cos^(-1/2) θ -- biased
    # toward the centre, missing energy at the high-NA rim.
    cos_safe = np.maximum(cos_theta, 1e-12)
    apod = 1.0 / np.sqrt(cos_safe)
    # Mask to exit pupil (in_pupil built from sin_theta_raw above).
    P = pupil * apod * in_pupil

    # S9-VD2 (audit review4a, pattern #3 -- degenerate resample/crop branch):
    # the ``N_focal < Np`` branch of ``_fft_field`` below centre-CROPS the
    # PUPIL, so asking for a smaller focal grid silently truncates the
    # APERTURE (and hence the effective NA) instead of only coarsening the
    # focal sampling.  Note the FFT-natural focal WINDOW extent
    # ``N_focal * dx_focal = wavelength*f / dx_pupil`` is INDEPENDENT of
    # ``N_focal``, so the crop buys nothing but lost aperture.  Measured on a
    # NA=0.5 / f=4 mm / dx_pupil=20 um / Np=256 pupil (aperture radius 100 px):
    # the focal-plane Parseval energy falls to 0.504 / 0.279 / 0.123 of the
    # full-pupil value at N_focal = 128 / 96 / 64, tracking the predicted
    # clipped-pupil area fraction 0.522 / 0.294 / 0.130 (the few-% residual is
    # the 1/sqrt(cos theta) rim apodisation the crop preferentially discards).
    # The returned field is then the PSF of an aperture the caller never asked
    # for.  Values are unchanged (bit-identical); the breach is surfaced as a
    # RuntimeWarning in the same style as the VD-1 immersion-NA guard and the
    # 4.10 dx_focal mismatch warning -- and ONLY when the crop actually
    # discards non-zero pupil content, so a zero-padded pupil whose support
    # fits inside the crop stays silent.
    if N_focal < Np:
        _c0 = Np // 2 - N_focal // 2
        _P2 = float(np.sum(np.abs(P) ** 2))
        _kept = float(np.sum(np.abs(
            P[_c0:_c0 + N_focal, _c0:_c0 + N_focal]) ** 2))
        if _P2 > 0.0 and (_P2 - _kept) > 1e-12 * _P2:
            import warnings
            warnings.warn(
                f"richards_wolf_focus: N_focal={N_focal} < pupil size "
                f"Np={Np}, so the pupil is centre-CROPPED to {N_focal}x"
                f"{N_focal} before the FFT -- this discards "
                f"{100.0 * (1.0 - _kept / _P2):.1f}% of the exit-pupil energy "
                f"and clips the effective NA to about "
                f"{min(NA, (N_focal * dx_pupil / 2.0) / f):.3f} (requested "
                f"{NA:.3f}).  The FFT-natural focal window "
                f"N_focal*dx_focal = wavelength*f/dx_pupil is independent of "
                f"N_focal, so cropping buys no field of view.  Pass "
                f"N_focal >= Np (the default) and crop/bin the RETURNED focal "
                f"field instead, or rebin the pupil.",
                RuntimeWarning, stacklevel=2)

    # Polarisation Jones vector
    if isinstance(polarization, str):
        if polarization == 'x':
            px, py = 1.0, 0.0
        elif polarization == 'y':
            px, py = 0.0, 1.0
        elif polarization == 'circular':
            px, py = 1.0 / np.sqrt(2), 1j / np.sqrt(2)
        else:
            raise ValueError(f"Unknown polarization {polarization!r}")
    else:
        px, py = complex(polarization[0]), complex(polarization[1])

    # Richards-Wolf: three integrals I0, I1, I2
    # I0 = integral P * (1 + cos_theta) * J0(k*rho_f*sin_theta) * exp(ikz*cos_theta) * sin_theta dtheta dphi
    # I1 = integral P * sin_theta^2 * J2(...) * exp(...)
    # I2 = integral P * sin_theta * (1 - cos_theta) * ...
    #
    # For computational efficiency on a grid we use the 2-D FFT approach:
    # the Richards-Wolf integral is equivalent to a Fourier transform of
    # the pupil field weighted by the vectorial transfer functions.

    # Build the three vectorial weighting functions in the pupil
    c = cos_theta
    s = sin_theta
    cp = np.cos(phi_p)
    sp = np.sin(phi_p)

    # Vectorial pupil functions for x-polarisation input (px, py)
    # Tx_x = cos(phi)^2 (cos_t + 1) + sin(phi)^2 (1 - cos_t) ... simplified:
    # Using Leutenegger et al. (2006) notation:
    # For (px, py) input:
    #   pupil_x = P * [px*(cp^2*ct + sp^2) + py*cp*sp*(ct-1)]
    #   pupil_y = P * [px*cp*sp*(ct-1) + py*(sp^2*ct + cp^2)]
    #   pupil_z = P * [-px*cp*s - py*sp*s]

    ct = c
    Px = P * (px * (cp ** 2 * ct + sp ** 2)
              + py * cp * sp * (ct - 1))
    Py = P * (px * cp * sp * (ct - 1)
              + py * (sp ** 2 * ct + cp ** 2))
    Pz = P * (-px * cp * s - py * sp * s)

    # Compute focal field via 2-D FFT for each z-plane.
    # 4.11.2: allocate output in the input pupil's dtype so the
    # precision='single' path stays complex64 end-to-end.
    n_z = len(z_planes)
    Ex = np.zeros((n_z, N_focal, N_focal), dtype=pupil.dtype)
    Ey = np.zeros_like(Ex)
    Ez = np.zeros_like(Ex)

    for iz, z in enumerate(z_planes):
        defocus = np.exp(1j * k * z * ct)
        # Zero-pad or crop pupil arrays to N_focal for FFT
        #
        # S9-VD1 (audit review4a, pattern #1 -- interpolation/registration
        # anchor): the transform chain is
        # ``fftshift(fft2(ifftshift(padded)))``, and ``ifftshift`` treats index
        # ``L//2`` (FLOOR) as the array centre.  Preserving the pupil's
        # registration across the pad/crop therefore requires the pupil centre
        # index ``Np//2`` to land exactly on ``N_focal//2``:
        #     pad = N_focal//2 - Np//2       c0 = Np//2 - N_focal//2
        # Pre-S9 this used ``(N_focal - Np)//2`` / ``(Np - N_focal)//2``, which
        # is the SAME integer whenever ``Np`` and ``N_focal`` share parity but
        # is off by exactly one index when they do not -- specifically for
        # (Np odd, N_focal even) on the pad branch and (Np even, N_focal odd)
        # on the crop branch.  That one-index slip translates the whole masked
        # + apodised pupil by one pupil pixel, so the returned COMPLEX focal
        # field picks up a spurious linear phase of ``2*pi/N_focal`` rad per
        # focal pixel: measured 0.098175 rad/px at (Np, N_focal) = (33, 64)
        # and 0.369599 rad/px at (32, 17), both matching the prediction
        # ``2*pi*delta/N_focal`` to 6 digits, with the two fields differing by
        # 145% in L2.  ``debye_wolf_psf`` (intensity) is blind to it; coherent
        # superposition with a reference arm is not -- the same failure class
        # the 4.11.2 ``exp(+i k f)`` sign fix addressed.  Both new expressions
        # are IDENTICAL integers to the old ones for every same-parity
        # (Np, N_focal) pair, so every even/even case -- i.e. every realistic
        # call, including the ``N_focal is None -> Np`` default -- is
        # bit-identical.
        def _fft_field(P_comp):
            if N_focal >= Np:
                pad = N_focal // 2 - Np // 2
                padded = np.pad(P_comp * defocus,
                                ((pad, N_focal - Np - pad),
                                 (pad, N_focal - Np - pad)),
                                mode='constant')
            else:
                c0 = Np // 2 - N_focal // 2
                padded = (P_comp * defocus)[c0:c0 + N_focal,
                                            c0:c0 + N_focal]
            return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))

        Ex[iz] = _fft_field(Px)
        Ey[iz] = _fft_field(Py)
        Ez[iz] = _fft_field(Pz)

    # 4.11.2: Richards-Wolf scalar prefactor with the correct 1/f² intensity
    # scaling AND the correct global-phase sign for the library's exp(-iωt)
    # forward-propagation convention.
    #
    # The Richards-Wolf integral (Richards & Wolf 1959 Eq. 3.7) reads
    #
    #     E(P) = (-i k f / 2π) · exp(+i k f) ·
    #             ∫∫ E_∞(θ,φ) exp(i k·r) sin θ dθ dφ
    #
    # The pre-4.11.2 code used `exp(-i k f)`, which is opposite-sign to
    # every other forward-prop in the library (angular_spectrum_propagate,
    # fresnel_propagate, ... all use exp(+i k z) under exp(-iωt)).
    # Coherent superposition with a reference arm therefore picked up a
    # spurious exp(-2 i k f) phase mismatch.
    #
    # When the angular integral is evaluated by FFT over a Cartesian
    # pupil grid, the change of variables (θ,φ) → (k_x, k_y) with
    # k_x = k sin θ cos φ, k_y = k sin θ sin φ introduces a Jacobian
    # 1/(k² cos θ), and the pupil-to-aperture mapping x_pupil = f sin θ
    # makes dk_x = (k/f) dx_pupil.  The two combined give
    #
    #     E(P) = (-i k / (2π f)) · exp(+i k f) · dx_pupil² · FFT[...]
    #
    # i.e. the prefactor scales as 1/f (amplitude) → 1/f² (intensity).
    # Pre-4.11.2 the prefactor was (-i k f / 2π) · dx² · exp(-i k f),
    # which scaled as f (amplitude) → f² (intensity) — the WRONG sign of
    # the f-dependence: a 1 m focal length and a 1 cm focal length gave
    # Airy peak intensities differing by 10⁴ in the wrong direction.
    rw_prefactor = (-1j * k / (2.0 * np.pi * f)) * np.exp(1j * k * f)
    # Multiply by the Cartesian pupil area element (the FFT's implicit
    # discretisation factor).
    rw_prefactor = rw_prefactor * (dx_pupil * dx_pupil)
    Ex = Ex * rw_prefactor
    Ey = Ey * rw_prefactor
    Ez = Ez * rw_prefactor

    # Squeeze single z-plane
    if n_z == 1:
        Ex = Ex[0]
        Ey = Ey[0]
        Ez = Ez[0]

    return Ex, Ey, Ez, x_f, y_f


def debye_wolf_psf(pupil, wavelength, NA, f, dx_pupil,
                   N_focal=None, dx_focal=None, z_planes=None,
                   polarization='x'):
    """Compute the vectorial PSF intensity |Ex|^2 + |Ey|^2 + |Ez|^2.

    Parameters match :func:`richards_wolf_focus`.

    Returns
    -------
    psf : ndarray
        Intensity PSF, shape (N_focal, N_focal) or
        (N_z, N_focal, N_focal).
    x_focal, y_focal : ndarray
    """
    # v4.15.4 (P2-NEW-V2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  Runs FIRST so the
    # PartialCoherenceMCF / 3-D ensemble message is returned
    # from ``debye_wolf_psf`` (with the right function name in
    # the error) rather than being raised at the downstream
    # ``richards_wolf_focus`` callsite with a wrong fn_name.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(pupil, 'debye_wolf_psf',
                           input_kind='pupil')

    Ex, Ey, Ez, x_f, y_f = richards_wolf_focus(
        pupil, wavelength, NA, f, dx_pupil,
        N_focal=N_focal, dx_focal=dx_focal,
        z_planes=z_planes, polarization=polarization)
    psf = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2
    return psf, x_f, y_f
