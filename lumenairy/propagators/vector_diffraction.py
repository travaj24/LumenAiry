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
    pupil = np.asarray(pupil, dtype=np.complex128)
    Np = pupil.shape[0]
    k = 2 * np.pi / wavelength
    theta_max = np.arcsin(min(NA, 0.9999))

    if N_focal is None:
        N_focal = Np
    if dx_focal is None:
        dx_focal = wavelength / (4.0 * NA)
    if z_planes is None:
        z_planes = np.array([0.0])
    else:
        z_planes = np.atleast_1d(np.asarray(z_planes, dtype=np.float64))

    # Focal coordinates
    x_f = (np.arange(N_focal) - N_focal / 2) * dx_focal
    y_f = (np.arange(N_focal) - N_focal / 2) * dx_focal
    Xf, Yf = np.meshgrid(x_f, y_f)
    rho_f = np.sqrt(Xf ** 2 + Yf ** 2)
    phi_f = np.arctan2(Yf, Xf)

    # Pupil coordinates -> angular mapping
    x_p = (np.arange(Np) - Np / 2) * dx_pupil
    y_p = (np.arange(Np) - Np / 2) * dx_pupil
    Xp, Yp = np.meshgrid(x_p, y_p)
    rho_p = np.sqrt(Xp ** 2 + Yp ** 2)
    # Map pupil radius to convergence angle: sin(theta) = rho / f
    sin_theta = np.clip(rho_p / f, 0, np.sin(theta_max))
    theta = np.arcsin(sin_theta)
    cos_theta = np.cos(theta)
    phi_p = np.arctan2(Yp, Xp)

    # Aplanatic apodisation factor: sqrt(cos(theta))
    apod = np.sqrt(np.maximum(cos_theta, 0))
    # Mask to exit pupil
    in_pupil = sin_theta < np.sin(theta_max)
    P = pupil * apod * in_pupil

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

    # Compute focal field via 2-D FFT for each z-plane
    n_z = len(z_planes)
    Ex = np.zeros((n_z, N_focal, N_focal), dtype=np.complex128)
    Ey = np.zeros_like(Ex)
    Ez = np.zeros_like(Ex)

    for iz, z in enumerate(z_planes):
        defocus = np.exp(1j * k * z * ct)
        # Zero-pad or crop pupil arrays to N_focal for FFT
        def _fft_field(P_comp):
            if N_focal >= Np:
                pad = (N_focal - Np) // 2
                padded = np.pad(P_comp * defocus,
                                ((pad, N_focal - Np - pad),
                                 (pad, N_focal - Np - pad)),
                                mode='constant')
            else:
                c0 = (Np - N_focal) // 2
                padded = (P_comp * defocus)[c0:c0 + N_focal,
                                            c0:c0 + N_focal]
            return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))

        Ex[iz] = _fft_field(Px)
        Ey[iz] = _fft_field(Py)
        Ez[iz] = _fft_field(Pz)

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
    Ex, Ey, Ez, x_f, y_f = richards_wolf_focus(
        pupil, wavelength, NA, f, dx_pupil,
        N_focal=N_focal, dx_focal=dx_focal,
        z_planes=z_planes, polarization=polarization)
    psf = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2
    return psf, x_f, y_f
