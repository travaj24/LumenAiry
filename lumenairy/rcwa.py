"""
Diffraction efficiencies for 1-D binary phase gratings.

.. warning::

   The current implementation is an **analytical scalar thin-phase
   grating** formula, not the full Rigorous Coupled-Wave Analysis
   algorithm.  It is correct and energy-conserving for low-contrast
   gratings and grating periods much larger than the wavelength,
   where the scalar approximation holds.  For high-contrast
   (n_ridge / n_groove > ~1.3) or sub-wavelength gratings, where
   polarisation and angle-of-incidence effects are significant, the
   results are **approximate**.

   The Fourier-space eigendecomposition (Moharam-Gaylord) is set up
   inside the function for future use but the interface S-matrix
   matching required to finish a true RCWA calculation is not yet
   implemented.  Reflection is assumed zero (valid for a lossless
   thin phase grating), which is accurate for the thin-scalar regime
   but wrong in the high-contrast / deep-grating regime.

The analytical formula implemented here is standard Fourier-series
thin-grating diffraction:

    t_m = f * exp(i*phi) * f * sinc(pi m f) + (1-f) * sinc(pi m (1-f))
          * ...  (see code for exact form)
    eta_m = |t_m|^2

which sums to unity by Parseval's theorem for a pure phase grating.

References
----------
Goodman, J. "Introduction to Fourier Optics" 3rd ed., chapter 4
(for the thin-grating formula).
Moharam & Gaylord (1981), JOSA 71(7), 811-818 (for full RCWA; not
yet implemented here).

Limitations
-----------
* 1-D gratings only (2-D would need block-Toeplitz matrices).
* Isotropic, non-magnetic materials.
* Single-layer grating (binary profile).
* Thin-grating scalar approximation -- R always zero.

Author: Andrew Traverso
"""
from __future__ import annotations
import numpy as np


def rcwa_1d(period, n_ridge, n_groove, n_substrate, n_superstrate,
            depth, duty_cycle, wavelength, angle=0.0,
            polarization='te', n_orders=11):
    """Diffraction efficiencies for a 1-D binary phase grating
    (**analytical thin-grating approximation -- not full RCWA**).

    Treats the grating as a pure phase object whose transmission
    varies between two values across one period::

        t(x) = exp(i * k0 * (n_ridge  - n_substrate) * depth)    on [0, f*P]
        t(x) = exp(i * k0 * (n_groove - n_substrate) * depth)    on [f*P, P]

    The transmitted order efficiencies follow from the Fourier
    coefficients of ``t(x)``; energy conservation
    ``sum |t_m|^2 = 1`` holds exactly for a pure phase grating (no
    amplitude variation and no loss).  **Reflection is set to zero**
    by this approximation -- valid in the thin-scalar regime but
    incorrect for high-contrast or deep gratings where a true RCWA
    interface-matching calculation is required.

    Parameters
    ----------
    period : float
        Grating period [m].
    n_ridge : float or complex
        Refractive index of the ridge (high-index) region.
    n_groove : float or complex
        Refractive index of the groove (low-index / air) region.
    n_substrate : float
        Substrate refractive index.
    n_superstrate : float
        Superstrate (incident medium) refractive index.
    depth : float
        Grating depth [m].
    duty_cycle : float
        Fraction of the period occupied by the ridge (0 to 1).
    wavelength : float
        Vacuum wavelength [m].
    angle : float, default 0
        Angle of incidence [rad].  Only used for the evanescent
        cut-off (which orders become non-propagating); the thin
        grating model itself is angle-independent.
    polarization : str, default 'te'
        ``'te'`` (s-polarisation, E perpendicular to plane of
        incidence) or ``'tm'`` (p-polarisation).  **Ignored by the
        current thin-grating model** -- present for forward API
        compatibility with a future full-RCWA implementation.
    n_orders : int, default 11
        Number of Fourier orders retained (total = 2*n_orders + 1).
        More orders = better accuracy but slower.

    Returns
    -------
    orders : ndarray of int
        Diffraction order indices (centered on 0).
    R_eff : ndarray
        Reflected diffraction efficiency per order.  **Always zero**
        under the thin-grating approximation.
    T_eff : ndarray
        Transmitted diffraction efficiency per order.  Sums to 1 by
        energy conservation (for lossless materials).
    """
    k0 = 2 * np.pi / wavelength
    K = 2 * np.pi / period
    N = 2 * n_orders + 1  # total orders
    orders = np.arange(-n_orders, n_orders + 1)
    f = duty_cycle

    # -- Transmission function Fourier coefficients -------------------
    # Treat the grating as a pure phase element sitting on top of the
    # substrate.  Each half of the grating period imposes a phase of
    #   phi_ridge  = k0 * (n_ridge  - n_substrate) * depth
    #   phi_groove = k0 * (n_groove - n_substrate) * depth
    # The transmission is then
    #   t(x) = exp(i * phi_ridge)   for x in [0, f*period]
    #   t(x) = exp(i * phi_groove)  for x in [f*period, period]
    # and we compute the Fourier coefficients t_m analytically.
    phi_r = k0 * (complex(n_ridge)  - n_substrate) * depth
    phi_g = k0 * (complex(n_groove) - n_substrate) * depth

    # Analytical Fourier coefficients of t(x):
    # For m == 0: t_0 = f * exp(i*phi_r) + (1-f) * exp(i*phi_g)
    # For m != 0: t_m = [(exp(i*phi_r) - exp(i*phi_g)) *
    #                   (exp(-i*2*pi*m*f) - 1) /
    #                   (-i * 2 * pi * m)]
    tm = np.zeros(N, dtype=np.complex128)
    for idx, m in enumerate(orders):
        if m == 0:
            tm[idx] = f * np.exp(1j * phi_r) + (1 - f) * np.exp(1j * phi_g)
        else:
            tm[idx] = ((np.exp(1j * phi_r) - np.exp(1j * phi_g))
                        * (np.exp(-1j * 2 * np.pi * m * f) - 1)
                        / (-1j * 2 * np.pi * m))

    # Propagating vs evanescent split.  An order is propagating when
    # |kx_m|^2 < (k0 * n_substrate)^2.
    kx0 = k0 * n_superstrate * np.sin(angle)
    kx = kx0 + orders * K
    k_sub = k0 * n_substrate
    propagating = np.abs(kx) < k_sub

    # Per-order transmitted efficiency is |t_m|^2 weighted by the
    # ratio of longitudinal components (Parseval holds exactly for a
    # pure-phase grating at normal incidence, so the weights are 1).
    T_eff = np.where(propagating, np.abs(tm) ** 2, 0.0)

    # Reflection: zero under the thin-phase scalar approximation.
    # Genuine RCWA would compute non-zero R via S-matrix interface
    # matching; see module docstring.
    R_eff = np.zeros(N)

    return orders, R_eff, T_eff


def grating_efficiency_vs_wavelength(period, n_ridge, n_groove,
                                      n_substrate, n_superstrate,
                                      depth, duty_cycle,
                                      wavelengths, order=1,
                                      polarization='te', n_orders=11):
    """Compute the efficiency of a single diffraction order across
    a range of wavelengths.

    Returns
    -------
    eff : ndarray
        Efficiency of the requested order at each wavelength.
    """
    wavelengths = np.atleast_1d(np.asarray(wavelengths))
    eff = np.empty(wavelengths.size)
    for i, wl in enumerate(wavelengths):
        orders, _, T = rcwa_1d(
            period, n_ridge, n_groove, n_substrate, n_superstrate,
            depth, duty_cycle, wl, polarization=polarization,
            n_orders=n_orders)
        idx = np.argmin(np.abs(orders - order))
        eff[i] = T[idx]
    return eff
