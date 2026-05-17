"""
Diffraction efficiencies for 1-D binary phase gratings.

This is the **analytical scalar thin-phase grating** model: a
closed-form Fourier-series solution that is correct and
energy-conserving for low-contrast gratings with periods much larger
than the wavelength.  The module name reflects the physics actually
implemented; a true Rigorous Coupled-Wave Analysis (RCWA) -- which
would handle high-contrast / sub-wavelength gratings with polarisation
and AOI effects -- is not implemented here.  Reflection is assumed zero
(exact for a lossless thin phase grating, an approximation for deep /
high-contrast gratings).

The analytical formula implemented here is the standard Fourier-series
thin-grating result:

    t_m = f * exp(i*phi) * f * sinc(pi m f) + (1-f) * sinc(pi m (1-f))
          * ...  (see code for exact form)
    eta_m = |t_m|^2

which sums to unity by Parseval's theorem for a pure phase grating.

See ``REFERENCES.txt`` Sections A and G.

Limitations
-----------
* 1-D gratings only (2-D would need block-Toeplitz matrices).
* Isotropic, non-magnetic materials.
* Single-layer grating (binary profile).
* Thin-grating scalar approximation -- R always zero.

Author: Andrew Traverso
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np


def thin_grating_efficiency_1d(
    period: float,
    n_ridge: Union[float, complex],
    n_groove: Union[float, complex],
    n_substrate: float,
    n_superstrate: float,
    depth: float,
    duty_cycle: float,
    wavelength: float,
    angle: float = 0.0,
    polarization: str = 'te',
    n_orders: int = 11,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diffraction efficiencies for a 1-D binary phase grating
    (analytical thin-grating scalar approximation).

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
    exp_r = np.exp(1j * phi_r)
    exp_g = np.exp(1j * phi_g)

    # Analytical Fourier coefficients of t(x), vectorised across m:
    #   m == 0:  t_0 = f * exp(i*phi_r) + (1-f) * exp(i*phi_g)
    #   m != 0:  t_m = (exp(i*phi_r) - exp(i*phi_g))
    #                 * (exp(-i*2*pi*m*f) - 1) / (-i * 2 * pi * m)
    # The m != 0 expression is evaluated over the full orders vector
    # then patched at the m == 0 index, which is faster than a Python
    # loop for the order counts the GUI / sweeps typically use
    # (n_orders >= 11, i.e. >= 23 evaluations).
    m_nonzero = np.where(orders == 0, 1, orders)  # avoid divide-by-zero
    tm = ((exp_r - exp_g)
          * (np.exp(-1j * 2 * np.pi * m_nonzero * f) - 1)
          / (-1j * 2 * np.pi * m_nonzero))
    # Patch the m == 0 entry (analytical limit of the above as m -> 0
    # is f*exp(i*phi_r) + (1-f)*exp(i*phi_g)).
    zero_idx = n_orders  # orders[n_orders] == 0 by construction
    tm[zero_idx] = f * exp_r + (1 - f) * exp_g

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


def grating_efficiency_vs_wavelength(
    period: float,
    n_ridge: Union[float, complex],
    n_groove: Union[float, complex],
    n_substrate: float,
    n_superstrate: float,
    depth: float,
    duty_cycle: float,
    wavelengths: Union[float, np.ndarray],
    order: int = 1,
    polarization: str = 'te',
    n_orders: int = 11,
) -> np.ndarray:
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
        orders, _, T = thin_grating_efficiency_1d(
            period, n_ridge, n_groove, n_substrate, n_superstrate,
            depth, duty_cycle, wl, polarization=polarization,
            n_orders=n_orders)
        idx = np.argmin(np.abs(orders - order))
        eff[i] = T[idx]
    return eff
