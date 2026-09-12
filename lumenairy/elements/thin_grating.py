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
thin-grating result.  Writing ``phi_r`` / ``phi_g`` for the ridge / groove
phase steps and ``f`` for the duty cycle, the Fourier coefficients of the
binary transmittance are

    t_0 = f * exp(i*phi_r) + (1 - f) * exp(i*phi_g)
    t_m = (exp(i*phi_r) - exp(i*phi_g))
          * (exp(-i*2*pi*m*f) - 1) / (-i * 2 * pi * m)        (m != 0)
    eta_m = |t_m|^2

which sums to unity by Parseval's theorem for a pure phase grating.

See ``REFERENCES.txt`` Sections A and G.

Limitations
-----------
* 1-D gratings only (2-D would need block-Toeplitz matrices).
* Isotropic, non-magnetic materials.
* Single-layer grating (binary profile).
* Thin-grating scalar approximation -- R always zero.
* Raman-Nath (thin) regime only: the Klein-Cook parameter
  ``Q = 2 pi lambda d / (n_bar Lambda^2)`` must be <~ 1 and the period must
  be several wavelengths.  Outside that the grating is Bragg-like and the
  returned per-order numbers are meaningless (they still sum to 1, which is
  why the regime must be checked rather than inferred from energy closure);
  :func:`thin_grating_efficiency_1d` warns.  Hand off to ``rcwa`` / ``pmm``.

Author: Andrew Traverso
"""
from __future__ import annotations

import warnings
from typing import Tuple, Union

import numpy as np

# Klein-Cook regime thresholds.  Q = 2 pi lambda d / (n_bar Lambda^2)
# separates Raman-Nath (thin, Q <~ 1) from Bragg (Q >> 1) diffraction;
# Klein & Cook, IEEE J. Quantum Electron. QE-3, 59 (1967); Moharam & Young,
# Appl. Opt. 17, 1757 (1978).  The scalar thin-phase model additionally
# needs a period several wavelengths wide for the "each ray sees the local
# phase" picture to hold at all.
_KLEIN_COOK_Q_WARN = 1.0
_PERIOD_OVER_WAVELENGTH_WARN = 10.0


def _warn_thin_grating_validity(func_name, period, depth, wavelength,
                                n_ridge, n_groove, duty_cycle):
    """Warn when (period, depth, wavelength) sits outside the Raman-Nath
    regime the thin-phase Fourier model is valid in.

    Sibling of ``emt._warn_rytov_validity``: the failure is silent and
    clean-looking (the kept orders still sum to 1 by Parseval), so energy
    closure cannot be used as the tripwire.
    """
    p, d, wl = float(period), float(depth), float(wavelength)
    if p <= 0 or wl <= 0 or d <= 0:
        return
    f = float(np.clip(duty_cycle, 0.0, 1.0))
    n_bar = abs(f * complex(n_ridge).real + (1.0 - f) * complex(n_groove).real)
    n_bar = n_bar if n_bar > 1e-12 else 1.0
    q = 2 * np.pi * wl * d / (n_bar * p * p)
    if q > _KLEIN_COOK_Q_WARN:
        warnings.warn(
            f"{func_name}: Klein-Cook Q = 2*pi*lambda*depth/(n_bar*period^2) "
            f"= {q:.3g} exceeds {_KLEIN_COOK_Q_WARN:g} (n_bar = {n_bar:.4g}) "
            f"-- this is the Bragg regime, not the Raman-Nath (thin) regime "
            f"the scalar Fourier model assumes.  The returned per-order "
            f"efficiencies are not meaningful here even though they still "
            f"sum to 1 by Parseval.  Use a rigorous grating solver "
            f"(rcwa/pmm) at this depth and period.",
            UserWarning, stacklevel=3)
        return
    if p < _PERIOD_OVER_WAVELENGTH_WARN * wl:
        warnings.warn(
            f"{func_name}: period/wavelength = {p / wl:.3g} is below "
            f"{_PERIOD_OVER_WAVELENGTH_WARN:g} -- the scalar thin-phase "
            f"picture (every ray sees the local transmittance, no coupling "
            f"between ridge and groove) degrades as the period approaches "
            f"the wavelength, and the model's R = 0 assumption with it.  "
            f"Use a rigorous grating solver (rcwa/pmm) for a quantitative "
            f"answer at this period.",
            UserWarning, stacklevel=3)


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
        Transmitted diffraction efficiency per order.  v5.4.6 (audit
        F-24): the kept orders sum to ~1 by energy conservation ONLY
        when all retained orders propagate AND enough orders are kept;
        once orders go evanescent (``|kx_m| >= k0*n_substrate``) or are
        truncated by ``n_orders``, their efficiency is set to 0 and the
        sum of ``T_eff`` is < 1 (the missing power is in evanescent /
        dropped orders).

    Warns
    -----
    UserWarning
        When the Klein-Cook parameter ``Q = 2*pi*lambda*depth/(n_bar*
        period**2)`` exceeds 1 (Bragg regime, where the thin-phase model
        does not apply), or -- failing that -- when the period is under
        10 wavelengths.  Neither condition shows up in the returned
        numbers: ``sum(T_eff)`` stays 1 by Parseval whatever the regime, so
        energy closure is not a usable validity check here.
    """
    # v5.5.2: validate the polarization (accepting the s/p aliases) so a typo
    # is caught instead of silently ignored.  The value is still unused by the
    # scalar thin-grating model -- it is checked for forward consistency with
    # the RCWA solver (CONVENTIONS Section 7 polarization bridge).
    pol = {'s': 'te', 'p': 'tm'}.get(str(polarization).lower(),
                                     str(polarization).lower())
    if pol not in ('te', 'tm'):
        raise ValueError(
            f"thin_grating_efficiency_1d: polarization must be 'te'/'tm' "
            f"(or the 's'/'p' aliases), got {polarization!r}.")
    _warn_thin_grating_validity(
        'thin_grating_efficiency_1d', period, depth, wavelength,
        n_ridge, n_groove, duty_cycle)
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
    # v5.4.6 (audit F-23): reject an out-of-range order rather than
    # silently returning the nearest available order's efficiency (the
    # retained orders are arange(-n_orders, n_orders+1)).
    if abs(int(order)) > n_orders:
        raise ValueError(
            f"grating_efficiency_vs_wavelength: requested order {order} is "
            f"outside the retained range [-{n_orders}, {n_orders}]; "
            f"increase n_orders to include it.")
    eff = np.empty(wavelengths.size)
    for i, wl in enumerate(wavelengths):
        orders, _, T = thin_grating_efficiency_1d(
            period, n_ridge, n_groove, n_substrate, n_superstrate,
            depth, duty_cycle, wl, polarization=polarization,
            n_orders=n_orders)
        idx = np.argmin(np.abs(orders - order))
        eff[i] = T[idx]
    return eff
