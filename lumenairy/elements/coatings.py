"""
Thin-film optical coating model (transfer matrix method).

Computes reflectance (R), transmittance (T), and phase shift of
multilayer dielectric coatings as a function of wavelength and angle
of incidence.  Standard Fresnel coefficients for uncoated interfaces
are available as the single-layer limit.

The transfer-matrix method (TMM) multiplies 2x2 characteristic
matrices for each layer:

    M_j = [[cos(delta_j),           -i*sin(delta_j)/eta_j],
           [-i*eta_j*sin(delta_j),   cos(delta_j)         ]]

where delta_j = 2*pi*n_j*d_j*cos(theta_j)/lambda is the phase
thickness and eta_j is the admittance (depends on polarisation).

Author: Andrew Traverso
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np


def coating_reflectance(
    layers: List[Tuple[Union[float, complex], float]],
    wavelengths: Union[float, np.ndarray],
    angle: float = 0.0,
    n_substrate: Union[float, complex] = 1.52,
    n_ambient: Union[float, complex] = 1.0,
    polarization: str = 'avg',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectral reflectance of a multilayer thin-film coating.

    Parameters
    ----------
    layers : list of (n, d)
        Each element is ``(refractive_index, physical_thickness_m)``.
        Ordered from ambient side inward (first layer is outermost).
        Refractive indices may be complex (absorbing layers).
    wavelengths : array-like of float
        Vacuum wavelengths [m] at which to evaluate.
    angle : float, default 0
        Angle of incidence [rad] in the ambient medium.
    n_substrate : float, default 1.52
        Substrate refractive index (real or complex).
    n_ambient : float, default 1.0
        Ambient (incident) medium refractive index.
    polarization : str, default 'avg'
        ``'s'``, ``'p'``, or ``'avg'`` (average of s and p).

    Returns
    -------
    R : ndarray
        Power reflectance at each wavelength (0 to 1).
    T : ndarray
        Power transmittance at each wavelength.
    phase_r : ndarray
        Reflection phase [rad] at each wavelength.

    Notes
    -----
    **Limitations (audit #2.4).**  Two assumptions in the internal
    Snell-step deserve calling out:

    1. **Complex index ``.imag`` is dropped at the Snell step.**
       The propagated refraction angle uses ``n.real`` only --
       fine for transparent dielectric AR / HR stacks where ``.imag``
       is essentially zero, but underestimates the absorbing-layer
       phase thickness for metallic mirrors or metal-dielectric
       hybrids.  For accurate metal-bearing stacks use a TMM solver
       that propagates the complex angle (e.g.
       ``tmm.coh_tmm`` in the ``tmm`` package).
    2. **TIR inside the stack is silently capped** via
       ``sin_t = min(sin_t, 0.9999)``, masking total internal
       reflection at intra-stack interfaces and reporting finite
       transmittance through what should be a totally reflecting
       interface.  Typical AR coatings don't reach TIR; high-AOI
       polarizing-beam-splitter coatings can, and this function
       will under-report their reflectance.

    Both items are on the roadmap for a proper TMM rewrite.
    """
    wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=np.float64))
    n_wv = wavelengths.size
    R = np.empty(n_wv)
    T = np.empty(n_wv)
    phase_r = np.empty(n_wv)

    pols = ['s', 'p'] if polarization == 'avg' else [polarization]

    for iw, lam in enumerate(wavelengths):
        rs, ts = [], []
        for pol in pols:
            M = np.eye(2, dtype=np.complex128)
            theta_prev = angle
            n_prev = complex(n_ambient)
            for n_layer, d in layers:
                n_layer = complex(n_layer)
                # 4.10: warn when intra-stack TIR is silently capped.
                # This module uses only n.real for the angle
                # propagation (audit limitation, documented), so true
                # complex-angle TIR is not supported here.  Until the
                # full complex-Snell rewrite, surface the cap so users
                # of polarising-beam-splitter / immersion coatings
                # know they're seeing an approximation.
                sin_t = n_prev.real * np.sin(theta_prev) / n_layer.real
                if sin_t > 1.0:
                    import warnings
                    warnings.warn(
                        f"thin_film_stack: intra-stack TIR at layer with "
                        f"n_layer={n_layer.real:.3f}, sin_t={sin_t:.3f}; "
                        f"capped at 0.9999 (real-Snell approximation).  "
                        f"For accurate TIR / immersion behaviour use a "
                        f"full complex-Snell solver.",
                        RuntimeWarning, stacklevel=2,
                    )
                sin_t = min(sin_t, 0.9999)
                cos_t = np.sqrt(1 - sin_t**2)
                delta = 2 * np.pi * n_layer * d * cos_t / lam
                if pol == 's':
                    eta = n_layer * cos_t
                else:
                    eta = n_layer / cos_t
                Mj = np.array([
                    [np.cos(delta), -1j * np.sin(delta) / eta],
                    [-1j * eta * np.sin(delta), np.cos(delta)],
                ], dtype=np.complex128)
                M = M @ Mj
                theta_prev = np.arcsin(sin_t)
                n_prev = n_layer
            # Substrate admittance
            sin_sub = n_prev.real * np.sin(theta_prev) / complex(n_substrate).real
            sin_sub = min(sin_sub, 0.9999)
            cos_sub = np.sqrt(1 - sin_sub**2)
            if pol == 's':
                eta_sub = complex(n_substrate) * cos_sub
                eta_amb = complex(n_ambient) * np.cos(angle)
            else:
                eta_sub = complex(n_substrate) / cos_sub
                eta_amb = complex(n_ambient) / np.cos(angle)
            # Reflection coefficient
            # 4.10: dropped the dead `num` / `den` lines (kept the
            # correct `B`, `C`, `r` formulas).  Documented sign
            # convention: this uses the Macleod p-pol form (eta_p =
            # n / cos_t), so r_p has the same sign as r_s at normal
            # incidence.  Born & Wolf use the opposite p-sign; the
            # reflectance R = |r|^2 is unaffected, but phase_r differs
            # by pi from those references.
            B = M[0, 0] + M[0, 1] * eta_sub
            C = M[1, 0] + M[1, 1] * eta_sub
            r = (eta_amb * B - C) / (eta_amb * B + C)
            # Amplitude transmission for absorbing/lossy stacks.
            t_amp = 2.0 * eta_amb / (eta_amb * B + C)
            ts.append(t_amp)
            rs.append(r)

        if polarization == 'avg':
            R_val = 0.5 * (abs(rs[0])**2 + abs(rs[1])**2)
            phase_val = 0.5 * (np.angle(rs[0]) + np.angle(rs[1]))
            # Power transmission via the amplitude coefficient (Macleod
            # eq. 2.99): T_s = Re(eta_sub) / Re(eta_amb) * |t|^2.
            T_s = float((eta_sub.real / max(eta_amb.real, 1e-30))
                         * abs(ts[0]) ** 2) if hasattr(eta_sub, 'real') \
                  else float((eta_sub / eta_amb) * abs(ts[0]) ** 2)
            T_p = float((eta_sub.real / max(eta_amb.real, 1e-30))
                         * abs(ts[1]) ** 2) if hasattr(eta_sub, 'real') \
                  else float((eta_sub / eta_amb) * abs(ts[1]) ** 2)
            T_val = 0.5 * (T_s + T_p)
        else:
            R_val = abs(rs[0])**2
            phase_val = np.angle(rs[0])
            T_val = float((eta_sub.real / max(eta_amb.real, 1e-30))
                           * abs(ts[0]) ** 2) if hasattr(eta_sub, 'real') \
                    else float((eta_sub / eta_amb) * abs(ts[0]) ** 2)
        R[iw] = R_val
        # 4.10: real transmission via the amplitude coefficient, so
        # absorbing stacks correctly produce R + T < 1.  Pre-4.10 used
        # T = 1 - R unconditionally which overstated transmission for
        # any lossy material (the specific case where complex n is
        # passed in, i.e. the only case where TMM matters).
        T[iw] = max(0.0, T_val)
        phase_r[iw] = phase_val

    return R, T, phase_r


def quarter_wave_ar(
    n_substrate: float,
    wavelength_center: float,
) -> List[Tuple[float, float]]:
    """Design a single-layer quarter-wave AR coating.

    Returns ``(n_layer, thickness)`` for a MgF2-like AR coating.
    """
    n_layer = np.sqrt(n_substrate)  # ideal
    d = wavelength_center / (4 * n_layer)
    return [(n_layer, d)]


def broadband_ar_v_coat(
    n_substrate: float,
    wavelength_center: float,
) -> List[Tuple[float, float]]:
    """Design a simple 2-layer V-coat AR for broadband use.

    Returns a list of (n, d) layers.
    """
    n_H = 2.3  # TiO2-like
    n_L = 1.38  # MgF2-like
    d_H = wavelength_center / (4 * n_H)
    d_L = wavelength_center / (4 * n_L)
    return [(n_L, d_L), (n_H, d_H)]
