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
import numpy as np


def coating_reflectance(layers, wavelengths, angle=0.0,
                        n_substrate=1.52, n_ambient=1.0,
                        polarization='avg'):
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
                # Snell's law for this layer
                sin_t = n_prev.real * np.sin(theta_prev) / n_layer.real
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
            num = M[0, 0] * eta_sub + M[0, 1] - eta_amb * (M[1, 0] * eta_sub + M[1, 1])
            den = M[0, 0] * eta_sub + M[0, 1] + eta_amb * (M[1, 0] * eta_sub + M[1, 1])
            # Actually the standard TMM formula:
            # r = (M[0,0]*eta_sub + M[0,1]*eta_amb*eta_sub - M[1,0] - M[1,1]*eta_amb) /
            #     (M[0,0]*eta_sub + M[0,1]*eta_amb*eta_sub + M[1,0] + M[1,1]*eta_amb)
            # Simplified: use B = M[0,0]*eta_sub + M[0,1], C = M[1,0]*eta_sub + M[1,1]
            B = M[0, 0] + M[0, 1] * eta_sub
            C = M[1, 0] + M[1, 1] * eta_sub
            r = (eta_amb * B - C) / (eta_amb * B + C)
            rs.append(r)

        if polarization == 'avg':
            R_val = 0.5 * (abs(rs[0])**2 + abs(rs[1])**2)
            phase_val = 0.5 * (np.angle(rs[0]) + np.angle(rs[1]))
        else:
            R_val = abs(rs[0])**2
            phase_val = np.angle(rs[0])
        R[iw] = R_val
        T[iw] = max(0, 1 - R_val)  # lossless approximation
        phase_r[iw] = phase_val

    return R, T, phase_r


def quarter_wave_ar(n_substrate, wavelength_center):
    """Design a single-layer quarter-wave AR coating.

    Returns ``(n_layer, thickness)`` for a MgF2-like AR coating.
    """
    n_layer = np.sqrt(n_substrate)  # ideal
    d = wavelength_center / (4 * n_layer)
    return [(n_layer, d)]


def broadband_ar_v_coat(n_substrate, wavelength_center):
    """Design a simple 2-layer V-coat AR for broadband use.

    Returns a list of (n, d) layers.
    """
    n_H = 2.3  # TiO2-like
    n_L = 1.38  # MgF2-like
    d_H = wavelength_center / (4 * n_H)
    d_L = wavelength_center / (4 * n_L)
    return [(n_L, d_L), (n_H, d_H)]
