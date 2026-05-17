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

import math
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
    # 4.14.0 (Tier-2 perf, audit group): track scalar-input back-compat
    # before the np.atleast_1d promotion.  When a scalar wavelength is
    # passed in we still return scalar R/T/phase, not length-1 arrays.
    _wv_in = np.asarray(wavelengths, dtype=np.float64)
    _scalar_in = (_wv_in.ndim == 0)
    wavelengths = np.atleast_1d(_wv_in)
    n_wv = wavelengths.size

    pols = ['s', 'p'] if polarization == 'avg' else [polarization]

    # The Snell-angle chain is intrinsically sequential per layer (each
    # layer's sin_t depends on the previous after the TIR cap), but it
    # does NOT depend on wavelength -- the real-Snell approximation
    # documented in the docstring drops n.imag and lambda from the
    # angle propagation entirely.  So we walk the layer stack once,
    # collecting (cos_t, eta_s, eta_p) per layer, then build a
    # (n_wv, n_layers, 2, 2) characteristic-matrix stack in one
    # vectorised pass.
    #
    # 4.13.0 (preserved): the inner Snell loop uses ``math.sin`` /
    # ``math.asin`` for ~10x speedup over the numpy ufuncs on
    # 50-layer-scale stacks; the result is bit-identical because both
    # routes go through libm.
    theta_prev = angle
    n_prev = complex(n_ambient)
    layer_cos_t: List[float] = []
    layer_n: List[complex] = []
    layer_d: List[float] = []
    for n_layer, d in layers:
        n_layer = complex(n_layer)
        sin_t = n_prev.real * math.sin(theta_prev) / n_layer.real
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
        cos_t = math.sqrt(1 - sin_t * sin_t)
        layer_cos_t.append(cos_t)
        layer_n.append(n_layer)
        layer_d.append(d)
        theta_prev = math.asin(sin_t)
        n_prev = n_layer

    n_layers = len(layer_cos_t)
    # Substrate / ambient angles (also wavelength-independent in the
    # real-Snell approximation).
    sin_sub = (n_prev.real * math.sin(theta_prev)
               / complex(n_substrate).real)
    sin_sub = min(sin_sub, 0.9999)
    cos_sub = math.sqrt(1 - sin_sub * sin_sub)
    cos_angle = math.cos(angle)

    # Layer arrays (n_layers,)
    if n_layers:
        n_arr = np.asarray(layer_n, dtype=np.complex128)
        d_arr = np.asarray(layer_d, dtype=np.float64)
        cos_t_arr = np.asarray(layer_cos_t, dtype=np.float64)
        # delta = 2*pi * n * d * cos_t / lambda
        # Broadcast wavelength: (n_wv, 1) * (n_layers,) -> (n_wv, n_layers)
        delta_all = (2.0 * math.pi
                     * (n_arr * d_arr * cos_t_arr)[None, :]
                     / wavelengths[:, None])  # (n_wv, n_layers) complex
    else:
        delta_all = np.empty((n_wv, 0), dtype=np.complex128)

    R = np.empty(n_wv)
    T = np.empty(n_wv)
    phase_r = np.empty(n_wv)

    rs_by_pol: dict = {}
    ts_by_pol: dict = {}
    eta_sub_by_pol: dict = {}
    eta_amb_by_pol: dict = {}

    for pol in pols:
        if n_layers:
            if pol == 's':
                eta_arr = n_arr * cos_t_arr  # (n_layers,)
            else:
                eta_arr = n_arr / cos_t_arr
            # Build (n_wv, n_layers, 2, 2) characteristic-matrix stack.
            # cos_d / sin_d are (n_wv, n_layers); etas broadcast to
            # (1, n_layers).
            cos_d = np.cos(delta_all)
            sin_d = np.sin(delta_all)
            eta_b = eta_arr[None, :]  # (1, n_layers)
            Mj = np.empty((n_wv, n_layers, 2, 2), dtype=np.complex128)
            Mj[:, :, 0, 0] = cos_d
            Mj[:, :, 0, 1] = -1j * sin_d / eta_b
            Mj[:, :, 1, 0] = -1j * eta_b * sin_d
            Mj[:, :, 1, 1] = cos_d
            # Tournament reduction over the layer axis (axis=1).
            # Preserves left-to-right order: each round pairs
            # Mj[:,0]@Mj[:,1], Mj[:,2]@Mj[:,3], ...  Odd trailing slab
            # carries forward unchanged.  Batched np.matmul on the
            # leading wavelength axis costs ~log2(n_layers) matmuls
            # of (n_wv, k, 2, 2) shape, vs n_wv * n_layers scalar
            # matmuls in the old per-wavelength loop.
            stack = Mj
            while stack.shape[1] > 1:
                n = stack.shape[1]
                even_n = n - (n & 1)
                left = stack[:, 0:even_n:2]
                right = stack[:, 1:even_n:2]
                paired = left @ right
                if n & 1:
                    stack = np.concatenate(
                        [paired, stack[:, even_n:even_n + 1]], axis=1)
                else:
                    stack = paired
            M = stack[:, 0]  # (n_wv, 2, 2)
        else:
            M = np.broadcast_to(np.eye(2, dtype=np.complex128),
                                 (n_wv, 2, 2)).copy()

        # Substrate / ambient admittance (scalar per pol, wavelength-
        # independent in this approximation).
        if pol == 's':
            eta_sub = complex(n_substrate) * cos_sub
            eta_amb = complex(n_ambient) * cos_angle
        else:
            eta_sub = complex(n_substrate) / cos_sub
            eta_amb = complex(n_ambient) / cos_angle
        eta_sub_by_pol[pol] = eta_sub
        eta_amb_by_pol[pol] = eta_amb

        # Reflection / transmission coefficients, vectorised over
        # wavelength.  M is (n_wv, 2, 2); slicing the four corners
        # yields (n_wv,) complex arrays.  Same algebra as the scalar
        # form -- Macleod p-pol sign convention (audit-documented).
        B = M[:, 0, 0] + M[:, 0, 1] * eta_sub
        C = M[:, 1, 0] + M[:, 1, 1] * eta_sub
        denom = eta_amb * B + C
        r = (eta_amb * B - C) / denom
        t_amp = 2.0 * eta_amb / denom
        rs_by_pol[pol] = r
        ts_by_pol[pol] = t_amp

    if polarization == 'avg':
        r_s = rs_by_pol['s']
        r_p = rs_by_pol['p']
        t_s = ts_by_pol['s']
        t_p = ts_by_pol['p']
        R[:] = 0.5 * (np.abs(r_s) ** 2 + np.abs(r_p) ** 2)
        # v4.14.1 (audit P1-NEW-2): aggregate the s/p reflection phases
        # via the complex sum (then angle), not the arithmetic mean of
        # the two individual angles.  ``r_p`` sign-flips through zero
        # at Brewster (~56 deg for fused silica at visible), and the
        # unwrapped arithmetic mean of two angles separated by ~pi is
        # off by pi/2 (or pi at the singularity).  The complex-sum
        # formulation is robust to that branch cut.
        phase_r[:] = np.angle(0.5 * (r_s + r_p))
        # Power transmission via the amplitude coefficient (Macleod
        # eq. 2.99): T_s = Re(eta_sub) / Re(eta_amb) * |t|^2.  Use the
        # per-polarization admittances (4.11.2 fix).
        _eta_sub_s = eta_sub_by_pol['s']
        _eta_amb_s = eta_amb_by_pol['s']
        _eta_sub_p = eta_sub_by_pol['p']
        _eta_amb_p = eta_amb_by_pol['p']
        T_s = ((_eta_sub_s.real / max(_eta_amb_s.real, 1e-30))
                * np.abs(t_s) ** 2)
        T_p = ((_eta_sub_p.real / max(_eta_amb_p.real, 1e-30))
                * np.abs(t_p) ** 2)
        T[:] = np.maximum(0.0, 0.5 * (T_s + T_p))
    else:
        r0 = rs_by_pol[pols[0]]
        t0 = ts_by_pol[pols[0]]
        eta_sub0 = eta_sub_by_pol[pols[0]]
        eta_amb0 = eta_amb_by_pol[pols[0]]
        R[:] = np.abs(r0) ** 2
        phase_r[:] = np.angle(r0)
        T_val = ((eta_sub0.real / max(eta_amb0.real, 1e-30))
                  * np.abs(t0) ** 2)
        T[:] = np.maximum(0.0, T_val)

    # 4.14.0: preserve scalar-input back-compat.  When the caller
    # passed a 0-d wavelength, return scalar (not length-1) outputs.
    if _scalar_in:
        return float(R[0]), float(T[0]), float(phase_r[0])
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
