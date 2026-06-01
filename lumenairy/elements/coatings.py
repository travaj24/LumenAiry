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
        # v4.15.1 (P3-4 / Agent E): always-emit TIR cap warning.
        # Physics: when ``sin_t > 1.0`` Snell's law has no real
        # solution -- the wave is totally internally reflected and
        # the transmitted amplitude is evanescent (purely imaginary
        # cos_t).  Capping ``sin_t`` at 0.9999 produces a finite
        # real cos_t and so a finite transmittance, which is
        # physically WRONG at TIR -- R should be 1.0, T should be
        # 0.0 in the steady state.  The warning category is
        # ``UserWarning`` (always-visible by Python's default
        # filters) rather than ``RuntimeWarning`` (which some test
        # frameworks / IDE notebooks silently filter) so the user
        # always sees the cap firing.
        if sin_t > 1.0:
            import warnings
            warnings.warn(
                f"thin_film_stack: intra-stack TIR at layer with "
                f"n_layer={n_layer.real:.3f}, sin_t={sin_t:.3f}; "
                f"capped at 0.9999 (real-Snell approximation).  "
                f"The reported R/T at this layer is NOT physically "
                f"correct at TIR -- the steady-state R is 1.0 and "
                f"T is 0.0, but the real-Snell approximation forces "
                f"a finite transmittance.  For accurate TIR / "
                f"immersion behaviour use a full complex-Snell "
                f"solver (e.g. ``tmm.coh_tmm`` in the ``tmm`` "
                f"package).",
                UserWarning, stacklevel=2,
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
    # v4.15.1 (P3-4 / Agent E): substrate TIR cap warning.  Pre-
    # v4.15.1 this site silently capped ``sin_sub`` with no
    # diagnostic, so a high-AOI polarizing-beam-splitter coating
    # where the substrate medium had a lower n than the final
    # layer could exhibit TIR at the substrate interface and the
    # caller would never know.  Mirrors the intra-stack cap above.
    if sin_sub > 1.0:
        import warnings
        warnings.warn(
            f"thin_film_stack: substrate TIR at the final interface "
            f"(n_substrate={complex(n_substrate).real:.3f}, "
            f"sin_sub={sin_sub:.3f}); capped at 0.9999 (real-Snell "
            f"approximation).  Same physics caveat as the intra-stack "
            f"TIR warning above -- use a complex-Snell solver for "
            f"accurate TIR R/T.",
            UserWarning, stacklevel=2,
        )
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


def coating_reflectance_jax(
    layers,
    wavelength,
    angle: float = 0.0,
    n_substrate=1.52,
    n_ambient=1.0,
    polarization: str = 'avg',
):
    """JAX-differentiable thin-film reflectance (Abeles TMM) for gradient-based
    coating inverse design -- the s/p sibling of the RCWA differentiable path
    (v5.5.3).

    Differentiable w.r.t. the layer THICKNESSES: pass them as ``jax.numpy``
    scalars/arrays and ``jax.grad`` flows through the matrix product.  Returns
    a scalar ``R`` (power reflectance at the single ``wavelength``).  Wrap this
    in :class:`lumenairy.optimize.JaxMeritTerm` (with ``needs_ray=False``) to
    optimize an AR / HR / band-pass stack against a target reflectance.

    Matches :func:`coating_reflectance` to its real-Snell approximation (the
    Snell angle chain uses the concrete real indices + angle, which do not
    depend on the differentiated thicknesses; ``n.imag`` is dropped at the
    Snell step and intra-stack TIR is capped, exactly as documented there).
    Requires the optional ``jax`` extra.

    Parameters mirror :func:`coating_reflectance` (``layers`` is a list of
    ``(index, thickness)`` ambient-side first; ``polarization`` is
    ``'s'`` / ``'p'`` / ``'avg'``).
    """
    from ..backend import JAX_AVAILABLE
    if not JAX_AVAILABLE:
        raise ImportError(
            "coating_reflectance_jax requires the optional 'jax' extra; "
            "install with `pip install lumenairy[jax]`.  Use "
            "coating_reflectance for non-differentiable evaluation.")
    import jax.numpy as jnp

    # --- Snell angle chain: CONCRETE (depends on indices + angle, not the
    # differentiated thicknesses).  Real-Snell + TIR cap, matching
    # coating_reflectance exactly.
    theta_prev = float(angle)
    n_prev = complex(n_ambient)
    cos_t, n_re = [], []
    for n_layer, _d in layers:
        n_layer = complex(n_layer)
        sin_t = min(n_prev.real * math.sin(theta_prev) / n_layer.real, 0.9999)
        cos_t.append(math.sqrt(1.0 - sin_t * sin_t))
        n_re.append(n_layer)
        theta_prev = math.asin(sin_t)
        n_prev = n_layer
    sin_sub = min(n_prev.real * math.sin(theta_prev)
                  / complex(n_substrate).real, 0.9999)
    cos_sub = math.sqrt(1.0 - sin_sub * sin_sub)
    cos_angle = math.cos(float(angle))

    pols = ['s', 'p'] if polarization == 'avg' else [polarization]
    R_terms = []
    for pol in pols:
        if pol == 's':
            eta = [n_re[j] * cos_t[j] for j in range(len(layers))]
            eta_sub = complex(n_substrate) * cos_sub
            eta_amb = complex(n_ambient) * cos_angle
        else:
            eta = [n_re[j] / cos_t[j] for j in range(len(layers))]
            eta_sub = complex(n_substrate) / cos_sub
            eta_amb = complex(n_ambient) / cos_angle
        # --- characteristic-matrix product (JAX; delta carries the thickness)
        M = jnp.eye(2, dtype=jnp.complex128)
        for j, (n_layer, d) in enumerate(layers):
            delta = (2.0 * jnp.pi * complex(n_re[j]) * cos_t[j]
                     * jnp.asarray(d) / wavelength)
            cd, sd = jnp.cos(delta), jnp.sin(delta)
            ej = complex(eta[j])
            Mj = jnp.array([[cd, -1j * sd / ej],
                            [-1j * ej * sd, cd]], dtype=jnp.complex128)
            M = M @ Mj
        B = M[0, 0] + M[0, 1] * eta_sub
        C = M[1, 0] + M[1, 1] * eta_sub
        r = (eta_amb * B - C) / (eta_amb * B + C)
        R_terms.append(jnp.abs(r) ** 2)
    return sum(R_terms) / len(R_terms)


def quarter_wave_ar(
    n_substrate: float,
    wavelength_center: float,
) -> List[Tuple[float, float]]:
    """Design a single-layer quarter-wave AR coating.

    Returns ``[(n_layer, thickness)]`` for a MgF2-like AR coating.

    v5.4.6 (audit P3-6): the returned layer list is in ``coating_reflectance``
    order -- ambient-side first (outermost layer first).  For this
    single-layer design the order is unobservable, but the convention is
    stated here so multi-layer designs (see ``broadband_ar_v_coat``) and
    callers agree.
    """
    n_layer = np.sqrt(n_substrate)  # ideal
    d = wavelength_center / (4 * n_layer)
    return [(n_layer, d)]


def broadband_ar_v_coat(
    n_substrate: float,
    wavelength_center: float,
) -> List[Tuple[float, float]]:
    """Design a simple 2-layer V-coat AR for broadband use.

    Returns a list of ``(n, d)`` layers in ``coating_reflectance`` order:
    AMBIENT-SIDE FIRST (outermost first).  The low-index (MgF2-like,
    n=1.38) layer is returned first because it sits on the air/ambient
    side; the high-index (TiO2-like, n=2.3) layer is next to the
    substrate.  v5.4.6 (audit P3-6): this order is load-bearing -- feeding
    the list to ``coating_reflectance`` substrate-first would model an HR
    stack, not an AR V-coat.  Pinned by a 550 nm round-trip test.
    """
    n_H = 2.3  # TiO2-like
    n_L = 1.38  # MgF2-like
    d_H = wavelength_center / (4 * n_H)
    d_L = wavelength_center / (4 * n_L)
    return [(n_L, d_L), (n_H, d_H)]


# =============================================================================
# v5.4 Phase 5 -- Thin-film coating material database
# =============================================================================
#
# Canonical lookup of refractive index for common thin-film AR / HR
# coating materials.  Modeled on the ``GLASS_REGISTRY`` / ``get_glass_
# index`` pattern in ``lumenairy/glass.py`` but scoped to dielectric
# coating materials (which the bulk-glass catalogue does not cover in
# a thin-film-relevant way).
#
# Each entry in :data:`COATING_MATERIAL_REGISTRY` is a dict carrying:
#
# * ``'n_constant'`` (float) -- flat refractive index at
#   ``'ref_wavelength'``.  Used as the fallback when no Sellmeier
#   coefficients are provided.
# * ``'ref_wavelength'`` (float, m) -- the wavelength at which
#   ``n_constant`` is quoted.  Always 550 nm for the dock's default
#   visible-band designs.
# * ``'range'`` ((float, float), m) -- documented validity range
#   (``lambda_min, lambda_max``).  ``get_coating_material_index`` warns
#   (``UserWarning``) when called outside this band.
# * ``'sellmeier'`` (optional, ((B1, B2, B3), (C1, C2, C3))) -- 3-term
#   Sellmeier coefficients with ``C_i`` in um^2 (same convention as
#   :data:`lumenairy.glass.SELLMEIER_COEFFICIENTS`).  When present the
#   dispersion is computed via the standard
#   ``n^2 - 1 = sum_i B_i lam^2 / (lam^2 - C_i)`` formula; otherwise
#   the constant ``n_constant`` value is returned for every wavelength.
#
# Sources for Sellmeier coefficients (top 4 AR/HR materials):
#
# * MgF2 (Dodge 1984): ordinary ray, refractiveindex.info
#   main/MgF2/Dodge-o; valid 0.2-7.0 um.
# * SiO2 (Malitson 1965): fused silica, JOSA 55, 1205; valid 0.21-6.7 um.
# * TiO2 (DeVore 1951): ordinary ray, JOSA 41, 416; valid 0.43-1.53 um.
# * Ta2O5 (Bright 2013): refractiveindex.info main/Ta2O5/Bright;
#   valid 0.5-1.0 um for the polynomial fit baked here.  Outside
#   that band the constant value is the safer choice; users with
#   broader bands should override the registry entry or pass an
#   explicit n at the layer level.
# =============================================================================

COATING_MATERIAL_REGISTRY: dict = {
    'MgF2':  {
        'n_constant':     1.38,
        'ref_wavelength': 550e-9,
        'range':          (200e-9, 7000e-9),
        # Dodge 1984 ordinary-ray Sellmeier (refractiveindex.info
        # main/MgF2/Dodge-o); valid 0.2-7.0 um.
        'sellmeier':      ((0.48755108, 0.39875031, 2.3120353),
                           (0.04338408**2, 0.09461442**2, 23.793604**2)),
    },
    'SiO2':  {
        'n_constant':     1.46,
        'ref_wavelength': 550e-9,
        # v5.4.5 (audit P2-3): tightened from 8000e-9 to match Malitson
        # 1965 source citation (0.21-6.7 um).  At lambda > 6.7 um the
        # Sellmeier extrapolation returns non-physical n < 1 for a
        # dielectric in air (e.g. n=0.945 at 7.5 um, n=0.642 at 8.0 um).
        'range':          (200e-9, 6700e-9),
        # Malitson 1965 fused silica (refractiveindex.info
        # main/SiO2/Malitson); valid 0.21-6.7 um.
        'sellmeier':      ((0.6961663, 0.4079426, 0.8974794),
                           (0.0684043**2, 0.1162414**2, 9.896161**2)),
    },
    'TiO2':  {
        'n_constant':     2.40,
        'ref_wavelength': 550e-9,
        # v5.4.6 (audit P2-1): range tightened to the cited DeVore-1951
        # fit validity (0.43-1.53 um).  The prior (400e-9, 5000e-9) was
        # ~3.3x wider than the 1-term ordinary-ray Sellmeier supports, so
        # the out-of-range UserWarning never fired for 1.53-5 um sweeps
        # that silently extrapolate.  For a wider design band swap to a
        # multi-pole fit (e.g. Sarkar 2019), do not widen 'range'.
        'range':          (430e-9, 1530e-9),
        # DeVore 1951 ordinary-ray Sellmeier (refractiveindex.info
        # main/TiO2/Devore-o); valid 0.43-1.53 um.
        # v5.4.1 (audit P2 NEW): dummy poles set to 0.0 instead of 1.0
        # to avoid lam2-C=0 division at lam=1um, which was inside
        # documented validity range.
        #
        # v5.4.1 (audit P3 #9): DeVore 1951 used here is the
        # ORDINARY-RAY Sellmeier of uniaxial-anisotropic rutile TiO2
        # (n_o).  The extraordinary-ray index n_e is ~11% higher
        # (n_e ~ 2.87 at 550nm vs n_o ~ 2.58); polarisation-sensitive
        # coating design should account for this.  Future v5.5+
        # candidate: add 'TiO2_e' (or similar) entry with the n_e
        # Sellmeier (Cardona 1965 / DeVore 1951 also has the
        # extraordinary fit).
        'sellmeier':      ((4.99048, 0.0, 0.0),
                           (0.19086**2, 0.0, 0.0)),
    },
    'Ta2O5': {
        'n_constant':     2.10,
        'ref_wavelength': 550e-9,
        # v5.4.6 (audit P2-1): range tightened to the cited Bright-2013
        # fit validity (0.5-1.0 um); the prior (350e-9, 8000e-9) was ~8x
        # wider than the single-pole Sellmeier supports.
        'range':          (500e-9, 1000e-9),
        # Bright 2013 Ta2O5 1-term Sellmeier
        # (refractiveindex.info main/Ta2O5/Bright); valid 0.5-1.0 um.
        # Single-pole approximation: n^2 - 1 = B*lam^2 / (lam^2 - C).
        # v5.4.1 (audit P2 NEW): dummy poles set to 0.0 instead of 1.0
        # to avoid lam2-C=0 division at lam=1um, which was inside
        # documented validity range.
        'sellmeier':      ((3.5820, 0.0, 0.0),
                           (0.16986**2, 0.0, 0.0)),
    },
    'MgO':   {
        'n_constant':     1.74,
        'ref_wavelength': 550e-9,
        'range':          (250e-9, 6000e-9),
    },
    'ZnS':   {
        'n_constant':     2.35,
        'ref_wavelength': 550e-9,
        'range':          (400e-9, 12000e-9),
    },
    'Al2O3': {
        'n_constant':     1.77,
        'ref_wavelength': 550e-9,
        'range':          (200e-9, 8000e-9),
    },
    'HfO2':  {
        'n_constant':     2.05,
        'ref_wavelength': 550e-9,
        'range':          (250e-9, 10000e-9),
    },
    'Y2O3':  {
        'n_constant':     1.93,
        'ref_wavelength': 550e-9,
        'range':          (300e-9, 12000e-9),
    },
    'ZrO2':  {
        'n_constant':     2.15,
        'ref_wavelength': 550e-9,
        'range':          (350e-9, 10000e-9),
    },
    'CeO2':  {
        'n_constant':     2.20,
        'ref_wavelength': 550e-9,
        'range':          (400e-9, 8000e-9),
    },
    'CaF2':  {
        'n_constant':     1.43,
        'ref_wavelength': 550e-9,
        'range':          (150e-9, 12000e-9),
    },
}


def _coating_sellmeier(
    wavelength_m,
    coeffs,
):
    """Evaluate a 3-term Sellmeier on scalar or array wavelength.

    ``coeffs`` is ``((B1, B2, B3), (C1, C2, C3))`` with ``C_i`` in
    um^2 (same convention as :data:`lumenairy.glass.SELLMEIER_COEFFI-
    CIENTS`).  Mirrors the array-aware behaviour of the polynomial
    evaluator so the coatings-dock visible-band sweeps can call this
    once with the full wavelength array.
    """
    # v5.4.6 (audit P3-7): share ``glass._guard_wavelength`` so this
    # evaluator and ``glass._sellmeier_index`` handle NaN / negative
    # wavelength identically (a guard fix in one no longer silently skips
    # the other).  Sellmeier is sign-symmetric, so the guard warns on a
    # negative wavelength and returns |lambda|; it also warns on NaN.  The
    # B==0 dummy-pole skip below remains a coatings-only optimisation, and
    # all four bundled coating poles lie out of band so no in-range
    # lam2==C blowup occurs (unlike glass._sellmeier_index, this array
    # evaluator tolerates an exact pole as NaN rather than raising).
    from ..glass import _guard_wavelength
    wl_g = _guard_wavelength(wavelength_m, "_coating_sellmeier",
                             sign_symmetric=True)
    lam2 = (np.abs(np.asarray(wl_g, dtype=float)) * 1e6) ** 2
    (B1, B2, B3), (C1, C2, C3) = coeffs
    # v5.4.1 (audit P2 NEW): defensive guard -- skip dummy poles where
    # B is zero (the term contributes nothing to n^2 and would NaN if C
    # also happens to equal lam2).
    n_sq_minus_1 = np.zeros_like(lam2, dtype=float)
    for B, C in ((B1, C1), (B2, C2), (B3, C3)):
        if B == 0:
            continue
        n_sq_minus_1 = n_sq_minus_1 + B * lam2 / (lam2 - C)
    return np.sqrt(1.0 + n_sq_minus_1)


def get_coating_material_index(
    material: str,
    wavelength,
):
    """Return the refractive index of a thin-film coating material at
    the given wavelength.

    Parameters
    ----------
    material : str
        Material name; must be a key in
        :data:`COATING_MATERIAL_REGISTRY`.
    wavelength : float or ndarray
        Vacuum wavelength(s) in metres.  Scalar input returns float;
        array input returns an ndarray of the same shape.

    Returns
    -------
    n : float or ndarray
        Real refractive index.  If the material has a ``'sellmeier'``
        entry, the dispersion is computed; otherwise the constant
        ``'n_constant'`` value is returned (broadcast for array
        input).

    Raises
    ------
    KeyError
        If ``material`` is not in :data:`COATING_MATERIAL_REGISTRY`.

    Warns
    -----
    UserWarning
        If ``wavelength`` falls outside the documented ``'range'``
        for ``material``.  The value is still returned (extrapolation
        is sometimes useful for design-space exploration); the warning
        only fires once per call.
    """
    if material not in COATING_MATERIAL_REGISTRY:
        raise KeyError(
            f"get_coating_material_index: unknown thin-film material "
            f"{material!r}.  Known materials: "
            f"{sorted(COATING_MATERIAL_REGISTRY.keys())}")
    entry = COATING_MATERIAL_REGISTRY[material]
    lmin, lmax = entry['range']

    import warnings

    wl_arr = np.asarray(wavelength, dtype=float)

    # v5.4.5 (audit P2-1): negative-wavelength guard.  The Sellmeier
    # formula uses lam2 = (wl*1e6)**2, so it is sign-symmetric:
    # n(-lam) == n(+lam) exactly.  A negative wavelength buried in an
    # array would silently land in a real band without this warning.
    # Use warnings.warn (not raise) because the formula IS
    # mathematically symmetric and vector callers (e.g. wavelength
    # sweeps that include 0 / near-0 values) may want to suppress.
    if np.any(wl_arr < 0):
        warnings.warn(
            f"get_coating_material_index({material!r}): negative "
            f"wavelength(s) detected; the Sellmeier formula is "
            f"sign-symmetric (n(-lam) == n(+lam)) so the returned "
            f"values are computed at |lam|.  This is almost certainly "
            f"a unit-conversion error in the caller; verify your "
            f"wavelength array.",
            UserWarning, stacklevel=2,
        )

    # v5.4.5 (audit P2-2): NaN-detection guard.  np.any(wl_arr < lmin)
    # returns False for NaN by IEEE 754 semantics, so a single NaN in
    # an array silently propagates to NaN output with no warning.
    # Place BEFORE the range check so the NaN warning surfaces first.
    if np.any(np.isnan(wl_arr)):
        warnings.warn(
            f"get_coating_material_index({material!r}): NaN "
            f"wavelength(s) detected in input; output will contain "
            f"NaN at those positions.  This usually indicates upstream "
            f"data corruption (NaN propagation from earlier "
            f"computation).",
            UserWarning, stacklevel=2,
        )

    # Range check -- scalar or array; fire UserWarning if ANY sample
    # falls outside the documented band.  v5.4.5 (audit P2-3): `>=` so
    # a wavelength exactly at lmax also triggers (covers the "n<1 at
    # the exact registry boundary" case).
    if np.any(wl_arr < lmin) or np.any(wl_arr >= lmax):
        warnings.warn(
            f"get_coating_material_index: {material} validity is "
            f"[{lmin:.3e}, {lmax:.3e}] m; got wavelength "
            f"{float(np.nanmin(wl_arr)):.3e} to "
            f"{float(np.nanmax(wl_arr)):.3e} m. "
            f"Extrapolated value may not be physical.",
            UserWarning, stacklevel=2,
        )

    if 'sellmeier' in entry:
        result = _coating_sellmeier(wavelength, entry['sellmeier'])
        # Scalar in -> scalar out, mirroring _sellmeier_index in glass.py
        if wl_arr.ndim == 0:
            return float(result)
        return result

    # Constant-n path.
    n_const = float(entry['n_constant'])
    if wl_arr.ndim == 0:
        return n_const
    return np.full_like(wl_arr, n_const, dtype=float)
