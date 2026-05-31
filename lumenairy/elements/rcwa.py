"""
Rigorous Coupled-Wave Analysis (RCWA / Fourier Modal Method)
============================================================

A frequency-domain, rigorous Maxwell solver for laterally PERIODIC,
layered structures: dielectric / metallic gratings, sub-wavelength
metasurfaces, and birefringent (liquid-crystal) stacks.  RCWA fills the
gap between LumenAiry's scalar :mod:`~lumenairy.elements.thin_grating`
(thin-phase, no metal, no polarization coupling) and the laterally-uniform
isotropic TMM in :mod:`~lumenairy.elements.coatings`: it solves the full
vector Maxwell equations inside a structured periodic layer and returns
rigorous diffraction efficiencies AND the complex zeroth-order Jones
reflection matrix that drops straight into the
:class:`~lumenairy.elements.polarization.JonesField` pipeline.

This is a clean-room implementation derived from the published Fourier
Modal Method literature -- principally

* M. G. Moharam, E. B. Grann, D. A. Pommet, T. K. Gaylord, "Formulation
  for stable and efficient implementation of the rigorous coupled-wave
  analysis of binary gratings," JOSA A 12, 1068 (1995); and the
  enhanced transmittance-matrix companion, JOSA A 12, 1077 (1995).
* L. Li, "Use of Fourier series in the analysis of discontinuous periodic
  structures," JOSA A 13, 1870 (1996)  -- the inverse rule / correct
  factorization for TM and metals.
* L. Li, "Formulation and comparison of two recursive matrix algorithms
  for modeling layered diffraction gratings," JOSA A 13, 1024 (1996) --
  the S-matrix (Redheffer) recursion, and the 1-D anisotropic
  factorization.
* L. Li, "New formulation of the Fourier modal method for crossed
  surface-relief gratings," JOSA A 14, 2758 (1997).
* R. C. Rumpf, "Improved formulation of scattering matrices for
  semi-analytical methods that is consistent with convention," PIER B 35,
  241 (2011) -- the gap-medium scattering-matrix bookkeeping used here.

No GPL/closed RCWA source (nannos, S4, RETICOLO, inkstone, grcwa) is
copied; those are used only as independent numerical validation oracles.

Conventions
-----------
* Time convention ``exp(-i omega t)``; forward propagation ``exp(+i k z)``
  (the library standard -- see CONVENTIONS.md section 7).  Consequently a
  passive absorber has ``n = n + i kappa`` with ``kappa > 0``, hence
  ``Im(epsilon) = Im(n^2) > 0``, and the physical (decaying) z-branch is
  the one with ``Im(k_z) >= 0``.
* All wavelengths, periods, thicknesses are in METRES (vacuum wavelength).
* Angles are in RADIANS.  ``theta`` is the polar angle from the +z axis;
  ``phi`` is the azimuth (conical mounting) measured from +x.
* Diffraction-order indices are integers centred on 0.  Efficiencies are
  REAL power fractions; amplitudes are COMPLEX field coefficients.
* Energy: for a lossless stack ``sum(R) + sum(T) == 1`` to the harmonic
  truncation; with loss, ``sum(R) + sum(T) + A == 1`` where ``A`` is the
  absorptance.

Internally the solver is the full VECTORIAL scattering-matrix method
(Rumpf 2011 / Moharam 1995): the 2N-component (x, y tangential-field)
eigenproblem is assembled per layer, so planar TE/TM, conical TE/TM
coupling, and (in later layers) 2-D crossed gratings all flow through one
stable Redheffer star-product recursion -- the algorithm never forms an
exponentially-growing T-matrix.

Author: Andrew Traverso (LumenAiry RCWA module).
"""
from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "rcwa_efficiency_1d",
    "rcwa_efficiency_vs_wavelength",
    "rcwa_efficiency_2d",
    "rcwa_efficiency_2d_shapes",
    "rcwa_jones_1d",
    "rcwa_jones_2d",
    "rcwa_efficiency_1d_jax",
    "uniaxial_tensor",
    "RCWAStack",
    "RCWAResult",
]

# Internal eigenproblem dtype -- always double complex regardless of the
# field-side default (RCWA conditioning is sensitive to single precision).
_C = np.complex128


# ===========================================================================
# Convention-aware square root (branch selection)
# ===========================================================================

def _sqrt_forward(x: np.ndarray) -> np.ndarray:
    """Square root on the branch that makes ``exp(+i k z)`` the forward
    (outgoing / evanescent-decaying) PLANE WAVE for the library's
    ``exp(-i omega t)`` convention: ``Im(result) >= 0``.

    Used for the homogeneous-region longitudinal wavevector ``k_z`` (where
    the field varies as ``exp(+i k_z z)``).  For a real positive argument
    this is the ordinary positive root (a propagating order); for a real
    negative argument it is ``+i|.|^{1/2}`` (an order that decays as
    ``z -> +inf``).
    """
    x = np.asarray(x, dtype=_C)
    r = np.sqrt(x)
    # numpy's principal branch already yields Im >= 0 except on the cut;
    # force the decaying root for any residual negative-imaginary roundoff.
    bad = (r.imag < 0) | ((np.abs(r.imag) <= 1e-300) & (r.real < 0))
    return np.where(bad, -r, r)


def _inv_lam(lam: np.ndarray) -> np.ndarray:
    """``1/lam`` with a floor on ``|lam|`` so a grazing mode (``kz -> 0`` so
    the modal eigenvalue ``lam -> 0``) does not produce ``inf``/``NaN`` in
    ``V = Q W diag(1/lam)``.  A ``kz = 0`` mode carries no z-directed power,
    so this regularisation never affects a physical diffraction efficiency;
    it only keeps the eigenvector matrix finite at an exact Wood anomaly.
    """
    safe = np.where(np.abs(lam) < 1e-12, 1e-12, lam)
    return 1.0 / safe


def _sqrt_decay(x: np.ndarray) -> np.ndarray:
    """Square root on the ``Re(result) >= 0`` (principal) branch, used for
    the LAYER modal eigenvalue ``lam`` that drives the propagator
    ``X = exp(-lam k0 L)``.

    Choosing ``Re(lam) >= 0`` guarantees ``|X| <= 1`` for forward thickness
    ``L > 0`` -- the unconditional-stability property of the S-matrix
    method.  This is NOT the ``Im >= 0`` branch: for a strongly evanescent
    mode the eigenvalue ``lam^2`` is a large positive real that ``eig``
    returns with a tiny spurious imaginary part, and an ``Im >= 0`` rule
    would flip ``lam`` to a NEGATIVE real, turning the decaying propagator
    into ``exp(+|gamma| k0 L)`` -- a catastrophic high-order blow-up.  The
    principal branch is immune (it never flips the sign of a positive-real
    root).  For propagating modes (``lam^2`` negative real) both branches
    agree on ``+i|kz|``, so physics is unchanged.
    """
    x = np.asarray(x, dtype=_C)
    r = np.sqrt(x)  # principal branch: Re(r) >= 0 by construction
    # On the cut (pure-imaginary r, i.e. lam^2 real negative) pin Im >= 0
    # so propagating modes use the outgoing root deterministically.
    on_cut = r.real == 0
    return np.where(on_cut & (r.imag < 0), -r, r)


# ===========================================================================
# Fourier factorization -- convolution matrices
# ===========================================================================

def _fourier_coeffs_1d(profile: np.ndarray, n_coeffs: int) -> np.ndarray:
    """Centred Fourier coefficients ``c_k`` (``k = -(n_coeffs-1) ..
    (n_coeffs-1)``, length ``2*n_coeffs-1``) of a uniformly-sampled,
    one-period profile, with ``c_k = <f(x) exp(-i k G x)>``.
    """
    profile = np.asarray(profile, dtype=_C)
    Nx = profile.shape[0]
    full = np.fft.fft(profile) / Nx  # full[k] holds c_k (periodic in k)
    K = 2 * n_coeffs - 1
    out = np.empty(K, dtype=_C)
    for j, k in enumerate(range(-(n_coeffs - 1), n_coeffs)):
        out[j] = full[k % Nx]
    return out


def _toeplitz_1d(coeffs: np.ndarray, n_orders: int) -> np.ndarray:
    """``(N, N)`` Toeplitz convolution matrix from centred Fourier
    coefficients, ``N = 2*n_orders + 1``; entry ``[m, n] = c_{m-n}``.
    """
    N = 2 * n_orders + 1
    centre = (coeffs.shape[0] - 1) // 2  # index of c_0
    out = np.empty((N, N), dtype=_C)
    for m in range(N):
        for n in range(N):
            out[m, n] = coeffs[centre + (m - n)]
    return out


def _binary_grating_convolutions(n_ridge, n_groove, duty_cycle, n_orders,
                                 n_samples=4096):
    """Convolution matrices for a 1-D binary grating: the Laurent ``[[eps]]``
    and the Li inverse-rule ``[[1/eps]]^{-1}``.  The ridge (index
    ``n_ridge``) occupies the fraction ``duty_cycle`` of the period.

    A closed-form Fourier series exists for a binary profile, but sampling
    + FFT keeps the path identical to the (future) arbitrary-profile and
    2-D cases and is exact to machine precision at this sampling.
    """
    x = (np.arange(n_samples) + 0.5) / n_samples
    eps_r = _C(n_ridge) ** 2
    eps_g = _C(n_groove) ** 2
    eps = np.where(x < duty_cycle, eps_r, eps_g).astype(_C)
    # The Toeplitz matrix needs coefficients c_k for k = -(N-1)..(N-1) with
    # N = 2*n_orders+1, i.e. n_coeffs = N.
    n_coeffs = 2 * n_orders + 1
    eps_coeffs = _fourier_coeffs_1d(eps, n_coeffs)
    inv_eps_coeffs = _fourier_coeffs_1d(1.0 / eps, n_coeffs)
    EPS = _toeplitz_1d(eps_coeffs, n_orders)               # Laurent rule
    EPS_II = np.linalg.inv(_toeplitz_1d(inv_eps_coeffs, n_orders))  # inverse rule
    return EPS, EPS_II


# ===========================================================================
# Layer eigen-solve (vectorial 2N system, Rumpf/Moharam)
# ===========================================================================

def _layer_Q_matrix(Kx, Ky, EPS, EPS_xx):
    """The ``Q`` block (``dE/dz' = Q H``) of the layer ODE system.

    ``EPS`` is the Laurent ``[[eps]]`` (used where E is tangential to the
    grating walls -- the ``E_y`` response); ``EPS_xx`` is the convolution
    for the wall-NORMAL field ``E_x``, which is ``[[eps]]`` for the Laurent
    rule but the Li inverse-rule ``[[1/eps]]^{-1}`` for fast TM / metal
    convergence.  Shared by the structured-layer eig solve and the analytic
    uniform-layer path so the magnetic eigenvectors ``V = Q W diag(1/lam)``
    use one convention everywhere (essential for evanescent-order interface
    consistency).
    """
    return np.block([
        [Kx @ Ky,           EPS - Kx @ Kx],
        [Ky @ Ky - EPS_xx,  -Ky @ Kx],
    ])


def _layer_eigenmodes(Kx, Ky, EPS, EPS_xx, ez_laurent_inv=None):
    """Eigenmodes of a single layer (structured or uniform).

    Dimension-agnostic: the harmonic count ``N`` is inferred from ``Kx`` so
    the SAME routine serves the 1-D core (``N = 2*n_orders+1``) and the 2-D
    crossed-grating path (``N = Nx*Ny``).

    The tangential field vector is ``[Sx; Sy]`` (x- and y- electric-field
    harmonics), giving a ``2N x 2N`` system.

    Factorization (Li 1996/1997): the wall-normal field ``E_x`` (and, in
    the ``P`` block, ``E_z``) is discontinuous and needs the inverse rule,
    while the wall-tangential ``E_y`` uses the Laurent rule.  Concretely the
    ``P`` block uses the Laurent inverse ``[[eps]]^{-1}`` (the ``E_z``
    elimination, which is wall-tangential -> direct rule on ``eps``, then
    inverted), and the ``Q`` block uses ``EPS_xx`` for the wall-normal
    ``E_x`` (the Li inverse-rule matrix ``[[1/eps]]^{-1}`` when requested)
    and the Laurent ``EPS`` for the tangential ``E_y``.  This placement is
    what gives the fast TM convergence; putting the inverse-rule matrix in
    the ``P`` inner term instead leaves TM converging at the slow Laurent
    rate.  Non-magnetic (``mu = 1``).

    A laterally UNIFORM layer (diagonal ``[[eps]]``) is detected and solved
    analytically: its 2N modes are doubly degenerate (TE/TM share ``kz``),
    so ``np.linalg.eig`` would return a singular eigenvector matrix.

    Returns ``(W, V, lam)`` -- the electric eigenvector block, the magnetic
    eigenvector block, and the modal eigenvalue ``lam = sqrt(-kz^2/k0^2)``
    (``Re >= 0`` branch; ``= i kz`` propagating, ``= |gamma|`` evanescent),
    which feeds the forward-decaying propagator ``X = exp(-lam k0 L)``.
    """
    Kx = np.asarray(Kx, dtype=_C)
    Ky = np.asarray(Ky, dtype=_C)
    N = Kx.shape[0]
    I = np.eye(N, dtype=_C)
    Q = _layer_Q_matrix(Kx, Ky, EPS, EPS_xx)

    offdiag = EPS - np.diag(np.diag(EPS))
    scale = max(1.0, float(np.max(np.abs(np.diag(EPS)))))
    if np.max(np.abs(offdiag)) < 1e-12 * scale:
        # Uniform layer: analytic modes (W = I), kz per order from eps.
        eps0 = EPS[0, 0]
        kx = np.diag(Kx)
        ky = np.diag(Ky)
        kz = _sqrt_forward(eps0 - kx ** 2 - ky ** 2)
        lam = _sqrt_decay(-np.concatenate([kz, kz]) ** 2)
        W = np.eye(2 * N, dtype=_C)
        V = Q @ np.diag(_inv_lam(lam))
        return W, V, lam

    # E_z elimination (P block): inv([[eps]]) by default, or the supplied
    # Laurent [[1/eps]] for the dual-Laurent (analytic-FT) formulation.
    EPS_inv = ez_laurent_inv if ez_laurent_inv is not None else np.linalg.inv(EPS)
    # P: dH/dz' = P E   (mu = 1 so URC = I).
    P = np.block([
        [Kx @ EPS_inv @ Ky,        I - Kx @ EPS_inv @ Kx],
        [Ky @ EPS_inv @ Ky - I,    -Ky @ EPS_inv @ Kx],
    ])
    OM2 = P @ Q  # Omega^2
    lam2, W = np.linalg.eig(OM2)
    lam = _sqrt_decay(lam2)              # Re >= 0: = i kz (prop.) / |gamma| (evan.)
    V = Q @ W @ np.diag(_inv_lam(lam))   # magnetic-field eigenvectors
    return W, V, lam


def _homogeneous_eigenmodes(Kx, Ky, eps):
    """Analytic eigenmodes of a UNIFORM half-space of scalar permittivity
    ``eps`` -- the reflection (superstrate) and transmission (substrate)
    regions.  Uses the SAME ``V = Q diag(1/lam)`` convention as
    :func:`_layer_eigenmodes` so propagating AND evanescent orders match at
    every interface.  Dimension-agnostic (``N`` inferred from ``Kx``).
    """
    N = np.asarray(Kx).shape[0]
    kx = np.diag(Kx).astype(_C)
    ky = np.diag(Ky).astype(_C)
    kz = _sqrt_forward(eps - kx ** 2 - ky ** 2)   # per-order kz/k0
    lam = _sqrt_decay(-np.concatenate([kz, kz]) ** 2)
    W = np.eye(2 * N, dtype=_C)
    eps_I = eps * np.eye(N, dtype=_C)             # uniform: Laurent == inverse rule
    Q = _layer_Q_matrix(np.asarray(Kx, dtype=_C), np.asarray(Ky, dtype=_C),
                        eps_I, eps_I)
    V = Q @ np.diag(_inv_lam(lam))
    return W, V, kz


# ===========================================================================
# Redheffer scattering-matrix algebra (gap-free interface + propagation)
# ===========================================================================
#
# The global S-matrix is assembled WITHOUT a free-space gap medium: each
# physical interface gets its own scattering matrix and each layer its own
# pure-propagation matrix, star-multiplied together.  The decaying
# propagator exp(-|gamma| L) <= 1 is thereby kept strictly separate from
# the interface matching, which is the textbook-stable decomposition (S4 /
# Whittaker-Culshaw).  A gap medium instead produces huge evanescent
# reflection blocks (vacuum-vs-high-index mode mismatch) that a near-
# singular star denominator then leaks into the propagating orders -- the
# observed nord=15 blow-up.

def _redheffer_star(SA, SB):
    """Redheffer star product of two block S-matrices, each a 4-tuple
    ``(S11, S12, S21, S22)`` of ``2N x 2N`` blocks."""
    A11, A12, A21, A22 = SA
    B11, B12, B21, B22 = SB
    n = A11.shape[0]
    I = np.eye(n, dtype=_C)
    D = np.linalg.inv(I - B11 @ A22)
    F = np.linalg.inv(I - A22 @ B11)
    C11 = A11 + A12 @ D @ B11 @ A21
    C12 = A12 @ D @ B12
    C21 = B21 @ F @ A21
    C22 = B22 + B21 @ F @ A22 @ B12
    return (C11, C12, C21, C22)


def _interface_smatrix(Wa, Va, Wb, Vb):
    """Scattering matrix of the interface from medium ``a`` to medium
    ``b`` (mode matrices ``W, V``), with NO propagation.

    Tangential E and H continuity across the interface gives, with
    ``a = Wb^{-1} Wa`` and ``b = Vb^{-1} Va``::

        S11 = -(a+b)^{-1}(a-b)   S12 = 2 (a+b)^{-1}
        S21 = (a+b)/2 - (a-b)(a+b)^{-1}(a-b)/2   S22 = (a-b)(a+b)^{-1}

    ``solve`` is used for the ``Wb^{-1}Wa`` / ``Vb^{-1}Va`` products so the
    deliberately tiny-columned evanescent eigenvectors do not blow up an
    explicit inverse.
    """
    a = np.linalg.solve(Wb, Wa)
    b = np.linalg.solve(Vb, Va)
    apb = a + b
    amb = a - b
    iapb = np.linalg.inv(apb)
    S11 = -iapb @ amb
    S12 = 2.0 * iapb
    S21 = 0.5 * (apb - amb @ iapb @ amb)
    S22 = amb @ iapb
    return (S11, S12, S21, S22)


def _propagation_smatrix(lam, k0_L):
    """Pure-propagation S-matrix of a layer: forward and backward modes
    each acquire ``X = exp(-lam k0 L)`` (a phase for propagating orders, a
    decay for evanescent ones), with zero self-reflection."""
    n = lam.shape[0]
    X = np.diag(np.exp(-lam * k0_L))
    Z = np.zeros((n, n), dtype=_C)
    return (Z, X, X, Z)


# ===========================================================================
# Public 1-D entry point
# ===========================================================================

def rcwa_efficiency_1d(
    period: float,
    n_ridge: complex,
    n_groove: complex,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    duty_cycle: float,
    wavelength: float,
    *,
    angle: float = 0.0,
    polarization: str = "te",
    n_orders: int = 11,
    formulation: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous diffraction efficiencies of a 1-D binary grating.

    Parameters
    ----------
    period : float
        Grating period (metres).
    n_ridge, n_groove : complex
        Refractive indices of the ridge and groove regions of the patterned
        layer (``n = n + i kappa``, ``kappa > 0`` for loss).
    n_substrate, n_superstrate : complex
        Indices of the transmission (below) and incidence (above)
        half-spaces.
    depth : float
        Grating (layer) thickness (metres).
    duty_cycle : float
        Ridge fraction of the period, in ``[0, 1]``.
    wavelength : float
        Vacuum wavelength (metres).
    angle : float, optional
        Incidence polar angle from the surface normal (radians), in the
        plane of periodicity (planar mounting).  Default 0.
    polarization : {'te', 'tm'}, optional
        ``'te'`` (s, E along grooves / y) or ``'tm'`` (p).  Default ``'te'``.
    n_orders : int, optional
        Number of retained orders per side; total harmonics ``2*n_orders+1``.
        Default 11.
    formulation : {'auto', 'laurent', 'li'}, optional
        Fourier factorization.  ``'laurent'`` (direct rule) converges fast
        for dielectrics; ``'li'`` (inverse rule) is required for metals /
        high-contrast TM.  ``'auto'`` picks ``'li'`` for TM or when any
        index is metallic, else ``'laurent'``.

    Returns
    -------
    orders : (2*n_orders+1,) int ndarray
        Diffraction-order indices, ascending.
    R_eff : (2*n_orders+1,) float ndarray
        Reflected diffraction efficiency per order (evanescent orders 0).
    T_eff : (2*n_orders+1,) float ndarray
        Transmitted diffraction efficiency per order.
    """
    if polarization not in ("te", "tm"):
        raise ValueError(
            f"rcwa_efficiency_1d: polarization must be 'te' or 'tm', got "
            f"{polarization!r}.")
    if not (0.0 <= duty_cycle <= 1.0):
        raise ValueError(
            f"rcwa_efficiency_1d: duty_cycle must be in [0, 1], got "
            f"{duty_cycle}.")

    # --- factorization choice (uses the PUBLIC n = n + i kappa) ----------
    def _metallic(n):
        nv = _C(n)
        return (nv.imag > 1e-6) or (nv.real ** 2 - nv.imag ** 2 < 0)
    is_metal = _metallic(n_ridge) or _metallic(n_groove)

    # Convention bridge: the eigenmode/S-matrix core is derived in the
    # engineering exp(+i omega t) convention (forward wave exp(-i kz z),
    # layer propagator exp(-lam k0 L) with Re(lam) >= 0), in which a passive
    # absorber has Im(eps) < 0.  LumenAiry's public convention is
    # exp(-i omega t) with n = n + i kappa (kappa > 0), i.e. Im(eps) > 0.
    # Conjugating the indices maps a public absorber to the internal loss
    # sign, giving POSITIVE absorptance; lossless (real n) is unaffected and
    # all the lossless validations are bit-identical.  (Complex reflection
    # amplitudes returned to the Jones bridge are conjugated back there.)
    n_ridge = np.conj(_C(n_ridge))
    n_groove = np.conj(_C(n_groove))
    n_substrate = np.conj(_C(n_substrate))
    n_superstrate = np.conj(_C(n_superstrate))
    if formulation == "auto":
        use_li = (polarization == "tm") or is_metal
    elif formulation == "li":
        use_li = True
    elif formulation == "laurent":
        use_li = False
    else:
        raise ValueError(
            f"rcwa_efficiency_1d: formulation must be 'auto', 'laurent' or "
            f"'li', got {formulation!r}.")

    M = int(n_orders)
    N = 2 * M + 1
    orders = np.arange(-M, M + 1)

    n_inc = _C(n_superstrate)
    eps_sup = _C(n_superstrate) ** 2
    eps_sub = _C(n_substrate) ** 2
    kx0 = np.real(n_inc) * np.sin(angle)

    # Wood-anomaly regularisation.  When a diffracted order sits EXACTLY at
    # grazing in a region (kx_m^2 == Re(eps), so kz = 0) the interface
    # S-matrix is singular.  Nudge the WAVELENGTH by a tiny RELATIVE amount
    # (only when an exact grazing is detected) to move that measure-zero
    # singularity away: everything stays real (so lossless energy is still
    # exact and normal-incidence +/-m symmetry is preserved, since kx0 is
    # unchanged), the grazing order -- which carries no z-power -- limits
    # continuously, and every matrix becomes invertible.
    def _grazing(wl):
        kxg = kx0 + orders * (wl / period)
        kzs = np.abs(np.real(eps_sup) - kxg ** 2)
        kzt = np.abs(np.real(eps_sub) - kxg ** 2)
        return np.min(np.concatenate([kzs, kzt]))
    wl_eff = wavelength
    for _ in range(4):
        if _grazing(wl_eff) > 1e-9:
            break
        wl_eff = wl_eff * (1.0 + 1e-7)

    k0 = 2.0 * np.pi / wl_eff
    # Tangential wavevector normalised by k0; planar mounting -> ky = 0.
    # Grating equation kx_m = kx0 + m * lambda/period (standard order sign:
    # order +m carries the +m'th grating vector G = 2*pi/period), matching
    # the diffraction-order labelling used across the RCWA literature.
    kx = kx0 + orders * (wl_eff / period)
    Kx = np.diag(kx.astype(_C))
    Ky = np.zeros((N, N), dtype=_C)

    # --- convolution matrices -------------------------------------------
    EPS, EPS_II = _binary_grating_convolutions(n_ridge, n_groove, duty_cycle, M)
    # Wall-normal E_x uses the Li inverse rule [[1/eps]]^{-1} when requested
    # (TM / metals); E_y (tangential) always uses the Laurent [[eps]].
    EPS_xx = EPS_II if use_li else EPS

    # --- region (half-space) modes --------------------------------------
    Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
    Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)

    # --- global S = (sup|layer) * propagate(layer) * (layer|sub) --------
    Wl, Vl, lam = _layer_eigenmodes(Kx, Ky, EPS, EPS_xx)
    S = _interface_smatrix(Wref, Vref, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
    S11, S12, S21, S22 = S

    # --- incident field (delta on 0th order, chosen polarization) -------
    delta = np.zeros(N, dtype=_C)
    delta[M] = 1.0
    if polarization == "te":
        cinc = np.concatenate([np.zeros(N, dtype=_C), delta])   # E along y
    else:
        cinc = np.concatenate([delta, np.zeros(N, dtype=_C)])   # E along x
    # Source is given in the reflection-region eigenbasis (W_ref = I).
    r = S11 @ cinc            # reflected tangential-E mode amplitudes
    t = S21 @ cinc            # transmitted

    rx, ry = r[:N], r[N:]
    tx, ty = t[:N], t[N:]
    kyv = np.diag(Ky)
    safe_r = np.where(np.abs(kz_ref) < 1e-12, 1.0, kz_ref)
    safe_t = np.where(np.abs(kz_trn) < 1e-12, 1.0, kz_trn)
    # Longitudinal field from div(D) = 0 in each homogeneous region: a
    # diffracted order's full E carries Ez = -(kx Ex + ky Ey)/kz, so the
    # transverse-only |E_t|^2 understates the power by |Ez|^2.
    rz = -(kx * rx + kyv * ry) / safe_r
    tz = -(kx * tx + kyv * ty) / safe_t

    # The injected source has UNIT tangential amplitude (Ex=1 for TM, Ey=1
    # for TE) but the physical incident plane wave also carries a
    # longitudinal Ez_inc, so |E_inc|^2 = 1 + (kx0/kz_inc)^2 = sec^2(theta)
    # for TM (and exactly 1 for TE).  Normalising the diffraction
    # efficiencies by this incident |E|^2 is what keeps sum(R)+sum(T)=1 at
    # oblique TM (without it the sums scale as sec^2 theta).
    kz_inc = np.real(_sqrt_forward(eps_sup - kx0 ** 2))
    if polarization == "te":
        einc_sq = 1.0
    else:
        einc_sq = 1.0 + (kx0 / kz_inc) ** 2
    R_eff = np.real(kz_ref / kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2
                                        + np.abs(rz) ** 2) / einc_sq
    T_eff = np.real(kz_trn / kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                        + np.abs(tz) ** 2) / einc_sq
    R_eff = np.where(np.real(kz_ref) > 0, np.real(R_eff), 0.0)
    T_eff = np.where(np.real(kz_trn) > 0, np.real(T_eff), 0.0)
    return orders, R_eff, T_eff


def rcwa_efficiency_vs_wavelength(
    period: float,
    n_ridge: complex,
    n_groove: complex,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    duty_cycle: float,
    wavelengths,
    *,
    order: int = 1,
    angle: float = 0.0,
    polarization: str = "te",
    n_orders: int = 11,
    formulation: str = "auto",
    quantity: str = "transmitted",
) -> np.ndarray:
    """Rigorous diffraction efficiency of a single order across a wavelength
    sweep -- the spectral companion to :func:`rcwa_efficiency_1d`, mirroring
    :func:`lumenairy.elements.thin_grating.grating_efficiency_vs_wavelength`
    but with the full vector solver (metals, TM, high contrast).

    Parameters
    ----------
    order : int, optional
        Diffraction order whose efficiency is returned (default ``+1``).
    quantity : {'transmitted', 'reflected'}, optional
        Which side's efficiency to return for ``order``.  Default
        ``'transmitted'``.
    wavelengths : float or array-like
        Vacuum wavelength(s) [m].  Scalar indices are assumed dispersionless
        across the sweep; for dispersive media call :func:`rcwa_efficiency_1d`
        per wavelength with the wavelength-specific indices.

    Returns
    -------
    eff : ndarray
        Efficiency of ``order`` at each wavelength (shape matches
        ``wavelengths``); ``0`` where the order is evanescent.

    Other parameters are as in :func:`rcwa_efficiency_1d`.
    """
    if quantity not in ("transmitted", "reflected"):
        raise ValueError(
            f"rcwa_efficiency_vs_wavelength: quantity must be 'transmitted' "
            f"or 'reflected', got {quantity!r}.")
    wl = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    out = np.empty(wl.shape[0], dtype=float)
    for i, w in enumerate(wl):
        orders, R, T = rcwa_efficiency_1d(
            period, n_ridge, n_groove, n_substrate, n_superstrate, depth,
            duty_cycle, float(w), angle=angle, polarization=polarization,
            n_orders=n_orders, formulation=formulation)
        idx = np.searchsorted(orders, order)
        if idx >= orders.shape[0] or orders[idx] != order:
            raise ValueError(
                f"rcwa_efficiency_vs_wavelength: order {order} is outside the "
                f"retained range +/-{n_orders}; increase n_orders.")
        out[i] = (T[idx] if quantity == "transmitted" else R[idx])
    return out if np.ndim(wavelengths) else out[0]


# ===========================================================================
# 2-D crossed gratings (doubly periodic)
# ===========================================================================
#
# The 2-D path reuses the dimension-agnostic vectorial machinery above
# (_layer_eigenmodes, _homogeneous_eigenmodes, the gap-free interface /
# propagation / Redheffer assembly, the _sqrt_forward/_sqrt_decay branches,
# the loss-conjugation bridge and the longitudinal-field efficiency
# formula) verbatim; only the harmonic indexing (a 2-D reciprocal lattice)
# and the permittivity convolution (block-Toeplitz-of-block-Toeplitz)
# become two-dimensional.

def _harmonic_orders_2d(n_orders_x, n_orders_y):
    """Flat list of integer ``(m, n)`` diffraction-order pairs on the 2-D
    reciprocal lattice (``m`` slow in ``[-Mx..Mx]``, ``n`` fast in
    ``[-My..My]``).  Returns ``(orders, N)`` with ``orders`` an ``(N, 2)``
    int array and ``N = (2 Mx + 1)(2 My + 1)``."""
    Mx, My = int(n_orders_x), int(n_orders_y)
    m = np.repeat(np.arange(-Mx, Mx + 1), 2 * My + 1)
    n = np.tile(np.arange(-My, My + 1), 2 * Mx + 1)
    orders = np.stack([m, n], axis=1)
    return orders, orders.shape[0]


def _eps_convolution_2d(eps_cell, orders, n_orders_x, n_orders_y):
    """``N x N`` Laurent (direct-rule) permittivity convolution matrix from a
    one-cell sampling ``eps_cell`` (shape ``(Sx, Sy)``).

    Entry ``[p, p'] = c_{(m-m'), (n-n')}`` where ``c`` are the centred 2-D
    Fourier coefficients of ``eps``; built by vectorised fancy-indexing into
    the coefficient table (the block-Toeplitz-Toeplitz structure).
    """
    Mx, My = int(n_orders_x), int(n_orders_y)
    eps_cell = np.asarray(eps_cell, dtype=_C)
    Sx, Sy = eps_cell.shape
    full = np.fft.fft2(eps_cell) / (Sx * Sy)  # full[k, l] = c_{k,l} (periodic)
    # Coefficient table over the difference range k in [-2Mx..2Mx], l in [-2My..2My].
    krange = np.arange(-2 * Mx, 2 * Mx + 1)
    lrange = np.arange(-2 * My, 2 * My + 1)
    table = full[np.ix_(krange % Sx, lrange % Sy)]  # (4Mx+1, 4My+1)
    dm = orders[:, 0][:, None] - orders[:, 0][None, :]   # (N, N)
    dn = orders[:, 1][:, None] - orders[:, 1][None, :]
    return table[dm + 2 * Mx, dn + 2 * My]


def rcwa_efficiency_2d(
    period_x: float,
    period_y: float,
    eps_cell,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    theta: float = 0.0,
    phi: float = 0.0,
    polarization: str = "te",
    n_orders_x: int = 5,
    n_orders_y: int = 5,
    formulation: str = "laurent",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous diffraction efficiencies of a 2-D (doubly periodic) crossed
    grating: a single patterned layer of permittivity ``eps_cell`` between a
    ``n_superstrate`` half-space and a ``n_substrate`` half-space.

    Parameters
    ----------
    period_x, period_y : float
        Lattice periods along x and y (metres).
    eps_cell : (Sx, Sy) array_like of complex
        Permittivity sampled over one unit cell (PUBLIC convention
        ``Im(eps) > 0`` for loss).  ``Sx``/``Sy`` must comfortably exceed
        ``4*n_orders_{x,y}`` to avoid Fourier aliasing.
    n_substrate, n_superstrate : complex
        Transmission and incidence half-space indices.
    depth : float
        Patterned-layer thickness (metres).
    wavelength : float
        Vacuum wavelength (metres).
    theta, phi : float, optional
        Polar angle from +z and azimuth from +x (radians) of the incident
        plane wave (conical mounting).  Defaults 0.
    polarization : {'te', 'tm'}, optional
        ``'te'`` (s) / ``'tm'`` (p) relative to the plane of incidence.
    n_orders_x, n_orders_y : int, optional
        Retained orders per side along each axis (default 5 -> 11x11 = 121
        harmonics).
    formulation : {'laurent'}, optional
        Fourier factorization.  ``'laurent'`` (direct rule) is correct and
        fast-converging for dielectrics; the fast-Fourier-factorization
        (normal-vector) rule for 2-D metals is provided separately.

    Returns
    -------
    orders : (N, 2) int ndarray
        Diffraction-order pairs ``(m, n)``.
    R_eff, T_eff : (N,) float ndarray
        Reflected / transmitted diffraction efficiency per order.
    """
    if polarization not in ("te", "tm"):
        raise ValueError(
            f"rcwa_efficiency_2d: polarization must be 'te' or 'tm', got "
            f"{polarization!r}.")
    if formulation != "laurent":
        raise ValueError(
            f"rcwa_efficiency_2d: only formulation='laurent' is available in "
            f"this build, got {formulation!r}.")

    # Loss-convention bridge (see rcwa_efficiency_1d): conjugate PUBLIC eps.
    eps_cell = np.conj(np.asarray(eps_cell, dtype=_C))
    eps_sup = np.conj(_C(n_superstrate) ** 2)
    eps_sub = np.conj(_C(n_substrate) ** 2)

    orders, N = _harmonic_orders_2d(n_orders_x, n_orders_y)
    k0 = 2.0 * np.pi / wavelength
    nre = np.real(np.sqrt(eps_sup))
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)
    kx = kx0 + orders[:, 0] * (wavelength / period_x)
    ky = ky0 + orders[:, 1] * (wavelength / period_y)
    Kx = np.diag(kx.astype(_C))
    Ky = np.diag(ky.astype(_C))

    EPS = _eps_convolution_2d(eps_cell, orders, n_orders_x, n_orders_y)
    EPS_xx = EPS  # Laurent rule: wall-normal convolution == [[eps]]

    Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
    Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)
    Wl, Vl, lam = _layer_eigenmodes(Kx, Ky, EPS, EPS_xx)
    S = _interface_smatrix(Wref, Vref, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
    S11, S12, S21, S22 = S

    # Incident unit plane wave on the (0, 0) order, TE/TM relative to the
    # plane of incidence (built from the in-plane azimuth direction).
    p0 = int(np.where((orders[:, 0] == 0) & (orders[:, 1] == 0))[0][0])
    delta = np.zeros(N, dtype=_C)
    delta[p0] = 1.0
    kz_inc = np.real(_sqrt_forward(eps_sup - kx0 ** 2 - ky0 ** 2))
    kt = np.hypot(kx0, ky0)
    if kt < 1e-12:                       # normal incidence
        ex0, ey0 = (0.0, 1.0) if polarization == "te" else (1.0, 0.0)
        einc_sq = 1.0
    else:
        ax, ay = kx0 / kt, ky0 / kt      # in-plane (rho) unit vector
        if polarization == "te":
            ex0, ey0 = -ay, ax           # s-pol: perpendicular, no z-component
            einc_sq = 1.0
        else:
            ex0, ey0 = ax, ay            # p-pol transverse part along rho
            einc_sq = 1.0 + (kt / kz_inc) ** 2
    cinc = np.concatenate([ex0 * delta, ey0 * delta])

    r = S11 @ cinc
    t = S21 @ cinc
    rx, ry = r[:N], r[N:]
    tx, ty = t[:N], t[N:]
    safe_r = np.where(np.abs(kz_ref) < 1e-12, 1.0, kz_ref)
    safe_t = np.where(np.abs(kz_trn) < 1e-12, 1.0, kz_trn)
    rz = -(kx * rx + ky * ry) / safe_r
    tz = -(kx * tx + ky * ty) / safe_t
    R_eff = np.real(kz_ref / kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2
                                        + np.abs(rz) ** 2) / einc_sq
    T_eff = np.real(kz_trn / kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                        + np.abs(tz) ** 2) / einc_sq
    R_eff = np.where(np.real(kz_ref) > 0, np.real(R_eff), 0.0)
    T_eff = np.where(np.real(kz_trn) > 0, np.real(T_eff), 0.0)
    return orders, R_eff, T_eff


# ===========================================================================
# 1-D anisotropic gratings (full in-plane permittivity tensor -- LC subset)
# ===========================================================================
#
# A layer whose permittivity is a 3x3 tensor with a full in-plane block
# [[exx, exy], [eyx, eyy]] (plus ezz; eps_xz = eps_yz = 0, the z-decoupled
# liquid-crystal subset).  The in-plane block couples TE and TM, so the
# reflection is a full 2x2 Jones matrix.  Factorization is Li 1996
# (anisotropic 1-D): the wall-normal x-direction uses the inverse rule, the
# tangential y the direct rule.

def uniaxial_tensor(n_o, n_e, theta, *, phi=0.0):
    """``(3, 3)`` rotated uniaxial permittivity tensor (the liquid-crystal /
    birefringent layer).

    The optic axis (director) starts along ``+z``
    (``eps = diag(n_o^2, n_o^2, n_e^2)``), is tilted by the polar angle
    ``theta`` from ``+z`` toward ``+x``, then rotated azimuthally by ``phi``
    about ``+z`` (``R = Rz(phi) @ Ry(theta)``).  ``theta = pi/2`` puts the
    director in the x-y plane (the in-plane-rotating LC), giving a full
    in-plane block with ``ezz = n_o^2`` -- the z-decoupled subset used by
    the 1-D / 2-D anisotropic solvers.

    Parameters
    ----------
    n_o, n_e : complex
        Ordinary and extraordinary indices (``n + i kappa`` for loss).
    theta : float
        Director polar tilt from ``+z`` (radians).
    phi : float, optional
        Director azimuth about ``+z`` (radians).  Default 0.

    Returns
    -------
    (3, 3) complex ndarray
        Permittivity tensor in the lab ``(x, y, z)`` basis.
    """
    eo = _C(n_o) ** 2
    ee = _C(n_e) ** 2
    eps_diag = np.diag([eo, eo, ee]).astype(_C)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    Ry = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]], dtype=_C)
    Rz = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]], dtype=_C)
    R = Rz @ Ry
    return R @ eps_diag @ R.T


def _toeplitz_of_profile(profile, n_orders):
    """Laurent (direct-rule) Toeplitz ``[[f]]`` of a sampled one-period
    profile."""
    return _toeplitz_1d(_fourier_coeffs_1d(profile, 2 * n_orders + 1), n_orders)


def _inv_toeplitz_of_profile(profile, n_orders):
    """Inverse-rule operator ``[[1/f]]^{-1}`` of a sampled one-period
    profile."""
    return np.linalg.inv(
        _toeplitz_1d(_fourier_coeffs_1d(1.0 / profile, 2 * n_orders + 1),
                     n_orders))


def _tensor_convolutions(profiles, n_orders):
    """Anisotropic 1-D Fourier operators (Li 1996; wall normal along x).

    ``profiles`` holds the one-period samplings of the tensor components
    ``xx, xy, yx, yy, zz``.  Returns ``(Cxx, Cxy, Cyx, Cyy, EZZ)`` where
    ``[Dx; Dy] = [[Cxx, Cxy], [Cyx, Cyy]] [Ex; Ey]`` and ``EZZ = [[ezz]]``
    (the wall-tangential ``E_z`` uses the direct rule, inverted later in the
    ``P`` block).  Reduces to ``Cxx = Cyy = [[eps]]``, ``Cxy = Cyx = 0`` for
    a scalar (isotropic) tensor.
    """
    a = np.asarray(profiles["xx"], dtype=_C)
    b = np.asarray(profiles["xy"], dtype=_C)
    c = np.asarray(profiles["yx"], dtype=_C)
    d = np.asarray(profiles["yy"], dtype=_C)
    ezz = np.asarray(profiles["zz"], dtype=_C)
    inv_a = _inv_toeplitz_of_profile(a, n_orders)             # [[1/exx]]^{-1}
    T_b_a = _toeplitz_of_profile(b / a, n_orders)             # [[exy/exx]]
    T_c_a = _toeplitz_of_profile(c / a, n_orders)             # [[eyx/exx]]
    T_schur = _toeplitz_of_profile(d - c * b / a, n_orders)   # [[eyy - eyx exy/exx]]
    Cxx = inv_a
    Cxy = inv_a @ T_b_a
    Cyx = T_c_a @ inv_a
    Cyy = T_schur + T_c_a @ inv_a @ T_b_a
    EZZ = _toeplitz_of_profile(ezz, n_orders)
    return Cxx, Cxy, Cyx, Cyy, EZZ


def _layer_eigenmodes_tensor(Kx, Ky, Cxx, Cxy, Cyx, Cyy, EZZ):
    """Eigenmodes of a full-in-plane-tensor layer (dimension-agnostic).

    The anisotropic ``Q`` block (rigorously derived and locked to the
    isotropic core by ``b = c = 0`` reduction; getting the ``Cyx`` sign
    wrong silently violates energy conservation at off-axis director
    angles) is::

        Q = [[ Cyx + Kx Ky,   Cyy - Kx Kx ],
             [ Ky Ky - Cxx,   -(Cxy + Ky Kx) ]]

    The ``P`` block is the core's, with the ``E_z`` elimination ``inv(EZZ)``.
    """
    Kx = np.asarray(Kx, dtype=_C)
    Ky = np.asarray(Ky, dtype=_C)
    N = Kx.shape[0]
    I = np.eye(N, dtype=_C)
    Ez_inv = np.linalg.inv(EZZ)
    P = np.block([
        [Kx @ Ez_inv @ Ky,        I - Kx @ Ez_inv @ Kx],
        [Ky @ Ez_inv @ Ky - I,    -Ky @ Ez_inv @ Kx],
    ])
    Q = np.block([
        [Cyx + Kx @ Ky,        Cyy - Kx @ Kx],
        [Ky @ Ky - Cxx,        -(Cxy + Ky @ Kx)],
    ])
    OM2 = P @ Q
    lam2, W = np.linalg.eig(OM2)
    lam = _sqrt_decay(lam2)
    V = Q @ W @ np.diag(_inv_lam(lam))
    return W, V, lam


def rcwa_jones_1d(
    period: float,
    eps_ridge,
    eps_groove,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    duty_cycle: float,
    wavelength: float,
    *,
    angle: float = 0.0,
    n_orders: int = 11,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous 1-D anisotropic grating: a binary grating whose ridge and
    groove are full ``(3, 3)`` permittivity tensors (the liquid-crystal /
    birefringent case).  Because the in-plane tensor couples TE and TM, the
    response is a full Jones matrix.

    Parameters
    ----------
    period : float
        Grating period (metres).
    eps_ridge, eps_groove : (3, 3) array_like of complex
        Permittivity tensors of the ridge and groove (PUBLIC convention
        ``Im(eps) > 0`` for loss).  Pass ``scalar * np.eye(3)`` for an
        isotropic region; build LC tensors with :func:`uniaxial_tensor`.
    n_substrate, n_superstrate : complex
        Transmission / incidence half-space (isotropic) indices.
    depth, duty_cycle, wavelength, angle, n_orders
        As in :func:`rcwa_efficiency_1d` (ridge occupies ``duty_cycle`` of
        the period; planar incidence at ``angle``).

    Returns
    -------
    orders : (2*n_orders+1,) int ndarray
        Diffraction-order indices.
    R_eff, T_eff : (2, 2*n_orders+1) float ndarray
        Reflected / transmitted diffraction efficiency per order; row 0 is
        the response to an incident ``E_x`` wave, row 1 to incident
        ``E_y``.
    jones_reflection : (2, 2) complex ndarray
        Zeroth-order Jones reflection matrix in the lab ``(x, y)`` basis
        (PUBLIC ``exp(-i w t)`` convention); columns are the responses to
        incident ``E_x`` / ``E_y``, rows are ``[E_x; E_y]`` reflected.
    """
    if not (0.0 <= duty_cycle <= 1.0):
        raise ValueError(
            f"rcwa_jones_1d: duty_cycle must be in [0, 1], got {duty_cycle}.")
    M = int(n_orders)
    N = 2 * M + 1
    orders = np.arange(-M, M + 1)

    # Loss-convention bridge: conjugate ALL public eps (tensor + regions).
    eps_ridge = np.conj(np.asarray(eps_ridge, dtype=_C))
    eps_groove = np.conj(np.asarray(eps_groove, dtype=_C))
    eps_sup = np.conj(_C(n_superstrate) ** 2)
    eps_sub = np.conj(_C(n_substrate) ** 2)
    n_inc = np.conj(_C(n_superstrate))

    k0 = 2.0 * np.pi / wavelength
    kx0 = np.real(n_inc) * np.sin(angle)
    kx = kx0 + orders * (wavelength / period)
    Kx = np.diag(kx.astype(_C))
    Ky = np.zeros((N, N), dtype=_C)

    # Sample the per-component profiles across one period (ridge over duty).
    n_samples = 4096
    xq = (np.arange(n_samples) + 0.5) / n_samples
    inside = xq < duty_cycle
    profiles = {}
    for key, (ii, jj) in {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0),
                          "yy": (1, 1), "zz": (2, 2)}.items():
        profiles[key] = np.where(inside, eps_ridge[ii, jj],
                                 eps_groove[ii, jj]).astype(_C)
    Cxx, Cxy, Cyx, Cyy, EZZ = _tensor_convolutions(profiles, M)

    Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
    Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)
    Wl, Vl, lam = _layer_eigenmodes_tensor(Kx, Ky, Cxx, Cxy, Cyx, Cyy, EZZ)
    S = _interface_smatrix(Wref, Vref, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
    S11, S12, S21, S22 = S

    delta = np.zeros(N, dtype=_C)
    delta[M] = 1.0
    kz_inc = np.real(_sqrt_forward(eps_sup - kx0 ** 2))
    R_eff = np.zeros((2, N))
    T_eff = np.zeros((2, N))
    jones_reflection = np.zeros((2, 2), dtype=_C)
    for col, pol in enumerate(("x", "y")):
        if pol == "x":
            cinc = np.concatenate([delta, np.zeros(N, dtype=_C)])
            einc_sq = 1.0 + (kx0 / kz_inc) ** 2 if kz_inc != 0 else 1.0
        else:
            cinc = np.concatenate([np.zeros(N, dtype=_C), delta])
            einc_sq = 1.0
        r = S11 @ cinc
        t = S21 @ cinc
        rx, ry = r[:N], r[N:]
        tx, ty = t[:N], t[N:]
        safe_r = np.where(np.abs(kz_ref) < 1e-12, 1.0, kz_ref)
        safe_t = np.where(np.abs(kz_trn) < 1e-12, 1.0, kz_trn)
        rz = -(kx * rx) / safe_r
        tz = -(kx * tx) / safe_t
        Re = np.real(kz_ref / kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2
                                         + np.abs(rz) ** 2) / einc_sq
        Te = np.real(kz_trn / kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                         + np.abs(tz) ** 2) / einc_sq
        R_eff[col] = np.where(np.real(kz_ref) > 0, np.real(Re), 0.0)
        T_eff[col] = np.where(np.real(kz_trn) > 0, np.real(Te), 0.0)
        # Zeroth-order Jones (conjugate back to the public exp(-i w t)).
        jones_reflection[0, col] = np.conj(rx[M])
        jones_reflection[1, col] = np.conj(ry[M])
    return orders, R_eff, T_eff, jones_reflection


def rcwa_jones_2d(
    period_x: float,
    period_y: float,
    eps_tensor_cell,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    theta: float = 0.0,
    phi: float = 0.0,
    n_orders_x: int = 5,
    n_orders_y: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous 2-D (doubly periodic) anisotropic grating: a single layer
    whose permittivity is a full in-plane TENSOR FIELD (the z-decoupled LC
    subset; Li 2003, direct-rule factorization).  Returns diffraction
    efficiencies for both incident linear polarizations plus the 2x2
    zeroth-order Jones reflection matrix.

    Parameters
    ----------
    period_x, period_y : float
        Lattice periods (metres).
    eps_tensor_cell : (Sx, Sy, 3, 3) array_like of complex
        Per-pixel permittivity tensor over one unit cell (PUBLIC convention
        ``Im(eps) > 0`` for loss).  Only the in-plane block ``[[xx, xy],
        [yx, yy]]`` and ``zz`` are used.  ``Sx``/``Sy`` must exceed
        ``4*n_orders_{x,y}``.
    n_substrate, n_superstrate, depth, wavelength, theta, phi,
    n_orders_x, n_orders_y
        As in :func:`rcwa_efficiency_2d`.

    Returns
    -------
    orders : (N, 2) int ndarray
        Diffraction-order pairs ``(m, n)``.
    R_eff, T_eff : (2, N) float ndarray
        Diffraction efficiencies per order; row 0 is the response to an
        incident ``E_x`` plane wave, row 1 to incident ``E_y``.
    jones_reflection : (2, 2) complex ndarray
        Zeroth-order Jones reflection matrix in the lab ``(x, y)`` basis
        (columns = response to incident ``E_x`` / ``E_y``).

    Notes
    -----
    Uses the direct (Laurent) tensor factorization, which is exactly
    energy-conserving for a lossless tensor and reduces to
    :func:`rcwa_efficiency_2d` for a scalar cell; it converges fastest for
    smooth / dielectric anisotropic media (e.g. liquid-crystal cells).
    """
    eps_t = np.conj(np.asarray(eps_tensor_cell, dtype=_C))
    eps_sup = np.conj(_C(n_superstrate) ** 2)
    eps_sub = np.conj(_C(n_substrate) ** 2)

    orders, N = _harmonic_orders_2d(n_orders_x, n_orders_y)
    k0 = 2.0 * np.pi / wavelength
    nre = np.real(np.sqrt(eps_sup))
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)
    kx = kx0 + orders[:, 0] * (wavelength / period_x)
    ky = ky0 + orders[:, 1] * (wavelength / period_y)
    Kx = np.diag(kx.astype(_C))
    Ky = np.diag(ky.astype(_C))

    # Direct-rule (Laurent) convolution of each tensor component.
    def _conv(comp):
        return _eps_convolution_2d(comp, orders, n_orders_x, n_orders_y)
    Cxx = _conv(eps_t[:, :, 0, 0])
    Cxy = _conv(eps_t[:, :, 0, 1])
    Cyx = _conv(eps_t[:, :, 1, 0])
    Cyy = _conv(eps_t[:, :, 1, 1])
    EZZ = _conv(eps_t[:, :, 2, 2])

    Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
    Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)
    Wl, Vl, lam = _layer_eigenmodes_tensor(Kx, Ky, Cxx, Cxy, Cyx, Cyy, EZZ)
    S = _interface_smatrix(Wref, Vref, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
    S11, S12, S21, S22 = S

    p0 = int(np.where((orders[:, 0] == 0) & (orders[:, 1] == 0))[0][0])
    delta = np.zeros(N, dtype=_C)
    delta[p0] = 1.0
    kz_inc = np.real(_sqrt_forward(eps_sup - kx0 ** 2 - ky0 ** 2))
    safe_r = np.where(np.abs(kz_ref) < 1e-12, 1.0, kz_ref)
    safe_t = np.where(np.abs(kz_trn) < 1e-12, 1.0, kz_trn)
    R_eff = np.zeros((2, N))
    T_eff = np.zeros((2, N))
    jones_reflection = np.zeros((2, 2), dtype=_C)
    for col, (ex0, ey0) in enumerate(((1.0, 0.0), (0.0, 1.0))):
        # Unit tangential E along (ex0, ey0); the incident wave's longitudinal
        # Ez = -(kx0 ex + ky0 ey)/kz_inc inflates |E_inc|^2 (cf. the 1-D sec^2).
        long_inc = (kx0 * ex0 + ky0 * ey0)
        einc_sq = 1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0 else 1.0
        cinc = np.concatenate([ex0 * delta, ey0 * delta])
        r = S11 @ cinc
        t = S21 @ cinc
        rx, ry = r[:N], r[N:]
        tx, ty = t[:N], t[N:]
        rz = -(kx * rx + ky * ry) / safe_r
        tz = -(kx * tx + ky * ty) / safe_t
        Re = np.real(kz_ref / kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2
                                         + np.abs(rz) ** 2) / einc_sq
        Te = np.real(kz_trn / kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                         + np.abs(tz) ** 2) / einc_sq
        R_eff[col] = np.where(np.real(kz_ref) > 0, np.real(Re), 0.0)
        T_eff[col] = np.where(np.real(kz_trn) > 0, np.real(Te), 0.0)
        jones_reflection[0, col] = np.conj(rx[p0])
        jones_reflection[1, col] = np.conj(ry[p0])
    return orders, R_eff, T_eff, jones_reflection


# ===========================================================================
# Analytic shape Fourier transforms + dual-Laurent 2-D factorization
# ===========================================================================
#
# For known shapes (rectangle, disk, ellipse) the permittivity Fourier
# coefficients are computed in CLOSED FORM (exact form factors) instead of
# by FFT-sampling a pixelated cell -- eliminating the aliasing / staircase
# error of pixelation, so the spectrum is exact and convergence is clean.
# Both [[eps]] and [[1/eps]] are built from the SAME analytic form factors,
# and the layer eigenproblem uses the dual-Laurent factorization (the
# in-plane Q block uses [[eps]]; the E_z elimination uses [[1/eps]]
# directly) -- the formulation used by mature analytic-FT FMM codes.

def _shape_form_factor(shape, gxv, gyv, period_x, period_y):
    """Analytic Fourier form factor ``F(G) = (1/A_cell) integral_shape
    exp(-i G.r) d^2r`` for one shape, at the (difference) reciprocal vectors
    ``(gxv, gyv)`` [1/m].  Closed form for ``rectangle`` / ``disk`` /
    ``ellipse``; the ``G = 0`` entry is the area fraction."""
    kind = shape["shape"]
    cx, cy = shape.get("center", (period_x / 2.0, period_y / 2.0))
    area = period_x * period_y
    phase = np.exp(-1j * (gxv * cx + gyv * cy))
    if kind == "rectangle":
        wx, wy = shape["size"]
        # np.sinc(z) = sin(pi z)/(pi z), so sinc(G w / (2 pi)) = sin(G w/2)/(G w/2).
        sx = np.sinc(gxv * wx / (2.0 * np.pi))
        sy = np.sinc(gyv * wy / (2.0 * np.pi))
        return (wx * wy / area) * sx * sy * phase
    if kind in ("disk", "ellipse"):
        from scipy.special import j1
        if kind == "disk":
            ax = ay = shape["radius"]
        else:
            ax, ay = shape["semi_axes"]
        q = np.sqrt((gxv * ax) ** 2 + (gyv * ay) ** 2)
        small = q < 1e-12
        qsafe = np.where(small, 1.0, q)
        bessel = np.where(small, 1.0, 2.0 * j1(qsafe) / qsafe)   # -> 1 as q -> 0
        return (np.pi * ax * ay / area) * bessel * phase
    raise ValueError(
        f"_shape_form_factor: unknown shape {kind!r} (expected 'rectangle', "
        f"'disk' or 'ellipse').")


def _analytic_convolutions_2d(eps_background, shapes, orders, n_orders_x,
                              n_orders_y, period_x, period_y):
    """Analytic ``[[eps]]`` and ``[[1/eps]]`` convolution matrices for a 2-D
    unit cell of background ``eps_background`` overlaid with ``shapes`` (each
    a dict ``{'shape', 'eps', geometry, ['center']}``).  Returns
    ``(EPS, EPS_inv_laurent)``."""
    Mx, My = int(n_orders_x), int(n_orders_y)
    ks = np.arange(-2 * Mx, 2 * Mx + 1)
    ls = np.arange(-2 * My, 2 * My + 1)
    KK, LL = np.meshgrid(ks, ls, indexing="ij")
    gxv = KK * (2.0 * np.pi / period_x)
    gyv = LL * (2.0 * np.pi / period_y)
    eps_bg = _C(eps_background)
    c_eps = np.zeros(KK.shape, dtype=_C)
    c_inv = np.zeros(KK.shape, dtype=_C)
    c_eps[2 * Mx, 2 * My] = eps_bg            # background (DC) term
    c_inv[2 * Mx, 2 * My] = 1.0 / eps_bg
    for sh in shapes:
        eps_s = _C(sh["eps"])
        F = _shape_form_factor(sh, gxv, gyv, period_x, period_y)
        c_eps = c_eps + (eps_s - eps_bg) * F
        c_inv = c_inv + (1.0 / eps_s - 1.0 / eps_bg) * F
    dm = orders[:, 0][:, None] - orders[:, 0][None, :]
    dn = orders[:, 1][:, None] - orders[:, 1][None, :]
    EPS = c_eps[dm + 2 * Mx, dn + 2 * My]
    EPS_inv = c_inv[dm + 2 * Mx, dn + 2 * My]
    return EPS, EPS_inv


def rcwa_efficiency_2d_shapes(
    period_x: float,
    period_y: float,
    eps_background: complex,
    shapes,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    theta: float = 0.0,
    phi: float = 0.0,
    polarization: str = "te",
    n_orders_x: int = 5,
    n_orders_y: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous 2-D crossed-grating efficiencies using **analytic** shape
    Fourier transforms and the dual-Laurent factorization.

    The patterned layer is a background permittivity ``eps_background``
    overlaid with analytically-described ``shapes`` (no pixelation), so the
    permittivity spectrum is exact and convergence is clean.

    Parameters
    ----------
    period_x, period_y : float
        Lattice periods (metres).
    eps_background : complex
        Permittivity of the unpatterned background (PUBLIC convention
        ``Im(eps) > 0`` lossy).
    shapes : list of dict
        Each shape is ``{'shape': 'rectangle'|'disk'|'ellipse', 'eps':
        complex, ...geometry..., 'center': (cx, cy) [m]}``; geometry is
        ``'size': (wx, wy)`` for a rectangle, ``'radius': r`` for a disk,
        ``'semi_axes': (ax, ay)`` for an ellipse (all metres).  Shapes are
        painted in order over the background.
    n_substrate, n_superstrate, depth, wavelength, theta, phi, polarization,
    n_orders_x, n_orders_y
        As in :func:`rcwa_efficiency_2d`.

    Returns
    -------
    orders : (N, 2) int ndarray
    R_eff, T_eff : (N,) float ndarray
    """
    if polarization not in ("te", "tm"):
        raise ValueError(
            f"rcwa_efficiency_2d_shapes: polarization must be 'te' or 'tm', "
            f"got {polarization!r}.")
    # Loss-sign bridge: conjugate every public permittivity.
    eps_bg = np.conj(_C(eps_background))
    shapes_c = [dict(s, eps=np.conj(_C(s["eps"]))) for s in shapes]
    eps_sup = np.conj(_C(n_superstrate) ** 2)
    eps_sub = np.conj(_C(n_substrate) ** 2)

    orders, N = _harmonic_orders_2d(n_orders_x, n_orders_y)
    k0 = 2.0 * np.pi / wavelength
    nre = np.real(np.sqrt(eps_sup))
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)
    kx = kx0 + orders[:, 0] * (wavelength / period_x)
    ky = ky0 + orders[:, 1] * (wavelength / period_y)
    Kx = np.diag(kx.astype(_C))
    Ky = np.diag(ky.astype(_C))

    EPS, EPS_inv = _analytic_convolutions_2d(
        eps_bg, shapes_c, orders, n_orders_x, n_orders_y, period_x, period_y)

    Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
    Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)
    # Dual-Laurent: in-plane uses [[eps]], E_z elimination uses [[1/eps]].
    Wl, Vl, lam = _layer_eigenmodes(Kx, Ky, EPS, EPS, ez_laurent_inv=EPS_inv)
    S = _interface_smatrix(Wref, Vref, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
    S11, S12, S21, S22 = S

    p0 = int(np.where((orders[:, 0] == 0) & (orders[:, 1] == 0))[0][0])
    delta = np.zeros(N, dtype=_C)
    delta[p0] = 1.0
    kz_inc = np.real(_sqrt_forward(eps_sup - kx0 ** 2 - ky0 ** 2))
    kt = np.hypot(kx0, ky0)
    if kt < 1e-12:
        ex0, ey0 = (0.0, 1.0) if polarization == "te" else (1.0, 0.0)
        einc_sq = 1.0
    else:
        ax, ay = kx0 / kt, ky0 / kt
        if polarization == "te":
            ex0, ey0 = -ay, ax
            einc_sq = 1.0
        else:
            ex0, ey0 = ax, ay
            einc_sq = 1.0 + (kt / kz_inc) ** 2
    cinc = np.concatenate([ex0 * delta, ey0 * delta])
    r = S11 @ cinc
    t = S21 @ cinc
    rx, ry = r[:N], r[N:]
    tx, ty = t[:N], t[N:]
    safe_r = np.where(np.abs(kz_ref) < 1e-12, 1.0, kz_ref)
    safe_t = np.where(np.abs(kz_trn) < 1e-12, 1.0, kz_trn)
    rz = -(kx * rx + ky * ry) / safe_r
    tz = -(kx * tx + ky * ty) / safe_t
    R_eff = np.real(kz_ref / kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2
                                        + np.abs(rz) ** 2) / einc_sq
    T_eff = np.real(kz_trn / kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                        + np.abs(tz) ** 2) / einc_sq
    R_eff = np.where(np.real(kz_ref) > 0, np.real(R_eff), 0.0)
    T_eff = np.where(np.real(kz_trn) > 0, np.real(T_eff), 0.0)
    return orders, R_eff, T_eff


# ===========================================================================
# JAX backend: differentiable 1-D RCWA for inverse design (adjoint/autodiff)
# ===========================================================================
#
# A JAX reimplementation of the 1-D isotropic core whose efficiencies are
# differentiable w.r.t. continuous design parameters (layer permittivities,
# depth, angle) -- the enabler for gradient-based metasurface inverse
# design.  JAX is an OPTIONAL dependency, imported lazily so the rest of the
# module (and CI without jax) is unaffected.
#
# The one hard gradient is through ``jnp.linalg.eig`` of the non-Hermitian
# layer matrix (reverse-mode AD of general eig is unstable near degenerate
# eigenvalues).  We register a custom VJP with the torcwa-style
# Lorentzian-broadened eigenvector gradient (``eps_broaden``) plus a
# canonical eigenvector gauge, so the decomposition is a deterministic,
# differentiable function of the matrix (validated entrywise vs complex
# finite differences to < 1e-5).

_JAX_EIG_STABLE = None


def _jax_eig_stable():
    """Lazily build (once) and return a differentiable, gauge-fixed ``eig``
    for complex non-Hermitian matrices.  Raises ImportError if JAX absent."""
    global _JAX_EIG_STABLE
    if _JAX_EIG_STABLE is not None:
        return _JAX_EIG_STABLE
    from functools import partial

    import jax
    import jax.numpy as jnp

    @partial(jax.custom_vjp, nondiff_argnums=(1,))
    def _eig_raw(A, eps_broaden=1e-10):
        return jnp.linalg.eig(A)

    def _eig_raw_fwd(A, eps_broaden):
        lam, V = jnp.linalg.eig(A)
        return (lam, V), (lam, V)

    def _eig_raw_bwd(eps_broaden, res, cot):
        lam, V = res
        lam_bar, V_bar = cot
        D = lam[None, :] - lam[:, None]
        n = lam.shape[0]
        offdiag = 1.0 - jnp.eye(n, dtype=D.dtype)
        denom = jnp.abs(D) ** 2 + eps_broaden
        F = jnp.where(offdiag != 0, D / jnp.where(denom == 0, 1.0, denom), 0.0)
        Vinv = jnp.linalg.inv(V)
        VinvH = jnp.conj(Vinv).T
        VH = jnp.conj(V).T
        Mmat = VH @ jnp.conj(V_bar)
        inner = jnp.diag(jnp.conj(lam_bar)) + F * Mmat
        # Conjugate the matrix cotangent to match JAX's Wirtinger convention
        # (dL/d conj(A)); without it the eig-path gradient is silently wrong
        # (verified by a gauge-invariant grid search over the conjugations).
        return (jnp.conj(VinvH @ inner @ VH),)

    _eig_raw.defvjp(_eig_raw_fwd, _eig_raw_bwd)

    # The physical RCWA efficiencies are GAUGE-INVARIANT (independent of the
    # eigenvector phase / ordering), so the solver uses the raw eig with the
    # Lorentzian-broadened custom VJP directly.  A canonical gauge fix
    # (unit-norm + phase pivot + sort) would only be needed if a downstream
    # quantity depended on the gauge -- and its argmax / lexsort are
    # non-differentiable, which corrupts the gradient of a gauge-invariant
    # loss (observed: a 3x error on the eig-path gradient).  So it is omitted.
    _JAX_EIG_STABLE = _eig_raw
    return _eig_raw


def rcwa_efficiency_1d_jax(
    period,
    n_ridge,
    n_groove,
    n_substrate,
    n_superstrate,
    depth,
    duty_cycle,
    wavelength,
    *,
    angle=0.0,
    polarization="te",
    n_orders=11,
    formulation="auto",
    n_samples=512,
):
    """JAX (differentiable) twin of :func:`rcwa_efficiency_1d`.

    Returns ``(orders, R_eff, T_eff)`` as JAX arrays; the efficiencies are
    differentiable w.r.t. ``n_ridge``, ``n_groove``, ``depth`` and ``angle``
    (pass them as JAX tracers / floats), enabling ``jax.grad`` /
    ``jax.value_and_grad`` gradient-based metasurface inverse design.

    Matches :func:`rcwa_efficiency_1d` numerically (same physics,
    conventions and loss-sign bridge); the binary profile is sampled on
    ``n_samples`` points with a soft edge (differentiable w.r.t. the index
    VALUES rather than the discrete duty-cycle threshold).  Requires the
    optional ``jax`` extra.  Assumes no order sits exactly at grazing (no
    Wood-anomaly nudge on the differentiable path).
    """
    import jax.numpy as jnp
    eig_stable = _jax_eig_stable()
    C = jnp.complex128

    if polarization not in ("te", "tm"):
        raise ValueError("polarization must be 'te' or 'tm'.")

    nr0 = complex(n_ridge) if not hasattr(n_ridge, "shape") else n_ridge
    ng0 = complex(n_groove) if not hasattr(n_groove, "shape") else n_groove
    is_metal = (np.imag(complex(nr0)) > 1e-6 if not hasattr(nr0, "shape") else False) \
        or (np.imag(complex(ng0)) > 1e-6 if not hasattr(ng0, "shape") else False)
    use_li = (formulation == "li") or (
        formulation == "auto" and (polarization == "tm" or is_metal))

    M = int(n_orders)
    N = 2 * M + 1
    orders = jnp.arange(-M, M + 1)
    # Loss-sign bridge: conjugate the public indices (engineering exp(+iwt)).
    nr = jnp.conj(jnp.asarray(n_ridge, dtype=C))
    ng = jnp.conj(jnp.asarray(n_groove, dtype=C))
    eps_sup = jnp.conj(jnp.asarray(n_superstrate, dtype=C) ** 2)
    eps_sub = jnp.conj(jnp.asarray(n_substrate, dtype=C) ** 2)

    k0 = 2.0 * jnp.pi / wavelength
    kx0 = jnp.real(jnp.sqrt(eps_sup)) * jnp.sin(angle)
    kx = kx0 + orders * (wavelength / period)
    Kx = jnp.diag(kx.astype(C))

    xq = (jnp.arange(n_samples) + 0.5) / n_samples
    soft = 0.5 * (1.0 + jnp.tanh((duty_cycle - xq) * n_samples))
    eps_prof = ng ** 2 + (nr ** 2 - ng ** 2) * soft
    ncoef = 2 * N - 1
    centre = ncoef // 2
    ks = jnp.arange(-centre, centre + 1)
    mi = jnp.arange(N)
    tidx = centre + (mi[:, None] - mi[None, :])

    def _toeplitz(prof):
        full = jnp.fft.fft(prof) / n_samples
        coef = full[ks % n_samples]
        return coef[tidx]

    EPS = _toeplitz(eps_prof)
    EPS_xx = jnp.linalg.inv(_toeplitz(1.0 / eps_prof)) if use_li else EPS

    def _sqrt_fwd(x):
        r = jnp.sqrt(x.astype(C))
        bad = (jnp.imag(r) < 0) | ((jnp.abs(jnp.imag(r)) <= 1e-300) & (jnp.real(r) < 0))
        return jnp.where(bad, -r, r)

    def _sqrt_dec(x):
        r = jnp.sqrt(x.astype(C))
        return jnp.where((jnp.real(r) == 0) & (jnp.imag(r) < 0), -r, r)

    Zb = jnp.zeros((N, N), dtype=C)
    Ib = jnp.eye(N, dtype=C)

    def _homog(eps):
        kz = _sqrt_fwd(eps - kx ** 2)
        lam_h = _sqrt_dec(-jnp.concatenate([kz, kz]) ** 2)
        Qh = jnp.block([[Zb, eps * Ib - Kx @ Kx], [-eps * Ib, Zb]])
        return jnp.eye(2 * N, dtype=C), Qh @ jnp.diag(1.0 / lam_h), kz

    Q = jnp.block([[Zb, EPS - Kx @ Kx], [-EPS_xx, Zb]])
    P = jnp.block([[Zb, Ib - Kx @ jnp.linalg.inv(EPS) @ Kx], [-Ib, Zb]])
    lam2, W = eig_stable(P @ Q)
    lam = _sqrt_dec(lam2)
    Vl = Q @ W @ jnp.diag(1.0 / lam)

    Wref, Vref, kz_ref = _homog(eps_sup)
    Wtrn, Vtrn, kz_trn = _homog(eps_sub)
    X = jnp.diag(jnp.exp(-lam * k0 * depth))

    def _iface(Wa, Va, Wb, Vb):
        a = jnp.linalg.solve(Wb, Wa)
        b = jnp.linalg.solve(Vb, Va)
        apb, amb = a + b, a - b
        iapb = jnp.linalg.inv(apb)
        return (-iapb @ amb, 2.0 * iapb,
                0.5 * (apb - amb @ iapb @ amb), amb @ iapb)

    def _star(SA, SB):
        A11, A12, A21, A22 = SA
        B11, B12, B21, B22 = SB
        Im = jnp.eye(A11.shape[0], dtype=C)
        Dm = jnp.linalg.inv(Im - B11 @ A22)
        Fm = jnp.linalg.inv(Im - A22 @ B11)
        return (A11 + A12 @ Dm @ B11 @ A21, A12 @ Dm @ B12,
                B21 @ Fm @ A21, B22 + B21 @ Fm @ A22 @ B12)

    Z2 = jnp.zeros((2 * N, 2 * N), dtype=C)
    Sp = (Z2, X, X, Z2)
    S = _star(_star(_iface(Wref, Vref, W, Vl), Sp), _iface(W, Vl, Wtrn, Vtrn))
    S11, S12, S21, S22 = S

    delta = jnp.zeros(N, dtype=C).at[M].set(1.0)
    kz_inc = jnp.real(_sqrt_fwd(eps_sup - kx0 ** 2))
    if polarization == "te":
        cinc = jnp.concatenate([jnp.zeros(N, dtype=C), delta])
        einc_sq = 1.0
    else:
        cinc = jnp.concatenate([delta, jnp.zeros(N, dtype=C)])
        einc_sq = 1.0 + (kx0 / kz_inc) ** 2
    r = S11 @ cinc
    t = S21 @ cinc
    rx, ry = r[:N], r[N:]
    tx, ty = t[:N], t[N:]
    safe_r = jnp.where(jnp.abs(kz_ref) < 1e-12, 1.0, kz_ref)
    safe_t = jnp.where(jnp.abs(kz_trn) < 1e-12, 1.0, kz_trn)
    rz = -(kx * rx) / safe_r
    tz = -(kx * tx) / safe_t
    R = jnp.real(kz_ref / kz_inc) * (jnp.abs(rx) ** 2 + jnp.abs(ry) ** 2
                                     + jnp.abs(rz) ** 2) / einc_sq
    T = jnp.real(kz_trn / kz_inc) * (jnp.abs(tx) ** 2 + jnp.abs(ty) ** 2
                                     + jnp.abs(tz) ** 2) / einc_sq
    R = jnp.where(jnp.real(kz_ref) > 0, jnp.real(R), 0.0)
    T = jnp.where(jnp.real(kz_trn) > 0, jnp.real(T), 0.0)
    return orders, R, T


# ===========================================================================
# Unified multi-layer API: RCWAStack / RCWAResult (+ caching, Jones bridge)
# ===========================================================================

# Cache of homogeneous half-space eigenmodes -- the most-repeated solve cost:
# the same superstrate / substrate / spacer recurs across a stack and across
# a wavelength or angle sweep.  Guarded by a Lock for thread safety; cleared
# via the library cache registry.
_HOMOG_CACHE: dict = {}
_HOMOG_LOCK = threading.Lock()


def _clear_rcwa_caches() -> None:
    """Clear the RCWA homogeneous-eigenmode cache (registered with the
    library cache registry)."""
    with _HOMOG_LOCK:
        _HOMOG_CACHE.clear()


def _cached_homogeneous_eigenmodes(eps, Kx, Ky, key):
    with _HOMOG_LOCK:
        hit = _HOMOG_CACHE.get(key)
    if hit is not None:
        return hit
    res = _homogeneous_eigenmodes(Kx, Ky, eps)
    with _HOMOG_LOCK:
        _HOMOG_CACHE[key] = res
    return res


class RCWAResult:
    """Result of an :class:`RCWAStack` solve.

    Accessors
    ---------
    efficiencies() -> (orders, R, T)
        ``R``, ``T`` are ``(2, N)`` real arrays: row 0 is the response to an
        incident ``E_x`` plane wave, row 1 to incident ``E_y``; ``orders`` is
        ``(N, 2)`` (2-D) or ``(N,)`` (1-D).
    absorptance() -> (2,) ndarray
        ``1 - sum(R) - sum(T)`` per incident polarization (``>= 0`` for
        passive media -- the loss-sign bridge guarantees the sign).
    jones_reflection() / jones_transmission() -> (2, 2) complex
        Zeroth-order Jones matrices (columns = incident ``E_x`` / ``E_y``,
        rows = ``[E_x; E_y]``).
    apply_reflection(jones_field) -> JonesField
        Apply the zeroth-order Jones reflection to a
        :class:`~lumenairy.elements.polarization.JonesField` -- the bridge
        from a rigorous metasurface reflection into the polarization
        pipeline.
    """

    def __init__(self, orders, R, T, jones_reflection, jones_transmission):
        self.orders = orders
        self._R = R
        self._T = T
        self._Jr = jones_reflection
        self._Jt = jones_transmission

    def efficiencies(self):
        return self.orders, self._R, self._T

    def absorptance(self):
        return 1.0 - self._R.sum(axis=1) - self._T.sum(axis=1)

    def jones_reflection(self):
        return self._Jr

    def jones_transmission(self):
        return self._Jt

    def apply_reflection(self, jones_field):
        from .polarization import apply_jones_matrix
        return apply_jones_matrix(jones_field, self._Jr)


class _RCWALayer:
    __slots__ = ("thickness", "kind", "data")

    def __init__(self, thickness, kind, data):
        self.thickness = float(thickness)
        self.kind = kind          # 'uniform' | 'iso' | 'tensor'
        self.data = data


class RCWAStack:
    """Builder + solver for a MULTI-LAYER RCWA stack (1-D or 2-D periodic).

    Compose a stack of uniform spacers and patterned layers (isotropic or
    full-tensor / liquid-crystal) between a superstrate and substrate, set
    the incident plane wave, and solve once for the diffraction efficiencies
    of both incident polarizations plus the zeroth-order Jones reflection.

    Example
    -------
    >>> stack = RCWAStack(period=1.0e-6, n_superstrate=1.0, n_substrate=1.5,
    ...                   n_orders=11)
    >>> stack.add_layer(0.1e-6, eps=2.1 ** 2)          # uniform spacer
    >>> stack.add_layer(0.2e-6, eps_cell=cell)         # 2-D patterned layer
    >>> res = stack.set_source(0.633e-6, theta=0.2).solve()
    >>> orders, R, T = res.efficiencies()

    Parameters
    ----------
    period : float
        Period along x (metres).
    period_y : float, optional
        Period along y for a 2-D (crossed) stack.  If omitted (and
        ``n_orders_y`` is omitted) the stack is 1-D (mono-periodic).
    n_superstrate, n_substrate : complex
        Incidence / transmission half-space indices.
    n_orders : int
        Retained orders per side along x.
    n_orders_y : int, optional
        Retained orders per side along y (2-D only; default = ``n_orders``).
    """

    def __init__(self, period, *, period_y=None, n_superstrate=1.0,
                 n_substrate=1.0, n_orders=11, n_orders_y=None):
        self.period_x = float(period)
        self.is_1d = period_y is None and n_orders_y is None
        self.period_y = float(period if period_y is None else period_y)
        self.n_superstrate = n_superstrate
        self.n_substrate = n_substrate
        self.nox = int(n_orders)
        self.noy = 0 if self.is_1d else int(
            n_orders if n_orders_y is None else n_orders_y)
        self._layers: List[_RCWALayer] = []
        self._source: Optional[dict] = None

    def add_layer(self, thickness, *, eps=None, eps_cell=None,
                  eps_tensor_cell=None, shapes=None, eps_background=None):
        """Append a layer.  Provide exactly one layer specification:

        * ``eps`` (scalar) -- uniform spacer;
        * ``eps_cell`` (``(Sx, Sy)``) -- isotropic patterned, FFT-sampled;
        * ``eps_tensor_cell`` (``(Sx, Sy, 3, 3)``) -- anisotropic patterned;
        * ``shapes`` (with ``eps_background``) -- isotropic patterned using
          ANALYTIC shape Fourier transforms + the dual-Laurent factorization
          (exact, no pixelation; see :func:`rcwa_efficiency_2d_shapes`).

        Permittivities are in the PUBLIC convention (``Im(eps) > 0`` lossy).
        """
        n = sum(x is not None for x in (eps, eps_cell, eps_tensor_cell, shapes))
        if n != 1:
            raise ValueError(
                "add_layer: provide exactly one of eps, eps_cell, "
                "eps_tensor_cell, shapes.")
        if eps is not None:
            self._layers.append(_RCWALayer(thickness, "uniform", _C(eps)))
        elif eps_cell is not None:
            cell = np.asarray(eps_cell, dtype=_C)
            if cell.ndim == 1:
                cell = cell[:, None]
            self._layers.append(_RCWALayer(thickness, "iso", cell))
        elif shapes is not None:
            if eps_background is None:
                raise ValueError(
                    "add_layer: shapes requires eps_background.")
            self._layers.append(
                _RCWALayer(thickness, "shapes", (_C(eps_background), shapes)))
        else:
            self._layers.append(
                _RCWALayer(thickness, "tensor",
                           np.asarray(eps_tensor_cell, dtype=_C)))
        return self

    def set_source(self, wavelength, *, theta=0.0, phi=0.0, polarization="te"):
        """Set the incident plane wave (vacuum ``wavelength`` [m], polar
        ``theta`` and azimuth ``phi`` [rad])."""
        self._source = dict(wavelength=float(wavelength), theta=float(theta),
                            phi=float(phi), polarization=polarization)
        return self

    def _layer_modes(self, layer, Kx, Ky, orders):
        if layer.kind == "uniform":
            W, V, kz = _homogeneous_eigenmodes(Kx, Ky, np.conj(layer.data))
            lam = _sqrt_decay(-np.concatenate([kz, kz]) ** 2)
            return W, V, lam
        if layer.kind == "iso":
            EPS = _eps_convolution_2d(np.conj(layer.data), orders,
                                      self.nox, self.noy)
            return _layer_eigenmodes(Kx, Ky, EPS, EPS)
        if layer.kind == "shapes":
            eps_bg, shapes = layer.data
            shapes_c = [dict(s, eps=np.conj(_C(s["eps"]))) for s in shapes]
            EPS, EPS_inv = _analytic_convolutions_2d(
                np.conj(eps_bg), shapes_c, orders, self.nox, self.noy,
                self.period_x, self.period_y)
            return _layer_eigenmodes(Kx, Ky, EPS, EPS, ez_laurent_inv=EPS_inv)
        et = np.conj(layer.data)

        def cv(comp):
            return _eps_convolution_2d(comp, orders, self.nox, self.noy)
        return _layer_eigenmodes_tensor(
            Kx, Ky, cv(et[:, :, 0, 0]), cv(et[:, :, 0, 1]),
            cv(et[:, :, 1, 0]), cv(et[:, :, 1, 1]), cv(et[:, :, 2, 2]))

    def solve(self) -> RCWAResult:
        """Solve the stack -> :class:`RCWAResult`."""
        if self._source is None:
            raise ValueError("RCWAStack.solve: call set_source first.")
        if not self._layers:
            raise ValueError("RCWAStack.solve: add at least one layer.")
        src = self._source
        wl, theta, phi = src["wavelength"], src["theta"], src["phi"]
        k0 = 2.0 * np.pi / wl
        orders, N = _harmonic_orders_2d(self.nox, self.noy)
        eps_sup = np.conj(_C(self.n_superstrate) ** 2)
        eps_sub = np.conj(_C(self.n_substrate) ** 2)
        nre = np.real(np.sqrt(eps_sup))
        kx0 = nre * np.sin(theta) * np.cos(phi)
        ky0 = nre * np.sin(theta) * np.sin(phi)
        kx = kx0 + orders[:, 0] * (wl / self.period_x)
        ky = ky0 + orders[:, 1] * (wl / self.period_y)
        Kx = np.diag(kx.astype(_C))
        Ky = np.diag(ky.astype(_C))
        geom = (self.nox, self.noy, wl, theta, phi, self.period_x,
                self.period_y)
        Wref, Vref, kz_ref = _cached_homogeneous_eigenmodes(
            eps_sup, Kx, Ky, ("sup", self.n_superstrate) + geom)
        Wtrn, Vtrn, kz_trn = _cached_homogeneous_eigenmodes(
            eps_sub, Kx, Ky, ("sub", self.n_substrate) + geom)

        modes = [self._layer_modes(L, Kx, Ky, orders) for L in self._layers]
        W0, V0, lam0 = modes[0]
        S = _interface_smatrix(Wref, Vref, W0, V0)
        S = _redheffer_star(S, _propagation_smatrix(lam0, k0 * self._layers[0].thickness))
        for i in range(1, len(modes)):
            Wp, Vp, _lp = modes[i - 1]
            Wc, Vc, lamc = modes[i]
            S = _redheffer_star(S, _interface_smatrix(Wp, Vp, Wc, Vc))
            S = _redheffer_star(S, _propagation_smatrix(lamc, k0 * self._layers[i].thickness))
        Wl, Vl, _ll = modes[-1]
        S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
        S11, S12, S21, S22 = S

        p0 = int(np.where((orders[:, 0] == 0) & (orders[:, 1] == 0))[0][0])
        delta = np.zeros(N, dtype=_C)
        delta[p0] = 1.0
        kz_inc = np.real(_sqrt_forward(eps_sup - kx0 ** 2 - ky0 ** 2))
        safe_r = np.where(np.abs(kz_ref) < 1e-12, 1.0, kz_ref)
        safe_t = np.where(np.abs(kz_trn) < 1e-12, 1.0, kz_trn)
        R = np.zeros((2, N))
        T = np.zeros((2, N))
        Jr = np.zeros((2, 2), dtype=_C)
        Jt = np.zeros((2, 2), dtype=_C)
        for col, (ex0, ey0) in enumerate(((1.0, 0.0), (0.0, 1.0))):
            long_inc = kx0 * ex0 + ky0 * ey0
            einc_sq = 1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0 else 1.0
            cinc = np.concatenate([ex0 * delta, ey0 * delta])
            r = S11 @ cinc
            t = S21 @ cinc
            rx, ry = r[:N], r[N:]
            tx, ty = t[:N], t[N:]
            rz = -(kx * rx + ky * ry) / safe_r
            tz = -(kx * tx + ky * ty) / safe_t
            Re = np.real(kz_ref / kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2
                                             + np.abs(rz) ** 2) / einc_sq
            Te = np.real(kz_trn / kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                             + np.abs(tz) ** 2) / einc_sq
            R[col] = np.where(np.real(kz_ref) > 0, np.real(Re), 0.0)
            T[col] = np.where(np.real(kz_trn) > 0, np.real(Te), 0.0)
            Jr[0, col], Jr[1, col] = np.conj(rx[p0]), np.conj(ry[p0])
            Jt[0, col], Jt[1, col] = np.conj(tx[p0]), np.conj(ty[p0])
        out_orders = orders[:, 0].copy() if self.is_1d else orders
        return RCWAResult(out_orders, R, T, Jr, Jt)


# Register the RCWA caches with the library cache registry (so the global
# "clear all caches" path empties them too).  Canonical v4.16.0 enrollment
# pattern (mirrors propagators/propagation.py).
try:
    import sys as _sys

    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    _this_mod = _sys.modules[__name__]
    _register_cache_clearer(
        "rcwa_homogeneous_modes",
        lambda: getattr(_this_mod, "_clear_rcwa_caches")(),
    )
except ImportError:  # pragma: no cover - registry always present in-tree
    pass
