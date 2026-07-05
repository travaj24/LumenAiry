"""Berreman 4x4 anisotropic planar multilayer solver.

The fast, EXACT tool for laterally-UNIFORM stacks whose layers may be fully
anisotropic (birefringent / uniaxial / biaxial / gyrotropic) -- liquid-crystal
retarders, waveplates, multilayer dichroics, magneto-optic films.  This is the
anisotropic generalization of the scalar transfer-matrix coating model
(:func:`~lumenairy.elements.coatings.coating_reflectance`): a scalar isotropic
stack reduces to it EXACTLY, while a tensor stack carries the full ``2x2`` Jones
response that the scalar s/p model cannot.

It is NOT a competitor to RCWA / PMM -- those solve laterally PATTERNED
(grating / metasurface) structures, where a transverse Fourier / nodal basis and
a per-layer eigenproblem are unavoidable.  Berreman is the planar limit: a
``4x4`` state ``[Ex, Ey, Hx, Hy]`` per layer (Berreman 1972 / Yeh 1979), so the
whole solve is microseconds -- ideal for unpatterned retarder / coating design
and as the effective-medium screen that FEEDS the patterned solvers.

Method.  In a planar layer the tangential field state obeys ``dpsi/dz' =
Delta psi`` (``z' = k0 z``) with the ``4x4`` Berreman matrix ``Delta(eps, Kx,
Ky)``; ``Ez, Hz`` are eliminated from the normalized Maxwell curl equations.
Eigendecomposition gives 2 forward + 2 backward modes; the layers are composed
by the SAME numerically-stable generalized scattering-matrix cascade the PMM /
RCWA out-of-plane paths use (forward referenced to the layer top, backward to
its bottom, so every exponential decays -- no transfer-matrix overflow through
a thick / lossy layer).

Conventions are PUBLIC throughout (``exp(-i w t)``, ``Im(eps) > 0`` for loss):
the permittivity enters raw (no conjugation), the forward / backward mode split
is by the physical decay / energy direction so the flux-based ``R`` / ``T`` are
PHYSICAL (validated against the complex-angle scalar TMM on lossy stacks), and
the returned Jones matches :func:`~lumenairy.rcwa_jones_1d` to machine precision
on a uniform-tensor slab.  (The standalone single-slab Berreman ORACLE in the
test suite conjugates ``eps`` in and the Jones out -- an equivalent pair for the
Jones, but it never validated loss, where only the raw-``eps`` split gives the
physically-correct power split.)
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from .rcwa import (
    _interface_smatrix_general,
    _modes_to_M,
    _propagation_smatrix_general,
    _redheffer_star,
)
from .rcwa._core import (
    _forward_flux_kz,
    _homogeneous_eigenmodes,
    _layer_eigenmodes_tensor,
    _propagation_star_general,
    _sqrt_decay,
    _sqrt_forward,
)

_C = np.complex128

__all__ = ["berreman_jones_1d", "BerremanStack"]


# =========================================================================== #
# core operators
# =========================================================================== #

def _as_eps_tensor(eps):
    """Scalar or ``(3,3)`` permittivity -> ``(3,3)`` complex tensor (PUBLIC
    convention, ``Im > 0`` for loss)."""
    M = np.asarray(eps, dtype=_C)
    if M.ndim == 0:
        return M * np.eye(3, dtype=_C)
    if M.shape != (3, 3):
        raise ValueError(
            "berreman: each eps must be a scalar or a (3, 3) tensor, got "
            f"shape {M.shape}.")
    return M


def _berreman_delta(eps, Kx, Ky):
    """Berreman ``4x4`` ``Delta`` for state ``[Ex, Ey, Hx, Hy]``,
    ``dpsi/dz' = Delta psi`` (``z' = k0 z``), eigenvalues ``q = kz/k0``.  Full
    ``3x3`` eps (INTERNAL convention), ``mu = 1``; ``Ez, Hz`` eliminated.  The
    conical ``Kx*Ky`` terms at ``[2,0]/[2,3]/[3,2]`` are the audit-F1 fix
    (rotation-covariance-pinned)."""
    exx, exy, exz = eps[0, 0], eps[0, 1], eps[0, 2]
    eyx, eyy, eyz = eps[1, 0], eps[1, 1], eps[1, 2]
    ezx, ezy, ezz = eps[2, 0], eps[2, 1], eps[2, 2]
    d = 1.0 / ezz
    D = np.zeros((4, 4), dtype=_C)
    D[0, 0] = -Kx * ezx * d
    D[0, 1] = -Kx * ezy * d
    D[0, 2] = Kx * Ky * d
    D[0, 3] = 1.0 - Kx * Kx * d
    D[1, 0] = -Ky * ezx * d
    D[1, 1] = -Ky * ezy * d
    D[1, 2] = Ky * Ky * d - 1.0
    D[1, 3] = -Ky * Kx * d
    D[2, 0] = eyx - eyz * ezx * d + Kx * Ky
    D[2, 1] = eyy - eyz * ezy * d - Kx * Kx
    D[2, 2] = eyz * Ky * d
    D[2, 3] = -eyz * Kx * d
    D[3, 0] = exz * ezx * d - exx + Ky * Ky
    D[3, 1] = exz * ezy * d - exy - Kx * Ky
    D[3, 2] = -exz * Ky * d
    D[3, 3] = exz * Kx * d
    return D


def _split_fwd_bwd(gam):
    """Indices of the 2 forward (+z) and 2 backward modes.  ``gam = kz/k0`` in
    the ``exp(gam k0 z)`` convention, so forward is ``Re(gam) < 0`` (decays as
    ``z -> +inf``) or, for a propagating mode, ``Im(gam) > 0``."""
    g = np.asarray(gam)
    tol = 1e-9 * max(1.0, float(np.max(np.abs(g))))
    fwd, bwd = [], []
    for i, v in enumerate(g):
        if v.real < -tol:
            is_fwd = True
        elif v.real > tol:
            is_fwd = False
        else:
            is_fwd = v.imag > 0
        (fwd if is_fwd else bwd).append(i)
    if len(fwd) != 2:
        order = np.argsort(g.real)
        fwd, bwd = list(order[:2]), list(order[2:])
    return np.array(fwd), np.array(bwd)


def _layer_modes(eps_tensor, Kx, Ky):
    """Forward / backward modal blocks of one Berreman layer.

    Returns ``(Wf, Vf, lamf, Wb, Vb, lamb)`` with ``W = [Ex; Ey]`` rows
    (``2x2``), ``V = [Hx; Hy]`` rows (``2x2``), and ``lam = -gam`` so the forward
    set has ``Re(lam) >= 0`` (decay ``exp(-lam k0 z)``) -- the convention the
    generalized scattering-matrix cascade expects."""
    D = _berreman_delta(eps_tensor, Kx, Ky)
    gam, Psi = np.linalg.eig(D)
    fwd, bwd = _split_fwd_bwd(gam)
    Wf, Vf = Psi[:2, fwd], Psi[2:, fwd]
    Wb, Vb = Psi[:2, bwd], Psi[2:, bwd]
    lamf, lamb = -gam[fwd], -gam[bwd]
    return Wf, Vf, lamf, Wb, Vb, lamb


def _flux(Etan, Hmodal):
    """Time-averaged z-Poynting flux from tangential ``E`` and the MODAL ``H``
    (the Berreman state's lower block).  The modal H carries an extra ``-i``
    vs the physical field (the same convention the PMM / RCWA internal-field
    machinery uses), so the physical pair is ``H_phys = -i H_modal`` and
    ``Sz = Re(Ex conj(Hy_phys) - Ey conj(Hx_phys))`` (the common 1/2 cancels in
    every R / T ratio)."""
    Ex, Ey = Etan[0], Etan[1]
    Hx, Hy = -1j * Hmodal[0], -1j * Hmodal[1]
    return float(np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx)))


# =========================================================================== #
# the cascade (shared by the functional entry and the class)
# =========================================================================== #

def _solve_core(eps_layers, thicks, eps_sup, eps_sub, wavelength, Kx, Ky,
                retain=False):
    """S-matrix cascade for a planar Berreman stack (INTERNAL convention).

    ``eps_layers`` are already-conjugated ``(3,3)`` tensors; ``eps_sup/eps_sub``
    already-conjugated scalars.  Returns a dict with the half-space forward /
    backward mode blocks, ``S11/S21``, and (when ``retain``) the per-layer modes
    + partial cascades for internal-field / absorption reconstruction."""
    k0 = 2.0 * np.pi / wavelength

    Wf_s, Vf_s, lamf_s, Wb_s, Vb_s, lamb_s = _layer_modes(
        eps_sup * np.eye(3, dtype=_C), Kx, Ky)
    Wf_b, Vf_b, lamf_b, Wb_b, Vb_b, lamb_b = _layer_modes(
        eps_sub * np.eye(3, dtype=_C), Kx, Ky)
    M_sup = _modes_to_M(Wf_s, Vf_s, Wb_s, Vb_s)
    M_sub = _modes_to_M(Wf_b, Vf_b, Wb_b, Vb_b)

    modes = [_layer_modes(e, Kx, Ky) for e in eps_layers]
    Ms = [_modes_to_M(Wf, Vf, Wb, Vb) for (Wf, Vf, _lf, Wb, Vb, _lb) in modes]
    nlay = len(modes)

    # full cascade sup -> [iface, prop]*layers -> sub
    if nlay == 0:
        S = _interface_smatrix_general(M_sup, M_sub)
    else:
        S = _interface_smatrix_general(M_sup, Ms[0])
        for i in range(nlay):
            lamf, lamb = modes[i][2], modes[i][5]
            S = _redheffer_star(
                S, _propagation_smatrix_general(lamf, lamb, k0 * thicks[i]))
            nxt = M_sub if i == nlay - 1 else Ms[i + 1]
            S = _redheffer_star(S, _interface_smatrix_general(Ms[i], nxt))
    S11, _S12, S21, _S22 = S

    out = dict(M_sup=M_sup, M_sub=M_sub, S11=S11, S21=S21,
               Wf_s=Wf_s, Vf_s=Vf_s, Wb_s=Wb_s, Vb_s=Vb_s,
               Wf_b=Wf_b, Vf_b=Vf_b, k0=k0)
    if retain and nlay:
        # Three bracketing partial cascades for the in-structure field (the
        # general RCWAResult._internal_cpm recovery, valid for the asymmetric
        # forward/backward Berreman modes):
        #   S_above[i]     : sup -> TOP of layer i (before its propagation)
        #   S_below[i]     : TOP of layer i -> sub (its OWN propagation first)
        #   S_below_bot[i] : BOTTOM of layer i -> sub (propagation removed)
        # The backward amplitude references the layer BOTTOM, so every modal
        # exponential decays (no overflow through a deep / lossy layer).
        ifc = [_interface_smatrix_general(M_sup, Ms[0])]
        for i in range(1, nlay):
            ifc.append(_interface_smatrix_general(Ms[i - 1], Ms[i]))
        ifc.append(_interface_smatrix_general(Ms[-1], M_sub))
        prop = [_propagation_smatrix_general(modes[i][2], modes[i][5],
                                             k0 * thicks[i])
                for i in range(nlay)]
        S_above = [None] * nlay
        S_above[0] = ifc[0]
        for i in range(1, nlay):
            S_above[i] = _redheffer_star(
                _redheffer_star(S_above[i - 1], prop[i - 1]), ifc[i])
        S_below = [None] * nlay
        S_below_bot = [None] * nlay
        for i in range(nlay - 1, -1, -1):
            S_below_bot[i] = (ifc[nlay] if i == nlay - 1
                              else _redheffer_star(ifc[i + 1], S_below[i + 1]))
            S_below[i] = _redheffer_star(prop[i], S_below_bot[i])
        out.update(modes=modes, thicks=thicks, S_above=S_above,
                   S_below=S_below, S_below_bot=S_below_bot,
                   eps_layers=eps_layers)
    return out


def _farfield(core, eps_sup, eps_sub, Kx, Ky):
    """Reflected / transmitted Jones (INTERNAL convention, ``2x2`` columns =
    incident lab ``[Ex; Ey]``) and per-incident-polarization power ``R, T``."""
    Wf_s, Wb_s, Vf_s, Vb_s = (core["Wf_s"], core["Wb_s"],
                              core["Vf_s"], core["Vb_s"])
    Wf_b, Vf_b = core["Wf_b"], core["Vf_b"]
    S11, S21 = core["S11"], core["S21"]
    Jr = np.zeros((2, 2), dtype=_C)
    Jt = np.zeros((2, 2), dtype=_C)
    R = np.zeros(2)
    T = np.zeros(2)
    for col, Einc in enumerate((np.array([1.0, 0.0], _C),
                                np.array([0.0, 1.0], _C))):
        c_inc = np.linalg.solve(Wf_s, Einc)          # forward sup modal amps
        c_ref = S11 @ c_inc                           # backward sup
        c_trn = S21 @ c_inc                           # forward sub
        E_inc = Wf_s @ c_inc
        H_inc = Vf_s @ c_inc
        E_ref = Wb_s @ c_ref
        H_ref = Vb_s @ c_ref
        E_trn = Wf_b @ c_trn
        H_trn = Vf_b @ c_trn
        Jr[:, col] = E_ref
        Jt[:, col] = E_trn
        F_inc = _flux(E_inc, H_inc)
        R[col] = -_flux(E_ref, H_ref) / F_inc        # backward flux is -z
        T[col] = _flux(E_trn, H_trn) / F_inc
    return Jr, Jt, R, T


# =========================================================================== #
# out-of-plane-tensor + oblique incidence: generalized S-matrix cascade
# =========================================================================== #
#
# The native Berreman 4x4 S-matrix cascade above is EXACT for isotropic,
# in-plane-anisotropic and out-of-plane-tensor-at-NORMAL layers, but its
# forward/backward mode convention is subtly wrong for an OUT-OF-PLANE tensor
# (``eps_xz/eps_yz != 0``) at OBLIQUE incidence -- the [W; -V] <-> -lam symmetry
# the native pairing implicitly relies on is BROKEN there, so the reflected
# amplitudes come out ~2% off (energy still conserves, so the error is silent).
# For that regime ONLY we route to the SAME generalized (Li 2003) S-matrix that
# ``rcwa_jones_1d`` / ``RCWAStack`` use: a planar stack is a single-Fourier-order
# grating, so the RCWA mode machinery (``_homogeneous_eigenmodes`` for the
# isotropic half-spaces / scalar spacers, ``_layer_eigenmodes_tensor`` fed the
# ezz-Schur-condensed in-plane block + raw off-plane operators for tensor layers)
# reduces to it EXACTLY.  Validated to match ``rcwa_jones_1d`` / ``RCWAStack``
# to machine precision across iso / in-plane / OOP-normal / OOP-oblique /
# OOP-conical / lossy / multilayer.  (The two harness transfer-matrix oracles
# that previously flagged this as a "berreman bug" each carried their own
# transfer-direction error, single-layer and multilayer; the generalized S-matrix
# is the independently-cross-checked truth.)


def _tensor_is_offplane(eps):
    """True if a ``(3, 3)`` permittivity has OUT-OF-PLANE coupling
    (``eps_xz / eps_yz / eps_zx / eps_zy`` above a tiny relative tolerance).
    Scalars / isotropic tensors return False."""
    M = np.asarray(eps, dtype=_C)
    if M.ndim != 2 or M.shape != (3, 3):
        return False
    off = max(abs(M[0, 2]), abs(M[1, 2]), abs(M[2, 0]), abs(M[2, 1]))
    diag = max(abs(M[0, 0]), abs(M[1, 1]), abs(M[2, 2]), 1.0)
    return bool(off > 1e-12 * diag)


def _offplane_oblique(eps_layers, Kx, Ky):
    """Route condition for the generalized S-matrix path: at least one
    out-of-plane tensor layer AND oblique incidence (``Kx`` or ``Ky != 0``).
    Everything else stays on the exact native Berreman cascade."""
    oblique = (abs(float(np.real(Kx))) > 1e-12
               or abs(float(np.real(Ky))) > 1e-12)
    return oblique and any(_tensor_is_offplane(e) for e in eps_layers)


def _offplane_condensed_M(eps_internal, Kx1, Ky1):
    """``(M, lam_f, lam_b)`` for one layer (INTERNAL-convention eps) via the RCWA
    mode functions.  Scalar / isotropic -> analytic homogeneous ``[W; -V]``
    (``lam, -lam``); tensor -> ezz-Schur-condensed in-plane block + raw off-plane
    (Li 2003), as the general 6-tuple (out-of-plane) or the symmetric 3-tuple
    (in-plane).  ``Kx1 / Ky1`` are ``1x1`` (single Fourier order)."""
    e = np.asarray(eps_internal, dtype=_C)
    if e.ndim == 0 or float(np.max(np.abs(e - e[0, 0] * np.eye(3)))) < \
            1e-12 * max(1.0, abs(e[0, 0] if e.ndim else e)):
        es = complex(e if e.ndim == 0 else e[0, 0])
        W, V, kz = _homogeneous_eigenmodes(Kx1, Ky1, es)
        kzc = kz.ravel()
        lam = _sqrt_decay(-np.concatenate([kzc, kzc]) ** 2)
        return _modes_to_M(W, V, W, -V), lam, -lam
    exx, exy, exz = e[0, 0], e[0, 1], e[0, 2]
    eyx, eyy, eyz = e[1, 0], e[1, 1], e[1, 2]
    ezx, ezy, ezz = e[2, 0], e[2, 1], e[2, 2]
    d = 1.0 / ezz
    o = lambda v: np.array([[v]], dtype=_C)     # noqa: E731 (1x1 Fourier op)
    Cxx, Cxy = o(exx - exz * ezx * d), o(exy - exz * ezy * d)
    Cyx, Cyy = o(eyx - eyz * ezx * d), o(eyy - eyz * ezy * d)
    if max(abs(exz), abs(eyz), abs(ezx), abs(ezy)) > 1e-12:
        Wf, Vf, lf, Wb, Vb, lb = _layer_eigenmodes_tensor(
            Kx1, Ky1, Cxx, Cxy, Cyx, Cyy, o(ezz),
            o(ezx), o(ezy), o(exz), o(eyz))
        return _modes_to_M(Wf, Vf, Wb, Vb), lf, lb
    W, V, lam = _layer_eigenmodes_tensor(Kx1, Ky1, Cxx, Cxy, Cyx, Cyy, o(ezz))
    return _modes_to_M(W, V, W, -V), lam, -lam


def _offplane_oblique_solve(eps_layers, thicks, eps_sup, eps_sub, wl, Kx, Ky):
    """``(R, T, Jr, Jt)`` for a stack with out-of-plane tensor layer(s) at oblique
    incidence, via the generalized (Li 2003) single-order S-matrix cascade.
    ``eps_*`` are PUBLIC (``Im > 0``); returns PUBLIC-convention Jones + per-lab-
    polarization power ``R, T``.  Validated to ``rcwa_jones_1d`` / ``RCWAStack``."""
    kx0, ky0 = float(np.real(Kx)), float(np.real(Ky))
    o = lambda v: np.array([[v]], dtype=_C)     # noqa: E731 (1x1 Fourier op)
    Kx1, Ky1 = o(kx0), o(ky0)
    k0 = 2.0 * np.pi / wl
    Wr, Vr, _kzr = _homogeneous_eigenmodes(Kx1, Ky1, np.conj(_C(eps_sup)))
    Wt, Vt, _kzt = _homogeneous_eigenmodes(Kx1, Ky1, np.conj(_C(eps_sub)))
    M_prev = _modes_to_M(Wr, Vr, Wr, -Vr)
    S = None
    for eps, thk in zip(eps_layers, thicks):
        Ml, lf, lb = _offplane_condensed_M(
            np.conj(np.asarray(eps, dtype=_C)), Kx1, Ky1)
        Si = _interface_smatrix_general(M_prev, Ml)
        S = Si if S is None else _redheffer_star(S, Si)
        S = _propagation_star_general(S, lf, lb, k0 * float(thk))
        M_prev = Ml
    S = _redheffer_star(
        S, _interface_smatrix_general(M_prev, _modes_to_M(Wt, Vt, Wt, -Vt)))
    S11, _S12, S21, _S22 = S
    kz_inc = float(np.real(_sqrt_forward(_C(eps_sup) - kx0 ** 2 - ky0 ** 2)))
    kzrf = _forward_flux_kz(_C(eps_sup), np.array([kx0]), np.array([ky0]))[0]
    kztf = _forward_flux_kz(_C(eps_sub), np.array([kx0]), np.array([ky0]))[0]
    Jr = np.zeros((2, 2), dtype=_C)
    Jt = np.zeros((2, 2), dtype=_C)
    R = np.zeros(2)
    T = np.zeros(2)
    for col, (ex0, ey0) in enumerate(((1.0, 0.0), (0.0, 1.0))):
        r = S11 @ np.array([ex0, ey0], dtype=_C)
        t = S21 @ np.array([ex0, ey0], dtype=_C)
        rx, ry, tx, ty = r[0], r[1], t[0], t[1]
        longi = kx0 * ex0 + ky0 * ey0
        einc_sq = 1.0 + (longi / kz_inc) ** 2 if kz_inc != 0 else 1.0
        # longitudinal (z) field from div(D)=0: BOTH kx and ky contribute at a
        # CONICAL mount (dropping ky under-counts the y-pol flux by ~1%).
        rz = -(kx0 * rx + ky0 * ry) / (kzrf if abs(kzrf) > 1e-12 else 1.0)
        tz = -(kx0 * tx + ky0 * ty) / (kztf if abs(kztf) > 1e-12 else 1.0)
        R[col] = (float(np.real(kzrf / kz_inc))
                  * (abs(rx) ** 2 + abs(ry) ** 2 + abs(rz) ** 2) / einc_sq
                  if np.real(kzrf) > 0 else 0.0)
        T[col] = (float(np.real(kztf / kz_inc))
                  * (abs(tx) ** 2 + abs(ty) ** 2 + abs(tz) ** 2) / einc_sq
                  if np.real(kztf) > 0 else 0.0)
        Jr[:, col] = [np.conj(rx), np.conj(ry)]
        Jt[:, col] = [np.conj(tx), np.conj(ty)]
    return R, T, Jr, Jt


# =========================================================================== #
# functional entry
# =========================================================================== #

def berreman_jones_1d(
    layers: List[Tuple[Union[complex, np.ndarray], float]],
    n_substrate: complex,
    n_superstrate: complex,
    wavelength: float,
    *,
    angle: float = 0.0,
    phi: float = 0.0,
    theta: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full ``2x2`` Jones reflection / transmission of an ANISOTROPIC PLANAR
    multilayer by the Berreman ``4x4`` method.

    Parameters
    ----------
    layers : list of (eps, thickness)
        Ordered from the SUPERSTRATE side inward (first = outermost).  Each
        ``eps`` is a scalar (isotropic) or a ``(3, 3)`` tensor (PUBLIC
        ``Im > 0`` for loss); ``thickness`` in metres.
    n_substrate, n_superstrate : complex
        Half-space refractive indices (isotropic).
    wavelength : float
        Vacuum wavelength [m].
    angle / theta : float, optional
        Polar incidence angle [rad] (``theta`` is the cross-suite alias and
        wins when both are given).
    phi : float, optional
        Azimuth [rad] (conical mount; ``0`` = classical x-z plane).

    Returns
    -------
    R, T : (2,) float ndarray
        Power reflectance / transmittance for incident lab ``E_x`` (index 0)
        and ``E_y`` (index 1), flux-normalized (``R + T = 1`` for a lossless
        stack).
    jones_r, jones_t : (2, 2) complex ndarray
        Reflection / transmission Jones; columns = incident lab ``[Ex; Ey]``,
        rows = reflected / transmitted lab ``[Ex; Ey]``.  PUBLIC ``exp(-i w t)``.

    Accuracy
    --------
    Exact (machine precision) for isotropic, in-plane-anisotropic and
    out-of-plane-tensor layers at any incidence, including OBLIQUE and CONICAL.
    Two internal paths: the native Berreman ``4x4`` S-matrix cascade for the
    symmetric cases (isotropic / in-plane / out-of-plane-at-NORMAL), and -- for
    an OUT-OF-PLANE tensor (``eps_xz/eps_yz != 0``) at OBLIQUE incidence, where
    that cascade's forward/backward mode convention would be ~a few percent off
    -- the generalized (Li 2003) single-Fourier-order S-matrix that
    :func:`~lumenairy.elements.rcwa.rcwa_jones_1d` / ``RCWAStack`` use, to which
    it is validated to machine precision (all regimes + lossy + multilayer).
    The route is automatic; both paths agree on every case they share.
    """
    # Differentiable (JAX) dispatch: any traced input routes to the jnp twin.
    from ..backend import is_jax_array
    if (is_jax_array(n_substrate) or is_jax_array(n_superstrate)
            or is_jax_array(wavelength) or is_jax_array(angle)
            or is_jax_array(phi) or is_jax_array(theta)
            or any(is_jax_array(e) or is_jax_array(t) for e, t in layers)):
        # The generalized-S-matrix out-of-plane-oblique fix is NumPy-only; the
        # jnp twin still uses the native cascade (the ~2%-off convention).  Guard
        # the concretely-detectable out-of-plane-tensor case at non-normal (or
        # traced) incidence so it fails loudly rather than returning a silently
        # inaccurate differentiable result.  (Normal incidence is exact on the
        # native path, so a provably-normal mount is allowed through.)
        _off = any(not is_jax_array(e) and _tensor_is_offplane(_as_eps_tensor(e))
                   for e, _t in layers)
        _ang = (float(theta) if (theta is not None and not is_jax_array(theta))
                else float(angle) if not is_jax_array(angle) else None)
        if _off and (_ang is None or abs(_ang) > 1e-12):
            raise NotImplementedError(
                "berreman_jones_1d: differentiable (JAX) evaluation of an "
                "out-of-plane tensor (eps_xz/eps_yz != 0) at OBLIQUE incidence "
                "is not supported -- the jnp twin uses the native 4x4 cascade, "
                "which is ~2% off in that regime (the generalized-S-matrix fix "
                "is NumPy-only).  Use NumPy inputs, restrict to normal "
                "incidence, or differentiate rcwa.rcwa_jones_1d / RCWAStack.")
        from ._berreman_jax import _berreman_jones_1d_jax
        return _berreman_jones_1d_jax(layers, n_substrate, n_superstrate,
                                      wavelength, angle=angle, phi=phi,
                                      theta=theta)
    if theta is not None:
        angle = float(theta)
    eps_layers = [_as_eps_tensor(e) for e, _t in layers]
    thicks = [float(t) for _e, t in layers]
    eps_sup = complex(_C(n_superstrate) ** 2)
    eps_sub = complex(_C(n_substrate) ** 2)
    nre = float(np.real(_C(n_superstrate)))
    Kx = nre * np.sin(angle) * np.cos(phi)
    Ky = nre * np.sin(angle) * np.sin(phi)
    if _offplane_oblique(eps_layers, Kx, Ky):
        # out-of-plane tensor at oblique incidence: the generalized S-matrix
        # cascade (matches rcwa_jones_1d / RCWAStack to machine precision).
        return _offplane_oblique_solve(eps_layers, thicks, eps_sup, eps_sub,
                                       wavelength, Kx, Ky)
    core = _solve_core(eps_layers, thicks, eps_sup, eps_sub, wavelength,
                       Kx, Ky)
    Jr, Jt, R, T = _farfield(core, eps_sup, eps_sub, Kx, Ky)
    return R, T, Jr, Jt


# =========================================================================== #
# builder class (mirrors RCWAStack / PMMStack)
# =========================================================================== #

class BerremanStack:
    """Builder for an anisotropic planar multilayer solved by the Berreman
    ``4x4`` method -- the planar / unpatterned member of the solver family
    (cf. :class:`~lumenairy.elements.rcwa.RCWAStack`,
    :class:`~lumenairy.elements.pmm.PMMStack`).

    >>> st = BerremanStack(n_substrate=1.5, n_superstrate=1.0)
    >>> st.add_layer(140e-9, eps=lc_tensor)       # a planar LC retarder
    >>> st.add_layer(100e-9, eps=2.1)             # an isotropic cap
    >>> R, T, jones = st.set_source(0.633e-6, theta=0.2).solve()
    """

    def __init__(self, *, n_substrate=1.0, n_superstrate=1.0):
        from ..backend import is_jax_array
        # JAX half-space indices stay RAW so their gradient flows.
        self.n_sub = (n_substrate if is_jax_array(n_substrate)
                      else _C(n_substrate))
        self.n_sup = (n_superstrate if is_jax_array(n_superstrate)
                      else _C(n_superstrate))
        self._layers = []                     # (thickness, eps_tensor_public)
        self._src = None
        self._internal = None

    def add_layer(self, thickness, *, eps):
        """Append a planar layer of thickness ``t`` [m] and permittivity
        ``eps`` (scalar isotropic or ``(3, 3)`` tensor, PUBLIC ``Im > 0``)."""
        from ..backend import is_jax_array
        # Keep traced thickness / eps raw (the differentiable solve path);
        # float()/_as_eps_tensor() would sever the trace.
        if is_jax_array(thickness):
            t_store = thickness
        else:
            t_store = float(thickness)
            if t_store <= 0:
                raise ValueError(
                    "BerremanStack.add_layer: thickness must be > 0.")
        eps_store = eps if is_jax_array(eps) else _as_eps_tensor(eps)
        self._layers.append((t_store, eps_store))
        return self

    def add_effective_grating(self, thickness, *, eps_ridge, eps_groove, fill,
                              period=None, order=0):
        """Append a sub-wavelength 1-D binary grating as its EFFECTIVE-MEDIUM
        (Rytov) uniaxial layer -- the EMT bridge that lets a whole design sweep
        run in microseconds before a rigorous RCWA / PMM cross-check.

        The grating (period along ``x``, ridge ``eps_ridge`` fill ``fill``,
        groove ``eps_groove``) homogenizes to ``diag(eps_perp, eps_par,
        eps_par)`` via :func:`~lumenairy.elements.emt.rytov_tensor`.  ``order=2``
        adds the second-order bulk-index correction and needs ``period`` (the
        wavelength comes from :meth:`set_source`, so call ``set_source`` first
        for ``order=2``).  EXACT only as ``period -> 0`` -- a screen, not a
        replacement for the rigorous solve."""
        from .emt import rytov_tensor
        wl = None
        if order == 2:
            if self._src is None:
                raise ValueError(
                    "BerremanStack.add_effective_grating: order=2 needs the "
                    "wavelength; call set_source(...) before adding the layer.")
            wl = self._src["wl"]
        eps_eff = rytov_tensor(eps_ridge, eps_groove, fill,
                               period=period, wavelength=wl, order=order)
        return self.add_layer(thickness, eps=eps_eff)

    def set_source(self, wavelength, *, angle=0.0, theta=None, phi=0.0):
        """Set the incident plane wave (vacuum ``wavelength`` [m], polar
        ``angle`` / ``theta`` [rad], azimuth ``phi`` [rad])."""
        from ..backend import is_jax_array
        if theta is not None:
            angle = theta
        self._src = dict(
            wl=wavelength if is_jax_array(wavelength) else float(wavelength),
            angle=angle if is_jax_array(angle) else float(angle),
            phi=phi if is_jax_array(phi) else float(phi))
        return self

    def _holds_traced(self):
        from ..backend import is_jax_array
        return (is_jax_array(self.n_sub) or is_jax_array(self.n_sup)
                or (self._src is not None
                    and (is_jax_array(self._src["wl"])
                         or is_jax_array(self._src["angle"])
                         or is_jax_array(self._src["phi"])))
                or any(is_jax_array(t) or is_jax_array(e)
                       for t, e in self._layers))

    def _geom(self):
        wl, ang, phi = self._src["wl"], self._src["angle"], self._src["phi"]
        eps_sup = complex(self.n_sup ** 2)
        eps_sub = complex(self.n_sub ** 2)
        nre = float(np.real(self.n_sup))
        Kx = nre * np.sin(ang) * np.cos(phi)
        Ky = nre * np.sin(ang) * np.sin(phi)
        eps_layers = [e for _t, e in self._layers]
        thicks = [t for t, _e in self._layers]
        return wl, eps_sup, eps_sub, Kx, Ky, eps_layers, thicks

    def solve(self, *, retain_internal=False):
        """Solve the stack.  Returns ``(R, T, jones)`` -- ``R``/``T`` are
        ``(2,)`` power per incident lab polarization (``E_x``, ``E_y``);
        ``jones`` is the ``2x2`` reflection Jones (PUBLIC convention)."""
        if self._src is None:
            raise ValueError("BerremanStack.solve: call set_source(...) first.")
        if not self._layers:
            raise ValueError("BerremanStack.solve: add at least one layer.")
        if self._holds_traced():
            if retain_internal:
                raise NotImplementedError(
                    "BerremanStack.solve(retain_internal=True): the internal-"
                    "field / absorption observables are NumPy-only; use "
                    "concrete inputs for fields.")
            from ._berreman_jax import _berreman_stack_solve_jax
            return _berreman_stack_solve_jax(self)
        wl, eps_sup, eps_sub, Kx, Ky, eps_layers, thicks = self._geom()
        eps_layers = [_as_eps_tensor(e) for e in eps_layers]
        if _offplane_oblique(eps_layers, Kx, Ky):
            # out-of-plane tensor at oblique incidence -> generalized S-matrix.
            if retain_internal:
                raise NotImplementedError(
                    "BerremanStack.solve(retain_internal=True): the internal-"
                    "field / absorption observables are not available for "
                    "out-of-plane tensor layers at oblique incidence (that "
                    "regime uses the generalized S-matrix cascade, which has no "
                    "per-layer Berreman-mode reconstruction).  Use "
                    "rcwa.RCWAStack for internal fields of such stacks, or "
                    "normal incidence.")
            R, T, Jr, _Jt = _offplane_oblique_solve(
                eps_layers, thicks, eps_sup, eps_sub, wl, Kx, Ky)
            return R, T, Jr
        core = _solve_core(eps_layers, thicks, eps_sup, eps_sub, wl, Kx, Ky,
                           retain=retain_internal)
        Jr, Jt, R, T = _farfield(core, eps_sup, eps_sub, Kx, Ky)
        if retain_internal:
            # incident modal amplitudes per lab polarization, for the field /
            # absorption reconstruction.
            cinc = np.zeros((2, 2), dtype=_C)
            for col, Einc in enumerate((np.array([1.0, 0.0], _C),
                                        np.array([0.0, 1.0], _C))):
                cinc[:, col] = np.linalg.solve(core["Wf_s"], Einc)
            core["cinc"] = cinc
            core["Kx"], core["Ky"] = Kx, Ky
            core["R"], core["T"] = R, T
            self._internal = core
        return R, T, Jr

    # -- internal observables ------------------------------------------- #

    def _amplitudes(self):
        """Per-layer ``(c_fwd_top, c_bwd_bot)`` modal amplitudes (each
        ``2x2``: cols = incident lab pol) from the retained partial cascades --
        the general (asymmetric-mode) RCWAResult._internal_cpm recovery:
        ``c+ = inv(I - Sa22 Sb11) Sa21 cinc`` at the layer TOP, then the
        backward amplitude at the BOTTOM from the forward field propagated
        down (a DECAYING ``Xf``)."""
        d = self._internal
        cinc = d["cinc"]
        out = []
        k0 = d["k0"]
        for i, (Wf, Vf, lamf, Wb, Vb, lamb) in enumerate(d["modes"]):
            Sa = d["S_above"][i]
            Sb11 = d["S_below"][i][0]
            Bb11 = d["S_below_bot"][i][0]
            A22, A21 = Sa[3], Sa[2]
            denom = np.linalg.inv(np.eye(2, dtype=_C) - A22 @ Sb11)
            c_fwd = denom @ (A21 @ cinc)              # forward at layer TOP
            Xf = np.exp(-lamf * k0 * d["thicks"][i])
            c_bwd = Bb11 @ (Xf[:, None] * c_fwd)      # backward at layer BOTTOM
            out.append((c_fwd, c_bwd))
        return out

    def internal_field(self, z, *, component="all", incident=(1.0, 0.0)):
        """Reconstruct the E and/or H field INSIDE the stack -- the planar
        counterpart of :meth:`RCWAResult.internal_field` /
        :meth:`PMMStack.internal_field`.

        Requires ``solve(retain_internal=True)``.

        Parameters
        ----------
        z : float or array-like
            Depth(s) [m] from the stack TOP (the superstrate interface).
        component : {'E', 'H', 'all'}, optional
            Which field(s) to return (default all six components).
        incident : (complex, complex), optional
            PUBLIC incident Jones ``(E_x, E_y)`` (default x-polarized).

        Returns
        -------
        dict
            ``{'Ex','Ey','Ez','Hx','Hy','Hz'}`` (per ``component``) each a
            ``(nz,)`` complex array, plus ``'z'`` and ``'layer'``.  PUBLIC
            ``exp(-i w t)`` convention.
        """
        d = self._internal
        if d is None or "cinc" not in d:
            raise ValueError(
                "BerremanStack.internal_field: call solve(retain_internal="
                "True) first.")
        names = {"E": ("Ex", "Ey", "Ez"), "H": ("Hx", "Hy", "Hz"),
                 "all": ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        if component not in names:
            raise ValueError(
                f"BerremanStack.internal_field: component must be 'E', 'H' or "
                f"'all', got {component!r}.")
        want = names[component]
        wx, wy = complex(incident[0]), complex(incident[1])
        amps = self._amplitudes()
        k0, Kx, Ky = d["k0"], d["Kx"], d["Ky"]
        thicks = d["thicks"]
        z_top = np.concatenate([[0.0], np.cumsum(thicks)])
        zs = np.atleast_1d(np.asarray(z, dtype=float))
        scalar_in = np.ndim(z) == 0
        out = {c: np.empty(zs.shape[0], dtype=_C) for c in want}
        layers_out = np.empty(zs.shape[0], dtype=int)
        cache = {}
        for zi, zz in enumerate(zs):
            i = int(np.clip(np.searchsorted(z_top, zz, side="right") - 1,
                            0, len(thicks) - 1))
            zloc = zz - z_top[i]
            layers_out[zi] = i
            Wf, Vf, lamf, Wb, Vb, lamb = d["modes"][i]
            if i not in cache:
                c_fwd, c_bwd = amps[i]
                cache[i] = (wx * c_fwd[:, 0] + wy * c_fwd[:, 1],
                            wx * c_bwd[:, 0] + wy * c_bwd[:, 1])
            cF, cB = cache[i]
            t_i = thicks[i]
            P = np.exp(-lamf * k0 * zloc)          # forward from layer TOP
            Q = np.exp(lamb * k0 * (t_i - zloc))   # backward from layer BOTTOM
            E = Wf @ (P * cF) + Wb @ (Q * cB)
            H = Vf @ (P * cF) + Vb @ (Q * cB)
            Ex, Ey = E[0], E[1]
            Hx, Hy = -1j * H[0], -1j * H[1]      # modal H -> -i eta0 scale
            eps_t = d["eps_layers"][i]            # raw (public) tensor
            # Ez from Dz = 0: eps_zx Ex + eps_zy Ey + eps_zz Ez = -(Kx Hy - Ky Hx)
            ezz = eps_t[2, 2]
            Ez = (-(Kx * Hy - Ky * Hx)
                  - eps_t[2, 0] * Ex - eps_t[2, 1] * Ey) / ezz
            Hz = Kx * Ey - Ky * Ex                # curl-E z-component
            full = dict(Ex=Ex, Ey=Ey, Ez=Ez, Hx=Hx, Hy=Hy, Hz=Hz)
            for c in want:
                out[c][zi] = full[c]
        res = {c: (out[c][0] if scalar_in else out[c]) for c in want}
        res["z"] = float(zs[0]) if scalar_in else zs
        res["layer"] = int(layers_out[0]) if scalar_in else layers_out
        return res

    def layer_absorption(self):
        """Per-layer absorbed power fraction ``(n_layers, 2)`` (cols = incident
        lab ``E_x`` / ``E_y``) from the internal z-Poynting flux difference
        across each layer.  Requires ``solve(retain_internal=True)``.  The
        cross-machinery invariant ``sum_i A_i ~= 1 - R - T`` is the honest
        closure check (internal amplitudes vs the half-space far field)."""
        d = self._internal
        if d is None or "cinc" not in d:
            raise ValueError(
                "BerremanStack.layer_absorption: call solve(retain_internal="
                "True) first.")
        amps = self._amplitudes()
        k0 = d["k0"]
        nlay = len(d["modes"])

        def flux_at(i, zfrac):
            Wf, Vf, lamf, Wb, Vb, lamb = d["modes"][i]
            c_fwd, c_bwd = amps[i]
            t = d["thicks"][i]
            P = np.exp(-lamf * k0 * (zfrac * t))[:, None]
            Q = np.exp(lamb * k0 * ((1.0 - zfrac) * t))[:, None]
            E = Wf @ (P * c_fwd) + Wb @ (Q * c_bwd)
            H = Vf @ (P * c_fwd) + Vb @ (Q * c_bwd)
            return np.array([_flux(E[:, c], H[:, c]) for c in range(2)])

        F_top = np.array([flux_at(i, 0.0) for i in range(nlay)])
        F_bot = np.array([flux_at(i, 1.0) for i in range(nlay)])
        # F_inc per pol: the top-of-stack internal forward flux equals the
        # incident flux (the cascade is referenced to unit incident modal amp).
        F_inc = np.array([
            self._incident_flux(c) for c in range(2)])
        return (F_top - F_bot) / F_inc[None, :]

    def _incident_flux(self, col):
        d = self._internal
        cinc = d["cinc"][:, col]
        return _flux(d["Wf_s"] @ cinc, d["Vf_s"] @ cinc)
