"""Scalar mode-matching diffraction from 2-D layer modes -- and the documented
finding that the EME's real-space modes are the WRONG basis for it.

A doubly-periodic layer ``eps(x, y)`` of thickness ``depth`` sits between a
uniform superstrate and substrate.  The scattering is set up by MODE MATCHING:

* the uniform half-spaces carry Rayleigh plane waves ``u_mp(x,y)`` (orders
  ``(m, p)``) with ``kz`` set by the dispersion;
* the layer carries 2-D modes ``psi_n(x,y)`` propagating as ``exp(+-i qz_n z)``
  -- the eigenpairs of the transverse Helmholtz operator, supplied either by the
  EME (``eme_2d.layer_modes`` + ``mode_field``) or by the 2-D-FD oracle;
* continuity of ``psi`` and ``d psi/dz`` at ``z = 0`` and ``z = depth``, projected
  onto the plane-wave orders, closes a square (``K = N_pw``) linear system for the
  reflected / transmitted amplitudes (`mode_match`).

WHAT WORKS (validated, regression-tested -- ``test_eme_diffraction.py``):
  The mode-matching MATH is exact.  A UNIFORM layer reproduces the analytic
  scalar slab (Airy / Fabry-Perot) -- to machine precision at normal incidence
  (the (0,0) mode is flat, exact on the grid) and to FD accuracy (~1e-4) at
  oblique incidence (the (0,0) mode is ``exp(i ky0 y)``, O(h^2) eigenvalue error),
  with energy ``R + T = 1`` exactly in both -- because a uniform layer's modes ARE
  the plane waves, so ``K = N_pw`` modes match the ``N_pw`` orders one-to-one.

THE FINDING (a real negative result, NOT a bug):
  For a STRUCTURED layer the same mode matching does NOT converge -- the
  efficiencies wander (e.g. ``T_00`` swinging 0.27 <-> 0.48) and energy sits at
  0.97-1.5 as the order/mode count grows, for EVERY mode selection tried (highest
  ``qz^2``; largest order-overlap; complete coarse grid).  The reason is
  structural, not numerical: a CONVERGENT modal grating method (Botten / classical
  lamellar) computes the layer modes IN the truncated ``N_pw``-order Fourier
  space, where there are exactly ``N_pw`` of them and they span that space.  The
  EME (and the FD oracle) instead compute modes at FULL real-space resolution
  (``Nx*Ny`` of them); truncating that real-space set to ``N_pw`` does not yield a
  basis of the retained-order space, so the square match is ill-conditioned.
  Efficiency truncation is a Fourier-space notion; the EME modes are real-space.

CONSEQUENCE:
  The EME lateral cascade is the right tool for 2-D MODES / band structure
  (``eme_2d`` -- validated, shipped) but NOT for diffraction EFFICIENCIES.  For
  efficiencies use ``lumenairy.rcwa_efficiency_2d`` (Fourier-space modes) or
  ``pmm_efficiency_2d``.  This module is kept as the validated slab math plus the
  recorded dead-end so the lateral-cascade route to efficiencies is not
  re-attempted blindly.

SCOPE: scalar Helmholtz (one polarization); a uniform-layer scalar slab equals
1-D TE.  ``rcwa_efficiency_2d`` solves the full VECTOR problem and does not reduce
to this scalar model for a genuine 2-D crossed grating.
"""
from __future__ import annotations

import numpy as np

from .eme_2d import layer_modes, mode_field, ref_2d_modes, strip_x_modes


# --------------------------------------------------------------------------- #
#  Rayleigh plane-wave basis of the uniform half-spaces                        #
# --------------------------------------------------------------------------- #
def plane_wave_orders(Mx, My):
    """Diffraction orders ``(m, p)``, ``|m| <= Mx``, ``|p| <= My``."""
    return [(m, p) for m in range(-Mx, Mx + 1) for p in range(-My, My + 1)]


def _grid(Lx, Ly, Nx, Ny):
    """Cell-centre sample grid (matches ``strip_x_modes`` / ``strips_to_eps_xy``
    / ``mode_field``)."""
    x = (np.arange(Nx) + 0.5) / Nx * Lx
    y = (np.arange(Ny) + 0.5) / Ny * Ly
    return x, y


def pw_matrix(orders, kx0, ky0, Lx, Ly, Nx, Ny):
    """Orthonormal plane-wave profiles on the grid.  Returns ``(U, kx, ky)`` with
    ``U[:, c]`` the flattened ``(Nx*Ny,)`` profile of order ``orders[c]``,
    normalised so ``<U_c, U_c> = 1`` under the cell inner product
    ``hx hy sum conj . .``."""
    x, y = _grid(Lx, Ly, Nx, Ny)
    kx = np.array([kx0 + 2 * np.pi * m / Lx for m, _ in orders])
    ky = np.array([ky0 + 2 * np.pi * p / Ly for _, p in orders])
    X, Y = np.meshgrid(x, y, indexing="ij")
    norm = np.sqrt(Lx * Ly)
    U = np.empty((Nx * Ny, len(orders)), dtype=complex)
    for c, (kxm, kyp) in enumerate(zip(kx, ky)):
        U[:, c] = (np.exp(1j * (kxm * X + kyp * Y)) / norm).ravel()
    return U, kx, ky


def _kz(eps, k0, kx, ky):
    """Longitudinal wavenumber, principal branch (Re>0 propagating / Im>0 decay)."""
    return np.sqrt(eps * k0 ** 2 - kx ** 2 - ky ** 2 + 0j)


# --------------------------------------------------------------------------- #
#  Basis-agnostic scalar mode matcher                                          #
# --------------------------------------------------------------------------- #
def mode_match(qz2, Psi, orders, *, kx0, ky0, k0, eps_sup, eps_sub, depth,
               Lx, Ly, Nx, Ny, inc_order=(0, 0)):
    """Scalar diffraction by mode matching for an ARBITRARY layer modal basis.

    ``qz2`` : layer-mode eigenvalues ``qz_n^2`` (length ``K``).
    ``Psi`` : layer-mode transverse fields, ``(Nx*Ny, K)`` (columns, any scale).
    Returns ``dict`` with ``r``/``t`` (order amplitudes), ``R``/``T`` (per-order
    efficiencies), ``energy = sum R + sum T`` and ``orders``.

    The matching algebra is EXACT (validated to machine precision against the
    analytic slab); it only delivers a convergent answer when ``Psi`` is a basis
    of the retained-order space -- true for a uniform layer (modes == plane
    waves), false for a structured layer's truncated real-space modes (see the
    module docstring).
    """
    U, kx, ky = pw_matrix(orders, kx0, ky0, Lx, Ly, Nx, Ny)
    w = Lx * Ly / (Nx * Ny)                              # cell quadrature weight
    # normalise modes, overlaps O[order, mode] = <U_order, psi_mode>
    Psi = Psi / np.sqrt(w * np.sum(np.abs(Psi) ** 2, axis=0))
    O = w * (U.conj().T @ Psi)                           # (Npw, K)
    qz = np.sqrt(np.asarray(qz2, complex) + 0j)          # forward branch
    kz_sup = _kz(eps_sup, k0, kx, ky)
    kz_sub = _kz(eps_sub, k0, kx, ky)
    Ksup, Ksub, Qz = np.diag(kz_sup), np.diag(kz_sub), np.diag(qz)
    E = np.diag(np.exp(1j * qz * depth))
    Einv = np.diag(np.exp(-1j * qz * depth))
    e_inc = np.array([1.0 if o == inc_order else 0.0 for o in orders], complex)
    kz0 = kz_sup[orders.index(inc_order)]
    # unknowns x = [c+, c-]; rows: (I) z=0 deriv-eliminated, (II) z=depth radiation
    #  (I)  Ksup O (c+ + c-) + O Qz (c+ - c-) = 2 Ksup e_inc
    #  (II) (O Qz - Ksub O) E c+ - (O Qz + Ksub O) Einv c- = 0
    KsO, OQ, KbO = Ksup @ O, O @ Qz, Ksub @ O
    top = np.hstack([KsO + OQ, KsO - OQ])
    bot = np.hstack([(OQ - KbO) @ E, -(OQ + KbO) @ Einv])
    A = np.vstack([top, bot])
    rhs = np.concatenate([2.0 * (Ksup @ e_inc), np.zeros(len(orders), complex)])
    cpm = np.linalg.lstsq(A, rhs, rcond=None)[0]
    cp, cm = cpm[:O.shape[1]], cpm[O.shape[1]:]
    r = O @ (cp + cm) - e_inc                            # reflected amplitudes
    t = O @ (E @ cp + Einv @ cm)                         # transmitted amplitudes
    prop_s, prop_b = kz_sup.real > 1e-9, kz_sub.real > 1e-9
    R = np.where(prop_s, kz_sup.real / kz0.real * np.abs(r) ** 2, 0.0)
    T = np.where(prop_b, kz_sub.real / kz0.real * np.abs(t) ** 2, 0.0)
    return {"orders": orders, "r": r, "t": t, "R": R, "T": T,
            "energy": float(R.sum() + T.sum())}


# --------------------------------------------------------------------------- #
#  Driver A -- 2-D-FD layer modes (EXACT for uniform; non-convergent structured) #
# --------------------------------------------------------------------------- #
def diffraction_fd(eps_xy, Lx, Ly, Nx, Ny, k0, eps_sup, eps_sub, depth,
                   Mx, My, *, kx0=0.0, ky0=0.0, n_modes=None):
    """Mode-matched diffraction using the direct 2-D-FD layer modes.

    EXACT for a uniform layer (the slab); for a STRUCTURED layer it inherits the
    module-level non-convergence (the FD modes are real-space, not order-space) --
    kept to demonstrate that the failure is independent of the EME mode-finding.

    Square method: keep ``K = N_pw`` modes for ``N_pw`` retained orders (highest
    ``qz^2`` by default; ``n_modes`` overrides ``K``)."""
    w, V = ref_2d_modes(eps_xy, Lx, Ly, Nx, Ny, k0, kx0=kx0, ky0=ky0,
                        return_vecs=True)
    K = (2 * Mx + 1) * (2 * My + 1) if n_modes is None else n_modes
    K = min(K, len(w))
    Psi = V[:, :, :K].reshape(Nx * Ny, K)
    return mode_match(w[:K], Psi, plane_wave_orders(Mx, My), kx0=kx0, ky0=ky0,
                      k0=k0, eps_sup=eps_sup, eps_sub=eps_sub, depth=depth,
                      Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)


# --------------------------------------------------------------------------- #
#  Driver B -- EME layer modes from 1-D strips (the attempted surface solver)   #
# --------------------------------------------------------------------------- #
def diffraction_eme(strips, Lx, Nx, Ly, k0, eps_sup, eps_sub, depth, Mx, My, *,
                    kx0=0.0, ky0=0.0, qz2_window=None, Ny=None, n_scan=600):
    """Mode-matched diffraction using the EME layer modes (``eme_2d``).

    EXPLORATORY -- records the dead end.  The layer modes are found by
    ``layer_modes`` over ``qz2_window`` and their fields reconstructed by
    ``mode_field``; the matcher is shared with ``diffraction_fd``.  For a
    structured layer this does NOT converge (module docstring); use
    ``rcwa_efficiency_2d`` / ``pmm_efficiency_2d`` for efficiencies.

    Returns the ``mode_match`` dict (same shape as ``diffraction_fd``) with the
    retained layer eigenvalues attached as ``res["qz2"]``.
    """
    Npw = (2 * Mx + 1) * (2 * My + 1)
    if qz2_window is None:
        hi = max(np.max(np.real(e)) for e, _ in strips) * k0 ** 2
        qz2_window = (-1.5 * hi, hi)
    if Ny is None:
        Ny = max(8 * (2 * My + 1), 4 * len(strips))
    qz2 = layer_modes(strips, Lx, Nx, Ly, k0, qz2_window, kx0=kx0, ky0=ky0,
                      n_scan=n_scan)
    if len(qz2) < Npw:
        raise ValueError(
            f"found {len(qz2)} layer modes < N_pw={Npw}; widen qz2_window (lower "
            f"floor) or raise n_scan -- the modal basis is under-resolved")
    qz2 = qz2[:Npw]                                  # square method: K = N_pw
    sm = [(strip_x_modes(e, Lx, Nx, k0, kx0), h) for e, h in strips]
    Psi = np.empty((Nx * Ny, Npw), dtype=complex)
    for n, q in enumerate(qz2):
        Psi[:, n] = mode_field(sm, q, ky0, Ly, Ny)[0].reshape(Nx * Ny)
    res = mode_match(qz2, Psi, plane_wave_orders(Mx, My), kx0=kx0, ky0=ky0,
                     k0=k0, eps_sup=eps_sup, eps_sub=eps_sub, depth=depth,
                     Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
    res["qz2"] = qz2                                 # bare dict, like diffraction_fd
    return res
