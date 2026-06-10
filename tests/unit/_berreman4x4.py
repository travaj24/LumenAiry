"""Independent Berreman 4x4 oracle for the full-3x3 (out-of-plane) anisotropic
RCWA 1-D path (v5.11.0 / GAP7-full).

A UNIFORM full-tensor slab between isotropic half-spaces is solved by a direct
first-order reduction of the layer ODE in the tangential state ``[Ex, Ey, Hx,
Hy]`` (Berreman 1972 / Yeh 1979), eigen-decomposed and matched at both
interfaces -- NO Fourier, no P/Q/G blocks, no shared algebra with the RCWA path.
This makes it a genuinely independent reference for the integrated solver.

Built in the SOLVER-INTERNAL convention (conjugated eps, ``z' = k0 z``, forward
``exp(+i kz z)``) and conjugated back at the end to the PUBLIC ``exp(-i w t)``
convention so it compares bit-for-bit with :func:`lumenairy.rcwa_jones_1d`.

Ported from the validated standalone ``C:/tmp/gap7_proto_2.py`` with one fix:
the transmitted Jones ``Jt`` is taken as the substrate forward-mode tangential
field AT THE EXIT PLANE (``Asub_fwd @ c_trn``), not the proto's version that
erroneously propagated it back through the layer (``T_layer @ Asub_fwd``), which
gave a wrong transmission amplitude (off by a complex ~0.84 factor).  With the
fix, an isotropic film reproduces the analytic Airy transmission bit-for-bit and
energy ``R + (n_sub/n_sup) |Jt|^2 = 1`` to ~1e-15.
"""
from __future__ import annotations

import numpy as np

_C = np.complex128


def berreman_jones(eps_lab, n_sup, n_sub, depth, wavelength, angle):
    """Analytic 2x2 Jones reflection AND transmission of a UNIFORM full-3x3
    tensor slab between isotropic half-spaces, via the Berreman 4x4 method.

    Parameters
    ----------
    eps_lab : (3, 3) complex
        Slab permittivity in the lab ``(x, y, z)`` basis (PUBLIC ``Im > 0`` loss).
    n_sup, n_sub : complex
        Superstrate (incidence) / substrate (transmission) indices.
    depth, wavelength, angle : float
        Slab thickness (m), vacuum wavelength (m), planar incidence (rad).

    Returns
    -------
    (Jr, Jt) : (2, 2) complex ndarrays
        Columns = incident lab ``[Ex; Ey]``; rows = reflected / transmitted lab
        ``[Ex; Ey]`` at the entrance (``z = 0``) / exit (``z = depth``) plane.
        PUBLIC ``exp(-i w t)`` convention.
    """
    eps = np.conj(np.asarray(eps_lab).astype(_C))         # internal convention
    eps_sup = complex(np.conj(_C(n_sup) ** 2))
    eps_sub = complex(np.conj(_C(n_sub) ** 2))
    k0 = 2.0 * np.pi / wavelength
    Kx = float(np.real(np.conj(_C(n_sup))) * np.sin(angle))   # planar mount: Ky=0
    Ky = 0.0

    Delta = _berreman_delta(eps, Kx, Ky)
    gam, Psi = np.linalg.eig(Delta)

    A_sup, qs_sup = _iso_field_modes(eps_sup, Kx, Ky)
    A_sub, qs_sub = _iso_field_modes(eps_sub, Kx, Ky)

    phase = np.exp(gam * k0 * depth)
    T_layer = Psi @ np.diag(phase) @ np.linalg.inv(Psi)

    sup_fwd, sup_bwd = _split_updown(qs_sup)
    sub_fwd, sub_bwd = _split_updown(qs_sub)

    Asup_fwd = A_sup[:, sup_fwd]
    Asup_bwd = A_sup[:, sup_bwd]
    Asub_fwd = A_sub[:, sub_fwd]
    # Field continuity at z=0:  Asup_fwd c_inc + Asup_bwd c_ref
    #                            = T_layer Asub_fwd c_trn  (substrate field at
    #                              z=depth back-propagated to z=0).
    # Unknown u = [c_ref(2); c_trn(2)]:
    Mmat = np.column_stack([Asup_bwd, -(T_layer @ Asub_fwd)])  # 4x4

    Jr = np.zeros((2, 2), dtype=_C)
    Jt = np.zeros((2, 2), dtype=_C)
    E_of_cinc = Asup_fwd[:2, :]      # Ex,Ey rows of the forward sup modes (2x2)
    for col, Einc in enumerate([np.array([1.0, 0.0], dtype=_C),
                                np.array([0.0, 1.0], dtype=_C)]):
        c_inc = np.linalg.solve(E_of_cinc, Einc)
        rhs = -(Asup_fwd @ c_inc)
        u = np.linalg.solve(Mmat, rhs)
        c_ref = u[:2]
        c_trn = u[2:]
        Eref = Asup_bwd[:2, :] @ c_ref
        # FIX: transmitted tangential E is the SUBSTRATE forward-mode field at the
        # EXIT plane z=depth (Asub_fwd c_trn), NOT propagated back through the
        # layer (the proto's T_layer @ Asub_fwd over-applied the layer phase).
        Etrn = Asub_fwd[:2, :] @ c_trn
        Jr[:, col] = Eref
        Jt[:, col] = Etrn
    return np.conj(Jr), np.conj(Jt)


def _berreman_delta(eps, Kx, Ky):
    """Berreman 4x4 ``Delta`` for state ``[Ex, Ey, Hx, Hy]``, ``dpsi/dz' = Delta
    psi`` (``z' = k0 z``), eigenvalues ``q = kz/k0``.  Full 3x3 eps (internal
    convention), ``mu = 1``; ``Ez, Hz`` eliminated from the normalized Maxwell
    curl equations."""
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
    # Conical (Kx*Ky != 0) entries fixed 2026-06-10 (audit F1-berreman):
    # rotation covariance blkdiag(R2,R2) Delta(eps_rot, kt, 0) blkdiag(R2,R2)^T
    # == Delta(eps, Kx, Ky) pinned the defect to exactly +/-Kx*Ky at [2,0],
    # [2,3], [3,2] (eps-independent); the uncorrected form gave R+T = 14.35 on
    # a lossless isotropic slab at (theta=20, phi=35) while being exact at
    # phi = 0 / 90 where Kx*Ky = 0 (so every planar test passed).
    D[2, 0] = eyx - eyz * ezx * d + Kx * Ky
    D[2, 1] = eyy - eyz * ezy * d - Kx * Kx
    D[2, 2] = eyz * Ky * d
    D[2, 3] = -eyz * Kx * d
    D[3, 0] = exz * ezx * d - exx + Ky * Ky
    D[3, 1] = exz * ezy * d - exy - Kx * Ky
    D[3, 2] = -exz * Ky * d
    D[3, 3] = exz * Kx * d
    return D


def _iso_field_modes(eps, Kx, Ky):
    """4 plane-wave field columns ``[Ex, Ey, Hx, Hy]`` of an isotropic
    half-space.  Returns ``(A (4x4), q (4,))`` with eigenvalues ``q = kz/k0``."""
    D = _berreman_delta(eps * np.eye(3, dtype=_C), Kx, Ky)
    q, A = np.linalg.eig(D)
    return A, q


def _split_updown(gam):
    """Indices of the 2 forward (+z, outgoing/decaying) and 2 backward modes.
    ``gam = kz/k0``; forward is the propagating root with ``Im(gam) > 0`` or,
    for an evanescent mode, the one decaying as ``z -> +inf`` (``Re < 0`` in the
    ``exp(gam k0 z)`` convention used by ``_berreman_delta``)."""
    g = np.asarray(gam)
    tol = 1e-9 * max(1.0, np.max(np.abs(g)))
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
        fwd = list(order[:2])
        bwd = list(order[2:])
    return np.array(fwd), np.array(bwd)
