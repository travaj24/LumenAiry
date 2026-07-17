"""Vector (TE/TM) EME lateral cascade -- full-Maxwell 2-D Bloch layer modes from
1-D-x building blocks.

The vector generalization of ``eme_2d.py`` (scalar Helmholtz).  A doubly-periodic
layer ``eps(x, y)`` that is piecewise-uniform in y (a stack of y-strips, each
``eps(x)``) has its full VECTOR Bloch modes -- propagation constant ``qz`` along z,
both polarizations -- built without a 2-D vector eigensolve:

1. Each y-strip ``eps(x)`` supplies its 1-D-x VECTOR modes propagating in y
   (``strip_vector_modes``): the y-tangential Berreman state ``[Ex, Ez, Hx, Hz](x)``
   (the fields continuous across a ``y = const`` interface), eigenvalue ``ky``.
2. The strips are joined laterally by the field-continuity conditions on the
   tangential state -- forward modal state ``[W; V]`` with ``W = [Ex; Ez]``
   (tangential E) and ``V = [Hx; Hz]`` (tangential H); backward is ``[W; -V]``
   (reversing y flips H, keeps E -- proven exact: the strip generator is
   block-anti-diagonal).
3. The 2-D modes are the ``qz^2`` where the GLOBAL lateral interface block matrix
   ``G(qz^2)`` is singular (``sigma_min(G) = 0``), found by a real-axis scan +
   Brent refinement + a degeneracy-agnostic rank-drop acceptance test.  ``G`` (one
   block per strip, no accumulation) is used instead of the Redheffer cascade
   residual ``sigma_min(M)``: the cascade physics is exact but its star-product
   loses conditioning as the propagating-strip-mode count grows toward low
   ``qz^2``, missing those modes (the forward finder recovered only 2/16 on the
   reference structured cell); the single block ``G`` is well-conditioned and
   recovers ~14-16/16 (sharpening with ``Nx``), needing no multi-basis rotation
   (it is basis-independent).  See ``layer_vector_modes`` for the validated regime
   (structured layers) and the high-degeneracy limitation.

TE / TM.  TE = E along the invariant z-axis (``Ez``); TM = ``Hz``.  Within a single
x-UNIFORM strip the two polarizations DECOUPLE (the strip is 1-D stratified).  They
HYBRIDIZE through the x-structure at conical incidence: the strip operator is
**qz-DEPENDENT** -- a structured strip's x-walls couple TE/TM with strength
``~ qz * d(eps)/dx`` (analytic coupling ``C(x) = (qz/k0) d/dx[1/(k0^2 eps - qz^2)]``,
exactly zero for a uniform strip or at ``qz = 0``).  The ky^2 SPECTRUM shifts
rigidly ``ky^2(qz) = ky^2(0) - qz^2`` for isotropic eps, but the EIGENVECTORS (the
``W, V`` the cascade consumes) rotate with qz -- so the strip modes are rebuilt at
each trial ``qz^2`` (NOT precomputed once as in the scalar path).

At ``qz = 0`` the solver reduces EXACTLY to two scalar ``eme_2d`` runs (TE = Ez =
the scalar Helmholtz field; TM = Hz with the inverse rule).  Validated against a
direct Yee-staggered 2-D vector FD solve (``ref_2d_modes_vector``); uniform ->
analytic (doubly degenerate TE+TM); structured -> the 2-D-FD converges to the EME.

SCOPE: a mode / band-structure solver (NOT diffraction efficiencies -- see
``eme_diffraction.py`` for why the lateral-cascade modes are the wrong basis for
efficiency truncation).  Conventions: ``exp(+i qz z)``, ``exp(+i ky y)``;
normalized Maxwell ``curl E = i k0 h``, ``curl h = -i k0 eps E`` (``h = Z0 H``).

Materials/geometry (see ``layer_vector_modes`` / ``strip_vector_modes`` /
``eps_xy_to_strips``): isotropic OR ``(Nx,3,3)`` anisotropic FULL out-of-plane 3x3
(DIAGONAL + ``exz``/``ezx`` + ``eyz``/``ezy``) lossless eps; scalar magnetic
``mu(x,y)``; arbitrary ``eps(x,y)`` via the ``eps_xy_to_strips`` y-staircase.  The
``eyz``/``ezy`` STRIP modes are rigorous (validated vs the analytic Christoffel
determinant ``det(k k^T - |k|^2 I + k0^2 eps) = 0``), but ``eyz`` breaks the
block-anti-diagonal ``[W; -V]`` backward mode, so the eyz LAYER finder
(``layer_vector_modes``) is GATED -- see ``_strip_modes_at``.  LOSSY layers: the FD
oracle ``ref_2d_modes_vector(return_complex=True)`` solves them EXACTLY (complex
``qz^2``); for the strip-solver a SEEDED Beyn refiner (``beyn_refine_complex`` /
``layer_vector_modes_complex``) reaches complex / lossy / leaky ``qz^2`` from a
coarse complex seed (x-FD floor ~1e-2).
Out of scope (research, premises probe-tested): the eyz LAYER cascade (the general
non-``[W; -V]`` backward); AUTONOMOUS complex-mode discovery (vs the seeded Beyn) and
lossy-anisotropy; in-plane ``exy``/``eyx`` coupling; tensor-eps with ``mu != 1``.
NOTE: high-degeneracy (uniform-slab) mode-finding is NOT a conditioning wall -- the
floor is the O(h^2) x-FD error on high-``|kx|`` strip modes (refuted premise); use
the oracle / analytic dispersion there.

The strip generator and the FD oracle were derived + numerically self-validated as
separate operators (uniform analytic; qz=0 scalar reduction byte-exact; qz-coupling
confirmed; spurious-free Yee oracle cross-checked vs an independent Fourier-PWE)
before this cascade was built on them.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eig, svdvals
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import eigs, splu

from ...backend.array import is_jax_array


# =========================================================================== #
#  1-D-x VECTOR strip modes (Berreman-in-y, Yee-staggered, qz-dependent)       #
# =========================================================================== #
def _strip_yee_diffs(Nx, h, kx0, Lx):
    """Periodic-cell staggered first differences (Yee).  ``Df`` maps an
    integer-node field to the half nodes, ``Db`` a half-node field to the
    integer nodes, with ``Db = -Df^H`` so ``Db @ Df`` is the exact 3-point
    Laplacian (no spurious Nyquist null -- a collocated centred first difference
    must NOT be used here).  Bloch wrap carries ``exp(+/- i kx0 Lx)``."""
    ph = np.exp(1j * kx0 * Lx)
    Df = np.zeros((Nx, Nx), dtype=complex)
    Db = np.zeros((Nx, Nx), dtype=complex)
    for i in range(Nx):
        Df[i, i] -= 1.0 / h
        Df[i, (i + 1) % Nx] += (ph if i == Nx - 1 else 1.0) / h
        Db[i, i] += 1.0 / h
        Db[i, (i - 1) % Nx] -= ((1.0 / ph) if i == 0 else 1.0) / h
    return Df, Db


def _strip_vector_generator(eps_x, Lx, Nx, k0, kx0, qz, mu_x=1.0):
    """The ``4Nx x 4Nx`` generator ``A`` with ``d/dy psi = A psi`` for the
    y-tangential Berreman state ``psi = [Ex, Ez, Hx, Hz](x)`` of a strip
    ``eps = eps(x)`` (invariant in y, z), propagating in y.  Eigenvalues of ``A``
    are ``i*ky``.  qz-DEPENDENT.  Yee placement: ``Ex, Hz`` on integer nodes,
    ``Ez, Hx`` on half nodes.

    ``eps_x`` is either a SCALAR / ``(Nx,)`` isotropic profile (the legacy path,
    byte-identical to the original) or a per-node ``(Nx, 3, 3)`` permittivity
    TENSOR (DIAGONAL + ``exz``/``ezx`` out-of-plane coupling, lossless), which
    routes to :func:`_strip_vector_generator_tensor`.  The tensor branch reduces
    BYTE-EXACTLY to this scalar body for an isotropic tensor ``e(x) * I3``."""
    eps = np.asarray(eps_x, dtype=complex)
    if eps.ndim == 3:                       # (Nx, 3, 3) anisotropic tensor branch
        return _strip_vector_generator_tensor(eps, Lx, Nx, k0, kx0, qz, mu_x)
    h = Lx / Nx
    Df, Db = _strip_yee_diffs(Nx, h, kx0, Lx)
    Z = np.zeros((Nx, Nx), dtype=complex)
    Ei = np.diag(1.0 / eps)         # 1/eps(x) pointwise
    Eps = np.diag(eps)
    # scalar permeability mu(x): curl E = i k0 mu h.  mu enters 6 terms; mu=1 (the
    # default) makes Mu = Mi = I -> byte-identical to the lossless mu=1 generator.
    mu = np.asarray(mu_x, dtype=complex)
    if mu.ndim == 0:
        mu = np.full(Nx, complex(mu))
    Mi, Mu = np.diag(1.0 / mu), np.diag(mu)
    # dEx/dy = Db[-(qz/k0)(1/eps)Hx - (i/k0)(1/eps)Df Hz] - i k0 mu Hz
    A_Ex_Hx = Db @ (-(qz / k0) * Ei)
    A_Ex_Hz = Db @ (-(1j / k0) * (Ei @ Df)) - 1j * k0 * Mu
    # dEz/dy = i k0 mu Hx - i(qz^2/k0)(1/eps)Hx + (qz/k0)(1/eps)Df Hz
    A_Ez_Hx = 1j * k0 * Mu - 1j * (qz ** 2 / k0) * Ei
    A_Ez_Hz = (qz / k0) * (Ei @ Df)
    # dHx/dy = Df[(qz/k0)(1/mu)Ex + (i/k0)(1/mu)Db Ez] + i k0 eps Ez
    A_Hx_Ex = Df @ ((qz / k0) * Mi)
    A_Hx_Ez = Df @ ((1j / k0) * (Mi @ Db)) + 1j * k0 * Eps
    # dHz/dy = -i k0 eps Ex + i(qz^2/k0)(1/mu)Ex - (qz/k0)(1/mu)Db Ez
    A_Hz_Ex = -1j * k0 * Eps + 1j * (qz ** 2 / k0) * Mi
    A_Hz_Ez = -(qz / k0) * (Mi @ Db)
    return np.block([
        [Z, Z, A_Ex_Hx, A_Ex_Hz],
        [Z, Z, A_Ez_Hx, A_Ez_Hz],
        [A_Hx_Ex, A_Hx_Ez, Z, Z],
        [A_Hz_Ex, A_Hz_Ez, Z, Z],
    ])


def _strip_vector_generator_tensor(eps_t, Lx, Nx, k0, kx0, qz, mu_x=1.0):
    """ANISOTROPIC strip generator (the tensor branch of
    :func:`_strip_vector_generator`).

    ``eps_t`` is a per-node ``(Nx, 3, 3)`` permittivity tensor.  SCOPE: the FULL
    out-of-plane 3x3 -- DIAGONAL entries, ``exz`` / ``ezx`` (``xz``) coupling, AND
    ``eyz`` / ``ezy`` (``yz``) coupling, LOSSLESS (real eps).  (The in-plane
    ``exy`` / ``eyx`` coupling is still out of scope.)

    Derived directly from the normalized Maxwell curl equations
    (``curl E = i k0 h``, ``curl h = -i k0 eps E``), propagating in y with
    ``Ey, Hy`` eliminated via the in-plane ``y`` curl rows.  The key role-swap is
    that ``1/eyy`` here plays the part ``1/ezz`` plays in the planar Berreman
    ``Delta`` (y is the strip's eliminated in-plane axis).  Yee placement matches
    the scalar body: E-rows take the OUTER ``Db`` / inner ``Df``; H-rows the outer
    ``Df`` / inner ``Db``.  For an isotropic tensor ``e(x) * I3`` every block
    equals the scalar generator's BYTE-FOR-BYTE (validated, gate 1).

    The DIAGONAL + ``exz``/``ezx`` blocks are written verbatim from the original
    (xz-only) body and the ``eyz``/``ezy`` terms are ADDED as separate
    contributions, every one proportional to ``Ey_Ez = -(1/eyy) eyz`` or to
    ``ezy``, so at ``eyz = ezy = 0`` they vanish to the exact zero matrix and the
    block reduces BYTE-FOR-BYTE to the original xz-only generator (GATE A,
    ``max|dA| = 0`` across ``Nx in {1,20}``, ``kx0 in {0,0.37}``,
    ``qz2 in {0,9,36}`` and diagonal / sym-exz / asym-exz).

    The ``eyz``/``ezy`` coupling enters through the ``Ey`` elimination
    ``Ey = -(1/eyy)[(curl h)_y / (i k0) + eyz Ez]``: its ``eyz Ez`` part feeds the
    E-rows (``dEz/dy += i qz Ey``, ``dEx/dy += Db Ey``), populating the E-E block;
    and the ``i k0 ezy Ey`` term in ``dHx/dy`` populates the H-H block (its ``Hx`` /
    ``Hz`` parts) plus an ``Hx <- Ez`` cross term (its ``Ey_Ez Ez`` part).  The E-E /
    H-H blocks are zero for diagonal / xz, so the role-swapped Berreman is the WRONG
    oracle for yz (its z-propagation axis maps onto the eliminated y-axis): the
    CORRECT independent oracle is the analytic CHRISTOFFEL determinant
    ``det(k k^T - |k|^2 I + k0^2 eps) = 0`` solved for ``ky`` at fixed
    ``(kx, qz)`` -- the uniform-strip forward ``|ky|`` multiset matches it to
    ~1e-13 at ``kx0 = 0`` (symmetric eyz=ezy, asymmetric eyz!=ezy, and a combined
    exz+eyz cell; GATE B), with O(h^2) Yee convergence at ``kx0 != 0``.

    The ``eyz``/``ezy`` E-E / H-H blocks BREAK the block-anti-diagonal structure
    (``S A S != -A``, where ``S`` flips the H sign) that ``strip_vector_modes``
    eig(B@C) reduction AND the global block-G ``[W; -V]`` backward mode rely on.
    ``strip_vector_modes`` detects this via its ``bc_ok`` guard and falls back to
    the full ``eig(A)`` with true E/H eigenvectors (so the STRIP modes stay
    rigorous), but ``[W; -V]`` is no longer the exact backward mode for eyz, so
    the LAYER mode-finder (``layer_vector_modes``) is gated on eyz -- see
    ``_strip_modes_at``."""
    e = np.asarray(eps_t, dtype=complex)
    if e.shape != (Nx, 3, 3):
        raise ValueError(
            "_strip_vector_generator: a tensor eps must have shape (Nx, 3, 3), "
            f"got {e.shape}.")
    if np.any(np.asarray(mu_x, dtype=complex) != 1.0):
        raise NotImplementedError(
            "_strip_vector_generator: anisotropic-tensor eps with mu != 1 is not "
            "implemented (scalar-eps + scalar-mu, or tensor-eps + mu=1, only).")
    h = Lx / Nx
    Df, Db = _strip_yee_diffs(Nx, h, kx0, Lx)
    I = np.eye(Nx, dtype=complex)
    Z = np.zeros((Nx, Nx), dtype=complex)
    Eyyi = np.diag(1.0 / e[:, 1, 1])     # 1/eyy(x) -- the role-swapped 1/ezz
    Exx = np.diag(e[:, 0, 0])
    Ezz = np.diag(e[:, 2, 2])
    Exz = np.diag(e[:, 0, 2])
    Ezx = np.diag(e[:, 2, 0])
    Eyz = np.diag(e[:, 1, 2])
    Ezy = np.diag(e[:, 2, 1])
    # --- BASE blocks (diagonal + exz/ezx): VERBATIM from the original xz-only body
    # E-rows: outer Db, inner Df (matches the scalar Yee placement exactly).
    A_Ex_Hx = Db @ (-(qz / k0) * Eyyi)
    A_Ex_Hz = Db @ (-(1j / k0) * (Eyyi @ Df)) - 1j * k0 * I
    A_Ez_Hx = 1j * k0 * I - 1j * (qz ** 2 / k0) * Eyyi
    A_Ez_Hz = (qz / k0) * (Eyyi @ Df)
    # H-rows: outer Df, inner Db.  exz/ezx enter the E<->H blocks (Ezx in dHx/dy,
    # Exx/Exz in dHz/dy).
    A_Hx_Ex = Df @ ((qz / k0) * I) + 1j * k0 * Ezx
    A_Hx_Ez = Df @ ((1j / k0) * Db) + 1j * k0 * Ezz
    A_Hz_Ex = -1j * k0 * Exx + 1j * (qz ** 2 / k0) * I
    A_Hz_Ez = -(qz / k0) * Db - 1j * k0 * Exz
    # --- yz-coupling additions (every term == 0 when eyz=ezy=0 -> byte-exact base)
    # Ey = -(1/eyy)[(qz/k0)Hx - (1/(i k0))Df Hz + eyz Ez]; the eyz part is Ey_Ez Ez.
    Ey_Ez = -(Eyyi @ Eyz)                       # eyz part of the Ey elimination
    Ey_Hx = -(Eyyi @ ((qz / k0) * I))           # (only needed for the ezy*Ey term)
    Ey_Hz = -(Eyyi @ (-(1.0 / (1j * k0)) * Df))
    # E-rows pick up Ey: dEz/dy += i qz Ey, dEx/dy += Db Ey -> the E-E block.
    A_Ex_Ez = Db @ Ey_Ez
    A_Ez_Ez = 1j * qz * Ey_Ez
    # dHx/dy picks up +i k0 ezy Ey: its Ez part adds to A_Hx_Ez, its Hx/Hz parts
    # populate the H-H block.
    A_Hx_Ez = A_Hx_Ez + 1j * k0 * (Ezy @ Ey_Ez)
    A_Hx_Hx = 1j * k0 * (Ezy @ Ey_Hx)
    A_Hx_Hz = 1j * k0 * (Ezy @ Ey_Hz)
    return np.block([
        [Z, A_Ex_Ez, A_Ex_Hx, A_Ex_Hz],
        [Z, A_Ez_Ez, A_Ez_Hx, A_Ez_Hz],
        [A_Hx_Ex, A_Hx_Ez, A_Hx_Hx, A_Hx_Hz],
        [A_Hz_Ex, A_Hz_Ez, Z, Z],
    ])


def _strip_split_forward(ky):
    """Indices of the ``2Nx`` FORWARD (+y) modes (mirrors
    ``berreman._split_fwd_bwd``): ``exp(i ky y)`` decays forward when
    ``Im(ky) > 0``; a real propagating ``ky`` is forward when ``Re(ky) > 0``."""
    tol = 1e-9 * max(1.0, float(np.max(np.abs(ky))))
    fwd = []
    for i, v in enumerate(ky):
        if v.imag > tol:
            fwd.append(i)
        elif v.imag < -tol:
            pass
        elif v.real > 0:
            fwd.append(i)
    return np.array(fwd, dtype=int)


def strip_vector_modes(eps_x, Lx, Nx, k0, kx0=0.0, qz2=0.0, mu_x=1.0):
    """1-D-x VECTOR eigenmodes of a y-strip (``eps = eps(x)``, invariant in y, z)
    propagating in y -- the vector analog of ``eme_2d.strip_x_modes`` + ``_wv``.

    Returns ``(ky, W, V)`` with the ``2Nx`` FORWARD modes:
      ``ky`` : (2Nx,) lateral wavenumbers (propagator ``exp(i ky h)``, used
               directly in ``_prop`` like the scalar path);
      ``W``  : (2Nx, 2Nx) tangential-E, rows ``[Ex; Ez]``, columns the modes;
      ``V``  : (2Nx, 2Nx) tangential-H, rows ``[Hx; Hz]``.
    ``[W; V]`` is the forward modal state, ``[W; -V]`` the backward, feeding the
    UNCHANGED ``eme_2d._interface / _prop / _star`` (a ``2Nx`` block).

    The operator is qz-DEPENDENT (rebuild per trial ``qz2``); see the module
    docstring.  ``qz = sqrt(qz2)``.  ``mu_x`` is the scalar (or ``(Nx,)``)
    permeability ``mu(x)`` (default 1); magnetic media give the dispersion
    ``eps mu k0^2 - kx^2 - ky^2`` (validated).
    """
    qz = np.sqrt(complex(qz2)) if qz2 != 0.0 else 0.0
    A = _strip_vector_generator(eps_x, Lx, Nx, k0, kx0, complex(qz), mu_x)
    n2 = 2 * Nx
    # A is block-ANTI-diagonal [[0, B], [C, 0]] (E-rows couple only to H, H-rows
    # only to E), so the 4Nx eig(A) reduces EXACTLY to the 2Nx eig(B@C) -- ~7.5x
    # cheaper, the dominant solver cost.  A[u; w] = mu[u; w] gives B w = mu u,
    # C u = mu w => (B C) u = mu^2 u; the A-eigenvalue is mu = i ky, the H-part is
    # w = C u / mu, and the two roots +-mu of mu^2 are the +-ky forward/backward
    # pair (same E-part u, H-part sign-flipped -- the [W; -V] structure).  This
    # holds for the SCALAR path and for DIAGONAL + exz/ezx tensors (gate 3); the
    # bc_ok guard below detects any future break (the E-E / H-H blocks becoming
    # nonzero -- e.g. an eyz/ezy term) and falls back to the full eig(A).
    AEE, AHH = A[:n2, :n2], A[n2:, n2:]
    bc_tol = 1e-10 * max(1.0, float(np.max(np.abs(A))))
    bc_ok = (np.max(np.abs(AEE)) < bc_tol and np.max(np.abs(AHH)) < bc_tol)
    B, C = A[:n2, n2:], A[n2:, :n2]
    if bc_ok:
        mu2, U = np.linalg.eig(B @ C)
        mu = np.sqrt(mu2)
        kys = np.concatenate([mu / 1j, -mu / 1j])
        Us = np.concatenate([U, U], axis=1)
        mus = np.concatenate([mu, -mu])
        fwd = _strip_split_forward(kys)
        if fwd.shape[0] != n2:           # degenerate-real fallback (Berreman-style)
            fwd = np.lexsort((-kys.real, -kys.imag))[:n2]
        ky_f, U_f, mu_f = kys[fwd], Us[:, fwd], mus[fwd]
        return ky_f, U_f, (C @ U_f) / mu_f[None, :]
    # general fallback: full 4Nx eig(A), then re-split E/H from the eigenvectors.
    mu_all, Psi = np.linalg.eig(A)       # mu = i ky
    kys = mu_all / 1j
    fwd = _strip_split_forward(kys)
    if fwd.shape[0] != n2:
        fwd = np.lexsort((-kys.real, -kys.imag))[:n2]
    ky_f = kys[fwd]
    W_f = Psi[:n2, fwd]                   # tangential-E rows [Ex; Ez]
    V_f = Psi[n2:, fwd]                   # tangential-H rows [Hx; Hz]
    return ky_f, W_f, V_f


# =========================================================================== #
#  Global block-G null-space mode-finder (well-conditioned, full-band)          #
# =========================================================================== #
#  The mode condition is sigma_min(G(qz^2)) = 0 with G the GLOBAL lateral
#  interface block matrix -- NOT the Redheffer cascade residual sigma_min(M).
#  The accumulated star-product cell_smatrix -> _bloch_residual_M loses
#  conditioning as the propagating-strip-mode count grows (toward low qz^2): the
#  cascade sigma_min floors ~1e-2 at genuine low-qz^2 modes and misses them
#  (adversarially measured: 2/16 modes found on the reference 2-strip cell).  The
#  single block G accumulates nothing -- each block carries one strip's
#  exp(+-i ky h) -- so sigma_min stays ~1e-8 throughout and the FULL band is
#  recovered (16/16).  The cascade physics is correct (signs / [W;-V] backward /
#  oracle all proven exact in adversarial review); only its residual conditioning
#  is poor, so the mode-finder uses G.  Bonus: G is basis-independent, so no
#  multi-basis rotation is needed (the cascade M lived in the first strip's basis).
def _strips_have_eyz(strips):
    """True if any strip's eps tensor has a nonzero ``eyz``/``ezy`` (``yz``)
    entry.  The global block-G ``[W; -V]`` backward mode is EXACT only for the
    block-anti-diagonal strip generator (diagonal + ``exz``/``ezx``); ``eyz``
    populates the E-E and H-H blocks (``S A S != -A``), so ``[W; -V]`` is no
    longer the strip's true backward mode and the cascade mode-finder would
    return WRONG layer modes -- the LAYER path is gated on ``eyz`` (the general
    non-``[W; -V]`` backward cascade is deferred).  The STRIP operator / modes
    themselves stay rigorous (``strip_vector_modes`` falls back to the full
    ``eig(A)`` with true eigenvectors)."""
    for s in strips:
        ex = np.asarray(s[0], dtype=complex)
        if ex.ndim == 3:                          # (Nx, 3, 3) tensor strip
            tol = 1e-12 * max(1.0, float(np.max(np.abs(ex))))
            if (np.max(np.abs(ex[:, 1, 2])) > tol
                    or np.max(np.abs(ex[:, 2, 1])) > tol):
                return True
    return False


def _strip_modes_at(strips, Lx, Nx, k0, kx0, qz2):
    """The ``(ky, W, V, h)`` vector strip modes of every strip at trial ``qz2``
    (the strip operator is qz-dependent, so this is evaluated per ``qz2``).  A
    strip is ``(eps_x, h)`` or ``(eps_x, h, mu_x)`` (magnetic; default ``mu = 1``).

    GATE: raises ``NotImplementedError`` if any strip carries ``eyz``/``ezy``
    coupling -- the global block-G cascade hard-codes the ``[W; -V]`` backward
    mode, which is rigorously WRONG for an eyz strip (it breaks the
    block-anti-diagonal structure ``[W; -V]`` relies on; degeneracy-proof check
    ``S A S = -A`` fails, ``max|SAS+A| ~ 2`` for eyz vs ``0`` for exz/diagonal).
    Returning modes here would feed those wrong backward states to the layer
    mode-finder, so it is gated rather than silently incorrect.  The general
    backward cascade (consuming the strip's TRUE -ky eigenvectors) is deferred;
    use the FD oracle for eyz LAYER modes meanwhile.  (The STRIP operator and its
    modes via :func:`strip_vector_modes` remain rigorous and validated for eyz.)"""
    if _strips_have_eyz(strips):
        raise NotImplementedError(
            "layer_vector_modes: the global block-G cascade hard-codes the "
            "[W; -V] backward mode, which is exact only for block-anti-diagonal "
            "strips (diagonal + exz/ezx).  eyz/ezy (yz) coupling breaks that, so "
            "the cascade would return wrong layer modes -- it is gated (the "
            "general non-[W;-V] backward cascade is deferred).  The eyz STRIP "
            "operator/modes (strip_vector_modes) are rigorous; use "
            "ref_2d_modes_vector for eyz LAYER modes.")
    out = []
    for s in strips:
        eps_x, h = s[0], s[1]
        mu_x = s[2] if len(s) > 2 else 1.0
        out.append((*strip_vector_modes(eps_x, Lx, Nx, k0, kx0, qz2, mu_x), h))
    return out


def _global_block_G(wvk, t):
    """Column-equilibrated global lateral interface block matrix ``G(qz^2)`` and
    its column norms.  Rows: per-interface continuity of the tangential E-state
    ``W = [Ex; Ez]`` and H-state ``V = [Hx; Hz]``; columns: per-strip forward /
    backward modal amplitudes ``(a_s, b_s)`` (each ``2Nx``).  ``G`` is SINGULAR at
    a 2-D layer mode; equilibration (unit column norms) makes ``sigma_min`` a clean
    scale-free singularity indicator.  Backward modes are ``[W; -V]`` (proven exact
    -- the strip generator is block-anti-diagonal)."""
    S = len(wvk)
    M = wvk[0][1].shape[0]                        # 2Nx block size
    n = 2 * M * S
    G = np.zeros((n, n), dtype=complex)
    for r in range(S):
        ky_c, W_c, V_c, h_c = wvk[r]
        E_c, Ei_c = np.exp(1j * ky_c * h_c), np.exp(-1j * ky_c * h_c)
        nxt = (r + 1) % S
        _, W_n, V_n, _ = wvk[nxt]
        tt = t if r == S - 1 else 1.0             # Bloch phase on the wrap only
        rE, rH = 2 * M * r, 2 * M * r + M         # E-continuity / H-continuity rows
        ca, cb = 2 * M * r, 2 * M * r + M         # a_r / b_r columns
        na, nb = 2 * M * nxt, 2 * M * nxt + M
        # tangential-E continuity: W_c(a_c E_c + b_c Ei_c) = tt W_n(a_n + b_n)
        G[rE:rE + M, ca:ca + M] += W_c * E_c[None, :]
        G[rE:rE + M, cb:cb + M] += W_c * Ei_c[None, :]
        G[rE:rE + M, na:na + M] -= tt * W_n
        G[rE:rE + M, nb:nb + M] -= tt * W_n
        # tangential-H continuity: V_c(a_c E_c - b_c Ei_c) = tt V_n(a_n - b_n)
        G[rH:rH + M, ca:ca + M] += V_c * E_c[None, :]
        G[rH:rH + M, cb:cb + M] -= V_c * Ei_c[None, :]
        G[rH:rH + M, na:na + M] -= tt * V_n
        G[rH:rH + M, nb:nb + M] += tt * V_n
    cn = np.linalg.norm(G, axis=0)
    cn[cn < 1e-300] = 1.0
    return G / cn[None, :], cn


_NULL_KMAX = 6      # max null-space dimension the rank-drop test scans for (a
#                     uniform layer's +-ky x 2-pol gives 4; >6 is rare)

# Robust mode detection (backend-invariant).  The structured band is DENSE --
# sigma_min stays low across the whole qz^2 range, so modes are closely packed --
# and a mode is a DIP of sigma_min toward its floor.  Detect on a grid of at
# least _DETECT_PPU points per unit qz^2 (resolves the close dips regardless of
# the caller's n_scan) and pick dips with scipy ``find_peaks`` (proper plateau /
# tie handling).  This replaces the strict 3-point grid local-minimum test, whose
# registration flipped with grid alignment and the per-point sigma_min wobble
# that varies across BLAS backends -- so real modes flickered in/out of the set
# (flaky recall 9..16/16 across CI runners).
_DETECT_PPU = 8     # detection-scan points per unit qz^2 (dense-band resolution)


def _sigma_min_invpow(Geq, iters=60, tol=1e-14):
    """``sigma_min(Geq)`` by inverse-power iteration on a SPARSE LU.

    The equilibrated block ``G`` is BLOCK-CYCLIC-BIDIAGONAL (each strip couples
    only to its neighbour + the Bloch-wrap corner), so a sparse LU is O(S) and
    inverse-iterating on ``(G^H G)^-1`` (whose top eigenvalue is ``1/sigma_min^2``)
    costs O(S) per ``qz^2`` -- vs the dense ``svdvals`` at O((2(2Nx)S)^3).  This is
    the fine-y-staircase (large ``S``) speedup; ~15-37x at S>=16 (probe-measured).
    A numerically-singular ``G`` (a mode) makes ``splu`` raise -> return 0.0 (the
    correct ``sigma_min``)."""
    Gs = sp.csc_matrix(Geq)
    n = Gs.shape[0]
    try:
        lu = splu(Gs)
        luH = splu(Gs.conj().T.tocsc())
    except RuntimeError:
        return 0.0
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x /= np.linalg.norm(x)
    prev = 0.0
    for _ in range(iters):
        z = lu.solve(luH.solve(x))
        nz = np.linalg.norm(z)
        x = z / nz
        cur = 1.0 / np.sqrt(nz)
        if abs(cur - prev) < tol * cur:
            break
        prev = cur
    return cur


def _block_singvals(strips, Lx, Nx, k0, kx0, qz2, ky0, Ly):
    """All singular values (descending) of the equilibrated global block
    ``G(qz^2)``.  ``[-1]`` is ``sigma_min``; a clean rank drop ``s_k << s_{k+1}``
    among the smallest values marks a k-fold-degenerate mode."""
    wvk = _strip_modes_at(strips, Lx, Nx, k0, kx0, qz2)
    Geq, _ = _global_block_G(wvk, np.exp(1j * ky0 * Ly))
    return svdvals(Geq)


def dispersion_vec(strips, Lx, Nx, k0, kx0, qz2, ky0, Ly, solver="dense"):
    """``sigma_min(G(qz^2))`` -- zero at a 2-D vector Bloch layer mode.
    ``solver="dense"`` (default, byte-identical) uses ``svdvals``; ``"banded"``
    uses the O(S) inverse-power ``sigma_min`` (same zeros, for fine staircases)."""
    wvk = _strip_modes_at(strips, Lx, Nx, k0, kx0, qz2)
    Geq, _ = _global_block_G(wvk, np.exp(1j * ky0 * Ly))
    if solver == "banded":
        return _sigma_min_invpow(Geq)
    return svdvals(Geq)[-1]


def _strips_to_mu_xy(strips, Nx, Ly, Ny):
    """Rasterize the optional per-strip scalar ``mu_x`` (magnetic ``(eps, h, mu)``
    3-tuple strips) onto an ``(Nx, Ny)`` grid for the FD oracle.  Returns ``None``
    when every strip has ``mu == 1`` (the byte-identical legacy oracle path)."""
    if all(len(s) < 3 or not np.any(np.asarray(s[2], dtype=complex) != 1.0)
           for s in strips):
        return None
    edges = np.cumsum([0.0] + [s[1] for s in strips])
    yc = (np.arange(Ny) + 0.5) / Ny * Ly
    mu = np.ones((Nx, Ny), dtype=complex)
    for s, st in enumerate(strips):
        msk = (yc >= edges[s]) & (yc < edges[s + 1])
        mu_s = np.asarray(st[2] if len(st) > 2 else 1.0, dtype=complex)
        if mu_s.ndim == 0:
            mu_s = np.full(Nx, complex(mu_s))
        mu[:, msk] = mu_s[:, None]
    return mu


def _fd_eig_dist(strips, Lx, Nx, Ly, k0, kx0, ky0, qz2, ny):
    """Distance from a candidate ``qz2`` to the NEAREST 2-D-FD oracle eigenvalue
    (a few sparse shift-invert modes near it).  A real mode has an FD mode within
    the y-FD error (~0.1); a spurious candidate has none nearby (O(1)).  The
    independent ORACLE-ASSISTED spurious discriminator used by ``verify=True``
    (a self-contained PDE-residual check fails -- piecewise-y Gibbs noise)."""
    eps_xy = strips_to_eps_xy(strips, Lx, Nx, Ly, ny)
    mu_xy = _strips_to_mu_xy(strips, Nx, Ly, ny)   # magnetic -> mu-consistent oracle
    G, _, _ = _build_generator(eps_xy, Lx, Ly, Nx, ny, k0, kx0, ky0, mu_xy)
    Gc = G.tocsc()
    # Shift-invert at a small OFFSET from the candidate, never AT it.  A real
    # mode's FD eigenvalue sits at ``gam = i*sqrt(qz2)``, so a sigma placed
    # exactly there makes ``(Gc - sigma)`` singular -> the shift-invert LU is
    # ill-conditioned and ARPACK's Ritz values become BLAS/backend-sensitive
    # (this, not the start vector, was the cross-environment flake: v5.18.1's
    # fixed ``v0`` removed only the random-start jitter within one BLAS; the
    # near-singular solve still diverged MKL-vs-OpenBLAS, dropping real modes).
    # Offsetting sigma along the imaginary axis by ``1e-3 * sqrt(qz2)`` (<< the
    # mode spacing, so the target stays the nearest eigenvalue) conditions the
    # solve deterministically; ``k=6`` gives ARPACK margin to converge the
    # target.  The test's own oracle (_oracle_band) already uses a band-CENTRE
    # sigma for the same reason.
    root = np.sqrt(complex(qz2))
    sigma = 1j * root * (1.0 + 1e-3)
    v0 = np.random.default_rng(0).standard_normal(Gc.shape[0]).astype(complex)
    gam, _ = eigs(Gc, k=6, sigma=sigma, v0=v0)
    return float(np.min(np.abs((-(gam ** 2)).real - qz2)))


def layer_vector_modes(strips, Lx, Nx, Ly, k0, qz2_range, *, kx0=0.0, ky0=0.0,
                       n_scan=400, tol=5e-2, ratio_tol=1e-3, merge_rtol=3e-3,
                       return_multiplicity=False, verify=False, verify_ny=None,
                       verify_tol=1.0, solver="dense"):
    """Full-vector 2-D Bloch modes ``qz^2`` of a y-strip-sectioned layer (the
    vector analog of ``eme_2d.layer_modes``).

    ``strips`` : list of ``(eps_x_array, height)`` (heights sum to ``Ly``).
    Returns the propagation eigenvalues ``qz^2`` (descending) in ``qz2_range``.

    Mode condition: ``G(qz^2)`` (``_global_block_G``) is SINGULAR at a mode.
    Because at finite ``Nx`` a real mode's ``sigma_min`` floors at the x-FD error
    (so a depth threshold alone cannot separate real modes from spurious dips),
    acceptance uses a degeneracy-agnostic RANK-DROP / GAP test: a genuine k-fold
    mode has a sharp jump ``s_k << s_{k+1}`` among the smallest singular values,
    while a spurious dip decays smoothly.  Detection: ``sigma_min(qz^2)`` is
    sampled on a grid dense enough to resolve the closely-packed structured band
    (at least ``_DETECT_PPU`` points per unit ``qz^2``, independent of ``n_scan``)
    and dips are picked with ``scipy.signal.find_peaks`` (plateau/tie-robust --
    the earlier strict 3-point grid-local-minimum test flickered with grid
    alignment and the per-point ``sigma_min`` wobble that varies across LAPACK
    backends, giving flaky mode counts).  Each dip is Brent-refined and kept iff
    the smallest gap among the bottom ``_NULL_KMAX`` singular values is
    ``< ratio_tol``; deduped within ``merge_rtol``.  (``n_scan`` is retained for
    API compatibility and as a floor on the detection density.)

    ``return_multiplicity`` : also return the per-mode degeneracy ``k`` (the
    rank-drop location -- 1 non-degenerate, 2 a TE/TM pair, ...).
    ``verify`` : drop any accepted candidate with no 2-D-FD oracle eigenvalue
    within ``verify_tol`` (an FD solve at ``verify_ny`` rows; default ~6*nstrips).
    This is an ORACLE-ASSISTED filter that removes the ~1 residual spurious
    candidate (recall 16/16, spurious 0 on the reference cell); default-off, so the
    legacy path is byte-identical.  (That ~1 spurious is a TRUE ``det(G)`` ghost-zero
    with a Maxwell-consistent field, NOT a numerical artifact -- probe-tested -- so no
    function of ``G`` alone separates it and the FD oracle is required.)

    VALIDATED REGIME -- STRUCTURED layers (TE/TM split): recall 16/16 on the
    reference 2-strip cell at Nx=20.  KNOWN LIMITATION: HIGH-degeneracy layers
    (e.g. a uniform slab, ``+-ky x 2-pol`` 4-fold-degenerate dense clusters) give
    unreliable mode-finding -- use the oracle / the analytic dispersion there.  The
    root cause was probe-tested: it is the O(h^2) x-FD error on the high-``|kx|``
    strip modes flooding ``sigma_min``'s floor, NOT a conditioning wall in ``G``'s
    degenerate null-space (a symmetry-reduced ``G`` inherits the identical floor), so
    refining the FINDER cannot help -- raise ``Nx`` or use the oracle.
    """
    if strips and is_jax_array(strips[0][0]):
        raise NotImplementedError(
            "layer_vector_modes: the sigma_min root-scan (real-axis scan + Brent "
            "root-find) is NOT differentiable.  For differentiable 2-D modes use "
            "ref_2d_modes_vector with a JAX eps (the eig-based FD-oracle twin), or "
            "solve on NumPy and apply implicit differentiation at the converged "
            "mode.")
    hsum = float(np.real(sum(s[1] for s in strips)))
    if abs(hsum - Ly) > 1e-9 * max(abs(Ly), 1.0):
        raise ValueError(
            f"layer_vector_modes: strip heights sum to {hsum:g} but Ly = {Ly:g}; "
            "the cell physics uses sum(h) while the Bloch wrap phase uses Ly, so "
            "they must agree (documented contract: heights sum to Ly).")
    lo, hi = qz2_range
    vny = verify_ny if verify_ny is not None else max(48, 6 * len(strips))

    def f(q):
        return dispersion_vec(strips, Lx, Nx, k0, kx0, q, ky0, Ly, solver)

    found = []                                        # (qz2, multiplicity)

    def _refine_accept(bracket):
        """Brent-refine a candidate within ``bracket`` and, iff it is a true
        (rank-drop) mode, append ``(qz2, multiplicity)`` to ``found``."""
        r = minimize_scalar(f, bracket=bracket, method="brent",
                            options={"xtol": 1e-7})
        s = _block_singvals(strips, Lx, Nx, k0, kx0, r.x, ky0, Ly)
        # clean rank-drop test, degeneracy-agnostic: a genuine k-fold mode has a
        # sharp GAP s_k << s_{k+1} somewhere in the smallest few singular values
        # (k=1 non-degenerate; k=2 TE/TM pair; k=4 a uniform layer's +-ky x 2-pol;
        # ...), while a spurious dip decays SMOOTHLY (no gap).
        sa = s[::-1]                                  # ascending
        gaps = sa[:_NULL_KMAX] / sa[1:_NULL_KMAX + 1]
        if s[-1] < tol and float(gaps.min()) < ratio_tol:
            if verify and _fd_eig_dist(
                    strips, Lx, Nx, Ly, k0, kx0, ky0, r.x, vny) > verify_tol:
                return                                # no FD mode nearby -> spurious
            found.append((r.x, int(np.argmin(gaps)) + 1))

    # Backend-invariant candidate detection.  The old strict 3-point grid
    # local-minimum test (d[i]<d[i-1] and d[i]<d[i+1] and d[i]<tol) flakily missed
    # a mode whenever its dip fell near-symmetrically between two grid points, or
    # two close dips in this DENSE band merged at the coarse resolution -- and
    # WHICH dips registered depended on per-point sigma_min values that wobble
    # across LAPACK backends (recall swung 9..16/16 across CI runners).  Fix:
    # detect on a grid dense enough to resolve the packed band (>= _DETECT_PPU
    # points per unit qz^2), and pick dips with find_peaks (plateau/tie-robust)
    # rather than the strict test.  The rank-drop acceptance (with the tightened
    # ratio_tol -- real modes rank-drop ~1e-6 while a spurious det(G) ghost-zero
    # only ~5e-3, a 2-3 decade margin) filters spurious; dedup is unchanged.
    from scipy.signal import find_peaks
    dscan = max(n_scan, int(_DETECT_PPU * (hi - lo)) + 1)
    grid = np.linspace(lo, hi, dscan)
    d = np.array([f(q) for q in grid])
    peaks, _ = find_peaks(-d, height=-tol)
    for i in peaks:
        _refine_accept((grid[i - 1], grid[i], grid[i + 1]))
    found.sort(key=lambda t: -t[0])
    out = []
    for q, m in found:
        if not out or abs(out[-1][0] - q) > merge_rtol * max(abs(q), 1.0):
            out.append((q, m))
    qz2 = np.array([q for q, _ in out])
    if return_multiplicity:
        return qz2, np.array([m for _, m in out], dtype=int)
    return qz2


def mode_field_vec(strips, Lx, Nx, Ly, k0, qz2, ky0, Ny, *, kx0=0.0):
    """Reconstruct the vector mode field at a layer mode ``qz2`` -- the per-strip
    modal amplitudes are the null vector of the same global block ``G`` the
    mode-finder uses.  Returns ``(Ex, Ez, sigma)`` with the tangential-E components
    on an ``(Nx, Ny)`` grid (cell-centre y, matching the oracle) and
    ``sigma = sigma_min(G)`` (small confirms a true mode)."""
    wvk = _strip_modes_at(strips, Lx, Nx, k0, kx0, qz2)
    Geq, cn = _global_block_G(wvk, np.exp(1j * ky0 * Ly))
    s = np.linalg.svd(Geq, compute_uv=True)
    c = s[2][-1].conj() / cn                      # null vector of the UNscaled G
    sigma = s[1][-1]
    S = len(strips)
    M = 2 * Nx
    # strips may be (eps_x, h) OR magnetic (eps_x, h, mu_x) -- like _strip_modes_at
    edges = np.concatenate([[0.0], np.cumsum([s[1] for s in strips])])
    yc = (np.arange(Ny) + 0.5) / Ny * Ly
    Ex = np.zeros((Nx, Ny), dtype=complex)
    Ez = np.zeros((Nx, Ny), dtype=complex)
    for j, y in enumerate(yc):
        si = min(int(np.searchsorted(edges, y, side="right")) - 1, S - 1)
        ky, W, _, _ = wvk[si]
        a = c[2 * M * si:2 * M * si + M]
        b = c[2 * M * si + M:2 * M * (si + 1)]
        eta = y - edges[si]
        estate = W @ (a * np.exp(1j * ky * eta) + b * np.exp(-1j * ky * eta))
        Ex[:, j] = estate[:Nx]
        Ez[:, j] = estate[Nx:]
    return Ex, Ez, sigma


# =========================================================================== #
#  Seeded Beyn refiner -- COMPLEX qz^2 modes (lossy / leaky), off the real axis #
# =========================================================================== #
def _beyn_moments(getG, n, center, R, ar, Nq, L, rng):
    """Beyn contour moments ``A0 = (1/2pi i) oint G(z)^-1 V dz`` and
    ``A1 = (1/2pi i) oint z G(z)^-1 V dz`` over an ellipse (semi-axes ``R``,
    ``R*ar``) centred at ``center``, ``Nq`` quadrature points, ``L`` random probe
    columns ``V``."""
    V = rng.standard_normal((n, L)) + 1j * rng.standard_normal((n, L))
    A0 = np.zeros((n, L), dtype=complex)
    A1 = np.zeros((n, L), dtype=complex)
    for j in range(Nq):
        th = 2 * np.pi * (j + 0.5) / Nq
        zt = center + R * (np.cos(th) + 1j * ar * np.sin(th))
        dz = R * (-np.sin(th) + 1j * ar * np.cos(th))
        Ti = np.linalg.solve(getG(zt), V)
        w = dz / Nq
        A0 += w * Ti
        A1 += (zt * w) * Ti
    return A0 / 1j, A1 / 1j


def beyn_refine_complex(strips, Lx, Nx, k0, ky0, Ly, seed, *, kx0=0.0,
                        R=0.9, ar=0.22, Nq=160, L=12, seed_rng=0, s0_accept=None):
    """Refine an approximate COMPLEX ``qz^2`` ``seed`` to the EME's own complex
    layer mode by a SEEDED Beyn contour integral of ``G(z)^-1`` -- reaching
    LOSSY / LEAKY / gain modes that the real-axis ``sigma_min`` scan structurally
    cannot (they sit off the real ``qz^2`` axis).

    Returns ``(mode, s0)``: ``mode`` is the residue-weighted centroid of the Beyn
    root cloud inside the contour (``complex``), ``s0`` the leading singular value
    of ``A0`` (the acceptance score).  If ``s0_accept`` is given and ``s0 <
    s0_accept`` (an empty contour -- no mode), returns ``(None, s0)``.

    SCOPE (honest): a SEEDED refiner, NOT an autonomous discoverer -- supply a
    coarse complex ``qz^2`` (e.g. ``ref_2d_modes_vector(return_complex=True)`` at
    a small ``Ny``, or an analytic guess).  One mode per contour; accuracy is
    x-FD-floored (~1e-2..1e-3, the satellite-cloud limit, NOT pole-sharp);
    weak-to-moderate loss.  Autonomous full-band complex discovery + clean
    multiplicity from the ``A0`` rank is deferred (the near-pole satellite cloud).
    """
    n = 2 * (2 * Nx) * len(strips)
    t = np.exp(1j * ky0 * Ly)
    seed = complex(seed)

    def getG(z):
        return _global_block_G(_strip_modes_at(strips, Lx, Nx, k0, kx0, z), t)[0]

    rng = np.random.default_rng(seed_rng)
    A0, A1 = _beyn_moments(getG, n, seed, R, ar, Nq, L, rng)
    U, s, Vh = np.linalg.svd(A0, full_matrices=False)
    if s0_accept is not None and s[0] < s0_accept:
        return None, s[0]
    kk = max(1, int(np.sum(s / s[0] > 1e-2)))            # numerical rank of A0
    B = U[:, :kk].conj().T @ A1 @ Vh[:kk].conj().T / s[:kk][None, :]
    keep, wts = [], []
    for lam in np.linalg.eigvals(B):                     # roots = Beyn eigenvalues
        u, v = (lam.real - seed.real) / R, (lam.imag - seed.imag) / (R * ar)
        if u * u + v * v <= 1.2:                         # inside the contour
            keep.append(lam)
            wts.append(1.0 / max(svdvals(getG(lam))[-1], 1e-12))   # residue weight
    if not keep:
        return None, s[0]
    keep, wts = np.array(keep), np.array(wts)
    return complex(np.sum(keep * wts) / np.sum(wts)), s[0]


def layer_vector_modes_complex(strips, Lx, Nx, Ly, k0, seeds, *, kx0=0.0, ky0=0.0,
                               **kw):
    """Refine a list of approximate COMPLEX ``qz^2`` ``seeds`` (e.g. from
    ``ref_2d_modes_vector(return_complex=True)`` at a coarse ``Ny``) to the EME's
    own complex layer modes via the seeded Beyn refiner -- the lossy / leaky-mode
    path the real-axis ``layer_vector_modes`` cannot reach.  Returns the array of
    complex ``qz^2``."""
    out = []
    for sd in seeds:
        m, _ = beyn_refine_complex(strips, Lx, Nx, k0, ky0, Ly, sd, kx0=kx0, **kw)
        if m is not None:
            out.append(m)
    return np.array(out)


# =========================================================================== #
#  Direct Yee-staggered 2-D VECTOR FD reference (the validation oracle)        #
# =========================================================================== #
def _fd_fwd(N, h, p):
    """Forward difference (f[i+1]-f[i])/h, periodic, Bloch wrap phase p."""
    rows = np.arange(N)
    I = np.concatenate([rows, rows])
    J = np.concatenate([rows, (rows + 1) % N])
    d = np.concatenate([-np.ones(N), np.ones(N)]).astype(complex) / h
    d[N + (N - 1)] *= p
    return sp.csr_matrix((d, (I, J)), shape=(N, N), dtype=complex)


def _fd_bwd(N, h, p):
    """Backward difference = -(forward)^H exactly (Yee-adjoint pairing)."""
    return -_fd_fwd(N, h, p).conj().T.tocsr()


def _yee_ops(Nx, Ny, Lx, Ly, kx0, ky0):
    hx, hy = Lx / Nx, Ly / Ny
    px, py = np.exp(1j * kx0 * Lx), np.exp(1j * ky0 * Ly)
    Ix = sp.identity(Nx, dtype=complex, format="csr")
    Iy = sp.identity(Ny, dtype=complex, format="csr")
    DxF = sp.kron(_fd_fwd(Nx, hx, px), Iy, format="csr")
    DxB = sp.kron(_fd_bwd(Nx, hx, px), Iy, format="csr")
    DyF = sp.kron(Ix, _fd_fwd(Ny, hy, py), format="csr")
    DyB = sp.kron(Ix, _fd_bwd(Ny, hy, py), format="csr")
    return DxF, DxB, DyF, DyB


def _build_generator(eps_xy, Lx, Ly, Nx, Ny, k0, kx0, ky0, mu_xy=None):
    """Sparse 4N x 4N Yee generator G (gamma = i qz) for the transverse state
    [Ex, Ey, hx, hy] (Ez, hz eliminated).

    ``eps_xy`` is either an ``(Nx, Ny)`` scalar isotropic grid (the legacy path,
    byte-identical) or an ``(Nx, Ny, 3, 3)`` per-node permittivity TENSOR
    (DIAGONAL + ``exz``/``ezx``, lossless) -- the latter routes to
    :func:`_build_generator_tensor` and reduces to this body BYTE-EXACTLY for an
    isotropic tensor.  ``mu_xy`` is an optional ``(Nx, Ny)`` scalar permeability
    (default ``None`` -> ``mu = 1``, byte-identical); it enters the ``hz``
    elimination (``1/mu``) and the ``i k0 mu`` source terms.  The returned ``eps``
    is the flattened scalar or the ``(N, 3, 3)`` tensor (for the divergence check)."""
    arr = np.asarray(eps_xy, dtype=complex)
    if arr.ndim == 4:                       # (Nx, Ny, 3, 3) anisotropic branch
        return _build_generator_tensor(arr, Lx, Ly, Nx, Ny, k0, kx0, ky0)
    N = Nx * Ny
    DxF, DxB, DyF, DyB = _yee_ops(Nx, Ny, Lx, Ly, kx0, ky0)
    eps = arr.reshape(N)
    inv = sp.diags(1.0 / (k0 * eps), format="csr")
    Eps = sp.diags(eps, format="csr")
    Z = sp.csr_matrix((N, N), dtype=complex)
    if mu_xy is None:
        Mu = sp.identity(N, dtype=complex, format="csr")
        invmu = sp.diags(np.full(N, 1.0 / k0), format="csr")
    else:
        mu = np.asarray(mu_xy, dtype=complex).reshape(N)
        Mu = sp.diags(mu, format="csr")
        invmu = sp.diags(1.0 / (k0 * mu), format="csr")
    Ez_hx = -1j * inv @ DyF
    Ez_hy = 1j * inv @ DxF
    hz_Ex = 1j * invmu @ DyB                    # mu=1 -> (1j/k0) DyB
    hz_Ey = -1j * invmu @ DxB                   # mu=1 -> -(1j/k0) DxB
    rEx = [Z, Z, DxB @ Ez_hx, 1j * k0 * Mu + DxB @ Ez_hy]
    rEy = [Z, Z, -1j * k0 * Mu + DyB @ Ez_hx, DyB @ Ez_hy]
    rhx = [DxF @ hz_Ex, -1j * k0 * Eps + DxF @ hz_Ey, Z, Z]
    rhy = [1j * k0 * Eps + DyF @ hz_Ex, DyF @ hz_Ey, Z, Z]
    G = sp.bmat([rEx, rEy, rhx, rhy], format="csc")
    return G, (DxF, DxB, DyF, DyB), eps


def _build_generator_tensor(eps_xy, Lx, Ly, Nx, Ny, k0, kx0, ky0):
    """Sparse 4N x 4N Yee oracle generator (gamma = i qz, state [Ex, Ey, hx, hy],
    Ez/hz eliminated) for a per-node ``(Nx, Ny, 3, 3)`` permittivity TENSOR,
    DIAGONAL + ``exz``/``ezx`` (lossless).  Raises ``NotImplementedError`` for
    ``eyz``/``ezy``.  The tensor analog of :func:`_build_generator`; reduces to it
    byte-exactly for an isotropic tensor (validated, gate 4a).  ``Ez`` is
    eliminated from ``Dz = 0`` (``ezx Ex + ezz Ez = -(curl h)_z / (i k0)``).

    VALIDATED REGIME -- DIAGONAL tensors: the oracle qz^2 converges to the planar
    Berreman analytic at O(h^2) (gate 4b, ratio ~4), and a structured diagonal
    cell's strip-generator EME band converges to it (gate 4d).  KNOWN LIMITATION
    -- the ``exz``/``ezx`` off-diagonal terms carry a residual Yee-stagger error
    (``Ex`` and the reconstructed ``Ez`` live on different half-nodes, so the
    constitutive product ``exx Ex + exz Ez`` is location-mismatched): for a
    UNIFORM exz cell the oracle qz^2 sits ~0.3 off the Berreman value and does NOT
    converge with ``Nx``.  This needs a half-node averaging operator (the FD
    analog of correct Fourier factorization) -- DEFERRED.  The ``exz`` STRIP
    generator itself is rigorously validated WITHOUT this oracle, via the
    role-swapped planar Berreman (gate 2, byte-exact)."""
    N = Nx * Ny
    e = np.asarray(eps_xy, dtype=complex).reshape(N, 3, 3)
    tol = 1e-12 * max(1.0, float(np.max(np.abs(e))))
    if np.max(np.abs(e[:, 1, 2])) > tol or np.max(np.abs(e[:, 2, 1])) > tol:
        raise NotImplementedError(
            "_build_generator: out-of-plane eyz/ezy (yz) coupling is not "
            "implemented in the 2-D-FD oracle (diagonal + exz/ezx only).")
    DxF, DxB, DyF, DyB = _yee_ops(Nx, Ny, Lx, Ly, kx0, ky0)
    exx = sp.diags(e[:, 0, 0], format="csr")
    exz = sp.diags(e[:, 0, 2], format="csr")
    ezx = sp.diags(e[:, 2, 0], format="csr")
    eyy = sp.diags(e[:, 1, 1], format="csr")
    ezzi = sp.diags(1.0 / e[:, 2, 2], format="csr")
    I = sp.identity(N, dtype=complex, format="csr")
    Z = sp.csr_matrix((N, N), dtype=complex)
    ik0 = 1j * k0
    # Ez = (1/ezz)[ -(DxF hy - DyF hx)/(i k0) - ezx Ex ]   (forward inner)
    Ez_hx = ezzi @ (DyF / ik0)
    Ez_hy = ezzi @ (-DxF / ik0)
    Ez_Ex = ezzi @ (-ezx)
    # hz = (DxB Ey - DyB Ex)/(i k0)   (backward inner)
    hz_Ex = -DyB / ik0
    hz_Ey = DxB / ik0
    rEx = [DxB @ Ez_Ex, Z, DxB @ Ez_hx, ik0 * I + DxB @ Ez_hy]
    rEy = [DyB @ Ez_Ex, Z, -ik0 * I + DyB @ Ez_hx, DyB @ Ez_hy]
    rhx = [DxF @ hz_Ex, -ik0 * eyy + DxF @ hz_Ey, Z, Z]
    rhy = [ik0 * exx + ik0 * exz @ Ez_Ex + DyF @ hz_Ex,
           DyF @ hz_Ey, ik0 * exz @ Ez_hx, ik0 * exz @ Ez_hy]
    G = sp.bmat([rEx, rEy, rhx, rhy], format="csc")
    return G, (DxF, DxB, DyF, DyB), e


def ref_2d_modes_vector(eps_xy, Lx, Ly, Nx, Ny, k0, kx0=0.0, ky0=0.0,
                        return_vecs=False, k=None, sigma=None,
                        return_complex=False, mu_xy=None):
    """Full-vectorial 2-D Bloch modes ``qz^2`` of a z-invariant
    ``eps(x, y)`` by a direct Yee-staggered finite-difference Maxwell solve -- the
    independent oracle (vector analog of ``eme_2d.ref_2d_modes``).  ``eps_xy`` is
    an ``(Nx, Ny)`` scalar isotropic grid OR an ``(Nx, Ny, 3, 3)`` per-node
    permittivity TENSOR (DIAGONAL + ``exz``/``ezx``, lossless).

    Returns ``qz^2`` (descending).  With ``return_vecs`` also returns
    ``(fields, reldiv)`` -- ``fields[c]`` (c in [Ex,Ey,hx,hy]) reshaped
    ``(Nx, Ny)`` per mode, and ``reldiv = |div(eps E)|/(k0|E|)`` per mode
    (physical ~1e-12, spurious O(1); the Yee operator is already spurious-free).

    By default a DENSE ``eig`` returns the full ``4 Nx Ny`` spectrum (``+-qz`` per
    polarization, collapsed by ``qz^2 = -gamma^2``) -- needed for mode-count /
    degeneracy checks.  Pass ``k`` to instead use a SPARSE shift-invert
    (``scipy.sparse.linalg.eigs``) that returns the ``k`` physical modes nearest
    ``sigma`` (default: the band centre ``~ i k0 sqrt(max eps) * 0.78``) --
    O(100x) faster for the top / in-band physical modes, and it returns the
    DISTINCT modes directly (only the ``+i gamma`` branch, no ``+-qz`` doubling).
    """
    if any(is_jax_array(a) for a in (eps_xy, k0, kx0, ky0)):
        # differentiable twin (eig-based; see ._jax_modes).  Scalar isotropic eps,
        # dense spectrum, qz^2 only -- the JAX guards reject tensor / sparse /
        # return_vecs with a clear message.
        from ._jax_modes import _ref_2d_modes_vector_jax
        return _ref_2d_modes_vector_jax(
            eps_xy, Lx, Ly, Nx, Ny, k0, kx0, ky0, return_vecs=return_vecs,
            k=k, sigma=sigma, return_complex=return_complex, mu_xy=mu_xy)
    N = Nx * Ny
    G, (DxF, DxB, DyF, DyB), eps = _build_generator(
        eps_xy, Lx, Ly, Nx, Ny, k0, kx0, ky0, mu_xy)
    is_tensor = eps.ndim == 3                            # (N, 3, 3) vs scalar (N,)
    if k is None:
        gam, V = eig(G.toarray())                    # dense full spectrum
    else:
        if sigma is None:
            sigma = 1j * k0 * np.sqrt(float(np.max(eps_xy.real))) * 0.78
        # fixed start vector -> deterministic ARPACK (eigenvalues are v0-independent;
        # this only removes the random-start run-to-run jitter, not the result)
        v0 = np.random.default_rng(0).standard_normal(4 * N)
        gam, V = eigs(G.tocsc(), k=min(k, 4 * N - 2), sigma=sigma, v0=v0)
    qz2 = -(gam ** 2)
    order = np.argsort(qz2.real)[::-1]
    qz2, gam, V = qz2[order], gam[order], V[:, order]
    out_qz2 = qz2.copy() if return_complex else qz2.real.copy()  # lossy -> complex
    if not return_vecs:
        return out_qz2
    reldiv = np.empty(V.shape[1])
    for m in range(V.shape[1]):
        Ex, Ey = V[0:N, m], V[N:2 * N, m]
        hx, hy = V[2 * N:3 * N, m], V[3 * N:4 * N, m]
        qz = -1j * gam[m]
        if is_tensor:
            # diagonal + exz/ezx: Dz = ezx Ex + ezz Ez = -(curl h)_z/(i k0)
            ezx, ezz = eps[:, 2, 0], eps[:, 2, 2]
            Ez = ((1j / k0) * (DxF @ hy - DyF @ hx) - ezx * Ex) / ezz
            Dx = eps[:, 0, 0] * Ex + eps[:, 0, 2] * Ez
            Dy = eps[:, 1, 1] * Ey
            Dz = ezx * Ex + ezz * Ez
            divD = DxF @ Dx + DyF @ Dy + 1j * qz * Dz
        else:
            Ez = (1j / (k0 * eps)) * (DxF @ hy - DyF @ hx)
            divD = DxF @ (eps * Ex) + DyF @ (eps * Ey) + 1j * qz * (eps * Ez)
        En = np.sqrt(np.sum(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2))
        reldiv[m] = float(np.linalg.norm(divD) / (k0 * En + 1e-300))
    fields = np.stack([
        V[0:N].reshape(Nx, Ny, -1), V[N:2 * N].reshape(Nx, Ny, -1),
        V[2 * N:3 * N].reshape(Nx, Ny, -1), V[3 * N:4 * N].reshape(Nx, Ny, -1),
    ], axis=0)
    return out_qz2, fields, reldiv


def strips_to_eps_xy(strips, Lx, Nx, Ly, Ny):
    """Rasterize a strip list onto an ``(Nx, Ny)`` grid for the 2-D reference.

    Each strip is ``(eps_x, h)`` or magnetic ``(eps_x, h, mu_x)`` (the ``mu_x`` is
    ignored here -- see ``_strips_to_mu_xy`` for the ``mu`` grid).  ``eps_x`` is a
    scalar isotropic ``(Nx,)`` profile (-> ``(Nx, Ny)`` output) OR a per-node
    ``(Nx, 3, 3)`` permittivity tensor (-> ``(Nx, Ny, 3, 3)`` output, the oracle's
    anisotropic input)."""
    edges = np.cumsum([0.0] + [s[1] for s in strips])
    yc = (np.arange(Ny) + 0.5) / Ny * Ly
    ex0 = np.asarray(strips[0][0], dtype=complex)
    is_tensor = ex0.ndim == 3                       # (Nx, 3, 3)
    eps = np.zeros((Nx, Ny, 3, 3) if is_tensor else (Nx, Ny), dtype=complex)
    for s, st in enumerate(strips):
        msk = (yc >= edges[s]) & (yc < edges[s + 1])
        exa = np.asarray(st[0], dtype=complex)
        if is_tensor:
            eps[:, msk] = exa[:, None, :, :]
        else:
            eps[:, msk] = exa[:, None]
    return eps


# =========================================================================== #
#  Arbitrary geometry -> y-strips (the lateral analog of RCWA z-staircasing)    #
# =========================================================================== #
def eps_xy_to_strips(eps, Nx, S, Lx, Ly):
    """Rasterize an arbitrary ``eps(x, y)`` into ``S`` equal-height y-strips for
    ``layer_vector_modes`` / ``layer_modes`` -- so the EME accepts an arbitrary
    cell, not only a hand-built strip list.

    ``eps`` is either a callable ``eps(x, y)`` (sampled at cell centres) or an
    ``(Nx_in, Ny_in)`` grid (nearest-resampled).  Returns ``[(eps_x, Ly/S), ...]``
    with ``eps_x`` complex on the cell-centre x grid ``(arange(Nx)+0.5)/Nx*Lx``.

    This is the exact lateral analog of how RCWA staircases ``z``: the accuracy is
    **1st-order staircase-limited**, so refine ``S`` for slanted / curved features
    (a y-uniform piecewise structure is exact at the matching ``S``).  A
    hand-built ``strips`` list bypasses this wrapper and is unaffected.
    """
    xc = (np.arange(Nx) + 0.5) / Nx * Lx
    yc = (np.arange(S) + 0.5) / S * Ly
    h = Ly / S
    if callable(eps):
        return [(np.array([eps(x, y) for x in xc], dtype=complex), h) for y in yc]
    arr = np.asarray(eps, dtype=complex)
    nxi, nyi = arr.shape
    ix = np.minimum((xc / Lx * nxi).astype(int), nxi - 1)
    jy = np.minimum((yc / Ly * nyi).astype(int), nyi - 1)
    return [(arr[ix, j], h) for j in jy]
