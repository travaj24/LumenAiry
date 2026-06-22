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

SCOPE: isotropic (or in-plane diagonal) lossless eps; a mode / band-structure
solver (NOT diffraction efficiencies -- see ``eme_diffraction.py`` for why the
lateral-cascade modes are the wrong basis for efficiency truncation).  Conventions:
``exp(+i qz z)``, ``exp(+i ky y)``; normalized Maxwell ``curl E = i k0 h``,
``curl h = -i k0 eps E`` (``h = Z0 H``); real eps (lossless).

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
from scipy.sparse.linalg import eigs


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


def _strip_vector_generator(eps_x, Lx, Nx, k0, kx0, qz):
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
        return _strip_vector_generator_tensor(eps, Lx, Nx, k0, kx0, qz)
    h = Lx / Nx
    Df, Db = _strip_yee_diffs(Nx, h, kx0, Lx)
    I = np.eye(Nx, dtype=complex)
    Z = np.zeros((Nx, Nx), dtype=complex)
    Ei = np.diag(1.0 / eps)         # 1/eps(x) pointwise
    Eps = np.diag(eps)
    # dEx/dy = Db[-(qz/k0)(1/eps)Hx - (i/k0)(1/eps)Df Hz] - i k0 Hz
    A_Ex_Hx = Db @ (-(qz / k0) * Ei)
    A_Ex_Hz = Db @ (-(1j / k0) * (Ei @ Df)) - 1j * k0 * I
    # dEz/dy = i k0 Hx - i(qz^2/k0)(1/eps)Hx + (qz/k0)(1/eps)Df Hz
    A_Ez_Hx = 1j * k0 * I - 1j * (qz ** 2 / k0) * Ei
    A_Ez_Hz = (qz / k0) * (Ei @ Df)
    # dHx/dy = Df[(qz/k0)Ex + (i/k0)Db Ez] + i k0 eps Ez
    A_Hx_Ex = Df @ ((qz / k0) * I)
    A_Hx_Ez = Df @ ((1j / k0) * Db) + 1j * k0 * Eps
    # dHz/dy = -i k0 eps Ex + i(qz^2/k0)Ex - (qz/k0)Db Ez
    A_Hz_Ex = -1j * k0 * Eps + 1j * (qz ** 2 / k0) * I
    A_Hz_Ez = -(qz / k0) * Db
    return np.block([
        [Z, Z, A_Ex_Hx, A_Ex_Hz],
        [Z, Z, A_Ez_Hx, A_Ez_Hz],
        [A_Hx_Ex, A_Hx_Ez, Z, Z],
        [A_Hz_Ex, A_Hz_Ez, Z, Z],
    ])


def _strip_vector_generator_tensor(eps_t, Lx, Nx, k0, kx0, qz):
    """ANISOTROPIC strip generator (the tensor branch of
    :func:`_strip_vector_generator`).

    ``eps_t`` is a per-node ``(Nx, 3, 3)`` permittivity tensor.  SCOPE: DIAGONAL
    entries + the out-of-plane ``exz`` / ``ezx`` (``xz``) coupling, LOSSLESS (real
    eps).  Raises ``NotImplementedError`` for ``eyz`` / ``ezy`` (the ``yz``
    coupling -- deferred; a prior hand-derivation got that block wrong and it also
    breaks the block-anti-diagonal structure ``eig(B C)`` relies on).

    Derived directly from the normalized Maxwell curl equations
    (``curl E = i k0 h``, ``curl h = -i k0 eps E``), propagating in y with
    ``Ey, Hy`` eliminated via the in-plane ``y`` curl rows.  The key role-swap is
    that ``1/eyy`` here plays the part ``1/ezz`` plays in the planar Berreman
    ``Delta`` (y is the strip's eliminated in-plane axis).  Yee placement matches
    the scalar body: E-rows take the OUTER ``Db`` / inner ``Df``; H-rows the outer
    ``Df`` / inner ``Db``.  For an isotropic tensor ``e(x) * I3`` every block
    equals the scalar generator's BYTE-FOR-BYTE (validated, gate 1).  For a
    uniform tensor strip the forward ``ky`` multiset equals the role-swapped
    planar Berreman ``Delta`` (swap eps axes ``y<->z``, ``Kx = kx/k0``,
    ``Ky = qz/k0``, ``ky/k0 = -i*gam``) -- validated to machine precision at
    ``kx0 = 0`` (gate 2); ``exz`` at ``kx0 != 0`` carries the expected O(h^2) Yee
    dispersion in its single-``Dx`` cross-terms (2nd-order convergent, gate 2c)."""
    e = np.asarray(eps_t, dtype=complex)
    if e.shape != (Nx, 3, 3):
        raise ValueError(
            "_strip_vector_generator: a tensor eps must have shape (Nx, 3, 3), "
            f"got {e.shape}.")
    tol = 1e-12 * max(1.0, float(np.max(np.abs(e))))
    if np.max(np.abs(e[:, 1, 2])) > tol or np.max(np.abs(e[:, 2, 1])) > tol:
        raise NotImplementedError(
            "_strip_vector_generator: out-of-plane eyz/ezy (yz) coupling is not "
            "implemented (deferred -- diagonal + exz/ezx only).")
    h = Lx / Nx
    Df, Db = _strip_yee_diffs(Nx, h, kx0, Lx)
    I = np.eye(Nx, dtype=complex)
    Z = np.zeros((Nx, Nx), dtype=complex)
    Eyyi = np.diag(1.0 / e[:, 1, 1])     # 1/eyy(x) -- the role-swapped 1/ezz
    Exx = np.diag(e[:, 0, 0])
    Ezz = np.diag(e[:, 2, 2])
    Exz = np.diag(e[:, 0, 2])
    Ezx = np.diag(e[:, 2, 0])
    # E-rows: outer Db, inner Df (matches the scalar Yee placement exactly).
    A_Ex_Hx = Db @ (-(qz / k0) * Eyyi)
    A_Ex_Hz = Db @ (-(1j / k0) * (Eyyi @ Df)) - 1j * k0 * I
    A_Ez_Hx = 1j * k0 * I - 1j * (qz ** 2 / k0) * Eyyi
    A_Ez_Hz = (qz / k0) * (Eyyi @ Df)
    # H-rows: outer Df, inner Db.  exz/ezx enter ONLY the E<->H blocks (Ezx in
    # dHx/dy, Exx/Exz in dHz/dy), so the E-E / H-H blocks stay zero -> the
    # block-anti-diagonal eig(B C) reduction survives (gate 3).
    A_Hx_Ex = Df @ ((qz / k0) * I) + 1j * k0 * Ezx
    A_Hx_Ez = Df @ ((1j / k0) * Db) + 1j * k0 * Ezz
    A_Hz_Ex = -1j * k0 * Exx + 1j * (qz ** 2 / k0) * I
    A_Hz_Ez = -(qz / k0) * Db - 1j * k0 * Exz
    return np.block([
        [Z, Z, A_Ex_Hx, A_Ex_Hz],
        [Z, Z, A_Ez_Hx, A_Ez_Hz],
        [A_Hx_Ex, A_Hx_Ez, Z, Z],
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


def strip_vector_modes(eps_x, Lx, Nx, k0, kx0=0.0, qz2=0.0):
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
    docstring.  ``qz = sqrt(qz2)``.
    """
    qz = np.sqrt(complex(qz2)) if qz2 != 0.0 else 0.0
    A = _strip_vector_generator(eps_x, Lx, Nx, k0, kx0, complex(qz))
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
def _strip_modes_at(strips, Lx, Nx, k0, kx0, qz2):
    """The ``(ky, W, V, h)`` vector strip modes of every strip at trial ``qz2``
    (the strip operator is qz-dependent, so this is evaluated per ``qz2``)."""
    return [(*strip_vector_modes(eps_x, Lx, Nx, k0, kx0, qz2), h)
            for eps_x, h in strips]


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


def _block_singvals(strips, Lx, Nx, k0, kx0, qz2, ky0, Ly):
    """All singular values (descending) of the equilibrated global block
    ``G(qz^2)``.  ``[-1]`` is ``sigma_min``; a clean rank drop ``s_k << s_{k+1}``
    among the smallest values marks a k-fold-degenerate mode."""
    wvk = _strip_modes_at(strips, Lx, Nx, k0, kx0, qz2)
    Geq, _ = _global_block_G(wvk, np.exp(1j * ky0 * Ly))
    return svdvals(Geq)


def dispersion_vec(strips, Lx, Nx, k0, kx0, qz2, ky0, Ly):
    """``sigma_min(G(qz^2))`` -- zero at a 2-D vector Bloch layer mode (the
    well-conditioned global block-``G`` residual, not the cascade residual)."""
    return _block_singvals(strips, Lx, Nx, k0, kx0, qz2, ky0, Ly)[-1]


def _fd_eig_dist(strips, Lx, Nx, Ly, k0, kx0, ky0, qz2, ny):
    """Distance from a candidate ``qz2`` to the NEAREST 2-D-FD oracle eigenvalue
    (a few sparse shift-invert modes near it).  A real mode has an FD mode within
    the y-FD error (~0.1); a spurious candidate has none nearby (O(1)).  The
    independent ORACLE-ASSISTED spurious discriminator used by ``verify=True``
    (a self-contained PDE-residual check fails -- piecewise-y Gibbs noise)."""
    eps_xy = strips_to_eps_xy(strips, Lx, Nx, Ly, ny)
    G, _, _ = _build_generator(eps_xy, Lx, Ly, Nx, ny, k0, kx0, ky0)
    gam, _ = eigs(G.tocsc(), k=4, sigma=1j * np.sqrt(complex(qz2)))
    return float(np.min(np.abs((-(gam ** 2)).real - qz2)))


def layer_vector_modes(strips, Lx, Nx, Ly, k0, qz2_range, *, kx0=0.0, ky0=0.0,
                       n_scan=400, tol=5e-2, ratio_tol=1e-2, merge_rtol=3e-3,
                       return_multiplicity=False, verify=False, verify_ny=None,
                       verify_tol=1.0):
    """Full-vector 2-D Bloch modes ``qz^2`` of a y-strip-sectioned layer (the
    vector analog of ``eme_2d.layer_modes``).

    ``strips`` : list of ``(eps_x_array, height)`` (heights sum to ``Ly``).
    Returns the propagation eigenvalues ``qz^2`` (descending) in ``qz2_range``.

    Mode condition: ``G(qz^2)`` (``_global_block_G``) is SINGULAR at a mode.
    Because at finite ``Nx`` a real mode's ``sigma_min`` floors at the x-FD error
    (so a depth threshold alone cannot separate real modes from spurious dips),
    acceptance uses a degeneracy-agnostic RANK-DROP / GAP test: a genuine k-fold
    mode has a sharp jump ``s_k << s_{k+1}`` among the smallest singular values,
    while a spurious dip decays smoothly.  Candidate minima of ``sigma_min`` below
    ``tol`` are Brent-refined and kept iff the smallest gap among the bottom
    ``_NULL_KMAX`` values is ``< ratio_tol``; deduped within ``merge_rtol``.

    ``return_multiplicity`` : also return the per-mode degeneracy ``k`` (the
    rank-drop location -- 1 non-degenerate, 2 a TE/TM pair, ...).
    ``verify`` : drop any accepted candidate with no 2-D-FD oracle eigenvalue
    within ``verify_tol`` (an FD solve at ``verify_ny`` rows; default ~6*nstrips).
    This is an ORACLE-ASSISTED filter that removes the ~1 residual spurious
    candidate (recall 16/16, spurious 0 on the reference cell); default-off, so the
    legacy path is byte-identical.

    VALIDATED REGIME -- STRUCTURED layers (TE/TM split): recall 16/16 on the
    reference 2-strip cell at Nx=20.  KNOWN LIMITATION: HIGH-degeneracy layers
    (e.g. a uniform slab, ``+-ky x 2-pol`` 4-fold-degenerate dense clusters) give
    unreliable mode-finding -- use the oracle / the analytic dispersion there.
    """
    lo, hi = qz2_range
    grid = np.linspace(lo, hi, n_scan)
    vny = verify_ny if verify_ny is not None else max(48, 6 * len(strips))

    def f(q):
        return dispersion_vec(strips, Lx, Nx, k0, kx0, q, ky0, Ly)

    d = np.array([f(q) for q in grid])
    found = []                                        # (qz2, multiplicity)
    for i in range(1, len(grid) - 1):
        if d[i] < d[i - 1] and d[i] < d[i + 1] and d[i] < tol:
            r = minimize_scalar(f, bracket=(grid[i - 1], grid[i], grid[i + 1]),
                                method="brent", options={"xtol": 1e-7})
            s = _block_singvals(strips, Lx, Nx, k0, kx0, r.x, ky0, Ly)
            # clean rank-drop test, degeneracy-agnostic: a genuine k-fold mode has
            # a sharp GAP s_k << s_{k+1} somewhere in the smallest few singular
            # values (k=1 non-degenerate; k=2 TE/TM pair; k=4 a uniform layer's
            # +-ky x 2-pol; ...), while a spurious dip decays SMOOTHLY (no gap).
            sa = s[::-1]                                  # ascending
            gaps = sa[:_NULL_KMAX] / sa[1:_NULL_KMAX + 1]
            if s[-1] < tol and float(gaps.min()) < ratio_tol:
                if verify and _fd_eig_dist(
                        strips, Lx, Nx, Ly, k0, kx0, ky0, r.x, vny) > verify_tol:
                    continue                              # no FD mode nearby -> spurious
                found.append((r.x, int(np.argmin(gaps)) + 1))
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
    edges = np.concatenate([[0.0], np.cumsum([h for _, h in strips])])
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


def _build_generator(eps_xy, Lx, Ly, Nx, Ny, k0, kx0, ky0):
    """Sparse 4N x 4N Yee generator G (gamma = i qz) for the transverse state
    [Ex, Ey, hx, hy] (Ez, hz eliminated).

    ``eps_xy`` is either an ``(Nx, Ny)`` scalar isotropic grid (the legacy path,
    byte-identical) or an ``(Nx, Ny, 3, 3)`` per-node permittivity TENSOR
    (DIAGONAL + ``exz``/``ezx``, lossless) -- the latter routes to
    :func:`_build_generator_tensor` and reduces to this body BYTE-EXACTLY for an
    isotropic tensor.  The returned ``eps`` is the flattened scalar (isotropic)
    or the ``(N, 3, 3)`` tensor (so the divergence check in
    ``ref_2d_modes_vector`` can use the full tensor)."""
    arr = np.asarray(eps_xy, dtype=complex)
    if arr.ndim == 4:                       # (Nx, Ny, 3, 3) anisotropic branch
        return _build_generator_tensor(arr, Lx, Ly, Nx, Ny, k0, kx0, ky0)
    N = Nx * Ny
    DxF, DxB, DyF, DyB = _yee_ops(Nx, Ny, Lx, Ly, kx0, ky0)
    eps = arr.reshape(N)
    inv = sp.diags(1.0 / (k0 * eps), format="csr")
    Eps = sp.diags(eps, format="csr")
    I = sp.identity(N, dtype=complex, format="csr")
    Z = sp.csr_matrix((N, N), dtype=complex)
    Ez_hx = -1j * inv @ DyF
    Ez_hy = 1j * inv @ DxF
    hz_Ex = (1j / k0) * DyB
    hz_Ey = -(1j / k0) * DxB
    rEx = [Z, Z, DxB @ Ez_hx, 1j * k0 * I + DxB @ Ez_hy]
    rEy = [Z, Z, -1j * k0 * I + DyB @ Ez_hx, DyB @ Ez_hy]
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
                        return_vecs=False, k=None, sigma=None):
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
    N = Nx * Ny
    G, (DxF, DxB, DyF, DyB), eps = _build_generator(
        eps_xy, Lx, Ly, Nx, Ny, k0, kx0, ky0)
    is_tensor = eps.ndim == 2 and eps.shape[1] == 3      # (N, 3, 3)
    if k is None:
        gam, V = eig(G.toarray())                    # dense full spectrum
    else:
        if sigma is None:
            sigma = 1j * k0 * np.sqrt(float(np.max(eps_xy.real))) * 0.78
        gam, V = eigs(G.tocsc(), k=min(k, 4 * N - 2), sigma=sigma)
    qz2 = -(gam ** 2)
    order = np.argsort(qz2.real)[::-1]
    qz2, gam, V = qz2[order], gam[order], V[:, order]
    if not return_vecs:
        return qz2.real.copy()
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
    return qz2.real.copy(), fields, reldiv


def strips_to_eps_xy(strips, Lx, Nx, Ly, Ny):
    """Rasterize a strip list onto an ``(Nx, Ny)`` grid for the 2-D reference.

    Each strip's ``eps_x`` is a scalar isotropic ``(Nx,)`` profile (-> ``(Nx, Ny)``
    output) OR a per-node ``(Nx, 3, 3)`` permittivity tensor (-> ``(Nx, Ny, 3, 3)``
    output, the oracle's anisotropic input)."""
    edges = np.cumsum([0.0] + [h for _, h in strips])
    yc = (np.arange(Ny) + 0.5) / Ny * Ly
    ex0 = np.asarray(strips[0][0], dtype=complex)
    is_tensor = ex0.ndim == 3                       # (Nx, 3, 3)
    eps = np.zeros((Nx, Ny, 3, 3) if is_tensor else (Nx, Ny), dtype=complex)
    for s, (ex, _) in enumerate(strips):
        msk = (yc >= edges[s]) & (yc < edges[s + 1])
        exa = np.asarray(ex, dtype=complex)
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
