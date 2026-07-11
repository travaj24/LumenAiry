"""EME (Eigenmode Expansion) lateral cascade — 2-D Bloch layer modes from 1-D
building blocks.

A doubly-periodic layer ``eps(x, y)`` (period ``Lx x Ly``) that is **piecewise
uniform in y** (a stack of y-strips, each varying only in x) has its 2-D Bloch
modes built WITHOUT a 2-D eigensolve: in each y-strip the field separates into
that strip's 1-D-x eigenmodes propagating laterally in y, and the strips are
joined by a lateral S-MATRIX cascade (Redheffer) with Bloch periodicity in y.
The 2-D modes are the ``qz^2`` where the unit-cell Bloch condition holds.

This is the "stack 1-D solvers laterally" idea done rigorously.  Two non-obvious
ingredients make it work (each was a failed first attempt; see the roadmap notes):

1. **S-matrices, NOT transfer matrices.**  A lateral T-matrix carries
   ``exp(+|ky| h)`` for evanescent strip modes and blows up exponentially (the
   classic reason S-matrices exist).  The Redheffer cascade only ever propagates
   decaying exponentials -> stable for the full mode basis.
2. **`det(M(qz^2)) = 0`, NOT eigenvalue tracking.**  The modes sit at per-strip
   band edges (``ky = 0``), where the Bloch *eigenvalue* ``mu`` dives toward the
   target and then vanishes from the spectrum (a branch transition) -- so
   tracking ``mu`` misses them.  The matrix
   ``M = -S12 t^2 + (I + S12 S21 - S22 S11) t - S21`` (``t = exp(i ky0 Ly)``,
   the Bloch QEP residual) instead goes **singular** at the modes, and its
   smallest singular value crosses zero cleanly through the band edges.  So a
   robust real-axis root-find on ``sigma_min(M(qz^2))`` finds every mode -- no
   contour (Beyn) method required.

Validated against a direct 2-D finite-difference eigensolve (``ref_2d_modes``):
uniform -> analytic ``eps k0^2 - kx^2 - ky^2``; structured -> the EME (analytic
in y) is the ``Ny -> inf`` limit the 2-D-FD converges to.

SCOPE: scalar Helmholtz layer modes (the core).  A full diffraction *surface*
solver adds the z-cascade onto the plane-wave half-spaces + the cylindrical/
Rayleigh far-field; the vector (TE/TM) extension doubles the strip-mode fields.
Both reuse this module's modes as the layer basis.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import svdvals
from scipy.optimize import minimize_scalar

from ...backend.array import is_jax_array


# --------------------------------------------------------------------------- #
#  1-D-x modal building block + stable lateral S-matrix primitives            #
# --------------------------------------------------------------------------- #
def strip_x_modes(eps_x, Lx, Nx, k0, kx0=0.0):
    """1-D-x eigenproblem ``[d2/dx2 + eps(x) k0^2] phi = lam phi`` on a periodic
    cell (Bloch phase ``kx0``).  Returns ``(lam, Phi)`` with ``Phi`` columns the
    orthonormal x-eigenfunctions (the strip's lateral cross-section modes)."""
    h = Lx / Nx
    ph = np.exp(1j * kx0 * Lx)
    D = np.zeros((Nx, Nx), dtype=complex)
    for i in range(Nx):
        D[i, i] = -2.0 / h ** 2
        D[i, (i + 1) % Nx] += (ph if i == Nx - 1 else 1.0) / h ** 2
        D[i, (i - 1) % Nx] += ((1.0 / ph) if i == 0 else 1.0) / h ** 2
    eps_c = np.asarray(eps_x, dtype=complex)
    A = D + np.diag(eps_c * k0 ** 2)
    if kx0 == 0.0 and not np.any(eps_c.imag):        # real symmetric -> orthonormal
        lam, Phi = np.linalg.eigh(A.real if np.isrealobj(A) else A)
    else:
        # A lossy eps (Im(eps) != 0) at kx0 = 0 makes A complex SYMMETRIC, not
        # Hermitian -- eigh would silently discard Im(eps) (LAPACK ignores the
        # imaginary diagonal), so route it to the general eig path (matching the
        # kx0 != 0 branch) and keep the complex spectrum.
        from scipy.linalg import eig
        # EME-nit (AUDIT_EME): unlike the ``eigh`` branch (ascending ``lam``),
        # ``eig`` returns the eigenpairs UNSORTED.  Downstream ``layer_modes``
        # scans ``qz^2`` and does not rely on ordering, so this is safe here;
        # a future consumer must NOT assume sorted ``lam`` from this branch.
        lam, Phi = eig(A)
        # EME-nit: floor the complex-SYMMETRIC bilinear norm ``sum(Phi^2)``
        # (the correct non-Hermitian normaliser here).  For a defective
        # (non-diagonalisable) ``A`` -- a degenerate/lossy edge -- it can
        # vanish, which would divide the eigenvector to inf/NaN; leave such a
        # column un-normalised (divide by 1) instead.
        _bilin = np.sum(Phi ** 2, axis=0)
        _bilin = np.where(np.abs(_bilin) < 1e-30, 1.0, _bilin)
        Phi = Phi / np.sqrt(_bilin)                     # complex-orthonormal
    return lam, Phi.astype(complex)


def _wv(lam, Phi, qz2):
    """The strip's forward modal field matrices: ``W = Phi`` (psi), ``V = Phi
    diag(i ky)`` (d psi/dy), with ``ky = sqrt(lam - qz2)`` on the principal
    (forward-decaying) branch.  Backward modes are ``[W; -V]``."""
    ky = np.sqrt(lam - qz2 + 0j)
    return Phi, Phi @ np.diag(1j * ky), ky


def _interface(Wa, Va, Wb, Vb):
    a = np.linalg.solve(Wb, Wa)
    b = np.linalg.solve(Vb, Va)
    apb, amb = a + b, a - b
    ia = np.linalg.inv(apb)
    return (-ia @ amb, 2.0 * ia, 0.5 * (apb - amb @ ia @ amb), amb @ ia)


def _prop(ky, h):
    X = np.diag(np.exp(1j * ky * h))
    Z = np.zeros_like(X)
    return (Z, X, X, Z)


def _star(SA, SB):
    A11, A12, A21, A22 = SA
    B11, B12, B21, B22 = SB
    I = np.eye(A11.shape[0], dtype=complex)
    D = np.linalg.inv(I - B11 @ A22)
    F = np.linalg.inv(I - A22 @ B11)
    return (A11 + A12 @ D @ B11 @ A21, A12 @ D @ B12,
            B21 @ F @ A21, B22 + B21 @ F @ A22 @ B12)


def cell_smatrix(strip_modes, qz2):
    """Unit-cell (one y-period) lateral S-matrix, in the FIRST strip's basis.
    ``strip_modes`` = list of ``((lam, Phi), height)``.  Cascade
    ``prop(0) * iface(0->1) * prop(1) * ... * iface(n-1->0)``."""
    wv = [(_wv(lam, Phi, qz2), h) for (lam, Phi), h in strip_modes]
    (W0, V0, ky0), h0 = wv[0]
    S = _prop(ky0, h0)
    for s in range(len(wv)):
        (Wc, Vc, _), _ = wv[s]
        (Wn, Vn, kyn), hn = wv[(s + 1) % len(wv)]
        S = _star(S, _interface(Wc, Vc, Wn, Vn))
        if s < len(wv) - 1:
            S = _star(S, _prop(kyn, hn))
    return S


def _bloch_residual_M(S, t):
    """The Bloch-QEP residual matrix ``M(qz^2)``; singular at a layer mode
    (Bloch eigenvalue ``= t``)."""
    S11, S12, S21, S22 = S
    I = np.eye(S11.shape[0], dtype=complex)
    return -S12 * t ** 2 + (I + S12 @ S21 - S22 @ S11) * t - S21


# --------------------------------------------------------------------------- #
#  The solver: 2-D Bloch layer modes via sigma_min(M) root-finding            #
# --------------------------------------------------------------------------- #
def dispersion(strip_modes, qz2, ky0, Ly):
    """``sigma_min(M(qz^2))`` -- zero at a 2-D Bloch layer mode."""
    S = cell_smatrix(strip_modes, qz2)
    M = _bloch_residual_M(S, np.exp(1j * ky0 * Ly))
    return svdvals(M)[-1]


def layer_modes(strips, Lx, Nx, Ly, k0, qz2_range, *, kx0=0.0, ky0=0.0,
                n_scan=600, tol=5e-2, merge_rtol=3e-3):
    """2-D Bloch modes of a y-strip-sectioned layer.

    ``strips`` : list of ``(eps_x_array, height)`` (heights sum to ``Ly``).
    Returns the propagation eigenvalues ``qz^2`` (descending) inside
    ``qz2_range = (lo, hi)``.

    The Bloch-residual ``M`` lives in the *starting* strip's modal basis, so a
    mode that is deeply EVANESCENT in that strip leaves an exponentially small
    (sub-precision) signature and is missed.  To capture every mode we build the
    cell starting from EACH strip in turn (rotating the list) -- each basis sees
    the modes PROPAGATING in its strip -- and take the union (deduped).  ``M``
    goes singular (``sigma_min(M) -> 0``) at a mode; minima below ``tol`` are
    refined by Brent.
    """
    if strips and is_jax_array(strips[0][0]):
        raise NotImplementedError(
            "layer_modes: the sigma_min root-scan (real-axis scan + Brent "
            "root-find) is NOT differentiable.  For differentiable 2-D modes use "
            "ref_2d_modes with a JAX eps (the eig-based FD-oracle twin), or solve "
            "on NumPy and apply implicit differentiation at the converged mode.")
    hsum = float(np.real(sum(s[1] for s in strips)))
    if abs(hsum - Ly) > 1e-9 * max(abs(Ly), 1.0):
        raise ValueError(
            f"layer_modes: strip heights sum to {hsum:g} but Ly = {Ly:g}; the "
            "cell physics uses sum(h) while the Bloch wrap phase uses Ly, so "
            "they must agree (documented contract: heights sum to Ly).")
    lo, hi = qz2_range
    grid = np.linspace(lo, hi, n_scan)
    sm_all = [(strip_x_modes(e, Lx, Nx, k0, kx0), h) for e, h in strips]
    found = []
    for start in range(len(strips)):
        sm = sm_all[start:] + sm_all[:start]          # rotate -> this strip's basis

        def f(q, sm=sm):
            try:
                return dispersion(sm, q, ky0, Ly)
            except np.linalg.LinAlgError:
                # a scan sample landing EXACTLY on a strip band edge (ky = 0,
                # e.g. hi = max(eps)*k0^2 for an x-uniform strip) makes the
                # _interface V-block exactly singular.  A band-edge sample is
                # not a mode candidate -- treat it as +inf so the scan skips it.
                return np.inf

        d = np.array([f(q) for q in grid])
        for i in range(1, len(grid) - 1):
            if d[i] < d[i - 1] and d[i] < d[i + 1] and d[i] < tol:
                r = minimize_scalar(f, bracket=(grid[i - 1], grid[i], grid[i + 1]),
                                    method="brent", options={"xtol": 1e-7})
                if r.fun < tol:
                    found.append(r.x)
    # dedup: a mode propagating in several strips is found in several bases,
    # each refined to slightly different values -- merge clusters within a small
    # relative window (so genuinely-distinct modes survive; near-degenerate
    # pairs may merge to one -- a documented v1 limitation).
    found = sorted(found, reverse=True)
    out = []
    for q in found:
        if not out or abs(out[-1] - q) > merge_rtol * max(abs(q), 1.0):
            out.append(q)
    return np.array(out)


# --------------------------------------------------------------------------- #
#  Mode field reconstruction (the eigenvector psi(x,y) at a found mode)         #
# --------------------------------------------------------------------------- #
def _global_lateral_nullspace(strip_modes, qz2, t):
    """Per-strip modal amplitudes ``(a_s, b_s)`` of the layer mode -- the null
    vector of the GLOBAL lateral interface system.

    Unlike the eigenvalue search (which cascades S-matrices), reconstructing the
    field needs every strip's amplitudes.  Assembling the strip-interface +
    Bloch-wrap conditions as ONE block matrix ``G(qz^2)`` and taking its null
    space is stable -- each block carries only a SINGLE strip's ``exp(+-i ky h)``
    (bounded), never the accumulated transfer-matrix product that blows up.
    Returns the stacked null vector ``c`` and ``sigma_min(G)`` (the mode residual).
    """
    S = len(strip_modes)
    Nx = strip_modes[0][0][1].shape[0]
    wv = []
    for (lam, Phi), h in strip_modes:
        ky = np.sqrt(lam - qz2 + 0j)
        wv.append((Phi, ky, h, np.exp(1j * ky * h), np.exp(-1j * ky * h)))
    n = 2 * Nx * S
    G = np.zeros((n, n), dtype=complex)
    for r in range(S):                              # strip r (right) -> strip nxt (left)
        Phi_c, ky_c, _, E_c, Ei_c = wv[r]
        nxt = (r + 1) % S
        Phi_n, ky_n, _, _, _ = wv[nxt]
        tt = t if r == S - 1 else 1.0               # Bloch phase on the wrap only
        rf, rd = 2 * Nx * r, 2 * Nx * r + Nx        # field rows / deriv rows
        ca, cb = 2 * Nx * r, 2 * Nx * r + Nx        # strip r: a_r / b_r columns
        na, nb = 2 * Nx * nxt, 2 * Nx * nxt + Nx    # strip nxt: a_n / b_n columns
        ikc, ikn = 1j * ky_c, 1j * ky_n
        # field continuity: Phi_c(a_c E_c + b_c Ei_c) = tt Phi_n(a_n + b_n)
        G[rf:rf + Nx, ca:ca + Nx] += Phi_c * E_c[None, :]
        G[rf:rf + Nx, cb:cb + Nx] += Phi_c * Ei_c[None, :]
        G[rf:rf + Nx, na:na + Nx] -= tt * Phi_n
        G[rf:rf + Nx, nb:nb + Nx] -= tt * Phi_n
        # deriv continuity: Phi_c(i ky_c)(a_c E_c - b_c Ei_c) = tt Phi_n(i ky_n)(a_n - b_n)
        G[rd:rd + Nx, ca:ca + Nx] += Phi_c * (ikc * E_c)[None, :]
        G[rd:rd + Nx, cb:cb + Nx] -= Phi_c * (ikc * Ei_c)[None, :]
        G[rd:rd + Nx, na:na + Nx] -= tt * Phi_n * ikn[None, :]
        G[rd:rd + Nx, nb:nb + Nx] += tt * Phi_n * ikn[None, :]
    # NOTE (audit P3-20, investigated and KEPT as two calls): the pair looks
    # redundant but is accuracy-load-bearing.  Unlike the vector sibling
    # (mode_field_vec), this G is NOT column-equilibrated -- deeply evanescent
    # strip modes put exp(+|ky|h) ~ 1e12 columns next to O(1) ones -- and on
    # that scaling the with-vectors gesdd SVD (np.linalg.svd) reports s[-1] up
    # to ~14x off (measured 4.97e-2 vs the true 3.58e-3 on the reference
    # structured cell), breaking the documented sigma diagnostic (tests pin
    # sigma < 1e-2 at a true mode).  The values-only svdvals path stays
    # accurate, so sigma comes from svdvals and only the null vector from the
    # full SVD.  (||G @ c|| is no substitute either: it is ~2.5e-2 vs sigma
    # 9.6e-5 on the same cell's top mode.)
    s = svdvals(G)
    c = np.linalg.svd(G)[2][-1].conj()              # right null vector
    return c, s[-1]


def mode_field(strip_modes, qz2, ky0, Ly, Ny):
    """Reconstruct the 2-D Bloch mode field ``psi(x, y)`` at a layer mode ``qz2``.

    ``strip_modes`` = ``[((lam, Phi), height), ...]`` (as built inside
    ``layer_modes``).  Samples ``psi`` on an ``(Nx, Ny)`` grid with cell-centre
    ``y`` (matching ``strips_to_eps_xy`` / the 2-D-FD oracle).  Returns
    ``(psi, sigma)`` with ``sigma = sigma_min(G)`` (small confirms a true mode).
    """
    Nx = strip_modes[0][0][1].shape[0]
    c, sigma = _global_lateral_nullspace(strip_modes, qz2, np.exp(1j * ky0 * Ly))
    S = len(strip_modes)
    edges = np.concatenate([[0.0], np.cumsum([h for _, h in strip_modes])])
    yc = (np.arange(Ny) + 0.5) / Ny * Ly
    psi = np.zeros((Nx, Ny), dtype=complex)
    for j, y in enumerate(yc):
        s = min(int(np.searchsorted(edges, y, side="right")) - 1, S - 1)
        (lam, Phi), _ = strip_modes[s]
        ky = np.sqrt(lam - qz2 + 0j)
        a = c[2 * Nx * s:2 * Nx * s + Nx]
        b = c[2 * Nx * s + Nx:2 * Nx * (s + 1)]
        eta = y - edges[s]
        psi[:, j] = Phi @ (a * np.exp(1j * ky * eta) + b * np.exp(-1j * ky * eta))
    return psi, sigma


# --------------------------------------------------------------------------- #
#  Direct 2-D finite-difference reference (validation oracle)                  #
# --------------------------------------------------------------------------- #
def ref_2d_modes(eps_xy, Lx, Ly, Nx, Ny, k0, kx0=0.0, ky0=0.0, return_vecs=False,
                 k=None, sigma=None, return_complex=False):
    """Bloch modes ``qz^2`` of ``[d2/dx2 + d2/dy2 + eps(x,y) k0^2] psi = qz^2 psi``
    by a direct 2-D FD eigensolve (the independent oracle).  With ``return_vecs``
    also returns the eigenvectors reshaped to ``(Nx, Ny)`` (descending by ``qz^2``).

    Default: dense ``eig`` of the full ``Nx*Ny`` spectrum.  Pass ``k`` for a SPARSE
    shift-invert (``scipy.sparse.linalg.eigs``) returning the ``k`` modes nearest
    ``sigma`` (default: the band top ``max(eps) k0^2``) -- O(100x) faster for the
    top / in-band modes.

    ``return_complex`` keeps the complex ``qz^2`` for a LOSSY ``eps`` (parity with
    ``ref_2d_modes_vector``); the default returns ``Re(qz^2)`` and WARNS if that
    discards a nonzero imaginary part."""
    if any(is_jax_array(a) for a in (eps_xy, k0, kx0, ky0)):
        # differentiable twin (eig-based; see ._jax_modes).  Scalar isotropic eps,
        # dense spectrum, qz^2 only.
        if return_complex:
            raise NotImplementedError(
                "ref_2d_modes: return_complex is not supported by the JAX twin "
                "(it returns the real spectrum); use the NumPy path.")
        from ._jax_modes import _ref_2d_modes_jax
        return _ref_2d_modes_jax(
            eps_xy, Lx, Ly, Nx, Ny, k0, kx0, ky0, return_vecs=return_vecs,
            k=k, sigma=sigma)
    if not return_complex and np.any(np.asarray(eps_xy).imag):
        import warnings
        warnings.warn(
            "ref_2d_modes: eps has a nonzero imaginary part (lossy) but "
            "return_complex=False -- the imaginary parts of qz^2 are DISCARDED; "
            "pass return_complex=True to keep the complex spectrum.",
            stacklevel=2)
    import scipy.sparse as _sp
    hx, hy = Lx / Nx, Ly / Ny
    N = Nx * Ny
    px, py = np.exp(1j * kx0 * Lx), np.exp(1j * ky0 * Ly)

    def ix(i, j):
        return (i % Nx) * Ny + (j % Ny)

    rows, cols, vals = [], [], []
    for i in range(Nx):
        for j in range(Ny):
            p = ix(i, j)
            rows += [p, p, p, p, p]
            cols += [p, ix(i + 1, j), ix(i - 1, j), ix(i, j + 1), ix(i, j - 1)]
            vals += [-2 / hx ** 2 - 2 / hy ** 2 + eps_xy[i, j] * k0 ** 2,
                     (px if i == Nx - 1 else 1) / hx ** 2,
                     ((1 / px) if i == 0 else 1) / hx ** 2,
                     (py if j == Ny - 1 else 1) / hy ** 2,
                     ((1 / py) if j == 0 else 1) / hy ** 2]
    A = _sp.coo_matrix((vals, (rows, cols)), shape=(N, N), dtype=complex).tocsc()
    if k is None:
        from scipy.linalg import eig
        Ad = A.toarray()
        if not return_vecs:
            w = eig(Ad, right=False)
            if not return_complex:
                return np.sort(w.real)[::-1]
            return w[np.argsort(w.real)[::-1]]
        w, Vv = eig(Ad)
    else:
        from scipy.sparse.linalg import eigs
        if sigma is None:
            sigma = float(np.max(eps_xy.real)) * k0 ** 2
        w, Vv = eigs(A, k=min(k, N - 2), sigma=sigma)
        if not return_vecs:
            if not return_complex:
                return np.sort(w.real)[::-1]
            return w[np.argsort(w.real)[::-1]]
    order = np.argsort(w.real)[::-1]
    w_out = w[order] if return_complex else w[order].real
    return w_out, Vv[:, order].reshape(Nx, Ny, -1)


def strips_to_eps_xy(strips, Lx, Nx, Ly, Ny):
    """Rasterize a strip list onto an ``(Nx, Ny)`` grid for the 2-D reference.

    Thin delegation to the ONE general (tensor-aware, magnetic-3-tuple-tolerant)
    implementation in :mod:`.eme_2d_vector` -- also the package-level export.
    (The two modules historically carried drifted same-named copies.)"""
    from .eme_2d_vector import strips_to_eps_xy as _general
    return _general(strips, Lx, Nx, Ly, Ny)
