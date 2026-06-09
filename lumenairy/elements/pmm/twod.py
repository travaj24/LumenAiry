"""
lumenairy.elements.pmm.twod -- 2-D crossed-grating Polynomial Modal Method (hybrid).
================================================================================

:func:`pmm_efficiency_2d` is the 2-D (doubly periodic) analogue of
:func:`lumenairy.elements.pmm.pmm_efficiency_1d`, for a **separable rectangular
pillar** (one ``eps_pillar`` rectangle embedded in an ``eps_host`` background).
It is the modal-method counterpart of :func:`lumenairy.elements.rcwa.rcwa_efficiency_2d`.

How it works (and how it differs from the 1-D PMM)
--------------------------------------------------
The 1-D PMM is a *no-floor* subsectional nodal modal method -- its accuracy is
limited only by the polynomial degree, with no Fourier-truncation plateau.  A
fully nodal **2-D** modal method on a tensor-product GLL grid is *not* viable:
the degenerate uniform-region nodal eigenproblem is flux-inconsistent and the
all-nodal solve violates energy conservation (the same wall RCWA hits, which is
why RCWA falls back to its analytic ``W = I`` plane-wave region path).

This solver is therefore a **hybrid**:

* **Structured layer** -- assembled as a tensor-product GLL spectral-element
  nodal operator (so the rectangular pillar is resolved geometrically, not by a
  staircased Fourier series), then **Fourier-Galerkin projected** into the
  Rayleigh (plane-wave) basis.  The projection ``O_F = T O T^+`` gives square
  Fourier-basis layer modes with the correct spectrum and spectral convergence.
* **Half-space regions** (substrate / superstrate) -- treated **analytically**
  as exact plane waves (``W = I``), giving a clean far field and exact flux.

Because the layer is represented in a truncated Rayleigh basis of half-width
``n_orders``, this method **has a Fourier-truncation floor**, exactly like the
FMM/RCWA -- it is *not* no-floor like the 1-D PMM.  Its advantage over RCWA is
that the pillar edge is captured by the nodal grid (geometry-conforming) rather
than by a Gibbs-limited Fourier series, so for a given ``n_orders`` it tends to
be at least as accurate; validated against :func:`rcwa_efficiency_2d` (Li rule,
17 orders) to ``~2e-4`` on the 0-th order at ``degree=11``, energy conserved to
``~1e-3``.

The "null-mode" fix that makes it work
--------------------------------------
A naive nodal ``[Sx; Sy] P@Q`` solve produced a fatal cloud of spurious modes.
The root cause was **not** a fundamental vector-Maxwell spurious-gradient
problem -- it was the classic **periodic-grid Nyquist null mode** of the nodal
first-derivative operator, which appears precisely when the per-axis node count
is *even*.  Forcing the per-axis node count **odd** (``3 * degree *
elements_per_strip`` odd, i.e. odd ``degree``) restores the correct
one-dimensional derivative kernel, and the divergence-reduced second-order
``[Sx; Sy]`` form then injects no spurious modes -- no grad-div penalty,
projection, or Lagrange multiplier needed.

Scope / limitations
-------------------
* **Single separable rectangular pillar** only (``eps_pillar`` in ``eps_host``).
  For arbitrary 2-D profiles or multilayer stacks use :func:`rcwa_efficiency_2d`
  / :class:`RCWAStack`.
* **Isotropic scalar** TE/TM only -- no anisotropy, no full Jones (use
  :func:`rcwa_jones_2d` for tensor cells).
* **Normal / near-normal incidence** is validated.  Oblique incidence is wired
  in via the Bloch shift ``d/dx -> d/dx + i*kx0`` but has *not* been validated
  against an oracle at large angles -- treat ``theta`` beyond a few degrees as
  experimental.
* Single layer of thickness ``depth``; the regions are uniform half-spaces.

CONVENTION (matches the public efficiency convention of ``rcwa_efficiency_2d``):
public ``exp(-i w t)``; ``Gx = (-i/k0) d/dx`` nodal drop-in for ``Kx``;
``lam = sqrt(-kz^2/k0^2)`` on the decay branch; forward ``X = exp(-lam k0 L)``;
**no** eps conjugation.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ..rcwa import Efficiency2D  # cross-suite 2-D result (unpacks (o,R,T), carries .dof)
from ._core import (
    _gll_nodes_weights,
    _graded_boundaries,
    _interface_smatrix,
    _lagrange_derivative_matrix,
    _propagation_smatrix,
    _redheffer_star,
)

__all__ = ["pmm_efficiency_2d"]

_C = np.complex128


# =========================================================================== #
# 1-D per-axis GLL spectral-element assembly (periodic C0, 3 strips: the pillar
# sits in the middle strip [wall0, wall1]; the two outer strips are host).
# =========================================================================== #

def _build_axis(period, walls, degree, n_el=1, grade=False):
    ref_nodes, ref_w = _gll_nodes_weights(degree)
    Dref = _lagrange_derivative_matrix(ref_nodes)
    bnds = [0.0] + list(walls) + [period]
    strips = list(zip(bnds[:-1], bnds[1:]))
    elem_bnds = []
    for s, (a, b) in enumerate(strips):
        eb = _graded_boundaries(a, b, n_el, grade)
        for e in range(len(eb) - 1):
            elem_bnds.append((eb[e], eb[e + 1], s))
    n_el_tot = len(elem_bnds)

    l2g = np.zeros((n_el_tot, degree + 1), dtype=int)
    gid = 0
    for e in range(n_el_tot):
        for a in range(degree + 1):
            if a == 0 and e > 0:
                l2g[e, a] = l2g[e - 1, degree]
            else:
                l2g[e, a] = gid
                gid += 1
    last = l2g[n_el_tot - 1, degree]      # wrap the last node onto node 0 (periodic)
    l2g[l2g == last] = 0
    n = last

    def _z():
        return np.zeros((n, n), dtype=_C)

    M = _z()
    D = _z()                              # M-weighted (consistent) first derivative
    nstr = len(strips)
    Mtile = [_z() for _ in range(nstr)]
    for e in range(n_el_tot):
        xl, xr, s = elem_bnds[e]
        J = 0.5 * (xr - xl)
        wel = ref_w * J
        Dphys = Dref / J
        Mloc = np.diag(wel)
        Dloc = Mloc @ Dphys
        idx = l2g[e]
        ix = np.ix_(idx, idx)
        M[ix] += Mloc
        D[ix] += Dloc
        Mtile[s][ix] += Mloc
    return dict(M=M, D=D, Mtile=Mtile, n=n, strips=strips, l2g=l2g,
                elem_bnds=elem_bnds, degree=degree, ref_nodes=ref_nodes,
                period=period)


# =========================================================================== #
# 2-D Kronecker assembly of the RCWA-normalized operators.
#   vec ordering x-inner: index I = jx + nx*jy  ->  kron(Y, X)
# =========================================================================== #

def _assemble_2d(ax, ay, eps_tile, k0):
    Mx, Dx = ax["M"], ax["D"]
    My, Dy = ay["M"], ay["D"]
    nx, ny = ax["n"], ay["n"]
    N = nx * ny
    M = np.kron(My, Mx)
    DX = np.kron(My, Dx)
    DY = np.kron(Dy, Mx)
    Minv = np.linalg.inv(M)
    Gx = (-1j / k0) * (Minv @ DX)
    Gy = (-1j / k0) * (Minv @ DY)

    nsx, nsy = len(ax["Mtile"]), len(ay["Mtile"])
    P_eps = np.zeros((N, N), dtype=_C)
    P_inv = np.zeros((N, N), dtype=_C)
    for sx in range(nsx):
        for sy in range(nsy):
            e = eps_tile[sx, sy]
            ker = np.kron(ay["Mtile"][sy], ax["Mtile"][sx])
            P_eps += e * ker
            P_inv += (1.0 / e) * ker
    Eps = Minv @ P_eps               # multiply-by-eps  (Laurent / tangential)
    Einv = Minv @ P_inv              # multiply-by-1/eps (E_z elimination, Li)
    Epn = np.linalg.solve(P_inv, M)  # inverse rule [[1/eps]]^-1 (wall-normal Ex)
    return dict(Gx=Gx, Gy=Gy, Eps=Eps, Einv=Einv, Epn=Epn, M=M, N=N)


# =========================================================================== #
# decay-branch helpers + plane-wave kz
# =========================================================================== #

def _sqrt_decay(x):
    r = np.sqrt(np.asarray(x, dtype=_C))
    on_cut = r.real == 0
    return np.where(on_cut & (r.imag < 0), -r, r)


def _inv_lam(lam):
    safe = np.where(np.abs(lam) < 1e-12, 1e-12, lam)
    return 1.0 / safe


def _kz_forward2(eps, kx, ky):
    """``kz/k0`` on the forward branch for ``exp(-i w t)`` (2-D)."""
    val = np.sqrt(np.asarray(eps - kx ** 2 - ky ** 2, dtype=_C))
    return np.where(val.imag < 0.0, -val, val)


# =========================================================================== #
# Far-field nodal -> Rayleigh projection  T[m,i] = (1/L) INT phi_i e^{-imGx}
# =========================================================================== #

def _axis_projection(ax, orders_1d):
    from numpy.polynomial.legendre import leggauss
    l2g, elem_bnds, degree = ax["l2g"], ax["elem_bnds"], ax["degree"]
    ref_nodes, n, L = ax["ref_nodes"], ax["n"], ax["period"]
    G = 2.0 * np.pi / L
    nq = max(2 * degree + 8, 24)
    xg, wg = leggauss(nq)
    wbary = np.ones(degree + 1)
    for j in range(degree + 1):
        for k in range(degree + 1):
            if k != j:
                wbary[j] /= (ref_nodes[j] - ref_nodes[k])

    def _vals(xi):
        V = np.zeros((len(xi), degree + 1))
        for r, x in enumerate(xi):
            diff = x - ref_nodes
            if np.any(np.abs(diff) < 1e-14):
                V[r, np.argmin(np.abs(diff))] = 1.0
            else:
                num = wbary / diff
                V[r, :] = num / num.sum()
        return V

    Lv = _vals(xg)
    T = np.zeros((len(orders_1d), n), dtype=_C)
    for e in range(len(elem_bnds)):
        xl, xr, _s = elem_bnds[e]
        J = 0.5 * (xr - xl)
        xphys = 0.5 * (xr + xl) + J * xg
        phase = np.exp(-1j * np.outer(np.asarray(orders_1d) * G, xphys))
        contrib = (phase * (wg * J / L)) @ Lv
        idx = l2g[e]
        for a in range(degree + 1):
            T[:, idx[a]] += contrib[:, a]
    return T


def _projectors(ax, ay, ox, oy):
    Tx = _axis_projection(ax, ox)
    Ty = _axis_projection(ay, oy)
    Tp = np.kron(Ty, Tx)
    Tpinv = np.linalg.pinv(Tp)
    return Tp, Tpinv


# =========================================================================== #
# layer modes (projected nodal) + analytic region modes
# =========================================================================== #

def _fourier_ops(ax, ay, eps_tile, k0, Tp, Tpinv, kx0, ky0):
    ops = _assemble_2d(ax, ay, eps_tile, k0)
    N = ops["N"]
    I = np.eye(N, dtype=_C)
    ops["Gx"] = ops["Gx"] + kx0 * I
    ops["Gy"] = ops["Gy"] + ky0 * I
    GxF = Tp @ ops["Gx"] @ Tpinv
    GyF = Tp @ ops["Gy"] @ Tpinv
    EpsF = Tp @ ops["Eps"] @ Tpinv
    EinvF = Tp @ ops["Einv"] @ Tpinv
    EpnF = Tp @ ops["Epn"] @ Tpinv
    return GxF, GyF, EpsF, EinvF, EpnF


def _layer_modes_projected(GxF, GyF, EpsF, EinvF, EpnF, formulation="li"):
    Nf = GxF.shape[0]
    I = np.eye(Nf, dtype=_C)
    EPS_normal = EpnF if formulation == "li" else EpsF
    Q = np.block([[GxF @ GyF, EpsF - GxF @ GxF],
                  [GyF @ GyF - EPS_normal, -GyF @ GxF]])
    EPS_inv = EinvF if formulation == "li" else np.linalg.inv(EpsF)
    P = np.block([[GxF @ EPS_inv @ GyF, I - GxF @ EPS_inv @ GxF],
                  [GyF @ EPS_inv @ GyF - I, -GyF @ EPS_inv @ GxF]])
    lam2, W = np.linalg.eig(P @ Q)
    lam = _sqrt_decay(lam2)
    V = Q @ W @ np.diag(_inv_lam(lam))
    return W, V, lam


def _homogeneous_modes(kx, ky, eps):
    """Analytic plane-wave (Rayleigh) modes: W=I, V=Q diag(1/lam)."""
    N = len(kx)
    Kx = np.diag(kx.astype(_C))
    Ky = np.diag(ky.astype(_C))
    kz = _kz_forward2(eps, kx, ky)
    lam = _sqrt_decay(-np.concatenate([kz, kz]) ** 2)
    eps_I = eps * np.eye(N, dtype=_C)
    Q = np.block([[Kx @ Ky, eps_I - Kx @ Kx],
                  [Ky @ Ky - eps_I, -Ky @ Kx]])
    W = np.eye(2 * N, dtype=_C)
    V = Q @ np.diag(_inv_lam(lam))
    return W, V, lam, kz


# =========================================================================== #
# public driver
# =========================================================================== #

def pmm_efficiency_2d(
    period_x: float,
    period_y: float,
    eps_pillar: complex,
    eps_host: complex,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    degree: int = 11,
    elements_per_strip: int = 1,
    grade: bool = False,
    polarization: str = "te",
    theta: float = 0.0,
    phi: float = 0.0,
    n_orders: int = 11,
    formulation: str = "li",
) -> Efficiency2D:
    r"""Diffraction efficiencies of a 2-D rectangular pillar via the hybrid PMM.

    The modal-method counterpart of :func:`rcwa_efficiency_2d` for a single
    separable rectangular pillar (``eps_pillar`` rectangle in an ``eps_host``
    background).  See the module docstring for the method, the null-mode fix,
    and -- importantly -- the **scope limitations** (single rect pillar,
    isotropic scalar TE/TM, normal/near-normal incidence, single layer; this is
    a hybrid with a Fourier-truncation ``n_orders`` floor, *not* no-floor like
    the 1-D PMM).

    Parameters
    ----------
    period_x, period_y : float
        Lattice periods (metres).
    eps_pillar, eps_host : complex
        Relative permittivities of the pillar and the surrounding host.
    x_bounds, y_bounds : (float, float)
        ``(x0, x1)`` and ``(y0, y1)`` rectangle edges (metres), ``0 <= x0 <
        x1 <= period_x`` (and likewise in y).
    n_substrate, n_superstrate : complex
        Refractive indices of the transmission (substrate) and incidence
        (superstrate) half-spaces.  Internally squared to permittivities.
    depth : float
        Pillar height / layer thickness (metres).
    wavelength : float
        Vacuum wavelength (metres).
    degree : int, optional
        GLL polynomial degree per element.  With 3 strips x
        ``elements_per_strip`` elements the per-axis node count is
        ``3 * degree * elements_per_strip``, which **must be odd** (avoids the
        periodic Nyquist null mode); an odd ``degree`` with the defaults
        satisfies this.
    elements_per_strip : int, optional
        Spectral elements per strip (per axis).  Default 1.
    grade : bool, optional
        Graded element boundaries within each strip (clustering toward walls).
    polarization : {"te", "tm"}, optional
        Incident linear polarization.
    theta, phi : float, optional
        Incidence polar / azimuth angles (radians).  Validated near normal;
        large ``theta`` is experimental.
    n_orders : int, optional
        Fourier (Rayleigh) truncation half-width for both regions and the
        projected layer.  Must satisfy ``2*n_orders + 1 <= 3 * degree *
        elements_per_strip`` so the nodal grid can represent the harmonics.
    formulation : {"li", "laurent"}, optional
        Factorization rule for the layer operator (``"li"`` = inverse rule for
        the wall-normal field component; recommended).

    Returns
    -------
    orders : ndarray, shape (M, 2)
        ``(m, n)`` diffraction-order indices.
    R, T : ndarray, shape (M,)
        Reflected / transmitted efficiencies per order (energy fractions).
    dof : int
        Modal degrees of freedom retained (``2 * (2*n_orders+1)**2``).
    """
    n_nodes_axis = 3 * degree * elements_per_strip
    if n_nodes_axis % 2 == 0:
        raise ValueError(
            "per-axis node count 3*degree*elements_per_strip must be ODD to "
            "avoid the periodic Nyquist null mode (use an odd polynomial "
            f"degree); got degree={degree}, elements_per_strip="
            f"{elements_per_strip} -> {n_nodes_axis} nodes")
    if 2 * n_orders + 1 > n_nodes_axis:
        raise ValueError(
            f"n_orders={n_orders} too large for degree={degree}: need "
            f"2*n_orders+1 ({2 * n_orders + 1}) <= per-axis nodes "
            f"({n_nodes_axis}); raise degree or lower n_orders")
    if polarization not in ("te", "tm"):
        raise ValueError("polarization must be 'te' or 'tm'")

    x0, x1 = float(x_bounds[0]), float(x_bounds[1])
    y0, y1 = float(y_bounds[0]), float(y_bounds[1])
    eps_p = _C(eps_pillar)
    eps_h = _C(eps_host)
    eps_sup = _C(n_superstrate) ** 2
    eps_sub = _C(n_substrate) ** 2

    k0 = 2.0 * np.pi / wavelength
    wl = wavelength
    ax = _build_axis(period_x, [x0, x1], degree, elements_per_strip, grade)
    ay = _build_axis(period_y, [y0, y1], degree, elements_per_strip, grade)
    eps_tile = np.full((3, 3), eps_h, dtype=_C)
    eps_tile[1, 1] = eps_p

    nre = float(np.real(np.sqrt(eps_sup)))
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)

    ox = np.arange(-n_orders, n_orders + 1)
    oy = np.arange(-n_orders, n_orders + 1)
    order_x = np.tile(ox, len(oy))
    order_y = np.repeat(oy, len(ox))
    Nf = len(order_x)
    kxv = kx0 + order_x * (wl / period_x)
    kyv = ky0 + order_y * (wl / period_y)

    # ---- analytic Fourier regions (exact plane waves: flux-exact far field) --
    Wsup, Vsup, _ls, kz_ref = _homogeneous_modes(kxv, kyv, eps_sup)
    Wsub, Vsub, _lb, kz_trn = _homogeneous_modes(kxv, kyv, eps_sub)

    # ---- layer modes: projected nodal, UNLESS laterally uniform (-> analytic)
    uniform_layer = abs(eps_p - eps_h) < 1e-12
    if uniform_layer:
        Wl, Vl, lam_l, _ = _homogeneous_modes(kxv, kyv, eps_h)
    else:
        Tp, Tpinv = _projectors(ax, ay, ox, oy)
        Wl, Vl, lam_l = _layer_modes_projected(
            *_fourier_ops(ax, ay, eps_tile, k0, Tp, Tpinv, kx0, ky0),
            formulation=formulation)

    # ---- Redheffer recursion (Fourier/Rayleigh basis) ----
    S = _interface_smatrix(Wsup, Vsup, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam_l, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wsub, Vsub))
    S11, _S12, S21, _S22 = S

    # ---- incident (0,0) plane wave (Rayleigh basis: amplitudes ARE coeffs) --
    kt = float(np.hypot(kx0, ky0))
    if kt < 1e-12:
        ex0, ey0 = (0.0, 1.0) if polarization == "te" else (1.0, 0.0)
        einc_sq = 1.0
    else:
        axu, ayu = kx0 / kt, ky0 / kt
        if polarization == "te":
            ex0, ey0 = -ayu, axu
            einc_sq = 1.0
        else:
            ex0, ey0 = axu, ayu
            kz_inc0 = float(np.real(_kz_forward2(eps_sup, kx0, ky0)))
            einc_sq = 1.0 + (kt / kz_inc0) ** 2
    delta = ((order_x == 0) & (order_y == 0)).astype(_C)
    cinc = np.concatenate([ex0 * delta, ey0 * delta])
    r = S11 @ cinc
    t = S21 @ cinc
    rx, ry = r[:Nf], r[Nf:]
    tx, ty = t[:Nf], t[Nf:]

    kz_inc = float(np.real(_kz_forward2(eps_sup, kx0, ky0)))
    safe_r = np.where(np.abs(kz_ref) < 1e-12, 1.0, kz_ref)
    safe_t = np.where(np.abs(kz_trn) < 1e-12, 1.0, kz_trn)
    rz = -(kxv * rx + kyv * ry) / safe_r
    tz = -(kxv * tx + kyv * ty) / safe_t
    R = np.real(kz_ref / kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2
                                    + np.abs(rz) ** 2) / einc_sq
    T = np.real(kz_trn / kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                    + np.abs(tz) ** 2) / einc_sq
    R = np.where(np.real(kz_ref) > 0, np.real(R), 0.0)
    T = np.where(np.real(kz_trn) > 0, np.real(T), 0.0)
    orders2d = np.stack([order_x, order_y], axis=1)
    # cross-suite return shape: unpacks as (orders, R, T); .dof = 2*Nf (the modal
    # eigenproblem dimension).  Was a bare 4-tuple (orders, R, T, dof) pre-v5.12.
    return Efficiency2D(orders2d, R, T, 2 * Nf)
