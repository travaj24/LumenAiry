"""Shared scratch solver for the z-profile planning probes (1-D, scalar TE / TM).

ANALYSIS ONLY -- nothing in ``lumenairy/`` is modified; the library is imported
read-only for its GLL nodes and its differentiation matrix.

What this file implements (the plan's "route B" in one dimension):

* a periodic C0 spectral-element basis in the computational coordinate ``u``
  on a FIXED wall grid (the walls never move in ``u``);
* a z-dependent covariant map ``x = X(u, w)``, piecewise affine in ``u``,
  that carries the fixed ``u`` walls onto the physical walls of a linear
  taper at height ``w`` (each ridge narrows about its own centre, the groove
  widens symmetrically);
* the transformation-optics coefficients of the VIRTUAL medium in ``(u, w)``
  (scalar TE: ``E_y``; scalar TM: ``H_y``)::

      Q_uu = (1 + S^2) / X_u ,  Q_uw = Q_wu = -S ,  Q_ww = X_u ,  m = X_u eps
      S = X_w (the tilt field) ,  X_u (the in-plane stretch)

  (TM divides ``Q`` by ``eps`` and uses ``m = X_u``);
* per slab, the coefficients FROZEN at the slab midpoint (the exponential
  midpoint rule) and the modal problem written in CONSERVATIVE form (no
  ``dQ_ww/dw`` term inside a slab: the w-variation of the virtual medium is
  carried by the slab-to-slab steps, as for any medium varying in w);
* slab-to-slab coupling by a SQUARE match of the virtual tangential fields
  (``E`` nodal and the flux dual ``<v|F>``, ``F = Q_wu d_u E + Q_ww d_w E``),
  because every slab shares the same ``u`` grid;
* half-spaces = the uniform medium under the FACE map (``w = 0`` or ``w = h``)
  with no tilt; the far field is the pulled-back Rayleigh projection
  ``a_m = (1/p) INT E(X(u)) exp(-i k_m X(u)) X_u du``.

Switches used by the probes as engineered defects / comparisons:
``tilt=False`` drops the tilt field (S := 0 inside the slabs);
``m5=True`` adds the non-conservative ``i beta <v|S'|phi>`` term of the M5
prototype (``validation/m5_covariant_taper.py``) to the slab pencil.
"""
from __future__ import annotations

import json
import os
import platform
import sys

import numpy as np
import scipy.linalg as sla

_C = np.complex128


def assert_tree():
    """Every probe asserts that it measures the worktree it lives in."""
    import lumenairy
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    lf = os.path.abspath(lumenairy.__file__)
    if not lf.startswith(os.path.abspath(here)):
        raise SystemExit(f"lumenairy imported from {lf}, expected under {here}"
                         " -- set PYTHONPATH to the worktree")
    return dict(lumenairy=lf, version=lumenairy.__version__,
                python=sys.version.split()[0], numpy=np.__version__,
                platform=platform.platform(),
                omp=os.environ.get("OMP_NUM_THREADS"))


def dump(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: (
            [o.real, o.imag] if isinstance(o, complex) else str(o)))


# ---------------------------------------------------------------------------
# reference element
# ---------------------------------------------------------------------------
def _ref(degree, nq):
    from lumenairy.elements.pmm._core import _gll_nodes_weights, _lagrange_derivative_matrix
    xi, _w = _gll_nodes_weights(degree)
    xi = np.asarray(xi, dtype=float)
    D = np.asarray(_lagrange_derivative_matrix(xi), dtype=float)
    gq, gw = np.polynomial.legendre.leggauss(nq)
    # barycentric Lagrange values at the Gauss points
    bw = np.array([1.0 / np.prod([xi[j] - xi[k] for k in range(len(xi))
                                  if k != j]) for j in range(len(xi))])
    B = np.zeros((nq, len(xi)))
    for i, x in enumerate(gq):
        d = x - xi
        hit = np.where(np.abs(d) < 1e-15)[0]
        if hit.size:
            B[i, hit[0]] = 1.0
            continue
        t = bw / d
        B[i] = t / t.sum()
    dB = B @ D            # derivative of the interpolant in xi
    return xi, gq, gw, B, dB


class Mesh1D:
    """Periodic C0 SEM on fixed ``u`` walls (``walls`` includes 0 and p)."""

    def __init__(self, period, walls, eps_regions, degree, els_per_region=1,
                 kx=0.0, nq=None):
        self.p = float(period)
        self.degree = int(degree)
        nq = nq or degree + 6
        self.xi, self.gq, self.gw, self.B, self.dB = _ref(degree, nq)
        els = []
        for i in range(len(walls) - 1):
            a, b = walls[i], walls[i + 1]
            for e in range(els_per_region):
                els.append((a + (b - a) * e / els_per_region,
                            a + (b - a) * (e + 1) / els_per_region,
                            i, eps_regions[i]))
        self.els = els
        n_el = len(els)
        l2g = np.zeros((n_el, degree + 1), dtype=int)
        ph = np.ones((n_el, degree + 1), dtype=_C)
        gid = 0
        for e in range(n_el):
            for a in range(degree + 1):
                if a == 0 and e > 0:
                    l2g[e, a] = l2g[e - 1, degree]
                else:
                    l2g[e, a] = gid
                    gid += 1
        last = l2g[n_el - 1, degree]
        l2g[l2g == last] = 0
        ph[n_el - 1, degree] = np.exp(1j * kx * self.p)   # Bloch glue
        self.n = int(last)
        self.l2g, self.ph = l2g, ph
        self.kx = float(kx)

    def assemble(self, local):
        """``local(e, a, b, region, eps) -> (nq x nq) weighted matrix`` blocks
        summed into the global matrix with the Bloch phases."""
        n = self.n
        G = np.zeros((n, n), dtype=_C)
        for e, (a, b, reg, eps) in enumerate(self.els):
            Ae = local(e, a, b, reg, eps)
            P = self.ph[e]
            idx = self.l2g[e]
            G[np.ix_(idx, idx)] += (np.conj(P)[:, None] * Ae) * P[None, :]
        return G

    def field_at_quad(self, coef):
        """Nodal coefficients -> values at every element's Gauss points."""
        out = []
        for e in range(len(self.els)):
            out.append(self.B @ (self.ph[e] * coef[self.l2g[e]]))
        return out


# ---------------------------------------------------------------------------
# the linear-taper map
# ---------------------------------------------------------------------------
class TaperMap:
    """``x = X(u, w)``: the fixed ``u`` walls ``uw`` go to the physical walls
    ``xw(w)``; between walls the map is affine in ``u`` (so ``X_u`` is a
    constant per region and the tilt ``S = X_w`` is linear in ``u``).

    ``xw_of(w)`` returns the physical wall array (0 and p included, fixed)."""

    def __init__(self, uw, xw_of, h, dxw_of=None):
        self.uw = np.asarray(uw, float)
        self.xw_of = xw_of
        self.dxw_of = dxw_of            # analytic d(walls)/dw, if given
        self.h = float(h)

    def region_geom(self, reg, w, dw=1e-7):
        u0, u1 = self.uw[reg], self.uw[reg + 1]
        x = np.asarray(self.xw_of(w), float)
        if self.dxw_of is not None:
            xp = np.asarray(self.dxw_of(w), float)
        else:
            xp = (np.asarray(self.xw_of(w + dw), float)
                  - np.asarray(self.xw_of(w - dw), float)) / (2 * dw)
        Xu = (x[reg + 1] - x[reg]) / (u1 - u0)
        return u0, u1, x[reg], x[reg + 1], Xu, xp[reg], xp[reg + 1]

    def X_at(self, reg, w, u):
        u0, u1, x0, x1, Xu, _s0, _s1 = self.region_geom(reg, w)
        return x0 + (u - u0) * Xu, Xu

    def S_at(self, reg, w, u):
        u0, u1, _x0, _x1, _Xu, s0, s1 = self.region_geom(reg, w)
        return s0 + (u - u0) / (u1 - u0) * (s1 - s0), (s1 - s0) / (u1 - u0)


# ---------------------------------------------------------------------------
# region operators and modes
# ---------------------------------------------------------------------------
def region_ops(mesh, tmap, w, pol, k0, *, eps_override=None, tilt=True,
               m5=False):
    """The frozen-slab (or half-space) operators at height ``w``.

    Returns dict with A0, A1 (linear), A2 (= Mww) and the flux pieces
    (C2f, Mwwf) such that the flux dual is ``f = -C2f phi + i beta Mwwf phi``.
    ``eps_override``: a scalar -> a uniform medium (half-space / film)."""
    gw = mesh.gw
    B, dB = mesh.B, mesh.dB
    gq = mesh.gq

    def coeffs(a, b, reg, eps):
        J = 0.5 * (b - a)
        u = a + (gq + 1.0) * J
        _X, Xu = tmap.X_at(reg, w, u)
        if tilt:
            S, dS = tmap.S_at(reg, w, u)
        else:                       # half-spaces and the NOTILT defect
            S, dS = np.zeros_like(u), 0.0
        e = eps if eps_override is None else eps_override
        if pol == "te":
            inv = 1.0
            mcoef = Xu * e
        else:
            inv = 1.0 / e
            mcoef = Xu * 1.0
        Quu = inv * (1.0 + S * S) / Xu
        Qwu = inv * (-S)
        Qww = inv * Xu * np.ones_like(u)
        return J, Quu, Qwu, Qww, mcoef * np.ones_like(u), inv * dS * np.ones_like(u)

    cache = {}

    def get(e, a, b, reg, eps):
        if e not in cache:
            cache[e] = coeffs(a, b, reg, eps)
        return cache[e]

    def Kuu(e, a, b, reg, eps):
        J, Quu, *_ = get(e, a, b, reg, eps)
        return dB.T @ ((gw * Quu / J)[:, None] * dB)

    def Mww(e, a, b, reg, eps):
        J, _Quu, _Qwu, Qww, *_ = get(e, a, b, reg, eps)
        return B.T @ ((gw * J * Qww)[:, None] * B)

    def Mm(e, a, b, reg, eps):
        J, _Quu, _Qwu, _Qww, mc, _d = get(e, a, b, reg, eps)
        return B.T @ ((gw * J * mc)[:, None] * B)

    def C2(e, a, b, reg, eps):          # INT v (-Q_wu) phi'  = INT v S phi'
        J, _Quu, Qwu, *_ = get(e, a, b, reg, eps)
        return B.T @ ((gw * (-Qwu))[:, None] * dB)

    def Msp(e, a, b, reg, eps):         # INT v S' phi  (the M5 term)
        J, _Quu, _Qwu, _Qww, _mc, dS = get(e, a, b, reg, eps)
        return B.T @ ((gw * J * dS)[:, None] * B)

    K = mesh.assemble(Kuu)
    M2 = mesh.assemble(Mww)
    Mmass = mesh.assemble(Mm)
    C2m = mesh.assemble(C2)
    C1m = C2m.conj().T
    A0 = k0 * k0 * Mmass - K
    A1 = 1j * (C1m - C2m)
    if m5:
        A1 = A1 + 1j * mesh.assemble(Msp)
    return dict(A0=A0, A1=A1, A2=M2, C2f=C2m, Mwwf=M2)


def modes(ops, n):
    """Companion eig of ``A0 + beta A1 - beta^2 A2``; split by flux."""
    I = np.eye(n, dtype=_C)
    Z = np.zeros((n, n), dtype=_C)
    A = np.block([[Z, I], [ops["A0"], ops["A1"]]])
    Bm = np.block([[I, Z], [Z, ops["A2"]]])
    beta, X = sla.eig(A, Bm)
    phi = X[:n]
    nrm = np.linalg.norm(phi, axis=0)
    phi = phi / np.where(nrm < 1e-300, 1.0, nrm)
    f = -ops["C2f"] @ phi + 1j * (ops["Mwwf"] @ phi) * beta[None, :]
    P = np.imag(np.einsum("in,in->n", np.conj(phi), f))
    bmax = max(float(np.max(np.abs(beta))), 1.0)
    prop = np.abs(beta.imag) < 1e-9 * bmax
    fwd = np.where(prop, P > 0.0, beta.imag > 0.0)
    fi = np.where(fwd)[0]
    if fi.size != n:
        sc = np.where(prop, P / max(np.max(np.abs(P)), 1e-300),
                      beta.imag / max(np.max(np.abs(beta.imag)), 1e-300))
        fi = np.argsort(-sc)[:n]
    bi = np.array(sorted(set(range(2 * n)) - set(fi.tolist())), dtype=int)
    return dict(beta=beta, Wf=phi[:, fi], Vf=f[:, fi], bf=beta[fi],
                Wb=phi[:, bi], Vb=f[:, bi], bb=beta[bi], nprop=int(prop.sum()),
                n_fwd_raw=int(np.sum(fwd)))


# ---------------------------------------------------------------------------
# S-matrix cascade
# ---------------------------------------------------------------------------
def s_interface(ma, mb):
    n = ma["Wf"].shape[0]
    L = np.block([[ma["Wb"], -mb["Wf"]], [ma["Vb"], -mb["Vf"]]])
    R = np.block([[-ma["Wf"], mb["Wb"]], [-ma["Vf"], mb["Vb"]]])
    S = np.linalg.solve(L, R)
    return S[:n, :n], S[:n, n:], S[n:, :n], S[n:, n:]


def s_layer(m, d, k0=None):
    Pf = np.diag(np.exp(1j * m["bf"] * d))
    Pb = np.diag(np.exp(-1j * m["bb"] * d))
    n = Pf.shape[0]
    Z = np.zeros((n, n), dtype=_C)
    return Z, Pb, Pf, Z


def star(A, B):
    A11, A12, A21, A22 = A
    B11, B12, B21, B22 = B
    n = A11.shape[0]
    I = np.eye(n, dtype=_C)
    X = np.linalg.solve(I - B11 @ A22, np.eye(n, dtype=_C))
    Y = np.linalg.solve(I - A22 @ B11, np.eye(n, dtype=_C))
    S11 = A11 + A12 @ X @ B11 @ A21
    S12 = A12 @ X @ B12
    S21 = B21 @ Y @ A21
    S22 = B22 + B21 @ Y @ A22 @ B12
    return S11, S12, S21, S22


# ---------------------------------------------------------------------------
# far field
# ---------------------------------------------------------------------------
def project(mesh, tmap, w, coef, orders, kx):
    vals = mesh.field_at_quad(coef)
    a = np.zeros(len(orders), dtype=_C)
    for e, (ea, eb, reg, _eps) in enumerate(mesh.els):
        J = 0.5 * (eb - ea)
        u = ea + (mesh.gq + 1.0) * J
        X, Xu = tmap.X_at(reg, w, u)
        for k, m in enumerate(orders):
            km = kx + 2 * np.pi * m / mesh.p
            a[k] += np.sum(mesh.gw * J * Xu * vals[e] * np.exp(-1j * km * X))
    return a / mesh.p


def solve_taper(*, period, uw, xw_of, h, eps_regions, eps_sup, eps_sub, wl,
                pol, degree, K, theta=0.0, els_per_region=1, tilt=True,
                m5=False, orders=(-2, -1, 0, 1, 2), slab_w=None, nq_far=None,
                eps_film=None, slab_mid=None, dxw_of=None):
    """Route B on a 1-D taper: K equal slabs, midpoint-frozen, square-matched.

    ``eps_film``: a scalar -> the layer is a UNIFORM film under the same map
    (the Airy null test).  ``slab_w`` (K + 1 edges) and ``slab_mid`` (K
    freezing heights) override the equal slabs and their midpoints (graded
    slabs for a rounded rim).  Returns a dict of R, T per order, r00, t00,
    closure."""
    k0 = 2 * np.pi / wl
    n_sup = np.sqrt(eps_sup + 0j)
    kx = float(np.real(n_sup) * k0 * np.sin(theta))
    mesh = Mesh1D(period, uw, eps_regions, degree, els_per_region, kx=kx,
                  nq=nq_far)
    tmap = TaperMap(uw, xw_of, h, dxw_of)
    n = mesh.n
    top = modes(region_ops(mesh, tmap, 0.0, pol, k0, eps_override=eps_sup,
                           tilt=False), n)
    bot = modes(region_ops(mesh, tmap, h, pol, k0, eps_override=eps_sub,
                           tilt=False), n)
    if slab_w is None:
        edges = np.linspace(0.0, h, K + 1)
    else:
        edges = np.asarray(slab_w, float)
    S = None
    prev = top
    for k in range(len(edges) - 1):
        wm = (0.5 * (edges[k] + edges[k + 1]) if slab_mid is None
              else float(slab_mid[k]))
        d = edges[k + 1] - edges[k]
        ops = region_ops(mesh, tmap, wm, pol, k0, eps_override=eps_film,
                         tilt=tilt, m5=m5)
        m = modes(ops, n)
        Si = s_interface(prev, m)
        S = Si if S is None else star(S, Si)
        S = star(S, s_layer(m, d))
        prev = m
    Si = s_interface(prev, bot)
    S = Si if S is None else star(S, Si)
    S11, _S12, S21, _S22 = S
    # incident: the pulled-back plane wave at w = 0 on the top half-space's
    # forward modes (nodal interpolation; exact at normal incidence)
    unodes = _global_nodes(mesh)
    Xn = np.array([tmap.X_at(_region_of(mesh, u), 0.0, u)[0] for u in unodes])
    Einc = np.exp(1j * kx * Xn)
    c = np.linalg.solve(top["Wf"], Einc)
    r_nodal = top["Wb"] @ (S11 @ c)
    t_nodal = bot["Wf"] @ (S21 @ c)
    orders = list(orders)
    ra = project(mesh, tmap, 0.0, r_nodal, orders, kx)
    ta = project(mesh, tmap, h, t_nodal, orders, kx)
    kzi = np.sqrt(eps_sup * k0 * k0 - kx * kx + 0j)
    R = np.zeros(len(orders))
    T = np.zeros(len(orders))
    for i, m_ in enumerate(orders):
        km = kx + 2 * np.pi * m_ / period
        kzs = np.sqrt(eps_sup * k0 * k0 - km * km + 0j)
        kzb = np.sqrt(eps_sub * k0 * k0 - km * km + 0j)
        if pol == "te":
            R[i] = abs(ra[i]) ** 2 * np.real(kzs) / np.real(kzi)
            T[i] = abs(ta[i]) ** 2 * np.real(kzb) / np.real(kzi)
        else:
            R[i] = (abs(ra[i]) ** 2 * np.real(kzs / eps_sup)
                    / np.real(kzi / eps_sup))
            T[i] = (abs(ta[i]) ** 2 * np.real(kzb / eps_sub)
                    / np.real(kzi / eps_sup))
    i0 = orders.index(0)
    return dict(orders=orders, R=R.tolist(), T=T.tolist(),
                r00=complex(ra[i0]), t00=complex(ta[i0]),
                closure=float(abs(1.0 - R.sum() - T.sum())), n=n)


def _global_nodes(mesh):
    u = np.zeros(mesh.n)
    for e, (a, b, _reg, _eps) in enumerate(mesh.els):
        J = 0.5 * (b - a)
        loc = a + (mesh.xi + 1.0) * J
        for k, g in enumerate(mesh.l2g[e]):
            if not (e == len(mesh.els) - 1 and k == mesh.degree):
                u[g] = loc[k]
    return u


def _region_of(mesh, u):
    for (a, b, reg, _eps) in mesh.els:
        if a - 1e-14 <= u <= b + 1e-14:
            return reg
    return mesh.els[-1][2]


def obs_vec(r):
    """The comparison vector: every R and T plus the complex r00, t00."""
    return np.concatenate([np.asarray(r["R"]), np.asarray(r["T"]),
                           [r["r00"].real, r["r00"].imag,
                            r["t00"].real, r["t00"].imag]])


def dist(a, b):
    va, vb = obs_vec(a), obs_vec(b)
    nR = len(a["R"]) + len(a["T"])
    return dict(eff=float(np.max(np.abs(va[:nR] - vb[:nR]))),
                amp=float(np.max(np.abs(va[nR:] - vb[nR:]))))
