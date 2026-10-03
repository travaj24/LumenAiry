"""Shared fixture of the INDEPENDENT verification of Phase A (curved-cell map).

Verifier's own fixture (NOT the build's): lambda = 1.3, square period 1.0,
depth 0.4, air above, n = 1.52 below, features eps = 6.25 (n = 2.5).
``stripe`` = y-uniform ridge x in [0.20, 0.65] (exact oracle
``pmm_efficiency_1d``, stabilize=False, self-gap measured between two
degrees); ``pillar`` = x in [0.20, 0.65], y in [0.25, 0.70];
``film`` = uniform eps 6.25 (exact oracle: the Airy slab, s and p).

Run with BLAS pinned ON THE COMMAND LINE and PYTHONPATH at the tree under
test; ``VA_ROOT`` names that tree and is ASSERTED against
``lumenairy.__file__`` (default: this worktree; ``/mnt/c/...`` under WSL).
"""
import json
import os
import platform
import sys
import time

import numpy as np

import lumenairy

_DEF = (r"C:\tmp\lum_vcurved_a" if os.name == "nt"
        else "/mnt/c/tmp/lum_vcurved_a")
ROOT = os.path.normcase(os.path.abspath(os.environ.get("VA_ROOT", _DEF)))
assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(ROOT), (
    f"lumenairy imported from {lumenairy.__file__}, not {ROOT}")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    assert os.environ.get(_v) == "1", f"{_v} must be 1 on the command line"

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = "win" if os.name == "nt" else "wsl"
P = 1.0
WL = 1.3
DEPTH = 0.4
N_SUP, N_SUB = 1.0, 1.52
EPS_F = 6.25 + 0j
XW = np.array([0.0, 0.20, 0.65, P])
YW = np.array([0.0, 0.25, 0.70, P])
K0 = 2 * np.pi / WL


def env():
    import scipy
    return dict(build=BUILD, python=sys.version.split()[0],
                numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(), lumenairy=lumenairy.__file__,
                date=time.strftime("%Y-%m-%d %H:%M"))


def dump(name, payload):
    payload = dict(payload)
    payload["env"] = env()
    fn = os.path.join(HERE, f"{name}_{BUILD}.json")
    with open(fn, "w") as f:
        json.dump(payload, f, indent=1, default=_js)
    print("wrote", fn)
    return fn


def _js(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    raise TypeError(type(o))


def cell(kind, eps=EPS_F, n=3):
    c = np.ones((n, n), complex)
    if kind == "stripe":
        c[1, :] = eps
    elif kind == "pillar":
        c[1, 1] = eps
    elif kind == "film":
        c[:] = eps
    elif kind == "vac":
        pass
    else:
        raise ValueError(kind)
    return c


# ----------------------------------------------------------------- stretches
class HarmonicStretch:
    """VERIFIER's 1-D stretch (duck-types SineStretch): two harmonics and a
    phase, so f has NO mirror symmetry:
        f(u) = u + a1 sin(k u) + a2 [sin(2 k u + phi) - sin(phi)]
    f(0) = 0, f(u + p) = f(u) + p.  ``a1``, ``a2`` in units of the period."""

    def __init__(self, a1, a2=0.0, phi=0.0):
        self.a1, self.a2, self.phi = float(a1), float(a2), float(phi)

    def __call__(self, u, period):
        k = 2 * np.pi / period
        u = np.asarray(u, float)
        a1, a2 = self.a1 * period, self.a2 * period
        f = u + a1 * np.sin(k * u) + a2 * (np.sin(2 * k * u + self.phi)
                                           - np.sin(self.phi))
        fp = 1 + a1 * k * np.cos(k * u) + 2 * a2 * k * np.cos(2 * k * u
                                                               + self.phi)
        return f, fp

    def min_slope(self, period=1.0):
        u = np.linspace(0, period, 200001)
        return float(np.min(self(u, period)[1]))

    def check(self, period, axis):
        if not self.min_slope(period) > 0:
            raise ValueError(f"HarmonicStretch on {axis} folds the plane")

    def inverse(self, x, period):
        x = np.asarray(x, float)
        u = x.copy()
        for _ in range(200):
            f, fp = self(u, period)
            du = (f - x) / fp
            u = u - du
            if float(np.max(np.abs(du), initial=0.0)) <= 1e-16 * period:
                break
        return u

    def key(self):
        return ("HarmonicStretch", self.a1, self.a2, self.phi)


def stretch_map(fx, x_walls=XW, y_walls=YW, fy=None):
    from lumenairy.elements.pmm._curvemap import SeparableStretch
    return SeparableStretch.from_physical_walls(x_walls, y_walls, fx=fx,
                                                fy=fy)


def sine(a_frac):
    from lumenairy.elements.pmm._curvemap import SineStretch
    return SineStretch(a_frac * P)


def make_shear_map(cx, cy, n=2, Fv=None, Gu=None, walls=None):
    """VERIFIER's NON-SEPARABLE, SHEARED map on the (u, v) grid with walls at
    0, p/2, p (n = 2) -- the walls stay STRAIGHT because sin(k u) vanishes on
    every one of them, while the map shears every cell interior:
        x = u + cx sin(k u) F(v),   y = v + cy sin(k v) G(u)
    F, G periodic (default F(v) = sin(k v) + 0.4 cos(2 k v), G(u) = cos(k u)
    + 0.3 sin(2 k u): no mirror symmetry).  So g12 != 0 inside every cell,
    the mixed tensor blocks (e12, e21, chi12, chi21) and the off-diagonal
    cofactor blocks (P12, P21) are all live, and the PHYSICAL device is still
    the straight-walled one the (u, v) cells describe.  ``cx``, ``cy`` in
    units of the period."""
    from lumenairy.elements.pmm._curvemap import CellMap
    k = 2 * np.pi / P
    if Fv is None:
        def Fv(v):
            return (np.sin(k * v) + 0.4 * np.cos(2 * k * v),
                    k * np.cos(k * v) - 0.8 * k * np.sin(2 * k * v))
    if Gu is None:
        def Gu(u):
            return (np.cos(k * u) + 0.3 * np.sin(2 * k * u),
                    -k * np.sin(k * u) + 0.6 * k * np.cos(2 * k * u))
    cxa, cya = cx * P, cy * P

    class ShearMap(CellMap):
        def __init__(self):
            w = (np.linspace(0, P, n + 1) if walls is None
                 else np.asarray(walls, float))
            self._init_walls(w, w, P, P)
            self.validate()

        def geom(self, sx, sy, U, V):
            U = np.asarray(U, float)[:, None]
            V = np.asarray(V, float)[None, :]
            F, Fp = Fv(V)
            G, Gp = Gu(U)
            X = U + cxa * np.sin(k * U) * F
            Y = V + cya * np.sin(k * V) * G
            xu = 1 + cxa * k * np.cos(k * U) * F
            xv = cxa * np.sin(k * U) * Fp
            yu = cya * np.sin(k * V) * Gp
            yv = 1 + cya * k * np.cos(k * V) * G
            shp = (U.size, V.size)
            return tuple(np.broadcast_to(t, shp).astype(float)
                         for t in (X, Y, xu, xv, yu, yv))

        def _key(self):
            return ("ShearMap", cx, cy, n)
    return ShearMap()


def detj_range(cmap, n=41):
    lo, hi = np.inf, -np.inf
    Nx, Ny = cmap.shape
    xg = np.linspace(-0.999, 0.999, n)
    for sx in range(Nx):
        U = 0.5 * (cmap.u_bounds[sx] + cmap.u_bounds[sx + 1]) + 0.5 * (
            cmap.u_bounds[sx + 1] - cmap.u_bounds[sx]) * xg
        for sy in range(Ny):
            V = 0.5 * (cmap.v_bounds[sy] + cmap.v_bounds[sy + 1]) + 0.5 * (
                cmap.v_bounds[sy + 1] - cmap.v_bounds[sy]) * xg
            _X, _Y, xu, xv, yu, yv = cmap.geom(sx, sy, U, V)
            d = xu * yv - xv * yu
            lo, hi = min(lo, float(d.min())), max(hi, float(d.max()))
    return lo, hi


def airy(theta=0.0, eps=EPS_F, d=DEPTH, wl=WL, n1=N_SUP, n3=N_SUB):
    """Exact (R, T) of a slab for s and p at polar angle theta in medium 1."""
    k0 = 2 * np.pi / wl
    kx = n1 * np.sin(theta)
    out = {}
    e1, e2, e3 = n1 ** 2, complex(eps), n3 ** 2
    kz1 = np.sqrt(e1 - kx ** 2 + 0j)
    kz2 = np.sqrt(e2 - kx ** 2 + 0j)
    kz3 = np.sqrt(e3 - kx ** 2 + 0j)
    for pol in ("s", "p"):
        if pol == "s":
            a1, a2, a3 = kz1, kz2, kz3
        else:
            a1, a2, a3 = kz1 / e1, kz2 / e2, kz3 / e3
        r12 = (a1 - a2) / (a1 + a2)
        r23 = (a2 - a3) / (a2 + a3)
        t12 = 2 * a1 / (a1 + a2)
        t23 = 2 * a2 / (a2 + a3)
        ph = np.exp(1j * kz2 * k0 * d)
        den = 1 + r12 * r23 * ph ** 2
        r = (r12 + r23 * ph ** 2) / den
        t = t12 * t23 * ph / den
        R = abs(r) ** 2
        T = abs(t) ** 2 * (a3.real / a1.real)
        out[pol] = (float(R), float(T))
    return out


def stack_solve(cmap, cells, M, *, theta=0.0, phi=0.0, retain=False,
                n_orders=3, depths=None, quiet=True):
    """One PMM2DStackPure solve.  ``cells`` = list of (Nx, Ny) eps arrays (or
    scalars for uniform layers); returns (orders, R, T, J, stack)."""
    import warnings

    from lumenairy.elements.pmm import PMM2DStackPure
    kw = {} if cmap is None else {"cmap": cmap}
    st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                        n_modes=M, n_orders=n_orders, **kw)
    if depths is None:
        depths = [DEPTH] * len(cells)
    for c, d in zip(cells, depths):
        if np.ndim(c) == 0:
            st.add_layer(d, eps=c)
        else:
            st.add_layer(d, eps_cell=c)
    st.set_source(WL, theta=theta, phi=phi)
    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("ignore")
        o, R, T, J = st.solve(retain_internal=retain)
    return o, R, T, J, st


def ref_perlayer(cells, M, x_walls, y_walls, *, theta=0.0, phi=0.0,
                 n_orders=3, depths=None):
    """UNMAPPED reference on NON-UNIFORM physical walls: a per-layer stack
    (the shared-grid stack takes a uniform lattice only)."""
    import warnings

    from lumenairy.elements.pmm import PMM2DStackPure
    st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                        n_modes=M, n_orders=n_orders,
                        layer_grids="per-layer")
    if depths is None:
        depths = [DEPTH] * len(cells)
    for c, d in zip(cells, depths):
        st.add_layer(d, eps_cell=c, x_walls=list(x_walls[1:-1]),
                     y_walls=list(y_walls[1:-1]))
    st.set_source(WL, theta=theta, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve()
    return o, R, T, J, st


def i00(o):
    return int(np.nonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])


_ORACLE = {}


def oracle_1d(pol, x_fill=0.45, theta=0.0, degree=40, eps=EPS_F):
    """{m: (R, T)} for the y-uniform stripe from pmm_efficiency_1d
    (stabilize=False: a clean single-degree solve)."""
    key = (pol, x_fill, theta, degree, complex(eps))
    if key in _ORACLE:
        return _ORACLE[key]
    from lumenairy.elements.pmm.oned import pmm_efficiency_1d
    o, R, T = pmm_efficiency_1d(P, np.sqrt(complex(eps)), 1.0, N_SUB, N_SUP,
                                DEPTH, x_fill, WL, polarization=pol,
                                degree=degree, far_field_orders=9,
                                theta=theta, stabilize=False)
    d = {int(m): (float(np.real(R[i])), float(np.real(T[i])))
         for i, m in enumerate(np.asarray(o))}
    _ORACLE[key] = d
    return d


def stripe_err(o, R, T, row, ref):
    """max |R - R_ref|, |T - T_ref| over every returned order (orders with
    m_y != 0 compared against 0)."""
    e = 0.0
    for i, (mx, my) in enumerate(o):
        r0, t0 = ref.get(int(mx), (0.0, 0.0)) if my == 0 else (0.0, 0.0)
        e = max(e, abs(R[row, i] - r0), abs(T[row, i] - t0))
    return float(e)


def closure(R, T):
    return float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1)))
