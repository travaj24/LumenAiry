"""Shared fixtures of the INDEPENDENT Phase E1 verifier probes (out-of-plane
tensors and slanted walls inside curved cells).

Pins ``lumenairy`` to the tree being measured (this worktree, or the tree
named by ``LUM_TREE`` for a BEFORE arm) and ASSERTS it.  Run every probe with
BLAS pinned on the command line:

  cd /c/tmp/lum_vcurved_e1/validation/probe_pmm2d_curved/verify_e1 && \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=C:/tmp/lum_vcurved_e1 python <probe>.py ...

Everything here is the verifier's OWN, chosen independently of the builder's
``_e1common.py``:

* tensors -- a uniaxial director in a GENERAL direction (polar 52 deg,
  azimuth 37 deg: no zero entry), its non-reciprocal (magneto-optic) twin,
  a LOSSY GYROTROPIC eps rotated by three Euler angles (fully populated,
  non-symmetric, out of plane), a lossy gyrotropic mu, a lossless gyrotropic
  mu, and a fully populated mu WITH an out-of-plane block (must be refused);
* maps -- an ASYMMETRIC two-harmonic separable stretch (not 180-degree
  symmetric), a 4 x 4 straight-edged transfinite map with five interior
  vertices moved (non-diagonal J, no symmetry), the 3 x 3 circle map
  centred and OFF-centre, the 5 x 5 circle map;
* the oracle -- ``eps_mu_slab``: an (eps, mu) Berreman slab solved by its
  EIGEN-decomposition with a z-referenced 8 x 8 interface system (no matrix
  exponential) and ANALYTIC s / p half-space modes; validated against the
  shipped ``berreman_jones_1d`` at mu = I and against the Airy formula of an
  isotropic (eps, mu) slab before use (``v0_oracle.py``).
"""
import json
import os
import sys
import time
import warnings
from contextlib import contextmanager

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normcase(os.path.abspath(
    os.environ.get("LUM_TREE") or os.path.join(HERE, "..", "..", "..")))
if ROOT not in [os.path.normcase(os.path.abspath(p)) for p in sys.path]:
    sys.path.insert(0, ROOT)
import lumenairy  # noqa: E402

assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(ROOT), (
    f"lumenairy imported from {lumenairy.__file__}, not {ROOT}")

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402,F401
from lumenairy.elements.pmm import (  # noqa: E402,F401
    Circle,
    PMM2DStackPure,
    Rect,
    _curvemap as CM,
    stack2d_pure as SP,
    twod_staggered as TS,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

TREE = "pre" if os.environ.get("LUM_TREE") else "post"
ENV = {"python": sys.version.split()[0], "numpy": np.__version__,
       "lumenairy": lumenairy.__file__,
       "threads": {k: os.environ.get(k) for k in
                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS")}}
EYE = np.eye(3, dtype=complex)


def dump(name, obj):
    obj = dict(obj)
    obj.setdefault("env", ENV)
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(obj, f, indent=1, default=_js)


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def _js(o):
    if isinstance(o, np.ndarray):
        if np.iscomplexobj(o):
            return {"re": o.real.tolist(), "im": o.imag.tolist()}
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


# --------------------------------------------------------------------------- #
# tensors
# --------------------------------------------------------------------------- #
def rot3(a, b, c):
    """Z-Y-Z Euler rotation."""
    def rz(t):
        return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0],
                         [0, 0, 1.0]])

    def ry(t):
        return np.array([[np.cos(t), 0, np.sin(t)], [0, 1.0, 0],
                         [-np.sin(t), 0, np.cos(t)]])
    return rz(a) @ ry(b) @ rz(c)


DIRGEN = uniaxial_tensor(1.52, 1.78, np.deg2rad(52.0), phi=np.deg2rad(37.0))
NRGEN = np.array(DIRGEN, dtype=complex)            # non-reciprocal twin
NRGEN[0, 2] = DIRGEN[0, 2] + 0.18j
NRGEN[2, 0] = np.conj(NRGEN[0, 2])
NRGEN[1, 2] = DIRGEN[1, 2] - 0.11j
NRGEN[2, 1] = np.conj(NRGEN[1, 2])
_G0 = np.array([[2.3 + 0.06j, 0.35j + 0.03, 0.0],
                [-0.35j + 0.03, 2.1 + 0.05j, 0.0],
                [0.0, 0.0, 2.6 + 0.04j]], complex)
_RG = rot3(0.5, 0.7, 0.3)
GYRO_EPS_LOSSY = _RG @ _G0 @ _RG.T                 # fully populated, OOP
MU_GYRO_LOSSY = np.array([[1.45 + 0.04j, 0.28j + 0.02, 0],
                          [-0.28j + 0.02, 1.3 + 0.03j, 0],
                          [0, 0, 1.15 + 0.02j]], complex)
MU_GYRO = np.array([[1.35, -0.25j, 0], [0.25j, 1.5, 0], [0, 0, 1.1]],
                   complex)
MU_ANISO = np.array([[1.4, 0.18, 0], [0.18, 1.25, 0], [0, 0, 1.3]], complex)
MU_SYM_LOSSY = np.array([[1.4 + 0.06j, 0.15 + 0.01j, 0],
                         [0.15 + 0.01j, 1.25 + 0.04j, 0],
                         [0, 0, 1.1 + 0.05j]], complex)
MU_FULL_OOP = np.array([[1.4, 0.1, 0.15], [0.1, 1.3, 0.12],
                        [0.15, 0.12, 1.2]], complex)
# the pillar director: 30 deg out of the plane (polar 60 deg), azimuth 30 deg
PIL_OOP = uniaxial_tensor(1.52, 1.78, np.deg2rad(60.0), phi=np.deg2rad(30.0))
PIL_NR = np.array(PIL_OOP, dtype=complex)
PIL_NR[0, 2] = PIL_OOP[0, 2] + 0.22j
PIL_NR[2, 0] = np.conj(PIL_NR[0, 2])

LCIN = uniaxial_tensor(1.52, 1.78, np.pi / 2, phi=np.deg2rad(37.0))
TENSORS = {"lc": LCIN, "dirgen": DIRGEN, "nrgen": NRGEN, "gyrol": GYRO_EPS_LOSSY,
           "piloop": PIL_OOP, "pilnr": PIL_NR}
MUS = {"none": None, "gyro": MU_GYRO, "gyrol": MU_GYRO_LOSSY,
       "aniso": MU_ANISO, "syml": MU_SYM_LOSSY}

# --------------------------------------------------------------------------- #
# maps
# --------------------------------------------------------------------------- #
class TwoHarm:
    """x = u + a1 sin(k u) + a2 sin(2 k u + ph): asymmetric, periodic."""

    def __init__(self, a1, a2, ph):
        self.a1, self.a2, self.ph = float(a1), float(a2), float(ph)

    def __call__(self, u, period):
        k = 2 * np.pi / float(period)
        u = np.asarray(u, dtype=float)
        return (u + self.a1 * np.sin(k * u) + self.a2 * np.sin(2 * k * u
                                                                + self.ph)
                - self.a2 * np.sin(self.ph),
                1.0 + self.a1 * k * np.cos(k * u)
                + 2 * k * self.a2 * np.cos(2 * k * u + self.ph))

    def check(self, period, axis):
        u = np.linspace(0, period, 4001)
        assert np.min(self(u, period)[1]) > 0, axis

    def inverse(self, x, period):
        x = np.asarray(x, dtype=float)
        u = x.copy()
        for _ in range(200):
            f, fp = self(u, period)
            du = (f - x) / fp
            u = u - du
            if float(np.max(np.abs(du), initial=0.0)) <= 1e-16 * period:
                break
        return u

    def key(self):
        return ("TwoHarm", self.a1, self.a2, self.ph)


def map_two_harm(P, n=3, big=False):
    s = 1.6 if big else 1.0
    fx = TwoHarm(0.07 * s * P, 0.025 * s * P, 0.9)
    fy = TwoHarm(-0.05 * s * P, 0.03 * s * P, -0.4)
    w = np.linspace(0.0, P, n + 1)
    return CM.SeparableStretch(w, w, fx=fx, fy=fy)


def map_shear4(P):
    """4 x 4 straight-edged transfinite map, five interior vertices moved
    (no symmetry; non-diagonal J in every interior cell)."""
    w = np.linspace(0.0, P, 5)
    V = np.stack(np.meshgrid(w, w, indexing="ij"), axis=-1)
    for (i, j), (dx, dy) in {(1, 1): (0.05, -0.03), (2, 1): (-0.04, 0.05),
                             (3, 2): (0.03, 0.04), (1, 3): (-0.05, -0.02),
                             (2, 2): (0.02, -0.06)}.items():
        V[i, j] += (dx * P, dy * P)
    return CM.TransfiniteMap(w, w, V, None)


def map_circle3(P, r=0.3, center=None):
    c = None if center is None else (center[0] * P, center[1] * P)
    return CM._circle_map_3x3(P, r * P, center=c)[0]


def map_circle5(P, r=0.3):
    return CM._circle_map_5x5(P, r * P)[0]


def make_map(name, P):
    if name == "none":
        return None
    if name == "th3":
        return map_two_harm(P, 3)
    if name == "th2b":
        return map_two_harm(P, 2, big=True)
    if name == "sh4":
        return map_shear4(P)
    if name == "c3":
        return map_circle3(P)
    if name == "c3off":
        return map_circle3(P, 0.27, center=(0.41, 0.56))
    if name == "c5":
        return map_circle5(P)
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# the (eps, mu) slab oracle (own formulation)
# --------------------------------------------------------------------------- #
SLAB = dict(P=0.8, WL=1.0, DEP=0.42, NSUB=1.52, NSUP=1.0)
MOUNTS = {"n": (0.0, 0.0), "o": (np.deg2rad(30.0), 0.0),
          "c": (np.deg2rad(22.0), np.deg2rad(63.0))}


def delta(eps, mu, Kx, Ky):
    """Berreman matrix, psi = (Ex, Ey, Hx, Hy) (H in units of 1/Z0),
    d psi / d(k0 z) = i Delta psi, exp(-i w t).  Ez and Hz from the two
    longitudinal equations (eps E)_z = Ky Hx - Kx Hy, (mu H)_z = Kx Ey - Ky
    Ex -- a FULL mu (out-of-plane block included) is allowed here."""
    e = np.asarray(eps, complex)
    m = np.asarray(mu, complex)
    D = np.zeros((4, 4), complex)
    # Ez = a . psi, Hz = b . psi
    a = np.array([-e[2, 0], -e[2, 1], Ky, -Kx]) / e[2, 2]
    b = np.array([-Ky, Kx, -m[2, 0], -m[2, 1]]) / m[2, 2]
    # E = SE psi, H = SH psi (3 x 4)
    SE = np.vstack([[1, 0, 0, 0], [0, 1, 0, 0], a])
    SH = np.vstack([[0, 0, 1, 0], [0, 0, 0, 1], b])
    eE = e @ SE
    mH = m @ SH
    D[0] = Kx * a + mH[1]
    D[1] = Ky * a - mH[0]
    D[2] = Kx * b - eE[1]
    D[3] = Ky * b + eE[0]
    return D


def _iso_modes(n, Kx, Ky):
    """Analytic forward / backward modes of an isotropic half-space: columns
    psi for (s, p), each (4, 2).  H = k x E (normalised units)."""
    n = complex(n)
    K = np.hypot(Kx, Ky)
    kz = np.sqrt(n * n - K * K + 0j)
    if kz.imag < 0:
        kz = -kz
    if K < 1e-14:
        cs, sn = 1.0, 0.0
    else:
        cs, sn = Kx / K, Ky / K
    out = []
    for sgn in (+1, -1):
        k = np.array([Kx, Ky, sgn * kz])
        es = np.array([-sn, cs, 0.0])
        ep = np.cross(k, es) / n          # p-polarised E, |E| = 1 (real k)
        cols = []
        for E in (es, ep):
            H = np.cross(k, E)
            cols.append(np.array([E[0], E[1], H[0], H[1]]))
        out.append(np.array(cols).T)
    return out[0], out[1], kz


def _flux(v):
    return float(np.real(v[0] * np.conj(v[3]) - v[1] * np.conj(v[2])))


def eps_mu_slab(eps, mu, theta, phi, f=SLAB):
    """(R[2], T[2], Jr, Jt) of one uniform (eps, mu) slab between isotropic
    half-spaces.  Inputs are unit TANGENTIAL E along x and along y (the
    convention of ``berreman_jones_1d``); Jones = tangential E out."""
    k0d = 2 * np.pi / f["WL"] * f["DEP"]
    Kx = f["NSUP"] * np.sin(theta) * np.cos(phi)
    Ky = f["NSUP"] * np.sin(theta) * np.sin(phi)
    mu = EYE if mu is None else mu
    D = delta(eps, mu, Kx, Ky)
    q, W = np.linalg.eig(D)
    fw = []
    for i in range(4):
        if abs(q[i].imag) > 1e-10:
            fw.append(q[i].imag > 0)
        else:
            fw.append(_flux(W[:, i]) > 0)
    fw = np.array(fw)
    assert fw.sum() == 2, (q, fw)
    Wf, Wb = W[:, fw], W[:, ~fw]
    qf, qb = q[fw], q[~fw]
    Pf = np.diag(np.exp(1j * qf * k0d))            # forward: 0 -> d
    Pb = np.diag(np.exp(-1j * qb * k0d))           # backward ref. at z = d
    U1f, U1b, _ = _iso_modes(f["NSUP"], Kx, Ky)
    U3f, _U3b, _ = _iso_modes(f["NSUB"], Kx, Ky)
    # unknowns x = (r[2], cf[2], cb[2], t[2])
    # z = 0:  U1f a + U1b r = Wf cf + Wb Pb cb
    # z = d:  Wf Pf cf + Wb cb = U3f t
    A = np.zeros((8, 8), complex)
    A[:4, 0:2] = U1b
    A[:4, 2:4] = -Wf
    A[:4, 4:6] = -Wb @ Pb
    A[4:, 2:4] = Wf @ Pf
    A[4:, 4:6] = Wb
    A[4:, 6:8] = -U3f
    R, T = np.zeros(2), np.zeros(2)
    Jr, Jt = np.zeros((2, 2), complex), np.zeros((2, 2), complex)
    for c in range(2):
        a = np.linalg.solve(U1f[:2], np.eye(2)[c])
        rhs = np.zeros(8, complex)
        rhs[:4] = -U1f @ a
        x = np.linalg.solve(A, rhs)
        vi, vr, vt = U1f @ a, U1b @ x[0:2], U3f @ x[6:8]
        Jr[:, c], Jt[:, c] = vr[:2], vt[:2]
        R[c] = -_flux(vr) / _flux(vi)
        T[c] = _flux(vt) / _flux(vi)
    return R, T, Jr, Jt


# --------------------------------------------------------------------------- #
# solver runners
# --------------------------------------------------------------------------- #
def jt_of(st):
    md = st._modal
    p0 = md["p0"]
    return np.array([[md["tx"][0][p0], md["tx"][1][p0]],
                     [md["ty"][0][p0], md["ty"][1][p0]]])


def slab_run(t33, cmap, M, mount="n", slant=None, mu=None, f=SLAB,
             n_orders=2, symmetry="auto"):
    """(dRT, dJr, dJt, closure, wall) of a uniform slab layer of a (possibly
    mapped) stack against the own oracle."""
    th, ph = MOUNTS[mount] if isinstance(mount, str) else mount
    t0 = time.perf_counter()
    st = PMM2DStackPure(f["P"], f["P"], n_superstrate=f["NSUP"],
                        n_substrate=f["NSUB"], n_modes=M, n_orders=n_orders,
                        cmap=cmap, symmetry=symmetry)
    st.add_layer(f["DEP"], eps=t33, slant=slant, mu=mu)
    st.set_source(f["WL"], theta=th, phi=ph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R, T, J = st.solve(jones=True)
    R, T = np.asarray(R), np.asarray(T)
    Rb, Tb, jr, jt = eps_mu_slab(t33, mu, th, ph, f)
    return dict(dRT=float(max(np.abs(R.sum(1) - Rb).max(),
                              np.abs(T.sum(1) - Tb).max())),
                dJr=float(np.abs(np.asarray(J) - jr).max()),
                dJt=float(np.abs(jt_of(st) - jt).max()),
                clo=float(np.abs(R.sum(1) + T.sum(1) - 1.0).max()),
                Jr=np.asarray(J), Jt=jt_of(st),
                wall=time.perf_counter() - t0)


def worst(r):
    return max(r["dRT"], r["dJr"], r["dJt"])


@contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


# --------------------------------------------------------------------------- #
# the pillar fixture (own): period 1.1, r 0.33, depth 0.45, air / 1.5
# --------------------------------------------------------------------------- #
PIL = dict(P=1.1, R0=0.33, DEP=0.45, WL=1.0, NSUB=1.5, NSUP=1.0)
ORD9 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1),
        (-1, -1)]


def idx(o, orders=ORD9):
    o = np.asarray(o)
    return [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0])
            for m, n in orders]


def vec36(o, R, T):
    i = idx(o)
    return np.concatenate([np.asarray(R)[:, i].ravel(),
                           np.asarray(T)[:, i].ravel()])


def disk_cell(kind, t33, f=PIL):
    P = f["P"]
    cm = (CM._circle_map_3x3(P, f["R0"])[0] if kind == "c3"
          else CM._circle_map_5x5(P, f["R0"])[0])
    N = cm.shape[0]
    eps = np.broadcast_to(EYE, (N, N, 3, 3)).copy()
    cells = ([(1, 1)] if N == 3 else
             [(i, j) for i in (1, 2, 3) for j in (1, 2, 3)])
    for c in cells:
        eps[c] = t33
    return cm, eps


def pillar_run(kind, t33, M, th=0.0, ph=0.0, slant=None, f=PIL,
               symmetry="auto"):
    cm, eps = disk_cell(kind, t33, f)
    t0 = time.perf_counter()
    st = PMM2DStackPure(f["P"], f["P"], n_superstrate=f["NSUP"],
                        n_substrate=f["NSUB"], n_modes=M, n_orders=3,
                        cmap=cm, symmetry=symmetry)
    st.add_layer(f["DEP"], eps_cell=eps, slant=slant)
    st.set_source(f["WL"], theta=th, phi=ph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve(jones=True)
    return st, np.asarray(o), np.asarray(R), np.asarray(T), \
        time.perf_counter() - t0


def stair_walls(k, f=PIL):
    c, r, P = f["P"] / 2, f["R0"], f["P"]
    inner = sorted([c - r * i / k for i in range(1, k + 1)]
                   + [c + r * i / k for i in range(1, k + 1)])
    return np.array([0.0] + inner + [P])


def stair_run(k, t33, M, th=0.0, ph=0.0, slant=None, f=PIL):
    w = stair_walls(k, f)
    c, r = f["P"] / 2, f["R0"]
    n = len(w) - 1
    mid = 0.5 * (w[:-1] + w[1:])
    eps = np.broadcast_to(EYE, (n, n, 3, 3)).copy()
    for i in range(n):
        for j in range(n):
            if (mid[i] - c) ** 2 + (mid[j] - c) ** 2 < r ** 2:
                eps[i, j] = t33
    t0 = time.perf_counter()
    st = PMM2DStackPure(f["P"], f["P"], n_superstrate=f["NSUP"],
                        n_substrate=f["NSUB"], n_modes=M, n_orders=3,
                        layer_grids="per-layer")
    st.add_layer(f["DEP"], eps_cell=eps, x_walls=w, y_walls=w, slant=slant)
    st.set_source(f["WL"], theta=th, phi=ph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve(jones=True)
    return st, np.asarray(o), np.asarray(R), np.asarray(T), \
        time.perf_counter() - t0


def jones_block(st, o, order):
    """Power-normalised 2 x 2 reflection Jones block of the channel
    (incidence -> reflected ``order``)."""
    md = st._modal
    k = idx(o, [order])[0]
    A = np.array([[md["rx"][c][k] for c in (0, 1)],
                  [md["ry"][c][k] for c in (0, 1)]])
    kx0, ky0, kzi = md["kx0"], md["ky0"], md["kz_inc"]
    kxo, kyo, kzo = md["kx"][k], md["ky"][k], float(np.real(md["kz_ref"][k]))
    Gin = np.eye(2) + np.outer([kx0, ky0], [kx0, ky0]) / kzi ** 2
    Wout = (kzo / kzi) * (np.eye(2) + np.outer([kxo, kyo], [kxo, kyo])
                          / kzo ** 2)

    def msqrt(S, inv=False):
        w, V = np.linalg.eigh(S)
        return (V * w ** (-0.5 if inv else 0.5)) @ V.conj().T
    return msqrt(Wout) @ A @ msqrt(Gin, inv=True)


def reverse_angles(th, ph, m, n, f=PIL):
    s = f["NSUP"] * np.sin(th)
    kx = -(s * np.cos(ph) + m * f["WL"] / f["P"])
    ky = -(s * np.sin(ph) + n * f["WL"] / f["P"])
    return (float(np.arcsin(np.hypot(kx, ky) / f["NSUP"])),
            float(np.arctan2(ky, kx)))
