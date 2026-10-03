"""Shared fixtures of the INDEPENDENT Phase D verifier probes (tensor and
magnetic materials inside curved cells).

Pins ``lumenairy`` to the tree being measured (this worktree, or the tree
named by ``LUM_TREE`` for a BEFORE arm) and ASSERTS it.  Run every probe with
BLAS pinned on the command line:

  cd /c/tmp/lum_vcurved_d/validation/probe_pmm2d_curved/verify_d && \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=C:/tmp/lum_vcurved_d python <probe>.py ...

Everything here is the verifier's OWN: its own maps (a 4 x 4 sheared
transfinite map with a different vertex pattern than the builder's, an
ASYMMETRIC two-harmonic stretch), its own tensors (LC at 0/30/45/90 deg, a
biaxial in-plane tensor, a REAL ASYMMETRIC non-Hermitian tensor, a lossy
tensor, a gyrotropic permeability) and its own Berreman 4x4 transfer matrix
with BOTH eps and mu (``berreman_eps_mu``), cross-checked against the shipped
``berreman_jones_1d`` on eps-only stacks before it is trusted with mu
(``v0_oracle.py``).
"""
import json
import os
import sys

import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normcase(os.path.abspath(
    os.environ.get("LUM_TREE") or os.path.join(HERE, "..", "..", "..")))
if ROOT not in [os.path.normcase(os.path.abspath(p)) for p in sys.path]:
    sys.path.insert(0, ROOT)
import lumenairy  # noqa: E402

assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(ROOT), (
    f"lumenairy imported from {lumenairy.__file__}, not {ROOT}")

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402,F401
from lumenairy.elements.pmm import (  # noqa: E402,F401 -- re-exported
    PMM2DStackPure,
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


def dump(name, obj):
    obj = dict(obj)
    obj.setdefault("env", ENV)
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(obj, f, indent=1, default=_js)


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
# tensors (block form)
# --------------------------------------------------------------------------- #
def lc(deg, no=1.5, ne=1.8):
    """In-plane uniaxial, director at ``deg`` degrees from x."""
    return uniaxial_tensor(no, ne, np.pi / 2, phi=np.deg2rad(deg))


def biaxial(rot_deg=20.0, d=(2.1, 3.0, 2.5)):
    """Biaxial with its in-plane axes rotated by ``rot_deg`` (eps_xx !=
    eps_yy != eps_zz)."""
    c, s = np.cos(np.deg2rad(rot_deg)), np.sin(np.deg2rad(rot_deg))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return (Rz @ np.diag(np.asarray(d, complex)) @ Rz.T).astype(complex)


#: gyrotropic of the OPPOSITE public gauge to the builder's GYRO, stronger,
#: unequal diagonal
GYRO_V = np.array([[2.4, -0.7j, 0], [0.7j, 2.1, 0], [0, 0, 2.2]], complex)
#: real, ASYMMETRIC, non-Hermitian (e12 != e21, both real): neither
#: reciprocal (eps != eps^T) nor lossless (eps != eps^H)
RASYM = np.array([[2.3, 0.45, 0], [-0.15, 2.0, 0], [0, 0, 2.6]], complex)
#: lossy rotated biaxial (complex diagonal, PUBLIC Im > 0 = loss)
LOSSY = biaxial(35.0, (2.2 + 0.35j, 3.1 + 0.08j, 2.4 + 0.2j))
#: gyrotropic permeability (Hermitian)
MU_GYRO = np.array([[1.5, 0.35j, 0], [-0.35j, 1.4, 0], [0, 0, 1.25]], complex)
#: real symmetric mu with off-diagonals
MU_SYM = np.array([[1.6, 0.3, 0], [0.3, 1.3, 0], [0, 0, 1.15]], complex)
#: lossy mu with off-diagonals
MU_LOSSY = np.array([[1.5 + 0.2j, 0.1, 0], [0.1, 1.35 + 0.05j, 0],
                     [0, 0, 1.2 + 0.1j]], complex)


# --------------------------------------------------------------------------- #
# maps (the verifier's own)
# --------------------------------------------------------------------------- #
class TwoHarmonicStretch:
    """ASYMMETRIC periodic stretch ``f(u) = u + a1 sin(k u) + a2 [sin(2 k u +
    ph) - sin(ph)]`` (``f(0) = 0``, ``f(u + p) = f(u) + p``); duck-typed to
    the :class:`SineStretch` protocol."""

    def __init__(self, a1, a2, ph):
        self.a1, self.a2, self.ph = float(a1), float(a2), float(ph)

    def __call__(self, u, period):
        k = 2.0 * np.pi / float(period)
        u = np.asarray(u, dtype=float)
        f = (u + self.a1 * np.sin(k * u)
             + self.a2 * (np.sin(2 * k * u + self.ph) - np.sin(self.ph)))
        fp = (1.0 + self.a1 * k * np.cos(k * u)
              + 2 * k * self.a2 * np.cos(2 * k * u + self.ph))
        return f, fp

    def check(self, period, axis):
        u = np.linspace(0, period, 4001)
        if np.min(self(u, period)[1]) <= 0:
            raise ValueError("TwoHarmonicStretch folds on " + axis)

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
        return ("TwoHarmonicStretch", self.a1, self.a2, self.ph)


def stretch2(Pp, a1=0.09, a2=0.05, ph=0.7, b1=-0.06, b2=0.035, phb=-1.1,
             n=3):
    """Both axes stretched by an asymmetric two-harmonic profile."""
    w = np.linspace(0.0, Pp, n + 1)
    fx = TwoHarmonicStretch(a1 * Pp, a2 * Pp, ph)
    fy = TwoHarmonicStretch(b1 * Pp, b2 * Pp, phb)
    return CM.SeparableStretch(w, w, fx=fx, fy=fy)


def stretch_range(cm, Pp):
    u = np.linspace(0, Pp, 2001)
    a = cm.fx(u, Pp)[1]
    b = cm.fy(u, Pp)[1]
    return float(a.max() / a.min()), float(b.max() / b.min())


def shear4(Pp, amp=1.0):
    """4 x 4 TransfiniteMap, straight edges, the nine interior vertices moved
    by a deterministic pseudo-random pattern of up to 7 % of the period
    (bilinear cells, NON-diagonal J, g12 != 0).  ``amp`` scales it."""
    w = np.linspace(0.0, Pp, 5)
    V = np.stack(np.meshgrid(w, w, indexing="ij"), axis=-1)
    rng = np.random.default_rng(20261003)
    for i in range(1, 4):
        for j in range(1, 4):
            V[i, j] += amp * rng.uniform(-0.07, 0.07, 2) * Pp
    return CM.TransfiniteMap(w, w, V, None)


def circle(Pp, kind="c3", r_frac=0.3):
    if kind == "c3":
        return CM._circle_map_3x3(Pp, r_frac * Pp)[0]
    return CM._circle_map_5x5(Pp, r_frac * Pp)[0]


def make_map(name, Pp):
    if name == "h2":
        return stretch2(Pp)
    if name == "sh4":
        return shear4(Pp)
    if name in ("c3", "c5"):
        return circle(Pp, name)
    if name == "none":
        return None
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# the verifier's own Berreman 4x4 with eps AND mu
# --------------------------------------------------------------------------- #
def _t3(a):
    a = np.asarray(a, complex)
    return a * np.eye(3) if a.ndim == 0 else a


def _delta(eps, mu, Kx, Ky):
    """d psi / dz = i k0 D psi, psi = (Ex, Ey, hx, hy), h = eta0 H,
    exp(-i w t): curl E = i k0 mu h, curl h = -i k0 eps E."""
    eps, mu = _t3(eps), _t3(mu)
    D = np.zeros((4, 4), complex)
    for c in range(4):
        Ex, Ey, hx, hy = np.eye(4)[c]
        hz = (Kx * Ey - Ky * Ex - mu[2, 0] * hx - mu[2, 1] * hy) / mu[2, 2]
        Ez = (-(Kx * hy - Ky * hx) - eps[2, 0] * Ex
              - eps[2, 1] * Ey) / eps[2, 2]
        E = np.array([Ex, Ey, Ez])
        h = np.array([hx, hy, hz])
        mh, eE = mu @ h, eps @ E
        D[:, c] = [Kx * Ez + mh[1], Ky * Ez - mh[0],
                   Kx * hz - eE[1], Ky * hz + eE[0]]
    return D


def _halfspace(n, Kx, Ky):
    w, v = np.linalg.eig(_delta(n * n, 1.0, Kx, Ky))
    fwd = (w.real > 1e-12) | ((np.abs(w.real) <= 1e-12) & (w.imag > 0))
    F, B = v[:, fwd], v[:, ~fwd]
    assert F.shape[1] == 2 and B.shape[1] == 2
    return F @ np.linalg.inv(F[:2]), B @ np.linalg.inv(B[:2])


def _sz(psi):
    return float(np.real(psi[0] * np.conj(psi[3]) - psi[1] * np.conj(psi[2])))


def berreman_eps_mu(layers, n_sub, n_sup, wl, theta=0.0, phi=0.0):
    """layers: [(eps, mu, d)] from the superstrate side.  Returns (R, T, r):
    per-input (lab Ex, Ey transverse) reflectance / transmittance and the
    reflection Jones in the lab (Ex, Ey) basis."""
    k0 = 2 * np.pi / wl
    Kx = n_sup * np.sin(theta) * np.cos(phi)
    Ky = n_sup * np.sin(theta) * np.sin(phi)
    Fs, Bs = _halfspace(n_sup, Kx, Ky)
    Ft, _Bt = _halfspace(n_sub, Kx, Ky)
    Mt = np.eye(4, dtype=complex)
    for eps, mu, d in layers:
        Mt = expm(1j * k0 * _delta(eps, mu, Kx, Ky) * d) @ Mt
    A = np.hstack([Ft, -Mt @ Bs])
    X = np.linalg.solve(A, Mt @ Fs)
    t, r = X[:2], X[2:]
    R, T = np.zeros(2), np.zeros(2)
    for j in range(2):
        s_in = _sz(Fs[:, j])
        R[j] = -_sz(Bs @ r[:, j]) / s_in
        T[j] = _sz(Ft @ t[:, j]) / s_in
    return R, T, r


# --------------------------------------------------------------------------- #
# film helpers
# --------------------------------------------------------------------------- #
G3 = dict(P=0.40e-6, WL=1.0e-6, DEP=0.55e-6, NSUB=1.5, NSUP=1.0)


def film_stack(t33, cmap, M, theta=0.0, phi=0.0, mu=None, f=None):
    f = G3 if f is None else f
    st = PMM2DStackPure(f["P"], f["P"], n_superstrate=f["NSUP"],
                        n_substrate=f["NSUB"], n_modes=M, n_orders=2,
                        cmap=cmap)
    if mu is None:
        st.add_layer(f["DEP"], eps=t33)
    else:
        st.add_layer(f["DEP"], eps=t33, mu=mu)
    st.set_source(f["WL"], theta=theta, phi=phi)
    o, R, T, J = st.solve(jones=True, retain_internal=True)
    return st, np.asarray(o), np.asarray(R), np.asarray(T), np.asarray(J)


def film_vs_berreman(t33, cmap, M, theta=0.0, phi=0.0, f=None):
    """(dRT, dJ, J, extra): the shipped Berreman as the oracle (complex
    reflection Jones)."""
    f = G3 if f is None else f
    st, _o, R, T, J = film_stack(t33, cmap, M, theta, phi, f=f)
    Rb, Tb, jr, _jt = berreman_jones_1d([(t33, f["DEP"])], f["NSUB"],
                                        f["NSUP"], f["WL"], angle=theta,
                                        phi=phi)
    dRT = max(float(np.max(np.abs(R.sum(1) - Rb))),
              float(np.max(np.abs(T.sum(1) - Tb))))
    return dRT, float(np.max(np.abs(J - jr))), J, (R, T, Rb, Tb, jr, st)


# --------------------------------------------------------------------------- #
# the verifier's own engineered defects (patch the ONE congruence kernel)
# --------------------------------------------------------------------------- #
VMUT = ("eps_T", "eps_side", "no_inv_sg", "chi_T", "chi_side", "chi33_no_sg",
        "e33_no_sg", "chi_vac", "chi33m_no_sg", "chi_side_m")


def vmutate(kind):
    """Context manager: patch ``TS._stag_map_eff_tensor`` with one defect.

    eps_T       -- eps transposed (reverses gyration);
    eps_side    -- J^-T eps J^-1 (adj(J)^T eps adj(J) / sg);
    no_inv_sg   -- adj(J) eps adj(J)^T WITHOUT the 1/sg (adj vs J^-1);
    chi_T       -- mu transposed inside chi;
    chi_side    -- J mu^-1 J^T / sg instead of J^T mu^-1 J / sg;
    chi33_no_sg -- chi_33 = 1 / mu_33 (sg dropped);
    e33_no_sg   -- eps'_33 = eps_33 (sg dropped);
    chi_vac     -- mu ignored (chi of vacuum);
    chi33m_no_sg -- chi_33 = 1 / mu_33 ONLY when a material mu is given
                   (the realistic one-token bug in the magnetic branch; the
                   broad chi33_no_sg also hits the vacuum chi of every
                   tensor-route cell);
    chi_side_m  -- chi_side ONLY when a material mu is given."""
    import contextlib

    orig = TS._stag_map_eff_tensor

    def f(eps, mu, xu, xv, yu, yv):
        sg = xu * yv - xv * yu
        if kind == "eps_T":
            return orig(np.swapaxes(eps, -1, -2), mu, xu, xv, yu, yv)
        if kind == "chi_T":
            return orig(eps, None if mu is None else np.swapaxes(mu, -1, -2),
                        xu, xv, yu, yv)
        if kind == "chi_vac":
            out = orig(eps, None, xu, xv, yu, yv)
            return out
        out = orig(eps, mu, xu, xv, yu, yv)
        if kind == "eps_side":
            bad = orig(eps, mu, xu, yu, xv, yv)
            out.update({k: bad[k] for k in ("e11", "e12", "e21", "e22")})
        elif kind == "no_inv_sg":
            for k in ("e11", "e12", "e21", "e22"):
                out[k] = out[k] * sg
        elif kind == "chi_side" or (kind == "chi_side_m"
                                    and mu is not None):
            bad = orig(eps, mu, xu, yu, xv, yv)
            out.update({k: bad[k] for k in ("c11", "c12", "c21", "c22")})
        elif kind == "chi33_no_sg" or (kind == "chi33m_no_sg"
                                       and mu is not None):
            out["c33"] = out["c33"] * sg
        elif kind in ("chi_side_m", "chi33m_no_sg"):
            pass
        elif kind == "e33_no_sg":
            out["e33"] = out["e33"] / sg
        else:
            raise ValueError(kind)
        return out

    @contextlib.contextmanager
    def cm():
        TS._stag_map_eff_tensor = f
        try:
            yield
        finally:
            TS._stag_map_eff_tensor = orig
    return cm()
