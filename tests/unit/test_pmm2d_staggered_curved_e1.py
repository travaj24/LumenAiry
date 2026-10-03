"""CURVED-CELL MAP, Phase E1, for the PURE staggered 2-D PMM: OUT-OF-PLANE
tensors and SLANTED walls inside curved cells -- gates E1-1 .. E1-9 of
``docs/audits/BUILD_PMM2D_CURVED_E1_2026_10_03.md`` (plan
``docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md`` section 4.5).

Words.  Under a coordinate map ``(x, y) = Phi(u, v)`` (Jacobian ``J``,
``sg = det J``) EVERY region is magnetic -- vacuum carries ``chi_t = g / sg``,
``chi33 = 1 / sg`` -- so the FIRST-ORDER out-of-plane generator (``4 q^2``,
state ``[E1; E2; G1; G2]``) needs PERMEABILITY BLOCKS: ``R = C[chi_t]C`` on
its B side, ``K_tz = C[chi_t][d2; -d1]`` in its E rows and the
``chi33``-weighted ``G3`` projection (``_assemble_oop_general``).  A SLANTED
layer under a map is the composite frame ``x = Phi(u, v) + t w``; the shear
reaches the generator as ``tau = J^-1 t`` and ``kappa = chi_t tau``.  The
two GAUGE CONSTANTS of the shipped out-of-plane path are ``_OOP_ROT_SIGN =
-1`` (a sign on the out-of-plane entries and the slant vector) and
``_OOP_H_GAUGE = -1j`` (the ``G``-to-``H`` constant).  A FAIL-BEFORE arm is a
deliberately broken variant through the real code (a patched function or
constant, restored afterwards) that must fail the bar.

EVERY BAR is derived from a measurement made by this build on 2026-10-03
(Windows 11, CPython 3.14.6, numpy 2.4.4, scipy 1.17.1, OMP/OPENBLAS/MKL = 1;
probe JSON under ``validation/probe_pmm2d_curved/build_e1/``, the unit-size
readings in ``e_unit_readings.json``), stated next to the assertion with its
gap on both sides.  Sizes: 3 x 3 grids at M <= 5, 5 x 5 at M = 4.

Fixtures: the shipped out-of-plane suite's uniform slab (period 0.9, depth
0.35, lambda 1, air over n = 1.5) with its tilted-LC tensors -- ``_OOP``
(tilt 35 deg, azimuth 25 deg) and the NON-RECIPROCAL ``_NONREC`` (Hermitian,
``e13 != e31``) -- against the exact Berreman 4x4 oracle; and the planning P3
fixture (period 1.2, depth 0.5, air over n = 1.45, r = 0.36) for the pillars
(``_OOP30``: a director tilted 30 deg out of the plane).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import warnings  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402
from lumenairy.elements.pmm import (  # noqa: E402
    Circle,
    PMM2DStackPure,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.pmm import _curvemap as CM  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE = os.path.join(_HERE, "..", "..", "validation", "probe_pmm2d_curved",
                      "build_e1")

_SLAB = dict(P=0.9, WL=1.0, DEP=0.35, NSUB=1.5, NSUP=1.0)
_OOP = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
_NONREC = np.array(_OOP, dtype=complex)
_NONREC[0, 2] = _OOP[0, 2] + 0.22j
_NONREC[2, 0] = np.conj(_NONREC[0, 2])
_OOP30 = uniaxial_tensor(1.5, 1.8, np.deg2rad(60.0), phi=np.deg2rad(30.0))
_NONREC30 = np.array(_OOP30, dtype=complex)
_NONREC30[0, 2] = _OOP30[0, 2] + 0.25j
_NONREC30[2, 0] = np.conj(_NONREC30[0, 2])
_EYE3 = np.eye(3, dtype=complex)
_P = 1.2
_ORD9 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1),
         (-1, -1)]
_CON = (np.deg2rad(25.0), np.deg2rad(40.0))
_OBL = (np.deg2rad(25.0), 0.0)
_MU_GYRO = np.array([[1.6, 0.3j, 0], [-0.3j, 1.6, 0], [0, 0, 1.2]], complex)
_MU_LOSSY = np.array([[1.5 + 0.08j, 0.2, 0], [0.2, 1.3 + 0.05j, 0],
                      [0, 0, 1.2 + 0.03j]], complex)


# =========================================================================== #
# helpers
# =========================================================================== #
def _shear(Pp):
    """3 x 3, straight edges, the four interior vertices MOVED: bilinear
    cells with g12 != 0 and a non-diagonal J (Phase D's sheared map)."""
    w = np.linspace(0.0, Pp, 4)
    V = np.stack(np.meshgrid(w, w, indexing="ij"), axis=-1)
    for (i, j), (dx, dy) in {(1, 1): (0.06, 0.04), (2, 1): (-0.05, 0.03),
                             (1, 2): (0.03, -0.05),
                             (2, 2): (-0.04, -0.06)}.items():
        V[i, j] += (dx * Pp, dy * Pp)
    return CM.TransfiniteMap(w, w, V, None)


def _ident(Pp, n=3):
    w = np.linspace(0.0, Pp, n + 1)
    return CM.TransfiniteMap(w, w)


def _stretch(Pp, a, n=2):
    w = np.linspace(0.0, Pp, n + 1)
    return CM.SeparableStretch(w, w, fx=CM.SineStretch(a * Pp),
                               fy=CM.SineStretch(-0.5 * a * Pp))


def _circle(Pp, r_frac=0.3):
    return CM._circle_map_3x3(Pp, r_frac * Pp)[0]


def _jt(st):
    md = st._modal
    p0 = md["p0"]
    return np.array([[md["tx"][0][p0], md["tx"][1][p0]],
                     [md["ty"][0][p0], md["ty"][1][p0]]])


def _slab(t33, cmap, M, theta=0.0, phi=0.0, slant=None, mu=None,
          oracle=None):
    """A uniform tensor slab (possibly slanted / magnetic) as a uniform
    layer of the (possibly mapped) stack against the exact 4x4 answer:
    (dRT, dJr, dJt, closure)."""
    f = _SLAB
    st = PMM2DStackPure(f["P"], f["P"], n_superstrate=f["NSUP"],
                        n_substrate=f["NSUB"], n_modes=M, n_orders=2,
                        cmap=cmap)
    st.add_layer(f["DEP"], eps=t33, slant=slant, mu=mu)
    st.set_source(f["WL"], theta=theta, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R, T, J = st.solve(jones=True)
    if oracle is None:
        Rb, Tb, jr, jt = berreman_jones_1d([(t33, f["DEP"])], f["NSUB"],
                                           f["NSUP"], f["WL"], angle=theta,
                                           phi=phi)
    else:
        Rb, Tb, jr, jt = oracle
    R, T = np.asarray(R), np.asarray(T)
    return (float(max(np.abs(R.sum(1) - Rb).max(),
                      np.abs(T.sum(1) - Tb).max())),
            float(np.abs(np.asarray(J) - jr).max()),
            float(np.abs(_jt(st) - jt).max()),
            float(np.abs(R.sum(1) + T.sum(1) - 1.0).max()))


@contextmanager
def _patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _mu_delta(eps, mu, Kx, Ky):
    """The 4 x 4 Berreman matrix of a homogeneous (eps, mu) medium,
    Psi = (Ex, Ey, Hx, Hy), dPsi/dz = i k0 Delta Psi (PUBLIC exp(-i w t));
    the independent oracle of ``validation/.../build_e1/_mu_oracle.py``."""
    D = np.zeros((4, 4), dtype=complex)
    for j in range(4):
        Ex, Ey, Hx, Hy = np.eye(4)[j]
        Ez = (-(Kx * Hy - Ky * Hx) - eps[2, 0] * Ex - eps[2, 1] * Ey) \
            / eps[2, 2]
        Hz = (Kx * Ey - Ky * Ex - mu[2, 0] * Hx - mu[2, 1] * Hy) / mu[2, 2]
        mH = mu @ np.array([Hx, Hy, Hz])
        eE = eps @ np.array([Ex, Ey, Ez])
        D[:, j] = (mH[1] + Kx * Ez, Ky * Ez - mH[0], Kx * Hz - eE[1],
                   Ky * Hz + eE[0])
    return D


def _mu_berreman(eps, mu, theta, phi):
    """(R, T, Jr, Jt) of one uniform (eps, mu) slab of the _SLAB fixture."""
    f = _SLAB
    k0 = 2 * np.pi / f["WL"]
    Kx = f["NSUP"] * np.sin(theta) * np.cos(phi)
    Ky = f["NSUP"] * np.sin(theta) * np.sin(phi)

    def flux(v):
        return float(np.real(v[0] * np.conj(v[3]) - v[1] * np.conj(v[2])))

    def split(n):
        q, W = np.linalg.eig(_mu_delta(n ** 2 * _EYE3, _EYE3, Kx, Ky))
        fw = [i for i in range(4) if q[i].imag > 1e-12 or
              (abs(q[i].imag) <= 1e-12 and flux(W[:, i]) > 0)]
        return W[:, fw], W[:, [i for i in range(4) if i not in fw]]
    Wf1, Wb1 = split(f["NSUP"])
    Wf3, _ = split(f["NSUB"])
    Pm = expm(1j * k0 * _mu_delta(np.asarray(eps, complex),
                                  np.asarray(mu, complex), Kx, Ky) * f["DEP"])
    A = np.concatenate([Pm @ Wb1, -Wf3], axis=1)
    R, T = np.zeros(2), np.zeros(2)
    Jr, Jt = np.zeros((2, 2), complex), np.zeros((2, 2), complex)
    for c in range(2):
        a = np.linalg.solve(Wf1[:2], np.eye(2)[c])
        x = np.linalg.solve(A, -Pm @ (Wf1 @ a))
        vi, vr, vt = Wf1 @ a, Wb1 @ x[:2], Wf3 @ x[2:]
        Jr[:, c], Jt[:, c] = vr[:2], vt[:2]
        R[c], T[c] = -flux(vr) / flux(vi), flux(vt) / flux(vi)
    return R, T, Jr, Jt


def _idx(o, orders=_ORD9):
    o = np.asarray(o)
    return [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0])
            for m, n in orders]


def _vec(o, R, T):
    i = _idx(o)
    return np.concatenate([np.asarray(R)[:, i].ravel(),
                           np.asarray(T)[:, i].ravel()])


def _disk(kind, t33, R0=0.36):
    cm = (CM._circle_map_3x3(_P, R0)[0] if kind == "c3"
          else CM._circle_map_5x5(_P, R0)[0])
    N = cm.shape[0]
    eps = np.broadcast_to(_EYE3, (N, N, 3, 3)).copy()
    for c in ([(1, 1)] if N == 3 else [(i, j) for i in (1, 2, 3)
                                       for j in (1, 2, 3)]):
        eps[c] = t33
    return cm, eps


def _pillar(kind, t33, M, theta=0.0, phi=0.0, slant=None):
    cm, eps = _disk(kind, t33)
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=3, cmap=cm)
    st.add_layer(0.5, eps_cell=eps, slant=slant)
    st.set_source(1.0, theta=theta, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve(jones=True)
    return st, np.asarray(o), np.asarray(R), np.asarray(T)


def _saved_vec(name):
    with open(os.path.join(_PROBE, name), encoding="utf-8") as fh:
        return np.array(json.load(fh)["vec"])


# =========================================================================== #
# E1-1 -- the shipped dispatch: no map, no mu -> the shipped arithmetic
# =========================================================================== #
def test_e1_1_shipped_dispatch_never_reaches_the_e1_code(monkeypatch):
    """No map = today's bytes (``e1_compare.json``: 200 / 200 SHA-256 equal
    against ``eae470d9`` -- Phase D's 122-key fixture set plus 78 E1 keys:
    every out-of-plane and slant branch, the parity reduction on and off,
    slanted stacks with the frame-anchor phase, an OOP mixed stack with
    absorption, and the Phase D MAPPED tensor and magnetic solves through the
    refactored permeability blocks; fail-before: the identity map through
    the mapped generator, 42 of 99 operator hashes differ -- the rest are
    ``None`` attributes).  The unit-test restatement: with the three Phase E1
    kernels booby-trapped, an unmapped out-of-plane, a slanted, an in-plane
    magnetic and a MAPPED block-form tensor solve all run."""
    def boom(*a, **k):
        raise AssertionError("a Phase E1 kernel was reached")
    monkeypatch.setattr(TS.Granet2DTransverseE, "_assemble_oop_general",
                        boom)
    for name in ("_stag_map_slant_weights", "_stag_scale_weight"):
        monkeypatch.setattr(TS, name, boom)
    cell = np.broadcast_to(_EYE3, (2, 2, 3, 3)).copy()
    cell[0, 0] = _OOP
    pmm_jones_2d_staggered(_P, _P, cell, 1.45, 1.0, 0.4, 1.0, degree=4,
                           n_orders=2, theta=0.2, phi=0.3)
    sc = np.array([[4.0, 1], [1, 1]], complex)
    pmm_jones_2d_staggered(_P, _P, sc, 1.45, 1.0, 0.4, 1.0, degree=4,
                           n_orders=2, slant=(0.2, -0.1), theta=0.2)
    pmm_jones_2d_staggered(_P, _P, sc, 1.45, 1.0, 0.4, 1.0, degree=4,
                           n_orders=2, mu_cell=np.full((2, 2), 1.3 + 0j))
    lc = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=np.pi / 6)
    pmm_jones_2d_staggered(_P, _P, None, 1.45, 1.0, 0.5, 1.0, n_modes=4,
                           n_orders=2, shapes=[Circle(0.6, 0.6, 0.36, lc)],
                           background_eps=1.0)


# =========================================================================== #
# E1-2 -- the identity map through the mapped generator = the shipped one
# =========================================================================== #
def _lcell(t33):
    c = np.broadcast_to(_EYE3, (3, 3, 3, 3)).copy()
    for i, j in ((0, 0), (1, 0), (0, 1)):
        c[i, j] = t33
    return c


def _identity_arms(slant, M=5, theta=0.0, phi=0.0, move=0.0):
    """(shipped solver, identity-mapped solver) on the re-entrant L cell."""
    a0x = 2 * np.pi * np.sin(theta) * np.cos(phi)
    a0y = 2 * np.pi * np.sin(theta) * np.sin(phi)
    kw = dict(alpha0x=a0x, alpha0y=a0y, slant=slant)
    s0 = TS.Granet2DTransverseE(_P, _P, 3, 3, M, _lcell(_OOP), **kw)
    w = np.linspace(0.0, _P, 4)
    V = np.stack(np.meshgrid(w, w, indexing="ij"), axis=-1)
    V[1, 1, 0] += move * _P
    cm = CM.TransfiniteMap(w, w, V, None)
    s1 = TS.Granet2DTransverseE(_P, _P, cm.u_walls, cm.v_walls, M,
                                _lcell(_OOP), cmap=cm, **kw)
    return s0, s1


def _rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


def _identity_full(slant):
    out = []
    for cm in (None, _ident(_P)):
        st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.5,
                            n_modes=5, n_orders=2, cmap=cm)
        st.add_layer(0.4, eps_cell=_lcell(_OOP), slant=slant)
        st.set_source(1.0)
        _o, R, T, J = st.solve(jones=True)
        out.append((np.asarray(R), np.asarray(T), np.asarray(J), _jt(st)))
    return [float(np.abs(x - y).max()) for x, y in zip(*out)]


@pytest.mark.parametrize("slant", [None, (0.25, -0.1)])
def test_e1_2_identity_map_is_the_shipped_out_of_plane_generator(slant):
    """The identity map through the mapped out-of-plane generator (chi =
    I, tau = t) is the shipped generator to round-off: the operators, the
    modal set, and the full R / T / both Jones at normal incidence.  The
    slanted arm is E1-6's first gate: a slanted RECTANGULAR (L-shaped)
    pillar under the identity transfinite map IS the shipped slant solve.
    Measured 2026-10-03 (``e_unit_readings.json`` key ``e1_2``, M = 5, at
    the Bloch phases of (0.3, 0.4) rad; ``e2_identity_M*.json`` for M = 7
    and more cells): A, B, the eigenvalues and R / T / Jones all <= ~5e-14.
    Bar 1e-11 (the Phase A A2 bar; >= 2.3 decades above).  Upper gap: one
    interior vertex moved by 1e-6 of the period moves A by ~1e-6 relative
    (asserted > 1e-9)."""
    s0, s1 = _identity_arms(slant, theta=0.3, phi=0.4)
    assert _rel(s1.Agen, s0.Agen) <= 1e-11
    assert _rel(s1.Bgen, s0.Bgen) <= 1e-11
    m0, m1 = TS._region_modes_oop(s0), TS._region_modes_oop(s1)
    d = np.abs(m0[2][:, None] - m1[2][None, :]).min(axis=1).max()
    assert d / np.abs(m0[2]).max() <= 1e-11
    _s0, sm = _identity_arms(slant, move=1e-6)
    assert _rel(sm.Agen, _s0.Agen) > 1e-9
    assert max(_identity_full(slant)) <= 1e-11


# =========================================================================== #
# E1-3 -- a uniform out-of-plane slab under a map is the Berreman slab;
#         both gauge constants, two-sided, in the mapped setting
# =========================================================================== #
def test_e1_3_oop_slab_under_a_sheared_map_is_the_berreman_slab():
    """The sheared bilinear map (non-diagonal J) is EXACT for a uniform slab
    at normal incidence and spectral off it.  Measured 2026-10-03
    (``e3_slab_shear_nonrec_*.json``): normal M = 4: R / T 3.3e-14,
    Jr 2.3e-15, Jt 2.9e-14; conical (25, 40 deg) M = 5: 1.9e-10 / 9.2e-11
    / 3.1e-10 (M = 7: 8.2e-14).  Bars 1e-11 (normal, 2.5 decades above) and
    3e-9 (conical M = 5, 1.0 decade above); the wrong gauges sit at 1e-3 ..
    0.4 (next test)."""
    r = _slab(_NONREC, _shear(_SLAB["P"]), 4)
    assert max(r[:3]) <= 1e-11, r
    r = _slab(_NONREC, _shear(_SLAB["P"]), 5, *_CON)
    assert max(r[:3]) <= 3e-9, r


def test_e1_3_both_gauge_constants_are_pinned_under_a_map():
    """``_OOP_ROT_SIGN`` and ``_OOP_H_GAUGE`` are MAP-INDEPENDENT: at their
    shipped values the sheared-map slab is Berreman (previous test); flipped,
    it misses by the SAME amounts as the unmapped slab (``e3_arms_*.json``,
    M = 6: rot +1 -> Jt 0.12 conical / 0.16 oblique, invisible at normal
    (7e-14: the rotation maps the normal mount onto itself); H gauge +1j ->
    Jr 1.4e-2 / Jt 0.40 at normal with R / T BLIND (4.3e-14); +1 -> the
    lossless trap broken, closure 0.10).  At the unit sizes see
    ``e_unit_readings.json`` key ``e1_3_gauge``.  Bars: conical rot +1
    >= 1e-3 on Jt, normal rot +1 <= 1e-11 (the blindness, two-sided);
    +1j: Jr >= 1e-3 while R / T <= 1e-11; +1: closure >= 1e-2."""
    cm = _shear(_SLAB["P"])
    with _patched(TS, "_OOP_ROT_SIGN", 1.0):
        rc = _slab(_NONREC, cm, 5, *_CON)
        rn = _slab(_NONREC, cm, 4)
    assert rc[2] >= 1e-3, rc
    assert max(rn[:3]) <= 1e-11, rn
    with _patched(TS, "_OOP_H_GAUGE", 1j):
        rh = _slab(_NONREC, cm, 4)
    assert rh[1] >= 1e-3 and rh[0] <= 1e-11, rh
    with _patched(TS, "_OOP_H_GAUGE", 1.0):
        r1 = _slab(_NONREC, cm, 4)
    assert r1[3] >= 1e-2, r1


def _no_mu_blocks(monkeypatch):
    """The permeability blocks dropped from the generator under a map (chi_t
    = I, chi33 = 1 in _assemble_oop_general; everything else intact)."""
    orig = TS.Granet2DTransverseE._assemble_oop_general

    def f(self, *a, **k):
        if self.cmap is None:
            return orig(self, *a, **k)
        W = self._mapw["c11"]

        def one(v):
            if isinstance(W, TS._StagNodeWeight):
                return TS._StagNodeWeight(
                    W.t * 0 + v, {k_: x * 0 + v for k_, x in W.p.items()})
            return W * 0 + v
        self._chi_maps = lambda: (one(1.0), one(0.0), one(0.0), one(1.0),
                                  one(1.0))
        try:
            return orig(self, *a, **k)
        finally:
            del self._chi_maps
    monkeypatch.setattr(TS.Granet2DTransverseE, "_assemble_oop_general", f)


def test_e1_3_the_permeability_blocks_are_load_bearing_under_a_map(
        monkeypatch):
    """E1-9's first row: drop the permeability blocks under a map and the
    sheared-map slab misses Berreman by 3.3e-3 (R / T) / 5.4e-2 (Jt) at
    M = 6 (``e3_arms_shear_nonrec_normal_M6.json``) -- with the LOSSLESS
    closure still 6.6e-8 (an energy check would pass it) -- while the
    IDENTITY map cannot see it (chi = I there: 6.9e-14) and neither can the
    unmapped solve.  At the unit size (``e_unit_readings.json`` key
    ``e1_3_mu``, M = 4): shear >= 1e-4 required, identity <= 1e-11."""
    _no_mu_blocks(monkeypatch)
    r = _slab(_NONREC, _shear(_SLAB["P"]), 4)
    assert max(r[:3]) >= 1e-4, r
    r = _slab(_NONREC, _ident(_SLAB["P"]), 4)
    assert max(r[:3]) <= 1e-11, r


def test_e1_3_chi33_is_seen_only_off_normal_incidence(monkeypatch):
    """chi33 without sqrt(g) (equivalently the G rows' G3 taken as the bare
    strong curl): a uniform slab at NORMAL incidence never excites the
    longitudinal curl (c3 = 0), so the defect is invisible there (sheared
    map, M = 6: 7.3e-14) and caught at oblique incidence (5.1e-4 R / T,
    2.5e-3 Jt) -- the out-of-plane twin of Phase D's no_sg_e33 finding
    (``e3_arms_shear_nonrec_*_M6.json``).  Unit size
    (``e_unit_readings.json`` key ``e1_3_chi33``, M = 5): oblique >= 1e-5,
    normal <= 1e-11."""
    orig = TS._stag_map_eff_tensor

    def f(eps, mu, xu, xv, yu, yv, **kw):
        out = orig(eps, mu, xu, xv, yu, yv, **kw)
        if kw.get("oop"):
            out["c33"] = out["c33"] * (xu * yv - xv * yu)
        return out
    monkeypatch.setattr(TS, "_stag_map_eff_tensor", f)
    cm = _shear(_SLAB["P"])
    assert max(_slab(_NONREC, cm, 5, *_OBL)[:3]) >= 1e-5
    assert max(_slab(_NONREC, cm, 5)[:3]) <= 1e-11


def test_e1_3_oop_slab_under_a_stretch_converges_like_the_inplane_film():
    """Under a separable sine stretch (a = 0.15 p, 2 x 2 cells; local stretch
    33:1) the slab is spectral, not exact, and the out-of-plane slab tracks
    the IN-PLANE LC film under the same map rung for rung
    (``e3_slab_s15_{nonrec,lc}_*``: normal M = 7 nonrec 3.0e-11 / 2.0e-10
    against LC 4.7e-10 / 1.2e-9) -- the residual is the map's resolution,
    not the out-of-plane machinery.  Unit size (``e_unit_readings.json`` key
    ``e1_3_stretch``): normal M = 5 measured 8.8e-7 (Jt; the plateau of the
    stretch's odd / even ladder) -> bar 2e-6, and the M = 4 -> 5 drop
    >= 2 decades (measured 4.0)."""
    cm = _stretch(_SLAB["P"], 0.15)
    r4 = _slab(_NONREC, cm, 4)
    r5 = _slab(_NONREC, cm, 5)
    assert max(r5[:3]) <= 2e-6, r5
    assert max(r4[:3]) / max(r5[:3]) >= 1e2, (r4, r5)


# =========================================================================== #
# E1-3m -- an out-of-plane eps WITH a material mu (the magnetic build's wall)
# =========================================================================== #
def test_e1_3m_the_mu_oracle_is_berreman_at_mu_one():
    """The (eps, mu) 4x4 oracle of this file against the shipped
    ``berreman_jones_1d`` at mu = I (out-of-plane non-reciprocal tensor,
    conical): R, T and both Jones.  Measured 2026-10-03
    (``e3m_oracle.json``): <= 3.0e-15 over three tensors and three mounts;
    the isotropic (eps, mu) slab against the analytic Airy formula 5.6e-16.
    Bar 1e-12."""
    Rb, Tb, jr, jt = berreman_jones_1d([(_NONREC, _SLAB["DEP"])],
                                       _SLAB["NSUB"], _SLAB["NSUP"],
                                       _SLAB["WL"], angle=_CON[0],
                                       phi=_CON[1])
    R, T, Jr, Jt = _mu_berreman(_NONREC, _EYE3, *_CON)
    for a, b in ((R, Rb), (T, Tb), (Jr, jr), (Jt, jt)):
        assert float(np.abs(a - b).max()) <= 1e-12


@pytest.mark.parametrize("mu,cmap", [
    ("gyro", None), ("lossy", None), ("gyro", "shear")])
def test_e1_3m_oop_eps_with_mu_matches_the_eps_mu_oracle(mu, cmap):
    """An out-of-plane eps with a block-form material mu -- refused until
    this phase -- against the independent (eps, mu) oracle, R / T / both
    Jones, conical.  GYRO (Hermitian, non-reciprocal) keeps the whitened
    eig; LOSSY makes chi_t non-Hermitian and takes the QZ branch.  Measured
    2026-10-03 (``e3m_*.json``; ``e_unit_readings.json`` key ``e1_3m``):
    unmapped M = 6: gyro 6.5e-12 / 1.25e-11 (Jt), lossy 3.3e-12 / 7.4e-12;
    the sheared map at M = 5 6.1e-10 (Jt; 1.4e-13 by M = 7).  Bars 1e-9
    (unmapped M = 6, 1.9 decades above) and 1e-8 (mapped M = 5, 1.2 decades
    above); the wrong-gauge arms of the slab sit at 1e-3 .. 0.4."""
    m = {"gyro": _MU_GYRO, "lossy": _MU_LOSSY}[mu]
    ref = _mu_berreman(_NONREC, m, *_CON)
    if cmap is None:
        r = _slab(_NONREC, None, 6, *_CON, mu=m, oracle=ref)
        assert max(r[:3]) <= 1e-9, r
    else:
        r = _slab(_NONREC, _shear(_SLAB["P"]), 5, *_CON, mu=m, oracle=ref)
        assert max(r[:3]) <= 1e-8, r


def test_e1_3m_a_transposed_mu_is_caught_by_the_non_symmetric_gates(
        monkeypatch):
    """Phase D verifier's surviving mutant, closed here: the permeability
    TRANSPOSED inside chi (``mu -> mu^T`` in the map's congruence and in the
    unmapped per-cell inverse).  Every Phase D gate used a diagonal mu, so it
    survived all 26 Phase D ids; the E1-3m gates carry a NON-SYMMETRIC
    (gyrotropic, ``m12 = -m21 = 0.3i``) mu on a CONICAL slab, unmapped and
    under the sheared map.  Measured 2026-10-03 (this build, the
    ``test_e1_3m_oop_eps_with_mu_matches_the_eps_mu_oracle[gyro-*]``
    fixtures): the mutant misses the (eps, mu) oracle by 8.4e-3 on R / T and
    0.63 on Jt (both arms), with the lossless closure untouched (1e-11 /
    6e-10: the lossless trap), against 1.25e-11 / 6.1e-10 correct.  Bar
    >= 1e-3 on R / T (0.9 decades below the defect, 6 above the correct
    arms)."""
    orig_t = TS._stag_map_eff_tensor

    def ft(eps, mu, *a, **k):
        return orig_t(eps, None if mu is None else np.swapaxes(mu, -1, -2),
                      *a, **k)
    orig_c = TS.Granet2DTransverseE._chi_maps

    def fc(self):
        m = self.mu_cell
        if self.cmap is None and m is not None and m.ndim == 4:
            self.mu_cell = np.swapaxes(m, -1, -2)
            try:
                return orig_c(self)
            finally:
                self.mu_cell = m
        return orig_c(self)
    monkeypatch.setattr(TS, "_stag_map_eff_tensor", ft)
    monkeypatch.setattr(TS.Granet2DTransverseE, "_chi_maps", fc)
    ref = _mu_berreman(_NONREC, _MU_GYRO, *_CON)
    r = _slab(_NONREC, None, 6, *_CON, mu=_MU_GYRO, oracle=ref)
    assert r[0] >= 1e-3, r
    r = _slab(_NONREC, _shear(_SLAB["P"]), 5, *_CON, mu=_MU_GYRO, oracle=ref)
    assert r[0] >= 1e-3, r


# =========================================================================== #
# E1-4 -- an out-of-plane film under the CIRCLE map (singular vertices)
# =========================================================================== #
def test_e1_4_oop_film_under_the_circle_map_is_spectral():
    """The four singular vertices of the circle map and the 4 q^2 generator
    together: a uniform out-of-plane film under the 3 x 3 circle map is
    spectral to round-off at normal incidence (``e3_slab_c3_nonrec_normal``:
    R / T 1.6e-6, 1.4e-8, 2.3e-11, 1.6e-13 at M = 4 .. 7; Jt 2.3e-6 ..
    6.6e-13; the 5 x 5 map 7.3e-12 by M = 5) and at oblique / conical
    incidence four decades behind (F-B6: 2.6e-9 / 1.1e-9 at M = 8).  Bars:
    M = 6 <= 1e-9 (0.9 decades above Jt 1.3e-10) and the M = 4 -> 6 drop
    >= 3 decades (measured 4.2)."""
    cm = _circle(_SLAB["P"])
    r4 = _slab(_NONREC, cm, 4)
    r6 = _slab(_NONREC, cm, 6)
    assert max(r6[:3]) <= 1e-9, r6
    assert max(r4[:3]) / max(r6[:3]) >= 1e3, (r4, r6)


# =========================================================================== #
# E1-5 -- an out-of-plane tensor circular pillar
# =========================================================================== #
def test_e1_5_oop_pillar_lands_on_its_converged_answer():
    """The OOP30 disk (director tilted 30 deg out of the plane), c5 M = 4
    (dof 900) against the saved c3 M = 10 rung (``e5_curved_oop30_c3_*_M10``;
    its own last rung change 1.0e-5, and the c5 M = 7 rung agrees with it to
    4.0e-6 -- two independent topologies).  Measured 2026-10-03
    (``e5_summary.json``): 3.2e-4.  Bar 1e-3 (0.5 decades).  Fail-before:
    the 4-step staircase on the shipped out-of-plane solver at M = 4,
    4.5e-2 (it converges in M to 1.9e-2, a different device) -- asserted
    >= 1e-2."""
    ref = _saved_vec("e5_curved_oop30_c3_t0.0000_p0.0000_M10.json")
    _st, o, R, T = _pillar("c5", _OOP30, 4)
    assert float(np.abs(_vec(o, R, T) - ref).max()) <= 1e-3
    w = np.array([0.0, 0.24, 0.96, _P])
    eps = np.broadcast_to(_EYE3, (3, 3, 3, 3)).copy()
    eps[1, 1] = _OOP30
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=4, n_orders=3, layer_grids="per-layer")
    st.add_layer(0.5, eps_cell=eps, x_walls=w, y_walls=w)
    st.set_source(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o2, R2, T2, _J = st.solve(jones=True)
    assert float(np.abs(_vec(o2, R2, T2) - ref).max()) >= 1e-2


# =========================================================================== #
# E1-6 -- slant under a map: the composite frame x = Phi(u, v) + t w
# =========================================================================== #
def _slant_weights_patch(monkeypatch, kind):
    """tau_unmapped: the shear NOT composed with the map (tau = kappa = t,
    the shipped constants); no_slant_blocks: tau = kappa = 0."""
    orig = TS._stag_map_slant_weights

    def f(out, A, sg, tvec):
        orig(out, A, sg, tvec)
        for k, v in (("t1", tvec[0]), ("t2", tvec[1]), ("k1", tvec[0]),
                     ("k2", tvec[1])):
            out[k] = out[k] * 0 + (float(v) if kind == "tau_unmapped"
                                   else 0.0)
    monkeypatch.setattr(TS, "_stag_map_slant_weights", f)


def test_e1_6_slanted_uniform_slab_under_a_map_is_the_slab():
    """A SLANTED uniform slab is the slab (a shear of a homogeneous medium is
    a coordinate change): under the sheared map the composite frame -- the
    congruence of the congruence, tau = J^-1 t varying across every cell,
    kappa = chi_t tau -- must reproduce Berreman.  Measured 2026-10-03
    (``e3_slab_shear_nonrec_*_slant+0.20-0.10.json``): normal M = 5 R / T
    7.1e-11, Jr 4.5e-15, Jt 2.6e-14 (M = 7 1.3e-13); oblique M = 5 8.0e-9 /
    1.6e-9 / 6.2e-8 (M = 7 8.5e-12 / 1.9e-12 / 8.0e-11).  Bars 1e-9 (normal
    M = 5, 1.1 decades above) and 1e-6 (oblique M = 5, 1.2 decades above).
    Fail-before arms below."""
    cm = _shear(_SLAB["P"])
    r = _slab(_NONREC, cm, 5, slant=(0.2, -0.1))
    assert max(r[:3]) <= 1e-9, r
    r = _slab(_NONREC, cm, 5, *_OBL, slant=(0.2, -0.1))
    assert max(r[:3]) <= 1e-6, r


def test_e1_6_the_shear_must_be_composed_with_the_map(monkeypatch):
    """Two-sided: the shipped constant slant blocks laid on a mapped cell
    (tau = t instead of J^-1 t: the shear not composed with the map) miss the
    slanted slab by 7.7e-4 at NORMAL incidence (M = 6) and 2.9e-3 (Jt) at
    oblique; dropping the slant blocks altogether is invisible at normal
    incidence (there tau . G'_t = t . G_t is constant, so its derivative
    vanishes: 2.4e-14) and caught at oblique (0.14 on Jt)
    (``e3_arms_shear_nonrec_*_slant+0.20-0.10_M6.json``).  Unit size
    (``e_unit_readings.json`` key ``e1_6_arms``, M = 5): tau_unmapped at
    normal >= 1e-4; no_slant_blocks at oblique >= 1e-3 on Jt."""
    cm = _shear(_SLAB["P"])
    _slant_weights_patch(monkeypatch, "tau_unmapped")
    r = _slab(_NONREC, cm, 5, slant=(0.2, -0.1))
    assert max(r[:3]) >= 1e-4, r
    monkeypatch.undo()
    _slant_weights_patch(monkeypatch, "no_slant_blocks")
    r = _slab(_NONREC, cm, 5, *_OBL, slant=(0.2, -0.1))
    assert r[2] >= 1e-3, r


def test_e1_6_slanted_circle_lands_on_its_converged_answer():
    """A slanted circular pillar (eps 4, r 0.36, public slant (0.2, 0)): the
    5 x 5 composite map at M = 4 against the saved 3 x 3 M = 10 rung
    (``e6_curved_eps4_c3_*_M10``; its own last rung change 3.8e-5; the c5
    M = 7 rung agrees with it to 2.3e-5).  The two staircase limits approach
    it (``e6_summary.json``): the shipped slant solver on 4 / 8 / 16-step
    in-plane staircases 7.6e-2 / 5.3e-2 / 9.1e-3 (each converged in M to its
    own device), and the exact-disk RCWA z-staircase (16 slices, 25 orders,
    Richardson in 1 / N and then in 1 / N_z^2) 2.2e-4.  Measured: c5 M = 4
    6.3e-4.  Bar 2e-3 (0.5 decades).  Fail-before: the 4-step staircase on
    the shipped slant solver at M = 4, 6.6e-2 (>= 2e-2), and the pillar
    slanted the OTHER way (``e6_curved_eps4_c3_*_s-0.200_0.000_M8``, read
    from the saved rung: 2.5e-2 from the answer, asserted >= 1e-2 -- and
    exactly the x-mirror image of the +0.2 rung, R(m, n; -t) = R(-m, n; +t)
    to 1.7e-13, so the slant's SIGN is a physical identity of the composite
    frame, not a convention that happens to fit)."""
    ref = _saved_vec("e6_curved_eps4_c3_t0.0000_p0.0000_M10.json")
    _st, o, R, T = _pillar("c5", 4.0 * _EYE3, 4, slant=(0.2, 0.0))
    assert float(np.abs(_vec(o, R, T) - ref).max()) <= 2e-3
    w = np.array([0.0, 0.24, 0.96, _P])
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=4, n_orders=3, layer_grids="per-layer")
    st.add_layer(0.5, eps_cell=eps, x_walls=w, y_walls=w, slant=(0.2, 0.0))
    st.set_source(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o2, R2, T2, _J = st.solve(jones=True)
    assert float(np.abs(_vec(o2, R2, T2) - ref).max()) >= 2e-2
    mirror = _saved_vec("e6_curved_eps4_c3_t0.0000_p0.0000_s-0.200_0.000_M8"
                        ".json")
    assert float(np.abs(mirror - ref).max()) >= 1e-2


# =========================================================================== #
# E1-7 / E1-8 -- slant x out-of-plane x curved, at conical incidence
# =========================================================================== #
def _jones_block(st, o, order):
    """Power-normalized 2 x 2 reflection Jones block of the channel
    (incidence -> reflected ``order``) (Phase D's d8 formula)."""
    md = st._modal
    k = _idx(o, [order])[0]
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


def _reverse(th, ph, m, n):
    st = np.sin(th)
    kx = -(st * np.cos(ph) + m / _P)
    ky = -(st * np.sin(ph) + n / _P)
    return float(np.arcsin(np.hypot(kx, ky))), float(np.arctan2(ky, kx))


def _sv(t33, M, th, ph, order):
    st, o, _R, _T = _pillar("c3", t33, M, th, ph, slant=(0.2, 0.0))
    return np.linalg.svd(_jones_block(st, o, order), compute_uv=False)


def test_e1_7_slant_oop_curved_is_reciprocal_and_a_nonreciprocal_one_obeys_its_transpose():
    """The hardest case, all three at once: the OOP30 disk SLANTED (0.2, 0)
    under the circle map, conical (25, 40 deg).  Reciprocity (an identity
    the solver does not impose): for eps = eps^T the power-normalized
    Jones block of (incidence -> reflected order (-1, 0)) and of its
    reversal have the same singular values; for the NON-reciprocal twin
    (``e13 != e31``) they do NOT, and the reversal of the TRANSPOSED tensor
    restores them (eps -> eps^T is the reciprocal partner of a
    non-reciprocal medium -- the transpose convention, seen through the
    fields).  Measured 2026-10-03 (``e7_summary.json``, c3, M = 5 / 7):
    reciprocal control 1.3e-4 / 1.5e-6, transposed partner 8.6e-5 / 1.4e-6
    (both converging; the shipped solver on the 4-step staircase reads
    3.2e-5 / 8.5e-8 and 4.0e-5 / 6.9e-8), the non-reciprocal residual 6.2e-3
    / 5.5e-3 (flat: physics, not discretisation); the wrong pairing
    (``e_unit_readings.json`` key ``e1_7``) 8.9e-2.  Bars: reciprocal and
    transposed-partner <= 1e-3 (0.9 decades above M = 5); non-reciprocal
    >= 2e-3 (0.5 decades below it, 1.4 above the transposed partner); wrong
    pairing >= 1e-2."""
    th, ph = _CON
    tr, pr = _reverse(th, ph, -1, 0)
    f = _sv(_OOP30, 5, th, ph, (-1, 0))
    r = _sv(_OOP30, 5, tr, pr, (-1, 0))
    w = _sv(_OOP30, 5, tr, pr, (0, 0))
    assert float(np.abs(f - r).max()) <= 1e-3
    assert float(np.abs(f - w).max()) >= 1e-2
    fn = _sv(_NONREC30, 5, th, ph, (-1, 0))
    rn = _sv(_NONREC30, 5, tr, pr, (-1, 0))
    rT = _sv(_NONREC30.T.copy(), 5, tr, pr, (-1, 0))
    assert float(np.abs(fn - rn).max()) >= 2e-3
    assert float(np.abs(fn - rT).max()) <= 1e-3
