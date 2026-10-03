"""CURVED-CELL MAP, Phase D, for the PURE staggered 2-D PMM: ANISOTROPIC and
MAGNETIC materials inside curved cells -- gates D1-D10 of
``docs/audits/BUILD_PMM2D_CURVED_D_2026_10_03.md`` (plan
``docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md`` section 4.4).

Words.  Under a coordinate map ``(x, y) = Phi(u, v)`` with Jacobian ``J``
(``sg = det J``) the solver works on the covariant fields ``E' = J^T E``,
and a BLOCK-FORM material becomes the EFFECTIVE tensors
``eps'_t = sg J^-1 eps_t J^-T``, ``eps'_33 = sg eps_33``,
``chi_t = [mu'_t]^-1 = J^T mu_t^-1 J / sg``, ``chi_33 = 1 / (sg mu_33)``
(``_stag_map_eff_tensor``), evaluated at every quadrature node.  A
FAIL-BEFORE arm is a deliberately broken variant (here: the ONE congruence
kernel or the weight / H-partner path patched with a specific defect) that
must fail the bar.  A GYROTROPIC tensor has ``e12 = -e21`` imaginary; its
transpose is the opposite magnetization, which no efficiency of a uniform
film can see and its Jones matrix does.

Fixtures: the shipped anisotropic suite's G3 film (period 0.40 um, lambda
1 um, depth 0.55 um, air over n = 1.5) with its two tensors -- the rotated
in-plane uniaxial ``LC`` (n_o 1.5, n_e 1.8, director at 0.55 rad) and the
gyrotropic ``GYRO`` -- against the exact Berreman 4x4 oracle; and the
planning P3 fixture (period 1.2, lambda 1, depth 0.5, air over n = 1.45)
for the curved pillars (``LC30``: the LC with its director at 30 deg).  Row 0
of R / T is the input E along x, row 1 E along y.

EVERY BAR is derived from a measurement made by this build on 2026-10-03
(Windows 11, CPython 3.14.6, numpy 2.4.4, scipy 1.17.1, OMP/OPENBLAS/MKL = 1;
probe JSON under ``validation/probe_pmm2d_curved/build_d/``, mostly
``d_unit_*.json``), stated next to the assertion with its gap on both
sides.  The discretisation numbers are deterministic; the round-off floor of
the mapped solve is ~1e-14 (Phase C, C9), far below every bar.  Sizes: 3 x 3
grids at M <= 6, 5 x 5 at M <= 4, the Li 2 x 2 cell at M = 8.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import warnings  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402
from lumenairy.elements.pmm import (  # noqa: E402
    Circle,
    PMM2DStackPure,
    Rect,
    compile_shapes,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.pmm import _curvemap as CM  # noqa: E402
from lumenairy.elements.pmm import stack2d_pure as SP  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE = os.path.join(_HERE, "..", "..", "validation", "probe_pmm2d_curved",
                      "build_d")

_G3 = dict(P=0.40e-6, WL=1.0e-6, DEP=0.55e-6, NSUB=1.5, NSUP=1.0)
_LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
_LC30 = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=np.pi / 6)
_GYRO = np.array([[2.25, 0.5j, 0.0], [-0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                 dtype=complex)
_P = 1.2
_ORD9 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1),
         (-1, -1)]
_EYE3 = np.eye(3, dtype=complex)


# =========================================================================== #
# helpers
# =========================================================================== #
def _stretch(Pp, a):
    w = np.linspace(0.0, Pp, 4)
    return CM.SeparableStretch(w, w, fx=CM.SineStretch(a * Pp),
                               fy=CM.SineStretch(-0.5 * a * Pp))


def _shear(Pp):
    """3 x 3, straight edges, the four interior vertices MOVED: bilinear
    cells with g12 != 0 and a non-diagonal J."""
    w = np.linspace(0.0, Pp, 4)
    V = np.stack(np.meshgrid(w, w, indexing="ij"), axis=-1)
    for (i, j), (dx, dy) in {(1, 1): (0.06, 0.04), (2, 1): (-0.05, 0.03),
                             (1, 2): (0.03, -0.05),
                             (2, 2): (-0.04, -0.06)}.items():
        V[i, j] += (dx * Pp, dy * Pp)
    return CM.TransfiniteMap(w, w, V, None)


def _circle(Pp):
    return CM._circle_map_3x3(Pp, 0.3 * Pp)[0]


def _film(t33, cmap, M, theta=0.0, phi=0.0, mu=None):
    """A uniform film under ``cmap`` vs Berreman: (dRT, dJ, J)."""
    f = _G3
    st = PMM2DStackPure(f["P"], f["P"], n_superstrate=f["NSUP"],
                        n_substrate=f["NSUB"], n_modes=M, n_orders=2,
                        cmap=cmap)
    st.add_layer(f["DEP"], eps=t33)
    st.set_source(f["WL"], theta=theta, phi=phi)
    _o, R, T, J = st.solve(jones=True)
    Rb, Tb, jr, _jt = berreman_jones_1d([(t33, f["DEP"])], f["NSUB"],
                                        f["NSUP"], f["WL"], angle=theta,
                                        phi=phi)
    dRT = max(float(np.abs(np.asarray(R).sum(1) - Rb).max()),
              float(np.abs(np.asarray(T).sum(1) - Tb).max()))
    J = np.asarray(J)
    return dRT, float(np.abs(J - jr).max()), J


def _idx(o, orders=_ORD9):
    o = np.asarray(o)
    return [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0])
            for m, n in orders]


def _vec(o, R, T, orders=_ORD9):
    i = _idx(o, orders)
    return np.concatenate([np.asarray(R)[:, i].ravel(),
                           np.asarray(T)[:, i].ravel()])


def _airy_mu(eps, mu, d, wl=1.0, n1=1.0, n3=1.45):
    """Normal-incidence (R, T) of an isotropic (eps, mu) slab."""
    k0 = 2 * np.pi / wl
    n2 = np.sqrt(eps * mu + 0j)
    a1, a2, a3 = n1, n2 / mu, n3
    r12, r23 = (a1 - a2) / (a1 + a2), (a2 - a3) / (a2 + a3)
    t12, t23 = 2 * a1 / (a1 + a2), 2 * a2 / (a2 + a3)
    ph = np.exp(1j * n2 * k0 * d)
    den = 1 + r12 * r23 * ph ** 2
    r, t = (r12 + r23 * ph ** 2) / den, t12 * t23 * ph / den
    return float(abs(r) ** 2), float(abs(t) ** 2 * a3 / a1)


_MU_F = np.diag([1.8, 1.8, 1.3]).astype(complex)


def _mag_film_err(M):
    """The magnetic film (eps 2, mu diag(1.8, 1.8, 1.3)) under the 3 x 3
    circle map vs the (eps, mu) Airy slab at normal incidence."""
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=2, cmap=_circle(_P))
    st.add_layer(0.5, eps=2.0, mu=_MU_F)
    st.set_source(1.0)
    o, R, T, _J = st.solve(jones=True)
    R, T = np.asarray(R).copy(), np.asarray(T).copy()
    Rx, Tx = _airy_mu(2.0, 1.8, 0.5)
    i0 = _idx(o, [(0, 0)])[0]
    R[:, i0] -= Rx
    T[:, i0] -= Tx
    return float(max(np.abs(R).max(), np.abs(T).max()))


# ---- engineered defects through the real code path (the D9 matrix) --------
@contextmanager
def _mutate(kind):
    """Patch ONE kernel with a specific defect for the duration (restored
    afterwards).  See ``validation/probe_pmm2d_curved/build_d/_dcommon.py``
    for the full matrix; the arms used here:

    * transpose  -- eps transposed inside the congruence;
    * side       -- J^-T eps J^-1 instead of J^-1 eps J^-T;
    * no_sg_e33  -- sqrt(g) dropped from eps'_33;
    * mixed_sign -- the mixed (e12', e21') weights negated;
    * mu_after   -- chi = inverse of the cell-AVERAGED mu' (after
      quadrature);
    * hgram_R    -- the Eq.-25 H partner of every mapped region through -R
      instead of the plain Gram (the incident decomposition keeps the plain
      Gram)."""
    orig = TS._stag_map_eff_tensor
    saved = []

    def setp(mod, name, fn):
        saved.append((mod, name, getattr(mod, name)))
        setattr(mod, name, fn)

    if kind == "transpose":
        setp(TS, "_stag_map_eff_tensor",
             lambda e, m, a, b, c, d: orig(np.swapaxes(e, -1, -2), m, a, b,
                                           c, d))
    elif kind == "side":
        def f(e, m, xu, xv, yu, yv):
            good = orig(e, m, xu, xv, yu, yv)
            bad = orig(e, m, xu, yu, xv, yv)
            good.update({k: bad[k] for k in ("e11", "e12", "e21", "e22")})
            return good
        setp(TS, "_stag_map_eff_tensor", f)
    elif kind == "no_sg_e33":
        def f(e, m, xu, xv, yu, yv):
            out = orig(e, m, xu, xv, yu, yv)
            out["e33"] = out["e33"] / (xu * yv - xv * yu)
            return out
        setp(TS, "_stag_map_eff_tensor", f)
    elif kind == "mixed_sign":
        def f(e, m, xu, xv, yu, yv):
            out = orig(e, m, xu, xv, yu, yv)
            out["e12"], out["e21"] = -out["e12"], -out["e21"]
            return out
        setp(TS, "_stag_map_eff_tensor", f)
    elif kind == "mu_after":
        orig_w = TS._stag_map_weights

        def w(bx, by, cmap, eps_cell, rule, mu_cell=None):
            out = orig_w(bx, by, cmap, eps_cell, rule, mu_cell=mu_cell)
            return out if mu_cell is None else _chi_after(out, rule)
        setp(TS, "_stag_map_weights", w)
    elif kind == "hgram_R":
        orig_a = TS.Granet2DTransverseE._assemble
        orig_hc = TS._homog_geom_cache
        orig_hm = TS._homog_region_modes

        def a(self):
            orig_a(self)
            if self.cmap is not None:
                self._true_gram = self.Ggram_blocks
                self.Ggram_blocks = None

        def hc(solver):
            if getattr(solver, "cmap", None) is None:
                return orig_hc(solver)
            solver.Ggram_blocks = solver._true_gram
            try:
                g = orig_hc(solver)
            finally:
                solver.Ggram_blocks = None
            return tuple(g) + (np.linalg.inv(-solver.Rmat),)

        def hm(geom, eps):
            if len(geom) == 7:
                geom = tuple(geom[:4]) + (geom[6], geom[5])
            return orig_hm(geom, eps)
        setp(TS.Granet2DTransverseE, "_assemble", a)
        for mod in (TS, SP):
            setp(mod, "_homog_geom_cache", hc)
            setp(mod, "_homog_region_modes", hm)
    else:
        raise ValueError(kind)
    try:
        yield
    finally:
        for mod, name, old in reversed(saved):
            setattr(mod, name, old)


def _chi_after(W, rule):
    keys = ("c11", "c12", "c21", "c22", "c33")
    w2 = np.outer(rule.tensor[1], rule.tensor[1])

    def inv2(c11, c12, c21, c22):
        d = c11 * c22 - c12 * c21
        return c22 / d, -c12 / d, -c21 / d, c11 / d
    out = dict(W)
    T = {k: W[k].t for k in keys}
    m = list(inv2(*(T[k] for k in keys[:4]))) + [1.0 / T["c33"]]
    new = {k: np.empty_like(T[k]) for k in keys}
    newp = {k: {} for k in keys}
    for sx in range(T["c11"].shape[0]):
        for sy in range(T["c11"].shape[1]):
            cell = (sx, sy)
            if cell in rule.points:
                wq = rule.points[cell][2]
                pm = list(inv2(*(W[k].p[cell] for k in keys[:4]))) + [
                    1.0 / W["c33"].p[cell]]
                mm = [np.sum(wq * v) / np.sum(wq) for v in pm]
            else:
                mm = [np.sum(w2 * a[sx, sy]) / np.sum(w2) for a in m]
            c = list(inv2(*mm[:4])) + [1.0 / mm[4]]
            for k, v in zip(keys, c):
                new[k][sx, sy] = v
                if cell in rule.points:
                    newp[k][cell] = np.full_like(W[k].p[cell], v)
    for k in keys:
        out[k] = TS._StagNodeWeight(new[k], newp[k])
    return out


# =========================================================================== #
# D1 -- no map, and scalar maps, never reach the Phase D kernels
# =========================================================================== #
def test_d1_unmapped_and_scalar_mapped_solves_never_reach_the_phase_d_code(
        monkeypatch):
    """No map = today's bytes (``d1_compare.json``: 122 / 122 SHA-256 equal
    against ``607d43b0`` on Phase C's fixture set extended by 13 MAPPED
    SCALAR keys, ``d1_v1compare.json``: 181 / 181 on the Phase A verifier's
    set with every tensor and magnetic fixture of 5.43 / 5.44; fail-before:
    the identity map through the quadrature path, 28 of 36 operator hashes
    differ).  The unit-test restatement: with the three Phase D kernels
    booby-trapped, an unmapped tensor and an unmapped magnetic Jones solve
    and a SCALAR shape circle (mapped, scalar route) all run -- the scalar
    route under a map keeps its Phases A-C formula, which is why the 13
    mapped keys are byte-identical too."""
    def boom(*a, **k):
        raise AssertionError("a Phase D kernel was reached")
    for name in ("_stag_map_eff_tensor", "_stag_map_weights_tensor",
                 "_stag_map_as33"):
        monkeypatch.setattr(TS, name, boom)
    cell = np.broadcast_to(_EYE3, (2, 2, 3, 3)).copy()
    cell[0, 0] = _LC
    pmm_jones_2d_staggered(_P, _P, cell, 1.45, 1.0, 0.4, 1.0, degree=4,
                           n_orders=2)
    pmm_jones_2d_staggered(_P, _P, np.array([[4.0, 1], [1, 1]], complex),
                           1.45, 1.0, 0.4, 1.0, degree=4, n_orders=2,
                           mu_cell=np.broadcast_to(np.diag([2.0, 1.5, 1.0]),
                                                   (2, 2, 3, 3)))
    pmm_jones_2d_staggered(_P, _P, None, 1.45, 1.0, 0.5, 1.0, n_modes=4,
                           n_orders=2, shapes=[Circle(0.6, 0.6, 0.36, 4.0)],
                           background_eps=1.0)


def test_d1_tensor_route_at_eps_times_identity_is_the_scalar_route():
    """The two expressions of ONE formula agree: a scalar cell promoted to
    ``eps I`` through the general congruence equals the scalar route.
    Measured 2026-10-03 (``d2_scalar.json``, M = 5): operators 1.1e-15
    (shear) / 3.1e-15 (circle) relative, R / T / Jones 6.0e-15 / 1.3e-14.
    Bars 1e-12 (2.5 decades above).  Fail-before: the tensor route with
    the mixed weights negated moves the operators by 0.13 (shear) / 0.97
    (circle) (``d_unit_scalar_mixed.json``) -- the scalar cell carries a
    mixed e12' = -eps g12 / sqrt(g) under a sheared map."""
    k0 = 2 * np.pi
    for cm in (_shear(_P), _circle(_P)):
        e = np.ones((3, 3), complex)
        e[1, 1] = 4.0
        e33 = np.zeros((3, 3, 3, 3), complex)
        for k in range(3):
            e33[..., k, k] = e
        s1 = TS.Granet2DTransverseE(_P, _P, cm.u_walls, cm.v_walls, 5, e,
                                    k0=k0, cmap=cm)
        s2 = TS.Granet2DTransverseE(_P, _P, cm.u_walls, cm.v_walls, 5, e33,
                                    k0=k0, cmap=cm)
        sc = max(np.abs(s1.Lmat).max(), np.abs(s1.Rmat).max())
        d = max(np.abs(s1.Lmat - s2.Lmat).max(),
                np.abs(s1.Rmat - s2.Rmat).max()) / sc
        assert d <= 1e-12, d
        with _mutate("mixed_sign"):
            s3 = TS.Granet2DTransverseE(_P, _P, cm.u_walls, cm.v_walls, 5,
                                        e33, k0=k0, cmap=cm)
        dfb = max(np.abs(s1.Lmat - s3.Lmat).max(),
                  np.abs(s1.Rmat - s3.Rmat).max()) / sc
        assert dfb >= 1e-2, dfb


# =========================================================================== #
# D2 -- a uniform tensor film under a map IS the film (Berreman, Jones too)
# =========================================================================== #
def test_d2_tensor_films_under_a_stretch_and_a_shear_match_berreman():
    """A uniform LC (rotated in-plane uniaxial) and a GYROTROPIC film under
    the a = 0.05 p sine stretch of both axes, against ``berreman_jones_1d``
    (R, T and the complex Jones matrix).  Measured 2026-10-03
    (``d_unit_film.json``): LC 5.1e-6 / 9.9e-10, gyro 7.3e-6 / 1.4e-9 (R / T)
    at M = 4 / 6, Jones 1.4e-6 / 2.8e-10 and 2.1e-6 / 4.2e-10.  Bars: M = 6
    <= 1e-8 (0.84 decades above the worst 1.44e-9) and a M = 4 -> 6 drop
    >= 3 decades (3.7 measured).  The build doc's ladder reaches 6.9e-14 /
    1.3e-13 at M = 8 and 5.6e-13 at the 33:1 stretch (the stop condition
    1e-9 by M = 8 is met by four decades).  The SHEARED bilinear map makes
    the plane-wave field an exact polynomial: <= 6.8e-15 at M = 4, bar 1e-11.
    Fail-before (the shear distinguishes J^-1 eps J^-T from J^-T eps J^-1,
    a stretch cannot -- its J is diagonal): the side-swapped congruence on
    the LC film reads 7.0e-4 (R / T) / 2.6e-3 (Jones) at M = 4."""
    cs = _stretch(_G3["P"], 0.05)
    for t in (_LC, _GYRO):
        r4 = _film(t, cs, 4)
        r6 = _film(t, cs, 6)
        assert max(r6[0], r6[1]) <= 1e-8, r6[:2]
        assert np.log10(r4[0] / r6[0]) >= 3.0, (r4[0], r6[0])
        rs = _film(t, _shear(_G3["P"]), 4)
        assert max(rs[0], rs[1]) <= 1e-11, rs[:2]
    with _mutate("side"):
        fb = _film(_LC, _shear(_G3["P"]), 4)
    assert fb[0] >= 1e-4 and fb[1] >= 1e-4, fb[:2]


def test_d2_gyrotropic_jones_sees_the_transpose_and_rt_does_not():
    """THE transpose-convention gate, both halves asserted.  For a uniform
    gyrotropic film the efficiencies are TRANSPOSE-BLIND (eps^T is the
    opposite magnetization; per-input power sums are equal), only the Jones
    matrix sees the sign.  Measured 2026-10-03 (``d_unit_film.json``), the
    s05 stretch at M = 5: correct Jones 4.7e-8; TRANSPOSED Jones 0.1496
    (bar >= 1e-2: 1.2 decades under it, 5.3 decades above the correct
    arm's bar 1e-7), while the transposed R / T error equals the correct
    one to 1.9e-15 (bar 1e-12 on the difference) -- an R / T gate would
    pass a transposed build.  The film's own gyrotropic signature: J01 / J10
    = -1 + 6.0e-7 (M = 5; bar 1e-5)."""
    cs = _stretch(_G3["P"], 0.05)
    dRT, dJ, J = _film(_GYRO, cs, 5)
    with _mutate("transpose"):
        tRT, tJ, _Jt = _film(_GYRO, cs, 5)
    assert dJ <= 1e-7, dJ
    assert tJ >= 1e-2, tJ
    assert abs(tRT - dRT) <= 1e-12, (tRT, dRT)
    assert abs(J[0, 1] / J[1, 0] + 1.0) <= 1e-5, J[0, 1] / J[1, 0]


# =========================================================================== #
# D3 -- a tensor weight through the four singular vertices of the circle
# =========================================================================== #
def test_d3_tensor_films_under_the_circle_map_are_spectral():
    """The LC and gyrotropic films under the 3 x 3 CIRCLE map (four singular
    vertices, the Duffy corner rule; a general, non-diagonal J everywhere).
    Measured 2026-10-03 (``d_unit_film.json``): LC 2.5e-6 -> 7.7e-11 (R / T),
    2.9e-6 -> 1.7e-10 (Jones); gyro 2.0e-6 -> 4.4e-11, 2.9e-6 -> 1.7e-10 at
    M = 4 -> 6.  Bars: M = 6 <= 1e-9 (0.77 decades above the worst 1.74e-10)
    and the R / T drop >= 3 decades (4.4 / 4.6 measured).  The build doc's
    ladder reaches 2.4e-13 / 1.4e-13 at M = 8 and is spectral at conical
    incidence too (8.8e-11 / 9.9e-11 at M = 8)."""
    cm = _circle(_G3["P"])
    for t in (_LC, _GYRO):
        r4 = _film(t, cm, 4)
        r6 = _film(t, cm, 6)
        assert max(r6[0], r6[1]) <= 1e-9, r6[:2]
        assert np.log10(r4[0] / r6[0]) >= 3.0, (r4[0], r6[0])


def test_d3_corner_rule_reaches_roundoff_with_tensor_and_magnetic_weights(
        monkeypatch):
    """The corner (Duffy) rule with a TENSOR weight (the LC30 disk) AND a
    material mu (diag(2, 2, 1) in the disk): the weights now carry the
    individual Jacobian products, not only the five metric combinations the
    node criterion measures.  Measured 2026-10-03 (``d3_quad_c3_M6.json``,
    ``d_unit_quad.json``, M = 6): the corner-rule operators at n = 16 vs
    n = 48 differ by 7.3e-13 (bar 1e-11, 1.1 decades above); the plain
    tensor rule at n = 24 is 4.5e-2 from them (bar >= 1e-2) -- its moments
    fall only algebraically (0.33, 0.16, 0.097, ..., 0.012 at n = 8 .. 48);
    the adaptive count picks n0 = 2 M + 8 = 20, within 5.5e-13 of 2 n0."""
    cm = _circle(_P)
    eps = np.broadcast_to(_EYE3, (3, 3, 3, 3)).copy()
    mu = eps.copy()
    eps[1, 1] = _LC30
    mu[1, 1] = np.diag([2.0, 2.0, 1.0])

    def ops(n, plain=False):
        monkeypatch.setattr(TS, "_stag_map_nodes", lambda *a, **k: n)
        if plain:
            monkeypatch.setattr(TS, "_stag_map_singular_corners",
                                lambda c: {})
        s = TS.Granet2DTransverseE(_P, _P, cm.u_walls, cm.v_walls, 6, eps,
                                   k0=2 * np.pi, mu_cell=mu, cmap=cm)
        monkeypatch.undo()
        return s.Lmat, s.Rmat
    a, top, pl = ops(16), ops(48), ops(24, plain=True)
    sc = max(np.abs(top[0]).max(), np.abs(top[1]).max())

    def rel(x):
        return max(np.abs(x[0] - top[0]).max(),
                   np.abs(x[1] - top[1]).max()) / sc
    assert rel(a) <= 1e-11, rel(a)
    assert rel(pl) >= 1e-2, rel(pl)
    n0 = TS._stag_map_nodes(TS.Basis1D(_P, cm.u_walls, 6),
                            TS.Basis1D(_P, cm.v_walls, 6), cm, 6)
    assert n0 == 20, n0


# =========================================================================== #
# D4 -- a TENSOR circular pillar against its converged answer and two oracles
# =========================================================================== #
def _lc30_pillar(cmap, M):
    N = cmap.shape[0]
    eps = np.broadcast_to(_EYE3, (N, N, 3, 3)).copy()
    for c in ([(1, 1)] if N == 3 else [(i, j) for i in (1, 2, 3)
                                       for j in (1, 2, 3)]):
        eps[c] = _LC30
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=3, cmap=cmap)
    st.add_layer(0.5, eps_cell=eps)
    st.set_source(1.0)
    o, R, T, _J = st.solve(jones=True)
    return o, np.asarray(R), np.asarray(T)


def test_d4_tensor_pillar_lands_on_its_converged_answer():
    """The LC30 disk (r 0.36, director at 30 deg) in air on n = 1.45: the
    5 x 5 circle map at M = 4 (dof 450) against the CONVERGED tensor answer
    -- the 3 x 3 map at M = 10, which the INDEPENDENT 5 x 5 topology at M = 7
    reproduces to 3.1e-6 (``d4_summary.json``).  Measured 2026-10-03: 3.3e-4
    (bar 1e-3, 0.48 decades above).  The two families the brief names as
    oracles approach that same answer from far away (build doc ladders): the
    shipped 2-D tensor RCWA with the EXACT disk form factor at 9 .. 29
    orders per axis 9.1e-3 .. 3.0e-3 (distance x N within 0.082 .. 0.088 --
    1 / N; Richardson pairs down to 5.6e-5), and the shipped staggered
    solver on the 4 / 8 / 16-step staircases 1.7e-2 / 1.4e-2 / 5.4e-3.
    Fail-before here: the 4-step staircase at M = 4 (the shipped code, no
    map), 4.8e-2 -- >= 1e-2, 1 decade above the bar."""
    with open(os.path.join(_PROBE, "d4_curved_c3_M10.json"),
              encoding="utf-8") as fh:
        ref = json.load(fh)
    ref = np.array(ref["vec"])
    o, R, T = _lc30_pillar(CM._circle_map_5x5(_P, 0.36)[0], 4)
    d = float(np.abs(_vec(o, R, T) - ref).max())
    assert d <= 1e-3, d
    w = np.array([0.0, 0.24, 0.96, _P])
    eps = np.broadcast_to(_EYE3, (3, 3, 3, 3)).copy()
    eps[1, 1] = _LC30
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=4, n_orders=3, layer_grids="per-layer")
    st.add_layer(0.5, eps_cell=eps, x_walls=w, y_walls=w)
    st.set_source(1.0)
    o, R, T, _J = st.solve(jones=True)
    ds = float(np.abs(_vec(o, R, T) - ref).max())
    assert ds >= 1e-2, ds


# =========================================================================== #
# D5 -- Li 2003 (gyrotropic, patterned) under the identity map and a stretch
# =========================================================================== #
_LAM = 1.0e-6
_LI_B = np.array([[2.25, -0.5j, 0.0], [0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                 dtype=complex)
_LI_A = np.conj(_LI_B)
_LI_T1 = {(0, 0): 0.2980, (1, 0): 0.1195, (2, 0): 0.0222,
          (0, -1): 0.0619, (1, -1): 0.0269, (1, 1): 0.0137}
_LI_T2 = {**_LI_T1, (1, -1): 0.0137, (1, 1): 0.0269}


def _li(M, cmap=None, swap=False):
    c = np.empty((2, 2, 3, 3), dtype=complex)
    c[:] = _LI_B if swap else _LI_A
    c[0, 0] = _LI_A if swap else _LI_B
    st = PMM2DStackPure(2.4 * _LAM, 1.4 * _LAM, n_superstrate=1.0,
                        n_substrate=1.0 + 5.0j, n_modes=M, n_orders=4,
                        cmap=cmap)
    st.add_layer(_LAM, eps_cell=c)
    st.set_source(_LAM)
    o, R, T, J = st.solve(jones=True)
    R0 = np.asarray(R)[0, _idx(o, list(_LI_T1))]
    dev = [max(abs(R0[k] - v) for k, v in enumerate(t.values()))
           for t in (_LI_T1, _LI_T2)]
    return dev, np.asarray(R), np.asarray(T), np.asarray(J)


def _li_stretch():
    xw = np.array([0.0, 1.2 * _LAM, 2.4 * _LAM])
    yw = np.array([0.0, 0.7 * _LAM, 1.4 * _LAM])
    return CM.SeparableStretch.from_physical_walls(
        xw, yw, fx=CM.SineStretch(0.05 * 2.4 * _LAM),
        fy=CM.SineStretch(-0.03 * 1.4 * _LAM))


def test_d5_li2003_under_the_identity_map_is_the_unmapped_solve():
    """Li's gyrotropic crossed grating (J. Opt. A 5:345 2003, Example 1 --
    the published Table 1, REFLECTED orders, incident E_x) through the
    TENSOR route under an IDENTITY TransfiniteMap on the same walls.
    Bit-for-bit is not available (the quadrature reassociates the sums,
    plan P1); measured 2026-10-03 (``d_unit_li.json``): R / T / Jones
    4.5e-14 from the unmapped solve at M = 6 (bar 1e-11, 2.3 decades
    above), and at M = 8 the identity map reproduces the shipped 8.74e-05
    table deviation to the printed digits (``d5_li_summary.json``,
    8.739349639e-05 vs 8.739349637e-05)."""
    cm = CM.TransfiniteMap(np.array([0.0, 1.2 * _LAM, 2.4 * _LAM]),
                           np.array([0.0, 0.7 * _LAM, 1.4 * _LAM]))
    du, Ru, Tu, Ju = _li(6)
    di, Ri, Ti, Ji = _li(6, cmap=cm)
    d = max(np.abs(Ru - Ri).max(), np.abs(Tu - Ti).max(),
            np.abs(Ju - Ji).max())
    assert d <= 1e-11, d


def test_d5_li2003_under_a_stretch_keeps_the_gyrotropic_sign():
    """The same grating under a sine stretch of both axes (a = 0.05 p_x,
    -0.03 p_y) with the walls at the PREIMAGES of the physical walls (the
    same device).  Measured 2026-10-03 (``d_unit_li.json``, M = 8): Li's
    first row 1.40e-4 (bar 5e-4 = the shipped G2 bar, Li's own four-decimal
    rounding at truncation order 23; 0.55 decades above), his SECOND row
    (the cross terms reversed) missed by 1.32e-2; the swapped cell lands on
    the second row (1.41e-4) and misses the first by 1.32e-2 (bar >= 1e-3,
    1.1 decades under).  So the gyrotropic sign survives the map -- on a
    PATTERNED gyrotropic grating the transpose IS R / T-visible (the two
    rows differ in the (1, -1) / (1, 1) pair), unlike on a uniform film.
    The build doc's ladder: 1.45e-2 .. 1.1e-4 from the unmapped M = 10
    answer over M = 5 .. 9, converging to it (the stretch costs a rung or
    two, as Phase A measured)."""
    cm = _li_stretch()
    dev, *_ = _li(8, cmap=cm)
    assert dev[0] <= 5e-4 and dev[1] >= 1e-3, dev
    dsw, *_ = _li(8, cmap=cm, swap=True)
    assert dsw[1] <= 5e-4 and dsw[0] >= 1e-3, dsw


# =========================================================================== #
# D6 -- a material permeability under the map
# =========================================================================== #
def test_d6_magnetic_film_under_the_circle_map_matches_the_airy_slab():
    """A uniform MAGNETIC film (eps 2, mu = diag(1.8, 1.8, 1.3)) under the
    3 x 3 circle map against the exact (eps, mu) slab.  Measured 2026-10-03
    (``d_unit_mag.json``): 1.4e-6 (M = 4), 1.7e-8 (M = 5); bar 1e-7 at M = 5
    (0.78 decades above).  Fail-before arms (flat in M -- wrong operators,
    not discretisation): the mu INVERSE taken after quadrature (chi = the
    inverse of the cell-averaged mu') 0.44; the H partner through -R in
    every region 1.9e-2 -- the plan's prediction (section 4.4) that a
    material mu makes the H-partner trap real WITHOUT mixing helpers, since
    -R then differs between regions.  Bars >= 1e-3 (1.3 / 2.6 decades)."""
    assert _mag_film_err(5) <= 1e-7
    with _mutate("mu_after"):
        assert _mag_film_err(5) >= 1e-3
    with _mutate("hgram_R"):
        assert _mag_film_err(5) >= 1e-3


def test_d6_magnetic_pillar_is_the_dual_of_its_dielectric_twin():
    """Electromagnetic duality with VACUUM half-spaces: the E_x input of a
    magnetic disk (eps 1, mu = diag(2, 2, 1)) is the E_y input of the
    dielectric disk (eps = diag(2, 2, 1), mu 1), order by order -- two
    DIFFERENT weight paths under the same circle map (the chi blocks vs the
    eps blocks).  Not a discrete identity: the residual falls with M to the
    rung-change level.  Measured 2026-10-03 (``d6_summary.json``, 3 x 3 map):
    5.4e-3, 8.9e-4, 5.8e-4, 8.6e-6, 3.2e-5, 1.2e-5 at M = 4 .. 9 (5 x 5:
    2.3e-4 .. 2.6e-5 at M = 4 .. 6); the comparison WITHOUT the polarization
    swap stays at 0.048.  Bars at M = 5: duality <= 3e-3 (0.53 decades above
    8.9e-4), control >= 1e-2 (0.69 decades under 4.85e-2)."""
    cm = _circle(_P)
    one = np.broadcast_to(_EYE3, (3, 3, 3, 3)).copy()
    mu = one.copy()
    mu[1, 1] = np.diag([2.0, 2.0, 1.0])

    def run(eps, mu_c):
        st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.0,
                            n_modes=5, n_orders=3, cmap=cm)
        if mu_c is None:
            st.add_layer(0.5, eps_cell=eps)
        else:
            st.add_layer(0.5, eps_cell=eps, mu_cell=mu_c)
        st.set_source(1.0)
        o, R, T, _J = st.solve(jones=True)
        i = _idx(o)
        return np.asarray(R)[:, i], np.asarray(T)[:, i]
    Rm, Tm = run(one, mu)
    Re, Te = run(mu, None)
    dual = max(np.abs(Rm[0] - Re[1]).max(), np.abs(Tm[0] - Te[1]).max(),
               np.abs(Rm[1] - Re[0]).max(), np.abs(Tm[1] - Te[0]).max())
    ctrl = max(np.abs(Rm[0] - Re[0]).max(), np.abs(Tm[0] - Te[0]).max())
    assert dual <= 3e-3, dual
    assert ctrl >= 1e-2, ctrl


def test_d6_the_h_partner_trap_needs_a_material_mu():
    """Two-sided on the plan's statement: with NO material mu every mapped
    region's -R is the same geometric operator, so the -R H partner in
    EVERY region is consistent-wrong and invisible -- measured 2026-10-03
    (``d_unit_mag.json``): the LC film under the circle map at M = 5 reads
    2.8147005e-8 correct and 2.8147008e-8 through -R (difference 3.4e-15;
    bar 1e-12), and the D4 tensor pillar moves 2.7e-14 (``d9_summary.json``)
    -- while the magnetic film above moves by 1.9e-2."""
    cm = _circle(_G3["P"])
    a = _film(_LC, cm, 5)
    with _mutate("hgram_R"):
        b = _film(_LC, cm, 5)
    assert abs(a[0] - b[0]) <= 1e-12 and np.abs(a[2] - b[2]).max() <= 1e-12


# =========================================================================== #
# D7 -- a tensor and a magnetic layer in ONE merged-map stack
# =========================================================================== #
_MU2 = np.diag([1.5, 1.5, 1.2]).astype(complex)


def _d7(M, e1, layer2, retain=False):
    """LC30 circle (r 0.3, layer 1) inside a magnetic Rect (0.9 x 0.9,
    layer 2) over a plain Rect layer: ONE merged 5 x 5 map."""
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=3)
    st.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.3, e1)], background_eps=1.0)
    if layer2 == "mag":
        st.add_layer(0.2, shapes=[Rect(0.6, 0.6, 0.9, 0.9, 2.25, mu=_MU2)],
                     background_eps=1.0)
    elif layer2 == "vac_mu":
        st.add_layer(0.2, shapes=[Rect(0.6, 0.6, 0.9, 0.9, 1.0, mu=_EYE3)],
                     background_eps=1.0)
    elif layer2 == "uniform":
        st.add_layer(0.2, eps=1.0)
    elif layer2 == "paint11":
        st.add_layer(0.2, shapes=[Rect(0.6, 0.6, 0.9, 0.9, 1.0, mu=1.1)],
                     background_eps=1.0)
    st.add_layer(0.1, shapes=[Rect(0.6, 0.6, 0.9, 0.9, 1.5)],
                 background_eps=1.0)
    st.set_source(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve(retain_internal=retain, jones=True)
    return st, np.asarray(R), np.asarray(T), np.asarray(J)


def test_d7_tensor_and_magnetic_layers_share_one_map_and_conserve_energy():
    """Measured 2026-10-03 (``d_unit_stack.json``), the 5 x 5 merged map:
    lossless closure 2.3e-2 (M = 3), 9.2e-4 (M = 4) -- bar 5e-3 at M = 4
    (0.73 decades above; the discretisation level, falling 1.4 decades per
    rung).  Absorption with the LC made lossy (LC30 + 0.3i I), M = 3: the
    two LOSSLESS layers absorb 1.1e-15 (bar 1e-10) against 2.0e-2 with -R as
    the flux Gram (Phase A's A7 defect; bar >= 1e-3)."""
    st, R, T, _J = _d7(4, _LC30, "mag")
    assert st.cmap.shape == (5, 5)
    assert np.abs(R.sum(1) + T.sum(1) - 1).max() <= 5e-3
    st, R, T, _J = _d7(3, _LC30 + 0.3j * _EYE3, "mag", retain=True)
    A = np.asarray(st.layer_absorption())
    assert np.abs(A[1:]).max() <= 1e-10, A
    G = st._internal["G"]
    st._internal["G"] = -TS.Granet2DTransverseE(
        _P, _P, st.cmap.u_walls, st.cmap.v_walls, 3,
        np.ones(st.cmap.shape, complex), cmap=st.cmap).Rmat
    Ab = np.asarray(st.layer_absorption())
    st._internal["G"] = G
    assert np.abs(Ab[1:]).max() >= 1e-3, Ab


def test_d7_vacuum_painted_with_mu_one_is_vacuum():
    """A physical identity across the two routes: layer 2 painted with a
    vacuum Rect that carries an explicit mu = I (the magnetic TENSOR route
    under the map) equals a uniform vacuum layer 2 (the scalar route) on
    the same merged map.  Measured 2026-10-03 (``d_unit_stack.json``, M = 3):
    3.1e-15 (bar 1e-9, 5.5 decades above); fail-before: the same Rect with
    mu = 1.1 moves R / T / Jones by 2.3e-2 (bar >= 1e-3)."""
    a = _d7(3, _LC30, "vac_mu")
    b = _d7(3, _LC30, "uniform")
    c = _d7(3, _LC30, "paint11")
    assert a[0].cmap.fingerprint == b[0].cmap.fingerprint

    def d(x, y):
        return max(np.abs(x[1] - y[1]).max(), np.abs(x[2] - y[2]).max(),
                   np.abs(x[3] - y[3]).max())
    assert d(a, b) <= 1e-9, d(a, b)
    assert d(c, b) >= 1e-3, d(c, b)


# =========================================================================== #
# D8 -- oblique / conical incidence through a tensor circle: reciprocity
# =========================================================================== #
def _reverse(th, ph, m, n, wl=1.0, p=_P):
    st = np.sin(np.deg2rad(th))
    kx = -(st * np.cos(np.deg2rad(ph)) + m * wl / p)
    ky = -(st * np.sin(np.deg2rad(ph)) + n * wl / p)
    return (float(np.rad2deg(np.arcsin(np.hypot(kx, ky)))),
            float(np.rad2deg(np.arctan2(ky, kx))))


def _jones_block(st, o, k):
    """Power-normalized 2 x 2 reflection Jones block of retained order k."""
    md = st._modal
    kx0, ky0, kzi = md["kx0"], md["ky0"], float(md["kz_inc"])
    A = np.array([[md["rx"][c][k] for c in (0, 1)],
                  [md["ry"][c][k] for c in (0, 1)]])
    kxo, kyo, kzo = md["kx"][k], md["ky"][k], float(np.real(md["kz_ref"][k]))
    Gin = np.eye(2) + np.outer([kx0, ky0], [kx0, ky0]) / kzi ** 2
    Wout = (kzo / kzi) * (np.eye(2) + np.outer([kxo, kyo], [kxo, kyo])
                          / kzo ** 2)

    def msqrt(S, inv=False):
        w, V = np.linalg.eigh(S)
        return (V * w ** (-0.5 if inv else 0.5)) @ V.conj().T
    return msqrt(Wout) @ A @ msqrt(Gin, inv=True)


def _d8(t33, th, ph):
    cm = _circle(_P)
    eps = np.broadcast_to(_EYE3, (3, 3, 3, 3)).copy()
    eps[1, 1] = t33
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=5, n_orders=3, cmap=cm)
    st.add_layer(0.5, eps_cell=eps)
    st.set_source(1.0, theta=np.deg2rad(th), phi=np.deg2rad(ph))
    o, R, T, _J = st.solve()
    return st, np.asarray(o), np.asarray(R), np.asarray(T)


def test_d8_conical_tensor_circle_is_reciprocal_and_a_gyrotropic_one_is_not():
    """Reciprocity is an independent identity the solver does not impose:
    for a RECIPROCAL medium (the LC is real symmetric) the channel
    (25 deg, 40 deg) -> reflected order (-1, 0) and its reversal have the
    same Jones singular values.  Measured 2026-10-03 (``d_unit_oblique.json``,
    3 x 3 circle, M = 5): LC 1.2e-4 (bar 1e-3, 0.94 decades above; the build
    doc's ladder falls with M); the WRONG pairing (the reversed run's
    specular channel) 0.12; the GYROTROPIC disk (eps != eps^T, physically
    non-reciprocal) 2.1e-2 -- the identity has teeth on both a pairing
    error and the material (bars >= 1e-2 and >= 3e-3).  Closure (lossless)
    <= 3e-2 (8.1e-3 measured, the M = 5 discretisation level)."""
    tr, pr = _reverse(25.0, 40.0, -1, 0)
    sv = {}
    for name, t in (("lc", _LC30), ("gyro", _GYRO)):
        stf, of, Rf, Tf = _d8(t, 25.0, 40.0)
        str_, orr, _Rr, _Tr = _d8(t, tr, pr)
        k = _idx(of, [(-1, 0)])[0]
        kr = _idx(orr, [(-1, 0)])[0]
        k0 = _idx(orr, [(0, 0)])[0]
        sf = np.linalg.svd(_jones_block(stf, of, k), compute_uv=False)
        sr = np.linalg.svd(_jones_block(str_, orr, kr), compute_uv=False)
        sw = np.linalg.svd(_jones_block(str_, orr, k0), compute_uv=False)
        sv[name] = (float(np.abs(sf - sr).max()), float(np.abs(sf - sw).max()))
        if name == "lc":
            assert np.abs(Rf.sum(1) + Tf.sum(1) - 1).max() <= 3e-2
    assert sv["lc"][0] <= 1e-3, sv
    assert sv["lc"][1] >= 1e-2, sv
    assert sv["gyro"][0] >= 3e-3, sv


# =========================================================================== #
# D9 -- the mutation matrix: every defect caught by its discriminating gate
# =========================================================================== #
def test_d9_no_sg_e33_is_caught_only_off_normal_incidence():
    """sqrt(g) dropped from eps'_33 changes only the div(D) = 0 Schur term,
    which a plane wave at NORMAL incidence never exercises (E_z = 0).
    Measured 2026-10-03 (``d_unit_no_sg_e33.json``, LC film under the
    circle map): normal incidence M = 5: 2.81462e-8 mutated vs 2.81470e-8
    correct (blind: the gate must be oblique -- recorded, asserted <= 1e-12
    on the difference); CONICAL (25, 40), M = 6: correct 8.7e-8 (bar 1e-6),
    mutated 1.2e-4, flat in M (bar >= 1e-5)."""
    cm = _circle(_G3["P"])
    th, ph = np.deg2rad(25.0), np.deg2rad(40.0)
    good = _film(_LC, cm, 6, theta=th, phi=ph)
    n_good = _film(_LC, cm, 5)
    with _mutate("no_sg_e33"):
        bad = _film(_LC, cm, 6, theta=th, phi=ph)
        n_bad = _film(_LC, cm, 5)
    assert good[0] <= 1e-6 and bad[0] >= 1e-5, (good[0], bad[0])
    assert abs(n_good[0] - n_bad[0]) <= 1e-12


def test_d9_mixed_sign_is_rt_blind_on_a_stretched_film_and_jones_visible():
    """The mixed (e12', e21') weights negated: on the LC film under the
    a = 0.05 p stretch (diagonal J) it flips the director's in-plane angle,
    which the efficiencies of a uniform film cannot see.  Measured
    2026-10-03 (``d_unit_mut.json``, M = 5): R / T 1.21966e-7 both arms
    (difference 1.4e-14; bar 1e-12), Jones 3.0e-8 correct vs 9.8e-3
    mutated (bars 1e-7 / >= 1e-3)."""
    cs = _stretch(_G3["P"], 0.05)
    good = _film(_LC, cs, 5)
    with _mutate("mixed_sign"):
        bad = _film(_LC, cs, 5)
    assert abs(good[0] - bad[0]) <= 1e-12
    assert good[1] <= 1e-7 and bad[1] >= 1e-3, (good[1], bad[1])


# =========================================================================== #
# The shape layer: mu on shapes, the explicit route, the refusals
# =========================================================================== #
def test_d_shapes_carry_mu_and_the_shapes_route_is_the_explicit_route():
    """``compile_shapes(..., with_mu=True)`` returns the permeability grid as
    a fifth output, painted like eps (a shape without mu paints 1, the
    background takes ``background_mu``); the four-output form REFUSES a
    magnetic layer (it would drop mu silently).  The shapes route with mu
    equals ``compile_shapes(with_mu=True)`` + ``cmap=`` + ``eps_cell=`` +
    ``mu_cell=`` BYTE FOR BYTE (``d_unit_shapes.json``: bytes equal)."""
    sh = [Circle(0.6, 0.6, 0.36, _LC30, mu=np.diag([1.5, 1.5, 1.0]))]
    with pytest.raises(ValueError, match="with_mu=True"):
        compile_shapes(_P, _P, sh, 1.0)
    eps, xw, yw, cm, mu = compile_shapes(_P, _P, sh, 1.0, background_mu=1.2,
                                         with_mu=True)
    assert mu.shape == (3, 3, 3, 3)
    assert np.allclose(mu[1, 1], np.diag([1.5, 1.5, 1.0]))
    assert np.allclose(mu[0, 0], 1.2 * _EYE3)
    assert compile_shapes(_P, _P, [Circle(0.6, 0.6, 0.36, 4.0)], 1.0,
                          with_mu=True)[4] is None
    a = pmm_jones_2d_staggered(_P, _P, None, 1.45, 1.0, 0.5, 1.0, n_modes=4,
                               n_orders=3, shapes=sh, background_eps=1.0,
                               background_mu=1.2)
    b = pmm_jones_2d_staggered(_P, _P, eps, 1.45, 1.0, 0.5, 1.0, n_modes=4,
                               n_orders=3, cmap=cm, mu_cell=mu)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    with pytest.raises(AttributeError, match="immutable"):
        sh[0].mu = 2.0


def test_d_refusals_keep_out_of_plane_and_raw_mu_cells_out():
    """An OUT-OF-PLANE PERMEABILITY stays refused under a map (the solver,
    the stack's uniform layer, a shape) -- it is refused without a map too.
    An OUT-OF-PLANE PERMITTIVITY under a map is accepted since Phase E1
    (gates in ``tests/unit/test_pmm2d_staggered_curved_e1.py``; here it only
    has to build).  A raw ``mu_cell`` cannot join a shape stack (its cells
    would refer to a grid the merge does not know); a UNIFORM ``mu`` layer
    can."""
    oop = np.diag([2.0, 2.5, 2.0]).astype(complex)
    oop[0, 2] = oop[2, 0] = 0.3
    cm = _circle(_P)
    s_oop = TS.Granet2DTransverseE(_P, _P, cm.u_walls, cm.v_walls, 4,
                                   np.broadcast_to(oop, (3, 3, 3, 3)),
                                   cmap=cm)
    assert s_oop.offplane
    with pytest.raises(NotImplementedError, match="OUT-OF-PLANE"):
        TS.Granet2DTransverseE(_P, _P, cm.u_walls, cm.v_walls, 4,
                               np.ones((3, 3), complex),
                               mu_cell=np.broadcast_to(oop, (3, 3, 3, 3)),
                               cmap=cm)
    st = PMM2DStackPure(_P, _P, n_modes=4, cmap=cm)
    st.add_layer(0.2, eps=oop)
    with pytest.raises(NotImplementedError, match="OUT-OF-PLANE"):
        st.add_layer(0.2, eps=2.0, mu=oop)
    with pytest.raises(NotImplementedError, match="OUT-OF-PLANE"):
        PMM2DStackPure(_P, _P, n_modes=4).add_layer(
            0.3, shapes=[Circle(0.6, 0.6, 0.3, 4.0, mu=oop)],
            background_eps=1.0)
    st = PMM2DStackPure(_P, _P, n_modes=4)
    st.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.3, 4.0)], background_eps=1.0)
    st.add_layer(0.2, eps=2.0, mu=1.5)                 # uniform mu: joins
    with pytest.raises(ValueError, match="mu_cell"):
        st.add_layer(0.2, eps=1.0, mu_cell=np.ones((3, 3)))
