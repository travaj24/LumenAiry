"""PMM lateral-structure registration: the point-mirror bug (v5.8 .. v5.50).

WHAT WAS WRONG.  Every PMM path that takes lateral structure solved the POINT
MIRROR ``eps(-x, -y)`` of the declared cell (incidence, order labels and
uniform layers were right) -- the PMM side of the RCWA fault fixed in
``tests/unit/test_rcwa_point_mirror_fix.py``.  Measured on 5.50.0: asymmetric
lossy 3-level staircase, theta = 0.5, TE, total absorptance 0.709845
(``PMMStack``), 0.7097 (``PMM2DStack`` along x and along y), 0.70378
(``PMM2DStackPure``, 4-cell variant) against the truth 0.477864 (grcwa 0.1.2).
It arrived three different ways, one per core family:

* 1-D nodal cores (public gauge, natively registered): a ``[::-1]`` in
  ``_core._segment_elem_bnds`` and the sign of the slant convection in
  ``_sem_modes_slant`` / ``_build_generator_metric`` -- calibrations against
  the then-mirrored RCWA.  The 1-D covariant core (internal gauge) solves the
  mirror of what it is handed and now gets the mirror (``_internal_mirror_1d``).
* 2-D hybrid (internal gauge, public-form operators): fixed at ONE place, the
  element table of ``twod._build_axis`` laid out in the internal frame
  ``x' = P - x``, plus the slant crossing into that frame
  (``twod._internal_slant``).
* 2-D staggered / ``PMM2DStackPure`` (public gauge, but the Bloch glue ran
  ``exp(-i alpha0 x)``: a 180-degree-rotated solve): the glue
  (``twod_staggered._stag_bloch_tau``), both far-field kernels and the mapped
  incident load now run the public carrier, so ``_OOP_ROT_SIGN`` is ``+1``
  and the slant is no longer negated.  Curved maps / ``shapes=`` need no
  geometry rewrite at all.
* Every sheared cascade re-references transmitted amplitudes with ONE kernel,
  ``_core._frame_anchor_phase`` (``exp(-i k0 alpha . W)``; the old ``+i`` was
  the mirror image's).

These tests are grcwa-free.  The discriminating ones are TRANSLATION (moving
the structure by ``a`` multiplies order ``m`` by ``exp(-i m G a)``; the
mirror gave ``exp(+i ...)``), the BLAZE direction, first-order BORN on a weak
lossy OOP tensor, field REGISTRATION against the (fixed) RCWA field, the
slant direction against the engines' own vertical staircases, and the
pinned grcwa absorptance.  A 180-degree rotation test is NOT one of them: a
point mirror about the origin commutes with that rotation.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy.elements.pmm as pmm
from lumenairy.elements.pmm._core import (
    _frame_anchor_phase,
    _internal_mirror_1d,
    _segment_elem_bnds,
)
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.pmm.twod import _build_axis, _internal_slant
from lumenairy.elements.pmm.twod_staggered import _OOP_ROT_SIGN, _stag_bloch_tau
from lumenairy.elements.rcwa import RCWAStack

WL, P, D = 1.0, 1.7, 0.35
A_, B_ = 6.0 + 2.0j, 3.0 + 0.3j
SEGS = [(0.30, A_), (0.25, B_), (0.45, 1.0 + 0j)]       # from x = 0
A_TRUE = 0.477864          # grcwa 0.1.2, TE, theta = 0.5 (5.50.0: 0.7098)
A_MIRR = 0.709830
A4_TRUE, A4_MIRR = 0.483870, 0.703778   # 4-pixel staircase (grcwa)


def _A(R, T, row):
    return 1.0 - float(np.sum(R[row])) - float(np.sum(T[row]))


# ---------------------------------------------------------------------------
# the bridge pieces themselves
# ---------------------------------------------------------------------------

def test_segment_layout_is_declared_order():
    eb = _segment_elem_bnds(1.0, [0.2, 0.3, 0.5], ["a", "b", "c"], 1, False)
    assert [m for (_l, _r, m) in eb] == ["a", "b", "c"]
    assert eb[0][0] == 0.0 and abs(eb[0][1] - 0.2) < 1e-15


def test_covariant_bridge_is_the_point_mirror():
    ta = np.arange(9.0).reshape(3, 3) + 1j
    w, t, s = _internal_mirror_1d([0.2, 0.3, 0.5], [ta, 2 * ta, 3 * ta], 0.4)
    assert w == [0.5, 0.3, 0.2] and s == -0.4
    m = np.diag([-1.0, 1.0, 1.0])                 # the x-mirror of a tensor
    for got, want in zip(t, (3 * ta, 2 * ta, ta)):
        assert np.array_equal(got, m @ want @ m)  # exy, eyx, exz, ezx flip


def test_hybrid_axis_is_laid_out_in_the_internal_frame():
    ax = _build_axis(2.0, [0.5], 4, 1, False)            # strips [0,.5) [.5,2)
    assert [s for (_l, _r, s) in ax["elem_bnds"]] == [1, 0]
    (l0, r0, _), (l1, r1, _) = ax["elem_bnds"]
    assert (l0, r0, l1, r1) == (0.0, 1.5, 1.5, 2.0)      # x' = P - x
    # a wall-less axis is its own mirror and is left untouched
    ay = _build_axis(2.0, [], 4, 2, False)
    assert [(l, r) for (l, r, _s) in ay["elem_bnds"]] == [(0.0, 1.0), (1.0, 2.0)]
    assert _internal_slant((0.3, -0.2)) == (-0.3, 0.2)
    assert _internal_slant(None) is None


def test_staggered_glue_runs_the_public_carrier():
    assert _OOP_ROT_SIGN == 1.0
    a0, p = 2.3, 1.7
    assert abs(_stag_bloch_tau(a0, p) - np.exp(1j * a0 * p)) < 1e-15


def test_frame_anchor_sign():
    k0, kx, w = 2 * np.pi, np.array([0.3, -0.4]), 0.25
    assert np.allclose(_frame_anchor_phase(k0, kx, w), np.exp(-1j * k0 * kx * w))


# ---------------------------------------------------------------------------
# the pinned measurement (grcwa truth) -- every 1-D entry and both 2-D stacks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry", ["stack", "jones_segments", "eff_segments"])
def test_1d_asymmetric_staircase_absorptance_is_the_declared_one(entry):
    th = 0.5
    if entry == "stack":
        st = pmm.PMMStack(P, n_substrate=1.0, n_superstrate=1.0, degree=16)
        st.add_layer(D, segments=SEGS)
        _o, R, T, _J = st.set_source(WL, angle=th).solve()
        A = _A(np.asarray(R), np.asarray(T), 1)
    elif entry == "jones_segments":
        _o, R, T, _J = pmm.pmm_jones_1d_segments(P, SEGS, 1.0, 1.0, D, WL, angle=th,
                                                 degree=16)
        A = _A(np.asarray(R), np.asarray(T), 1)
    else:
        _o, R, T = pmm.pmm_efficiency_1d_segments(
            P, [(w, np.sqrt(e)) for w, e in SEGS], 1.0, 1.0, D, WL, angle=th,
            polarization="te", degree=16)
        A = 1.0 - float(np.sum(R)) - float(np.sum(T))
    assert abs(A - A_TRUE) < 2e-4, A
    assert abs(A - A_MIRR) > 0.2


def test_2d_stacks_on_the_staircase_along_x_and_along_y():
    row = np.array([A_, B_, 1, 1], complex)               # 4 footprint pixels
    for cell, phi, k in ((row[:, None], 0.0, 1), (row[None, :], np.pi / 2, 0)):
        h = pmm.PMM2DStack(P, P, degree=9, n_orders=8)
        h.add_layer(D, eps_cell=cell)
        _o, R, T = tuple(h.set_source(WL, theta=0.5, phi=phi).solve())[:3]
        A = _A(np.asarray(R), np.asarray(T), k)
        # the 4-pixel staircase (edges 0.25 / 0.5): grcwa 0.1.2 0.483870,
        # its point mirror 0.703778 (5.50.0 returned the latter)
        assert abs(A - A4_TRUE) < 5e-3 and abs(A - A4_MIRR) > 0.2, A
    p = PMM2DStackPure(P, P, n_modes=6, n_orders=5)
    p.add_layer(D, eps_cell=np.repeat(row[:, None], 4, 1))
    _o, R, T = tuple(p.set_source(WL, theta=0.5, phi=0.0).solve())[:3]
    A = _A(np.asarray(R), np.asarray(T), 1)
    assert abs(A - A4_TRUE) < 2e-3 and abs(A - A4_MIRR) > 0.2, A


# ---------------------------------------------------------------------------
# TRANSLATION: exp(-i m G a), not exp(+i m G a)
# ---------------------------------------------------------------------------

def _translation_errors(a0, a1, ph):
    """(error of the declared law, error of the mirrored law) over both
    ports' tangential amplitudes, relative to the largest amplitude."""
    e_d = e_m = 0.0
    for port in ("reflection", "transmission"):
        for c in ("Ex", "Ey"):
            u, v = np.asarray(a0[port][c]), np.asarray(a1[port][c])
            sc = max(np.max(np.abs(u)), 1e-30)
            e_d = max(e_d, np.max(np.abs(v - u * ph)) / sc)
            e_m = max(e_m, np.max(np.abs(v - u / ph)) / sc)
    return e_d, e_m


def _amps(st):
    return {p: st.per_order_amplitudes(p) for p in ("reflection", "transmission")}


def test_1d_translation_law():
    rot = [SEGS[2], SEGS[0], SEGS[1]]                     # shifted by +0.45 P
    out = []
    for segs in (SEGS, rot):
        st = pmm.PMMStack(P, n_substrate=1.5, degree=14)
        st.add_layer(D, segments=segs)
        st.set_source(WL, angle=0.4).solve()
        out.append(_amps(st))
    o = np.asarray(out[0]["reflection"]["orders"])
    ph = np.exp(-2j * np.pi * o * 0.45)
    e_d, e_m = _translation_errors(out[0], out[1], ph)
    assert e_d < 1e-9 and e_m > 0.1, (e_d, e_m)


@pytest.mark.parametrize("engine", ["hybrid", "pure"])
def test_2d_translation_law(engine):
    # 5 x 5 so that every axis has an ODD node count (no periodic-Nyquist
    # parity bump, which would put its extra element on a different PHYSICAL
    # strip of the rolled cell): the rolled mesh is then an exact translate
    c5 = np.ones((5, 5), complex)
    c5[0, 0] = c5[1, 0] = c5[0, 1] = A_
    c5[2, 0] = c5[0, 2] = c5[1, 1] = B_
    out = []
    for cell in (c5, np.roll(np.roll(c5, 1, axis=0), 1, axis=1)):
        if engine == "hybrid":
            st = pmm.PMM2DStack(P, 1.5, degree=7, n_orders=4, n_substrate=1.5)
        else:
            st = PMM2DStackPure(P, 1.5, n_modes=5, n_orders=3, n_substrate=1.5)
        st.add_layer(D, eps_cell=cell)
        st.set_source(WL, theta=0.4, phi=0.7).solve()
        out.append(_amps(st))
    o = np.asarray(out[0]["reflection"]["orders"])
    ph = np.exp(-2j * np.pi * (o[:, 0] + o[:, 1]) / 5.0)   # one pixel each axis
    e_d, e_m = _translation_errors(out[0], out[1], ph)
    assert e_d < 1e-6 and e_m > 0.1, (e_d, e_m)


# ---------------------------------------------------------------------------
# BLAZE: a staircase rising toward +x sends its power toward +x
# ---------------------------------------------------------------------------

def test_blaze_direction_all_engines():
    Pb, db = 5.7, 1.0
    vals = np.array([2.25, 3.0, 4.0], complex)            # thick side at +x
    res = {}
    st = pmm.PMMStack(Pb, degree=14)
    st.add_layer(db, segments=[(1 / 3, v) for v in vals])
    o, _R, T, _J = st.set_source(WL, angle=0.0).solve()
    res["PMMStack"] = (np.asarray(o), np.asarray(T)[1])
    h = pmm.PMM2DStack(Pb, 1.0, degree=9, n_orders=8)
    h.add_layer(db, eps_cell=vals[:, None])
    o, _R, T = tuple(h.set_source(WL, theta=0.0, phi=0.0).solve())[:3]
    res["PMM2DStack"] = (np.asarray(o)[:, 0], np.asarray(T)[1])
    p = PMM2DStackPure(Pb, 1.0, n_modes=6, n_orders=7)
    p.add_layer(db, eps_cell=np.repeat(vals[:, None], 3, 1))
    o, _R, T = tuple(p.set_source(WL, theta=0.0, phi=0.0).solve())[:3]
    o = np.asarray(o)
    keep = o[:, 1] == 0
    res["PMM2DStackPure"] = (o[keep, 0], np.asarray(T)[1][keep])
    for name, (o, t) in res.items():
        assert o[int(np.argmax(t))] == 1, name               # (5.50.0: m = -1)
        assert t[o > 0].sum() > 2.0 * t[o < 0].sum(), name


# ---------------------------------------------------------------------------
# BORN: weak lossy OOP tensor segments (catches the mirror and an xz/yz flip)
# ---------------------------------------------------------------------------

def test_1d_oop_segments_first_order_born():
    rng = np.random.default_rng(7)

    def sym(a):
        m = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) * a
        return 0.5 * (m + m.T)
    w = [0.2, 0.3, 0.15, 0.35]
    e = np.r_[0.0, np.cumsum(w)]
    v = [sym(0.003) for _ in w]
    bg = 1.0 + 0.025j
    Pp, dd, th = 2.3, 0.04, 0.35
    o, R, T, _J = pmm.pmm_jones_1d_segments(Pp, [(wi, bg * np.eye(3) + vi) for wi, vi in zip(w, v)],
                                            1.0, 1.0, dd, WL, angle=th, degree=12)
    o, R, T = np.asarray(o), np.asarray(R), np.asarray(T)

    def born(q, row, flip):
        qq = -q if flip else q
        dq = sum(vk * wk * np.sinc(qq * wk) * np.exp(-1j * np.pi * qq * (e[k] + e[k + 1]))
                 for k, (wk, vk) in enumerate(zip(w, v)))
        k0 = 2 * np.pi / WL
        a0, g0 = np.sin(th), np.cos(th)
        inc = (1.0, 0.0) if row == 0 else (0.0, 1.0)
        ev = np.array([inc[0], inc[1], -a0 * inc[0] / g0], complex)
        ev /= np.linalg.norm(ev)
        a = a0 + q * WL / Pp
        g = np.sqrt(1 - a * a)
        etas = []
        for sgn, phz in ((-1, g0 + g), (1, g0 - g)):
            kv = np.array([a, 0.0, sgn * g])
            integ = (np.exp(1j * phz * dd * k0) - 1) / (1j * phz)
            t = 1j / (2 * g) * ((np.eye(3) - np.outer(kv, kv)) @ (dq @ ev)) * integ
            etas.append(g / g0 * np.sum(np.abs(t) ** 2))
        return etas
    err = {False: 0.0, True: 0.0}
    for flip in (False, True):
        for row in (0, 1):
            for i, q in enumerate(o):
                a = np.sin(th) + q * WL / Pp
                if q == 0 or abs(a) >= 1:
                    continue
                eR, eT = born(int(q), row, flip)
                err[flip] = max(err[flip], abs(R[row, i] - eR) / eR, abs(T[row, i] - eT) / eT)
    assert err[False] < 0.08 and err[True] > 0.5, err


# ---------------------------------------------------------------------------
# SLANT direction (walls walk toward +x with depth) and the frame anchor
# ---------------------------------------------------------------------------

def _sheared(fac, shear):
    st = pmm.PMMStack(P, degree=12, factorization=fac)
    return st.add_sheared_grating(D, eps_ridge=3.0 + 0.5j, eps_groove=1.0, duty=0.35,
                                  shear=shear, centre=0.4)


def _staircase(shear, nz=16):
    st = pmm.PMMStack(P, degree=10)
    return st.add_tapered_grating(D, eps_ridge=3.0 + 0.5j, eps_groove=1.0, duty_bottom=0.35,
                                  duty_top=0.35, n_slices=nz, shear=shear)


@pytest.mark.parametrize("fac", ["convection", "covariant"])
def test_1d_slanted_layer_walks_toward_plus_x(fac):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R, T, _J = _sheared(fac, 0.3).set_source(WL, angle=0.4).solve()
        ref = {}
        for sh in (0.3, -0.3):
            _o2, R2, T2, _J2 = _staircase(sh).set_source(WL, angle=0.4).solve()
            ref[sh] = max(np.max(np.abs(np.asarray(R) - np.asarray(R2))),
                          np.max(np.abs(np.asarray(T) - np.asarray(T2))))
    assert ref[0.3] < 0.01 and ref[-0.3] > 3 * ref[0.3], ref


def test_1d_frame_anchor_matches_the_staircase():
    st = pmm.PMMStack(P, n_substrate=1.5, degree=12, factorization="convection")
    st.add_sheared_grating(D, eps_ridge=3.0 + 0.2j, eps_groove=1.0, duty=0.4, shear=0.25)
    st.set_source(WL, angle=0.4).solve()
    sc = pmm.PMMStack(P, n_substrate=1.5, degree=10)
    sc.add_tapered_grating(D, eps_ridge=3.0 + 0.2j, eps_groove=1.0, duty_bottom=0.4,
                           duty_top=0.4, n_slices=24, shear=0.25)
    sc.set_source(WL, angle=0.4).solve()
    a, b = st.per_order_amplitudes("transmission"), sc.per_order_amplitudes("transmission")
    prop = np.real(np.asarray(a["kz"])) > 1e-6
    for c in ("Ex", "Ey"):
        u, v = np.asarray(a[c])[:, prop], np.asarray(b[c])[:, prop]
        assert np.max(np.abs(u - v)) / np.max(np.abs(v)) < 0.03


@pytest.mark.parametrize("engine", ["hybrid", "pure"])
def test_2d_slant_on_a_y_uniform_cell_matches_the_1d_slant(engine):
    t = 0.5
    row = np.array([4.0 + 1.0j, 2.5 + 0.3j, 1.0, 1.0], complex)
    ref = {}
    for sgn in (1.0, -1.0):
        st = pmm.PMMStack(P, degree=14, factorization="convection")
        st.add_layer(D, segments=[(0.25, row[0]), (0.25, row[1]), (0.5, 1.0)],
                     slant_angle=np.arctan(sgn * t))
        o, R, T, _J = st.set_source(WL, angle=0.4).solve()
        ref[sgn] = (np.asarray(o), np.asarray(R), np.asarray(T))
    if engine == "hybrid":
        s = pmm.PMM2DStack(P, P, degree=9, n_orders=8)
        s.add_layer(D, eps_cell=row[:, None], slant=(t, 0.0))
    else:
        s = PMM2DStackPure(P, P, n_modes=6, n_orders=5)
        s.add_layer(D, eps_cell=np.repeat(row[:, None], 4, 1), slant=(t, 0.0))
    o2, R2, T2 = tuple(s.set_source(WL, theta=0.4, phi=0.0).solve())[:3]
    o2 = np.asarray(o2)
    keep = o2[:, 1] == 0

    def err(sgn):
        o1, R1, T1 = ref[sgn]
        e = 0.0
        for j, m in zip(np.nonzero(keep)[0], o2[keep, 0]):
            if m in o1:
                i = int(np.nonzero(o1 == m)[0][0])
                e = max(e, abs(R2[1, j] - R1[1, i]), abs(T2[1, j] - T1[1, i]))
        return e
    assert err(1.0) < 0.01 and err(-1.0) > 3 * err(1.0), (err(1.0), err(-1.0))


# ---------------------------------------------------------------------------
# FIELD registration against the (fixed, grcwa-validated) RCWA field
# ---------------------------------------------------------------------------

def test_1d_internal_field_is_registered():
    th, z = 0.5, 0.5 * D
    edges = np.r_[0.0, np.cumsum([w for w, _ in SEGS])]

    def eps(u):
        u = np.asarray(u) % 1.0
        i = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, len(SEGS) - 1)
        return np.array([SEGS[k][1] for k in i])
    S, nx = 1024, 32
    u = np.arange(S) / S
    st = RCWAStack(P, n_orders=30)
    st.add_layer(D, eps_cell=eps(u)[:, None], formulation="li")
    fr = st.set_source(WL, theta=th).solve(retain_internal=True).internal_field(
        z, component="E", nx=nx, incident=(0.0, 1.0))
    ps = pmm.PMMStack(P, degree=16)
    ps.add_layer(D, segments=SEGS)
    ps.set_source(WL, angle=th).solve(retain_internal=True)
    fp = ps.internal_field(z, component="E", pol=1, nx=nx)
    xr, er = np.asarray(fr["x"]).ravel(), np.asarray(fr["Ey"]).ravel()
    xp, ep = np.asarray(fp["x"]).ravel(), np.asarray(fp["Ey"]).ravel()
    k0, kx0 = 2 * np.pi / WL, np.sin(th) * 2 * np.pi / WL
    idx = [int(np.argmin(np.abs((xp - x) / P - np.round((xp - x) / P)))) for x in xr]
    em = ep[idx] * np.exp(1j * kx0 * (xr - xp[idx]))
    assert k0 > 0
    assert np.linalg.norm(em - er) / np.linalg.norm(er) < 1e-2
