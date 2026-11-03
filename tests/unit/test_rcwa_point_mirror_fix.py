"""RCWA lateral-structure registration: the v5.5.0..v5.50.0 POINT-MIRROR bug.

WHAT WAS WRONG.  The public gauge is ``exp(-i w t)`` with harmonics
``exp(+i kx_m x)``; the eigenmode core runs in ``exp(+i w t)``, reached by
complex conjugation.  Conjugation flips the transverse carrier too, so the
internal Toeplitz must be ``conj(c_{m-m'})``.  Every bridge fed ``conj(eps)``
to kernels that implement the PUBLIC rule ``FT_{m-m'}``, which returns
``conj(c_{m'-m})`` -- the Toeplitz of ``conj(eps(-x, -y))``.  ``RCWAStack``
(1-D/2-D, ``eps_cell``, ``shapes=``, tensors) and the standalone 2-D /
segment entries therefore solved the POINT MIRROR of the declared cell while
incidence and order labels kept their public meaning.  Measured (scratch
harness ``t5``): asymmetric lossy staircase at theta = 0.5, TE, total
absorptance 0.709828 against the truth 0.477864 (grcwa 0.1.2 and an
independent TE RCWA); after the fix 0.477867.

THE FIX (``rcwa._core._internal_cell`` / ``_internal_profiles_1d``,
``rcwa.twod._internal_analytic_convolution``) is a pure coordinate flip of the
pattern at the bridge -- node ``j -> -j mod S`` per lateral axis, mirrored
segment edges -- with tensor COMPONENTS untouched; for analytic shapes, the
equivalent conjugate of the PUBLIC operator (exact for any form factor).

WHY THESE TESTS AND NOT A ROTATION TEST.  The bug is a coordinate point
mirror ``Pi`` about the ORIGIN.  ``Pi`` commutes with every origin-fixed
linear map ``g`` (rotation by pi, mirrors), so a ``Pi``-solver obeys
``F(Pi g eps, g k) = F(Pi eps, k)``: rotation / mirror equivariance holds on
the BUGGY build too (measured: 11/11 on 5.50.0).  It is kept below only as a
consistency gate for the tensor paths.  What does discriminate:

* the first-order BORN oracle for a weak, thin, LOSSY pattern whose Fourier
  coefficients are not Hermitian-paired (``|t_+m| != |t_-m|``), which also
  catches an xz / yz sign error -- analytic, so no third-party solver in CI;
* TRANSLATION: ``Pi T_a = T_{-a} Pi``, so shifting the structure by ``a``
  must shift the reconstructed field by ``+a`` (the bug moved it by ``-a``),
  and the order phases must pick up ``exp(-i m K a)``;
* absolute REGISTRATION of ``internal_field``: TM ``E_x`` jumps at the
  declared walls;
* the blaze physics: a staircase whose index rises toward +x sends its power
  toward +x at normal incidence;
* the bridge invariant itself: the kernels' Toeplitz of the internal cell is
  the complex conjugate of the PUBLIC Toeplitz.

The full external-oracle evidence (grcwa per-order R/T in 1-D and 2-D, at
phi = 0, pi/2, 0.7; grcwa real-space fields; FD-checked JAX gradients; the
5.50.0 A/B) lives in the fix's scratch gates, not in CI: grcwa is not a test
dependency.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.rcwa import (
    RCWAStack,
    prepare_rcwa_2d,
    rcwa_efficiency_2d,
    rcwa_efficiency_2d_vs_wavelength,
    rcwa_jones_1d_segments,
    rcwa_jones_2d,
)
from lumenairy.elements.rcwa import stack as _stack_mod
from lumenairy.elements.rcwa._core import (
    _internal_cell,
    _internal_profiles_1d,
    _toeplitz_of_profile,
)
from lumenairy.elements.rcwa.twod import (
    _analytic_convolutions_2d,
    _eps_convolution_2d,
    _harmonic_orders_2d,
    _internal_analytic_convolution,
)

WL = 1.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _born(delta, px, py, d, theta, phi, inc_t, *, flip=False, oop_sign=1.0):
    """First-order Born efficiencies ``{q: (eta_R, eta_T)}`` of a thin weak
    layer ``eps = eps_bg I + sum_q delta_q exp(i q.G r)`` in vacuum
    (``exp(-i w t)``, ``k0 = 1`` units):
    ``t_m = i/(2 g_m) (I - k_m k_m^T) delta_m e_inc int_0^d exp(i (g_0 -+ g_m) z) dz``.
    ``flip`` = the point-mirrored pattern (``delta_q -> delta_-q``),
    ``oop_sign=-1`` = xz / yz (and zx / zy) sign-flipped."""
    k0 = 2 * np.pi / WL
    D = d * k0
    a0, b0 = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)
    g0 = np.cos(theta)
    e = np.array([inc_t[0], inc_t[1], -(a0 * inc_t[0] + b0 * inc_t[1]) / g0], complex)
    e /= np.linalg.norm(e)
    S = np.diag([oop_sign, oop_sign, 1.0])
    out = {}
    for q, dq in delta.items():
        dm = S @ (delta[(-q[0], -q[1])] if flip else dq) @ S
        a = a0 + q[0] * WL / px
        b = b0 + q[1] * WL / py
        g2 = 1 - a * a - b * b
        if g2 <= 0:
            continue
        g = np.sqrt(g2)
        etas = []
        for sgn, kz in ((-1, g0 + g), (+1, g0 - g)):
            k = np.array([a, b, sgn * g])
            integ = (np.exp(1j * kz * D) - 1) / (1j * kz) if abs(kz) > 1e-12 else D
            t = 1j / (2 * g) * ((np.eye(3) - np.outer(k, k)) @ (dm @ e)) * integ
            etas.append(g / g0 * float(np.sum(np.abs(t) ** 2)))
        out[q] = tuple(etas)
    return out


def _max_rel_vs_born(orders, R, T, delta, px, py, d, theta, phi, rows, **kw):
    o = np.asarray(orders).reshape(len(orders), -1)
    if o.shape[1] == 1:
        o = np.c_[o, np.zeros_like(o)]
    err = 0.0
    for row in rows:
        inc_t = (1.0, 0.0) if row == 0 else (0.0, 1.0)
        for q, (eR, eT) in _born(delta, px, py, d, theta, phi, inc_t, **kw).items():
            i = int(np.nonzero((o[:, 0] == q[0]) & (o[:, 1] == q[1]))[0][0])
            for lv, bv in ((R[row][i], eR), (T[row][i], eT)):
                err = max(err, abs(float(lv) - bv) / bv)
    return err


def _sym_rand(rng, amp):
    A = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) * amp
    return 0.5 * (A + A.T)


def _band_limited_cell(delta, sx, sy, bg):
    X, Y = np.meshgrid(np.arange(sx) / sx, np.arange(sy) / sy, indexing="ij")
    c = np.zeros((sx, sy, 3, 3), complex) + bg * np.eye(3)
    for q, dq in delta.items():
        c = c + np.exp(2j * np.pi * (q[0] * X + q[1] * Y))[..., None, None] * dq
    return c


BG = 1.0 + 0.05j      # lossy background: keeps the weak pattern passive


# ---------------------------------------------------------------------------
# 1. the bridge invariant
# ---------------------------------------------------------------------------

def test_internal_cell_toeplitz_is_conjugate_of_public_toeplitz():
    """THE contract: the kernels (public rule FT_{m-m'}) on the internal cell
    must return conj of the PUBLIC Toeplitz, conj(c_{m-m'}).  Odd and even S,
    1-D (Sy = 1), 2-D, and a (Sx, Sy, 3, 3) tensor (components untouched)."""
    rng = np.random.default_rng(1)
    for sx, sy, mx, my in ((17, 1, 4, 0), (24, 1, 5, 0), (12, 15, 2, 3)):
        orders, _ = _harmonic_orders_2d(mx, my)
        cell = rng.normal(size=(sx, sy)) + 1j * rng.normal(size=(sx, sy))
        pub = _eps_convolution_2d(cell, orders, mx, my)
        intl = _eps_convolution_2d(_internal_cell(cell, np), orders, mx, my)
        np.testing.assert_allclose(intl, np.conj(pub), atol=1e-14)
        # and it is NOT the old bridge (conj only), which is the transpose
        old = _eps_convolution_2d(np.conj(cell), orders, mx, my)
        assert np.max(np.abs(old - np.conj(pub))) > 0.1
        np.testing.assert_allclose(old, np.conj(pub).T, atol=1e-14)
    t = rng.normal(size=(9, 7, 3, 3)) + 1j * rng.normal(size=(9, 7, 3, 3))
    ti = _internal_cell(t, np)
    for a in range(3):
        for b in range(3):
            np.testing.assert_array_equal(
                ti[..., a, b], _internal_cell(t[..., a, b], np))


def test_internal_shapes_toeplitz_is_conjugate_of_public_toeplitz():
    px, py, mx, my = 1.3, 1.1, 3, 2
    orders, _ = _harmonic_orders_2d(mx, my)
    shapes = [dict(shape="rectangle", eps=4.0 + 1.0j, size=(0.4, 0.3), center=(0.3, 0.2)),
              dict(shape="ellipse", eps=2.0 + 0.2j, semi_axes=(0.12, 0.2), center=(0.9, 0.7)),
              dict(shape="disk", eps=6.0, radius=0.1)]           # default centre
    pub, _ = _analytic_convolutions_2d(1.5 + 0.1j, shapes, orders, mx, my, px, py)
    intl = _internal_analytic_convolution(1.5 + 0.1j, shapes, orders, mx, my, px, py)
    np.testing.assert_array_equal(intl, np.conj(pub))
    # equivalently: the same shapes with NEGATED centres (each primitive is
    # centro-symmetric about its own centre) and conjugated eps
    neg = [dict(s, eps=np.conj(s["eps"]),
                center=tuple(-c for c in s.get("center", (px / 2, py / 2))))
           for s in shapes]
    alt, _ = _analytic_convolutions_2d(np.conj(1.5 + 0.1j), neg, orders, mx, my, px, py)
    np.testing.assert_allclose(intl, alt, atol=1e-14)
    # and NOT the old bridge (conjugated eps only) -- that is the transpose
    old, _ = _analytic_convolutions_2d(
        np.conj(1.5 + 0.1j), [dict(s, eps=np.conj(s["eps"])) for s in shapes],
        orders, mx, my, px, py)
    np.testing.assert_allclose(old, intl.T, atol=1e-14)
    assert np.max(np.abs(old - intl)) > 0.01


def test_internal_profiles_1d_toeplitz_is_conjugate_of_public_toeplitz():
    rng = np.random.default_rng(2)
    vals = rng.normal(size=4) + 1j * rng.normal(size=4)
    edges = (0.0, 0.13, 0.4, 0.71, 1.0)
    pub = _toeplitz_of_profile(vals, 5, edges)
    prof, e2 = _internal_profiles_1d({"xx": np.conj(vals)}, edges, np)
    assert e2[0] == 0.0 and e2[-1] == 1.0
    np.testing.assert_allclose(_toeplitz_of_profile(prof["xx"], 5, e2), np.conj(pub),
                               atol=1e-14)
    samp = rng.normal(size=23) + 1j * rng.normal(size=23)          # node sampling
    pub = _toeplitz_of_profile(samp, 5)
    prof, e2 = _internal_profiles_1d({"xx": np.conj(samp)}, None, np)
    assert e2 is None
    np.testing.assert_allclose(_toeplitz_of_profile(prof["xx"], 5), np.conj(pub), atol=1e-14)


# ---------------------------------------------------------------------------
# 2. Born oracle (discriminating, analytic)
# ---------------------------------------------------------------------------

def test_born_oracle_iso_1d_cell_shapes_segments():
    """Weak thin lossy two-bar 1-D grating, TE and TM, theta = 0 and 0.35:
    RCWAStack(eps_cell), RCWAStack(shapes=) and rcwa_jones_1d_segments match
    first-order Born to its O(delta) residual and NOT the mirrored pattern."""
    P, d, M = 2.3, 0.04, 8
    bars = [(0.10, 0.45, 0.012 + 0.010j), (0.55, 0.70, 0.004 + 0.018j)]   # (lo, hi, delta)
    delta = {}
    for q in (1, 2, -1, -2):
        c = sum(dl * (hi - lo) * np.sinc(q * (hi - lo)) * np.exp(-1j * np.pi * q * (lo + hi))
                for lo, hi, dl in bars)
        delta[(q, 0)] = c * np.eye(3)
    segs, x = [], 0.0
    for lo, hi, dl in bars:
        segs += [(lo - x, BG), (hi - lo, BG + dl)]
        x = hi
    segs.append((1 - x, BG))
    for theta in (0.0, 0.35):
        res = {}
        st = RCWAStack(P, n_superstrate=1.0, n_substrate=1.0, n_orders=M)
        st.add_layer(d, shapes=[dict(shape="rectangle", eps=BG + dl, size=((hi - lo) * P, st.period_y),
                                     center=(0.5 * (lo + hi) * P, 0.5 * st.period_y))
                                for lo, hi, dl in bars], eps_background=BG)
        res["shapes"] = st.set_source(WL, theta=theta).solve().efficiencies()
        o, R, T, _J = rcwa_jones_1d_segments(P, segs, 1.0, 1.0, d, WL, theta=theta, n_orders=M)
        res["segments"] = (o, R, T)
        for name, (o, R, T) in res.items():
            good = _max_rel_vs_born(o, np.asarray(R), np.asarray(T), delta, P, 1.0, d,
                                    theta, 0.0, (0, 1))
            bad = _max_rel_vs_born(o, np.asarray(R), np.asarray(T), delta, P, 1.0, d,
                                   theta, 0.0, (0, 1), flip=True)
            assert good < 0.1, (name, theta, good)
            assert bad > 0.5, (name, theta, bad)
    # band-limited raster through eps_cell (exact Fourier data on the node lattice)
    rng = np.random.default_rng(3)
    dl = {(q, 0): (rng.normal() + 1j * rng.normal()) * 0.006 * np.eye(3)
          for q in (1, 2, -1, -2)}
    cell = _band_limited_cell(dl, 64, 1, BG)[:, :, 0, 0]
    st = RCWAStack(P, n_superstrate=1.0, n_substrate=1.0, n_orders=M)
    st.add_layer(d, eps_cell=cell)
    o, R, T = st.set_source(WL, theta=0.35).solve().efficiencies()
    assert _max_rel_vs_born(o, np.asarray(R), np.asarray(T), dl, P, 1.0, d, 0.35, 0.0,
                            (0, 1)) < 0.1
    assert _max_rel_vs_born(o, np.asarray(R), np.asarray(T), dl, P, 1.0, d, 0.35, 0.0,
                            (0, 1), flip=True) > 0.5


@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (0.4, 0.7)])
def test_born_oracle_offplane_tensor_2d(theta, phi):
    """Weak lossy band-limited FULL symmetric tensor pattern (xz, yz, xy all
    patterned), 2-D, conical: RCWAStack(eps_tensor_cell) and rcwa_jones_2d
    match Born, not the mirrored pattern, not the xz/yz-sign-flipped one."""
    rng = np.random.default_rng(7)
    qs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    delta = {q: _sym_rand(rng, 0.003) for q in qs}
    px, py, d, M = 1.9, 1.6, 0.04, 2
    cell = _band_limited_cell(delta, 16, 16, BG)
    st = RCWAStack(px, period_y=py, n_superstrate=1.0, n_substrate=1.0, n_orders=M,
                   n_orders_y=M)
    st.add_layer(d, eps_tensor_cell=cell)
    o, R, T = st.set_source(WL, theta=theta, phi=phi).solve().efficiencies()
    o2, R2, T2, _J = rcwa_jones_2d(px, py, cell, 1.0, 1.0, d, WL, theta=theta, phi=phi,
                                   n_orders_x=M, n_orders_y=M)
    for oo, RR, TT in ((o, R, T), (o2, R2, T2)):
        RR, TT = np.asarray(RR), np.asarray(TT)
        args = (oo, RR, TT, delta, px, py, d, theta, phi, (0, 1))
        assert _max_rel_vs_born(*args) < 0.12
        assert _max_rel_vs_born(*args, flip=True) > 0.5
        assert _max_rel_vs_born(*args, oop_sign=-1.0) > 0.5


def test_born_oracle_offplane_tensor_1d_stack():
    rng = np.random.default_rng(11)
    delta = {(q, 0): _sym_rand(rng, 0.0035) for q in (1, 2, -1, -2)}
    P, d, M = 2.3, 0.04, 6
    cell = _band_limited_cell(delta, 48, 1, BG)
    for theta in (0.0, 0.35):
        st = RCWAStack(P, n_superstrate=1.0, n_substrate=1.0, n_orders=M)
        st.add_layer(d, eps_tensor_cell=cell)
        o, R, T = st.set_source(WL, theta=theta).solve().efficiencies()
        args = (o, np.asarray(R), np.asarray(T), delta, P, 1.0, d, theta, 0.0, (0, 1))
        assert _max_rel_vs_born(*args) < 0.1
        assert _max_rel_vs_born(*args, flip=True) > 0.5
        assert _max_rel_vs_born(*args, oop_sign=-1.0) > 0.5


# ---------------------------------------------------------------------------
# 3. translation (discriminating: Pi T_a = T_-a Pi)
# ---------------------------------------------------------------------------

def _bar_stack(lo, hi, eps, mode, P=0.8, d=0.3, M=12, S=240):
    st = RCWAStack(P, n_superstrate=1.0, n_substrate=1.0, n_orders=M)
    if mode == "cell":
        u = np.arange(S) / S
        st.add_layer(d, eps_cell=np.where((u >= lo) & (u < hi), eps, 1.0 + 0j)[:, None])
    else:
        st.add_layer(d, shapes=[dict(shape="rectangle", eps=eps, size=((hi - lo) * P, st.period_y),
                                     center=(0.5 * (lo + hi) * P, 0.5 * st.period_y))],
                     eps_background=1.0)
    return st


@pytest.mark.parametrize("mode", ["cell", "shapes"])
def test_order_phases_follow_a_translation(mode):
    """Bar shifted by a: t_m -> t_m exp(-i m K a) (exp(-i w t), kx_m = kx0 + m K)."""
    P, a = 0.8, 0.25
    amps = []
    for lo in (0.10, 0.10 + a):
        st = _bar_stack(lo, lo + 0.3, 12.0 + 3.0j, mode, P=P)
        r = st.set_source(WL, theta=0.3).solve()
        amps.append(r.per_order_amplitudes(port="transmission"))
    o = np.asarray(amps[0]["orders"]).ravel()
    ph = np.exp(-2j * np.pi * o * a)
    for c in ("Ex", "Ey"):
        np.testing.assert_allclose(amps[1][c], amps[0][c] * ph[None, :], atol=1e-9)


def test_internal_field_follows_a_translation():
    """The reconstructed field of the bar shifted by a = P/4 is the original
    field shifted by +a (|E| on the node grid, exact sample shift)."""
    P, a, nx = 0.8, 0.25, 240
    F = []
    for lo in (0.10, 0.35):
        st = _bar_stack(lo, lo + 0.3, 12.0 + 3.0j, "cell", P=P)
        r = st.set_source(WL, theta=0.3).solve(retain_internal=True)
        F.append(r.internal_field(0.15, nx=nx, incident=(0.0, 1.0), component="E"))
    E0 = np.abs(np.asarray(F[0]["Ey"])[0])
    E1 = np.abs(np.asarray(F[1]["Ey"])[0])
    sh = int(round(a * nx))
    np.testing.assert_allclose(E1, np.roll(E0, sh), rtol=0, atol=1e-9 * E0.max())
    assert np.max(np.abs(E1 - np.roll(E0, -sh))) > 0.05 * E0.max()


def test_internal_field_tm_jumps_at_declared_walls():
    """TM E_x (wall-normal) drops entering eps = 12 at x = 0.1P and recovers at
    0.4P -- the DECLARED walls (the mirror put them at 0.6P / 0.9P)."""
    P, nx = 0.8, 400
    st = _bar_stack(0.10, 0.40, 12.0 + 0.0j, "cell", P=P, M=20, S=400)
    r = st.set_source(WL, theta=0.3).solve(retain_internal=True)
    F = r.internal_field(0.15, nx=nx, incident=(1.0, 0.0), component="E")
    x = np.asarray(F["x"]) % P
    o = np.argsort(x)
    a = np.abs(np.asarray(F["Ex"])[0])[o]
    k = np.ones(9) / 9
    a_s = np.convolve(np.r_[a[-4:], a, a[:4]], k, mode="valid")
    g = np.roll(a_s, -1) - np.roll(a_s, 1)
    xs = x[o] / P
    assert abs(xs[np.argmin(g)] - 0.10) < 0.02
    assert abs(xs[np.argmax(g)] - 0.40) < 0.02


# ---------------------------------------------------------------------------
# 4. blaze physics (label-free in the field)
# ---------------------------------------------------------------------------

def test_blaze_sends_power_toward_the_thick_side():
    """Index 2.25 | 3 | 4 on thirds of the period (optically thick side +x),
    normal incidence, TE: T(kx > 0) > T(kx < 0) for cell, shapes and
    segments, and the field's phase gradient below the grating is > 0."""
    P, d, gap, M = 5.7, 1.0, 3.0, 14
    bl = [(0.0, 1 / 3, 2.25 + 0j), (1 / 3, 2 / 3, 3.0 + 0j), (2 / 3, 1.0, 4.0 + 0j)]
    S = 513
    u = np.arange(S) / S
    prof = np.select([(u >= lo) & (u < hi) for lo, hi, _ in bl], [e for _, _, e in bl])
    for mode in ("cell", "shapes"):
        st = RCWAStack(P, n_superstrate=1.0, n_substrate=1.0, n_orders=M)
        if mode == "cell":
            st.add_layer(d, eps_cell=prof[:, None].astype(complex))
        else:
            st.add_layer(d, shapes=[dict(shape="rectangle", eps=e, size=((hi - lo) * P, st.period_y),
                                         center=(0.5 * (lo + hi) * P, 0.5 * st.period_y))
                                    for lo, hi, e in bl], eps_background=1.0)
        st.add_layer(gap, eps=1.0)
        r = st.set_source(WL, theta=0.0).solve(retain_internal=True)
        o, R, T = r.efficiencies()
        kx = np.asarray(o) * WL / P
        Te = np.asarray(T)[1]
        assert Te[kx > 1e-9].sum() > 2.0 * Te[kx < -1e-9].sum(), mode
        F = r.internal_field(d + 0.9 * gap, nx=2 * S, incident=(0.0, 1.0), component="E")
        E = np.asarray(F["Ey"])[0]
        dx = P / E.shape[0]
        dE = (np.roll(E, -1) - np.roll(E, 1)) / (2 * dx)
        assert np.imag(np.sum(np.conj(E) * dE)) > 0, mode
    o, R, T, _J = rcwa_jones_1d_segments(P, [(hi - lo, e) for lo, hi, e in bl], 1.0, 1.0, d, WL,
                                         n_orders=M)
    kx = np.asarray(o) * WL / P
    Te = np.asarray(T)[1]
    assert Te[kx > 1e-9].sum() > 2.0 * Te[kx < -1e-9].sum()


# ---------------------------------------------------------------------------
# 5. rotation / mirror equivariance (CONSISTENCY only -- passes on the bug too)
# ---------------------------------------------------------------------------

def _nm(c, axes):
    for ax in axes:
        c = np.roll(np.flip(c, axis=ax), 1, axis=ax)
    return c


def test_rotation_by_pi_equivariance_tensor_2d():
    R3 = np.diag([-1.0, -1.0, 1.0])
    px, py, d, M, S = 1.3, 1.1, 0.3, 2, 20
    U, V = np.meshgrid(np.arange(S) / S, np.arange(S) / S, indexing="ij")
    n = np.array([np.cos(0.6) * np.cos(0.4), np.cos(0.6) * np.sin(0.4), np.sin(0.6)])
    T_lc = 2.25 * np.eye(3) + 0.8125 * np.outer(n, n) + 0.08j * np.eye(3)
    cell = np.zeros((S, S, 3, 3), complex) + np.eye(3)
    cell[(U >= 0.1) & (U < 0.45) & (V >= 0.05) & (V < 0.3)] = (6 + 2j) * np.eye(3)
    cell[(U >= 0.5) & (U < 0.9) & (V >= 0.45) & (V < 0.8)] = T_lc
    rot = np.einsum("ab,...bc,dc->...ad", R3, _nm(cell, (0, 1)), R3)
    out = []
    for c, ph in ((cell, 0.7), (rot, 0.7 + np.pi)):
        o, R, T, _J = rcwa_jones_2d(px, py, c, 1.0, 1.0, d, WL, theta=0.45, phi=ph,
                                    n_orders_x=M, n_orders_y=M)
        out.append((np.asarray(o), np.asarray(R), np.asarray(T)))
    (oa, Ra, Ta), (ob, Rb, Tb) = out
    np.testing.assert_allclose(Ra.sum(1), Rb.sum(1), atol=1e-10)
    np.testing.assert_allclose(Ta.sum(1), Tb.sum(1), atol=1e-10)
    key = {tuple(v): i for i, v in enumerate(ob)}
    j = [key[tuple(-v)] for v in oa]
    np.testing.assert_allclose(Ta, Tb[:, j], atol=1e-10)


# ---------------------------------------------------------------------------
# 6. the standalone 2-D entry agrees with the stack on an asymmetric cell
# ---------------------------------------------------------------------------

def test_efficiency_2d_matches_stack_on_asymmetric_cell():
    px, py, d, M, S = 1.3, 1.3, 0.3, 3, 30
    U, V = np.meshgrid(np.arange(S) / S, np.arange(S) / S, indexing="ij")
    cell = np.ones((S, S), complex)
    cell[(U >= 0.1) & (U < 0.45) & (V >= 0.05) & (V < 0.3)] = 6 + 2j
    cell[(U >= 0.6) & (U < 0.75) & (V >= 0.1) & (V < 0.45)] = 4.0
    st = RCWAStack(px, period_y=py, n_superstrate=1.0, n_substrate=1.0, n_orders=M,
                   n_orders_y=M)
    st.add_layer(d, eps_cell=cell)
    o, R, T = st.set_source(WL, theta=0.45, phi=0.0).solve().efficiencies()
    o2, R2, T2 = rcwa_efficiency_2d(px, py, cell, 1.0, 1.0, d, WL, theta=0.45, phi=0.0,
                                    polarization="tm", n_orders_x=M, n_orders_y=M)
    np.testing.assert_allclose(np.asarray(R2), np.asarray(R)[0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(T2), np.asarray(T)[0], atol=1e-12)


# ---------------------------------------------------------------------------
# 6b. the bridge sites a full-solve comparison cannot see on its own
#     (each test below FAILS when its one site is reverted to conj-only)
# ---------------------------------------------------------------------------

def _asym_cell(S=30):
    U, V = np.meshgrid(np.arange(S) / S, np.arange(S) / S, indexing="ij")
    cell = np.ones((S, S), complex)
    cell[(U >= 0.1) & (U < 0.45) & (V >= 0.05) & (V < 0.3)] = 6 + 2j
    cell[(U >= 0.6) & (U < 0.75) & (V >= 0.1) & (V < 0.45)] = 4.0
    return cell


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_prepared_and_vs_wavelength_match_efficiency_2d(pol):
    """``prepare_rcwa_2d`` / ``PreparedRCWA2D.solve`` and
    ``rcwa_efficiency_2d_vs_wavelength`` carry their OWN bridge (the prepared
    cache) -- per order they must equal ``rcwa_efficiency_2d`` on an
    asymmetric cell at oblique conical incidence."""
    px, py, d, M = 1.3, 1.2, 0.3, 3
    cell = _asym_cell()
    kw = dict(theta=0.45, phi=0.7, polarization=pol, n_orders_x=M, n_orders_y=M)
    wls = (1.0, 1.07)
    prep = prepare_rcwa_2d(px, py, cell, 1.5, 1.0, d, **kw)
    o_s, R_s, T_s = rcwa_efficiency_2d_vs_wavelength(px, py, cell, 1.5, 1.0, d, wls, **kw)
    for k, wl in enumerate(wls):
        o, R, T = rcwa_efficiency_2d(px, py, cell, 1.5, 1.0, d, wl, **kw)
        op, Rp, Tp = prep.solve(wl)
        assert np.array_equal(np.asarray(o), np.asarray(op))
        assert np.array_equal(np.asarray(o), np.asarray(o_s))
        for a_, b_ in ((R, Rp), (T, Tp), (R, np.asarray(R_s)[k]), (T, np.asarray(T_s)[k])):
            np.testing.assert_allclose(np.asarray(b_), np.asarray(a_), rtol=0, atol=1e-10)


def test_even_parity_fold_keeps_the_order_phases_of_an_off_origin_cell(monkeypatch):
    """At normal incidence a cell centro-symmetric about an OFF-origin point
    takes the even-parity fold (``_layer_even_spec``).  Its efficiencies are
    blind to a mirror (the mirror is a translation), but the per-order COMPLEX
    amplitudes are not: they must equal the full solve's."""
    calls = []
    real = _stack_mod._symmetric_cascade_rt

    def spy(*a, **k):
        out = real(*a, **k)
        calls.append(out is not None)
        return out
    monkeypatch.setattr(_stack_mod, "_symmetric_cascade_rt", spy)
    px, py, S = 1.3, 1.2, 40
    U, V = np.meshgrid(np.arange(S) / S, np.arange(S) / S, indexing="ij")
    cell = np.ones((S, S), complex)
    # a rectangle centred off-origin (half-pixel centre (11.5, 7.5)/S) and a
    # second one point-symmetric to nothing but the common centre
    cell[(U >= 0.2) & (U < 0.4) & (V >= 0.1) & (V < 0.3)] = 5.0 + 1.0j
    out = {}
    for sym in ("auto", False):
        for f in ("laurent", "li"):
            st = RCWAStack(px, period_y=py, n_superstrate=1.0, n_substrate=1.5, n_orders=3,
                           n_orders_y=3)
            st.add_layer(0.3, eps_cell=cell, formulation=f)
            st.add_layer(0.2, eps=2.1)
            r = st.set_source(WL, theta=0.0).solve(symmetry=sym)
            out[sym, f] = r
    assert calls and all(calls), calls             # the fold actually ran
    for f in ("laurent", "li"):
        for port in ("reflection", "transmission"):
            a = out["auto", f].per_order_amplitudes(port=port)
            b = out[False, f].per_order_amplitudes(port=port)
            for c in ("Ex", "Ey"):
                np.testing.assert_allclose(a[c], b[c], rtol=0, atol=1e-9)


def test_offplane_internal_field_ez_follows_a_translation():
    """The OOP ``internal_field`` recovers E_z through the retained EZX / EZY
    convolutions (a separate bridge site from the modes): on a tilted-tensor
    bar, shifting the structure by a = P/4 must shift |E_z| (and |E_x|) by +a,
    and ``layer_absorption`` must be translation-invariant."""
    P, a, nx, S = 0.9, 0.25, 240, 240
    n = np.array([np.cos(0.7), 0.0, np.sin(0.7)])
    T_lc = 2.25 * np.eye(3) + 0.9 * np.outer(n, n) + 0.1j * np.eye(3)
    u = np.arange(S) / S
    F, A = [], []
    for lo in (0.10, 0.35):
        cell = np.zeros((S, 1, 3, 3), complex) + np.eye(3)
        cell[(u >= lo) & (u < lo + 0.3), 0] = T_lc
        cell[(u >= lo + 0.3) & (u < lo + 0.4), 0] = (3.0 + 0.4j) * np.eye(3)
        st = RCWAStack(P, n_superstrate=1.0, n_substrate=1.5, n_orders=10)
        st.add_layer(0.3, eps_tensor_cell=cell)
        st.add_layer(0.1, eps=2.0 + 0.15j)      # lossy spacer: a real split
        r = st.set_source(WL, theta=0.3).solve(retain_internal=True)
        F.append(r.internal_field(0.15, nx=nx, incident=(1.0, 0.0), component="E"))
        A.append(r.layer_absorption(nx=nx, nz_per_layer=6))
    sh = int(round(a * nx))
    for c in ("Ez", "Ex"):
        e0 = np.abs(np.asarray(F[0][c])[0])
        e1 = np.abs(np.asarray(F[1][c])[0])
        np.testing.assert_allclose(e1, np.roll(e0, sh), rtol=0, atol=1e-8 * e0.max())
    np.testing.assert_allclose(A[1], A[0], rtol=0, atol=1e-9)


def test_li_companion_pair_is_bridged_with_the_cell():
    """``eps_cell_normal=(cell, cell)`` (the raster='harmonic' companion
    route through ``_li_blocks``) reduces to the plain 'li' layer (the
    library's documented reduction: Cxx bit-identical, Cyy ~4e-16 on a
    y-uniform cell) -- only if the companions are bridged like the cell."""
    P, S, M = 1.7, 240, 10
    u = np.arange(S) / S
    cell = np.where(u < 0.3, 6.0 + 2.0j, np.where(u < 0.55, 3.0 + 0.3j, 1.0 + 0j))[:, None]
    out = []
    for pair in (None, (cell, cell)):
        st = RCWAStack(P, n_superstrate=1.0, n_substrate=1.5, n_orders=M)
        st.add_layer(0.35, eps_cell=cell, formulation="li", eps_cell_normal=pair)
        o, R, T = st.set_source(WL, theta=0.4).solve().efficiencies()
        out.append((np.asarray(R), np.asarray(T)))
    np.testing.assert_allclose(out[1][0], out[0][0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(out[1][1], out[0][1], rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# 7. JAX: the flip is a differentiable gather
# ---------------------------------------------------------------------------

def test_jax_gradient_wrt_feature_centre_matches_fd():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    P, S, M, th = 1.7, 128, 6, 0.4
    u = np.arange(S) / S

    def bar(xp, c, w, s=0.03):
        return sum(0.5 * (xp.tanh((u - c - k + w / 2) / s) - xp.tanh((u - c - k - w / 2) / s))
                   for k in (-1, 0, 1))

    def f(xp, c):
        cell = (1.0 + (5.0 + 1.5j) * bar(xp, c, 0.3) + (2.0 + 0.3j) * bar(xp, 0.7, 0.15))
        st = RCWAStack(P, n_superstrate=1.0, n_substrate=1.5, n_orders=M)
        st.add_layer(0.33, eps_cell=cell.astype(complex)[:, None])
        o, R, T = st.set_source(WL, theta=th).solve().efficiencies()
        kx = np.sin(th) + np.asarray(o) * WL / P
        return xp.sum(xp.where(xp.asarray(kx) > 0, T[1], -T[1]))

    c0, h = 0.31, 1e-5
    g = float(jax.grad(lambda c: jnp.real(f(jnp, c)))(jnp.asarray(c0)))
    fd = (float(np.real(f(np, c0 + h))) - float(np.real(f(np, c0 - h)))) / (2 * h)
    assert abs(g - fd) < 1e-6 * max(1.0, abs(fd))
    assert abs(float(jnp.real(f(jnp, jnp.asarray(c0)))) - float(np.real(f(np, c0)))) < 1e-12
