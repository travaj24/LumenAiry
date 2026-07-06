"""Backlog batch 1 (docs/BACKLOG_2026_06_10.md items A1/A2/B1/B2/D1).

* **A2** -- PMM 1-D homogeneous-eig share: one eps-free geometric eig serves
  every uniform half-space (``q^2(eps) = q^2(1) + (eps - 1)``, shared
  eigenvectors; ~x5.5-6 on the half-space mode pair).
* **A1** -- RCWA even-parity scope: the generalized (P, Q) even-sector
  cascade extends the x4 symmetry fast path to the sequential-rule
  ``'li'``, ``rcwa_jones_2d`` (in-plane, normal incidence), and
  ``RCWAStack.solve(symmetry=True)`` (measured x4.5 on a 4-layer stack).
* **B1/B2** -- PMM2DStack ``retain_internal`` + ``layer_absorption``;
  ``PMMStack.internal_field(z)``.
* **D1** -- docs/TOLERANCE_POLICY.md + the ``_tolerances`` constants module
  (used throughout this file).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm import PMM2DStack, PMMStack
from lumenairy.elements.pmm._core import (
    _build_sem_tensor_segments,
    _sem_modes_tensor,
    _sem_modes_uniform,
    _tensor3_dict,
    _uniform_geo_eig,
)
from lumenairy.elements.rcwa import (
    RCWAStack,
    prepare_rcwa_2d,
    rcwa_efficiency_2d,
    rcwa_jones_2d,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor

from ._tolerances import GAUGE_CROSS_PATH, LOSSLESS_CLOSURE

_P, _WL = 0.5e-6, 0.6e-6


# =========================================================================== #
# A2 -- shared homogeneous eig
# =========================================================================== #

@pytest.mark.parametrize("kx0_frac", [0.0, 0.4])
def test_uniform_modes_spectrum_matches_full_eig(kx0_frac):
    """The eps-shifted geometric spectrum equals the full 2n eig's spectrum
    as a SET (the construction is exact, not approximate)."""
    t = _tensor3_dict((2.25 + 0.1j) * np.eye(3))
    mats = _build_sem_tensor_segments(0.8e-6, [0.3, 0.2, 0.5], [t] * 3,
                                      14, 1, True)
    k0 = 2 * np.pi / 0.633e-6
    kx0 = kx0_frac * k0
    W2, V2, lam, q = _sem_modes_uniform(mats, k0, kx0, 2.25 + 0.1j)
    Wr, Vr, lamr, qr = _sem_modes_tensor(mats, k0, kx0, True)
    a = np.sort_complex(q ** 2)
    b = np.sort_complex(qr ** 2)
    assert np.max(np.abs(a - b)) < 1e-9


def test_uniform_modes_shared_geo_eig_between_media():
    """One geometric eig serves two media: explicit eigen-relation check
    Mbig(eps) W = W diag(q^2) for both eps values from the SAME geo."""
    t = _tensor3_dict(1.0 * np.eye(3))
    mats = _build_sem_tensor_segments(0.8e-6, [0.5, 0.5], [t] * 2, 12, 1,
                                      True)
    k0 = 2 * np.pi / 0.633e-6
    geo = _uniform_geo_eig(mats, k0, 0.0)
    mu, w, Kx2 = geo
    for eps in (1.0, 2.25, 2.25 + 0.5j):
        q2 = eps - mu
        # block eigen-relation on the n-space: (eps I - Kx2) w = w diag(q2)
        lhs = eps * w - Kx2 @ w
        rhs = w * q2[None, :]
        assert np.max(np.abs(lhs - rhs)) < 1e-8 * max(1.0, abs(eps))


def test_pmm_solvers_unchanged_through_shared_eig():
    """Solver-level oracle through the new half-space gauge: every medium of
    a uniform thin film routes through the shared geometric eig, and the
    specular reflectance must equal the analytic Fresnel film value."""
    import lumenairy as la
    n1, n2, n3 = 1.0, 1.5, 1.0  # superstrate / film / substrate
    t, wl = 0.3e-6, _WL
    # analytic Airy film at normal incidence
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    beta = 2 * np.pi * n2 * t / wl
    r = (r12 + r23 * np.exp(2j * beta)) / (1 + r12 * r23 * np.exp(2j * beta))
    R_exact = abs(r) ** 2
    st = PMMStack(0.8e-6, n_substrate=n3, n_superstrate=n1, degree=12)
    st.add_layer(t, eps=n2 ** 2)
    o2, R2, T2, J2 = st.set_source(wl).solve()
    i0 = int(np.argmin(np.abs(np.asarray(o2))))
    assert abs(float(R2[0, i0]) - R_exact) < GAUGE_CROSS_PATH
    assert abs(float(R2.sum() + T2.sum()) / 2 - 1.0) < LOSSLESS_CLOSURE
    # the segments entry point shares the same half-space gauge
    o, R, T, J = la.pmm_jones_1d_segments(
        0.8e-6, [(0.5, n2 ** 2), (0.5, n2 ** 2)], n3, n1, t, wl,
        degree=12, stabilize=False)
    i0 = int(np.argmin(np.abs(np.asarray(o))))
    assert abs(float(R[0, i0]) - R_exact) < GAUGE_CROSS_PATH


# =========================================================================== #
# A1 -- even-parity scope
# =========================================================================== #

def _pillar(S, hw=0.22, hi=-3.0 + 4.0j, lo=2.1):
    x = (np.arange(S) + 0.5) / S - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    c = np.full((S, S), lo, dtype=complex)
    c[(np.abs(X) < hw) & (np.abs(Y) < hw)] = hi
    return c


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_li_even_sector_matches_full(pol):
    """The sequential-rule 'li' regains the even-parity fast path (it was
    gated out when 'li' moved to the tensor eigensolver in v5.14.1)."""
    cell = _pillar(64)
    a = rcwa_efficiency_2d(_P, _P, cell, 1.5, 1.0, 0.2e-6, _WL,
                           polarization=pol, n_orders_x=6, n_orders_y=6,
                           formulation="li", symmetry=False)
    b = rcwa_efficiency_2d(_P, _P, cell, 1.5, 1.0, 0.2e-6, _WL,
                           polarization=pol, n_orders_x=6, n_orders_y=6,
                           formulation="li", symmetry=True)
    assert np.max(np.abs(a[1] - b[1])) < 1e-11
    assert np.max(np.abs(a[2] - b[2])) < 1e-11
    assert not np.array_equal(a[1], b[1])      # genuinely the fast path


def test_li_even_prepared_matches_direct():
    cell = _pillar(64)
    b = rcwa_efficiency_2d(_P, _P, cell, 1.5, 1.0, 0.2e-6, _WL,
                           polarization="tm", n_orders_x=6, n_orders_y=6,
                           formulation="li", symmetry=True)
    prep = prepare_rcwa_2d(_P, _P, cell, 1.5, 1.0, 0.2e-6,
                           polarization="tm", n_orders_x=6, n_orders_y=6,
                           formulation="li", symmetry=True)
    o, R, T = prep.solve(_WL)
    assert np.max(np.abs(np.asarray(R) - b[1])) < 1e-12


def test_jones_2d_even_sector_matches_full():
    S = 48
    tc = np.zeros((S, S, 3, 3), complex)
    for i in range(3):
        tc[:, :, i, i] = 2.25
    no2, ne2 = 1.5 ** 2, 1.7 ** 2
    c0, s0 = np.cos(0.7), np.sin(0.7)
    x = (np.arange(S) + 0.5) / S - 0.5
    m = (np.abs(x[:, None]) < 0.25) & (np.abs(x[None, :]) < 0.25)
    tc[m, 0, 0] = ne2 * c0 * c0 + no2 * s0 * s0
    tc[m, 1, 1] = ne2 * s0 * s0 + no2 * c0 * c0
    tc[m, 0, 1] = tc[m, 1, 0] = (ne2 - no2) * c0 * s0
    tc[m, 2, 2] = no2
    # explicit symmetry=False forces the full 2N solve (the even-parity fold is
    # ON by default since the v5.20.5 "auto" default -- so the comparison must
    # request the full path explicitly to stay a full-vs-even test).
    a = rcwa_jones_2d(_P, _P, tc, 1.5, 1.0, 0.2e-6, _WL, n_orders_x=5,
                      n_orders_y=5, symmetry=False)
    b = rcwa_jones_2d(_P, _P, tc, 1.5, 1.0, 0.2e-6, _WL, n_orders_x=5,
                      n_orders_y=5, symmetry=True)
    assert np.max(np.abs(a[1] - b[1])) < 1e-8
    assert np.max(np.abs(a[3] - b[3])) < 1e-8
    assert not np.array_equal(a[1], b[1])


def _mixed_stack(shapes_center):
    st = RCWAStack(period=_P, period_y=_P, n_superstrate=1.0,
                   n_substrate=1.5, n_orders=6, n_orders_y=6)
    st.add_layer(0.05e-6, eps=2.25)
    st.add_layer(0.12e-6, eps_cell=_pillar(64))
    st.add_layer(0.10e-6, eps_cell=_pillar(64, hw=0.15), formulation="li")
    st.add_layer(0.06e-6, eps_background=1.0, shapes=[
        dict(shape="rectangle", eps=4.0, size=(0.2e-6, 0.2e-6),
             center=(shapes_center, shapes_center))])
    return st


def test_stack_even_sector_matches_full():
    """The even-parity cascade runs the WHOLE mixed stack (uniform +
    laurent-iso + sequential-li + analytic shapes) in the even sector when
    every layer shares one symmetry centre.  NB the pixel-grid feature
    centre sits at the half-pixel (31.5/64)*P, NOT P/2 -- a shapes layer at
    exactly P/2 is a GENUINE half-pixel centre mismatch and must fall back
    (tested below)."""
    cc = (31.5 / 64) * _P
    rf = _mixed_stack(cc).set_source(_WL).solve(symmetry=False)   # explicit full
    rs = _mixed_stack(cc).set_source(_WL).solve(symmetry=True)
    of, Rf, Tf = rf.efficiencies()
    os_, Rs, Ts = rs.efficiencies()
    assert np.max(np.abs(np.asarray(Rf) - np.asarray(Rs))) < 1e-11
    assert np.max(np.abs(np.asarray(Tf) - np.asarray(Ts))) < 1e-11
    assert not np.array_equal(np.asarray(Rf), np.asarray(Rs))


def test_stack_even_sector_fallbacks_bit_identical():
    cc = (31.5 / 64) * _P
    # half-pixel centre mismatch -> joint symmetry fails -> full solve
    rf = _mixed_stack(_P / 2).set_source(_WL).solve()
    rs = _mixed_stack(_P / 2).set_source(_WL).solve(symmetry=True)
    assert np.array_equal(np.asarray(rf.efficiencies()[1]),
                          np.asarray(rs.efficiencies()[1]))
    # oblique incidence -> full solve
    rf = _mixed_stack(cc).set_source(_WL, theta=0.2).solve()
    rs = _mixed_stack(cc).set_source(_WL, theta=0.2).solve(symmetry=True)
    assert np.array_equal(np.asarray(rf.efficiencies()[1]),
                          np.asarray(rs.efficiencies()[1]))
    # an out-of-plane layer -> full solve
    oop = uniaxial_tensor(1.5, 1.7, np.deg2rad(30.0))
    sto = RCWAStack(period=_P, period_y=_P, n_superstrate=1.0,
                    n_substrate=1.5, n_orders=3, n_orders_y=3)
    sto.add_layer(0.1e-6,
                  eps_tensor_cell=np.broadcast_to(oop, (16, 16, 3, 3)).copy())
    rf = sto.set_source(_WL).solve()
    rs = sto.set_source(_WL).solve(symmetry=True)
    assert np.array_equal(np.asarray(rf.efficiencies()[1]),
                          np.asarray(rs.efficiencies()[1]))


# =========================================================================== #
# B1/B2 -- PMM internal field + 2-D stack absorption
# =========================================================================== #

def test_pmm_internal_field_continuity_and_grid():
    st = PMMStack(0.8e-6, n_substrate=1.5, n_superstrate=1.0, degree=12)
    st.add_layer(0.15e-6, segments=[(0.5, 4.0 + 2.0j), (0.5, 1.0)])
    st.add_layer(0.10e-6, eps=2.25)
    st.set_source(0.633e-6).solve(retain_internal=True)
    f1 = st.internal_field(0.15e-6 - 1e-12)
    f2 = st.internal_field(0.15e-6 + 1e-12, pol=0)
    scale = max(np.max(np.abs(f1["Ex"])), np.max(np.abs(f1["Ey"])))
    dE = max(np.max(np.abs(f1["Ex"] - f2["Ex"])),
             np.max(np.abs(f1["Ey"] - f2["Ey"]))) / scale
    assert dE < 1e-3                 # tangential-E continuity (truncation)
    assert bool(np.all(np.diff(f1["x"]) >= 0))
    assert f1["layer"] == 0 and f2["layer"] == 1
    f3 = st.internal_field(0.05e-6, pol=1)
    assert f3["Ex"].shape == f1["Ex"].shape
    with pytest.raises(ValueError, match="pol"):
        st.internal_field(0.05e-6, pol=2)


def test_pmm2d_stack_layer_absorption_closes():
    st = PMM2DStack(0.6e-6, n_substrate=1.5, n_superstrate=1.0, degree=9,
                    n_orders=4)
    cell = np.full((6, 6), 1.0 + 0j)
    cell[1:4, 1:4] = 6.0 + 3.0j
    st.add_layer(0.15e-6, eps_cell=cell)
    st.add_layer(0.08e-6, eps=2.25)
    cell2 = np.full((6, 6), 2.25 + 0j)
    cell2[2:5, 2:5] = -15.0 + 2.0j
    st.add_layer(0.10e-6, eps_cell=cell2)
    o, R, T, J = st.set_source(0.633e-6).solve(retain_internal=True)
    A = st.layer_absorption()
    budget = 1.0 - R.sum(axis=1) - T.sum(axis=1)
    assert np.max(np.abs(A.sum(axis=0) - budget)) < 1e-10
    assert np.max(np.abs(A[1])) < 1e-12        # lossless spacer
    st2 = PMM2DStack(0.6e-6, degree=9, n_orders=4)
    st2.add_layer(0.1e-6, eps=2.25)
    with pytest.raises(ValueError, match="retain_internal"):
        st2.layer_absorption()
