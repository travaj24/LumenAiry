"""Oblique-incidence far-field validation of the staggered 2-D PMM (v5.21).

Locks in the two projection-kernel fixes in ``_stag_fourier_projection``:

* the Bloch ``alpha0`` shift (without it the one-period Rayleigh projection is
  non-orthogonal at oblique -- energy loss even in the specular order), and
* the order-index ORIENTATION: the kernel must be the conjugate of the
  order-m plane wave ``e^{-i(alpha0 + mG)x}``, i.e. ``e^{+i(mG + alpha0)x}``.
  The mirrored kernel (``e^{-i(mG - alpha0)x}``) deposits physical order -m
  into slot m: exact at normal incidence (kz even in kx) and for
  reflection-symmetric cells, but at oblique every |m|>0 order gets the wrong
  kz flux factor -- the asymmetric-cell oracle test below measured
  ``stag[m] = oracle[-m] * kz(m)/kz(-m)`` (several-% energy non-conservation,
  sign flipping with theta) before the fix.

Oracles: the EXACT 1-D pure PMM (``PMMStack``, no Fourier floor, no corner
residual) for y-uniform stripe cells -- identical physics to the 2-D solve at
``phi = 0`` -- and energy conservation for genuinely 2-D pillars.
"""
import numpy as np
import pytest

from lumenairy import PMMStack
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import pmm_efficiency_2d_staggered

PX = PY = 0.9e-6
WL = 0.60e-6
DEPTH = 0.30e-6
# ASYMMETRIC y-uniform stripe (2|4|9): the only geometry class that can SEE a
# label mirror (reflection-symmetric cells have mirror-identical magnitudes).
ASYM = np.array([[2.0] * 3, [4.0] * 3, [9.0] * 3], np.complex128)
ASYM_SEGS = [(1 / 3, 2.0), (1 / 3, 4.0), (1 / 3, 9.0)]


def _n0_row(orders, R, T):
    orders = np.asarray(orders)
    sel = orders[:, 1] == 0
    m = orders[sel, 0]
    idx = np.argsort(m)
    return m[idx], R[sel][idx], T[sel][idx]


def _oracle_1d(layers, theta):
    """Exact 1-D pure-PMM per-order (m, R, T), TE row, kx_m = kx0 + m wl/px."""
    stk = PMMStack(PX, n_superstrate=1.0, n_substrate=1.0, degree=14)
    for t, segs in layers:
        stk.add_layer(t, segments=segs)
    stk.set_source(WL, theta=theta)
    o, R, T = stk.solve()[:3]
    o = np.asarray(o).ravel()
    idx = np.argsort(o)
    return o[idx], R[1][idx], T[1][idx]        # row 1 = incident E_y (TE)


@pytest.mark.parametrize("cell", [
    np.array([[2.0, 2.0], [6.0, 6.0]], np.complex128),   # y-uniform stripe
    np.array([[6.0, 2.0], [2.0, 2.0]], np.complex128),   # single pillar
])
@pytest.mark.parametrize("theta", [0.10, 0.20])
def test_staggered_oblique_energy(cell, theta):
    """Lossless patterned cells at oblique conserve energy (leaked 0.82-1.05
    with the mirrored kernel; the converged stripe plateaued at 0.878)."""
    _, R, T = pmm_efficiency_2d_staggered(
        PX, PY, cell, 1.0, 1.0, DEPTH, WL, degree=6, n_orders=8,
        polarization="te", theta=theta, phi=0.0)
    assert abs(R.sum() + T.sum() - 1.0) < 1e-4


# v5.32.1 marker inversion (AUDIT_CI_TEST_TIME_2026_08_03 §4/chunk 6): the
# file's only ``slow`` mark was on ``test_stack_pure_multilayer_ab_vs_1d``
# (correct -- it stays), while THESE two cells were unmarked and measured
# 53.8 s of the file's 56.1 s: each solves a degree-8 staggered 2-D cell AND a
# degree-14 exact 1-D pure-PMM oracle.  The anti-mirror tripwire they carry is
# an orientation/labelling contract, not a Python-version one.
@pytest.mark.slow
@pytest.mark.parametrize("theta", [0.0, 0.20])
def test_staggered_per_order_orientation_vs_1d(theta):
    """Asymmetric stripe: per-order R/T match the EXACT 1-D pure PMM in the
    SAME order slots, and beat the mirrored assignment by >=5x.  theta=0 pins
    the orientation at normal incidence too (the mirror was energy-invisible
    but label-wrong there)."""
    o1, R1, T1 = pmm_efficiency_2d_staggered(
        PX, PY, ASYM, 1.0, 1.0, DEPTH, WL, degree=8, n_orders=6,
        polarization="te", theta=theta, phi=0.0)
    m1, r1, t1 = _n0_row(o1, R1, T1)
    m2, r2, t2 = _oracle_1d([(DEPTH, ASYM_SEGS)], theta)
    # align the order windows (1-D far_field_orders may differ)
    keep = np.isin(m2, m1)
    m2, r2, t2 = m2[keep], r2[keep], t2[keep]
    assert np.array_equal(m1, m2)
    assert abs(R1.sum() + T1.sum() - 1.0) < 1e-4
    # exact-vs-exact: measured 7.0e-4 (theta=0) / 1.7e-4 (theta=0.2) at
    # degree 8; the mirrored assignment sits at 0.29 / 0.50.
    direct = max(np.abs(r1 - r2).max(), np.abs(t1 - t2).max())
    mirrored = max(np.abs(r1 - r2[::-1]).max(), np.abs(t1 - t2[::-1]).max())
    assert direct < 5e-3
    assert mirrored > 5.0 * direct  # anti-mirror tripwire


def test_staggered_vacuum_oblique_exact():
    """Empty cell at oblique: specular-only, R+T = 1 to machine-ish level
    (guards the alpha0 half of the kernel)."""
    vac = np.ones((2, 2), np.complex128)
    _, R, T = pmm_efficiency_2d_staggered(
        PX, PY, vac, 1.0, 1.0, DEPTH, WL, degree=6, n_orders=6,
        polarization="te", theta=0.20, phi=0.0)
    assert abs(R.sum() + T.sum() - 1.0) < 1e-9


@pytest.mark.slow      # two staggered modal eigs + a degree-14 1-D
                       # multilayer oracle -- minutes-scale on CI
def test_stack_pure_multilayer_ab_vs_1d():
    """DIRECT patterned<->patterned (A|B) staggered cascade against the exact
    1-D multilayer oracle at oblique.  This interface 'blew up energy
    (1.2-3.4)' before the projection-kernel mirror fix -- the blow-up was the
    far field, not the cascade; post-fix it is exact and per-order correct,
    so PMM2DStackPure supports multiple patterned layers."""
    A = ASYM
    B = np.array([[9.0] * 3, [2.0] * 3, [4.0] * 3], np.complex128)
    tA, tB = 0.20e-6, 0.15e-6
    stk = PMM2DStackPure(PX, PY, n_modes=8, n_orders=6)
    stk.add_layer(tA, eps_cell=A)
    stk.add_layer(tB, eps_cell=B)
    stk.set_source(WL, theta=0.20, phi=0.0)
    o, R, T = [np.asarray(x) for x in stk.solve(jones=False)[:3]]
    # both incident polarizations conserve
    for p in (0, 1):
        assert abs(R[p].sum() + T[p].sum() - 1.0) < 1e-4
    m1, r1, t1 = _n0_row(o, R[1], T[1])
    B_SEGS = [(1 / 3, 9.0), (1 / 3, 2.0), (1 / 3, 4.0)]
    m2, r2, t2 = _oracle_1d([(tA, ASYM_SEGS), (tB, B_SEGS)], 0.20)
    keep = np.isin(m2, m1)
    m2, r2, t2 = m2[keep], r2[keep], t2[keep]
    assert np.array_equal(m1, m2)
    direct = max(np.abs(r1 - r2).max(), np.abs(t1 - t2).max())
    assert direct < 5e-3
