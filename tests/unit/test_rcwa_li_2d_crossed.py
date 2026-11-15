"""The 2-D (crossed-grating) inverse rule, ``formulation='li'``, of
``RCWAStack``: what it builds, against what it is measured, and what
``formulation='auto'`` resolves to on a 2-D stack because of those
measurements.

What the operator is (Li 1997, J. Opt. Soc. Am. A 14, 2758, Eqs. 8, 9, 27):
``E_x`` takes ``[[ [[1/eps]]_x^-1 ]]_y`` -- per y, the x-Toeplitz of ``1/eps``
is INVERTED (inverse rule across the x-walls, where ``D_x`` is continuous and
``E_x`` jumps), then the resulting matrix function of y is Fourier
transformed along y (direct rule: ``E_x`` is continuous across the y-walls).
``E_y`` takes the transposed construction, and ``E_z`` is eliminated with
the plain 2-D ``[[eps]]^-1`` (``E_z`` is tangential to every vertical wall).
This is the rule Li derived for boundaries made of segments parallel to the
axes; Schuster et al. 2007 (J. Opt. Soc. Am. A 24, 2880, Sec. 4-5) state it
is the correct factorization for that class ("perfectly appropriate" on their
axis-aligned checkerboard cell) and that it is NOT for oblique or curved
walls, which it sees as zigzags.  ``test_li_operators_are_lis_eqs_8_and_9``
pins the library's operators to an independent transcription of Eqs. 8 / 9.

Measured (docs/audits/AUDIT_RCWA_2D_LI_FACTORIZATION_2026_11_14.md): the
operator is not defective, but on 2-D cells with sharp METAL corners neither
Fourier rule converges reliably at affordable truncations, and the inverse
rule is not reliably the better of the two there.  The references are the
library's own pure staggered 2-D polynomial modal solver (``PMM2DStackPure``,
no Fourier factorization at all) ladders, plus the Laurent ``1/N`` tail
extrapolation where that tail is clean.
"""
import numpy as np
import pytest

import lumenairy as lum
from lumenairy.elements.rcwa.twod import (
    _eps_convolution_2d,
    _harmonic_orders_2d,
    _li_convolutions_2d,
)


def _li_eq8_eq9(cell, Mx, My):
    """Li 1997 Eqs. (8) and (9) transcribed with explicit sums over the node
    lattice ``x_j = j Px / Sx`` (the library's pixel meaning), in the
    library's flat order layout (m slow, n fast).  Independent of the
    library's FFT / batched-inverse implementation."""
    Sx, Sy = cell.shape
    inv = 1.0 / cell
    jx = np.arange(Sx)
    jy = np.arange(Sy)
    am = np.arange(-Mx, Mx + 1)
    an = np.arange(-My, My + 1)
    # Eq. (8): for each y_k the x-Toeplitz of 1/eps, inverted ...
    A = np.empty((Sy, am.size, am.size), dtype=complex)
    for k in range(Sy):
        T = np.empty((am.size, am.size), dtype=complex)
        for a, m in enumerate(am):
            for b, mp in enumerate(am):
                T[a, b] = np.sum(inv[:, k] * np.exp(-2j * np.pi * (m - mp)
                                                    * jx / Sx)) / Sx
        A[k] = np.linalg.inv(T)
    # Eq. (9): for each x_j the y-Toeplitz of 1/eps, inverted
    B = np.empty((Sx, an.size, an.size), dtype=complex)
    for j in range(Sx):
        U = np.empty((an.size, an.size), dtype=complex)
        for a, n in enumerate(an):
            for b, nq in enumerate(an):
                U[a, b] = np.sum(inv[j, :] * np.exp(-2j * np.pi * (n - nq)
                                                    * jy / Sy)) / Sy
        B[j] = np.linalg.inv(U)
    orders, N = _harmonic_orders_2d(Mx, My)
    Cxx = np.empty((N, N), dtype=complex)
    Cyy = np.empty((N, N), dtype=complex)
    for p, (m, n) in enumerate(orders):
        for q, (mp, nq) in enumerate(orders):
            # ... then the direct rule (a plain Fourier coefficient) along
            # the other axis
            Cxx[p, q] = np.sum(A[:, m + Mx, mp + Mx]
                               * np.exp(-2j * np.pi * (n - nq) * jy / Sy)) / Sy
            Cyy[p, q] = np.sum(B[:, n + My, nq + My]
                               * np.exp(-2j * np.pi * (m - mp) * jx / Sx)) / Sx
    return Cxx, Cyy


def test_li_operators_are_lis_eqs_8_and_9():
    """The 2-D 'li' blocks ARE Li's Eqs. (8) / (9), on a cell with two
    disjoint metal / dielectric rectangles of different sizes (so x and y
    are not interchangeable and a transposed or swapped construction
    differs), Mx != My.  The swapped assignment (Cxx <-> Cyy), the plain
    Laurent [[eps]] and the global 2-D inverse [[1/eps]]^-1 (the 1-D rule
    applied to the whole crossed cell) are all far from it."""
    S = 16
    cell = np.ones((S, S), dtype=complex)
    cell[2:9, 3:7] = -20.0 + 2.0j
    cell[11:14, 9:15] = 6.0 + 0.5j
    Mx, My = 3, 2
    orders, _N = _harmonic_orders_2d(Mx, My)
    Cxx, Cyy = _li_convolutions_2d(cell, orders, Mx, My, np)
    Rxx, Ryy = _li_eq8_eq9(cell, Mx, My)
    scale = np.max(np.abs(Rxx))
    assert np.max(np.abs(Cxx - Rxx)) < 1e-12 * scale
    assert np.max(np.abs(Cyy - Ryy)) < 1e-12 * np.max(np.abs(Ryy))
    # discrimination: the swapped assignment, the direct rule and the global
    # 2-D inverse rule are different operators
    E = _eps_convolution_2d(cell, orders, Mx, My)
    G = np.linalg.inv(_eps_convolution_2d(1.0 / cell, orders, Mx, My))
    for wrong in (Ryy, E, G):
        assert np.max(np.abs(Cxx - wrong)) > 1e-2 * scale


# ---------------------------------------------------------------------------
# Measured cells and their references
# ---------------------------------------------------------------------------
#
# A 300 nm square (or a disk of radius 180 nm) centred in a 600 nm square
# cell, in air, over a uniform spacer, on n = 1.5, at 1310 nm, normal
# incidence, E along x.  "weak": 60 nm of eps -3+1i over 150 nm of eps 2.1;
# "strong": 50 nm of eps -70+5i over 60 nm of eps 2.1 over a 100 nm mirror of
# eps -70+5i; "diel": 200 nm of eps 12 over 150 nm of eps 2.1.  The
# checkerboard is Li 1997's Example 1 in his unit cell A: period 2.5 lambda,
# two eps-2.25 squares of side 1.25 lambda touching at their corners, lambda
# thick, incident from n = 1.5 onto air.
#
# REFERENCES: the pure staggered 2-D polynomial modal solver
# (PMM2DStackPure, Granet's modified-Legendre basis on the walls of the cell;
# no Fourier factorization), highest affordable n_modes M, with the
# uncertainty taken as the largest change over the last two rungs (it
# converges algebraically on these sharp corners).  Measured 2026-11-14,
# Windows/MKL, BLAS 2 threads:
#
#   cell     M    J_xx (reflection)          uncertainty
#   weak     12   -0.1824439 - 0.2025169i    4.4e-4
#   strong   12   -0.8418916 + 0.3109818i    5.9e-3
#   diel     12   -0.1225103 + 0.0140120i    1.5e-4
#   diel     9    -0.1301575 + 0.0234269i    3.4e-4   (disk; 3 x 3 map)
#   weak     9    -0.1830686 - 0.2174318i    3.6e-4   (disk)
#   checker  16   +0.1443999 - 0.0622678i    2.8e-5   (T00 0.175621)
#
# and, for the checkerboard, Li 1997's own Table 1, T(0,0) = 17.486 % (his
# unit cell B; his Fig. 6 puts the cell-A / cell-B spread near 1e-3).
_P = 600e-9
_LAM = 1.31e-6
_REF = {
    "weak": (-0.1824439 - 0.2025169j, 4.4e-4),
    "strong": (-0.8418916 + 0.3109818j, 5.9e-3),
    "diel": (-0.1225103 + 0.0140120j, 1.5e-4),
    "diel_disk": (-0.1301575 + 0.0234269j, 3.4e-4),
    "weak_disk": (-0.1830686 - 0.2174318j, 3.6e-4),
    "checker": (0.1443999 - 0.0622678j, 2.8e-5),
}
_LI1997_T00 = 0.17486
_PMM_CHECKER_T00 = 0.175621

_CELLS = {
    "weak": ((60e-9, -3 + 1j), [(150e-9, 2.1)]),
    "strong": ((50e-9, -70 + 5j), [(60e-9, 2.1), (100e-9, -70 + 5j)]),
    "diel": ((200e-9, 12.0), [(150e-9, 2.1)]),
}


def _raster_disk(S, eps):
    u = np.arange(S) * _P / S                  # node lattice (pixel meaning)
    X, Y = np.meshgrid(u, u, indexing="ij")
    c = np.ones((S, S), dtype=complex)
    c[np.hypot(X - _P / 2, Y - _P / 2) < 0.3 * _P] = eps
    return c


def _solve(name, N, formulation, *, disk=None):
    (d, eps), under = _CELLS[name]
    st = lum.RCWAStack(_P, period_y=_P, n_superstrate=1.0, n_substrate=1.5,
                       n_orders=N, n_orders_y=N)
    if disk == "raster":
        st.add_layer(d, eps_cell=_raster_disk(1024, eps),
                     formulation=formulation)
    elif disk == "analytic":
        st.add_layer(d, shapes=[dict(shape="disk", eps=eps, radius=0.3 * _P,
                                     center=(_P / 2, _P / 2))],
                     eps_background=1.0, formulation=formulation)
    else:
        st.add_layer(d, shapes=[dict(shape="rectangle", eps=eps,
                                     size=(300e-9, 300e-9),
                                     center=(_P / 2, _P / 2))],
                     eps_background=1.0, formulation=formulation)
    for t, e in under:
        st.add_layer(t, eps=e)
    return st, st.set_source(_LAM).solve()


def _checker(N, formulation):
    a = 1.25 * _LAM
    st = lum.RCWAStack(2 * a, period_y=2 * a, n_superstrate=1.5,
                       n_substrate=1.0, n_orders=N, n_orders_y=N)
    st.add_layer(_LAM, shapes=[
        dict(shape="rectangle", eps=2.25, size=(a, a), center=(a / 2, a / 2)),
        dict(shape="rectangle", eps=2.25, size=(a, a),
             center=(1.5 * a, 1.5 * a))],
        eps_background=1.0, formulation=formulation)
    res = st.set_source(_LAM).solve()
    o, _R, T = res.efficiencies()
    o = np.asarray(o)
    i0 = int(np.flatnonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0])
    return st, res, float(np.asarray(T)[0][i0])


def _err(res, ref):
    return abs(complex(np.asarray(res.jones_reflection())[0, 0]) - ref)


@pytest.mark.parametrize("name, N, li_bar, laurent_floor", [
    # measured |J - J_ref|: 'li' 5.32e-4 / 'laurent' 3.34e-2
    ("diel", 8, 1.5e-3, 2.0e-2),
    # measured 3.72e-2 / 2.69e-1 (both erratic in N on this cell; 'li' is
    # 3x-50x closer on every rung from 4 to 24)
    ("strong", 8, 7.0e-2, 1.5e-1),
    # measured 2.07e-3 / 6.98e-3 -- the weak metal near the corner-singular
    # eps = -3: 'li' leads below n_orders ~14, trails above (slow test)
    ("weak", 8, 4.0e-3, 5.0e-3),
])
def test_rectangle_cells_against_the_polynomial_modal_reference(
        name, N, li_bar, laurent_floor):
    ref, unc = _REF[name]
    _st, r_li = _solve(name, N, "li")
    _st, r_la = _solve(name, N, "laurent")
    assert unc < li_bar / 2                      # the reference resolves it
    assert _err(r_li, ref) < li_bar
    assert _err(r_la, ref) > laurent_floor


def test_auto_is_explicit_li_on_rectangles_whatever_the_material():
    """'auto' on a 2-D rectangle layer IS the explicit 'li' call (same route,
    bit for bit), for a dielectric and for a metal."""
    for name in ("diel", "strong"):
        st_a, r_a = _solve(name, 3, "auto")
        _st, r_l = _solve(name, 3, "li")
        assert st_a.layers[0].formulation == "li"
        assert st_a.layers[0].formulation_requested == "auto"
        assert np.array_equal(np.asarray(r_a.jones_reflection()),
                              np.asarray(r_l.jones_reflection()))


def test_disk_cells_raster_li_against_analytic_laurent():
    """A disk: 'auto' keeps the ANALYTIC disk on the direct rule ('li' takes
    rectangles only); the same disk rasterized (1024 x 1024 node lattice)
    under 'li' -- Li's staircase -- is measured closer to the reference on a
    dielectric (6.33e-3 against 3.62e-2 at n_orders 8) and level with it on
    the weak metal (2.56e-3 against 3.64e-3)."""
    st, r_an = _solve("diel", 8, "auto", disk="analytic")
    assert st.layers[0].formulation == "laurent"
    st_r, r_ra = _solve("diel", 8, "auto", disk="raster")
    assert st_r.layers[0].formulation == "li"
    ref, _unc = _REF["diel_disk"]
    assert _err(r_ra, ref) < 1.0e-2
    assert _err(r_an, ref) > 2.5e-2
    ref, _unc = _REF["weak_disk"]
    _st, r_wa = _solve("weak", 8, "laurent", disk="analytic")
    _st, r_wr = _solve("weak", 8, "li", disk="raster")
    assert max(_err(r_wa, ref), _err(r_wr, ref)) < 5.0e-3


def test_li_1997_checkerboard_benchmark():
    """Li 1997, Example 1 (unit cell A here): T(0,0) = 17.486 % in his
    Table 1.  Measured at n_orders 10: 'li' 0.174861, 'laurent' 0.183838;
    the polynomial modal reference (M = 16) 0.175621, and J_xx 6.8e-5 from
    'li' against 1.74e-3 from 'laurent'."""
    _st, res, t00 = _checker(10, "li")
    _st, res_la, t00_la = _checker(10, "laurent")
    assert abs(t00 - _LI1997_T00) < 1.5e-3
    assert abs(t00 - _PMM_CHECKER_T00) < 1.5e-3
    assert abs(t00_la - _LI1997_T00) > 5e-3
    ref, _unc = _REF["checker"]
    assert _err(res, ref) < 3e-4
    assert _err(res_la, ref) > 1e-3
    # 'auto' takes 'li' on this all-dielectric rectangle cell
    st_a, _res, _t = _checker(4, "auto")
    assert st_a.layers[0].formulation == "li"


@pytest.mark.slow
def test_weak_metal_crossover_ladder():
    """The regime where Li's rule is NOT the more accurate one: the
    eps -3+1i square, whose corners sit at the border of the irregular
    corner singularity (Li and Granet 2011).  Measured |J - J_ref|:

        n_orders   'li'      'laurent'
        8          2.07e-3   6.98e-3
        12         2.14e-3   3.06e-3
        16         2.03e-3   1.14e-3
        20         1.89e-3   2.59e-4

    'li' leads up to n_orders ~14 and trails beyond; both stay within
    ~2e-3 of the reference (whose own uncertainty is 4.4e-4)."""
    ref, _unc = _REF["weak"]
    _s, a = _solve("weak", 12, "li")
    _s, b = _solve("weak", 12, "laurent")
    assert _err(a, ref) < _err(b, ref)
    _s, a = _solve("weak", 20, "li")
    _s, b = _solve("weak", 20, "laurent")
    assert _err(b, ref) < _err(a, ref) < 2.5e-3


@pytest.mark.slow
def test_polynomial_modal_reference_reproduces():
    """The dielectric-square reference two rungs below the tabulated one
    lands within the tabulated uncertainty (M = 10: |dJ| 1.44e-4 against
    1.5e-4; M = 11 is 7.6e-5 away)."""
    from lumenairy.elements.pmm.shapes2d import Rect
    from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
    sp = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.5,
                        n_modes=10, n_orders=3)
    sp.add_layer(200e-9, shapes=[Rect(_P / 2, _P / 2, 300e-9, 300e-9,
                                      eps=12.0)], background_eps=1.0)
    sp.add_layer(150e-9, eps=2.1)
    sp.set_source(_LAM)
    _o, _R, _T, J = sp.solve()
    ref, unc = _REF["diel"]
    assert abs(complex(J[0, 0]) - ref) <= unc
