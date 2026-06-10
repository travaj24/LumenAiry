"""v5.14.0 -- multi-region 2-D hybrid PMM (``pmm_efficiency_2d_cell``).

The hybrid PMM's geometry input generalizes from a single rectangular pillar to
an ARBITRARY axis-aligned ``eps_cell`` pixel grid (the RCWA input convention,
resolved EXACTLY: pixel boundaries where the adjacent column/row differs become
spectral-element walls -- no Fourier staircase).  These tests pin:

* the single-pillar cell reduces BYTE-IDENTICALLY to ``pmm_efficiency_2d``
  (same walls -> same operators -> same numbers);
* a two-pillar cell (not expressible by the old API) agrees with the
  independent ``rcwa_efficiency_2d`` and ``pmm_efficiency_2d_staggered``
  solvers and conserves energy;
* the Nyquist parity bump (even strip-count axis) keeps the solve well-posed;
* the prepared/sweep companions reproduce the direct per-wavelength calls;
* the cost guard catches sampled-smooth (anti-aliased) cells, and the
  validation errors fire.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm import (
    pmm_efficiency_2d,
    pmm_efficiency_2d_cell,
    pmm_efficiency_2d_cell_vs_wavelength,
    pmm_efficiency_2d_staggered,
    prepare_pmm_2d_cell,
)
from lumenairy.elements.rcwa import rcwa_efficiency_2d

_P = 0.6e-6
_WL = 0.55e-6
_DEP = 0.25e-6


def _order0(o, A):
    return float(np.asarray(A)[(o[:, 0] == 0) & (o[:, 1] == 0)][0])


def _two_pillar_cell(loss=0.0):
    """Two pillars on a COARSE 6x6 segment grid (walls at multiples of P/6;
    keeps the staggered cross-check's per-pixel-segment modal dim small)."""
    cell = np.full((6, 6), 1.0 + 0j)
    cell[1:3, 1:3] = 6.0 + loss * 1j
    cell[4:6, 4:5] = 4.0
    return cell


# --------------------------------------------------------------------------- #
# (1) single-pillar reduction: BYTE-identical to the pillar API
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("pol", ["te", "tm"])
def test_single_pillar_cell_matches_pillar_api_exactly(pol):
    S = 20
    cell = np.full((S, S), 2.0 + 0j)
    cell[4:12, 6:14] = 6.0
    # the pillar API gets the SAME wall floats the cell derivation computes
    xb = (4 * _P / S, 12 * _P / S)
    yb = (6 * _P / S, 14 * _P / S)
    o1, R1, T1 = pmm_efficiency_2d(_P, _P, 6.0, 2.0, xb, yb, 1.5, 1.0,
                                   _DEP, _WL, degree=7, n_orders=5,
                                   polarization=pol)
    o2, R2, T2 = pmm_efficiency_2d_cell(_P, _P, cell, 1.5, 1.0, _DEP, _WL,
                                        degree=7, n_orders=5,
                                        polarization=pol)
    assert np.array_equal(o1, o2)
    assert np.array_equal(np.asarray(R1), np.asarray(R2))
    assert np.array_equal(np.asarray(T1), np.asarray(T2))


# --------------------------------------------------------------------------- #
# (2) two-pillar cell vs independent solvers + energy
# --------------------------------------------------------------------------- #

def test_two_pillar_cell_vs_rcwa_and_staggered():
    # The staggered solver treats EVERY pixel as a segment (modal dim
    # 2*(Nx*(M-1))^2), so this shared-geometry cross-check lives on the
    # coarse 6x6 grid.
    cell = _two_pillar_cell()
    o1, R1, T1 = pmm_efficiency_2d_cell(_P, _P, cell, 1.5, 1.0, _DEP, _WL,
                                        degree=9, n_orders=6)
    # energy at the hybrid floor
    assert abs(float(R1.sum() + T1.sum()) - 1.0) < 2e-2
    # RCWA on an exact 5x upsampling of the SAME geometry (needs >= 4*n+1 px)
    up = np.kron(cell, np.ones((5, 5)))
    o2, R2, T2 = rcwa_efficiency_2d(_P, _P, up, 1.5, 1.0, _DEP, _WL,
                                    n_orders_x=6, n_orders_y=6,
                                    formulation="li")
    assert abs(_order0(o1, T1) - _order0(o2, T2)) < 2e-2
    # staggered solver shares the walls-on-grid eps_cell convention
    o3, R3, T3 = pmm_efficiency_2d_staggered(_P, _P, cell, 1.5, 1.0, _DEP,
                                             _WL, degree=6, n_orders=6)
    assert abs(_order0(o1, T1) - _order0(o3, T3)) < 2e-2


def test_lossy_region_absorbs_not_gains():
    cell = _two_pillar_cell(loss=0.8)
    o, R, T = pmm_efficiency_2d_cell(_P, _P, cell, 1.5, 1.0, _DEP, _WL,
                                     degree=9, n_orders=6)
    tot = float(R.sum() + T.sum())
    assert tot < 1.0 + 2e-2          # no gain
    assert tot < 0.97                # genuinely absorbs


# --------------------------------------------------------------------------- #
# (3) Nyquist parity bump: even strip count stays well-posed
# --------------------------------------------------------------------------- #

def test_parity_bump_two_strip_axis_conserves_energy():
    # ONE wall per axis -> 2 strips -> n_el_tot would be even -> widest strip
    # gets the +1 element bump; the solve must stay well-posed.
    S = 16
    cell = np.full((S, S), 1.0 + 0j)
    cell[:8, :8] = 4.0               # quarter block: walls at P/2 on both axes
    o, R, T = pmm_efficiency_2d_cell(_P, _P, cell, 1.5, 1.0, _DEP, _WL,
                                     degree=9, n_orders=6)
    assert abs(float(R.sum() + T.sum()) - 1.0) < 2e-2


# --------------------------------------------------------------------------- #
# (4) prepared / sweep companions
# --------------------------------------------------------------------------- #

def test_prepared_and_sweep_match_direct():
    cell = _two_pillar_cell()
    wls = (0.5e-6, 0.55e-6, 0.62e-6)
    prep = prepare_pmm_2d_cell(_P, _P, cell, 1.5, 1.0, _DEP,
                               degree=9, n_orders=6)
    o_sw, R_sw, T_sw = pmm_efficiency_2d_cell_vs_wavelength(
        _P, _P, cell, 1.5, 1.0, _DEP, wls, degree=9, n_orders=6)
    for i, w in enumerate(wls):
        o_d, R_d, T_d = pmm_efficiency_2d_cell(_P, _P, cell, 1.5, 1.0, _DEP,
                                               w, degree=9, n_orders=6)
        o_p, R_p, T_p = prep.solve(w)
        assert np.allclose(np.asarray(R_p), np.asarray(R_d), rtol=0,
                           atol=1e-10)
        assert np.allclose(np.asarray(T_p), np.asarray(T_d), rtol=0,
                           atol=1e-10)
        assert np.allclose(R_sw[i], np.asarray(R_p), rtol=0, atol=0)
        assert np.allclose(T_sw[i], np.asarray(T_p), rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# (5) guards
# --------------------------------------------------------------------------- #

def test_cost_guard_catches_sampled_smooth_cell():
    # anti-aliased disk: nearly every pixel boundary becomes a wall
    S = 64
    x = (np.arange(S) + 0.5) / S - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    disk = 1.0 + 3.0 / (1.0 + np.exp((np.hypot(X, Y) - 0.3) * 200))
    with pytest.raises(ValueError, match="max_nodal_dof"):
        pmm_efficiency_2d_cell(_P, _P, disk.astype(complex), 1.5, 1.0,
                               _DEP, _WL, degree=7, n_orders=5)


def test_validation_errors():
    cell = _two_pillar_cell()
    with pytest.raises(ValueError, match="ODD"):
        pmm_efficiency_2d_cell(_P, _P, cell, 1.5, 1.0, _DEP, _WL,
                               degree=8, n_orders=5)       # even degree
    with pytest.raises(ValueError, match="n_orders"):
        pmm_efficiency_2d_cell(_P, _P, cell, 1.5, 1.0, _DEP, _WL,
                               degree=3, n_orders=40)      # orders > nodes
    with pytest.raises(ValueError, match="2-D"):
        pmm_efficiency_2d_cell(_P, _P, np.ones(8, complex), 1.5, 1.0,
                               _DEP, _WL)                  # 1-D input
    with pytest.raises(ValueError, match="polarization"):
        pmm_efficiency_2d_cell(_P, _P, cell, 1.5, 1.0, _DEP, _WL,
                               polarization="circular")
