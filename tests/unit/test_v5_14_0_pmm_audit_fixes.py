"""v5.14.0 -- PMM accuracy/speed audit fixes (the 22-agent audit's confirmed
findings, each pinned by its own failing case):

* P1 dense resonances: the 1-D normal-incidence binary path's legacy forward
  branch flipped propagating modes on ~1e-15 QZ noise (8 of 13 degrees in
  12..24 returned sum(R)+sum(T) up to 65.7); the noise-robust branch is now
  unconditional -- every degree conserves energy;
* P2 Wood cutoff: the 1-D solvers now nudge off exact Rayleigh-cutoff
  wavelengths (was tot = 1.000253 silently);
* P1 staggered Wood divergence: warns inside the ~1/sqrt(distance) divergence
  band;
* P2 pillar bounds: inverted/degenerate bounds raise ValueError (was a
  silently wrong geometry / raw LinAlgError);
* P1 factorized 2-D assembly: machine-identical to the dense path and the
  dominant (88-97%) solve cost removed; the raised max_nodal_dof admits
  staircased curved cells;
* P2 stabilize pseudo-plateau: the consensus returns the energy-cleanest
  cluster member on lossless structures.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm import (
    pmm_efficiency_1d,
    pmm_efficiency_2d,
    pmm_efficiency_2d_cell,
    pmm_efficiency_2d_staggered,
    pmm_jones_1d,
)


def test_normal_incidence_degree_scan_all_conserving():
    """The P1 dense-resonance fix: EVERY degree 12..24 conserves energy at
    normal incidence with stabilize=False (was 8 of 13 catastrophically
    non-conserving, up to tot=65.7)."""
    for d in range(12, 25):
        o, R, T = pmm_efficiency_1d(1e-6, 2.0, 1.0, 1.45, 1.0, 0.3e-6, 0.5,
                                    632.8e-9, degree=d, polarization="te",
                                    stabilize=False)
        tot = float(R.sum() + T.sum())
        assert abs(tot - 1.0) < 1e-6, f"degree {d}: tot={tot}"


def test_jones_normal_incidence_degree_scan_all_conserving():
    eps_r = np.diag([4.0, 4.0, 4.0]).astype(complex)
    eps_g = np.eye(3, dtype=complex)
    for d in range(12, 19):
        o, R, T, _J = pmm_jones_1d(1e-6, eps_r, eps_g, 1.45, 1.0, 0.3e-6,
                                   0.5, 632.8e-9, degree=d, stabilize=False)
        for row in (0, 1):
            tot = float(R[row].sum() + T[row].sum())
            assert abs(tot - 1.0) < 1e-6, f"degree {d} row {row}: tot={tot}"


def test_1d_wood_cutoff_nudged():
    """wl = P*(1 - 1e-9) sits on the +/-1-order Rayleigh cutoff; the nudge
    keeps energy conserved (was a silent 2.5e-4 violation)."""
    o, R, T = pmm_efficiency_1d(1e-6, 2.0, 1.0, 1.0, 1.0, 0.3e-6, 0.5,
                                1e-6 * (1 - 1e-9), degree=16,
                                polarization="te", stabilize=False)
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-6


def test_staggered_warns_near_rayleigh_cutoff():
    cell = np.full((2, 2), 4.0 + 0j)
    cell[0, 0] = 2.0                       # patterned
    with pytest.warns(UserWarning, match="Rayleigh"):
        pmm_efficiency_2d_staggered(1e-6, 1e-6, cell, 1.0, 1.0, 0.5e-6,
                                    1e-6 * (1 - 1e-6), degree=4, n_orders=2)


def test_pillar_bounds_validation():
    kw = dict(degree=7, n_orders=4)
    P = 1e-6
    with pytest.raises(ValueError, match="x_bounds"):
        pmm_efficiency_2d(P, P, 6.0, 1.0, (0.75 * P, 0.25 * P),
                          (0.2 * P, 0.6 * P), 1.5, 1.0, 0.2e-6, 0.6e-6, **kw)
    with pytest.raises(ValueError, match="x_bounds"):
        pmm_efficiency_2d(P, P, 6.0, 1.0, (0.0, 0.6 * P),
                          (0.2 * P, 0.6 * P), 1.5, 1.0, 0.2e-6, 0.6e-6, **kw)
    with pytest.raises(ValueError, match="y_bounds"):
        pmm_efficiency_2d(P, P, 6.0, 1.0, (0.2 * P, 0.6 * P),
                          (0.4 * P, 0.4 * P), 1.5, 1.0, 0.2e-6, 0.6e-6, **kw)


def test_factorized_assembly_matches_dense_reference():
    """The factorized projected operators are machine-identical to the legacy
    dense-kron reference (kept as _assemble_2d) -- the P1 perf finding's
    correctness gate."""
    from lumenairy.elements.pmm.twod import (
        _assemble_2d,
        _axis_elem_counts,
        _build_axis,
        _cell_to_walls_tile,
        _projectors,
        _scalar_projected_ops,
    )
    P = 0.6e-6
    cell = np.full((6, 6), 2.25 + 0j)
    cell[1:3, 1:3] = 12.0
    cell[4:6, 4:5] = 12.0 + 0.5j
    xw, yw, tile = _cell_to_walls_tile(cell, P, P, "t")
    tile_i = np.conj(tile)
    deg = 9
    el_x = _axis_elem_counts(P, xw, deg, 1, "t", "x")
    el_y = _axis_elem_counts(P, yw, deg, 1, "t", "y")
    ax = _build_axis(P, xw, deg, el_x, False)
    ay = _build_axis(P, yw, deg, el_y, False)
    ox = np.arange(-4, 5)
    ops = _assemble_2d(ax, ay, tile_i, 1.0)
    Tp, Tpinv = _projectors(ax, ay, ox, ox)
    ref = dict(Gx0F=Tp @ ops["Gx"] @ Tpinv, Gy0F=Tp @ ops["Gy"] @ Tpinv,
               EpsF=Tp @ ops["Eps"] @ Tpinv, EinvF=Tp @ ops["Einv"] @ Tpinv,
               EpnF=Tp @ ops["Epn"] @ Tpinv)
    new = _scalar_projected_ops(ax, ay, tile_i, ox, ox, P, P)
    for k, refv in ref.items():
        sc = max(float(np.max(np.abs(refv))), 1e-300)
        assert np.max(np.abs(refv - new[k])) / sc < 1e-12, k


def test_raised_cap_admits_staircased_disk():
    """A 16x16 staircased disk (was blocked by the 4000-DOF dense cap) now
    solves under the factorized assembly and conserves energy at the
    staircase + Fourier floor."""
    S = 16
    x = (np.arange(S) + 0.5) / S - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    cell = np.where(np.hypot(X, Y) < 0.32, 6.25 + 0j, 1.0 + 0j)
    o, R, T = pmm_efficiency_2d_cell(0.6e-6, 0.6e-6, cell, 1.5, 1.0,
                                     0.25e-6, 0.55e-6, degree=7, n_orders=4)
    assert abs(float(R.sum() + T.sum()) - 1.0) < 3e-2


def test_stabilize_returns_energy_clean_member():
    """On a lossless cell the consensus pick is the energy-cleanest cluster
    member (the pseudo-plateau guard) -- the returned total must be within
    1e-5 of unity whenever any scanned degree achieves that."""
    o, R, T = pmm_efficiency_1d(1e-6, 2.0, 1.0, 1.45, 1.0, 0.3e-6, 0.5,
                                632.8e-9, degree=13, polarization="te",
                                stabilize=True)
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-5


def test_dispersive_sweeps():
    """v5.14 generality: the PMM dispersive wavelength sweeps (callable
    materials), mirroring the RCWA sweep API."""
    from lumenairy.elements.pmm import (
        pmm_efficiency_1d_vs_wavelength,
        pmm_jones_1d_vs_wavelength,
    )
    n_disp = lambda w: 2.0 + 0.1 * (w / 0.5e-6 - 1.0)   # noqa: E731
    wl, R, T = pmm_efficiency_1d_vs_wavelength(
        1e-6, n_disp, 1.0, 1.45, 1.0, 0.3e-6, 0.5, (0.5e-6, 0.6e-6),
        degree=12)
    assert R.shape == (2,) and np.all(np.abs(R + T - 1.0) < 1e-6)
    eps_d = lambda w: np.diag([n_disp(w) ** 2] * 3).astype(complex)  # noqa: E731
    wl, J, R2, T2 = pmm_jones_1d_vs_wavelength(
        1e-6, eps_d, np.eye(3, dtype=complex), 1.45, 1.0, 0.3e-6, 0.5,
        (0.5e-6, 0.6e-6), degree=12)
    assert J.shape == (2, 2, 2)
    assert np.all(np.abs(R2 + T2 - 1.0) < 1e-6)
    # scalar-in -> scalar-out convention
    w1, R1, T1 = pmm_efficiency_1d_vs_wavelength(
        1e-6, 2.0, 1.0, 1.45, 1.0, 0.3e-6, 0.5, 0.55e-6, degree=12)
    assert np.isscalar(R1) or np.ndim(R1) == 0
