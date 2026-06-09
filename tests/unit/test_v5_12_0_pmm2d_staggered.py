"""v5.12.0 -- canonical no-floor 2-D crossed-grating PMM (Granet 2023 staggered
modified-Legendre basis), ``pmm_efficiency_2d_staggered``.

The defining property vs the FMM-floored hybrid ``pmm_efficiency_2d`` is that the
energy balance is ``n_orders``-INDEPENDENT (no Fourier floor) and tracks only the
modal degree -- exercised directly below.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm.twod_staggered import pmm_efficiency_2d_staggered

_G = dict(period_x=0.8e-6, period_y=0.8e-6, depth=0.3e-6, wavelength=0.633e-6)


def _T00(o, T):
    return float(T[(o[:, 0] == 0) & (o[:, 1] == 0)][0])


def _R00(o, R):
    return float(R[(o[:, 0] == 0) & (o[:, 1] == 0)][0])


# quarter-cell square pillar (Nx=Ny=2, pillar in segment (0,0))
_PILLAR = np.array([[2.25, 1.0], [1.0, 1.0]], dtype=complex)


# ===================================================================
# Vacuum exactness
# ===================================================================
@pytest.mark.parametrize("n_orders", [3, 5, 7])
def test_vacuum_exact(n_orders):
    o, R, T = pmm_efficiency_2d_staggered(
        eps_cell=np.ones((2, 2)), n_substrate=1.0, n_superstrate=1.0,
        degree=6, n_orders=n_orders, **_G)
    assert abs(_T00(o, T) - 1.0) < 1e-10
    assert R.sum() < 1e-10
    assert abs(R.sum() + T.sum() - 1.0) < 1e-10


# ===================================================================
# THE no-floor signatures
# ===================================================================
def test_no_fourier_floor():
    """Energy balance is BYTE-IDENTICAL across n_orders -- the no-Fourier-floor
    property that distinguishes this from the FMM-floored hybrid."""
    vals = []
    for no in (3, 5, 7, 9):
        o, R, T = pmm_efficiency_2d_staggered(
            eps_cell=_PILLAR, n_substrate=1.0, n_superstrate=1.0,
            degree=8, n_orders=no, **_G)
        vals.append(R.sum() + T.sum())
    assert max(vals) - min(vals) < 1e-13          # n_orders-independent
    assert abs(vals[0] - 1.0) < 1e-7              # and small


def test_no_floor_in_degree():
    """Pillar energy error decreases monotonically with degree to round-off
    (no plateau)."""
    errs = []
    for M in (6, 8, 10):
        o, R, T = pmm_efficiency_2d_staggered(
            eps_cell=_PILLAR, n_substrate=1.0, n_superstrate=1.0,
            degree=M, n_orders=5, **_G)
        errs.append(abs(R.sum() + T.sum() - 1.0))
    assert errs[1] < errs[0] and errs[2] < errs[1]
    assert errs[2] < 1e-11


# ===================================================================
# Uniform-slab Fabry-Perot vs analytic (independent oracle)
# ===================================================================
def test_uniform_slab_fabry_perot():
    """A uniform layer must reproduce the analytic single-layer (characteristic
    matrix) reflectance/transmittance, with all non-zero orders vanishing."""
    n_sup, n_sub, n_f = 1.0, 1.5, 2.0
    o, R, T = pmm_efficiency_2d_staggered(
        eps_cell=np.full((2, 2), n_f ** 2), n_substrate=n_sub,
        n_superstrate=n_sup, degree=6, n_orders=4, **_G)
    # analytic normal-incidence slab (admittances ~ index):
    delta = 2.0 * np.pi * n_f * _G["depth"] / _G["wavelength"]
    B = np.cos(delta) + 1j * np.sin(delta) / n_f * n_sub
    Cc = 1j * n_f * np.sin(delta) + np.cos(delta) * n_sub
    r = (n_sup * B - Cc) / (n_sup * B + Cc)
    R_an = abs(r) ** 2
    T_an = 1.0 - R_an
    assert abs(_R00(o, R) - R_an) < 1e-9
    assert abs(_T00(o, T) - T_an) < 1e-9
    # non-zero orders carry no power for a uniform layer
    nz = ~((o[:, 0] == 0) & (o[:, 1] == 0))
    assert R[nz].sum() + T[nz].sum() < 1e-9


# ===================================================================
# Position invariance (lattice translation preserves per-order efficiency)
# ===================================================================
def test_position_invariance():
    """The pillar translated by half a period gives identical per-order
    efficiencies (a correctness check the FMM staircase cannot match exactly)."""
    base = dict(n_substrate=1.0, n_superstrate=1.0, degree=8, n_orders=5, **_G)
    o1, R1, T1 = pmm_efficiency_2d_staggered(eps_cell=_PILLAR, **base)
    pillar2 = np.array([[1.0, 1.0], [1.0, 2.25]], dtype=complex)  # shifted (1,1)
    o2, R2, T2 = pmm_efficiency_2d_staggered(eps_cell=pillar2, **base)
    assert np.array_equal(o1, o2)
    assert np.max(np.abs(R1 - R2)) < 1e-9
    assert np.max(np.abs(T1 - T2)) < 1e-9


# ===================================================================
# Physical absorption (R+T=1 in the lossless case is physics, not identity)
# ===================================================================
def test_lossy_pillar_absorbs():
    eps = np.array([[2.25 + 0.5j, 1.0], [1.0, 1.0]], dtype=complex)
    o, R, T = pmm_efficiency_2d_staggered(
        eps_cell=eps, n_substrate=1.0, n_superstrate=1.0, degree=8,
        n_orders=5, **_G)
    A = 1.0 - R.sum() - T.sum()
    assert A > 1e-3                       # genuine absorption
    assert A < 1.0


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_polarization_runs_and_conserves(pol):
    o, R, T = pmm_efficiency_2d_staggered(
        eps_cell=_PILLAR, n_substrate=1.0, n_superstrate=1.0, degree=8,
        n_orders=5, polarization=pol, **_G)
    assert abs(R.sum() + T.sum() - 1.0) < 1e-7


# ===================================================================
# Cross-check vs the Fourier method (brackets the same value)
# ===================================================================
def test_matches_rcwa():
    """PMM (converged, position-invariant) and a high-order RCWA-li agree on
    the dominant orders within RCWA's own truncation error on this hard
    high-contrast large-period case."""
    rcwa = pytest.importorskip("lumenairy.elements.rcwa")
    o, R, T = pmm_efficiency_2d_staggered(
        eps_cell=_PILLAR, n_substrate=1.0, n_superstrate=1.0, degree=12,
        n_orders=7, **_G)
    # rasterize the same quarter-cell pillar for RCWA
    n_ord = 15
    S = 8 * n_ord + 1
    e = np.ones((S, S), dtype=complex)
    xc = (np.arange(S) + 0.5) / S
    XX, YY = np.meshgrid(xc, xc, indexing="ij")
    e[(XX < 0.5) & (YY < 0.5)] = 2.25
    orc, Rr, Tr = rcwa.rcwa_efficiency_2d(
        _G["period_x"], _G["period_y"], e, 1.0, 1.0, _G["depth"],
        _G["wavelength"], polarization="te", n_orders_x=n_ord, n_orders_y=n_ord,
        formulation="li", stabilize=True)
    assert abs(_T00(o, T) - _T00(orc, Tr)) < 3e-3
    assert abs(R.sum() - Rr.sum()) < 3e-3


# ===================================================================
# Input validation
# ===================================================================
def test_validation():
    base = dict(n_substrate=1.0, n_superstrate=1.0, **_G)
    with pytest.raises(ValueError):
        pmm_efficiency_2d_staggered(eps_cell=_PILLAR, polarization="xx", **base)
    with pytest.raises(ValueError):
        pmm_efficiency_2d_staggered(eps_cell=_PILLAR, degree=1, **base)
    with pytest.raises(ValueError):
        pmm_efficiency_2d_staggered(eps_cell=np.ones(3), **base)  # not 2-D
