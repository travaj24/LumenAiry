"""Fast CI regression pins for the native RCWA module
(:mod:`lumenairy.elements.rcwa`).

These mirror a subset of the thorough physics validation in
``validation/elements/test_rcwa.py`` but run quickly (small truncation) as
unit-suite gates.  No external/GPL RCWA oracle is used here -- the
references are the analytic Airy thin-film, the library's own TMM
(``coating_reflectance``), and exact energy conservation.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.rcwa import (
    RCWAStack,
    rcwa_efficiency_1d,
    rcwa_efficiency_2d,
    rcwa_efficiency_2d_shapes,
    rcwa_efficiency_vs_wavelength,
    rcwa_jones_1d,
    rcwa_jones_2d,
    uniaxial_tensor,
)

WL = 0.633e-6
_C = np.complex128


def _airy_R(n_sup, n_film, n_sub, d, wl, angle, pol):
    kx = np.real(n_sup) * np.sin(angle)

    def kz(n):
        return np.sqrt(_C(n) ** 2 - kx ** 2)
    kzs, kzf, kzt = kz(n_sup), kz(n_film), kz(n_sub)
    if pol == "te":
        r01 = (kzs - kzf) / (kzs + kzf)
        r12 = (kzf - kzt) / (kzf + kzt)
    else:
        e0, e1, e2 = _C(n_sup) ** 2, _C(n_film) ** 2, _C(n_sub) ** 2
        r01 = (e1 * kzs - e0 * kzf) / (e1 * kzs + e0 * kzf)
        r12 = (e2 * kzf - e1 * kzt) / (e2 * kzf + e1 * kzt)
    ph = np.exp(2j * (2 * np.pi / wl) * kzf * d)
    r = (r01 + r12 * ph) / (1 + r01 * r12 * ph)
    return float(np.abs(r) ** 2)


@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("ang_deg", [0.0, 30.0])
def test_uniform_layer_matches_airy(pol, ang_deg):
    """A grating with n_ridge == n_groove is a homogeneous slab; its rigorous
    zeroth-order reflectance must equal the exact Airy thin-film result."""
    ang = np.deg2rad(ang_deg)
    n_film, d = 2.2, 0.3e-6
    o, R, T = rcwa_efficiency_1d(0.5e-6, n_film, n_film, 1.5, 1.0, d, 0.5,
                                 WL, angle=ang, polarization=pol, n_orders=5)
    R0 = R[len(o) // 2]
    assert abs(R0 - _airy_R(1.0, n_film, 1.5, d, WL, ang, pol)) < 1e-10


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_uniform_layer_matches_library_tmm(pol):
    """Cross-check the uniform limit against the library's own coatings TMM."""
    n_film, d, ang = 1.9, 0.45e-6, np.deg2rad(20)
    Rc, _, _ = la.coating_reflectance(
        [(n_film, d)], WL, angle=ang, n_substrate=1.5, n_ambient=1.0,
        polarization=("s" if pol == "te" else "p"))
    o, R, T = rcwa_efficiency_1d(0.5e-6, n_film, n_film, 1.5, 1.0, d, 0.5,
                                 WL, angle=ang, polarization=pol, n_orders=4)
    assert abs(R[len(o) // 2] - float(np.atleast_1d(Rc)[0])) < 1e-10


@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("M", [5, 11, 21])
def test_energy_conservation_lossless(pol, M):
    o, R, T = rcwa_efficiency_1d(1.2e-6, 2.0, 1.0, 1.5, 1.0, 0.5e-6, 0.4, WL,
                                 angle=np.deg2rad(20), polarization=pol,
                                 n_orders=M)
    assert abs(R.sum() + T.sum() - 1.0) < 1e-9


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_metal_grating_positive_absorptance(pol):
    """Silver grating: the loss-sign convention bridge must yield POSITIVE
    absorptance A = 1 - R - T (a passive metal absorbs, never emits)."""
    n_ag = 0.056 + 4.28j
    o, R, T = rcwa_efficiency_1d(0.6e-6, n_ag, 1.0, 1.5, 1.0, 0.12e-6, 0.5, WL,
                                 angle=np.deg2rad(25), polarization=pol,
                                 n_orders=21)
    A = 1.0 - R.sum() - T.sum()
    assert np.all(np.isfinite(R)) and np.all(np.isfinite(T))
    assert A > 1e-3, f"absorptance {A} must be positive"
    assert R.sum() + T.sum() < 1.0 + 1e-9


def test_high_order_stability():
    """The Re>=0 layer-eigenvalue branch keeps the solver stable to large
    truncation (no exponential evanescent blow-up)."""
    last = None
    for M in (11, 21, 31):
        o, R, T = rcwa_efficiency_1d(1.2e-6, 2.0, 1.0, 1.5, 1.0, 0.5e-6, 0.4,
                                     WL, polarization="tm", n_orders=M)
        assert np.all(np.isfinite(R)) and abs(R.sum() + T.sum() - 1) < 1e-8
        last = R[len(o) // 2]
    assert 0.0 <= last <= 1.0


def test_order_symmetry_normal_vs_oblique():
    o, R, _ = rcwa_efficiency_1d(2.0e-6, 1.6, 1.0, 1.0, 1.0, 0.6e-6, 0.3, WL,
                                 angle=0.0, polarization="te", n_orders=8)
    mid = len(o) // 2
    # Single binary ridge is mirror-symmetric -> +/-m symmetric at normal.
    assert max(abs(R[mid + m] - R[mid - m]) for m in range(1, 6)) < 1e-9
    o2, R2, _ = rcwa_efficiency_1d(2.0e-6, 1.6, 1.0, 1.0, 1.0, 0.6e-6, 0.3, WL,
                                   angle=np.deg2rad(18), polarization="te",
                                   n_orders=8)
    mid2 = len(o2) // 2
    # Oblique breaks the symmetry (also pins the kx0 + m order-sign).
    assert max(abs(R2[mid2 + m] - R2[mid2 - m]) for m in range(1, 6)) > 1e-3


def test_wood_anomaly_does_not_crash():
    """An order at EXACT grazing (kx_m = +/-n) must not raise -- the
    wavelength-nudge regularisation keeps every matrix invertible."""
    # period 1.4 um, n_sup=1: order +/-2 grazes at wl=0.7 um (kx=+/-1.0).
    o, R, T = rcwa_efficiency_1d(1.4e-6, 2.0, 1.0, 1.5, 1.0, 0.5e-6, 0.5,
                                 0.7e-6, polarization="te", n_orders=11)
    assert np.all(np.isfinite(R)) and np.all(np.isfinite(T))
    assert abs(R.sum() + T.sum() - 1.0) < 1e-6


@pytest.mark.parametrize("bad", [
    dict(polarization="circular"),
    dict(duty_cycle=1.5),
    dict(formulation="spectral"),
])
def test_input_validation(bad):
    base = dict(period=1e-6, n_ridge=2.0, n_groove=1.0, n_substrate=1.5,
                n_superstrate=1.0, depth=0.4e-6, duty_cycle=0.5, wavelength=WL)
    base.update(bad)
    with pytest.raises(ValueError):
        rcwa_efficiency_1d(**base)


def test_output_shapes():
    o, R, T = rcwa_efficiency_1d(1e-6, 2.0, 1.0, 1.5, 1.0, 0.4e-6, 0.5, WL,
                                 n_orders=7)
    assert o.shape == R.shape == T.shape == (15,)
    assert o[0] == -7 and o[-1] == 7
    assert np.all(R >= 0) and np.all(T >= 0)


def test_wavelength_sweep_matches_per_call():
    wls = np.linspace(0.5e-6, 0.7e-6, 5)
    eff = rcwa_efficiency_vs_wavelength(1.4e-6, 2.0, 1.0, 1.5, 1.0, 0.5e-6,
                                        0.5, wls, order=1, polarization="te",
                                        n_orders=11)
    assert eff.shape == (5,)
    for i, w in enumerate(wls):
        o, R, T = rcwa_efficiency_1d(1.4e-6, 2.0, 1.0, 1.5, 1.0, 0.5e-6, 0.5,
                                     float(w), polarization="te", n_orders=11)
        assert abs(eff[i] - T[np.searchsorted(o, 1)]) < 1e-12
    # Scalar wavelength -> scalar return.
    s = rcwa_efficiency_vs_wavelength(1.4e-6, 2.0, 1.0, 1.5, 1.0, 0.5e-6, 0.5,
                                      0.6e-6, order=1)
    assert np.isscalar(s)


# ============================== 2-D crossed gratings =======================

@pytest.mark.parametrize("pol", ["te", "tm"])
def test_2d_uniform_cell_matches_airy(pol):
    """A spatially-uniform 2-D cell is a homogeneous slab; its 0th-order
    rigorous reflectance must equal the analytic Airy result (conical
    incidence; a uniform slab depends only on the polar angle)."""
    n_film, d, th, ph = 2.1, 0.35e-6, np.deg2rad(25), np.deg2rad(40)
    cell = np.full((8, 8), n_film ** 2, dtype=complex)
    o, R, T = rcwa_efficiency_2d(0.5e-6, 0.5e-6, cell, 1.5, 1.0, d, WL,
                                 theta=th, phi=ph, polarization=pol,
                                 n_orders_x=2, n_orders_y=2)
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    assert abs(R[p0] - _airy_R(1.0, n_film, 1.5, d, WL, th, pol)) < 1e-9


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_2d_energy_conservation(pol):
    """Lossless 2-D dielectric pillar, conical incidence: sum(R)+sum(T)=1."""
    S = 48
    xx, yy = np.meshgrid((np.arange(S) + .5) / S, (np.arange(S) + .5) / S,
                         indexing="ij")
    cell = np.where((np.abs(xx - .5) < .25) & (np.abs(yy - .5) < .25),
                    2.5 ** 2, 1.0).astype(complex)
    o, R, T = rcwa_efficiency_2d(0.8e-6, 0.8e-6, cell, 1.45, 1.0, 0.4e-6, WL,
                                 theta=np.deg2rad(12), phi=np.deg2rad(25),
                                 polarization=pol, n_orders_x=4, n_orders_y=4)
    assert abs(R.sum() + T.sum() - 1.0) < 1e-9


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_2d_reduces_to_1d(pol):
    """A y-invariant 2-D cell must reproduce the 1-D core (Laurent) totals,
    and put no power into n!=0 orders (no y-momentum transfer)."""
    Sx = 512
    x = (np.arange(Sx) + 0.5) / Sx
    cell = np.tile(np.where(x < 0.4, 2.0 ** 2, 1.0)[:, None], (1, 8)).astype(complex)
    o, R, T = rcwa_efficiency_2d(1.2e-6, 1.0e-6, cell, 1.5, 1.0, 0.5e-6, WL,
                                 theta=np.deg2rad(20), phi=0.0,
                                 polarization=pol, n_orders_x=8, n_orders_y=1)
    o1, R1, T1 = rcwa_efficiency_1d(1.2e-6, 2.0, 1.0, 1.5, 1.0, 0.5e-6, 0.4,
                                    WL, angle=np.deg2rad(20), polarization=pol,
                                    n_orders=8, formulation="laurent")
    assert abs(R.sum() - R1.sum()) < 5e-3 and abs(T.sum() - T1.sum()) < 5e-3
    spurious = R[o[:, 1] != 0].sum() + T[o[:, 1] != 0].sum()
    assert spurious < 1e-12, f"y-invariant cell leaked {spurious} into n!=0"


def test_2d_input_validation():
    cell = np.full((8, 8), 2.0, dtype=complex)
    with pytest.raises(ValueError):
        rcwa_efficiency_2d(0.5e-6, 0.5e-6, cell, 1.5, 1.0, 0.3e-6, WL,
                           polarization="diagonal")
    with pytest.raises(ValueError):
        rcwa_efficiency_2d(0.5e-6, 0.5e-6, cell, 1.5, 1.0, 0.3e-6, WL,
                           formulation="fff")


# ============================== 1-D anisotropic (LC) =======================

@pytest.mark.parametrize("ang_deg", [0.0, 20.0])
def test_aniso_reduces_to_isotropic_core(ang_deg):
    """tensor = scalar*I must reproduce the isotropic core: incident-Ey == TE,
    incident-Ex == TM, bit-identically."""
    nr, ng = 2.2, 1.0
    er, eg = np.eye(3) * nr ** 2, np.eye(3) * ng ** 2
    ang = np.deg2rad(ang_deg)
    o, R, T, J = rcwa_jones_1d(1.2e-6, er.astype(complex), eg.astype(complex),
                               1.5, 1.0, 0.5e-6, 0.4, WL, angle=ang, n_orders=11)
    _, Rte, Tte = rcwa_efficiency_1d(1.2e-6, nr, ng, 1.5, 1.0, 0.5e-6, 0.4, WL,
                                     angle=ang, polarization="te", n_orders=11)
    _, Rtm, Ttm = rcwa_efficiency_1d(1.2e-6, nr, ng, 1.5, 1.0, 0.5e-6, 0.4, WL,
                                     angle=ang, polarization="tm", n_orders=11)
    assert np.max(np.abs(R[1] - Rte)) < 1e-10  # Ey == TE
    assert np.max(np.abs(R[0] - Rtm)) < 1e-10  # Ex == TM
    # Isotropic -> no cross-polarization in the Jones reflection.
    assert abs(J[0, 1]) < 1e-10 and abs(J[1, 0]) < 1e-10


def test_lc_grating_energy_and_coupling():
    """In-plane LC grating: energy conserved and the in-plane director
    rotation produces genuine TE/TM (Jones cross-) coupling."""
    elc = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=np.deg2rad(35))
    o, R, T, J = rcwa_jones_1d(0.9e-6, elc, np.eye(3, dtype=complex), 1.45, 1.0,
                               0.4e-6, 0.5, WL, angle=np.deg2rad(12), n_orders=20)
    for row in (0, 1):
        assert abs(R[row].sum() + T[row].sum() - 1.0) < 1e-9
    assert abs(J[0, 1]) > 1e-3 and abs(J[1, 0]) > 1e-3  # genuine coupling


def test_uniaxial_tensor_properties():
    # In-plane director (theta=pi/2): ezz = n_o^2, eigenvalues = {no^2,no^2,ne^2}.
    e = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=0.3)
    assert np.isclose(e[2, 2], 1.5 ** 2)
    assert np.allclose(np.sort(np.linalg.eigvals(e).real),
                       [1.5 ** 2, 1.5 ** 2, 1.7 ** 2])
    # Zero birefringence -> isotropic.
    assert np.allclose(uniaxial_tensor(1.6, 1.6, 0.7, phi=1.1),
                       1.6 ** 2 * np.eye(3))


# ============================== 2-D anisotropic ============================

def _scalar_tensor_cell(S, mask, n_in, n_out=1.0):
    cell = np.empty((S, S, 3, 3), dtype=complex)
    cell[mask] = np.eye(3) * n_in ** 2
    cell[~mask] = np.eye(3) * n_out ** 2
    return cell


def test_jones_2d_reduces_to_isotropic_at_normal():
    """At normal incidence a scalar tensor cell must reproduce the isotropic
    2-D solver (Ey==TE, Ex==TM) bit-exactly, with zero cross-polarization for
    a symmetric pillar."""
    S = 32
    xx, yy = np.meshgrid((np.arange(S) + .5) / S, (np.arange(S) + .5) / S,
                         indexing="ij")
    mask = (np.abs(xx - .5) < .25) & (np.abs(yy - .5) < .25)
    ct = _scalar_tensor_cell(S, mask, 2.3)
    cs = np.where(mask, 2.3 ** 2, 1.0).astype(complex)
    o, R, T, J = rcwa_jones_2d(0.8e-6, 0.8e-6, ct, 1.45, 1.0, 0.4e-6, WL,
                               theta=0.0, phi=0.0, n_orders_x=4, n_orders_y=4)
    _, Rte, _ = rcwa_efficiency_2d(0.8e-6, 0.8e-6, cs, 1.45, 1.0, 0.4e-6, WL,
                                   polarization="te", n_orders_x=4, n_orders_y=4)
    _, Rtm, _ = rcwa_efficiency_2d(0.8e-6, 0.8e-6, cs, 1.45, 1.0, 0.4e-6, WL,
                                   polarization="tm", n_orders_x=4, n_orders_y=4)
    assert np.max(np.abs(R[1] - Rte)) < 1e-10  # Ey == TE
    assert np.max(np.abs(R[0] - Rtm)) < 1e-10  # Ex == TM
    assert abs(J[0, 1]) < 1e-10 and abs(J[1, 0]) < 1e-10  # no cross-pol


def test_jones_2d_lc_energy_and_coupling():
    """In-plane LC tensor on a 2-D lattice: energy conserved, real TE/TM
    (Jones cross-) coupling from the director rotation."""
    S = 36
    xx, yy = np.meshgrid((np.arange(S) + .5) / S, (np.arange(S) + .5) / S,
                         indexing="ij")
    mask = (np.abs(xx - .5) < .25) & (np.abs(yy - .5) < .25)
    elc = uniaxial_tensor(1.5, 1.75, np.pi / 2, phi=np.deg2rad(30))
    cell = np.empty((S, S, 3, 3), dtype=complex)
    cell[mask] = elc
    cell[~mask] = np.eye(3)
    o, R, T, J = rcwa_jones_2d(0.9e-6, 0.9e-6, cell, 1.45, 1.0, 0.4e-6, WL,
                               theta=np.deg2rad(10), phi=np.deg2rad(25),
                               n_orders_x=5, n_orders_y=5)
    assert abs(R[0].sum() + T[0].sum() - 1.0) < 1e-9
    assert abs(R[1].sum() + T[1].sum() - 1.0) < 1e-9
    assert abs(J[0, 1]) > 1e-3 and abs(J[1, 0]) > 1e-3


# ============================== Analytic-FT + dual-Laurent =================

@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("ang_deg", [0.0, 30.0])
def test_analytic_uniform_limit_matches_airy(pol, ang_deg):
    """No shapes -> background-only slab; the analytic-FT dual-Laurent solver
    must reproduce the exact Airy thin-film (factorization-independent gold
    standard) at conical incidence."""
    nf, d = 2.2, 0.3e-6
    ang = np.deg2rad(ang_deg)
    o, R, T = rcwa_efficiency_2d_shapes(0.5e-6, 0.5e-6, nf ** 2, [], 1.5, 1.0,
                                        d, WL, theta=ang, phi=np.deg2rad(35),
                                        polarization=pol, n_orders_x=2,
                                        n_orders_y=2)
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    assert abs(R[p0] - _airy_R(1.0, nf, 1.5, d, WL, ang, pol)) < 1e-10


def test_analytic_ft_is_exact_vs_fine_fft():
    """The analytic rectangle [[eps]] is the exact spectrum; an FFT-sampled
    cell approaches it as the sampling is refined."""
    from lumenairy.elements.rcwa import (
        _analytic_convolutions_2d,
        _eps_convolution_2d,
        _harmonic_orders_2d,
    )
    P, nox, noy = 0.7e-6, 4, 4
    orders, _ = _harmonic_orders_2d(nox, noy)
    shapes = [{"shape": "rectangle", "eps": 2.5 ** 2,
               "size": (0.35e-6, 0.35e-6), "center": (0.35e-6, 0.35e-6)}]
    EPSa, _ = _analytic_convolutions_2d(1.0, shapes, orders, nox, noy, P, P)
    prev = None
    for S in (256, 1024):
        xx, yy = np.meshgrid((np.arange(S) + .5) / S, (np.arange(S) + .5) / S,
                             indexing="ij")
        cell = np.where((np.abs(xx - .5) < .25) & (np.abs(yy - .5) < .25),
                        2.5 ** 2, 1.0).astype(complex)
        err = np.max(np.abs(_eps_convolution_2d(cell, orders, nox, noy) - EPSa))
        if prev is not None:
            assert err < prev          # FFT converges toward the analytic
        prev = err


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_analytic_energy_and_clean_convergence(pol):
    """Analytic-FT dual-Laurent conserves energy exactly and converges
    monotonically/cleanly for a high-contrast dielectric pillar."""
    sh = [{"shape": "rectangle", "eps": 3.5 ** 2, "size": (0.3e-6, 0.3e-6),
           "center": (0.3e-6, 0.3e-6)}]
    vals = []
    for M in (6, 10, 14):
        o, R, T = rcwa_efficiency_2d_shapes(0.6e-6, 0.6e-6, 1.0, sh, 1.0, 1.0,
                                            0.2e-6, WL, polarization=pol,
                                            n_orders_x=M, n_orders_y=M)
        assert abs(R.sum() + T.sum() - 1.0) < 1e-9
        vals.append(R.sum())
    assert abs(vals[-1] - vals[-2]) < abs(vals[-2] - vals[0])  # converging


def test_analytic_metal_disk_positive_absorptance():
    sh = [{"shape": "disk", "eps": (0.2 + 3.4j) ** 2, "radius": 0.18e-6,
           "center": (0.25e-6, 0.25e-6)}]
    o, R, T = rcwa_efficiency_2d_shapes(0.5e-6, 0.5e-6, 1.0, sh, 1.0, 1.0,
                                        0.06e-6, WL, polarization="tm",
                                        n_orders_x=8, n_orders_y=8)
    A = 1.0 - R.sum() - T.sum()
    assert A > 1e-3 and R.sum() + T.sum() <= 1.0 + 1e-9


def test_analytic_input_validation():
    with pytest.raises(ValueError):
        rcwa_efficiency_2d_shapes(0.5e-6, 0.5e-6, 1.0,
                                  [{"shape": "triangle", "eps": 2.0,
                                    "size": (1e-7, 1e-7)}],
                                  1.0, 1.0, 0.1e-6, WL, n_orders_x=2,
                                  n_orders_y=2)
    with pytest.raises(ValueError):
        rcwa_efficiency_2d_shapes(0.5e-6, 0.5e-6, 1.0, [], 1.0, 1.0, 0.1e-6,
                                  WL, polarization="x", n_orders_x=2,
                                  n_orders_y=2)


def test_stack_analytic_shape_layer_matches_standalone():
    """A single analytic-shape stack layer reproduces rcwa_efficiency_2d_shapes
    (Ey row == TE at normal incidence)."""
    sh = [{"shape": "disk", "eps": 2.5 ** 2, "radius": 0.15e-6,
           "center": (0.3e-6, 0.3e-6)}]
    st = RCWAStack(0.6e-6, period_y=0.6e-6, n_superstrate=1.0, n_substrate=1.5,
                   n_orders=6, n_orders_y=6)
    st.add_layer(0.25e-6, shapes=sh, eps_background=1.0)
    res = st.set_source(WL).solve()
    o, R, T = res.efficiencies()
    o2, R2, T2 = rcwa_efficiency_2d_shapes(0.6e-6, 0.6e-6, 1.0, sh, 1.5, 1.0,
                                           0.25e-6, WL, polarization="te",
                                           n_orders_x=6, n_orders_y=6)
    assert np.max(np.abs(R[1] - R2)) < 1e-12  # Ey == TE at normal


# ============================== RCWAStack (multi-layer) ====================

def test_stack_single_tensor_layer_matches_jones_2d():
    """A single-layer stack must reproduce the standalone rcwa_jones_2d
    bit-exactly (efficiencies AND the Jones reflection)."""
    S = 32
    xx, yy = np.meshgrid((np.arange(S) + .5) / S, (np.arange(S) + .5) / S,
                         indexing="ij")
    mask = (np.abs(xx - .5) < .25) & (np.abs(yy - .5) < .25)
    elc = uniaxial_tensor(1.5, 1.75, np.pi / 2, phi=np.deg2rad(30))
    cell = np.empty((S, S, 3, 3), dtype=complex)
    cell[mask] = elc
    cell[~mask] = np.eye(3)
    st = RCWAStack(0.9e-6, period_y=0.9e-6, n_superstrate=1.0,
                   n_substrate=1.45, n_orders=5, n_orders_y=5)
    st.add_layer(0.4e-6, eps_tensor_cell=cell)
    res = st.set_source(WL, theta=np.deg2rad(10), phi=np.deg2rad(25)).solve()
    o, R, T = res.efficiencies()
    o2, R2, T2, J2 = rcwa_jones_2d(0.9e-6, 0.9e-6, cell, 1.45, 1.0, 0.4e-6, WL,
                                   theta=np.deg2rad(10), phi=np.deg2rad(25),
                                   n_orders_x=5, n_orders_y=5)
    assert np.max(np.abs(R - R2)) < 1e-12
    assert np.max(np.abs(res.jones_reflection() - J2)) < 1e-12


def test_stack_multilayer_energy_and_absorptance():
    """A multi-layer stack (spacer + patterned + spacer) conserves energy;
    a lossy layer gives positive absorptance."""
    S = 32
    xx, yy = np.meshgrid((np.arange(S) + .5) / S, (np.arange(S) + .5) / S,
                         indexing="ij")
    mask = (np.abs(xx - .5) < .25) & (np.abs(yy - .5) < .25)
    cell = np.where(mask, 2.3 ** 2, 1.0).astype(complex)
    st = RCWAStack(0.8e-6, period_y=0.8e-6, n_superstrate=1.0,
                   n_substrate=1.5, n_orders=4, n_orders_y=4)
    st.add_layer(0.15e-6, eps=2.1 ** 2).add_layer(0.3e-6, eps_cell=cell)
    st.add_layer(0.1e-6, eps=1.6 ** 2)
    res = st.set_source(WL, theta=np.deg2rad(8)).solve()
    o, R, T = res.efficiencies()
    assert abs(R[0].sum() + T[0].sum() - 1.0) < 1e-9
    assert np.allclose(res.absorptance(), 0.0, atol=1e-9)  # lossless
    # lossy patterned layer -> positive absorptance
    cell_l = np.where(mask, (0.2 + 3.4j) ** 2, 1.0).astype(complex)
    st2 = RCWAStack(0.5e-6, period_y=0.5e-6, n_superstrate=1.0,
                    n_substrate=1.0, n_orders=5, n_orders_y=5)
    st2.add_layer(0.06e-6, eps_cell=cell_l)
    res2 = st2.set_source(WL).solve()
    o2, R2, T2 = res2.efficiencies()
    assert np.all(res2.absorptance() > 1e-3)
    assert np.all(R2.sum(axis=1) + T2.sum(axis=1) <= 1.0 + 1e-9)


def test_stack_input_validation():
    st = RCWAStack(1e-6, n_orders=5)
    with pytest.raises(ValueError):
        st.solve()  # no source / no layers
    with pytest.raises(ValueError):
        st.add_layer(0.1e-6, eps=2.0, eps_cell=np.ones((4, 4)))  # two specs


def test_stack_jones_bridge_to_jonesfield():
    """RCWAResult.apply_reflection drops the rigorous Jones reflection into
    the JonesField polarization pipeline."""
    from lumenairy.elements.polarization import JonesField
    elc = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=np.deg2rad(30))
    S = 24
    cell = np.broadcast_to(elc, (S, S, 3, 3)).copy()
    st = RCWAStack(0.8e-6, period_y=0.8e-6, n_orders=4, n_orders_y=4)
    res = st.add_layer(0.3e-6, eps_tensor_cell=cell).set_source(WL).solve()
    n = 8
    jf = JonesField(np.ones((n, n), complex), np.zeros((n, n), complex), dx=1e-6)
    out = res.apply_reflection(jf)
    # The applied 2x2 maps incident (Ex=1, Ey=0) to the Jones column 0.
    J = res.jones_reflection()
    assert np.allclose(out.Ex, J[0, 0]) and np.allclose(out.Ey, J[1, 0])


# ============================== JAX autodiff ===============================

jax = pytest.importorskip("jax")


def test_jax_value_matches_numpy():
    """The differentiable JAX twin matches the NumPy core (soft-edge approx)
    and conserves energy."""
    jax.config.update("jax_enable_x64", True)
    from lumenairy.elements.rcwa import rcwa_efficiency_1d_jax
    o, R, T = rcwa_efficiency_1d(1.2e-6, 2.2, 1.0, 1.5, 1.0, 0.5e-6, 0.5, WL,
                                 angle=np.deg2rad(15), polarization="tm",
                                 n_orders=11)
    oj, Rj, Tj = rcwa_efficiency_1d_jax(1.2e-6, 2.2, 1.0, 1.5, 1.0, 0.5e-6, 0.5,
                                        WL, angle=np.deg2rad(15),
                                        polarization="tm", n_orders=11,
                                        n_samples=2048)
    m = len(o) // 2
    assert abs(T[m] - float(Tj[m])) < 5e-3
    assert abs(float(Rj.sum() + Tj.sum()) - 1.0) < 1e-6


def test_jax_gradients_match_finite_difference():
    """Autodiff gradients (incl. the non-Hermitian eig path) match central
    finite differences -- the inverse-design enabler.  The eig custom VJP's
    output conjugation (JAX Wirtinger convention) is what makes the
    permittivity-path gradient correct."""
    jax.config.update("jax_enable_x64", True)
    from lumenairy.elements.rcwa import rcwa_efficiency_1d_jax

    def t0(nr):
        _, R, T = rcwa_efficiency_1d_jax(1.2e-6, nr, 1.0, 1.5, 1.0, 0.5e-6, 0.5,
                                         WL, angle=np.deg2rad(15),
                                         polarization="tm", n_orders=11,
                                         n_samples=512)
        return T[11]
    g = float(jax.grad(t0)(2.2))
    fd = float((t0(2.2 + 1e-6) - t0(2.2 - 1e-6)) / 2e-6)
    assert abs(g - fd) / abs(fd) < 1e-5  # eig-path gradient correct

    def t0d(depth):
        _, R, T = rcwa_efficiency_1d_jax(1.2e-6, 2.2, 1.0, 1.5, 1.0, depth, 0.5,
                                         WL, angle=np.deg2rad(15),
                                         polarization="tm", n_orders=11,
                                         n_samples=512)
        return T[11]
    gd = float(jax.grad(t0d)(0.5e-6))
    fdd = float((t0d(0.5e-6 + 1e-10) - t0d(0.5e-6 - 1e-10)) / 2e-10)
    assert abs(gd - fdd) / abs(fdd) < 1e-3
