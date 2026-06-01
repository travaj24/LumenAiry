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


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_p2a_layer_mode_grazing_pin(pol):
    """Regression pin for P2-A: a diffracted order grazing inside the *LAYER*
    (not just a half-space) made the interface S-matrix singular -> LinAlgError
    before the layer-index Wood nudge.  period = 2*wl, normal incidence, ridge
    n=2.0 -> order +/-4 has kx_4 = 4*wl/period = 2.0 = n_ridge, i.e. kz = 0
    INSIDE the ridge.  Must stay finite + energy-conserving."""
    o, R, T = rcwa_efficiency_1d(1.1e-6, 2.0, 1.0, 1.5, 1.0, 0.3e-6, 0.5,
                                 0.55e-6, angle=0.0, polarization=pol,
                                 n_orders=8)
    assert np.all(np.isfinite(R)) and np.all(np.isfinite(T))
    assert abs(R.sum() + T.sum() - 1.0) < 1e-6
    # anisotropic (Jones) layer path shares the nudge -- pin it too.
    er = uniaxial_tensor(2.0, 2.0, np.pi / 2)   # n=2.0 -> order +/-4 grazes
    eg = 1.0 * np.eye(3)
    o, R, T, J = rcwa_jones_1d(1.1e-6, er, eg, 1.5, 1.0, 0.3e-6, 0.5, 0.55e-6,
                               angle=0.0, n_orders=8)
    assert np.all(np.isfinite(R)) and np.all(np.isfinite(J))


@pytest.mark.parametrize("P,M", [(20e-6, 15), (15e-6, 15), (20e-6, 8)])
def test_large_period_energy_blowup_guarded(P, M):
    """v5.5.3 (audit/workflow cross-confirmed): a near-degenerate layer
    eigenproblem at certain large-period / low-contrast geometries silently
    returned R+T up to 1e30+ (erratic in n_orders).  The post-solve energy
    guard now turns that silent wrong answer into a clear error."""
    with pytest.raises(ValueError, match="energy non-conservation"):
        rcwa_efficiency_1d(P, 1.52, 1.5, 1.5, 1.5, 0.8e-6, 0.5, WL, n_orders=M)
    # a valid lossy (metal) solve with R+T < 1 must NOT be tripped
    o, R, T = rcwa_efficiency_1d(0.6e-6, 0.056 + 4.28j, 1.0, 1.5, 1.0, 0.12e-6,
                                 0.5, WL, n_orders=15)
    assert R.sum() + T.sum() < 1.0


@pytest.mark.parametrize("fn,kw", [
    ("rcwa_efficiency_1d", None),
    ("rcwa_efficiency_2d", None),
])
def test_p2b_non_propagating_incidence_raises(fn, kw):
    """Regression pin for P2-B: a non-propagating incidence half-space
    (Re(eps_superstrate) <= kx0^2+ky0^2 -- here a metallic n=0.5+2j with
    Re(eps) = -3.75) must raise a clear ValueError, not silently return
    negative / NaN efficiencies."""
    n_metal = 0.5 + 2j
    if fn == "rcwa_efficiency_1d":
        with pytest.raises(ValueError, match="non-propagating"):
            rcwa_efficiency_1d(1.0e-6, 2.2, 1.0, 1.5, n_metal, 0.3e-6, 0.5, WL,
                               angle=np.deg2rad(20), polarization="te",
                               n_orders=5)
    else:
        with pytest.raises(ValueError, match="non-propagating"):
            rcwa_efficiency_2d(0.5e-6, 0.5e-6, np.full((16, 16), 2.0, complex),
                               1.5, n_metal, 0.3e-6, WL, theta=np.deg2rad(20),
                               n_orders_x=3, n_orders_y=3)


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


def test_stack_homog_cache_substrate_key_collision_oblique():
    """ADVERSARIAL: the module-global homogeneous-mode cache must not return
    the wrong substrate modes when two stacks share n_substrate + geom but
    differ in n_superstrate at oblique incidence."""
    from lumenairy.elements import rcwa as _r

    S = 16  # >= 4*n_orders + 1 = 9 (patterned cell must clear the alias bound)
    xx, yy = np.meshgrid((np.arange(S) + .5) / S, (np.arange(S) + .5) / S,
                         indexing="ij")
    mask = (np.abs(xx - .5) < .25) & (np.abs(yy - .5) < .25)
    cell = np.where(mask, 2.3 ** 2, 1.5 ** 2).astype(complex)
    theta = np.deg2rad(30)

    def build(nsup, nsub):
        st = RCWAStack(0.8e-6, period_y=0.8e-6, n_superstrate=nsup,
                       n_substrate=nsub, n_orders=2, n_orders_y=2)
        st.add_layer(0.3e-6, eps_cell=cell)
        return st.set_source(WL, theta=theta).solve()

    # clean reference for B (n_sup=2.2), cache empty
    la.clear_asm_caches()
    o, Rc, Tc = build(2.2, 1.5).efficiencies()
    # dirty: A (n_sup=1.0) populates ('sub',1.5,geom); B (n_sup=2.2) shares it
    la.clear_asm_caches()
    _ = build(1.0, 1.5)
    o2, Rd, Td = build(2.2, 1.5).efficiencies()
    la.clear_asm_caches()
    # correct behavior: cache must NOT change B's result, energy stays <= 1
    # (regression pin for P1-B: substrate homog-mode cache key now folds in
    # n_superstrate via `geom`, so the A->B oblique collision can't occur).
    assert np.max(np.abs(Td - Tc)) < 1e-12
    assert np.max(np.abs(Rd - Rc)) < 1e-12  # Rd == Rc
    assert np.all(Rd.sum(axis=1) + Td.sum(axis=1) <= 1.0 + 1e-9)


def test_stack_apply_reflection_does_not_mutate_input():
    """ADVERSARIAL: apply_reflection must not mutate the incident JonesField."""
    from lumenairy.elements.polarization import JonesField

    elc = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=np.deg2rad(30))
    cell = np.broadcast_to(elc, (8, 8, 3, 3)).copy()
    st = RCWAStack(0.8e-6, period_y=0.8e-6, n_orders=2, n_orders_y=2)
    res = st.add_layer(0.3e-6, eps_tensor_cell=cell).set_source(WL).solve()
    jf = JonesField(np.ones((6, 6), complex), np.zeros((6, 6), complex), dx=1e-6)
    ex0, ey0 = jf.Ex.copy(), jf.Ey.copy()
    out = res.apply_reflection(jf)
    assert np.allclose(jf.Ex, ex0) and np.allclose(jf.Ey, ey0)  # input intact
    assert out is not jf                                         # new object


# =========================== input validation =============================
# v5.5.1: every entry point rejects non-physical geometry with a fn_name:
# prefix instead of the v5.5.0 silent-wrong-answer / cryptic-LinAlgError modes.


def test_validate_1d_geometry_rejects_bad_inputs():
    bad = [
        dict(period=0.0), dict(period=-1e-6), dict(depth=-0.3e-6),
        dict(depth=0.0), dict(n_orders=0), dict(n_orders=-3),
        dict(n_orders=5.5), dict(wavelength=0.0),
    ]
    base = dict(period=1e-6, n_ridge=2.2, n_groove=1.0, n_substrate=1.5,
                n_superstrate=1.0, depth=0.3e-6, duty_cycle=0.5, wavelength=WL)
    for override in bad:
        kw = dict(base, **override)
        with pytest.raises(ValueError, match="rcwa_efficiency_1d:"):
            rcwa_efficiency_1d(kw.pop("period"), kw.pop("n_ridge"),
                               kw.pop("n_groove"), kw.pop("n_substrate"),
                               kw.pop("n_superstrate"), kw.pop("depth"),
                               kw.pop("duty_cycle"), kw.pop("wavelength"), **kw)


def _patterned_cell(S):
    """A genuinely patterned SxS cell (centred square) -- has high-order
    Fourier content, so the aliasing bound applies (unlike a uniform cell)."""
    xx, yy = np.meshgrid((np.arange(S) + .5) / S, (np.arange(S) + .5) / S,
                         indexing="ij")
    mask = (np.abs(xx - .5) < .25) & (np.abs(yy - .5) < .25)
    return np.where(mask, 6.0, 2.0).astype(complex)


def test_validate_2d_aliasing_bound_rejects_undersampled_cell():
    # S = 8 < 4*n_orders + 1 = 13 -> would alias -> must raise, not silently
    # return a wrong spectrum (patterned cell, so the bound truly applies).
    with pytest.raises(ValueError, match="too coarse"):
        rcwa_efficiency_2d(0.8e-6, 0.8e-6, _patterned_cell(8), 1.5, 1.0,
                           0.3e-6, WL, n_orders_x=3, n_orders_y=3)
    # exactly 4*n_orders + 1 = 13 is accepted and conserves energy
    o, R, T = rcwa_efficiency_2d(0.8e-6, 0.8e-6, _patterned_cell(13), 1.5, 1.0,
                                 0.3e-6, WL, n_orders_x=3, n_orders_y=3)
    assert abs(R.sum() + T.sum() - 1.0) < 1e-9
    # a UNIFORM cell uses the relaxed 2*n_orders+1 floor: 8 >= 2*3+1 = 7 passes
    # (8 < 4*3+1 = 13 would reject a *patterned* cell of the same size).
    o, R, T = rcwa_efficiency_2d(0.8e-6, 0.8e-6, np.full((8, 8), 2.25, complex),
                                 1.5, 1.0, 0.3e-6, WL, n_orders_x=3,
                                 n_orders_y=3)
    assert abs(R.sum() + T.sum() - 1.0) < 1e-9
    # but a uniform cell below the 2*n_orders+1 DC-alias floor still raises
    with pytest.raises(ValueError, match="uniform cell"):
        rcwa_efficiency_2d(0.8e-6, 0.8e-6, np.full((4, 4), 2.25, complex),
                           1.5, 1.0, 0.3e-6, WL, n_orders_x=3, n_orders_y=3)


def test_validate_shapes_rejects_nonpositive_dims():
    sh = [{"shape": "disk", "eps": 6.0, "radius": 0.0,
           "center": (0.4e-6, 0.4e-6)}]
    with pytest.raises(ValueError, match="non-positive"):
        rcwa_efficiency_2d_shapes(0.8e-6, 0.8e-6, 1.0, sh, 1.5, 1.0, 0.3e-6, WL,
                                  n_orders_x=3, n_orders_y=3)


def test_validate_shapes_rejects_over_cell():
    # An over-cell disk (radius > period / sqrt(pi), i.e. area fraction > 1)
    # drives the DC permittivity past the shape's own eps -- physically
    # impossible.  P / sqrt(pi) ~= 0.451e-6 for P = 0.8e-6, so r = 0.6e-6 gives
    # area fraction pi*r^2/P^2 ~= 1.77 > 1.  Must raise, not silently model a
    # non-physical structure (and the disk would conserve energy if it ran).
    big_disk = [{"shape": "disk", "eps": 6.0, "radius": 0.6e-6,
                 "center": (0.4e-6, 0.4e-6)}]
    with pytest.raises(ValueError, match="area fraction"):
        rcwa_efficiency_2d_shapes(0.8e-6, 0.8e-6, 1.0, big_disk, 1.5, 1.0,
                                  0.3e-6, WL, n_orders_x=3, n_orders_y=3)
    # A shape with area fraction <= 1 can still wrap and self-overlap: a
    # 1.2e-6 x 0.4e-6 rectangle in a 0.8e-6 cell is fraction 0.75 but its
    # x-extent 1.2e-6 > 0.8e-6.  The bounding-extent guard catches it.
    wrap_rect = [{"shape": "rectangle", "eps": 6.0, "size": (1.2e-6, 0.4e-6),
                  "center": (0.4e-6, 0.4e-6)}]
    with pytest.raises(ValueError, match="extent"):
        rcwa_efficiency_2d_shapes(0.8e-6, 0.8e-6, 1.0, wrap_rect, 1.5, 1.0,
                                  0.3e-6, WL, n_orders_x=3, n_orders_y=3)
    # The same guard fires on the RCWAStack.add_layer path (uses the stack's
    # own period), with the fn_name prefix.
    with pytest.raises(ValueError, match="add_layer:.*area fraction"):
        RCWAStack(0.8e-6, period_y=0.8e-6, n_orders=3, n_orders_y=3).add_layer(
            0.3e-6, shapes=big_disk, eps_background=1.0)
    # A shape that exactly tiles / is inscribed must still be accepted: a disk
    # of radius P/2 (inscribed, extent == period, fraction pi/4) does not raise.
    fit_disk = [{"shape": "disk", "eps": 6.0, "radius": 0.4e-6,
                 "center": (0.4e-6, 0.4e-6)}]
    o, R, T = rcwa_efficiency_2d_shapes(0.8e-6, 0.8e-6, 1.0, fit_disk, 1.5, 1.0,
                                        0.3e-6, WL, n_orders_x=3, n_orders_y=3)
    assert abs(R.sum() + T.sum() - 1.0) < 1e-9


def test_validate_shapes_rejects_cumulative_over_cell():
    """v5.5.3 (audit P2): the over-cell check must be CUMULATIVE -- two
    disjoint disks at area fraction 0.6 each (total 1.2) drive the DC
    permittivity past the shapes' eps just as a single oversized shape would,
    but each passes the per-shape check.  The v5.5.2 per-shape-only check let
    them slip through."""
    r = float(np.sqrt(0.6 * (0.8e-6) ** 2 / np.pi))   # fraction 0.6 each
    two = [{"shape": "disk", "eps": 6.0, "radius": r, "center": (0.2e-6, 0.4e-6)},
           {"shape": "disk", "eps": 6.0, "radius": r, "center": (0.6e-6, 0.4e-6)}]
    with pytest.raises(ValueError, match="CUMULATIVE area fraction"):
        rcwa_efficiency_2d_shapes(0.8e-6, 0.8e-6, 1.0, two, 1.5, 1.0, 0.3e-6,
                                  WL, n_orders_x=4, n_orders_y=4)
    with pytest.raises(ValueError, match="add_layer:.*CUMULATIVE"):
        RCWAStack(0.8e-6, period_y=0.8e-6, n_orders=4, n_orders_y=4).add_layer(
            0.3e-6, shapes=two, eps_background=1.0)
    # two small disjoint disks (total fraction << 1) still solve fine
    small = [{"shape": "disk", "eps": 6.0, "radius": 0.15e-6,
              "center": (0.2e-6, 0.4e-6)},
             {"shape": "disk", "eps": 6.0, "radius": 0.15e-6,
              "center": (0.6e-6, 0.4e-6)}]
    o, R, T = rcwa_efficiency_2d_shapes(0.8e-6, 0.8e-6, 1.0, small, 1.5, 1.0,
                                        0.3e-6, WL, n_orders_x=4, n_orders_y=4)
    assert abs(R.sum() + T.sum() - 1.0) < 1e-9


def test_validate_stack_geometry_and_dead_param():
    with pytest.raises(ValueError, match="RCWAStack: period"):
        RCWAStack(0.0)
    with pytest.raises(ValueError, match="add_layer: depth"):
        RCWAStack(0.8e-6, period_y=0.8e-6, n_orders=2,
                  n_orders_y=2).add_layer(-1e-6, eps=2.0)
    with pytest.raises(ValueError, match="set_source: wavelength"):
        RCWAStack(0.8e-6, period_y=0.8e-6, n_orders=2,
                  n_orders_y=2).set_source(-WL)
    with pytest.raises(ValueError, match="too coarse"):
        RCWAStack(0.8e-6, period_y=0.8e-6, n_orders=3,
                  n_orders_y=3).add_layer(0.3e-6, eps_cell=_patterned_cell(8))
    # the inert v5.5.0 set_source(polarization=...) param is gone
    with pytest.raises(TypeError):
        RCWAStack(0.8e-6, period_y=0.8e-6, n_orders=2,
                  n_orders_y=2).set_source(WL, polarization="tm")


def test_n_orders_ceiling_guards_oom():
    """v5.5.2: a fat-finger n_orders that would build an unsolvable 2N x 2N
    eig must raise (not OOM-hang).  1-D harmonic count and the 2-D product
    are both bounded."""
    with pytest.raises(ValueError, match="harmonic count"):
        rcwa_efficiency_1d(1e-6, 2.0, 1.0, 1.5, 1.0, 0.4e-6, 0.5, WL,
                           n_orders=10 ** 9)
    with pytest.raises(ValueError, match="harmonic count"):
        rcwa_efficiency_2d(0.8e-6, 0.8e-6, np.full((16, 16), 2.0, complex),
                           1.5, 1.0, 0.3e-6, WL, n_orders_x=100, n_orders_y=100)


def test_vs_wavelength_validation():
    """v5.5.2: rcwa_efficiency_vs_wavelength reports geometry errors with ITS
    OWN prefix (not the inner per-call prefix) and rejects an empty / non-
    positive wavelength list instead of silently returning empty."""
    base = (1.4e-6, 2.0, 1.0, 1.5, 1.0, 0.5e-6, 0.5)
    with pytest.raises(ValueError, match="rcwa_efficiency_vs_wavelength: peri"):
        rcwa_efficiency_vs_wavelength(-1.0, *base[1:], [0.6e-6], order=1)
    with pytest.raises(ValueError, match="wavelengths is empty"):
        rcwa_efficiency_vs_wavelength(*base, np.array([]), order=1)
    with pytest.raises(ValueError, match="every wavelength must be"):
        rcwa_efficiency_vs_wavelength(*base, [0.6e-6, -1.0], order=1)


# ===================== convergence / physics oracles ======================
# Committed value pins that auto-catch a future regression in the delicate
# paths (the v5.5.0 audit's P1-A oblique-TM claim was a convergence artifact,
# triangulated to lumen's converged Rtot ~= 0.5317; this pins that value).


def test_oblique_tm_converged_reflectance_pin():
    """Oblique high-contrast TM total reflectance, pinned to the value
    triangulated against grcwa (Laurent, converging down) and inkstone
    (converging up) -- both bracket ~0.5317.  Locks the Li inverse-rule TM
    path against silent regression (and documents that the v5.5.0 'P1-A'
    split was an under-converged-oracle artifact, not a solver bug)."""
    o, R, T = rcwa_efficiency_1d(1.2e-6, 2.5, 1.0, 1.0, 1.0, 0.4e-6, 0.4,
                                 0.55e-6, angle=np.deg2rad(15),
                                 polarization="tm", n_orders=60)
    assert abs(float(R.sum()) - 0.5317) < 5e-3          # converged value
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-9   # lossless: energy = 1


def test_conical_2d_jones_energy_and_crosspol():
    """Conical-incidence 2-D anisotropic Jones solve conserves energy for both
    incident polarizations AND produces real cross-polarization coupling
    (a guard on the anisotropic Q-block Cyx sign: getting it wrong silently
    breaks energy at off-axis director angles)."""
    elc = uniaxial_tensor(1.5, 1.9, np.pi / 2, phi=np.deg2rad(35))
    cell = np.broadcast_to(elc, (16, 16, 3, 3)).copy()
    o, R, T, J = rcwa_jones_2d(0.7e-6, 0.7e-6, cell, 1.0, 1.0, 0.5e-6, WL,
                               theta=np.deg2rad(12), phi=np.deg2rad(25),
                               n_orders_x=4, n_orders_y=4)
    assert abs(float(R[0].sum() + T[0].sum()) - 1.0) < 1e-6
    assert abs(float(R[1].sum() + T[1].sum()) - 1.0) < 1e-6
    # off-diagonal Jones (cross-pol) must be non-trivial for a rotated director
    assert float(np.abs(J[0, 1])) > 1e-3 and float(np.abs(J[1, 0])) > 1e-3


def test_numpy_value_regression_pins():
    """Committed value pins for the NumPy path -- any future backend / helper
    refactor that perturbs the physics is caught here (these are the exact
    pre-keystone v5.5.0 values)."""
    o, R, T = rcwa_efficiency_1d(1.2e-6, 2.5, 1.0, 1.5, 1.0, 0.4e-6, 0.4,
                                 0.55e-6, angle=np.deg2rad(15),
                                 polarization="tm", n_orders=40)
    assert abs(float(R.sum()) - 0.1905690334313482) < 1e-12
    assert abs(float(T.sum()) - 0.809430966568596) < 1e-12
    o, R, T = rcwa_efficiency_1d(1.2e-6, 2.2, 1.0, 1.5, 1.0, 0.5e-6, 0.5, WL,
                                 polarization="te", n_orders=11)
    assert abs(float(T[len(T) // 2]) - 0.6835550591286147) < 1e-12


# ====================== backend dispatch (v5.5.1) =========================
# The solvers became backend-dispatched (NumPy / CuPy via use_gpu / JAX).
# CuPy GPU execution can't run in CI without CUDA libs; these pins lock the
# NumPy results, the s/p aliases, and the use_gpu routing contract.


@pytest.mark.parametrize("alias,canon", [("s", "te"), ("p", "tm"),
                                         ("S", "te"), ("P", "tm")])
def test_polarization_s_p_aliases(alias, canon):
    """'s'/'p' (coatings convention) map exactly to 'te'/'tm' (RCWA)."""
    args = (1.2e-6, 2.5, 1.0, 1.5, 1.0, 0.4e-6, 0.4, 0.55e-6)
    o, Ra, Ta = rcwa_efficiency_1d(*args, angle=np.deg2rad(12),
                                   polarization=alias, n_orders=20)
    o, Rc, Tc = rcwa_efficiency_1d(*args, angle=np.deg2rad(12),
                                   polarization=canon, n_orders=20)
    assert np.array_equal(Ra, Rc) and np.array_equal(Ta, Tc)


def test_use_gpu_without_cupy_raises_clear_error(monkeypatch):
    """use_gpu=True must raise a clear, actionable error when CuPy is absent
    (rather than a cryptic NumPy/AttributeError)."""
    import lumenairy.elements.rcwa as _r
    monkeypatch.setattr(_r, "CUPY_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="use_gpu=True but CuPy is not"):
        rcwa_efficiency_1d(1.2e-6, 2.2, 1.0, 1.5, 1.0, 0.5e-6, 0.5, WL,
                           n_orders=11, use_gpu=True)


def test_block_helper_matches_numpy_block():
    """The portable concatenate-based _block reproduces numpy.block exactly
    (some CuPy builds lack cupy.block)."""
    from lumenairy.elements.rcwa import _block
    rng = np.random.default_rng(0)
    A, B, C, D = (rng.standard_normal((3, 3)) for _ in range(4))
    assert np.array_equal(_block(np, [[A, B], [C, D]]),
                          np.block([[A, B], [C, D]]))


def test_set_blas_threads_numerically_equivalent():
    """v5.5.2: the opt-in BLAS thread cap is numerically equivalent to the
    default (it changes results only at the ~1e-14 floating-point-
    reassociation level from the thread-count change, not physically); the
    context manager restores the prior setting."""
    import lumenairy.elements.rcwa as _r
    from lumenairy.elements.rcwa import rcwa_blas_threads
    args = (1.2e-6, 2.5, 1.0, 1.5, 1.0, 0.4e-6, 0.4, 0.55e-6)
    o, Rd, Td = rcwa_efficiency_1d(*args, angle=np.deg2rad(15),
                                   polarization="tm", n_orders=40)
    with rcwa_blas_threads(2):
        assert _r._get_blas_threads() == 2
        o, Rc, Tc = rcwa_efficiency_1d(*args, angle=np.deg2rad(15),
                                       polarization="tm", n_orders=40)
    assert _r._get_blas_threads() is None    # restored on context exit
    assert float(np.max(np.abs(Rc - Rd))) < 1e-10
    assert abs(float(Rc.sum() + Tc.sum()) - 1.0) < 1e-9


def test_blas_threads_thread_local_isolation():
    """v5.5.3 (audit P3): the BLAS cap is thread-local, so a cap set in one
    thread does not leak into another running concurrently."""
    import threading

    import lumenairy.elements.rcwa as _r
    from lumenairy.elements.rcwa import rcwa_blas_threads, set_blas_threads
    seen = {}
    barrier = threading.Barrier(2)

    def worker(name, n):
        with rcwa_blas_threads(n):
            barrier.wait()                    # both inside their cap at once
            seen[name] = _r._get_blas_threads()
        seen[name + "_after"] = _r._get_blas_threads()

    set_blas_threads(None)
    t1 = threading.Thread(target=worker, args=("a", 2))
    t2 = threading.Thread(target=worker, args=("b", 4))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert seen["a"] == 2 and seen["b"] == 4          # no cross-thread leak
    assert seen["a_after"] is None and seen["b_after"] is None
    assert _r._get_blas_threads() is None             # main thread untouched


def test_to_jones_field_specular_bridge():
    """RCWAResult.to_jones_field builds a uniform specular JonesField whose
    value is the zeroth-order Jones applied to the incident vector."""
    from lumenairy.elements.polarization import JonesField
    elc = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=np.deg2rad(30))
    cell = np.broadcast_to(elc, (16, 16, 3, 3)).copy()
    res = (RCWAStack(0.8e-6, period_y=0.8e-6, n_orders=3, n_orders_y=3)
           .add_layer(0.3e-6, eps_tensor_cell=cell)
           .set_source(WL).solve())
    jf = res.to_jones_field(8, 6, dx=1e-6, incident=(1.0, 0.0))
    assert isinstance(jf, JonesField) and jf.Ex.shape == (6, 8)
    Jr = res.jones_reflection()
    assert np.allclose(jf.Ex, Jr[0, 0]) and np.allclose(jf.Ey, Jr[1, 0])
    jt = res.to_jones_field(4, 4, dx=1e-6, incident=(0.0, 1.0),
                            port="transmission")
    assert np.allclose(jt.Ex, res.jones_transmission()[0, 1])
    with pytest.raises(ValueError, match="port must be"):
        res.to_jones_field(4, 4, dx=1e-6, port="bogus")


def _diffracting_grating_result():
    """A high-contrast 1-D grating (most reflected power in non-zero orders)."""
    S = 64
    x = (np.arange(S) + 0.5) / S
    cell = np.where(x < 0.5, 2.5 ** 2, 1.0).astype(complex)
    return (RCWAStack(2.0e-6, n_superstrate=1.0, n_substrate=1.0, n_orders=12)
            .add_layer(0.4e-6, eps_cell=cell)
            .set_source(WL, theta=np.deg2rad(8)).solve())


def test_rcwa_reduces_to_thin_grating_limit():
    """v5.5.2 cross-solver pin: in the thin-grating regime (period >> lambda,
    low index contrast, shallow, normal incidence, index-matched half-spaces
    so reflection -> 0) the rigorous RCWA transmitted-order efficiencies must
    agree with the analytic scalar thin_grating model -- the documented
    upgrade path from thin_grating to RCWA."""
    from lumenairy.elements.thin_grating import thin_grating_efficiency_1d
    cfg = dict(period=10e-6, n_ridge=1.55, n_groove=1.5, n_substrate=1.5,
               n_superstrate=1.5, depth=0.5e-6, duty_cycle=0.5, wavelength=WL,
               angle=0.0, n_orders=15)
    o, R, T = rcwa_efficiency_1d(cfg["period"], cfg["n_ridge"], cfg["n_groove"],
                                 cfg["n_substrate"], cfg["n_superstrate"],
                                 cfg["depth"], cfg["duty_cycle"],
                                 cfg["wavelength"], angle=cfg["angle"],
                                 polarization="te", n_orders=cfg["n_orders"])
    ot, Rt, Tt = thin_grating_efficiency_1d(**cfg, polarization="te")
    assert R.sum() < 2e-3                          # thin regime: R ~ 0
    mid = len(o) // 2
    for m in (-2, -1, 0, 1, 2):
        assert abs(T[mid + m] - Tt[mid + m]) < 2e-3
    # thin_grating now also validates polarization (s/p aliases; rejects typos)
    with pytest.raises(ValueError, match="polarization must be"):
        thin_grating_efficiency_1d(**cfg, polarization="circular")


def test_rcwa_system_element():
    """v5.5.2: an 'rcwa' element in propagate_through_system applies the
    rigorous specular scalar amplitude to a scalar field (explicit amplitude
    or from an RCWAResult's co-pol Jones), and composes with other stages."""
    from lumenairy.propagators.system import propagate_through_system
    E0 = np.ones((32, 32), complex)

    def _field(out):
        return out[0] if isinstance(out, tuple) else out

    # explicit complex amplitude -> uniform multiplier
    E = _field(propagate_through_system(
        E0, [{"type": "rcwa", "amplitude": 0.5 + 0.2j}], WL, dx=2e-6,
        verbose=False))
    assert np.allclose(E, 0.5 + 0.2j)

    # from an RCWAResult (uniform slab -> co-pol == Jt[0,0])
    res = (RCWAStack(0.5e-6, period_y=0.5e-6, n_substrate=1.5, n_orders=3,
                     n_orders_y=3)
           .add_layer(0.3e-6, eps=2.1 ** 2).set_source(WL).solve())
    E = _field(propagate_through_system(
        E0, [{"type": "rcwa", "result": res, "port": "transmission",
              "incident": (1, 0)}], WL, dx=2e-6, verbose=False))
    assert np.allclose(E[0, 0], res.jones_transmission()[0, 0])

    # composes with a propagation stage; missing spec raises clearly
    E = _field(propagate_through_system(
        E0, [{"type": "rcwa", "amplitude": 0.8}, {"type": "propagate",
              "z": 5e-3}], WL, dx=2e-6, verbose=False))
    assert np.all(np.isfinite(E))
    with pytest.raises(ValueError, match="rcwa.*needs either|needs either"):
        propagate_through_system(E0, [{"type": "rcwa"}], WL, dx=2e-6,
                                 verbose=False)


def test_multiorder_bridge():
    """v5.5.2: RCWAResult exposes per-order amplitudes + a multi-order field
    bridge -- the data the solver used to discard.  A strongly diffracting
    grating puts most power in non-zero orders, which to_multiorder_field
    reconstructs as a superposition of tilted plane-wave carriers."""
    res = _diffracting_grating_result()
    o, R, T = res.efficiencies()
    p0 = int(np.where(o == 0)[0][0])
    # most reflected power is NOT in the specular order (the whole point)
    assert R[0, p0] / R[0].sum() < 0.5

    pa = res.per_order_amplitudes("reflection")
    assert pa["Ex"].shape == (2, o.shape[0]) and pa["kx"].shape == o.shape
    assert pa["wavelength"] == WL

    dx = 2.0e-6 / 64
    # specular order=None is a uniform field (backward-compatible)
    jf0 = res.to_jones_field(48, 48, dx=dx, order=None)
    assert np.allclose(jf0.Ex, jf0.Ex.flat[0])
    # a non-zero order is a tilted carrier (uniform magnitude, varying phase)
    jf1 = res.to_jones_field(48, 48, dx=dx, order=1)
    assert np.allclose(np.abs(jf1.Ex), np.abs(jf1.Ex).flat[0])
    assert not np.allclose(jf1.Ex, jf1.Ex.flat[0])      # phase actually tilts
    # a single-order multi-order field equals to_jones_field of that order
    mf1 = res.to_multiorder_field(48, 48, dx=dx, orders=[1])
    assert np.allclose(mf1.Ex, jf1.Ex) and np.allclose(mf1.Ey, jf1.Ey)
    # the full reconstruction is finite and superposes the propagating orders
    mf = res.to_multiorder_field(48, 48, dx=dx)
    assert np.all(np.isfinite(mf.Ex)) and np.max(np.abs(mf.Ex)) > 0
    # out-of-range order / bad port raise clearly
    with pytest.raises(ValueError, match="outside the retained range"):
        res.to_jones_field(8, 8, dx=dx, order=99)
    with pytest.raises(ValueError, match="port must be"):
        res.to_multiorder_field(8, 8, dx=dx, port="bogus")


@pytest.mark.parametrize("port", ["reflection", "transmission"])
@pytest.mark.parametrize("theta", [0.0, 10.0])
def test_multiorder_field_conserves_energy(port, theta):
    """v5.5.3 (audit P1): the power-normalized multi-order field must carry the
    correct per-order power -- on a one-cell grid Parseval gives
    mean|field|^2 == sum of the propagating-order EFFICIENCIES.  The v5.5.2
    field dropped the Poynting flux weight + the longitudinal component and
    was off by -39% to +148%; this pins the fix."""
    S = 64
    x = (np.arange(S) + 0.5) / S
    cell = np.where(x < 0.5, 2.5 ** 2, 1.0).astype(complex)
    res = (RCWAStack(2.0e-6, n_superstrate=1.0, n_substrate=1.5, n_orders=12)
           .add_layer(0.4e-6, eps_cell=cell)
           .set_source(WL, theta=np.deg2rad(theta)).solve())
    o, R, T = res.efficiencies()
    pa = res.per_order_amplitudes(port)
    prop = np.real(pa["kz"]) > 1e-12
    eff_total = float((R if port == "reflection" else T)[0][prop].sum())
    # one-cell grid -> the order carriers are orthonormal (Parseval exact)
    period = res._modal["period_x"]
    nx = 60
    jf = res.to_multiorder_field(nx, nx, dx=period / nx, incident=(1.0, 0.0),
                                 port=port, normalize="power")
    field_power = float(np.mean(np.abs(jf.Ex) ** 2 + np.abs(jf.Ey) ** 2))
    assert abs(field_power - eff_total) < 1e-6
    # the raw 'field' mode is NOT power-calibrated (documents the distinction)
    jf_raw = res.to_multiorder_field(nx, nx, dx=period / nx, port=port,
                                     normalize="field")
    raw_power = float(np.mean(np.abs(jf_raw.Ex) ** 2 + np.abs(jf_raw.Ey) ** 2))
    assert abs(raw_power - eff_total) > 1e-4 or theta == 0.0  # generally differs


# ============================== JAX autodiff ===============================

jax = pytest.importorskip("jax")


def _square_cell(S):
    xx, yy = np.meshgrid((np.arange(S) + .5) / S - .5,
                         (np.arange(S) + .5) / S - .5, indexing="ij")
    return np.where(xx ** 2 + yy ** 2 < 0.0625, 6.0, 2.0).astype(complex)


def test_jax_execution_parity_all_entry_points():
    """v5.5.1 execution-parity contract: every backend-dispatched solver
    returns bit-for-eig-precision identical results on NumPy and JAX (the
    folded core runs the SAME algorithm on both backends)."""
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    cell = _square_cell(16)
    elc = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=np.deg2rad(30))
    tcell = np.broadcast_to(elc, (16, 16, 3, 3)).copy()

    # 1-D efficiency
    a = (1.2e-6, 2.2, 1.0, 1.5, 1.0, 0.5e-6, 0.5, WL)
    on, Rn, Tn = rcwa_efficiency_1d(*a, angle=np.deg2rad(15),
                                    polarization="tm", n_orders=11)
    oj, Rj, Tj = rcwa_efficiency_1d(
        1.2e-6, jnp.asarray(2.2 + 0j), jnp.asarray(1.0 + 0j),
        jnp.asarray(1.5 + 0j), jnp.asarray(1.0 + 0j), jnp.asarray(0.5e-6),
        0.5, jnp.asarray(WL), angle=np.deg2rad(15), polarization="tm",
        n_orders=11)
    assert float(np.max(np.abs(np.asarray(Rj) - Rn))) < 1e-10
    assert float(np.max(np.abs(np.asarray(Tj) - Tn))) < 1e-10

    # 2-D efficiency
    on, Rn, Tn = rcwa_efficiency_2d(0.8e-6, 0.8e-6, cell, 1.5, 1.0, 0.3e-6, WL,
                                    theta=np.deg2rad(10), n_orders_x=3,
                                    n_orders_y=3)
    oj, Rj, Tj = rcwa_efficiency_2d(0.8e-6, 0.8e-6, jnp.asarray(cell), 1.5, 1.0,
                                    0.3e-6, WL, theta=np.deg2rad(10),
                                    n_orders_x=3, n_orders_y=3)
    assert float(np.max(np.abs(np.asarray(Rj) - Rn))) < 1e-10

    # 1-D Jones (anisotropic)
    er = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=np.deg2rad(20))
    eg = (1.5 ** 2) * np.eye(3)
    on, Rn, Tn, Jn = rcwa_jones_1d(1.0e-6, er, eg, 1.5, 1.0, 0.4e-6, 0.5, WL,
                                   angle=np.deg2rad(10), n_orders=15)
    oj, Rj, Tj, Jj = rcwa_jones_1d(1.0e-6, jnp.asarray(er), jnp.asarray(eg),
                                   1.5, 1.0, 0.4e-6, 0.5, WL,
                                   angle=np.deg2rad(10), n_orders=15)
    assert float(np.max(np.abs(np.asarray(Jj) - Jn))) < 1e-10

    # 2-D Jones (tensor field)
    on, Rn, Tn, Jn = rcwa_jones_2d(0.8e-6, 0.8e-6, tcell, 1.5, 1.0, 0.3e-6, WL,
                                   theta=np.deg2rad(8), n_orders_x=3,
                                   n_orders_y=3)
    oj, Rj, Tj, Jj = rcwa_jones_2d(0.8e-6, 0.8e-6, jnp.asarray(tcell), 1.5, 1.0,
                                   0.3e-6, WL, theta=np.deg2rad(8),
                                   n_orders_x=3, n_orders_y=3)
    assert float(np.max(np.abs(np.asarray(Jj) - Jn))) < 1e-10


def test_jax_uniform_layer_energy_conserved():
    """ADVERSARIAL (v5.5.1): a laterally-UNIFORM isotropic layer is doubly
    degenerate; its eig basis is gauge-arbitrary and ill-conditioned, which
    silently broke energy on the JAX path at oblique incidence (R+T=2.2).  The
    analytic uniform-mode branch is now reachable on JAX (tracer-safe where),
    so a uniform iso cell AND a uniform isotropic tensor conserve energy."""
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    kw = dict(period_x=1e-6, period_y=1e-6, n_substrate=1.5, n_superstrate=1.0,
              depth=0.3e-6, wavelength=WL, n_orders_x=5, n_orders_y=5)
    for theta in (0.0, 0.2):  # the failure was oblique-specific
        o, R, T = rcwa_efficiency_2d(eps_cell=jnp.full((21, 21), 2.25 + 0j),
                                     theta=theta, phi=0.0, polarization="te",
                                     **kw)
        assert abs(float(np.asarray(R).sum() + np.asarray(T).sum()) - 1.0) < 1e-6
    iso_t = (2.25 + 0j) * np.eye(3)
    tcell = jnp.asarray(np.broadcast_to(iso_t, (21, 21, 3, 3)).copy())
    o, R, T, J = rcwa_jones_2d(1e-6, 1e-6, tcell, 1.5, 1.0, 0.3e-6, WL,
                               theta=0.2, phi=0.0, n_orders_x=5, n_orders_y=5)
    assert abs(float(R[0].sum() + T[0].sum()) - 1.0) < 1e-6
    assert abs(float(R[1].sum() + T[1].sum()) - 1.0) < 1e-6


def test_jax_wood_anomaly_no_nan():
    """ADVERSARIAL (v5.5.1): at an EXACT Rayleigh/Wood anomaly a diffracted
    order has k_z = 0 in a region, making the interface S-matrix singular ->
    NaN.  The NumPy path nudges the wavelength off the anomaly; the JAX path
    used to skip that nudge and return NaN (poisoning jax.grad).  The guards
    now run against the concrete geometry on JAX too, so forward value AND
    gradient stay finite at the anomaly."""
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    # period = wavelength, normal incidence -> m = +/-1 graze in the n=1 regions
    a = dict(angle=jnp.asarray(0.0), polarization="te", n_orders=5)
    o, R, T = rcwa_efficiency_1d(1e-6, jnp.asarray(1.5 + 0j), jnp.asarray(1.0 + 0j),
                                 jnp.asarray(1.0 + 0j), jnp.asarray(1.0 + 0j),
                                 jnp.asarray(0.5e-6), 0.5, jnp.asarray(1e-6), **a)
    assert not bool(np.isnan(np.asarray(R)).any())
    assert abs(float(np.asarray(R).sum() + np.asarray(T).sum()) - 1.0) < 1e-6

    def loss(nr):
        _, R, T = rcwa_efficiency_1d(1e-6, nr, jnp.asarray(1.0 + 0j),
                                     jnp.asarray(1.0 + 0j), jnp.asarray(1.0 + 0j),
                                     jnp.asarray(0.5e-6), 0.5, jnp.asarray(1e-6),
                                     **a)
        return T.sum()
    g = jax.grad(loss)(jnp.asarray(1.5 + 0j))
    assert not bool(np.isnan(np.asarray(g)))


def test_jax_2d_gradient_matches_finite_difference():
    """The 2-D solver is differentiable through the eig path (cell scaling)."""
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    cell = _square_cell(16)

    def loss(scale):
        o, R, T = rcwa_efficiency_2d(0.8e-6, 0.8e-6, jnp.asarray(cell) * scale,
                                     1.5, 1.0, 0.3e-6, WL, theta=np.deg2rad(10),
                                     n_orders_x=3, n_orders_y=3)
        return T[len(T) // 2]
    g = float(jax.grad(loss)(1.0))
    h = 1e-6
    fd = (float(loss(1.0 + h)) - float(loss(1.0 - h))) / (2 * h)
    assert abs(g - fd) / max(abs(fd), 1e-30) < 1e-5


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_jax_validation_and_error_prefixes():
    """v5.5.1: the folded JAX twin delegates to the unified solver, so input
    validation (no longer silently dropped) is enforced there.  ``n_samples``
    is now accepted-but-ignored (exact analytic coefficients need no
    sampling)."""
    jax.config.update("jax_enable_x64", True)
    from lumenairy.elements.rcwa import rcwa_efficiency_1d_jax
    base = dict(period=1.2e-6, n_ridge=2.2, n_groove=1.0, n_substrate=1.5,
                n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5, wavelength=WL)

    def call(**over):
        kw = dict(base, **over)
        return rcwa_efficiency_1d_jax(
            kw["period"], kw["n_ridge"], kw["n_groove"], kw["n_substrate"],
            kw["n_superstrate"], kw["depth"], kw["duty_cycle"], kw["wavelength"],
            **{k: v for k, v in over.items() if k not in base})

    with pytest.raises(ValueError, match="rcwa_efficiency_1d: polarization"):
        call(polarization="circular")
    with pytest.raises(ValueError, match="rcwa_efficiency_1d: formulation"):
        call(formulation="bogus")
    with pytest.raises(ValueError, match="rcwa_efficiency_1d: duty_cycle"):
        call(duty_cycle=1.7)
    with pytest.raises(ValueError, match="rcwa_efficiency_1d: depth"):
        call(depth=-1e-6)
    # n_samples is accepted (back-compat) but no longer triggers validation
    o, R, T = call(n_orders=8, n_samples=5)
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-6


def test_rcwa_efficiency_1d_jax_deprecation_warning():
    """v5.5.2: the folded alias emits a DeprecationWarning steering callers to
    the unified rcwa_efficiency_1d (with jax.numpy inputs)."""
    jax.config.update("jax_enable_x64", True)
    from lumenairy.elements.rcwa import rcwa_efficiency_1d_jax
    with pytest.warns(DeprecationWarning, match="rcwa_efficiency_1d_jax"):
        rcwa_efficiency_1d_jax(1.2e-6, 2.2, 1.0, 1.5, 1.0, 0.5e-6, 0.5, WL,
                               n_orders=8)


def test_jax_plus_use_gpu_rejected():
    """v5.5.2: a JAX input combined with use_gpu (or a CuPy input) is an
    ambiguous mixed-backend call and must raise, not silently pick JAX."""
    import jax.numpy as jnp
    with pytest.raises(ValueError, match="cannot be combined with use_gpu"):
        rcwa_efficiency_1d(1.2e-6, jnp.asarray(2.2 + 0j), 1.0, 1.5, 1.0,
                           0.5e-6, 0.5, WL, n_orders=5, use_gpu=True)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_jax_value_matches_numpy():
    """v5.5.1: the folded JAX path uses the SAME exact binary-grating Fourier
    coefficients as the NumPy core (not the old soft-edge sample), so the two
    agree to eig precision (was ~5e-3) and energy is conserved."""
    jax.config.update("jax_enable_x64", True)
    from lumenairy.elements.rcwa import rcwa_efficiency_1d_jax
    o, R, T = rcwa_efficiency_1d(1.2e-6, 2.2, 1.0, 1.5, 1.0, 0.5e-6, 0.5, WL,
                                 angle=np.deg2rad(15), polarization="tm",
                                 n_orders=11)
    oj, Rj, Tj = rcwa_efficiency_1d_jax(1.2e-6, 2.2, 1.0, 1.5, 1.0, 0.5e-6, 0.5,
                                        WL, angle=np.deg2rad(15),
                                        polarization="tm", n_orders=11)
    assert float(np.max(np.abs(np.asarray(Rj) - R))) < 1e-10
    assert float(np.max(np.abs(np.asarray(Tj) - T))) < 1e-10
    assert abs(float(Rj.sum() + Tj.sum()) - 1.0) < 1e-9


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
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
