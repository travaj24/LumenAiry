"""PMM internal-field parity with RCWA (PMMStack v2 + PMM2DStack new).

``PMMStack.internal_field`` is upgraded to the RCWA interface (z arrays, all
SIX components, ``incident=`` Jones, Bloch carrier, optional uniform-grid
resampling) and ``PMM2DStack.internal_field`` is the new crossed-grating
mirror (same grid conventions/carriers, so the two co-register pointwise).

Conventions were pinned by ORACLES, not derivation:
* a uniform absorbing film must match ``RCWAResult.internal_field``
  pointwise in all six components (normal, oblique, conical, complex
  incident) -- this pinned the PMM modal-H sign (``+i u``, OPPOSITE to
  RCWA's ``-i u``: the PMM tensor ``Q`` block carries the other sign) and
  caught a real RCWA bug (below);
* a patterned 1-D grating: RCWA's near field CONVERGES TOWARD the
  nodal-exact PMM as n_orders rises (3.4e-2 @ 30 -> 9e-3 @ 120 interior);
* the 2-D patterned field converges to the nodal-exact 1-D PMM on a
  y-uniform stripe as (n_orders, degree) rise together;
* the volume-integral absorption identity ``k0 Im(eps)|E|^2`` closes
  against the flux-based ``layer_absorption``.

RCWA BUG found by the co-registration oracle and fixed here:
``RCWAResult.internal_field`` built the internal-gauge ``cinc`` directly
from the PUBLIC incident Jones and conjugated the output -- so a COMPLEX
(circular/elliptical) incident returned the field of the CONJUGATED
incident, i.e. the opposite handedness.  Real incidents were unaffected.
The public-linearity property pinned below is the regression guard.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.pmm import PMM2DStack, PMMStack
from lumenairy.elements.rcwa import RCWAStack

P, WL = 0.8e-6, 0.633e-6
COMPS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


# =========================================================================== #
# 1-D PMMStack vs RCWA
# =========================================================================== #

@pytest.mark.parametrize("ang", [0.0, 0.3])
@pytest.mark.parametrize("inc", [(1.0, 0.0), (0.0, 1.0), (0.7, 0.7j)])
def test_uniform_film_co_registers_with_rcwa_1d(ang, inc):
    eps_f, t = 4.0 + 0.8j, 0.30e-6
    rst = RCWAStack(period=P, n_substrate=1.5, n_superstrate=1.0, n_orders=5)
    rst.add_layer(t, eps=eps_f)
    rres = rst.set_source(WL, theta=ang).solve(retain_internal=True)
    pst = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=12)
    pst.add_layer(t, eps=eps_f)
    pst.set_source(WL, angle=ang).solve(retain_internal=True)
    zs = np.linspace(0.02e-6, 0.28e-6, 5)
    rf = rres.internal_field(zs, nx=16, incident=inc)
    pf = pst.internal_field(zs, incident=inc)
    k0 = 2 * np.pi / WL
    kx0 = np.sin(ang) * k0
    glob = max(np.max(np.abs(rf[c])) for c in COMPS)
    for c in COMPS:
        # de-carrier -> x-independent for a uniform film -> compare medians
        rv = rf[c] * np.exp(-1j * kx0 * rf["x"])[None, :]
        pv = pf[c] * np.exp(-1j * kx0 * pf["x"])[None, :]
        rmed = np.median(rv.real, axis=1) + 1j * np.median(rv.imag, axis=1)
        pmed = np.median(pv.real, axis=1) + 1j * np.median(pv.imag, axis=1)
        assert np.max(np.abs(rmed - pmed)) / glob < 1e-10, c


def test_patterned_interior_rcwa_converges_toward_pmm():
    """RCWA's near field at n_orders=60 sits within 2e-2 of the
    nodal-exact PMM in the interior (away from the Gibbs wall zones); the
    measured ladder is 3.4e-2 @ 30 -> 9.5e-3 @ 60 -> 9.3e-3 @ 120 (Ex)."""
    t = 0.25e-6
    S = 4096
    xc = (np.arange(S) + 0.5) / S
    cell = np.where(np.abs(xc - 0.5) < 0.225, 4.0 + 0.8j, 1.0 + 0j)
    rst = RCWAStack(period=P, n_substrate=1.5, n_superstrate=1.0,
                    n_orders=60)
    rst.add_layer(t, eps_cell=cell)
    rres = rst.set_source(WL).solve(retain_internal=True)
    pst = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=18,
                   elements_per_region=2)
    pst.add_layer(t, segments=[(0.275, 1.0), (0.45, 4.0 + 0.8j),
                               (0.275, 1.0)])
    pst.set_source(WL).solve(retain_internal=True)
    NX = 64
    zs = np.array([0.05e-6, 0.125e-6, 0.2e-6])
    rf = rres.internal_field(zs, nx=NX, incident=(1.0, 0.0))
    pf = pst.internal_field(zs, nx=NX, incident=(1.0, 0.0))
    kk = np.arange(NX)
    idx_p = (kk - NX // 2) % NX               # PMM [0,P) -> RCWA centred grid
    xw = idx_p * (P / NX)
    dist = np.minimum(np.abs(xw - 0.275 * P), np.abs(xw - 0.725 * P))
    mask = np.minimum(dist, P - dist) > 0.05 * P
    for c, tol in (("Ex", 2e-2), ("Ez", 3e-2), ("Hy", 1e-2)):
        rv, pv = rf[c], pf[c][:, idx_p]
        scale = np.max(np.abs(rv))
        assert np.max(np.abs(rv[:, mask] - pv[:, mask])) / scale < tol, c
    for c in ("Ey", "Hx", "Hz"):              # TM drive: TE channel silent
        assert np.max(np.abs(pf[c])) < 1e-10, c


def test_absorption_volume_identity_1d():
    """k0 * integral Im(eps)|E|^2 (GLL in x, midpoint in z) closes against
    the flux-based layer_absorption -- validates Ez inclusion + the flux
    normalization.  TE is wall-free (tight); TM carries the documented
    wall-interpolation error of the C0 |Ex|^2 (5.6e-3 @ deg 24)."""
    from lumenairy.elements.pmm._core import _gll_nodes_weights
    ang, t = 0.25, 0.25e-6
    pst = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=24)
    pst.add_layer(t, segments=[(0.45, 4.0 + 0.8j), (0.55, 1.0)])
    pst.add_layer(0.1e-6, eps=2.25)
    pst.set_source(WL, angle=ang).solve(retain_internal=True)
    A_flux = pst.layer_absorption()
    d = pst._internal
    mats0 = d["layer_mats"][0]
    ref_nodes, ref_w = _gll_nodes_weights(mats0["degree"])
    n_glob = d["n_glob"]
    wq = np.zeros(n_glob)
    x_glob = np.zeros(n_glob)
    for e, (xl, xr, _t) in enumerate(mats0["elem_bnds"]):
        wq[mats0["l2g"][e]] += ref_w * 0.5 * (xr - xl)
        x_glob[mats0["l2g"][e]] = (0.5 * (xl + xr)
                                   + 0.5 * (xr - xl) * np.asarray(ref_nodes))
    order = np.argsort(x_glob)
    mats_i = d["layer_mats"][0]
    im_n = np.zeros(n_glob)
    cnt = np.zeros(n_glob)
    for e, (xl, xr, tns) in enumerate(mats_i["elem_bnds"]):
        im_n[mats_i["l2g"][e]] += np.imag(tns["exx"])
        cnt[mats_i["l2g"][e]] += 1.0
    im_s = (im_n / cnt)[order]
    wq_s = wq[order]
    NZ = 200
    k0 = 2 * np.pi / WL
    kzi, kx0n = np.cos(ang), np.sin(ang)
    zq = (np.arange(NZ) + 0.5) / NZ * t
    A_vol = np.empty(2)
    for pol in (0, 1):
        f = pst.internal_field(zq, pol=pol)
        E2 = (np.abs(f["Ex"]) ** 2 + np.abs(f["Ey"]) ** 2
              + np.abs(f["Ez"]) ** 2)
        einc = (1.0 + (kx0n / kzi) ** 2) if pol == 0 else 1.0
        A_vol[pol] = (k0 * np.sum(E2 @ (im_s * wq_s)) * (t / NZ)
                      / (P * kzi * einc))
    assert abs(A_vol[1] - A_flux[0, 1]) / A_flux[0, 1] < 5e-3      # TE
    assert abs(A_vol[0] - A_flux[0, 0]) / A_flux[0, 0] < 2e-2      # TM


def test_internal_field_api_1d():
    pst = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=12)
    pst.add_layer(0.2e-6, segments=[(0.5, 4.0 + 0.5j), (0.5, 1.0)])
    pst.set_source(WL).solve(retain_internal=True)
    # scalar z keeps the (n_x,) legacy shape + pol= still works
    f = pst.internal_field(0.1e-6, pol=1)
    assert f["Ex"].ndim == 1 and isinstance(f["layer"], int)
    assert bool(np.all(np.diff(f["x"]) >= 0))
    # pol=1 == incident=(0,1)
    g = pst.internal_field(0.1e-6, incident=(0.0, 1.0))
    assert np.allclose(f["Ey"], g["Ey"], atol=0, rtol=0)
    # array z -> (nz, n_x); component subset; resampling grid
    h = pst.internal_field(np.array([0.05e-6, 0.15e-6]), component="E",
                           nx=48)
    assert h["Ex"].shape == (2, 48) and "Hx" not in h
    with pytest.raises(ValueError, match="component"):
        pst.internal_field(0.1e-6, component="bogus")
    with pytest.raises(ValueError, match="not both"):
        pst.internal_field(0.1e-6, pol=0, incident=(1.0, 0.0))
    st2 = PMMStack(P, degree=10)
    st2.add_layer(0.1e-6, eps=2.25)
    st2.set_source(WL).solve()
    with pytest.raises(ValueError, match="retain_internal"):
        st2.internal_field(0.05e-6)


def test_rcwa_complex_incident_public_linearity():
    """REGRESSION (RCWA bug found by the PMM oracle): a complex incident
    Jones must be the PUBLIC-linear superposition of the basis drives --
    previously it returned the conjugated incident's field (opposite
    handedness for circular drives)."""
    rst = RCWAStack(period=P, n_substrate=1.5, n_superstrate=1.0, n_orders=5)
    rst.add_layer(0.3e-6, eps=4.0 + 0.8j)
    rres = rst.set_source(WL, theta=0.3).solve(retain_internal=True)
    zs = np.array([0.1e-6])
    fx = rres.internal_field(zs, nx=8, incident=(1.0, 0.0))
    fy = rres.internal_field(zs, nx=8, incident=(0.0, 1.0))
    fc = rres.internal_field(zs, nx=8, incident=(0.7, 0.7j))
    for c in COMPS:
        lin = 0.7 * fx[c] + 0.7j * fy[c]
        assert np.max(np.abs(fc[c] - lin)) < 1e-12, c


# =========================================================================== #
# 2-D PMM2DStack
# =========================================================================== #

P2 = 0.6e-6


def test_uniform_film_co_registers_with_rcwa_2d_conical():
    eps_f, t = 4.0 + 0.8j, 0.25e-6
    th, ph = 0.3, 0.5
    rst = RCWAStack(period=P2, period_y=P2, n_substrate=1.5,
                    n_superstrate=1.0, n_orders=3, n_orders_y=3)
    rst.add_layer(t, eps=eps_f)
    rres = rst.set_source(WL, theta=th, phi=ph).solve(retain_internal=True)
    pst = PMM2DStack(P2, n_substrate=1.5, n_superstrate=1.0, degree=7,
                     n_orders=3)
    pst.add_layer(t, eps=eps_f)
    pst.set_source(WL, theta=th, phi=ph).solve(retain_internal=True)
    zs = np.linspace(0.03e-6, 0.22e-6, 4)
    for inc in ((1.0, 0.0), (0.7, 0.7j)):
        rf = rres.internal_field(zs, nx=8, ny=8, incident=inc)
        pf = pst.internal_field(zs, nx=8, ny=8, incident=inc)
        glob = max(np.max(np.abs(rf[c])) for c in COMPS)
        for c in COMPS:
            assert np.max(np.abs(rf[c] - pf[c])) / glob < 1e-12, (c, inc)


def test_patterned_2d_matches_nodal_exact_1d_oracle():
    """A y-uniform stripe through the 2-D machinery vs the nodal-exact 1-D
    PMM: interior agreement at the measured Fourier-truncation level
    (n=13/deg=17: Ex 6.2e-2, Ez 8.2e-2, Hy 1.5e-2; ~2x headroom), and the
    whole ladder converges as (n_orders, degree) rise together."""
    t = 0.25e-6
    S2 = 8
    x2 = (np.arange(S2) + 0.5) / S2
    stripe2 = np.where(np.abs(x2 - 0.5) < 0.25, 4.0 + 0.8j, 1.0 + 0j)
    cell2 = np.tile(stripe2[:, None], (1, S2))
    p1 = PMMStack(P2, n_substrate=1.5, n_superstrate=1.0, degree=17,
                  elements_per_region=2)
    p1.add_layer(t, segments=[(0.25, 1.0), (0.5, 4.0 + 0.8j), (0.25, 1.0)])
    p1.set_source(WL).solve(retain_internal=True)
    p2 = PMM2DStack(P2, n_substrate=1.5, n_superstrate=1.0, degree=17,
                    n_orders=13, formulation="laurent")
    p2.add_layer(t, eps_cell=cell2)
    p2.set_source(WL).solve(retain_internal=True)
    NX = 64
    zs = np.array([0.125e-6])
    pf1 = p1.internal_field(zs, nx=NX, incident=(1.0, 0.0))
    pf2 = p2.internal_field(zs, nx=NX, ny=1, incident=(1.0, 0.0))
    kk = np.arange(NX)
    idx_p = (kk - NX // 2) % NX
    xg = (np.arange(NX) - NX // 2) * (P2 / NX)
    mask = np.abs(np.abs(xg) - 0.25 * P2) > 0.05 * P2
    for c, tol in (("Ex", 1.5e-1), ("Ez", 1.7e-1), ("Hy", 4e-2)):
        ref = pf1[c][:, idx_p]
        scale = np.max(np.abs(ref))
        assert np.max(np.abs((ref - pf2[c])[:, mask])) / scale < tol, c


def test_internal_interface_continuity_2d():
    t = 0.25e-6
    S2 = 8
    x2 = (np.arange(S2) + 0.5) / S2
    X2, Y2 = np.meshgrid(x2, x2, indexing="ij")
    cellp = np.where((np.abs(X2 - 0.5) < 0.25) & (np.abs(Y2 - 0.5) < 0.25),
                     4.0 + 0.8j, 1.0 + 0j)
    pst = PMM2DStack(P2, n_substrate=1.5, n_superstrate=1.0, degree=9,
                     n_orders=5)
    pst.add_layer(t, eps_cell=cellp)
    pst.add_layer(0.08e-6, eps=2.25)
    pst.set_source(WL).solve(retain_internal=True)
    fa = pst.internal_field(np.array([t - 1e-12]), nx=24, ny=24,
                            incident=(1.0, 0.0))
    fb = pst.internal_field(np.array([t + 1e-12]), nx=24, ny=24,
                            incident=(1.0, 0.0))
    for c in ("Ex", "Ey", "Hx", "Hy"):
        scale = max(np.max(np.abs(fa[c])), 1e-12)
        assert np.max(np.abs(fa[c] - fb[c])) / scale < 1e-3, c


def test_absorption_volume_identity_2d_uniform():
    """A UNIFORM lossy layer carries only the (0,0) order at normal
    incidence, so the volume integral is quadrature-exact -- the tight 2-D
    identity (the patterned case is wall-quadrature-limited)."""
    t = 0.2e-6
    pst = PMM2DStack(P2, n_substrate=1.5, n_superstrate=1.0, degree=7,
                     n_orders=3)
    pst.add_layer(t, eps=4.0 + 0.8j)
    pst.add_layer(0.08e-6, eps=2.25)
    pst.set_source(WL).solve(retain_internal=True)
    A_flux = pst.layer_absorption()
    NZ = 400
    zq = (np.arange(NZ) + 0.5) / NZ * t
    f = pst.internal_field(zq, nx=4, ny=4, incident=(1.0, 0.0))
    E2 = (np.abs(f["Ex"]) ** 2 + np.abs(f["Ey"]) ** 2
          + np.abs(f["Ez"]) ** 2)
    k0 = 2 * np.pi / WL
    A_vol = k0 * 0.8 * np.mean(E2.reshape(NZ, -1), axis=1).sum() * (t / NZ)
    assert abs(A_vol - A_flux[0, 0]) / A_flux[0, 0] < 1e-3
    assert abs(A_flux[1].sum()) < 1e-12               # lossless spacer


def test_internal_field_api_2d():
    pst = PMM2DStack(P2, n_substrate=1.5, n_superstrate=1.0, degree=7,
                     n_orders=3)
    pst.add_layer(0.15e-6, eps=2.25)
    pst.set_source(WL).solve(retain_internal=True)
    f = pst.internal_field(np.array([0.05e-6]), nx=8, ny=8,
                           filter="lanczos")
    assert f["Ex"].shape == (1, 8, 8)
    g = pst.internal_field(np.array([0.05e-6]), nx=8, ny=1, component="H")
    assert g["Hx"].shape == (1, 8) and "Ex" not in g
    with pytest.raises(ValueError, match="filter"):
        pst.internal_field(0.05e-6, filter="bogus")
    st2 = PMM2DStack(P2, degree=7, n_orders=3)
    st2.add_layer(0.1e-6, eps=2.25)
    st2.set_source(WL).solve()
    with pytest.raises(ValueError, match="retain_internal"):
        st2.internal_field(0.05e-6)
