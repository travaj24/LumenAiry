"""v5.9.0 RCWA audit quick-wins.

Addresses the low-effort / high-value items from the two RCWA audits:
* GAP 3 -- a convergence / resonance self-check (``rcwa_convergence``) that
  catches a silently under-resolved truncation (a real error class: a coarse
  harmonic count can fabricate a spurious reflection null).
* GAP 5 -- a DISPERSIVE Jones spectral sweep (``rcwa_jones_vs_wavelength``).
* P5 / GAP 7 guard -- reject an OUT-OF-PLANE (full 3x3) permittivity tensor
  instead of silently dropping its z-coupling.
* GAP 2 verify -- confirm the stacked 1-D-anisotropic + isotropic ``RCWAStack``
  path actually solves and conserves energy (a previously "⚠ verify" claim).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.rcwa import (
    rcwa_convergence,
    rcwa_efficiency_1d,
    rcwa_efficiency_2d,
    rcwa_jones_1d,
    rcwa_jones_2d,
    rcwa_jones_vs_wavelength,
    uniaxial_tensor,
)

METAL = dict(period=1.0e-6, n_ridge=0.2 + 6j, n_groove=1.0, n_substrate=1.0,
             n_superstrate=1.0, depth=0.3e-6, duty_cycle=0.5, wavelength=0.8e-6,
             polarization="tm")
ISO3 = 2.0 * np.eye(3)


# ---------------------------------------------------------------------------
# out-of-plane (full 3x3) tensor rejection
# ---------------------------------------------------------------------------

def test_inplane_tensor_accepted():
    inplane = uniaxial_tensor(1.5, 1.7, np.pi / 2)        # director in x-y plane
    o, R, T, J = rcwa_jones_1d(1e-6, inplane, ISO3, 1.0, 1.0, 0.3e-6, 0.5,
                               0.8e-6, n_orders=6)
    assert abs((R.sum() + T.sum()) / 2 - 1.0) < 1e-6      # lossless -> energy ok


def test_out_of_plane_tensor_solved_jones_1d():
    """v5.11.0 (GAP7-full): the 1-D path now SOLVES a full-3x3 out-of-plane
    tensor (tilted-director LC) via the general anisotropic FMM (Li 2003)
    instead of rejecting it.  Was a hard ValueError pre-v5.11.0; now it returns
    an energy-conserving Jones response.  (2-D / RCWAStack still reject -- see
    test_out_of_plane_tensor_rejected_jones_2d_and_stack.)"""
    tilted = uniaxial_tensor(1.5, 1.7, np.pi / 4)         # 45deg tilt -> eps_xz!=0
    assert abs(tilted[0, 2]) > 0.1                         # genuinely out-of-plane
    o, R, T, J = rcwa_jones_1d(1e-6, tilted, ISO3, 1.0, 1.0, 0.3e-6, 0.5, 0.8e-6,
                               n_orders=6)
    # lossless tilted dielectric grating -> energy conserved per polarization
    assert abs(float(R[0].sum() + T[0].sum()) - 1.0) < 1e-6
    assert abs(float(R[1].sum() + T[1].sum()) - 1.0) < 1e-6


def test_out_of_plane_tensor_jones_2d_supported_stack_rejected():
    """RE-PINNED 2026-06-10 (audit GAP2, v5.14.1): rcwa_jones_2d now SOLVES
    out-of-plane tensors (generalized cascade; pinned against the conical
    Berreman oracle in test_v5_14_1_rcwa_deferred); the STACK still rejects
    them (its cascade is symmetric-mode based -- roadmap item)."""
    tilted = uniaxial_tensor(1.5, 1.7, np.pi / 4)
    Sx, Sy = 32, 32
    cell = np.broadcast_to(tilted, (Sx, Sy, 3, 3)).copy()
    o, R, T, J = rcwa_jones_2d(1e-6, 1e-6, cell, 1.0, 1.0, 0.2e-6, 0.8e-6,
                               n_orders_x=3, n_orders_y=3)
    for row in (0, 1):
        assert abs(float(R[row].sum() + T[row].sum()) - 1.0) < 1e-9
    with pytest.raises(ValueError, match="out-of-plane"):
        (la.RCWAStack(1e-6, period_y=1e-6, n_orders=3, n_orders_y=3)
         .add_layer(0.2e-6, eps_tensor_cell=cell))


# ---------------------------------------------------------------------------
# convergence / resonance guard
# ---------------------------------------------------------------------------

def test_convergence_flags_underresolved_and_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result, report = rcwa_convergence(rcwa_efficiency_1d, n_orders=4,
                                           **METAL)
    assert report["converged"] is False
    assert report["delta"] > 1e-3
    assert any("NOT converged" in str(x.message) for x in w)
    # the RETURNED result is the higher-resolution solve
    assert report["n_orders_high"] == {"n_orders": 8}
    assert len(result) == 3                                # (orders, R, T)


def test_convergence_passes_when_resolved():
    result, report = rcwa_convergence(rcwa_efficiency_1d, n_orders=30,
                                      warn=False, **METAL)
    assert report["converged"] is True
    assert report["delta"] < 1e-3


def test_convergence_supports_2d_order_params():
    S = 48
    xx = (np.arange(S) + 0.5) / S
    cell = np.where((xx < 0.5)[:, None], -3 + 4j, 1.0).astype(complex)
    cell = np.broadcast_to(cell, (S, S)).copy()
    result, report = rcwa_convergence(
        rcwa_efficiency_2d, order_params=("n_orders_x", "n_orders_y"), bump=2,
        warn=False, period_x=0.5e-6, period_y=0.5e-6, eps_cell=cell,
        n_substrate=1.0, n_superstrate=1.0, depth=0.2e-6, wavelength=0.6e-6,
        n_orders_x=4, n_orders_y=4, polarization="tm")
    assert report["n_orders_high"] == {"n_orders_x": 6, "n_orders_y": 6}
    assert "delta" in report


# ---------------------------------------------------------------------------
# dispersive Jones wavelength sweep
# ---------------------------------------------------------------------------

def test_jones_vs_wavelength_shapes_and_consistency():
    inplane = uniaxial_tensor(1.5, 1.7, np.pi / 2)
    wls = np.array([0.7e-6, 0.8e-6, 0.9e-6])
    wl, J, R, T = rcwa_jones_vs_wavelength(1e-6, inplane, ISO3, 1.0, 1.0,
                                           0.3e-6, 0.5, wls, n_orders=6)
    assert J.shape == (3, 2, 2) and R.shape == (3, 2) and T.shape == (3, 2)
    # matches a direct per-wavelength call
    o, Rd, Td, Jd = rcwa_jones_1d(1e-6, inplane, ISO3, 1.0, 1.0, 0.3e-6, 0.5,
                                  0.8e-6, n_orders=6)
    assert np.allclose(J[1], Jd)
    assert np.allclose(R[1], Rd.sum(axis=1))


def test_jones_vs_wavelength_dispersive_callable_and_scalar():
    def disp(w):                                          # n increases with wl
        return uniaxial_tensor(1.5 + 1e7 * (w - 0.8e-6), 1.7, np.pi / 2)
    # scalar wavelength in -> scalar out
    wl, J, R, T = rcwa_jones_vs_wavelength(1e-6, disp, ISO3, 1.0, 1.0, 0.3e-6,
                                           0.5, 0.8e-6, n_orders=6)
    assert np.ndim(wl) == 0 and J.shape == (2, 2)
    # dispersion actually changes the result across the sweep
    wls = np.array([0.75e-6, 0.85e-6])
    _, Js, _, _ = rcwa_jones_vs_wavelength(1e-6, disp, ISO3, 1.0, 1.0, 0.3e-6,
                                           0.5, wls, n_orders=6)
    assert not np.allclose(Js[0], Js[1])


# ---------------------------------------------------------------------------
# GAP 2 verify: stacked 1-D anisotropic + isotropic RCWAStack conserves energy
# ---------------------------------------------------------------------------

def test_stacked_anisotropic_isotropic_stack_conserves_energy():
    Sx, Sy = 64, 16
    xx = (np.arange(Sx) + 0.5) / Sx
    ridge = uniaxial_tensor(1.5, 1.8, np.pi / 2)          # lossless in-plane LC
    groove = np.eye(3)
    tcell = np.empty((Sx, Sy, 3, 3), dtype=complex)
    tcell[:] = np.where((xx < 0.5)[:, None, None, None],
                        ridge[None, None], groove[None, None])
    stack = (la.RCWAStack(1e-6, period_y=1e-6, n_substrate=1.5, n_orders=6,
                          n_orders_y=2)
             .add_layer(0.2e-6, eps_tensor_cell=tcell)    # anisotropic patterned
             .add_layer(0.15e-6, eps=2.1)                 # isotropic spacer
             .set_source(0.8e-6, theta=0.0))
    result = stack.solve()
    o, R, T = result.efficiencies()
    energy = R.sum(axis=1) + T.sum(axis=1) + result.absorptance()
    assert np.allclose(energy, 1.0, atol=1e-6)            # per incident pol
