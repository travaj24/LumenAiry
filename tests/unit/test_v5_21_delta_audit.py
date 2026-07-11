"""v5.18.1 -> v5.21 delta-audit remediation (AUDIT_V5_21_DELTA_2026_07_07).

Locks in the fixes for the delta audit's new findings:

* D1  -- conical PMM entries + PMMStack conical solve now run the suite-standard
         incidence / grazing-safe guards (gain / metallic incidence rejected;
         the grazing nudge is a no-op away from a Rayleigh cutoff, so valid
         solves are byte-identical).
* D2  -- the finite-difference ray-transfer Jacobian ANDs companion-aliveness
         into ``alive`` and scrubs NaNs, so a rim-adjacent base ray whose +/-h
         companion vignettes is masked out instead of leaking a NaN Jacobian
         into the coherent GBD sum.
* D3  -- the multibranch KMAH index counts in-glass (surface-to-surface) fold
         caustics per leg (0 for an air-focus system, so existing charts are
         byte-identical) instead of relying on a mod-2 parity closure that
         missed an even internal-crossing count.
"""
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.rcwa._core import _grazing_safe_wavelength

_P, _WL, _DEP = 0.6e-6, 0.55e-6, 0.25e-6
_TH, _PHI = np.deg2rad(30.0), np.deg2rad(40.0)


# ----------------------------------------------------------------- D1 ------
def test_d1_conical_isotropic_rejects_metallic_and_gain():
    with pytest.raises(ValueError, match="non-propagating"):
        la.pmm_jones_1d_conical(_P, 6.0, 1.0, 1.5, 0.1 + 3.0j, _DEP, 0.5, _WL,
                                theta=_TH, phi=_PHI, degree=8, n_orders=3)
    with pytest.raises(ValueError, match="gain incidence medium"):
        la.pmm_jones_1d_conical(_P, 6.0, 1.0, 1.5, 1.0 - 1e-6j, _DEP, 0.5, _WL,
                                theta=_TH, phi=_PHI, degree=8, n_orders=3)


def test_d1_conical_tensor_and_stack_reject_gain():
    eps_t = np.broadcast_to(np.diag([2.25, 2.25, 2.25]).astype(complex),
                            (2, 3, 3)).copy()
    with pytest.raises(ValueError, match="gain incidence medium"):
        la.pmm_jones_1d_conical_tensor(_P, eps_t, 1.5, 1.0 - 1e-6j, _DEP, _WL,
                                       theta=_TH, phi=_PHI, degree=8,
                                       n_orders=3)
    s = la.PMMStack(_P, n_substrate=1.5, n_superstrate=1.0 - 1e-6j, degree=10)
    s.add_layer(_DEP, segments=[(0.5, 6.0), (0.5, 1.0)])
    s.set_source(_WL, theta=_TH, phi=_PHI)
    with pytest.raises(ValueError, match="gain incidence medium"):
        s.solve()


def test_d1_grazing_nudge_is_noop_away_from_cutoff():
    """A valid conical solve is byte-identical: the grazing-safe nudge returns
    the wavelength unchanged when no order is on a Rayleigh cutoff."""
    ox = np.arange(-5, 6)
    kx0 = np.sin(_TH) * np.cos(_PHI)
    ky0 = np.sin(_TH) * np.sin(_PHI)
    wl2 = _grazing_safe_wavelength(_WL, kx0, ky0, ox, ox * 0, _P, _P,
                                   [1.0, 2.25, 6.0, 1.0])
    assert wl2 == _WL


# ----------------------------------------------------------------- D2 ------
def test_d2_fd_jacobian_masks_dead_companions():
    """A base ray at the aperture rim (its +h companion vignettes) is marked
    dead and carries no NaN Jacobian row."""
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.differential import ray_transfer_jacobian
    surfs = surfaces_from_prescription({'aperture_diameter': 6e-3, 'surfaces': [
        {'radius': 25e-3, 'conic': -0.6, 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 3.0e-3},
        {'radius': -25e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 3.0e-3}],
        'thicknesses': [4e-3, 40e-3]})
    xs = np.array([0.0, 1.0e-3, 2.99e-3, 3.0e-3])   # last is exactly at the rim
    ze = np.zeros_like(xs)
    for per in (False, True):
        dt = ray_transfer_jacobian(xs, ze, ze, ze, surfs, 0.633e-6,
                                   per_surface=per)
        J = np.asarray(dt.jacobian)
        assert not np.isnan(J).any()               # NaNs scrubbed
        alive = np.asarray(dt.alive)
        assert not alive[-1]                        # rim ray masked out
        for i in np.where(alive)[0]:               # kept rows fully finite
            row = J[:, i] if per else J[i]
            assert np.isfinite(row).all()


# ----------------------------------------------------------------- D3 ------
def _singlet():
    return {'aperture_diameter': 6e-3, 'surfaces': [
        {'radius': 25e-3, 'conic': 0., 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 3e-3},
        {'radius': -25e-3, 'conic': 0., 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 3e-3}],
        'thicknesses': [3e-3, 40e-3]}


def test_d3_in_glass_count_zero_for_air_focus():
    """No ray focuses inside a simple singlet -> the in-glass KMAH count is 0
    at every node, so the multibranch index is byte-identical to the prior
    release (the parity-only closure is only extended, never perturbed here)."""
    from lumenairy.elements._lens_traced_multibranch import (
        _kmah_in_glass,
        _trace_launch_grid,
    )
    g = _trace_launch_grid(_singlet(), 0.633e-6, 2.5e-3, 21, 40e-3, 1.0)
    assert len(g['history']) == 2
    m_ig = _kmah_in_glass(g)
    assert int((m_ig > 0).sum()) == 0


def test_d3_shared_counter_matches_manual_quadratic():
    """The factored fold-root counter reproduces the exact-quadratic count."""
    from lumenairy.elements._lens_traced_multibranch import _count_fold_roots
    # detQ(z) = 1 - 3 z + 2 z^2 = (1-z)(1-2z): roots at z=0.5 and z=1.0
    detQ0 = np.array([[1.0]])
    trK = np.array([[-3.0]])
    detQd = np.array([[2.0]])
    assert _count_fold_roots(detQ0, trK, detQd, 0.4)[0, 0] == 0
    assert _count_fold_roots(detQ0, trK, detQd, 0.9)[0, 0] == 1
    assert _count_fold_roots(detQ0, trK, detQd, 1.5)[0, 0] == 2


def test_d3_air_focus_multibranch_runs_without_warning():
    """The internal-focus warning does NOT fire for an air-focus singlet."""
    from lumenairy.elements.lenses import apply_real_lens_traced_multibranch
    N, dx = 48, 25e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # any warn -> failure
        out = apply_real_lens_traced_multibranch(
            E, prescription=_singlet(), wavelength=0.633e-6, dx=dx,
            output_plane_distance=38e-3)
    assert np.isfinite(out).all()
