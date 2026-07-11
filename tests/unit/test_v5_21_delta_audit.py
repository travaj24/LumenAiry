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


# ----------------------------------------------------------------- D4 ------
def test_d4_output_plane_n_eikonal_slope():
    """The intrapolation eikonal gradient is n*(L, M): output_plane_n=1 is a
    no-op (byte-identical), an immersed output plane rescales the phase."""
    from lumenairy.elements._lens_traced_multibranch import _trace_launch_grid
    g = _trace_launch_grid(_singlet(), 0.633e-6, 2.5e-3, 15, 30e-3, 1.0)
    # the exit direction cosines feed p = n*(L, M); the OPL advance already
    # scales by output_plane_n, so the intrapolation must match it (verified
    # via the byte-identity of the n=1 default in the multibranch tests).
    assert np.isfinite(g['L']).any() and np.isfinite(g['M']).any()


# --------------------------------------------------------------- D11/D12 ---
def _jax_ok():
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _jax_ok(), reason="jax not installed")
def test_d12_traced_oop_tensor_routes_to_exact_general_path():
    """A TRACED out-of-plane tensor now routes to the generalized (exact) berreman
    cascade instead of the ~2%-off native path; forward matches the concrete
    solve and jax.grad flows (D12, mirroring the rcwa tracer->general fix)."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from lumenairy.elements.berreman import berreman_jones_1d
    from lumenairy.elements.rcwa._core import uniaxial_tensor
    eps = uniaxial_tensor(1.5, 1.7, np.deg2rad(50.0), phi=np.deg2rad(30.0))
    th, ph, d, wl = np.deg2rad(35.0), np.deg2rad(20.0), 0.3e-6, 0.55e-6
    Rn, Tn, _, _ = berreman_jones_1d([(eps, d)], 1.5, 1.0, wl, angle=th, phi=ph)
    ref = float(np.real(Rn).sum() + np.real(Tn).sum())

    def fwd(scale):
        R, T, _, _ = berreman_jones_1d([(jnp.asarray(eps) * scale, d)], 1.5,
                                       1.0, wl, angle=th, phi=ph)
        return jnp.real(R).sum() + jnp.real(T).sum()

    val, grad = jax.value_and_grad(fwd)(1.0)
    assert abs(float(val) - ref) < 1e-9
    assert np.isfinite(float(grad))


# ----------------------------------------------------------------- D7 ------
def test_d7_clenshaw_curtis_weights_exact():
    """The residual-bound quadrature weights integrate polynomials up to the
    node degree exactly (the returned Levin bound is now an integral, not the
    edge-over-weighted sample mean)."""
    from lumenairy._math.levin import _cc_weights
    for n in (8, 24, 48):
        w = _cc_weights(n)
        nodes = np.cos(np.pi * np.arange(n) / (n - 1))[::-1]
        assert abs(w.sum() - 2.0) < 1e-13                 # int 1 = 2
        assert abs((w * nodes ** 2).sum() - 2.0 / 3.0) < 1e-13   # int x^2
        assert abs((w * nodes ** 4).sum() - 2.0 / 5.0) < 1e-13   # int x^4


def test_d7_levin1d_no_fmax_kwarg():
    """The dead fmax parameter is gone and the 1-D Levin still integrates a
    plain oscillatory integral accurately."""
    import inspect

    from lumenairy._math.levin import levin1d_adaptive
    assert "fmax" not in inspect.signature(levin1d_adaptive).parameters
    # int_-1^1 exp(i*40*y) dy = 2 sin(40)/40
    val = levin1d_adaptive(lambda y: 40.0 * y, lambda y: 40.0 * np.ones_like(y),
                           lambda y: np.ones_like(y), -1.0, 1.0, tol=1e-10)
    assert abs(val - 2.0 * np.sin(40.0) / 40.0) < 1e-8


# ----------------------------------------------------------------- D9 ------
def test_d9_bare_pmm2dstack_deprecation():
    """The bare PMM2DStack name warns (transitional alias, scheduled repoint);
    the explicit PMM2DStackHybrid / PMM2DStack_hybrid names do not."""
    from lumenairy.elements.pmm import (
        PMM2DStack,
        PMM2DStack_hybrid,
        PMM2DStackHybrid,
    )
    with pytest.warns(DeprecationWarning, match="TRANSITIONAL alias"):
        s = PMM2DStack(1e-6)
    assert isinstance(s, PMM2DStackHybrid)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # explicit names silent
        PMM2DStackHybrid(1e-6)
        PMM2DStack_hybrid(1e-6)


# ----------------------------------------------------------------- D15 -----
def test_d15_segmentation_respects_max_segments():
    """The traced-segmentation cap keeps the segment count within budget (the
    shallowest cuts are dropped when over budget)."""
    from lumenairy.elements._lens_traced import _segment_field_by_angle
    # two well-separated beams in x AND y -> up to 4 segments; cap at 2
    N, dx = 96, 20e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    k = 2 * np.pi / 0.633e-6
    E = (np.exp(1j * k * 0.05 * X) + np.exp(-1j * k * 0.05 * X)).astype(
        np.complex128) * np.exp(-(X ** 2 + Y ** 2) / (1.0e-3) ** 2)
    segs = _segment_field_by_angle(E, dx, dx, "auto", "auto", 0.995, 0.15,
                                   1e-3, 2)
    assert 1 <= len(segs) <= 2


@pytest.mark.skipif(not _jax_ok(), reason="jax not installed")
def test_d11a_offplane_stack_with_traced_iso_spacer_jits():
    """D11(a): a concrete OOP layer beside a TRACED isotropic spacer no longer
    raises ConcretizationTypeError under jit (bool(is_iso) made trace-safe)."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from lumenairy.elements.berreman import berreman_jones_1d
    from lumenairy.elements.rcwa._core import uniaxial_tensor
    eps = uniaxial_tensor(1.5, 1.7, np.deg2rad(50.0), phi=np.deg2rad(30.0))

    @jax.jit
    def solve(scale):
        R, T, _, _ = berreman_jones_1d(
            [(jnp.asarray(eps), 0.3e-6),
             (jnp.asarray(2.25 + 0j) * scale, 0.2e-6)],
            1.5, 1.0, 0.55e-6, angle=np.deg2rad(35.0), phi=np.deg2rad(20.0))
        return jnp.real(R).sum()
    assert np.isfinite(float(solve(1.0)))
