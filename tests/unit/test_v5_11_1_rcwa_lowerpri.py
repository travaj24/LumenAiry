"""v5.11.1 RCWA lower-priority ergonomics / hardening batch.

Five additive, bit-identical-to-existing-behaviour items:

1. DIFFERENTIABILITY guard + pin -- the IN-PLANE ``rcwa_jones_1d_segments`` path
   differentiates under ``jax.grad`` (matches finite difference); a JAX tensor
   with OUT-OF-PLANE coupling raises ``NotImplementedError`` (its full-3x3 solver
   uses a host ``np.where`` / ``argsort`` forward-mode split that breaks autodiff,
   and the in-plane router would otherwise SILENTLY drop the z-coupling).
2. ``RCWAResult.layer_absorption`` for TENSOR layers -- the per-layer absorption
   of a lossy anisotropic stack sums to the total absorptance, and a lossless
   tensor stack deposits ~0 raw loss (the quadratic form ``Im(E* . eps . E)``
   correctly localises loss).
3. ``reflective_outcoupling`` JAX twin -- backend-agnostic; the NumPy result is a
   float (bit-identical to the historical path) and ``jax.grad`` flows through.
4. ``rcwa_convergence`` for an ``RCWAStack`` -- accepts a configured stack, solves
   at its ``n_orders`` and a bumped count, restores the truncation, reports delta.
5. ``rcwa_jones_vs_wavelength_segments`` -- dispersive Jones sweep of a
   multi-segment grating; a single-wavelength entry matches the direct
   ``rcwa_jones_1d_segments`` call.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.rcwa import (
    rcwa_convergence,
    rcwa_jones_1d,
    rcwa_jones_1d_segments,
    rcwa_jones_vs_wavelength_segments,
    reflective_outcoupling,
    uniaxial_tensor,
)

WL = 0.55e-6
DEPTH = 0.4e-6
PERIOD = 0.8e-6
N_SUB, N_SUP = 1.5, 1.0

# in-plane (theta=pi/2): NO off-plane coupling -> the differentiable subset.
RIDGE_INPLANE = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=0.3)
GROOVE_ISO = (1.0 + 0.0j) ** 2 * np.eye(3)
# tilted (theta=pi/4): genuine out-of-plane coupling.
RIDGE_OFFPLANE = uniaxial_tensor(1.5, 1.7, np.pi / 4, phi=0.3)


# ===========================================================================
# Item 1 -- differentiability finding: in-plane diffs, off-plane rejected.
# ===========================================================================
def _jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


def test_inplane_segments_gradient_matches_finite_difference():
    """IN-PLANE rcwa_jones_1d_segments differentiates: jax.grad == FD."""
    jax = _jax()
    import jax.numpy as jnp

    def fom(scale):
        ridge = scale * jnp.asarray(RIDGE_INPLANE)
        groove = jnp.asarray(GROOVE_ISO)
        o, R, T, J = rcwa_jones_1d_segments(
            PERIOD, [(0.4, ridge), (0.6, groove)], N_SUB, N_SUP, DEPTH, WL,
            n_orders=6)
        return jnp.real(jnp.sum(T))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = float(jax.grad(fom)(jnp.asarray(1.0)))
        h = 1e-6
        fd = float((fom(jnp.asarray(1.0 + h)) - fom(jnp.asarray(1.0 - h)))
                   / (2 * h))
    assert np.isfinite(g)
    assert abs(g) > 1e-9                                 # a real, non-zero grad
    assert np.isclose(g, fd, rtol=1e-4, atol=1e-9)


def test_offplane_jax_tensor_raises_notimplemented():
    """A JAX off-plane tensor (tilted director) has no differentiable path."""
    _jax()
    import jax.numpy as jnp

    jr = jnp.asarray(RIDGE_OFFPLANE)
    jg = jnp.asarray(GROOVE_ISO)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotImplementedError, match="OUT-OF-PLANE"):
            rcwa_jones_1d_segments(PERIOD, [(0.4, jr), (0.6, jg)], N_SUB, N_SUP,
                                   DEPTH, WL, n_orders=4)
        # rcwa_jones_1d shares the guard.
        with pytest.raises(NotImplementedError, match="OUT-OF-PLANE"):
            rcwa_jones_1d(PERIOD, jr, jg, N_SUB, N_SUP, DEPTH, 0.4, WL,
                          n_orders=4)


def test_offplane_numpy_still_runs_rigorously():
    """The off-plane guard is JAX-only: a NumPy off-plane solve is unchanged
    (rigorous, energy-conserving)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = rcwa_jones_1d_segments(
            PERIOD, [(0.4, RIDGE_OFFPLANE), (0.6, GROOVE_ISO)], N_SUB, N_SUP,
            DEPTH, WL, n_orders=8)
    # lossless -> R + T = 1 per incident polarization.
    assert np.allclose(R.sum(axis=1) + T.sum(axis=1), 1.0, atol=1e-6)


# ===========================================================================
# Item 2 -- layer_absorption for TENSOR layers.
# ===========================================================================
def _lc_grating_cell(no, ne, S=64):
    """In-plane (theta=pi/2) uniaxial LC patterned as an x-grating ridge."""
    T = uniaxial_tensor(no, ne, np.pi / 2, phi=0.4)      # in-plane tensor
    iso = (1.0 + 0.0j) ** 2 * np.eye(3)
    x = (np.arange(S) + 0.5) / S
    cell = np.empty((S, 1, 3, 3), dtype=np.complex128)
    for k in range(S):
        cell[k, 0] = T if x[k] < 0.5 else iso
    return cell


def _tensor_stack(no, ne):
    cell = _lc_grating_cell(no, ne)
    st = la.RCWAStack(period=0.8e-6, n_superstrate=1.0, n_substrate=1.5,
                      n_orders=10)
    st.add_layer(0.1e-6, eps=2.1)                        # lossless spacer
    st.add_layer(0.4e-6, eps_tensor_cell=cell)           # tensor grating
    st.add_layer(0.1e-6, eps=2.25)                       # lossless spacer
    return st


def test_tensor_layer_absorption_sums_to_total():
    """Per-layer absorption of a LOSSY tensor stack sums to 1 - R - T, and the
    lossless spacers carry ~0 (the loss localises on the lossy tensor layer)."""
    st = _tensor_stack(no=1.5 + 0.02j, ne=1.7 + 0.05j)
    res = st.set_source(WL, theta=0.0).solve(retain_internal=True)
    A = res.absorptance()
    assert np.all(A > 1e-3)                               # genuinely lossy
    LA = res.layer_absorption(nx=64, nz_per_layer=10)
    assert LA.shape == (2, 3)
    # (a) energy closure: per-layer sum == total absorptance.
    assert np.allclose(LA.sum(axis=1), A, atol=1e-6)
    # (b) loss localisation: the lossless spacers get ~0.
    assert np.allclose(LA[:, 0], 0.0, atol=1e-9)
    assert np.allclose(LA[:, 2], 0.0, atol=1e-9)
    # (c) the lossy tensor layer carries essentially all of it.
    assert np.allclose(LA[:, 1], A, atol=1e-6)


def test_lossless_tensor_layer_has_zero_raw_absorption():
    """A LOSSLESS tensor stack has ~0 absorptance and the quadratic form does
    NOT manufacture spurious loss from the off-diagonal tensor terms."""
    st = _tensor_stack(no=1.5 + 0.0j, ne=1.7 + 0.0j)
    res = st.set_source(WL, theta=0.0).solve(retain_internal=True)
    assert np.allclose(res.absorptance(), 0.0, atol=1e-6)
    LA = res.layer_absorption(nx=48, nz_per_layer=6)
    assert np.allclose(LA, 0.0, atol=1e-6)


def test_layer_absorption_iso_path_unchanged():
    """A purely isotropic lossy stack still works (the iso branch is the old
    Im(eps)|E|^2 path, untouched)."""
    S = 128
    x = (np.arange(S) + 0.5) / S
    cell = np.where(x < 0.5, (2.0 + 0.1j), 1.0 + 0j)[:, None].astype(complex)
    st = la.RCWAStack(period=0.8e-6, n_superstrate=1.0, n_substrate=1.5,
                      n_orders=10)
    st.add_layer(0.3e-6, eps_cell=cell)
    st.add_layer(0.1e-6, eps=2.25)
    res = st.set_source(WL).solve(retain_internal=True)
    LA = res.layer_absorption(nx=64, nz_per_layer=8)
    assert np.allclose(LA.sum(axis=1), res.absorptance(), atol=1e-6)
    assert np.allclose(LA[:, 1], 0.0, atol=1e-9)          # lossless spacer


# ===========================================================================
# Item 3 -- reflective_outcoupling JAX twin.
# ===========================================================================
JONES = np.array([[0.8 + 0.1j, 0.05 - 0.02j],
                  [0.03 + 0.01j, 0.7 - 0.15j]], dtype=np.complex128)


def test_reflective_outcoupling_numpy_returns_float():
    """NumPy path returns a Python float, bit-identical to the historical
    formula |[Q J Q]_yx|^2."""
    from lumenairy.elements.rcwa import _qwp_matrix

    v = reflective_outcoupling(JONES)
    assert isinstance(v, float)
    Q = _qwp_matrix(np.pi / 4)
    expected = float(np.abs((Q @ JONES @ Q)[1, 0]) ** 2)
    assert v == expected                                 # bit-identical
    # list input also works (array_like) and matches.
    assert np.isclose(reflective_outcoupling(JONES.tolist()), v)


def test_reflective_outcoupling_qwp_angle_passthrough():
    v0 = reflective_outcoupling(JONES, qwp_angle=0.3)
    assert isinstance(v0, float) and np.isfinite(v0)
    assert not np.isclose(v0, reflective_outcoupling(JONES))   # angle matters


def test_reflective_outcoupling_jax_gradient_matches_fd():
    """A JAX Jones returns a traced scalar and jax.grad flows through; the
    value matches the NumPy result and the gradient matches FD."""
    jax = _jax()
    import jax.numpy as jnp

    def fom(g):
        J = jnp.array([[0.8 + 0.1j, g * (0.05 - 0.02j)],
                       [g * (0.03 + 0.01j), 0.7 - 0.15j]], dtype=jnp.complex128)
        return reflective_outcoupling(J)

    assert np.isclose(float(fom(jnp.asarray(1.0))),
                      reflective_outcoupling(JONES))
    g = float(jax.grad(fom)(jnp.asarray(1.0)))
    h = 1e-6
    fd = float((fom(jnp.asarray(1.0 + h)) - fom(jnp.asarray(1.0 - h))) / (2 * h))
    assert np.isfinite(g)
    assert np.isclose(g, fd, rtol=1e-5, atol=1e-9)


# ===========================================================================
# Item 4 -- rcwa_convergence for an RCWAStack.
# ===========================================================================
def _dielectric_stack_1d(n_orders=8):
    S = 256
    x = (np.arange(S) + 0.5) / S
    cell = np.where(x < 0.5, 2.5 + 0j, 1.0 + 0j)[:, None]
    st = la.RCWAStack(period=0.7e-6, n_superstrate=1.0, n_substrate=1.5,
                      n_orders=n_orders)
    st.add_layer(0.3e-6, eps_cell=cell)
    return st.set_source(WL, theta=0.1)


def test_rcwa_convergence_stack_1d():
    st = _dielectric_stack_1d(n_orders=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        high, rep = rcwa_convergence(st, bump=4)
    # truncation bumped then RESTORED
    assert st.nox == 8
    assert rep["n_orders_low"] == {"n_orders": 8}
    assert rep["n_orders_high"] == {"n_orders": 12}
    assert "converged" in rep and "delta" in rep
    assert rep["delta"] >= 0.0
    # the returned high-resolution result is an energy-conserving RCWAResult
    o, R, T = high.efficiencies()
    assert np.allclose(R.sum(axis=1) + T.sum(axis=1), 1.0, atol=1e-6)


def test_rcwa_convergence_stack_2d_bumps_both_axes():
    S = 48
    xx = (np.arange(S) + 0.5) / S - 0.5
    X, Y = np.meshgrid(xx, xx, indexing="ij")
    cell = np.where((np.abs(X) < 0.25) & (np.abs(Y) < 0.25), 2.5 + 0j, 1.0 + 0j)
    st = la.RCWAStack(period=0.6e-6, period_y=0.6e-6, n_superstrate=1.0,
                      n_substrate=1.5, n_orders=3)
    st.add_layer(0.2e-6, eps_cell=cell)
    st.set_source(WL, theta=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        high, rep = rcwa_convergence(st, bump=2)
    assert (st.nox, st.noy) == (3, 3)                    # restored
    assert rep["n_orders_low"] == {"n_orders": 3, "n_orders_y": 3}
    assert rep["n_orders_high"] == {"n_orders": 5, "n_orders_y": 5}


def test_rcwa_convergence_callable_path_unchanged():
    """The callable-solver path still works (item 4 is an additive branch)."""
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, rep = rcwa_convergence(
            rcwa_efficiency_1d, period=1e-6, n_ridge=1.5, n_groove=1.0,
            n_substrate=1.0, n_superstrate=1.0, depth=0.3e-6, duty_cycle=0.5,
            wavelength=0.55e-6, n_orders=8, polarization="te")
    assert isinstance(out, tuple) and len(out) >= 3
    assert "converged" in rep


# ===========================================================================
# Item 5 -- rcwa_jones_vs_wavelength_segments.
# ===========================================================================
def test_jones_vs_wavelength_segments_matches_single_call():
    segs = [(0.4, RIDGE_INPLANE), (0.6, GROOVE_ISO)]
    wls = np.array([0.50e-6, 0.55e-6, 0.60e-6])
    wl, J, R, T = rcwa_jones_vs_wavelength_segments(
        PERIOD, segs, N_SUB, N_SUP, DEPTH, wls, n_orders=8)
    assert J.shape == (3, 2, 2) and R.shape == (3, 2) and T.shape == (3, 2)
    o, Rd, Td, Jd = rcwa_jones_1d_segments(
        PERIOD, segs, N_SUB, N_SUP, DEPTH, 0.55e-6, n_orders=8)
    assert np.allclose(J[1], Jd)
    assert np.allclose(R[1], Rd.sum(axis=1))
    assert np.allclose(T[1], Td.sum(axis=1))


def test_jones_vs_wavelength_segments_scalar_in_scalar_out():
    segs = [(0.4, RIDGE_INPLANE), (0.6, GROOVE_ISO)]
    wl0, J0, R0, T0 = rcwa_jones_vs_wavelength_segments(
        PERIOD, segs, N_SUB, N_SUP, DEPTH, 0.55e-6, n_orders=8)
    assert np.ndim(wl0) == 0
    assert J0.shape == (2, 2) and R0.shape == (2,) and T0.shape == (2,)


def test_jones_vs_wavelength_segments_dispersive_callable():
    def disp_ridge(w):
        return uniaxial_tensor(1.5 + (w - 0.55e-6) * 1e6, 1.7, np.pi / 2,
                               phi=0.3)
    segs = [(0.4, disp_ridge), (0.6, GROOVE_ISO)]
    wls = np.array([0.50e-6, 0.60e-6])
    wl, J, R, T = rcwa_jones_vs_wavelength_segments(
        PERIOD, segs, N_SUB, N_SUP, DEPTH, wls, n_orders=8)
    # the dispersive ridge makes the two wavelengths genuinely differ.
    assert not np.allclose(J[0], J[1])


def test_jones_vs_wavelength_segments_validates_inputs():
    segs = [(0.4, RIDGE_INPLANE), (0.6, GROOVE_ISO)]
    with pytest.raises(ValueError, match="finite"):
        rcwa_jones_vs_wavelength_segments(PERIOD, segs, N_SUB, N_SUP, DEPTH,
                                          np.array([]), n_orders=4)
    with pytest.raises(ValueError, match="non-empty"):
        rcwa_jones_vs_wavelength_segments(PERIOD, [], N_SUB, N_SUP, DEPTH, WL,
                                          n_orders=4)


# ===========================================================================
# Export surface.
# ===========================================================================
def test_exports():
    assert hasattr(la, "rcwa_jones_vs_wavelength_segments")
    assert "rcwa_jones_vs_wavelength_segments" in la.__all__
    from lumenairy.elements import rcwa_jones_vs_wavelength_segments as _f
    assert _f is la.rcwa_jones_vs_wavelength_segments
