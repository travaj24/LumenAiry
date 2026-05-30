"""v5.4.6 Wave 2 regression pins: the three new P1s (GBD phase
conjugation F-1, seidel internal-stop crash F-2, RS buffer aliasing F-3)
plus the JAX/backend parity items (jax_trace direction-aware root pick
P1-1/P3-1, backend RNG dtype F-30/F-31, through_focus best-focus parity
F-28).

Each test fails on the pre-v5.4.6 behaviour and passes after the fix.
Author: v5.4.6 audit-closure cycle.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import lumenairy as la

_HAS_JAX = importlib.util.find_spec('jax') is not None


# ----------------------------------------------------------------------
# F-1: GBD reconstructed transverse phase must MATCH (not conjugate) ASM
# ----------------------------------------------------------------------

def test_gbd_freespace_phase_matches_asm_not_conjugate():
    """A Gaussian propagated past its Rayleigh range by GBD must carry the
    SAME-sign transverse wavefront curvature as ASM (the physics
    exp(+ikz) convention), not the complex conjugate.  Pre-fix the GBD
    radial phase was the negative of ASM (sum ~ 0)."""
    N, wl, dx, w0 = 128, 633e-9, 4e-6, 24e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X * X + Y * Y) / w0 ** 2).astype(np.complex128)
    z = 3e-3  # > z_R = pi w0^2 / lambda ~ 2.86 mm

    Ea = la.angular_spectrum_propagate(E0, z, wl, dx)
    Eg = la.propagate_gbd_freespace(E0, dx, z=z, wavelength=wl, sample_step=2)

    ic = N // 2

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    pa = np.angle(Ea[ic, :])
    pg = np.angle(Eg[ic, :])
    # Sample a few radii; the radial phase of a diverging beam increases
    # outward (positive) under exp(+ikz).
    for rp in (4, 8, 12):
        d_asm = wrap(pa[ic + rp] - pa[ic])
        d_gbd = wrap(pg[ic + rp] - pg[ic])
        assert d_asm > 0.05, "ASM diverging-beam radial phase should be positive"
        # Same sign as ASM, agreeing to GBD discretisation (~few %); the
        # pre-fix conjugate would give d_gbd ~ -d_asm (diff ~ 2*d_asm).
        assert abs(wrap(d_asm - d_gbd)) < 0.15 * abs(d_asm) + 0.02, (
            f"GBD radial phase {d_gbd:+.4f} must match ASM {d_asm:+.4f} "
            f"(same sign); a conjugated phase would be ~{-d_asm:+.4f}.")


def test_gbd_freespace_intensity_unchanged_by_phase_fix():
    """The F-1 phase fix must not perturb |E| (it only flips Re(Q))."""
    N, wl, dx, w0 = 128, 633e-9, 4e-6, 24e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X * X + Y * Y) / w0 ** 2).astype(np.complex128)
    Eg = la.propagate_gbd_freespace(E0, dx, z=3e-3, wavelength=wl, sample_step=2)
    Ea = la.angular_spectrum_propagate(E0, 3e-3, wl, dx)
    # Peak intensities agree to a few percent (GBD vs ASM envelope).
    assert abs(np.abs(Eg).max() - np.abs(Ea).max()) < 0.1 * np.abs(Ea).max()


# ----------------------------------------------------------------------
# F-2: seidel.compute_pupils must not crash on an internal stop
# ----------------------------------------------------------------------

def test_compute_pupils_internal_stop_no_unbound_local():
    """A system whose aperture stop is NOT surface 0 must return finite
    pupil data, not raise UnboundLocalError on ``ep_z``."""
    import lumenairy.raytrace.seidel as sd
    S = sd.Surface
    surfaces = [
        S(radius=0.05, thickness=0.005, glass_before='air',
          glass_after='N-BK7', semi_diameter=0.01, surf_num=0),
        S(radius=np.inf, thickness=0.02, glass_before='N-BK7',
          glass_after='air', semi_diameter=0.008, is_stop=True, surf_num=1),
        S(radius=-0.05, thickness=0.1, glass_before='air',
          glass_after='air', semi_diameter=0.01, surf_num=2),
    ]
    info = sd.compute_pupils(surfaces, 550e-9)
    assert np.isfinite(info.ep_z), "ep_z must be assigned/finite on internal stop"
    assert np.isfinite(info.ep_radius)
    assert np.isfinite(info.xp_z)


# ----------------------------------------------------------------------
# F-3: rayleigh_sommerfeld_propagate must return an independent array
# ----------------------------------------------------------------------

def test_rs_propagate_returns_independent_copy():
    """Repeated RS calls at the same grid must not corrupt earlier
    results -- the return must be a copy, not a view into the reused
    pyFFTW ping-pong buffer."""
    N, wl, dx, w0 = 128, 633e-9, 3e-6, 18e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    R2 = X * X + Y * Y

    def mk(scale):
        return (np.exp(-R2 / w0 ** 2) * scale).astype(np.complex128)

    A = la.rayleigh_sommerfeld_propagate(mk(1.0), 1e-3, wl, dx)
    A0 = A.copy()
    for s in (2.0, 3.0, 4.0, 5.0):
        la.rayleigh_sommerfeld_propagate(mk(s), 1e-3, wl, dx)
    assert np.array_equal(A, A0), (
        "Earlier RS result was mutated by later same-grid RS calls -- "
        "the return aliases the reused inverse-FFT buffer.")


# ----------------------------------------------------------------------
# F-28: through_focus_scan (NumPy) must populate the best-focus summary
# ----------------------------------------------------------------------

def test_through_focus_numpy_populates_best_focus_fields():
    """The NumPy through_focus path must set best_focus_strehl /
    best_focus_spot (the JAX twin already did); pre-fix they were left
    at the dataclass NaN default."""
    N, wl, dx = 96, 633e-9, 2e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X * X + Y * Y) / (12e-6) ** 2).astype(np.complex128)
    z_values = np.linspace(-2e-4, 2e-4, 9)
    # ideal_peak enables the Strehl ratio (relative to a reference peak).
    ideal_peak = float((np.abs(la.angular_spectrum_propagate(E, 0.0, wl, dx)) ** 2).max())
    res = la.through_focus_scan(E, dx, wl, z_values, ideal_peak=ideal_peak)
    assert np.isfinite(res.best_focus_strehl), "best_focus_strehl must be populated"
    assert np.isfinite(res.best_focus_spot), "best_focus_spot must be populated"
    assert res.best_focus_spot > 0.0


# ----------------------------------------------------------------------
# F-30/F-31: backend RandomState JAX dtype parity with the x64-aware default
# ----------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
def test_random_jax_default_dtype_is_x64_aware():
    """The JAX RNG default dtype must follow jax_enable_x64 (matching
    NumPy/CuPy when x64 is on), not be hard-coded to 32-bit."""
    import jax
    import jax.numpy as jnp

    from lumenairy.backend import RandomState
    rs = RandomState(rng=jax.random.PRNGKey(0))
    assert rs.backend == 'jax'
    u = rs.uniform((4,))
    n = rs.normal((4,))
    i = rs.integers((4,), low=0, high=10)
    assert u.dtype == jnp.result_type(float), (
        f"uniform default {u.dtype} must equal x64-aware result_type(float) "
        f"{jnp.result_type(float)}")
    assert n.dtype == jnp.result_type(float)
    assert i.dtype == jnp.result_type(int), (
        f"integers default {i.dtype} must equal result_type(int) "
        f"{jnp.result_type(int)}")


# ----------------------------------------------------------------------
# P1-1 / P3-1: JAX trace direction-aware root pick == NumPy trace
# ----------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
def test_jax_trace_refractive_matches_numpy():
    """trace_jax must agree with the NumPy ``trace`` on a refractive
    singlet after the v5.4.6 direction-aware root-pick edit (P1-1/P3-1).

    Note: the wrong-root scenario the prior audit described (Cassegrain /
    backward mirror leg) is NOT reachable via the public ``trace_jax``
    today -- it raises NotImplementedError for ``is_mirror`` surfaces.  The
    direction-aware fix is therefore PRE-EMPTIVE (the kernel is now correct
    for when reflective JAX trace lands, mirroring the v5.4.1 NumPy fix that
    is pinned by test_v5_4_1_raytrace_mirror_backward_ray).  This test pins
    that the edit did not regress the supported forward-refractive path."""
    from lumenairy.raytrace import RayBundle, surfaces_from_prescription, trace
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax

    singlet = la.make_singlet(R1=20e-3, R2=float('inf'), d=2e-3,
                              glass='N-BK7', aperture=4e-3)
    x0 = np.array([1.0e-3, -0.7e-3, 0.3e-3])
    y0 = np.array([0.0, 0.5e-3, -0.4e-3])
    zeros = np.zeros_like(x0)
    L0 = np.zeros_like(x0)
    M0 = np.zeros_like(x0)
    N0 = np.ones_like(x0)

    surfaces = surfaces_from_prescription(singlet)
    rays = RayBundle(
        x=x0.copy(), y=y0.copy(), z=zeros.copy(),
        L=L0.copy(), M=M0.copy(), N=N0.copy(),
        wavelength=633e-9, alive=np.ones_like(x0, dtype=bool),
        opd=zeros.copy(),
    )
    res = trace(rays, surfaces, wavelength=633e-9).ray_history[-1]

    state = make_jax_ray_state(x=x0, y=y0, z=zeros, L=L0, M=M0, N=N0)
    out = trace_jax(state, singlet, 633e-9)

    assert np.allclose(np.asarray(out.x), res.x, atol=1e-9), (
        "trace_jax x must match NumPy trace on the refractive path")
    assert np.allclose(np.asarray(out.y), res.y, atol=1e-9)
    assert np.allclose(np.asarray(out.z), res.z, atol=1e-9)
