"""G04 guards -- propagator robustness guards (v5.24.2 exhaustive audit).

Regression tests for the ``G04-guards-prop`` remediation group.  Each test
fails on the pre-remediation code and passes after, and prefers an INDEPENDENT
oracle (a hand formula / analytic argument) over the code's own path:

* S2-15 -- FGA auto ``n_p`` clamp at 61 re-enters the under-sampled ``dp``
  regime for ``p_max > ~0.24``; ``_resolve_sampling`` now warns when the
  post-clamp spacing exceeds ``_DP_TARGET``.
* S2-17 -- GBD tensor free-space amplitude branch assumes ``Im(lambda) < 0``;
  ``propagate_beamlets_freespace`` now warns when a curvature eigenvalue
  violates it (principal-sqrt sign-flip risk).
* S2-18 -- Maslov Levin integrator now WARNS when the depth-6 per-pixel
  fallback still cannot meet the residual tolerance (was: only a progress
  string).
* S2-20 -- ``reconstruct_field_from_beamlets`` warns on the dense
  ``window=None`` footgun at large workloads; the propagate entry points warn
  when an already-tilted input is decomposed with ``direction_sampling=False``.
* S3-12 -- JAX ``_transfer_jax`` freezes DEAD rays (position + OPL) to match
  the NumPy backend instead of advancing them.
"""
import warnings

import numpy as np
import pytest

LAM = 0.633e-6


# ---------------------------------------------------------------------------
# S2-15  FGA auto n_p clamp re-enters the under-sampled dp regime
# ---------------------------------------------------------------------------
def test_s2_15_fga_np_clamp_warns_when_dp_exceeds_target():
    from lumenairy.propagators.fga import _DP_TARGET, _resolve_sampling

    E = np.ones((8, 8), dtype=complex)
    # Explicit wide p_max (> ~0.24) with n_p auto -> the <=61 cap forces a
    # spacing coarser than the target.  Independent oracle: with the cap the
    # achieved dp = 2*p_max/(n_p-1) exceeds _DP_TARGET.
    p_max_in = 0.30
    with pytest.warns(RuntimeWarning, match="under-sampled"):
        p_max, n_p = _resolve_sampling(
            E, 1e-6, 1e-6, LAM, 1e-6, None, p_max_in, None)
    assert n_p == 61
    dp_achieved = 2.0 * p_max / (n_p - 1)
    assert dp_achieved > _DP_TARGET          # oracle: the cap under-samples


def test_s2_15_no_warn_when_dp_within_target():
    from lumenairy.propagators.fga import _resolve_sampling

    E = np.ones((8, 8), dtype=complex)
    # A modest p_max resolves WITHOUT hitting the 61 cap -> the clamp does not
    # bite, so the under-sampling warning must stay silent (any small dp offset
    # here is the auto-sizer's odd-rounding, not the clamp footgun).
    p_max_in = 0.10
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _p_max, n_p = _resolve_sampling(
            E, 1e-6, 1e-6, LAM, 1e-6, None, p_max_in, None)
    assert n_p < 61
    assert not any("under-sampled" in str(w.message) for w in rec)


# ---------------------------------------------------------------------------
# S2-17  GBD tensor free-space principal-sqrt branch assumes Im(lambda) < 0
# ---------------------------------------------------------------------------
def _tensor_bundle(lam0, lam1):
    """BeamletBundle with a diagonal tensor Q = diag(lam0, lam1) launched along
    +z (so the free-space leg is t = z_distance).  A diagonal 2x2 has its
    diagonal as eigenvalues, giving direct control over the sqrt argument."""
    from lumenairy.propagators.gbd import BeamletBundle

    n = 4
    pos = np.zeros((n, 3))
    dirs = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    Q = np.zeros((n, 2, 2), dtype=complex)
    Q[:, 0, 0] = lam0
    Q[:, 1, 1] = lam1
    amp = np.ones(n, dtype=complex)
    w0 = np.full(n, 1e-6)
    return BeamletBundle(positions=pos, directions=dirs, Q=Q,
                         amplitude=amp, waist0=w0)


_ZR = np.pi * (1e-6) ** 2 / LAM      # physical beamlet Rayleigh range


def test_s2_17_tensor_branch_guard_warns_on_branch_cut():
    from lumenairy.propagators.gbd import propagate_beamlets_freespace

    # A NEAR-REAL curvature eigenvalue driven past its focus: lam0 = -1000 with
    # a negligible imaginary part; at z=2e-3 the sqrt argument is
    # w = 1 + t*lam0 = 1 - 2 = -1 (on the negative real branch cut).  Independent
    # oracle: Re(w) < 0 and Im(w) ~ 0 -> the principal-sqrt sign is ambiguous.
    bad = _tensor_bundle(-1000.0 - 1e-9j, -1j / _ZR)
    with pytest.warns(RuntimeWarning, match="negative real axis"):
        propagate_beamlets_freespace(bad, 2e-3, LAM)


def test_s2_17_tensor_branch_guard_silent_on_physical_beam():
    from lumenairy.propagators.gbd import propagate_beamlets_freespace

    good = _tensor_bundle(-1j / _ZR, -1j / _ZR)   # physical (Im < 0), off cut
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        propagate_beamlets_freespace(good, 2e-3, LAM)
    assert not any("negative real axis" in str(w.message) for w in rec)


def test_s2_17_tensor_branch_guard_silent_on_reflected_frame():
    from lumenairy.propagators.gbd import propagate_beamlets_freespace

    # A fold / periscope reflection legitimately flips Im(lambda) POSITIVE, but
    # the sqrt argument w = 1 + t*(+i/zR) stays far off the branch cut
    # (Re(w) = 1 > 0).  The precise guard must NOT false-fire here (the crude
    # "Im >= 0" form did, on the folded-periscope path).
    reflected = _tensor_bundle(+1j / _ZR, -1j / _ZR)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        propagate_beamlets_freespace(reflected, 2e-3, LAM)
    assert not any("negative real axis" in str(w.message) for w in rec)


# ---------------------------------------------------------------------------
# S2-18  Maslov Levin integrator warns on over-tolerance fallback pixels
# ---------------------------------------------------------------------------
def test_s2_18_levin_over_tolerance_warns():
    # The over-tolerance detection the Levin fallback loop now performs is
    # factored into ``_warn_levin_over_tolerance`` (the loop fills the achieved
    # residual bounds + per-pixel tolerances and calls it).  Driving the full
    # integrator to over-tolerance end-to-end costs minutes (the depth-12 deep
    # re-pass refines to its 200k pair-box cap before the per-pixel fallback),
    # so the warning LOGIC is verified directly here with an independent oracle
    # (over-tolerance <=> achieved bound > tolerance).
    from lumenairy.elements.lenses_maslov import _warn_levin_over_tolerance

    # One pixel's bound (1e-3) exceeds its tolerance (1e-6); the other meets it.
    with pytest.warns(RuntimeWarning, match="did not meet the residual"):
        n_over = _warn_levin_over_tolerance(
            np.array([1e-3, 1e-9]), np.array([1e-6, 1e-6]), 10, 1e-3)
    assert n_over == 1


def test_s2_18_levin_no_warn_when_within_tolerance():
    from lumenairy.elements.lenses_maslov import _warn_levin_over_tolerance

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        n_over = _warn_levin_over_tolerance(
            np.array([1e-9, 5e-8]), np.array([1e-6, 1e-6]), 10, 1e-3)
    assert n_over == 0
    assert not any("did not meet" in str(w.message) for w in rec)


def test_s2_18_levin_dark_pixels_ignored():
    # tolerance <= 0 marks a dark / zero-amplitude pixel (f_scale == 0); it must
    # never count as over-tolerance even with a nonzero bound.
    from lumenairy.elements.lenses_maslov import _warn_levin_over_tolerance

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        n_over = _warn_levin_over_tolerance(
            np.array([1.0]), np.array([0.0]), 5, 1e-3)
    assert n_over == 0
    assert not any("did not meet" in str(w.message) for w in rec)


# ---------------------------------------------------------------------------
# S2-20  reconstruct window=None dense footgun + tilted-input direction_sampling
# ---------------------------------------------------------------------------
def _scalar_bundle(n, x_extent=1e-4):
    from lumenairy.propagators.gbd import BeamletBundle

    zR = np.pi * (1e-6) ** 2 / LAM
    xs = np.linspace(-x_extent, x_extent, n)
    pos = np.stack([xs, np.zeros(n), np.zeros(n)], axis=-1)
    dirs = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    Q = np.full(n, -1j / zR, dtype=complex)
    amp = np.ones(n, dtype=complex)
    w0 = np.full(n, 1e-6)
    return BeamletBundle(positions=pos, directions=dirs, Q=Q,
                         amplitude=amp, waist0=w0)


def test_s2_20_reconstruct_dense_footgun_warns():
    from lumenairy.propagators.gbd import (
        _GBD_DENSE_WARN_WORK,
        reconstruct_field_from_beamlets,
    )

    # Large workload with window=None -> dense O(beamlets*Ny*Nx) path warns.
    Ny = Nx = 64
    n = 30000
    assert float(Ny) * float(Nx) * float(n) > _GBD_DENSE_WARN_WORK   # oracle
    bundle = _scalar_bundle(n)
    with pytest.warns(RuntimeWarning, match="dense"):
        reconstruct_field_from_beamlets(
            bundle, Ny=Ny, Nx=Nx, dx=2e-6, wavelength=LAM, window=None)


def test_s2_20_reconstruct_small_workload_no_warn():
    from lumenairy.propagators.gbd import reconstruct_field_from_beamlets

    bundle = _scalar_bundle(10)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        reconstruct_field_from_beamlets(
            bundle, Ny=8, Nx=8, dx=2e-6, wavelength=LAM, window=None)
    assert not any("dense" in str(w.message) for w in rec)


def _tilted_field(n=32, dx=5e-8, tilt=0.1):
    x = (np.arange(n) - n / 2) * dx
    k = 2.0 * np.pi / LAM
    ramp = np.exp(1j * k * tilt * x)[None, :]
    return np.broadcast_to(ramp, (n, n)).astype(complex).copy()


def test_s2_20_mean_input_tilt_matches_hand_formula():
    from lumenairy.propagators.gbd import _mean_input_tilt

    dx = 5e-8
    tilt = 0.1
    E = _tilted_field(n=32, dx=dx, tilt=tilt)
    est = _mean_input_tilt(E, dx, dx, LAM)
    # Central-difference gradient of exp(i*a*x) recovers sin(a*dx)/(dx) for the
    # local wavevector; independent oracle for the intensity-weighted mean.
    a = 2.0 * np.pi / LAM * tilt
    oracle = np.sin(a * dx) / (dx * 2.0 * np.pi / LAM)
    assert abs(est - oracle) < 1e-3
    assert est > 0.02


def test_s2_20_tilted_input_warns_without_direction_sampling():
    from lumenairy.propagators.gbd import propagate_gbd_freespace

    E = _tilted_field()
    with pytest.warns(RuntimeWarning, match="direction_sampling"):
        propagate_gbd_freespace(E, 5e-8, z=1e-4, wavelength=LAM,
                                direction_sampling=False)


def test_s2_20_tilted_input_no_warn_with_direction_sampling():
    from lumenairy.propagators.gbd import propagate_gbd_freespace

    E = _tilted_field()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        propagate_gbd_freespace(E, 5e-8, z=1e-4, wavelength=LAM,
                                direction_sampling=True)
    assert not any("direction_sampling" in str(w.message) for w in rec)


def test_s2_20_flat_input_no_warn():
    from lumenairy.propagators.gbd import propagate_gbd_freespace

    E = np.ones((32, 32), dtype=complex)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        propagate_gbd_freespace(E, 5e-8, z=1e-4, wavelength=LAM,
                                direction_sampling=False)
    assert not any("direction_sampling" in str(w.message) for w in rec)


# ---------------------------------------------------------------------------
# S3-12  JAX _transfer_jax freezes dead rays (position + OPL) like NumPy
# ---------------------------------------------------------------------------
def test_s3_12_jax_transfer_freezes_dead_rays():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from lumenairy.raytrace.jax_trace import _transfer_jax, make_jax_ray_state

    # Two rays: ray 0 alive, ray 1 dead.  Give the dead ray a nonzero direction
    # and OPL so an unconditional advance WOULD move it.
    x = jnp.array([0.1e-3, 0.2e-3])
    y = jnp.array([-0.1e-3, 0.3e-3])
    z = jnp.array([0.0, 0.0])
    L = jnp.array([0.02, 0.05])
    M = jnp.array([-0.03, 0.04])
    N = jnp.sqrt(1.0 - L ** 2 - M ** 2)
    opd = jnp.array([1e-6, 2e-6])
    alive = jnp.array([True, False])
    state = make_jax_ray_state(x, y, z, L, M, N, opd=opd, alive=alive)

    thickness = 5e-3
    n_medium = 1.5
    out = _transfer_jax(state, thickness, n_medium)

    ox = np.asarray(out.x)
    oy = np.asarray(out.y)
    oz = np.asarray(out.z)
    oopd = np.asarray(out.opd)

    # Dead ray (index 1): position + OPL frozen EXACTLY at the input state
    # (leg == 0 for a dead ray), regardless of the backend's float width.
    assert ox[1] == float(state.x[1])
    assert oy[1] == float(state.y[1])
    assert oz[1] == float(state.z[1])
    assert oopd[1] == float(state.opd[1])

    # Alive ray (index 0): advanced by the hand formula (float32 backend).
    assert ox[0] == pytest.approx(float(x[0]) + float(L[0]) * thickness, rel=1e-5)
    assert oy[0] == pytest.approx(float(y[0]) + float(M[0]) * thickness, rel=1e-5)
    assert oopd[0] == pytest.approx(
        float(opd[0]) + n_medium * thickness, rel=1e-5)


def test_s3_12_jax_transfer_matches_numpy_dead_ray_transverse():
    pytest.importorskip("jax")
    from types import SimpleNamespace

    import jax.numpy as jnp

    from lumenairy.raytrace.intersection import _transfer as _transfer_np
    from lumenairy.raytrace.jax_trace import _transfer_jax, make_jax_ray_state

    x = np.array([0.15e-3, 0.22e-3])
    y = np.array([-0.11e-3, 0.31e-3])
    L = np.array([0.02, 0.05])
    M = np.array([-0.03, 0.04])
    Nn = np.sqrt(1.0 - L ** 2 - M ** 2)
    opd = np.array([1e-6, 2e-6])
    alive = np.array([True, False])
    thickness = 4e-3
    n_medium = 1.0

    r = SimpleNamespace(
        x=x.copy(), y=y.copy(), z=np.zeros(2),
        L=L.copy(), M=M.copy(), N=Nn.copy(),
        opd=opd.copy(), alive=alive.copy())
    _transfer_np(r, thickness, n_medium)

    state = make_jax_ray_state(
        jnp.asarray(x), jnp.asarray(y), jnp.zeros(2),
        jnp.asarray(L), jnp.asarray(M), jnp.asarray(Nn),
        opd=jnp.asarray(opd), alive=jnp.asarray(alive))
    out = _transfer_jax(state, thickness, n_medium)

    # The dead ray's transverse position + OPL agree across backends now (to
    # float32 precision -- the JAX default width).  Before the fix the JAX path
    # advanced the dead ray while NumPy froze it, a gross (thickness-scale)
    # mismatch, so a loose relative tolerance still distinguishes the two.
    assert np.asarray(out.x)[1] == pytest.approx(r.x[1], rel=1e-5)
    assert np.asarray(out.y)[1] == pytest.approx(r.y[1], rel=1e-5)
    assert np.asarray(out.opd)[1] == pytest.approx(r.opd[1], rel=1e-5)
