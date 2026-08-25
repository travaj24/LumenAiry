"""v5.21 GBD speed/memory/accuracy features + Maslov default.

Covers:
 * windowed (bounded-support) reconstruction == dense sum to machine precision,
   across scalar / tensor / direction-ramp / anamorphic beamlets;
 * the dense-path memory-budget auto-chunk leaves small-N results unchanged;
 * decompose_field_adaptive: partition-of-unity (energy) + matches uniform-fine
   edge fidelity at far fewer beamlets;
 * soft-edge (analytic partial-vignetting) aperture: correct half-open-plane
   energy fraction, and improves the hard-aperture Airy-focus accuracy;
 * apply_real_lens_maslov default integration_method is now 'auto'.
"""
import numpy as np
import pytest

from lumenairy.propagators.gbd import (
    BeamletBundle,
    apply_aperture_to_beamlets,
    apply_thin_lens_to_beamlets,
    decompose_field_adaptive,
    decompose_field_to_beamlets,
    propagate_beamlets_freespace,
    reconstruct_field_from_beamlets,
)

LAM = 0.633e-6


def _relerr(A, B):
    return float(np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-300))


def _gauss(N, dx, w0=0.15e-3, tilt=0.0, dy=None):
    dy = dy or dx
    xs = (np.arange(N) - N // 2) * dx
    ys = (np.arange(N) - N // 2) * dy
    X, Y = np.meshgrid(xs, ys)
    E = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    if tilt:
        E = E * np.exp(1j * 2 * np.pi / LAM * tilt * X)
    return E


# --------------------------------------------------------------------------
# windowed reconstruction == dense
# --------------------------------------------------------------------------
@pytest.mark.parametrize("z_prop,lens_f,tilt,aniso", [
    (0.0, None, 0.0, False),          # scalar, at plane
    (30e-3, 30e-3, 0.0, False),       # scalar, near a focus
    (2e-3, None, 0.02, False),        # direction ramp (Husimi tilt)
    (1e-3, None, 0.0, True),          # anamorphic dy != dx
])
def test_windowed_reconstruct_matches_dense(z_prop, lens_f, tilt, aniso):
    N, dx = 96, 5e-6
    dy = 6e-6 if aniso else dx
    E = _gauss(N, dx, tilt=tilt, dy=(dy if aniso else None))
    b = decompose_field_to_beamlets(
        E, dx, wavelength=LAM, dy=(dy if aniso else None),
        sample_step=2, waist_factor=1.5, direction_sampling=bool(tilt))
    if lens_f is not None:
        b = apply_thin_lens_to_beamlets(b, lens_f, LAM)
    if z_prop:
        b = propagate_beamlets_freespace(b, z_prop, LAM)
    dense = reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx, dy=dy,
                                            wavelength=LAM, window=None)
    win = reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx, dy=dy,
                                          wavelength=LAM, window=5.0)
    assert _relerr(win, dense) < 1e-9


def test_windowed_reconstruct_tensor_Q():
    N, dx = 96, 5e-6
    E = _gauss(N, dx)
    b = decompose_field_to_beamlets(E, dx, wavelength=LAM, sample_step=2,
                                    waist_factor=1.5)
    n = len(b)
    zRx = np.pi * (1.5 * dx) ** 2 / LAM
    Q = np.zeros((n, 2, 2), dtype=np.complex128)
    Q[:, 0, 0] = -1j / zRx
    Q[:, 1, 1] = -1j / (1.4 * zRx)
    Q[:, 0, 1] = Q[:, 1, 0] = 0.06j / zRx     # skew astigmatism
    bt = BeamletBundle(positions=b.positions, directions=b.directions, Q=Q,
                       amplitude=b.amplitude, waist0=b.waist0)
    dense = reconstruct_field_from_beamlets(bt, Ny=N, Nx=N, dx=dx,
                                            wavelength=LAM, window=None)
    win = reconstruct_field_from_beamlets(bt, Ny=N, Nx=N, dx=dx,
                                          wavelength=LAM, window=5.0)
    assert _relerr(win, dense) < 1e-9


def test_mem_budget_does_not_change_small_N():
    N, dx = 64, 5e-6
    E = _gauss(N, dx)
    b = decompose_field_to_beamlets(E, dx, wavelength=LAM, sample_step=2,
                                    waist_factor=1.5)
    a = reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx, wavelength=LAM,
                                        window=None, mem_budget_mb=512.0)
    # A tiny budget forces chunk auto-shrink; result must be unchanged to ULP.
    c = reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx, wavelength=LAM,
                                        window=None, mem_budget_mb=1.0)
    assert _relerr(c, a) < 1e-12


# --------------------------------------------------------------------------
# the reconstruct-route decision must not be taken by the machine's free RAM
# --------------------------------------------------------------------------
def _uniform_bundle(N=64, dx=5e-6):
    """A bundle the FFT-convolution route DOES apply to (uniform Q, uniform
    direction, centres on grid) -- so the route decision is live."""
    E = _gauss(N, dx)
    return E, dx, decompose_field_to_beamlets(
        E, dx, wavelength=LAM, sample_step=2, waist_factor=1.5)


def test_fft_route_decision_falls_back_on_inspection_failure():
    """The trace-safe half of the contract: an applicability check that cannot
    INSPECT the bundle (a JAX tracer raises on ``np.asarray``) falls back to
    the summed route and still returns a field."""
    from lumenairy.propagators import gbd as _gbd_mod
    E, dx, b = _uniform_bundle()
    N = E.shape[0]
    ref = reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx,
                                          wavelength=LAM, window=5.0)
    orig = _gbd_mod._fft_applicable_impl

    def _raise_value(*a, **k):
        raise ValueError('pretend tracer')

    _gbd_mod._fft_applicable_impl = _raise_value
    try:
        out = reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx,
                                              wavelength=LAM, window=5.0)
    finally:
        _gbd_mod._fft_applicable_impl = orig
    assert np.all(np.isfinite(out))
    # The two routes are the same sum in a different order, so they agree to
    # rounding and NOT bit for bit -- which is exactly why the DECISION between
    # them must not be taken by anything but the data (see below).  Measured
    # 2026-08-24 over three bundles the FFT route applies to: 6.61e-16 /
    # 9.79e-16 / 5.28e-16 relative, byte-identical on none of them.  The bar
    # sits ~3 decades above that and ~7 below the windowing truncation the
    # module already tolerates (exp(-25) ~ 1.4e-11).
    assert _relerr(out, ref) < 1e-12


def test_fft_route_decision_does_not_swallow_memory_error():
    """P4 close-out (2026-08-24): ``MemoryError`` is a subclass of
    ``Exception``, so the bare handler that exists for JAX tracers used to
    catch it -- and a reconstruct would then take a DIFFERENT summation route
    because the box was short of memory at that instant.  Route-by-free-RAM is
    the silent-wrongness shape; the error propagates instead.

    Two-sided with the test above: an inspection failure that really is an
    inspection failure still falls back."""
    from lumenairy.propagators import gbd as _gbd_mod
    E, dx, b = _uniform_bundle()
    N = E.shape[0]
    orig = _gbd_mod._fft_applicable_impl

    def _raise_mem(*a, **k):
        raise MemoryError('pretend the box is full')

    _gbd_mod._fft_applicable_impl = _raise_mem
    try:
        with pytest.raises(MemoryError):
            reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx,
                                            wavelength=LAM, window=5.0)
    finally:
        _gbd_mod._fft_applicable_impl = orig


# --------------------------------------------------------------------------
# adaptive decomposition
# --------------------------------------------------------------------------
def test_adaptive_matches_uniform_fine_at_fewer_beamlets():
    """Residual-refined adaptive reaches uniform-fine edge fidelity with far
    fewer beamlets than the uniform-fine grid."""
    N, dx = 128, 5e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    r = np.sqrt(X ** 2 + Y ** 2)
    E = (r <= 0.2e-3).astype(np.complex128)         # hard-edged top hat

    def recon(bundle):
        return reconstruct_field_from_beamlets(bundle, Ny=N, Nx=N, dx=dx,
                                               wavelength=LAM, window=5.0)
    fine = decompose_field_to_beamlets(E, dx, wavelength=LAM, sample_step=1,
                                       waist_factor=1.5)
    adap, st = decompose_field_adaptive(E, dx, wavelength=LAM, base_step=4,
                                        refine_step=1, return_stats=True)
    e_fine = _relerr(recon(fine), E)
    e_adap = _relerr(recon(adap), E)
    # adaptive is at least as faithful as uniform-fine, with fewer beamlets
    assert e_adap <= e_fine * 1.10
    assert st['n_total'] < 0.4 * len(fine)


def test_adaptive_partition_of_unity_no_double_count():
    """Residual refinement ADDS a correction (fine carries E - recon(coarse)),
    so reconstruct(adaptive) at z=0 conserves energy / matches the field better
    than the coarse alone -- proving no double counting at the coarse/fine seam
    (a double-counted overlay would inflate the energy)."""
    N, dx = 96, 5e-6
    E = _gauss(N, dx, w0=0.12e-3)

    def recon(bundle):
        return reconstruct_field_from_beamlets(bundle, Ny=N, Nx=N, dx=dx,
                                               wavelength=LAM, window=5.0)
    coarse = decompose_field_to_beamlets(E, dx, wavelength=LAM, sample_step=4,
                                         waist_factor=1.5 * 4)
    adap = decompose_field_adaptive(E, dx, wavelength=LAM, base_step=4,
                                    refine_step=1)
    e_in = float(np.sum(np.abs(E) ** 2))
    e_adap = float(np.sum(np.abs(recon(adap)) ** 2))
    # adaptive reconstruction energy is within a few % of the input (not
    # inflated -> no double counting) and closer to E than the coarse alone
    assert abs(e_adap - e_in) / e_in < 0.05
    assert _relerr(recon(adap), E) < _relerr(recon(coarse), E)


# --------------------------------------------------------------------------
# soft-edge aperture
# --------------------------------------------------------------------------
def test_soft_edge_energy_fraction_straight_edge():
    """A single beamlet far from a straight (rectangular) edge passes ~all its
    energy inside and ~none outside; on the edge it passes ~half."""
    from lumenairy.propagators.gbd import _erf_xp
    # erf sanity across backends
    assert abs(float(_erf_xp(np, np.array([0.0]))[0])) < 1e-12
    assert abs(float(_erf_xp(np, np.array([5.0]))[0]) - 1.0) < 1e-6

    N, dx = 64, 5e-6
    E = _gauss(N, dx, w0=0.08e-3)
    b = decompose_field_to_beamlets(E, dx, wavelength=LAM, sample_step=2,
                                    waist_factor=1.5)
    # rectangular half-width huge -> everything passes (soft ~ binary ~ 1)
    big = apply_aperture_to_beamlets(b, 1.0, shape='rectangular', soft_edge=True)
    assert _relerr(big.amplitude, b.amplitude) < 1e-4
    # half-width 0 (edge at centre): each beamlet centred on axis passes ~1/2
    zero = apply_aperture_to_beamlets(b, 0.0, shape='rectangular', soft_edge=True)
    on_axis = np.argmin(np.abs(b.positions[:, 0]) + np.abs(b.positions[:, 1]))
    frac = abs(zero.amplitude[on_axis] / b.amplitude[on_axis])
    assert abs(frac - 0.5) < 0.05


# v5.32.1 (AUDIT_CI_TEST_TIME_2026_08_03 §4/chunk 6): 27.6 s of this file's
# 40.7 s -- a full ASM oracle plus two GBD focus reconstructions to compare
# soft-edge vs binary vignetting.  Pure-NumPy comparison, no version contract.
@pytest.mark.slow
def test_soft_edge_improves_hard_aperture_focus():
    """Soft-edge aperture lowers the hard-aperture Airy-focus intensity error
    vs binary vignetting."""
    from lumenairy.propagators.asm import angular_spectrum_propagate
    from lumenairy.propagators.gbd import propagate_gbd_thin_lens
    N, dx, f, R = 160, 6e-6, 30e-3, 0.5e-3
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    r = np.sqrt(X ** 2 + Y ** 2)
    r2 = X ** 2 + Y ** 2
    E0 = (r <= R).astype(np.complex128)
    k = 2 * np.pi / LAM
    Ea = angular_spectrum_propagate(E0 * np.exp(-1j * k * r2 / (2 * f)), f, LAM, dx)

    def relI(A, B):
        IA = np.abs(A) ** 2
        IB = np.abs(B) ** 2
        m = IB > 0.01 * IB.max()
        return float(np.linalg.norm((IA - IB)[m]) / np.linalg.norm(IB[m]))
    kw = dict(z_to_lens=0.0, focal_length=f, z_lens_to_output=f, wavelength=LAM,
              output_dx=dx, sample_step=2, waist_factor=3.0,
              aperture_semi_diameter=R)
    e_bin = relI(propagate_gbd_thin_lens(E0, dx, aperture_soft_edge=False, **kw), Ea)
    e_soft = relI(propagate_gbd_thin_lens(E0, dx, aperture_soft_edge=True, **kw), Ea)
    assert e_soft < e_bin           # soft-edge is strictly better here
    assert e_soft < 0.03            # and lands well under the binary ~3.3%


def _single_beamlet(Q, w0, lam):
    """One on-axis beamlet launched at its waist (Q, waist0=w0)."""
    return BeamletBundle(
        positions=np.zeros((1, 3), dtype=np.float64),
        directions=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        Q=Q,
        amplitude=np.ones((1,), dtype=np.complex128),
        waist0=np.asarray(w0, dtype=np.float64).reshape(-1),
    )


def test_soft_edge_uses_propagated_width_not_launch_waist0():
    """Audit S2-5: after a free-space leg the soft-edge aperture must vignette
    on the beamlet's *propagated* 1/e amplitude radius (derived from Im(Q)),
    not the stale launch-time ``waist0``.

    Independent oracle: reconstruct the single propagated beamlet on a fine
    grid and read the 1/e amplitude radius directly off ``|E|`` (this path
    never touches the alpha = 0.5 k lam_min formula the aperture uses); the
    analytic ABCD width ``w0*sqrt(1+(z/zR)^2)`` must match it, and the
    soft-edge vignetting fraction with ``wavelength`` supplied must equal
    ``0.5(1+erf(d*sqrt2/w_true))`` using THAT propagated width -- while the
    legacy ``wavelength=None`` path (waist0) collapses to a near-hard cut.
    """
    from lumenairy.propagators.gbd import _erf_xp
    lam = LAM
    w0 = 8e-6
    zR = np.pi * w0 ** 2 / lam
    z = 10.0 * zR                                 # 10 Rayleigh ranges
    w_true = w0 * np.sqrt(1.0 + (z / zR) ** 2)     # ~80 um analytic ABCD width

    b0 = _single_beamlet(np.array([-1j / zR], dtype=np.complex128), w0, lam)
    b = propagate_beamlets_freespace(b0, z_distance=z, wavelength=lam)

    # Independent grid oracle: 1/e amplitude radius of the rendered beamlet.
    Ng, dxg = 401, 2e-6
    field = reconstruct_field_from_beamlets(b, Ny=Ng, Nx=Ng, dx=dxg,
                                            wavelength=lam)
    prof = np.abs(field)[Ng // 2, Ng // 2:]        # x >= 0 half of central row
    xr = np.arange(Ng - Ng // 2) * dxg
    w_grid = xr[int(np.argmin(np.abs(prof - prof[0] / np.e)))]
    assert abs(w_grid - w_true) / w_true < 0.02    # analytic width is physical
    assert w_true > 8.0 * w0                        # genuinely widened (~10x)

    # On-axis beamlet, circular stop with d = semi_diameter = 0.3*w_true.
    d = 0.3 * w_true
    expected = 0.5 * (1.0 + float(
        _erf_xp(np, np.array([d * np.sqrt(2.0) / w_true]))[0]))

    out_fix = apply_aperture_to_beamlets(b, d, shape='circular',
                                         soft_edge=True, wavelength=lam)
    frac_fix = abs(out_fix.amplitude[0] / b.amplitude[0])
    assert abs(frac_fix - expected) < 2e-3          # uses the propagated width

    out_stale = apply_aperture_to_beamlets(b, d, shape='circular',
                                           soft_edge=True)   # wavelength=None
    frac_stale = abs(out_stale.amplitude[0] / b.amplitude[0])
    assert frac_stale > 0.999                        # stale waist0 -> hard cut
    assert frac_fix < 0.80                           # fix is a real partial
    assert (frac_stale - frac_fix) > 0.15            # materially different


def test_soft_edge_tensor_width_tracks_wider_axis():
    """Tensor-Q branch of the S2-5 fix: the propagated soft-edge width is the
    widest principal axis (smallest eigenvalue of -Im(Q)); the aperture
    fraction matches an erf built from that independently-computed width."""
    from lumenairy.propagators.gbd import _erf_xp
    lam = LAM
    wx, wy = 8e-6, 16e-6
    zRx = np.pi * wx ** 2 / lam
    zRy = np.pi * wy ** 2 / lam
    Q = np.array([[[-1j / zRx, 0.0], [0.0, -1j / zRy]]], dtype=np.complex128)
    b0 = _single_beamlet(Q, np.sqrt(wx * wy), lam)
    z = 8.0 * zRx
    b = propagate_beamlets_freespace(b0, z_distance=z, wavelength=lam)

    wx_z = wx * np.sqrt(1.0 + (z / zRx) ** 2)
    wy_z = wy * np.sqrt(1.0 + (z / zRy) ** 2)
    w_wide = max(wx_z, wy_z)                          # min-eigenvalue width

    d = 0.4 * w_wide
    expected = 0.5 * (1.0 + float(
        _erf_xp(np, np.array([d * np.sqrt(2.0) / w_wide]))[0]))
    out = apply_aperture_to_beamlets(b, d, shape='circular',
                                     soft_edge=True, wavelength=lam)
    frac = abs(out.amplitude[0] / b.amplitude[0])
    assert abs(frac - expected) < 2e-3


# --------------------------------------------------------------------------
# Maslov default
# --------------------------------------------------------------------------
def test_maslov_default_is_auto():
    import inspect

    from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
    sig = inspect.signature(apply_real_lens_maslov)
    assert sig.parameters['integration_method'].default == 'auto'
