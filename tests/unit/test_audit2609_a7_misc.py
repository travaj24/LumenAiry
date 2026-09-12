"""A5 / A6 / A7 -- the P2 and P3 rows of the analysis audit.

AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11 §8 rows A5, A6, A7.

Each test names the defect it pins and the measured before / after.
"""
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.analysis.aberration import caustic_diagnostic
from lumenairy.analysis.ghost import _ring_area_weights, _weighted_median
from lumenairy.analysis.interferometry import simulate_interferogram
from lumenairy.analysis.phase_retrieval import gerchberg_saxton
from lumenairy.analysis.plotting import _auto_extent
from lumenairy.analysis.polychromatic import radial_power_bands
from lumenairy.analysis.through_focus import (single_plane_metrics,
                                              through_focus_scan)
from lumenairy.analysis.zernike import _zernike_basis_cache_key

# ===========================================================================
# A5 -- gerchberg_saxton's reported error was off by N_pix
# ===========================================================================


def _exact_gs_problem(N=64, seed=7):
    """A target for which the supplied initial phase is an EXACT solution,
    so the reported error MUST be able to reach 0."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    source = np.exp(-(X ** 2 + Y ** 2) / 0.3 ** 2)
    phi0 = rng.uniform(-np.pi, np.pi, (N, N))
    target = np.abs(np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(source * np.exp(1j * phi0)))))
    return source, target, phi0


def test_gs_error_reaches_zero_for_an_exact_solution():
    """The metric compared an UNNORMALISED DFT (``sum |F|^2 = N_pix *
    sum |E|^2``) against a target rescaled to the SOURCE power, leaving a
    hard-wired factor N_pix.

    Measured on this fixture: reported error 3.775380e+02 -- 97 % of the
    target energy 3.896184e+02, and flat over 50 iterations -- against
    3.26e-29 after.  Bar: 1e-20, which is 9 decades above the measured
    residual and 22 decades below the pre-fix value.
    """
    source, target, phi0 = _exact_gs_problem()
    _, err = gerchberg_saxton(source, target, n_iter=10, initial_phase=phi0)
    assert err < 1e-20, (
        f'an exact solution reports error {err:.6e}; the metric cannot '
        f'reach zero.')


def test_gs_history_moves_and_does_not_sit_at_the_target_energy():
    source, target, phi0 = _exact_gs_problem()
    _, _, hist = gerchberg_saxton(source, target, n_iter=20,
                                  initial_phase=phi0, return_history=True)
    assert max(hist) < 1e-20


def test_gs_retrieved_phase_is_unchanged_by_the_metric_fix():
    """Both amplitude-replacement steps are scale invariant, so the
    rescale must not move the answer -- only the number reported.

    Bar: the retrieved field (not the raw phase, which is defined modulo
    2 pi at a sample whose amplitude is ~0) reproduces the reference to
    1e-9 relative.
    """
    N = 48
    rng = np.random.default_rng(2)
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    source = np.exp(-(X ** 2 + Y ** 2) / 0.4 ** 2)
    target = ((np.abs(X) < 0.3) & (np.abs(Y) < 0.3)).astype(float)
    phase, _ = gerchberg_saxton(source, target, n_iter=25, seed=1)
    # Reference: the same iteration with the OLD (source-power) scaling.
    rng_ref = np.random.default_rng(1)
    ph = rng_ref.uniform(-np.pi, np.pi, (N, N))
    tgt = target * np.sqrt(np.sum(source ** 2) / np.sum(target ** 2))
    field = source * np.exp(1j * ph)
    for _ in range(25):
        F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
        F = tgt * np.exp(1j * np.angle(F))
        field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F)))
        field = source * np.exp(1j * np.angle(field))
    ref = np.angle(field)
    d = np.abs(np.exp(1j * phase) - np.exp(1j * ref))
    assert float(np.max(d)) < 1e-9


def test_gs_numpy_and_jax_report_the_same_error_scale():
    """The JAX twin never rescaled at all, so the two backends' ``err``
    differed by N_pix despite documenting the same physics."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from lumenairy.analysis.phase_retrieval import gerchberg_saxton_jax
    source, target, phi0 = _exact_gs_problem(N=48, seed=4)
    _, e_np = gerchberg_saxton(source, target, n_iter=10,
                               initial_phase=phi0)
    _, e_jx = gerchberg_saxton_jax(source, target, n_iter=10,
                                   initial_phase=phi0, dtype=np.float64)
    # Both are at the float64 noise floor on an exact solution; what is
    # pinned is that neither carries a factor N_pix = 2304.
    assert e_np < 1e-20 and e_jx < 1e-20


# ===========================================================================
# A5 -- plot_stokes used the x extent for both axes
# ===========================================================================


def test_auto_extent_addresses_pixel_edges_not_sample_centres():
    """``extent`` is the OUTER EDGE of the first and last pixel, while the
    field is sampled on ``(arange(N) - N / 2) * dx``: the edges are half a
    sample outside the first and last sample.  The symmetric
    ``+-N/2*dx`` form drew every pixel centre half a sample to the RIGHT
    (samples [-4..3]*dx drew at [-3.5..3.5]*dx)."""
    N, dx = 8, 1e-6
    ext, unit, scale = _auto_extent(N, dx, 'um')
    x = (np.arange(N) - N / 2) * dx * scale
    centres = ext[0] + (np.arange(N) + 0.5) * (ext[1] - ext[0]) / N
    np.testing.assert_allclose(centres, x, atol=1e-12)


def test_plot_psf_and_auto_extent_use_the_same_convention():
    """``plot_psf`` used ``(x[0], x[-1])`` -- the opposite half-pixel
    error -- so the same field plotted two ways landed a whole pixel
    apart."""
    N, dx = 16, 2e-6
    ext, _, scale = _auto_extent(N, dx, 'um')
    x = (np.arange(N) - N / 2) * dx * scale
    h = 0.5 * dx * scale
    assert ext[0] == pytest.approx(x[0] - h)
    assert ext[1] == pytest.approx(x[-1] + h)


def test_plot_stokes_takes_a_dy():
    """``plot_stokes`` had no ``dy`` and called ``_auto_extent(Nx, dx)``
    for BOTH axes, so a non-square or anamorphic Jones field got a
    mislabelled y axis -- and ``_auto_extent(64, 1e-6)`` equals
    ``_auto_extent(32, 2e-6)``, so the error is invisible in the drawn
    figure."""
    import inspect
    from lumenairy.analysis.plotting import plot_stokes
    assert 'dy' in inspect.signature(plot_stokes).parameters


# ===========================================================================
# A6 -- performance, with the numerics they are required to preserve
# ===========================================================================


def _focusing_field(N=96, dx=6e-6, wl=1.31e-6, f=20e-3, sigma=60e-6):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return (np.exp(-(X ** 2 + Y ** 2) / sigma ** 2)
            * np.exp(-1j * 2 * np.pi / wl * (X ** 2 + Y ** 2) / (2 * f)))


def test_transfer_function_recurrence_engages_only_on_a_uniform_scan():
    """The recurrence ``H(z + dz) = H(z) * H(dz)`` is valid only for
    uniformly spaced z.  On a non-uniform scan the direct ``exp`` must be
    used, and the result must then be bit-identical to the per-plane
    propagator.
    """
    from lumenairy.propagators.propagation import angular_spectrum_propagate
    wl, dx = 1.31e-6, 6e-6
    E = _focusing_field()
    z = np.array([15e-3, 17e-3, 20e-3, 24e-3, 25e-3])      # NOT uniform
    scan = through_focus_scan(E, dx, wl, z, bandlimit=True, verbose=False)
    for i, zi in enumerate(z):
        Ez = angular_spectrum_propagate(E, float(zi), wl, dx, bandlimit=True)
        assert scan.peak_I[i] == float((np.abs(Ez) ** 2).max()), (
            f'non-uniform z must take the direct exp path and stay '
            f'bit-identical; plane {i} drifted.')


def test_transfer_function_recurrence_matches_the_direct_form(recwarn):
    """On a uniform scan the recurrence is used.  Both forms evaluate the
    same exact function and both round the ARGUMENT at ``|kz z| eps / 2``,
    so their difference is bounded by ``2 |kz z|_max eps`` -- neither is
    the more accurate one.  Here ``|kz z|_max = 2 pi / 1.31e-6 * 25e-3 =
    1.20e5`` rad, giving 5.3e-11; measured worst relative metric drift
    1.2e-11.  Bar 1e-9: two decades above the measurement, and six below
    any real defect in the transfer function.
    """
    from lumenairy.propagators.propagation import angular_spectrum_propagate
    wl, dx = 1.31e-6, 6e-6
    E = _focusing_field()
    z = np.linspace(15e-3, 25e-3, 11)
    scan = through_focus_scan(E, dx, wl, z, bandlimit=True, verbose=False)
    for i, zi in enumerate(z):
        Ez = angular_spectrum_propagate(E, float(zi), wl, dx, bandlimit=True)
        ref = float((np.abs(Ez) ** 2).max())
        assert scan.peak_I[i] == pytest.approx(ref, rel=1e-9)


def test_transfer_function_modulus_does_not_drift():
    """A recurrence on unit-modulus factors could bleed energy.  Measured
    ``max | |H| - 1 |`` after 21 steps: 1.8e-15."""
    N, dx, wl = 128, 4e-6, 633e-9
    k = 2 * np.pi / wl
    fx = np.fft.fftshift(np.fft.fftfreq(N, dx)) * 2 * np.pi
    kz2 = k * k - fx[None, :] ** 2 - fx[:, None] ** 2
    prop = kz2 > 0
    kzs = np.sqrt(np.where(prop, kz2, 0.0))
    dz = 1e-4
    H = np.where(prop, np.exp(1j * kzs * dz), 0.0)
    step = H.copy()
    for _ in range(20):
        H = H * step
    assert float(np.max(np.abs(np.abs(H[prop]) - 1.0))) < 1e-13


def test_single_plane_metrics_are_unchanged_by_the_single_pass_rewrite():
    """The default path now shares one ``|E|**2``, one meshgrid and one
    set of moment sums with ``beam_centroid`` / ``beam_d4sigma`` instead
    of building them three times.  Same helper, same order of operations,
    so the numbers must be BIT-identical."""
    from lumenairy.analysis.beam_stats import (beam_centroid, beam_d4sigma,
                                               beam_power)
    E = _focusing_field(N=128)
    dx, wl = 6e-6, 1.31e-6
    m = single_plane_metrics(E, dx, wl)
    cx, cy = beam_centroid(E, dx, dx)
    d4x, d4y = beam_d4sigma(E, dx, dx)
    assert m['centroid_x'] == cx
    assert m['centroid_y'] == cy
    assert m['d4sigma_x'] == d4x
    assert m['d4sigma_y'] == d4y
    assert m['peak_I'] == float((np.abs(E) ** 2).max())


def test_single_plane_metrics_iso_path_is_untouched():
    """ISO 11146 background / aperture conditioning changes the intensity
    the D4sigma moments run on while ``centroid_x/y`` stay whole-grid;
    that split must survive the rewrite."""
    from lumenairy.analysis.beam_stats import beam_centroid, beam_d4sigma
    E = _focusing_field(N=96)
    dx, wl = 6e-6, 1.31e-6
    m = single_plane_metrics(E, dx, wl, background='corner', aperture=True)
    cx, cy = beam_centroid(E, dx, dx)
    d4x, d4y = beam_d4sigma(E, dx, dx, background='corner', aperture=True)
    assert m['centroid_x'] == cx and m['centroid_y'] == cy
    assert m['d4sigma_x'] == d4x and m['d4sigma_y'] == d4y


@pytest.mark.parametrize('n_radii', [1, 8, 64])
def test_radial_power_bands_below_the_crossover_is_bit_identical(n_radii):
    """Small band counts keep the mask-and-sum loop, so every existing
    caller -- ``single_plane_metrics(bucket_radius=...)`` asks for ONE
    radius per plane -- gets the same float64 it always did."""
    N, dx = 256, 2e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (50e-6) ** 2).astype(complex)
    R2 = X ** 2 + Y ** 2
    I = np.abs(E) ** 2
    radii = np.linspace(5e-6, 200e-6, n_radii)
    got = radial_power_bands(E, dx, radii)
    ref = np.array([float(np.sum(I[R2 <= r * r]) * dx * dx) for r in radii])
    np.testing.assert_array_equal(got, ref)


def test_radial_power_bands_above_the_crossover_matches_the_masked_sum():
    """Above the crossover the sorted construction is used.  It sums the
    SAME addends in radius order rather than in grid order, so it differs
    only by float64 summation associativity: the sequential ``cumsum``
    bound is O(Ny*Nx * eps) = 7e-12 relative here, measured 2.1e-13."""
    N, dx = 256, 2e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (50e-6) ** 2).astype(complex)
    R2 = X ** 2 + Y ** 2
    I = np.abs(E) ** 2
    radii = np.linspace(5e-6, 200e-6, 256)
    got = radial_power_bands(E, dx, radii)
    ref = np.array([float(np.sum(I[R2 <= r * r]) * dx * dx) for r in radii])
    np.testing.assert_allclose(got, ref, rtol=1e-11, atol=0)


def test_radial_power_bands_preserves_input_order():
    N, dx = 128, 2e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (40e-6) ** 2).astype(complex)
    radii = np.array([100e-6, 20e-6, 200e-6])
    p = radial_power_bands(E, dx, radii)
    assert p[1] < p[0] < p[2]


# ===========================================================================
# A7 -- the P3 rows
# ===========================================================================


def test_ghost_fifty_percent_radius_is_area_weighted():
    """``retrace_ghost_path`` took the MEDIAN RAY radius as the 50 %
    encircled-energy radius.  ``make_rings`` puts the same ray count on
    every ring, so the areal density falls as 1/r and the median ray is
    not the median of the ENERGY: on a uniform disc it reads 0.500 R
    where the truth is ``1 / sqrt(2) = 0.7071 R``, 29 % low."""
    R = 1.0
    for n_rings, rpr in ((6, 36), (12, 36), (24, 36)):
        xs, ys = [0.0], [0.0]
        for k in range(1, n_rings + 1):
            th = np.linspace(0, 2 * np.pi, rpr, endpoint=False)
            xs.append(R * k / n_rings * np.cos(th))
            ys.append(R * k / n_rings * np.sin(th))
        x = np.concatenate([np.atleast_1d(a) for a in xs])
        y = np.concatenate([np.atleast_1d(a) for a in ys])
        r = np.hypot(x, y)
        w = _ring_area_weights(x, y, R, n_rings)
        assert w.sum() == pytest.approx(1.0)
        est = _weighted_median(r, w)
        # Bar: within one ring spacing of the analytic 0.70711 R.  The
        # estimator can only resolve the energy to the ring grid, and the
        # un-weighted median it replaces is 0.500 for EVERY ring count --
        # i.e. it does not converge at all.
        assert abs(est - 1 / np.sqrt(2)) <= R / n_rings
        assert abs(float(np.median(r)) - 1 / np.sqrt(2)) > 0.15


def test_caustic_diagnostic_flags_complex_eigenvalues():
    """``disc = max(0.25 tr^2 - det, 0)`` silently collapsed a
    complex-conjugate eigenvalue pair onto ``tr / 2``, reporting two
    spurious coincident real eigenvalues (and a wrong Maslov index) for a
    system whose transverse map rotates.  It must at least say so."""
    import lumenairy.analysis.aberration as ab
    src = __import__('inspect').getsource(ab.caustic_diagnostic)
    assert 'n_complex_eig' in src, (
        'the complex-eigenvalue branch must be counted, not clamped away')
    # An axisymmetric system must NOT trip the new warning.
    p = la.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3, glass='N-BK7',
                        aperture=10e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        caustic_diagnostic(p, wavelength=587.6e-9, n_z_per_gap=16)


def test_simulate_interferogram_has_a_dy_and_uses_it():
    """The y tilt ramp was built on ``dx``, so an anamorphic grid got the
    wrong fringe frequency along y."""
    Ny, Nx = 64, 64
    lam = 633e-9
    opd = np.zeros((Ny, Nx))
    dx, dy = 4e-6, 1e-6
    tilt_y = 1.0 / (Ny * dy)             # exactly one fringe down the field
    a = simulate_interferogram(opd, lam, tilt_y=tilt_y, dx=dx, dy=dy)
    b = simulate_interferogram(opd, lam, tilt_y=tilt_y, dx=dx)
    assert not np.allclose(a, b), 'dy is ignored'
    # ``tilt_y = 1 / (Ny * dy)`` is exactly ONE fringe down the field, so
    # the dominant DFT bin of the column profile is 1.  Without dy the
    # ramp is built on dx and the frequency is off by dx / dy = 4.
    def _peak_bin(img):
        sp = np.abs(np.fft.rfft(img[:, 0] - img[:, 0].mean()))
        return int(np.argmax(sp))
    assert _peak_bin(a) == 1
    assert _peak_bin(b) == 4


def test_simulate_interferogram_documents_its_real_output_range():
    """The docstring promised ``[0, 1]``; the implementation returns
    ``background * (1 + V cos phi)`` which spans ``[0, 2 background]``."""
    from lumenairy.analysis.interferometry import simulate_interferogram as f
    doc = f.__doc__
    assert '2 * background' in doc
    opd = np.linspace(-1e-6, 1e-6, 64)[None, :] * np.ones((8, 1))
    out = f(opd, 633e-9, visibility=1.0, background=1.0)
    assert float(out.max()) > 1.5


def test_analysis_package_exports_the_functions_that_live_in_it():
    """``ao_closed_loop``, ``make_shack_hartmann_wfs`` and
    ``coronagraph_contrast_curve`` live in ``analysis/`` and were exported
    by the package ROOT but absent from ``lumenairy.analysis``; the cache
    accessors were exported nowhere."""
    import lumenairy.analysis as A
    for name in ('ao_closed_loop', 'make_shack_hartmann_wfs',
                 'coronagraph_contrast_curve', 'clear_meshgrid_cache',
                 'meshgrid_cache_bytes', 'zernike_basis_cache_bytes',
                 'unwrap_phase_2d'):
        assert hasattr(A, name), f'lumenairy.analysis.{name} missing'
        assert name in A.__all__, f'{name} missing from analysis.__all__'
        assert callable(getattr(A, name))


def test_strehl_phase_integral_documents_the_tilt_convention():
    """``strehl_phase_integral`` and ``strehl_ratio`` disagree by three
    orders of magnitude on a tilted wavefront -- both legitimately -- and
    the docstring never said so."""
    from lumenairy.analysis.strehl import strehl_phase_integral, strehl_ratio
    assert 'tilt' in strehl_phase_integral.__doc__.lower()
    N = 256
    x = (np.arange(N) - N / 2) / (N / 2)
    X, Y = np.meshgrid(x, x)
    ap = (X ** 2 + Y ** 2) <= 1.0
    # 1 wave rms of pure tilt over the pupil.
    W = X / float(np.sqrt(np.mean((X[ap]) ** 2)))
    P = ap * np.exp(1j * 2 * np.pi * W)
    s_int = strehl_phase_integral(P)
    ideal = np.abs(np.fft.fftshift(np.fft.fft2(
        np.fft.ifftshift(ap.astype(complex))))) ** 2
    got = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(P)))) ** 2
    s_peak = float(got.max() / ideal.max())
    assert s_peak > 0.99 and s_int < 0.01, (
        f'the documented disagreement is the point: peak-ratio '
        f'{s_peak:.5f} vs phase-integral {s_int:.5f}')


def test_zernike_cache_key_separates_grids_that_differ_in_x():
    """The key's single "mid-point sample" indexed ``X.flat[N * N / 2]``
    = ``X[N / 2, 0]`` = ``x[0]`` -- identical to ``X.flat[0]`` for the
    row-repeating meshgrid every caller passes -- so it carried NO
    information for X and the key was corners-only in that axis."""
    N = 64
    x = (np.arange(N) - N / 2) / (N / 2)
    X, Y = np.meshgrid(x, x)
    # A grid with the same shape / dtype / corners but a different
    # interior in X: squeeze the middle columns towards the axis.
    X2 = X.copy()
    X2[:, 1:-1] *= 0.5
    assert float(X2.flat[0]) == float(X.flat[0])
    assert float(X2.flat[-1]) == float(X.flat[-1])
    assert _zernike_basis_cache_key(6, X, Y, 1.0) != _zernike_basis_cache_key(
        6, X2, Y, 1.0)
    # Same grid, same key (the cache must still hit).
    Xc, Yc = np.meshgrid(x, x)
    assert _zernike_basis_cache_key(6, X, Y, 1.0) == _zernike_basis_cache_key(
        6, Xc, Yc, 1.0)
