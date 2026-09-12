"""VERIFY-A7 -- independent re-verification of the WP-A7 analysis fixes.

These are the checks the WP's own test files do NOT make.  Every oracle here
is analytic or hand-built in this file; none of it is produced by the code
under test.  Fixtures deliberately differ from the WP's (wavelength, grid
parity, pitch anisotropy, lens shape) so a fixture-specific coincidence
cannot carry both sets.

Written against ``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11`` rows
A1-A7 and ``docs/TESTING_STANDARDS.md`` (derived, two-sided bars with the
measured values and the decades of gap recorded at the bar).
"""
import warnings

import numpy as np
import pytest

from lumenairy.analysis.detector import _reconstruct_wavefront
from lumenairy.analysis.opd import wave_opd_2d
from lumenairy.analysis.polychromatic import (radial_power_bands,
                                              _RADIAL_SORT_CROSSOVER)
from lumenairy.analysis.through_focus import (diffraction_limited_peak,
                                              through_focus_scan)
from lumenairy.propagators.propagation import angular_spectrum_propagate

# 1064 nm and 532 nm -- neither is a wavelength the WP's fixtures use.
LAM = 532e-9
K0 = 2.0 * np.pi / LAM

# Bar on "the unwrapped map IS the analytic wavefront it was built from".
#
# Derivation: the masked Itoh integration is a cumulative sum of wrapped
# neighbour differences over a path of at most ~max(Ny, Nx) samples.  Each
# step is exact to ``eps * |phase|``, so the accumulated error is
# ``N * eps * k0 * |W|`` = 400 * 2.2e-16 * 2 pi * 2 waves = 1.1e-12 rad
# = 1.8e-13 waves.  Measured across the fixtures below (annulus, off-centre,
# odd N, dy != dx, float32): 2.4e-15 .. 4.6e-7 waves, the largest being the
# complex64 case whose own input phase only carries ~6e-8 rad.  The defect
# this bar exists to catch is an INTEGER wave slip (>= 1.0 waves, measured
# 1.000 / 4.000 / 19.000 pre-fix).  1e-4 waves sits 3 decades above the
# worst measurement and 4 decades below the smallest real failure.
UNWRAP_TOL_WAVES = 1e-4


def _aberrated_pupil(Ny, Nx, dx, dy, ap, ox=0.0, oy=0.0, eps=0.0,
                     coma=0.9, spher=0.5, dtype=np.complex128):
    """Pupil carrying coma + spherical on an (optionally annular, optionally
    decentred) support.  Returns (E, W_true, mask)."""
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X - ox, Y - oy)
    rho = R / (ap / 2)
    th = np.arctan2(Y - oy, X - ox)
    mask = (rho <= 1.0) & (rho >= eps)
    W = LAM * (coma * np.sqrt(8) * (3 * rho ** 3 - 2 * rho) * np.cos(th)
               + spher * np.sqrt(5) * (6 * rho ** 4 - 6 * rho ** 2 + 1))
    return (mask * np.exp(1j * K0 * W)).astype(dtype), W, mask


def _shape_error_waves(opd, W, mask):
    m = mask & np.isfinite(opd)
    d = opd[m] - W[m]
    return float(np.max(np.abs(d - np.median(d)))) / LAM


# ===========================================================================
# A1 -- masked 2-D unwrap, on grids the WP's tests never exercise
# ===========================================================================

@pytest.mark.parametrize('Ny,Nx,dxv,dyv', [
    (257, 257, 1.1e-6, 1.1e-6),          # odd square
    (255, 321, 1.1e-6, 1.1e-6),          # odd x odd, rectangular
    (257, 257, 1.1e-6, 2.7e-6),          # odd + anamorphic
    (198, 311, 0.9e-6, 2.2e-6),          # even x odd + anamorphic
])
def test_unwrap_is_exact_on_odd_and_anamorphic_grids(Ny, Nx, dxv, dyv):
    """``dy != dx`` and odd N change the grid the aperture mask and the
    piston anchor are built on.  The unwrap itself must not care."""
    ap = 0.9 * min(Nx * dxv, Ny * dyv)
    E, W, mask = _aberrated_pupil(Ny, Nx, dxv, dyv, ap)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        _, _, opd = wave_opd_2d(E, dxv, LAM, dy=dyv)
    err = _shape_error_waves(opd, W, mask)
    assert err < UNWRAP_TOL_WAVES, (
        f'{Ny}x{Nx} dx={dxv:.1e} dy={dyv:.1e}: max |OPD error| = '
        f'{err:.3e} waves')
    assert np.all(np.round((opd[mask] - W[mask]
                            - np.median(opd[mask] - W[mask])) / LAM) == 0)


def test_unwrap_survives_a_float32_field():
    """``np.angle`` of a complex64 field is float32, and the kernel builds
    its difference buffer in float64.  The result must still be a valid
    unwrap -- the float32 input phase itself only resolves ~6e-8 rad, so
    the bar is the input's own floor, not the kernel's."""
    E64, W, mask = _aberrated_pupil(512, 512, 1e-6, 1e-6, 400e-6)
    E32 = E64.astype(np.complex64)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        _, _, o64 = wave_opd_2d(E64, 1e-6, LAM, aperture=400e-6)
        _, _, o32 = wave_opd_2d(E32, 1e-6, LAM, aperture=400e-6)
    e64 = _shape_error_waves(o64, W, mask)
    e32 = _shape_error_waves(o32, W, mask)
    # measured: 1.15e-14 (complex128) and 4.56e-07 (complex64)
    assert e64 < UNWRAP_TOL_WAVES and e32 < UNWRAP_TOL_WAVES, (e64, e32)
    assert o32.dtype == np.float64


@pytest.mark.parametrize('layout', ['fortran', 'strided'])
def test_unwrap_is_layout_independent(layout):
    """Non-C-contiguous inputs must give the identical map -- the kernel
    writes into its own buffers, so any difference would mean it is
    reading a stride it should not."""
    E, W, mask = _aberrated_pupil(256, 256, 1.5e-6, 1.5e-6, 300e-6)
    alt = (np.asfortranarray(E) if layout == 'fortran'
           else np.ascontiguousarray(np.repeat(E, 2, axis=1))[:, ::2])
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        _, _, a = wave_opd_2d(E, 1.5e-6, LAM, aperture=300e-6)
        _, _, b = wave_opd_2d(alt, 1.5e-6, LAM, aperture=300e-6)
    np.testing.assert_array_equal(a[mask], b[mask])


@pytest.mark.parametrize('eps', [0.0, 0.4, 0.7])
def test_unwrap_is_exact_across_a_central_obstruction(eps):
    """An annulus is multiply connected; both sides of the obscuration must
    land on the same whole-wave branch, and no residue/disconnection
    warning may fire (the annulus IS one connected region)."""
    E, W, mask = _aberrated_pupil(384, 384, 1.3e-6, 1.3e-6, 300e-6, eps=eps)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        _, _, opd = wave_opd_2d(E, 1.3e-6, LAM, aperture=300e-6)
    assert _shape_error_waves(opd, W, mask) < UNWRAP_TOL_WAVES


@pytest.mark.parametrize('ox,oy', [(0.0, 0.0), (60e-6, -40e-6), (110e-6, 90e-6)])
def test_offcentre_pupil_shape_is_exact_and_the_piston_gauge_is_documented(ox, oy):
    """The SHAPE is exact wherever the pupil sits.  The ABSOLUTE piston is
    a documented gauge: the map is anchored on the principal value of the
    valid sample nearest ``x = y = 0``, which is the wavefront's stationary
    point only for a CENTRED pupil.

    Measured on this fixture (1.2 waves rms of coma, 160 um pupil on a
    320 x 320 / 1.5 um grid): absolute piston offset 0.0000 waves centred,
    +1.0000 at (+60, -40) um, +2.0000 at (+110, +90) um -- always an exact
    whole number of waves, never a fraction, which is what makes it a
    gauge rather than an error.  This test pins both halves: shape exact,
    piston an integer.  (Open item VERIFY-A7/A1-N1, resolved: the
    ``wave_opd_2d`` Notes now name the centred-pupil condition explicitly
    and quote these numbers; ``test_the_piston_anchor_note_names_the_
    centred_pupil_condition`` pins the claim.)
    """
    E, W, mask = _aberrated_pupil(320, 320, 1.5e-6, 1.5e-6, 160e-6,
                                  ox=ox, oy=oy, coma=1.2, spher=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        _, _, opd = wave_opd_2d(E, 1.5e-6, LAM)
    assert _shape_error_waves(opd, W, mask) < UNWRAP_TOL_WAVES
    piston = float(np.median(opd[mask] - W[mask])) / LAM
    assert abs(piston - round(piston)) < UNWRAP_TOL_WAVES, (
        f'piston offset {piston:.6f} waves is not a whole number, so the '
        f'map is not congruent to the wrapped phase.')
    if ox == 0.0 and oy == 0.0:
        assert round(piston) == 0


@pytest.mark.parametrize('charge', [1, 2, -1])
def test_vortex_of_any_charge_is_reported_as_a_residue(charge):
    """A charge-m vortex carries a genuine 2 pi m circulation: no
    single-valued map exists and the function must say so.  Measured
    residue 1.000 waves for every charge (the kernel reports the worst
    single LINK, which is one wave regardless of m)."""
    N, dx, ap = 256, 2e-6, 400e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = ((X ** 2 + Y ** 2) <= (ap / 2) ** 2) * np.exp(1j * charge
                                                      * np.arctan2(Y, X))
    with pytest.warns(RuntimeWarning, match='residues'):
        wave_opd_2d(E, dx, LAM, aperture=ap)


# ===========================================================================
# A2 -- the exact-sphere Strehl reference on geometries the WP did not use
# ===========================================================================

# Bar on "Strehl of an aberration-free pupil is exactly 1".
#
# Derivation: numerator and denominator are the SAME field through the SAME
# propagator, so in exact arithmetic the ratio is 1 and the residual is the
# float64 FFT floor (~Ny*Nx*eps ~ 1e-12 at N = 1024).  Measured 1.000000000
# (nine figures) on every row below, both signs of f, decentred, odd N,
# rectangular and annular.  Pre-fix the same quantity read 1.0015 at f/5,
# 1.0988 at f/2.5 and 1.4292 at f/2.  1e-6 is 6 decades above the floor and
# 3 decades below the smallest pre-fix error.
STREHL_TOL = 1e-6


def _sphere_pupil(N, dx, D, f, ox=0.0, oy=0.0, eps=0.0, Ny=None):
    Ny = Ny or N
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dx
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X - ox, Y - oy)
    amp = ((R <= D / 2) & (R >= eps * D / 2)).astype(float)
    sag = np.sign(f) * (np.sqrt(X ** 2 + Y ** 2 + f * f) - abs(f))
    return amp * np.exp(-1j * K0 * sag)


@pytest.mark.parametrize('f_mm', [+0.5, -0.5, +2.0, -2.0])
def test_strehl_is_one_for_both_signs_of_f(f_mm):
    """The f < 0 branch: ``sign(f) (sqrt(r^2+f^2) - |f|)``.  An
    ``abs(f)``-only 'exact' form would flip the diverging reference to a
    converging one, which THIS test catches and an algebraic
    re-derivation of the formula inside the test does not.

    D = 200 um at 532 nm, so |f| = 0.5 mm is f/2.5 and 2 mm is f/10.
    Measured with the pre-fix quadratic reference restored in-process:
    1.127579 at f/2.5 (both signs) and 1.000063 at f/10 (both signs),
    against a measured post-fix 1.000000000.  The bar separates them at
    every arm; the f/2.5 arms clear it by 5 decades and the f/10 arms --
    kept because they are the regime the fix must NOT disturb -- by 1.8.
    """
    f = f_mm * 1e-3
    N, dx, D = 1024, 0.3e-6, 200e-6
    E = _sphere_pupil(N, dx, D, f)
    ref = diffraction_limited_peak(E, LAM, f, dx)
    peak = float(np.max(np.abs(angular_spectrum_propagate(E, f, LAM, dx)) ** 2))
    assert abs(peak / ref - 1.0) < STREHL_TOL, (
        f'f = {f_mm:+.2f} mm: Strehl {peak / ref:.9f}')


@pytest.mark.parametrize('kind', ['centred', 'decentred', 'odd_N',
                                  'rectangular', 'annular'])
def test_strehl_is_one_for_offcentre_odd_and_annular_pupils(kind):
    """The reference sag is built on the grid centre, not on the pupil
    centre; a decentred or annular amplitude must not disturb the ratio,
    and odd / rectangular grids must not shift the reference by half a
    sample.

    Geometry f/2.5 (D = 200 um, f = 500 um, 532 nm), where the removed
    inflation is large: ``W040 = (D/lam) / (128 (f/#)^3)`` = 0.19 waves,
    so the paraxial reference's own Strehl is ~0.7.  Measured with the
    pre-fix quadratic reference restored in-process: 1.127579 (centred),
    1.468475 (decentred), 1.126358 (odd N), 1.126121 (rectangular),
    1.106679 (annular) -- 5 decades above the 1e-6 bar, which is itself
    3 decades above the measured post-fix 1.000000000.
    """
    dx, D, f = 0.3e-6, 200e-6, 500e-6
    kw = {'centred': dict(N=1024),
          'decentred': dict(N=1024, ox=30e-6, oy=-25e-6),
          'odd_N': dict(N=1023),
          'rectangular': dict(N=1025, Ny=801),
          'annular': dict(N=1024, eps=0.5)}[kind]
    E = _sphere_pupil(dx=dx, D=D, f=f, **kw)
    ref = diffraction_limited_peak(E, LAM, f, dx)
    peak = float(np.max(np.abs(angular_spectrum_propagate(E, f, LAM, dx)) ** 2))
    assert abs(peak / ref - 1.0) < STREHL_TOL, f'{kind}: {peak / ref:.9f}'


# ===========================================================================
# A3 -- the Southwell solve, isolated from the sensor
# ===========================================================================

# Bar on the zonal solve.  The Southwell relation
# ``(W[q] - W[p]) / pitch == (s[q] + s[p]) / 2`` is the trapezoid rule, so
# it is EXACT for any slope field linear in the pupil coordinate -- tilt,
# defocus and astigmatism all are.  The only error is the sparse
# factorisation's, ~cond(L) * eps; measured 1.7e-15 .. 3.4e-14 of the
# wavefront span over 172 .. 3112 nodes on discs and annuli.  1e-10 sits
# 4 decades above that and 10 decades below the factor-2 defect this row
# is about.
ZONAL_TOL = 1e-10


@pytest.mark.parametrize('n,eps', [(16, 0.0), (17, 0.45), (33, 0.0),
                                   (33, 0.45)])
def test_southwell_is_exact_on_disc_and_annular_lenslet_masks(n, eps):
    """Exact analytic slopes of a NON-separable wavefront
    ``W = 2c x y + a r^2 + tilt`` over a circular / annular measured set.
    45-degree astigmatism is the case the pre-fix average of two one-sided
    integrals could not even describe as 'half'."""
    p = 1.0e-4
    ii = (np.arange(n) - (n - 1) / 2) * p
    X, Y = np.meshgrid(ii, ii)
    R = np.hypot(X, Y)
    rmax = float(ii.max())
    good = (R <= rmax * 1.001) & (R >= eps * rmax)
    W = 2 * 5.0 * X * Y + 1.5 * (X ** 2 + Y ** 2) + 3e-4 * X
    Sx = 2 * 5.0 * Y + 3.0 * X + 3e-4
    Sy = 2 * 5.0 * X + 3.0 * Y
    w = _reconstruct_wavefront(np.where(good, Sx, np.nan),
                               np.where(good, Sy, np.nan), p, 'southwell')
    g = np.isfinite(w)
    np.testing.assert_array_equal(g, good)
    t = W - W[g][0]
    d = (w - t)[g]
    d = d - d[0]
    span = float(t[g].max() - t[g].min())
    assert float(np.max(np.abs(d))) / span < ZONAL_TOL, (
        f'n={n} eps={eps}: max |W - truth| / span = '
        f'{np.max(np.abs(d)) / span:.3e}')


def test_itoh_integrates_through_holes_where_southwell_does_not():
    """Guards the ``_reconstruct_wavefront`` docstring claim.  The NaN
    exclusion belongs to the Southwell solve only: ``'itoh'`` still
    replaces an un-measured lenslet's slope with ZERO and integrates
    through it, so on an obstructed mask it is wrong by a large fraction
    of the span.  Measured on the annulus below: southwell 8.7e-15 of the
    span, itoh 4.9e-1.  Pinned so the difference is a documented choice,
    not a surprise.  (Open item VERIFY-A7/A3-N1, resolved: the docstring
    now says so explicitly, on ``_reconstruct_wavefront``, on
    ``_itoh_wavefront`` and under ``shack_hartmann(reconstruction=)``;
    the accompanying ``test_the_itoh_caveat_is_documented`` pins that.)
    """
    n, p, eps = 33, 1.0e-4, 0.45
    ii = (np.arange(n) - (n - 1) / 2) * p
    X, Y = np.meshgrid(ii, ii)
    R = np.hypot(X, Y)
    rmax = float(ii.max())
    good = (R <= rmax * 1.001) & (R >= eps * rmax)
    W = 2 * 5.0 * X * Y + 1.5 * (X ** 2 + Y ** 2)
    Sx = np.where(good, 2 * 5.0 * Y + 3.0 * X, np.nan)
    Sy = np.where(good, 2 * 5.0 * X + 3.0 * Y, np.nan)
    span = float((W[good]).max() - (W[good]).min())
    errs = {}
    for meth in ('southwell', 'itoh'):
        w = _reconstruct_wavefront(Sx, Sy, p, meth)
        g = np.isfinite(w)
        d = (w - (W - W[g][0]))[g]
        errs[meth] = float(np.max(np.abs(d - d[0]))) / span
    assert errs['southwell'] < ZONAL_TOL
    assert errs['itoh'] > 0.1, (
        "if 'itoh' has learned to skip un-measured lenslets, tighten this "
        "and update the _reconstruct_wavefront docstring, which currently "
        "claims the NaN exclusion for both methods")


def test_disconnected_measured_sets_are_gauged_independently():
    """A mask split in two has an undetermined relative piston.  The solve
    must pin each component separately (lowest-index member to zero) and
    NOT invent a relationship -- and it must not raise."""
    n, p = 6, 1.0e-4
    Sx = np.full((n, n), 3e-4)
    Sy = np.full((n, n), -1e-4)
    Sx[:, 3] = np.nan
    Sy[:, 3] = np.nan
    w = _reconstruct_wavefront(Sx, Sy, p, 'southwell')
    left = w[:, :3]
    right = w[:, 4:]
    assert np.all(np.isnan(w[:, 3]))
    assert np.all(np.isfinite(left)) and np.all(np.isfinite(right))
    assert w[0, 0] == 0.0
    assert w[0, 4] == 0.0, (
        'the second component must be gauged on its own lowest-index '
        'member, not carried across the gap')


# ===========================================================================
# A6 -- the transfer-function recurrence: does it actually engage?
# ===========================================================================

def _focus_field(N=128, dx=4e-6, f=20e-3):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return (np.exp(-(X ** 2 + Y ** 2) / (0.25 * N * dx) ** 2)
            * np.exp(-1j * K0 * (X ** 2 + Y ** 2) / (2 * f)))


def test_uniform_scan_really_takes_the_recurrence_path(monkeypatch):
    """The WP's own tests check that the two paths AGREE; nothing checks
    that the fast path is entered at all.  If the uniformity gate
    (``k * max|z - model| <= 1e-12``) ever rejected a plain ``linspace``,
    every accuracy test would still pass and the whole speedup would be
    silently gone.

    Structural, build-free discriminator: the direct form calls
    ``np.exp`` once per propagating plane; the recurrence calls it twice
    in total (``H_step`` plus the first plane's ``H_cur``).  Counted here
    on a 21-plane scan -- 2 vs 21 is not a tolerance, it is a branch.
    """
    import lumenairy.analysis.through_focus as tf
    E = _focus_field()
    dx, f = 4e-6, 20e-3
    calls = {'n': 0}
    real_exp = np.exp

    def counting_exp(*a, **kw):
        calls['n'] += 1
        return real_exp(*a, **kw)

    monkeypatch.setattr(tf.np, 'exp', counting_exp)
    z_uni = np.linspace(0.9 * f, 1.1 * f, 21)
    calls['n'] = 0
    through_focus_scan(E, dx, LAM, z_uni, verbose=False)
    n_uniform = calls['n']
    z_non = z_uni.copy()
    z_non[10] += (z_uni[1] - z_uni[0]) * 1e-3
    calls['n'] = 0
    through_focus_scan(E, dx, LAM, z_non, verbose=False)
    n_direct = calls['n']
    assert n_uniform < n_direct / 3, (
        f'uniform scan made {n_uniform} np.exp calls and the non-uniform '
        f'scan {n_direct}; the recurrence is not engaging on a linspace.')
    assert n_direct >= 20


def test_recurrence_holds_over_a_long_scan():
    """200 planes instead of 21: the accumulated rounding of
    ``H_n = H_0 * H_step^n`` grows linearly in n, so a long scan is where
    a recurrence would show.

    Bound (derived): both forms evaluate the same exact function and round
    the ARGUMENT at ``|kz z| eps / 2``, so their difference is at most
    ``2 |kz z|_max eps``.  Here ``|kz z|_max = k * 30 mm = 3.5e5`` rad,
    giving 1.6e-10 in phase.  Measured worst relative drift of ``peak_I``
    against 200 independent ``angular_spectrum_propagate`` calls: 4.5e-12
    on this class of machine.  Bar 1e-8: three decades above the
    measurement, five below the 1e-3 that any sign / band-limit / shift
    defect in the transfer function would produce.
    """
    E = _focus_field(N=96)
    dx, f = 4e-6, 20e-3
    z = np.linspace(0.5 * f, 1.5 * f, 200)
    scan = through_focus_scan(E, dx, LAM, z, verbose=False)
    ref = np.array([float((np.abs(angular_spectrum_propagate(
        E, float(zi), LAM, dx)) ** 2).max()) for zi in z])
    drift = float(np.max(np.abs(scan.peak_I - ref) / ref))
    assert drift < 1e-8, f'worst relative peak drift over 200 planes {drift:.3e}'


# ===========================================================================
# A6 -- radial_power_bands at the crossover
# ===========================================================================

@pytest.mark.parametrize('n', [_RADIAL_SORT_CROSSOVER - 1,
                               _RADIAL_SORT_CROSSOVER,
                               _RADIAL_SORT_CROSSOVER + 1])
def test_radial_band_crossover_is_seamless(n):
    """Straddling the threshold: below it the masked loop must be
    BIT-identical to the historical result; at and above it the sorted
    construction may differ only by summation associativity.

    Bound: a sequential ``cumsum`` over Ny*Nx = 65536 addends against
    numpy's pairwise reduction differs by O(Ny*Nx * eps) = 1.5e-11
    relative; measured 5.5e-14 on this Gaussian.  1e-11 is at the bound
    and 8 decades below a wrong-bin defect (which moves a band by a whole
    annulus, >= 1e-3).
    """
    N, dx = 256, 2e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (60e-6) ** 2).astype(complex)
    R2 = X ** 2 + Y ** 2
    I = np.abs(E) ** 2
    radii = np.linspace(4e-6, 220e-6, n)
    got = radial_power_bands(E, dx, radii)
    ref = np.array([float(np.sum(I[R2 <= r * r]) * dx * dx) for r in radii])
    if n < _RADIAL_SORT_CROSSOVER:
        np.testing.assert_array_equal(got, ref)
    else:
        np.testing.assert_allclose(got, ref, rtol=1e-11, atol=0)


@pytest.mark.parametrize('n', [4, _RADIAL_SORT_CROSSOVER + 8])
def test_radial_band_nan_radius_answers_the_same_on_both_paths(n):
    """A NaN radius is the one query where ``searchsorted`` and
    ``R2 <= r*r`` disagree by construction: NaN sorts ABOVE every finite
    key, so the sorted path would hand back the WHOLE grid's power where
    the masked loop returns 0.  Measured before the guard: 2.513e-09
    (total) vs 0.0.  Both paths must agree with the masked loop, which is
    what every release before the crossover returned.
    """
    N, dx = 128, 2e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (40e-6) ** 2).astype(complex)
    R2 = X ** 2 + Y ** 2
    I = np.abs(E) ** 2
    radii = np.concatenate(([np.nan], np.linspace(4e-6, 120e-6, n - 1)))
    got = radial_power_bands(E, dx, radii)
    ref = np.array([float(np.sum(I[R2 <= r * r]) * dx * dx) for r in radii])
    assert got[0] == 0.0, (
        f'NaN radius returned {got[0]:.4e}; the masked loop returns 0.0 '
        f'and the total grid power is {float(I.sum() * dx * dx):.4e}.')
    np.testing.assert_allclose(got, ref, rtol=1e-11, atol=0)


@pytest.mark.parametrize('n', [4, _RADIAL_SORT_CROSSOVER + 8])
def test_radial_band_infinite_radius_is_the_total_on_both_paths(n):
    """The +/- inf companion of the NaN case, which needs no special
    handling and must not acquire one: ``r*r`` is ``+inf`` on both paths
    and both must return the whole grid's power."""
    N, dx = 96, 2e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (40e-6) ** 2).astype(complex)
    total = float((np.abs(E) ** 2).sum() * dx * dx)
    radii = np.concatenate(([np.inf, -np.inf],
                            np.linspace(4e-6, 120e-6, n - 2)))
    got = radial_power_bands(E, dx, radii)
    assert got[0] == pytest.approx(total, rel=1e-11)
    assert got[1] == pytest.approx(total, rel=1e-11)


# ===========================================================================
# Open-item resolutions (coordinator rulings of 2026-09-12)
# ===========================================================================

def test_the_itoh_caveat_is_documented():
    """VERIFY-A7/A3-N1.  The NaN-exclusion sentence used to sit in a
    paragraph that read as covering BOTH reconstructions while being true
    of only one, which is the "documentation says X, code does Y" shape
    this audit exists to remove.  The behaviour is the documented choice
    (a path integral cannot route around a hole); what must not drift is
    the claim.  Structural, not a rendering test.
    """
    from lumenairy.analysis.detector import (_itoh_wavefront,
                                             _reconstruct_wavefront,
                                             shack_hartmann)

    def flat(doc):
        return ' '.join(doc.split()).lower()

    shared = flat(_reconstruct_wavefront.__doc__)
    assert 'southwell solve' in shared, (
        'the NaN-exclusion sentence must name the method it applies to')
    assert 'does not get the nan exclusion' in shared
    for doc in (shared, flat(_itoh_wavefront.__doc__)):
        assert 'zero slope' in doc and 'integrated through' in doc
    assert 'integrates through it' in flat(shack_hartmann.__doc__), (
        'the public reconstruction= parameter must carry the caveat too')


def test_the_piston_anchor_note_names_the_centred_pupil_condition():
    """VERIFY-A7/A1-N1.  The claim "a pupil carrying a known defocus comes
    back with the right integer wave count" holds only when the pupil
    straddles the grid origin; measured +1 and +2 whole waves of piston on
    decentred pupils (see
    ``test_offcentre_pupil_shape_is_exact_and_the_piston_gauge_is_documented``).
    """
    from lumenairy.analysis.opd import wave_opd_2d
    doc = ' '.join(wave_opd_2d.__doc__.split()).lower()
    assert 'centred pupil carrying a known defocus' in doc
    assert 'does not straddle the grid origin' in doc
    assert 'arbitrary whole number of waves' in doc


# --- VERIFY-A7/A5-N1: gerchberg_saxton_jax dtype handling ------------------

def _gs_fixture(N=32, seed=1):
    rng = np.random.default_rng(seed)
    src = np.exp(-((np.arange(N)[:, None] - N / 2) ** 2
                   + (np.arange(N)[None, :] - N / 2) ** 2) / 40.0)
    phi0 = rng.uniform(-np.pi, np.pi, (N, N))
    tgt = np.abs(np.fft.fftshift(np.fft.fft2(
        np.fft.ifftshift(src * np.exp(1j * phi0)))))
    return src, tgt, tgt * (1 + 0.3 * rng.random((N, N))), phi0


@pytest.mark.parametrize('cdtype', [np.complex64, np.complex128])
def test_gs_jax_accepts_a_complex_dtype_and_returns_a_real_error(cdtype):
    """VERIFY-A7/A5-N1.  A complex ``dtype`` names the FIELD type; the
    amplitude arrays -- and therefore ``err`` -- take its real
    counterpart.  Pre-fix a complex request fell through to the
    'unrecognised real dtype' branch, made ``src``/``tgt`` complex, and
    the call died in ``float(err)`` with
    ``TypeError: float() argument must be ... not 'complex'``, so this
    test fails hard (not by a tolerance) on the pre-fix code.
    """
    jax = pytest.importorskip('jax')
    from lumenairy.analysis.phase_retrieval import gerchberg_saxton_jax
    src, _tgt, tgt2, phi0 = _gs_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        phase, err = gerchberg_saxton_jax(src, tgt2, n_iter=10,
                                          initial_phase=phi0, dtype=cdtype)
    assert isinstance(err, float) and np.isfinite(err) and err > 0
    assert np.asarray(phase).dtype == np.dtype(cdtype).type(0).real.dtype
    assert np.all(np.isfinite(phase))


def test_gs_jax_default_precision_follows_x64():
    """VERIFY-A7/A5-N1.  With ``jax_enable_x64`` on, the default working
    precision must be float64 -- the twin documented "the same physics" as
    the NumPy path while returning a float32 answer, so the two agreed to
    six digits at best.

    Bar: with x64 on the two backends' ``err`` must agree to 1e-9
    relative.  Derived -- both run the identical iteration on float64 FFTs,
    so the residual is the two FFT implementations' rounding,
    ~N_pix * eps = 2e-13 relative at N = 32; measured 0.0e+00 (the two
    print 1.5027588862e-01 to every digit float64 carries), against
    2.0e-05 relative at float32.  1e-9 sits 4 decades above the analytic
    floor and 4 below the float32 reading, so the assertion is a decision
    between the two precisions, not a tolerance on either.
    """
    jax = pytest.importorskip('jax')
    if not bool(jax.config.jax_enable_x64):
        jax.config.update('jax_enable_x64', True)
    from lumenairy.analysis.phase_retrieval import (gerchberg_saxton,
                                                    gerchberg_saxton_jax)
    src, _tgt, tgt2, phi0 = _gs_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, err_np, _ = gerchberg_saxton(src, tgt2, n_iter=10,
                                        initial_phase=phi0,
                                        return_history=True)
        phase_d, err_d = gerchberg_saxton_jax(src, tgt2, n_iter=10,
                                              initial_phase=phi0)
        phase_32, err_32 = gerchberg_saxton_jax(src, tgt2, n_iter=10,
                                                initial_phase=phi0,
                                                dtype=np.float32)
    assert np.asarray(phase_d).dtype == np.float64, (
        'default dtype must follow jax_enable_x64')
    assert err_d == pytest.approx(err_np, rel=1e-9), (
        f'jax {err_d!r} vs numpy {err_np!r}')
    # ... and an explicit float32 request still pins the historical path.
    assert np.asarray(phase_32).dtype == np.float32
    assert abs(err_32 - err_np) / err_np > 1e-7, (
        'float32 is expected to differ from the NumPy reference; if it no '
        'longer does, this test no longer discriminates the two precisions')
