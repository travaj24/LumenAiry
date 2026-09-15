"""WP-B8 (Wave 4) -- the analysis / sources performance designs WP-A7 and
WP-A11 deferred, plus the MFT-based PSF sampler of audit section 15.9.

Covered
-------
A6.1 / sec. 15.9  ``compute_psf`` transient memory, and ``method='mft'``.
A6.2              ``encircled_energy_profile`` shared by the curve and the
                  radius (no content-keyed cache -- audit sec. 15.5).
A6.3              the Zernike recurrence and the DM influence-function
                  banding.
Z3 / A11 sec. 6.1 Gori pseudo-modes behind ``generator='modes'``.
Z3 / A11 sec. 6.2 ``create_gaussian_beam(geometry_dtype=)``.
Z3 / A11 sec. 6.3 ``apply_jones_matrix`` accumulation.

Discipline (``docs/TESTING_STANDARDS.md``)
-----------------------------------------
* No wall-clock assertion anywhere.  Times are measured and reported in
  ``WP-B8_REPORT.md``; what is ASSERTED here is either an exact/bit-identity
  claim, an oracle the library did not produce, or a peak-memory bar derived
  from a COUNT OF FULL-GRID ARRAYS (an integer, with half a grid of gap on
  each side) -- never a byte figure read off one run.
* Every "before" is produced IN THIS PROCESS, by restating the pre-fix
  expression locally, so "fails before / passes after" is measured on the
  running build rather than quoted (rule 3 / S2).
* The oracles are lumenairy-free: an exact ``fractions.Fraction`` evaluation
  of the Zernike radial polynomial (its coefficients are integers, so a
  rational ``rho`` gives an exactly rational value), a brute-force centred
  Fourier sum, the closed-form Gaussian and Airy patterns, and the exact
  finite-M moment ``E[I^2]/E[I]^2 = 2 - 1/M`` of a random-phasor sum.
"""
from __future__ import annotations

import gc
import math
import tracemalloc
import warnings
from fractions import Fraction

import numpy as np
import pytest

from lumenairy.analysis import (
    DeformableMirror,
    compute_otf,
    compute_psf,
    encircled_energy_curve,
    encircled_energy_profile,
    encircled_energy_radius,
    zernike_basis_matrix,
    zernike_polynomial,
)
from lumenairy.analysis import ao as _ao
from lumenairy.analysis import psf_mtf_otf as _psf
from lumenairy.analysis import zernike as _zk
from lumenairy.elements.polarization import JonesField, apply_jones_matrix
from lumenairy.sources.core import (
    _GORI_MAX_MODES,
    _GORI_MIN_MODES,
    _gori_mode_count,
    _schell_phase_realizations,
    create_gaussian_beam,
    create_gaussian_schell_source,
    create_schell_model_source,
)

WL = 633e-9
FOCAL = 0.05


def _raw(a):
    """Raw bytes of an array, for ``==`` that means BIT-identical."""
    return np.ascontiguousarray(a).view(np.uint8)


def _same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and np.array_equal(_raw(a), _raw(b)))


def _peak_grids(fn, grid_bytes, reps=3):
    """Median peak transient of ``fn``, in units of ONE full grid.

    The unit is what makes the bars derivable: an implementation that
    holds k full-grid arrays at its peak measures k, and the bars below
    sit half a grid away from the integers they separate.
    """
    peaks = []
    for _ in range(reps):
        gc.collect()
        tracemalloc.start()
        out = fn()
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        del out
    return float(np.median(peaks)) / grid_bytes


# ===========================================================================
# A6.1 -- the centred transform, its bit-identity, and its array count
# ===========================================================================

def _stock_centred_fft2(a):
    """The expression ``compute_psf`` / ``compute_otf`` used before B8."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))


@pytest.mark.parametrize('shape', [(8, 8), (16, 16), (64, 64), (100, 100),
                                   (192, 192), (6, 10), (64, 128), (2, 2),
                                   (33, 33), (17, 64), (64, 17), (35, 21)])
@pytest.mark.parametrize('dt', [np.complex128, np.complex64, np.float64])
@pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
def test_b8_centred_fft2_is_bit_identical_to_the_explicit_shifts(
        shape, dt, layout):
    """The in-place quadrant exchange is a PERMUTATION, so the FFT sees
    the same bits ``ifftshift`` would have handed it.  Bit-identity is a
    construction property here, not a build coincidence -- which is the
    whole reason this form was chosen over the chessboard identity (see
    the next test).  Odd lengths fall back to the explicit shifts and
    are included so the fallback is pinned too."""
    ny, nx = shape
    rng = np.random.default_rng(1000 + ny * 31 + nx)
    if dt is np.float64:
        base = rng.standard_normal(shape)
    else:
        base = (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape)).astype(dt)
    if layout == 'C':
        a = np.ascontiguousarray(base)
    elif layout == 'F':
        a = np.asfortranarray(base)
    else:
        big = np.zeros((ny * 2, nx), dtype=base.dtype)
        big[::2] = base
        a = big[::2]

    want = _stock_centred_fft2(a)
    got = _psf._centred_fft2(a, np)
    assert _same_bits(want, got)
    # And the consuming form, which is what compute_psf uses.
    box = [np.array(a, copy=True, order='C')]
    assert _same_bits(want, _psf._centred_fft2_take(box, np))
    assert box[0] is None, 'the box must be emptied so the input can be freed'


@pytest.mark.parametrize('n', [8, 16, 32, 64, 128, 256, 512])
def test_b8_fft2_is_exactly_its_two_axis_passes(n):
    """``_centred_fft2_take`` splits ``fft2`` so the padded input can be
    released between the two passes.  NumPy's ``_raw_fftnd`` walks the
    axis list in reverse, so ``fft2`` IS ``fft(axis=-1)`` then
    ``fft(axis=-2)``; pin it, because the split would otherwise be a
    silent change of arithmetic if that ever moved."""
    rng = np.random.default_rng(7 * n)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    split = np.fft.fft(np.fft.fft(a, axis=-1), axis=-2)
    assert _same_bits(np.fft.fft2(a), split)


@pytest.mark.parametrize('n', [8, 16, 32, 64, 100, 128, 192, 256, 384, 512])
def test_b8_chessboard_identity_is_bit_identical_only_on_powers_of_two(n):
    """WHY the chessboard identity the A7 design named is not used.

    ``chess * fft2(chess * a)`` equals ``fftshift(fft2(ifftshift(a)))``
    up to the global sign ``(-1)^(Ny/2 + Nx/2)`` for any EVEN length --
    mathematically.  In floating point the two forms are different
    reductions, and on this build they agree bit for bit only when the
    length is a power of two; on an even non-power-of-two they agree to
    a few ULP, which is a moved default.  This test is the measurement
    that gates the choice, and it is two-sided: exact where it is
    exact, and NOT exact (but within the float64 reduction bound) where
    it is not."""
    rng = np.random.default_rng(n)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    chess = np.ones(n)
    chess[1::2] = -1.0
    b = a * chess[:, None]
    b *= chess[None, :]
    F = np.fft.fft2(b)
    F *= chess[:, None]
    F *= chess[None, :]
    if ((n // 2) * 2) % 4:
        F = -F
    want = _stock_centred_fft2(a)
    is_pow2 = (n & (n - 1)) == 0
    if is_pow2:
        assert _same_bits(want, F), (
            'the chessboard form was bit-identical on powers of two when '
            'this bar was derived (2026-09-13); if that has changed the '
            'choice in _centred_fft2_take can be revisited')
    else:
        assert not _same_bits(want, F)
        # Bound: both forms are the same exact sum evaluated by different
        # reductions over n terms per axis, so the gap is O(n * eps) of
        # the transform's own scale.  n = 512 -> 1.1e-13 relative.
        bound = 2.0 * n * np.finfo(np.float64).eps
        assert np.abs(want - F).max() / np.abs(want).max() < bound


def test_b8_chessboard_identity_is_a_different_array_on_an_odd_length():
    """The other half of the same measurement: at an odd length the
    identity is off by a cyclic shift, so it is not a tolerance question
    at all (relative ~2)."""
    n = 63
    rng = np.random.default_rng(63)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    chess = np.ones(n)
    chess[1::2] = -1.0
    F = np.fft.fft2(a * chess[:, None] * chess[None, :])
    F = F * chess[:, None] * chess[None, :]
    want = _stock_centred_fft2(a)
    rel = np.abs(want - F).max() / np.abs(want).max()
    assert rel > 0.5, f'expected an O(1) disagreement, measured {rel:.3g}'


@pytest.mark.parametrize('oversample', [2, 4])
def test_b8_compute_psf_transient_falls_by_one_full_padded_grid(oversample):
    """Peak transient, in units of ONE padded complex grid.

    Derived, not read off a run.  The pre-B8 expression holds, at the
    moment of the second FFT pass: the padded pupil, the ``ifftshift``
    copy and both FFT intermediates = **4**.  The new one holds the
    padded pupil (which IS the shift buffer) and both intermediates =
    **3**; its extra quadrant scratch is a quarter grid and never
    coexists with all three.  The bar sits at 3.5 -- half a grid from
    each integer -- and the same run measures the pre-fix arm."""
    Np = 256
    dx = 2e-6
    N_psf = Np * oversample
    grid = N_psf * N_psf * 16
    x = (np.arange(Np) - Np / 2) * dx
    X, Y = np.meshgrid(x, x)
    pupil = (np.sqrt(X ** 2 + Y ** 2) <= 200e-6).astype(complex)

    def pre_fix():
        pad = (N_psf - Np) // 2
        a = np.pad(pupil, ((pad, N_psf - Np - pad),) * 2, mode='constant')
        amp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))
        psf = np.abs(amp) ** 2
        s = float(np.sum(np.abs(a) ** 2)) * dx ** 2
        d = WL * FOCAL / (N_psf * dx)
        return psf * (s / (float(np.sum(psf)) * d ** 2))

    new = _peak_grids(lambda: compute_psf(pupil, WL, FOCAL, dx,
                                          oversample=oversample)[0], grid)
    old = _peak_grids(pre_fix, grid)
    assert old > 3.5, f'pre-fix arm measured {old:.2f} padded grids'
    assert new < 3.5, f'post-fix arm measured {new:.2f} padded grids'
    assert old - new > 0.8, (
        f'expected one whole padded grid removed; measured '
        f'{old:.2f} -> {new:.2f}')


def test_b8_centred_transform_transient_falls_by_two_grids():
    """The helper in isolation, with the input allocated INSIDE the
    traced region so the count is the whole cost of one centred
    transform.

    Derived: the explicit form holds the input, the ``ifftshift`` copy
    and both FFT intermediates at once = **4**.  The consuming form
    shifts in place and runs the two FFT passes separately, releasing
    the input between them, so it never holds more than two = **2**
    (plus a quarter-grid quadrant scratch that does not coexist with
    them).  Bar at 3.0, a full grid from each."""
    n = 512
    grid = n * n * 16
    rng = np.random.default_rng(5)
    base = (rng.standard_normal((n, n))
            + 1j * rng.standard_normal((n, n)))

    def stock():
        return _stock_centred_fft2(base.copy())

    def new():
        return _psf._centred_fft2_take([base.copy()], np)

    old_p = _peak_grids(stock, grid)
    new_p = _peak_grids(new, grid)
    assert old_p > 3.0, f'pre-fix arm measured {old_p:.2f} grids'
    assert new_p < 3.0, f'post-fix arm measured {new_p:.2f} grids'
    assert old_p - new_p > 1.5, (old_p, new_p)


def test_b8_compute_otf_transient_does_not_rise():
    """``compute_otf``'s input is a REAL intensity PSF, and NumPy's first
    FFT pass makes its own complex copy of a real input -- a floor of
    2.5 grids (0.5 real copy + 1 conversion + 1 output) that neither
    form can go under.  What the rewrite removes there is the
    ``fftshift`` copy and the ``/ dc`` copy, neither of which is at the
    peak.  So the claim here is the honest one: the peak does not rise,
    and the bit-identity is pinned above."""
    n = 512
    grid = n * n * 16
    rng = np.random.default_rng(5)
    psf = np.abs(rng.standard_normal((n, n))) ** 2

    def pre_fix():
        otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
        dc = otf[n // 2, n // 2]
        return otf / dc

    new = _peak_grids(lambda: compute_otf(psf), grid)
    old = _peak_grids(pre_fix, grid)
    assert new <= old + 0.05, (new, old)
    assert _same_bits(compute_otf(psf), pre_fix())


# ---------------------------------------------------------------------------
# A6.1 / sec. 15.9 -- method='mft'
# ---------------------------------------------------------------------------

def _brute_force_psf(pupil, dx_pupil, xs, ys, wavelength, f):
    """The DEFINITION of the centred Fraunhofer sum, in four lines and
    with no lumenairy in them:
    ``F(xf, yf) = sum_n pupil(x_n, y_n) exp(-2 pi i (x_n xf + y_n yf)
    / (lambda f))`` on the package grid ``(arange(N) - N/2) * dx``."""
    Np = pupil.shape[0]
    xn = (np.arange(Np) - Np / 2.0) * dx_pupil
    ey = np.exp(-2j * np.pi * np.outer(ys, xn) / (wavelength * f))
    ex = np.exp(-2j * np.pi * np.outer(xs, xn) / (wavelength * f))
    return np.abs(ey @ pupil @ ex.T) ** 2


@pytest.mark.parametrize('Np,oversample', [(64, 1), (64, 2), (128, 2),
                                           (96, 1), (32, 4)])
def test_b8_mft_reproduces_the_fft_grid_on_the_natural_pitch(Np, oversample):
    """The PSF grid contract does not move: with ``dx_psf`` left None the
    MFT samples the lattice the padded FFT delivers, and on even lengths
    the two agree to the Bluestein's own floor.

    Bar: 2e-13 of the peak.  Derived as the chirp-Z's error floor --
    a Bluestein of length ~(N_in + N_out) is three FFTs and two chirp
    multiplies, so ~5 * n * eps with n <= 1024 is 1.1e-12; the measured
    worst over these five points is 1.1e-15, three decades under the
    bar, and the FFT's own pixel-to-pixel structure is O(1)."""
    dx = 4e-6
    x = (np.arange(Np) - Np / 2) * dx
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X ** 2 + Y ** 2)
    pupil = ((R <= Np * dx / 4)
             * np.exp(2j * (X / (Np * dx / 4)) ** 3)).astype(complex)
    for norm in ('power', 'peak', 'none'):
        a, dxa = compute_psf(pupil, WL, FOCAL, dx, oversample=oversample,
                             normalize=norm)
        b, dxb = compute_psf(pupil, WL, FOCAL, dx, oversample=oversample,
                             normalize=norm, method='mft')
        assert dxa == dxb
        assert a.shape == b.shape
        rel = np.abs(a - b).max() / a.max()
        assert rel < 2e-13, f'{norm}: {rel:.3e}'


@pytest.mark.parametrize('zoom', [1.0, 4.0, 11.0])
def test_b8_mft_matches_a_brute_force_fourier_sum_at_any_pitch(zoom):
    """The independent oracle for the sampler itself: an explicit
    ``O(N^2 M^2)`` centred Fourier sum at the MFT's own output
    coordinates.  Bar 1e-12 relative -- the Bluestein floor ~5 n eps
    with n ~ 100 is 1e-13; measured 7.7e-15 at worst."""
    Np, dx, N_out = 48, 5e-6, 24
    rng = np.random.default_rng(3)
    pupil = (rng.standard_normal((Np, Np))
             + 1j * rng.standard_normal((Np, Np)))
    dx_out = WL * FOCAL / (Np * dx) / zoom
    psf, dxo = compute_psf(pupil, WL, FOCAL, dx, N_psf=N_out,
                           normalize='none', method='mft', dx_psf=dx_out)
    assert dxo == dx_out
    xs = (np.arange(N_out) - N_out / 2.0) * dx_out
    ref = _brute_force_psf(pupil, dx, xs, xs, WL, FOCAL)
    rel = np.abs(psf - ref).max() / ref.max()
    assert rel < 1e-12, f'{rel:.3e}'


@pytest.mark.parametrize('zoom', [1.0, 4.0, 16.0])
def test_b8_mft_matches_the_closed_form_gaussian_psf(zoom):
    """A Gaussian pupil has an exactly Gaussian Fraunhofer transform, and
    on a grid chosen so that BOTH discretisation errors are below the
    float64 floor the closed form is a tight oracle:

    * aliasing (Poisson summation of the transform) ~ exp(-pi^2 w^2/dx^2)
    * truncation at the grid edge   ~ exp(-2 (L/2)^2 / w^2)

    which the assertion computes from the grid it actually built.  Bar
    1e-12 relative; measured 1.1e-15 to 2.2e-15."""
    Np, dx, w, N_out = 256, 2e-6, 16e-6, 64
    x = (np.arange(Np) - Np / 2) * dx
    X, Y = np.meshgrid(x, x)
    pupil = np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(complex)
    alias = math.exp(-np.pi ** 2 * w ** 2 / dx ** 2)
    trunc = math.exp(-2.0 * (Np * dx / 2) ** 2 / w ** 2)
    assert max(alias, trunc) < 1e-200, (alias, trunc)

    dx_out = WL * FOCAL / (Np * dx) / zoom
    psf, _ = compute_psf(pupil, WL, FOCAL, dx, N_psf=N_out,
                         normalize='none', method='mft', dx_psf=dx_out)
    xs = (np.arange(N_out) - N_out / 2.0) * dx_out
    XS, YS = np.meshgrid(xs, xs)
    amp = (np.pi * w ** 2
           * np.exp(-np.pi ** 2 * w ** 2 * (XS ** 2 + YS ** 2)
                    / (WL * FOCAL) ** 2) / (dx * dx))
    ana = amp ** 2
    rel = np.abs(psf - ana).max() / ana.max()
    assert rel < 1e-12, f'{rel:.3e}'


@pytest.mark.parametrize('zoom', [1.0, 8.0])
def test_b8_mft_matches_the_analytic_airy_pattern(zoom):
    """The circular-pupil oracle.  Unlike the Gaussian this one is
    PIXELATION limited: the sampled disc is not the continuum disc, and
    the gap does not shrink with the focal-plane pitch (measured 8.3e-5
    at 300 pixels across the diameter and 7.4e-5 at 600).  Bar 5e-4 --
    above the measured envelope by ~6x and three decades below any
    sampler defect, which would show as an O(1) change of the ring
    structure.  The SAME bar at two zooms is the claim: the error is the
    aperture's, not the sampler's."""
    j1 = pytest.importorskip('scipy.special').j1
    Np, dx, D, N_out = 512, 2e-6, 600e-6, 64
    x = (np.arange(Np) - Np / 2) * dx
    X, Y = np.meshgrid(x, x)
    pupil = (np.sqrt(X ** 2 + Y ** 2) <= D / 2).astype(complex)
    dx_out = WL * FOCAL / (Np * dx) / zoom
    psf, _ = compute_psf(pupil, WL, FOCAL, dx, N_psf=N_out,
                         normalize='peak', method='mft', dx_psf=dx_out)
    xs = (np.arange(N_out) - N_out / 2.0) * dx_out
    XS, YS = np.meshgrid(xs, xs)
    v = np.pi * D * np.sqrt(XS ** 2 + YS ** 2) / (WL * FOCAL)
    safe = np.where(v == 0, 1.0, v)
    airy = np.where(v == 0, 1.0, (2 * j1(safe) / safe) ** 2)
    rel = np.abs(psf - airy).max() / airy.max()
    assert rel < 5e-4, f'{rel:.3e}'


def test_b8_mft_power_normalisation_is_the_analytic_parseval_constant():
    """``normalize='power'`` on the MFT path is the closed-form constant
    ``(dx_pupil^2 / (lambda f))^2``, not an in-window rescale.

    Two claims: on the full natural grid it agrees with the FFT path's
    empirical ratio (so nothing moved), and on a ZOOMED window it keeps
    the same physical scale -- where an in-window rescale would inflate
    the sub-field to carry the whole pupil's power."""
    Np, dx = 128, 3e-6
    x = (np.arange(Np) - Np / 2) * dx
    X, Y = np.meshgrid(x, x)
    pupil = np.exp(-(X ** 2 + Y ** 2) / (60e-6) ** 2).astype(complex)
    nat = WL * FOCAL / (Np * dx)

    full, _ = compute_psf(pupil, WL, FOCAL, dx, normalize='power')
    pupil_power = float(np.sum(np.abs(pupil) ** 2)) * dx ** 2
    assert abs(float(np.sum(full)) * nat ** 2 / pupil_power - 1.0) < 1e-12

    core, _ = compute_psf(pupil, WL, FOCAL, dx, N_psf=8, normalize='power',
                          method='mft', dx_psf=nat / 8.0)
    wide, _ = compute_psf(pupil, WL, FOCAL, dx, N_psf=256, normalize='power',
                          method='mft', dx_psf=nat / 8.0)
    # Same physical intensity at the same physical point.
    assert abs(core[4, 4] / wide[128, 128] - 1.0) < 1e-12
    assert abs(core.max() / full.max() - 1.0) < 1e-9
    # The 8 x 8 window really is a sub-field: it holds well under half
    # the energy, so an in-window rescale -- which is what the FFT
    # path's empirical ratio is -- would have inflated it by 1/frac.
    frac = float(np.sum(core)) * (nat / 8.0) ** 2 / pupil_power
    assert frac < 0.5, frac
    assert 1.0 / frac > 2.0


def test_b8_mft_none_normalisation_is_the_fft_raw_convention():
    Np, dx = 64, 4e-6
    x = (np.arange(Np) - Np / 2) * dx
    X, Y = np.meshgrid(x, x)
    pupil = (np.sqrt(X ** 2 + Y ** 2) <= 100e-6).astype(complex)
    a, _ = compute_psf(pupil, WL, FOCAL, dx, normalize='none')
    b, _ = compute_psf(pupil, WL, FOCAL, dx, normalize='none', method='mft')
    assert np.abs(a - b).max() / a.max() < 2e-13
    c, _ = compute_psf(pupil, WL, FOCAL, dx, normalize='peak', method='mft')
    assert c.max() == pytest.approx(1.0, abs=0.0, rel=1e-15)


def test_b8_mft_warns_on_an_odd_grid_and_says_what_moves():
    """An odd axis is the one place the two samplers disagree, and by a
    half pixel: ``ifftshift`` centres on ``N // 2``, the package grid on
    ``N / 2``.  The warning is the contract; the measurement below is
    why it exists."""
    Np, dx = 65, 4e-6
    x = (np.arange(Np) - Np / 2) * dx
    X, Y = np.meshgrid(x, x)
    pupil = ((np.sqrt(X ** 2 + Y ** 2) <= 100e-6)
             * np.exp(1j * X / 40e-6)).astype(complex)
    with pytest.warns(UserWarning, match='half a pixel'):
        odd, _ = compute_psf(pupil, WL, FOCAL, dx, method='mft')
    ref, _ = compute_psf(pupil, WL, FOCAL, dx)
    assert np.abs(odd - ref).max() / ref.max() > 1e-3
    # Even grids are silent and agree (covered above); assert the silence.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        compute_psf(pupil[:64, :64], WL, FOCAL, dx, method='mft')


def test_b8_compute_psf_rejects_a_bad_method_and_a_misplaced_dx_psf():
    pupil = np.ones((16, 16), dtype=complex)
    with pytest.raises(ValueError, match="method must be 'fft' or 'mft'"):
        compute_psf(pupil, WL, FOCAL, 1e-6, method='bluestein')
    with pytest.raises(ValueError, match='only meaningful with'):
        compute_psf(pupil, WL, FOCAL, 1e-6, dx_psf=1e-6)
    with pytest.raises(ValueError, match='positive finite focal'):
        compute_psf(pupil, WL, -FOCAL, 1e-6, method='mft')


# ===========================================================================
# A6.2 -- encircled_energy_profile
# ===========================================================================

def _ee_field(N=96, dx=1e-6, offset=False):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    cx, cy = (4e-6, -3e-6) if offset else (0.0, 0.0)
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (12e-6) ** 2).astype(
        complex), dx


@pytest.mark.parametrize('offset', [False, True])
def test_b8_profile_reproduces_both_consumers_exactly(offset):
    """Sharing the profile is not an approximation of the two calls: the
    curve and the radius come back bit-identical to the un-shared
    path."""
    E, dx = _ee_field(offset=offset)
    prof = encircled_energy_profile(E, dx)
    r0, ee0 = encircled_energy_curve(E, dx, n_radii=37)
    r1, ee1 = encircled_energy_curve(E, dx, n_radii=37, profile=prof)
    assert _same_bits(r0, r1) and _same_bits(ee0, ee1)
    for th in (0.1, 0.5, 0.84, 0.865, 0.99, 1.0):
        assert (encircled_energy_radius(E, dx, threshold=th)
                == encircled_energy_radius(E, dx, threshold=th,
                                           profile=prof))


def _shell_energy_at(prof, th):
    """Energy of the pixel-radius SHELL the threshold falls in.

    A square grid produces ties in the pixel radii constantly ((i, j)
    and (j, i) share one), so the cumulative curve is a staircase whose
    riser at the crossing is a whole shell of pixels.  The forward
    sampler brackets that riser with ``searchsorted(r, ..., 'right')``
    and the inverse with ``searchsorted(p, ..., 'left')``; the two pick
    opposite ends of a riser, so the round-trip cannot be tighter than
    one riser.  This computes that riser FROM THE PROFILE, which makes
    the bar below a property of the data rather than a constant."""
    r_sorted, p_cum, _ = prof
    idx = int(np.searchsorted(p_cum, th, side='left'))
    idx = min(max(idx, 1), r_sorted.size - 1)
    r_hi = r_sorted[idx]
    lo = int(np.searchsorted(r_sorted, r_hi, side='left'))
    hi = int(np.searchsorted(r_sorted, r_hi, side='right'))
    return float(p_cum[min(hi, p_cum.size) - 1]
                 - p_cum[max(lo - 1, 0)])


def test_b8_the_radius_inverts_the_curve_to_within_one_pixel_shell():
    """The v5.30 A-2 contract, now pinned on the SHARED profile: the
    radius inverts the SAME cumulative construction the curve samples,
    so the round trip closes to within ONE pixel-radius shell -- the
    intrinsic resolution of a staircase built from a square grid, and
    the bar is computed from the profile itself at each threshold.

    Two-sided: the same round trip through a profile built on a
    DIFFERENT centre misses by orders more, so the bar is separating a
    real signal and not measuring noise."""
    E, dx = _ee_field()
    prof = encircled_energy_profile(E, dx)
    worst_shell = 0.0
    for th in (0.05, 0.2, 0.4, 0.6, 0.75, 0.84, 0.9, 0.95):
        r = encircled_energy_radius(E, dx, threshold=th, profile=prof)
        _, ee = encircled_energy_curve(E, dx, radii=np.array([r]),
                                       profile=prof)
        shell = _shell_energy_at(prof, th)
        err = abs(float(ee[0]) - th)
        worst_shell = max(worst_shell, shell)
        assert err <= shell + 8 * np.finfo(float).eps, (th, err, shell)

    # Signal above the bar: the SAME round trip through a profile
    # centred 15 um off reads the thresholds at visibly different radii
    # -- measured 0.55 against a worst shell of 8.4e-03, i.e. 66x the
    # bar, so the bar is separating a real disagreement rather than
    # measuring float noise.
    off = encircled_energy_profile(E, dx, centroid=(15e-6, 0.0))
    gap = 0.0
    for th in (0.4, 0.84):
        r = encircled_energy_radius(E, dx, threshold=th, profile=off)
        _, ee = encircled_energy_curve(E, dx, radii=np.array([r]),
                                       profile=prof)
        gap = max(gap, abs(float(ee[0]) - th))
    assert gap > 20 * worst_shell, (gap, worst_shell)


def test_b8_a_supplied_profile_is_never_written_through():
    """``encircled_energy_radius`` clamps ``p_cum`` into [0, 1] in place
    when it owns it.  A caller's profile must come back untouched --
    otherwise handing the same profile to the curve afterwards would be
    reading a mutated array."""
    E, dx = _ee_field()
    prof = encircled_energy_profile(E, dx)
    before = prof[1].copy()
    encircled_energy_radius(E, dx, threshold=0.84, profile=prof)
    assert _same_bits(before, prof[1])


def test_b8_profile_rejects_what_it_cannot_honour():
    E, dx = _ee_field(N=32)
    prof = encircled_energy_profile(E, dx)
    with pytest.raises(ValueError, match='already carries the centre'):
        encircled_energy_radius(E, dx, profile=prof, centroid=(0.0, 0.0))
    with pytest.raises(ValueError, match='already carries the centre'):
        encircled_energy_curve(E, dx, profile=prof, dy=dx)
    with pytest.raises(ValueError, match='3-tuple'):
        encircled_energy_curve(E, dx, profile=object())
    with pytest.raises(ValueError, match='same length'):
        encircled_energy_curve(E, dx,
                               profile=(np.zeros(3), np.zeros(4), 1.0))


def test_b8_profile_of_a_degenerate_field_is_empty_and_both_consumers_cope():
    N, dx = 24, 1e-6
    E = np.zeros((N, N), dtype=complex)
    r_sorted, p_cum, r_max = encircled_energy_profile(E, dx)
    assert r_sorted.size == 0 and p_cum.size == 0
    assert r_max > 0
    r, ee = encircled_energy_curve(E, dx, n_radii=5,
                                   profile=(r_sorted, p_cum, r_max))
    assert np.all(ee == 0.0)
    assert (encircled_energy_radius(E, dx,
                                    profile=(r_sorted, p_cum, r_max))
            == encircled_energy_radius(E, dx))


def test_b8_the_shared_profile_sorts_once_not_twice(monkeypatch):
    """A DECISION, not a wall clock: count the full-grid ``argsort``
    calls the pair makes.  Two without the profile, one with it."""
    E, dx = _ee_field(N=64)
    calls = []
    real = np.argsort

    def counting(a, *args, **kw):
        calls.append(np.size(a))
        return real(a, *args, **kw)

    monkeypatch.setattr(np, 'argsort', counting)
    encircled_energy_curve(E, dx)
    encircled_energy_radius(E, dx)
    unshared = sum(1 for n in calls if n == E.size)
    calls.clear()
    prof = encircled_energy_profile(E, dx)
    encircled_energy_curve(E, dx, profile=prof)
    encircled_energy_radius(E, dx, profile=prof)
    shared = sum(1 for n in calls if n == E.size)
    assert unshared == 2 and shared == 1


def test_b8_profile_holds_no_reference_to_the_field_and_is_not_cached():
    """Audit sec. 15.5: the profile is a plain value.  Two calls on
    equal-but-distinct arrays must each compute (no content key), and
    the returned arrays must not alias the input."""
    E, dx = _ee_field(N=32)
    p1 = encircled_energy_profile(E, dx)
    p2 = encircled_energy_profile(E.copy(), dx)
    assert p1[0] is not p2[0] and p1[1] is not p2[1]
    assert _same_bits(p1[1], p2[1])
    assert not np.shares_memory(p1[0], E) and not np.shares_memory(p1[1], E)


# ===========================================================================
# A6.3 -- the Zernike recurrence and the shared power table
# ===========================================================================

def _R_exact(n, m, r: Fraction) -> Fraction:
    """R_n^m(r) in EXACT rational arithmetic.  The radial polynomial has
    integer coefficients, so a rational rho gives a rational value: this
    is an oracle with no floating point and no library in it at all."""
    m = abs(m)
    tot = Fraction(0)
    for s in range((n - m) // 2 + 1):
        num = (-1) ** s * math.factorial(n - s)
        den = (math.factorial(s) * math.factorial((n + m) // 2 - s)
               * math.factorial((n - m) // 2 - s))
        tot += Fraction(num, den) * r ** (n - 2 * s)
    return tot


def _factorial_sum(n, m, rho):
    """The pre-B8 closed-form sum, restated locally."""
    m = abs(m)
    R = np.zeros_like(rho)
    for s in range((n - m) // 2 + 1):
        num = ((-1) ** s) * math.factorial(n - s)
        den = (math.factorial(s) * math.factorial((n + m) // 2 - s)
               * math.factorial((n - m) // 2 - s))
        R = R + (num / den) * rho ** (n - 2 * s)
    return R


_RHO_DEN = 64
_RHO_K = np.arange(_RHO_DEN + 1)
_RHO = _RHO_K / _RHO_DEN


@pytest.mark.parametrize('n', list(range(0, 9)))
def test_b8_shipped_table_orders_are_untouched_and_still_exact(n):
    """Every (n, m) at or below the highest order this module ships a
    name for (n = 8, ``_zernike_classical_name``) must still be the
    factorial sum, bit for bit, AND must still match the exact rational
    oracle to the float64 floor.

    Measured worst over n <= 8: the sum is EXACT (0.0) through n = 6 and
    7.1e-15 at n = 8.  Bar 1e-13 -- two decades over the measurement and
    ten below the 1.5e-9 the sum reaches at n = 22, where the
    implementation switches."""
    for m in range(-n, n + 1):
        if (n - abs(m)) % 2:
            continue
        got = _zk._zernike_radial(n, m, _RHO)
        assert _same_bits(got, _factorial_sum(n, m, _RHO)), (n, m)
        exact = np.array([float(_R_exact(n, m, Fraction(int(k), _RHO_DEN)))
                          for k in _RHO_K])
        scale = max(float(np.abs(exact).max()), 1e-300)
        assert np.abs(got - exact).max() / scale < 1e-13, (n, m)


def test_b8_the_recurrence_takes_over_exactly_where_the_sum_has_lost_nine_digits():
    """The switch is one order, and both sides of it are asserted.

    At ``n = _ZERNIKE_RECURRENCE_MIN_N - 1`` the shipped function is
    still bit-for-bit the factorial sum.  At ``_ZERNIKE_RECURRENCE_MIN_N``
    it is no longer, and the exact rational oracle says the NEW answer is
    the better one by more than three decades -- which is what makes the
    boundary a defect line rather than a preference."""
    n_lo = _zk._ZERNIKE_RECURRENCE_MIN_N - 1
    n_hi = _zk._ZERNIKE_RECURRENCE_MIN_N
    for m in (0, 2, n_lo):
        if (n_lo - abs(m)) % 2 == 0:
            assert _same_bits(_zk._zernike_radial(n_lo, m, _RHO),
                              _factorial_sum(n_lo, m, _RHO)), (n_lo, m)

    worse = better = None
    for m in range(0, n_hi + 1, 2):
        exact = np.array([float(_R_exact(n_hi, m, Fraction(int(k), _RHO_DEN)))
                          for k in _RHO_K])
        scale = max(float(np.abs(exact).max()), 1e-300)
        e_sum = np.abs(_factorial_sum(n_hi, m, _RHO) - exact).max() / scale
        e_new = np.abs(_zk._zernike_radial(n_hi, m, _RHO) - exact).max() / scale
        worse = e_sum if worse is None else max(worse, e_sum)
        better = e_new if better is None else max(better, e_new)
    assert worse > 1e-9, f'the factorial sum measured {worse:.3e} at n={n_hi}'
    assert better < 1e-13, f'the recurrence measured {better:.3e}'
    assert worse / max(better, 1e-300) > 1e3


@pytest.mark.parametrize('n', [22, 26, 30, 34, 40])
def test_b8_the_recurrence_holds_the_float64_floor_out_to_its_stability_limit(n):
    """Stability limit, stated and measured: Kintner stays at the float64
    floor where the factorial sum has fallen apart.  Bar 1e-13 (measured
    worst 3.9e-15 at n <= 32, 3.0e-15 at n = 40); the factorial sum at
    n = 40 measures 3.2e-03, ten decades worse."""
    for m in (0, 2, n // 2 * 2 if (n - n // 2 * 2) % 2 == 0 else 0, n):
        if (n - abs(m)) % 2:
            continue
        exact = np.array([float(_R_exact(n, m, Fraction(int(k), _RHO_DEN)))
                          for k in _RHO_K])
        scale = max(float(np.abs(exact).max()), 1e-300)
        got = _zk._zernike_radial_kintner(n, m, _RHO)
        assert np.abs(got - exact).max() / scale < 1e-13, (n, m)


@pytest.mark.parametrize('N,n_modes', [(64, 15), (64, 21), (96, 36),
                                       (48, 66)])
def test_b8_basis_build_is_bit_identical_to_the_per_mode_loop(N, n_modes):
    """The shared power table and the hoisted pupil mask change which
    array object an exponent comes from, never its value.  Compared
    against the pre-B8 per-mode loop restated locally."""
    x = (np.arange(N) - N / 2) / (N / 2)
    X, Y = np.meshgrid(x, x)
    _zk.clear_zernike_basis_cache()
    basis, mask = zernike_basis_matrix(n_modes, X, Y, 0.87)

    r_sq = (X ** 2 + Y ** 2) / (0.87 ** 2)
    pm = r_sq <= 1.0
    rho = np.sqrt(r_sq[pm])
    theta = np.arctan2(Y[pm], X[pm])
    want = np.empty((rho.size, n_modes), dtype=np.float64)
    for j in range(n_modes):
        n, m = _zk.zernike_index_to_nm(j)
        Nn = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
        R = _factorial_sum(n, m, rho)
        ang = np.cos(m * theta) if m >= 0 else np.sin(-m * theta)
        want[:, j] = np.where(rho <= 1.0, Nn * R * ang, 0.0)
    assert _same_bits(mask, pm)
    assert _same_bits(basis, want)


def test_b8_public_zernike_polynomial_is_untouched():
    rho = np.linspace(0, 1.4, 401)
    theta = np.linspace(-np.pi, np.pi, 401)
    for n in range(0, 10):
        for m in range(-n, n + 1):
            if (n - abs(m)) % 2:
                continue
            Nn = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
            R = _factorial_sum(n, m, rho)
            ang = np.cos(m * theta) if m >= 0 else np.sin(-m * theta)
            want = np.where(rho <= 1.0, Nn * R * ang, 0.0)
            assert _same_bits(zernike_polynomial(n, m, rho, theta), want)


def test_b8_zernike_polynomial_still_handles_the_awkward_argument_shapes():
    """The in-place multiply is gated on float64 and a shape that
    broadcasts INTO the accumulator; everything else keeps the original
    expression.  Pin the gated-out cases against it."""
    rho32 = np.linspace(0, 1, 17, dtype=np.float32)
    th32 = np.linspace(0, 1, 17, dtype=np.float32)
    R = _factorial_sum(2, 0, rho32)
    want = np.where(rho32 <= 1.0, np.sqrt(3.0) * R * np.cos(0 * th32), 0.0)
    assert _same_bits(zernike_polynomial(2, 0, rho32, th32), want)

    # scalar rho against an array theta: the result must take theta's
    # shape, which an out= into the scalar accumulator could not.
    out = zernike_polynomial(3, 1, np.float64(0.5), np.linspace(0, 1, 9))
    assert out.shape == (9,)

    rho_int = np.arange(3)
    out_i = zernike_polynomial(2, 0, rho_int, np.zeros(3))
    assert out_i.dtype == np.float64


def test_b8_power_table_is_bounded_by_the_basis_it_builds():
    """The memo needs no budget knob: it holds ``n_max + 1`` columns
    against the basis's ``n_modes``, i.e. ``2 / (n_max + 2)`` of the
    array the function already returns."""
    for n_modes in (1, 3, 6, 15, 21, 36, 66, 231):
        n_max = _zk.zernike_index_to_nm(n_modes - 1)[0]
        ratio = (n_max + 1) / n_modes
        assert ratio <= 1.0
        if n_modes >= 3:
            assert ratio <= 2.0 / 3.0 + 1e-12, (n_modes, ratio)


# ---------------------------------------------------------------------------
# A6.3 -- the DM influence functions
# ---------------------------------------------------------------------------

def test_b8_grid_views_are_bit_identical_to_the_dense_meshgrid():
    dm = DeformableMirror(n_actuators=4, pitch=5e-3, dx=1e-3, N=48,
                          cache_basis=False)
    X, Y = dm._grid_views()
    x = (np.arange(48) - 48 / 2) * 1e-3
    Xd, Yd = np.meshgrid(x, x)
    assert _same_bits((X - 1e-3) ** 2 + (Y - 2e-3) ** 2,
                      (Xd - 1e-3) ** 2 + (Yd - 2e-3) ** 2)
    Xb, Yb = dm._grid_views(7, 19)
    Xdb, Ydb = np.meshgrid(x, x[7:19])
    assert _same_bits(Xb + Yb, Xdb + Ydb)


@pytest.mark.parametrize('n_act,N,rows_band', [(3, 32, 7), (4, 24, 24),
                                               (5, 20, 1)])
def test_b8_banded_if_apply_matches_the_materialised_normal_equations(
        n_act, N, rows_band):
    """``_banded_IF_apply`` is the hoisted band construction.  Its normal
    equations must be the ones an explicit design matrix gives, to the
    gemm-blocking floor.

    Bar: ``A^T A`` is a sum of ``N^2`` products, so a different blocking
    of the same reduction differs by ``O(N^2 eps)`` relative = 5.7e-13
    at N = 32.  Measured worst 1e-15 relative."""
    dm = DeformableMirror(n_actuators=n_act, pitch=6e-3, dx=1e-3, N=N,
                          cache_basis=False)
    n2 = n_act ** 2
    rng = np.random.default_rng(11)
    target = rng.standard_normal((N, N))
    AtA, Atb = dm._banded_IF_apply(target, n2, rows_band)
    A = np.empty((N * N, n2))
    for k in range(n2):
        A[:, k] = dm._influence_function_kth(k).ravel()
    want_AtA = A.T @ A
    want_Atb = A.T @ target.ravel()
    bar = 4.0 * (N * N) * np.finfo(float).eps
    assert np.abs(AtA - want_AtA).max() / np.abs(want_AtA).max() < bar
    assert np.abs(Atb - want_Atb).max() / np.abs(want_Atb).max() < bar


def test_b8_dm_cache_decision_did_not_move_and_the_warning_is_one_shot():
    """The ``'auto'`` boundary is untouched -- the cached and the lazy
    ``phase()`` sum in different orders, so moving it would move the
    delivered phase map.  What is new is that an eager stack above
    ``_IF_CACHE_WARN_BYTES`` says so, once, at construction."""
    ceiling = _ao._DEFAULT_CACHE_CEILING_BYTES
    warn_at = _ao._IF_CACHE_WARN_BYTES
    assert warn_at == ceiling // 2

    # Exactly ON the ceiling: still cached (the comparison is <=), which
    # is the audit's own 16x16-on-512 case, and now warned about.
    n_act, N = 16, 512
    assert (n_act ** 2) * (N ** 2) * 8 == ceiling
    with pytest.warns(UserWarning, match='influence-function stack'):
        dm = DeformableMirror(n_actuators=n_act, pitch=1e-3, dx=1e-4, N=N)
    assert dm._cache_active is True
    assert dm._IF_basis is not None
    del dm
    gc.collect()

    # Below the warn threshold: silent, and still cached.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        small = DeformableMirror(n_actuators=4, pitch=6e-3, dx=1e-3, N=64)
    assert small._cache_active is True

    # cache_basis=False never warns, whatever the size.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        lazy = DeformableMirror(n_actuators=16, pitch=1e-3, dx=1e-4, N=512,
                                cache_basis=False)
    assert lazy._cache_active is False and lazy._IF_basis is None


def test_b8_dm_phase_and_fit_are_unchanged_on_both_cache_paths():
    """Nothing about the DM's numbers moved: both ``phase()`` paths and
    both ``fit_phase`` paths are pinned against a local restatement."""
    n_act, N = 5, 40
    for cache in (True, False):
        dm = DeformableMirror(n_actuators=n_act, pitch=6e-3, dx=1e-3, N=N,
                              cache_basis=cache)
        cmd = np.linspace(-1, 1, n_act ** 2).reshape(n_act, n_act)
        dm.set_command(cmd)
        x = (np.arange(N) - N / 2) * 1e-3
        Xd, Yd = np.meshgrid(x, x)
        s2 = dm._sigma_IF ** 2
        if cache:
            stack = np.empty((n_act, n_act, N, N))
            for i, xi in enumerate(dm._act_centres):
                for j, yj in enumerate(dm._act_centres):
                    stack[j, i] = np.exp(
                        -((Xd - xi) ** 2 + (Yd - yj) ** 2) / (2.0 * s2))
            want = np.einsum('ij,ijkl->kl', dm.command, stack)
        else:
            want = np.zeros((N, N))
            for j, yj in enumerate(dm._act_centres):
                for i, xi in enumerate(dm._act_centres):
                    a = dm.command[j, i]
                    if a == 0.0:
                        continue
                    want += a * np.exp(
                        -((Xd - xi) ** 2 + (Yd - yj) ** 2) / (2.0 * s2))
        assert _same_bits(dm.phase(), want)


# ===========================================================================
# A11 sec. 6.1 -- Gori pseudo-modes
# ===========================================================================

def _mu_along_x(phi):
    """``<phi(r) conj(phi(r + d x))> / <|phi|^2>`` by LINEAR (unwrapped)
    pair averaging -- it counts no wrapped pair, so it cannot itself
    manufacture periodicity."""
    n, Ny, Nx = phi.shape
    denom = float(np.mean(np.abs(phi) ** 2))
    return np.array([np.mean(phi[:, :, :Nx - d] * np.conj(phi[:, :, d:]))
                     / denom for d in range(Nx)])


def test_b8_default_generator_is_fft_and_byte_identical():
    """The default must not move: naming ``'fft'`` explicitly, and
    saying nothing, must give the same array bit for bit -- and that
    array must be the one the pre-B8 signature produced for the same
    seed (which the ``pad_sigma=0.0`` escape hatch and the A11 pins
    already cover)."""
    kw = dict(Ny=24, Nx=24, dx=1e-6, dy=1e-6, coherence_length=3e-6,
              n_realizations=3)
    a = _schell_phase_realizations(rng=np.random.default_rng(5), **kw)
    b = _schell_phase_realizations(rng=np.random.default_rng(5),
                                   generator='fft', **kw)
    assert _same_bits(a, b)
    c, *_ = create_gaussian_schell_source(
        N=24, dx=1e-6, wavelength=WL, w0=8e-6, sigma_g=3e-6,
        n_realizations=3, rng=5)
    d, *_ = create_gaussian_schell_source(
        N=24, dx=1e-6, wavelength=WL, w0=8e-6, sigma_g=3e-6,
        n_realizations=3, rng=5, generator='fft')
    assert _same_bits(c, d)


@pytest.mark.parametrize('frac', [3.0, 8.0])
def test_b8_pseudo_modes_realise_the_gaussian_kernel(frac):
    """The realised two-point correlation against the target, for BOTH
    generators on the same grid and the same ensemble size.

    Bar: the ensemble estimate of a correlation from ``n_real``
    realisations of a field holding ``(L/sigma)^2`` coherence cells has
    a standard error ~``1 / sqrt(n_real * cells)``; at 400 x 64 that is
    0.0063, and the bar is 6 sigma of it.  The claim is not that the
    modes generator is better -- it is that both land inside the
    SAMPLING error of the same target, which the pre-Z2 periodised
    kernel does not (asserted in the next test)."""
    N, dx = 64, 1e-6
    L = N * dx
    sigma = L / frac
    n_real = 400
    cells = (L / sigma) ** 2
    se = 1.0 / math.sqrt(n_real * cells)
    bar = 6.0 * se
    d = np.arange(N) * dx
    target = np.exp(-d ** 2 / (2 * sigma ** 2))
    for gen in ('fft', 'modes'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            phi = _schell_phase_realizations(
                Ny=N, Nx=N, dx=dx, dy=dx, coherence_length=sigma,
                n_realizations=n_real, rng=np.random.default_rng(1234),
                generator=gen)
        err = np.abs(np.abs(_mu_along_x(phi)) - target).max()
        assert err < bar, f'{gen}: max|mu - target| = {err:.4f} vs {bar:.4f}'


def test_b8_pseudo_modes_cannot_manufacture_edge_coherence():
    """The failure Z2 fixed cannot arise in this generator at all: there
    is no grid in the construction, so there is nothing to periodise.

    Fail-before is measured in this process on the pre-Z2 path
    (``generator='fft', pad_sigma=0.0``), which reads ~1 at the largest
    separation the grid can form where the model says ~1e-13."""
    N, dx = 64, 1e-6
    L = N * dx
    sigma = L / 8
    true_edge = math.exp(-((N - 1) * dx) ** 2 / (2 * sigma ** 2))
    assert true_edge < 1e-12

    kw = dict(Ny=N, Nx=N, dx=dx, dy=dx, coherence_length=sigma,
              n_realizations=300)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pre = _schell_phase_realizations(
            rng=np.random.default_rng(2), pad_sigma=0.0, **kw)
        modes = _schell_phase_realizations(
            rng=np.random.default_rng(2), generator='modes', **kw)
    edge_pre = abs(_mu_along_x(pre)[-1])
    edge_modes = abs(_mu_along_x(modes)[-1])
    assert edge_pre > 0.5, f'pre-Z2 arm measured {edge_pre:.4f}'
    assert edge_modes < 0.05, f'modes arm measured {edge_modes:.4f}'


@pytest.mark.parametrize('M', [8, 32, 128, 512])
def test_b8_pseudo_mode_marginal_is_circular_gaussian_as_M_grows(M):
    """Two statements about the marginal at a fixed point.

    1. The EXACT finite-M moment of a random-phasor sum,
       ``E[I^2] / E[I]^2 = 2 - 1/M`` -- an oracle with a closed form at
       every M, not an asymptotic one.  Bar: the estimator's own
       standard error.  ``I`` is (nearly) exponential, so ``I^2`` has
       relative standard deviation ``sqrt(20)`` and the mean of ``n``
       independent samples carries ``sqrt(20 / n)``; the bar is 5 of
       those.
    2. A chi-square goodness-of-fit of ``|phi|^2`` against ``Exp(1)`` on
       20 equiprobable bins.  Samples are taken one per ``4 sigma`` cell
       and one per realisation is not enough, so the grid is subsampled
       at a spacing where the model's own correlation is
       ``exp(-8) = 3e-4``; the statistic then has 19 degrees of freedom
       and the bar is the 0.1 % critical value, 43.82.  The test is
       two-sided in M: the SAME machinery is applied to the 'fft'
       generator as a control, which must also pass."""
    N, dx = 64, 1e-6
    sigma = 4.0 * dx
    n_real = 240
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        phi = _schell_phase_realizations(
            Ny=N, Nx=N, dx=dx, dy=dx, coherence_length=sigma,
            n_realizations=n_real, rng=np.random.default_rng(20260913),
            generator='modes', n_pseudo_modes=M)
    step = max(1, int(round(4.0 * sigma / dx)))
    sub = phi[:, ::step, ::step]
    I = (np.abs(sub) ** 2).ravel()
    n = I.size
    mean_I = float(I.mean())
    ratio = float((I ** 2).mean()) / mean_I ** 2
    exact = 2.0 - 1.0 / M
    se = math.sqrt(20.0 / n)
    assert abs(ratio - exact) < 5.0 * se, (
        f'M={M}: measured E[I^2]/E[I]^2 = {ratio:.4f} against the exact '
        f'{exact:.4f} (5 s.e. = {5 * se:.4f})')
    assert abs(mean_I - 1.0) < 5.0 / math.sqrt(n)

    # 20 equiprobable Exp(1) bins; the last edge is +inf by definition
    # (built separately so the 1 - 20/20 = 0 log is never taken).
    edges = np.concatenate([-np.log(1.0 - np.arange(20) / 20.0), [np.inf]])
    obs, _ = np.histogram(I / mean_I, bins=edges)
    expect = np.full(20, n / 20.0)
    chi2 = float(((obs - expect) ** 2 / expect).sum())
    if M >= _GORI_MIN_MODES:
        assert chi2 < 43.82, f'M={M}: chi2(19) = {chi2:.1f}'


def test_b8_the_fft_generator_passes_the_same_marginal_test():
    """Control arm for the chi-square above: the estimator is not
    rigged, the padded-FFT generator passes it too."""
    N, dx = 64, 1e-6
    sigma = 4.0 * dx
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        phi = _schell_phase_realizations(
            Ny=N, Nx=N, dx=dx, dy=dx, coherence_length=sigma,
            n_realizations=240, rng=np.random.default_rng(4242))
    sub = phi[:, ::16, ::16]
    I = (np.abs(sub) ** 2).ravel()
    mean_I = float(I.mean())
    # 20 equiprobable Exp(1) bins; the last edge is +inf by definition
    # (built separately so the 1 - 20/20 = 0 log is never taken).
    edges = np.concatenate([-np.log(1.0 - np.arange(20) / 20.0), [np.inf]])
    obs, _ = np.histogram(I / mean_I, bins=edges)
    expect = np.full(20, I.size / 20.0)
    assert float(((obs - expect) ** 2 / expect).sum()) < 43.82


def test_b8_mode_count_heuristic_is_the_coherence_cell_census():
    L = 64e-6
    # Mid-range: the raw (L/sigma)^2 census.
    sigma = L / 40.0
    assert _gori_mode_count(L, L, sigma) == 1600
    # Clamped below by the 1/M contrast floor.
    assert _gori_mode_count(L, L, L / 4.0) == _GORI_MIN_MODES
    # Clamped above by the cost cap.
    assert _gori_mode_count(L, L, L / 500.0) == _GORI_MAX_MODES
    # Anisotropic grid: the census is the product of the two axes.
    assert _gori_mode_count(2 * L, L, L / 20.0) == 800


def test_b8_mode_generator_warns_when_the_cap_binds_and_is_silent_otherwise():
    N, dx = 32, 1e-6
    kw = dict(Ny=N, Nx=N, dx=dx, dy=dx, n_realizations=1,
              generator='modes')
    with pytest.warns(UserWarning, match='capped at'):
        _schell_phase_realizations(
            coherence_length=N * dx / 200.0,
            rng=np.random.default_rng(1), **kw)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _schell_phase_realizations(
            coherence_length=N * dx / 10.0,
            rng=np.random.default_rng(1), **kw)
    # An explicit n_pseudo_modes overrides the cap without a warning.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = _schell_phase_realizations(
            coherence_length=N * dx / 200.0, n_pseudo_modes=64,
            rng=np.random.default_rng(1), **kw)
    assert out.shape == (1, N, N)


def test_b8_mode_generator_rejects_what_it_cannot_honour():
    kw = dict(Ny=16, Nx=16, dx=1e-6, dy=1e-6, coherence_length=3e-6,
              n_realizations=1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="'fft' or 'modes'"):
        _schell_phase_realizations(generator='gori', **kw)
    with pytest.raises(ValueError, match='no meaning for'):
        _schell_phase_realizations(generator='modes', pad_sigma=0.0, **kw)
    with pytest.raises(ValueError, match='only meaningful with'):
        _schell_phase_realizations(n_pseudo_modes=64, **kw)
    with pytest.raises(ValueError, match='positive integer'):
        _schell_phase_realizations(generator='modes', n_pseudo_modes=0, **kw)


def test_b8_public_factories_carry_the_generator_through():
    N = 32
    for factory, extra in (
            (create_gaussian_schell_source,
             dict(w0=8e-6, sigma_g=4e-6)),
            (create_schell_model_source,
             dict(intensity_profile=np.ones((N, N)),
                  coherence_length=4e-6))):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E, dx_o, dy_o, wl = factory(
                N=N, dx=1e-6, wavelength=WL, n_realizations=4, rng=3,
                generator='modes', **extra)
        assert E.shape == (4, N, N)
        assert np.isfinite(E).all()
        # Unit mean intensity in expectation is the ensemble contract the
        # FFT generator also carries; the pseudo-mode sum has it exactly.
        env = np.abs(factory(N=N, dx=1e-6, wavelength=WL, n_realizations=1,
                             rng=3, **extra)[0][0])
        assert env.shape == (N, N)


def test_b8_pseudo_mode_transient_is_a_fraction_of_the_padded_fft():
    """Peak transient in units of the OUTPUT ensemble.

    Derived: the pseudo-mode path holds the ensemble it is filling plus
    a bounded number of rank-``M`` panels -- ``A`` is ``(Ny, M)``, ``B``
    is ``(M, Nx)``, and building each costs one same-sized temporary --
    so at most ~6 panels, each ``M / (n_real * N)`` of the ensemble.
    The FFT path instead works on a grid padded by ``4 sigma`` per side,
    which at ``sigma = L / 8`` is 2x the grid per axis, i.e. 4x the
    AREA, several times over."""
    N, dx = 128, 1e-6
    sigma = N * dx / 8
    n_real = 4
    out_bytes = n_real * N * N * 16
    kw = dict(Ny=N, Nx=N, dx=dx, dy=dx, coherence_length=sigma,
              n_realizations=n_real)
    M = _gori_mode_count(N * dx, N * dx, sigma)
    panel = M / (n_real * N)

    def run(gen):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return _schell_phase_realizations(
                rng=np.random.default_rng(5), generator=gen, **kw)

    fft_peak = _peak_grids(lambda: run('fft'), out_bytes)
    mod_peak = _peak_grids(lambda: run('modes'), out_bytes)
    assert mod_peak < 1.0 + 6.0 * panel, (mod_peak, panel)
    assert fft_peak > 2.0 + 6.0 * panel, fft_peak
    assert mod_peak < fft_peak, (mod_peak, fft_peak)


# ===========================================================================
# A11 sec. 6.2 -- create_gaussian_beam(geometry_dtype=)
# ===========================================================================

@pytest.mark.parametrize('N', [17, 64, 256])
@pytest.mark.parametrize('normalize', ['peak', 'power', 'none'])
@pytest.mark.parametrize('dt', [np.complex128, np.complex64])
def test_b8_geometry_dtype_default_is_bit_identical(N, normalize, dt):
    """The default must not move: ``geometry_dtype=None`` and
    ``geometry_dtype=np.float64`` are the same call."""
    kw = dict(w0=8e-6, x0=2e-6, y0=-1e-6, dy=2e-6, normalize=normalize,
              dtype=dt)
    a, ax, ay = create_gaussian_beam(N, 1e-6, WL, **kw)
    b, bx, by = create_gaussian_beam(N, 1e-6, WL,
                                     geometry_dtype=np.float64, **kw)
    assert _same_bits(a, b) and _same_bits(ax, bx) and _same_bits(ay, by)


@pytest.mark.parametrize('N', [64, 512])
@pytest.mark.parametrize('normalize', ['peak', 'power', 'none'])
@pytest.mark.parametrize('off', [False, True])
def test_b8_geometry_dtype_float32_stays_inside_one_float32_ulp(
        N, normalize, off):
    """Documented tolerance for the opt-in.  A float32 exponent carries
    24 bits, so the field it produces can differ from the
    float64-then-cast one by a few ULP of float32.

    Bar: 8 * eps(float32) = 9.5e-07 relative to the peak -- three ULP of
    the container the result lives in.  Measured worst over these twelve
    configurations: 1.2e-07."""
    kw = dict(w0=9e-6, normalize=normalize, dtype=np.complex64)
    if off:
        kw.update(x0=3e-6, y0=-4e-6)
    ref, _, _ = create_gaussian_beam(N, 1e-6, WL, **kw)
    got, gx, gy = create_gaussian_beam(N, 1e-6, WL,
                                       geometry_dtype=np.float32, **kw)
    assert got.dtype == np.dtype(np.complex64)
    assert gx.dtype == np.dtype(np.float64) and gy.dtype == np.dtype(
        np.float64), 'the returned axes are the caller\'s coordinates'
    peak = float(np.abs(ref).max())
    rel = float(np.abs(got - ref).max()) / peak
    assert rel < 8 * np.finfo(np.float32).eps, f'{rel:.3e}'


def test_b8_geometry_dtype_float32_transient_is_smaller():
    """Derived count: the exponent buffer is one real full grid.  In
    float64 that is 8 bytes/pixel against the complex64 output's 8, so
    the peak is 2.00 outputs; in float32 it is 4, so 1.50."""
    N = 1024
    out_bytes = N * N * 8
    f64 = _peak_grids(lambda: create_gaussian_beam(
        N, 1e-6, WL, w0=20e-6, dtype=np.complex64)[0], out_bytes)
    f32 = _peak_grids(lambda: create_gaussian_beam(
        N, 1e-6, WL, w0=20e-6, dtype=np.complex64,
        geometry_dtype=np.float32)[0], out_bytes)
    assert f64 > 1.75, f64
    assert f32 < 1.75, f32


def test_b8_geometry_dtype_rejects_a_double_precision_request():
    with pytest.raises(ValueError, match='needs a complex64 output'):
        create_gaussian_beam(32, 1e-6, WL, w0=8e-6,
                             dtype=np.complex128,
                             geometry_dtype=np.float32)
    with pytest.raises(ValueError, match='must be'):
        create_gaussian_beam(32, 1e-6, WL, w0=8e-6, dtype=np.complex64,
                             geometry_dtype=np.float16)


# ===========================================================================
# A11 sec. 6.3 -- apply_jones_matrix
# ===========================================================================

def _pre_fix_jones(J, Ex, Ey):
    """The pre-B8 expression, verbatim."""
    return (J[0, 0] * Ex + J[0, 1] * Ey,
            J[1, 0] * Ex + J[1, 1] * Ey)


@pytest.mark.parametrize('N', [8, 33, 64])
@pytest.mark.parametrize('dt', [np.complex128, np.complex64])
@pytest.mark.parametrize('special', [None, 'dark', 'nan', 'tiny'])
def test_b8_apply_jones_matrix_is_bit_identical(N, dt, special):
    rng = np.random.default_rng(N * 7 + len(str(special)))
    Ex = (rng.standard_normal((N, N))
          + 1j * rng.standard_normal((N, N))).astype(dt)
    Ey = (rng.standard_normal((N, N))
          + 1j * rng.standard_normal((N, N))).astype(dt)
    if special == 'dark':
        Ex[0, 0] = Ey[0, 0] = 0
    elif special == 'nan':
        Ex[1, 1] = np.nan
        Ey[2, 2] = np.inf
    elif special == 'tiny':
        Ex[3, 3] = Ey[3, 3] = 1e-160
    J = np.array([[0.3 + 0.4j, -0.5 + 0.1j],
                  [0.2 - 0.7j, 0.9 + 0.05j]], dtype=complex)
    with np.errstate(all='ignore'):
        want_x, want_y = _pre_fix_jones(J, Ex, Ey)
        out = apply_jones_matrix(JonesField(Ex.copy(), Ey.copy(), 1e-6, WL),
                                 J)
    assert _same_bits(out.Ex, want_x)
    assert _same_bits(out.Ey, want_y)


def test_b8_apply_jones_matrix_is_bit_identical_for_a_spatial_callable():
    N = 48
    rng = np.random.default_rng(9)
    Ex = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    Ey = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))

    def cb(X, Y):
        s = np.sin(X / 1e-5)
        return np.array([[0.5 + s, 0.2 * s], [-0.3 * s, 0.8 - s]],
                        dtype=complex)

    x = (np.arange(N) - N / 2) * 1e-6
    X, Y = np.meshgrid(x, x)
    J = cb(X, Y)
    want_x, want_y = _pre_fix_jones(J, Ex, Ey)
    out = apply_jones_matrix(JonesField(Ex.copy(), Ey.copy(), 1e-6, WL), cb)
    assert _same_bits(out.Ex, want_x) and _same_bits(out.Ey, want_y)


def test_b8_apply_jones_matrix_is_bit_identical_on_a_mixed_precision_field():
    """A mixed-precision field with a ``complex128`` MATRIX, which does
    NOT engage the dtype gate and is pinned for that reason.

    ``np.asarray(matrix, dtype=complex)`` makes every array-form Jones
    matrix ``complex128``, and under NEP 50 a NumPy scalar is strong, so
    ``J[0,0] * Ex`` is ``complex128`` even for a ``complex64`` ``Ex``:
    both products land in ``complex128``, the in-place add is taken, and
    it is bit-identical because nothing narrows.  The case that DOES
    engage the gate needs a lower-precision matrix and lives in
    ``test_verifyb8_apply_jones_matrix_falls_back_when_the_two_products_
    disagree`` below."""
    N = 16
    rng = np.random.default_rng(2)
    Ex = (rng.standard_normal((N, N))
          + 1j * rng.standard_normal((N, N))).astype(np.complex64)
    Ey = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    J = np.array([[0.3 + 0.4j, -0.5 + 0.1j],
                  [0.2 - 0.7j, 0.9 + 0.05j]], dtype=complex)
    want_x, want_y = _pre_fix_jones(J, Ex, Ey)
    out = apply_jones_matrix(JonesField(Ex.copy(), Ey.copy(), 1e-6, WL), J)
    assert _same_bits(out.Ex, want_x) and _same_bits(out.Ey, want_y)
    assert out.Ex.dtype == want_x.dtype


#: How far above a whole number of full grids a tracemalloc peak may sit.
#: DERIVED, two-sided: the reading is an EXACT allocation count in units of one
#: full COMPLEX grid plus tracemalloc's own bookkeeping, and that bookkeeping
#: measured 448 .. 7760 B over 9 repeats on each of two arms (Windows py3.14 /
#: numpy 2.4.4 and WSL py3.12 / numpy 2.4.6), i.e. at most 4.6e-04 grids at
#: N = 1024.  0.05 is two decades above that spread and 1.3 decades below the
#: 1.0 that separates one allocation count from the next.
_B8_PEAK_SLACK = 0.05


def _numpy_elides_the_sum_of_two_temporaries(J, Ex, Ey, grid):
    """MEASURED premise: does this build rewrite ``a*X + b*Y`` into one of the
    two products' own buffers instead of allocating a third array?

    NumPy's ``temp_elide.c`` does that when an operand of a binary op is an
    unreferenced temporary, but only where the optimisation is compiled in (it
    needs ``backtrace()``) and only when its stack walk can confirm the
    temporary came from the interpreter.  Both are BUILD properties, and they
    do NOT follow the operating system: on CI run 34914295323 this sum was
    elided on the py3.14 Linux runner and NOT elided on the py3.12 and py3.13
    ones, while a DIFFERENT elidable pattern (``Ex * conj(Ey)``, gated in
    ``test_audit2609_a11_polar_sources_infra.py``) was elided on py3.11.  Two
    patterns, two premises, each measured where it is used -- never inferred
    from the platform, and never from each other.

    Returns ``(free, held)`` in full-grid COMPLEX arrays.  ``held`` binds both
    products to names, which lifts their reference counts to 2 and puts
    elision out of reach on every build: it is 3.00 everywhere, and is
    asserted, so a reading of "no elision here" can never come from an
    instrument that measured nothing at all.  ``free`` is 3.00 where elision
    is unavailable and 2.00 where it is.  MEASURED: 3.000 / 3.000 on Windows
    py3.14, 2.000 / 3.000 on WSL py3.12.
    """
    free = _peak_grids(lambda: J[0, 0] * Ex + J[0, 1] * Ey, grid)

    def _held():
        a = J[0, 0] * Ex                  # named -> refcount 2 -> not elidable
        b = J[0, 1] * Ey
        return a + b

    held = _peak_grids(_held, grid)
    assert 2.9 < held < 3.1, (
        f"the elision instrument read {held:.3f} full grids for two products "
        f"and their sum, which must cost exactly 3.00 -- it is not measuring "
        f"what it claims, so no premise can be drawn from it")
    return free, held


def _b8_jones_peaks(N=1024):
    """``(grid_bytes, J, Ex, Ey, old, new)`` -- the two peak readings, in units
    of ONE full-grid complex array."""
    grid = N * N * 16
    rng = np.random.default_rng(1)
    Ex = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    Ey = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    J = np.array([[0.3 + 0.4j, -0.5 + 0.1j],
                  [0.2 - 0.7j, 0.9 + 0.05j]], dtype=complex)
    old = _peak_grids(lambda: _pre_fix_jones(J, Ex, Ey), grid)
    new = _peak_grids(lambda: apply_jones_matrix(
        JonesField(Ex, Ey, 1e-6, WL), J), grid)
    return grid, J, Ex, Ey, old, new


def test_b8_apply_jones_matrix_peak_full_grid_arrays():
    """The UNCONDITIONAL half: the shipped path sits ON its derived floor, in
    units of ONE full-grid complex array.

    3.00 is that floor -- two results plus one shared scratch.  Neither result
    can be written before both of its terms exist, and the inputs belong to
    the caller, so nothing can do better and the reading is two-sided: at
    least 3.00 and less than 4.00.

    MEASURED, 9 repeats on each arm: 3.000448 .. 3.000457 on Windows py3.14 /
    numpy 2.4.4 and 3.000448 .. 3.000457 on WSL py3.12 / numpy 2.4.6, i.e. the
    exact count plus <= 7.7 kB of tracemalloc bookkeeping on both.  No arm may
    make the shipped path the more expensive of the two, which is asserted
    here as well and holds whether or not NumPy elides the pre-fix form's
    third array.

    The reading that IS build-dependent -- the pre-fix expression's fourth
    grid -- is premise-gated in
    :func:`test_b8_the_pre_fix_jones_expression_holds_a_fourth_grid`.
    """
    _grid, _J, _Ex, _Ey, old, new = _b8_jones_peaks()
    assert 3.0 <= new < 3.0 + _B8_PEAK_SLACK, (old, new)
    assert new <= old + _B8_PEAK_SLACK, (old, new)


def test_b8_the_pre_fix_jones_expression_holds_a_fourth_grid():
    """PREMISE-GATED (TESTING_STANDARDS S3).  The pathology the shipped
    ``apply_jones_matrix`` removed -- the third full-grid array that
    ``J[1,0]*Ex + J[1,1]*Ey`` costs while the first component's result is
    still live -- is observable only on a build where NumPy does not elide
    that sum into one of its own operands.

    The premise is measured on the running arm by
    :func:`_numpy_elides_the_sum_of_two_temporaries`, never assumed from the
    platform.  Where it holds, the pre-fix expression peaks at 4.00 grids
    against the shipped path's 3.00 (Windows py3.14 / numpy 2.4.4: 4.000027 vs
    3.000460; and the py3.12 and py3.13 Linux runners of CI run 34914295323,
    which read 4.00 and passed).  Where NumPy elides, the pre-fix expression
    is rewritten into the same 3.00 and there is no fourth grid to see: WSL
    py3.12 / numpy 2.4.6 reads 3.000032 and so did the py3.14 runner of that CI
    run, which is what red this gate's previous ``old > 3.5`` form.  That arm
    asserts the collapse explicitly and then skips WITH the reading, so it can
    never pass silently.
    """
    grid, J, Ex, Ey, old, new = _b8_jones_peaks()
    free, held = _numpy_elides_the_sum_of_two_temporaries(J, Ex, Ey, grid)
    if free < held - 0.5:
        assert 3.0 <= old < 3.0 + _B8_PEAK_SLACK, (
            f"NumPy elided the pre-fix expression's third array, so it must "
            f"land on the same 3.00-grid floor as the shipped path, but it "
            f"read {old:.6f} (new={new:.6f})")
        pytest.skip(
            f"premise absent on this arm: NumPy's temporary elision is ACTIVE "
            f"for a*X + b*Y (it peaks at {free:.3f} full grids free, "
            f"{held:.3f} with both products name-bound), so the pre-fix "
            f"expression allocates no fourth grid -- measured old={old:.6f} "
            f"against new={new:.6f}, both on the 3.00 floor")
    assert 4.0 <= old < 4.0 + _B8_PEAK_SLACK, (
        f"elision is inactive here (free={free:.3f}, held={held:.3f}), so the "
        f"pre-fix expression must hold its fourth grid: old={old:.6f}, "
        f"new={new:.6f}")
    assert old - new > 0.95, (old, new)


# ===========================================================================
# VERIFY-WP-B8 -- pins added by the adversarial re-verification pass
# ===========================================================================


def test_verifyb8_apply_jones_matrix_falls_back_when_products_disagree():
    """The dtype gate, on inputs that actually engage it.

    ``Ex_new = j00*Ex`` and ``scratch = j01*Ey`` only land in different
    dtypes when the matrix is narrower than one of the components: a
    ``complex64`` SPATIALLY-VARYING matrix (the callable path is the only
    one that can carry a dtype other than ``complex128``) against ``Ex``
    ``complex64`` and ``Ey`` ``complex128`` gives ``complex64`` and
    ``complex128``.  ``Ex_new += scratch`` would then compute in
    ``complex128`` and NARROW back to ``complex64`` -- a different answer
    AND a different dtype from ``Ex_new + scratch``.

    Fail-before is measured, not quoted: the narrowed form is evaluated
    here and asserted to differ from the pre-fix expression, so the pass
    arm cannot be satisfied by a build where the two happen to agree.

    Found by VERIFY-WP-B8: dropping the ``Ex_new.dtype ==
    scratch.dtype`` guard left all 256 WP-B8 tests green.
    """
    N = 12
    rng = np.random.default_rng(20260913)
    base = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    Ex = base.astype(np.complex64)
    Ey = base * (0.37 - 1.9j)

    def cb(X, Y):
        s = (np.cos(X / 1e-6) + 1j * np.sin(Y / 1e-6)).astype(np.complex64)
        return np.array([[s, 0.5 * s], [-0.25 * s, s * s]],
                        dtype=np.complex64)

    # dy defaults to dx, so the callable sees this same square grid
    x = (np.arange(N) - N / 2) * 1e-6
    X, Y = np.meshgrid(x, x)
    J = cb(X, Y)
    want_x, want_y = _pre_fix_jones(J, Ex, Ey)

    # the state this gate exists for: the two products really do differ
    assert (J[0, 0] * Ex).dtype != (J[0, 1] * Ey).dtype

    # fail-before: the narrowed accumulation is a different answer
    narrowed = (J[0, 0] * Ex).copy()
    narrowed += J[0, 1] * Ey
    assert narrowed.dtype != want_x.dtype
    assert not _same_bits(narrowed.astype(want_x.dtype), want_x)

    out = apply_jones_matrix(JonesField(Ex.copy(), Ey.copy(), 1e-6), cb)
    assert out.Ex.dtype == want_x.dtype and out.Ey.dtype == want_y.dtype
    assert _same_bits(out.Ex, want_x) and _same_bits(out.Ey, want_y)


@pytest.mark.parametrize('shape', [(16, 16), (16, 14), (17, 15), (20, 21)])
def test_verifyb8_centred_fft2_keeps_the_input_memory_order(shape):
    """``_centred_fft2`` must return what ``fftshift(fft2(ifftshift(a)))``
    returns, LAYOUT included.

    ``ifftshift`` goes through ``np.roll``, whose ``empty_like`` carries
    the input's order, and the FFT carries it through; a plain ``copy()``
    inside ``_centred_fft2`` is C-ordered, so a Fortran-ordered PSF came
    back C-ordered -- same values, different buffer, a contract change
    nobody asked for.  Asserted on the FLAGS, since the values are
    already pinned above.

    Found by VERIFY-WP-B8.
    """
    rng = np.random.default_rng(sum(shape))
    a = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    for arr in (np.ascontiguousarray(a), np.asfortranarray(a),
                np.pad(a, 2)[2:-2, 2:-2]):
        want = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr)))
        got = _psf._centred_fft2(arr, np)
        assert _same_bits(want, got)
        assert got.flags['C_CONTIGUOUS'] == want.flags['C_CONTIGUOUS']
        assert got.flags['F_CONTIGUOUS'] == want.flags['F_CONTIGUOUS']


def test_verifyb8_compute_otf_keeps_a_fortran_psf_fortran():
    """The public consequence of the test above."""
    rng = np.random.default_rng(5)
    psf = np.asfortranarray(rng.random((32, 32)))
    otf = compute_otf(psf)
    assert otf.flags['F_CONTIGUOUS'] and not otf.flags['C_CONTIGUOUS']
    ref = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
    assert _same_bits(otf, ref / ref[16, 16])


@pytest.mark.parametrize('bad,why', [
    ((np.arange(9.0), np.linspace(1.0, 0.0, 9), 9.0), 'p_cum decreasing'),
    ((np.arange(9.0), np.linspace(0.0, 5.0, 9), 9.0), 'p_cum past 1'),
    ((np.linspace(9.0, 0.0, 9), np.linspace(0.0, 1.0, 9), 9.0),
     'radii descending'),
    ((np.arange(9.0), np.full(9, np.nan), 9.0), 'p_cum NaN'),
])
def test_verifyb8_a_structurally_invalid_profile_is_refused(bad, why):
    """``profile=`` carried no endpoint check, so a profile that is not a
    cumulative-energy profile at all was accepted and silently answered a
    different question -- while both docstrings claimed the endpoints
    were validated.  O(1), so it costs nothing the argument was bought
    for.

    Found by VERIFY-WP-B8.
    """
    E = np.exp(-np.add.outer((np.arange(3) - 1.0) ** 2,
                             (np.arange(3) - 1.0) ** 2)) + 0j
    for fn in (encircled_energy_curve, encircled_energy_radius):
        with pytest.raises(ValueError, match='endpoints'):
            fn(E, 1e-6, profile=bad)


def test_verifyb8_a_profile_from_a_differently_sized_field_is_refused():
    """The one wrong-field case that IS catchable in O(1).  A profile
    from a same-sized field still cannot be, and the docstring now says
    so.

    Found by VERIFY-WP-B8.
    """
    rng = np.random.default_rng(3)
    E_small = rng.random((8, 8)) + 0j
    E_big = rng.random((16, 16)) + 0j
    prof = encircled_energy_profile(E_small, 1e-6)
    for fn in (encircled_energy_curve, encircled_energy_radius):
        with pytest.raises(ValueError, match='different field'):
            fn(E_big, 1e-6, profile=prof)
    # and the right-sized one still goes through, bit for bit
    good = encircled_energy_profile(E_big, 1e-6)
    r0, ee0 = encircled_energy_curve(E_big, 1e-6)
    r1, ee1 = encircled_energy_curve(E_big, 1e-6, profile=good)
    assert _same_bits(r0, r1) and _same_bits(ee0, ee1)


def test_verifyb8_profile_endpoint_slack_admits_every_real_profile():
    """The bar has a gap on both sides.  The ``len(p_cum) * eps`` slack
    sits above every real ``cumsum`` drift with room to spare -- measured
    over N = 16...2048 x {Gaussian, noise, Airy, near-delta}
    (2026-09-13), the drift is 23x under the slack at N = 16, where the
    reduction is shortest and the ratio worst, and 105x under it at
    N = 2048 (8.8e-12 against n eps = 9.3e-10) -- and ten decades below
    the O(1) violations the guard rejects.  One decade is asserted."""
    rng = np.random.default_rng(17)
    for N in (16, 64, 256):
        x = (np.arange(N) - N / 2) * 1e-6
        X, Y = np.meshgrid(x, x)
        for E in (np.exp(-(X ** 2 + Y ** 2) / (N * 1e-7) ** 2) + 0j,
                  rng.standard_normal((N, N))
                  + 1j * rng.standard_normal((N, N)),
                  (rng.random((N, N)) ** 12).astype(complex)):
            prof = encircled_energy_profile(E, 1e-6)
            tol = prof[1].size * float(np.finfo(np.float64).eps)
            assert abs(float(prof[1][-1]) - 1.0) < tol / 10.0
            encircled_energy_curve(E, 1e-6, profile=prof)
            encircled_energy_radius(E, 1e-6, profile=prof)


@pytest.mark.parametrize('sigma_g', [0.0, -4e-6, np.inf, np.nan])
def test_verifyb8_modes_refuses_a_degenerate_coherence_length(sigma_g):
    """``generator='modes'`` draws ``k ~ N(0, 1/sigma_g)``, so a zero /
    negative / non-finite coherence length is undefined.  Before the fix
    ``0.0`` raised ``ZeroDivisionError`` from inside ``_gori_mode_count``
    (whose own ``cells <= 0`` guard could never run, because a Python
    float divide by zero raises first), ``nan`` returned an all-NaN
    field, and ``inf`` / a negative value returned a field for
    ``|sigma_g|`` without a word.

    Found by VERIFY-WP-B8.  ``generator='fft'`` is a default and is NOT
    touched -- this asserts only that ``'modes'`` no longer answers
    silently.
    """
    with pytest.raises(ValueError, match='coherence_length'):
        _schell_phase_realizations(
            Ny=8, Nx=8, dx=1e-6, dy=1e-6, coherence_length=sigma_g,
            n_realizations=1, rng=np.random.default_rng(0),
            generator='modes')
    # the helper itself is total now, whatever it is handed
    assert _gori_mode_count(8e-6, 8e-6, sigma_g) == _GORI_MIN_MODES


def test_verifyb8_float32_geometry_survives_a_numpy_scalar_centre():
    """NEP 50 makes a NumPy scalar STRONG, so ``float32_array -
    np.float64(x0)`` came back float64 and ``geometry_dtype=np.float32``
    silently did nothing -- a different field AND the full
    double-precision transient.  Both arms are asserted: same bits either
    way, and the peak stays under 1.75 full grids (1.50 after, 2.00
    before).

    Found by VERIFY-WP-B8.
    """
    N = 512
    out = N * N * 8                       # one complex64 full grid
    kw = dict(dx=1e-6, wavelength=WL, w0=40e-6, dtype=np.complex64,
              geometry_dtype=np.float32)
    E_py, _, _ = create_gaussian_beam(N, x0=3e-6, y0=-2e-6, **kw)
    E_np, _, _ = create_gaussian_beam(N, x0=np.float64(3e-6),
                                      y0=np.float64(-2e-6), **kw)
    assert _same_bits(E_py, E_np)
    peak = _peak_grids(
        lambda: create_gaussian_beam(N, x0=np.float64(3e-6),
                                     y0=np.float64(-2e-6), **kw), out)
    assert peak < 1.75, peak


def test_verifyb8_the_fft_path_refuses_n_psf_below_the_pupil_size():
    """``N_psf < N_pupil`` is the one place the two samplers part company.
    The FFT sampler cannot crop, so it used to return an
    ``N_pupil x N_pupil`` array while reporting ``wavelength*f/(N_psf*dx_pupil)``
    as its pitch (measured on the pre-fix library: shape (32, 32) for
    ``N_psf=16``, the same reported pitch as the MFT's (16, 16)).  It now
    refuses with a message that names the remedy; the MFT path honours
    ``N_psf``.

    Found by VERIFY-WP-B8; the refusal is the orchestrator's ruling.
    """
    N, dx = 32, 5e-6
    g = np.arange(N) - N / 2
    pupil = (np.hypot(*np.meshgrid(g, g)) <= 12).astype(complex)
    with pytest.raises(ValueError,
                       match=r"compute_psf: N_psf=16 is smaller than the pupil"):
        compute_psf(pupil, WL, FOCAL, dx, N_psf=16)
    b, dxb = compute_psf(pupil, WL, FOCAL, dx, N_psf=16, method='mft')
    assert b.shape == (16, 16)          # the MFT path honours N_psf
    assert dxb == pytest.approx(WL * FOCAL / (16 * dx))
    # at or above the pupil size they agree, which is the contract
    c, _ = compute_psf(pupil, WL, FOCAL, dx, N_psf=N)
    d, _ = compute_psf(pupil, WL, FOCAL, dx, N_psf=N, method='mft')
    assert c.shape == d.shape == (N, N)
    assert np.abs(c - d).max() / c.max() < 1e-13
