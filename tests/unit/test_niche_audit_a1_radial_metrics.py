"""Audit AUDIT_ADVERSARIAL_CODEBASE_2026_07_25, Territory A finding A-1
(CRITICAL-physics): the default ``axis='radial'`` path of
``fwhm_resolution`` and ``rayleigh_resolution`` was biased by
integer-pixel radial binning.

Mechanism
---------
``_psf_1d_profile(axis='radial')`` rounds every pixel's Euclidean
radius to the nearest multiple of ``d_bin = sqrt(dx*dy)`` and
azimuthally averages inside those integer shells.  Near the peak the
shells hold few, lopsidedly-placed pixels, so the shell mean sits
systematically BELOW ``I(k * d_bin)`` -- which drags any half-max or
first-zero crossing inward.  The same module already ships the
unbiased alternative, ``_radial_profile_subpixel`` (a polar resample
at ~4 samples per pixel, 64 azimuthal samples), and
``sparrow_resolution(axis='radial')`` had used it since v4.15.1.

Measured on an analytic Airy PSF (lambda = 600 nm, f/# = 4; first zero
``j1_zero/pi * lam * f/# = 2.92721 um``, FWHM ``1.029 lam f/# =
2.46960 um``) BEFORE the fix / AFTER the fix:

=====================  ===============  ==============
samples / first zero   FWHM v5.29       FWHM v5.30
=====================  ===============  ==============
19.5                   -0.12%           -0.00%
9.8                    -1.92%           +0.01%
4.9                    **-8.04%**       +0.01%
2.4                    **-21.08%**      -1.03%
=====================  ===============  ==============

=====================  ===================  ==============
samples / first zero   Rayleigh v5.29       Rayleigh v5.30
=====================  ===================  ==============
19.5                   -0.12%               +0.02%
9.8                    +1.54%               +0.10%
4.9                    +6.76%               +0.18%
2.4                    **NaN + warning**    +3.26%
=====================  ===================  ==============

The v5.29 NaN at 2.4 samples per first zero came with a
``RuntimeWarning`` that blamed the *PSF shape* ("the criterion is not
defined for Gaussian-like PSFs without a true first zero") on a
**perfect Airy pattern** -- a false accusation that sent the user
looking for a physics bug instead of increasing the sampling.  v5.30
splits that diagnostic in two: a monotonically-decaying profile still
gets the Gaussian-like message; a profile that retains ring structure
gets a message naming COARSE SAMPLING as the likely cause.

Fix: ``axis='radial'`` routes through ``_radial_metric_profile``,
which prefers ``_radial_profile_subpixel`` and falls back to the
historical integer-bin profile only when the array is too small for a
4-pixel polar radius (or SciPy is missing), preserving the tiny-grid
and degenerate-input return contracts.  ``axis='x'`` / ``axis='y'``
are untouched.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.special import j1

from lumenairy.analysis.psf_mtf_otf import (
    _psf_1d_profile,
    _radial_metric_profile,
    _radial_profile_subpixel,
    compute_psf,
    fwhm_resolution,
    rayleigh_resolution,
    sparrow_resolution,
)

# --------------------------------------------------------------------------
# Analytic Airy oracle.  I(r) = [2 J1(v)/v]^2 with v = pi r / (lam f/#);
# first zero at v = j1_zero -> r = (j1_zero/pi) lam f/#.
# --------------------------------------------------------------------------
_LAM = 600e-9
_FN = 4.0
_J1_ZERO = 3.8317059702075123
_R_ZERO = _J1_ZERO / np.pi * _LAM * _FN          # 2.92721 um
_FWHM_TRUE = 1.029 * _LAM * _FN                  # 2.46960 um
_SPARROW_TRUE = 0.947 * _LAM * _FN


def _airy_analytic(N: int, samples_per_first_zero: float):
    """Analytic Airy intensity, peak on the exact grid centre.

    ``samples_per_first_zero`` fixes ``dx = r_first_zero / spz``, which
    is the single variable finding A-1 is about.
    """
    dx = _R_ZERO / float(samples_per_first_zero)
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    v = np.pi * np.sqrt(X ** 2 + Y ** 2) / (_LAM * _FN)
    out = np.ones_like(v)
    nz = v > 0
    out[nz] = (2.0 * j1(v[nz]) / v[nz]) ** 2
    return out, dx


def _airy_fft(pad: int):
    """Independent oracle: ``compute_psf`` of a circular pupil.

    128 pixels across a 25 mm f/4 pupil zero-padded ``pad`` times gives
    ``dx_psf = lam (f/#) / pad``, i.e. ``1.22 * pad`` samples per first
    zero.
    """
    D, f = 0.025, 0.100
    Ng = 128 * pad
    dx_pupil = D / 128
    x = (np.arange(Ng) - Ng / 2) * dx_pupil
    X, Y = np.meshgrid(x, x)
    pupil = (np.sqrt(X ** 2 + Y ** 2) <= D / 2).astype(complex)
    return compute_psf(pupil, _LAM, f, dx_pupil)


def _gaussian_psf(N: int, dx: float, w0: float) -> np.ndarray:
    """Intensity of a Gaussian beam (no first ring -> no Rayleigh)."""
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-2.0 * (X ** 2 + Y ** 2) / w0 ** 2)


# ==========================================================================
# A-1 pin 1 -- fwhm_resolution(axis='radial') vs the analytic Airy FWHM
# ==========================================================================

# (samples per first zero, tolerance, pre-fix measured relative error).
# The tolerances carry modest headroom over the measured post-fix error
# (1.3e-4 at 4.88, 1.03e-2 at 2.44) and every one of them is violated by
# the pre-fix value in the third column.
_FWHM_CASES = [
    pytest.param(4.88, 0.005, 0.0804, id='4.9-samples-per-first-zero'),
    pytest.param(2.44, 0.020, 0.2108, id='2.4-samples-per-first-zero'),
]


@pytest.mark.parametrize('spz,tol,prefix_err', _FWHM_CASES)
def test_a1_fwhm_radial_matches_analytic_airy(spz, tol, prefix_err):
    """``fwhm_resolution(axis='radial')`` recovers ``1.029 lam f/#``.

    Pre-fix this read -8.04% (spz=4.88) and -21.08% (spz=2.44); both
    are far outside the tolerances asserted here, so this test is a
    genuine regression pin and not a restatement of the old behaviour
    (the ``prefix_err`` guard below makes that explicit).
    """
    psf, dx = _airy_analytic(512, spz)
    d_fwhm = fwhm_resolution(psf, dx, axis='radial')
    rel = abs(d_fwhm - _FWHM_TRUE) / _FWHM_TRUE
    assert rel < tol, (
        f"fwhm_resolution(axis='radial') on an analytic Airy at {spz} "
        f"samples per first zero: got {d_fwhm * 1e6:.5f} um, analytic "
        f"1.029*lam*f/# = {_FWHM_TRUE * 1e6:.5f} um (rel {rel:.4%} > "
        f"{tol:.2%}).  Regression to integer-pixel radial binning?")
    # Counter-check that the tolerance really does exclude the biased
    # pre-fix answer -- a pin that both behaviours satisfy is no pin.
    assert prefix_err > tol, (
        f"tolerance {tol} does not exclude the pre-fix error "
        f"{prefix_err}; tighten the pin.")


@pytest.mark.parametrize('pad,tol', [(4, 0.005), (2, 0.020)])
def test_a1_fwhm_radial_matches_on_fft_airy_oracle(pad, tol):
    """Same pin on the independent ``compute_psf`` Airy oracle
    (pad=4 -> 4.88 samples per first zero, pad=2 -> 2.44).  Pre-fix:
    -7.98% and -21.02%."""
    psf, dx = _airy_fft(pad)
    d_fwhm = fwhm_resolution(psf, dx, axis='radial')
    rel = abs(d_fwhm - _FWHM_TRUE) / _FWHM_TRUE
    assert rel < tol, (
        f"fwhm_resolution(axis='radial') on the FFT Airy at pad={pad}: "
        f"got {d_fwhm * 1e6:.5f} um vs {_FWHM_TRUE * 1e6:.5f} um "
        f"(rel {rel:.4%} > {tol:.2%}).")


def test_a1_binned_profile_is_measurably_worse_than_the_subpixel_one():
    """Mechanism pin: the two 'radial' profiles the module ships
    disagree at the half-max crossing, and the integer-binned one is
    the biased member of the pair.  If a future refactor points the
    metrics back at the binned profile, the FWHM pins above break --
    this test says *why*."""
    psf, dx = _airy_analytic(512, 4.88)
    r_bin, p_bin = _psf_1d_profile(psf, dx, axis='radial')
    r_sub, p_sub = _radial_profile_subpixel(psf, dx, dx)

    def half_radius(r, p):
        p = p / p.max()
        for i in range(1, p.size):
            if p[i] <= 0.5:
                t = (0.5 - p[i - 1]) / (p[i] - p[i - 1])
                return r[i - 1] + t * (r[i] - r[i - 1])
        return np.nan

    h_bin = half_radius(r_bin, p_bin)
    h_sub = half_radius(r_sub, p_sub)
    h_true = 0.5 * _FWHM_TRUE
    assert abs(h_sub - h_true) / h_true < 0.005, (
        f"sub-pixel profile half-radius {h_sub * 1e6:.5f} um vs analytic "
        f"{h_true * 1e6:.5f} um.")
    assert (h_true - h_bin) / h_true > 0.03, (
        f"the integer-binned profile is expected to under-read the "
        f"half-radius by >3% at 4.88 samples per first zero; measured "
        f"{100 * (h_true - h_bin) / h_true:.2f}% "
        f"(bin {h_bin * 1e6:.5f} um, analytic {h_true * 1e6:.5f} um).")


# ==========================================================================
# A-1 pin 2 -- rayleigh_resolution(axis='radial') is finite, accurate,
#              and does NOT warn on a perfect Airy
# ==========================================================================

_RAYLEIGH_CASES = [
    # spz, tolerance.  Measured post-fix: +0.18% at 4.88, +3.26% at
    # 2.44 (2.44 samples per first zero is close to the sampling floor
    # for a resolvable ring, hence the looser band); pre-fix: +6.76%
    # and NaN.
    pytest.param(4.88, 0.010, id='4.9-samples-per-first-zero'),
    pytest.param(2.44, 0.050, id='2.4-samples-per-first-zero'),
]


@pytest.mark.parametrize('spz,tol', _RAYLEIGH_CASES)
def test_a1_rayleigh_radial_finite_accurate_and_silent(spz, tol):
    """A perfect Airy must yield a finite first zero with NO warning.

    Pre-fix, ``spz=2.44`` returned NaN and emitted a ``RuntimeWarning``
    blaming the PSF for being "Gaussian-like" -- on an analytic Airy.
    """
    psf, dx = _airy_analytic(512, spz)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        d_rayleigh = rayleigh_resolution(psf, dx, _LAM, axis='radial')
    runtime = [str(w.message) for w in caught
               if issubclass(w.category, RuntimeWarning)]
    assert not runtime, (
        f"rayleigh_resolution warned on an analytic Airy at {spz} "
        f"samples per first zero: {runtime}")
    assert np.isfinite(d_rayleigh), (
        f"rayleigh_resolution returned {d_rayleigh!r} on an analytic "
        f"Airy at {spz} samples per first zero; expected a finite "
        f"first-zero radius near {_R_ZERO * 1e6:.5f} um.")
    rel = abs(d_rayleigh - _R_ZERO) / _R_ZERO
    assert rel < tol, (
        f"rayleigh_resolution(axis='radial') at {spz} samples per first "
        f"zero: got {d_rayleigh * 1e6:.5f} um, analytic first zero "
        f"{_R_ZERO * 1e6:.5f} um (= 1.2197*lam*f/#, the 1.22*lam*f/D "
        f"Rayleigh separation); rel {rel:.4%} > {tol:.2%}.")


def test_a1_rayleigh_radial_silent_on_fft_airy_oracle():
    """Independent oracle: no warning and <4% on the ``compute_psf``
    Airy at both paddings (pre-fix: NaN + warning at pad=2)."""
    for pad, tol in ((4, 0.010), (2, 0.050)):
        psf, dx = _airy_fft(pad)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            val = rayleigh_resolution(psf, dx, _LAM, axis='radial')
        assert not [w for w in caught
                    if issubclass(w.category, RuntimeWarning)], (
            f"FFT Airy pad={pad} raised a RuntimeWarning.")
        assert np.isfinite(val) and abs(val - _R_ZERO) / _R_ZERO < tol, (
            f"FFT Airy pad={pad}: rayleigh = {val!r} vs "
            f"{_R_ZERO * 1e6:.5f} um.")


def test_a1_rayleigh_still_nan_and_warns_for_a_gaussian():
    """Counter-pin (v4.15.1 P1-F1-4 must not regress): a Gaussian PSF
    really has no first ring, so NaN + ``RuntimeWarning`` is correct --
    and the message must now blame the monotone DECAY, not invoke the
    coarse-sampling branch."""
    psf = _gaussian_psf(256, 0.1e-6, 0.5e-6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        val = rayleigh_resolution(psf, 0.1e-6, _LAM, axis='radial')
    assert np.isnan(val), f"Gaussian PSF should give NaN; got {val!r}."
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)]
    assert msgs, "Gaussian PSF must still emit a RuntimeWarning."
    joined = ' '.join(msgs).lower()
    assert 'decays monotonically' in joined, (
        f"expected the monotone-decay diagnosis for a Gaussian; got "
        f"{msgs}.")
    assert 'sampled too coarsely' not in joined, (
        f"a well-sampled Gaussian must NOT be diagnosed as an "
        f"under-sampling problem; got {msgs}.")
    # The v4.15.1 contract (message points at the alternatives) holds.
    assert 'fwhm_resolution' in joined and 'sparrow_resolution' in joined


def test_a1_coarse_sampling_diagnosis_is_reachable_and_named():
    """The other half of the split diagnostic: when the first ring is
    present but unresolvable, the warning must name SAMPLING rather
    than accuse the PSF of being Gaussian-like.

    Driven through the binned-fallback profile of a deliberately tiny
    Airy grid, which is the regime where the ring survives as structure
    but no qualifying below-5%-of-peak minimum exists.
    """
    # A ring-bearing profile whose minimum never drops below 5% of peak:
    # an Airy core plus a pedestal.  Constructed directly so the test
    # exercises the diagnostic, not a particular PSF generator.
    psf, dx = _airy_analytic(64, 2.0)
    psf = psf + 0.10 * psf.max()          # pedestal lifts the null > 5%
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        val = rayleigh_resolution(psf, dx, _LAM, axis='radial')
    msgs = [str(w.message).lower() for w in caught
            if issubclass(w.category, RuntimeWarning)]
    assert np.isnan(val)
    assert msgs, "expected a RuntimeWarning for the unresolvable ring."
    joined = ' '.join(msgs)
    assert 'sampled too coarsely' in joined, (
        f"a ring-bearing profile with no qualifying minimum must be "
        f"diagnosed as a sampling problem; got {msgs}.")
    assert 'fwhm_resolution' in joined


# ==========================================================================
# A-1 pin 3 -- cross-consistency: axis='radial' vs axis='x'
# ==========================================================================

@pytest.mark.parametrize('spz,tol', [(4.88, 0.010), (2.44, 0.030)])
def test_a1_fwhm_radial_agrees_with_axis_x_on_a_symmetric_psf(spz, tol):
    """On a rotationally symmetric PSF the radial average and the row
    cut measure the same profile, so the two FWHMs must agree.

    Pre-fix they disagreed by 8.3% (spz=4.88) and 22.1% (spz=2.44) --
    the binning bias showed up precisely as this inconsistency.
    Measured post-fix: 0.35% and 2.34%.
    """
    psf, dx = _airy_analytic(512, spz)
    d_rad = fwhm_resolution(psf, dx, axis='radial')
    d_x = fwhm_resolution(psf, dx, axis='x')
    d_y = fwhm_resolution(psf, dx, axis='y')
    assert abs(d_rad / d_x - 1.0) < tol, (
        f"fwhm_resolution radial={d_rad * 1e6:.5f} um vs "
        f"x={d_x * 1e6:.5f} um at {spz} samples per first zero: "
        f"{abs(d_rad / d_x - 1.0):.4%} > {tol:.2%} apart on a "
        f"rotationally symmetric PSF.")
    assert d_x == pytest.approx(d_y, rel=1e-12), (
        f"x and y cuts of a symmetric PSF must match: {d_x!r} vs {d_y!r}")


def test_a1_rayleigh_radial_agrees_with_axis_x_where_x_is_defined():
    """Same cross-consistency for the Rayleigh first zero at 4.88
    samples per first zero.  The band is wider than the FWHM one
    because the row cut resolves the first zero with only ~5 samples
    and its own parabolic refinement reads +6.4% there -- the radial
    average (which pools 64 azimuthal samples per radius) is the more
    accurate of the two, so this pin is a sanity band, not an accuracy
    claim about the cut."""
    psf, dx = _airy_analytic(512, 4.88)
    d_rad = rayleigh_resolution(psf, dx, _LAM, axis='radial')
    d_x = rayleigh_resolution(psf, dx, _LAM, axis='x')
    assert np.isfinite(d_rad) and np.isfinite(d_x)
    assert abs(d_rad / d_x - 1.0) < 0.10, (
        f"rayleigh radial={d_rad * 1e6:.5f} um vs x={d_x * 1e6:.5f} um "
        f"differ by {abs(d_rad / d_x - 1.0):.2%}.")
    # The radial reading is the one that must be close to analytic.
    assert abs(d_rad - _R_ZERO) / _R_ZERO < 0.01


def test_a1_metric_ordering_sparrow_lt_fwhm_lt_rayleigh():
    """Physical ordering on an Airy: Sparrow (0.947) < FWHM (1.029) <
    Rayleigh (1.220), all in units of ``lam f/#``.  Pre-fix the radial
    FWHM had slipped BELOW the Sparrow value at 2.44 samples per first
    zero (1.9491 um vs 2.1372 um), inverting the ordering."""
    for spz in (9.76, 4.88, 2.44):
        psf, dx = _airy_analytic(512, spz)
        d_sp = sparrow_resolution(psf, dx, axis='radial')
        d_fw = fwhm_resolution(psf, dx, axis='radial')
        d_ra = rayleigh_resolution(psf, dx, _LAM, axis='radial')
        assert d_sp < d_fw < d_ra, (
            f"ordering broken at {spz} samples per first zero: "
            f"sparrow={d_sp * 1e6:.4f} fwhm={d_fw * 1e6:.4f} "
            f"rayleigh={d_ra * 1e6:.4f} um.")


# ==========================================================================
# Contract preservation -- signatures, dy threading, degenerate inputs
# ==========================================================================

def test_a1_dy_none_is_bit_identical_to_dy_equal_dx():
    """v4.15.1 C.5 contract: on a square grid ``dy=None`` and
    ``dy=dx`` must agree EXACTLY (the sub-pixel helper branches the
    square case out for precisely this reason)."""
    psf, dx = _airy_analytic(256, 4.88)
    assert (fwhm_resolution(psf, dx, axis='radial') ==
            fwhm_resolution(psf, dx, axis='radial', dy=dx))
    assert (rayleigh_resolution(psf, dx, _LAM, axis='radial') ==
            rayleigh_resolution(psf, dx, _LAM, axis='radial', dy=dx))


def test_a1_dy_is_actually_threaded_on_the_radial_path():
    """An anamorphic grid must change the radial metric -- otherwise
    ``dy`` is silently inert (the failure mode S9-AN1 caught in
    ``_radial_profile_subpixel`` itself)."""
    psf, dx = _airy_analytic(256, 4.88)
    iso = fwhm_resolution(psf, dx, axis='radial')
    ana = fwhm_resolution(psf, dx, axis='radial', dy=2 * dx)
    assert ana != iso, (
        f"dy=2*dx returned the isotropic value {iso!r}; dy threading "
        f"appears broken on the radial path.")


def test_a1_degenerate_and_tiny_inputs_keep_their_contracts():
    """The sub-pixel helper returns an empty profile for arrays too
    small for a 4-pixel polar radius; the fallback must keep the
    historical NaN / small-grid answers."""
    flat = np.zeros((16, 16))
    assert np.isnan(fwhm_resolution(flat, 1e-6, axis='radial'))
    assert np.isnan(rayleigh_resolution(flat, 1e-6, _LAM, axis='radial'))
    # 8x8 delta: too small for the polar resample -> binned fallback,
    # which reads a one-pixel FWHM exactly as it did pre-fix.
    tiny = np.zeros((8, 8))
    tiny[3, 3] = 1.0
    r, prof, subpixel = _radial_metric_profile(tiny, 1e-6, 1e-6)
    assert not subpixel, "8x8 grid should take the binned fallback."
    assert fwhm_resolution(tiny, 1e-6, axis='radial') == pytest.approx(1e-6)
    assert prof.size > 0 and r.size == prof.size


def test_a1_non_2d_input_still_raises_valueerror_naming_2d():
    """The radial path no longer goes through ``_psf_1d_profile``, so
    the ndim guard has to be restated -- v4.15.0 C.3 pins the
    ``ValueError`` mentioning '2-D'."""
    for bad in (np.ones(64), np.ones((8, 8, 8))):
        with pytest.raises(ValueError, match='2-D'):
            fwhm_resolution(bad, 1e-6, axis='radial')
        with pytest.raises(ValueError, match='2-D'):
            rayleigh_resolution(bad, 1e-6, _LAM, axis='radial')


def test_a1_axis_x_and_y_paths_are_unchanged():
    """``axis='x'`` / ``axis='y'`` must still measure the raw
    pixel-aligned cut from ``_psf_1d_profile``: recomputed here from
    the documented algorithm and required to match bit-for-bit, so the
    A-1 fix provably did not leak into the cut paths."""
    psf, dx = _airy_analytic(256, 4.88)
    for axis in ('x', 'y'):
        r, profile = _psf_1d_profile(psf, dx, axis=axis)
        peak_idx = int(np.argmax(profile))
        half = 0.5 * float(profile.max())
        j = next(i for i in range(peak_idx + 1, profile.size)
                 if profile[i] <= half)
        y_lo, y_hi = profile[j - 1], profile[j]
        t = (half - y_lo) / (y_hi - y_lo)
        r_step = float(r[1] - r[0])
        expect = 2.0 * (abs(r[j - 1] - r[peak_idx]) + t * r_step)
        assert fwhm_resolution(psf, dx, axis=axis) == expect, (
            f"axis={axis!r} FWHM drifted from the raw cut computation.")


def test_a1_invalid_axis_still_raises():
    """Signature contract: an unknown axis is still rejected."""
    psf, dx = _airy_analytic(64, 4.88)
    with pytest.raises(ValueError, match='axis'):
        fwhm_resolution(psf, dx, axis='diagonal')
    with pytest.raises(ValueError, match='axis'):
        rayleigh_resolution(psf, dx, _LAM, axis='diagonal')


def test_a1_docstrings_record_the_measured_radial_accuracy():
    """Doc pin: both functions must document that the default radial
    path uses the sub-pixel profile, so the next reader does not
    reintroduce the binning as an 'optimisation'."""
    for fn in (fwhm_resolution, rayleigh_resolution):
        doc = (fn.__doc__ or '')
        assert '_radial_profile_subpixel' in doc, (
            f"{fn.__name__} docstring does not name the sub-pixel "
            f"radial profile it measures.")
        assert 'samples / first zero' in doc, (
            f"{fn.__name__} docstring is missing the measured "
            f"accuracy-vs-sampling table.")


def test_a1_sparrow_is_untouched_by_the_fix():
    """``sparrow_resolution`` already used the sub-pixel profile; its
    analytic accuracy pin (<1% on a well-sampled Airy) must be
    unaffected."""
    psf, dx = _airy_analytic(512, 9.76)
    d_sp = sparrow_resolution(psf, dx, axis='radial')
    assert abs(d_sp - _SPARROW_TRUE) / _SPARROW_TRUE < 0.01, (
        f"sparrow_resolution = {d_sp * 1e6:.5f} um vs analytic "
        f"{_SPARROW_TRUE * 1e6:.5f} um.")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
