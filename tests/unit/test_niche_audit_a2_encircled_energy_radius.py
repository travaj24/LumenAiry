"""Audit AUDIT_ADVERSARIAL_CODEBASE_2026_07_25, Territory A finding A-2
(HIGH-bug): ``encircled_energy_radius`` drifted with zero-padding.

Mechanism
---------
Through v5.29 the function hunted the threshold crossing on a
hard-coded 256-point radius ladder that ran from 0 to the *grid
corner*::

    radii, ee = encircled_energy_curve(..., n_radii=256)   # 0 .. corner

The corner grows with the array size while the beam does not, so the
ladder step -- and with it the interpolation error at the crossing --
scaled with the padding.  Measured on a FIXED physical Gaussian
(``dx = 0.5 um``, ``w = 5 um`` = 10 px) over ``N = 64 -> 2048``:

======  ==============  ===============  ===============
N       corner / um     ladder step/um   r(86.5%) error
======  ==============  ===============  ===============
64      22.63           0.0887           +0.20%
256     90.51           0.3549           +0.77%
1024    362.04          1.4198           +3.14%
2048    724.08          2.8395           **+6.05%**
======  ==============  ===============  ===============

while inverting the underlying cumulative curve on a beam-scaled grid
read -0.80% at *every* N -- i.e. the curve was fine and the wrapper's
sampling grid was the whole defect.  The comment above the call even
claimed "256 samples ... gives sub-percent accuracy".

Fix: invert the exact cumulative-energy curve.  ``_ee_sorted_cumulative``
(now shared with :func:`encircled_energy_curve`, which samples the very
same construction) sorts the pixel radii and accumulates the powers;
the radius is located with one ``searchsorted`` plus one linear
interpolation inside the straddling pixel-radius interval.  No radius
ladder exists, so the answer cannot depend on the array size:
post-fix the whole ``N = 64 -> 2048`` sweep returns **bit-identical**
radii.

What is intentionally NOT claimed
---------------------------------
Two intrinsic, array-size-independent limits remain and are documented
in the function's Notes rather than papered over:

* a half-pixel quadrature bias (``p_cum[k]`` books the power including
  pixel ``k`` at that pixel's own radius) worth a few tenths of a
  percent at 10-40 px per waist, shrinking with ``dx``, not ``N``;
* ``dEE/dr = 2 pi r I(r)`` vanishes at a dark ring, so inverting a
  threshold that sits on an Airy null is intrinsically unstable -- the
  *fraction* is recovered to <1e-4 there while the *radius* can move a
  few percent.  The Airy checks below therefore pin the forward
  direction plus round-trip consistency, which is what is actually
  well-posed.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j1

from lumenairy.analysis.psf_mtf_otf import (
    compute_psf,
    encircled_energy_curve,
    encircled_energy_radius,
)

_EE_1E2 = 1.0 - np.exp(-2.0)                 # 0.8646647 -- EE at r = w
_LAM = 600e-9
_FN = 4.0
_J1_ZERO = 3.8317059702075123
_R_ZERO = _J1_ZERO / np.pi * _LAM * _FN      # Airy first dark ring


# --------------------------------------------------------------------- fixtures
def _gaussian_field(N: int, dx: float, w0: float) -> np.ndarray:
    """``E = exp(-r^2/w^2)`` so ``I = exp(-2r^2/w^2)`` and
    ``EE(r) = 1 - exp(-2 r^2 / w^2)`` analytically."""
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(complex)


def _r_gauss(w0: float, ee: float) -> float:
    """Analytic inverse of the Gaussian encircled-energy curve."""
    return w0 * np.sqrt(-np.log(1.0 - ee) / 2.0)


def _airy_field_fft(pad: int):
    """Airy amplitude from ``compute_psf`` (Parseval-exact total power,
    so the EE normalisation is not distorted by tail truncation)."""
    D, f = 0.025, 0.100
    Ng = 128 * pad
    dx_pupil = D / 128
    x = (np.arange(Ng) - Ng / 2) * dx_pupil
    X, Y = np.meshgrid(x, x)
    pupil = (np.sqrt(X ** 2 + Y ** 2) <= D / 2).astype(complex)
    psf, dx_psf = compute_psf(pupil, _LAM, f, dx_pupil)
    return np.sqrt(np.maximum(psf, 0.0)).astype(complex), dx_psf


def _airy_field_analytic(N: int, samples_per_first_zero: float):
    dx = _R_ZERO / float(samples_per_first_zero)
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    v = np.pi * np.sqrt(X ** 2 + Y ** 2) / (_LAM * _FN)
    out = np.ones_like(v)
    nz = v > 0
    out[nz] = (2.0 * j1(v[nz]) / v[nz]) ** 2
    return np.sqrt(out).astype(complex), dx


def _embed(field: np.ndarray, N: int) -> np.ndarray:
    """Zero-pad ``field`` into an ``N x N`` grid, centre pixel aligned.

    Nothing physical changes: same ``dx``, same beam, same total power.
    Only the array size (and hence the old corner-spanning ladder) moves.
    """
    n = field.shape[0]
    assert N >= n and (N - n) % 2 == 0
    out = np.zeros((N, N), dtype=field.dtype)
    k = N // 2 - n // 2
    out[k:k + n, k:k + n] = field
    return out


# ==========================================================================
# A-2 pin 1 -- zero-padding invariance
# ==========================================================================

# The measured post-fix drift over the whole sweep is exactly 0 (the
# padding adds only exact zeros, which change neither the sorted
# cumulative sum nor the centroid).  1e-6 leaves four orders of headroom
# under the requested ~0.5% band while still excluding the +6.05% bug by
# four orders in the other direction.
_PAD_TOL = 1e-6


@pytest.mark.parametrize('threshold', [0.50, 0.80, _EE_1E2])
def test_a2_gaussian_ee_radius_is_invariant_under_zero_padding(threshold):
    """Same physical Gaussian embedded at N = 64 / 256 / 1024 must give
    the same EE radius.  Pre-fix the 86.5% radius drifted +6.05% across
    a comparable sweep purely from the array size."""
    dx, w0 = 1e-6, 20.3e-6            # w/dx deliberately non-integer
    base = _gaussian_field(128, dx, w0)
    radii = {}
    for N in (128, 256, 512, 1024):
        radii[N] = encircled_energy_radius(
            _embed(base, N), dx, threshold=threshold)
    ref = radii[128]
    for N, r in radii.items():
        assert abs(r / ref - 1.0) < _PAD_TOL, (
            f"encircled_energy_radius(threshold={threshold}) drifted "
            f"with zero-padding: N={N} -> {r * 1e6:.6f} um vs N=128 -> "
            f"{ref * 1e6:.6f} um ({abs(r / ref - 1.0):.3e} relative). "
            f"The radius must not depend on the array size.")
    # Also assert against the smaller 64-px base grid (the audit's own
    # N=64 -> 2048 framing) so the sweep spans a 32x array-area range.
    base64 = _gaussian_field(64, dx, 10e-6)
    ref64 = encircled_energy_radius(base64, dx, threshold=threshold)
    for N in (128, 512, 2048):
        r = encircled_energy_radius(
            _embed(base64, N), dx, threshold=threshold)
        assert abs(r / ref64 - 1.0) < _PAD_TOL, (
            f"64 -> {N} padding sweep drifted: {r!r} vs {ref64!r}.")


@pytest.mark.parametrize('threshold', [0.50, 0.80])
def test_a2_airy_ee_radius_is_invariant_under_zero_padding(threshold):
    """Same invariance on an Airy amplitude (a beam with real ring
    structure rather than a monotone Gaussian tail)."""
    base, dx = _airy_field_analytic(256, 9.76)
    ref = encircled_energy_radius(base, dx, threshold=threshold)
    for N in (512, 1024):
        r = encircled_energy_radius(_embed(base, N), dx,
                                    threshold=threshold)
        assert abs(r / ref - 1.0) < _PAD_TOL, (
            f"Airy EE radius drifted N=256 -> {N}: {r!r} vs {ref!r}.")


def test_a2_padding_drift_pin_would_have_caught_the_v5_29_behaviour():
    """Counter-pin: reproduce the v5.29 algorithm (256-point ladder to
    the grid corner) on the same fixture and show it violates the
    invariance pin above -- i.e. the pin has teeth."""
    dx, w0 = 1e-6, 20.3e-6
    base = _gaussian_field(128, dx, w0)

    def v5_29_radius(E, threshold):
        radii, ee = encircled_energy_curve(E, dx, n_radii=256)
        if ee[-1] < threshold:
            return float(radii[-1])
        idx = int(np.searchsorted(ee, threshold, side='left'))
        if idx <= 0:
            return float(radii[0])
        t = ((threshold - ee[idx - 1]) /
             (ee[idx] - ee[idx - 1]) if ee[idx] != ee[idx - 1] else 0.0)
        return float(radii[idx - 1] + t * (radii[idx] - radii[idx - 1]))

    ref = v5_29_radius(base, _EE_1E2)
    drifts = [abs(v5_29_radius(_embed(base, N), _EE_1E2) / ref - 1.0)
              for N in (256, 512, 1024)]
    assert max(drifts) > 100 * _PAD_TOL, (
        f"the v5.29 corner-ladder algorithm was expected to drift with "
        f"padding; measured drifts {drifts}.  If it no longer drifts, "
        f"this counter-pin (and the tolerance above) needs revisiting.")


# ==========================================================================
# A-2 pin 2 -- analytic Gaussian: EE radius at 1 - e^-2 is the waist
# ==========================================================================

@pytest.mark.parametrize('w0,dx,N', [
    (20.3e-6, 1e-6, 128),
    (17.37e-6, 1e-6, 128),
    (13.7e-6, 0.5e-6, 256),
    (25e-6, 1e-6, 256),
])
def test_a2_gaussian_radius_at_86_5_percent_is_the_waist(w0, dx, N):
    """Textbook identity: a TEM00 Gaussian encircles ``1 - e^-2 =
    86.47%`` of its power inside ``r = w``.  Inverting the curve at that
    fraction must return the waist.  Measured errors on these four
    fixtures: 0.25% / 0.62% / 0.32% / 0.08% -- all half-pixel quadrature
    bias, none of it array-size dependent."""
    E = _gaussian_field(N, dx, w0)
    r = encircled_energy_radius(E, dx, threshold=_EE_1E2)
    rel = abs(r - w0) / w0
    assert rel < 0.01, (
        f"encircled_energy_radius(1-e^-2) = {r * 1e6:.5f} um for a "
        f"w0 = {w0 * 1e6:.2f} um Gaussian (rel {rel:.4%} > 1%).")
    # Forward consistency: the curve really does read ~0.8647 at r = w.
    _, ee = encircled_energy_curve(E, dx, radii=np.array([w0]))
    assert abs(float(ee[0]) - _EE_1E2) < 5e-3, (
        f"EE(w0) = {ee[0]:.6f}, analytic {_EE_1E2:.6f}.")


@pytest.mark.parametrize('threshold',
                         [0.20, 0.30, 0.50, 0.80, _EE_1E2, 0.95, 0.99])
@pytest.mark.parametrize('dx,w0,N', [
    (0.5e-6, 13.7e-6, 256),
    (1e-6, 20.3e-6, 128),
    (0.25e-6, 13.7e-6, 512),
])
def test_a2_gaussian_radius_tracks_the_analytic_inverse(
        threshold, dx, w0, N):
    """Beyond the single 86.5% identity: the returned radius tracks
    ``w sqrt(-ln(1-EE)/2)`` across the whole curve, to within HALF A
    PIXEL.

    Half a pixel is the honest bound, not a fudge: the cumulative curve
    books each pixel's power at that pixel's own radius, so the
    crossing can only be localised to within the pixel shell it lands
    in.  The measured worst case over these 21 combinations is
    0.17*dx.  Expressing the tolerance in pixels rather than percent is
    what makes it meaningful at small radii too -- at ``r = 9 dx`` half
    a pixel already IS 5.5%, so a flat 1% relative band would be
    asserting something the grid cannot deliver.
    """
    E = _gaussian_field(N, dx, w0)
    got = encircled_energy_radius(E, dx, threshold=threshold)
    want = _r_gauss(w0, threshold)
    assert abs(got - want) <= 0.5 * dx, (
        f"threshold={threshold}, dx={dx * 1e6:.2f} um, "
        f"w0={w0 * 1e6:.2f} um: got {got * 1e6:.5f} um, analytic "
        f"{want * 1e6:.5f} um -- off by {abs(got - want) / dx:.3f} "
        f"pixels ({abs(got - want) / want:.3%}), more than the "
        f"half-pixel quadrature bound.")


# ==========================================================================
# A-2 pin 3 -- Airy: 83.8% at the first dark ring
# ==========================================================================

@pytest.mark.parametrize('pad', [4, 8, 16])
def test_a2_airy_encircles_83_8_percent_at_the_first_dark_ring(pad):
    """Textbook Airy value: 83.8% of the power falls inside the first
    dark ring.  Pinned in the FORWARD direction, which is the
    well-posed one -- ``dEE/dr`` vanishes at the null, so the inverse
    at that threshold is intrinsically unstable (see the module
    docstring).  Measured forward deviations: 2.3e-4 / 5.2e-5 /
    6.9e-5."""
    E, dx = _airy_field_fft(pad)
    _, ee = encircled_energy_curve(E, dx, radii=np.array([_R_ZERO]))
    assert abs(float(ee[0]) - 0.83785) < 1e-3, (
        f"EE at the Airy first dark ring = {ee[0]:.6f}, textbook "
        f"0.83785 (pad={pad}, {_R_ZERO / dx:.2f} samples per first "
        f"zero).")


@pytest.mark.parametrize('pad,tol', [(8, 0.015), (16, 0.015), (32, 0.015)])
def test_a2_airy_84_percent_radius_matches_the_true_airy_value(pad, tol):
    """The default ``threshold=0.84`` against the CORRECT analytic
    reference.

    The spec-sheet shorthand "84% encircled energy ~ 1.22 lam f/#" is
    a convenient lie: the Airy encircles 83.7785% inside the first
    dark ring, so the exact 84% crossing sits further out.  Solving
    ``1 - J0(v)^2 - J1(v)^2 = 0.84`` gives ``v = 4.290940``, i.e.
    ``r(84%) = 1.11955 * (1.22 lam f/#)`` -- the shorthand is +12% low,
    which is the real reason the v4.14.0 pin
    (``test_audit_analysis.py::test_airy_84_percent_radius_matches_
    rayleigh``) needs 20% of slop: it compares against a reference
    that is wrong by construction, on a 2.44-samples-per-first-zero
    grid where half a pixel is already 18% of the radius.

    Measured here against the true value: -1.38% at 4.9 samples per
    first zero, then +0.67% / +0.39% / +0.27% at 9.8 / 19.5 / 39.
    """
    v84 = 4.290939900208022                 # brentq root, see docstring
    r84_true = v84 / np.pi * _LAM * _FN
    E, dx = _airy_field_fft(pad)
    got = encircled_energy_radius(E, dx, threshold=0.84)
    rel = abs(got - r84_true) / r84_true
    assert rel < tol, (
        f"pad={pad} ({_R_ZERO / dx:.2f} samples per first zero): "
        f"r(84%) = {got * 1e6:.5f} um, analytic {r84_true * 1e6:.5f} um "
        f"(rel {rel:.4%} > {tol:.2%}).")
    # And the shorthand really is ~12% low -- documented, not asserted
    # as an accuracy claim, so the number in the docstring stays honest.
    assert 1.09 < r84_true / (1.22 * _LAM * _FN) < 1.15


@pytest.mark.parametrize('pad', [8, 16])
def test_a2_airy_radius_at_83_8_percent_round_trips(pad):
    """Consistency check that survives the ill-conditioning: whatever
    radius the inverse returns for 0.838, evaluating the curve there
    must give 0.838 back.  Measured: 1.5e-6 (pad=8), 0.0 (pad=16).
    Pre-fix the 256-point ladder cut the corner off the curve and this
    round trip missed by up to 1e-2."""
    E, dx = _airy_field_fft(pad)
    r = encircled_energy_radius(E, dx, threshold=0.83785)
    _, ee = encircled_energy_curve(E, dx, radii=np.array([r]))
    assert abs(float(ee[0]) - 0.83785) < 1e-3, (
        f"round trip: r(0.83785) = {r * 1e6:.5f} um but EE there is "
        f"{ee[0]:.8f}.")


# ==========================================================================
# A-2 -- the exact-inversion contract itself
# ==========================================================================

_BRACKET_CASES = [
    pytest.param('gaussian-w20.3', _gaussian_field(256, 1e-6, 20.3e-6),
                 1e-6, id='gaussian-w20.3'),
    pytest.param('gaussian-w5', _gaussian_field(512, 0.5e-6, 5e-6),
                 0.5e-6, id='gaussian-w5'),
]


@pytest.mark.parametrize('label,E,dx', _BRACKET_CASES)
@pytest.mark.parametrize('threshold', [0.30, 0.50, 0.80, _EE_1E2, 0.95])
def test_a2_returned_radius_is_the_first_crossing_of_the_curve(
        label, E, dx, threshold):
    """The documented contract -- "the smallest radius at which the
    curve reaches ``threshold``" -- stated as a two-sided bracket:

    * ``EE(r) >= threshold`` (the disc really does hold the fraction);
    * ``EE(0.999 r) <= threshold`` (no smaller radius already did).

    v5.29 violated BOTH sides on this fixture: at threshold 0.50 it
    returned a radius holding only 0.4916 (the curve had not crossed
    yet), and at 0.80 on the w=5 um beam it returned a radius holding
    0.8101 (the crossing was well inside).  Because the inversion now
    shares ``_ee_sorted_cumulative`` with the forward curve, the
    bracket is exact by construction.
    """
    r = encircled_energy_radius(E, dx, threshold=threshold)
    _, ee_at = encircled_energy_curve(E, dx, radii=np.array([r]))
    _, ee_below = encircled_energy_curve(
        E, dx, radii=np.array([r * (1.0 - 1e-3)]))
    assert float(ee_at[0]) >= threshold - 1e-12, (
        f"{label}: EE({r * 1e6:.6f} um) = {ee_at[0]:.10f} < requested "
        f"{threshold} -- the returned radius does not hold the "
        f"requested fraction.")
    assert float(ee_below[0]) <= threshold + 1e-12, (
        f"{label}: EE(0.999 r) = {ee_below[0]:.10f} already exceeds "
        f"{threshold} -- the returned radius is not the FIRST crossing.")


def test_a2_radius_is_monotone_in_threshold():
    """Sanity: asking for more power can never give a smaller radius."""
    E = _gaussian_field(256, 1e-6, 20e-6)
    thresholds = np.linspace(0.05, 0.99, 40)
    radii = [encircled_energy_radius(E, 1e-6, threshold=float(t))
             for t in thresholds]
    assert all(b >= a - 1e-18 for a, b in zip(radii, radii[1:])), (
        f"non-monotone EE radius vs threshold: {radii}")


# ==========================================================================
# Contract preservation
# ==========================================================================

def test_a2_hot_centre_delta_still_returns_zero():
    """v4.14.1 P1-NEW-6 contract: a delta at the centre pixel holds all
    the power at r = 0, so any threshold in (0, 1] short-circuits to
    0 m."""
    E = np.zeros((128, 128), dtype=complex)
    E[64, 64] = 1.0 + 0.0j
    assert encircled_energy_radius(E, dx=1e-6, threshold=0.5) == 0.0
    assert encircled_energy_radius(E, dx=1e-6, threshold=0.99) == 0.0


def test_a2_threshold_one_returns_the_grid_extent():
    """v4.14.0 5A.4 contract: threshold = 1.0 returns the maximum
    in-grid radius (the corner), never something inside the beam."""
    N, dx, w0 = 128, 1e-6, 15e-6
    E = _gaussian_field(N, dx, w0)
    r_full = encircled_energy_radius(E, dx, threshold=1.0)
    r_corner = float(np.sqrt(2.0) * (N / 2) * dx)
    assert (N / 2) * dx <= r_full <= r_corner + 1e-9, (
        f"threshold=1.0 -> {r_full * 1e6:.4f} um; expected "
        f"[{(N / 2) * dx * 1e6:.4f}, {r_corner * 1e6:.4f}] um.")


def test_a2_degenerate_zero_field_returns_the_grid_extent():
    """A zero-power field has an identically-zero curve that never
    reaches any threshold -- report the grid extent, as the documented
    clip behaviour says (and as the v5.29 zero-curve path did)."""
    N, dx = 32, 1e-6
    r = encircled_energy_radius(np.zeros((N, N), dtype=complex), dx,
                                threshold=0.5)
    assert r == pytest.approx(np.sqrt(2.0) * (N / 2) * dx, rel=1e-12)


def test_a2_anamorphic_dy_and_explicit_centroid_are_threaded():
    """``dy`` and ``centroid`` must still reach the computation."""
    E = _gaussian_field(256, 1e-6, 20e-6)
    iso = encircled_energy_radius(E, 1e-6, threshold=0.5)
    ana = encircled_energy_radius(E, 1e-6, dy=2e-6, threshold=0.5)
    assert ana != iso, "dy is inert on encircled_energy_radius."
    off = encircled_energy_radius(E, 1e-6, centroid=(5e-6, 0.0),
                                  threshold=0.5)
    assert off > iso, (
        f"an off-beam centre must need a larger radius: {off!r} vs "
        f"{iso!r}.")


def test_a2_threshold_validation_unchanged():
    """Signature contract: threshold must be in (0, 1]."""
    E = _gaussian_field(64, 1e-6, 10e-6)
    for bad in (0.0, -0.1, 1.01, float('nan')):
        with pytest.raises(ValueError, match='threshold'):
            encircled_energy_radius(E, 1e-6, threshold=bad)


def test_a2_non_2d_input_is_rejected_by_name():
    """The function no longer borrows ``encircled_energy_curve``'s
    guard, so it carries its own -- and the message must name the
    actual entry point the user called."""
    with pytest.raises(ValueError) as exc:
        encircled_energy_radius(np.ones((4, 4, 4), dtype=complex), 1e-6)
    assert '3-D' in str(exc.value)
    assert 'encircled_energy_radius' in str(exc.value)


def test_a2_curve_and_radius_share_one_construction():
    """Structural pin on the fix: both public functions must go through
    ``_ee_sorted_cumulative``.  If either grows its own copy of the
    sort-and-accumulate step, the exact-inversion bracket above becomes
    an accident waiting to drift."""
    import inspect

    from lumenairy.analysis import psf_mtf_otf as mod

    for fn in (mod.encircled_energy_curve, mod.encircled_energy_radius):
        src = inspect.getsource(fn)
        assert '_ee_sorted_cumulative' in src, (
            f"{fn.__name__} no longer uses the shared exact "
            f"cumulative-energy construction.")
    assert 'n_radii=256' not in inspect.getsource(
        mod.encircled_energy_radius), (
        "encircled_energy_radius is back on a fixed radius ladder.")


def test_a2_docstring_no_longer_claims_sub_percent_accuracy():
    """Doc pin: the v5.29 comment asserted the 256-point ladder gave
    "sub-percent accuracy on the threshold crossing" while it drifted
    6% with padding.  The replacement must state the array-size
    independence and both residual limits instead."""
    doc = encircled_energy_radius.__doc__ or ''
    assert 'sub-percent' not in doc, (
        "the retracted sub-percent accuracy claim is back in the "
        "docstring.")
    assert 'independent of the array size' in doc
    assert 'dark ring' in doc, (
        "the docstring must warn that inverting a threshold at a dark "
        "ring is ill-conditioned.")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
