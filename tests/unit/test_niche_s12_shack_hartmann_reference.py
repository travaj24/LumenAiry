"""S12 -- what ``shack_hartmann`` really does with a global tilt, and the
reference-centroid calibration defect that measuring it turned up (fixed).

Provenance
----------
``54a2dcf`` (AUDIT_ADVERSARIAL_CODEBASE_2026_07_25, Territory A) flagged
but did not claim: *"shack_hartmann returns ~0 slopes for a global 2 mrad
tilt (possibly legitimate per-lenslet reference cancellation)"*.  Measured
here; neither of the two candidate explanations survives.

Verdict on the flag (S12-2 contract, :class:`TestGlobalTiltContract`)
---------------------------------------------------------------------
The "~0" was a *spatial-spread* statistic, and ~0 is the physically
correct value for one.  A GLOBAL tilt tilts every sub-aperture by the same
amount, so the recovered slope MAP is uniform: ``ptp(slopes_x)`` lands at
3e-19..1e-17 rad (fp noise -- the flag's 4.8e-14 urad is 4.8e-20 rad),
while ``nanmean(slopes_x)`` carries the whole signal.  Specifically:

* NOT reference cancellation.  The reference is one flat-field constant
  and cannot cancel a tilt-proportional displacement -- doubling the tilt
  doubles the reported slope (pinned, ratio 2.00 +/- 5%).
* NOT a lost centroid either.  Slope tracks tilt with gain 0.945-0.949 for
  theta <= 1 mrad (pinned).  The ~5% deficit is finite sub-aperture
  truncation of the shifted spot, and the SAME 0.947 gain shows up
  independently in the defocus and astigmatism oracles -- so it is a
  sampling property, not a tilt-specific bug.

S12-1, the real defect -- FIXED IN v5.30
----------------------------------------
Measuring the flag turned up a separate, genuine defect: the sensor's
slope ZERO POINT was wrong.  The reference-centroid pass propagated its
flat calibration field with a bare ``fftshift(fft2(ifftshift(...)))``
while the measurement pass used the bandlimited angular-spectrum kernel at
``z = lenslet_focal``.  So (1) the subtracted reference was not the
centroid the pipeline yields for a flat wavefront, and (2) a bare ``fft2``
lands in the Fraunhofer plane, sample pitch
``wavelength * lenslet_focal / (sa_pixels * dx)`` rather than ``dx``, so
centroiding it against ``Xsa`` was dimensionally wrong too.

Observable: a PERFECTLY FLAT wavefront read back as a uniform tilt,
against the analytic oracle *flat in => zero slopes out*.  Measured
``nanmean(slopes_x)`` [rad] on a 256x256 flat field at 632.8 nm:

    dx     pitch/dx  focal   pre-v5.30    v5.30
    5 um    4        2 mm    -1.245e-03   0.0
    5 um   32        5 mm    -2.472e-05   0.0
    10 um  16        2 mm    +6.514e-03   0.0
    20 um   8        5 mm    +2.075e-03   0.0
    20 um  32        2 mm    -1.523e-02   0.0

Worst case over the 45-configuration sweep: 1.5235e-02 rad, i.e. 15.2 mrad
of invented tilt -- larger than the 2 mrad signal such a sensor is asked
to measure.

The fix folds the reference in as slice 0 of the same ASM batch that
carries the measurements, making it the measurement path's own zero-slope
reference by construction.  A flat wavefront now reads EXACTLY ``0.0`` --
not merely small -- at all 45 configurations, because slice 0 is
bit-identical to every measurement slice when the input is flat.

Because the pre-fix bias was a single additive CONSTANT it left the slope
map's uniformity and every least-squares SHAPE gain untouched and
corrupted only the zero point.  That is why every aberration oracle below
reads the same before and after, and why this is a pure CALIBRATION
correction rather than a change of physics.
:class:`TestAdditiveConstantProof` pins that character directly: the
constant fitted out of a DEFOCUS slope map equalled the flat-field slope
pre-fix (-2.4548e-05 vs -2.472095e-05) and is now negligible (<= 2.9e-06).

Collateral, corrected with justification in the same pass:
``test_v4_16_1_agent_a.py``'s three ``test_bug2_shack_hartmann_*`` pins
recovered the reconstruction pitch from a ``lenslet_pitch = 1.7 * dx``
probe -- ``sa_pixels = 2``, where the diffraction spot is ~65x WIDER than
the propagation window, the focal intensity is uniform, and the centroid
degenerates to the window mean.  Their slopes were nonzero ONLY because of
the bias this fix removes.  They are re-based onto a resolvable
``sa_pixels = 9`` probe there; see that file for the measured numbers.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import lumenairy as la

LAM = 632.8e-9
DETECTOR_PY = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..',
    'lumenairy', 'analysis', 'detector.py'))

# Well-sampled reference configuration for the aberration oracles: 32 px
# per sub-aperture, diffraction spot lam*f/D = 19.8 um = 4 px, comfortably
# inside the 160 um sub-aperture.
GOOD = dict(N=512, dx=5e-6, pitch=32 * 5e-6, focal=5e-3)

# The flat-field slope at GOOD as measured BEFORE the S12-1 fix.  Retained
# as the historical magnitude the additive-constant proof is calibrated
# against; post-fix the same quantity is exactly 0.0.
PRE_FIX_BIAS_AT_GOOD = -2.472095e-05


def _field(N, dx, lam=LAM, tilt=0.0, defocus=0.0, astig=0.0):
    """Unit-amplitude pupil carrying an analytic OPD.

    ``tilt`` is an angle [rad]; ``defocus`` / ``astig`` are OPD amplitudes
    [m] at the edge radius ``R = N * dx / 2``.
    """
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    R = (N / 2) * dx
    opd = (tilt * X
           + defocus * (X ** 2 + Y ** 2) / R ** 2
           + astig * (X ** 2 - Y ** 2) / R ** 2)
    return np.exp(1j * (2 * np.pi / lam) * opd), R


def _run(cfg=None, **field_kw):
    cfg = cfg or GOOD
    E, R = _field(cfg['N'], cfg['dx'], **field_kw)
    out = la.shack_hartmann(
        E, cfg['dx'], wavelength=LAM,
        lenslet_pitch=cfg['pitch'], lenslet_focal=cfg['focal'])
    return out, R


def _mean_slope(**field_kw):
    return float(np.nanmean(_run(**field_kw)[0][0]))


def _lenslet_centres(n_lenslets, cfg=None):
    """Sub-aperture centre coordinates [m], matching the library geometry
    (``x0 = N // 2 - (n_lenslets * sa) // 2``, centre at ``r0 + sa / 2``)."""
    cfg = cfg or GOOD
    sa = int(round(cfg['pitch'] / cfg['dx']))
    x0 = cfg['N'] // 2 - (n_lenslets * sa) // 2
    return (((x0 + np.arange(n_lenslets) * sa + sa / 2) - cfg['N'] / 2)
            * cfg['dx'])


def _fit_gain_offset(measured, oracle):
    """Least-squares ``measured ~ gain * oracle + offset``.

    Returns ``(gain, offset, residual_rms / oracle_peak)``.  Fitting the
    offset separates the two things sampling and calibration each control:
    the SHAPE gain, and the zero point that S12-1 corrupted.
    """
    m = np.isfinite(measured)
    A = np.stack([oracle[m], np.ones(int(m.sum()))], axis=1)
    (gain, offset), *_ = np.linalg.lstsq(A, measured[m], rcond=None)
    resid = float(np.sqrt(np.mean(
        (measured[m] - (gain * oracle[m] + offset)) ** 2)))
    return float(gain), float(offset), resid / float(np.max(np.abs(oracle)))


# ===========================================================================
# The analytic oracle S12-1 was violating: flat in => zero slopes out
# ===========================================================================

class TestFlatWavefrontOracle:
    """A flat wavefront has zero slope everywhere, so the sensor must
    report zero.  Post-fix it reports EXACTLY 0.0, because the reference
    slice is bit-identical to every measurement slice on a flat input.
    The 1e-18 rad bound is fifteen orders below the smallest pre-fix error
    (2.472e-05 rad) and immune to any fp reassociation.
    """

    # (dx, pitch/dx, focal) -- the five rows tabulated in the module
    # docstring plus three more corners of the 45-point sweep.
    @pytest.mark.parametrize('dx,sa_ratio,focal', [
        (5e-6, 4, 2e-3),
        (5e-6, 32, 5e-3),
        (5e-6, 16, 20e-3),
        (10e-6, 16, 2e-3),
        (10e-6, 8, 5e-3),
        (20e-6, 8, 5e-3),
        (20e-6, 32, 2e-3),
        (20e-6, 4, 20e-3),
    ])
    def test_flat_field_reports_no_slope(self, dx, sa_ratio, focal):
        E, _ = _field(256, dx)
        sx, sy, wf, cx, cy = la.shack_hartmann(
            E, dx, wavelength=LAM,
            lenslet_pitch=sa_ratio * dx, lenslet_focal=focal)
        for name, arr in (('slopes_x', sx), ('slopes_y', sy)):
            peak = float(np.nanmax(np.abs(arr)))
            assert peak <= 1e-18, (
                f'flat wavefront must give zero {name}; got peak '
                f'{peak:.4e} rad at dx={dx * 1e6:g} um, '
                f'pitch/dx={sa_ratio}, f={focal * 1e3:g} mm.  A nonzero '
                f'value means the reference centroid is no longer measured '
                f'through the same transform as the measurement -- S12-1, '
                f'which produced up to 1.52e-02 rad of phantom tilt.')
        for name, arr in (('centroids_x', cx), ('centroids_y', cy)):
            peak = float(np.nanmax(np.abs(arr)))
            assert peak <= 1e-20, (
                f'flat wavefront must give zero {name}; got {peak:.4e} m.')
        wf_ptp = float(np.nanmax(wf) - np.nanmin(wf))
        assert wf_ptp <= 1e-24, (
            f'flat wavefront must reconstruct flat; got ptp {wf_ptp:.4e} m.')

    def test_flat_field_zero_is_exact_not_merely_small(self):
        """The fix makes this bit-exact, a stronger statement than "small",
        and it is what proves the reference is the measurement path's own:
        slice 0 and every measurement slice see identical input."""
        E, _ = _field(256, GOOD['dx'])
        out = la.shack_hartmann(
            E, GOOD['dx'], wavelength=LAM,
            lenslet_pitch=GOOD['pitch'], lenslet_focal=GOOD['focal'])
        for name, arr in zip(('slopes_x', 'slopes_y',
                              'centroids_x', 'centroids_y'),
                             (out[0], out[1], out[3], out[4])):
            finite = arr[np.isfinite(arr)]
            assert np.all(finite == 0.0), (
                f'{name} should be exactly 0.0 on a flat field; got '
                f'max|.| = {np.max(np.abs(finite)):.4e}')

    def test_the_zero_point_is_uniform_across_sub_apertures(self):
        """Whatever the zero point is, it must be the SAME for every
        sub-aperture; a per-lenslet-varying bias would corrupt shapes too,
        not just the zero point.  Held before the fix and after it."""
        E, _ = _field(256, GOOD['dx'])
        sx, sy, *_ = la.shack_hartmann(
            E, GOOD['dx'], wavelength=LAM,
            lenslet_pitch=GOOD['pitch'], lenslet_focal=GOOD['focal'])
        for name, arr in (('slopes_x', sx), ('slopes_y', sy)):
            assert float(np.nanmax(arr) - np.nanmin(arr)) <= 1e-18, (
                f'the flat-field {name} zero point must be uniform across '
                f'sub-apertures')


# ===========================================================================
# The flag's verdict, pinned as a contract
# ===========================================================================

class TestGlobalTiltContract:
    """**Contract.**  A global tilt gives a UNIFORM slope map: its MEAN
    carries the tilt, its spatial spread is fp noise.  Summarising the
    sensor with ``ptp`` / ``std`` reads ~0 for a pure tilt and must not be
    read as "blind to tilt" -- that is exactly what ``54a2dcf`` saw.
    """

    def test_global_tilt_slope_map_is_uniform(self):
        (sx, _sy, _wf, _cx, _cy), _R = _run(tilt=1e-3)
        spread = float(np.nanmax(sx) - np.nanmin(sx))
        assert spread <= 1e-15, (
            f'a global tilt must produce a UNIFORM slope map; got '
            f'ptp={spread:.4e} rad (measured 3e-19..1e-17 across configs)')

    def test_the_mean_carries_the_tilt_even_though_the_spread_is_zero(self):
        assert _mean_slope(tilt=0.0) == 0.0
        assert _mean_slope(tilt=1e-3) > 5e-4, (
            'the MEAN slope must carry a 1 mrad tilt (measured 9.450e-04 '
            'rad).  A near-zero ptp is NOT evidence of a blind sensor -- '
            'read the mean.')

    def test_tilt_is_not_removed_by_the_reference_subtraction(self):
        """Refutes hypothesis (a): the reference is one constant, so the
        tilt-proportional response survives it and scales."""
        s1 = _mean_slope(tilt=0.5e-3)
        s2 = _mean_slope(tilt=1.0e-3)
        assert s1 > 1e-4
        assert s2 / s1 == pytest.approx(2.0, rel=0.05), (
            f'tilt response must scale linearly; got {s1:.4e} -> {s2:.4e} '
            f'(ratio {s2 / s1:.3f}, expected 2.0)')

    def test_x_tilt_does_not_leak_into_y_slopes(self):
        (sx, sy, *_), _ = _run(tilt=1e-3)
        leak = abs(float(np.nanmean(sy)))
        signal = abs(float(np.nanmean(sx)))
        assert leak < 1e-3 * signal, (
            f'a pure x tilt must not move slopes_y: leak {leak:.4e} rad vs '
            f'x signal {signal:.4e} rad')


# ===========================================================================
# Oracle: pure tilt -> centroid = f tan(theta)
# ===========================================================================

class TestTiltOracle:
    """``centroid = lenslet_focal * tan(theta)``, i.e. ``slope = theta``.

    Now that the zero point is clean this is pinned ABSOLUTELY (pre-fix it
    could only be pinned differentially, since the S12-1 constant polluted
    every reading).  Measured gains at GOOD: 0.948129 (0.2 mrad), 0.948777
    (0.5 mrad), 0.944992 (1.0 mrad).  The ~5% deficit is finite
    sub-aperture truncation of the shifted spot and grows with the
    walk-out fraction ``f theta / (pitch / 2)``: 0.911687 at 2 mrad, whose
    spot has walked 12% of the way to the sub-aperture edge.  The pin
    covers theta <= 1 mrad and bounds the wider case separately.
    """

    @pytest.mark.parametrize('theta', [0.2e-3, 0.5e-3, 1.0e-3])
    def test_slope_matches_theta(self, theta):
        gain = _mean_slope(tilt=theta) / theta
        assert 0.90 <= gain <= 1.00, (
            f'slope / theta = {gain:.6f} outside [0.90, 1.00] at '
            f'theta={theta * 1e3:.1f} mrad (measured 0.945-0.949)')

    @pytest.mark.parametrize('theta', [0.2e-3, 0.5e-3, 1.0e-3])
    def test_centroid_matches_f_tan_theta(self, theta):
        c = float(np.nanmean(_run(tilt=theta)[0][3]))
        oracle = GOOD['focal'] * np.tan(theta)
        assert c == pytest.approx(oracle, rel=0.10), (
            f'centroid {c * 1e6:.5f} um vs oracle f*tan(theta) = '
            f'{oracle * 1e6:.5f} um ({100 * (c - oracle) / oracle:+.2f}%)')

    def test_large_tilt_walks_the_spot_out_and_loses_gain(self):
        """Documents the truncation mechanism rather than treating it as a
        defect: gain 0.911687 at 2 mrad, and 0.878 for the 1 -> 2 mrad
        differential step."""
        gain = _mean_slope(tilt=2e-3) / 2e-3
        assert 0.85 <= gain <= 0.95, f'2 mrad gain {gain:.6f}'
        step = (_mean_slope(tilt=2e-3) - _mean_slope(tilt=1e-3)) / 1e-3
        assert 0.80 <= step <= 0.95, f'1->2 mrad differential {step:.6f}'

    def test_tilt_response_is_odd(self):
        plus = _mean_slope(tilt=+1e-3)
        minus = _mean_slope(tilt=-1e-3)
        assert plus > 0 > minus
        assert plus == pytest.approx(-minus, rel=0.02)

    def test_reconstruction_integrates_the_uniform_slope(self):
        """``wavefront[i, j] = 0.5 * pitch_actual * (s_x * j + s_y * i)``
        for uniform slopes, so the extremes sit at opposite corners and

            ptp = 0.5 * pitch_actual * (n - 1) * (|s_x| + |s_y|)

        (each leg contributes its magnitude independently, whatever the
        relative sign).  With the zero point clean, ``s_y == 0`` for a pure
        x tilt, so this reduces to the single-leg form.  All 16x16
        sub-apertures are in bounds at GOOD, so no NaN-zeroing enters the
        cumsum.
        """
        (sx, sy, wf, _cx, _cy), _R = _run(tilt=1e-3)
        assert np.all(np.isfinite(sx)) and np.all(np.isfinite(sy))
        n = sx.shape[0]
        pitch_actual = int(round(GOOD['pitch'] / GOOD['dx'])) * GOOD['dx']
        oracle = (0.5 * pitch_actual * (n - 1)
                  * (abs(float(np.nanmean(sx))) + abs(float(np.nanmean(sy)))))
        assert float(np.nanmax(wf) - np.nanmin(wf)) == pytest.approx(
            oracle, rel=0.02)


# ===========================================================================
# Oracle: defocus / astigmatism -> slope linear in the pupil coordinate
# ===========================================================================

class TestDefocusOracle:
    """``OPD = W (x^2 + y^2) / R^2``  =>  ``dOPD/dx = 2 W x / R^2``: the
    slope map must be LINEAR in the sub-aperture centre coordinate.

    Measured gains 0.948176 / 0.946454 at W = 0.2 / 1.0 um -- the same
    truncation gain the tilt oracle shows -- with residual/peak 4.11e-04 /
    3.68e-03.  Gains and residuals are UNCHANGED by the S12-1 fix (the
    oracle is odd in x, so a constant projects onto the offset term rather
    than the gain); what changed is the fitted offset, now ~0.
    """

    @pytest.mark.parametrize('W', [0.2e-6, 0.5e-6, 1.0e-6])
    def test_defocus_slope_is_linear_in_the_pupil_coordinate(self, W):
        (sx, sy, _wf, _cx, _cy), R = _run(defocus=W)
        xc = _lenslet_centres(sx.shape[0])
        oracle_x = 2.0 * W * np.broadcast_to(xc[None, :], sx.shape) / R ** 2
        gain, offset, rel = _fit_gain_offset(sx, oracle_x)
        assert 0.90 <= gain <= 1.00, (
            f'defocus slope gain {gain:.6f} outside [0.90, 1.00] '
            f'(measured 0.9465-0.9491)')
        assert rel <= 1e-2, (
            f'defocus slope must be linear in x; residual/peak {rel:.4e} '
            f'(measured <= 3.7e-3)')
        assert abs(offset) <= 1e-5, (
            f'the fitted zero point is {offset:.4e} rad; post-S12-1 it must '
            f'be negligible (measured 1.7e-07..2.8e-06).  The pre-fix value '
            f'was {PRE_FIX_BIAS_AT_GOOD:.4e}.')
        oracle_y = 2.0 * W * np.broadcast_to(xc[:, None], sy.shape) / R ** 2
        gain_y, _off_y, rel_y = _fit_gain_offset(sy, oracle_y)
        assert gain_y == pytest.approx(gain, rel=0.05), (
            'defocus must be rotationally symmetric: x and y gains equal')
        assert rel_y <= 1e-2

    def test_defocus_is_not_mistaken_for_tilt(self):
        """Defocus is odd in the pupil coordinate, so its slope map has a
        large spread and a ~zero mean -- the opposite signature to a global
        tilt.  With the zero point clean this is now a clean statement
        about the mean itself."""
        (sx, *_), _ = _run(defocus=1e-6)
        spread = float(np.nanmax(sx) - np.nanmin(sx))
        assert spread > 1e-3
        assert abs(float(np.nanmean(sx))) < 0.05 * spread


class TestAstigmatismOracle:
    """``OPD = W (x^2 - y^2) / R^2``  =>  ``dOPD/dx = +2 W x / R^2`` and
    ``dOPD/dy = -2 W y / R^2``: the two slope maps must carry OPPOSITE
    gains.  Measured 0.949102 / 0.946455 at W = 0.5 / 1.0 um,
    residual/peak 4.13e-04 / 3.68e-03.  This is the cross-coupled case the
    reconstruction docstring warns about, so only the SLOPES are pinned,
    never the integral.
    """

    @pytest.mark.parametrize('W', [0.5e-6, 1.0e-6])
    def test_astigmatism_slopes_have_opposite_gains(self, W):
        (sx, sy, _wf, _cx, _cy), R = _run(astig=W)
        xc = _lenslet_centres(sx.shape[0])
        oracle_x = 2.0 * W * np.broadcast_to(xc[None, :], sx.shape) / R ** 2
        oracle_y = -2.0 * W * np.broadcast_to(xc[:, None], sy.shape) / R ** 2
        gx, _ox, rx = _fit_gain_offset(sx, oracle_x)
        gy, _oy, ry = _fit_gain_offset(sy, oracle_y)
        assert 0.90 <= gx <= 1.00, f'astig slopes_x gain {gx:.6f}'
        assert 0.90 <= gy <= 1.00, (
            f'astig slopes_y gain {gy:.6f} -- a positive gain against the '
            f'NEGATED oracle is the sign flip astigmatism requires')
        assert rx <= 1e-2 and ry <= 1e-2, (
            f'astigmatism slope maps must be linear; residual/peak '
            f'x={rx:.4e} y={ry:.4e}')

    def test_astigmatism_is_distinguishable_from_defocus(self):
        """Defocus curves x and y the same way, astigmatism oppositely, so
        ``slopes_x + slopes_y.T`` cancels for astigmatism and adds for
        defocus."""
        (ax, ay, *_), _ = _run(astig=1e-6)
        (dx_, dy_, *_), _ = _run(defocus=1e-6)
        astig_sum = float(np.nanmax(np.abs(ax + ay.T)))
        defoc_sum = float(np.nanmax(np.abs(dx_ + dy_.T)))
        assert astig_sum < 0.2 * defoc_sum, (
            f'astigmatism should cancel under x + y.T ({astig_sum:.4e}) '
            f'while defocus adds ({defoc_sum:.4e})')


class TestSuperposition:
    """Tilt and defocus superpose in the OPD, so their measured slope maps
    must add.  With the zero point clean this needs no bias bookkeeping.
    Measured residual 2.58e-02 (1 mrad + 0.5 um) and 5.16e-02 (2 mrad +
    1.0 um) relative to peak; the residual is spot-truncation
    nonlinearity, so the pin allows 10%."""

    @pytest.mark.parametrize('theta,W', [(1e-3, 0.5e-6), (2e-3, 1.0e-6)])
    def test_tilt_plus_defocus_equals_tilt_plus_defocus(self, theta, W):
        (st, *_), _ = _run(tilt=theta)
        (sd, *_), _ = _run(defocus=W)
        (sb, *_), _ = _run(tilt=theta, defocus=W)
        resid = float(np.nanmax(np.abs(sb - (st + sd))))
        peak = float(np.nanmax(np.abs(sb)))
        assert resid / peak <= 0.10, (
            f'superposition residual {resid:.4e} rad is '
            f'{100 * resid / peak:.2f}% of peak {peak:.4e} '
            f'(measured 2.6-5.2%)')


# ===========================================================================
# S12-1: the additive-constant character, and why the bug existed
# ===========================================================================

class TestAdditiveConstantProof:
    """S12-1 was ONE ADDITIVE CONSTANT on every slope.  That is what makes
    the fix a pure calibration correction -- shapes, gains and uniformity
    untouched -- and it is worth pinning permanently, because a future
    reference regression that varied per sub-aperture, or that scaled with
    the signal, would be a far worse failure than the one just fixed.
    """

    def test_the_flat_field_slope_is_the_zero_point_of_every_measurement(self):
        """Fit ``gain * oracle + offset`` to a DEFOCUS slope map: the
        fitted constant must equal the FLAT-field slope.  Pre-fix that was
        -2.4548e-05 (W=0.2 um) / -2.7531e-05 (W=1.0 um) against a
        flat-field slope of -2.472095e-05 -- agreement to ~1%, which is how
        the single-constant character was established.  Post-fix both
        sides are ~0 and the relation still holds.
        """
        flat = _mean_slope(tilt=0.0)
        assert flat == 0.0, (
            f'post-S12-1 the flat-field slope must be exactly 0.0; got '
            f'{flat:.6e} (pre-fix {PRE_FIX_BIAS_AT_GOOD:.6e})')
        for W in (0.2e-6, 1.0e-6):
            (sx, _sy, _wf, _cx, _cy), R = _run(defocus=W)
            xc = _lenslet_centres(sx.shape[0])
            oracle = 2.0 * W * np.broadcast_to(
                xc[None, :], sx.shape) / R ** 2
            _gain, offset, _rel = _fit_gain_offset(sx, oracle)
            assert abs(offset - flat) <= 0.15 * abs(flat) + 1e-5, (
                f'the constant fitted out of the W={W * 1e6:.1f} um defocus '
                f'slope map ({offset:.6e}) must be the flat-field slope '
                f'({flat:.6e}); if they diverge the zero-point error is no '
                f'longer a single additive constant and the shape oracles '
                f'stop being valid')

    def test_the_zero_point_does_not_scale_with_the_signal(self):
        """A constant, not a gain error: tripling the defocus must not move
        the fitted offset."""
        offsets = []
        for W in (0.2e-6, 0.6e-6):
            (sx, _sy, _wf, _cx, _cy), R = _run(defocus=W)
            xc = _lenslet_centres(sx.shape[0])
            oracle = 2.0 * W * np.broadcast_to(
                xc[None, :], sx.shape) / R ** 2
            offsets.append(_fit_gain_offset(sx, oracle)[1])
        assert abs(offsets[1] - offsets[0]) <= 1e-5, (
            f'fitted zero point moved from {offsets[0]:.4e} to '
            f'{offsets[1]:.4e} when the defocus tripled; it must be '
            f'signal-independent')


class TestWhyTheBugExisted:
    """Discriminators that reconstruct the pre-fix quantities locally, so
    they document the mechanism without depending on the old behaviour."""

    def test_a_bare_fft_reference_lands_in_a_different_plane(self):
        """The bare-``fft2`` reference plane has sample pitch
        ``lam * f / (sa * dx)``, not ``dx``.  They coincide only when
        ``sa * dx**2 == lam * f`` -- which is why the pre-fix error swung
        from 1.2e-05 to 1.5e-02 rad across sampling choices instead of
        being a fixed offset."""
        # accidental near-coincidence that masked the bug
        assert (LAM * 5e-3 / (8 * 20e-6)) / 20e-6 == pytest.approx(
            0.989, rel=0.01)
        # ...and a config where the assumed pitch is ~4x wrong
        assert (LAM * 5e-3 / (32 * 5e-6)) / 5e-6 == pytest.approx(
            3.955, rel=0.01)

    def test_the_two_candidate_references_differ_by_a_multi_mrad_tilt(self):
        """Recompute both references at the worst-measured configuration
        and show the gap is multi-mrad of phantom tilt."""
        from lumenairy.propagators.propagation import _build_asm_H_square
        dx, sa, focal = 20e-6, 32, 2e-3
        k0 = 2 * np.pi / LAM
        xsa = (np.arange(sa) - sa / 2) * dx
        Xsa, Ysa = np.meshgrid(xsa, xsa)
        chirp = np.exp(-1j * k0 * (Xsa ** 2 + Ysa ** 2) / (2 * focal))

        def centroid(I):
            return float((Xsa * I).sum() / I.sum())

        spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(chirp)))
        I_fft = np.abs(spectrum) ** 2
        H = _build_asm_H_square(sa, dx, focal, LAM,
                                dtype=np.complex128, bandlimit=True)
        I_asm = np.abs(np.fft.fftshift(np.fft.ifft2(
            np.fft.ifftshift(spectrum * H)))) ** 2
        fake_tilt = (centroid(I_asm) - centroid(I_fft)) / focal
        assert abs(fake_tilt) > 1e-3, (
            f'the bare-fft and ASM references differ by only '
            f'{fake_tilt:.4e} rad at dx=20 um, 32 px/sa, f=2 mm; the '
            f'pre-fix flat-field slope there was -1.523e-02 rad')

    def test_an_unresolvable_spot_carries_no_centroid_information(self):
        """Why ``test_v4_16_1_agent_a.py``'s pitch pins had to be re-based:
        the sub-aperture is propagated on a window equal to its own width,
        so at ``sa_pixels = 2`` the spot is ~65x wider than the window, the
        focal intensity is uniform, and the centroid degenerates to the
        window mean.  Those pins' nonzero slopes were entirely S12-1."""
        dx, lam, focal = 5e-6, 1.31e-6, 5e-3
        overfill = {}
        for sa in (2, 9):
            window = sa * dx
            overfill[sa] = (lam * focal / window) / window
        assert overfill[2] > 50.0, (
            f'sa=2 spot/window = {overfill[2]:.1f}x -- degenerate')
        assert overfill[9] < 10.0, (
            f'sa=9 spot/window = {overfill[9]:.2f}x -- the re-based probe')


class TestFixIsRecorded:
    """The code and its documentation must not drift apart: the fix must be
    present in the source AND recorded in the docstring.  Both halves FAIL
    on a pre-fix (865e922) worktree."""

    def test_the_reference_is_a_slice_of_the_measurement_batch(self):
        with open(DETECTOR_PY, encoding='utf-8') as fh:
            src = fh.read()
        assert 'E_focus_ref' not in src, (
            'shack_hartmann has a separate reference-plane transform again '
            '(E_focus_ref): the reference centroid must be measured '
            'through the SAME bandlimited-ASM batch as the measurements, '
            'or a flat wavefront reads back as a tilt (S12-1).')
        assert 'E_all = np.concatenate(' in src, (
            'the flat-field reference slice is no longer concatenated onto '
            'the measurement batch; the exact-zero flat-field oracle '
            'depends on that construction.')

    def test_the_docstring_records_the_fix_not_a_deviation(self):
        with open(DETECTOR_PY, encoding='utf-8') as fh:
            src = fh.read()
        assert 'FIXED IN v5.30 -- S12-1' in src, (
            "shack_hartmann's Notes no longer record the S12-1 calibration "
            'fix; the measured before/after table is the only place a '
            'caller learns their pre-v5.30 SH slopes carried up to '
            '15.2 mrad of phantom tilt.')
        assert 'KNOWN DEVIATION' not in src, (
            'the S12-1 KNOWN DEVIATION block is still present even though '
            'the fix is applied -- delete it (or, if the fix was reverted, '
            'restore the xfail markers this file used to carry).')

    def test_the_global_tilt_contract_is_documented(self):
        """The flag's actual confusion -- ptp vs mean -- must stay
        documented where a caller will see it."""
        with open(DETECTOR_PY, encoding='utf-8') as fh:
            src = fh.read()
        assert 'UNIFORM slope map' in src and 'read the MEAN' in src, (
            "shack_hartmann's docstring no longer states that a global "
            'tilt gives a uniform slope map whose MEAN carries the signal; '
            'that omission is what produced the 54a2dcf flag.')

    def test_the_sampling_caveat_is_documented(self):
        """The degeneracy that made the v4.16.1 probe meaningless must be
        documented, or the next probe will hit it too."""
        with open(DETECTOR_PY, encoding='utf-8') as fh:
            src = fh.read()
        assert 'Sampling caveat' in src, (
            "shack_hartmann's Notes no longer warn that the focal spot "
            'must fit the sub-aperture window '
            '(lam * lenslet_focal / (sa_pixels * dx) < sa_pixels * dx), '
            'outside which slopes read 0 regardless of the input.')
