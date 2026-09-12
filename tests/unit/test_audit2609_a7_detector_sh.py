"""A3 -- the Shack-Hartmann ``wavefront`` must carry the slopes' own scale.

AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11 §8 row A3.

What was wrong
--------------
The reconstruction averaged two ONE-SIDED cumulative integrals::

    wf_x = cumsum(sx, axis=1) * pitch;  wf_x -= wf_x[0, 0]
    wf_y = cumsum(sy, axis=0) * pitch;  wf_y -= wf_y[0, 0]
    wavefront = 0.5 * (wf_x + wf_y)

``wf_x[i, j]`` is ``W(x_j, y_i) - W(x_0, y_i)`` and ``wf_y[i, j]`` is
``W(x_j, y_i) - W(x_j, y_0)``, so for any wavefront separable in x and y
-- tilt, defocus, astigmatism, i.e. essentially every use -- their
average is ``(W - W_00) / 2``.  Exactly half.  Anchoring removes each
half's piston but not the factor.

Measured before (N = 256, dx = 5 um, pitch = 32 dx, f = 5 mm, 632.8 nm),
as the ratio of the reconstruction to the analytic truth scaled by the
sensor's own measured slope gain:

    tilt 0.20 mrad   0.4741   (gain 0.9481)
    tilt 0.50 mrad   0.4744   (gain 0.9488)
    tilt 1.00 mrad   0.4725   (gain 0.9450)
    defocus, 1 um edge   0.4626

After: 0.9481 / 0.9488 / 0.9450 and 0.9340 -- the slope gain itself, as
it must be.  ``max|wf|`` on the defocus case went 5.4263e-07 m ->
1.4351e-06 m against a truth of 1.5000e-06 m.

The oracle in every test below is the ANALYTIC wavefront that was put in,
divided by the slope gain the same run reports, so the ~5 % sub-aperture
truncation deficit is separated from the factor being pinned.
"""
import numpy as np
import pytest

import lumenairy as la
from lumenairy.analysis.detector import _reconstruct_wavefront

LAM = 632.8e-9
CFG = dict(N=256, dx=5e-6, pitch=32 * 5e-6, focal=5e-3)


def _run(**field):
    x = (np.arange(CFG['N']) - CFG['N'] / 2) * CFG['dx']
    X, Y = np.meshgrid(x, x)
    R = (CFG['N'] / 2) * CFG['dx']
    opd = (field.get('tilt', 0.0) * X
           + field.get('defocus', 0.0) * (X ** 2 + Y ** 2) / R ** 2
           + field.get('astig', 0.0) * (X ** 2 - Y ** 2) / R ** 2)
    E = np.exp(1j * (2 * np.pi / LAM) * opd)
    out = la.shack_hartmann(E, CFG['dx'], LAM, CFG['pitch'], CFG['focal'],
                            **field.get('kw', {}))
    return out, X, Y, R, opd


def _lenslet_grid(n):
    sa = int(round(CFG['pitch'] / CFG['dx']))
    x0 = CFG['N'] // 2 - (n * sa) // 2
    c = (((x0 + np.arange(n) * sa + sa / 2) - CFG['N'] / 2) * CFG['dx'])
    return np.meshgrid(c, c)


@pytest.mark.parametrize('theta', [0.2e-3, 0.5e-3, 1.0e-3])
def test_tilt_reconstruction_is_full_scale_not_half(theta):
    """``W = theta * x``: the reconstruction slope against the lenslet
    coordinate must equal the MEASURED slope, not half of it.

    Bar: 3 % relative.  The reconstruction is an exact integral of the
    slope map for a uniform slope (both the Southwell trapezoid and the
    single path integral are exact on a constant), so the only
    discrepancy is the edge lenslets' own gain spread -- measured 0.9450
    -- and the quantity being separated is a factor of 2.  3 % is 30x
    inside that separation.
    """
    (sx, sy, wf, _, _), _, _, _, _ = _run(tilt=theta)
    n = sx.shape[0]
    XL, _ = _lenslet_grid(n)
    gain = float(np.nanmean(sx)) / theta
    m = np.isfinite(wf)
    # Fit wf ~ a * x + b over the measured lenslets.
    A = np.stack([XL[m], np.ones(int(m.sum()))], axis=1)
    (a, _b), *_ = np.linalg.lstsq(A, wf[m], rcond=None)
    ratio = float(a) / (theta * gain)
    assert ratio == pytest.approx(1.0, rel=0.03), (
        f'reconstruction slope / (theta * slope-gain) = {ratio:.4f}; '
        f'pre-fix this was 0.5 by construction for any separable '
        f'wavefront.')


def test_defocus_reconstruction_is_full_scale_not_half():
    """The audit's fourth row: a 1 um-edge defocus read back at 0.4626 of
    truth before the fix, 0.9340 after (the sensor's slope gain)."""
    W = 1e-6
    (sx, sy, wf, _, _), _, _, R, _ = _run(defocus=W)
    n = sx.shape[0]
    XL, YL = _lenslet_grid(n)
    truth = W * (XL ** 2 + YL ** 2) / R ** 2
    truth = truth - truth[0, 0]
    m = np.isfinite(wf)
    scale = float(np.polyfit(truth[m].ravel(), wf[m].ravel(), 1)[0])
    assert 0.85 <= scale <= 1.05, (
        f'reconstructed / true defocus scale = {scale:.4f}; pre-fix '
        f'0.4626.')
    assert float(np.nanmax(np.abs(wf))) > 0.8 * float(np.max(np.abs(truth)))


def test_astigmatism_is_reconstructed_at_scale():
    """Astigmatism is separable but has OPPOSITE-signed x and y slopes,
    so an averaging reconstruction does not merely halve it -- it is the
    case the old code's own comment admitted 'mis-reconstructs'."""
    W = 0.6e-6
    (sx, sy, wf, _, _), _, _, R, _ = _run(astig=W)
    n = sx.shape[0]
    XL, YL = _lenslet_grid(n)
    truth = W * (XL ** 2 - YL ** 2) / R ** 2
    truth = truth - truth[0, 0]
    m = np.isfinite(wf)
    scale = float(np.polyfit(truth[m].ravel(), wf[m].ravel(), 1)[0])
    resid = wf[m] - scale * truth[m]
    assert 0.85 <= scale <= 1.05, f'astigmatism scale {scale:.4f}'
    assert float(np.std(resid)) < 0.1 * float(np.max(np.abs(truth)))


@pytest.mark.parametrize('field', [dict(tilt=1e-3), dict(defocus=1e-6),
                                   dict(astig=0.6e-6)])
def test_southwell_and_itoh_agree_on_a_consistent_slope_field(field):
    """Two independent integrators of the same slope map.  Both use the
    same trapezoid neighbour relation, and a noise-free slope field is
    path-consistent, so the least-squares solve over ALL neighbour pairs
    and the single path integral must land on the same wavefront -- which
    is what makes either of them a check on the other.

    Bar: 1 % of the reconstruction's own peak-to-valley.  They are not
    identical -- Southwell distributes the residual inconsistency of the
    measured slopes over every pair while the path integral carries it
    along one route -- and that residual is the measurement's own noise,
    not an algorithmic difference.  Measured worst case over the three
    fields: 0.5 % (astigmatism).
    """
    (sx, sy, wf_s, _, _), _, _, _, _ = _run(**field)
    pitch = int(round(CFG['pitch'] / CFG['dx'])) * CFG['dx']
    wf_i = _reconstruct_wavefront(sx, sy, pitch, 'itoh')
    m = np.isfinite(wf_s) & np.isfinite(wf_i)
    span = float(np.nanmax(wf_s[m]) - np.nanmin(wf_s[m]))
    assert float(np.max(np.abs(wf_s[m] - wf_i[m]))) < 0.01 * span


def test_unmeasured_lenslets_are_nan_not_integrated_through():
    """Out-of-bounds lenslets carry NaN slopes.  Treating them as zero
    slope -- what the cumsum did -- silently extends the wavefront across
    the dead region as if it had been measured flat."""
    N, dx = 32, 4e-6
    E = np.ones((N, N), dtype=complex)
    sx, sy, wf, _, _ = la.shack_hartmann(
        E, dx, 1.31e-6, lenslet_pitch=20e-6, lenslet_focal=200e-6,
        n_lenslets=12)
    assert np.isnan(sx[0, 0])
    assert np.isnan(wf[0, 0]), (
        'a lenslet with no measurement must not be given a wavefront '
        'value')
    assert np.any(np.isfinite(wf))
    np.testing.assert_array_equal(np.isnan(wf), np.isnan(sx))


def test_flat_wavefront_reconstructs_exactly_flat():
    N, dx = 256, 5e-6
    E = np.ones((N, N), dtype=complex)
    _, _, wf, _, _ = la.shack_hartmann(E, dx, LAM, CFG['pitch'],
                                        CFG['focal'])
    finite = wf[np.isfinite(wf)]
    assert np.all(finite == 0.0)


def test_reconstruction_kwarg_is_validated_with_the_function_prefix():
    N, dx = 64, 5e-6
    E = np.ones((N, N), dtype=complex)
    with pytest.raises(ValueError, match=r'shack_hartmann: reconstruction'):
        la.shack_hartmann(E, dx, LAM, 16 * dx, 2e-3,
                          reconstruction='cumsum-average')


def test_reconstruction_is_deterministic():
    out1 = _run(tilt=5e-4)[0]
    out2 = _run(tilt=5e-4)[0]
    for a, b in zip(out1, out2):
        np.testing.assert_array_equal(a, b)
