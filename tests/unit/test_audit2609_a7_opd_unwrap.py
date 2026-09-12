"""A1 -- ``wave_opd_2d`` must not slip whole waves on an aberrated pupil.

AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11 §8 row A1.

What was wrong
--------------
``np.angle(field)`` was taken over the WHOLE grid, zero-amplitude exterior
included (``np.angle(0) == 0``), and unwrapped ``axis=1`` then ``axis=0``.
The row pass anchors at column 0 -- outside the pupil -- so each in-pupil
row picks up its own ``2 pi k``; the column pass then re-anchors
INDEPENDENTLY IN EVERY COLUMN at that column's first in-pupil row, and the
wrap of the entry jump injects a column-dependent whole-wave offset.
Masking to ``valid`` happened only afterwards.

The failure needs no under-sampling: it appears at phase gradients 10-25x
BELOW the Nyquist limit, on smooth, single-valued, simply connected
wavefronts, while ``check_opd_sampling`` reports SAFE.  Measured on a flat
circular pupil (N = 512, dx = 1 um, aperture 400 um, 633 nm) carrying pure
primary coma and NO defocus:

    coma [waves rms]   max |OPD err| pre-fix   post-fix
    0.10                    0.000 waves        0.000
    0.20                    1.000               0.000
    0.50                    1.000               0.000
    1.00                    4.000               0.000
    5.00                   19.000               0.000

and the fitted OSA c8 came back +0.1024 waves for a true +0.2000 pre-fix,
+0.2000 post-fix.

The oracles here are analytic wavefronts, so "correct" is an absolute
statement, not a comparison against another code path.
"""
import numpy as np
import pytest

from lumenairy.analysis.opd import unwrap_phase_2d, wave_opd_2d

LAM = 633e-9
K0 = 2 * np.pi / LAM

# Bar for "the map is the analytic wavefront".
#
# Derivation: the unwrap integrates wrapped neighbour differences along a
# path of at most ~N samples; each step is exact to eps * |phase|, so the
# accumulated error is ~N * eps * |k0 W| = 512 * 2.2e-16 * 2 pi * 5 waves
# = 3.5e-12 rad = 5.6e-13 waves.  Measured max |err| is 0.000 waves to
# every digit the repro prints (max 1.0e-15 waves on the 1-wave-coma
# case).  The defect this bar exists to catch is an INTEGER wave slip,
# i.e. >= 1.0 waves.  1e-6 waves sits 6 decades above the float floor and
# 6 decades below the smallest real failure.
WAVE_TOL = 1e-6


def _coma_pupil(A_waves, N=512, dx=1e-6, ap=400e-6):
    """Flat circular pupil carrying ``A_waves`` rms of OSA j=8 coma."""
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)
    rho = R / (ap / 2)
    th = np.arctan2(Y, X)
    inside = R <= ap / 2
    coma = np.sqrt(8) * (3 * rho ** 3 - 2 * rho) * np.cos(th)
    opd_true = A_waves * LAM * coma
    E = inside * np.exp(1j * K0 * opd_true)
    return E, opd_true, inside, dx, ap


@pytest.mark.parametrize('A', [0.2, 0.3, 0.5, 1.0, 2.0, 5.0])
def test_no_integer_wave_slips_on_a_coma_pupil(A):
    """The exact row from the audit table.  Pre-fix every one of these
    returned a map wrong by 1-19 whole waves over 5-95 % of the pupil."""
    E, opd_true, inside, dx, ap = _coma_pupil(A)
    _, _, opd = wave_opd_2d(E, dx, LAM, aperture=ap)
    m = np.isfinite(opd)
    d = opd[m] - opd_true[m]
    d = d - np.median(d)
    err_waves = float(np.max(np.abs(d))) / LAM
    assert err_waves < WAVE_TOL, (
        f'{A} waves rms of coma gives max |OPD error| = {err_waves:.3f} '
        f'waves; pre-fix this pupil slipped by whole waves over '
        f'{100 * np.mean(np.abs(d) > 0.4 * LAM):.1f} % of the aperture.')
    # And the error is not merely small on average: no sample is a whole
    # wave out.  (The pre-fix failure was EXACTLY integer, so a
    # round-to-nearest-wave histogram is the sharpest discriminator.)
    assert np.all(np.round(d / LAM) == 0)


def test_phase_gradient_is_far_below_nyquist_when_it_used_to_fail():
    """Guards the premise: these pupils are NOT under-sampled, so nothing
    about the failure can be excused as an aliasing limitation."""
    E, opd_true, inside, dx, ap = _coma_pupil(0.5)
    ph = K0 * opd_true
    gx = np.abs(np.diff(ph, axis=1))[inside[:, :-1] & inside[:, 1:]].max()
    gy = np.abs(np.diff(ph, axis=0))[inside[:-1] & inside[1:]].max()
    worst = float(max(gx, gy))
    assert worst < 0.5, (
        f'max |dphi| per sample = {worst:.3f} rad; the fixture is meant '
        f'to sit an order below the pi Nyquist limit (measured 0.27 rad).')


def test_defocus_pupil_returns_the_right_integer_wave_count():
    """Absolute, not median-subtracted: a pupil carrying a known defocus
    must come back with the right WHOLE-WAVE count, which is what makes
    the map usable as an absolute OPD rather than a shape.

    Oracle: the analytic sag ``-r^2 / (2 f)`` of the field that was
    built, evaluated on the same grid.  The centre sample is the anchor
    (a converging wavefront is stationary there), so no gauge freedom is
    being hidden by a piston subtraction.
    """
    N, dx, ap = 512, 1e-6, 400e-6
    f = 2.0e-3                       # ~15.8 waves of edge sag
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    R2 = X ** 2 + Y ** 2
    inside = R2 <= (ap / 2) ** 2
    opd_true = -R2 / (2 * f)
    E = inside * np.exp(1j * K0 * opd_true)
    _, _, opd = wave_opd_2d(E, dx, LAM, aperture=ap)
    m = np.isfinite(opd)
    err_waves = float(np.max(np.abs(opd[m] - opd_true[m]))) / LAM
    assert err_waves < WAVE_TOL, (
        f'absolute OPD error {err_waves:.4f} waves on a {abs(opd_true).max() / LAM:.1f}-wave '
        f'defocus pupil -- the returned map is off by whole waves.')


def test_annular_pupil_is_unwrapped_as_one_wavefront():
    """A centrally obscured pupil is multiply connected: the interior
    zero-amplitude hole is exactly the exterior condition that broke the
    row-then-column unwrap, and both sides of the obscuration must come
    back on the SAME whole-wave branch."""
    N, dx, ap = 384, 1e-6, 300e-6
    obsc = 0.4
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)
    rho = R / (ap / 2)
    th = np.arctan2(Y, X)
    ring = (rho <= 1.0) & (rho >= obsc)
    opd_true = 1.5 * LAM * np.sqrt(8) * (3 * rho ** 3 - 2 * rho) * np.cos(th)
    E = ring * np.exp(1j * K0 * opd_true)
    _, _, opd = wave_opd_2d(E, dx, LAM, aperture=ap)
    m = np.isfinite(opd)
    assert m.sum() > 1000
    d = opd[m] - opd_true[m]
    d = d - np.median(d)
    assert float(np.max(np.abs(d))) / LAM < WAVE_TOL


def test_reliability_method_agrees_with_the_default_on_clean_data():
    """The quality-guided kernel is a different traversal of the same
    path integral, so on residue-free data it must land on the same
    branch everywhere."""
    E, opd_true, inside, dx, ap = _coma_pupil(0.5, N=192, ap=150e-6)
    _, _, a = wave_opd_2d(E, dx, LAM, aperture=ap, unwrap='itoh')
    _, _, b = wave_opd_2d(E, dx, LAM, aperture=ap, unwrap='reliability')
    m = np.isfinite(a) & np.isfinite(b)
    assert m.sum() > 100
    assert float(np.max(np.abs(a[m] - b[m]))) / LAM < WAVE_TOL


def test_unknown_unwrap_method_is_refused_with_the_function_name():
    E, _, _, dx, ap = _coma_pupil(0.2, N=64, ap=40e-6)
    with pytest.raises(ValueError, match=r'wave_opd_2d: unwrap method'):
        wave_opd_2d(E, dx, LAM, aperture=ap, unwrap='quality-guided')


@pytest.mark.parametrize('A', [5.0, 30.0])
def test_aliased_pupil_warns_instead_of_returning_silently_wrong(A):
    """Above the Nyquist gradient no unwrap can succeed -- the wrapped
    data no longer determines the branch.  The function must SAY so
    rather than return a map that looks fine.  The residue it reports is
    measured on the supplied field, not estimated from the sampling:
    1.0 waves at 5 waves rms of coma on the 200 um / 256-sample pupil
    (6.1 rad/sample), 7.0 at 30 waves (36.8 rad/sample)."""
    E, _, _, dx, ap = _coma_pupil(A, N=256, ap=200e-6)
    with pytest.warns(RuntimeWarning, match='residues'):
        wave_opd_2d(E, dx, LAM, aperture=ap)


def test_optical_vortex_is_reported_as_a_residue():
    """A charge-1 vortex has a genuine 2 pi circulation: no single-valued
    OPD map exists, and the residue is exactly one wave."""
    N, dx, ap = 192, 1e-6, 150e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    inside = (X ** 2 + Y ** 2) <= (ap / 2) ** 2
    E = inside * np.exp(1j * np.arctan2(Y, X))
    with pytest.warns(RuntimeWarning, match='residues'):
        wave_opd_2d(E, dx, LAM, aperture=ap)


def test_symmetric_aliased_defocus_has_no_residue_and_is_caught_by_sampling():
    """The residue check's documented blind spot, pinned so nobody is
    later surprised by it.

    A radially symmetric aliased wavefront wraps onto the EXACTLY
    self-consistent phase of a lower-frequency wavefront -- there is no
    inconsistency to find, and no unwrap of any kind can distinguish the
    two.  Measured residue: 0.0000 waves at 16.5, 32.9 and 98.8 rad per
    sample.  The separate SAMPLING gate is what covers this case, and
    fires when ``focal_length`` is supplied.
    """
    from lumenairy.analysis.opd import (_unwrap_2d_itoh,
                                        _unwrap_residue_waves)
    N, dx, ap = 256, 1e-6, 200e-6
    f = 60e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    R2 = X ** 2 + Y ** 2
    inside = R2 <= (ap / 2) ** 2
    E = inside * np.exp(-1j * K0 * R2 / (2 * f))
    phase = np.angle(E)
    out, _, _, resid = _unwrap_2d_itoh(phase, inside)
    assert resid < 1e-9
    # ... and the generic full-grid checker agrees, so the cheap
    # vertical-links-only form the kernel reports is not hiding anything.
    assert _unwrap_residue_waves(phase, out, inside) < 1e-9
    # ... and the sampling criterion does catch it.
    with pytest.warns(RuntimeWarning, match='Nyquist'):
        wave_opd_2d(E, dx, LAM, aperture=ap, focal_length=f)


def test_well_sampled_pupil_does_not_warn():
    """The residue gate must not fire on data the unwrap handles."""
    import warnings
    E, _, _, dx, ap = _coma_pupil(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        wave_opd_2d(E, dx, LAM, aperture=ap)


def test_disconnected_support_warns_and_anchors_each_region():
    """Two separate islands have no phase relationship; saying so is the
    difference between a diagnostic and a fabrication."""
    N, dx = 96, 1e-6
    E = np.zeros((N, N), dtype=complex)
    E[10:30, 10:30] = 1.0
    E[60:80, 60:80] = np.exp(1j * 1.0)
    with pytest.warns(RuntimeWarning, match='disconnected'):
        _, _, opd = wave_opd_2d(E, dx, LAM)
    assert np.isfinite(opd[20, 20]) and np.isfinite(opd[70, 70])
    assert np.isnan(opd[45, 45])


def test_unwrap_phase_2d_is_congruent_to_the_wrapped_phase():
    """The public kernel's defining property: the answer differs from the
    wrapped input by a whole number of cycles at EVERY in-mask sample.
    Any unwrap that smooths (a least-squares one, say) breaks this."""
    rng = np.random.default_rng(11)
    N = 128
    x = (np.arange(N) - N / 2) / (N / 2)
    X, Y = np.meshgrid(x, x)
    mask = (X ** 2 + Y ** 2) <= 0.9 ** 2
    W = 7.0 * (X ** 2 + Y ** 2) + 3.0 * X + 1.5 * Y ** 3
    wrapped = np.angle(np.exp(1j * W))
    out = unwrap_phase_2d(wrapped, mask)
    k = (out[mask] - wrapped[mask]) / (2 * np.pi)
    assert float(np.max(np.abs(k - np.round(k)))) < 1e-9
    d = out[mask] - W[mask]
    assert float(np.max(np.abs(d - np.median(d)))) < 1e-9
