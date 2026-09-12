"""A4 -- ``eval_image_plane_wfe`` must model an infinite conjugate, and
must not return a silently wrong answer for a very distant finite one.

AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11 §8 row A4.

What was wrong
--------------
The function REQUIRED ``object_distance > 0`` and finite, and launched the
bundle at the object (``bundle.z = -object_distance``).  Someone modelling
an infinite conjugate had no option but a large finite number, and the
object-side ray-surface intersection then cancels catastrophically: the
quadratic carries ``|P - C|^2 ~ object_distance^2`` against ``R^2``, and
the two roots are separated by only ``~2|R|``, so the float64 rounding of
``b^2`` lands in the root as ``~eps * object_distance^2 / |R|``.

Measured on a biconvex N-BK7 singlet (R = +-50 mm, d = 3 mm, D = 10 mm,
587.6 nm, EFL 48.87 mm, BFL 47.875 mm) -- the on-axis CHIEF ray, which
must land at z = 0 by definition:

    object_distance   chief z0        reported PV    reported RMS
    1e3 m             +0.0006 um       3.4675 w       1.0430 w
    1e4 m             +0.0238 um       3.6912 w       1.0649 w
    1e5 m             -5.34 um        47.2747 w      10.7591 w
    1e6 m           +589.41 um       213.5098 w      61.9051 w

and ``image_plane='best_rms'`` then "corrected" the 1e6 m case by moving
the reference sphere to 96.6 mm on a lens whose BFL is 47.9 mm.  No
warning was emitted at any distance.

After: ``object_distance = float('inf')`` (or ``None``) launches a
collimated bundle on a plane wavefront a few aperture widths before
surface 0 and reports img_d_m = 47.87519 mm = the BFL, PV = 3.5944 waves;
every finite distance past the precision cliff warns.

Oracle
------
An independent transverse-ray-aberration integral computed in this file
from a hand-built collimated trace: ``W(rho) = -(a / R) * int eps drho``
with ``eps`` the image-plane transverse ray error.  Cross-checked against
the traced longitudinal spherical aberration,
``W040 = -LSA * a^2 / (4 lambda R^2)``.  Neither touches
``eval_image_plane_wfe``.
"""
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.analysis.image_plane_wfe import eval_image_plane_wfe
from lumenairy.raytrace import surfaces_from_prescription, system_abcd, trace
from lumenairy.raytrace.trace import _make_bundle

LAM = 587.6e-9
APERTURE = 10e-3


@pytest.fixture(scope='module')
def singlet():
    p = la.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3, glass='N-BK7',
                        aperture=APERTURE)
    surfs = surfaces_from_prescription(p)
    _, efl, bfl, _ = system_abcd(surfs, LAM)
    return p, surfs, float(efl), float(bfl)


def _transverse_ray_oracle(surfs, R_img, n=201):
    """PV wavefront error from the transverse ray aberration of a
    collimated bundle, integrated over the pupil.  Independent of
    ``eval_image_plane_wfe``."""
    a = APERTURE / 2
    rho = np.linspace(0.0, 1.0, n)
    b = _make_bundle(x=np.zeros(n), y=rho * a, L=np.zeros(n),
                     M=np.zeros(n), wavelength=LAM)
    b.z = np.full(n, -20e-3)
    b.opd = np.zeros(n)
    f = trace(b, surfs, LAM, output_filter='last').image_rays
    y2 = np.asarray(f.y)
    z2 = np.asarray(f.z)
    M2 = np.asarray(f.M)
    N2 = np.asarray(f.N)
    eps = y2 + M2 * (R_img - z2) / N2
    eps = eps - eps[0]
    a_exit = float(np.nanmax(np.abs(y2)))
    W = -(a_exit / R_img) * np.concatenate(([0.0], np.cumsum(
        0.5 * (eps[1:] + eps[:-1]) * np.diff(rho)))) / LAM
    # Longitudinal-SA cross-check of the same trace.
    z_marginal = z2[-1] - y2[-1] * N2[-1] / M2[-1]
    lsa = z_marginal - R_img
    w040 = -lsa * a_exit ** 2 / (4 * LAM * R_img ** 2)
    return rho, W, float(W.max() - W.min()), float(w040)


def test_infinite_object_distance_is_accepted_and_lands_on_the_bfl(singlet):
    p, surfs, efl, bfl = singlet
    pres = dict(p)
    pres['object_distance'] = float('inf')
    w = eval_image_plane_wfe(pres, LAM, n_pupil=31)
    assert w.img_d_m == pytest.approx(bfl, rel=1e-9), (
        f'infinite conjugate must image at the BFL ({bfl * 1e3:.5f} mm), '
        f'got {w.img_d_m * 1e3:.5f} mm.')
    assert np.isfinite(w.pv_waves) and w.pv_waves > 0


def test_none_object_distance_means_infinity(singlet):
    p, _, _, bfl = singlet
    pres = dict(p)
    pres['object_distance'] = None
    w = eval_image_plane_wfe(pres, LAM, n_pupil=15)
    assert w.img_d_m == pytest.approx(bfl, rel=1e-9)


def test_infinite_conjugate_wfe_matches_the_transverse_ray_oracle(singlet):
    """The audit's headline number: PV ~ 3.7 waves, not 213.

    Bar: 5 % on PV.  The oracle integrates a trapezoid over 201 pupil
    samples of a quantity the library evaluates on a different pupil
    partition and against a reference SPHERE rather than a plane, so a
    few percent is the honest agreement floor -- the audit quotes ~3 %
    for the same comparison in the valid finite regime.  Measured here:
    library 3.5944 waves, transverse-ray oracle 3.6586 (-1.8 %),
    longitudinal-SA cross-check W040 = 3.651 waves (LSA -0.8144 mm).
    The quantity being separated is a factor of 59 (213.5 / 3.59).
    """
    p, surfs, _, bfl = singlet
    pres = dict(p)
    pres['object_distance'] = float('inf')
    n = 201
    rho = np.linspace(0.0, 1.0, n)
    w = eval_image_plane_wfe(pres, LAM, pupil_grid=(np.zeros(n), rho))
    _, _, pv_oracle, w040 = _transverse_ray_oracle(surfs, w.img_d_m, n=n)
    assert w.pv_waves == pytest.approx(pv_oracle, rel=0.05), (
        f'library PV {w.pv_waves:.4f} waves vs transverse-ray oracle '
        f'{pv_oracle:.4f} waves.')
    assert w.pv_waves == pytest.approx(w040, rel=0.05), (
        f'library PV {w.pv_waves:.4f} waves vs longitudinal-SA '
        f'W040 = {w040:.4f} waves.')


def test_chief_ray_is_exactly_on_the_sphere_at_infinity(singlet):
    """The chief has zero OPD by construction; a nonzero value means the
    launch geometry drifted (the 1e6 m case put it 589 um off the
    vertex)."""
    p, _, _, _ = singlet
    pres = dict(p)
    pres['object_distance'] = float('inf')
    w = eval_image_plane_wfe(pres, LAM, n_pupil=9)
    chief = int(np.argmin(w.px ** 2 + w.py ** 2))
    assert abs(float(w.opd_w[chief])) < 1e-12


@pytest.mark.parametrize('obj_d', [1e4, 1e5, 1e6])
def test_distant_finite_object_warns(obj_d, singlet):
    """A distance past the float64 cliff must say so rather than return
    a degraded number silently."""
    p, _, _, _ = singlet
    pres = dict(p)
    pres['object_distance'] = obj_d
    with pytest.warns(RuntimeWarning, match='object_distance'):
        eval_image_plane_wfe(pres, LAM, n_pupil=9)


@pytest.mark.parametrize('obj_d', [1.0, 1e2, 1e3])
def test_usable_finite_object_does_not_warn(obj_d, singlet):
    """The gate must be silent where the answer is still right -- the
    reported PV is flat at 3.465-3.468 waves from 1e2 to 1e3 m and only
    starts moving at 1e4."""
    p, _, _, _ = singlet
    pres = dict(p)
    pres['object_distance'] = obj_d
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        eval_image_plane_wfe(pres, LAM, n_pupil=9)


def test_finite_object_converges_to_the_infinite_conjugate(singlet):
    """Physics check on the new launch path: as the object recedes, the
    finite-conjugate wavefront must approach the collimated one.  Compared
    on the SAME pupil samples so the comparison is not confounded by
    which marginal ray happens to vignette."""
    p, _, _, _ = singlet
    n = 41
    rho = np.linspace(0.0, 0.9, n)
    grid = (np.zeros(n), rho)
    pres_inf = dict(p)
    pres_inf['object_distance'] = float('inf')
    w_inf = eval_image_plane_wfe(pres_inf, LAM, pupil_grid=grid)
    pres_far = dict(p)
    pres_far['object_distance'] = 1e3
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        w_far = eval_image_plane_wfe(pres_far, LAM, pupil_grid=grid)
    m = np.isfinite(w_inf.opd_w) & np.isfinite(w_far.opd_w)
    span = float(np.nanmax(w_inf.opd_w[m]) - np.nanmin(w_inf.opd_w[m]))
    assert float(np.max(np.abs(w_inf.opd_w[m] - w_far.opd_w[m]))) < 0.02 * span


def test_off_axis_at_infinity_requires_a_field_angle(singlet):
    """A field point at infinity is a DIRECTION; ``field_max_m`` (a
    height) cannot define it, and guessing would silently produce the
    wrong chief ray."""
    p, _, _, _ = singlet
    pres = dict(p)
    pres['object_distance'] = float('inf')
    with pytest.raises(ValueError, match='field_max_rad'):
        eval_image_plane_wfe(pres, LAM, field=(0.0, 1.0), n_pupil=9,
                             field_max_m=1e-3)
    w = eval_image_plane_wfe(pres, LAM, field=(0.0, 1.0), n_pupil=21,
                             field_max_rad=np.deg2rad(1.0))
    assert np.isfinite(w.pv_waves) and w.pv_waves > 0


def test_zero_or_negative_object_distance_still_refused(singlet):
    p, _, _, _ = singlet
    pres = dict(p)
    pres['object_distance'] = -1.0
    with pytest.raises(ValueError, match='object_distance'):
        eval_image_plane_wfe(pres, LAM, n_pupil=9)
