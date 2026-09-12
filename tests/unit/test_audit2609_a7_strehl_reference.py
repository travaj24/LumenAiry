"""A2 -- the Strehl denominator must be the EXACT converging sphere.

AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11 §8 row A2.

What was wrong
--------------
``diffraction_limited_peak`` built its aberration-free reference from the
PARAXIAL quadratic phase ``exp(-i k r^2 / 2 f)`` while
``angular_spectrum_propagate`` -- the propagator both the reference and the
measurement go through -- is exact.  The quadratic differs from the sphere
by a genuine spherical-aberration term

    W040 = (D / 2)^4 / (8 f^3 lambda) = (D / lambda) / (128 (f/#)^3)  waves

so the "diffraction-limited" DENOMINATOR was itself an aberrated
wavefront: its focal peak is depressed by its own Strehl ``S_ref`` and
every ratio taken against it is inflated by ``1 / S_ref``.  Measured
end-to-end before the fix, on a PERFECT exact-spherical converging
wavefront at f/2 (D = 614 um, f = 1.228 mm, 600 nm, N = 4096):

    diffraction_limited_peak (paraxial ref)           = 1.342173e+04
    peak of the perfect sphere at z = f               = 1.519335e+05
    through_focus_scan best Strehl for a PERFECT lens = 11.3200

After: the reference equals the perfect sphere's own peak
(1.519335e+05) and the scan reports 1.0000.

Oracles
-------
1. A pupil that IS aberration-free by construction must report Strehl 1.
   Independent of the library: the numerator is built and propagated in
   the test.
2. The inflation the fix removes is predicted by the extended-Marechal
   Strehl of the paraxial-minus-sphere phase error, computed through
   ``strehl_phase_integral`` -- a different module, verified correct by
   the same audit.
3. At f/50 the two references agree to 1e-9, i.e. the change is confined
   to the non-paraxial regime it is about.
"""
import numpy as np
import pytest

from lumenairy.analysis.strehl import strehl_phase_integral
from lumenairy.analysis.through_focus import (diffraction_limited_peak,
                                              find_best_focus,
                                              through_focus_scan)
from lumenairy.propagators.propagation import angular_spectrum_propagate

LAM = 600e-9
K0 = 2 * np.pi / LAM

# Bar on "Strehl of an aberration-free pupil is 1".
#
# Derivation: the numerator and the denominator are the SAME field put
# through the SAME propagator, so in exact arithmetic the ratio is 1 and
# the residual is the float64 FFT floor, ~N * eps ~ 1e-13 at N = 2048.
# Measured: 1.000000000 at f/50, f/20, f/10, f/5, f/3.9, f/2.5 and f/2
# (nine significant figures).  Pre-fix the same quantity was 1.0015 at
# f/5, 1.0988 at f/2.5 and 1.4292 at f/2 on this fixture -- so 1e-6 sits
# 7 decades above the floor and 3 decades below the smallest pre-fix
# error, and the f/2 case clears it by 5 decades.
STREHL_TOL = 1e-6


def _sphere_pupil(fnum, D=200e-6, N=2048):
    """Aberration-free converging pupil at the requested f/#."""
    f = fnum * D
    dx = min(LAM * fnum / 4.0, 0.5e-6)      # keep the focal spot resolved
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    R2 = X ** 2 + Y ** 2
    ap = (R2 <= (D / 2) ** 2).astype(float)
    E = ap * np.exp(-1j * K0 * (np.sqrt(R2 + f * f) - f))
    return E, f, dx, R2, ap


@pytest.mark.parametrize('fnum', [50.0, 10.0, 5.0, 2.5, 2.0])
def test_perfect_sphere_reports_strehl_one(fnum):
    E, f, dx, _, _ = _sphere_pupil(fnum)
    ref = diffraction_limited_peak(E, LAM, f, dx)
    peak = float(np.max(np.abs(
        angular_spectrum_propagate(E, f, LAM, dx)) ** 2))
    s = peak / ref
    assert abs(s - 1.0) < STREHL_TOL, (
        f'a PERFECT f/{fnum:.1f} pupil reports Strehl {s:.6f}; the '
        f'reference is aberrated by W040 = '
        f'{(200e-6 / LAM) / (128 * fnum ** 3):.3f} waves.')


@pytest.mark.parametrize('fnum', [5.0, 2.5, 2.0])
def test_inflation_removed_matches_the_paraxial_aberration_oracle(fnum):
    """The size of the removed error is predicted, not just observed.

    ``S_ref`` -- the Strehl of the paraxial reference against the sphere
    -- comes from ``strehl_phase_integral`` on the phase DIFFERENCE
    between the two references.  The fix must raise the denominator by
    exactly ``1 / S_ref``.
    """
    E, f, dx, R2, ap = _sphere_pupil(fnum)
    ref_new = diffraction_limited_peak(E, LAM, f, dx)
    E_old = np.abs(E) * np.exp(-1j * K0 * R2 / (2.0 * f))
    ref_old = float(np.max(np.abs(
        angular_spectrum_propagate(E_old, f, LAM, dx)) ** 2))
    # Phase error of the paraxial reference relative to the sphere.
    dphi = -K0 * (R2 / (2.0 * f) - (np.sqrt(R2 + f * f) - f))
    s_ref = strehl_phase_integral(ap * np.exp(1j * dphi))
    measured = ref_new / ref_old
    predicted = 1.0 / s_ref
    # 12 % bar: `strehl_phase_integral` is the on-axis intensity of the
    # aberrated reference at the geometric focus, which is what the
    # denominator is, but the two differ by the focal shift of the
    # spherically aberrated spot -- a second-order effect that grows with
    # W040.  Measured ratio measured/predicted: 1.000 at f/5,
    # 1.005 at f/2.5, 1.10 at f/2 (W040 = 0.33 waves).  The quantity
    # being bounded spans 1.002 -> 1.43, so a 12 % bar still separates
    # every row from "no inflation at all".
    assert measured == pytest.approx(predicted, rel=0.12), (
        f'f/{fnum:.1f}: denominator rose by {measured:.4f}x, '
        f'extended-Marechal oracle predicts {predicted:.4f}x '
        f'(S_ref = {s_ref:.5f}).')
    assert measured > 1.0


def test_paraxial_limit_is_unchanged():
    """At f/50 the sphere and its quadratic expansion are the same
    wavefront to 3e-5 waves, so the change must be invisible there --
    the fix is confined to the regime that was wrong."""
    E, f, dx, R2, _ = _sphere_pupil(50.0)
    ref_new = diffraction_limited_peak(E, LAM, f, dx)
    E_old = np.abs(E) * np.exp(-1j * K0 * R2 / (2.0 * f))
    ref_old = float(np.max(np.abs(
        angular_spectrum_propagate(E_old, f, LAM, dx)) ** 2))
    assert ref_new == pytest.approx(ref_old, rel=1e-6)


def test_through_focus_scan_of_a_perfect_lens_never_exceeds_one():
    """The end-to-end symptom: ``through_focus_scan`` reported best
    Strehl 11.32 for a diffraction-limited f/2 pupil.  A Strehl above 1
    is unphysical for an aberration-free pupil at its own focus."""
    D, f, dx, N = 200e-6, 400e-6, 0.3e-6, 2048      # f/2
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    R2 = X ** 2 + Y ** 2
    E = (R2 <= (D / 2) ** 2) * np.exp(-1j * K0 * (np.sqrt(R2 + f * f) - f))
    ideal = diffraction_limited_peak(E, LAM, f, dx)
    z = np.linspace(0.97 * f, 1.03 * f, 13)
    scan = through_focus_scan(E, dx, LAM, z, ideal_peak=ideal, verbose=False)
    _, s_best = find_best_focus(scan, 'strehl')
    assert s_best <= 1.0 + 1e-3, (
        f'best Strehl {s_best:.4f} for a perfect f/2 lens (pre-fix: '
        f'11.32 on the audit fixture).')
    assert s_best > 0.99


def test_negative_focal_length_keeps_the_paraxial_sign():
    """A diverging reference must still reduce to ``r^2 / (2 f)`` with
    f < 0; the exact form is ``sign(f) (sqrt(r^2 + f^2) - |f|)``."""
    N, dx, D = 256, 2e-6, 200e-6
    f = -5e-3
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    R2 = X ** 2 + Y ** 2
    inside = R2 <= (D / 2) ** 2
    E = inside.astype(complex)
    sag = np.sign(f) * (np.sqrt(R2 + f * f) - abs(f))
    paraxial = R2 / (2.0 * f)
    # f/25 geometry: inside the PUPIL the two agree to
    # (D/2)^4 / (8 |f|^3) = 1.0e-10 m = 1.7e-4 waves, and they must agree
    # in SIGN -- the pre-fix quadratic form is negative for f < 0, so an
    # ``abs(f)``-only exact form would flip the reference to converging.
    d_waves = float(np.max(np.abs((sag - paraxial)[inside]))) / LAM
    assert d_waves < 1e-3, f'diverging sag differs by {d_waves:.3e} waves'
    assert np.all(sag[inside] <= 0.0) and np.all(paraxial[inside] <= 0.0)
    # And the call itself must not raise or return a non-finite value.
    val = diffraction_limited_peak(E, LAM, f, dx)
    assert np.isfinite(val) and val > 0
