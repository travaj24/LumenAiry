"""Hammer-audit H1 (2026-07-19): slant_correction obliquity OPD sign.

The pre-v5.25.0 ``slant_correction=True`` OPD divided the sag by the local
cosines (``n*sag/cos`` -- the geometric ray path-length through a slab).  The
inversion sign-flipped the leading obliquity (spherical-aberration) term; on a
symmetric biconvex f/5 singlet the two wrong-signed surface corrections
CANCELLED the pupil SA, producing an impossible near-diffraction-limited
3.6 um spot where the dual oracles (Zemax POP via ZOS-API + an independent
exact-raytrace Debye/Huygens integral, 2026-07-19 hammer campaign) both give
~65 um.  That cancellation is what this file exists to keep out.

Oracle provenance (f/5 case: R=+/-51.68 mm, t=5 mm, n=1.5168, lambda=1.31 um,
Gaussian w0=5 mm, image 49.163 mm behind the exit vertex): ZOS POP
r2m = 64.7 um, Debye r2m = 64.98 um, EE80 = 55.2 um.  A near-cancellation
(r2m ~ 3.6 um) is physically impossible for ~8 waves of SA at ANY plane.

UPDATED 2026-09-12 (audit 2026-09-11, finding L12).  v5.25.0's replacement,
``(n2 cos_tt - n1 cos_ti)*sag``, referenced both cosines to the facet NORMAL.
The module's own axial-translation identity needs them referenced to the
Z-AXIS -- a facet is displaced along z, not along its own normal -- which for
a collimated input is ``(n2 cos(theta_i - theta_t) - n1)*sag``.  Expanding
both to O(theta**2) with n1 = 1: exact ``(n2-1) - theta^2 (n2-1)^2/(2 n2)``,
v5.25.0 ``(n2-1) + theta^2 (n2-1)/(2 n2)`` -- opposite SIGN and 1/(n2-1) =
1.94x too large, so its total error was n2/(n2-1) = 2.94x the paraxial
screen's.  Measured against an exact vertex-plane eikonal at four radii the
ratio matched 2.9414-2.9417; the z-axis form is 290x-4000x better than
paraxial on a single face.  ``apply_real_lens`` now ships the z-axis form.

WHAT THAT COSTS ON THIS FIXTURE, MEASURED.  r2m at the image plane against the
65 um dual-oracle truth, at three samplings (the pre-fix numbers reproduced
exactly by rebuilding the v5.25.0 coefficient through ``form_error`` on the
paraxial path):

    dx [um]   paraxial   v5.25.0 slant   z-axis slant
      6.00      25.26        50.31          26.76
      3.00      40.55        76.70          43.06
      1.50      40.55        76.70          43.05

The measurement converges at dx <= 3 um; the old 50.3 um "corrected-slant"
figure quoted in this docstring was taken at dx = 6 um, which is not
converged.  Converged, v5.25.0 OVERSHOOTS the truth (76.7 vs 65) and the
z-axis form UNDERSHOOTS it (43.1 vs 65).  Both gaps have the same cause and it
is NOT the facet coefficient: on a symmetric biconvex the bundle at the SECOND
surface is converging at ~0.1 rad, so ``pz1 = n1 cos(alpha_in) != n1`` and the
collimated assumption the identity is exact under is broken there.  v5.25.0's
wrong-signed, oversized error happened to point the other way and partly
cancel it -- which is the "cancellation between two oppositely-signed
surfaces, not correctness" the 2026-09-11 audit identified in the biconvex
family.  For a converging bundle use ``carrier=`` (``screen_obliquity``),
``surface_model='displaced'`` / ``'tangent_facet'``, or
``apply_real_lens_traced``: all three carry the true local ray angle.

The facet coefficient's own correctness is pinned separately and directly, on
a SINGLE face against the exact one-facet eikonal, in
``tests/unit/test_audit2609_a2_analytic_lens.py::
TestL12SlantCorrectionAxialTranslationIdentity``.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la

_WL = 1.31e-6

# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_H1_FIX_GLASS': lambda wl: 1.5168}


def _singlet_f5():
    return {
        'wavelength': _WL,
        'aperture_diameter': 24e-3,
        'surfaces': [
            {'radius': 51.68e-3, 'thickness': 5e-3,
             'glass_before': 'air', 'glass_after': '_H1_FIX_GLASS',
             'semi_diameter': 12e-3},
            {'radius': -51.68e-3, 'thickness': 0.0,
             'glass_before': '_H1_FIX_GLASS', 'glass_after': 'air',
             'semi_diameter': 12e-3},
        ],
        'thicknesses': [5e-3],
        'stop_index': 0,
    }


def _r2m_at_image(E_exit, dx, z_img, crop_m=500e-6):
    E = la.angular_spectrum_propagate(E_exit, z_img, _WL, dx)
    I = np.abs(E) ** 2
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    j, i = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(x - x[i], x - x[j])
    r = np.sqrt(X ** 2 + Y ** 2)
    m = r <= crop_m
    return float(np.sqrt((I * r ** 2)[m].sum() / I[m].sum()))


def test_h1_slant_no_longer_cancels_spherical_aberration():
    """The SA-cancellation tripwire: on the f/5 singlet the slant screen
    must NOT produce a near-diffraction-limited spot (the bug's 3.6 um),
    and the obliquity term ADDS aberration relative to the paraxial screen
    (moves the analytic model TOWARD the 65 um oracle), so
    r2m(slant) >= r2m(paraxial).

    SAMPLING (updated 2026-09-12).  This ran at N = 2048 / dx = 12 um, which
    is a factor 2 coarser than the exit wavefront's Nyquist pitch
    ``lambda / (2 NA_exit) = 6.55 um`` for this f/5 element -- the windowed
    r2m aliases LOW there, and the PARAXIAL arm reads 5.53 um on the same
    grid.  The 10 um bar was therefore measuring the sampling, not the screen.
    Moved to N = 4096 / dx = 6 um, which is Nyquist-compliant: paraxial
    25.26 um, slant 26.76 um.  (r2m keeps rising to 40.55 / 43.06 um at
    dx = 3 um -- see the module docstring -- but the CANCELLATION signature
    this test guards is absent at both.)
    """
    N, dx = 4096, 6e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (5e-3) ** 2).astype(np.complex128)
    presc = _singlet_f5()
    z = 49.163e-3
    r2m = {}
    for name, kw in (('paraxial', {}), ('slant', {'slant_correction': True})):
        E_exit = la.apply_real_lens(E0, prescription=presc, wavelength=_WL,
                                    dx=dx, **kw)
        r2m[name] = _r2m_at_image(E_exit, dx, z)
    assert r2m['slant'] > 10e-6, (
        f"slant r2m={r2m['slant']*1e6:.2f} um -- the H1 SA-cancellation "
        f"signature (bug gave 3.6 um vs the 65 um dual-oracle truth)")
    assert r2m['slant'] >= r2m['paraxial'] * 0.98, (
        f"correct-signed obliquity must ADD |SA| vs the paraxial screen: "
        f"slant {r2m['slant']*1e6:.2f} vs paraxial {r2m['paraxial']*1e6:.2f}")


def test_h1_slant_moves_toward_the_oracle_without_overshooting():
    """The DIRECTION and the MAGNITUDE of the correction on this symmetric
    biconvex, bracketed on both sides.

    REPLACES an "oracle window [40, 90] um" pin that was taken at dx = 6 um
    and labelled "converged sampling".  It is not converged: the same
    measurement reads 25.26 um (paraxial) / 26.76 um (slant) at dx = 6 um and
    40.55 / 43.06 um at dx = 3 um and 1.50 um.  A window that a Nyquist-
    compliant run misses by 13 um is measuring the grid.

    DERIVATION OF THE BARS.  Both arms run on the SAME grid, so the sampling
    bias is common-mode and the RATIO is the stable quantity.  Measured
    slant/paraxial: 1.059 at dx = 6 um, 1.062 at dx = 3 um, 1.062 at
    dx = 1.5 um -- converged to three digits two samplings before the absolute
    number is.  The two failure modes it brackets:

    * a return of the pre-v5.25.0 ``n*sag/cos`` form CANCELS the pupil SA and
      drives the ratio far BELOW 1 (the 3.6 um spot);
    * a return of the v5.25.0 normal-referenced form, whose obliquity term has
      the wrong sign and 1.94x the magnitude, drives it to 50.31/25.26 = 1.992
      at dx = 6 um and 76.70/40.55 = 1.891 at dx = 3 um.

    The bar ``1.0 <= ratio < 1.5`` sits 0.06 above the measured value on the
    low side and 0.39 below the nearest defect on the high side.  The absolute
    dual-oracle comparison is NOT asserted here: at converged sampling this
    screen reads 43.1 um against the 65 um truth, an under-correction whose
    cause is the collimated-input assumption at the second surface, not the
    facet coefficient (module docstring).
    """
    N, dx = 4096, 6e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (5e-3) ** 2).astype(np.complex128)
    r2m = {}
    for name, kw in (('paraxial', {}), ('slant', {'slant_correction': True})):
        r2m[name] = _r2m_at_image(
            la.apply_real_lens(E0, prescription=_singlet_f5(),
                               wavelength=_WL, dx=dx, **kw),
            dx, 49.163e-3)
    ratio = r2m['slant'] / r2m['paraxial']
    assert 1.0 <= ratio < 1.5, (
        f"slant/paraxial r2m = {ratio:.3f} ({r2m['slant']*1e6:.2f} vs "
        f"{r2m['paraxial']*1e6:.2f} um).  Below 1 is the pre-v5.25.0 "
        f"SA-cancellation; 1.9-2.0 is the v5.25.0 normal-referenced "
        f"over-correction.")
    assert r2m['slant'] > 10e-6, (
        f"slant r2m={r2m['slant']*1e6:.2f} um -- the SA-cancellation "
        f"signature (the old bug gave 3.6 um against a 65 um truth)")


def test_h1_slant_benign_regime_stays_close_to_paraxial():
    """Benign f/50-class regime (w0=0.5 mm through the same singlet --
    validated 4-digit-exact vs Zemax): the obliquity correction is a
    SMALL perturbation there; slant must stay within a few percent of
    paraxial and must not regress the oracle agreement."""
    N, dx = 1024, 8e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (0.5e-3) ** 2).astype(np.complex128)
    presc = _singlet_f5()
    z = 49.163e-3
    r2m = {}
    for name, kw in (('paraxial', {}), ('slant', {'slant_correction': True})):
        E_exit = la.apply_real_lens(E0, prescription=presc, wavelength=_WL,
                                    dx=dx, **kw)
        r2m[name] = _r2m_at_image(E_exit, dx, z)
    assert abs(r2m['slant'] - r2m['paraxial']) / r2m['paraxial'] < 0.05, (
        f"benign regime: slant {r2m['slant']*1e6:.3f} um must be a small "
        f"perturbation of paraxial {r2m['paraxial']*1e6:.3f} um")
    # oracle: ZOS POP 29.98 um for this exact configuration
    assert abs(r2m['slant'] - 29.98e-6) / 29.98e-6 < 0.10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
