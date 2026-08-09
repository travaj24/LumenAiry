"""Hammer-audit H1 (2026-07-19): slant_correction obliquity OPD sign.

The historical ``slant_correction=True`` OPD divided the sag by the local
cosines (``n*sag/cos`` -- the geometric ray path-length through a slab)
where the wavefront OPD of a locally-tilted refracting facet is
``(n2*cos_tt - n1*cos_ti)*sag`` -- cosines in the NUMERATOR (the
plane-parallel-plate result).  The inversion sign-flips the leading
obliquity (spherical-aberration) term; on a symmetric biconvex f/5
singlet the two wrong-signed surface corrections CANCELLED the pupil SA,
producing an impossible near-diffraction-limited 3.6 um spot where the
dual oracles (Zemax POP via ZOS-API + an independent exact-raytrace
Debye/Huygens integral, 2026-07-19 hammer campaign) both give ~65 um.

Oracle provenance (f/5 case: R=+/-51.68 mm, t=5 mm, n=1.5168,
lambda=1.31 um, Gaussian w0=5 mm, image 49.163 mm behind the exit
vertex): ZOS POP r2m=64.7 um, Debye r2m=64.98 um, EE80=55.2 um;
corrected-slant measured 50.3 um, plain paraxial-screen 25.3 um at
converged sampling.  A near-cancellation (r2m ~ 3.6 um) is physically
impossible for ~8 waves of SA at ANY plane.
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
    and the correct-signed obliquity term ADDS aberration relative to
    the paraxial screen (moves the analytic model TOWARD the 65 um
    oracle), so r2m(slant) >= r2m(paraxial)."""
    N, dx = 2048, 12e-6
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


def test_h1_slant_lands_in_oracle_window_at_converged_sampling():
    """Oracle-window pin at converged sampling (N=4096, dx=6 um): the
    corrected slant screen measured 50.3 um against the 65 um dual-oracle
    truth; accept [40, 90] um -- excludes BOTH the 3.6 um bug and the
    25 um paraxial plateau."""
    N, dx = 4096, 6e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (5e-3) ** 2).astype(np.complex128)
    E_exit = la.apply_real_lens(E0, prescription=_singlet_f5(),
                                wavelength=_WL, dx=dx,
                                slant_correction=True)
    r2m = _r2m_at_image(E_exit, dx, 49.163e-3)
    assert 40e-6 < r2m < 90e-6, (
        f"slant r2m={r2m*1e6:.2f} um outside the oracle window [40, 90] "
        f"(dual-oracle 65 um; bug 3.6 um; paraxial 25 um)")


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
