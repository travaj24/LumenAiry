"""VERIFY-WP-B7c's driver for the shared fold oracle.

``validation/oracles/caustic_fold_truth.py`` is the lumenairy-free,
energy-correct DIRECT Rayleigh-Sommerfeld ring integral over an exact
meridional conic trace that WP-B7b, VERIFY-B7b and WP-B7c all scored against.
This module drives it for the VERIFY fixtures and shares no code with WP-B7c's
own ``validation/probe_multibranch_zeta/oracle.py``: the Sellmeier
coefficients are typed HERE from the Schott catalogue (so the refractive index
the oracle traces never comes from the library under test), the doublet's
buried cemented interface is handled, and the delta against
``lumenairy.glass.get_glass_index`` is reported as a control, never used.

Nothing here imports lumenairy except that control.
"""
# ruff: noqa: E402, I001
from __future__ import annotations

import os
import sys

import numpy as np

_ORACLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'oracles')
if _ORACLES not in sys.path:
    sys.path.insert(0, _ORACLES)

import caustic_fold_truth as cft            # noqa: E402

# Schott catalogue Sellmeier coefficients (B1..B3; C1..C3 in um^2), typed from
# the catalogue for the four glasses these fixtures use.
SELLMEIER = {
    'N-LASF9': ((2.00029547, 0.298926886, 1.80691843),
                (0.0121426017, 0.0538736236, 156.530829)),
    'N-BK7':   ((1.03961212, 0.231792344, 1.01046945),
                (0.00600069867, 0.0200179144, 103.560653)),
    'N-SF6':   ((1.77931763, 0.338149866, 2.08734474),
                (0.0133714182, 0.0617533621, 174.017590)),
    'N-SK16':  ((1.34317774, 0.241144399, 0.994317969),
                (0.00704687339, 0.0229005000, 92.7508526)),
}


def index_of(glass, wavelength_m):
    B, C = SELLMEIER[glass]
    wl2 = (wavelength_m * 1e6) ** 2
    n2 = 1.0
    for b, c in zip(B, C):
        n2 += b * wl2 / (wl2 - c)
    return float(np.sqrt(n2))


def index_control(glass, wavelength_m):
    """Delta against the library's own dispersion -- reported, never used."""
    from lumenairy.glass import get_glass_index
    return abs(index_of(glass, wavelength_m)
               - float(get_glass_index(glass, wavelength_m)))


def job_for(fx, z, n_fan=6000):
    """The oracle's job dict for fixture ``fx`` at output plane ``z`` [m].

    Handles any surface count, so the cemented doublet's buried interface is
    just another row whose ``index`` is the glass AFTER it.
    """
    presc = fx['prescription']
    wl = fx['wavelength']
    surfs = []
    for s in presc['surfaces']:
        gl = s['glass_after']
        n = 1.0 if gl == 'air' else index_of(gl, wl)
        R = s['radius']
        surfs.append({'radius_mm': (0.0 if not np.isfinite(R) else R * 1e3),
                      'thickness_mm': s['thickness'] * 1e3,
                      'index': ('air' if gl == 'air' else n),
                      'conic': 0.0})
    surfs[-1]['thickness_mm'] = z * 1e3
    return {'wavelength_um': wl * 1e6, 'surfaces': surfs,
            'aperture_mm': float(presc['aperture_diameter']) * 1e3 * 0.98,
            'input': {'w0_mm': fx['w0'] * 1e3},
            'n_fan': int(n_fan)}


def oracle_field(fx, z, n_fan=6000, n_rho=2400):
    """(N, N) complex reference field on the fixture's own grid."""
    job = job_for(fx, z, n_fan=n_fan)
    h, ys, opl, amp, z_exit, ex = cft.build_exit_field(job, n_fan=n_fan)
    N, dx = fx['N'], fx['dx']
    rho_max = 0.5 * N * dx * np.sqrt(2.0) * 1.001
    rho = np.linspace(0.0, rho_max, n_rho)
    E_rho = cft._rs_integral(h, ys, opl, amp, z, fx['wavelength'], rho)
    return cft.radial_to_2d(rho, E_rho, N, dx), rho, E_rho, ex


def fidelity(Ea, Eb):
    a = np.asarray(Ea).ravel()
    b = np.asarray(Eb).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(abs(np.vdot(a, b)) / (na * nb))


def power(E, dx):
    return float(np.sum(np.abs(np.asarray(E)) ** 2)) * dx * dx
