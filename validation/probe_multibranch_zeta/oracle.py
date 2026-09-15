"""Exact reference for the WP-B7c ladders -- the SAME oracle VERIFY-B7b used.

``validation/oracles/caustic_fold_truth.py`` is a lumenairy-free, energy-correct
DIRECT Rayleigh-Sommerfeld ring integral over an exact meridional conic trace.
This module only drives it: it turns a probe fixture into that oracle's job
schema, types the Schott Sellmeier coefficients HERE (so the refractive index
does not come from the library under test, with the delta against
``get_glass_index`` reported as a control), and rotates the radial answer onto
the library's own (N, N) grid.

Nothing here imports lumenairy except the control.
"""
# ruff: noqa: E402, I001  (sys.path is set up between the imports)
from __future__ import annotations

import os
import sys

import numpy as np

_ORACLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'oracles')
if _ORACLES not in sys.path:
    sys.path.insert(0, _ORACLES)

import caustic_fold_truth as cft            # noqa: E402

# Schott catalogue Sellmeier coefficients, typed here (B1..B3, C1..C3 [um^2]).
SELLMEIER = {
    'N-LAK22': ((1.14229781, 0.535138441, 1.04088385),
                (0.00585778594, 0.0198546147, 100.834017)),
    'N-BK7':   ((1.03961212, 0.231792344, 1.01046945),
                (0.00600069867, 0.0200179144, 103.560653)),
    'N-BAF10': ((1.5851495, 0.143559385, 1.08521269),
                (0.00926681282, 0.0424489805, 105.613573)),
    'N-SF11':  ((1.73759695, 0.313747346, 1.89878101),
                (0.013188707, 0.0623068142, 155.23629)),
}


def index_of(glass, wavelength_m):
    """Sellmeier index, independent of the library."""
    B, C = SELLMEIER[glass]
    wl2 = (wavelength_m * 1e6) ** 2
    n2 = 1.0
    for b, c in zip(B, C):
        n2 += b * wl2 / (wl2 - c)
    return float(np.sqrt(n2))


def index_control(glass, wavelength_m):
    """Delta against the library's own dispersion, for the report."""
    from lumenairy.glass import get_glass_index
    return abs(index_of(glass, wavelength_m)
               - float(get_glass_index(glass, wavelength_m)))


def job_for(fx, z, n_fan=6000):
    """The oracle's job dict for fixture ``fx`` at output plane ``z`` [m]."""
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
                      'conic': float(s.get('conic', 0.0) or 0.0)})
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
    """|<a,b>| / (|a| |b|) -- phase- and scale-invariant overlap."""
    a = np.asarray(Ea).ravel()
    b = np.asarray(Eb).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(abs(np.vdot(a, b)) / (na * nb))


def power(E, dx):
    return float(np.sum(np.abs(np.asarray(E)) ** 2)) * dx * dx
