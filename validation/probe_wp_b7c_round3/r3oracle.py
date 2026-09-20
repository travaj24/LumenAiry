"""WP-B7c round 3 -- the oracle, extended to ASPHERIC surfaces and used at
FULL RADIUS.

The three propagators (``rs_j0``, ``rs_exact``, ``asm_field``) and the scoring
helpers are taken verbatim from the round-2 VERIFIER's own oracle
(``validation/probe_verify_b7c_round2/vroracle.py``) -- independently written,
convergence-controlled there, and deliberately NOT re-derived here, so that a
round-3 number and a round-2 number differ by the measurement and not by a
third oracle's bias.  What this module adds is:

* an EVEN-ASPHERIC term in the sag and its slope, so the aspheric singlet
  this round adds can be traced exactly (round 2's optics were spherical or
  conic only, and ``vroracle`` has no aspheric term);
* ``exit_field`` through that aspheric-capable trace, with a CONTROL against
  ``vroracle.exit_field`` on a spherical prescription -- the two must agree to
  machine precision, so the extension cannot have perturbed the spherical
  path;
* ``exact_full_radius`` -- the exact azimuthal quadrature applied at EVERY
  radius out to the grid CORNER, which is what E6 says round 2's oracle-floor
  column was not: that column substituted the exact azimuth only inside the
  99.95 %-energy core, outside which the two fields are identical by
  construction and the dropped quadratic term is largest.

Nothing here imports lumenairy except the index control.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

_V2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'probe_verify_b7c_round2', 'vroracle.py')
_spec = importlib.util.spec_from_file_location('_b7c3_vroracle', _V2)
VOR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(VOR)

# re-exported verbatim: the propagators and the scoring
rs_j0 = VOR.rs_j0
rs_exact = VOR.rs_exact
asm_field = VOR.asm_field
to_2d = VOR.to_2d
fidelity = VOR.fidelity
rel_l2 = VOR.rel_l2
power = VOR.power
core_radius = VOR.core_radius
index_of = VOR.index_of
index_control = VOR.index_control
SELLMEIER = VOR.SELLMEIER


# ---------------------------------------------------------------------------
# Sag with an EVEN-ASPHERIC term.
#   z(h) = c h^2 / (1 + sqrt(1 - (1 + kappa) c^2 h^2)) + sum_p a_p h^p
# matching ``lumenairy.elements._lens_kernels.surface_sag_general`` (even
# powers only; that function evaluates ``h_sq ** (power // 2)`` and the
# library rejects odd powers at the entry point).
# ---------------------------------------------------------------------------
def _asph(y, coeffs):
    y = np.asarray(y, dtype=float)
    s = np.zeros_like(y, dtype=float)
    if not coeffs:
        return s
    for p, a in coeffs.items():
        s = s + float(a) * (y * y) ** (int(p) // 2)
    return s


def _dasph(y, coeffs):
    y = np.asarray(y, dtype=float)
    s = np.zeros_like(y, dtype=float)
    if not coeffs:
        return s
    for p, a in coeffs.items():
        p = int(p)
        s = s + float(a) * p * np.sign(y) * np.abs(y) ** (p - 1)
    return s


def _sag(y, c, kap, asph=None):
    return VOR._sag(y, c, kap) + _asph(y, asph)


def _dsag(y, c, kap, asph=None):
    return VOR._dsag(y, c, kap) + _dasph(y, asph)


def _intersect(p, d, c, kap, asph=None, n_iter=60):
    """Newton intersection of the ray ``p + t d`` with the surface at z = 0."""
    pz, py = p
    dz, dy = d
    t = -pz / dz
    for _ in range(n_iter):
        y = py + t * dy
        z = pz + t * dz
        s = float(_sag(y, c, kap, asph))
        if not np.isfinite(s):
            return np.nan
        f = z - s
        fp = dz - float(_dsag(y, c, kap, asph)) * dy
        if not np.isfinite(fp) or fp == 0.0:
            return np.nan
        step = f / fp
        t = t - step
        if abs(step) < 1e-16:
            break
    if not np.isfinite(t) or t <= 0.0:
        return np.nan
    return float(t)


def trace_fan(surfaces, wavelength, heights, z_stop_from_last):
    """Exact sequential meridional trace through conic + even-aspheric
    surfaces.  Same signature and return as ``vroracle.trace_fan``."""
    m = len(heights)
    y_ex = np.full(m, np.nan)
    opl_ex = np.full(m, np.nan)
    y_land = np.full(m, np.nan)
    ns = [1.0] + [index_of(s.get('glass_after'), wavelength) for s in surfaces]
    for i, h in enumerate(heights):
        p = np.array([-1e-3, float(h)])
        d = np.array([1.0, 0.0])
        opl = 0.0
        ok = True
        for j, s in enumerate(surfaces):
            R = s['radius']
            c = 0.0 if (R is None or not np.isfinite(R) or R == 0.0) else 1.0 / R
            kap = float(s.get('conic', 0.0) or 0.0)
            asph = s.get('aspheric_coeffs')
            t = _intersect(p, d, c, kap, asph)
            if not np.isfinite(t):
                ok = False
                break
            p = p + t * d
            opl += ns[j] * t
            semi = s.get('semi_diameter')
            if semi is not None and abs(p[1]) > float(semi):
                ok = False
                break
            g = np.array([1.0, -float(_dsag(p[1], c, kap, asph))])
            g = g / np.linalg.norm(g)
            n1, n2 = ns[j], ns[j + 1]
            cos_i = -float(np.dot(d, g))
            if cos_i < 0.0:
                g = -g
                cos_i = -cos_i
            eta = n1 / n2
            k2 = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
            if k2 < 0.0:
                ok = False
                break
            d = eta * d + (eta * cos_i - np.sqrt(k2)) * g
            d = d / np.linalg.norm(d)
            p = np.array([p[0] - float(s['thickness']), p[1]])
        if not ok:
            continue
        te = -p[0] / d[0]
        y_ex[i] = p[1] + te * d[1]
        opl_ex[i] = opl + ns[-1] * te
        ti = (z_stop_from_last - p[0]) / d[0]
        y_land[i] = p[1] + ti * d[1]
    return y_ex, opl_ex, y_land, np.isfinite(y_ex) & np.isfinite(opl_ex)


def exit_field(prescription, wavelength, w0, z, n_fan=6000, aper_frac=0.98):
    """Energy-correct exit-plane field of a collimated Gaussian, through the
    aspheric-capable trace.  Same return as ``vroracle.exit_field``."""
    surfs = prescription['surfaces']
    aper = prescription.get('aperture_diameter')
    r_edge = 0.5 * float(aper) * float(aper_frac)
    h = np.linspace(r_edge / n_fan, r_edge, n_fan)
    y_ex, opl_ex, y_land, alive = trace_fan(surfs, wavelength, h, float(z))
    h = h[alive]
    y = y_ex[alive]
    opl = opl_ex[alive]
    yl = y_land[alive]
    A = np.exp(-(h ** 2) / (w0 ** 2))
    J = np.abs(np.gradient(y, h))
    amp = A * np.sqrt(np.clip(h / np.clip(y * J, 1e-30, None), 0.0, None))
    P_in = 2.0 * np.pi * float(np.trapezoid(A ** 2 * h, h))
    return h, y, opl, amp, yl, P_in


def spherical_path_control(prescription, wavelength, w0, z, n_fan=2000):
    """|this trace - vroracle's| on a prescription with no aspheric term.

    The aspheric extension must be EXACTLY inert on a spherical / conic
    surface; this is the control that says so, reported rather than assumed.
    """
    a = exit_field(prescription, wavelength, w0, z, n_fan=n_fan)
    b = VOR.exit_field(prescription, wavelength, w0, z, n_fan=n_fan)
    out = {}
    for nm, i in (('y', 1), ('opl', 2), ('amp', 3), ('y_land', 4)):
        u, v = np.asarray(a[i]), np.asarray(b[i])
        if u.shape != v.shape:
            out[nm] = float('inf')
            continue
        sc = max(float(np.max(np.abs(v))), 1e-300)
        out[nm] = float(np.max(np.abs(u - v)) / sc)
    out['P_in'] = abs(a[5] - b[5]) / max(abs(b[5]), 1e-300)
    return out


def exact_full_radius(y, opl, amp, z, wl, N, dx, n_rho=320, safety=6.0):
    """The EXACT azimuthal quadrature at every radius out to the grid CORNER.

    E6: round 2's oracle-floor column substituted the exact azimuth only
    inside the 99.95 %-energy core and kept the Debye ``J0`` form outside,
    where the dropped quadratic term is LARGEST -- so its "Debye vs exact"
    relative L2 is identically zero over most of the grid by construction.
    This applies it everywhere.  Expensive (the node count grows with rho), so
    it is the cross-check on a SAMPLE of planes, not the scoring arm of the
    300-plane scan.
    """
    rho_max = 0.5 * N * dx * np.sqrt(2.0) * 1.001
    rho = np.linspace(0.0, rho_max, int(n_rho))
    E = rs_exact(y, opl, amp, z, wl, rho, safety=safety)
    return rho, E
