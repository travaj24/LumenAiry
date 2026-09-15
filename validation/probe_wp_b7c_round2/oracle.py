"""WP-B7c round 2's driver for the shared fold oracle, plus the EXACT
azimuthal quadrature that removes its NA ceiling.

``validation/oracles/caustic_fold_truth.py`` is the lumenairy-free, direct
Rayleigh-Sommerfeld ring integral WP-B7b, VERIFY-B7b, WP-B7c and
VERIFY-WP-B7c all scored against.  Its azimuthal integral is taken in the
Debye ``J0`` form,

    INT_0^2pi dphi (z/R) exp(i k R) / R  ->  2 pi J0(k y rho / R0)
                                             (z/R0) exp(i k R0) / R0,

which expands ``R = sqrt(z^2 + y^2 + rho^2 - 2 y rho cos phi)`` about
``R0 = sqrt(z^2 + y^2 + rho^2)`` and keeps only the linear term in the phase.
The neglected term is ``O(k (y rho)^2 / (2 R0^3))``; VERIFY-WP-B7c's D5
measured it at 2.76 rad at f/1.2 and recorded that the oracle therefore has an
unstated NA ceiling.

``phi='exact'`` here integrates ``phi`` NUMERICALLY instead -- the integrand is
smooth and 2pi-periodic, so the uniform trapezoid converges spectrally and a
few dozen nodes settle it.  Nothing else changes: the same exit field, the
same measure, the same prefactors.  ``phi='debye'`` calls the shared oracle's
own ``_rs_integral`` verbatim, so the two modes differ in exactly one term and
their difference IS the ceiling.

Sellmeier coefficients are typed here from the Schott catalogue, so the index
the oracle traces never comes from the library under test; the delta against
``lumenairy.glass.get_glass_index`` is reported as a control and never used.
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

SELLMEIER = {
    'N-LASF9': ((2.00029547, 0.298926886, 1.80691843),
                (0.0121426017, 0.0538736236, 156.530829)),
    'N-BK7':   ((1.03961212, 0.231792344, 1.01046945),
                (0.00600069867, 0.0200179144, 103.560653)),
    'N-SF6':   ((1.77931763, 0.338149866, 2.08734474),
                (0.0133714182, 0.0617533621, 174.017590)),
    'N-SK16':  ((1.34317774, 0.241144399, 0.994317969),
                (0.00704687339, 0.0229005000, 92.7508526)),
    'N-LAK22': ((1.14229781, 0.535138441, 1.04088385),
                (0.00585778594, 0.0198546147, 100.834017)),
    'N-BAF10': ((1.5851495, 0.143559385, 1.08521269),
                (0.00926681282, 0.0424489805, 105.613573)),
    'N-SF11':  ((1.73759695, 0.313747346, 1.89878101),
                (0.013188707, 0.0623068142, 155.23629)),
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

    Any surface count, any glass/air sequence (so an AIR GAP between two
    elements is just two more rows) and the surface ``conic`` if the
    prescription carries one.
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
                      'conic': float(s.get('conic', 0.0) or 0.0)})
    surfs[-1]['thickness_mm'] = z * 1e3
    return {'wavelength_um': wl * 1e6, 'surfaces': surfs,
            'aperture_mm': float(presc['aperture_diameter']) * 1e3 * 0.98,
            'input': {'w0_mm': fx['w0'] * 1e3},
            'n_fan': int(n_fan)}


# --------------------------------------------------------------------------
# the azimuthal integral, exactly
# --------------------------------------------------------------------------
#: Working-set budget for one block of the exact azimuthal quadrature, in
#: array entries.  The integrand is built as (n_rho_block, n_fan, n_phi); at
#: 8e6 entries each float64 temporary is 64 MB.
_EXACT_CHUNK_ENTRIES = 8_000_000


def _n_phi_for(b_max, safety=1.5, lo=64, hi=16384):
    """Azimuthal nodes needed for the midpoint rule on ``exp(-i b cos phi)``.

    The midpoint/trapezoid rule on a period aliases that integrand onto
    ``sum_m J_{m n_phi}(b)``, so the error is ``~2 |J_{n_phi}(b)|`` -- which
    is negligible as soon as ``n_phi`` clears ``b`` by a few ``b^{1/3}``
    (Bessel functions fall off super-exponentially past their turning point).
    ``safety`` multiplies that requirement and is the knob every measurement
    taken through this path reports a doubling of.
    """
    need = safety * (float(b_max) + 6.0 * max(float(b_max), 1.0) ** (1.0 / 3.0))
    n = int(lo)
    while n < need and n < hi:
        n *= 2
    return min(n, int(hi))


def rs_integral_exact(ys, opl, amp, zrel, wl, rho, phi_safety=1.5,
                      n_phi_min=64, n_phi_max=16384, return_n_phi=False):
    """The same direct Rayleigh-Sommerfeld ring integral as
    ``caustic_fold_truth._rs_integral``, with the azimuthal integral taken
    NUMERICALLY rather than in the Debye ``J0`` form.

    ``E(rho) = (1/(i lambda)) INT y dy E_exit(y)
    INT_0^2pi dphi (z/R) exp(i k R) / R``, with
    ``R = sqrt(z^2 + y^2 + rho^2 - 2 y rho cos phi)``.

    The azimuthal integrand oscillates like ``exp(-i b cos phi)`` with
    ``b = k y rho / R0``, and ``b`` runs from 0 on the axis to ~1000 rad at
    the corner of a 512-pixel grid -- which is exactly why the shared oracle
    does this integral in closed form and pays a ``J0`` truncation for it.
    The node count is therefore chosen PER RHO BLOCK from that block's own
    ``b`` (see :func:`_n_phi_for`), not fixed: a uniform ``n_phi`` is either
    unconverged at the rim or a hundred times more work than the core needs.

    Vectorised over the exit fan AND the azimuth, blocked over ``rho``
    against :data:`_EXACT_CHUNK_ENTRIES`.
    """
    k = 2.0 * np.pi / wl
    order = np.argsort(ys)
    ys_s, amp_s, opl_s = ys[order], amp[order], opl[order]
    Es = amp_s * np.exp(1j * k * opl_s)
    dys = np.gradient(ys_s)
    ring = Es * 2.0 * np.pi * ys_s * dys
    pref = 1.0 / (1j * wl)
    rho = np.asarray(rho, dtype=float)
    out = np.zeros(rho.size, dtype=complex)
    y2 = (ys_s ** 2)[None, :, None]
    y1 = ys_s[None, :, None]
    y_max = float(np.nanmax(np.abs(ys_s)))
    n_phi_used = np.zeros(rho.size, dtype=np.int64)

    lo = 0
    while lo < rho.size:
        # grow a block until either the entry budget or a doubling of the
        # required node count stops it
        r_lo = rho[lo]
        n_phi = _n_phi_for(k * y_max * r_lo
                           / np.sqrt(zrel ** 2 + y_max ** 2 + r_lo ** 2),
                           safety=phi_safety, lo=n_phi_min, hi=n_phi_max)
        hi = lo
        while hi < rho.size:
            r = rho[hi]
            need = _n_phi_for(k * y_max * r
                              / np.sqrt(zrel ** 2 + y_max ** 2 + r ** 2),
                              safety=phi_safety, lo=n_phi_min, hi=n_phi_max)
            if need > n_phi:
                break
            if (hi - lo + 1) * ys_s.size * n_phi > _EXACT_CHUNK_ENTRIES:
                break
            hi += 1
        hi = max(hi, lo + 1)
        phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
        cphi = np.cos(phi)[None, None, :]
        r = rho[lo:hi][:, None, None]
        R2 = zrel ** 2 + r * r + y2 - (2.0 * r) * y1 * cphi
        R = np.sqrt(R2)
        kR = k * R
        re = np.cos(kR)
        np.multiply(re, zrel / R2, out=re)
        im = np.sin(kR)
        np.multiply(im, zrel / R2, out=im)
        acc = (re.sum(axis=2) + 1j * im.sum(axis=2)) / n_phi
        out[lo:hi] = pref * (acc @ ring)
        n_phi_used[lo:hi] = n_phi
        del R2, R, kR, re, im, acc
        lo = hi
    if return_n_phi:
        return out, n_phi_used
    return out


def debye_phase_error(ys, zrel, wl, rho_ref):
    """The term the ``J0`` form drops, in radians, at the radius the light
    occupies: ``k (y rho)^2 / (2 R0^3)`` maximised over the exit fan."""
    k = 2.0 * np.pi / wl
    R0 = np.sqrt(zrel ** 2 + ys ** 2 + rho_ref ** 2)
    return float(np.nanmax(k * (ys * rho_ref) ** 2 / (2.0 * R0 ** 3)))


def rho_grid(fx, n_inner=1600, n_outer=400, r_split=None):
    """Radial sample points: dense where a near-focus field lives, coarse out
    to the grid corner.  A single uniform grid at the inner density would cost
    the exact quadrature an order of magnitude for samples that carry no
    structure."""
    N, dx = fx['N'], fx['dx']
    rho_max = 0.5 * N * dx * np.sqrt(2.0) * 1.001
    r_split = (90e-6 if r_split is None else float(r_split))
    r_split = min(r_split, 0.5 * rho_max)
    inner = np.linspace(0.0, r_split, n_inner, endpoint=False)
    outer = np.linspace(r_split, rho_max, n_outer)
    return np.concatenate([inner, outer])


def core_radius(rho, E_rho, keep=0.9995):
    """The radius holding ``keep`` of the radial field's energy."""
    w = np.abs(E_rho) ** 2 * rho
    c = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1])
                                         * np.diff(rho))])
    if c[-1] <= 0:
        return float(rho[-1])
    return float(np.interp(keep * c[-1], c, rho))


def oracle_field(fx, z, n_fan=6000, n_rho=2400, phi='debye', n_phi=1.5,
                 rho=None, r_core=None, return_detail=False):
    """(N, N) complex reference field on the fixture's own grid.

    ``phi='debye'`` is the shared oracle verbatim (uniform ``rho``, the ``J0``
    form).

    ``phi='exact'`` replaces the ``J0`` form by the numerically integrated
    azimuth INSIDE a core radius and keeps the ``J0`` form outside it.  The
    split is not a convenience: the azimuthal integrand oscillates like
    ``exp(-i k y rho / R0 cos phi)``, whose argument reaches ~1000 rad at the
    corner of a 512-pixel grid, so an exact quadrature that reached the rim
    would need ~1500 azimuthal nodes there for a region carrying <0.05 % of
    the energy.  ``r_core`` defaults to the radius holding 99.95 % of the
    ``J0`` field's energy, and both the radius and that fraction are
    REPORTED, so the substitution is visible rather than assumed.
    """
    job = job_for(fx, z, n_fan=n_fan)
    h, ys, opl, amp, z_exit, ex = cft.build_exit_field(job, n_fan=n_fan)
    N, dx = fx['N'], fx['dx']
    if rho is None:
        if phi == 'exact':
            rho = rho_grid(fx)
        else:
            rho_max = 0.5 * N * dx * np.sqrt(2.0) * 1.001
            rho = np.linspace(0.0, rho_max, n_rho)
    E_deb = cft._rs_integral(h, ys, opl, amp, z, fx['wavelength'], rho)
    detail = {'phi': phi}
    if phi == 'exact':
        rc = core_radius(rho, E_deb) if r_core is None else float(r_core)
        rc = max(rc, 4.0 * dx)
        core = rho <= rc
        E_rho = E_deb.copy()
        E_core, n_used = rs_integral_exact(
            ys, opl, amp, z, fx['wavelength'], rho[core],
            phi_safety=n_phi, return_n_phi=True)
        E_rho[core] = E_core
        w = np.abs(E_deb) ** 2 * rho
        tot = float(np.trapezoid(w, rho))
        detail.update(
            r_core_um=rc * 1e6, n_core=int(core.sum()),
            core_energy_fraction=(float(np.trapezoid(w[core], rho[core]))
                                  / tot if tot > 0 else None),
            n_phi_min=int(n_used.min()), n_phi_max=int(n_used.max()),
            core_relL2_vs_debye=float(
                np.linalg.norm(E_core - E_deb[core])
                / max(np.linalg.norm(E_core), 1e-300)))
    else:
        E_rho = E_deb
    E2d = cft.radial_to_2d(rho, E_rho, N, dx)
    if return_detail:
        return E2d, rho, E_rho, ex, detail
    return E2d, rho, E_rho, ex


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


def rms_radius(rho, E_rho):
    w = np.abs(E_rho) ** 2 * rho
    tot = np.trapezoid(w, rho)
    if tot <= 0:
        return float('nan')
    return float(np.sqrt(np.trapezoid(w * rho ** 2, rho) / tot))
