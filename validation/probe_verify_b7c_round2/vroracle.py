"""VERIFY-WP-B7c round 2 -- the INDEPENDENT verifier's own oracle.

Shares no code with ``validation/oracles/caustic_fold_truth.py`` nor with
either builder probe (``probe_verify_b7c/oracle.py``,
``probe_wp_b7c_round2/oracle.py``).  Everything here is written from the
physics:

* Sellmeier coefficients typed HERE from the Schott catalogue; the delta
  against ``lumenairy.glass.get_glass_index`` is reported as a CONTROL, never
  used, so a typo shows up as a control failure rather than as a silent
  oracle bias;
* an exact sequential meridional CONIC ray trace (Newton intersection, vector
  Snell), written here;
* THREE independent propagators of the traced exit field:
    - ``rs_j0``      the Debye ``J0`` azimuthal form (the shared oracle's),
                     re-implemented here so the comparison is like for like;
    - ``rs_exact``   the EXACT azimuthal quadrature, midpoint rule on
                     ``phi in (0, pi)`` with the node count derived per rho
                     from that rho's own oscillation rate;
    - ``asm_field``  a band-limited ANGULAR-SPECTRUM propagation of the same
                     exit field on a refined 2-D grid.  The angular spectrum
                     with the exact kernel ``exp(i z sqrt(k^2 - kx^2 - ky^2))``
                     is an exact solution of the scalar Helmholtz equation for
                     the given boundary field -- it makes NO Debye, paraxial or
                     azimuthal approximation -- so it is the arbiter of the
                     ``J0`` question from a completely different direction.

Nothing here imports lumenairy except the index control.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Schott catalogue Sellmeier coefficients, typed here.
# n^2 - 1 = sum_i B_i lam^2 / (lam^2 - C_i),  lam in um, C_i in um^2.
# ---------------------------------------------------------------------------
SELLMEIER = {
    'N-BK7':   ((1.03961212, 0.231792344, 1.01046945),
                (0.00600069867, 0.0200179144, 103.560653)),
    'N-BAK4':  ((1.28834642, 0.132817724, 0.945395373),
                (0.00779980626, 0.0315631177, 105.965875)),
    'N-F2':    ((1.39757037, 0.159201403, 1.26865430),
                (0.00995906143, 0.0546931752, 119.248346)),
    'N-SF10':  ((1.62153902, 0.256287842, 1.64447552),
                (0.0122241457, 0.0595736775, 147.468793)),
    'N-SF11':  ((1.73759695, 0.313747346, 1.89878101),
                (0.013188707, 0.0623068142, 155.23629)),
    'N-LASF9': ((2.00029547, 0.298926886, 1.80691843),
                (0.0121426017, 0.0538736236, 156.530829)),
    'N-BAF10': ((1.5851495, 0.143559385, 1.08521269),
                (0.00926681282, 0.0424489805, 105.613573)),
    'N-SK16':  ((1.34317774, 0.241144399, 0.994317969),
                (0.00704687339, 0.0229005000, 92.7508526)),
    'N-SF6':   ((1.77931763, 0.338149866, 2.08734474),
                (0.0133714182, 0.0617533621, 174.017590)),
    'N-LAK22': ((1.14229781, 0.535138441, 1.04088385),
                (0.00585778594, 0.0198546147, 100.834017)),
}


def index_of(glass, wavelength_m):
    """Refractive index from THIS module's own coefficients."""
    if glass in (None, 'air', 'AIR', 'Air'):
        return 1.0
    B, C = SELLMEIER[glass]
    l2 = (float(wavelength_m) * 1e6) ** 2
    n2 = 1.0
    for b, c in zip(B, C):
        n2 += b * l2 / (l2 - c)
    return float(np.sqrt(n2))


def index_control(glass, wavelength_m):
    """|this module's n - the library's n|.  Reported, never used."""
    if glass in (None, 'air'):
        return 0.0
    from lumenairy.glass import get_glass_index
    return abs(index_of(glass, wavelength_m)
               - float(get_glass_index(glass, wavelength_m)))


# ---------------------------------------------------------------------------
# Exact sequential meridional conic ray trace.
# Coordinates: z along the axis, y transverse.  A surface with vertex at
# z = 0, curvature c = 1/R and conic kappa has sag
#     z = c y^2 / (1 + sqrt(1 - (1 + kappa) c^2 y^2)).
# ---------------------------------------------------------------------------
def _sag(y, c, kap):
    if c == 0.0:
        return np.zeros_like(np.asarray(y, dtype=float))
    y = np.asarray(y, dtype=float)
    disc = 1.0 - (1.0 + kap) * c * c * y * y
    disc = np.where(disc < 0.0, np.nan, disc)
    return c * y * y / (1.0 + np.sqrt(disc))


def _dsag(y, c, kap):
    """dz/dy of the conic sag (the surface's meridional slope)."""
    if c == 0.0:
        return np.zeros_like(np.asarray(y, dtype=float))
    y = np.asarray(y, dtype=float)
    disc = 1.0 - (1.0 + kap) * c * c * y * y
    disc = np.where(disc < 0.0, np.nan, disc)
    return c * y / np.sqrt(disc)


def _intersect(p, d, c, kap, n_iter=60):
    """Newton intersection of the ray ``p + t d`` with the conic at z = 0.

    ``p = (z, y)``, ``d = (dz, dy)`` a unit vector, ``dz > 0``.  Returns
    ``t`` or NaN.
    """
    pz, py = p
    dz, dy = d
    # start from the plane z = 0 crossing
    t = -pz / dz
    for _ in range(n_iter):
        y = py + t * dy
        z = pz + t * dz
        s = _sag(y, c, kap)
        if not np.isfinite(s):
            return np.nan
        f = z - s
        fp = dz - _dsag(y, c, kap) * dy
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
    """Trace a collimated meridional fan through ``surfaces``.

    ``surfaces`` is a list of dicts with ``radius`` [m] (``inf`` for a plane),
    ``conic``, ``thickness`` [m] to the NEXT vertex, ``glass_after``.
    Returns ``(y_exit, opl_exit, y_land, alive)`` at the plane
    ``z_stop_from_last`` past the LAST vertex, with ``y_exit``/``opl_exit``
    taken at the last vertex plane itself (the exit plane).
    """
    m = len(heights)
    y_ex = np.full(m, np.nan)
    opl_ex = np.full(m, np.nan)
    y_land = np.full(m, np.nan)
    ns = [1.0] + [index_of(s.get('glass_after'), wavelength) for s in surfaces]
    for i, h in enumerate(heights):
        p = np.array([0.0, float(h)])          # (z, y) just before surface 0
        p[0] = -1e-3                            # start well before the vertex
        d = np.array([1.0, 0.0])
        opl = 0.0
        ok = True
        for j, s in enumerate(surfaces):
            R = s['radius']
            c = 0.0 if (R is None or not np.isfinite(R) or R == 0.0) else 1.0 / R
            kap = float(s.get('conic', 0.0) or 0.0)
            t = _intersect(p, d, c, kap)
            if not np.isfinite(t):
                ok = False
                break
            p = p + t * d
            opl += ns[j] * t
            semi = s.get('semi_diameter')
            if semi is not None and abs(p[1]) > float(semi):
                ok = False
                break
            # surface normal: the surface is F(z, y) = z - sag(y) = 0, so
            # grad F = (1, -dsag/dy); normalise.
            g = np.array([1.0, -float(_dsag(p[1], c, kap))])
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
            # advance to the next vertex plane
            p = np.array([p[0] - float(s['thickness']), p[1]])
        if not ok:
            continue
        # p[0] is now the axial coordinate relative to the LAST vertex plane
        # (the thickness of the last surface is 0 by construction of these
        # prescriptions, so p[0] is the distance BEHIND the exit plane).
        te = -p[0] / d[0]
        y_ex[i] = p[1] + te * d[1]
        opl_ex[i] = opl + ns[-1] * te
        ti = (z_stop_from_last - p[0]) / d[0]
        y_land[i] = p[1] + ti * d[1]
    return y_ex, opl_ex, y_land, np.isfinite(y_ex) & np.isfinite(opl_ex)


def exit_field(prescription, wavelength, w0, z, n_fan=6000, aper_frac=0.98):
    """Energy-correct exit-plane field of a collimated Gaussian.

    Returns ``(y, opl, amp, y_land, P_in)``, all on the surviving rays.
    """
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


# ---------------------------------------------------------------------------
# Propagators.
# ---------------------------------------------------------------------------
def _ring_weights(y, opl, amp, wl):
    order = np.argsort(y)
    ys = y[order]
    Es = amp[order] * np.exp(1j * (2.0 * np.pi / wl) * opl[order])
    dy = np.gradient(ys)
    return ys, Es, dy


def rs_j0(y, opl, amp, z, wl, rho):
    """The Debye ``J0`` azimuthal form (what the shared oracle does).

    ``R ~ R0 - y rho cos(phi) / R0`` inside the exponent, everything else at
    ``R0``, so the azimuthal integral closes as ``2 pi J0(k y rho / R0)``.
    """
    from scipy.special import j0 as _j0
    k = 2.0 * np.pi / wl
    ys, Es, dy = _ring_weights(y, opl, amp, wl)
    out = np.zeros(np.size(rho), dtype=complex)
    rho = np.asarray(rho, dtype=float)
    pref = 1.0 / (1j * wl)
    for i in range(ys.size):
        yi = ys[i]
        R0 = np.sqrt(z * z + yi * yi + rho * rho)
        out += (pref * Es[i] * _j0(k * yi * rho / R0) * (z / R0)
                * np.exp(1j * k * R0) / R0 * (2.0 * np.pi * yi * dy[i]))
    return out


def rs_exact(y, opl, amp, z, wl, rho, safety=6.0, n_phi_min=64,
             n_phi_cap=1 << 17):
    """EXACT azimuthal quadrature of the same Rayleigh-Sommerfeld integral.

    ``E(rho) = (1/(i lam)) INT 2 pi y dy E(y) <(z/R) e^{ikR}/R>_phi`` with
    ``R = sqrt(z^2 + y^2 + rho^2 - 2 y rho cos phi)`` kept EXACTLY -- the
    shared oracle's Debye step replaces ``R`` inside the exponent by
    ``R0 - y rho cos(phi) / R0`` and everything else by ``R0``, which closes
    the integral as ``2 pi J0(k y rho / R0)``.

    The integrand is even in ``phi``, so the midpoint rule runs on ``(0, pi)``.
    The node count is derived PER RHO from that rho's own oscillation rate
    ``b = k y_max rho / R0``: the midpoint rule on a ``cos``-modulated
    exponential aliases onto ``sum_m J_{m n}(b)``, so ``n`` must clear ``b``
    by a few ``b^(1/3)``.  ``safety`` is the knob the convergence control
    doubles.
    """
    k = 2.0 * np.pi / wl
    ys, Es, dy = _ring_weights(y, opl, amp, wl)
    src = (1.0 / (1j * wl)) * Es * (2.0 * np.pi * ys * dy)
    y2 = (ys * ys)[:, None]
    ysc = ys[:, None]
    rho = np.asarray(rho, dtype=float)
    out = np.zeros(rho.size, dtype=complex)
    ymax = float(ys.max())
    for i in range(rho.size):
        r = float(rho[i])
        if r == 0.0:
            R = np.sqrt(z * z + ys * ys)
            out[i] = np.sum(src * (z / R) * (np.cos(k * R)
                                             + 1j * np.sin(k * R)) / R)
            continue
        R0 = np.sqrt(z * z + ymax * ymax + r * r)
        b = k * ymax * r / R0
        n_phi = int(np.ceil(safety * (b + 10.0 * (b ** (1.0 / 3.0)) + 16.0)))
        n_phi = int(min(max(n_phi, n_phi_min), n_phi_cap))
        cph = np.cos((np.arange(n_phi) + 0.5) * (np.pi / n_phi))[None, :]
        R = np.sqrt(z * z + y2 + r * r - (2.0 * r) * ysc * cph)
        kr = k * R
        w = (z / (R * R))
        re = np.einsum('ij->i', w * np.cos(kr))
        im = np.einsum('ij->i', w * np.sin(kr))
        # midpoint on (0, pi) with weight pi/n_phi, doubled for (pi, 2pi),
        # i.e. the full azimuthal integral is 2 * (pi / n_phi) * sum
        f = 2.0 * (np.pi / n_phi)
        out[i] = np.sum(src * ((re + 1j * im) * f)) / (2.0 * np.pi)
    return out


def to_2d(rho, E_rho, N, dx):
    """Rotate a radial complex profile onto the (N, N) output grid."""
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X * X + Y * Y)
    re = np.interp(r, rho, E_rho.real, left=E_rho.real[0], right=0.0)
    im = np.interp(r, rho, E_rho.imag, left=E_rho.imag[0], right=0.0)
    return (re + 1j * im).astype(np.complex128)


def asm_field(y, opl, amp, z, wl, N, dx, refine=4, pad=2, Nf=None):
    """Band-limited ANGULAR-SPECTRUM propagation of the same exit field.

    Exact for the scalar Helmholtz equation given the boundary field: no
    Debye, no paraxial, no azimuthal approximation.  The exit field is laid on
    a grid of pitch ``dx / refine``; ``Nf`` (default ``pad * refine * N``) is
    the fine grid size, which must satisfy ``(Nf - refine * N)`` even so that
    the OUTPUT grid's pixel centres are an exact subset of the fine centres
    (coarse ``j`` -> fine ``refine * j + (Nf - refine * N) / 2``).
    """
    k = 2.0 * np.pi / wl
    if Nf is None:
        Nf = int(pad * refine * N)
    Nf = int(Nf)
    off2 = Nf - refine * N
    if off2 % 2 != 0:
        raise ValueError('Nf - refine*N must be even')
    dxf = dx / float(refine)
    xf = (np.arange(Nf) - Nf / 2.0) * dxf
    Xf, Yf = np.meshgrid(xf, xf)
    rf = np.sqrt(Xf * Xf + Yf * Yf)
    order = np.argsort(y)
    ys = y[order]
    Es = amp[order] * np.exp(1j * k * opl[order])
    re = np.interp(rf, ys, Es.real, left=Es.real[0], right=0.0)
    im = np.interp(rf, ys, Es.imag, left=Es.imag[0], right=0.0)
    E0 = (re + 1j * im).astype(np.complex128)
    E0[rf > ys.max()] = 0.0
    del Xf, Yf, rf, re, im
    fx = np.fft.fftfreq(Nf, d=dxf)
    FX2, FY2 = np.meshgrid(fx, fx)
    kt2 = (2.0 * np.pi) ** 2 * (FX2 * FX2 + FY2 * FY2)
    kz2 = k * k - kt2
    prop = np.where(kz2 > 0.0, np.exp(1j * np.sqrt(np.maximum(kz2, 0.0)) * z),
                    0.0)
    flim = 1.0 / (wl * np.sqrt((2.0 * z / (Nf * dxf)) ** 2 + 1.0))
    prop = np.where((np.abs(FX2) <= flim) & (np.abs(FY2) <= flim), prop, 0.0)
    del FX2, FY2, kt2, kz2
    Ez = np.fft.ifft2(np.fft.fft2(E0) * prop)
    del prop, E0
    idx = refine * np.arange(N) + off2 // 2
    return Ez[np.ix_(idx, idx)].astype(np.complex128)


# ---------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------
def fidelity(Ea, Eb):
    a = np.asarray(Ea).ravel()
    b = np.asarray(Eb).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(abs(np.vdot(a, b)) / (na * nb))


def rel_l2(Ea, Eb):
    a = np.asarray(Ea).ravel()
    b = np.asarray(Eb).ravel()
    nb = float(np.linalg.norm(b))
    if nb == 0.0:
        return float('inf')
    return float(np.linalg.norm(a - b) / nb)


def power(E, dx):
    return float(np.sum(np.abs(np.asarray(E)) ** 2)) * dx * dx


def core_radius(rho, E_rho, frac=0.9995):
    """Radius holding ``frac`` of the radial field's energy."""
    w = np.abs(E_rho) ** 2 * rho
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1])
                                           * np.diff(rho))])
    if cum[-1] <= 0.0:
        return float(rho[-1])
    return float(np.interp(frac * cum[-1], cum, rho))
