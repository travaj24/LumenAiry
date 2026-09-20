"""VERIFY-WP-B7c round 3 -- the independent verifier's OWN oracle.

Written for this verification.  It shares no code with
``validation/oracles/caustic_fold_truth.py``, with the builder probes
(``probe_verify_b7c/oracle.py``, ``probe_wp_b7c_round2/oracle.py``,
``probe_wp_b7c_round3/r3oracle.py``) nor with the round-2 verifier's
``probe_verify_b7c_round2/vroracle.py`` -- which matters here because WP-B7c
round 3 imported that last one VERBATIM, so re-using it would make this
verification a re-run of the builder's own arithmetic rather than an
independent measurement.

Every piece is derived here and DIFFERS in method from the module round 3
used, so that a shared mistake shows up as a disagreement rather than
cancelling:

===========================  ==============================  ==================
quantity                     round 3 (``r3oracle`` =         this module
                             ``vroracle``)
===========================  ==============================  ==================
surface intersection         scalar Newton on ``z - sag``,   VECTORISED damped
                             fixed 60 iterations, step       Newton with a
                             tolerance                       BISECTION bracket
                                                             fallback, residual
                                                             tolerance
refraction                   vector Snell                    ANGLE-form Snell
                                                             (asin of the
                                                             sine-law)
ray-map Jacobian ``dy/dh``   ``np.gradient`` (2nd order)     5-point central
                                                             differences (4th
                                                             order), 4th-order
                                                             one-sided at the
                                                             ends
radial quadrature            trapezoid via ``np.gradient``   composite SIMPSON
                             ring weights                    in the LAUNCH
                                                             height ``h``
azimuthal quadrature         MIDPOINT rule on (0, pi)        GAUSS-LEGENDRE on
                                                             (0, pi)
exit field onto the ASM      linear interpolation of the     CUBIC interpolation
grid                         field's REAL and IMAGINARY      of AMPLITUDE and
                             parts                           OPTICAL PATH, the
                                                             phase re-formed
                                                             after
aspheres                     conic only (round 3 added an    conic AND even
                             even-aspheric term to           aspheric from the
                             ``vroracle``)                   start
===========================  ==============================  ==================

The amplitude/phase interpolation is not only "different": interpolating the
REAL and IMAGINARY parts of a field whose phase turns by ~0.18 rad between
source samples commits a relative error of order ``(dphi)^2 / 8`` (~4e-3 at
NA 0.4), while the optical path and the amplitude are both smooth in ``h``.

Nothing here imports lumenairy except :func:`index_control`, which is
reported and never used.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Sellmeier coefficients, typed here from the Schott catalogue.
#   n^2 - 1 = sum_i B_i lam^2 / (lam^2 - C_i),   lam in um, C_i in um^2.
# The delta against ``lumenairy.glass.get_glass_index`` is a CONTROL: it is
# printed, never used, so a typo shows up as a control failure rather than as
# a silent oracle bias.
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

_AIR = (None, 'air', 'AIR', 'Air', '')


def index_of(glass, wavelength_m):
    """Refractive index from THIS module's own coefficients."""
    if glass in _AIR:
        return 1.0
    B, C = SELLMEIER[glass]
    lam2 = (float(wavelength_m) * 1e6) ** 2
    acc = 1.0
    for b, c in zip(B, C):
        acc += b * lam2 / (lam2 - c)
    return float(np.sqrt(acc))


def index_control(glass, wavelength_m):
    """``|this module's n - the library's n|``.  Reported, never used."""
    if glass in _AIR:
        return 0.0
    from lumenairy.glass import get_glass_index
    return abs(index_of(glass, wavelength_m)
               - float(get_glass_index(glass, wavelength_m)))


# ---------------------------------------------------------------------------
# Surfaces: conic + EVEN ASPHERIC sag and its meridional slope.
#   z(y) = c y^2 / (1 + sqrt(1 - (1 + k) c^2 y^2)) + sum_m a_m y^m
# ---------------------------------------------------------------------------
def _curv(surface):
    R = surface.get('radius')
    if R is None or R == 0.0 or not np.isfinite(R):
        return 0.0
    return 1.0 / float(R)


def _coeffs(surface):
    a = surface.get('aspheric_coeffs') or {}
    return tuple(sorted((int(m), float(v)) for m, v in a.items()))


def sag(y, c, k, aco=()):
    """Sag of the conic + even asphere at meridional height ``y``."""
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    if c != 0.0:
        disc = 1.0 - (1.0 + k) * c * c * y * y
        disc = np.where(disc > 0.0, disc, np.nan)
        out = c * y * y / (1.0 + np.sqrt(disc))
    for m, v in aco:
        out = out + v * y ** m
    return out


def dsag(y, c, k, aco=()):
    """``dz/dy`` of the same surface."""
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    if c != 0.0:
        disc = 1.0 - (1.0 + k) * c * c * y * y
        disc = np.where(disc > 0.0, disc, np.nan)
        out = c * y / np.sqrt(disc)
    for m, v in aco:
        out = out + m * v * y ** (m - 1)
    return out


# ---------------------------------------------------------------------------
# Vectorised damped-Newton intersection with a bisection fallback.
# ---------------------------------------------------------------------------
def _intersect(pz, py, dz, dy, c, k, aco, n_newton=80):
    """``t >= 0`` with ``(pz + t dz) == sag(py + t dy)``, elementwise.

    Damped Newton from the plane crossing; any entry whose residual has not
    come down is finished by BISECTION on a bracket grown outwards from the
    plane crossing.  Returns NaN where no root is found.
    """
    pz = np.asarray(pz, float)
    py = np.asarray(py, float)
    dz = np.asarray(dz, float)
    dy = np.asarray(dy, float)

    def F(t):
        y = py + t * dy
        with np.errstate(invalid='ignore'):
            return (pz + t * dz) - sag(y, c, k, aco)

    def Fp(t):
        y = py + t * dy
        with np.errstate(invalid='ignore'):
            return dz - dsag(y, c, k, aco) * dy

    t = -pz / dz                                    # the z = 0 plane crossing
    scale = np.maximum(np.abs(t), 1e-12)
    for _ in range(n_newton):
        f = F(t)
        fp = Fp(t)
        step = np.where(np.isfinite(fp) & (np.abs(fp) > 1e-14), f / fp, 0.0)
        # damping: never move more than a tenth of the current path in one go
        step = np.clip(step, -0.1 * scale, 0.1 * scale)
        t_new = t - step
        t = np.where(np.isfinite(t_new), t_new, t)
    res = np.abs(F(t))
    bad = ~np.isfinite(res) | (res > 1e-14 * scale) | (t <= 0.0)
    if np.any(bad):
        # bisection fallback on a bracket grown outwards from the plane
        # crossing, which is where the sag is zero
        lo = np.where(bad, 0.5 * (-pz / dz), t)
        hi = np.where(bad, 1.5 * (-pz / dz), t)
        for _ in range(8):
            same = np.sign(F(lo)) == np.sign(F(hi))
            if not np.any(same & bad):
                break
            lo = np.where(same & bad, lo * 0.5, lo)
            hi = np.where(same & bad, hi * 1.5, hi)
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            fm = F(mid)
            left = np.sign(fm) == np.sign(F(lo))
            lo = np.where(bad & left, mid, lo)
            hi = np.where(bad & ~left, mid, hi)
        t = np.where(bad, 0.5 * (lo + hi), t)
        res = np.abs(F(t))
        t = np.where(np.isfinite(res) & (res <= 1e-11 * scale) & (t > 0.0),
                     t, np.nan)
    return t


def _refract_angle(ang_in, ang_norm, n1, n2):
    """ANGLE-form Snell in the meridional plane.

    ``ang_in`` is the ray direction's angle from ``+z``, ``ang_norm`` the
    surface normal's.  ``sin(theta_t) = (n1/n2) sin(theta_i)`` with
    ``theta`` measured from the normal; the refracted direction's angle is
    ``ang_norm + theta_t``.  NaN on total internal reflection.
    """
    ti = ang_in - ang_norm
    s = (n1 / n2) * np.sin(ti)
    s = np.where(np.abs(s) <= 1.0, s, np.nan)
    return ang_norm + np.arcsin(s)


def trace_fan(surfaces, wavelength, heights, z_stop_from_last):
    """Trace a collimated meridional fan.

    Returns ``(y_exit, opl_exit, y_land, alive)`` -- heights and optical paths
    at the LAST vertex plane, and the landing height at ``z_stop_from_last``
    past it.  Vectorised over ``heights``.
    """
    h = np.asarray(heights, dtype=float)
    ns = [1.0] + [index_of(s.get('glass_after'), wavelength) for s in surfaces]
    pz = np.full(h.shape, -1.0e-3)
    py = h.copy()
    ang = np.zeros_like(h)                      # direction angle from +z
    opl = np.zeros_like(h)
    alive = np.ones(h.shape, dtype=bool)
    for j, s in enumerate(surfaces):
        c, k, aco = _curv(s), float(s.get('conic', 0.0) or 0.0), _coeffs(s)
        dz, dy = np.cos(ang), np.sin(ang)
        t = _intersect(pz, py, dz, dy, c, k, aco)
        alive &= np.isfinite(t)
        t = np.where(alive, t, 0.0)
        pz = pz + t * dz
        py = py + t * dy
        opl = opl + ns[j] * t
        semi = s.get('semi_diameter')
        if semi is not None:
            alive &= np.abs(py) <= float(semi) * (1.0 + 1e-12)
        # normal of F(z, y) = z - sag(y): grad = (1, -sag'), whose angle from
        # +z is atan2(-sag', 1)
        with np.errstate(invalid='ignore'):
            slope = dsag(py, c, k, aco)
        ang_n = np.arctan2(-slope, 1.0)
        ang = _refract_angle(ang, ang_n, ns[j], ns[j + 1])
        alive &= np.isfinite(ang)
        ang = np.where(alive, ang, 0.0)
        pz = pz - float(s['thickness'])          # next vertex to z = 0
    # ``pz`` is now the axial coordinate relative to the LAST vertex plane
    dz, dy = np.cos(ang), np.sin(ang)
    te = -pz / dz
    y_ex = py + te * dy
    opl_ex = opl + ns[-1] * te
    ti = (float(z_stop_from_last) - pz) / dz
    y_land = py + ti * dy
    alive &= np.isfinite(y_ex) & np.isfinite(opl_ex) & np.isfinite(y_land)
    nan = np.full(h.shape, np.nan)
    return (np.where(alive, y_ex, nan), np.where(alive, opl_ex, nan),
            np.where(alive, y_land, nan), alive)


# ---------------------------------------------------------------------------
# dy/dh at 4th order on a uniform h lattice.
# ---------------------------------------------------------------------------
def _d4(f, step):
    """4th-order derivative of ``f`` on a uniform lattice of pitch ``step``."""
    f = np.asarray(f, dtype=float)
    n = f.size
    out = np.empty(n, dtype=float)
    if n < 5:
        return np.gradient(f, step)
    out[2:-2] = (f[:-4] - 8.0 * f[1:-3] + 8.0 * f[3:-1] - f[4:]) / (12.0 * step)
    # 4th-order one-sided at the two ends
    out[0] = (-25 * f[0] + 48 * f[1] - 36 * f[2] + 16 * f[3] - 3 * f[4]) / (
        12.0 * step)
    out[1] = (-3 * f[0] - 10 * f[1] + 18 * f[2] - 6 * f[3] + f[4]) / (
        12.0 * step)
    out[-1] = (25 * f[-1] - 48 * f[-2] + 36 * f[-3] - 16 * f[-4]
               + 3 * f[-5]) / (12.0 * step)
    out[-2] = (3 * f[-1] + 10 * f[-2] - 18 * f[-3] + 6 * f[-4] - f[-5]) / (
        12.0 * step)
    return out


def exit_field(prescription, wavelength, w0, n_fan=6001, aper_frac=0.98):
    """Energy-correct exit-plane field of a collimated Gaussian.

    The launch heights are a UNIFORM lattice (Simpson needs one) with an ODD
    count, starting a half-step off the axis so the ``1/y`` of the ring
    Jacobian is never evaluated at zero.

    Returns a dict with ``h`` (launch heights), ``dh``, ``y`` (exit heights),
    ``opl``, ``amp``, ``P_in`` and ``alive``.
    """
    surfs = prescription['surfaces']
    r_edge = 0.5 * float(prescription['aperture_diameter']) * float(aper_frac)
    n_fan = int(n_fan) | 1                       # Simpson wants an odd count
    h = np.linspace(r_edge / n_fan, r_edge, n_fan)
    dh = float(h[1] - h[0])
    y, opl, _yl, alive = trace_fan(surfs, wavelength, h, 0.0)
    if not np.all(alive):
        # keep the lattice uniform: truncate to the longest alive prefix
        stop = int(np.argmin(alive)) if not alive[0] else int(np.argmin(alive))
        stop = stop if stop > 0 else 0
        if stop < 5:
            raise RuntimeError('the fan does not survive the prescription')
        stop = stop - ((stop + 1) % 2)           # keep it odd
        h, y, opl = h[:stop], y[:stop], opl[:stop]
    amp0 = np.exp(-(h ** 2) / (w0 ** 2))
    dydh = _d4(y, dh)
    with np.errstate(divide='ignore', invalid='ignore'):
        amp = amp0 * np.sqrt(np.abs(h / (y * dydh)))
    P_in = 2.0 * np.pi * _simpson(amp0 ** 2 * h, dh)
    return dict(h=h, dh=dh, y=y, opl=opl, amp=amp, dydh=dydh, P_in=P_in,
                n_fan=h.size)


def _simpson(f, step):
    """Composite Simpson on an ODD-length uniform lattice."""
    f = np.asarray(f, dtype=float)
    n = f.size
    if n % 2 == 0:                               # fall back on the last panel
        return _simpson(f[:-1], step) + 0.5 * step * (f[-1] + f[-2])
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return float(np.sum(w * f) * step / 3.0)


# ---------------------------------------------------------------------------
# Propagators.
# ---------------------------------------------------------------------------
def _simpson_weights(n, step):
    w = np.ones(n)
    if n % 2 == 1:
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        return w * (step / 3.0)
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0
    w[-1] = 0.0
    out = w * (step / 3.0)
    out[-1] += 0.5 * step
    out[-2] += 0.5 * step
    return out


def _ring_source(ef, wavelength):
    """``w_i``  such that ``sum_i w_i g(y_i)`` is ``INT g(y) E(y) 2 pi y dy``
    with the radial integral taken by SIMPSON in the launch height."""
    k = 2.0 * np.pi / float(wavelength)
    w = _simpson_weights(ef['h'].size, ef['dh'])
    phase = np.exp(1j * k * ef['opl'])
    return (ef['amp'] * phase * (2.0 * np.pi) * ef['y'] * ef['dydh'] * w,
            ef['y'])


def rs_exact(ef, z, wavelength, rho, safety=2.0, n_phi_min=48,
             n_phi_cap=1 << 16):
    """EXACT azimuthal quadrature of the Rayleigh-Sommerfeld integral.

    ``E(rho) = (1/(i lam)) INT 2 pi y dy E(y) <(z/R) e^{ikR} / R>_phi``, with

        ``R = sqrt(z^2 + y^2 + rho^2 - 2 y rho cos phi)``

    kept EXACTLY.  (The Debye ``J0`` form of the shared oracle replaces ``R``
    inside the exponent by ``R0 - y rho cos phi / R0`` and everything else by
    ``R0``, which closes the azimuth as ``2 pi J0(k y rho / R0)``.)

    The integrand is even in ``phi``, so the quadrature runs on ``(0, pi)`` and
    doubles.  Nodes are GAUSS-LEGENDRE, count derived per ``rho`` from that
    ``rho``'s own oscillation rate ``b = k y_max rho / R0``: Gauss-Legendre
    resolves ``e^{i b cos phi}`` once ``n`` clears ``b`` by a few ``b^(1/3)``.
    ``safety`` is the knob the convergence control doubles.
    """
    from scipy.special import roots_legendre
    k = 2.0 * np.pi / float(wavelength)
    src, ys = _ring_source(ef, wavelength)
    src = src / (1j * float(wavelength))
    y2 = (ys * ys)[:, None]
    ysc = ys[:, None]
    ymax = float(np.nanmax(np.abs(ys)))
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    out = np.zeros(rho.size, dtype=complex)
    cache = {}
    for i in range(rho.size):
        r = float(rho[i])
        if r == 0.0:
            R = np.sqrt(z * z + ys * ys)
            out[i] = np.sum(src * (z / R) * np.exp(1j * k * R) / R
                            * (2.0 * np.pi)) / (2.0 * np.pi)
            continue
        R0 = np.sqrt(z * z + ymax * ymax + r * r)
        b = k * ymax * r / R0
        n_phi = int(np.ceil(safety * (b + 8.0 * (b ** (1.0 / 3.0)) + 16.0)))
        n_phi = int(min(max(n_phi, n_phi_min), n_phi_cap))
        if n_phi not in cache:
            x, wgl = roots_legendre(n_phi)
            cache[n_phi] = (0.5 * np.pi * (x + 1.0), 0.5 * np.pi * wgl)
        ph, wph = cache[n_phi]
        cph = np.cos(ph)[None, :]
        R = np.sqrt(z * z + y2 + r * r - (2.0 * r) * ysc * cph)
        ker = (z / (R * R)) * np.exp(1j * k * R)
        # 2 x for (pi, 2pi), and the 1/(2 pi) normalises the phi-average
        out[i] = np.sum(src * (2.0 * (ker @ wph))) / (2.0 * np.pi)
    return out


def rs_j0(ef, z, wavelength, rho):
    """The Debye ``J0`` azimuthal form -- what the shared oracle computes."""
    from scipy.special import j0 as _j0
    k = 2.0 * np.pi / float(wavelength)
    src, ys = _ring_source(ef, wavelength)
    src = src / (1j * float(wavelength))
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    out = np.zeros(rho.size, dtype=complex)
    for i in range(rho.size):
        r = float(rho[i])
        R0 = np.sqrt(z * z + ys * ys + r * r)
        out[i] = np.sum(src * _j0(k * ys * r / R0) * (z / R0)
                        * np.exp(1j * k * R0) / R0)
    return out


def asm_field(ef, z, wavelength, N, dx, refine=4, Nf=None, pad=2.0):
    """Band-limited ANGULAR-SPECTRUM propagation of the same exit field.

    Exact for the scalar Helmholtz equation given the boundary field: no
    Debye, no paraxial and no azimuthal approximation at any radius.

    The exit field is laid on a grid of pitch ``dx / refine`` by CUBIC
    interpolation of the AMPLITUDE and of the OPTICAL PATH separately, the
    phase re-formed after -- not by interpolating the real and imaginary
    parts, which at NA 0.4 commits ~4e-3 of relative error because the field's
    phase turns by ~0.18 rad between source samples.

    ``Nf - refine * N`` is kept EVEN so the output grid's pixel centres are an
    exact subset of the fine lattice and no second interpolation is needed.
    """
    k = 2.0 * np.pi / float(wavelength)
    refine = int(refine)
    if Nf is None:
        Nf = int(pad * refine * int(N))
    Nf = int(Nf)
    if (Nf - refine * int(N)) % 2:
        Nf += 1
    dxf = float(dx) / refine
    xf = (np.arange(Nf) - Nf / 2.0) * dxf
    order = np.argsort(ef['y'])
    ys = ef['y'][order]
    amps = ef['amp'][order]
    opls = ef['opl'][order]
    E0 = np.zeros((Nf, Nf), dtype=np.complex128)
    ymax = float(ys[-1])
    # row by row, so the fine grid never needs a second (Nf, Nf) float array
    for i in range(Nf):
        rr = np.sqrt(xf * xf + xf[i] * xf[i])
        inside = rr <= ymax
        if not np.any(inside):
            continue
        r = rr[inside]
        a = _cubic(ys, amps, r)
        p = _cubic(ys, opls, r)
        row = np.zeros(Nf, dtype=np.complex128)
        row[inside] = a * np.exp(1j * k * p)
        E0[i] = row
    fx = np.fft.fftfreq(Nf, d=dxf)
    FX, FY = np.meshgrid(fx, fx)
    kt2 = (2.0 * np.pi) ** 2 * (FX * FX + FY * FY)
    kz2 = k * k - kt2
    prop = np.where(kz2 > 0.0,
                    np.exp(1j * np.sqrt(np.maximum(kz2, 0.0)) * float(z)), 0.0)
    # Matsushima band limit: frequencies whose ray leaves the periodic window
    flim = 1.0 / (float(wavelength)
                  * np.sqrt((2.0 * float(z) / (Nf * dxf)) ** 2 + 1.0))
    prop = np.where((np.abs(FX) <= flim) & (np.abs(FY) <= flim), prop, 0.0)
    del FX, FY, kt2, kz2
    Ez = np.fft.ifft2(np.fft.fft2(E0) * prop)
    del prop, E0
    off = (Nf - refine * int(N)) // 2
    idx = refine * np.arange(int(N)) + off
    return np.ascontiguousarray(Ez[np.ix_(idx, idx)]).astype(np.complex128)


def _cubic(xs, fs, x):
    """Catmull-Rom cubic interpolation of ``f(x)`` from a sorted sample."""
    xs = np.asarray(xs, float)
    fs = np.asarray(fs, float)
    n = xs.size
    j = np.clip(np.searchsorted(xs, x) - 1, 1, n - 3)
    x0, x1 = xs[j], xs[j + 1]
    t = np.where(x1 > x0, (x - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0)
    fm1, f0, f1, f2 = fs[j - 1], fs[j], fs[j + 1], fs[j + 2]
    m0 = 0.5 * (f1 - fm1)
    m1 = 0.5 * (f2 - f0)
    t2 = t * t
    t3 = t2 * t
    return ((2 * t3 - 3 * t2 + 1) * f0 + (t3 - 2 * t2 + t) * m0
            + (-2 * t3 + 3 * t2) * f1 + (t3 - t2) * m1)


def radial_to_2d(rho, E_rho, N, dx):
    """Rotate a radial complex profile onto the ``(N, N)`` output grid."""
    x = (np.arange(int(N)) - int(N) / 2.0) * float(dx)
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X * X + Y * Y)
    re = np.interp(r, rho, np.real(E_rho), left=np.real(E_rho)[0], right=0.0)
    im = np.interp(r, rho, np.imag(E_rho), left=np.imag(E_rho)[0], right=0.0)
    return (re + 1j * im).astype(np.complex128)


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


def rel_l2(Ea, Eb, w=None):
    a = np.asarray(Ea).ravel()
    b = np.asarray(Eb).ravel()
    if w is None:
        num = float(np.linalg.norm(a - b))
        den = float(np.linalg.norm(b))
    else:
        w = np.asarray(w, float).ravel()
        num = float(np.sqrt(np.sum(w * np.abs(a - b) ** 2)))
        den = float(np.sqrt(np.sum(w * np.abs(b) ** 2)))
    return float('inf') if den == 0.0 else num / den


def w_fidelity(Ea, Eb, w):
    a = np.asarray(Ea).ravel()
    b = np.asarray(Eb).ravel()
    w = np.asarray(w, float).ravel()
    num = abs(np.sum(w * np.conj(a) * b))
    den = np.sqrt(np.sum(w * np.abs(a) ** 2) * np.sum(w * np.abs(b) ** 2))
    return 0.0 if den == 0.0 else float(num / den)


def power(E, dx):
    return float(np.sum(np.abs(np.asarray(E)) ** 2)) * float(dx) ** 2


# ---------------------------------------------------------------------------
# Controls.
# ---------------------------------------------------------------------------
def aspheric_inertness_control(prescription, wavelength, z):
    """Worst column change when an all-zero even-aspheric term is ADDED.

    On a spherical or conic prescription the aspheric extension must be
    bit-inert; this is the control that says so.
    """
    import copy
    a = trace_fan(prescription['surfaces'], wavelength,
                  np.linspace(1e-6, 0.45e-3, 501), z)
    p2 = copy.deepcopy(prescription)
    for s in p2['surfaces']:
        s['aspheric_coeffs'] = {4: 0.0, 6: 0.0, 8: 0.0}
    b = trace_fan(p2['surfaces'], wavelength,
                  np.linspace(1e-6, 0.45e-3, 501), z)
    out = []
    for u, v in zip(a[:3], b[:3]):
        m = np.isfinite(u) & np.isfinite(v)
        out.append(float(np.max(np.abs(u[m] - v[m]))) if np.any(m) else 0.0)
    return out
