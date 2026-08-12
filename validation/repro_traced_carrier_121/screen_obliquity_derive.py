# SCREEN OBLIQUITY -- the closed form, derived and checked against exact rays.
#
# LOCAL-ONLY.  The oracle is the shipped exact ray tracer; the "screen model"
# arm is the FICTITIOUS split-step system apply_real_lens actually implements
# (zero-thickness sag screens separated by homogeneous slabs), traced as a
# Hamiltonian ray system so both arms are eikonals of the SAME kind and can be
# differenced at the EXIT plane.
#
# MODES
#   plate    -- (a) plane-parallel plate: the correction must be EXACTLY zero.
#   sphere   -- (b) single spherical surface, several angles / indices:
#               closed form vs exact rays.
#   d121     -- (c) design 121 group 5 at 54.87 mrad.
#   ablate   -- which surface's correction carries the residual, + the
#               TANGENT-FACET arm that isolates the screen's deflection defect.
#   field    -- the refutation's own exit-plane common-mode control.
#   fieldb   -- the same control on a case it CAN resolve.
#   guard    -- the accuracy estimator on design 121's six groups.
#   all
#
# Run:  python screen_obliquity_derive.py all
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Bind THIS checkout's lumenairy before _d121_common hard-inserts the dev-box
# tree at sys.path[0]; once the package is in sys.modules that insert is inert.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import lumenairy as la  # noqa: E402,I001
from lumenairy.glass import get_glass_index  # noqa: E402
from lumenairy.raytrace import _make_bundle, surfaces_from_prescription, trace  # noqa: E402
import _d121_common as C  # noqa: E402
import hmap_consumer2_121 as H2  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
LAM = C.LAM
# mode_ablate restricts the correction to a subset of surfaces (None = all).
_ALLOW_SURF = None


# ---------------------------------------------------------------------------
# sag + analytic d(sag)/d(h^2) for a plain conic + even asphere
# ---------------------------------------------------------------------------
def sag_and_dsag(h_sq, R, kc=0.0, asph=None):
    """Return ``(sag, d sag / d(h^2))`` for the conic + even asphere."""
    h_sq = np.asarray(h_sq, dtype=float)
    if not np.isfinite(R) or R == 0.0:
        s = np.zeros_like(h_sq)
        d = np.zeros_like(h_sq)
    else:
        c = 1.0 / R
        rad = 1.0 - (1.0 + kc) * c ** 2 * h_sq
        rad = np.maximum(rad, 1e-300)
        sq = np.sqrt(rad)
        D = 1.0 + sq
        s = c * h_sq / D
        # d/du [c u / (1 + sqrt(1 - (1+k) c^2 u))]
        dsq = -(1.0 + kc) * c ** 2 / (2.0 * sq)
        d = c / D - c * h_sq * dsq / D ** 2
    if asph:
        for power, coeff in asph.items():
            m = int(power) // 2
            if m <= 0:
                continue
            s = s + coeff * h_sq ** m
            d = d + coeff * m * h_sq ** (m - 1)
    return s, d


def _resolved_indices(surfaces, wavelength):
    return [(float(get_glass_index(s['glass_before'], wavelength)),
             float(get_glass_index(s['glass_after'], wavelength)))
            for s in surfaces]


# ---------------------------------------------------------------------------
# THE CLOSED FORM
# ---------------------------------------------------------------------------
def phi(p_sq, n1, n2):
    """``Phi(p) = sqrt(n2^2 - |p|^2) - sqrt(n1^2 - |p|^2)`` -- the exact
    generalised thin-facet OPD coefficient in the MOMENTUM representation.
    ``Phi(0) = n2 - n1`` is the shipped paraxial screen."""
    return (np.sqrt(np.maximum(n2 ** 2 - p_sq, 1e-300))
            - np.sqrt(np.maximum(n1 ** 2 - p_sq, 1e-300)))


def _walk(sag, ppx, ppy, n2):
    """Transverse walk of the REFRACTED ray across the sag height."""
    pz = np.sqrt(np.maximum(n2 ** 2 - ppx ** 2 - ppy ** 2, 1e-300))
    return sag * ppx / pz, sag * ppy / pz


def _pz_out(px, py, gx, gy, n1, n2):
    """``(n1 cos a1, n2 cos a2)`` -- the axial optical-momentum components
    before and after refraction at the LOCAL facet whose unit normal is
    ``nu = (-grad sag, 1) / sqrt(1 + |grad sag|^2)``.  Exact vector Snell."""
    pz1 = np.sqrt(np.maximum(n1 ** 2 - px ** 2 - py ** 2, 0.0))
    inv = 1.0 / np.sqrt(1.0 + gx ** 2 + gy ** 2)
    ax, ay, az = -gx * inv, -gy * inv, inv
    A = px * ax + py * ay + pz1 * az
    B = np.sqrt(np.maximum(n2 ** 2 - n1 ** 2 + A ** 2, 0.0))
    return pz1, pz1 + (B - A) * az


def delta_opd(qx, qy, p0x, p0y, sag, gx, gy, n1, n2, form='full'):
    """The ANGULAR correction to the shipped screen OPD at one surface.

    ``OPD_obl(x) = Phi(|p|) * sag(x + w)`` with ``Phi(u) = sqrt(n2^2-u^2) -
    sqrt(n1^2-u^2)`` and ``w`` the transverse walk of the refracted ray across
    the sag height; the correction is that MINUS its carrier-free value, so it
    vanishes identically at zero carrier and for zero sag.
    """
    px, py = p0x + qx, p0y + qy
    dn = n2 - n1
    if form == 'snell':
        z1, z2 = _pz_out(px, py, gx, gy, n1, n2)
        w1, w2 = _pz_out(p0x, p0y, gx, gy, n1, n2)
        return ((z2 - z1) - (w2 - w1)) * sag
    if form == 'quad':
        return dn / (2.0 * n1 * n2) * ((px ** 2 + py ** 2)
                                       - (p0x ** 2 + p0y ** 2)) * sag
    f1 = phi(px ** 2 + py ** 2, n1, n2)
    f0 = phi(p0x ** 2 + p0y ** 2, n1, n2)
    if form == 'nowalk':
        return (f1 - f0) * sag
    if form == 'nocross':                    # carrier-only, no p0 cross term
        return (phi(qx ** 2 + qy ** 2, n1, n2) - dn) * sag
    if form != 'full':
        raise ValueError(form)
    wx, wy = _walk(sag, px - dn * gx, py - dn * gy, n2)
    w0x, w0y = _walk(sag, p0x - dn * gx, p0y - dn * gy, n2)
    return (f1 * (sag + gx * wx + gy * wy)
            - f0 * (sag + gx * w0x + gy * w0y))


def _sag_grad(surf, x, y):
    sag, dsag = sag_and_dsag(x ** 2 + y ** 2, surf['radius'],
                             surf.get('conic', 0.0),
                             surf.get('aspheric_coeffs'))
    return sag, 2.0 * x * dsag, 2.0 * y * dsag


def _corr_at(surfaces, idx, i, x, y, qx, qy, form):
    """The angular correction at surface ``i``, as a closed-form function of
    the field-plane coordinate alone -- the carrier-free momentum ``p0`` is the
    grid-local accumulation of the preceding screens' own deflections."""
    p0x = np.zeros_like(x)
    p0y = np.zeros_like(y)
    for j in range(i):
        n1j, n2j = idx[j]
        _s, gxj, gyj = _sag_grad(surfaces[j], x, y)
        p0x = p0x - (n2j - n1j) * gxj
        p0y = p0y - (n2j - n1j) * gyj
    n1, n2 = idx[i]
    sag, gx, gy = _sag_grad(surfaces[i], x, y)
    return delta_opd(qx, qy, p0x, p0y, sag, gx, gy, n1, n2, form=form)


# ---------------------------------------------------------------------------
# THE SCREEN MODEL AS A RAY SYSTEM (S6's fictitious Hamiltonian trace)
# ---------------------------------------------------------------------------
def screen_trace(prescription, wavelength, x0, y0, L, M, correction=None,
                 form='snell', p0_mode='grid'):
    """Trace apply_real_lens's OWN model -- zero-thickness sag screens with
    homogeneous ASM slabs between them -- as a Hamiltonian ray system.

    Field convention ``exp(i k0 Lam)``; each screen multiplies
    ``exp(-i k0 OPD)`` so ``Lam -= OPD`` and ``p -= grad OPD``.  A slab of
    thickness ``t`` and index ``n`` advances ``x += t p / pz``,
    ``Lam += t n^2 / pz`` with ``pz = sqrt(n^2 - |p|^2)`` -- the Legendre
    transform of the ASM kernel ``exp(i k0 t pz)``, so the trace reproduces
    the wave model's geometric limit by construction.

    ``correction='obliquity'`` additionally imprints the angular screen
    correction derived in BUILD_SCREEN_OBLIQUITY_2026_08_11.

    Returns ``(x, y, Lam, px, py)`` at the exit VERTEX plane.
    """
    surfaces = prescription['surfaces']
    thick = prescription['thicknesses']
    idx = _resolved_indices(surfaces, wavelength)
    x = np.array(x0, dtype=float, copy=True)
    y = np.array(y0, dtype=float, copy=True)
    px = np.full_like(x, float(L))
    py = np.full_like(y, float(M))
    # W_in for a uniform tilt carrier
    Lam = L * x + M * y
    # carrier momentum (constant for a uniform tilt); the screen-model
    # accumulated momentum p0 is built up from the sag gradients alone.
    qx = np.full_like(x, float(L))
    qy = np.full_like(y, float(M))
    p0x = np.zeros_like(x)
    p0y = np.zeros_like(y)
    for i, surf in enumerate(surfaces):
        n1, n2 = idx[i]
        h_sq = x ** 2 + y ** 2
        sag, dsag = sag_and_dsag(h_sq, surf['radius'], surf.get('conic', 0.0),
                                 surf.get('aspheric_coeffs'))
        gx = 2.0 * x * dsag
        gy = 2.0 * y * dsag
        opd = (n2 - n1) * sag
        gopdx = (n2 - n1) * gx
        gopdy = (n2 - n1) * gy
        if correction == 'obliquity' and (_ALLOW_SURF is None
                                          or i in _ALLOW_SURF):
            if p0_mode == 'grid':
                pax, pay = p0x, p0y          # accumulated at THIS (x, y)
            else:
                pax, pay = px - qx, py - qy  # the true model momentum
            d = delta_opd(qx, qy, pax, pay, sag, gx, gy, n1, n2, form=form)
            opd = opd + d
            # The correction is a closed-form function of (x, y) alone, so its
            # gradient (the momentum kick the WAVE gets for free from the phase
            # screen) is a central difference of that function.
            hh = 1e-7
            dxp = _corr_at(surfaces, idx, i, x + hh, y, qx, qy, form)
            dxm = _corr_at(surfaces, idx, i, x - hh, y, qx, qy, form)
            dyp = _corr_at(surfaces, idx, i, x, y + hh, qx, qy, form)
            dym = _corr_at(surfaces, idx, i, x, y - hh, qx, qy, form)
            gopdx = gopdx + (dxp - dxm) / (2.0 * hh)
            gopdy = gopdy + (dyp - dym) / (2.0 * hh)
        Lam = Lam - opd
        px = px - gopdx
        py = py - gopdy
        p0x = p0x - (n2 - n1) * gx
        p0y = p0y - (n2 - n1) * gy
        if i < len(surfaces) - 1:
            t = float(thick[i])
            pz = np.sqrt(np.maximum(n2 ** 2 - px ** 2 - py ** 2, 1e-300))
            x = x + t * px / pz
            y = y + t * py / pz
            Lam = Lam + t * n2 ** 2 / pz
    return x, y, Lam, px, py


def exact_trace(prescription, wavelength, x0, y0, L, M):
    """The oracle: the shipped exact ray trace, entrance plane to exit VERTEX
    plane, with the input plane-wave eikonal ``L x0 + M y0`` added so both
    arms share one origin."""
    presc = dict(prescription)
    presc.pop('aperture_diameter', None)
    surfs = surfaces_from_prescription(presc)
    n_exit = float(get_glass_index(surfs[-1].glass_after, wavelength))
    rays = _make_bundle(x=np.ravel(x0), y=np.ravel(y0),
                        L=np.full(np.size(x0), float(L)),
                        M=np.full(np.size(y0), float(M)), wavelength=wavelength)
    fin = trace(rays, surfs, wavelength, output_filter='last').image_rays
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(fin.alive & (np.abs(fin.N) > 1e-30), -fin.z / fin.N, 0.0)
    return (fin.x + fin.L * t, fin.y + fin.M * t,
            L * np.ravel(x0) + M * np.ravel(y0) + fin.opd + n_exit * t,
            np.asarray(fin.alive, dtype=bool))


# ---------------------------------------------------------------------------
# the exit-referenced model error, and its ANGULAR part
# ---------------------------------------------------------------------------
def model_error(prescription, wavelength, x0, y0, L, M, correction=None,
                form='snell', p0_mode='grid'):
    """``Lam_model - Lam_exact`` at the EXACT ray's exit point.

    Both arms start from the same entrance point with the same direction.  The
    model arm lands at its own exit point ``x_m``; its eikonal is carried to
    the exact ray's exit point ``x_e`` by ``Lam += p . (x_e - x_m)`` (first
    order in the model's own exit-point error, which is the same order the
    thin-screen model is accurate to).  THAT is the exit-plane referencing the
    refutation demands."""
    xe, ye, Le, alive = exact_trace(prescription, wavelength, x0, y0, L, M)
    xm, ym, Lm, pxm, pym = screen_trace(prescription, wavelength, x0, y0, L, M,
                                        correction=correction, form=form,
                                        p0_mode=p0_mode)
    Lm_at_e = Lm + pxm * (xe - xm) + pym * (ye - ym)
    return Lm_at_e - Le, alive


def angular_error(prescription, wavelength, x0, y0, L, M, correction=None,
                  form='snell', p0_mode='grid'):
    """The COMMON-MODE-controlled angular error, in waves.

    ``D(theta) - D(0)`` with ``D = Lam_model - Lam_exact`` at the exit plane,
    piston and tilt removed over the pupil.  Everything angle-independent --
    which is everything that separates the two models at normal incidence, the
    model's documented accuracy ceiling -- cancels."""
    d_t, a1 = model_error(prescription, wavelength, x0, y0, L, M,
                          correction=correction, form=form, p0_mode=p0_mode)
    d_0, a0 = model_error(prescription, wavelength, x0, y0, 0.0, 0.0,
                          correction=correction, form=form, p0_mode=p0_mode)
    ok = a1 & a0 & np.isfinite(d_t) & np.isfinite(d_0)
    d = (d_t - d_0)[ok]
    px, py = np.ravel(x0)[ok], np.ravel(y0)[ok]
    A = np.stack([np.ones_like(px), px, py], axis=1)
    c, *_ = np.linalg.lstsq(A, d, rcond=None)
    res = d - A @ c
    return {'rms': float(np.sqrt((res ** 2).mean())) / wavelength,
            'max': float(np.abs(res).max()) / wavelength,
            'piston': float(c[0]) / wavelength,
            'tilt': float(np.abs(A[:, 1:] @ c[1:]).max()) / wavelength,
            'n': int(ok.sum())}


def disc(rad, n=65):
    t = np.linspace(-1.0, 1.0, n)
    PX, PY = np.meshgrid(rad * t, rad * t, indexing='ij')
    keep = (PX ** 2 + PY ** 2) <= rad ** 2
    return PX[keep], PY[keep]


# ---------------------------------------------------------------------------
# (a) the plane plate -- the correction must be EXACTLY zero
# ---------------------------------------------------------------------------
def mode_plate(verbose=True):
    presc = {'surfaces': [
        {'radius': np.inf, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': 'AIR', 'glass_after': 'N-BK7'},
        {'radius': np.inf, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': 'N-BK7', 'glass_after': 'AIR'}],
        'thicknesses': [25.4e-3], 'name': 'plate'}
    px, py = disc(3e-3)
    rows = []
    for tilt in (0.0, 10.0e-3, 20.0e-3, 41.5e-3, 100.0e-3):
        L = float(tilt)
        _x, _y, Lb, _px, _py = screen_trace(presc, LAM, px, py, L, 0.0)
        _x, _y, Lc, _px, _py = screen_trace(presc, LAM, px, py, L, 0.0,
                                            correction='obliquity')
        dmax = float(np.abs(Lc - Lb).max())
        same = bool(np.array_equal(Lb.view(np.uint8), Lc.view(np.uint8)))
        # and against the exact rays
        eb = angular_error(presc, LAM, px, py, L, 0.0)
        rows.append({'tilt_mrad': L * 1e3, 'corr_minus_shipped_m': dmax,
                     'byte_identical': same,
                     'shipped_angular_rms_waves': eb['rms']})
        if verbose:
            print('  tilt %7.3f mrad:  correction - shipped = %.3e m  '
                  '(byte-identical %s)   shipped angular error %.3e w'
                  % (L * 1e3, dmax, same, eb['rms']))
    return {'rows': rows}


# ---------------------------------------------------------------------------
# (b) a single spherical surface
# ---------------------------------------------------------------------------
def mode_sphere(verbose=True):
    rows = []
    cases = [('N-BK7', 50.0e-3), ('N-BK7', 25.0e-3), ('N-SF11', 50.0e-3),
             ('N-SF11', 25.0e-3), ('N-BK7', -50.0e-3)]
    tilts = (10.0e-3, 30.0e-3, 54.87e-3, 100.0e-3)
    for glass, R in cases:
        n2 = float(get_glass_index(glass, LAM))
        presc = {'surfaces': [
            {'radius': R, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'AIR', 'glass_after': glass},
            {'radius': np.inf, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': glass, 'glass_after': 'AIR'}],
            'thicknesses': [4.0e-3], 'name': 'singlet'}
        px, py = disc(3e-3)
        h = 3e-3
        sag = float(sag_and_dsag(np.array([h ** 2]), R)[0][0])
        for tilt in tilts:
            e_ship = angular_error(presc, LAM, px, py, tilt, 0.0)
            e_ex = angular_error(presc, LAM, px, py, tilt, 0.0,
                                 correction='obliquity', form='snell')
            e_w = angular_error(presc, LAM, px, py, tilt, 0.0,
                                correction='obliquity', form='full')
            e_q = angular_error(presc, LAM, px, py, tilt, 0.0,
                                correction='obliquity', form='quad')
            e_nc = angular_error(presc, LAM, px, py, tilt, 0.0,
                                 correction='obliquity', form='nowalk')
            pred = abs(sag) * (n2 - 1.0) / (2.0 * n2) * tilt ** 2 / LAM
            rows.append({'glass': glass, 'n': n2, 'R_mm': R * 1e3,
                         'sag_3mm_um': sag * 1e6, 'tilt_mrad': tilt * 1e3,
                         'shipped_rms': e_ship['rms'],
                         'shipped_max': e_ship['max'],
                         'snell_rms': e_ex['rms'], 'snell_max': e_ex['max'],
                         'walk_rms': e_w['rms'],
                         'quad_rms': e_q['rms'], 'nowalk_rms': e_nc['rms'],
                         'scale_pred_waves': pred})
            if verbose:
                print('  %-7s R=%+7.1f mm sag(3mm)=%8.1f um  %6.2f mrad:  '
                      'SHIPPED %9.5f w rms  ->  snell %9.6f  walk %9.6f  '
                      'quad %9.5f  nowalk %9.5f'
                      '   [scale (n-1)sag th^2/2n = %7.4f w]'
                      % (glass, R * 1e3, sag * 1e6, tilt * 1e3,
                         e_ship['rms'], e_ex['rms'], e_w['rms'],
                         e_q['rms'], e_nc['rms'], pred))
    return {'rows': rows}


# ---------------------------------------------------------------------------
# (c) design 121, group 5, 54.87 mrad
# ---------------------------------------------------------------------------
def mode_d121(verbose=True):
    groups_post, tilts, orders, _r = H2.geometry()
    out = []
    for gi in range(len(groups_post)):
        presc = dict(groups_post[gi]['prescription'])
        presc.pop('aperture_diameter', None)
        best = None
        for i, (mm, nn) in enumerate(orders):
            _xc, _yc, L, M = tilts[gi][i]
            if best is None or np.hypot(L, M) > np.hypot(best[2], best[3]):
                best = (int(mm), int(nn), L, M)
        L, M = best[2], best[3]
        for rad in (1e-3, 2e-3, 3e-3):
            px, py = disc(rad)
            e_ship = angular_error(presc, LAM, px, py, L, M)
            e_ex = angular_error(presc, LAM, px, py, L, M,
                                 correction='obliquity', form='snell')
            e_w = angular_error(presc, LAM, px, py, L, M,
                                correction='obliquity', form='full')
            e_q = angular_error(presc, LAM, px, py, L, M,
                                correction='obliquity', form='quad')
            e_nc = angular_error(presc, LAM, px, py, L, M,
                                 correction='obliquity', form='nowalk')
            e_true = angular_error(presc, LAM, px, py, L, M,
                                   correction='obliquity', form='snell',
                                   p0_mode='model')
            row = {'group': gi, 'radius_mm': rad * 1e3,
                   'order': [best[0], best[1]],
                   'tilt_mrad': float(np.hypot(L, M) * 1e3),
                   'shipped_rms': e_ship['rms'], 'shipped_max': e_ship['max'],
                   'snell_rms': e_ex['rms'], 'snell_max': e_ex['max'],
                   'walk_rms': e_w['rms'],
                   'quad_rms': e_q['rms'], 'nowalk_rms': e_nc['rms'],
                   'modelp0_rms': e_true['rms']}
            out.append(row)
            if verbose:
                print('  group %d r=%.1f mm %6.2f mrad:  SHIPPED %9.5f w rms  '
                      '->  snell %9.6f  (model-p0 %9.6f)  walk %9.5f  '
                      'quad %9.5f  nowalk %9.5f'
                      % (gi, rad * 1e3, row['tilt_mrad'], e_ship['rms'],
                         e_ex['rms'], e_true['rms'], e_w['rms'],
                         e_q['rms'], e_nc['rms']))
    return {'rows': out}


# ---------------------------------------------------------------------------
# ablate -- WHICH surface's correction carries the residual
# ---------------------------------------------------------------------------
def mode_ablate(gi=5, rad=3e-3, verbose=True):
    """Apply the correction at each SUBSET of the group's surfaces.

    On design 121's group 5 the correction at surface 0 ALONE makes the error
    worse, which looks like a broken closed form and is not: what is left after
    the correction is the screen's angle-blind DEFLECTION kick (S3.4 of the
    build note), and correcting one surface's OPD without correcting any
    surface's kick can move the two out of the partial cancellation the shipped
    model happens to sit in.  The tangent-facet arm below settles it."""
    global _ALLOW_SURF
    groups_post, tilts, orders, _r = H2.geometry()
    presc = dict(groups_post[gi]['prescription'])
    presc.pop('aperture_diameter', None)
    best = None
    for i, _o in enumerate(orders):
        _xc, _yc, L, M = tilts[gi][i]
        if best is None or np.hypot(L, M) > np.hypot(best[0], best[1]):
            best = (L, M)
    L, M = best
    px, py = disc(rad)
    n_surf = len(presc['surfaces'])
    subsets = [()]
    for k in range(1, n_surf + 1):
        subsets += [tuple(c) for c in _combos(range(n_surf), k)]
    rows = []
    for sub in subsets:
        _ALLOW_SURF = sub
        e = angular_error(presc, LAM, px, py, L, M, correction='obliquity',
                          form='snell')
        rows.append({'surfaces': list(sub), 'rms_waves': e['rms'],
                     'max_waves': e['max']})
        if verbose:
            print('  correction at surfaces %-12s: %9.5f w rms  %9.5f w max'
                  % (str(sub) if sub else '(none)', e['rms'], e['max']))
    _ALLOW_SURF = None
    # the TANGENT-FACET arm: exact refraction at the local tangent plane on the
    # vertex plane, plus the exact translation OPD.  If the identity is right,
    # this is the exact rays.
    for translate in (False, True):
        e = _facet_arm_error(presc, px, py, L, M, translate)
        rows.append({'tangent_facet': True, 'translate': translate,
                     'rms_waves': e})
        if verbose:
            print('  tangent facet, translation OPD %-5s : %9.6f w rms'
                  % (translate, e))
    return {'group': gi, 'radius_mm': rad * 1e3, 'rows': rows}


def _combos(items, k):
    items = list(items)
    if k == 0:
        yield []
        return
    for i, v in enumerate(items):
        for rest in _combos(items[i + 1:], k - 1):
            yield [v] + rest


def _facet_arm(prescription, x0, y0, L, M, translate):
    """Each surface refracts EXACTLY at its tangent plane placed on the vertex
    plane; gaps homogeneous; optionally plus the exact axial-translation OPD
    ``(n2 cos a2 - n1 cos a1) * sag``."""
    surfaces, thick = prescription['surfaces'], prescription['thicknesses']
    idx = _resolved_indices(surfaces, LAM)
    x, y = np.array(x0, float), np.array(y0, float)
    px, py = np.full_like(x, float(L)), np.full_like(y, float(M))
    lam = L * x + M * y
    for i, surf in enumerate(surfaces):
        n1, n2 = idx[i]
        sag, gx, gy = _sag_grad(surf, x, y)
        pz1, pz2 = _pz_out(px, py, gx, gy, n1, n2)
        inv = 1.0 / np.sqrt(1.0 + gx ** 2 + gy ** 2)
        ax, ay, az = -gx * inv, -gy * inv, inv
        a = px * ax + py * ay + pz1 * az
        b = np.sqrt(np.maximum(n2 ** 2 - n1 ** 2 + a ** 2, 0.0))
        px, py = px + (b - a) * ax, py + (b - a) * ay
        if translate:
            lam = lam - (pz2 - pz1) * sag
        if i < len(surfaces) - 1:
            t = float(thick[i])
            pz = np.sqrt(np.maximum(n2 ** 2 - px ** 2 - py ** 2, 1e-300))
            x, y = x + t * px / pz, y + t * py / pz
            lam = lam + t * n2 ** 2 / pz
    return x, y, lam, px, py


def _facet_arm_error(presc, px, py, L, M, translate):
    d = []
    for (l_, m_) in ((L, M), (0.0, 0.0)):
        xe, ye, le, _al = exact_trace(presc, LAM, px, py, l_, m_)
        xm, ym, lm, pxm, pym = _facet_arm(presc, px, py, l_, m_, translate)
        d.append(lm + pxm * (xe - xm) + pym * (ye - ym) - le)
    g = d[0] - d[1]
    A = np.stack([np.ones_like(px), px, py], axis=1)
    c, *_ = np.linalg.lstsq(A, g, rcond=None)
    return float(np.sqrt(((g - A @ c) ** 2).mean())) / LAM


# ---------------------------------------------------------------------------
# field -- THE REFERENCE-PLANE DISCIPLINE.  The refutation's own 'field' arm
# (hmap_consumer2_121.mode_field), verbatim in configuration and in the
# common-mode control, with the SHIPPED library correction as a third arm.
# ---------------------------------------------------------------------------
def mode_field(gi=5, n=512, dx=12.0e-6, w=2.5e-3, r_probe=1.5e-3,
               select='refutation', verbose=True):
    import warnings

    from lumenairy.elements import apply_real_lens, apply_real_lens_traced
    from lumenairy.elements._lens_traced import _tilted_carrier_parts
    groups_post, tilts, orders, _r = H2.geometry()
    presc = groups_post[gi]['prescription']
    best = None
    for i, (mm, nn) in enumerate(orders):
        x_c, y_c, L, M = tilts[gi][i]
        # ``select='refutation'`` reproduces hmap_consumer2_121.mode_field's
        # OWN order pick verbatim -- it scores hypot(best[3], best[4]), i.e.
        # (y_chief, L) rather than (L, M), and lands on the 50.52 mrad order
        # the refutation's 0.374 / 0.407 bar was measured at.  Reproduced
        # exactly so the bar is apples-to-apples; ``select='max_tilt'`` is the
        # corrected pick (54.87 mrad, the same order the eikonal arms use).
        if best is None or (np.hypot(L, M) > np.hypot(best[4], best[5])
                            if select == 'max_tilt'
                            else np.hypot(L, M) > np.hypot(best[3], best[4])):
            best = (int(mm), int(nn), x_c, y_c, L, M)
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x)
    mask = (X ** 2 + Y ** 2) <= r_probe ** 2
    gaps = {}
    for tag, (L, M) in (('zero', (0.0, 0.0)), ('tilt', (best[4], best[5]))):
        car = la.TiltedCarrier(float('inf'), L, M)
        W, _l, _m = _tilted_carrier_parts(car, X, Y)
        E = (np.exp(-(X ** 2 + Y ** 2) / w ** 2)
             * np.exp(1j * 2 * np.pi / LAM * W)).astype(np.complex128)
        kw = dict(prescription=presc, wavelength=LAM, dx=dx)
        E_blind = apply_real_lens(E, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E_obl = apply_real_lens(E, carrier=car,
                                    on_screen_obliquity='silent', **kw)
            E_tr = apply_real_lens_traced(
                E, prescription=presc, wavelength=LAM, dx=dx, carrier=car,
                tilt_aware_rays=True, on_noncollimated='off')
        gaps[tag] = {
            'blind': H2._wavefront_gap(E_blind, E_tr, X, Y, mask),
            'obliquity': H2._wavefront_gap(E_obl, E_tr, X, Y, mask)}
    out = {'group': gi, 'n': n, 'dx_um': dx * 1e6, 'w_mm': w * 1e3,
           'r_probe_mm': r_probe * 1e3, 'order': [best[0], best[1]],
           'tilt_mrad': float(np.hypot(best[4], best[5]) * 1e3),
           'n_mask': int(mask.sum())}
    for arm in ('blind', 'obliquity'):
        d = gaps['tilt'][arm] - gaps['zero'][arm]
        d = d - d.mean()
        out[arm] = {'angular_rms_waves': float(np.sqrt((d ** 2).mean())),
                    'angular_max_waves': float(np.abs(d).max()),
                    'raw_zero_rms_waves': float(np.sqrt(
                        (gaps['zero'][arm] ** 2).mean())),
                    'raw_tilt_rms_waves': float(np.sqrt(
                        (gaps['tilt'][arm] ** 2).mean()))}
        if verbose:
            print('  %-9s: raw vs traced %.4f w rms at 0 mrad, %.4f w rms at '
                  '%.2f mrad  ->  ANGULAR part %.5f w rms / %.5f w max'
                  % (arm, out[arm]['raw_zero_rms_waves'],
                     out[arm]['raw_tilt_rms_waves'], out['tilt_mrad'],
                     out[arm]['angular_rms_waves'],
                     out[arm]['angular_max_waves']))
    if verbose:
        print('  ANGULAR-ERROR RECOVERY (blind rms / corrected rms): %.3f x'
              % (out['blind']['angular_rms_waves']
                 / max(out['obliquity']['angular_rms_waves'], 1e-30)))
        print('  BAR: corrected must be BELOW the blind %.5f w'
              % out['blind']['angular_rms_waves'])
    return out


# ---------------------------------------------------------------------------
# fieldb -- the field arm on a case it CAN resolve.
# ---------------------------------------------------------------------------
def mode_fieldb(R=19.6e-3, glass='N-SSK2', t=4.0e-3, ap=3.0e-3, n=1536,
                dx=4.0e-6, w=0.9e-3, r_probe=0.7e-3,
                tilts=(0.02, 0.05, 0.10, 0.15), verbose=True):
    """The same exit-plane common-mode control as :func:`mode_field`, on a
    NYQUIST-SAMPLED fast singlet instead of design 121's last group.

    S4.4 of the refutation is right that the d121 field arm cannot resolve
    this feature -- its analytic-vs-traced floor AT NORMAL INCIDENCE (0.29
    waves rms) is an order of magnitude larger than the whole term, and the
    exit wavefront is 13x under-sampled there.  Removing both (a slower exit
    NA, ``dx <= lambda/(2 NA)``, a beam inside the aperture) drops the floor
    to ~0.0016 waves and the arm becomes a real measurement."""
    import warnings

    from lumenairy.elements import apply_real_lens, apply_real_lens_traced
    from lumenairy.elements._lens_traced import _tilted_carrier_parts
    presc = {'surfaces': [
        {'radius': R, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': 'AIR', 'glass_after': glass},
        {'radius': np.inf, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': glass, 'glass_after': 'AIR'}],
        'thicknesses': [t], 'aperture_diameter': ap, 'name': 'singlet'}
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x)
    mask = (X ** 2 + Y ** 2) <= r_probe ** 2
    rows, base = [], None
    for tilt in (0.0,) + tuple(tilts):
        car = la.TiltedCarrier(float('inf'), float(tilt), 0.0)
        W, _l, _m = _tilted_carrier_parts(car, X, Y)
        E = (np.exp(-(X ** 2 + Y ** 2) / w ** 2)
             * np.exp(1j * 2 * np.pi / LAM * W)).astype(np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Eb = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx)
            Eo = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                                 carrier=car, on_screen_obliquity='silent')
            Et = apply_real_lens_traced(
                E, prescription=presc, wavelength=LAM, dx=dx, carrier=car,
                on_noncollimated='off', on_aperture_beam='silent',
                on_undersample='silent')
        g = {'blind': H2._wavefront_gap(Eb, Et, X, Y, mask),
             'obliquity': H2._wavefront_gap(Eo, Et, X, Y, mask)}
        if base is None:
            base = g
            floor = float(np.sqrt((g['blind'] ** 2).mean()))
            if verbose:
                print('  model-vs-model floor at 0 mrad: %.5f waves rms'
                      % floor)
            continue
        vals = {}
        for arm in ('blind', 'obliquity'):
            d = g[arm] - base[arm]
            d = d - d.mean()
            vals[arm] = float(np.sqrt((d ** 2).mean()))
        rows.append({'tilt_mrad': tilt * 1e3, 'blind_rms_waves': vals['blind'],
                     'obliquity_rms_waves': vals['obliquity'],
                     'recovery': vals['blind'] / max(vals['obliquity'], 1e-30)})
        if verbose:
            print('  %6.1f mrad:  blind %.5f w rms  ->  corrected %.5f w rms'
                  '   recovery %5.2f x' % (tilt * 1e3, vals['blind'],
                                           vals['obliquity'],
                                           rows[-1]['recovery']))
    return {'R_mm': R * 1e3, 'glass': glass, 'aperture_mm': ap * 1e3,
            'n': n, 'dx_um': dx * 1e6, 'r_probe_mm': r_probe * 1e3,
            'floor_rms_waves': floor, 'rows': rows}


# ---------------------------------------------------------------------------
# guard -- the estimator, on design 121's own groups
# ---------------------------------------------------------------------------
def mode_guard(n=512, dx=12.0e-6, verbose=True):
    """What ``on_screen_obliquity`` actually does on the six post-DOE groups,
    at each group's own extreme-order carrier."""
    import warnings

    from lumenairy.elements import apply_real_lens
    from lumenairy.elements._lens_real import _SCREEN_OBLIQUITY_TOL_WAVES

    groups_post, tilts, orders, _r = H2.geometry()
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x)
    rows = []
    for gi in range(len(groups_post)):
        presc = dict(groups_post[gi]['prescription'])
        presc['aperture_diameter'] = 6.0e-3
        best = None
        for i, _o in enumerate(orders):
            _xc, _yc, L, M = tilts[gi][i]
            if best is None or np.hypot(L, M) > np.hypot(best[0], best[1]):
                best = (L, M)
        L, M = best
        car = la.TiltedCarrier(float('inf'), L, M)
        E = (np.exp(-(X ** 2 + Y ** 2) / (2.0e-3) ** 2)).astype(np.complex128)
        est = {}
        for tag, kw in (('off', {'screen_obliquity': False}),
                        ('on', {})):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                                carrier=car, **kw)
            est[tag] = any('angle-blind' in str(rec.message) for rec in w)
        # the estimate itself, read back through the same public expression
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                            carrier=car, screen_obliquity=False)
        val = None
        for rec in w:
            msg = str(rec.message)
            if 'angle-blind by an estimated' in msg:
                val = float(msg.split('estimated ')[1].split(' waves')[0])
        rows.append({'group': gi, 'tilt_mrad': float(np.hypot(L, M) * 1e3),
                     'estimate_waves': val, 'fires_off': est['off'],
                     'fires_on': est['on']})
        if verbose:
            print('  group %d %6.2f mrad:  estimate %s waves   guard fires: '
                  'correction OFF %-5s   correction ON %-5s'
                  % (gi, rows[-1]['tilt_mrad'],
                     ('%.5f' % val) if val is not None else
                     ('<= %.2f' % _SCREEN_OBLIQUITY_TOL_WAVES),
                     est['off'], est['on']))
    return {'tol_waves': _SCREEN_OBLIQUITY_TOL_WAVES, 'rows': rows}


# ---------------------------------------------------------------------------
def main(argv):
    mode = argv[1] if len(argv) > 1 else 'all'
    order = (['plate', 'sphere', 'd121', 'ablate', 'field', 'fieldb',
              'guard']
             if mode == 'all'
             else [mode])
    res = {}
    for m in order:
        print('\n===== %s %s' % (m.upper(), '=' * (60 - len(m))))
        res[m] = {'plate': mode_plate, 'sphere': mode_sphere,
                  'd121': mode_d121, 'field': mode_field,
                  'fieldb': mode_fieldb, 'guard': mode_guard,
                  'ablate': mode_ablate}[m]()
    fn = os.path.join(_HERE, '_screen_obl_%s.json' % mode)
    with open(fn, 'w', encoding='ascii') as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    print('\nwrote %s' % os.path.basename(fn))


if __name__ == '__main__':
    main(sys.argv)
