# ROUTE 3 -- the per-pixel TANGENT-FACET screen, derived and scored against
# exact rays.  LOCAL-ONLY (needs design 121's .zmx).
#
# Two independent oracles, because the two claims are different in kind:
#
#   facet  -- the CLOSED FORM.  For a tilted PLANE facet under a plane wave the
#             screen a wave model must imprint is exactly ``S_in - S_out``, and
#             both eikonals are known in closed form.  No ray-vs-wave scoring,
#             no common-mode control, no tolerance: the identity is right to
#             1e-13 RELATIVE or it is wrong.
#
#   d121 /  -- the MODEL.  The shipped exact ray tracer, differenced at the exit
#   ladder     plane with the common-mode control D(theta) - D(0), piston and
#              tilt removed, exactly as screen_obliquity_derive.py does it.  The
#              model arm is traced on a REGULAR BUNDLE rather than a scattered
#              disc, so that every gradient the library takes with xp.gradient
#              on its grid is taken here as the same physical gradient of the
#              same ray-carried field (via the bundle's own Jacobian).  That is
#              the only faithful way to score a model whose screens are
#              gradients of accumulator FIELDS.
#
# Run:  python tangent_facet_derive.py all
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import lumenairy as la  # noqa: E402,F401,I001
import screen_obliquity_derive as S  # noqa: E402
import hmap_consumer2_121 as H2  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
LAM = S.LAM


# ---------------------------------------------------------------------------
# the bundle, and physical gradients on it
# ---------------------------------------------------------------------------
def bundle(rad=3e-3, n=65, pad=6):
    """A square ray bundle whose INSCRIBED disc of radius ``rad`` is scored;
    the pad rows exist so every scored point has a two-sided stencil."""
    h = rad * (1.0 + 2.0 * pad / (n - 1.0))
    t = np.linspace(-h, h, n + 2 * pad)
    X, Y = np.meshgrid(t, t, indexing='ij')
    return X, Y, (X ** 2 + Y ** 2) <= rad ** 2


def _jac(x, y):
    xu, xv = np.gradient(x)
    yu, yv = np.gradient(y)
    return xu, xv, yu, yv, (xu * yv - xv * yu)


def _grad(f, xu, xv, yu, yv, det):
    fu, fv = np.gradient(f)
    return ((fu * yv - fv * yu) / det, (-fu * xv + fv * xu) / det)


# ---------------------------------------------------------------------------
# the model, term by term
# ---------------------------------------------------------------------------
def screen(surf, x, y, px, py, n1, n2, J, terms=(1, 1, 1)):
    """(T1) + (T2) + (T3) of the derivation above ``_tangent_facet_screen``."""
    s, gx, gy = S._sag_grad(surf, x, y)
    inv = 1.0 / np.sqrt(1.0 + gx ** 2 + gy ** 2)
    pz1 = np.sqrt(np.maximum(n1 ** 2 - px ** 2 - py ** 2, 1e-300))
    a = (-gx * px - gy * py + pz1) * inv
    b = np.sqrt(np.maximum(n2 ** 2 - n1 ** 2 + a ** 2, 0.0))
    dz = (b - a) * inv
    opd = dz * s                                              # (T1)
    wx, wy = s * px / pz1, s * py / pz1
    if terms[0]:                                              # (T2a)
        dzx, dzy = _grad(dz, *J)
        opd = opd + s * (wx * dzx + wy * dzy)
    if terms[1]:                                              # (T2b)
        hxx, hxy = _grad(gx, *J)
        hyx, hyy = _grad(gy, *J)
        opd = opd - 0.5 * dz * (wx * (wx * hxx + wy * hxy)
                                + wy * (wx * hyx + wy * hyy))
    if terms[2]:                                              # (T3)
        ox, oy = px - dz * gx, py - dz * gy
        pz2 = pz1 + dz
        ux = wx - s * ox / pz2
        uy = wy - s * oy / pz2
        oxx, oxy = _grad(ox, *J)
        oyx, oyy = _grad(oy, *J)
        opd = opd - 0.5 * (ux * (ux * oxx + uy * oxy)
                           + uy * (ux * oyx + uy * oyy))
    return opd


def trace(presc, x0, y0, L, M, terms=(1, 1, 1), blind=False):
    """The model as a Hamiltonian ray system -- ``Lam -= OPD``,
    ``p -= grad OPD`` at each screen, ``x += t p/pz``, ``Lam += t n^2/pz``
    through each gap -- with the SAME grid-gradient kick the wave gets."""
    surfaces, thick = presc['surfaces'], presc['thicknesses']
    idx = S._resolved_indices(surfaces, LAM)
    x, y = np.array(x0, float), np.array(y0, float)
    px = np.full_like(x, float(L))
    py = np.full_like(y, float(M))
    lam = L * x + M * y
    for i, surf in enumerate(surfaces):
        n1, n2 = idx[i]
        J = _jac(x, y)
        opd = ((n2 - n1) * S._sag_grad(surf, x, y)[0] if blind
               else screen(surf, x, y, px, py, n1, n2, J, terms))
        kx, ky = _grad(opd, *J)
        lam = lam - opd
        px, py = px - kx, py - ky
        if i < len(surfaces) - 1:
            t = float(thick[i])
            pz = np.sqrt(np.maximum(n2 ** 2 - px ** 2 - py ** 2, 1e-300))
            x, y = x + t * px / pz, y + t * py / pz
            lam = lam + t * n2 ** 2 / pz
    return x, y, lam, px, py


def angular_error(presc, x0, y0, L, M, keep, **kw):
    d = []
    for (l_, m_) in ((L, M), (0.0, 0.0)):
        xe, ye, le, _a = S.exact_trace(presc, LAM, x0, y0, l_, m_)
        xm, ym, lm, pxm, pym = trace(presc, x0, y0, l_, m_, **kw)
        d.append(lm + pxm * (xe.reshape(x0.shape) - xm)
                 + pym * (ye.reshape(x0.shape) - ym) - le.reshape(x0.shape))
    g = (d[0] - d[1])[keep]
    A = np.stack([np.ones_like(g), x0[keep], y0[keep]], axis=1)
    c, *_ = np.linalg.lstsq(A, g, rcond=None)
    return float(np.sqrt(((g - A @ c) ** 2).mean())) / LAM


def arm_error(presc, x0, y0, L, M, keep):
    """The TANGENT-FACET ARM of BUILD_R1_WIRING S1.2: the identity OPD with the
    EXACT facet kick ``-dz grad sag``.  Not a wave model -- its kick is not the
    gradient of its own value -- but it is the number route 3 is measured
    against."""
    d = []
    for (l_, m_) in ((L, M), (0.0, 0.0)):
        xe, ye, le, _a = S.exact_trace(presc, LAM, x0, y0, l_, m_)
        xm, ym, lm, pxm, pym = S._facet_arm(presc, x0, y0, l_, m_, True)
        d.append(lm + pxm * (xe.reshape(x0.shape) - xm)
                 + pym * (ye.reshape(x0.shape) - ym) - le.reshape(x0.shape))
    g = (d[0] - d[1])[keep]
    A = np.stack([np.ones_like(g), x0[keep], y0[keep]], axis=1)
    c, *_ = np.linalg.lstsq(A, g, rcond=None)
    return float(np.sqrt(((g - A @ c) ** 2).mean())) / LAM


def group(gi):
    groups_post, tilts, orders, _r = H2.geometry()
    presc = dict(groups_post[gi]['prescription'])
    presc.pop('aperture_diameter', None)
    best = None
    for i, _o in enumerate(orders):
        _xc, _yc, L, M = tilts[gi][i]
        if best is None or np.hypot(L, M) > np.hypot(best[0], best[1]):
            best = (L, M)
    return presc, best[0], best[1]


# ---------------------------------------------------------------------------
# THE REMAP RUNG -- surface_model='tangent_facet_remap'
#
# Route 3 REFERENCES the transverse walk away (its (T3) is the second order of
# that referencing).  This one REPRESENTS it: the element is a screen PLUS a
# coordinate remap, so the exit eikonal is evaluated where the ray actually is.
# The scoring arm below is the SAME Hamiltonian bundle instrument -- same
# common-mode control, same regular bundle, same physical gradients via the
# bundle Jacobian -- with two lines changed: the ray POSITION advances by the
# walk W at each screen, and the momentum becomes the exact refracted p_out
# (which is what the composite screen+remap kick equals identically; see (R3)
# above ``_tangent_facet_remap_screen``).
# ---------------------------------------------------------------------------
def remap_screen(surf, x, y, px, py, n1, n2, J, hit=2, curv=True):
    """(R1)-(R5).  Returns ``(opd, Wx, Wy, p_out_x, p_out_y)``.

    ``hit`` selects how the facet height at the ray's own crossing is found:
    0 = at the vertex plane (the walk's rise ignored), 1 = the linear fixed
    point ``sag/(1-a)`` (EXACT for a plane facet), 2 = plus the curvature term
    (what ships), 3 = a per-pixel Newton driven to convergence, which needs an
    analytic sag off the grid and is the measured-and-not-taken arm.
    ``curv`` takes the facet NORMAL at the crossing rather than at the pixel."""
    s, gx, gy = S._sag_grad(surf, x, y)
    pz1 = np.sqrt(np.maximum(n1 ** 2 - px ** 2 - py ** 2, 1e-300))
    qx, qy = px / pz1, py / pz1
    a = gx * qx + gy * qy
    hxx, hxy = _grad(gx, *J)
    hyx, hyy = _grad(gy, *J)
    b = qx * (qx * hxx + qy * hxy) + qy * (qx * hyx + qy * hyy)
    if hit == 0:
        sh = s
    elif hit == 1:
        sh = s / (1.0 - a)
    else:
        s1 = s / (1.0 - a)
        sh = s1 + 0.5 * b * s1 * s1 / (1.0 - a)
        if hit >= 3:
            for _ in range(2):
                s2, g2x, g2y = S._sag_grad(surf, x + sh * qx, y + sh * qy)
                sh = sh - (sh - s2) / (1.0 - (g2x * qx + g2y * qy))
    wx, wy = sh * qx, sh * qy
    ghx = gx + (wx * hxx + wy * hxy) if curv else gx
    ghy = gy + (wx * hyx + wy * hyy) if curv else gy
    inv = 1.0 / np.sqrt(1.0 + ghx ** 2 + ghy ** 2)
    a_dot = (-ghx * px - ghy * py + pz1) * inv
    b_sq = np.sqrt(np.maximum(n2 ** 2 - n1 ** 2 + a_dot ** 2, 0.0))
    dz = (b_sq - a_dot) * inv
    pox, poy = px - dz * ghx, py - dz * ghy
    pz2 = pz1 + dz
    return (sh * (n2 ** 2 / pz2 - n1 ** 2 / pz1),
            sh * (qx - pox / pz2), sh * (qy - poy / pz2), pox, poy)


def trace_remap(presc, x0, y0, L, M, **kw):
    surfaces, thick = presc['surfaces'], presc['thicknesses']
    idx = S._resolved_indices(surfaces, LAM)
    x, y = np.array(x0, float), np.array(y0, float)
    px = np.full_like(x, float(L))
    py = np.full_like(y, float(M))
    lam = L * x + M * y
    for i, surf in enumerate(surfaces):
        n1, n2 = idx[i]
        J = _jac(x, y)
        opd, wx, wy, pox, poy = remap_screen(surf, x, y, px, py, n1, n2, J,
                                             **kw)
        lam = lam - opd
        x, y = x + wx, y + wy          # <- the walk, REPRESENTED
        px, py = pox, poy              # <- (R3): the composite kick
        if i < len(surfaces) - 1:
            t = float(thick[i])
            pz = np.sqrt(np.maximum(n2 ** 2 - px ** 2 - py ** 2, 1e-300))
            x, y = x + t * px / pz, y + t * py / pz
            lam = lam + t * n2 ** 2 / pz
    return x, y, lam, px, py


def remap_error(presc, x0, y0, L, M, keep, **kw):
    d = []
    for (l_, m_) in ((L, M), (0.0, 0.0)):
        xe, ye, le, _a = S.exact_trace(presc, LAM, x0, y0, l_, m_)
        xm, ym, lm, pxm, pym = trace_remap(presc, x0, y0, l_, m_, **kw)
        d.append(lm + pxm * (xe.reshape(x0.shape) - xm)
                 + pym * (ye.reshape(x0.shape) - ym) - le.reshape(x0.shape))
    g = (d[0] - d[1])[keep]
    A = np.stack([np.ones_like(g), x0[keep], y0[keep]], axis=1)
    c, *_ = np.linalg.lstsq(A, g, rcond=None)
    return float(np.sqrt(((g - A @ c) ** 2).mean())) / LAM


# ---------------------------------------------------------------------------
# (a) the closed form vs exact plane-facet ray algebra
# ---------------------------------------------------------------------------
def mode_facet(verbose=True):
    """The identity is EXACT for a plane facet -- checked against ``S_in -
    S_out`` from exact ray algebra, not against a ray trace."""
    from lumenairy.elements._lens_real import _tangent_facet_screen
    n, s0 = 129, 2.3e-4
    xs = np.linspace(-1.5e-3, 1.5e-3, n)
    X, _Y = np.meshgrid(xs, xs)
    dx = float(xs[1] - xs[0])
    sl = (slice(2, -2), slice(2, -2))
    rows = []
    for slope in (0.05, 0.12, 0.24):
        for (n1, n2) in ((1.0, 1.8047), (1.8047, 1.0), (1.5917, 1.8047)):
            for tilt in (0.0, 0.055, 0.15):
                p = np.full_like(X, n1 * tilt)
                got, ok = _tangent_facet_screen(
                    s0 + slope * X, np.full_like(X, slope), np.zeros_like(X),
                    p, np.zeros_like(p), n1, n2, dx, dx, np)
                pz1 = np.sqrt(n1 ** 2 - (n1 * tilt) ** 2)
                inv = 1.0 / np.sqrt(1.0 + slope ** 2)
                a = -slope * inv * n1 * tilt + inv * pz1
                b = np.sqrt(n2 ** 2 - n1 ** 2 + a ** 2)
                po = n1 * tilt + (b - a) * (-slope * inv)
                pz2 = pz1 + (b - a) * inv
                z = (s0 + slope * X) / (1.0 - slope * n1 * tilt / pz1)
                x1 = X + z * n1 * tilt / pz1 - z * po / pz2
                c = (n1 * tilt * X + z * (n1 ** 2 / pz1 - n2 ** 2 / pz2)
                     - po * x1)
                want = n1 * tilt * X - (po * X + c)
                rel = float((np.abs(got[sl] - want[sl])
                             / np.abs(want[sl])).max())
                rows.append({'slope': slope, 'n1': n1, 'n2': n2,
                             'tilt_mrad': tilt * 1e3, 'rel_err': rel,
                             'all_ok': bool(np.all(ok))})
                if verbose:
                    print('  slope %.2f  n %6.4f->%6.4f  %5.1f mrad : '
                          'rel err %9.3e' % (slope, n1, n2, tilt * 1e3, rel))
    worst = max(r['rel_err'] for r in rows)
    if verbose:
        print('  WORST relative error over %d cells: %.3e' % (len(rows), worst))
    return {'rows': rows, 'worst_rel_err': worst}


# ---------------------------------------------------------------------------
# (b) design 121's four powered groups
# ---------------------------------------------------------------------------
def mode_d121(verbose=True):
    rows = []
    for gi in (2, 3, 4, 5):
        presc, L, M = group(gi)
        for rad in (1e-3, 2e-3, 3e-3):
            X, Y, keep = bundle(rad, 65)
            r = {'group': gi, 'radius_mm': rad * 1e3,
                 'tilt_mrad': float(1e3 * np.hypot(L, M)),
                 'blind': angular_error(presc, X, Y, L, M, keep, blind=True),
                 'route3': angular_error(presc, X, Y, L, M, keep),
                 'facet_arm': arm_error(presc, X, Y, L, M, keep)}
            rows.append(r)
            if verbose:
                print('  g%d %.0f mm  blind %10.6f  ROUTE 3 %11.7f  '
                      'arm %11.7f  (%6.0fx over blind)'
                      % (gi, rad * 1e3, r['blind'], r['route3'],
                         r['facet_arm'], r['blind'] / max(r['route3'], 1e-30)))
    return {'rows': rows}


# ---------------------------------------------------------------------------
# (c) the term ladder -- what each piece of the closed form buys
# ---------------------------------------------------------------------------
def mode_ladder(verbose=True):
    rows = []
    for gi in (5, 2, 3, 4):
        presc, L, M = group(gi)
        X, Y, keep = bundle(3e-3, 65)
        r = {'group': gi,
             'T1_identity': angular_error(presc, X, Y, L, M, keep,
                                          terms=(0, 0, 0)),
             'T1_T2': angular_error(presc, X, Y, L, M, keep, terms=(1, 1, 0)),
             'T1_T2_T3': angular_error(presc, X, Y, L, M, keep),
             'facet_arm': arm_error(presc, X, Y, L, M, keep)}
        rows.append(r)
        if verbose:
            print('  g%d  (T1) %11.7f  +(T2) %11.7f  +(T3) %11.7f  '
                  'arm %11.7f' % (gi, r['T1_identity'], r['T1_T2'],
                                  r['T1_T2_T3'], r['facet_arm']))
    return {'rows': rows}


# ---------------------------------------------------------------------------
# (d) THE REMAP RUNG -- the acceptance table, and its term ladder
# ---------------------------------------------------------------------------
def mode_remap(verbose=True):
    """The remap arm on the four powered groups x the three pupils, against
    route 3 and against the tangent-facet ARM (BUILD_TANGENT_FACET's prize)."""
    rows = []
    for gi in (2, 3, 4, 5):
        presc, L, M = group(gi)
        for rad in (1e-3, 2e-3, 3e-3):
            X, Y, keep = bundle(rad, 65)
            r = {'group': gi, 'radius_mm': rad * 1e3,
                 'tilt_mrad': float(1e3 * np.hypot(L, M)),
                 'blind': angular_error(presc, X, Y, L, M, keep, blind=True),
                 'route3': angular_error(presc, X, Y, L, M, keep),
                 'remap': remap_error(presc, X, Y, L, M, keep),
                 'facet_arm': arm_error(presc, X, Y, L, M, keep)}
            rows.append(r)
            if verbose:
                print('  g%d %.0f mm  blind %10.6f  route3 %11.7f  '
                      'REMAP %11.4e  arm %11.7f'
                      % (gi, rad * 1e3, r['blind'], r['route3'], r['remap'],
                         r['facet_arm']))
    return {'rows': rows}


def mode_remap_ladder(verbose=True):
    """What each piece of the remap buys, and where its residual comes from.

    ``hit0`` samples the facet at the vertex plane; ``hit1`` solves the hit
    point's linear fixed point (exact for a plane); ``nocurv`` keeps the facet
    NORMAL at the pixel instead of at the crossing; ``hit2`` is what ships;
    ``hit3`` drives the fixed point to convergence with a per-pixel Newton and
    is the measured-and-not-taken bound on the shipped truncation."""
    rows = []
    for gi in (5, 2, 3, 4):
        presc, L, M = group(gi)
        X, Y, keep = bundle(3e-3, 65)
        r = {'group': gi,
             'route3': angular_error(presc, X, Y, L, M, keep),
             'hit0': remap_error(presc, X, Y, L, M, keep, hit=0),
             'hit1': remap_error(presc, X, Y, L, M, keep, hit=1),
             'hit2_nocurv': remap_error(presc, X, Y, L, M, keep, hit=2,
                                        curv=False),
             'hit2': remap_error(presc, X, Y, L, M, keep, hit=2),
             'hit3_newton': remap_error(presc, X, Y, L, M, keep, hit=3)}
        rows.append(r)
        if verbose:
            print('  g%d  route3 %10.3e  hit0 %10.3e  hit1 %10.3e  '
                  'hit2/nocurv %10.3e  HIT2 %10.3e  hit3 %10.3e'
                  % (gi, r['route3'], r['hit0'], r['hit1'], r['hit2_nocurv'],
                     r['hit2'], r['hit3_newton']))
    return {'rows': rows}


def mode_remap_facet(verbose=True):
    """The remap screen is EXACT for a plane facet -- NO ORACLE.

    For a tilted plane facet under a plane wave both eikonals are closed form,
    so the pair (OPD, W) a wave model must apply is known exactly: the screen
    is ``S_in(x) - S_out(x + W)`` and the walk is where the ray lands.  The
    library's own ``_tangent_facet_remap_screen`` is scored against that, on
    the same 27 cells route 3's ``facet`` mode uses.  There is no tolerance to
    choose: the identity is right to 1e-13 relative or it is wrong."""
    from lumenairy.elements._lens_real import _tangent_facet_remap_screen
    n, s0 = 129, 2.3e-4
    xs = np.linspace(-1.5e-3, 1.5e-3, n)
    X, _Y = np.meshgrid(xs, xs)
    sl = (slice(2, -2), slice(2, -2))
    zero = np.zeros_like(X)
    rows = []
    for slope in (0.05, 0.12, 0.24):
        for (n1, n2) in ((1.0, 1.8047), (1.8047, 1.0), (1.5917, 1.8047)):
            for tilt in (0.0, 0.055, 0.15):
                p = n1 * tilt
                pv = np.full_like(X, p)
                got, gwx, gwy, gox, _goy, ok = _tangent_facet_remap_screen(
                    s0 + slope * X, np.full_like(X, slope), zero,
                    zero, zero, zero, zero, pv, zero, n1, n2, np)
                # exact plane-facet algebra
                pz1 = np.sqrt(n1 ** 2 - p ** 2)
                q = p / pz1
                sh = (s0 + slope * X) / (1.0 - slope * q)
                inv = 1.0 / np.sqrt(1.0 + slope ** 2)
                a_ = (-slope * p + pz1) * inv
                b_ = np.sqrt(n2 ** 2 - n1 ** 2 + a_ ** 2)
                dz = (b_ - a_) * inv
                po = p - dz * slope
                pz2 = pz1 + dz
                w_want = sh * (q - po / pz2)
                x1 = X + w_want
                c = p * X + sh * (n1 ** 2 / pz1 - n2 ** 2 / pz2) - po * x1
                opd_want = p * X - (po * x1 + c)
                e_opd = float((np.abs(got[sl] - opd_want[sl])
                               / np.abs(opd_want[sl])).max())
                e_w = float((np.abs(gwx[sl] - w_want[sl])
                             / np.maximum(np.abs(w_want[sl]), 1e-30)).max())
                e_p = float(np.abs(gox[sl] - po).max() / max(abs(po), 1e-30))
                rows.append({'slope': slope, 'n1': n1, 'n2': n2,
                             'tilt_mrad': tilt * 1e3, 'rel_err_opd': e_opd,
                             'rel_err_walk': e_w, 'rel_err_pout': e_p,
                             'walk_y_zero': bool(np.all(gwy == 0.0)),
                             'all_ok': bool(np.all(ok))})
                if verbose:
                    print('  slope %.2f  n %6.4f->%6.4f  %5.1f mrad : '
                          'OPD %9.3e  W %9.3e  p_out %9.3e'
                          % (slope, n1, n2, tilt * 1e3, e_opd, e_w, e_p))
    worst = max(max(r['rel_err_opd'], r['rel_err_walk'], r['rel_err_pout'])
                for r in rows)
    if verbose:
        print('  WORST relative error over %d cells: %.3e' % (len(rows), worst))
    return {'rows': rows, 'worst_rel_err': worst}


_MODES = {'facet': mode_facet, 'd121': mode_d121, 'ladder': mode_ladder,
          'remap': mode_remap, 'remap_ladder': mode_remap_ladder,
          'remap_facet': mode_remap_facet}


def main(argv):
    what = (argv[1:] or ['all'])[0]
    names = list(_MODES) if what == 'all' else [what]
    out = {}
    for name in names:
        print('== %s ==' % name)
        out[name] = _MODES[name]()
    path = os.path.join(_HERE, '_tangent_facet_%s.json'
                        % ('all' if what == 'all' else what))
    with open(path, 'w', encoding='ascii') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print('wrote %s' % path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
