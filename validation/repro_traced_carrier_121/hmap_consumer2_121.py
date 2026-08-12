# THE ANGLE-AWARE ANALYTIC LENS -- acceptance against the traced oracle.
#
# LOCAL-ONLY (needs the 121 .zmx via _d121_common).  Drives the SHIPPED library
# only: nothing here reimplements the map, and the oracle is the exact ray
# trace apply_real_lens_traced itself is built on.
#
# MODES
#   gap     -- reproduce PROTO_HAMILTON_MAP_2026_08_11 S7: how many waves the
#              ANGLE-BLIND screen loses, per group, per pupil radius.  This is
#              the quantity the feature exists to recover.
#   map     -- the shipped CongruenceMap's OWN error against direct ray traces
#              on exactly those cases, plus the plane-plate exact-zero /
#              piston property.
#   degree  -- the pupil-degree ladder behind _HMAP_PUPIL_DEGREE.
#   field   -- the FIELD-level arm: apply_real_lens_traced as the oracle
#              against apply_real_lens with and without a carrier.
#   nodes   -- the angular node ladder on a genuinely 2-D angular box (the
#              design's own finite-R congruence at the last group).
#   all     -- every mode above.
#
# Run:  python hmap_consumer2_121.py all
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _d121_common as C  # noqa: E402
import hmap_screen_proto as HM  # noqa: E402

import lumenairy as la  # noqa: E402
from lumenairy.elements import apply_real_lens, apply_real_lens_traced  # noqa: E402
from lumenairy.elements._lens_traced import _tilted_carrier_parts  # noqa: E402
from lumenairy.glass import get_glass_index  # noqa: E402
from lumenairy.propagators.carrier import (  # noqa: E402
    _chain_chief_ray_at_target,
    _paraxial_group_r_out,
)
from lumenairy.raytrace import _make_bundle, surfaces_from_prescription, trace  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
LAM = C.LAM
RADII = (1e-3, 2e-3, 3e-3)
N_PUP = 65


# ---------------------------------------------------------------------------
def _surfaces(groups_post, gi):
    presc = dict(groups_post[gi]['prescription'])
    presc.pop('aperture_diameter', None)
    surfs = surfaces_from_prescription(presc)
    n_exit = get_glass_index(surfs[-1].glass_after, LAM)
    return surfs, n_exit


def _characteristic(surfs, n_exit, px, py, thx, thy):
    """The oracle: the group's traced point characteristic, entrance plane to
    exit VERTEX plane, exactly as apply_real_lens_traced computes it."""
    rays = _make_bundle(x=np.ravel(px), y=np.ravel(py),
                        L=np.ravel(thx), M=np.ravel(thy), wavelength=LAM)
    fin = trace(rays, surfs, LAM, output_filter='last').image_rays
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(fin.alive & (np.abs(fin.N) > 1e-30), -fin.z / fin.N, 0.0)
    return (fin.x + fin.L * t, fin.y + fin.M * t, fin.opd + n_exit * t,
            np.asarray(fin.alive, dtype=bool))


def _disc(rad, n=N_PUP):
    t = np.linspace(-1.0, 1.0, n)
    PX, PY = np.meshgrid(rad * t, rad * t, indexing='ij')
    keep = (PX ** 2 + PY ** 2) <= rad ** 2
    return PX[keep], PY[keep]


def _decompose(px, py, d):
    """PISTON / TILT / RESIDUAL of an OPL map over a disc, in waves -- the
    proto's own least-squares decomposition, verbatim."""
    A = np.stack([np.ones_like(px), px, py], axis=1)
    A4 = np.stack([np.ones_like(px), px, py, px ** 2 + py ** 2], axis=1)
    c, *_ = np.linalg.lstsq(A, d, rcond=None)
    res = d - A @ c
    c4, *_ = np.linalg.lstsq(A4, d, rcond=None)
    res4 = d - A4 @ c4
    return {'piston': float(c[0]) / LAM,
            'tilt': float(np.abs(A[:, 1:] @ c[1:]).max()) / LAM,
            'resid': float(np.sqrt((res ** 2).mean())) / LAM,
            'resid4': float(np.sqrt((res4 ** 2).mean())) / LAM}


def geometry():
    """(groups_post, chief-ray table per group, R at each group's entrance)."""
    _pre, groups_post, _gap, period = C.geometry()
    mx, my, _amps = C.order_table(period)
    _env, R_doe, _dxd, _pin = C.chain_a()
    tilts, r_ent = [], []
    R_run = float(R_doe)
    for gi in range(len(groups_post)):
        head = groups_post[:gi]
        gap = float(groups_post[gi]['gap_before'])
        row = []
        for i in range(len(mx)):
            car = la.TiltedCarrier(float(R_doe), float(mx[i]) * LAM / period,
                                   float(my[i]) * LAM / period)
            row.append(_chain_chief_ray_at_target(head, LAM, car, gap,
                                                  'hmap_consumer2_121'))
        tilts.append(row)
        if gi:
            g = groups_post[gi - 1]
            R_run = float(_paraxial_group_r_out(
                g['prescription'], R_run + float(g['gap_before']), LAM))
        r_ent.append(R_run + gap)
    return groups_post, tilts, list(zip(mx, my)), r_ent


# ---------------------------------------------------------------------------
# gap -- the proto's S7, reproduced
# ---------------------------------------------------------------------------
def mode_gap(verbose=True):
    groups_post, tilts, orders, _r = geometry()
    out = []
    for gi in range(len(groups_post)):
        surfs, n_exit = _surfaces(groups_post, gi)
        for rad in RADII:
            px, py = _disc(rad)
            _x0, _y0, opl0, al0 = _characteristic(
                surfs, n_exit, px, py, np.zeros_like(px), np.zeros_like(py))
            if not al0.all():
                continue
            worst, worst_i = None, None
            for i, (mm, nn) in enumerate(orders):
                _xc, _yc, L, M = tilts[gi][i]
                _x, _y, opl, al = _characteristic(
                    surfs, n_exit, px, py, np.full_like(px, L),
                    np.full_like(py, M))
                if not al.all():
                    continue
                dec = _decompose(px, py, opl - opl0)
                if worst is None or dec['resid'] > worst['resid']:
                    worst, worst_i = dec, (int(mm), int(nn), L, M)
            if worst is None:
                continue
            out.append({'group': gi, 'radius_mm': rad * 1e3,
                        'order': list(worst_i[:2]),
                        'tilt_mrad': float(np.hypot(worst_i[2],
                                                    worst_i[3]) * 1e3),
                        **worst})
            if verbose:
                print('  group %d r=%.1f mm  worst %-9s %6.2f mrad:  '
                      'piston %10.3f w  tilt %9.3f w  RESID %9.4f w  '
                      '+defocus %9.4f w'
                      % (gi, rad * 1e3, str(tuple(worst_i[:2])),
                         out[-1]['tilt_mrad'], worst['piston'], worst['tilt'],
                         worst['resid'], worst['resid4']))
    return {'rows': out}


# ---------------------------------------------------------------------------
# map -- the shipped object against the same oracle
# ---------------------------------------------------------------------------
def _map_delta(prescription, carrier, px, py):
    """The SHIPPED module's correction at the probe points."""
    def angle_fn(qx, qy):
        _w, l, m = _tilted_carrier_parts(carrier, np.asarray(qx, dtype=float),
                                         np.asarray(qy, dtype=float))
        return l, m
    L, M = angle_fn(px, py)
    d, hmap = HM.angle_aware_delta_opl(
        prescription, LAM, px, py, L, M, float(carrier.R), angle_fn,
        where='hmap_consumer2_121')
    return d, hmap


def mode_map(verbose=True):
    groups_post, tilts, orders, _r = geometry()
    HM.clear_congruence_map_cache()
    out = []
    for gi in range(len(groups_post)):
        surfs, n_exit = _surfaces(groups_post, gi)
        presc = groups_post[gi]['prescription']
        for rad in RADII:
            px, py = _disc(rad)
            _x0, _y0, opl0, al0 = _characteristic(
                surfs, n_exit, px, py, np.zeros_like(px), np.zeros_like(py))
            if not al0.all():
                continue
            worst, worst_i, worst_d = None, None, None
            for i, (mm, nn) in enumerate(orders):
                _xc, _yc, L, M = tilts[gi][i]
                _x, _y, opl, al = _characteristic(
                    surfs, n_exit, px, py, np.full_like(px, L),
                    np.full_like(py, M))
                if not al.all():
                    continue
                dec = _decompose(px, py, opl - opl0)
                if worst is None or dec['resid'] > worst['resid']:
                    worst, worst_i = dec, (int(mm), int(nn), L, M)
                    worst_d = opl - opl0
            if worst is None:
                continue
            car = la.TiltedCarrier(float('inf'), worst_i[2], worst_i[3])
            t0 = time.perf_counter()
            got, hmap = _map_delta(presc, car, px, py)
            dt = time.perf_counter() - t0
            if got is None:
                out.append({'group': gi, 'radius_mm': rad * 1e3,
                            'refused': True})
                if verbose:
                    print('  group %d r=%.1f mm  REFUSED' % (gi, rad * 1e3))
                continue
            err = got - worst_d
            edec = _decompose(px, py, err)
            row = {'group': gi, 'radius_mm': rad * 1e3,
                   'order': list(worst_i[:2]),
                   'blind_resid_waves': worst['resid'],
                   'blind_piston_waves': worst['piston'],
                   'blind_tilt_waves': worst['tilt'],
                   'map_err_max_waves': float(np.abs(err).max()) / LAM,
                   'map_err_rms_waves': float(np.sqrt((err ** 2).mean())) / LAM,
                   'map_err_resid_waves': edec['resid'],
                   'map_piston_waves': float(_decompose(px, py,
                                                        got)['piston']),
                   'map_nodes': list(hmap.nodes),
                   'map_kB': hmap.nbytes / 1024.0,
                   'map_rays': hmap.n_rays,
                   'map_G7_waves': hmap.check_max_waves,
                   'map_detJ_range': hmap.det_j_range,
                   'seconds': dt, 'refused': False}
            out.append(row)
            if verbose:
                print('  group %d r=%.1f mm  %-9s  BLIND resid %8.4f w  ->  '
                      'MAP err max %.3e w  rms %.3e w  (piston %10.3f vs '
                      '%10.3f w)  nodes %s  %.1f kB  %.3f s'
                      % (gi, rad * 1e3, str(tuple(worst_i[:2])),
                         worst['resid'], row['map_err_max_waves'],
                         row['map_err_rms_waves'], row['map_piston_waves'],
                         worst['piston'], hmap.nodes, row['map_kB'], dt))
    return {'rows': out}


# ---------------------------------------------------------------------------
# degree -- the pupil-degree ladder
# ---------------------------------------------------------------------------
def mode_degree(gi=5, rad=3e-3, degrees=(4, 6, 8, 10, 12), verbose=True):
    groups_post, tilts, orders, _r = geometry()
    surfs, n_exit = _surfaces(groups_post, gi)
    presc = groups_post[gi]['prescription']
    px, py = _disc(rad)
    _x0, _y0, opl0, _a0 = _characteristic(
        surfs, n_exit, px, py, np.zeros_like(px), np.zeros_like(py))
    best, worst_d = None, None
    for i, _o in enumerate(orders):
        _xc, _yc, L, M = tilts[gi][i]
        _x, _y, opl, al = _characteristic(surfs, n_exit, px, py,
                                          np.full_like(px, L),
                                          np.full_like(py, M))
        if not al.all():
            continue
        dec = _decompose(px, py, opl - opl0)
        if best is None or dec['resid'] > best[0]:
            best, worst_d = (dec['resid'], L, M), opl - opl0
    car = la.TiltedCarrier(float('inf'), best[1], best[2])

    def angle_fn(qx, qy):
        _w, l, m = _tilted_carrier_parts(car, np.asarray(qx, dtype=float),
                                         np.asarray(qy, dtype=float))
        return l, m
    L, M = angle_fn(px, py)
    rows = []
    for deg in degrees:
        HM.clear_congruence_map_cache()
        t0 = time.perf_counter()
        got, hmap = HM.angle_aware_delta_opl(
            presc, LAM, px, py, L, M, float(car.R), angle_fn,
            pupil_degree=deg, where='hmap_consumer2_121')
        dt = time.perf_counter() - t0
        if got is None:
            rows.append({'degree': deg, 'refused': True})
            if verbose:
                print('  degree %2d: REFUSED' % deg)
            continue
        err = float(np.abs(got - worst_d).max()) / LAM
        rows.append({'degree': deg, 'err_max_waves': err,
                     'kB': hmap.nbytes / 1024.0, 'seconds': dt,
                     'refused': False})
        if verbose:
            print('  degree %2d: err max %.4e waves   %.1f kB   %.3f s'
                  % (deg, err, hmap.nbytes / 1024.0, dt))
    return {'group': gi, 'radius_mm': rad * 1e3, 'rows': rows}


# ---------------------------------------------------------------------------
# nodes -- the ANGULAR ladder, on the design's own finite-R congruence
# ---------------------------------------------------------------------------
def mode_nodes(gi=5, rad=3e-3, ladder=((1, 1), (3, 3), (4, 4), (5, 4),
                                       (6, 5), (7, 5), (9, 9)), verbose=True):
    groups_post, tilts, orders, r_ent = geometry()
    surfs, n_exit = _surfaces(groups_post, gi)
    presc = groups_post[gi]['prescription']
    px, py = _disc(rad)
    _x0, _y0, opl0, _a0 = _characteristic(
        surfs, n_exit, px, py, np.zeros_like(px), np.zeros_like(py))
    R = float(r_ent[gi])
    # Pick the EXTREME-TILT order, not the largest blind residual: at a finite
    # R the on-axis order (0,0) IS the reference sphere, so its reduced-angle
    # box is identically degenerate while its blind residual (the whole
    # convergence obliquity) is the largest in the fan.  Selecting on the
    # residual therefore picks the one congruence that exercises NO angular
    # axis, which is the opposite of what this ladder is for.
    best = None
    for i, _o in enumerate(orders):
        x_c, y_c, L, M = tilts[gi][i]
        car = la.TiltedCarrier(R, L, M, x_c, y_c)
        _w, Lg, Mg = _tilted_carrier_parts(car, px, py)
        _x, _y, opl, al = _characteristic(surfs, n_exit, px, py, Lg, Mg)
        if not al.all():
            continue
        d = opl - opl0
        dec = _decompose(px, py, d)
        score = float(np.hypot(L, M))
        if best is None or score > best[0]:
            best = (score, car, d, dec['resid'])
    if best is None:
        return {'rows': []}
    car, truth = best[1], best[2]

    def angle_fn(qx, qy):
        _w, l, m = _tilted_carrier_parts(car, np.asarray(qx, dtype=float),
                                         np.asarray(qy, dtype=float))
        return l, m
    L, M = angle_fn(px, py)
    rx, ry = HM._reference_sphere_gradient(R, px, py)
    if verbose:
        print('  congruence %s' % (car,))
        print('  reduced-angle spread over the %.1f mm disc: '
              'sx %.4e  sy %.4e rad'
              % (rad * 1e3, float(np.ptp(L - rx)), float(np.ptp(M - ry))))
        print('  angle-blind residual on this congruence: %.4f waves'
              % best[3])
    rows = []
    for nd in ladder:
        HM.clear_congruence_map_cache()
        t0 = time.perf_counter()
        got, hmap = HM.angle_aware_delta_opl(
            presc, LAM, px, py, L, M, R, angle_fn, nodes=nd,
            where='hmap_consumer2_121')
        dt = time.perf_counter() - t0
        if got is None:
            rows.append({'nodes': list(nd), 'refused': True})
            if verbose:
                print('  nodes %s: REFUSED' % (nd,))
            continue
        err = float(np.abs(got - truth).max()) / LAM
        rows.append({'nodes': list(nd), 'err_max_waves': err,
                     'kB': hmap.nbytes / 1024.0, 'rays': hmap.n_rays,
                     'seconds': dt, 'refused': False})
        if verbose:
            bar = ('PASS(3x)' if err <= 1e-2 / 3 else
                   ('inside' if err <= 1e-2 else 'FAIL'))
            print('  nodes %-6s = %2d: err max %.4e waves  %5.1f kB  '
                  '%6d rays  %.3f s  %s'
                  % (str(nd), nd[0] * nd[1], err, hmap.nbytes / 1024.0,
                     hmap.n_rays, dt, bar))
    return {'blind_resid_waves': best[3], 'rows': rows}


# ---------------------------------------------------------------------------
# field -- apply_real_lens_traced AS THE ORACLE, on a propagated field
# ---------------------------------------------------------------------------
def _apply_proto_screen(E, presc, car, X, Y):
    """The REVERTED wiring, reproduced locally: multiply the shipped analytic
    exit field by exp(+i k0 dOPL) with dOPL from the prototype map.  Kept as a
    local helper because the library edit that did this was reverted."""
    def angle_fn(qx, qy):
        _w, l, m = _tilted_carrier_parts(car, np.asarray(qx, dtype=float),
                                         np.asarray(qy, dtype=float))
        return l, m
    L, M = angle_fn(X, Y)
    d, hmap = HM.angle_aware_delta_opl(presc, LAM, X, Y, L, M, float(car.R),
                                       angle_fn, where='hmap_consumer2_121')
    if d is None:
        return E
    return E * np.exp(2j * np.pi / LAM * d)


def _wavefront_gap(Ea, Etr, X, Y, mask):
    """``phase(arm) - phase(traced)`` over ``mask``, piston and tilt removed,
    in waves.  Returned as the FIELD, not a scalar, so two tilts can be
    differenced pointwise."""
    ph = np.angle(Ea * np.conj(Etr))[mask] / (2.0 * np.pi)
    px, py = X[mask], Y[mask]
    A = np.stack([np.ones_like(px), px, py], axis=1)
    c, *_ = np.linalg.lstsq(A, ph, rcond=None)
    return ph - A @ c


def mode_field(gi=5, n=512, dx=12.0e-6, w=2.5e-3, r_probe=1.5e-3,
               verbose=True):
    """THE FIELD-LEVEL ARM, with the common-mode control that makes it mean
    something.

    A raw ``analytic vs traced`` wavefront comparison CANNOT resolve this
    feature: the two models differ by far more than the angle term (different
    amplitude model, different exit-support treatment, a distributed OPL versus
    a stack of screens), and that difference is present at normal incidence
    too.  So measure the ANGULAR PART by differencing:

        D_arm(theta) = phase(arm at theta) - phase(traced at theta)
        angular error of the arm = D_arm(theta) - D_arm(0)

    Everything angle-independent -- which is everything that separates the two
    models at normal incidence -- cancels in the difference, and what is left
    is exactly the quantity S7 sized and this feature exists to remove.  Both
    tilts are probed on the SAME axis-centred disc, with a beam wide enough to
    fill it at both, so the difference is pointwise."""
    groups_post, tilts, orders, r_ent = geometry()
    presc = groups_post[gi]['prescription']
    best = None
    for i, (mm, nn) in enumerate(orders):
        x_c, y_c, L, M = tilts[gi][i]
        if best is None or np.hypot(L, M) > np.hypot(best[3], best[4]):
            best = (int(mm), int(nn), x_c, y_c, L, M)
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x)
    mask = (X ** 2 + Y ** 2) <= r_probe ** 2

    gaps, secs = {}, {}
    for tag, (L, M) in (('zero', (0.0, 0.0)),
                        ('tilt', (best[4], best[5]))):
        car = la.TiltedCarrier(float('inf'), L, M)
        W, _l, _m = _tilted_carrier_parts(car, X, Y)
        E = (np.exp(-(X ** 2 + Y ** 2) / w ** 2)
             * np.exp(1j * 2 * np.pi / LAM * W)).astype(np.complex128)
        kw = dict(prescription=presc, wavelength=LAM, dx=dx)
        t0 = time.perf_counter()
        E_blind = apply_real_lens(E, **kw)
        t1 = time.perf_counter()
        E_aware = _apply_proto_screen(E_blind, presc, car, X, Y)
        t2 = time.perf_counter()
        E_tr = apply_real_lens_traced(
            E, prescription=presc, wavelength=LAM, dx=dx, carrier=car,
            tilt_aware_rays=True, on_noncollimated='off')
        t3 = time.perf_counter()
        gaps[tag] = {'blind': _wavefront_gap(E_blind, E_tr, X, Y, mask),
                     'aware': _wavefront_gap(E_aware, E_tr, X, Y, mask)}
        secs[tag] = {'blind': t1 - t0, 'aware': t2 - t1, 'traced': t3 - t2}

    out = {'group': gi, 'n': n, 'dx_um': dx * 1e6, 'w_mm': w * 1e3,
           'r_probe_mm': r_probe * 1e3, 'order': [best[0], best[1]],
           'tilt_mrad': float(np.hypot(best[4], best[5]) * 1e3),
           'n_mask': int(mask.sum()), 'seconds': secs}
    for arm in ('blind', 'aware'):
        d = gaps['tilt'][arm] - gaps['zero'][arm]
        d = d - d.mean()
        out[arm] = {
            'angular_rms_waves': float(np.sqrt((d ** 2).mean())),
            'angular_max_waves': float(np.abs(d).max()),
            'raw_zero_rms_waves': float(np.sqrt(
                (gaps['zero'][arm] ** 2).mean())),
            'raw_tilt_rms_waves': float(np.sqrt(
                (gaps['tilt'][arm] ** 2).mean()))}
        if verbose:
            print('  %-6s: raw vs traced %.4f w rms at 0 mrad, %.4f w rms at '
                  '%.2f mrad  ->  ANGULAR part %.5f w rms / %.5f w max'
                  % (arm, out[arm]['raw_zero_rms_waves'],
                     out[arm]['raw_tilt_rms_waves'], out['tilt_mrad'],
                     out[arm]['angular_rms_waves'],
                     out[arm]['angular_max_waves']))
    if verbose:
        print('  ANGULAR-ERROR RECOVERY (blind rms / aware rms): %.1f x'
              % (out['blind']['angular_rms_waves']
                 / max(out['aware']['angular_rms_waves'], 1e-30)))
        print('  timings (tilt arm) blind %.2f s  aware %.2f s  traced %.2f s'
              % (secs['tilt']['blind'], secs['tilt']['aware'],
                 secs['tilt']['traced']))
    return out


# ---------------------------------------------------------------------------
# plate -- THE REFUTATION, against a CLOSED FORM
# ---------------------------------------------------------------------------
def mode_plate(N=256, dx=20.0e-6, thickness=25.4e-3, glass='N-BK7',
               verbose=True):
    """The decisive test, and it needs no oracle but arithmetic.

    A plane-parallel plate has zero sag, so ``apply_real_lens`` reduces to a
    single in-glass ANGULAR-SPECTRUM step and its exit field for an incident
    plane wave ``exp(i k0 L x)`` is EXACTLY

        exp(i k0 L x) * exp(i k0 t sqrt(n^2 - L^2)),

    because the transverse frequency is preserved across both flat interfaces.
    The ray picture gives the same answer: the ray entering at ``x0`` carries
    ``n t / cos(theta_t)`` of optical path and exits at
    ``x1 = x0 + t tan(theta_t)``, and substituting ``x0 = x1 - t tan(theta_t)``
    with ``sin(theta_i) = n sin(theta_t)`` turns the entrance-referenced sum
    into ``k0 L x1 + k0 n t cos(theta_t)`` -- the SAME constant.  **The
    transverse walk-off and the entrance-referenced path length are not two
    separate terms an angle-blind screen misses; they are one term, and the
    angular spectrum already carries it exactly.**

    So the S7 quantity ``OPL(theta) - OPL(0)`` at fixed ENTRANCE point --
    ``n t (1/cos(theta_t) - 1)``, POSITIVE -- is the mirror image of the true
    exit-referenced differential ``n t (cos(theta_t) - 1)``, NEGATIVE.  Adding
    it to a model that is already exact is wrong by twice the term.

    THE TILT MUST BE GRID-EXACT AND BELOW NYQUIST or the measurement is an
    aliasing artefact rather than a model reading: ``L = m lambda / (N dx)``
    with ``m <= N/2``.  (Taking ``m = 162`` at ``N = 256`` -- the design's own
    41.5 mrad -- reads a spurious 0.35-wave residual on the SHIPPED model,
    which is the grid folding the plane wave, not the physics.)"""
    k0 = 2.0 * np.pi / LAM
    n = float(get_glass_index(glass, LAM))
    presc = {'surfaces': [
        {'radius': np.inf, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': 'AIR', 'glass_after': glass},
        {'radius': np.inf, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': glass, 'glass_after': 'AIR'}],
        'thicknesses': [thickness], 'name': 'plate'}
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    core = np.hypot(X, Y) <= 1.5e-3
    rows = []
    for m in (0, 40, 81, 128):
        L = m * LAM / (N * dx)
        E = np.exp(1j * k0 * L * X).astype(np.complex128)
        E_blind = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx)
        E_aware = _apply_proto_screen(
            E_blind, presc, la.TiltedCarrier(float('inf'), L, 0.0), X, Y)
        exact = k0 * thickness * np.sqrt(n ** 2 - L ** 2)

        def frac(v):
            r = v / (2.0 * np.pi)
            return float(r - np.round(r))
        blind = frac(float(np.angle((E_blind / E)[core]).mean()) - exact)
        aware = frac(float(np.angle((E_aware / E)[core]).mean()) - exact)
        ray = (n * thickness / np.sqrt(1.0 - (L / n) ** 2)
               - n * thickness) / LAM
        exl = k0 * thickness * (np.sqrt(n ** 2 - L ** 2) - n) / (2 * np.pi)
        rows.append({'m': m, 'L': L, 'tilt_mrad': L * 1e3,
                     'blind_vs_closed_waves': blind,
                     'aware_vs_closed_waves': aware,
                     'entrance_ray_diff_waves': ray,
                     'exit_exact_diff_waves': exl})
        if verbose:
            print('  m=%3d  %6.3f mrad:  SHIPPED vs closed form %+10.3e w   '
                  'WITH SCREEN %+10.3e w   |   exit-exact d %+8.4f w   '
                  'entrance-ray d %+8.4f w'
                  % (m, L * 1e3, blind, aware, exl, ray))
    if verbose:
        print('  (residuals are mod 1 wave; the shipped column is machine '
              'zero at every tilt, so the screen has nothing to add)')
    return {'glass': glass, 'n': n, 'thickness_mm': thickness * 1e3,
            'N': N, 'dx_um': dx * 1e6, 'rows': rows}


# ---------------------------------------------------------------------------
def main(argv):
    mode = argv[1] if len(argv) > 1 else 'all'
    order = (['plate', 'gap', 'map', 'degree', 'nodes', 'field']
             if mode == 'all' else [mode])
    res = {}
    for m in order:
        print('\n===== %s %s' % (m.upper(), '=' * (60 - len(m))))
        t0 = time.perf_counter()
        res[m] = {'gap': mode_gap, 'map': mode_map, 'degree': mode_degree,
                  'nodes': mode_nodes, 'field': mode_field,
                  'plate': mode_plate}[m]()
        print('[%s: %.1f s]' % (m, time.perf_counter() - t0))
    fn = os.path.join(_HERE, '_hmap_c2_%s.json' % mode)
    with open(fn, 'w', encoding='ascii') as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    print('\nwrote %s' % os.path.basename(fn))


if __name__ == '__main__':
    main(sys.argv)
