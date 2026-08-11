# HAMILTON-MAP feasibility prototype -- the MEASURED gate.
#
# LOCAL-ONLY (needs the 121 .zmx via _d121_common).  NO library edit; every
# lumenairy symbol used here is imported and called, never modified.  Nothing
# in this file propagates a field: all six measurements below are ray traces
# through the design's OWN post-DOE groups, so the whole script is minutes, not
# hours.
#
# THE OBJECT UNDER TEST.  ``apply_real_lens_traced`` launches a lattice of rays
# from the entrance plane ALONG ONE CONGRUENCE (``grad(W + a_fit)``) and keeps
#
#     P(x, y; theta_x, theta_y) -> (x_out, y_out, OPL)          [+ the exit
#                                    direction cosines, from which the
#                                    ray-tube Jacobian follows]
#
# for that congruence only.  Every DOE order re-traces the whole lattice
# because its congruence differs.  A SHARED-PUPIL HAMILTON MAP samples the same
# ``P`` on a PRODUCT grid -- pupil lattice x Chebyshev nodes in angle -- once,
# and evaluates each order by interpolating in angle.  This script measures
# whether that is (a) accurate enough, (b) geometrically legal, and (c) cheaper.
#
# THE PARAMETRISATION THAT MATTERS.  Indexed by ABSOLUTE angle the domain is
# the beam's own NA (+-0.395 rad on design 121 -- hmap_geom_121.py) and no
# affordable node count reaches lambda/100.  Indexed by the REDUCED angle
#
#     s = theta - grad S_R(x, y)
#
# -- the angle measured against the ONE order-independent carrier sphere every
# order shares at this plane -- the domain is the FAN alone (+-0.082 rad).  The
# node lattice in ``theta`` is then SHEARED across the pupil; it is the same 4-D
# function either way.  Both are measured below (mode ``smooth``).
#
# MODES
#   smooth    -- (2) Chebyshev order vs OPL interpolation error, in waves, at
#                the 32 orders' OWN angles; plus the theta^2 coefficient.
#   vig       -- (3) vignetting census over the union domain, with margins.
#   caustic   -- (5) Jacobian sign census / single-valuedness.
#   amp       -- (4b) amplitude (ray-tube Jacobian) interpolation accuracy.
#   consumer2 -- (6) angle-blind thin-element phase vs the angle-true map.
#   trace     -- (4a) ray-trace throughput, for the cost table.
#   all       -- every mode above, in order.
#
# Run:  python hmap_probe_121.py all
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _d121_common as C  # noqa: E402

import lumenairy as la  # noqa: E402
from lumenairy.elements._lens_traced import _tilted_carrier_parts  # noqa: E402
from lumenairy.glass import get_glass_index  # noqa: E402
from lumenairy.propagators.carrier import (  # noqa: E402
    _chain_chief_ray_at_target,
    _paraxial_group_r_out,
)
from lumenairy.raytrace import _make_bundle, surfaces_from_prescription, trace  # noqa: E402


_HERE = os.path.dirname(os.path.abspath(__file__))
LAM = C.LAM
K0 = 2.0 * np.pi / LAM

# The MEASURED retrace window of the shipped design-121 leg at NFC=8192
# (PROBE_SUM_AT_APERTURE_2026_08_11 sec 4: 8192 x 1.5324 um), i.e.
# window_factor * w_entrance.  Used for the per-order pupil half-window so the
# prototype's pupil is the shipped one and not a re-derived estimate.
RETRACE_WINDOW = 8192 * 1.5324e-6            # 12.5534 mm
R_PUPIL_ORDER = 0.5 * RETRACE_WINDOW         # 6.2767 mm


# ===========================================================================
# geometry
# ===========================================================================
def geometry(verbose=False):
    """``(groups_post, orders, R_ent, chief)`` -- the last post-DOE group, the
    32-order table, the carrier radius at that group's ENTRANCE and every
    order's chief ray there."""
    _pre, groups_post, _gap, period = C.geometry()
    mx, my, amps = C.order_table(period)
    _env, R_doe, _dxd, _pin = C.chain_a()
    head = groups_post[:-1]
    gap_last = float(groups_post[-1]['gap_before'])
    R = float(R_doe)
    for g in head:
        R = R + float(g['gap_before'])
        R = float(_paraxial_group_r_out(g['prescription'], R, LAM))
    R_ent = R + gap_last
    rows = []
    for i in range(len(mx)):
        L0 = float(mx[i]) * LAM / period
        M0 = float(my[i]) * LAM / period
        car = la.TiltedCarrier(float(R_doe), L0, M0)
        x_c, y_c, L, M = _chain_chief_ray_at_target(
            head, LAM, car, gap_last, 'hmap_probe_121')
        rows.append({'order': (int(mx[i]), int(my[i])), 'x_c': x_c,
                     'y_c': y_c, 'L': L, 'M': M, 'w': abs(complex(amps[i])),
                     'L_doe': L0, 'M_doe': M0})
    return groups_post, rows, R_ent, period, float(R_doe)


def group_surfaces(groups_post, idx=-1):
    """The surface list ``apply_real_lens_traced`` itself builds: the
    prescription with ``aperture_diameter`` popped (per-surface semi-diameters
    are KEPT -- they are the physical clear apertures)."""
    presc = dict(groups_post[idx]['prescription'])
    presc.pop('aperture_diameter', None)
    return surfaces_from_prescription(presc), presc


def characteristic(surfaces, px, py, thx, thy, history=False):
    """The point characteristic ``P(x, y; theta) -> (x_out, y_out, OPL)``.

    Reproduces ``apply_real_lens_traced``'s own trace step VERBATIM, including
    the EXIT-VERTEX CORRECTION (rays land on the last surface's sag, and the
    wave model's exit plane is the flat vertex plane).  It deliberately does
    NOT add the H6 carrier eikonal ``W`` or the niche-C6 residual ``a_fit``:
    those are ANALYTIC and PER-ORDER, so a shared map must not bake them in --
    the map stores the group's geometric path and each order adds its own
    entrance eikonal afterwards.  That factorisation is exact."""
    rays = _make_bundle(x=np.ravel(px), y=np.ravel(py),
                        L=np.ravel(thx), M=np.ravel(thy), wavelength=LAM)
    res = trace(rays, surfaces, LAM, output_filter=('all' if history
                                                    else 'last'))
    fin = res.image_rays
    n_exit = get_glass_index(surfaces[-1].glass_after, LAM)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(fin.alive & (np.abs(fin.N) > 1e-30), -fin.z / fin.N, 0.0)
    out = {'x': fin.x + fin.L * t, 'y': fin.y + fin.M * t,
           'opl': fin.opd + n_exit * t, 'L': fin.L, 'M': fin.M,
           'alive': fin.alive}
    if history:
        out['res'] = res
    return out


def shear(R_ent, px, py):
    """``grad S_R(x, y)`` -- the ONE order-independent sphere every order at
    this plane shares.  Uses the library's own closed form."""
    ref = la.TiltedCarrier(float(R_ent), 0.0, 0.0, 0.0, 0.0)
    _W, Lg, Mg = _tilted_carrier_parts(ref, np.asarray(px, dtype=float),
                                       np.asarray(py, dtype=float))
    return Lg, Mg


def order_angle(R_ent, row, px, py):
    """``grad W_k(x, y)`` -- order ``k``'s exact launch direction field."""
    car = la.TiltedCarrier(float(R_ent), row['L'], row['M'],
                           row['x_c'], row['y_c'])
    _W, Lg, Mg = _tilted_carrier_parts(car, np.asarray(px, dtype=float),
                                       np.asarray(py, dtype=float))
    return Lg, Mg


# ===========================================================================
# Chebyshev tensor interpolation in the 2-D angle
# ===========================================================================
def cheb_nodes(n):
    """Chebyshev-Lobatto nodes on [-1, 1], ascending."""
    return np.cos(np.pi * np.arange(n - 1, -1, -1) / (n - 1)) if n > 1 \
        else np.array([0.0])


def cheb_fit2d(F, n):
    """Tensor Chebyshev coefficients of the interpolant through ``F`` (n x n)
    sampled on ``cheb_nodes(n)`` x ``cheb_nodes(n)``."""
    t = cheb_nodes(n)
    V = np.polynomial.chebyshev.chebvander(t, n - 1)
    Vi = np.linalg.inv(V)
    return Vi @ F @ Vi.T


def cheb_fit2d_rect(F, nx, ny):
    """Tensor Chebyshev coefficients on a RECTANGULAR node grid
    ``cheb_nodes(nx) x cheb_nodes(ny)``."""
    Vx = np.polynomial.chebyshev.chebvander(cheb_nodes(nx), nx - 1)
    Vy = np.polynomial.chebyshev.chebvander(cheb_nodes(ny), ny - 1)
    return np.linalg.inv(Vx) @ F @ np.linalg.inv(Vy).T


def cheb_ev2d(Cf, tx, ty):
    return np.polynomial.chebyshev.chebval2d(tx, ty, Cf)


def _to_unit(v, c, h):
    return (np.asarray(v, dtype=float) - c) / h


# ===========================================================================
# (2) SMOOTHNESS
# ===========================================================================
def mode_smooth(nlist=((3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (9, 9),
                       (5, 4), (6, 4), (6, 5), (7, 4), (7, 5), (8, 4)),
                verbose=True):
    """Chebyshev interpolation error of the characteristic in ANGLE.

    ``nlist`` entries are ``(nx, ny)`` node counts per angular axis.  The box
    is ANISOTROPIC by construction on design 121: the 32 DOE orders sit on a
    near-lattice of 8 columns x 4 rows in reduced angle, so the y half-width
    is ~2x smaller than the x one and the two axes do not need the same node
    count.  ``nodes = nx * ny`` is the quantity the cost table compares
    against the 32 direct traces."""
    groups_post, rows, R_ent, _per, _rd = geometry()
    surfs, _p = group_surfaces(groups_post)

    reach = max(np.hypot(r['x_c'], r['y_c']) for r in rows)
    r_union = reach + R_PUPIL_ORDER
    frac = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])
    aa = np.array([0.0, 0.0, np.pi / 2, np.pi, 1.5 * np.pi,
                   np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4])
    pxs = r_union * frac * np.cos(aa)
    pys = r_union * frac * np.sin(aa)

    S, A = [], []
    for px, py in zip(pxs, pys):
        Sx0, Sy0 = shear(R_ent, px, py)
        for r in rows:
            lx, ly = order_angle(R_ent, r, px, py)
            S.append((float(lx - Sx0), float(ly - Sy0)))
            A.append((float(lx), float(ly)))
    S = np.array(S)
    A = np.array(A)
    s_lo, s_hi = S.min(axis=0), S.max(axis=0)
    ctr = 0.5 * (s_lo + s_hi)
    Hx, Hy = 0.5 * (s_hi - s_lo)
    a_lo, a_hi = A.min(axis=0), A.max(axis=0)
    ctr_a = 0.5 * (a_lo + a_hi)
    HAx, HAy = 0.5 * (a_hi - a_lo)
    if verbose:
        print("REDUCED-ANGLE BOX over %d pupil probes x %d orders:"
              % (len(pxs), len(rows)))
        print("  sx [%+.6f, %+.6f]  half-width Hx = %.6f rad"
              % (s_lo[0], s_hi[0], Hx))
        print("  sy [%+.6f, %+.6f]  half-width Hy = %.6f rad  (Hx/Hy = %.3f)"
              % (s_lo[1], s_hi[1], Hy, Hx / Hy))
        print("  ABSOLUTE box half-widths %.6f / %.6f rad (%.1fx / %.1fx "
              "wider)" % (HAx, HAy, HAx / Hx, HAy / Hy))

    results = {'Hx': Hx, 'Hy': Hy, 'HAx': HAx, 'HAy': HAy,
               'centre': ctr.tolist(), 'centre_abs': ctr_a.tolist(),
               'r_union': r_union, 'n_pupil_probes': int(len(pxs)),
               'rows': []}

    for label, hx, hy, cc, sheared in (
            ('reduced', Hx, Hy, ctr, True),
            ('absolute', HAx, HAy, ctr_a, False)):
        for (nx, ny) in nlist:
            tx, ty = cheb_nodes(nx), cheb_nodes(ny)
            errs, errs_x, errs_y = [], [], []
            dead_nodes = 0
            tot_nodes = 0
            skipped = 0
            for px, py in zip(pxs, pys):
                Sx0, Sy0 = shear(R_ent, px, py)
                base = (float(Sx0), float(Sy0)) if sheared else (0.0, 0.0)
                TX, TY = np.meshgrid(tx, ty, indexing='ij')
                thx = base[0] + cc[0] + hx * TX
                thy = base[1] + cc[1] + hy * TY
                ch = characteristic(surfs, np.full(thx.size, px),
                                    np.full(thx.size, py), thx.ravel(),
                                    thy.ravel())
                tot_nodes += thx.size
                if not ch['alive'].all():
                    # A dead NODE ray is not a small error: the interpolant
                    # cannot be built at all at this pupil point.  Tally it and
                    # move on -- the tally IS the result for that domain.
                    dead_nodes += int((~ch['alive']).sum())
                    skipped += 1
                    continue
                Co = cheb_fit2d_rect(ch['opl'].reshape(nx, ny), nx, ny)
                Cx = cheb_fit2d_rect(ch['x'].reshape(nx, ny), nx, ny)
                Cy = cheb_fit2d_rect(ch['y'].reshape(nx, ny), nx, ny)
                qx, qy = [], []
                for r in rows:
                    lx, ly = order_angle(R_ent, r, px, py)
                    qx.append(float(lx))
                    qy.append(float(ly))
                qx, qy = np.array(qx), np.array(qy)
                ref = characteristic(surfs, np.full(qx.size, px),
                                     np.full(qx.size, py), qx, qy)
                ok = ref['alive']
                ux = _to_unit(qx - base[0], cc[0], hx)
                uy = _to_unit(qy - base[1], cc[1], hy)
                errs.append(np.abs(cheb_ev2d(Co, ux, uy) - ref['opl'])[ok]
                            / LAM)
                errs_x.append(np.abs(cheb_ev2d(Cx, ux, uy) - ref['x'])[ok])
                errs_y.append(np.abs(cheb_ev2d(Cy, ux, uy) - ref['y'])[ok])
            row = {'domain': label, 'nx': int(nx), 'ny': int(ny),
                   'nodes': int(nx * ny), 'hx': hx, 'hy': hy,
                   'dead_nodes': int(dead_nodes),
                   'node_rays': int(tot_nodes),
                   'pupil_probes_unusable': int(skipped)}
            if errs:
                e = np.concatenate(errs)
                ex = np.concatenate(errs_x)
                ey = np.concatenate(errs_y)
                row['opl_max_waves'] = float(e.max())
                row['opl_rms_waves'] = float(np.sqrt((e ** 2).mean()))
                row['pos_max_nm'] = float(max(ex.max(), ey.max()) * 1e9)
            else:
                row['opl_max_waves'] = float('nan')
                row['opl_rms_waves'] = float('nan')
                row['pos_max_nm'] = float('nan')
            results['rows'].append(row)
            if verbose:
                v = row['opl_max_waves']
                bar = ('PASS(3x)' if v <= 0.01 / 3.0 else
                       ('inside' if v <= 0.01 else 'FAIL'))
                print("  [%-8s] %dx%d = %2d nodes: OPL max %.4e w  "
                      "rms %.4e w  pos %.3e nm  dead %5d/%5d  %s"
                      % (label, nx, ny, nx * ny, v, row['opl_rms_waves'],
                         row['pos_max_nm'], dead_nodes, tot_nodes, bar))

    # ---- the theta^2 coefficient -----------------------------------------
    px, py = 0.0, 0.0
    Sx0, Sy0 = shear(R_ent, px, py)
    tscan = np.linspace(-1.0, 1.0, 41)
    thx = float(Sx0) + ctr[0] + Hx * tscan
    thy = np.full_like(thx, float(Sy0) + ctr[1])
    direct = characteristic(surfs, np.full(thx.size, px),
                            np.full(thx.size, py), thx, thy)
    sx = Hx * tscan
    c_dir = np.polyfit(sx, direct['opl'], 2)
    quad = {'c2_direct': float(c_dir[0])}
    for (nx, ny) in nlist:
        tx, ty = cheb_nodes(nx), cheb_nodes(ny)
        TX, TY = np.meshgrid(tx, ty, indexing='ij')
        ch = characteristic(surfs, np.full(TX.size, px), np.full(TX.size, py),
                            (float(Sx0) + ctr[0] + Hx * TX).ravel(),
                            (float(Sy0) + ctr[1] + Hy * TY).ravel())
        Co = cheb_fit2d_rect(ch['opl'].reshape(nx, ny), nx, ny)
        pred = cheb_ev2d(Co, tscan, np.zeros_like(tscan))
        c_int = np.polyfit(sx, pred, 2)
        key = '%dx%d' % (nx, ny)
        quad['c2_' + key] = float(c_int[0])
        quad['c2_relerr_' + key] = float(abs(c_int[0] - c_dir[0])
                                         / abs(c_dir[0]))
        quad['c2_wave_err_' + key] = float(abs(c_int[0] - c_dir[0])
                                           * Hx ** 2 / LAM)
    results['theta2'] = quad
    if verbose:
        print()
        print("THETA^2 COEFFICIENT (pupil centre, reduced x axis, "
              "|s| <= %.4f rad)" % Hx)
        print("  direct trace  c2 = %+.9e m/rad^2  (= %.2f waves across the "
              "box)" % (quad['c2_direct'],
                        abs(quad['c2_direct']) * Hx ** 2 / LAM))
        for (nx, ny) in nlist:
            k = '%dx%d' % (nx, ny)
            print("  %-6s c2 = %+.9e  rel %.3e  -> %.3e waves across the box"
                  % (k, quad['c2_' + k], quad['c2_relerr_' + k],
                     quad['c2_wave_err_' + k]))
    return results


# ===========================================================================
# (2b) THE EXACT PARAMETRISATION -- Chebyshev nodes in the SOURCE POSITION
# ===========================================================================
# ``_tilted_carrier_parts`` evaluates
#
#     uu = (x - x0) + R L / N,   vv = (y - y0) + R M / N
#     grad W = sign(R) * (uu, vv) / sqrt(uu^2 + vv^2 + R^2)
#
# i.e. the launch-direction field of a ``TiltedCarrier`` depends on
# ``(x0, y0, L, M)`` ONLY through the SOURCE'S TRANSVERSE PROJECTION
#
#     (x_src, y_src) = (x0 - R L / N,  y0 - R M / N).
#
# So at a plane where every order shares one carrier RADIUS -- which design
# 121's last-group entrance does, exactly, because the chain's ``R`` closure is
# order-independent -- the whole 32-order fan is a TWO-PARAMETER family of
# congruences indexed by the source position.  A map sampled at Chebyshev nodes
# in ``(x_src, y_src)`` therefore represents every order EXACTLY in its
# parametrisation: there is no residual from labelling a congruence by a local
# angle that varies across the pupil.  This is the production form.
def source_positions(R_ent, rows):
    """``(x_src, y_src)`` of every order's congruence at this plane."""
    out = []
    for r in rows:
        L, M = float(r['L']), float(r['M'])
        N = np.sqrt(max(1.0 - L * L - M * M, 0.0))
        out.append((r['x_c'] - R_ent * L / N, r['y_c'] - R_ent * M / N))
    return np.array(out)


def src_angle(R_ent, xs, ys, px, py):
    """The launch-direction field of the congruence whose source projects to
    ``(xs, ys)`` -- the library's own closed form, with ``L = M = 0`` and the
    sphere centre at the source (the two spellings are the same function)."""
    car = la.TiltedCarrier(float(R_ent), 0.0, 0.0, float(xs), float(ys))
    _W, Lg, Mg = _tilted_carrier_parts(car, np.asarray(px, dtype=float),
                                       np.asarray(py, dtype=float))
    return Lg, Mg


def mode_src(nlist=((3, 3), (4, 4), (5, 4), (5, 5), (6, 4), (6, 5), (6, 6),
                    (7, 5), (7, 6), (7, 7), (8, 7), (9, 9)),
             pad=0.0, verbose=True):
    groups_post, rows, R_ent, _per, _rd = geometry()
    surfs, _p = group_surfaces(groups_post)
    src = source_positions(R_ent, rows)
    lo, hi = src.min(axis=0), src.max(axis=0)
    ctr = 0.5 * (lo + hi)
    Hx, Hy = 0.5 * (hi - lo)
    # niche-C6 widening: the element launches along grad(W + a_fit), and
    # a_fit is ORDER-DEPENDENT, so its launch gradient is an angular
    # offset no shared congruence label can carry.  Expressed as a source
    # displacement it is max|grad a_fit| x |R| (hmap_cost_121.py afit).
    Hx = Hx + float(pad)
    Hy = Hy + float(pad)
    reach = max(np.hypot(r['x_c'], r['y_c']) for r in rows)
    r_union = reach + R_PUPIL_ORDER

    # the SAME 9 pupil probes mode_smooth uses
    frac = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])
    aa = np.array([0.0, 0.0, np.pi / 2, np.pi, 1.5 * np.pi,
                   np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4])
    pxs = r_union * frac * np.cos(aa)
    pys = r_union * frac * np.sin(aa)

    if verbose:
        print("SOURCE-POSITION BOX (the EXACT congruence label):")
        print("  x_src [%+.6f, %+.6f] mm  half-width %.6f mm"
              % (lo[0] * 1e3, hi[0] * 1e3, Hx * 1e3))
        print("  y_src [%+.6f, %+.6f] mm  half-width %.6f mm"
              % (lo[1] * 1e3, hi[1] * 1e3, Hy * 1e3))
        print("  (R at this plane %.6f mm, order-independent -> the fan is a "
              "2-parameter family)" % (R_ent * 1e3))
        print("  C6 pad %+.6f mm -> box half-widths %.6f / %.6f mm"
              % (pad * 1e3, Hx * 1e3, Hy * 1e3))

    results = {'Hx_src': Hx, 'Hy_src': Hy, 'pad': float(pad),
               'centre_src': ctr.tolist(),
               'R_ent': R_ent, 'r_union': r_union,
               'src': src.tolist(), 'rows': []}
    for (nx, ny) in nlist:
        tx, ty = cheb_nodes(nx), cheb_nodes(ny)
        errs, errs_p, dead = [], [], 0
        for px, py in zip(pxs, pys):
            F_o = np.empty((nx, ny))
            F_x = np.empty((nx, ny))
            F_y = np.empty((nx, ny))
            ok_all = True
            for i, a in enumerate(tx):
                for j, b in enumerate(ty):
                    xs = ctr[0] + Hx * a
                    ys = ctr[1] + Hy * b
                    lx, ly = src_angle(R_ent, xs, ys, px, py)
                    ch = characteristic(surfs, np.array([px]), np.array([py]),
                                        lx, ly)
                    if not ch['alive'].all():
                        ok_all = False
                        dead += 1
                    F_o[i, j] = ch['opl'][0]
                    F_x[i, j] = ch['x'][0]
                    F_y[i, j] = ch['y'][0]
            if not ok_all:
                continue
            Co = cheb_fit2d_rect(F_o, nx, ny)
            Cx = cheb_fit2d_rect(F_x, nx, ny)
            Cy = cheb_fit2d_rect(F_y, nx, ny)
            for k, r in enumerate(rows):
                lx, ly = order_angle(R_ent, r, px, py)
                ref = characteristic(surfs, np.array([px]), np.array([py]),
                                     lx, ly)
                if not ref['alive'].all():
                    continue
                ux = _to_unit(src[k, 0], ctr[0], Hx)
                uy = _to_unit(src[k, 1], ctr[1], Hy)
                errs.append(abs(float(cheb_ev2d(Co, ux, uy)) - ref['opl'][0])
                            / LAM)
                errs_p.append(max(
                    abs(float(cheb_ev2d(Cx, ux, uy)) - ref['x'][0]),
                    abs(float(cheb_ev2d(Cy, ux, uy)) - ref['y'][0])))
        e = np.array(errs)
        row = {'nx': int(nx), 'ny': int(ny), 'nodes': int(nx * ny),
               'opl_max_waves': float(e.max()),
               'opl_rms_waves': float(np.sqrt((e ** 2).mean())),
               'pos_max_nm': float(np.max(errs_p) * 1e9),
               'dead_nodes': int(dead)}
        results['rows'].append(row)
        if verbose:
            v = row['opl_max_waves']
            bar = ('PASS(3x)' if v <= 0.01 / 3.0 else
                   ('inside' if v <= 0.01 else 'FAIL'))
            print("  %dx%d = %2d nodes: OPL max %.4e w  rms %.4e w  "
                  "pos %.3e nm  dead %d  %s"
                  % (nx, ny, nx * ny, v, row['opl_rms_waves'],
                     row['pos_max_nm'], dead, bar))
    return results


# ===========================================================================
# (3) VIGNETTING
# ===========================================================================
def mode_vig(n_pupil=97, n_ang=5, verbose=True):
    """Does any ray of the union domain clip anything?

    Three separate clips exist and they are NOT the same thing:

      1. the per-surface ``semi_diameter`` the TRACER enforces (a dead ray);
      2. the prescription's ``aperture_diameter``, which
         ``apply_real_lens_traced`` POPS before building surfaces (so it never
         kills a ray) but READS to size its launch disc -- it is the group's
         declared clear aperture and the physically meaningful bound;
      3. the launch disc itself, ``0.75 * aperture_diameter``, axis-centred and
         ORDER-INDEPENDENT, which is what the shipped per-order retrace already
         pays for.
    """
    groups_post, rows, R_ent, _per, _rd = geometry()
    surfs, presc = group_surfaces(groups_post)
    ap = groups_post[-1]['prescription'].get('aperture_diameter')
    ap = float(ap) if ap is not None else float('inf')
    lr_shipped = 0.75 * ap
    reach = max(np.hypot(r['x_c'], r['y_c']) for r in rows)
    r_union = reach + R_PUPIL_ORDER

    t = np.linspace(-1.0, 1.0, n_pupil)
    PX, PY = np.meshgrid(r_union * t, r_union * t, indexing='ij')
    keep = (PX ** 2 + PY ** 2) <= r_union ** 2
    px, py = PX[keep], PY[keep]
    Sx0, Sy0 = shear(R_ent, px, py)

    S = []
    for r in rows:
        lx, ly = order_angle(R_ent, r, px, py)
        S.append(np.stack([lx - Sx0, ly - Sy0]))
    S = np.stack(S)
    s_lo = S.min(axis=(0, 2))
    s_hi = S.max(axis=(0, 2))
    cc = 0.5 * (s_lo + s_hi)
    Hx, Hy = 0.5 * (s_hi - s_lo)

    tn = cheb_nodes(n_ang)
    n_dead = 0
    n_tot = 0
    hit_max = [0.0] * len(surfs)
    exit_max = 0.0
    for a in tn:
        for b in tn:
            thx = Sx0 + cc[0] + Hx * a
            thy = Sy0 + cc[1] + Hy * b
            ch = characteristic(surfs, px, py, thx, thy, history=True)
            n_tot += px.size
            n_dead += int((~ch['alive']).sum())
            res = ch['res']
            for i in range(len(surfs)):
                rb = res.ray_history[i]
                hit_max[i] = max(hit_max[i],
                                 float(np.nanmax(np.hypot(rb.x, rb.y))))
            exit_max = max(exit_max, float(np.nanmax(np.hypot(ch['x'],
                                                              ch['y']))))
    out = {'n_pupil_rays': int(px.size), 'n_angle_nodes': int(n_ang ** 2),
           'n_rays_total': int(n_tot), 'n_dead': int(n_dead),
           'r_union': r_union, 'Hx': Hx, 'Hy': Hy,
           'aperture_diameter': ap, 'semi_aperture': 0.5 * ap,
           'launch_radius_shipped': lr_shipped,
           'surface_hit_max': hit_max,
           'surface_semi_diameters': [float(sf.semi_diameter)
                                      for sf in surfs],
           'exit_radius_max': exit_max,
           'margin_vs_semi_aperture': float(0.5 * ap - max(hit_max)),
           'margin_frac_vs_semi_aperture': float(
               (0.5 * ap - max(hit_max)) / (0.5 * ap)),
           'margin_frac_vs_launch_disc': float(
               (lr_shipped - r_union) / lr_shipped)}
    if verbose:
        print("VIGNETTING census: %d rays (%d pupil x %d angular nodes) over "
              "the union disc r <= %.4f mm, |sx| <= %.4f / |sy| <= %.4f rad"
              % (n_tot, px.size, n_ang ** 2, r_union * 1e3, Hx, Hy))
        print("  dead rays (tracer semi_diameter clip): %d (%.4f %%)"
              % (n_dead, 100.0 * n_dead / max(n_tot, 1)))
        print("  group aperture_diameter %.4f mm -> semi-aperture %.4f mm"
              % (ap * 1e3, 0.5 * ap * 1e3))
        for i, sf in enumerate(surfs):
            sd = sf.semi_diameter
            print("    surface %d (%-12s) max hit radius %8.4f mm   "
                  "semi_diameter %s   vs semi-aperture %+7.4f mm (%+6.2f %%)"
                  % (i, (sf.label or '-'), hit_max[i] * 1e3,
                     ('INF' if not np.isfinite(sd) else '%.4f mm' % (sd * 1e3)),
                     (0.5 * ap - hit_max[i]) * 1e3,
                     100.0 * (0.5 * ap - hit_max[i]) / (0.5 * ap)))
        print("  exit landing radius max %.4f mm" % (exit_max * 1e3))
        print("  the union pupil sits INSIDE the shipped launch disc "
              "(%.4f mm) by %.2f %% -- the shared map needs NO pupil the "
              "shipped per-order retrace does not already launch"
              % (lr_shipped * 1e3, 100.0 * out['margin_frac_vs_launch_disc']))
    return out


# ===========================================================================
# (5) CAUSTIC GUARD -- Jacobian sign census
# ===========================================================================
def _congruence_jacobian(surfs, R_ent, row, px, py, h):
    """``det d(x_out, y_out)/d(x_in, y_in)`` ALONG order ``row``'s own
    congruence (the angle moves with the pupil point, exactly as the shipped
    launch does), by a central difference of step ``h``."""
    P = []
    for dx, dy in ((h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)):
        qx, qy = px + dx, py + dy
        lx, ly = order_angle(R_ent, row, qx, qy)
        P.append(characteristic(surfs, qx, qy, lx, ly))
    dXdx = (P[0]['x'] - P[1]['x']) / (2 * h)
    dYdx = (P[0]['y'] - P[1]['y']) / (2 * h)
    dXdy = (P[2]['x'] - P[3]['x']) / (2 * h)
    dYdy = (P[2]['y'] - P[3]['y']) / (2 * h)
    alive = P[0]['alive'] & P[1]['alive'] & P[2]['alive'] & P[3]['alive']
    return dXdx * dYdy - dXdy * dYdx, alive


def mode_caustic(n_pupil=65, verbose=True):
    groups_post, rows, R_ent, _per, _rd = geometry()
    surfs, _p = group_surfaces(groups_post)
    h = 5e-6
    census = []
    for r in rows:
        t = np.linspace(-1.0, 1.0, n_pupil)
        PX, PY = np.meshgrid(r['x_c'] + R_PUPIL_ORDER * t,
                             r['y_c'] + R_PUPIL_ORDER * t, indexing='ij')
        keep = ((PX - r['x_c']) ** 2 + (PY - r['y_c']) ** 2) \
            <= R_PUPIL_ORDER ** 2
        det, alive = _congruence_jacobian(surfs, R_ent, r, PX[keep], PY[keep],
                                          h)
        d = det[alive]
        census.append({'order': list(r['order']),
                       'n': int(d.size),
                       'n_pos': int((d > 0).sum()),
                       'n_neg': int((d < 0).sum()),
                       'n_zero': int((d == 0).sum()),
                       'min_abs': float(np.abs(d).min()),
                       'max_abs': float(np.abs(d).max()),
                       'median': float(np.median(d))})
    sign_ok = all(c['n_neg'] == 0 or c['n_pos'] == 0 for c in census)
    ratio = max(c['max_abs'] / max(c['min_abs'], 1e-300) for c in census)
    if verbose:
        print(f"CAUSTIC / SINGLE-VALUEDNESS census -- det J along each of the "
              f"{len(rows)} orders' own congruences, {census[0]['n']} pupil "
              f"points each")
        print(f"  sign-consistent on every order: {sign_ok}")
        print(f"  worst |det J| dynamic range within one order: {ratio:.4f}")
        for c in census[:4] + census[-2:]:
            print(f"    {str(tuple(c['order'])):>9s}  pos {c['n_pos']:6d} "
                  f"neg {c['n_neg']:6d}  |det| in "
                  f"[{c['min_abs']:.6e}, {c['max_abs']:.6e}]  "
                  f"median {c['median']:+.6e}")
    # and the NODE congruences the map is actually built on
    node_census = []
    for n_ang in (3, 5):
        tn = cheb_nodes(n_ang)
        reach = max(np.hypot(rr['x_c'], rr['y_c']) for rr in rows)
        r_union = reach + R_PUPIL_ORDER
        t = np.linspace(-1.0, 1.0, 49)
        PX, PY = np.meshgrid(r_union * t, r_union * t, indexing='ij')
        keep = (PX ** 2 + PY ** 2) <= r_union ** 2
        px, py = PX[keep], PY[keep]
        # the same reduced box as mode_vig
        Sx0, Sy0 = shear(R_ent, px, py)
        SS = []
        for r in rows:
            lx, ly = order_angle(R_ent, r, px, py)
            SS.append(np.stack([lx - Sx0, ly - Sy0]))
        SS = np.stack(SS)
        cxy = 0.5 * (SS.min(axis=(0, 2)) + SS.max(axis=(0, 2)))
        H = float(max(SS.max(axis=(0, 2)) - SS.min(axis=(0, 2))) * 0.5)
        bad = 0
        tot = 0
        for a in tn:
            for b in tn:
                def _ang(qx, qy, a=a, b=b):
                    sx, sy = shear(R_ent, qx, qy)
                    return sx + cxy[0] + H * a, sy + cxy[1] + H * b
                P = []
                for dx, dy in ((h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)):
                    lx, ly = _ang(px + dx, py + dy)
                    P.append(characteristic(surfs, px + dx, py + dy, lx, ly))
                det = ((P[0]['x'] - P[1]['x']) * (P[2]['y'] - P[3]['y'])
                       - (P[2]['x'] - P[3]['x']) * (P[0]['y'] - P[1]['y'])) \
                    / (4 * h * h)
                al = P[0]['alive'] & P[1]['alive'] & P[2]['alive'] \
                    & P[3]['alive']
                d = det[al]
                tot += d.size
                bad += int((d <= 0).sum()) if np.median(d) > 0 else \
                    int((d >= 0).sum())
        node_census.append({'n_ang': n_ang, 'n': tot, 'sign_flips': bad})
        if verbose:
            print(f"  node congruences (n_ang={n_ang}, sheared): {tot} "
                  f"samples, sign flips {bad}")
    return {'per_order': census, 'sign_consistent': bool(sign_ok),
            'dyn_range': float(ratio), 'node_census': node_census}


# ===========================================================================
# (4b) AMPLITUDE / JACOBIAN interpolation accuracy
# ===========================================================================
def mode_amp(nlist=(3, 4, 5, 6, 7), verbose=True):
    """Interpolate the map, then differentiate the INTERPOLANT along each
    order's congruence, and score the ray-tube amplitude ``|det J|^-1/2``
    against the same quantity from direct traces."""
    groups_post, rows, R_ent, _per, _rd = geometry()
    surfs, _p = group_surfaces(groups_post)
    reach = max(np.hypot(r['x_c'], r['y_c']) for r in rows)
    r_union = reach + R_PUPIL_ORDER
    h = 20e-6

    aa = np.array([0.0, 0.0, np.pi / 2, np.pi, 1.5 * np.pi,
                   np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4])
    frac = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.95, 0.95, 0.95, 0.95])
    pxs = r_union * frac * np.cos(aa)
    pys = r_union * frac * np.sin(aa)

    # reduced-angle domain over these probes
    S = []
    for px, py in zip(pxs, pys):
        sx, sy = shear(R_ent, px, py)
        for r in rows:
            lx, ly = order_angle(R_ent, r, px, py)
            S.append((float(lx - sx), float(ly - sy)))
    S = np.array(S)
    ctr = 0.5 * (S.min(axis=0) + S.max(axis=0))
    H = float(max(S.max(axis=0) - S.min(axis=0)) * 0.5)

    stencil = ((h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h))
    out = []
    for n in nlist:
        tn = cheb_nodes(n)
        errs = []
        for px, py in zip(pxs, pys):
            # build the map on the 4-point pupil stencil x the angular nodes
            maps = []
            for dx, dy in stencil:
                qx, qy = px + dx, py + dy
                sx, sy = shear(R_ent, qx, qy)
                TX, TY = np.meshgrid(tn, tn, indexing='ij')
                ch = characteristic(surfs, np.full(TX.size, qx),
                                    np.full(TX.size, qy),
                                    (float(sx) + ctr[0] + H * TX).ravel(),
                                    (float(sy) + ctr[1] + H * TY).ravel())
                maps.append((cheb_fit2d(ch['x'].reshape(n, n), n),
                             cheb_fit2d(ch['y'].reshape(n, n), n)))
            for r in rows:
                # interpolated congruence Jacobian
                vals = []
                for (dx, dy), (Cx, Cy) in zip(stencil, maps):
                    qx, qy = px + dx, py + dy
                    sx, sy = shear(R_ent, qx, qy)
                    lx, ly = order_angle(R_ent, r, qx, qy)
                    ux = _to_unit(float(lx) - float(sx), ctr[0], H)
                    uy = _to_unit(float(ly) - float(sy), ctr[1], H)
                    vals.append((float(cheb_ev2d(Cx, ux, uy)),
                                 float(cheb_ev2d(Cy, ux, uy))))
                det_i = ((vals[0][0] - vals[1][0]) * (vals[2][1] - vals[3][1])
                         - (vals[2][0] - vals[3][0])
                         * (vals[0][1] - vals[1][1])) / (4 * h * h)
                det_d, al = _congruence_jacobian(surfs, R_ent, r,
                                                 np.array([px]),
                                                 np.array([py]), h)
                det_d = float(det_d[0])
                a_i = abs(det_i) ** -0.5
                a_d = abs(det_d) ** -0.5
                errs.append(abs(a_i - a_d) / a_d)
        e = np.array(errs)
        out.append({'n': int(n), 'amp_rel_max': float(e.max()),
                    'amp_rel_rms': float(np.sqrt((e ** 2).mean()))})
        if verbose:
            print(f"  n={n} (deg {n - 1}): ray-tube amplitude "
                  f"|det J|^-1/2 relative error  max {e.max():.4e}  "
                  f"rms {np.sqrt((e ** 2).mean()):.4e}")
    return {'H': H, 'rows': out, 'stencil_h': h}


# ===========================================================================
# (6) CONSUMER 2 -- the angle-blind thin-element phase
# ===========================================================================
def mode_consumer2(radii=(1e-3, 2e-3, 3e-3), n_pup=65, verbose=True):
    """For every post-DOE group and every DOE order: how many waves does the
    NORMAL-INCIDENCE element model lose against the angle-true map?

    Measured as the change in the group's own traced OPL map when the launch
    congruence is tilted to that order's actual chief-ray direction at that
    group, decomposed into PISTON / TILT / RESIDUAL (piston+tilt removed) and
    RESIDUAL-after-defocus.  The angle-blind model carries NONE of it -- its
    screen is a function of ``(x, y)`` alone."""
    _pre, groups_post, _gap, period = C.geometry()
    mx, my, _amps = C.order_table(period)
    _env, R_doe, _dxd, _pin = C.chain_a()

    # chief-ray tilt of every order at every group's ENTRANCE
    tilts = []
    for gi in range(len(groups_post)):
        head = groups_post[:gi]
        gap = float(groups_post[gi]['gap_before'])
        row = []
        for i in range(len(mx)):
            car = la.TiltedCarrier(float(R_doe),
                                   float(mx[i]) * LAM / period,
                                   float(my[i]) * LAM / period)
            x_c, y_c, L, M = _chain_chief_ray_at_target(
                head, LAM, car, gap, 'hmap_probe_121')
            row.append((x_c, y_c, L, M))
        tilts.append(row)

    out = []
    for gi in range(len(groups_post)):
        surfs, presc = group_surfaces(groups_post, gi)
        for rad in radii:
            t = np.linspace(-1.0, 1.0, n_pup)
            PX, PY = np.meshgrid(rad * t, rad * t, indexing='ij')
            keep = (PX ** 2 + PY ** 2) <= rad ** 2
            px, py = PX[keep], PY[keep]
            base = characteristic(surfs, px, py, np.zeros_like(px),
                                  np.zeros_like(py))
            if not base['alive'].all():
                continue
            A = np.stack([np.ones_like(px), px, py], axis=1)
            A4 = np.stack([np.ones_like(px), px, py, px ** 2 + py ** 2],
                          axis=1)
            worst = {'piston': 0.0, 'tilt': 0.0, 'resid': 0.0, 'resid4': 0.0}
            worst_ord = None
            for i in range(len(mx)):
                _xc, _yc, L, M = tilts[gi][i]
                ch = characteristic(surfs, px, py, np.full_like(px, L),
                                    np.full_like(py, M))
                if not ch['alive'].all():
                    continue
                d = ch['opl'] - base['opl']
                c, *_ = np.linalg.lstsq(A, d, rcond=None)
                res = d - A @ c
                c4, *_ = np.linalg.lstsq(A4, d, rcond=None)
                res4 = d - A4 @ c4
                piston = abs(c[0]) / LAM
                tiltw = float(np.abs(A[:, 1:] @ c[1:]).max()) / LAM
                rr = float(np.sqrt((res ** 2).mean())) / LAM
                rr4 = float(np.sqrt((res4 ** 2).mean())) / LAM
                if rr > worst['resid']:
                    worst = {'piston': piston, 'tilt': tiltw, 'resid': rr,
                             'resid4': rr4}
                    worst_ord = (int(mx[i]), int(my[i]), L, M)
            if worst_ord is None:
                continue
            out.append({'group': gi, 'radius_mm': rad * 1e3,
                        'worst_order': [worst_ord[0], worst_ord[1]],
                        'tilt_rad': float(np.hypot(worst_ord[2],
                                                   worst_ord[3])),
                        **worst})
            if verbose:
                print(f"  group {gi}  r={rad * 1e3:.1f} mm  worst order "
                      f"{str(tuple(worst_ord[:2])):>9s} at "
                      f"{np.hypot(worst_ord[2], worst_ord[3]) * 1e3:6.2f} "
                      f"mrad:  piston {worst['piston']:10.3f} w   "
                      f"tilt {worst['tilt']:9.3f} w   "
                      f"RESID {worst['resid']:9.4f} w   "
                      f"resid+defocus {worst['resid4']:9.4f} w")
    return {'rows': out}


# ===========================================================================
# (4a) RAY-TRACE THROUGHPUT
# ===========================================================================
def mode_trace(verbose=True):
    groups_post, rows, R_ent, _per, _rd = geometry()
    surfs, _p = group_surfaces(groups_post)
    reach = max(np.hypot(r['x_c'], r['y_c']) for r in rows)
    r_union = reach + R_PUPIL_ORDER
    out = []
    for n in (200, 500, 1000, 2000):
        t = np.linspace(-r_union, r_union, n)
        PX, PY = np.meshgrid(t, t, indexing='ij')
        sx, sy = shear(R_ent, PX.ravel(), PY.ravel())
        t0 = time.perf_counter()
        characteristic(surfs, PX.ravel(), PY.ravel(), sx, sy)
        dt = time.perf_counter() - t0
        out.append({'n_launch': int(n), 'n_rays': int(n * n),
                    'seconds': dt, 'rays_per_s': n * n / dt})
        if verbose:
            print(f"  n_launch {n:5d} ({n * n:10d} rays): {dt:8.3f} s  "
                  f"= {n * n / dt:.3e} rays/s")
        del PX, PY, sx, sy
    return {'rows': out, 'n_surfaces': len(surfs), 'r_union': r_union}


# ===========================================================================
def main(argv):
    mode = argv[1] if len(argv) > 1 else 'all'
    res = {}
    order = (['smooth', 'src', 'vig', 'caustic', 'amp', 'consumer2', 'trace']
             if mode == 'all' else [mode])
    for m in order:
        print(f"\n===== {m.upper()} " + "=" * (60 - len(m)))
        t0 = time.perf_counter()
        res[m] = {'smooth': mode_smooth, 'src': mode_src, 'vig': mode_vig,
                  'caustic': mode_caustic, 'amp': mode_amp,
                  'consumer2': mode_consumer2, 'trace': mode_trace}[m]()
        print(f"[{m}: {time.perf_counter() - t0:.1f} s]")
    fn = os.path.join(_HERE, f'_hmap_{mode}.json')
    with open(fn, 'w', encoding='ascii') as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    print(f"\nwrote {os.path.basename(fn)}")


if __name__ == '__main__':
    main(sys.argv)
