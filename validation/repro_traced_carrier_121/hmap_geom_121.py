# HAMILTON-MAP feasibility prototype, step 1: the ANGULAR DOMAIN census.
#
# LOCAL-ONLY (needs the 121 .zmx via _d121_common).  NO library edit.  Nothing
# here propagates a field: every number is an analytic chief-ray trace plus the
# library's OWN closed-form carrier gradient, so it costs ~15 s and is free to
# re-run.
#
# WHAT IT ANSWERS
#   1. the chief-ray tilt (L, M) and position (x_c, y_c) of ALL 32 DOE orders at
#      the ENTRANCE of the last post-DOE group -- the plane a shared-pupil
#      characteristic map would be built on (the sum-at-aperture probe measured
#      the EXIT chief-ray spread, 7.7e-04 rad, which is a different plane and a
#      different quantity);
#   2. the ABSOLUTE launch-angle box the map would have to span if it were
#      indexed by (x, y, theta_x, theta_y) directly -- dominated by the beam's
#      own NA, NOT by the fan;
#   3. the REDUCED (sheared) angle box  s = theta - grad S_R(x, y),  i.e. the
#      angle measured against the ONE order-independent carrier sphere every
#      order shares at this plane.  This is the domain that actually has to be
#      Chebyshev-sampled, and it is ~2 decades smaller than (2).
#   4. the union PUPIL the map must cover (axis-centred, holding every order's
#      beam) against the per-order chief-ray-centred window the shipped retrace
#      uses -- the area ratio is the cost multiplier in the break-even table.
#
# Run:  python hmap_geom_121.py            (prints + writes _hmap_geom.json)
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _d121_common as C  # noqa: E402

import lumenairy as la  # noqa: E402
from lumenairy.elements._lens_traced import _tilted_carrier_parts  # noqa: E402
from lumenairy.propagators.carrier import (  # noqa: E402
    _chain_chief_ray_at_target,
    _paraxial_group_r_out,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(_HERE, '_hmap_geom.json')

# The probe's own configuration (PROBE_SUM_AT_APERTURE_2026_08_11 sec 1):
# window_factor = 4.0 is the value that reproduces the D9 note's measured
# 12.50 mm on-axis retrace window at the last group.
WINDOW_FACTOR = 4.0


def _amp_radius(env, dx):
    """Second-moment amplitude radius of an envelope (metres)."""
    a = np.abs(np.asarray(env))
    n = a.shape[0]
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    w = a ** 2
    tot = float(w.sum())
    cx = float((w * X).sum()) / tot
    cy = float((w * Y).sum()) / tot
    r2 = float((w * ((X - cx) ** 2 + (Y - cy) ** 2)).sum()) / tot
    return float(np.sqrt(2.0 * r2))       # 1/e^2 amplitude radius convention


def geometry_at_last_entrance(verbose=True):
    """Chief rays + carrier radius of every DOE order at the LAST post-DOE
    group's ENTRANCE, plus the beam radius and the union pupil."""
    _pre, groups_post, _gap, period = C.geometry()
    mx, my, amps = C.order_table(period)
    env, R_doe, dx_doe, P_in = C.chain_a()
    w_doe = _amp_radius(env, dx_doe)

    n_grp = len(groups_post)
    head = groups_post[:-1]                    # every group BEFORE the last
    gap_last = float(groups_post[-1]['gap_before'])

    # --- carrier radius at the last group's ENTRANCE (order-independent) ---
    # exactly the chain's own closure: R += gap, then the group's Moebius map.
    R = float(R_doe)
    m_cum = 1.0                                # co-moving (Sziklas) magnification
    for g in head:
        gb = float(g['gap_before'])
        m_cum *= (R + gb) / R
        R = R + gb
        R_next = float(_paraxial_group_r_out(g['prescription'], R, C.LAM))
        # a group maps the wavefront R -> R_next; the marginal ray height is
        # continuous ACROSS a thin group in the paraxial limit, so the
        # magnification the envelope sees on the leg is the free-space one.
        R = R_next
    m_cum *= (R + gap_last) / R
    R_ent = R + gap_last
    w_ent = abs(m_cum) * w_doe

    rows = []
    for i in range(len(mx)):
        L0 = float(mx[i]) * C.LAM / period
        M0 = float(my[i]) * C.LAM / period
        car = la.TiltedCarrier(float(R_doe), L0, M0)
        x_c, y_c, L, M = _chain_chief_ray_at_target(
            head, C.LAM, car, gap_last, 'hmap_geom_121')
        rows.append({'order': (int(mx[i]), int(my[i])),
                     'weight': complex(amps[i]),
                     'L_doe': L0, 'M_doe': M0,
                     'x_c': x_c, 'y_c': y_c, 'L': L, 'M': M})

    reach = max(float(np.hypot(r['x_c'], r['y_c'])) for r in rows)
    r_pupil_order = 0.5 * WINDOW_FACTOR * w_ent          # per-order half-window
    r_pupil_union = reach + r_pupil_order                # axis-centred union
    if verbose:
        print(f"groups_post: {n_grp};  last gap_before {gap_last * 1e3:.4f} mm")
        print(f"R at last-group ENTRANCE {R_ent * 1e3:.6f} mm "
              f"(order-independent);  co-moving magnification {m_cum:+.6f}")
        print(f"beam amp radius at the DOE {w_doe * 1e3:.4f} mm "
              f"-> at the last entrance {w_ent * 1e3:.4f} mm")
        print(f"chief-ray reach {reach * 1e3:.4f} mm;  per-order half-window "
              f"{r_pupil_order * 1e3:.4f} mm (wf={WINDOW_FACTOR});  "
              f"UNION half-window {r_pupil_union * 1e3:.4f} mm")
        print(f"  -> union/per-order pupil AREA ratio "
              f"{(r_pupil_union / r_pupil_order) ** 2:.4f}")
    return {'rows': rows, 'period': period, 'R_doe': R_doe, 'dx_doe': dx_doe,
            'P_in': P_in, 'w_doe': w_doe, 'w_ent': w_ent, 'R_ent': R_ent,
            'm_cum': m_cum, 'gap_last': gap_last, 'reach': reach,
            'r_pupil_order': r_pupil_order, 'r_pupil_union': r_pupil_union,
            'groups_post': groups_post, 'n_groups': n_grp}


def angular_domain(geo, n_pupil=65, verbose=True):
    """The ABSOLUTE and the REDUCED angular boxes the map must span.

    ``theta_k(x, y) = grad W_k(x, y)`` is the library's OWN launch direction
    (:func:`_tilted_carrier_parts`) for order ``k``'s congruence at this plane.
    The REDUCED angle subtracts the ONE order-independent sphere every order
    shares, ``grad S_R(x, y)`` with ``S_R`` the axis-centred, untilted sphere of
    the same radius:

        s_k(x, y) = grad W_k(x, y) - grad S_R(x, y)

    Sampling the map at Chebyshev nodes in ``s`` instead of in ``theta`` is a
    pure change of variable on the SAME 4-D function -- the node angle at pupil
    point ``(x, y)`` is ``grad S_R(x, y) + s_j``, a SHEARED angular lattice --
    but it removes the beam's own NA from the domain, which is the whole
    difficulty.
    """
    R = float(geo['R_ent'])
    rows = geo['rows']
    rp = geo['r_pupil_order']
    ru = geo['r_pupil_union']

    # per-order pupil lattice (the order's OWN support, chief-ray-centred) and
    # the union lattice (axis-centred), both square.
    t = np.linspace(-1.0, 1.0, n_pupil)

    # the shared, order-independent reference sphere: an untilted, axis-centred
    # TiltedCarrier of the same radius (its grad is the library's exact form).
    ref = la.TiltedCarrier(R, 0.0, 0.0, 0.0, 0.0)

    th_abs = []          # absolute launch angles, over each order's support
    s_red = []           # reduced angles, same samples
    s_chief = []         # reduced angle AT each order's own chief ray
    for r in rows:
        X = r['x_c'] + rp * t[None, :] * np.ones((n_pupil, 1))
        Y = r['y_c'] + rp * t[:, None] * np.ones((1, n_pupil))
        car = la.TiltedCarrier(R, r['L'], r['M'], r['x_c'], r['y_c'])
        _W, Lg, Mg = _tilted_carrier_parts(car, X, Y)
        _Wr, Lr, Mr = _tilted_carrier_parts(ref, X, Y)
        th_abs.append((Lg, Mg))
        s_red.append((Lg - Lr, Mg - Mr))
        _w0, l0, m0 = _tilted_carrier_parts(car, r['x_c'], r['y_c'])
        _w1, l1, m1 = _tilted_carrier_parts(ref, r['x_c'], r['y_c'])
        s_chief.append((float(l0 - l1), float(m0 - m1)))

    ax_lo = min(float(a[0].min()) for a in th_abs)
    ax_hi = max(float(a[0].max()) for a in th_abs)
    ay_lo = min(float(a[1].min()) for a in th_abs)
    ay_hi = max(float(a[1].max()) for a in th_abs)
    sx_lo = min(float(a[0].min()) for a in s_red)
    sx_hi = max(float(a[0].max()) for a in s_red)
    sy_lo = min(float(a[1].min()) for a in s_red)
    sy_hi = max(float(a[1].max()) for a in s_red)

    # how much of the reduced box is the FAN (chief-ray spread) and how much is
    # the residual pupil-dependence of s within one order (the shear's own
    # non-linearity -- the part a pure "one value per order" model would miss)
    scx = np.array([a[0] for a in s_chief])
    scy = np.array([a[1] for a in s_chief])
    fan_x = float(scx.max() - scx.min())
    fan_y = float(scy.max() - scy.min())
    within = max(float(np.ptp(a[0])) for a in s_red), \
        max(float(np.ptp(a[1])) for a in s_red)

    # the chief-ray tilt spread the task asks for literally
    Ls = np.array([r['L'] for r in rows])
    Ms = np.array([r['M'] for r in rows])
    tilt_spread = float(np.hypot(Ls.max() - Ls.min(), Ms.max() - Ms.min()))

    out = {
        'R_ent': R, 'n_pupil': n_pupil,
        'chief_tilt_spread': tilt_spread,
        'chief_L_range': [float(Ls.min()), float(Ls.max())],
        'chief_M_range': [float(Ms.min()), float(Ms.max())],
        'abs_box': [ax_lo, ax_hi, ay_lo, ay_hi],
        'abs_halfwidth': 0.5 * max(ax_hi - ax_lo, ay_hi - ay_lo),
        'red_box': [sx_lo, sx_hi, sy_lo, sy_hi],
        'red_halfwidth': 0.5 * max(sx_hi - sx_lo, sy_hi - sy_lo),
        'red_centre': [0.5 * (sx_lo + sx_hi), 0.5 * (sy_lo + sy_hi)],
        'fan_spread_reduced': [fan_x, fan_y],
        'within_order_reduced': [within[0], within[1]],
        's_chief': [[float(a), float(b)] for a, b in s_chief],
    }
    if verbose:
        print()
        print("ORDER TABLE at the LAST GROUP'S ENTRANCE")
        print("  order      |a|       chief (mm)              tilt (mrad)      "
              " reduced angle s_chief (mrad)")
        for r, (sx, sy) in zip(rows, s_chief):
            print(f"  {str(r['order']):>9s} {abs(r['weight']):.5f}  "
                  f"({r['x_c'] * 1e3:+8.4f},{r['y_c'] * 1e3:+8.4f})  "
                  f"({r['L'] * 1e3:+8.4f},{r['M'] * 1e3:+8.4f})   "
                  f"({sx * 1e3:+9.4f},{sy * 1e3:+9.4f})")
        print()
        print(f"(1) CHIEF-RAY tilt spread at this plane: "
              f"{tilt_spread:.6f} rad "
              f"({tilt_spread * 1e3:.3f} mrad)")
        print(f"(2) ABSOLUTE launch-angle box over every order's own support:")
        print(f"      x [{ax_lo:+.6f}, {ax_hi:+.6f}]   "
              f"y [{ay_lo:+.6f}, {ay_hi:+.6f}]   "
              f"half-width {out['abs_halfwidth']:.6f} rad")
        print(f"(3) REDUCED (sheared) angle box, s = grad W_k - grad S_R:")
        print(f"      x [{sx_lo:+.6e}, {sx_hi:+.6e}]   "
              f"y [{sy_lo:+.6e}, {sy_hi:+.6e}]")
        print(f"      half-width {out['red_halfwidth']:.6e} rad   "
              f"= {out['abs_halfwidth'] / max(out['red_halfwidth'], 1e-300):.1f}x "
              f"SMALLER than (2)")
        print(f"      of which FAN spread {fan_x:.6e} / {fan_y:.6e} rad, "
              f"WITHIN-order pupil variation {within[0]:.6e} / "
              f"{within[1]:.6e} rad")
    return out


def main():
    geo = geometry_at_last_entrance()
    dom = angular_domain(geo)
    blob = {
        'R_ent': geo['R_ent'], 'w_ent': geo['w_ent'], 'reach': geo['reach'],
        'r_pupil_order': geo['r_pupil_order'],
        'r_pupil_union': geo['r_pupil_union'],
        'pupil_area_ratio': (geo['r_pupil_union'] / geo['r_pupil_order']) ** 2,
        'gap_last': geo['gap_last'], 'period': geo['period'],
        'n_orders': len(geo['rows']),
        'orders': [{'order': list(r['order']), 'x_c': r['x_c'],
                    'y_c': r['y_c'], 'L': r['L'], 'M': r['M'],
                    'w_abs': abs(r['weight'])} for r in geo['rows']],
        'domain': dom,
        'window_factor': WINDOW_FACTOR,
    }
    with open(OUT_JSON, 'w', encoding='ascii') as fh:
        json.dump(blob, fh, indent=1, sort_keys=True)
    print()
    print(f"wrote {os.path.basename(OUT_JSON)}")


if __name__ == '__main__':
    main()
