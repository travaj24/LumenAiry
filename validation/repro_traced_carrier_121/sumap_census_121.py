# SUM-AT-APERTURE probe, step 0: the GEOMETRY CENSUS the grid choice rests on.
#
# LOCAL-ONLY (needs the 121 .zmx via _d121_common).  No library edit; nothing
# here propagates a field -- every number below is an analytic chief-ray trace
# through the post-DOE groups, so it costs ~2 s and can be re-run freely.
#
# WHAT IT ANSWERS, for the 3 probe orders (0,0) / (-2,0) / (-4,-2):
#   1. the exit congruence of each order at the LAST GROUP'S BACK APERTURE --
#      chief ray (x_c, y_c) and direction cosines (L, M) -- which is the plane
#      the sum-at-aperture architecture sums on;
#   2. the INTER-ORDER TILT SPREAD at that plane -- which is NOT the chief-ray
#      cosine spread (that is ~0 here) but the CARRIER-OFFSET RAMP
#      |c_k - c_0|/|R| a common carrier leaves behind;
#   3. each beam's OWN absolute angular band, which is what actually binds the
#      common pitch, and which does NOT add to (2);
#   4. the chief-ray spread at that plane, hence the common window.
#
# Run: python sumap_census_121.py
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _d121_common as C  # noqa: E402

import lumenairy as la  # noqa: E402
from lumenairy.propagators.carrier import (  # noqa: E402
    _chain_chief_ray_at_target,
    _paraxial_group_r_out,
)

ORDERS_PROBE = ((0, 0), (-2, 0), (-4, -2))
# The measured exit NA of the design-121 fine leg, from the shipped per-order
# banner (ADJUDICATION_NFC_8192_2026_08_10, column ``na_exit_measured``).  Used
# ONLY to state the (over-conservative) union-band bound (b'); the bound that
# actually governs is (c) below.
NA_EXIT_MEASURED = 0.5321
# The beam's own support at the back aperture, MEASURED on arm A's aperture
# field (``sumap_probe_121.py arma`` then the enclosed-power radius): 99.999 %
# of an order's power lies inside this radius of its chief ray.  It is the
# quantity bound (c) needs -- a beam's ABSOLUTE angular band is set by how far
# its own sphere is sampled from its own centre, and NOT by how far off axis
# that centre sits, so the carrier-offset ramp (b) does NOT add to it.
R_SUPPORT_99999 = 2.64e-3


def census(orders=ORDERS_PROBE, verbose=True):
    """Return a dict of the geometry the sum-at-aperture grid is sized from."""
    _pre, groups_post, _gap, period = C.geometry()
    mx, my, amps = C.order_table(period)
    sel = []
    for (m_, n_) in orders:
        idx = [i for i in range(len(mx))
               if int(mx[i]) == m_ and int(my[i]) == n_]
        if not idx:
            raise SystemExit(f"order {(m_, n_)} not in the design's own table "
                             f"{sorted(zip(mx.tolist(), my.tolist()))}")
        sel.append(idx[0])
    rows = []
    for i, (m_, n_) in zip(sel, orders):
        L = float(mx[i]) * C.LAM / period
        M = float(my[i]) * C.LAM / period
        car = la.TiltedCarrier(np.nan, L, M)   # R filled per-call below
        rows.append({'order': (int(mx[i]), int(my[i])), 'L_doe': L, 'M_doe': M,
                     'weight': complex(amps[i]), 'carrier': car})
    return _finish(rows, groups_post, period, verbose)


def _finish(rows, groups_post, period, verbose):
    # chain A only sets R at the DOE; it is order-independent, so read it from
    # the cached chain-A field rather than re-deriving it.
    _env, R_doe, dx_doe, P_in = C.chain_a()
    for r in rows:
        car = la.TiltedCarrier(float(R_doe), r['L_doe'], r['M_doe'])
        r['carrier'] = car
        # BACK APERTURE = the last group's exit vertex: final_distance = 0.
        x_a, y_a, L_a, M_a = _chain_chief_ray_at_target(
            groups_post, C.LAM, car, 0.0, 'sumap_census_121')
        # and the image plane, for the readout windows
        x_t, y_t, L_t, M_t = _chain_chief_ray_at_target(
            groups_post, C.LAM, car, C.TRAILING, 'sumap_census_121')
        r.update({'x_ap': x_a, 'y_ap': y_a, 'L_ap': L_a, 'M_ap': M_a,
                  'x_tgt': x_t, 'y_tgt': y_t})
    # paraxial carrier radius at the back aperture (the chain's own closure:
    # R += gap, then the group's Moebius map)
    R_ap = float(R_doe)
    for g in groups_post:
        R_ap = R_ap + float(g['gap_before'])
        R_ap = float(_paraxial_group_r_out(g['prescription'], R_ap, C.LAM))
    for r in rows:
        r['R_ap'] = R_ap
    L = np.array([r['L_ap'] for r in rows])
    M = np.array([r['M_ap'] for r in rows])
    dL = float(L.max() - L.min())
    dM = float(M.max() - M.min())
    dtilt = float(np.hypot(dL, dM))
    # (a) TILT-ONLY Nyquist: the residual ramp between the extreme orders,
    #     sampled at 2 px/period, then the campaign's 2x margin.
    dx_tilt = C.LAM / (2.0 * max(dtilt, 1e-30))
    # (b) UNION BAND: the sum's angular support about the COMMON carrier is
    #     max|L_k - L_common| + NA_k, and a real grid must hold the largest
    #     |f| of that support, not just the spread.
    Lc, Mc = L[0], M[0]                 # common carrier = the central frame
    off = float(np.max(np.hypot(L - Lc, M - Mc)))
    dx_union = C.LAM / (2.0 * (off + NA_EXIT_MEASURED))
    # (b) THE INTER-ORDER RAMP.  At this plane the orders are separated
    #     SPATIALLY, not angularly: every exit chief ray is (near) parallel to
    #     the axis and each order's converging sphere is the SAME radius
    #     DISPLACED to its own chief ray.  Referenced against ONE common
    #     sphere, order k's residual wavefront therefore carries a ramp of
    #     |c_k - c_common| / |R| radians of direction cosine -- which is the
    #     inter-order "tilt" the common grid has to sample, and it is 300x the
    #     chief-ray cosine spread in (a).  It does NOT add to (c): see (c).
    cx = np.array([r['x_ap'] for r in rows])
    cy = np.array([r['y_ap'] for r in rows])
    slope = float(np.max(np.hypot(cx - cx[0], cy - cy[0])) / abs(R_ap))
    dx_slope = C.LAM / (2.0 * max(slope, 1e-30))
    # (b') the sum-of-both bound, kept only to show it is NOT the operative
    # one: a beam's absolute band does not move with its chief ray.
    dx_full = C.LAM / (2.0 * (slope + NA_EXIT_MEASURED))
    # (c) THE OPERATIVE BOUND: each beam's own absolute band over its own
    # support, sin(theta) = r / sqrt(r^2 + R^2) at r = R_SUPPORT_99999.
    na_own = R_SUPPORT_99999 / np.hypot(R_SUPPORT_99999, R_ap)
    dx_own = C.LAM / (2.0 * na_own)
    out = {'rows': rows, 'period': period, 'R_doe': R_doe, 'dx_doe': dx_doe,
           'P_in': P_in, 'dL': dL, 'dM': dM, 'dtilt': dtilt, 'R_ap': R_ap,
           'dx_tilt_bound': dx_tilt, 'dx_tilt_margin2': dx_tilt / 2.0,
           'tilt_offset_max': off, 'dx_union_bound': dx_union,
           'dx_union_margin2': dx_union / 2.0,
           'carrier_slope': slope, 'dx_slope_bound': dx_slope,
           'dx_slope_margin2': dx_slope / 2.0,
           'dx_full_bound': dx_full, 'dx_full_margin2': dx_full / 2.0,
           'na_own': float(na_own), 'dx_own_bound': float(dx_own),
           'dx_own_margin2': float(dx_own) / 2.0,
           'x_ap_spread': float(np.ptp([r['x_ap'] for r in rows])),
           'y_ap_spread': float(np.ptp([r['y_ap'] for r in rows]))}
    if verbose:
        print(f"DOE period {period * 1e6:.4f} um, R_doe {R_doe * 1e3:.4f} mm, "
              f"dx_doe {dx_doe * 1e6:.4f} um")
        print()
        print("order    a(m,n)                 tilt@DOE (mrad)   "
              "BACK APERTURE chief (mm)   exit cosines      image (mm)")
        for r in rows:
            print(f"  {str(r['order']):>9s}  {abs(r['weight']):.6f}"
                  f"e^i{np.angle(r['weight']):+.4f}  "
                  f"({r['L_doe'] * 1e3:+8.4f},{r['M_doe'] * 1e3:+8.4f})  "
                  f"({r['x_ap'] * 1e3:+9.5f},{r['y_ap'] * 1e3:+9.5f})  "
                  f"({r['L_ap']:+.6f},{r['M_ap']:+.6f})  "
                  f"({r['x_tgt'] * 1e3:+8.4f},{r['y_tgt'] * 1e3:+8.4f})")
        print()
        print(f"inter-order TILT SPREAD at the back aperture: "
              f"dL={dL:.6f}  dM={dM:.6f}  |d|={dtilt:.6f} rad")
        print(f"  (a) tilt-only Nyquist   dx <= lambda/(2*|d|) = "
              f"{dx_tilt * 1e6:.4f} um   -> 2x margin {dx_tilt / 2 * 1e6:.4f} um")
        print(f"  (b) union band          max|L-Lc| = {off:.6f} + NA "
              f"{NA_EXIT_MEASURED:.4f}  -> dx <= {dx_union * 1e6:.4f} um   "
              f"-> 2x margin {dx_union / 2 * 1e6:.4f} um")
        print()
        print(f"back-aperture carrier radius R = {R_ap * 1e3:.6f} mm "
              f"(paraxial closure, order-independent)")
        print(f"chief-ray spread at the back aperture: "
              f"x {out['x_ap_spread'] * 1e6:.3f} um, "
              f"y {out['y_ap_spread'] * 1e6:.3f} um")
        print(f"  (b) CARRIER-OFFSET RAMP -- the inter-order 'tilt' a common "
              f"carrier leaves: max|c_k - c_0|/|R| = {slope:.6f} rad")
        print(f"      ramp-only Nyquist  dx <= {dx_slope * 1e6:.4f} um  "
              f"-> 2x margin {dx_slope / 2 * 1e6:.4f} um")
        print(f"  (b') ramp + measured exit NA (NOT operative -- an order's")
        print(f"       absolute band does not move with its chief ray):")
        print(f"       dx <= {dx_full * 1e6:.4f} um  -> 2x margin "
              f"{dx_full / 2 * 1e6:.4f} um")
        print(f"  (c) EACH BEAM'S OWN ABSOLUTE BAND over its own support "
              f"(r99.999 = {R_SUPPORT_99999 * 1e3:.2f} mm):  NA {na_own:.4f}")
        print(f"      dx <= {dx_own * 1e6:.4f} um  -> 2x margin "
              f"{dx_own / 2 * 1e6:.4f} um   <== GRID PITCH BOUND")
        print(f"  chosen: dx_c = 1.2292 um, N_c = 8192 (window "
              f"{8192 * 1.2292e-6 * 1e3:.4f} mm); pitch-converged against "
              f"0.6146/16384 -- PROBE_SUM_AT_APERTURE_2026_08_11 sec 8.1")
    return out


if __name__ == '__main__':
    census()
