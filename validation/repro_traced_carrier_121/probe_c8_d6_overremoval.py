# niche C8 on niche D6's decentred fixture: IS THE REMOVAL ALL MANUFACTURED?
#
# ORACLE_ENERGY_AND_D6_HALO_2026_08_01 S7.1(2) raises the right question: on
# D6's fixture the C8 support bound drops the chain-exit power by 3.17e-03 of
# the input while the halo the C7 check REPORTS carries only 6.45e-04.  4.9x
# more is removed than the visible lobe.  Either the rest was also
# manufactured, or the bound is eating legitimate skirt light -- and the
# difference is not a matter of opinion.
#
# THE PARTITION THAT SETTLES IT.  For the element call the bound acts on, take
# the EXACT traced ray bundle (the same one the library's own halo check reads)
# and build TWO convex hulls of the exit landing points:
#
#   H_all   every ALIVE ray, whatever its entrance                   \
#   H_ap    only the rays the ENTRANCE STOP passes  (= C8's own)     /
#
# H_ap is contained in H_all.  Every removed pixel then falls in exactly one of
# three classes:
#
#   (a) OUTSIDE H_all  -- no traced ray of this call reaches it, at all.  That
#       light was manufactured by definition; removing it is the fix working.
#   (b) between H_ap and H_all -- reachable only by rays the aperture BLOCKS.
#       Geometrically those carry no energy (``_ray_density_amp_grid`` already
#       NaNs a pixel whose converged entrance is outside the stop), so light
#       there exists only because the Newton inverse found a DIFFERENT,
#       spurious entrance inside the stop.  Also manufactured.
#   (c) INSIDE H_ap -- legitimate skirt.  Must be EXACTLY ZERO: the feather
#       band lies entirely outside the hull.  A nonzero (c) is an
#       implementation bug, and this probe is its detector.
#
# Plus a feather sweep, because "the feather is eating the skirt" makes a
# prediction the hull cut does not: the removal would keep growing as the
# feather narrows.
#
# usage:  python probe_c8_d6_overremoval.py
#         LEG=exact FEATHERS='0,0.5,1,2,4' python probe_c8_d6_overremoval.py
import hashlib
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tests', 'unit'))
sys.path.insert(0, _HERE)

import d6halo_probe as DP                                   # noqa: E402,I001
import lumenairy as la                                      # noqa: E402
import lumenairy.elements._lens_traced as LT                # noqa: E402
import test_niche_d6_exact_tilted_leg as D6                 # noqa: E402


def _run(leg, bound, feather):
    old = (LT.REMAP_INVERSE_SUPPORT_BOUND, LT._SUPPORT_BOUND_FEATHER_CELLS)
    LT.REMAP_INVERSE_SUPPORT_BOUND = bool(bound)
    LT._SUPPORT_BOUND_FEATHER_CELLS = float(feather)
    car = la.TiltedCarrier(np.inf, 0.0, 0.0, D6._X0, 0.0)
    try:
        with DP.Capture() as cap:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = D6._run_chain(car, final_leg=leg, centre_out=(0.0, 0.0))
    finally:
        (LT.REMAP_INVERSE_SUPPORT_BOUND,
         LT._SUPPORT_BOUND_FEATHER_CELLS) = old
    return cap.calls, res


def _hulls(rec):
    """(H_all, H_ap) as (A, b) half-plane sets from the call's OWN exact ray
    bundle -- the same object the library builds its bound from."""
    from scipy.spatial import ConvexHull
    tr = rec['trace']
    fin = tr['result'].image_rays
    n_l = int(round(np.sqrt(fin.x.size)))
    XE = np.asarray(fin.x).reshape(n_l, n_l)
    YE = np.asarray(fin.y).reshape(n_l, n_l)
    ok = (np.asarray(fin.alive, bool).reshape(n_l, n_l)
          & np.isfinite(XE) & np.isfinite(YE))
    xs = tr['h_x'].reshape(n_l, n_l)[:, 0]
    ap = rec['kwargs'].get('aperture')
    if ap is None:
        presc = rec['kwargs'].get('prescription')
        if presc is None and rec['args']:
            presc = rec['args'][0]
        ap = (presc or {}).get('aperture_diameter')
    ok_ap = ok
    if ap is not None:
        ok_ap = ok & ((xs[:, None] ** 2 + xs[None, :] ** 2)
                      <= (0.5 * float(ap)) ** 2)
    out = []
    for m in (ok, ok_ap):
        eq = ConvexHull(np.column_stack([XE[m], YE[m]])).equations
        out.append((np.ascontiguousarray(eq[:, :2].T),
                    np.ascontiguousarray(eq[:, 2])))
    return out[0], out[1], (XE, YE, ok, ok_ap, float(ap) if ap else None)


def _signed(hull, X, Y):
    A, b = hull
    s = (np.column_stack([X.ravel(), Y.ravel()]) @ A + b).max(axis=1)
    return s.reshape(X.shape)


def _outside(hull, X, Y):
    return _signed(hull, X, Y) > 0.0


def _within(hull, X, Y, d):
    return _signed(hull, X, Y) <= d


def main():
    leg = os.environ.get('LEG', 'paraxial')
    feathers = [float(v) for v in
                os.environ.get('FEATHERS', '0,0.5,1,2,4').split(',')]
    print("   lib sha256 %s" % hashlib.sha256(
        open(LT.__file__, 'rb').read()).hexdigest()[:16])
    print(f"niche C8 vs niche D6's decentred fixture, final_leg={leg!r}.")
    print("  Is the power the support bound removes MANUFACTURED, or is it "
          "legitimate skirt?")
    print()

    off_calls, off_res = _run(leg, False, 1.0)
    p_in = float((np.abs(off_calls[0]['E_in']) ** 2).sum())
    # the call the bound acts on = the LAST group call of the chain
    k = len(off_calls) - 1
    rec = off_calls[k]
    dx = DP._dx_of(rec)
    N = rec['E_out'].shape[0]
    ax = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(ax, ax, indexing='xy')
    H_all, H_ap, (XE, YE, ok, ok_ap, ap) = _hulls(rec)
    out_all = _outside(H_all, X, Y)
    out_ap = _outside(H_ap, X, Y)
    print(f"  scored call [{k}] of {len(off_calls)}: N={N} dx={dx*1e6:.4f} um "
          f"aperture={'%.4f mm' % (ap*1e3) if ap else 'None'}")
    print(f"  alive rays {int(ok.sum())}, of which the stop passes "
          f"{int(ok_ap.sum())}")
    print(f"  exit map |r| max (alive) {float(np.hypot(XE[ok], YE[ok]).max())*1e3:.4f}"
          f" mm, (stop-passing) "
          f"{float(np.hypot(XE[ok_ap], YE[ok_ap]).max())*1e3:.4f} mm")
    print(f"  exit pixels outside H_all {int(out_all.sum())}, outside H_ap "
          f"{int(out_ap.sum())}, of {N*N}")
    print()

    # The band the bound acts in -- everything within 6 coarse cells of the
    # hull, where a truncation step would live.  ``jump`` below is measured
    # over it and nowhere else, so the beam's own structure cannot enter.
    _sub = 4
    _band = out_all & _within(H_all, X, Y, 6.0 * _sub * dx)

    def _jump(F):
        """Largest nearest-neighbour |E| step, over the peak, inside the
        boundary band -- a grid-local measure of how sharp a discontinuity the
        cut leaves in the returned field."""
        a = np.abs(F)
        pk = float(a.max())
        m = 0.0
        for d, sl_a, sl_b in ((0, (slice(None), slice(1, None)),
                               (slice(None), slice(None, -1))),
                              (1, (slice(1, None), slice(None)),
                               (slice(None, -1), slice(None)))):
            j = np.abs(a[sl_a] - a[sl_b])
            mm = _band[sl_a] | _band[sl_b]
            if mm.any():
                m = max(m, float(j[mm].max()))
        return m / pk if pk > 0 else 0.0

    hdr = (f"{'feather':>8} {'dP/P_in':>11} "
           f"{'(a) out H_all':>13} {'(b) H_ap..H_all':>15} "
           f"{'(c) INSIDE H_ap':>15} {'amax_all':>9} {'edge jump':>10}")
    print(hdr)
    print('-' * len(hdr))

    def _row(lbl, calls):
        F = np.asarray(calls[k]['E_out'])
        d = np.abs(np.asarray(off_calls[k]['E_out'])) ** 2 - np.abs(F) ** 2
        tot = float(d.sum()) / p_in
        a = float(d[out_all].sum()) / p_in
        b = float(d[out_ap & ~out_all].sum()) / p_in
        c = float(d[~out_ap].sum()) / p_in
        aE = np.abs(F)
        pk = float(aE.max())
        amax = (float(aE[out_all].max()) / pk
                if out_all.any() and pk > 0 else 0.0)
        print(f"{lbl:>8} {tot:>11.3e} {a:>13.3e} {b:>15.3e} "
              f"{c:>15.3e} {amax:>9.2e} {_jump(F):>10.3e}", flush=True)

    _row('OFF', off_calls)
    for f in feathers:
        c2, _r2 = _run(leg, True, f)
        _row(f'{f:g}', c2)
    print()
    print("dP = P(bound OFF) - P(this row), over the chain INPUT power.")
    print("(a) no traced ray of the call reaches there at all -> "
          "manufactured by definition.")
    print("(b) reachable only by rays the entrance stop BLOCKS -> light there "
          "requires a spurious inverse root.")
    print("(c) inside the support the bound itself uses -> MUST be exactly 0; "
          "anything else is a bug.")
    print("amax_all  = max |E| outside H_all over the peak, on THIS row's "
          "field.")
    print("edge jump = largest nearest-neighbour |E| step over the peak, "
          "within 6 coarse cells of the hull;")
    print("            the OFF row is the field's own, so a feather that "
          "matches it introduces no new edge.")


if __name__ == '__main__':
    sys.exit(main())
