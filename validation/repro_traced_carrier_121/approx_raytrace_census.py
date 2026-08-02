# Approximation audit -- the RAYTRACE / GLASS layer under the traced-carrier
# path, measured on design 121's own post-DOE relay.
#
# Nothing here is a knob sweep: each number is a direct residual of the thing
# the library actually computed, so there is no differential floor to worry
# about (the floor is float64 round-off and is reported).
#
# Measures:
#   1. surface census -- which of the relay's surfaces take the CLOSED-FORM
#      quadratic intersection and which take the 10-iteration Newton one;
#   2. the Newton intersection's own residual |z - sag(x,y)| at every recorded
#      intersection of a dense real bundle, converted to an OPL error in waves;
#   3. the ray-death census (does the 10-iteration cap ever bind, i.e. does a
#      live ray get silently reclassified as a vignette?);
#   4. glass -- the picometre wavelength quantisation of the value cache, and
#      which of design 121's glasses resolve through the external
#      ``refractiveindex`` package rather than a closed-form Sellmeier;
#   5. the DOE's medium (does the missing 1/n2 factor in the grating kick bite
#      on this design?).
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import approx_common as A                                      # noqa: E402

from lumenairy.raytrace import (RAY_APERTURE, RAY_EVANESCENT,  # noqa: E402
                                RAY_MISSED_SURFACE, RAY_NAN, RAY_OK, RAY_TIR,
                                trace)
from lumenairy.raytrace.trace import _make_bundle              # noqa: E402
from lumenairy.raytrace.surface import _surface_sag_xy          # noqa: E402
from lumenairy.glass import get_glass_index, GLASS_REGISTRY     # noqa: E402

LAM = A.LAM
K0 = A.K0


def census(surfs):
    print("1. SURFACE CENSUS (post-DOE relay, %d surfaces)" % len(surfs))
    n_closed = n_newton = 0
    for i, s in enumerate(surfs):
        asph = bool(s.aspheric_coeffs)
        flat = not np.isfinite(s.radius)
        closed = flat or (s.conic == 0.0 and not asph)
        n_closed += closed
        n_newton += (not closed)
        if not closed:
            print("   surf %2d %-14s R %12.6f mm  conic %+.6f  asph %s"
                  "  -> NEWTON" % (i, s.label, s.radius * 1e3, s.conic,
                                   sorted(s.aspheric_coeffs or {})))
    print("   closed-form (flat or conic==0, no asphere): %d" % n_closed)
    print("   Newton (conic != 0 or aspheric):            %d" % n_newton)
    return n_newton


def intersection_residual(surfs, L, M, r_max, n=181):
    """|z - sag(x,y)| at every recorded intersection of a real bundle."""
    t = np.linspace(-r_max, r_max, n)
    X, Y = np.meshgrid(t, t, indexing='ij')
    keep = (X ** 2 + Y ** 2) <= r_max ** 2
    x, y = X[keep], Y[keep]
    rays = _make_bundle(x, y, np.full_like(x, L), np.full_like(x, M), LAM)
    res = trace(rays, surfs, LAM, output_filter='all')
    print("\n2. INTERSECTION RESIDUAL |z - sag(x,y)| at every surface")
    print("   bundle: %d rays, radius %.3f mm, launch (L,M)=(%.5f,%.5f)"
          % (x.size, r_max * 1e3, L, M))
    worst = 0.0
    for i, (r, s) in enumerate(zip(res.ray_history, surfs)):
        al = r.alive
        if not np.any(al):
            continue
        sag = _surface_sag_xy(r.x[al], r.y[al], s)
        d = np.abs(r.z[al] - sag)
        n_pre = get_glass_index(s.glass_before, LAM)
        waves = d.max() * n_pre / LAM
        worst = max(worst, float(waves))
        if d.max() > 0:
            print("   surf %2d %-14s alive %6d  max|dz| %9.3e m "
                  "(%.3e waves of OPL at n=%.4f)"
                  % (i, s.label, int(al.sum()), d.max(), waves, n_pre))
    print("   WORST intersection-residual OPL error: %.3e waves" % worst)
    return worst


def death_census(surfs, L, M, r_max, n=181):
    t = np.linspace(-r_max, r_max, n)
    X, Y = np.meshgrid(t, t, indexing='ij')
    keep = (X ** 2 + Y ** 2) <= r_max ** 2
    x, y = X[keep], Y[keep]
    rays = _make_bundle(x, y, np.full_like(x, L), np.full_like(x, M), LAM)
    res = trace(rays, surfs, LAM, output_filter='all')
    print("\n3. RAY-DEATH CENSUS (which code kills a ray, and where)")
    names = {RAY_OK: 'OK', RAY_TIR: 'TIR', RAY_APERTURE: 'APERTURE',
             RAY_MISSED_SURFACE: 'MISSED/NEWTON-FAIL', RAY_NAN: 'NAN',
             RAY_EVANESCENT: 'EVANESCENT'}
    prev = np.zeros(x.size, dtype=bool)
    for i, (r, s) in enumerate(zip(res.ray_history, surfs)):
        dead = ~r.alive
        new = dead & ~prev
        if new.any():
            codes, cnt = np.unique(r.error_code[new], return_counts=True)
            print("   surf %2d %-14s +%d dead: %s" % (
                i, s.label, int(new.sum()),
                ", ".join("%s x%d" % (names.get(int(c), str(c)), int(k))
                          for c, k in zip(codes, cnt))))
        prev = dead
    print("   survivors: %d / %d" % (int(res.ray_history[-1].alive.sum()),
                                     x.size))
    n_missed = int((res.ray_history[-1].error_code == RAY_MISSED_SURFACE).sum())
    print("   RAY_MISSED_SURFACE total (this is where a 10-iteration Newton "
          "non-convergence would hide): %d" % n_missed)
    return n_missed


def glass_report(surfs):
    print("\n4. GLASS / DISPERSION")
    gl = sorted({s.glass_before for s in surfs} | {s.glass_after for s in surfs}
                - {'air'})
    print("   glasses on the relay: %s" % ", ".join(gl))
    print("   %-10s %-26s %-14s %-14s" %
          ('glass', 'registry entry', 'n(lam)', 'dn over 0.5 pm'))
    for g in gl:
        e = GLASS_REGISTRY.get(g)
        n0 = get_glass_index(g, LAM)
        n1 = get_glass_index(g, LAM + 0.5e-12)
        kind = e if isinstance(e, str) else 'TUPLE %s' % (e,)
        print("   %-10s %-26s %-14.10f %-14.3e" % (g, str(kind)[:26], n0,
                                                   n1 - n0))
    # OPL sensitivity: worst glass thickness on the relay
    tmax = max((s.thickness for s in surfs
                if s.glass_after not in ('air',)), default=0.0)
    dn = 1e-8
    print("   worst glass thickness on the relay: %.3f mm" % (tmax * 1e3))
    print("   an index quantisation of dn=1e-8 over that path is "
          "%.3e waves of OPL" % (dn * tmax / LAM))


def doe_report():
    print("\n5. DOE MEDIUM (the grating kick has no 1/n2 factor)")
    rx = A.la.load_zemax_zmx(A.C.ZMX)
    for d in rx.get('diffractives', []):
        print("   DGRATING: keys %s" % sorted(d))
        for k in ('period', 'order', 'angle_deg', 'glass_before',
                  'glass_after', 'gap_before', 'gap_after'):
            if k in d:
                print("      %-12s %s" % (k, d[k]))
    surfs = rx['surfaces']
    for i, s in enumerate(surfs):
        if 'DGRAT' in str(getattr(s, 'label', '')).upper() or \
                str(rx.get('surface_types', [''] * len(surfs))[i]).upper() \
                .startswith('DGRAT'):
            print("      surface %d glass_before=%r glass_after=%r"
                  % (i, s.glass_before, s.glass_after))


def main():
    order = tuple(int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    _pre, post, _gap, period = A.C.geometry()
    surfs = A.C.post_surfaces(post)
    L = order[0] * LAM / period
    M = order[1] * LAM / period
    r_max = float(os.environ.get('RMAX', 3.2)) * 1e-3
    census(surfs)
    intersection_residual(surfs, L, M, r_max)
    death_census(surfs, L, M, r_max)
    glass_report(surfs)
    try:
        doe_report()
    except Exception as exc:                                    # noqa: BLE001
        print("   DOE report unavailable: %s" % exc)


if __name__ == '__main__':
    main()
