# Conservation audit (2026-07-31), ELEMENT level: conservation + halo + the
# library's own energy self-check, on design 121's last group.
#
# THE QUESTION.  ``_lens_traced.py`` ships a post-hoc energy self-check for
# ``amplitude_model='ray_density'`` (``_RD_ENERGY_DEFICIT_BASE`` 0.080,
# ``_RD_ENERGY_DEFICIT_PER_SUB`` 0.010, ``_RD_ENERGY_GAIN_TOL`` 0.050).  The
# prior audit measured configurations returning ``P/Pin`` 1.82 and 2.21 and did
# NOT report whether the guard fired.  This script measures, per configuration:
#
#   * ``P/Pin`` on the whole grid, AND the guard's OWN ratio (exit power over
#     APERTURE-TRANSMITTED input power) recomputed here from E_in / E_out, so
#     the guard's arithmetic is reproduced rather than trusted;
#   * whether the guard's RuntimeWarning actually fired (matched on
#     "energy self-check FAILED"), alongside the fold-caustic and exit-NA
#     under-sampling warnings;
#   * halo shells about the TRACED exit chief ray and the second moment
#     (``probe_ghost_c6.halo``, reused VERBATIM so the numbers are comparable
#     with APPROXIMATION_AUDIT_POST_C6 S3.1).
#
# NOTHING is a new instrument: the element call, the halo metric and the
# capture cache are ``probe_c6_element`` / ``probe_ghost_c6``.  What is new is
# recording the guard's verdict next to the ratio it is supposed to police.
#
# A NULL INTERVENTION (two identical shipped runs) runs first for every order
# and its ``max|dE|`` is printed; no delta below that floor is a measurement.
#
# usage:  ORDERS='0,0 -4,0 -4,-2' python energy_guard_probe.py
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _d121_common as C                                          # noqa: E402
import probe_c6_element as E6                                     # noqa: E402
import probe_ghost_c6 as G6                                       # noqa: E402
import wfe_probe_common as P                                      # noqa: E402
import lumenairy.elements._lens_traced as LT                      # noqa: E402
from lumenairy.elements import apply_real_lens_traced             # noqa: E402
from lumenairy.elements._lens_traced import (TiltedCarrier,       # noqa: E402
                                             _input_beam_amp_radius)

LAM = P.LAM


def element(E_in, presc, dx, car, rs, flag, c5=None, deg=None, **over):
    """One element call with C5 / C6 controlled, returning the warning census.

    ``c5`` toggles ``TILTED_CARRIER_EXACT_EIKONAL`` (niche C5); ``flag``
    toggles ``REMAP_STATIONARY_PHASE_LAUNCH`` (niche C6)."""
    old = (LT.REMAP_STATIONARY_PHASE_LAUNCH, LT._REMAP_RESID_EIKONAL_DEGREE,
           LT._REMAP_RESID_DEGREE_CAP, LT.TILTED_CARRIER_EXACT_EIKONAL)
    LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(flag)
    if c5 is not None:
        LT.TILTED_CARRIER_EXACT_EIKONAL = bool(c5)
    if deg is not None:
        LT._REMAP_RESID_EIKONAL_DEGREE = int(deg)
        LT._REMAP_RESID_DEGREE_CAP = max(int(deg), LT._REMAP_RESID_DEGREE_CAP)
    d = {}
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            opts = dict(E6.OPTS)
            opts.update(over)
            out = np.asarray(apply_real_lens_traced(
                E_in, prescription=presc, wavelength=LAM, dx=dx, carrier=car,
                ray_subsample=rs, _remap_launch_out=d, **opts))
    finally:
        (LT.REMAP_STATIONARY_PHASE_LAUNCH, LT._REMAP_RESID_EIKONAL_DEGREE,
         LT._REMAP_RESID_DEGREE_CAP, LT.TILTED_CARRIER_EXACT_EIKONAL) = old
    fl = {'energy': 0, 'fold': 0, 'undersample': 0, 'other': 0}
    texts = []
    for w in wl:
        t = str(w.message)
        if 'energy self-check FAILED' in t:
            fl['energy'] += 1
            texts.append(t)
        elif 'fold caustic' in t:
            fl['fold'] += 1
        elif 'far-halo energy lands at wrong radii' in t or 'NA_exit' in t:
            fl['undersample'] += 1
        else:
            fl['other'] += 1
    return out, d, fl, texts


def cases_for(m, n):
    """Configurations.  ``newton_poly_order=14`` is the prior audit's
    ``decentred_fit_poly_order=14`` equivalent on the weighted branch:
    ``_fit_poly_order = max(newton_poly_order, _dec_order)``."""
    cs = [
        ('C5on  C6off  SHIPPED-minus-C6', dict(flag=False)),
        ('C5on  C6on   SHIPPED', dict(flag=True)),
        ('C5off C6off  no-C5 no-C6', dict(flag=False, c5=False)),
        ('C5off C6on   no-C5', dict(flag=True, c5=False)),
        ('C5on  C6on   fitorder 10', dict(flag=True, newton_poly_order=10)),
        ('C5on  C6on   fitorder 14', dict(flag=True, newton_poly_order=14)),
        ('C5on  C6off  fitorder 14', dict(flag=False, newton_poly_order=14)),
    ]
    return cs


def main():
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get('ORDERS', '0,0 -4,0 -4,-2').split()]
    rs = int(os.environ.get('RS', '4'))
    rn = int(os.environ.get('RN', '1024'))

    import hashlib
    print("   lib %s  sha256 %s" % (
        os.path.basename(LT.__file__),
        hashlib.sha256(open(LT.__file__, 'rb').read()).hexdigest()[:16]))
    _lo = 1.0 - (LT._RD_ENERGY_DEFICIT_BASE
                 + LT._RD_ENERGY_DEFICIT_PER_SUB * (rs - 1))
    _hi = 1.0 + LT._RD_ENERGY_GAIN_TOL
    print(f"   guard band at ray_subsample={rs}: [{_lo:.4f}, {_hi:.4f}]")
    print("   P/Pin(grid) = exit power / whole-grid input power")
    print("   P/Pin(ap)   = the GUARD's own ratio (aperture-transmitted "
          "denominator), recomputed here")
    print("   ENG = did the library's energy self-check WARN?  "
          "FLD = fold-caustic warnings.  NYQ = exit-NA undersample warnings.")
    print()

    _pre, post, _g, _p = C.geometry()
    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)
    ap = presc.get('aperture_diameter')

    for (m, n) in orders:
        E_in, _Eo, carv, dx = E6.get_call(m, n, rn=rn, rs=rs)
        car = TiltedCarrier(*carv)
        N = E_in.shape[0]
        w = float(_input_beam_amp_radius(E_in, dx, dx,
                                         centre=(car.x0, car.y0)))
        axg = (np.arange(N) - N / 2) * dx
        Xg, Yg = np.meshgrid(axg, axg)
        p_grid = float((np.abs(E_in) ** 2).sum())
        if ap is None:
            p_ap = p_grid
        else:
            _mk = Xg ** 2 + Yg ** 2 <= (ap / 2) ** 2
            p_ap = float((np.abs(E_in) ** 2)[_mk].sum())
        xo, yo, _a, _b, _c, _d = P.trace_forward([car.x0], [car.y0], car, surfs)
        Rr = np.hypot(Xg - float(xo[0]), Yg - float(yo[0]))

        # NULL INTERVENTION -- before any delta is quoted.
        a1, _q1, _q2, _q3 = element(E_in, presc, dx, car, rs, True)
        a2, _q4, _q5, _q6 = element(E_in, presc, dx, car, rs, True)
        print(f"order ({m},{n})  w {w * 1e3:.4f} mm  dec {np.hypot(car.x0, car.y0) / w:.3f} w  "
              f"exit chief ({float(xo[0]) * 1e3:+.4f},"
              f"{float(yo[0]) * 1e3:+.4f}) mm  ap cuts "
              f"{1.0 - p_ap / p_grid:.2e}")
        print(f"   [NULL: array_equal={np.array_equal(a1, a2)} "
              f"max|dE|={float(np.abs(a1 - a2).max()):.3e}]")
        hdr = (f"{'config':>30} {'ord':>4} {'P/Pin(grid)':>12} "
               f"{'P/Pin(ap)':>11} {'ENG':>4} {'FLD':>4} {'NYQ':>4} "
               f"{'g2':>10} {'g4':>10} {'amax4':>9} {'r_rms/mm':>9} "
               f"{'core/mm':>8}")
        print(hdr)
        print('-' * len(hdr))
        for lbl, kw in cases_for(m, n):
            t0 = time.time()
            F, dg, fl, txt = element(E_in, presc, dx, car, rs, **kw)
            h = G6.halo(F, p_grid, Rr)
            pw = np.abs(F) ** 2
            r_ap = (float(pw.sum()) / p_ap) if p_ap > 0 else float('nan')
            inband = _lo <= r_ap <= _hi
            print(f"{lbl:>30} {str(dg.get('degree', '-')):>4} "
                  f"{h['P']:>12.5f} {r_ap:>11.5f} "
                  f"{('WARN' if fl['energy'] else ('.' if inband else 'MISS')):>4} "
                  f"{fl['fold']:>4d} {fl['undersample']:>4d} "
                  f"{h['g2']:>10.3e} {h['g4']:>10.3e} {h['amax4']:>9.2e} "
                  f"{h['r_rms'] * 1e3:>9.4f} {h['r_rms_core'] * 1e3:>8.4f}"
                  f"   [{time.time() - t0:.0f}s]", flush=True)
        print()
    print("ENG legend: WARN = the guard fired.  '.' = did not fire AND the "
          "ratio is inside its band (correct silence).")
    print("            MISS = the ratio is OUTSIDE the band but the guard did "
          "NOT fire -- a guard defect.")


if __name__ == '__main__':
    sys.exit(main())
