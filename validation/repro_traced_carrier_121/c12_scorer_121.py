# C12 -- the SCORER's blind spot, measured on design 121's own element calls.
#
# Niche C11's arbiter scores each candidate ray fit with the beam's own
# intensity ``exp(-2 r^2 / w^2)`` about the chief ray.  That weight is ~0
# beyond 2 w, so it cannot see what the fitted map does over the rest of the
# launch square -- which is where the Newton inversion evaluates it, and where
# the concentric hard mask leaves the fit unconstrained.  C11 S10 item 1 records
# the cost: a mis-pick at (-1,0) worth 0.026 EE3 points.
#
# This sweeps a FLOOR on that weight
#
#     w_i = max( exp(-2 |r_i - c|^2 / w^2), floor )     over the traced support
#
# and reports, per group per order, the two candidates' scores and the pick.
# ``floor = 0`` is C11 exactly.
#
# Also reports the C12 PHYSICS PREDICTOR's own verdict (the spectral-tail
# model, see c12_predict_121.py) so the two can be compared group by group.
#
# LOCAL-ONLY; no library edit.
#
# usage:  ORDERS='-1,0 -2,0' FLOORS='0 1e-8 1e-6 1e-4 1e-3 1e-2' \
#             python c12_scorer_121.py
import hashlib
import os
import sys
import time

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c11_discrim_121 as D                                     # noqa: E402
import energy_stage_audit_121 as SA                             # noqa: E402
import lumenairy.elements._lens_traced as LT                    # noqa: E402
from lumenairy.elements._lens_traced import (                   # noqa: E402
    _input_beam_amp_radius)

WL = SA.LAM


def candidates(E_in, kw, c, w):
    """The two candidate fits' (disc, weights, order) plus the traced OPL, all
    built through the library's own helpers -- so what is scored here is what
    the fit site would apply."""
    so = D.fits_for(E_in, kw, 'off')
    if len(so) < 3:
        return None
    xs = so[0]['xs']
    opl = so[2]['values']                     # weighted arm keeps every sample
    frbf = float(kw.get('fit_radius_beam_factor') or 0.0)
    ap = kw.get('aperture')
    Lr = ((0.5 * float(ap) * 1.50) if ap is not None
          else 0.5 * float(np.asarray(E_in).shape[-1]) * float(kw['dx']))
    dxk = float(kw['dx'])
    w_orig = float(_input_beam_amp_radius(E_in, dxk, dxk, centre=None))
    r_off = min(frbf * w, Lr)
    r_conc = min(frbf * w_orig, Lr)
    disc_o = (((xs[:, None] - c[0]) ** 2 + (xs[None, :] - c[1]) ** 2)
              <= r_off ** 2)
    disc_c = (xs[:, None] ** 2 + xs[None, :] ** 2) <= r_conc ** 2
    p = int(kw.get('newton_poly_order', 6) or 6)
    P = int(LT._DECENTRED_FIT_POLY_ORDER)
    wo, oo = LT._decentred_fit_restriction(disc_o, True, p, P)
    wc, oc = LT._decentred_fit_restriction(disc_c, False, p, P)
    return dict(xs=xs, opl=opl, disc_o=disc_o, disc_c=disc_c, wo=wo, oo=oo,
                wc=wc, oc=oc, r_off=r_off, r_conc=r_conc, p=p, P=P)


def main():
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get('ORDERS', '-1,0 -2,0').split()]
    floors = [float(v) for v in os.environ.get(
        'FLOORS', '0 1e-8 1e-6 1e-4 1e-3 1e-2 1e-1').split()]
    rs = int(os.environ.get('RS', '4'))
    print(f"   lib {os.path.basename(LT.__file__)}  sha256 "
          f"{hashlib.sha256(open(LT.__file__, 'rb').read()).hexdigest()[:16]}",
          flush=True)
    hdr = (f"{'order':>9} {'grp':>4} {'|c|/w':>7}"
           + ''.join(f"{('f=' + f'{f:.0e}'):>12}" for f in floors))
    print()
    print(hdr)
    print('-' * len(hdr), flush=True)
    for order in orders:
        t0 = time.time()
        calls, _r, _c, _s = SA.run(order, 'ship', rs=rs)
        for k, rec in enumerate(calls):
            kw = dict(rec['kw'])
            E_in = rec['E_in']
            car = kw.get('carrier')
            x0 = float(getattr(car, 'x0', 0.0) or 0.0)
            y0 = float(getattr(car, 'y0', 0.0) or 0.0)
            dxk = float(kw['dx'])
            dec = float(np.hypot(x0, y0))
            w = float(_input_beam_amp_radius(
                E_in, dxk, dxk, centre=((x0, y0) if dec > 0 else None)))
            u = dec / w if w > 0 else float('nan')
            C = candidates(E_in, kw, (x0, y0), w)
            if C is None:
                print(f"{str(order):>9} {k:>4} {u:>7.4f}   NO FITS")
                continue
            xs = C['xs']
            g = np.exp(-2.0 * (((xs[:, None] - x0) ** 2
                                + (xs[None, :] - y0) ** 2) / (w * w)))
            cells = []
            for f in floors:
                wgt = np.maximum(g, f) if f > 0 else g
                s_o = LT._decentred_fit_score(xs, C['opl'], wgt, C['disc_o'],
                                              C['wo'], C['oo'])
                s_c = LT._decentred_fit_score(xs, C['opl'], wgt, C['disc_c'],
                                              C['wc'], C['oc'])
                pick = 'conc' if s_c <= s_o else 'off'
                cells.append(f"{pick:>5}{s_c / max(s_o, 1e-300):7.3f}")
            print(f"{str(order):>9} {k:>4} {u:>7.4f}" + ''.join(cells),
                  flush=True)
        print(f"          [{time.time() - t0:.0f}s]\n", flush=True)
    print("cell = PICK and the concentric/off-centre score ratio at that "
          "weight floor.\nfloor 0 is niche C11 exactly.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
