# C11 -- the DIRECT DISCRIMINATOR on design 121's OWN element calls.
#
# ``c11_synth_sweep.py`` shows, on three synthetic geometries, that the two
# candidate ray fits can be RANKED at the decision site itself -- build both
# from the SAME traced samples, score each against those samples with the
# beam's own intensity as the weight, and the cheaper-residual one is the one
# that produces the better exit field (42/42 against the fit-domain-free
# spline oracle).
#
# Design 121 is the regime that oracle CANNOT reach (niche C2: at the chain's
# 33 um element pitch ``newton_fit='spline'`` fails to converge for 100 % of
# pixels and returns an ALL-ZERO field), so this asks the narrower question the
# design can answer: WHAT WOULD THE DISCRIMINATOR DECIDE, per group, per order?
# The chain-level EE3 arms (``rc_gate_121.py``) then say whether that decision
# was right.
#
# Each of the chain's six post-DOE element calls is captured once and REPLAYED
# with each branch forced.  The replay is aborted as soon as the three
# ``_Cheb2DEvaluator`` fits exist -- the Newton inversion below them is the
# expensive part and nothing here reads it.
#
# LOCAL-ONLY; no library edit (branches forced through the module attributes,
# fits captured by a script-side monkeypatch, both inside try/finally).
#
# usage:  ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python c11_discrim_121.py
import hashlib
import os
import sys
import time
import warnings

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import energy_stage_audit_121 as SA                            # noqa: E402
import lumenairy.elements._lens_traced as LT                   # noqa: E402
from lumenairy.elements._lens_traced import (                  # noqa: E402
    _input_beam_amp_radius)

WL = SA.LAM


class _Stop(Exception):
    pass


class FitSpy(object):
    """Capture the three ray fits and ABORT the call (the Newton inversion
    under them costs minutes and nothing here reads it)."""

    def __init__(self, n_stop=3):
        self.n_stop = int(n_stop)

    def __enter__(self):
        self.seen = []
        self._orig = LT._Cheb2DEvaluator.__init__
        seen = self.seen
        n_stop = self.n_stop

        def spy(zelf, xs_in, ys_in, values, order=6, xp=None, weights=None):
            self._orig(zelf, xs_in, ys_in, values, order=order, xp=xp,
                       weights=weights)
            seen.append({'ev': zelf, 'xs': np.asarray(xs_in),
                         'ys': np.asarray(ys_in),
                         'values': np.asarray(values), 'order': int(order),
                         'weights': (None if weights is None
                                     else np.asarray(weights))})
            if len(seen) >= n_stop:
                raise _Stop()

        LT._Cheb2DEvaluator.__init__ = spy
        return self

    def __exit__(self, *e):
        LT._Cheb2DEvaluator.__init__ = self._orig
        return False


def fits_for(E_in, kw, arm):
    old = (LT._DECENTRE_GATE_PIXELS, LT._DECENTRE_GATE_W_FRAC)
    if arm == 'conc':
        LT._DECENTRE_GATE_W_FRAC = float('inf')
    elif arm == 'off':
        LT._DECENTRE_GATE_PIXELS = 0.0
        LT._DECENTRE_GATE_W_FRAC = 0.0
    try:
        with FitSpy() as sp:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    LT.apply_real_lens_traced(E_in, **kw)
                except _Stop:
                    pass
        return sp.seen
    finally:
        LT._DECENTRE_GATE_PIXELS, LT._DECENTRE_GATE_W_FRAC = old


def discriminate(E_in, kw, c, w):
    sc = fits_for(E_in, kw, 'conc')
    so = fits_for(E_in, kw, 'off')
    if len(sc) < 3 or len(so) < 3:
        return None
    xs, ys = so[0]['xs'], so[0]['ys']
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    wgt = np.exp(-2.0 * (((X - c[0]) ** 2 + (Y - c[1]) ** 2) / (w * w)))
    out = {}
    for name, S in (('conc', sc), ('off', so)):
        r_xy, r_opl, cov = 0.0, 0.0, 1.0
        for k, rec in enumerate(S):
            truth = so[k]['values']              # the traced map itself
            pred = np.asarray(rec['ev'].ev(X, Y))
            ok = np.isfinite(truth) & np.isfinite(pred)
            ww = np.where(ok, wgt, 0.0)
            tot = float(ww.sum())
            if tot <= 0:
                continue
            cov = min(cov, tot / float(wgt.sum()))
            r = np.where(ok, pred - truth, 0.0)
            v = float(np.sqrt((ww * r * r).sum() / tot))
            if k < 2:
                r_xy = float(np.hypot(r_xy, v))
            else:
                r_opl = v
        out[name] = {'xy_rms': r_xy, 'opl_waves': r_opl / WL,
                     'order': S[0]['order'],
                     'weighted': S[0]['weights'] is not None, 'cov': cov}
    out['pick'] = ('conc' if out['conc']['opl_waves'] < out['off']['opl_waves']
                   else 'off')
    out['ratio'] = (out['conc']['opl_waves']
                    / max(out['off']['opl_waves'], 1e-300))
    return out


def main():
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get(
                  'ORDERS', '0,0 -1,0 -2,0 -3,0 -4,0 -4,-2').split()]
    rs = int(os.environ.get('RS', '4'))
    print(f"   lib {os.path.basename(LT.__file__)}  sha256 "
          f"{hashlib.sha256(open(LT.__file__, 'rb').read()).hexdigest()[:16]}")
    print(f"   shipped gate: pixels={LT._DECENTRE_GATE_PIXELS} "
          f"w_frac={LT._DECENTRE_GATE_W_FRAC}   "
          f"_DECENTRED_FIT_POLY_ORDER={LT._DECENTRED_FIT_POLY_ORDER}  "
          f"_REMAP_RESID_EIKONAL_DEGREE={LT._REMAP_RESID_EIKONAL_DEGREE}")
    hdr = (f"{'order':>9} {'grp':>4} {'|c|/w':>8} {'w mm':>8} {'ship':>6} "
           f"{'ord_c':>6} {'ord_o':>6} {'D:opl_c wv':>11} {'D:opl_o wv':>11} "
           f"{'c/o':>9} {'PICK':>6} {'cov_c':>6}")
    print()
    print(hdr)
    print('-' * len(hdr), flush=True)
    for order in orders:
        t0 = time.time()
        calls, _res, _cw, _s = SA.run(order, 'ship', rs=rs)
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
            ship = 'conc' if u <= LT._DECENTRE_GATE_W_FRAC else 'off'
            d = discriminate(E_in, kw, (x0, y0), w)
            if d is None:
                print(f"{str(order):>9} {k:>4} {u:>8.4f}   NO FITS")
                continue
            print(f"{str(order):>9} {k:>4} {u:>8.4f} {w*1e3:>8.4f} "
                  f"{ship:>6} {d['conc']['order']:>6} {d['off']['order']:>6} "
                  f"{d['conc']['opl_waves']:>11.4e} "
                  f"{d['off']['opl_waves']:>11.4e} {d['ratio']:>9.3f} "
                  f"{d['pick']:>6} {d['conc']['cov']:>6.3f}", flush=True)
        print(f"          [{time.time() - t0:.0f}s]\n", flush=True)
    print("D:opl_* = beam-intensity-weighted rms of each candidate fit's OPL "
          "residual\n          against the TRACED samples, in waves.  "
          "cov_c = fraction of the beam\n          weight the concentric "
          "candidate's finite (un-masked) samples cover.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
