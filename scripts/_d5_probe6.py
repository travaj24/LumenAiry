"""THE PRODUCTION COMPARISON G8 IS TRYING TO MAKE.

G8 scores both arms at HELD-OUT LAUNCH NODES.  Exit pixels are never launch
nodes, so what decides the returned field is each arm's error BETWEEN the
nodes.  Score all three there, against the element's own landings from a trace
at half the ray subsample (the lattices nest exactly; verified).

  arm A   the inverse-characteristic model, shipped coefficients
  arm B   the incumbent Newton, polynomial basis
  arm B'  the incumbent Newton, spline basis
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)
sys.path.insert(1, os.path.join(_ROOT, 'tests', 'unit'))

import warnings

import numpy as np
import test_niche_c6_stationary_phase_launch as C6  # noqa: E402

from lumenairy.elements import _lens_imap as IM  # noqa: E402

_K0 = C6._K0


def capture(basis, rs):
    cap = {}
    orig = IM.build_inverse_map

    def wrapper(xs_in, XO, YO, OP, *a, **kw):
        cap['xs'] = np.asarray(xs_in).copy()
        cap['XO'] = np.asarray(XO).copy()
        cap['YO'] = np.asarray(YO).copy()
        cap['OP'] = np.asarray(OP).copy()
        cap['inv'] = kw.get('parity_invert')
        out = orig(xs_in, XO, YO, OP, *a, **kw)
        cap['model'] = out
        return out

    IM.build_inverse_map = wrapper
    try:
        E, _X, _Y = C6._field()
        rec = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            C6._run(E, True, newton_fit=basis, ray_subsample=rs,
                    _imap_out=rec)
    finally:
        IM.build_inverse_map = orig
    cap['rec'] = rec
    return cap


if __name__ == '__main__':
    IM.TRACED_INVERSE_MAP = True
    IM._IMAP_CACHE_SIZE = 0
    fine = capture('polynomial', 2)
    xs2, XO2, YO2, OP2 = fine['xs'], fine['XO'], fine['YO'], fine['OP']

    cp = capture('polynomial', 4)
    cs = capture('spline', 4)
    xs4 = cp['xs']
    X4, Y4 = np.meshgrid(xs4, xs4, indexing='ij')
    ok4 = np.isfinite(cp['XO'])
    rmax = float(np.nanmax((X4 ** 2 + Y4 ** 2)[ok4]))

    xm = xs2[1::2]
    XM, YM = np.meshgrid(xm, xm, indexing='ij')
    XOm, YOm, OPm = XO2[1::2, 1::2], YO2[1::2, 1::2], OP2[1::2, 1::2]
    m = (np.isfinite(XOm) & np.isfinite(YOm) & np.isfinite(OPm)
         & ((XM ** 2 + YM ** 2) <= rmax))
    xq, yq = XOm[m], YOm[m]
    xt, yt, ot = XM[m], YM[m], OPm[m]
    print('%d midpoint probes inside the model\'s own fit disc' % int(m.sum()))

    model = cp['model']
    assert model is not None, 'the polynomial arm refused; nothing to score'
    ch = [np.empty(xq.shape), np.empty(xq.shape), np.empty(xq.shape)]
    model.eval_into(xq, yq, ch,
                    channels=[IM.InverseCharacteristic.CH_X_IN,
                              IM.InverseCharacteristic.CH_Y_IN,
                              IM.InverseCharacteristic.CH_OPL])

    def score(px, py, po, tag):
        px = np.ravel(px)
        py = np.ravel(py)
        po = np.ravel(po)
        e_pos = float(np.nanmax(np.maximum(np.abs(px - xt), np.abs(py - yt))))
        d = po - ot
        d = d - np.nanmedian(d)            # OPL is referenced, piston-free
        e_opl = float(np.nanmax(np.abs(d)) / C6._WL)
        print('  %-34s  pos %.4e m   OPL %.4e waves' % (tag, e_pos, e_opl))
        return e_pos, e_opl

    print('AT MIDPOINTS -- the production comparison:')
    a = score(ch[0], ch[1], ch[2], 'A   the model (shipped coeffs)')
    b = score(*cp['inv'](xq, yq), 'B   incumbent Newton, polynomial')
    c = score(*cs['inv'](xq, yq), "B'  incumbent Newton, spline")
    print('  model / poly-incumbent   pos %.4f   OPL %.4f'
          % (a[0] / b[0], a[1] / b[1]))
    print('  model / spline-incumbent pos %.4f   OPL %.4f'
          % (a[0] / c[0], a[1] / c[1]))
    print('AT THE HELD-OUT NODES -- what G8 records:')
    for tag, r in (('polynomial', cp['rec']), ('spline', cs['rec'])):
        print('  %-10s a_pos %.4e  b_pos %.4e  ratio %.3f -> %s'
              % (tag, r.get('parity_map_pos_m'),
                 r.get('parity_incumbent_pos_m'),
                 r.get('parity_ratio_pos'),
                 'ENGAGE' if r.get('engaged') else 'REFUSE ' + str(r.get('refused'))))
