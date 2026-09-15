"""Mutation evidence for every bar restated in

    tests/unit/test_audit2609_a6_carrier.py
    tests/unit/test_niche_audit_w3_oracles.py

Three of the four mutations need no library edit and run here.  The two that
DO need one are applied to an archived copy of the tree (never the shared
working tree); the recipe is at the bottom of this docstring.

  C2-M3   a GENUINE centring error (the defect the 16-ulp bar guards):
          feed the same estimator a beam decentred by 0.5 / 1 / 2 waists.
  W3-M4   the Van Vleck derivation, broken four ways, against the 1e-11
          frozen-to-frozen bar.
  W3-M5   the measured ``L``, perturbed, against the 1e-6 bar.
  W3-M6   the LG-merit rescales the [1e8, 1e11] band must catch.

  C4-M1/M2 (archived tree required -- carrier.py is owned elsewhere):
      D=/c/tmp/lum_mut; rm -rf $D; mkdir -p $D
      git -C /c/tmp/lum_reds archive HEAD lumenairy | tar -x -C $D
      cp -r /c/tmp/lum_reds/tests $D/
      # M1: in $D/lumenairy/propagators/carrier.py, replace the untilted
      #     fast path's three in-place lines
      #         phase -= root0 ; phase *= z_eff ; phase += k * z_eff
      #     with   phase = z_eff * (phase - root0 + k)
      # M2: after the cos/sin block in _exact_envelope_tf_step, insert
      #         H *= (1.0 + 1e-12)
      cd $D && OMP_NUM_THREADS=1 PYTHONPATH=$D python -m pytest \
        tests/unit/test_audit2609_a6_carrier.py -q --capture=sys \
        -k tf_step_is_bit_identical -p no:cacheprovider

  MEASURED 2026-09-14 (Windows, py3.14.6, numpy 2.4.4):
      M1 -> the byte claim fails at EVERY untilted shape (63/64/65/128/256)
      M2 -> byte claim fails, and the shipped-dispatch bar reads rel
            1.000e-12 against its 1e-13 (clean tree: 0.000e+00)

Writes <out>/probe_mutations_a6_w3_<ARM>.json.
"""
import importlib
import json
import os
import sys
import warnings

import numpy as np

import lumenairy
from lumenairy.propagators import carrier as C

LAM_C2 = 1.31e-6
K0_C2 = 2.0 * np.pi / LAM_C2
N, DX, W, R = 512, 2e-6, 100e-6, 50e-3


def _decentred(x0):
    g = (np.arange(N, dtype=np.float64) - N / 2) * DX
    xx = g[None, :] - x0
    yy = g[:, None]
    r2 = xx * xx + yy * yy
    return np.exp(-r2 / W ** 2) * np.exp(1j * K0_C2 * r2 / (2.0 * R))


def _c2_m3():
    rows = []
    for est in ('gradient', 'increment'):
        for f in (0.0, 0.5, 1.0, 2.0):
            e = _decentred(f * W)
            a = C.carrier_referenced_fit_radius(
                e, LAM_C2, DX, estimator=est, on_aliased='silent')
            b = C.carrier_referenced_fit_radius(
                e, LAM_C2, DX, estimator=est, on_aliased='silent',
                centre='origin')
            u = float(abs(a - b) / np.spacing(abs(b)))
            rows.append(dict(estimator=est, decentre_waists=f, a=float(a),
                             b=float(b), ulp=u, passes_bar_16=bool(u <= 16.0)))
    return rows


def _w3():
    m = importlib.import_module('tests.unit.test_niche_audit_w3_oracles')
    wl = m._WL_T3B
    m4, m5 = [], []
    for (R1, _w, want), (_R, old) in zip(m._Y2_FROZEN_T3B,
                                         m._Y2_FROZEN_T3B_OLD):
        res = m._tensor_t3b(R1, ((0, 0),))
        sq = abs(complex(res.van_vleck_weight)) * wl
        for lab, f in (('shipped -1j/(lam*sqrt|detJ|)', -1j / (wl * sq)),
                       ('M4a +1j not -1j', +1j / (wl * sq)),
                       ('M4b |detJ| not sqrt|detJ|', -1j / (wl * sq * sq)),
                       ('M4c 1/lambda dropped', -1j / sq),
                       ('M4d no rotation', 1.0 / (wl * sq))):
            pred = old * f
            rel = abs(want - pred) / abs(pred)
            m4.append(dict(R1=R1, variant=lab, rel=rel,
                           passes_bar_1e_11=bool(rel < 1e-11)))
        got = complex(res.L[0, 0])
        for lab, g in (('shipped', got), ('M5a x(1+1e-5)', got * (1 + 1e-5)),
                       ('M5b x(1+1e-7)', got * (1 + 1e-7)),
                       ('M5c x(-1j)', got * (-1j))):
            rel = abs(g - want) / abs(want)
            m5.append(dict(R1=R1, variant=lab, rel=rel,
                           passes_bar_1e_6=bool(rel < 1e-6)))

    class _Ctx:
        wavelength = wl
        N = 64
        dx = 20e-6

    merit = lumenairy.LGAberrationMerit(
        targets={(2, 0): 1.0}, field_points=[(0.0, 0.0)],
        w_s=m._WS_T3B, w_p=m._WP_T3B, fit_kwargs=m._FITKW_T3B)
    c = _Ctx()
    c.prescription = m._singlet_t3b(51.5e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        va = float(merit.evaluate(c))
    ref_sq = 6.16e-10          # |L_ref(0,0)|^2, the library's own reading
    y2 = 1.490953e+13          # 1/(lambda^2 |det J|) at R1 = 51.5 mm
    m6 = [dict(variant=lab, val=v, in_band=bool(1e8 < v < 1e11))
          for lab, v in (('shipped', va),
                         ('M6a reference normalisation off', va * ref_sq),
                         ('M6b Y2 Van Vleck factor undone', va / y2),
                         ('M6c Y2 factor applied twice', va * y2))]
    return m4, m5, m6


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else 'LOCAL'
    outdir = sys.argv[2] if len(sys.argv) > 2 else (
        os.path.dirname(os.path.abspath(__file__)))
    assert 'lum_reds' in lumenairy.__file__, lumenairy.__file__
    c2 = _c2_m3()
    m4, m5, m6 = _w3()
    env = dict(arm=arm, python=sys.version.split()[0], numpy=np.__version__,
               platform=sys.platform,
               OPENBLAS_CORETYPE=os.environ.get('OPENBLAS_CORETYPE', ''))
    out = dict(env=env, c2_m3=c2, w3_m4=m4, w3_m5=m5, w3_m6=m6)
    path = os.path.join(outdir, 'probe_mutations_a6_w3_%s.json' % arm)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(env))
    print('C2-M3  bar 16 ulp:')
    for r in c2:
        print('  %-10s x0=%.1f w  ulp=%-12.4g PASS=%s'
              % (r['estimator'], r['decentre_waists'], r['ulp'],
                 r['passes_bar_16']))
    print('W3-M4  bar 1e-11 (frozen-to-frozen derivation):')
    for r in m4:
        print('  R1=%.4f %-30s rel=%-11.4g PASS=%s'
              % (r['R1'], r['variant'], r['rel'], r['passes_bar_1e_11']))
    print('W3-M5  bar 1e-6 (measured L):')
    for r in m5:
        print('  R1=%.4f %-16s rel=%-11.4g PASS=%s'
              % (r['R1'], r['variant'], r['rel'], r['passes_bar_1e_6']))
    print('W3-M6  band [1e8, 1e11] (LG merit val_a):')
    for r in m6:
        print('  %-34s val=%-14.6e IN_BAND=%s'
              % (r['variant'], r['val'], r['in_band']))
    print('WROTE', path)


if __name__ == '__main__':
    main()
