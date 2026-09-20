"""VERIFY-WAVE5-E D8 -- E3's products term re-derived with FIVE roundings.

The pin's docstring derived the PRODUCTS half of its ULP bar as "``fl(s*a)``
then ``fl(re^2)``, ``fl(im^2)``, ``fl(+)`` is at most three roundings", i.e.
``3u`` per leg and ``6u`` for the ratio.  Per COMPLEX element the chain is
``fl(s*re)``, ``fl(s*im)``, ``fl((s*re)^2)``, ``fl((s*im)^2)``, ``fl(+)`` --
five, so ``5u`` per leg and ``10u`` for the ratio.

This probe reports BOTH bars on the SAME arm, so the change is measured and
not asserted, together with the readings each has to clear and the
independent-per-leg signal each has to sit below.  The reduction term is taken
from the pin's own ``_power_sum_rounding_spread``, imported from the test
module, so there is only one definition of it in play.

Run once per arm:

    OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T \
    OPENBLAS_CORETYPE=HASWELL PYTHONPATH=<tree> \
    python validation/probe_wave5_e/probe_e3_d8_products_term.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import warnings

import numpy as np

import lumenairy
from lumenairy import apply_real_lens_maslov_vector

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEST = os.path.join(os.path.dirname(os.path.dirname(_HERE)), 'tests', 'unit',
                     'test_audit2609_a4_verify_maslov_asymptotic.py')
_spec = importlib.util.spec_from_file_location('a4pin', _TEST)
A4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A4)

_spread = A4._power_sum_rounding_spread
_SAFETY = 4.0


def run():
    E_vec, kw = A4._vector_fixture()
    ratios, terms, outs = {}, {}, {}
    for mode in ('none', 'power', 'peak'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = apply_real_lens_maslov_vector(
                E_vec.copy(), normalize_output=mode, **kw)
        outs[mode] = out
        tx, ty = np.abs(out[0]) ** 2, np.abs(out[1]) ** 2
        ratios[mode] = float(np.sum(tx)) / float(np.sum(ty))
        terms[mode] = (tx, ty)
    r0 = ratios['none']
    ulp = float(np.spacing(r0))
    u = float(np.finfo(np.float64).eps) / 2.0
    reduction_rel = max(_spread(terms[m][0]) + _spread(terms[m][1])
                        for m in ('none', 'power', 'peak'))

    def bar(n_roundings_per_leg):
        product_rel = 2.0 * n_roundings_per_leg * u
        return (_SAFETY * (product_rel + reduction_rel) * r0 / ulp,
                product_rel * r0 / ulp)

    bar10, prod10 = bar(5)          # the corrected term
    bar6, prod6 = bar(3)            # what the docstring said
    readings = {m: abs(ratios[m] - r0) / ulp for m in ('power', 'peak')}
    widest = max(readings.values())

    ox, oy = outs['none'][0], outs['none'][1]
    pxin = float(np.sum(np.abs(E_vec[0]) ** 2))
    pyin = float(np.sum(np.abs(E_vec[1]) ** 2))
    ix = ox * float(np.sqrt(pxin / float(np.sum(np.abs(ox) ** 2))))
    iy = oy * float(np.sqrt(pyin / float(np.sum(np.abs(oy) ** 2))))
    r_indep = (float(np.sum(np.abs(ix) ** 2))
               / float(np.sum(np.abs(iy) ** 2)))
    indep_ulp = abs(r_indep - r0) / ulp
    return dict(
        r_none=r0, ulp=ulp,
        dev_from_input=r0 / (pxin / pyin) - 1.0,
        reading_power_ulp=readings['power'], reading_peak_ulp=readings['peak'],
        widest_reading_ulp=widest,
        reduction_ulp=reduction_rel * r0 / ulp,
        products_ulp_6u=prod6, products_ulp_10u=prod10,
        bar_ulp_6u=bar6, bar_ulp_10u=bar10,
        bar10_over_widest=bar10 / max(widest, 1e-300),
        indep_ulp=indep_ulp,
        indep_over_bar10=indep_ulp / bar10,
        bar10_clears_reading=bool(bar10 > widest),
        indep_over_1e3_bar10=bool(indep_ulp > 1e3 * bar10))


def main():
    try:
        import threadpoolctl
        pools = [dict(prefix=p.get('prefix'), arch=p.get('architecture'),
                      threads=p.get('num_threads'))
                 for p in threadpoolctl.threadpool_info()]
    except Exception as exc:                               # pragma: no cover
        pools = [{'error': str(exc)}]
    out = dict(lumenairy_file=lumenairy.__file__,
               python=sys.version.split()[0], platform=sys.platform,
               numpy=np.__version__,
               coretype=os.environ.get('OPENBLAS_CORETYPE'),
               threads=os.environ.get('OPENBLAS_NUM_THREADS'),
               threadpools=pools, result=run())
    tag = 'win32_314' if sys.platform.startswith('win') else 'linux_312'
    name = 'e3_d8_%s_%s_t%s.json' % (
        tag, os.environ.get('OPENBLAS_CORETYPE', 'default'),
        os.environ.get('OPENBLAS_NUM_THREADS', '?'))
    with open(os.path.join(_HERE, name), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps(out, indent=1, sort_keys=True))


main()
