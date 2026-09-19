"""VERIFY-WAVE5-E D4 / D5 / D7 -- the E2 ladder re-measured after the fixes.

Three things are measured here, on both builds, through the SHIPPED element
call (no fit order is passed anywhere):

D4  the trip count under the OLD predicate (``sup > 0 and 1/sup >= TRIP``) and
    the NEW one (``sup <= 1/TRIP``), on a ladder wide enough to contain rungs
    the bound empties COMPLETELY -- which is where the two differ.
D5  how many of the pin's seven rungs clear 10x at n = 256 and n = 512 (the
    item report and the test header said seven of seven at 512; the table
    printed above the sentence says 0.216 at the 2.0 w rung, i.e. 4.63x).
D7  an INERT cell: the four annulus ratios, and whether the two fields are
    bit-identical -- the justification for the bar's lower gap said they were.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=<tree> python validation/probe_wave5_e/probe_e2_d4_d5_d7.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

import lumenairy as la

print('lumenairy.__file__ =', la.__file__)
print('python =', sys.version.split()[0], ' numpy =', np.__version__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEST = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                     'tests', 'unit', 'test_wave5_e_c8_default_order.py')
_spec = importlib.util.spec_from_file_location('c8pin', _TEST)
C8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C8)

#: The wider ladder, out to where the bound empties the annulus completely.
_WIDE = tuple(round(2.0 + 0.25 * i, 2) for i in range(25))


def _rows(cand, n, factors):
    s = dict(C8._BASE)
    s['n'] = int(n)
    s['dx'] = (C8._GHOST['n'] * C8._GHOST['dx']) / float(n)
    s['alpha'], s['z'] = cand['alpha'], cand['z']
    off = C8._call(s, bound=False, cx=cand['cx'], frbf=cand['frbf'])
    on = C8._call(s, bound=True, cx=cand['cx'], frbf=cand['frbf'])
    old_factors = C8._FACTORS
    C8._FACTORS = tuple(factors)
    try:
        rows = C8._ladder(off, on, s, cand['cx'])
    finally:
        C8._FACTORS = old_factors
    return s, off, on, rows


def _trips_old(rows, trip):
    return [r for r in rows if r[3] > 0.0 and (1.0 / r[3]) >= trip]


def _trips_new(rows, trip):
    return [r for r in rows if r[3] <= 1.0 / trip]


def main():
    out = {'python': sys.version.split()[0], 'numpy': np.__version__,
           'lumenairy': la.__file__,
           'fit_order': int(C8.LT._DECENTRED_FIT_POLY_ORDER),
           'TRIP': C8._TRIP, 'REQUIRED_TRIPS': C8._REQUIRED_TRIPS}
    cand = C8._CANDIDATES[0]
    out['cand'] = {k: cand[k] for k in ('alpha', 'cx', 'z', 'frbf')}

    # ---- D5: the pin's own seven rungs, at both samplings -----------------
    for n in (256, 512):
        _s, _off, _on, rows = _rows(cand, n, C8._FACTORS)
        out['pin_rungs_n%d' % n] = [
            {'f': r[0], 'off': r[1], 'on': r[2], 'sup': r[3],
             'ratio': (1.0 / r[3]) if r[3] > 0 else float('inf')}
            for r in rows]
        out['pin_clear_%gx_n%d' % (C8._TRIP, n)] = len(
            _trips_new(rows, C8._TRIP))

    # ---- D4: the wide ladder, where a rung reads exactly zero -------------
    for n in (256,):
        _s, _off, _on, rows = _rows(cand, n, _WIDE)
        zero = [r[0] for r in rows if r[2] == 0.0 and r[1] > 0.0]
        out['wide_n%d' % n] = {
            'n_rungs': len(rows),
            'rungs_with_on_exactly_zero': zero,
            'trips_old_predicate': len(_trips_old(rows, C8._TRIP)),
            'trips_new_predicate': len(_trips_new(rows, C8._TRIP)),
            'rows': [{'f': r[0], 'off': r[1], 'on': r[2], 'sup': r[3]}
                     for r in rows]}

    # ---- D7: an INERT cell -----------------------------------------------
    inert = dict(alpha=3.0, cx=1.40e-3, z=12e-3, frbf=2.0)
    s, off, on, rows = _rows(inert, 256, C8._FACTORS)
    p_off = float((np.abs(off) ** 2).sum())
    p_on = float((np.abs(on) ** 2).sum())
    same_bytes = bool(np.array_equal(off.view(np.uint8), on.view(np.uint8)))
    out['inert'] = {
        'cell': inert,
        'annulus_ratios': [(r[3] if r[3] > 0 else None) for r in rows],
        'all_ratios_exactly_one': all(r[3] == 1.0 for r in rows),
        'fields_bit_identical': same_bytes,
        'power_off': p_off, 'power_on': p_on,
        'power_rel_move': (p_on - p_off) / p_off,
        'max_abs_field_move': float(np.max(np.abs(off - on)))}

    tag = 'win32_314' if sys.platform.startswith('win') else 'linux_312'
    path = os.path.join(_HERE, 'e2_d4d5d7_%s.json' % tag)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items()
                      if not k.startswith('wide_')}, indent=1,
                     sort_keys=True)[:4000])
    print('wide:', {k: v for k, v in out['wide_n256'].items() if k != 'rows'})
    print('wrote', path)


main()
