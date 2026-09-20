"""VERIFY-WP-C3 CLAIM 12.1 -- the d3 mux/linearity restatement, re-measured.

Drives the SHIPPED test module's own helpers on both transports and reports

  * ``||E6 - E4|| / ||E6||`` with the niche-C6 launch ON  (the test's bar: > 1)
  * ``max|E6 - E4|``          with the launch OFF          (the test's bar: == 0)
  * ``_linearity_error``'s bad6 / good6 / bad4 on both transports

so the report's "reads exactly 0.0 on the flipped default" can be checked, and
so the RESTATED test's own residual and bar can be read back.
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pytest                                                   # noqa: F401

sys.path.insert(0, 'tests/unit')
sys.path.insert(0, 'tests')

import lumenairy as la                                          # noqa: E402
import lumenairy.propagators.carrier as CA                      # noqa: E402
import test_niche_d3_guards as D3                               # noqa: E402
import lumenairy.elements._lens_traced as _lt                   # noqa: E402


def mux_field(tilt, *, degree, launch, transport):
    """``_mux_chain_field`` with the transport made a PARAMETER -- ``None``
    means "pass nothing", i.e. the library default of the running tree."""
    kw = {} if transport is None else {'transport': transport}
    _deg = _lt._REMAP_RESID_EIKONAL_DEGREE
    _lch = _lt.REMAP_STATIONARY_PHASE_LAUNCH
    _lt._REMAP_RESID_EIKONAL_DEGREE = degree
    _lt.REMAP_STATIONARY_PHASE_LAUNCH = launch
    try:
        return D3._chain(D3._mux_fan(tilt), quiet=True, focus_readout=None,
                         on_multi_congruence='ignore', **kw).field
    finally:
        _lt._REMAP_RESID_EIKONAL_DEGREE = _deg
        _lt.REMAP_STATIONARY_PHASE_LAUNCH = _lch


def lin_err(tilt, transport):
    kw = dict(focus_readout=None, on_multi_congruence='ignore')
    if transport is not None:
        kw['transport'] = transport
    X, Y = D3._grid(D3._CN, D3._CDX)
    G = D3._gauss(D3._CN, D3._CDX, D3._CW)
    parts = [G * np.exp(1j * D3._K0 * tilt * (sx * X + sy * Y))
             for sx in (-1, 1) for sy in (-1, 1)]
    ref = None
    shapes = []
    for p in parts:
        f = D3._chain(p, quiet=True, **kw).field
        shapes.append(f.shape)
        ref = f if ref is None else ref + f
    mux = D3._chain(sum(parts), quiet=True, **kw).field
    shapes.append(mux.shape)
    if mux.shape != ref.shape:
        return float('nan'), shapes
    return float(np.linalg.norm(mux - ref) / np.linalg.norm(ref)), shapes


def main(out):
    res = {'tree': CA.__file__,
           'default_transport': _default_transport()}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for tr, tag in ((None, 'LIBRARY-DEFAULT'), ('sziklas', 'sziklas'),
                        ('collins', 'collins')):
            row = {}
            try:
                off6 = mux_field(0.023, degree=6, launch=False, transport=tr)
                off4 = mux_field(0.023, degree=4, launch=False, transport=tr)
                row['launch_off_max_abs'] = float(np.max(np.abs(off6 - off4)))
                row['launch_off_equal'] = bool(np.array_equal(off6, off4))
                row['shape_off'] = list(off6.shape)
                on6 = mux_field(0.023, degree=6, launch=True, transport=tr)
                on4 = mux_field(0.023, degree=4, launch=True, transport=tr)
                row['norm_on6'] = float(np.linalg.norm(on6))
                row['shape_on6'] = list(on6.shape)
                row['shape_on4'] = list(on4.shape)
                if on6.shape == on4.shape:
                    row['moved'] = (float(np.linalg.norm(on6 - on4))
                                    / float(np.linalg.norm(on6)))
                    row['abs_diff'] = float(np.max(np.abs(on6 - on4)))
                else:
                    row['moved'] = None
                row['bar'] = 1.0
                row['passes_bar'] = (row['moved'] is not None
                                     and row['moved'] > 1.0)
            except Exception as exc:                            # noqa: BLE001
                row['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:200])
            try:
                bad6, sh6 = lin_err(0.023, tr)
                good6, _ = lin_err(0.0005, tr)
                _deg = _lt._REMAP_RESID_EIKONAL_DEGREE
                _lt._REMAP_RESID_EIKONAL_DEGREE = 4
                try:
                    bad4, _ = lin_err(0.023, tr)
                finally:
                    _lt._REMAP_RESID_EIKONAL_DEGREE = _deg
                row.update(bad6=bad6, good6=good6, bad4=bad4,
                           lin_shapes=[list(s) for s in sh6],
                           sep=(bad6 / good6 if good6 else None),
                           deg_ratio=(bad4 / bad6 if bad6 else None))
            except Exception as exc:                            # noqa: BLE001
                row['lin_error'] = '%s: %s' % (type(exc).__name__,
                                               str(exc)[:200])
            res[tag] = row
            print('---', tag)
            for k, v in row.items():
                print('   %-22s %s' % (k, v))
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, default=str)
    print('wrote', out)


def _default_transport():
    import inspect
    return str(inspect.signature(la.propagate_traced_carrier_chain)
               .parameters['transport'].default)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'vc3_d3_mux.json')
