"""VERIFY-WP-C3 -- arbitrate the moved bare-final-leg answer by GRID
REFINEMENT, which needs no new oracle.

One single-group chain, one physical window (N*dx = 10.24 mm held fixed), one
bare final leg of 10 mm, refined N = 256 / 512 / 1024.  A transport that is
evaluating the leg correctly converges; one whose chirp-Z integrand is aliased
does not.  Run on both trees and read the two ladders side by side.

    python probe_gridladder.py <tree> <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import numpy as np                                            # noqa: E402

import vlib3                                                  # noqa: E402

vlib3.bind(_TREE)

import lumenairy.propagators.carrier as CA                    # noqa: E402

WL = 1.31e-6
TKW = dict(on_undersample='silent', on_noncollimated='silent')
WINDOW = 256 * 40e-6


def sq(n, dx, w):
    x = (np.arange(n) - n / 2.0) * dx
    return np.exp(-((x[:, None] / w) ** 2
                    + (x[None, :] / w) ** 2)).astype(np.complex128)


BIG = {'name': 'big', 'aperture_diameter': 25.4e-3, 'thicknesses': [6e-3],
       'surfaces': [
           {'radius': 120e-3, 'glass_before': 'air', 'glass_after': 'N-BK7',
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
           {'radius': -120e-3, 'glass_before': 'N-BK7', 'glass_after': 'air',
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}

OUT = {}


def one(N, transport=None):
    dx = WINDOW / N
    kw = {} if transport is None else {'transport': transport}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            res = CA.propagate_traced_carrier_chain(
                sq(N, dx, 2.0e-3),
                [{'prescription': BIG, 'gap_before': 20e-3}],
                WL, dx, r_in=np.inf, ray_subsample=16, n_workers=1,
                traced_kwargs=TKW, final_leg='paraxial',
                final_distance=10e-3, **kw)
            f = np.asarray(res.field)
            # the peak and the total power on the returned window, plus the
            # SECOND-MOMENT radius, which a wrapped window inflates
            I = np.abs(f) ** 2
            ax = (np.arange(f.shape[-1]) - f.shape[-1] / 2.0) * float(
                res.dx if not isinstance(res.dx, tuple) else res.dx[0])
            tot = float(I.sum())
            r2 = float((I * (ax[None, :] ** 2 + ax[:, None] ** 2)).sum() / tot)
            rec = {'outcome': 'ok', 'peak': float(I.max()), 'power': tot,
                   'r2m_um': float(np.sqrt(r2) * 1e6),
                   'dx_out_um': float(
                       (res.dx if not isinstance(res.dx, tuple)
                        else res.dx[0]) * 1e6)}
        except BaseException as exc:                           # noqa: BLE001
            rec = {'outcome': 'raised', 'exc': type(exc).__name__,
                   'msg': str(exc)[:200]}
    rec['collins_warn'] = [str(w.message)[:110] for w in caught
                           if 'Collins chirp-Z' in str(w.message)]
    rec['n_warn'] = len(caught)
    return rec


def main():
    for N in (256, 512, 1024):
        OUT['N%d-default' % N] = one(N, None)
        OUT['N%d-sziklas' % N] = one(N, 'sziklas')
        OUT['N%d-collins' % N] = one(N, 'collins')
    OUT['_meta'] = {'tree': _TREE, 'build': vlib3.build_tag(),
                    'window_mm': WINDOW * 1e3}
    with open(sys.argv[2], 'w', encoding='utf-8') as fh:
        json.dump(OUT, fh, indent=1, sort_keys=True, default=str)
    sys.stdout.write('[probe_gridladder] -> %s\n' % sys.argv[2])


if __name__ == '__main__':
    main()
