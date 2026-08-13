"""BYTE-IDENTITY of the SPLINE path with the inverse map OFF.

The fit-domain symmetry fix is reachable only when the resolved basis cannot
restrict its own forward fit AND a model is being built.  With the shipped
default (``TRACED_INVERSE_MAP = False``) the second clause is never true, so
the spline path must not move either.  Measured, not argued.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)
sys.path.insert(1, os.path.join(_ROOT, 'scripts'))

import warnings  # noqa: E402

import _d5_byteid as B  # noqa: E402
import numpy as np  # noqa: E402

import lumenairy as la  # noqa: E402

CASES = [
    ('spl_bare', {}),
    ('spl_frbf2', dict(fit_radius_beam_factor=2.0)),
    ('spl_dec', dict(fit_radius_beam_factor=2.0, cx=0.35e-3,
                     beam_centre=(0.35e-3, 0.0))),
    ('spl_rd', dict(fit_radius_beam_factor=2.0,
                    amplitude_model='ray_density')),
]

if __name__ == '__main__':
    out = {}
    for tag, kw in CASES:
        kw = dict(kw)
        cx = kw.pop('cx', 0.0)
        kw.update(newton_fit='spline', on_undersample='silent',
                  on_noncollimated='silent', on_aperture_beam='silent',
                  on_fit_domain_basis='silent', ray_subsample=4, n_workers=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out[tag] = np.asarray(la.apply_real_lens_traced(
                B._field(cx), prescription=B._singlet(3.0e-3),
                wavelength=B._WL, dx=B._DX, **kw))
    np.savez(sys.argv[1], **out)
    print('wrote', sys.argv[1])
