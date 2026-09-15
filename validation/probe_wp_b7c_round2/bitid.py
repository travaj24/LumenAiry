"""Bit identity: SHA-256 over ``ndarray.tobytes()`` of the branch sum and the
completion at a matrix of fixtures, run in a CHILD process whose ``cwd`` and
``PYTHONPATH`` are set to ONE tree and whose ``lumenairy.__file__`` is
ASSERTED to live under it before any field is built.

Usage::

    python bitid.py <tree> <out.json>
"""
# ruff: noqa: E402, I001
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'probe_verify_b7c'))
import fixtures as FX                                     # noqa: E402

import lumenairy                                          # noqa: E402
from lumenairy.elements._lens_traced_multibranch import (  # noqa: E402
    apply_real_lens_traced_multibranch)
from lumenairy.elements._lens_traced_uniform import (      # noqa: E402
    apply_real_lens_traced_uniform)


def digest(E):
    return hashlib.sha256(np.asarray(E).tobytes()).hexdigest()


# (tag, fixture, z_um, member, kwargs)
MATRIX = [
    ('S_mb_healthy', 'S', 3214.78, 'mb', {}),
    ('S_uni_healthy', 'S', 3214.78, 'uni', {}),
    ('S_mb_blow', 'S', 3274.02, 'mb', {}),
    ('S_uni_blow', 'S', 3274.02, 'uni', {}),
    ('S_uni_vertex', 'S', 0.0, 'uni', {}),
    ('S_mb_vertex', 'S', 0.0, 'mb', {}),
    ('S_uni_plain', 'S', 3214.78, 'uni', {'caustic_band': 'plain'}),
    ('S_uni_sub4', 'S', 3214.78, 'uni', {'ray_subsample': 4}),
    ('S_alt_uni', 'S_alt', 3214.78, 'uni', {}),
    ('M_uni_healthy', 'M', 2201.74, 'uni', {}),
    ('M_mb_healthy', 'M', 2201.74, 'mb', {}),
    ('M_uni_fallback', 'M', 2330.0, 'uni', {}),
    ('M_uni_blow', 'M', 2343.0, 'uni', {}),
    ('M_alt_uni', 'M_alt', 2201.74, 'uni', {}),
    ('Q_uni_healthy', 'Q', 5680.0, 'uni', {}),
    ('Q_uni_gap', 'Q', 5400.0, 'uni', {}),
    ('Q_uni_gap2', 'Q', 5412.0, 'uni', {}),
    ('Q_uni_gap3', 'Q', 5420.0, 'uni', {}),
    ('Q_uni_refused', 'Q', 5460.0, 'uni', {}),
    ('Q_mb_refused', 'Q', 5460.0, 'mb', {}),
    ('Q_alt_uni', 'Q_alt', 5680.0, 'uni', {}),
    ('F_uni_healthy', 'F_alt', 1056.0, 'uni', {}),
    ('F_uni_d1', 'F_alt', 1076.0, 'uni', {}),
    ('F_uni_refused', 'F_alt', 1080.0, 'uni', {}),
    ('F_mb_refused', 'F_alt', 1080.0, 'mb', {}),
    ('F_uni_c64', 'F_alt', 1056.0, 'uni', {'_c64': True}),
    ('P_uni_healthy', 'P', 888.0, 'uni', {}),
    ('P_uni_refused', 'P', 1015.34, 'uni', {}),
    ('P_uni_blow', 'P', 1016.0, 'uni', {}),
    ('P_alt_uni', 'P_alt', 888.0, 'uni', {}),
]


def run_one(tag, name, zum, member, kw):
    fx = FX.FIXTURES[name]
    kw = dict(kw)
    c64 = kw.pop('_c64', False)
    E = FX.gauss(fx['N'], fx['dx'], fx['w0'],
                 dtype=np.complex64 if c64 else np.complex128)
    fn = (apply_real_lens_traced_multibranch if member == 'mb'
          else apply_real_lens_traced_uniform)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            out = fn(E, prescription=fx['prescription'],
                     wavelength=fx['wavelength'], dx=fx['dx'],
                     output_plane_distance=zum * 1e-6,
                     return_diagnostics=True, **kw)
        E_out = out[0]
        d = out[1]
        return {'tag': tag, 'ok': True, 'sha256': digest(E_out),
                'power': float(np.sum(np.abs(np.asarray(E_out)) ** 2))
                * fx['dx'] ** 2,
                'dtype': str(np.asarray(E_out).dtype),
                'n_warnings': len(rec),
                'power_ratio': (float(d['power_ratio'])
                                if d.get('power_ratio') is not None else None),
                'decision': d.get('power_ratio_decision'),
                'continuity': d.get('pixel_continuity'),
                'continuity_mb': d.get('multibranch_pixel_continuity'),
                'continuity_of': d.get('pixel_continuity_of'),
                'continuity_decision': d.get('pixel_continuity_decision')}
    except RuntimeError as e:
        return {'tag': tag, 'ok': False, 'error': str(e)[:300]}


def main(argv):
    tree = os.path.abspath(argv[1])
    assert os.path.abspath(lumenairy.__file__).startswith(tree), (
        f'lumenairy.__file__ = {lumenairy.__file__} is not under {tree}')
    res = {'tree': tree, 'lumenairy_file': lumenairy.__file__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'rows': []}
    for row in MATRIX:
        r = run_one(*row)
        res['rows'].append(r)
        print(f"{r['tag']:18s} {'OK ' if r['ok'] else 'ERR'} "
              f"{r.get('sha256', r.get('error', ''))[:24]} "
              f"cont={r.get('continuity')!s:>10.10} "
              f"{r.get('continuity_decision') or ''}", flush=True)
        with open(argv[2], 'w') as fh:
            json.dump(res, fh, indent=1, default=str)


if __name__ == '__main__':
    main(sys.argv)
