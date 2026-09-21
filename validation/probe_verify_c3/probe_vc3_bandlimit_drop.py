"""VERIFY-WP-C3 -- ``focus_readout['bandlimit']`` is a SZIKLAS-readout-only key
that is neither refused nor route-selecting on ``transport='collins'``: on the
COLLINS readout route it is silently DROPPED.

WP-C3 found this exact shape for ``standoff`` and ``on_focus_containment`` and
made them SELECT the Sziklas route.  ``bandlimit`` is in the SAME position --
it is in the sziklas ``_par_kw`` tuple and absent from the collins one, and no
guard mentions it -- and was left alone.  Before this branch that was reachable
only by opting in to ``transport='collins'``; the flip makes it the DEFAULT.

Fixture that actually TAKES the collins readout route (measured, not assumed):
N = 512 at 8 um, w = 0.30 mm, f = 300 mm singlet, final_distance = 50 mm,
K1 = 0.7698.
"""
import hashlib
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import lumenairy  # noqa: E402
from lumenairy.propagators import carrier as C  # noqa: E402

TREE = os.path.abspath(os.environ.get('VC3_TREE', os.getcwd()))
assert os.path.abspath(lumenairy.__file__).startswith(TREE)
LAM = 1.31e-6


def gauss(n, dx, w):
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


def main():
    from tests.unit.test_audit2609_b4_collins_transport import (
        _singlet, _CHAIN_TKW)
    out = {'lumenairy': lumenairy.__file__, 'python': sys.version.split()[0],
           'numpy': np.__version__, 'rows': []}
    n, dx, w, f = 512, 8e-6, 0.30e-3, 300e-3
    presc = _singlet(2 * f, -2 * f, 3e-3, 'N-BK7', 6e-3, 'p')
    groups = [{'prescription': presc, 'gap_before': 10e-3}]
    env = gauss(n, dx, w)
    fr0 = dict(dx_out=0.5e-6, N_out=64)

    def run(fr, transport, fd):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            r = C.propagate_traced_carrier_chain(
                env, groups, LAM, dx, r_in=np.inf, ray_subsample=16,
                n_workers=1, traced_kwargs=_CHAIN_TKW, final_leg='paraxial',
                final_distance=fd, focus_readout=fr, transport=transport)
        h = hashlib.sha256(np.ascontiguousarray(
            r.field, dtype=np.complex128).tobytes()).hexdigest()
        return r, h, [f'{type(x.message).__name__}: {str(x.message)[:70]}'
                      for x in wl]

    for fd, want in ((50e-3, 'collins'), (8e-3, 'sziklas')):
        base, h0, w0 = run(dict(fr0), 'collins', fd)
        bl, h1, w1 = run(dict(fr0, bandlimit=False), 'collins', fd)
        szk0, s0, _ = run(dict(fr0), 'sziklas', fd)
        szk1, s1, _ = run(dict(fr0, bandlimit=False), 'sziklas', fd)
        out['rows'].append(dict(
            final_distance=fd, expected_route=want,
            route=base.stages[-1].get('readout_route'),
            k1=base.stages[-1].get('readout_route_k1'),
            reason=base.stages[-1].get('readout_route_reason'),
            collins_sha_no_bandlimit=h0, collins_sha_with_bandlimit=h1,
            collins_bandlimit_is_INERT=(h0 == h1),
            sziklas_sha_no_bandlimit=s0, sziklas_sha_with_bandlimit=s1,
            sziklas_bandlimit_is_INERT=(s0 == s1),
            warnings_with_bandlimit_on_collins=w1,
            refused=False))
    # the same for the standoff key, which WP-C3 DID handle -- the contrast
    st, hs, _ = run(dict(fr0, standoff=2e-3), 'collins', 50e-3)
    out['standoff_contrast'] = dict(
        route=st.stages[-1].get('readout_route'),
        reason=st.stages[-1].get('readout_route_reason'),
        k1=st.stages[-1].get('readout_route_k1'))
    tag = os.environ.get('VC3_TAG', 'x')
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f'bandlimit_drop_{tag}.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    print('WROTE', p)


if __name__ == '__main__':
    main()
