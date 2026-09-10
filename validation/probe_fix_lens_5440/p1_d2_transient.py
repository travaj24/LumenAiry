"""P1 (D2) -- the PER-CALL transient of ``carrier_referenced_envelope`` /
``carrier_referenced_reconstruct`` on a complex64 field.

VERIFY_LENS_BANDED_COMPLEX64_2026_09_10 D2: ``_build_carrier_phase``
(carrier.py:1663) takes no ``dtype=``, so a complex64 chain still materialises
a FULL-GRID complex128 phasor per call (5 per two-group chain; 4.29 GB each at
N=16384).  This measures the whole-call ``tracemalloc`` peak of ONE public
helper call, at N=2048 and N=4096, on both the scalar (radial) and the
ASTIGMATIC ``(R_x, R_y)`` carrier branch, and hashes the returned field so the
fix can be shown to be value-preserving on BOTH dtypes.

Usage:  python p1_d2_transient.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixp  # noqa: E402

WL = _fixp.WL
NS = (2048, 4096)


def _field(N, dx, dtype):
    x = (np.arange(N) - N // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    w = 0.28 * N * dx
    return np.exp(-r2 / w ** 2).astype(dtype)


def main():
    la = _fixp.banner()
    from lumenairy.propagators import carrier as C
    print('# free RAM %.1f GB' % _fixp.free_gb(), flush=True)
    out = {'version': la.__version__, 'file': la.__file__, 'cases': {}}
    for N in NS:
        dx = 13.0e-6 * (1024.0 / N)
        grid64 = 8.0 * N * N                       # one complex64 grid
        for carrier_tag, R in (('sph', 55e-3), ('astig', (55e-3, -70e-3))):
            for fn_tag, fn in (('envelope', C.carrier_referenced_envelope),
                               ('reconstruct',
                                C.carrier_referenced_reconstruct)):
                for dt_tag, dt in (('c128', np.complex128),
                                   ('c64', np.complex64)):
                    E = _field(N, dx, dt)
                    fn(E, R, WL, dx)                       # warm
                    tracemalloc.start()
                    tracemalloc.reset_peak()
                    r = fn(E, R, WL, dx)
                    _, pk = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    key = '%d/%s/%s/%s' % (N, carrier_tag, fn_tag, dt_tag)
                    out['cases'][key] = {
                        'N': N, 'dx': dx, 'peak_bytes': int(pk),
                        'peak_MiB': pk / 2 ** 20,
                        'peak_c64_grids': pk / grid64,
                        'out_dtype': str(np.asarray(r).dtype),
                        'hash': _fixp.h(r)}
                    print('%-28s peak %9.2f MiB = %5.2f c64-grids  out %s  %s'
                          % (key, pk / 2 ** 20, pk / grid64,
                             np.asarray(r).dtype, _fixp.h(r)), flush=True)
                    del E, r
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
