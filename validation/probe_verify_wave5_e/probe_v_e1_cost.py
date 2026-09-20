"""VERIFY-WAVE5-E / E1(a): what would remedy (a) cost?

Remedy (a) = the FFT dispatchers hand back a PRIVATE COPY.  Emulated by
wrapping ``_fft2``/``_ifft2`` in the consumer modules' own namespaces with a
``.copy()``-returning shim -- the cheapest faithful stand-in for the edit, and
an UPPER bound on it only in the sense that the shim adds one python frame,
which is priced separately below (the ``nocopy_shim`` arm re-wraps WITHOUT the
copy, so the copy's own cost is shim-to-shim).

Arms, best-of-N over warm plans, fresh field each trial:
  * baseline           -- shipped dispatchers
  * nocopy_shim        -- wrapped, no copy (prices the wrapper itself)
  * copy               -- wrapped, ``np.ascontiguousarray(r.copy())``
and, separately, one ``buf.copy()`` against one forward ``_fft2`` at each n.
"""
import json
import os
import platform
import sys
import time

import numpy as np

import lumenairy
from lumenairy.propagators import asm as _asm, fft_infra as _fi, fresnel as _fres

REP = int(os.environ.get('VE1_REP', '7'))
NS = [int(v) for v in os.environ.get('VE1_NS', '512,1024,2048').split(',')]


def _interleaved(fns, rep=REP):
    """Time several arms ROUND-ROBIN so box contention hits them equally."""
    best = {k: float('inf') for k in fns}
    for _ in range(rep):
        for k, fn in fns.items():
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
            if dt < best[k]:
                best[k] = dt
    return best


def _best(fn, rep=REP):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts))


def _field(n):
    rng = np.random.default_rng(12345 + n)
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / 0.2).astype(np.complex128)
    E *= np.exp(1j * 3.0 * rng.standard_normal((n, n)))
    return np.ascontiguousarray(E)


def _install(mode):
    """mode in {'base', 'shim', 'copy'}"""
    f2, i2 = _fi._fft2, _fi._ifft2
    if mode == 'base':
        new2, newi = f2, i2
    elif mode == 'shim':
        def new2(x, _f=f2):
            return _f(x)

        def newi(x, _f=i2):
            return _f(x)
    else:
        def new2(x, _f=f2):
            return _f(x).copy()

        def newi(x, _f=i2):
            return _f(x).copy()
    for mod in (_asm, _fres):
        if hasattr(mod, '_fft2'):
            mod._fft2 = new2
        if hasattr(mod, '_ifft2'):
            mod._ifft2 = newi


def main():
    lam, dx, z = 633e-9, 2e-6, 0.01
    out = dict(
        lumenairy_file=lumenairy.__file__, version=lumenairy.__version__,
        python=sys.version.split()[0], platform=sys.platform,
        machine=platform.machine(), numpy=np.__version__, rep=REP,
        threads={k: os.environ.get(k) for k in
                 ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                  'MKL_NUM_THREADS')},
        double_buffer=_fi.get_fft_double_buffer(), cells=[])
    for n in NS:
        E = _field(n)
        # Fresnel is only valid beyond max(N*dx^2)/lambda -- stay 2x above it
        zf = 2.0 * n * dx * dx / lam
        cell = dict(n=n, z_asm=z, z_fresnel=zf)
        # warm the plans in every mode first
        def _mk(mode, which):
            def run():
                _install(mode)
                if which == 'asm':
                    _asm.angular_spectrum_propagate(E, z, lam, dx)
                elif which == 'asm_nobl':
                    _asm.angular_spectrum_propagate(E, z, lam, dx,
                                                    bandlimit=False)
                else:
                    _fres.fresnel_propagate(E, zf, lam, dx)
            return run

        arms = {'%s_%s' % (w, m): _mk(m, w)
                for w in ('asm', 'asm_nobl', 'fres')
                for m in ('base', 'shim', 'copy')}
        for fn in arms.values():        # warm every plan in every mode
            fn()
            fn()
        cell.update(_interleaved(arms))
        _install('base')
        # one buf.copy() vs one forward transform
        spec = _fi._fft2(E)
        cell['t_fft2'] = _best(lambda: _fi._fft2(E))[0]
        cell['t_copy'] = _best(lambda: spec.copy(), rep=max(REP, 25))[0]
        cell['copy_over_fft'] = cell['t_copy'] / cell['t_fft2']
        cell['spec_owndata'] = bool(spec.flags.owndata)
        for k in ('asm', 'asm_nobl', 'fres'):
            b, c = cell[k + '_base'], cell[k + '_copy']
            s = cell[k + '_shim']
            cell[k + '_pct_vs_base'] = 100.0 * (c - b) / b
            cell[k + '_pct_vs_shim'] = 100.0 * (c - s) / s
        out['cells'].append(cell)
    print(json.dumps(out, indent=1))


main()
