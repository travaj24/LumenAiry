"""VERIFY-WP-C4 ROUND 2, item 11 -- round 2's neutrality over round 1's suite.

Round 2 adds a SECOND condition to ``_auto_selects_direct``.  Everything round 1
measured has to be untouched by it:

* an explicitly named route (``'bluestein'`` / ``'separable'``) never consults
  the rule at all, so its bytes cannot move;
* ``_MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER`` refuses before either
  condition, so the no-keyword bytes are the pre-C4 dispatch's;
* a SQUARE captured shape has work/entry ``(N + M)/2``, which clears 16
  whenever ``N >= 31``, so every square shape round 1 captured is still
  captured and its bytes must be identical.

24 keys, digested from raw bytes, on a tree given as ``argv[1]``.  Run on a
``git archive bedcabe8`` tree (pre round 2) and on the tip, on both builds, and
compare.

    PYTHONPATH=<tree> python vc4b_neutrality.py <tree> <tag>

Author:  Andrew Traverso
"""
from __future__ import annotations

import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vc4blib as L                                              # noqa: E402

#: 8 shapes: four square CAPTURED, two square refused, two anisotropic.
SHAPES = [(96, 96, 3, 3), (128, 128, 4, 4), (256, 256, 8, 8),
          (64, 64, 2, 2), (24, 24, 12, 12), (64, 64, 8, 8),
          (512, 1024, 16, 32), (512, 1024, 64, 32)]

#: The four square CAPTURED shapes, where the no-keyword call takes the dense
#: route on BOTH trees and therefore must be byte-identical.
SQUARE_CAPTURED = SHAPES[:4]


def main(tree, tag):
    import numpy as np
    la = L.anchor(tree)
    L.single_thread_ffts()
    from lumenairy.propagators import _bluestein as B
    from lumenairy.propagators._bluestein import _bluestein_2d
    from lumenairy.propagators.fft_infra import _fft2, _ifft2

    out = {'build': L.build(), 'tag': tag, 'python': sys.version.split()[0],
           'numpy': np.__version__, 'lumenairy_version': la.__version__,
           'keys': {}}

    def rand(ny, nx):
        rng = np.random.default_rng(20260920)
        return (rng.standard_normal((ny, nx))
                + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)

    def run(E, a, my, mx, **kw):
        return _bluestein_2d(E, a, a, my, mx, sign=-1, xp=np, fft2=_fft2,
                             ifft2=_ifft2, **kw)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # 8 explicit 'bluestein' + 8 explicit 'separable'
        for (ny, nx, my, mx) in SHAPES:
            E = rand(ny, nx)
            a = 1.0e3 / float(max(ny, nx, my, mx)) ** 2
            for spell in ('bluestein', 'separable'):
                k = "%dx%d->%dx%d|%s" % (ny, nx, my, mx, spell)
                out['keys'][k] = L.digest(run(E, a, my, mx, method=spell))
        # 4 no-keyword at square CAPTURED shapes
        for (ny, nx, my, mx) in SQUARE_CAPTURED:
            E = rand(ny, nx)
            a = 1.0e3 / float(max(ny, nx, my, mx)) ** 2
            k = "%dx%d->%dx%d|auto" % (ny, nx, my, mx)
            out['keys'][k] = L.digest(run(E, a, my, mx))
        # 4 no-keyword with the rule switched off process-wide
        saved = B._MFT_DIRECT_MAX_RATIO
        try:
            B._MFT_DIRECT_MAX_RATIO = B._MFT_DIRECT_NEVER
            for (ny, nx, my, mx) in SQUARE_CAPTURED:
                E = rand(ny, nx)
                a = 1.0e3 / float(max(ny, nx, my, mx)) ** 2
                k = "%dx%d->%dx%d|NEVER" % (ny, nx, my, mx)
                out['keys'][k] = L.digest(run(E, a, my, mx))
        finally:
            B._MFT_DIRECT_MAX_RATIO = saved

    out['n_keys'] = len(out['keys'])
    for k in sorted(out['keys']):
        print("  %-32s %s" % (k, out['keys'][k][:16]))
    print("%d keys" % out['n_keys'])
    L.write(out, os.path.join(HERE, "vc4b_neutrality_%s_%s.json"
                              % (tag, L.tag())))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
