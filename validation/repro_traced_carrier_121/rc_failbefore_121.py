# D121 RESIDUAL CLOSURE -- the niche-C10 FAIL-BEFORE, on the DESIGN.
#
# In a FRESH process on the POST-change tree, patch
# ``_REMAP_RESID_EIKONAL_DEGREE`` back to 4 and check that design 121's chain
# returns the pre-change intensity array BIT FOR BIT on every order.
#
# The reference shas were recorded by ``rc_readout_121.py`` BEFORE the constant
# was touched; two of them (`8db002a1c1bd58ef` on axis, `5e8550468cb6061b` at
# (-4,-2)) are also `D121_FINAL_CLOSURE_2026_08_02`'s own, so this is anchored
# to a document written before this study existed.
#
# Why this is enough where niche C9 needed a whole ``git archive`` device: C9
# changed a BRANCH reached from six call sites, so a shadow module would have
# been reached by some and not others.  C10 changes one integer that is read
# once, at ``_fit_residual_eikonal``.
#
# usage:  python rc_failbefore_121.py
import hashlib
import os
import sys

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as C                                       # noqa: E402
import fc_instrument_121 as FI                                 # noqa: E402
import hybrid_localize_121 as H                                # noqa: E402
import lumenairy.elements._lens_traced as LT                   # noqa: E402
from approx_common import Patch                                # noqa: E402
from lumenairy.raytrace import RayBundle, trace                # noqa: E402

#: intensity sha256(16) of the chain arm (taper off, exact split, RN=1024,
#: rs=4, CLIP=3.0, dx_out=0.4 um, n_out=61) at ``degree = 4``.
REF = {'0,0': '8db002a1c1bd58ef', '-1,0': 'eef5a64eb2f808a3',
       '-2,0': '41a950e7767eb956', '-3,0': '7db7995c34afbdec',
       '-4,0': '8a589d3d7013ade5', '-4,-2': '5e8550468cb6061b'}


def main():
    print(FI._provenance(), flush=True)
    print(f"module default _REMAP_RESID_EIKONAL_DEGREE = "
          f"{LT._REMAP_RESID_EIKONAL_DEGREE}\n", flush=True)
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=1024, rs=4)
    ok = 0
    print(f"{'order':>8} {'pre-C10 (recorded)':>20} "
          f"{'fail-before deg=4':>20} {'':>8}", flush=True)
    for o, ref in REF.items():
        m, n = (int(v) for v in o.split(','))
        L, M = m * C.LAM / period, n * C.LAM / period
        ch = trace(RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                             L=np.array([L]), M=np.array([M]),
                             N=np.array([np.sqrt(1 - L * L - M * M)]),
                             wavelength=C.LAM, alive=np.ones(1, bool),
                             opd=np.zeros(1)),
                   C.post_surfaces(post), C.LAM,
                   output_filter='last').image_rays
        with Patch([(LT, '_REMAP_RESID_EIKONAL_DEGREE', 4)]):
            res, _w = FI.run_chain(post, env, R, dx, L, M, 4, 'off', 0)
            b = FI.chain_launch(res, L, M, 9999, 3.0, 1, post, 5e-3, 'exact')
        r = H.rs_spot(*b[:7], 5e-3, float(ch.x[0]), float(ch.y[0]),
                      dx_out=0.4e-6, n_out=61, nl=b[7])
        d = hashlib.sha256(np.ascontiguousarray(r['I']).tobytes()
                           ).hexdigest()[:16]
        good = d == ref
        ok += good
        print(f"{o:>8} {ref:>20} {d:>20} {'OK' if good else 'FAILED':>8}",
              flush=True)
    print(f"\nFAIL-BEFORE: {ok} of {len(REF)} bit-identical to the pre-C10 "
          f"tree", flush=True)
    return 0 if ok == len(REF) else 1


if __name__ == '__main__':
    sys.exit(main())
