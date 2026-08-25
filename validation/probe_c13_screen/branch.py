"""Which BRANCH of _solve_lstsq_thread_safe does each captured fit take on the
SHIPPED defaults, and does the C13 residual vote ever run?"""
import glob
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.environ['LUMENAIRY_ROOT'])
import lumenairy as la
from lumenairy.elements import _lens_traced as LT

assert os.path.realpath(os.path.dirname(la.__file__)) == \
    os.path.realpath(os.path.join(os.environ['LUMENAIRY_ROOT'], 'lumenairy'))

state = {}
_refine, _qr, _resid = LT._det_refine, LT._solve_lstsq_qr, LT._lstsq_residual


def s_refine(A, b, x, small):
    out = _refine(A, b, x, small)
    state['refine'] = 'REFUSED' if out is None else 'APPLIED'
    return out


def s_qr(A, b):
    state['qr'] = True
    return _qr(A, b)


def s_resid(A, b, x):
    state['vote'] = state.get('vote', 0) + 1
    return _resid(A, b, x)


LT._det_refine, LT._solve_lstsq_qr, LT._lstsq_residual = s_refine, s_qr, s_resid
print(f"defaults: STEPDOWN={LT.LSTSQ_CONDITIONING_STEPDOWN} "
      f"DET_TRACED={LT.DETERMINISTIC_TRACED_FIT} "
      f"DET_NE={LT.DETERMINISTIC_NORMAL_EQUATIONS} "
      f"screen={LT._LSTSQ_GRAM_RCOND_MIN:.0e} margin={LT._LSTSQ_RESID_MARGIN:.0e}")
for p in sorted(glob.glob(os.path.join(sys.argv[1], '*.npz'))):
    d = np.load(p)
    A, b, det = d['A'], d['b'], bool(d['det'][0])
    for flag in (True, False):
        state.clear()
        LT.DETERMINISTIC_TRACED_FIT = flag
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            LT._solve_lstsq_thread_safe(A, b, deterministic=(det and flag) or
                                        (det and A.shape[1] < 8))
        print(f"  {os.path.basename(p):26s} M={A.shape[1]:3d} "
              f"DET_TRACED={flag!s:5s} refine={state.get('refine','-'):8s} "
              f"qr={state.get('qr',False)!s:5s} "
              f"residual_votes={state.get('vote',0)}")
    LT.DETERMINISTIC_TRACED_FIT = True
