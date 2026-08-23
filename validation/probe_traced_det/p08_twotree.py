"""Two-tree byte comparison of the D14 kernel at the CARRIER fit's shapes.

The shipped tree (5.41.0) and this branch must return IDENTICAL bytes from
_det_normal_equations for every term count BELOW _DET_EINSUM_MIN_TERMS, or
niche D14's carrier bits -- and the analytic path's field -- have moved.
Run once per tree; the driver diffs the two outputs.
"""
import hashlib
import os

import numpy as np

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_want = os.path.realpath(os.path.join(os.environ['LUMENAIRY_ROOT'], 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__


def _h(a):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


rng = np.random.default_rng(20260823)
for n, M, K in ((119936, 5, 1), (67348, 5, 1), (30000, 6, 1), (5000, 7, 2),
                (1800000, 5, 1), (4000, 4, 3), (999, 2, 1)):
    A = np.ascontiguousarray(rng.normal(size=(n, M)) * 1e-2)
    B = np.ascontiguousarray(rng.normal(size=(n, K)))
    if K == 1:
        B = np.ascontiguousarray(B[:, 0])
    G, r = LT._det_normal_equations(A, B)
    print('n=%-9d M=%-3d K=%d  G=%s rhs=%s' % (n, M, K, _h(G), _h(r)))
