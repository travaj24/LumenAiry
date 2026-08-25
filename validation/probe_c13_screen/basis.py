"""ITEM 2 -- is the 1.6e-9 a property of the DATA, the WEIGHTING, or the BASIS?

Reads a captured 28-term traced fit and asks what, if anything, a cheaper
conditioning of the fit would buy over the C13 re-solve.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ['LUMENAIRY_ROOT'])
import lumenairy as la
from lumenairy.elements import _lens_traced as LT

assert os.path.realpath(os.path.dirname(la.__file__)) == \
    os.path.realpath(os.path.join(os.environ['LUMENAIRY_ROOT'], 'lumenairy'))

for tag in sys.argv[1:]:
    d = np.load(tag)
    A = np.ascontiguousarray(d['A'], dtype=np.float64)
    G = A.T @ A
    s = np.linalg.svd(A, compute_uv=False)
    # 1. raw Gram vs diagonally EQUILIBRATED Gram: does column SCALING alone
    #    explain (or fix) the conditioning?
    ev_raw = np.linalg.eigvalsh(G)
    rc_raw = float(ev_raw.min() / ev_raw.max())
    rc_eq = float(LT._gram_rcond(G))
    # 2. column-normalised A (unit 2-norm columns) -- the same scaling, applied
    #    to A rather than to G
    An = A / np.linalg.norm(A, axis=0)[None, :]
    sn = np.linalg.svd(An, compute_uv=False)
    # 3. an ORTHONORMAL basis of the SAME 28-dimensional space (Q from the QR
    #    of A): the best any change of basis can do
    from scipy.linalg import qr
    Q = qr(np.asfortranarray(A), mode='economic')[0]
    sq = np.linalg.svd(Q, compute_uv=False)
    print(f"{os.path.basename(tag):26s} A={A.shape}")
    print(f"   cond(A) shipped basis      {s[0]/s[-1]:.4e}"
          f"   -> cond(G) {(s[0]/s[-1])**2:.4e}")
    print(f"   rcond(G) raw               {rc_raw:.4e}")
    print(f"   rcond(G) equilibrated      {rc_eq:.4e}   (what the screen reads)")
    print(f"   cond(A) columns normalised {sn[0]/sn[-1]:.4e}"
          f"   (scaling alone buys {(s[0]/s[-1])/(sn[0]/sn[-1]):.2f}x)")
    print(f"   cond(A) orthonormal basis  {sq[0]/sq[-1]:.4e}"
          f"   (the floor -- costs one QR of A, i.e. the C13 re-solve itself)")
