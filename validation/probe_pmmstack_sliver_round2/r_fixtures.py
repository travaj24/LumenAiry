"""ROUND 2 -- the shared fixtures for the arbiter probes.

The O-11 stack (two slices of the F5 taper, the second's walls opened by
``delta``), its exact ``delta -> 0`` reference, and the unguarded solve every
probe scores against.  Nothing here touches the library's guard except through
its own fail-before switch ``PMM_SLIVER_GUARD``.
"""
import os
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as ps

P, WL, THETA = 1.2e-6, 0.85e-6, 0.15
EH, EP = 2.25, 9.0
A0, B0 = 0.27865, 0.62505
DZ = 0.32e-6 / 4
NO_SNAP = P * 1e-12               # a min_feature far below every wall spacing


def segs(a, b, eh=EH, ep=EP):
    return [(a, eh), (b - a, ep), (1.0 - b, eh)]


def build(d, deg, *, mf=NO_SNAP, nsub=1.0, nsup=1.0, th=THETA, nl=2,
          eps=EP, period=P, wl=WL, ffo=None):
    """A staircase of ``nl`` slices whose walls open by ``d`` in total."""
    kw = {} if ffo is None else dict(far_field_orders=ffo)
    st = PMMStack(period, n_superstrate=nsup, n_substrate=nsub, degree=deg,
                  min_feature=mf, **kw)
    for k in range(nl):
        dd = d * k / max(nl - 1, 1)
        st.add_layer(DZ, segments=segs(A0 - dd, B0 + dd, ep=eps))
    st.set_source(wl, theta=th)
    return st


def raw(st):
    """Unguarded solve -> (orders, R1, T1, worst).  The guard is OFF, so this
    is the pre-fix code path bit for bit."""
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, _J = st.solve()
    finally:
        ps.PMM_SLIVER_GUARD = was
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
    return (o[i], np.asarray(R)[1][i], np.asarray(T)[1][i], float(np.max(tot)),
            np.asarray(R)[:, i], np.asarray(T)[:, i])


def err(a, b):
    """Max |dR|, |dT| over the orders the two solves share."""
    c = np.intersect1d(a[0], b[0])
    ia, ib = np.searchsorted(a[0], c), np.searchsorted(b[0], c)
    return float(max(np.abs(a[1][ia] - b[1][ib]).max(),
                     np.abs(a[2][ia] - b[2][ib]).max()))


def err0(a, b):
    """The ZEROTH-order move only -- the shape-independent statistic the
    library falls back to when the snapped grid changes the order count."""
    ia = int(np.argmin(np.abs(a[0])))
    ib = int(np.argmin(np.abs(b[0])))
    return float(max(abs(a[1][ia] - b[1][ib]), abs(a[2][ia] - b[2][ib])))


def move_both(a, b):
    """The move the LIBRARY computes: max |dR|, |dT| over BOTH incident
    polarizations, on the orders the two solves share.  The campaign's `err`
    column scores polarization 1 only (the fix's own convention, and the
    verification's S4.1 definitional correction), so this is measured
    separately wherever a bar is read off it."""
    c = np.intersect1d(a[0], b[0])
    ia, ib = np.searchsorted(a[0], c), np.searchsorted(b[0], c)
    return float(max(np.abs(a[4][:, ia] - b[4][:, ib]).max(),
                     np.abs(a[5][:, ia] - b[5][:, ib]).max()))


def kind_of(e, d):
    """The campaign's fitted-constant-free continuity rule."""
    return "wrong" if e > 100.0 * d else "right" if e <= 10.0 * d else "grey"


def prescribed_mf(st):
    """The ``min_feature`` the refusal prescribes for this stack, read off the
    screen itself (``2 * the WIDEST manufactured cell``)."""
    hit = ps._cross_layer_sliver([L[1] for L in st._layers],
                                 float(st.min_feature) / float(st.period))
    return None if hit is None else 2.0 * hit[3] * float(st.period)
