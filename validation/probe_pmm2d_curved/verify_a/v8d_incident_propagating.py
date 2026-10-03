"""V8d -- DIAGNOSTIC (not a proposed patch): how much of the mapped stripe's
slow convergence is the incident overlap?  The stack's cinc = min-norm
lstsq(Hsup, delta_00) spreads the incident order over evanescent superstrate
modes under a non-polynomial map (v8c).  This arm restricts the overlap to
the PROPAGATING superstrate modes (|Re lam| < 1e-6), by monkeypatching the
two names the stack calls, and compares the TE / TM stripe vs the 1-D oracle
and the n_orders sensitivity of the Jones matrix."""
import _vcommon as C
import numpy as np

from lumenairy.elements.pmm import stack2d_pure as SP

_orig_h = SP._homog_region_modes
_orig_l = SP._guarded_lstsq
STASH = {}


def homog(geom, eps):
    W, V, lam = _orig_h(geom, eps)
    STASH.setdefault("sup_lam", lam)          # the FIRST call is the superstrate
    return W, V, lam


def lstsq(A, b, site, hint=None):
    if "far-field Rayleigh" not in site:
        return _orig_l(A, b, site, hint)
    lam = STASH["sup_lam"]
    prop = np.abs(lam.real) < 1e-6
    c = np.zeros(A.shape[1], complex)
    c[prop] = np.linalg.lstsq(A[:, prop], b, rcond=None)[0]
    return c


def solve(cm, M, n_orders, patched):
    STASH.clear()
    if patched:
        SP._homog_region_modes, SP._guarded_lstsq = homog, lstsq
    try:
        return C.stack_solve(cm, [C.cell("stripe")], M, n_orders=n_orders)
    finally:
        SP._homog_region_modes, SP._guarded_lstsq = _orig_h, _orig_l


rows = []
cm = C.stretch_map(C.HarmonicStretch(0.10, 0.04, 0.9))
for M in (5, 6, 7, 8):
    row = dict(M=M)
    for patched in (False, True):
        o, R, T, J, _ = solve(cm, M, 3, patched)
        o2, R2, T2, J2, _ = solve(cm, M, 2, patched)
        k = "prop" if patched else "shipped"
        row[k + "_te"] = C.stripe_err(o, R, T, 1, C.oracle_1d("te"))
        row[k + "_tm"] = C.stripe_err(o, R, T, 0, C.oracle_1d("tm"))
        row[k + "_dJ_orders"] = float(np.abs(J - J2).max())
        row[k + "_closure"] = C.closure(R, T)
    rows.append(row)
    print(row, flush=True)
C.dump("v8d_incident_propagating", {"rows": rows})
