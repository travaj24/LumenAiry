"""ROUND 3 -- the fixtures and helpers the D-5 repair is measured on.

Self-contained: nothing here imports from
``validation/probe_verify_sliver_round2/``, so the round-3 numbers are a
measurement of the running library rather than a re-reading of the round-2
verification's JSON.  The GEOMETRY is deliberately the same as the
verification's, because the two populations D-5 turns on are the ones that
verification measured and the repair has to be scored on the same devices.

Three families:

* ``CENSUS`` -- the 648-configuration realistic staircase box (dense
  superstrate, lossy substrate, grazing mount) and the 660-row false-negative
  grid on the O-11 fixture.  These are the two populations the arbiter's
  closure arm has to separate.
* ``FIXTURES`` -- five two-slice staircases whose measured continuity slopes
  span 0.47 .. 31.4, for the DROP-factor population statistic.
* ``gmr`` -- the guided-mode-resonance grating in a dense-superstrate grazing
  mount whose sliver-FREE truncation floor sits BETWEEN the absolute closure
  bar and the trigger.  That is the D-5 class.

Every statistic is computed with the guard DISARMED, so nothing the library
decides can feed back into the measurement, and the verdicts are then scored
two ways: ANALYTICALLY, from the recorded ``(su_snap, move, w_wide, worst)``
under each criterion, and by asking the LIBRARY what it actually did.
"""
import os
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as ps

TRIG = 1.0e-3
NO_SNAP_FRAC = 1e-12


# ------------------------------------------------------------- solving -----
def unguarded(st):
    """The pre-fix code path, bit for bit: ``dict(o, R, T, worst)`` with the
    orders sorted and ``R``/``T`` the full real ``(2, n_orders)`` arrays."""
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = st.solve()
    finally:
        ps.PMM_SLIVER_GUARD = was
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    R = np.real(np.asarray(R))[:, i]
    T = np.real(np.asarray(T))[:, i]
    tot = R.sum(axis=-1) + T.sum(axis=-1)
    return dict(o=o[i], R=R, T=T, worst=float(np.max(tot)))


def guarded(st):
    """``(refused, message, payload_or_None, warning_texts)`` -- what the
    LIBRARY actually does on this stack, end to end."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            o, R, T, _J = st.solve()
        except ValueError as exc:
            return (True, str(exc), None, [str(w.message) for w in rec])
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    return (False, "", dict(o=o[i], R=np.real(np.asarray(R))[:, i],
                            T=np.real(np.asarray(T))[:, i]),
            [str(w.message) for w in rec])


def shared_move(a, b, *, pol=None):
    """Max ``|dR|``, ``|dT|`` over the orders two solves SHARE.  ``pol=None``
    scores both incident polarizations (what the library's arbiter compares);
    ``pol=1`` is the campaign's ``err`` convention."""
    c = np.intersect1d(a["o"], b["o"])
    if c.size == 0:
        return None
    ia = np.searchsorted(a["o"], c)
    ib = np.searchsorted(b["o"], c)
    sl = slice(None) if pol is None else slice(pol, pol + 1)
    return float(max(np.abs(a["R"][sl][:, ia] - b["R"][sl][:, ib]).max(),
                     np.abs(a["T"][sl][:, ia] - b["T"][sl][:, ib]).max()))


def classify(e, d):
    """The campaign's fitted-constant-free continuity rule: an answer that
    tracks the physical wall shift to 10x is RIGHT, past 100x it is WRONG."""
    return "wrong" if e > 100.0 * d else "right" if e <= 10.0 * d else "grey"


def prescribed(st):
    """The ``min_feature`` (metres) this stack's refusal would prescribe and
    the widest manufactured cell it is built from -- read off the library's
    own screen so the probe cannot drift from it."""
    hit = ps._cross_layer_sliver([L[1] for L in st._layers],
                                 float(st.min_feature) / float(st.period))
    if hit is None:
        return None
    return dict(mf=2.0 * hit[3] * float(st.period), w_wide=hit[3],
                w_narrow=hit[0], own=hit[4], n_hit=hit[5])


def snapped(st, mf):
    """The unguarded solve of the same stack on the ``min_feature = mf`` grid
    -- the arbiter's own single measurement, taken here independently."""
    clone = st._min_feature_clone(float(mf))
    clone._src = dict(st._src)
    return unguarded(clone)


# ------------------------------------------------- the two criteria ---------
def drop_factor(worst, su_snap):
    """How much of the super-unity the prescribed snap REMOVES.  ``inf`` when
    the snap removes all of it (the round-2 sliver population's usual read)."""
    v = max(worst - 1.0, 0.0)
    if su_snap <= 0.0:
        return float("inf")
    return v / su_snap


def verdict_round2(worst, su_snap, move, w_wide):
    """Round 2's arbiter, as an expression of the recorded quantities:
    ``su <= 1e-5 AND move > 100 * w_wide``."""
    return ("sliver" if (su_snap <= 1.0e-5 and move > 100.0 * w_wide)
            else "truncation")


def verdict_round3(worst, su_snap, move, w_wide, frac):
    """The candidate round-3 arbiter: the closure made RELATIVE, with the
    round-2 absolute value kept as the lower arm of a ``max``."""
    bar = max(1.0e-5, max(worst - 1.0, 0.0) * frac)
    return ("sliver" if (su_snap <= bar and move > 100.0 * w_wide)
            else "truncation")


# ------------------------------------------- the two-slice staircase --------
def wbuild(d, deg, *, period, wl, th, a0, b0, eh, ep, dz, nl=2, nsub=1.0,
           nsup=1.0, mf=None, ffo=None):
    """``nl`` z-slices whose walls open by ``d`` (a fraction of a period) in
    total across the stack -- the geometry that manufactures the sliver."""
    kw = {} if ffo is None else dict(far_field_orders=ffo)
    st = PMMStack(period, n_superstrate=nsup, n_substrate=nsub, degree=deg,
                  min_feature=(period * NO_SNAP_FRAC if mf is None
                               else float(mf)), **kw)
    for k in range(nl):
        dd = d * k / max(nl - 1, 1)
        st.add_layer(dz, segments=[(a0 - dd, eh), (b0 + dd - (a0 - dd), ep),
                                   (1.0 - (b0 + dd), eh)])
    st.set_source(wl, theta=th)
    return st


#: Five staircases whose measured smooth-regime continuity slopes ``dR/dx``
#: span 0.47 .. 31.4 (measured degree 14, delta 3e-3..1e-4, 2026-09-11).  Only
#: ``O11`` is the fix's own geometry.
FIXTURES = {
    "C_nir": dict(period=1.05e-6, wl=0.98e-6, th=0.42, a0=0.2350, b0=0.6650,
                  eh=2.10, ep=4.00, dz=0.15e-6),                    # 0.47
    "O11": dict(period=1.2e-6, wl=0.85e-6, th=0.15, a0=0.27865, b0=0.62505,
                eh=2.25, ep=9.0, dz=0.32e-6 / 4),                   # 1.15
    "B_vis": dict(period=0.74e-6, wl=0.53e-6, th=0.31, a0=0.19137,
                  b0=0.71429, eh=1.96, ep=6.25, dz=0.06e-6),        # 3.15
    "D_tele": dict(period=1.8e-6, wl=1.31e-6, th=0.11, a0=0.31250,
                   b0=0.58750, eh=2.10, ep=11.7, dz=0.10e-6),       # 10.2
    "S_steep": dict(period=0.74e-6, wl=0.53e-6, th=0.31, a0=0.19137,
                    b0=0.71429, eh=1.96, ep=6.25, dz=0.21e-6 / 3),  # 31.4
}


# ------------------------------------------------------- the census box -----
CP, CWL = 1.2e-6, 0.85e-6
CA0, CB0 = 0.27865, 0.62505
CEH = 2.25
CDZ = 0.32e-6 / 4


def cbuild(d, deg, nsub, nsup, th, nl, eps):
    """One configuration of the realistic staircase box: a dense superstrate,
    a lossy substrate and a grazing mount, whose super-unity is ORDINARY
    under-convergence."""
    st = PMMStack(CP, n_superstrate=nsup, n_substrate=nsub, degree=deg,
                  min_feature=CP * NO_SNAP_FRAC, far_field_orders=31)
    for k in range(nl):
        dd = d * k / max(nl - 1, 1)
        st.add_layer(CDZ, segments=[(CA0 - dd, CEH),
                                    (CB0 + dd - (CA0 - dd), eps),
                                    (1.0 - (CB0 + dd), CEH)])
    st.set_source(CWL, theta=th)
    return st


# --------------------------------------------------------- the D-5 class ----
def gmr(d, deg, *, wl=9.3e-7, nsup=2.4, nsub=complex(1.45, 0.05), th=1.22,
        duty=0.5, period=1.0e-6, e_lo=3.6, e_hi=4.0, t_gr=0.30e-6,
        t_slab=0.10e-6, mf=None, nl=2):
    """The guided-mode-resonance grating in a DENSE-superstrate grazing mount.

    Its sliver-FREE super-unity is ordinary truncation with a clean degree
    ladder, and at degree 8 that floor is 3.73e-05 -- BETWEEN the absolute
    closure bar (1e-5) and the trigger (1e-3).  That is exactly the class on
    which an absolute closure can never be met."""
    a = 0.5 - duty / 2.0
    st = PMMStack(period, n_substrate=nsub, n_superstrate=nsup, degree=deg,
                  far_field_orders=15,
                  min_feature=(period * 1e-12 if mf is None else float(mf)))
    for k in range(nl):
        dd = d * k / max(nl - 1, 1)
        st.add_layer(t_gr / nl, segments=[(a - dd, e_lo),
                                          (duty + 2 * dd, e_hi),
                                          (1.0 - a - duty - dd, e_lo)])
    st.add_layer(t_slab, eps=e_hi)
    st.set_source(wl, theta=th)
    return st
