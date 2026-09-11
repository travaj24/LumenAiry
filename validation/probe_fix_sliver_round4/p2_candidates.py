"""ROUND 4, P2 -- the CANDIDATE DISCRIMINANTS, measured side by side.

Round 4's brief proposes a two-snap continuity test.  This probe does not
assume it: it records, per row, every statistic any of the candidate criteria
needs, so the criterion can be CHOSEN from the numbers instead of asserted.

Every solve is UNGUARDED, so nothing the library decides feeds back.

Solves per row (six):
  A_s   the stack as given -- the sliver solve;
  A_M   the PRESCRIBED grid, ``min_feature = 2 w_wide P`` (walls to midpoint);
  A_L   every flagged wall group CLOSED onto its LEFT wall;
  A_R   the same, onto its RIGHT wall;
  A_s'  the sliver stack at ``degree - 2``;
  A_M'  the prescribed grid at ``degree - 2``.

Statistics:
  d0    |A_s - A_M|                    the shipped ``_sliver_answer_move``
  dLR   |A_L - A_R|                    the device's own answer change for a
                                       wall displacement of one sliver width
  exc   how far A_s sits OUTSIDE the componentwise bracket [A_L, A_R]
  ds_deg|A_s - A_s'|                   degree stability WITH the sliver
  dM_deg|A_M - A_M'|                   degree stability WITHOUT it
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g_fixtures as G  # noqa: E402
import numpy as np  # noqa: E402

from lumenairy.elements.pmm import stack as ps  # noqa: E402


def collapse(st, side):
    """The UNGUARDED solve of this stack with the sliver closed to ``side``."""
    segs = ps._sliver_collapsed_segments(
        [L[1] for L in st._layers],
        float(st.min_feature) / float(st.period), side)
    if segs is None:
        return None
    c = st._min_feature_clone(float(st.min_feature))
    c._layers = [(L[0], segs[i], L[2]) for i, L in enumerate(st._layers)]
    c._src = dict(st._src)
    return G.unguarded(c)


def _shared(a, b):
    c = np.intersect1d(a["o"], b["o"])
    if c.size == 0:
        return None, None, None
    return c, np.searchsorted(a["o"], c), np.searchsorted(b["o"], c)


def excursion(s, lo, hi):
    """How far ``s`` sits OUTSIDE the componentwise bracket the two closed
    geometries span, in the same units as the move (a per-order efficiency).

    A device that is merely SENSITIVE to where the contested wall sits gives
    an answer inside (or barely outside) that bracket, however wide it is; a
    solve the sliver corrupted leaves it entirely."""
    out = 0.0
    for key in ("R", "T"):
        c, i_s, i_l = _shared(s, lo)
        c2, _i_s2, i_h = _shared(s, hi)
        if c is None or c2 is None or c.size != c2.size:
            return None
        A = s[key][:, i_s]
        B = lo[key][:, i_l]
        C = hi[key][:, i_h]
        low = np.minimum(B, C)
        high = np.maximum(B, C)
        out = max(out, float(np.max(np.maximum(A - high, low - A))))
    return max(out, 0.0)


def degree_clone(st, deg):
    c = ps.PMMStack(st.period, n_substrate=st.n_sub, n_superstrate=st.n_sup,
                    degree=int(deg), elements_per_region=st.n_el,
                    grade=st.grade, far_field_orders=st.ffo,
                    factorization=st.factorization,
                    min_feature=float(st.min_feature),
                    layer_grids=st.layer_grids,
                    window_halfwidth=st.window_halfwidth)
    c._layers = list(st._layers)
    if st._src is not None:
        c._src = dict(st._src)
    return c


def measure(st, ref, delta, *, label, degree):
    """One row: every candidate's inputs, or ``screened=False``."""
    base = G.unguarded(st)
    err = G.shared_move(base, ref, pol=1) if ref is not None else None
    rec = dict(label=label, degree=int(degree), delta=float(delta),
               worst=base["worst"], err=err,
               err_over_delta=(err / delta) if (err is not None and delta)
               else None,
               kind=G.classify(err, delta) if (err is not None and delta)
               else "ref")
    pre = G.prescribed(st)
    if pre is None:
        rec["screened"] = False
        return rec
    rec["screened"] = True
    rec.update(w_wide=pre["w_wide"], mf1=pre["mf"], own=pre["own"],
               n_hit=pre["n_hit"])
    M = G.snapped(st, pre["mf"])
    L = collapse(st, "left")
    R = collapse(st, "right")
    if L is None or R is None:
        rec["collapsed"] = False
        return rec
    rec["collapsed"] = True
    rec.update(su1=max(M["worst"] - 1.0, 0.0),
               suL=max(L["worst"] - 1.0, 0.0),
               suR=max(R["worst"] - 1.0, 0.0),
               d0=G.shared_move(base, M), dLR=G.shared_move(L, R),
               d0L=G.shared_move(base, L), d0R=G.shared_move(base, R),
               dML=G.shared_move(M, L), dMR=G.shared_move(M, R),
               exc=excursion(base, L, R),
               err_M=(G.shared_move(M, ref, pol=1) if ref is not None
                      else None),
               err_L=(G.shared_move(L, ref, pol=1) if ref is not None
                      else None))
    dm = max(int(degree) - 2, 2)
    try:
        s2 = G.unguarded(degree_clone(st, dm))
        mc = st._min_feature_clone(pre["mf"])
        mc._src = dict(st._src)
        m2 = G.unguarded(degree_clone(mc, dm))
        rec.update(ds_deg=G.shared_move(base, s2),
                   dM_deg=G.shared_move(M, m2), degree_lo=dm)
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        rec.update(ds_deg=None, dM_deg=None, degree_err=str(exc)[:120])
    return rec
