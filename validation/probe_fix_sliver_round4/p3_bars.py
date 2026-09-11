"""ROUND 4, P3 -- the two bars, derived two-sided on every population.

Measures, with the guard DISARMED, for every row of every population:

  d0   = |A_sliver - A_M|   the shipped move, against the PRESCRIBED
                            ``min_feature = 2 w_wide P`` grid;
  d0L  = |A_sliver - A_L|   the same against the CLOSED grid (remedy 2);
  d12  = |A_L - A_L(+w)|    the device's OWN answer change when that closed
                            wall is displaced by one widest manufactured cell
                            -- the ``dR/dx`` the move bar has always assumed;
  err  against the exact ``delta -> 0`` reference, scored BOTH on the
       campaign's pol-1 convention and on BOTH polarizations, which is the
       statistic the arbiter's move actually uses.

Populations: the round-3 ladder (5 staircases x 3 degrees), the realistic
staircase box (the false-positive population), the D-5 mounts, the V-4
degree-4 mount, the round-2 guided-mode resonance at its own resonance, the
many-slice tapers and the LC directors.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g_fixtures as G  # noqa: E402
import numpy as np  # noqa: E402
import v_fixtures as V  # noqa: E402

from lumenairy.elements.pmm import stack as ps  # noqa: E402

G.assert_tree()


def collapse(st, side, shift=0.0):
    segs = ps._sliver_collapsed_segments(
        [L[1] for L in st._layers],
        float(st.min_feature) / float(st.period), side, shift)
    if segs is None:
        return None
    c = st._min_feature_clone(float(st.min_feature))
    c._layers = [(L[0], segs[i], L[2]) for i, L in enumerate(st._layers)]
    c._src = dict(st._src)
    return G.unguarded(c)


def row(st, ref, delta, label, degree):
    base = G.unguarded(st)
    rec = dict(label=label, degree=int(degree), delta=float(delta),
               worst=base["worst"])
    e1 = G.shared_move(base, ref, pol=1)
    eb = G.shared_move(base, ref)
    rec.update(err_pol1=e1, err_both=eb,
               eod_pol1=e1 / delta, eod_both=eb / delta,
               kind_pol1=G.classify(e1, delta), kind=G.classify(eb, delta))
    pre = G.prescribed(st)
    if pre is None:
        rec["screened"] = False
        return rec
    w = pre["w_wide"]
    rec.update(screened=True, w_wide=w, mf1=pre["mf"], own=pre["own"],
               n_hit=pre["n_hit"])
    M = G.snapped(st, pre["mf"])
    L = collapse(st, "left")
    S = collapse(st, "left", w)
    if L is None or S is None:
        rec["measured"] = False
        return rec
    rec["measured"] = True
    d0 = G.shared_move(base, M)
    d0L = G.shared_move(base, L)
    d12 = G.shared_move(L, S)
    dML = G.shared_move(M, L)
    rec.update(su_M=max(M["worst"] - 1.0, 0.0), d0=d0, d0L=d0L, d12=d12,
               dML=dML, d0_over_w=d0 / w, d0L_over_w=d0L / w,
               d12_over_w=d12 / w,
               d0_over_d12=(d0 / d12) if d12 > 0 else float("inf"),
               d0L_over_d12=(d0L / d12) if d12 > 0 else float("inf"),
               err_M_pol1=G.shared_move(M, ref, pol=1),
               err_M_both=G.shared_move(M, ref),
               err_L_both=G.shared_move(L, ref))
    return rec


def _v4(d, **kw):
    return V.vstair(d, period=1.02e-6, wl=0.633e-6, theta=1.35, a0=0.3120,
                    b0=0.6790, e_lo=2.56, e_hi=12.25, dz=0.12e-6, nl=3,
                    nsup=3.10, nsub=complex(2.90, 1.10), ffo=21, **kw)


def _gmr2(delta, wl=1.749000e-6, degree=16, duty=0.5, nl=2):
    a = 0.5 - duty / 2.0
    st = V.PMMStack(1.0e-6, n_superstrate=1.0, n_substrate=1.45, degree=degree,
                    far_field_orders=15, min_feature=1.0e-6 * V.NO_SNAP)
    for k in range(nl):
        dd = delta * k / max(nl - 1, 1)
        st.add_layer(0.30e-6 / nl,
                     segments=[(a - dd, 3.6), (duty + 2 * dd, 4.0),
                               (1.0 - a - duty - dd, 3.6)])
    st.add_layer(0.10e-6, eps=4.0)
    st.set_source(wl, theta=0.10)
    return st


def _graze(d, **k):
    return V.vstair(d, period=1.18e-6, wl=0.72e-6, theta=1.31, a0=0.2870,
                    b0=0.7150, e_lo=3.24, e_hi=4.41, dz=0.19e-6, nl=2,
                    nsup=2.28, nsub=complex(1.46, 0.06), ffo=15, **k)


D5 = {
    "gmr_deg6": (V.vgmr, dict(degree=6)),
    "gmr_deg8": (V.vgmr, dict(degree=8)),
    "fp_deg8": (V.vfp, dict(degree=8)),
    "wood_deg8": (V.vwood, dict(degree=8)),
    "graze_deg6": (_graze, dict(degree=6)),
    "graze_deg8": (_graze, dict(degree=8)),
}

LAD = [float(x) for x in np.geomspace(3e-3, 1e-6, 18)]
D5D = [float(x) for x in np.geomspace(1e-4, 1e-6, 12)]


def ladder(out):
    for name, cfg in G.FIXTURES.items():
        for deg in (12, 14, 20):
            ref = G.unguarded(G.wbuild(0.0, deg, **cfg))
            for d in LAD:
                out.append(row(G.wbuild(d, deg, **cfg), ref, d,
                               "ladder:" + name, deg))
        print("ladder", name, flush=True)


def census(out):
    for nsub in (complex(1.45, 0.08), complex(2.0, 0.35), complex(3.4, 1.7)):
        for nsup in (2.4, 3.2):
            for th in (1.22, 1.44):
                for deg in (6, 10):
                    for nl in (2, 4):
                        ref = G.unguarded(G.cbuild(0.0, deg, nsub, nsup, th,
                                                   nl, 10.5))
                        for d in (3e-3, 1e-3, 3e-4):
                            out.append(row(G.cbuild(d, deg, nsub, nsup, th,
                                                    nl, 10.5), ref, d,
                                           "census", deg))
        print("census", nsub, flush=True)


def d5(out):
    for name in D5:
        fn, kw = D5[name]
        ref = G.unguarded(fn(0.0, **kw))
        for d in D5D:
            out.append(row(fn(d, **kw), ref, d, "d5:" + name, kw["degree"]))
        print("d5", name, flush=True)


def v4(out):
    ref = G.unguarded(_v4(0.0, degree=4))
    ds = [1.6622e-05, 1.2690e-05, 7.3955e-06]
    ds += [float(x) for x in np.geomspace(3e-5, 3e-6, 8)]
    for d in ds:
        out.append(row(_v4(d, degree=4), ref, d, "v4", 4))
    print("v4", flush=True)


def steep(out):
    for wl in (1.748920e-6, 1.749000e-6, 1.749040e-6):
        for deg in (16, 20):
            ref = G.unguarded(_gmr2(0.0, wl, deg))
            for d in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6):
                out.append(row(_gmr2(d, wl, deg), ref, d, "steep:gmr2", deg))
    print("steep gmr2", flush=True)
    cfg = dict(G.FIXTURES["O11"])
    # The many-slice taper is the round-3 verification own MOVE counter-
    # fixture: ``move`` saturates while ``w_wide`` falls 215x, so a CORRECT
    # row reaches ``move/w_wide`` = 906.56 against a bar of 100.  Degree 10
    # only above 12 slices -- a 24-slice degree-14 stack is four solves of a
    # 25-layer cascade per row and dominates the probe wall time.
    for nl, degs in ((6, (10, 14)), (12, (10, 14)), (24, (10,))):
        for deg in degs:
            ref = G.unguarded(G.wbuild(0.0, deg, nl=nl, **cfg))
            for d in [float(x) for x in np.geomspace(3e-3, 1e-5, 8)]:
                out.append(row(G.wbuild(d, deg, nl=nl, **cfg), ref, d,
                               "steep:taper%d" % nl, deg))
        print("steep taper", nl, flush=True)


def tensor(out):
    for name, fn in (("tensor", V.vtensor), ("oop", V.voop)):
        ref = G.unguarded(fn(0.0))
        for d in [float(x) for x in np.geomspace(3e-4, 1e-6, 12)]:
            out.append(row(fn(d), ref, d, "tensor:" + name, 10))
        print("tensor", name, flush=True)


PARTS = dict(ladder=ladder, census=census, d5=d5, v4=v4, steep=steep,
             tensor=tensor)


def main():
    want = sys.argv[1:] or list(PARTS)
    out = []
    for k in want:
        PARTS[k](out)
    name = "p3_bars" if len(want) == len(PARTS) else "p3_bars_" + "_".join(want)
    G.dump(dict(rows=out, parts=want), name)


if __name__ == "__main__":
    main()
