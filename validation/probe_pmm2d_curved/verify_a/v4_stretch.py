"""V4 -- A3 / A4 / F1 on the verifier's own stretches.

ladder <name>   TE (incident E_y) and TM (incident E_x) y-uniform stripe under
                the named stretch vs pmm_efficiency_1d (degree 40,
                stabilize=False; self-gap vs degree 48 recorded), M = 4..10,
                with the UNMAPPED per-layer solve on the same physical walls
                alongside.  Names: sine0.02 sine0.08 sine0.12 harm_sym
                (a1 0.08, a2 0.03, phi 0: odd, so mirror-symmetric) harm_asym
                (a1 0.10, a2 0.04, phi 0.9: no mirror symmetry), film_asym
                (uniform film under harm_asym vs Airy).
nq <M>          R / T of the stripe under harm_asym at FORCED node counts vs
                a 1024-node reference: where the quadrature floor sits, and
                what the adaptive rule picks.
cap             near-singular stretches (min f' = 0.05 and 0.01): does the
                256 cap fire, what does the warning say, and what R / T error
                is left at the cap (vs a forced 1024-node rule).
"""
import sys
import warnings

import _vcommon as C
import numpy as np

from lumenairy.elements.pmm import twod_staggered as TS

ARM = sys.argv[1]
STRETCH = {
    "sine0.02": lambda: C.sine(0.02),
    "sine0.08": lambda: C.sine(0.08),
    "sine0.12": lambda: C.sine(0.12),
    "harm_sym": lambda: C.HarmonicStretch(0.08, 0.03, 0.0),
    "harm_asym": lambda: C.HarmonicStretch(0.10, 0.04, 0.9),
}


def min_slope(f):
    u = np.linspace(0, C.P, 200001)
    return float(np.min(f(u, C.P)[1]))


def run_ladder(name):
    M_HI = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    film = name.startswith("film")
    key = name.split("_", 1)[1] if film else name
    key = "harm_asym" if key == "asym" else key
    f = STRETCH[key]()
    cm = C.stretch_map(f)
    res = dict(stretch=name, min_fprime=min_slope(f),
               u_walls=list(cm.u_bounds), rows=[])
    if not film:
        refs = {p: C.oracle_1d(p) for p in ("te", "tm")}
        res["oracle_selfgap_40_48"] = {
            p: max(abs(a - b) for m in refs[p] for a, b in
                   zip(refs[p][m], C.oracle_1d(p, degree=48)[m]))
            for p in refs}
    for M in range(4, M_HI + 1):
        bx = TS.Basis1D(C.P, cm.u_walls, M)
        by = TS.Basis1D(C.P, cm.v_walls, M)
        nq = TS._stag_map_nodes(bx, by, cm, M)
        if film:
            o, R, T, J, _ = C.stack_solve(cm, [C.cell("film")], M)
            ex = C.airy()
            i0 = C.i00(o)
            Rr, Tr = R.copy(), T.copy()
            Rr[:, i0] -= ex["s"][0]
            Tr[:, i0] -= ex["s"][1]
            row = dict(M=M, nq=nq, err_tm=float(max(abs(Rr[0]).max(),
                                                     abs(Tr[0]).max())),
                       err_te=float(max(abs(Rr[1]).max(), abs(Tr[1]).max())),
                       closure=C.closure(R, T))
        else:
            o, R, T, J, _ = C.stack_solve(cm, [C.cell("stripe")], M)
            o0, R0, T0, J0, _ = C.ref_perlayer([C.cell("stripe")], M, C.XW,
                                               C.YW)
            row = dict(M=M, nq=nq,
                       map_te=C.stripe_err(o, R, T, 1, refs["te"]),
                       map_tm=C.stripe_err(o, R, T, 0, refs["tm"]),
                       unm_te=C.stripe_err(o0, R0, T0, 1, refs["te"]),
                       unm_tm=C.stripe_err(o0, R0, T0, 0, refs["tm"]),
                       closure=C.closure(R, T),
                       closure_unm=C.closure(R0, T0))
        res["rows"].append(row)
        print(row, flush=True)
    C.dump(f"v4_ladder_{name}", res)


def forced(nq):
    orig = TS._stag_map_nodes
    TS._stag_map_nodes = (lambda *a, _n=nq, **k: _n)
    return orig


def run_nq():
    M = int(sys.argv[2])
    f = STRETCH["harm_asym"]()
    cm = C.stretch_map(f)
    bx = TS.Basis1D(C.P, cm.u_walls, M)
    by = TS.Basis1D(C.P, cm.v_walls, M)
    nad = TS._stag_map_nodes(bx, by, cm, M)
    res = dict(M=M, adaptive=nad, rows=[])
    out = {}
    base = 2 * M + 8
    for nq in sorted({base, 2 * base, 3 * base, 4 * base, 6 * base,
                      8 * base, nad, 256, 512, 1024}):
        orig = forced(nq)
        try:
            o, R, T, J, _ = C.stack_solve(cm, [C.cell("stripe")], M)
        finally:
            TS._stag_map_nodes = orig
        out[nq] = (R, T, J)
    Rr, Tr, Jr = out[1024]
    for nq in sorted(out):
        R, T, J = out[nq]
        row = dict(nq=nq, adaptive=(nq == nad),
                   dRT=float(max(np.abs(R - Rr).max(), np.abs(T - Tr).max())),
                   dJ=float(np.abs(J - Jr).max()))
        res["rows"].append(row)
        print(row, flush=True)
    C.dump(f"v4_nq_M{M}", res)


def run_cap():
    res = {"rows": []}
    for target in (0.05, 0.01):
        a = (1 - target) / (2 * np.pi)          # min f' = 1 - 2 pi a = target
        f = C.sine(a)
        cm = C.stretch_map(f)
        for M in (5, 6, 8):
            bx = TS.Basis1D(C.P, cm.u_walls, M)
            by = TS.Basis1D(C.P, cm.v_walls, M)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                n = TS._stag_map_nodes(bx, by, cm, M)
            msgs = [str(x.message) for x in w]
            with warnings.catch_warnings(record=True) as w2:
                warnings.simplefilter("always")
                o, R, T, J, _ = C.stack_solve(cm, [C.cell("stripe")], M,
                                              quiet=False)
            solve_msgs = sorted({str(x.message)[:160] for x in w2})
            orig = forced(1024)
            try:
                o2, R2, T2, J2, _ = C.stack_solve(cm, [C.cell("stripe")], M)
            finally:
                TS._stag_map_nodes = orig
            row = dict(min_fprime=target, a_over_p=a, M=M, nq=n,
                       cap_warning=msgs,
                       solve_warnings=solve_msgs,
                       dRT_vs_1024=float(max(np.abs(R - R2).max(),
                                             np.abs(T - T2).max())),
                       err_te_vs_1d=C.stripe_err(o, R, T, 1,
                                                 C.oracle_1d("te")),
                       err_te_1024_vs_1d=C.stripe_err(o2, R2, T2, 1,
                                                      C.oracle_1d("te")))
            res["rows"].append(row)
            print(row, flush=True)
    C.dump("v4_cap", res)


if ARM == "ladder":
    run_ladder(sys.argv[2])
elif ARM == "nq":
    run_nq()
elif ARM == "cap":
    run_cap()
