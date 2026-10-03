"""V6 -- OBLIQUE and CONICAL incidence under a stretch on PATTERNED cells.

stripe  y-uniform stripe under the asymmetric stretch (harm_asym) and
        sine 0.08: (a) theta 25 deg in the x-z plane vs pmm_efficiency_1d
        at the same angle (TE / TM), M = 4..9; (b) conical (25, 40 deg) and
        (25, 90 deg) [incidence plane ALONG the stripe]: y-MOMENTUM
        conservation -- every order with m_y != 0 must carry zero power on a
        y-uniform cell under an x-only map (measured max R/T over m_y != 0);
        and the complex zero-order Jones vs the UNMAPPED per-layer solve at
        the same physical walls (the pulled-back incident phase).
pillar  pillar under harm_asym, conical (25, 40 deg): mapped ladder vs the
        unmapped per-layer solve at M = 10 (both must approach one limit),
        plus the closure.
"""
import sys

import _vcommon as C
import numpy as np

ARM = sys.argv[1]
TH = np.radians(25.0)


def my_leak(o, R, T):
    mask = o[:, 1] != 0
    return float(max(np.abs(R[:, mask]).max(), np.abs(T[:, mask]).max()))


def run_stripe():
    res = {}
    for name, f in (("harm_asym", C.HarmonicStretch(0.10, 0.04, 0.9)),
                    ("sine0.08", C.sine(0.08))):
        cm = C.stretch_map(f)
        refs = {p: C.oracle_1d(p, theta=TH) for p in ("te", "tm")}
        rows = []
        for M in range(4, 10):
            o, R, T, J, _ = C.stack_solve(cm, [C.cell("stripe")], M,
                                          theta=TH)
            o0, R0, T0, J0, _ = C.ref_perlayer([C.cell("stripe")], M, C.XW,
                                               C.YW, theta=TH)
            row = dict(M=M, obl_map_te=C.stripe_err(o, R, T, 1, refs["te"]),
                       obl_map_tm=C.stripe_err(o, R, T, 0, refs["tm"]),
                       obl_unm_te=C.stripe_err(o0, R0, T0, 1, refs["te"]),
                       obl_unm_tm=C.stripe_err(o0, R0, T0, 0, refs["tm"]),
                       obl_dJ_vs_unm=float(np.abs(J - J0).max()))
            for tag, ph in (("con40", np.radians(40)), ("con90",
                                                        np.radians(90))):
                o, R, T, J, _ = C.stack_solve(cm, [C.cell("stripe")], M,
                                              theta=TH, phi=ph)
                o0, R0, T0, J0, _ = C.ref_perlayer([C.cell("stripe")], M,
                                                   C.XW, C.YW, theta=TH,
                                                   phi=ph)
                row[f"{tag}_ymom_leak_map"] = my_leak(o, R, T)
                row[f"{tag}_ymom_leak_unm"] = my_leak(o0, R0, T0)
                row[f"{tag}_dRT_vs_unm"] = float(max(np.abs(R - R0).max(),
                                                     np.abs(T - T0).max()))
                row[f"{tag}_dJ_vs_unm"] = float(np.abs(J - J0).max())
                row[f"{tag}_closure"] = C.closure(R, T)
            rows.append(row)
            print(name, row, flush=True)
        res[name] = rows
    C.dump("v6_stripe_oblique", res)


def run_pillar():
    cm = C.stretch_map(C.HarmonicStretch(0.10, 0.04, 0.9))
    ph = np.radians(40)
    o0, R0, T0, J0, _ = C.ref_perlayer([C.cell("pillar")], 10, C.XW, C.YW,
                                       theta=TH, phi=ph)
    rows = []
    for M in range(4, 10):
        o, R, T, J, _ = C.stack_solve(cm, [C.cell("pillar")], M, theta=TH,
                                      phi=ph)
        oa, Ra, Ta, Ja, _ = C.ref_perlayer([C.cell("pillar")], M, C.XW, C.YW,
                                           theta=TH, phi=ph)
        row = dict(M=M,
                   map_vs_unm10=float(max(np.abs(R - R0).max(),
                                          np.abs(T - T0).max())),
                   unm_vs_unm10=float(max(np.abs(Ra - R0).max(),
                                          np.abs(Ta - T0).max())),
                   map_dJ_vs_unm10=float(np.abs(J - J0).max()),
                   closure=C.closure(R, T))
        rows.append(row)
        print(row, flush=True)
    C.dump("v6_pillar_conical", {"rows": rows})


{"stripe": run_stripe, "pillar": run_pillar}[ARM]()
