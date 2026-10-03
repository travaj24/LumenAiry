"""V8 -- a MAPPED STACK with two patterned layers on the shared grid and a
lossy layer (the build did not measure multi-patterned mapped stacks).

ladder  [stripe eps 6.25, d 0.25] / [uniform eps 2.1, d 0.15] / [pillar
        eps 3 + 0.5i, d 0.2] under the asymmetric stretch, M = 4..8, normal
        and conical (20, 35 deg): closure sum A == 1 - sum R - sum T
        (layer_absorption, per layer), the unmapped per-layer solve on the
        same physical walls at the same M and at M = 9 (common limit).
vacuum  a PATTERNED all-vacuum layer (eps_cell = 1) of thickness 0.2 put on
        TOP (against the air superstrate) is a no-op for R / T and shifts
        only the zero-order Jones phases: R / T vs the stack without it, and
        r00 * exp(2 i k0 d) / t00 * exp(i k0 d) vs the stack without it.
        It runs its OWN region eig (patterned path), so the agreement is a
        cross-check of the patterned and geometric-eig routes under a map.
refuse  layer_grids='per-layer' + cmap must raise naming Phase E.
"""
import sys

import _vcommon as C
import numpy as np

ARM = sys.argv[1]
CM = C.stretch_map(C.HarmonicStretch(0.10, 0.04, 0.9))
LOSSY = 3.0 + 0.5j


def layers():
    return [C.cell("stripe"), 2.1 + 0j, C.cell("pillar", LOSSY)]


DEPTHS = [0.25, 0.15, 0.2]


def run_ladder():
    res = {}
    for tag, th, ph in (("normal", 0.0, 0.0),
                        ("conical", np.radians(20), np.radians(35))):
        o9, R9, T9, J9, _ = C.ref_perlayer(
            [C.cell("stripe"), np.full((3, 3), 2.1 + 0j),
             C.cell("pillar", LOSSY)], 9, C.XW, C.YW, theta=th, phi=ph,
            depths=DEPTHS)
        rows = []
        for M in range(4, 9):
            o, R, T, J, st = C.stack_solve(CM, layers(), M, theta=th, phi=ph,
                                           retain=True, depths=DEPTHS)
            A = st.layer_absorption()
            target = 1 - R.sum(axis=1) - T.sum(axis=1)
            row = dict(M=M, closure_abs=float(np.max(np.abs(A.sum(axis=0)
                                                            - target))),
                       A_per_layer=np.real(A).tolist(),
                       lossless_layers_A=float(np.abs(A[:2]).max()),
                       dRT_vs_unm9=float(max(np.abs(R - R9).max(),
                                             np.abs(T - T9).max())))
            if M <= 8:
                oa, Ra, Ta, Ja, _ = C.ref_perlayer(
                    [C.cell("stripe"), np.full((3, 3), 2.1 + 0j),
                     C.cell("pillar", LOSSY)], M, C.XW, C.YW, theta=th,
                    phi=ph, depths=DEPTHS)
                row["unm_dRT_vs_unm9"] = float(max(np.abs(Ra - R9).max(),
                                                   np.abs(Ta - T9).max()))
            rows.append(row)
            print(tag, row, flush=True)
        res[tag] = rows
    C.dump("v8_stack_ladder", res)


def run_vacuum():
    rows = []
    d = 0.2
    for M in (5, 7):
        o1, R1, T1, J1, _ = C.stack_solve(CM, [C.cell("stripe")], M)
        o2, R2, T2, J2, _ = C.stack_solve(CM, [C.cell("vac"),
                                               C.cell("stripe")], M,
                                          depths=[d, C.DEPTH])
        i0 = C.i00(o1)
        ph = np.exp(2j * C.K0 * d)
        rows.append(dict(M=M, dRT=float(max(np.abs(R1 - R2).max(),
                                            np.abs(T1 - T2).max())),
                         dJ_phase=float(np.abs(J2 - J1 * ph).max()),
                         dJ_phase_conj=float(np.abs(J2 - J1 / ph).max()),
                         dJ_nophase=float(np.abs(J2 - J1).max()),
                         r00=[complex(J1[0, 0]), complex(J2[0, 0])]))
        print(rows[-1], flush=True)
        del i0
    C.dump("v8_stack_vacuum", {"rows": rows})


def run_refuse():
    from lumenairy.elements.pmm import PMM2DStackPure
    try:
        PMM2DStackPure(C.P, C.P, layer_grids="per-layer", cmap=CM)
        msg = None
    except NotImplementedError as e:
        msg = str(e)
    print(msg)
    C.dump("v8_stack_refuse", {"message": msg,
                               "names_phase_E": bool(msg and "Phase E" in msg)})


{"ladder": run_ladder, "vacuum": run_vacuum, "refuse": run_refuse}[ARM]()
