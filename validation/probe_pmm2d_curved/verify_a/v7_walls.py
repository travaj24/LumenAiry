"""V7 -- F5, where the (u, v) walls go, and the user-facing wall trap.

film   a uniform film under sine 0.12 on three (u, v) grids: the uniform
       3 x 3 lattice, the stripe's PREIMAGE walls (one cell spans the whole
       compression), and a 5 x 5 grid whose u walls SUBDIVIDE the steep
       region (preimages of x = 0.1, 0.2, 0.65, 0.9); M = 4..8 vs Airy (TM,
       incident E_x, the polarisation that sees 1 / f').  Records, per grid,
       the largest stretch RANGE inside one cell (max f' / min f').
trap   the stripe given on the PHYSICAL walls XW but passed to
       SeparableStretch(u_walls=XW, ...) directly (instead of
       from_physical_walls): the solve then models a DIFFERENT device whose
       physical walls are f(XW).  Measured against the 1-D oracle of the
       INTENDED device and of the device actually built (fill f(0.65) -
       f(0.2)).
"""
import sys

import _vcommon as C
import numpy as np

from lumenairy.elements.pmm._curvemap import SeparableStretch

ARM = sys.argv[1]


def cell_ranges(cm, f):
    out = []
    for s in range(cm.shape[0]):
        u = np.linspace(cm.u_bounds[s], cm.u_bounds[s + 1], 2001)
        fp = f(u, C.P)[1]
        out.append(float(fp.max() / fp.min()))
    return out


def run_film():
    f = C.sine(0.12)
    xw5 = np.array([0.0, 0.1, 0.2, 0.65, 0.9, C.P])
    grids = {
        "uniform3": SeparableStretch(3, 3, fx=f, period_x=C.P, period_y=C.P),
        "preimage3": C.stretch_map(f),
        "subdiv5": C.stretch_map(f, x_walls=xw5, y_walls=np.linspace(0, 1, 6)),
        # the walls ADDED inside the wide middle preimage cell (where the
        # compression f' -> 0.25 sits): physical x = 0.35, 0.5 between the
        # stripe walls 0.2 and 0.65
        "subdiv_mid5": C.stretch_map(f, x_walls=np.array(
            [0.0, 0.2, 0.35, 0.5, 0.65, C.P]), y_walls=np.linspace(0, 1, 6)),
    }
    only = sys.argv[2:] or list(grids)
    grids = {k: v for k, v in grids.items() if k in only}
    res = {}
    ex = C.airy()
    for name, cm in grids.items():
        n = cm.shape[0]
        rows = []
        for M in range(4, 9):
            o, R, T, J, _ = C.stack_solve(cm, [C.cell("film", n=n)], M)
            i0 = C.i00(o)
            Rr, Tr = R[0].copy(), T[0].copy()
            Rr[i0] -= ex["p"][0]
            Tr[i0] -= ex["p"][1]
            rows.append(dict(M=M, err_tm=float(max(abs(Rr).max(),
                                                   abs(Tr).max()))))
            print(name, rows[-1], flush=True)
        res[name] = dict(u_bounds=list(cm.u_bounds),
                         stretch_range_per_cell=cell_ranges(cm, f),
                         rows=rows)
    C.dump("v7_walls_film" + ("" if len(grids) > 1 else "_" + next(iter(grids))), res)


def run_trap():
    f = C.sine(0.08)
    wrong = SeparableStretch(C.XW, C.YW, fx=f)
    built = wrong.physical_walls()[0]
    fill_built = float(built[2] - built[1])
    rows = []
    for M in (6, 8):
        o, R, T, J, _ = C.stack_solve(wrong, [C.cell("stripe")], M)
        rows.append(dict(
            M=M, physical_walls_built=list(built),
            err_vs_intended=C.stripe_err(o, R, T, 1, C.oracle_1d("te")),
            err_vs_built=C.stripe_err(o, R, T, 1, C.oracle_1d(
                "te", x_fill=fill_built))))
        print(rows[-1], flush=True)
    C.dump("v7_walls_trap", {"rows": rows})


{"film": run_film, "trap": run_trap}[ARM]()
