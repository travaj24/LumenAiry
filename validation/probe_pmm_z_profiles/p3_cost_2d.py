"""P3 -- what one route-B slab and one route-A slice cost in the shipped 2-D
solver (wall times, this box, BLAS threads pinned on the command line).

One route-B slab is a curved cell under its own map WITH a tilt: the shipped
solver's closest equivalent is a slanted circle layer (the composite frame of
Phase E1, first-order ``4 q^2`` generator).  One route-A slice is an
in-plane circle layer (``2 q^2`` pencil); a route-A CURVED staircase also
pays one curved mortar per slice interface (Phase E2), measured here on two
concentric circles 0.01 of the period apart, the step a 20-slice cone with a
0.2-period radius change takes.

Each configuration is solved twice in one process; the second (warm) wall
time is reported with the first.  Single-layer stacks between air and
n = 1.45, period 1.2, lambda 1, radius 0.36, height 0.5, normal incidence.

Run:  PYTHONPATH=<worktree> OMP_NUM_THREADS=2 python p3_cost_2d.py
Writes p3_cost_2d.json.
"""
from __future__ import annotations

import os
import sys
import time
import tracemalloc
import warnings

import _zcommon as zc
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PER, WL, R0, H = 1.2, 1.0, 0.36, 0.5


def one(M, *, slant=None, concentric=False):
    from lumenairy.elements.pmm import Circle, PMM2DStackPure
    kw = {}
    if concentric:
        kw["layer_grids"] = "per-layer"
    st = PMM2DStackPure(PER, PER, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, **kw)
    if concentric:
        st.add_layer(0.5 * H, shapes=[Circle(0.6, 0.6, R0, eps=4.0)],
                     background_eps=1.0)
        st.add_layer(0.5 * H, shapes=[Circle(0.6, 0.6, R0 - 0.012, eps=4.0)],
                     background_eps=1.0)
    else:
        lk = {} if slant is None else {"slant": slant}
        st.add_layer(H, shapes=[Circle(0.6, 0.6, R0, eps=4.0)],
                     background_eps=1.0, **lk)
    st.set_source(WL)
    times = []
    out = None
    for _ in range(2):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = st.solve()
        times.append(time.perf_counter() - t0)
    _o, R, T, _J = out
    clos = float(np.max(np.abs(1.0 - np.sum(R, axis=-1) - np.sum(T, axis=-1))))
    return dict(times=times, closure=clos)


def main():
    info = zc.assert_tree()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = dict(info=info)
    for M in (5, 6, 7):
        for name, kw in (("inplane", {}), ("slanted", {"slant": (0.1, 0.0)}),
                         ("concentric_pair", {"concentric": True})):
            if which != "all" and which != name:
                continue
            if name == "concentric_pair" and M > 6:
                continue
            tracemalloc.start()
            r = one(M, **kw)
            _cur, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            r["py_peak_MB"] = peak / 2 ** 20
            res[f"{name}_M{M}"] = r
            print(f"{name:16s} M={M}: cold {r['times'][0]:.2f}s warm "
                  f"{r['times'][1]:.2f}s closure {r['closure']:.1e} "
                  f"py-peak {r['py_peak_MB']:.0f} MB", flush=True)
    zc.dump(os.path.join(HERE, f"p3_cost_2d_{which}.json"), res)


if __name__ == "__main__":
    main()
