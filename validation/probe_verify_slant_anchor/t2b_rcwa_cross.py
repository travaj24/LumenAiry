"""TASK 2 (continued) -- V2 against a THIRD engine: RCWA.

The z-staircase oracle of ``t2_v2_sign`` is the SAME engine as the thing under
test, and ``PMM2DStackPure`` is a different spectral-element discretisation of
the same family.  ``RCWAStack`` is neither: a Fourier-modal solver with its own
z-staircase, its own factorisation rules and its own far-field bookkeeping,
and its transmitted amplitudes are lab-referenced by construction.  If the
anchored 1-D answer is right it must agree with RCWA's zeroth-order
transmitted Jones -- which is invariant under a lateral translation of the
whole structure, so the two engines' ``centre`` conventions cannot contaminate
it -- and no other re-referencing may.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

if os.environ.get("LUM_ARM_TREE"):
    sys.path.insert(0, os.environ["LUM_ARM_TREE"])
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

import _lib  # noqa: E402
import numpy as np  # noqa: E402
import t2_v2_sign as t2  # noqa: E402


def main():
    from lumenairy.elements.rcwa import RCWAStack
    t0 = time.time()
    st = t2.sheared_run(t2.ONE)
    t2._solve(st)
    _Ex, _Ey, _al, Jt, a0 = t2._read(st)
    rows, prev = {}, None
    for ns in (8, 16, 32):
        r = RCWAStack(t2.P, n_superstrate=1.0, n_substrate=t2.N_SUB,
                      n_orders=15)
        r.add_tapered_grating(t2.D, eps_ridge=t2.E_R, eps_groove=t2.E_G,
                              duty_bottom=t2.DUTY, duty_top=t2.DUTY,
                              shear=t2.SHEAR, n_slices=ns, n_x=512,
                              raster="area")
        r.set_source(t2.WL, theta=t2.THETA)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = r.solve()
        RJ = np.asarray(res.jones_transmission())
        arms = t2._armsJ(Jt, a0, t2.SHEAR * t2.P, t2.WL)
        rows[str(ns)] = dict(
            {k: _lib.rel(v, RJ) for k, v in arms.items()},
            rcwa_own_step=(None if prev is None else _lib.rel(RJ, prev)))
        prev = RJ
        print(f"  rcwa ns={ns}: "
              f"{ {k: round(v, 6) for k, v in rows[str(ns)].items() if v} }")
    _lib.save("t2b_rcwa_cross",
              dict(rows=rows, jones_1d=[[float(np.real(x)), float(np.imag(x))]
                                        for x in np.asarray(Jt).ravel()],
                   total_secs=round(time.time() - t0, 1)))


if __name__ == "__main__":
    main()
