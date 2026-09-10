"""R4 -- does the answer depend on the BLAS REDUCTION ORDER?

One process, one thread setting (read from the environment by the BLAS at
import time, so the sweep is driven from the shell).  Prints one line the
caller's loop collects.  If ``dR`` moves by decades as the thread count moves,
the quantity the test asserts on is not a property of the mathematics: it is a
property of the order in which the sums were accumulated, which is exactly what
an ill-conditioned solve amplifies.
"""
from __future__ import annotations

import json
import os
import sys

import _lib as L
import numpy as np

from lumenairy.elements.rcwa import rcwa_jones_2d


def main():
    import warnings
    a = L.arm()
    tc = L.even_sector_cell()
    kw = dict(n_orders_x=5, n_orders_y=5)
    res = {}
    for tag, sym in (("full", False), ("even", True)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = rcwa_jones_2d(L.P_DEFAULT, L.P_DEFAULT, tc, 1.5, 1.0, 0.2e-6,
                              L.WL_DEFAULT, symmetry=sym, **kw)
            res[tag] = (r, [str(x.message).split(":")[1].split(",")[0].strip()
                            for x in w])
    full, even = res["full"][0], res["even"][0]
    dR = float(np.max(np.abs(full[1] - even[1])))
    dJ = float(np.max(np.abs(full[3] - even[3])))
    out = dict(build=a["build"], tree=a["tree"],
               threads={k: os.environ.get(k) for k in
                        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
               dR=dR, dJ=dJ,
               RT_full=float(np.sum(full[1]) + np.sum(full[2])),
               RT_even=float(np.sum(even[1]) + np.sum(even[2])),
               R_full_head=[float(v) for v in np.asarray(full[1]).ravel()[:4]],
               R_even_head=[float(v) for v in np.asarray(even[1]).ravel()[:4]],
               warn_full=res["full"][1], warn_even=res["even"][1],
               passes=bool(dR < 1e-8 and dJ < 1e-8))
    print("R4JSON " + json.dumps(out))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
