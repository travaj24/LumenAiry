"""R5 -- is the coincidence ``eps_layer_background == eps_substrate`` the cause?

The fixture's layer background is ``2.25`` and its substrate index is ``1.5``,
so ``eps_sub = 1.5^2 = 2.25`` EXACTLY; the block's ``zz`` component is the same
``no^2 = 2.25`` again.  ``_check_energy``'s own text names that coincidence as
the library's exactly-degenerate layer<->region mode-match class.  This probe
walks a RELATIVE detune of the substrate index away from the coincidence and
records, at each detune, the lossless-closure defect of each path and the
full-vs-even disagreement the failing test asserts on.

A mechanism is established only if the defect and the disagreement COLLAPSE as
the detune leaves the coincidence and RETURN as it approaches it -- one number
at one detune proves nothing.
"""
from __future__ import annotations

import json
import warnings

import _lib as L
import numpy as np

from lumenairy.elements.rcwa import rcwa_jones_2d

DETUNES = (0.0, 1e-13, 1e-11, 1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def solve(tc, n_sub, sym):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rcwa_jones_2d(L.P_DEFAULT, L.P_DEFAULT, tc, n_sub, 1.0, 0.2e-6,
                             L.WL_DEFAULT, n_orders_x=5, n_orders_y=5,
                             symmetry=sym)


def main():
    a = L.arm()
    tc = L.even_sector_cell()
    rows = []
    for d in DETUNES:
        n_sub = 1.5 * (1.0 + d)
        full = solve(tc, n_sub, False)
        even = solve(tc, n_sub, True)
        row = dict(detune=d, n_sub=n_sub,
                   dR=float(np.max(np.abs(full[1] - even[1]))),
                   dT=float(np.max(np.abs(full[2] - even[2]))),
                   dJ=float(np.max(np.abs(full[3] - even[3]))),
                   defect_full=float(np.sum(full[1]) + np.sum(full[2]) - 2.0),
                   defect_even=float(np.sum(even[1]) + np.sum(even[2]) - 2.0))
        rows.append(row)
        print("detune %-8.0e  defect_full %+.3e  defect_even %+.3e  "
              "dR %.3e  dT %.3e  dJ %.3e"
              % (d, row["defect_full"], row["defect_even"], row["dR"],
                 row["dT"], row["dJ"]), flush=True)
    print("R5JSON " + json.dumps(dict(build=a["build"], rows=rows)))
    L.dump("r5_detune", dict(rows=rows))


if __name__ == "__main__":
    main()
