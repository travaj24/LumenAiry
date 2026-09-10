"""V10 -- the coincidence the audit does not name: a UNIFORM LAYER, not a region.

The library's warning text, and the audit built on it, describe the failure as
"a LAYER permittivity EXACTLY EQUAL to a REGION's".  But a UNIFORM layer of an
``RCWAStack`` is built by ``_homogeneous_eigenmodes`` in exact arithmetic,
exactly as a half-space region is -- so a uniform spacer at ``eps = 2.25``
sitting next to a STRUCTURED layer whose background is ``2.25`` reproduces the
same singular mode-match, with no region involved and with the substrate index
free.

Measured here across thread counts and both builds, on a three-layer stack
(uniform 2.25 / anisotropic block-in-2.25 / uniform 2.25) at ``n_orders = 4``:
the PRE arm returns ``sum R + T`` far from 2 (or is refused by the library's own
energy tripwire) and the POST arm returns 2 to the arithmetic floor -- INCLUDING
at ``n_substrate = 1.63``, where no permittivity coincides with a REGION at all.

Usage:  OPENBLAS_NUM_THREADS=<n> python v10_stack_spacer.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_fixtures as V  # noqa: E402


def _stack(n_sub, sym, spacer_eps=2.25, n_orders=4):
    from lumenairy.elements.rcwa import RCWAStack
    st = RCWAStack(period=V._P, period_y=V._P, n_superstrate=1.0,
                   n_substrate=n_sub, n_orders=n_orders, n_orders_y=n_orders)
    st.add_layer(0.05e-6, eps=spacer_eps)
    st.add_layer(0.12e-6, eps_tensor_cell=V.uniaxial_cell(S=32))
    st.add_layer(0.06e-6, eps=spacer_eps)
    return st.set_source(V._WL).solve(symmetry=sym)


def observe(n_sub, sym, spacer_eps):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = _stack(n_sub, sym, spacer_eps)
            e = res.efficiencies()
            tot = float(np.sum(np.asarray(e[1])) + np.sum(np.asarray(e[2])))
            return dict(verdict=("WARNED" if caught else "SILENT"),
                        sum_RT=tot, defect=tot - 2.0,
                        warnings=sorted({w.category.__name__ for w in caught}))
        except Exception as exc:
            return dict(verdict="RAISED", error=type(exc).__name__,
                        sum_RT=None, defect=None, warnings=[])


CASES = [
    # (n_substrate, symmetry, spacer eps, does anything coincide with a REGION?)
    (1.5, "auto", 2.25, True),
    (1.5, True, 2.25, True),
    (1.5, False, 2.25, True),
    (1.63, "auto", 2.25, False),
    (1.63, True, 2.25, False),
    (1.63, False, 2.25, False),
    # control: move the SPACER off the structured layer's background, so no
    # layer-layer coincidence remains either
    (1.63, "auto", 2.56, False),
    (1.63, True, 2.56, False),
    (1.63, False, 2.56, False),
]


def main():
    V.require_local_tree()
    out = sys.argv[1]
    rows = []
    for n_sub, sym, sp, reg in CASES:
        post = observe(n_sub, sym, sp)
        with V.PreSqrtDecay():
            pre = observe(n_sub, sym, sp)
        rows.append(dict(n_substrate=n_sub, symmetry=str(sym), spacer_eps=sp,
                         coincides_with_a_region=reg, post=post, pre=pre))
        print("n_sub=%-5s sym=%-5s spacer=%-5s regionCoinc=%-5s  "
              "PRE %-6s %s   POST %-6s %s" % (
                  n_sub, sym, sp, reg, pre["verdict"],
                  ("sumR+T=%.9f" % pre["sum_RT"]) if pre["sum_RT"] is not None
                  else "(refused)",
                  post["verdict"],
                  ("sumR+T=%.15f" % post["sum_RT"])
                  if post["sum_RT"] is not None else "(refused)"))
    V.dump(out, dict(rows=rows,
                     openblas_num_threads=os.environ.get(
                         "OPENBLAS_NUM_THREADS", "unpinned")))


if __name__ == "__main__":
    main()
