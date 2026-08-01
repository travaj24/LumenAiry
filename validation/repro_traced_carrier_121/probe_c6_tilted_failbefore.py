"""Niche C6 byte-identity arm (c) -- NOT covered by probe_c6_byte_identity.py,
which checks only the untilted path and the C5 contract.

With ``REMAP_STATIONARY_PHASE_LAUNCH = False``, does a TILTED order reproduce
the COMMITTED library bit for bit?  A fail-before switch that restores only the
untilted path is not a fail-before: every tilted order is exactly what C6 is
meant to change, so that is where the switch has to be proven.
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import _d121_common as C                                       # noqa: E402
import lumenairy.elements as _EL                               # noqa: E402
import lumenairy.elements._lens_traced as _LT                  # noqa: E402
from probe_c6_byte_identity import load_head_module            # noqa: E402

LAM = C.LAM
ORDERS = [(-4, -2), (-4, 0), (-1, 0), (0, 0)]


def run(order, head_mod=None, rn=1024, rs=4):
    """One tilted chain run with C6 DISABLED; ``head_mod`` swaps the element
    for the committed one."""
    prev_flag = _LT.REMAP_STATIONARY_PHASE_LAUNCH
    prev_mod = _EL._lens_traced
    _LT.REMAP_STATIONARY_PHASE_LAUNCH = False
    if head_mod is not None:
        head_mod.REMAP_STATIONARY_PHASE_LAUNCH = False
        _EL._lens_traced = head_mod
    try:
        pre, post, _g, period = C.geometry()
        env, R, dx, _P = C.chain_a(n=rn, rs=rs)
        L, M = order[0] * LAM / period, order[1] * LAM / period
        res = C.la.propagate_traced_carrier_chain(
            env, post, LAM, dx, r_in=C.la.TiltedCarrier(R, L, M),
            ray_subsample=rs, n_workers=8, final_distance=0.0,
            final_leg='paraxial', on_decentred_fit='ignore',
            on_na_proximity='ignore', on_tilt_exact_grid='ignore',
            on_rs_fine_clamp='ignore')
        return np.asarray(res.field)
    finally:
        _EL._lens_traced = prev_mod
        _LT.REMAP_STATIONARY_PHASE_LAUNCH = prev_flag


if __name__ == '__main__':
    head, path = load_head_module()
    print(f"shadow: HEAD:lumenairy/elements/_lens_traced.py -> {path}\n")
    ok = True
    for o in ORDERS:
        a = run(o)                    # current tree, C6 OFF
        b = run(o, head_mod=head)     # committed tree
        eq = bool(np.array_equal(a, b))
        d = (float(np.abs(a - b).max()) if a.shape == b.shape
             else float('nan'))
        ok &= eq
        print(f"  order {str(o):9s}  C6-OFF vs COMMITTED   "
              f"array_equal={eq}  max|dE| {d:.3e}")
    print("\nOK -- the fail-before restores prior behaviour for TILTED orders"
          if ok else
          "\nMISMATCH -- the fail-before does NOT restore tilted behaviour")
