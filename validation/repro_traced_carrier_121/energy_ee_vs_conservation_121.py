# Conservation audit (2026-07-31): the TRADE, priced in both currencies.
#
# The campaign scored niche C6 in EE3 points.  ``energy_stage_audit_121.py``
# scores the same flag in conservation.  This script runs the COMPLETE shipped
# path -- the one the EE numbers came from, ``approx_common.run_chain`` with
# the exact Bluestein readout on a fixed lattice centred on the order's chief
# ray -- and prints EE3/EE6/EE12 and the last group's energy budget from the
# SAME run, so the two currencies are quoted for one field and not two.
#
# A NULL row (two identical baseline runs) establishes the differential floor
# before any delta is quoted.
#
# usage:  LUMEN_PIN=0 ORDERS='0,0 -4,-2' python energy_ee_vs_conservation_121.py
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('LUMEN_PIN', '0')       # measure the LIVE working tree

import approx_common as A                                         # noqa: E402
import probe_ghost_c6 as G6                                       # noqa: E402
import wfe_probe_common as P                                      # noqa: E402
import _d121_common as C                                          # noqa: E402
import lumenairy.elements._lens_traced as LT                      # noqa: E402

CONFIGS = {
    'ship': {'c5': True,  'c6': True},
    'noC6': {'c5': True,  'c6': False},
    'noC5': {'c5': False, 'c6': True},
    # 2026-07-31 fit-guard decision study: the SAME production path with
    # REMAP_STATIONARY_PHASE_FIT_GUARD on.  This is the EE half of the trade
    # the guard's chain-level cost was never measured on.
    'shipG': {'c5': True,  'c6': True,  'guard': True},
    'noC6G': {'c5': True,  'c6': False, 'guard': True},
}


class _cap(object):
    """Record the LAST element call only (E_in, E_out, dx, carrier)."""

    def __init__(self):
        self.last = None
        self._orig = None

    def __enter__(self):
        import lumenairy.elements as _el
        self._orig = _el.apply_real_lens_traced

        def _w(E_in, *a, **kw):
            out = self._orig(E_in, *a, **kw)
            self.last = {'E_in': np.array(E_in, copy=True),
                         'E_out': np.array(out, copy=True), 'kw': dict(kw)}
            return out

        _el.apply_real_lens_traced = _w
        return self

    def __exit__(self, *e):
        import lumenairy.elements as _el
        _el.apply_real_lens_traced = self._orig
        return False


def one(order, cfg, post, env, R, dx, P_in, L, M, centre):
    cf = CONFIGS[cfg]
    old = (LT.TILTED_CARRIER_EXACT_EIKONAL, LT.REMAP_STATIONARY_PHASE_LAUNCH,
           LT.REMAP_STATIONARY_PHASE_FIT_GUARD)
    LT.TILTED_CARRIER_EXACT_EIKONAL = bool(cf['c5'])
    LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(cf['c6'])
    LT.REMAP_STATIONARY_PHASE_FIT_GUARD = bool(cf.get('guard', False))
    t0 = time.time()
    try:
        with _cap() as cap:
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter('always')
                E, _stg, _s = A.run_chain(post, env, R, dx, L, M, centre)
    finally:
        (LT.TILTED_CARRIER_EXACT_EIKONAL,
         LT.REMAP_STATIONARY_PHASE_LAUNCH,
         LT.REMAP_STATIONARY_PHASE_FIT_GUARD) = old
    eng = sum(1 for w in wl if 'energy self-check FAILED' in str(w.message))
    hal = sum(1 for w in wl if 'HALO self-check FAILED' in str(w.message))
    return np.asarray(E), cap.last, (eng, hal), time.time() - t0


def main():
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get('ORDERS', '0,0 -4,-2').split()]
    cfgs = [c.strip() for c in
            os.environ.get('CONFIGS', 'ship,noC6').split(',') if c.strip()]
    _pre, post_all, _g, _p = C.geometry()
    surfs_last = P.element_surfaces(post_all[-1]['prescription'])
    print(f"   lib {LT.__file__}")
    print("   COMPLETE shipped path (exact Bluestein readout), fixed lattice "
          "on the order's chief ray.")
    print("   EE* are normalised to the CHAIN INPUT power, so a variant that "
          "loses or gains energy is visible in them.")
    print()
    for order in orders:
        post, env, R, dx, P_in, L, M, centre = A.setup(order)
        # NULL: two identical baseline runs, before any delta.  NULL=0 skips
        # it where the floor for THIS instrument is already on record
        # (it read array_equal=True, max|dE|=0.000e+00 on order (0,0)).
        if os.environ.get('NULL', '1') == '1':
            E1, _c1, _e1, _s1 = one(order, cfgs[0], post, env, R, dx, P_in,
                                    L, M, centre)
            E2, _c2, _e2, _s2 = one(order, cfgs[0], post, env, R, dx, P_in,
                                    L, M, centre)
            print(f"order {order}  NULL: "
                  f"array_equal={np.array_equal(E1, E2)} "
                  f"max|dE|={float(np.abs(E1 - E2).max()):.3e}")
            del E1, E2
        else:
            print(f"order {order}  NULL skipped (floor already on record: "
                  f"array_equal=True, max|dE|=0.000e+00)")
        hdr = (f"{'config':>6} {'EE3 %':>8} {'EE6 %':>8} {'EE12 %':>8} "
               f"{'Ptile %':>9} {'elem(last)':>11} {'g4':>10} {'amax4':>9} "
               f"{'r_rms/mm':>9} {'ENG':>4} {'HAL':>4} {'s':>5}")
        print(hdr)
        print('-' * len(hdr))
        for cfg in cfgs:
            E, last, eng, secs = one(order, cfg, post, env, R, dx, P_in, L, M,
                                     centre)
            mt = A.metrics(E, P_in)
            dxl = float(last['kw']['dx'])
            Nl = last['E_out'].shape[0]
            axl = (np.arange(Nl) - Nl / 2) * dxl
            Xl, Yl = np.meshgrid(axl, axl)
            car = last['kw']['carrier']
            if not hasattr(car, 'R'):
                # ON-AXIS: the chain hands the element a bare float R, not a
                # TiltedCarrier (the same case probe_c6_element.get_call
                # guards).  Without this the (0,0) row dies AFTER the run.
                from lumenairy.elements._lens_traced import TiltedCarrier as _T
                car = _T(float(car), 0.0, 0.0, 0.0, 0.0)
            xo, yo, _a, _b, _c, _d = P.trace_forward([car.x0], [car.y0], car,
                                                     surfs_last)
            Rr = np.hypot(Xl - float(xo[0]), Yl - float(yo[0]))
            pin_l = float((np.abs(last['E_in']) ** 2).sum())
            h = G6.halo(last['E_out'], pin_l, Rr)
            print(f"{cfg:>6} {mt['EE3'] * 100:>8.3f} {mt['EE6'] * 100:>8.3f} "
                  f"{mt['EE12'] * 100:>8.3f} {mt['P_tile'] * 100:>9.4f} "
                  f"{h['P']:>11.6f} {h['g4']:>10.3e} {h['amax4']:>9.2e} "
                  f"{h['r_rms'] * 1e3:>9.4f} {eng[0]:>4d} {eng[1]:>4d} "
                  f"{secs:>5.0f}", flush=True)
        print()


if __name__ == '__main__':
    sys.exit(main())
