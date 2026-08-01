# Conservation audit (2026-07-31) -- RECON only.  Prints the facts the energy
# guard's arithmetic depends on, so nothing downstream is assumed:
#   * the pinned library file identity (sha256) actually imported,
#   * design 121's post-DOE group prescriptions and their aperture_diameter,
#   * for each cached element call: whole-grid P_in vs the guard's
#     APERTURE-MASKED P_in (the guard's denominator), and the fraction the
#     aperture removes -- i.e. how much legitimate vignetting the guard is
#     blind to by construction.
# LOCAL-ONLY.  No library edit; reads only.
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _d121_common as C                                          # noqa: E402
import probe_c6_element as E6                                     # noqa: E402
import wfe_probe_common as P                                      # noqa: E402
import lumenairy.elements._lens_traced as LT                      # noqa: E402
from lumenairy.elements._lens_traced import (TiltedCarrier,       # noqa: E402
                                             _input_beam_amp_radius)

ORDERS = [tuple(int(v) for v in o.split(','))
          for o in os.environ.get('ORDERS', '0,0 -1,0 -4,0 -4,-2').split()]


def _sha(fn):
    return hashlib.sha256(open(fn, 'rb').read()).hexdigest()[:16]


def main():
    import lumenairy.propagators.carrier as CM
    print("LIBRARY ACTUALLY IMPORTED")
    print(f"  _lens_traced.py {LT.__file__}\n    sha256 {_sha(LT.__file__)}")
    print(f"  carrier.py      {CM.__file__}\n    sha256 {_sha(CM.__file__)}")
    print(f"  C5 TILTED_CARRIER_EXACT_EIKONAL     = "
          f"{LT.TILTED_CARRIER_EXACT_EIKONAL}")
    print(f"  C6 REMAP_STATIONARY_PHASE_LAUNCH    = "
          f"{LT.REMAP_STATIONARY_PHASE_LAUNCH}")
    print(f"  C6 REMAP_STATIONARY_PHASE_FIT_GUARD = "
          f"{getattr(LT, 'REMAP_STATIONARY_PHASE_FIT_GUARD', 'ABSENT')}")
    print(f"  guard band constants: DEFICIT_BASE "
          f"{LT._RD_ENERGY_DEFICIT_BASE} PER_SUB "
          f"{LT._RD_ENERGY_DEFICIT_PER_SUB} GAIN_TOL "
          f"{LT._RD_ENERGY_GAIN_TOL}")
    for sub in (1, 2, 4, 8):
        lo = 1.0 - (LT._RD_ENERGY_DEFICIT_BASE
                    + LT._RD_ENERGY_DEFICIT_PER_SUB * (sub - 1))
        print(f"    ray_subsample={sub}: band "
              f"[{lo:.4f}, {1.0 + LT._RD_ENERGY_GAIN_TOL:.4f}]  "
              f"width {1.0 + LT._RD_ENERGY_GAIN_TOL - lo:.4f}")
    print()

    _pre, post, gap, period = C.geometry()
    print(f"POST-DOE GROUPS: {len(post)}   DOE period {period * 1e6:.4f} um  "
          f"gap_to_doe {gap * 1e3:.4f} mm  trailing {C.TRAILING * 1e3:.4f} mm")
    for j, g in enumerate(post):
        ap = g['prescription'].get('aperture_diameter')
        ns = len(g['prescription'].get('surfaces',
                                       g['prescription'].get('radii', [])))
        print(f"  group {j}: gap_before {g['gap_before'] * 1e3:9.4f} mm  "
              f"aperture_diameter "
              f"{'None' if ap is None else f'{ap * 1e3:.4f} mm'}  "
              f"keys {sorted(g['prescription'])[:6]}")
    print()

    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)
    ap = presc.get('aperture_diameter')
    print("CACHED LAST-GROUP ELEMENT CALLS (C6 OFF capture = post-C5 shipped)")
    hdr = (f"{'order':>8} {'N':>5} {'dx/um':>8} {'w/mm':>8} {'dec/w':>7} "
           f"{'P_in(grid)':>12} {'P_in(ap)':>12} {'ap cuts':>9} "
           f"{'chief x,y /mm':>20}")
    print(hdr)
    print('-' * len(hdr))
    for (m, n) in ORDERS:
        try:
            E_in, E_out, carv, dx = E6.get_call(m, n)
        except Exception as exc:                       # noqa: BLE001
            print(f"{f'({m},{n})':>8}  NO CACHE -- {type(exc).__name__}: "
                  f"{str(exc)[:70]}")
            continue
        car = TiltedCarrier(*carv)
        N = E_in.shape[0]
        axg = (np.arange(N) - N / 2) * dx
        Xg, Yg = np.meshgrid(axg, axg)
        w = float(_input_beam_amp_radius(E_in, dx, dx,
                                         centre=(car.x0, car.y0)))
        p_grid = float((np.abs(E_in) ** 2).sum())
        if ap is None:
            p_ap = p_grid
        else:
            msk = Xg ** 2 + Yg ** 2 <= (ap / 2) ** 2
            p_ap = float((np.abs(E_in) ** 2)[msk].sum())
        xo, yo, _a, _b, _c, _d = P.trace_forward([car.x0], [car.y0], car, surfs)
        print(f"{f'({m},{n})':>8} {N:>5} {dx * 1e6:>8.3f} {w * 1e3:>8.4f} "
              f"{np.hypot(car.x0, car.y0) / w:>7.3f} {p_grid:>12.5e} "
              f"{p_ap:>12.5e} {1.0 - p_ap / p_grid:>9.2e} "
              f"{f'{float(xo[0]) * 1e3:+.4f},{float(yo[0]) * 1e3:+.4f}':>20}")
    print()
    print("ap cuts = fraction of the ELEMENT INPUT power the aperture removes "
          "= the vignetting the guard's denominator hides by construction.")


if __name__ == '__main__':
    sys.exit(main())
