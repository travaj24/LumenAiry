"""F-B1 -- which Jacobian entries must be lattice-periodic?

Phase A's ``CellMap.validate`` required ALL FOUR Jacobian entries to be equal
on the two sides of the periodic seam.  The unknown that must be continuous
across the side u = 0 ~ p_x is E'_v = E . dPhi/dv (the staggered basis is
broken in u for E'_u), so only the TANGENTIAL derivative (dPhi/dv on the
u-sides, dPhi/du on the v-sides) is needed.  This probe measures, for every
Phase-B gate map, the seam mismatch of the full Jacobian (Phase A's check)
and of the tangential part (the amended check), plus the position mismatch.

  python validation/probe_pmm2d_curved/build_b/b0_validate.py
Output: b0_validate.json
"""
import _common as C
import numpy as np
from numpy.polynomial.legendre import leggauss

MAPS = {
    "circle3": lambda: C.CM._circle_map_3x3(C.P, C.R_CIRC)[0],
    "circle5": lambda: C.CM._circle_map_5x5(C.P, C.R_CIRC)[0],
    "fillet5_r0.2": lambda: C.CM._fillet_map_5x5(C.P, 0.3, 0.12)[0],
    "ellipse3": lambda: C.CM._ellipse_map_3x3(C.P, (0.40, 0.28))[0],
    "sine3": lambda: C.CM._sine_stripe_map_3x3(C.P, 0.3, 0.9, 0.12)[0],
}


def seam(cm, n=12):
    xg, _ = leggauss(n)
    Nx, Ny = cm.shape
    pos = full = tang = 0.0
    for sy in range(Ny):
        V = 0.5 * (cm.v_bounds[sy] + cm.v_bounds[sy + 1]) + 0.5 * (
            cm.v_bounds[sy + 1] - cm.v_bounds[sy]) * xg
        a = cm.geom(0, sy, np.array([cm.u_bounds[0]]), V)
        b = cm.geom(Nx - 1, sy, np.array([cm.u_bounds[-1]]), V)
        pos = max(pos, float(np.max(np.abs(b[0] - a[0] - cm.period_x))),
                  float(np.max(np.abs(b[1] - a[1]))))
        full = max(full, max(float(np.max(np.abs(b[k] - a[k])))
                             for k in (2, 3, 4, 5)))
        tang = max(tang, max(float(np.max(np.abs(b[k] - a[k])))
                             for k in (3, 5)))
    for sx in range(Nx):
        U = 0.5 * (cm.u_bounds[sx] + cm.u_bounds[sx + 1]) + 0.5 * (
            cm.u_bounds[sx + 1] - cm.u_bounds[sx]) * xg
        a = cm.geom(sx, 0, U, np.array([cm.v_bounds[0]]))
        b = cm.geom(sx, Ny - 1, U, np.array([cm.v_bounds[-1]]))
        pos = max(pos, float(np.max(np.abs(b[0] - a[0]))),
                  float(np.max(np.abs(b[1] - a[1] - cm.period_y))))
        full = max(full, max(float(np.max(np.abs(b[k] - a[k])))
                             for k in (2, 3, 4, 5)))
        tang = max(tang, max(float(np.max(np.abs(b[k] - a[k])))
                             for k in (2, 4)))
    return pos, full, tang


def main():
    res = {"env": C.env_record(), "maps": {}}
    for name, mk in MAPS.items():
        cm = mk()
        pos, full, tang = seam(cm)
        res["maps"][name] = {"position": pos, "full_jacobian": full,
                             "tangential": tang,
                             "singular_vertices": cm.singular_vertices}
        print(name, f"pos {pos:.1e} full {full:.2e} tangential {tang:.1e}")
    C.dump("b0_validate.json", res)


if __name__ == "__main__":
    main()
