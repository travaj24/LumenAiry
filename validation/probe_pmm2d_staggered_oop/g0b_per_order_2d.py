"""GATE 0b -- PER-ORDER localization of the out-of-plane assembly.

GATE 0's first arm shows the prototype's PER-ORDER transmission on the (3,3)
L cell departing from BOTH Fourier oracles by ~2.7e-02 while its own M-ladder
is converged to 1e-06 -- an order of magnitude worse than the 2.2e-03 the
prototype doc reports on ``sum R``, because the doc's 2-D comparisons (M5, M8,
M9) compare TOTALS (``R.sum(axis=1)``), never per order.

This probe asks WHICH BLOCK.  The nine eps-weighted masses and six div-D
blocks split by axis: ``A13 / A31 / K13`` carry an x-derivative or an x-mixed
mass, ``A23 / A32 / K23`` carry the y ones.  M4 exercised a PATTERNED cell
with ``azim = 0`` (``e23 = 0``) only, so the y-axis out-of-plane blocks have
never been tested against a per-order oracle on a patterned cell.

Arms (all per order, both polarizations, against Fourier oracles):
  A  x-patterned stripe (y-uniform), director azimuth 0 / 90 / 25
  B  y-patterned stripe (x-uniform), same three azimuths  (the transpose)
  C  the M5 CONVEX (2,2) pillar, azimuth 25, normal + conical
each with its IN-PLANE control (out-of-plane entries zeroed) on the same walls.

Run:
  cd /c/tmp/lum_aniso_oopint && PYTHONPATH=/c/tmp/lum_aniso_oopint \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python validation/probe_pmm2d_staggered_oop/g0b_per_order_2d.py
"""
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import probe_common as pc  # noqa: E402

from lumenairy.elements.rcwa import rcwa_jones_2d  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

WL = 1.0
PX = PY = 1.2
DEPTH = 0.4
NSUB, NSUP = 1.5, 1.0
MSTAG = 7
RCWA_NO = 9


def strip_oop(t):
    o = np.array(t, dtype=complex)
    o[0, 2] = o[1, 2] = o[2, 0] = o[2, 1] = 0.0
    return o


def upsample(ec, n):
    return np.repeat(np.repeat(np.asarray(ec), n, axis=0), n, axis=1)


def cmp_per_order(tag, o_s, R_s, T_s, J_s, o_r, R_r, T_r, J_r, keep=None):
    """Max |stag - oracle| over the orders both carry (optionally restricted
    to the PROPAGATING ones, where the efficiency is not identically zero)."""
    idx_r = {tuple(int(v) for v in row): j for j, row in enumerate(np.asarray(o_r))}
    dR = dT = 0.0
    worst = None
    for i, row in enumerate(np.asarray(o_s)):
        k = tuple(int(v) for v in row)
        j = idx_r.get(k)
        if j is None:
            continue
        if keep is not None and k not in keep:
            continue
        a = float(np.max(np.abs(np.asarray(R_s)[:, i] - np.asarray(R_r)[:, j])))
        b = float(np.max(np.abs(np.asarray(T_s)[:, i] - np.asarray(T_r)[:, j])))
        if max(a, b) > max(dR, dT):
            worst = k
        dR, dT = max(dR, a), max(dT, b)
    dJ = float(np.max(np.abs(np.asarray(J_s) - np.asarray(J_r))))
    print(f"      {tag:34s} per-order dR = {dR:.3e}  dT = {dT:.3e}  "
          f"dJones = {dJ:.3e}   (worst order {worst})")
    return dict(dR=dR, dT=dT, dJ=dJ, worst=str(worst))


def run_cell(name, ec, th, ph, res):
    print(f"\n=== {name}  theta={np.rad2deg(th):.0f} phi={np.rad2deg(ph):.0f} ===")
    n = ec.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fine = upsample(ec, int(np.ceil((4 * RCWA_NO + 1) / n)))
        o_r, R_r, T_r, J_r = rcwa_jones_2d(PX, PY, fine, NSUB, NSUP, DEPTH, WL,
                                           theta=th, phi=ph,
                                           n_orders_x=RCWA_NO,
                                           n_orders_y=RCWA_NO)
        fine2 = upsample(ec, int(np.ceil((4 * (RCWA_NO - 2) + 1) / n)))
        o_r2, R_r2, T_r2, J_r2 = rcwa_jones_2d(PX, PY, fine2, NSUB, NSUP,
                                               DEPTH, WL, theta=th, phi=ph,
                                               n_orders_x=RCWA_NO - 2,
                                               n_orders_y=RCWA_NO - 2)
    # the oracle's OWN per-order drift is the bar
    own = cmp_per_order(f"rcwa({RCWA_NO-2}) vs rcwa({RCWA_NO})", o_r2, R_r2,
                        T_r2, J_r2, o_r, R_r, T_r, J_r)
    cand = "a" if np.max(np.abs(ec[..., [0, 1, 2, 2], [2, 2, 0, 1]])) > 1e-12 \
        else "eform"
    rows = {}
    for M in (MSTAG - 1, MSTAG):
        o_s, R_s, T_s, J_s = pc.solve_slab(PX, PY, ec, NSUB, NSUP, DEPTH, WL,
                                           M=M, theta=th, phi=ph,
                                           candidate=cand)
        rows[M] = (o_s, R_s, T_s, J_s)
        d = cmp_per_order(f"staggered('{cand}') M={M} vs rcwa({RCWA_NO})",
                          o_s, R_s, T_s, J_s, o_r, R_r, T_r, J_r)
        res[f"{name}|th{int(np.rad2deg(th))}|M{M}"] = d
    self_move = cmp_per_order("staggered self-move (M-1 -> M)",
                              *rows[MSTAG - 1], *rows[MSTAG])
    res[f"{name}|th{int(np.rad2deg(th))}|selfmove"] = self_move
    res[f"{name}|th{int(np.rad2deg(th))}|oracle_drift"] = own
    return res


def main():
    pc.banner("GATE 0b -- per-order localization of the OOP assembly")
    res = {}
    eg = np.eye(3, dtype=complex)

    for azim in (0.0, 90.0, 25.0):
        er = pc.uniaxial(1.5, 1.7, 35.0, azim_deg=azim)
        print(f"\n############ director tilt 35, azimuth {azim:.0f}: "
              f"e13 = {er[0,2]:+.5f}  e23 = {er[1,2]:+.5f} ############")
        for tag, t33 in (("OOP", er), ("INPLANE", strip_oop(er))):
            # A: x-patterned stripe (y-uniform)
            ex = np.zeros((2, 2, 3, 3), dtype=complex)
            ex[:, :] = eg
            ex[0, :] = t33
            run_cell(f"A x-stripe azim{azim:.0f} {tag}", ex, 0.0, 0.0, res)
            # B: y-patterned stripe (x-uniform) -- the transpose
            ey = np.zeros((2, 2, 3, 3), dtype=complex)
            ey[:, :] = eg
            ey[:, 0] = t33
            run_cell(f"B y-stripe azim{azim:.0f} {tag}", ey, 0.0, 0.0, res)

    # C: the M5 CONVEX pillar, azimuth 25, normal + conical
    er = pc.uniaxial(1.5, 1.7, 35.0, azim_deg=25.0)
    for tag, t33 in (("OOP", er), ("INPLANE", strip_oop(er))):
        ec = np.zeros((2, 2, 3, 3), dtype=complex)
        ec[:, :] = eg
        ec[0, 0] = t33
        for th, ph in ((0.0, 0.0), (np.deg2rad(20.0), np.deg2rad(35.0))):
            run_cell(f"C pillar {tag}", ec, th, ph, res)

    with open(os.path.join(OUT, "g0b_per_order_2d.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("\nwrote results/g0b_per_order_2d.json")


if __name__ == "__main__":
    main()
