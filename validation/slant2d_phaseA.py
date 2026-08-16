"""PHASE A validation battery for the native 2-D slant metric.

Gates
-----
A1  sign pin      -- 4 convection-sign candidates vs the EXACT 1-D slant oracle
A2  null control  -- a UNIFORM cell with any slant is a NO-OP (a shear of a
                     homogeneous medium is a coordinate change)
A3  null control  -- a y-uniform cell sheared ALONG Y is a NO-OP (shearing an
                     invariant direction does nothing); and (tx, ty) on such a
                     cell must equal (tx, 0)
A4  1-D oracle    -- y-uniform cell, slant along x, vs pmm_jones_1d_slanted,
                     tracked against the VERTICAL control at the same n_orders
A5  staircase     -- an N-slice laterally-shifted staircase must CONVERGE
                     TOWARD the single metric layer as N grows

The lossless trap is explicitly avoided: every gate is a PER-ORDER comparison
against an independent oracle, never an energy-closure check.
"""
import time
import warnings

import numpy as np

import slant2d_proto as SP
from lumenairy.elements.pmm import (pmm_jones_2d, pmm_jones_1d,
                                    pmm_jones_1d_slanted)
from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid

P = 700e-9
WL = 633e-9
D = 300e-9
DUTY = 0.5
ER, EG = 4.0 + 0j, 1.0 + 0j
NSUB, NSUP = 1.5, 1.0
NX = 480
DEG = 11

warnings.simplefilter("ignore")


def _n0(o, R, T):
    o = np.asarray(o)
    keep = o[:, 1] == 0
    idx = np.argsort(o[keep][:, 0])
    return (o[keep][:, 0][idx], np.asarray(R)[:, keep][:, idx],
            np.asarray(T)[:, keep][:, idx])


def run2d(tx, ty, n_ord, theta, c=-1j, shift_px=0, cell=None):
    SP.set_slant(tx, ty, c)
    try:
        cl = SP.binary_cell(NX, DUTY, ER, EG, shift_px) if cell is None else cell
        o, R, T, J = pmm_jones_2d(P, P, cl, NSUB, NSUP, D, WL, theta=theta,
                                  phi=0.0, degree=DEG, n_orders=n_ord)
    finally:
        SP.set_slant(0.0, 0.0)
    return _n0(o, R, T)


def run1d(phi, theta, n_ff=21):
    if phi == 0.0:
        o, R, T, J = pmm_jones_1d(P, ER * np.eye(3), EG * np.eye(3), NSUB, NSUP,
                                  D, DUTY, WL, angle=theta, degree=16,
                                  far_field_orders=n_ff)
    else:
        o, R, T, J = pmm_jones_1d_slanted(P, ER * np.eye(3), EG * np.eye(3),
                                          NSUB, NSUP, D, DUTY, WL, phi,
                                          angle=theta, degree=16,
                                          far_field_orders=n_ff,
                                          factorization="convection")
    return np.asarray(o), np.asarray(R), np.asarray(T)


def cmp_o(m2, R2, T2, o1, R1, T1):
    common = np.intersect1d(m2, o1)
    i2, i1 = np.searchsorted(m2, common), np.searchsorted(o1, common)
    d = 0.0
    for r in (0, 1):
        d = max(d, float(np.max(np.abs(R2[r][i2] - R1[r][i1]))))
        d = max(d, float(np.max(np.abs(T2[r][i2] - T1[r][i1]))))
    return d


def cmp_2d(a, b):
    _m, Ra, Ta = a
    _m2, Rb, Tb = b
    return max(float(np.max(np.abs(Ra - Rb))), float(np.max(np.abs(Ta - Tb))))


PHI = np.radians(20.0)
TX = float(np.tan(PHI))


def gate_A1():
    print("\n=== A1  convection SIGN pin (theta=0.20, y-uniform, slant along x) ===")
    print("    winner must TRACK the vertical control; others must not")
    for n_ord in (7, 11, 15):
        o1v, R1v, T1v = run1d(0.0, 0.20)
        ctrl = cmp_o(*run2d(0.0, 0.0, n_ord, 0.20), o1v, R1v, T1v)
        o1, R1, T1 = run1d(PHI, 0.20)
        out = [f"ctrl={ctrl:.2e}"]
        for name, c in (("+1", 1 + 0j), ("-1", -1 + 0j), ("+i", 1j), ("-i", -1j)):
            try:
                d = cmp_o(*run2d(TX, 0.0, n_ord, 0.20, c=c), o1, R1, T1)
                out.append(f"{name}:{d:.2e}")
            except Exception as e:  # noqa: BLE001
                out.append(f"{name}:{type(e).__name__}")
        print(f"    n_orders={n_ord:>2}  " + "  ".join(out))


def gate_A2():
    print("\n=== A2  NULL: uniform cell + slant is a NO-OP ===")
    uni = SP.binary_cell(NX, DUTY, ER, ER)     # ridge == groove
    for theta in (0.0, 0.20):
        base = run2d(0.0, 0.0, 11, theta, cell=uni)
        for txd in (5.0, 20.0, 40.0):
            t = float(np.tan(np.radians(txd)))
            d = cmp_2d(run2d(t, 0.0, 11, theta, cell=uni), base)
            dxy = cmp_2d(run2d(t, 0.7 * t, 11, theta, cell=uni), base)
            print(f"    theta={theta}  slant={txd:4.1f}deg  "
                  f"|d| x-only={d:.2e}   x+y={dxy:.2e}")


def gate_A3():
    print("\n=== A3  NULL: y-uniform cell sheared ALONG Y is a NO-OP ===")
    for theta in (0.0, 0.20):
        base = run2d(0.0, 0.0, 11, theta)
        for tyd in (10.0, 30.0):
            ty = float(np.tan(np.radians(tyd)))
            d = cmp_2d(run2d(0.0, ty, 11, theta), base)
            print(f"    theta={theta}  ty={tyd:4.1f}deg alone      : {d:.2e}")
        bx = run2d(TX, 0.0, 11, theta)
        for tyd in (10.0, 30.0):
            ty = float(np.tan(np.radians(tyd)))
            d = cmp_2d(run2d(TX, ty, 11, theta), bx)
            print(f"    theta={theta}  (tx,ty={tyd:4.1f}deg) vs (tx,0): {d:.2e}")


def gate_A4():
    print("\n=== A4  1-D ORACLE: y-uniform slanted grating vs pmm_jones_1d_slanted ===")
    print("    slant residual must TRACK the vertical control (same floor)")
    for theta in (0.0, 0.20):
        for phid in (10.0, 20.0, 35.0):
            phi = np.radians(phid)
            tx = float(np.tan(phi))
            o1v, R1v, T1v = run1d(0.0, theta)
            o1s, R1s, T1s = run1d(phi, theta)
            row = []
            for n_ord in (7, 11, 15):
                ctrl = cmp_o(*run2d(0.0, 0.0, n_ord, theta), o1v, R1v, T1v)
                slnt = cmp_o(*run2d(tx, 0.0, n_ord, theta, c=-1j), o1s, R1s, T1s)
                row.append(f"n{n_ord}: ctrl {ctrl:.1e} / slant {slnt:.1e}")
            print(f"    theta={theta} phi={phid:4.1f}deg  " + " | ".join(row))


def gate_A4b():
    """Guard against a MIRRORED order-label pairing.

    _core.py:5285-5289 warns that the slant sign convention can flip the
    diffraction order labels m -> -m ("identical physics, only the order
    labels flip").  If the 2-D (+t_x) convention pairs with the 1-D (-phi)
    convention, A4 would fail for every sign candidate while the formulation
    is in fact correct.  This gate resolves the pairing explicitly.
    """
    print("\n=== A4b  order-label pairing: 2-D(+tx) vs 1-D(+phi) or 1-D(-phi) ===")
    theta, n_ord = 0.20, 11
    for phid in (10.0, 20.0):
        phi = np.radians(phid)
        tx = float(np.tan(phi))
        got = run2d(tx, 0.0, n_ord, theta, c=-1j)
        dp = cmp_o(*got, *run1d(phi, theta))
        dm = cmp_o(*got, *run1d(-phi, theta))
        print(f"    phi={phid:4.1f}deg   vs 1-D(+phi)={dp:.3e}   "
              f"vs 1-D(-phi)={dm:.3e}   -> {'+phi' if dp < dm else '-phi'}")


def gate_A5():
    print("\n=== A5  STAIRCASE convergence toward the single metric layer ===")
    theta, n_ord = 0.20, 11
    # total lateral walk chosen as an INTEGER pixel count so every slice is exact
    walk_px = 96                       # 96/480 of a period
    walk = walk_px / NX * P
    tx = walk / D
    phid = np.degrees(np.arctan(tx))
    print(f"    walk={walk*1e9:.1f}nm over d={D*1e9:.0f}nm -> {phid:.2f} deg (tx={tx:.4f})")
    t0 = time.time()
    metric = run2d(tx, 0.0, n_ord, theta, c=-1j)
    print(f"    metric layer (1 eigensolve): {time.time()-t0:.1f}s")
    prev = None
    for ns in (1, 2, 4, 8, 16, 32):
        if walk_px % ns:
            continue
        t0 = time.time()
        # formulation MUST match pmm_jones_2d's default ('laurent'); the stack
        # defaults to 'li' and the two differ by ~1e-2 at n_orders=7 (measured).
        # Matched, stack(ns=1) vs the single-layer solver is EXACTLY 0.0.
        st = PMM2DStackHybrid(P, P, n_substrate=NSUB, n_superstrate=NSUP,
                              degree=DEG, n_orders=n_ord, formulation="laurent")
        for k in range(ns):
            # slice k, TOP-down (stack is superstrate-first); frame anchored at
            # the layer TOP to match add_sheared_grating's convention
            sh = int(round(walk_px * (k + 0.5) / ns))
            st.add_layer(D / ns, eps_tensor_cell=SP.binary_cell(NX, DUTY, ER, EG, sh))
        st.set_source(WL, theta=theta, phi=0.0)
        o, R, T, J = st.solve()
        dt = time.time() - t0
        stair = _n0(o, R, T)
        d = cmp_2d(stair, metric)
        rat = f"  ratio {prev/d:.2f}" if prev else ""
        print(f"    ns={ns:>2}: |staircase - metric| = {d:.3e}{rat}   ({dt:.1f}s)")
        prev = d


if __name__ == "__main__":
    gate_A2()
    gate_A3()
    gate_A1()
    gate_A4()
    gate_A5()
