"""M5 -- SPURIOUS CENSUS and CASCADE STABILITY for a lossless slanted cell.

T5a  CENSUS.  The full ``4 q^2`` spectrum of a slanted region, against the same
     cell at slant 0 (the shipped out-of-plane generator's own census is the
     control).  Reported: the forward/backward split (must be exactly
     ``2 q^2 / 2 q^2`` BEFORE the selector's defensive rebalance),
     ``min Re(lam_f)`` (a negative value is a GROWING mode classified forward --
     the failure that blows a cascade up), the largest ``|q|`` in the spectrum
     (the "advection spurious" the 1-D convection route grows with slant), the
     count above the physical band ``|q| > max n``, and the worst modal flux
     carried by any of those.

T5b  CASCADE.  ``|R + T - 1|`` for a LOSSLESS slanted cell at depths 0.25 / 1 /
     3 wavelengths and the forward growth factor ``max exp(-Re(lam_f) k0 L)``
     at the deepest.  Closure that does NOT grow with depth is the two-sided
     statement (a growing-mode leak multiplies by ``exp(2 |Re gam| k0 L)``).

T5c  LAYER SPLIT.  One slanted layer of depth ``d`` == two stacked slanted
     layers of depth ``d/2`` (same slant): a consistency identity on the
     propagator and the interface that only holds if the modes and the frame
     anchor compose.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import (  # noqa: E402
    SlantSolver, assert_worktree, slant_region_modes, solve_slant_stack,
    tensor_uniaxial,
)
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE, _region_modes, _region_modes_oop,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
res = {"lumenairy": assert_worktree()}
t00 = time.time()
PX = PY = 1.2
WL = 1.0
K0 = 2.0 * np.pi / WL
NSUP, NSUB = 1.0, 1.5

CELLS = {}
c = np.full((2, 2), 1.0 + 0j)
c[0, 0] = 4.0
CELLS["scalar_pillar_eps4"] = c
c2 = np.zeros((2, 2, 3, 3), dtype=complex)
c2[:, :] = np.eye(3)
c2[0, 0] = tensor_uniaxial(1.5, 1.7, np.deg2rad(35), np.deg2rad(25))
CELLS["oop_uniaxial_pillar"] = c2
c3 = np.full((2, 2), 1.0 + 0j)
c3[0, 0] = 12.0
CELLS["high_contrast_eps12"] = c3
c4 = np.full((2, 2), 1.0 + 0j)
c4[0, 0] = 4.0 + 0.6j            # PUBLIC convention Im(eps) > 0 = loss
CELLS["lossy_pillar"] = c4


def census(cell, slant, M, kx0, ky0):
    a0x, a0y = kx0 * K0, ky0 * K0
    if slant == (0.0, 0.0):
        sol = Granet2DTransverseE(PX, PY, 2, 2, M, cell, alpha0x=a0x,
                                  alpha0y=a0y, k0=K0)
        if sol.offplane:
            Wf, Vf, lf, Wb, Vb, lb = _region_modes_oop(sol)
            qv = np.concatenate([1j * lf, 1j * lb])
        else:
            Wl, Vl, lam, _g2 = _region_modes(sol)
            lf, lb = lam, -lam
            qv = np.concatenate([1j * lf, 1j * lb])
        nf, nb = lf.size, lb.size
    else:
        sol = SlantSolver(PX, PY, 2, 2, M, cell, slant=slant, alpha0x=a0x,
                          alpha0y=a0y, k0=K0)
        out = slant_region_modes(sol)
        lf, lb, qv, fidx, bidx = out[2], out[5], out[6], out[7], out[8]
        nf, nb = fidx.size, bidx.size
    qq = sol.q * sol.q
    nmax = float(np.sqrt(np.max(np.abs(np.asarray(cell, dtype=complex)))))
    big = np.abs(qv) > 3.0 * max(nmax, 1.0)
    return dict(dim=int(qv.size), qq2=int(2 * qq), nf=int(nf), nb=int(nb),
                min_Re_lamf=float(np.min(np.real(lf))),
                max_abs_q=float(np.max(np.abs(qv))),
                max_Re_q=float(np.max(np.abs(np.real(qv)))),
                n_above_band=int(np.sum(big)),
                frac_above_band=float(np.mean(big)))


print("T5a  CENSUS (M = 6, conical 20/35)")
kx0 = np.sin(np.deg2rad(20)) * np.cos(np.deg2rad(35))
ky0 = np.sin(np.deg2rad(20)) * np.sin(np.deg2rad(35))
rows = []
for cname, cell in CELLS.items():
    for sname, sl in (("vertical", (0.0, 0.0)), ("x20", (np.tan(np.deg2rad(20)), 0.0)),
                      ("x45", (1.0, 0.0)), ("diag45", (1.0, 1.0)),
                      ("x60", (np.tan(np.deg2rad(60)), 0.0))):
        r = census(cell, sl, 6, kx0, ky0)
        r.update(cell=cname, slant=sname)
        rows.append(r)
        print(f"  {cname:22s} {sname:9s} dim {r['dim']:5d} split "
              f"{r['nf']}/{r['nb']} (want {r['qq2']}) minRe(lam_f) "
              f"{r['min_Re_lamf']:+.2e} max|q| {r['max_abs_q']:8.2f} "
              f"above-band {r['n_above_band']:4d} "
              f"({100 * r['frac_above_band']:.1f}%)")
res["T5a_census"] = rows

print("\nT5b  CASCADE vs DEPTH (lossless)")
rows = []
for cname in ("scalar_pillar_eps4", "oop_uniaxial_pillar", "lossy_pillar"):
    cell = CELLS[cname]
    for sname, sl in (("vertical", (0.0, 0.0)), ("x36.9", (0.75, 0.0)),
                      ("diag45", (1.0, 1.0))):
        for mount, (th, ph) in (("normal", (0.0, 0.0)),
                                ("conical20_35", (np.deg2rad(20),
                                                  np.deg2rad(35)))):
            row = {"cell": cname, "slant": sname, "mount": mount}
            for depth in (0.25, 1.0, 3.0):
                _o, R, T, _Jr, _Jt, _i = solve_slant_stack(
                    PX, PY, [{"thickness": depth, "cell": cell, "slant": sl}],
                    NSUP, NSUB, WL, M=5, n_orders=3, theta=th, phi=ph)
                row[f"clo_{depth}"] = float(np.max(np.abs(
                    R.sum(1) + T.sum(1) - 1.0)))
            # forward growth at the deepest
            if sl == (0.0, 0.0):
                sol = Granet2DTransverseE(PX, PY, 2, 2, 5, cell,
                                          alpha0x=np.sin(th) * np.cos(ph) * K0,
                                          alpha0y=np.sin(th) * np.sin(ph) * K0,
                                          k0=K0)
                lf = (_region_modes_oop(sol)[2] if sol.offplane
                      else _region_modes(sol)[2])
            else:
                sol = SlantSolver(PX, PY, 2, 2, 5, cell, slant=sl,
                                  alpha0x=np.sin(th) * np.cos(ph) * K0,
                                  alpha0y=np.sin(th) * np.sin(ph) * K0, k0=K0)
                lf = slant_region_modes(sol)[2]
            row["max_fwd_growth"] = float(np.max(np.exp(
                -np.real(lf) * K0 * 3.0)))
            rows.append(row)
            print(f"  {cname:22s} {sname:8s} {mount:12s} clo "
                  f"{row['clo_0.25']:.2e} / {row['clo_1.0']:.2e} / "
                  f"{row['clo_3.0']:.2e}  fwd-growth "
                  f"{row['max_fwd_growth']:.4e}")
res["T5b_cascade"] = rows

print("\nT5c  LAYER SPLIT (one d == two d/2, same slant)")
rows = []
for cname in ("scalar_pillar_eps4", "oop_uniaxial_pillar"):
    cell = CELLS[cname]
    for sname, sl in (("x36.9", (0.75, 0.0)), ("diag45", (1.0, 1.0))):
        for mount, (th, ph) in (("normal", (0.0, 0.0)),
                                ("conical20_35", (np.deg2rad(20),
                                                  np.deg2rad(35)))):
            o1, R1, T1, J1, Jt1, _i = solve_slant_stack(
                PX, PY, [{"thickness": 0.8, "cell": cell, "slant": sl}],
                NSUP, NSUB, WL, M=5, n_orders=3, theta=th, phi=ph)
            # NOTE the second half's cell is the cross-section at ITS top,
            # i.e. the first half's bottom -- the frame anchor is per layer.
            o2, R2, T2, J2, Jt2, _j = solve_slant_stack(
                PX, PY,
                [{"thickness": 0.4, "cell": cell, "slant": sl},
                 {"thickness": 0.4, "cell": cell, "slant": sl}],
                NSUP, NSUB, WL, M=5, n_orders=3, theta=th, phi=ph)
            row = {"cell": cname, "slant": sname, "mount": mount,
                   "dR": float(np.max(np.abs(R1 - R2))),
                   "dT": float(np.max(np.abs(T1 - T2))),
                   "dJr": float(np.max(np.abs(J1 - J2)))}
            rows.append(row)
            print(f"  {cname:22s} {sname:8s} {mount:12s} dR {row['dR']:.2e} "
                  f"dT {row['dT']:.2e} dJr {row['dJr']:.2e}")
res["T5c_split"] = rows

print("\nT5d  NO-FLOOR: does the far-field order count move a SLANTED answer?")
rows = []
for cname in ("scalar_pillar_eps4", "oop_uniaxial_pillar"):
    cell = CELLS[cname]
    for sname, sl in (("x36.9", (0.75, 0.0)), ("diag45", (1.0, 1.0))):
        for mount, (th, ph) in (("normal", (0.0, 0.0)),
                                ("conical20_35", (np.deg2rad(20),
                                                  np.deg2rad(35)))):
            base = None
            row = {"cell": cname, "slant": sname, "mount": mount}
            for no in (3, 5, 8):
                o, R, T, Jr, _Jt, _i = solve_slant_stack(
                    PX, PY, [{"thickness": 0.5, "cell": cell, "slant": sl}],
                    NSUP, NSUB, WL, M=5, n_orders=no, theta=th, phi=ph)
                idx = {(int(m), int(n)): i for i, (m, n) in enumerate(o)}
                v = {k: (float(R[0, idx[k]]), float(T[0, idx[k]]))
                     for k in ((0, 0), (1, 0), (0, 1), (-1, 0))}
                if base is None:
                    base, jbase = v, Jr
                else:
                    row[f"move_n{no}"] = max(
                        max(abs(a - b) for a, b in zip(v[k], base[k]))
                        for k in v)
                    row[f"dJ_n{no}"] = float(np.max(np.abs(Jr - jbase)))
            rows.append(row)
            print(f"  {cname:22s} {sname:8s} {mount:12s} "
                  f"n_orders 3->5 {row['move_n5']:.2e} (dJ {row['dJ_n5']:.1e})"
                  f"  3->8 {row['move_n8']:.2e} (dJ {row['dJ_n8']:.1e})")
res["T5d_no_floor"] = rows

res["wall_s"] = time.time() - t00
with open(os.path.join(OUT, "m5_census_cascade.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"\nWROTE results/m5_census_cascade.json  ({res['wall_s']:.1f} s)")
