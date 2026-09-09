"""M7 -- SLANT x ANISOTROPY, including the combination NO 2-D engine in the
suite currently covers.

T7a  the COVERAGE GAP, stated by measurement: ``PMM2DStackHybrid.add_layer``
     raises on ``slant`` combined with OUT-OF-PLANE tensor components.  The
     exception text is recorded.
T7b  a y-uniform SLANTED OUT-OF-PLANE stripe (a tilted LC director in a slanted
     ridge) against ``pmm_jones_1d_slanted`` -- the 1-D engine that DOES cover
     the combination (validated in-library against an RCWA tensor z-staircase,
     slant 15-60 deg).  Per order, both incident polarizations, M ladder.
T7c  a slanted IN-PLANE tensor PILLAR (genuinely 2-D) against the shipped
     hybrid slant metric, which does cover that case -- an independent
     formulation, so this is a cross-engine check on the covariant congruence
     with a non-scalar tensor.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import (  # noqa: E402
    assert_worktree, solve_slant_stack, tensor_uniaxial,
)
from lumenairy.elements.pmm import (  # noqa: E402
    PMM2DStackHybrid, pmm_jones_1d_slanted,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
res = {"lumenairy": assert_worktree()}
t00 = time.time()

PX = PY = 0.75
WL, DEPTH = 1.0, 0.30
NSUP, NSUB = 1.0, 1.5
ORD_CMP = [-2, -1, 0, 1, 2]
TILT_LC = tensor_uniaxial(1.5, 1.7, np.deg2rad(35), np.deg2rad(25))   # OOP
INPL_LC = tensor_uniaxial(1.5, 1.7, np.pi / 2, np.deg2rad(25))        # in-plane

# ------------------------------------------------------------------ T7a gap
print("T7a  the hybrid's slant x out-of-plane REFUSAL")
gap = {}
for nm, t33 in (("oop", TILT_LC), ("inplane", INPL_LC)):
    cellpx = np.zeros((16, 16, 3, 3), dtype=complex)
    cellpx[:, :] = np.eye(3)
    cellpx[:8, :] = t33
    st = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                          degree=9, n_orders=5, symmetry=False)
    try:
        st.add_layer(DEPTH, eps_tensor_cell=cellpx, slant=(0.5, 0.0))
        gap[nm] = "ACCEPTED"
    except Exception as exc:                                # noqa: BLE001
        gap[nm] = f"{type(exc).__name__}: {exc}"
    print(f"  hybrid slant + {nm:8s} -> {gap[nm][:150]}")
res["T7a_hybrid_gap"] = gap

# ------------------------------------ T7b  y-uniform slanted OOP stripe vs 1-D
print("\nT7b  y-uniform SLANTED OUT-OF-PLANE stripe vs pmm_jones_1d_slanted")


def cell_stripe(t33):
    c = np.zeros((2, 2, 3, 3), dtype=complex)
    c[0, :] = t33
    c[1, :] = np.eye(3)
    return c


def oracle_1d(t33, phi_deg, theta, degree=24, fact="convection"):
    o, R, T, _J = pmm_jones_1d_slanted(
        PX, t33, np.eye(3), NSUB, NSUP, DEPTH, 0.5, WL, np.deg2rad(phi_deg),
        angle=theta, degree=degree, far_field_orders=15, factorization=fact)
    return {"ex": {int(m): (float(R[0, i]), float(T[0, i]))
                   for i, m in enumerate(o)},
            "ey": {int(m): (float(R[1, i]), float(T[1, i]))
                   for i, m in enumerate(o)}}


def two_d(t33, t, theta, M):
    o, R, T, _Jr, _Jt, _i = solve_slant_stack(
        PX, PY, [{"thickness": DEPTH, "cell": cell_stripe(t33),
                  "slant": (t, 0.0)}],
        NSUP, NSUB, WL, M=M, n_orders=4, theta=theta, phi=0.0)
    idx = {int(m): i for i, (m, n) in enumerate(o) if n == 0}
    ylk = max([0.0] + [float(np.max(np.abs(R[:, i]))) for i, (m, n)
                       in enumerate(o) if n != 0])
    return ({"ex": {m: (float(R[0, i]), float(T[0, i])) for m, i in idx.items()},
             "ey": {m: (float(R[1, i]), float(T[1, i])) for m, i in idx.items()}},
            ylk, float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))))


def d(a, b):
    return max(max(abs(x[0] - y[0]), abs(x[1] - y[1]))
               for k in ORD_CMP if k in a and k in b
               for x, y in ((a[k], b[k]),))


rows = []
for phi_deg in (0.0, 20.0, 35.0):
    for mount, th in (("normal", 0.0), ("oblique25", np.deg2rad(25))):
        orc24 = oracle_1d(TILT_LC, phi_deg, th, 24)
        orc30 = oracle_1d(TILT_LC, phi_deg, th, 30)
        own = max(d(orc24["ex"], orc30["ex"]), d(orc24["ey"], orc30["ey"]))
        for M in (5, 6, 7, 8):
            d2, ylk, clo = two_d(TILT_LC, -np.tan(np.deg2rad(phi_deg)), th, M)
            row = {"phi": phi_deg, "mount": mount, "M": M,
                   "ex": d(d2["ex"], orc30["ex"]),
                   "ey": d(d2["ey"], orc30["ey"]),
                   "oracle_own_drift": own, "yleak": ylk, "closure": clo}
            rows.append(row)
            print(f"  phi={phi_deg:4.0f} {mount:9s} M={M} Ex {row['ex']:.2e} "
                  f"Ey {row['ey']:.2e} (oracle own {own:.1e}) "
                  f"yleak {ylk:.1e} |R+T-1| {clo:.2e}")
res["T7b_oop_stripe"] = rows

# ------------------------------------- T7c  slanted IN-PLANE tensor pillar
print("\nT7c  slanted IN-PLANE tensor PILLAR vs the hybrid slant metric")
PX2 = PY2 = 1.2
DEPTH2 = 0.8
TX = 0.75


def cell_pillar(N, t33, shift=0.0):
    h = PX2 / N
    c = np.zeros((N, N, 3, 3), dtype=complex)
    c[:, :] = np.eye(3)
    for i in range(N):
        for j in range(N):
            xc = (i * h + 0.5 * h - shift) % PX2
            yc = (j * h + 0.5 * h) % PY2
            if 0.3 <= xc < 0.6 and 0.3 <= yc < 0.6:
                c[i, j] = t33
    return c


ORD2 = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]


def po(o, R, T):
    idx = {(int(m), int(n)): i for i, (m, n) in enumerate(o)}
    return {k: (float(R[0, idx[k]]), float(R[1, idx[k]]),
                float(T[0, idx[k]]), float(T[1, idx[k]]))
            for k in ORD2 if k in idx}


def dd(a, b):
    return max(max(abs(x - y) for x, y in zip(a[k], b[k]))
               for k in a if k in b)


rows = []
for mount, (th, ph) in (("normal", (0.0, 0.0)),
                        ("conical20_35", (np.deg2rad(20), np.deg2rad(35)))):
    proto_v = {}
    for M in (5, 6):
        o, R, T, _Jr, _Jt, _i = solve_slant_stack(
            PX2, PY2, [{"thickness": DEPTH2, "cell": cell_pillar(4, INPL_LC),
                        "slant": (TX, 0.0)}],
            NSUP, NSUB, WL, M=M, n_orders=3, theta=th, phi=ph)
        proto_v[M] = po(o, R, T)
    row = {"mount": mount,
           "proto_selfmove_M5_M6": dd(proto_v[5], proto_v[6])}
    for sgn in (+1.0, -1.0):
        for no in (5, 7):
            st = PMM2DStackHybrid(PX2, PY2, n_superstrate=NSUP,
                                  n_substrate=NSUB, degree=9, n_orders=no,
                                  symmetry=False)
            st.add_layer(DEPTH2, eps_tensor_cell=cell_pillar(48, INPL_LC),
                         slant=(sgn * TX, 0.0))
            st.set_source(WL, theta=th, phi=ph)
            out = st.solve()
            hv = po(out[0], np.atleast_2d(out[1]), np.atleast_2d(out[2]))
            row[f"hyb_s{sgn:+.0f}_n{no}"] = dd(hv, proto_v[6])
    rows.append(row)
    print(f"  {mount:12s} proto self-move M5->M6 "
          f"{row['proto_selfmove_M5_M6']:.2e} | "
          + " ".join(f"{k.replace('hyb_', '')} {v:.2e}"
                     for k, v in row.items() if k.startswith("hyb_")))
res["T7c_inplane_pillar"] = rows

res["wall_s"] = time.time() - t00
with open(os.path.join(OUT, "m7_slant_x_aniso.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print(f"\nWROTE results/m7_slant_x_aniso.json  ({res['wall_s']:.1f} s)")
