"""B1 (slant-0 BIT-IDENTITY) + B2 (the NULL test) + B3 (the frame-anchor phase).

B1 -- ``slant=None`` vs ``slant=0.0`` vs ``slant=(0,0)``, on the PENCIL and end
to end, by sha256 of the raw bytes.  Same build, two arms.

B2 -- a UNIFORM layer at any slant is a pure coordinate change, so it must
return the unslanted answer.  Reported as an M-ladder (the residual is
DISCRETIZATION and must fall spectrally).

B3 -- the frame-anchor phase ``exp(-i alpha_m . t d)`` on the TRANSMITTED
amplitudes, three arms: none / ``-i`` (shipped) / ``+i``.
"""
import hashlib
import time

import numpy as np
from _lib import tile, uniaxial, write

from lumenairy.elements.pmm import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import (
    Granet2DTransverseE,
    pmm_jones_2d_staggered,
)

PX = PY = 1.10e-6
WL = 0.68e-6
DEP = 0.34e-6
NSUP, NSUB = 1.0, 1.5


def h(*arrs):
    m = hashlib.sha256()
    for a in arrs:
        m.update(np.ascontiguousarray(a).tobytes())
    return m.hexdigest()


TIL = uniaxial(1.5, 1.7, np.deg2rad(35.0), np.deg2rad(25.0))
OOP = tile(np.eye(3, dtype=complex), 2)
OOP[0, 0] = TIL
SCA = np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex)

res = {}
t00 = time.time()

# ------------------------------------------------------------------ B1
print("B1  slant-0 BIT-IDENTITY")
rows = []
for nm, cell in (("scalar", SCA), ("oop", OOP)):
    for a0 in ((0.0, 0.0), (0.4, 0.2)):
        hs = []
        for sl in (None, 0.0, (0.0, 0.0), [0.0, 0.0]):
            s = Granet2DTransverseE(0.9, 0.9, 2, 2, 5, cell, alpha0x=a0[0],
                                    alpha0y=a0[1], k0=2 * np.pi, slant=sl)
            hs.append(h(s.Agen, s.Bgen) if s.offplane else h(s.Rmat, s.Lmat))
        rows.append({"cell": nm, "alpha0": list(a0),
                     "all_equal": len(set(hs)) == 1, "sha": hs[0][:16]})
        print("  pencil %-7s a0=%s  identical=%s"
              % (nm, a0, rows[-1]["all_equal"]))
for nm, cell in (("scalar", SCA), ("oop", OOP)):
    for th, ph in ((0.0, 0.0), (0.3, 0.4)):
        hs = []
        for sl in (None, 0.0, (0.0, 0.0)):
            o, R, T, J = pmm_jones_2d_staggered(
                PX, PY, cell, NSUB, NSUP, DEP, WL, n_modes=4, n_orders=3,
                theta=th, phi=ph, slant=sl)
            hs.append(h(R, T, J))
        rows.append({"cell": "e2e_" + nm, "theta": th, "phi": ph,
                     "all_equal": len(set(hs)) == 1, "sha": hs[0][:16]})
        print("  e2e    %-7s th=%s ph=%s  identical=%s"
              % (nm, th, ph, rows[-1]["all_equal"]))
res["B1"] = rows

# ------------------------------------------------------------------ B2 null
print("")
print("B2  UNIFORM-LAYER NULL (a shear of a homogeneous medium is a no-op)")


def uniform_solve(eps33, sl, M, theta, phi, n_orders=3):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=n_orders)
    if np.ndim(eps33) == 0:
        st.add_layer(DEP, eps=complex(eps33), slant=sl)
    else:
        st.add_layer(DEP, eps=np.asarray(eps33, dtype=complex), slant=sl)
    st.set_source(WL, theta=theta, phi=phi)
    return st.solve(jones=True)


ISO = 2.25
INP = uniaxial(1.5, 1.7, np.pi / 2, np.deg2rad(25.0))
OOPU = TIL
GYR = np.array([[2.25, 0.3j, 0.0], [-0.3j, 2.25, 0.0], [0.0, 0.0, 2.25]],
               dtype=complex)
LOSS = uniaxial(1.5 + 0.05j, 1.7 + 0.02j, np.deg2rad(35.0), np.deg2rad(25.0))
T35 = float(np.tan(np.deg2rad(35.0)))
T10 = float(np.tan(np.deg2rad(10.0)))
SLANTS = {"x10": (T10, 0.0), "x35": (T35, 0.0), "y35": (0.0, T35),
          "diag35": (T35 / np.sqrt(2), T35 / np.sqrt(2))}
MOUNTS = {"normal": (0.0, 0.0), "oblique25": (np.deg2rad(25.0), 0.0),
          "conical": (np.deg2rad(25.0), np.deg2rad(40.0))}
TENSORS = {"iso": ISO, "inplane": INP, "oop": OOPU, "gyro": GYR, "lossy": LOSS}

null_rows = []
M_NULL = 5
for tn, tv in TENSORS.items():
    for mn, (th, ph) in MOUNTS.items():
        o0, R0, T0, J0 = uniform_solve(tv, None, M_NULL, th, ph)
        i0 = int(np.where((o0[:, 0] == 0) & (o0[:, 1] == 0))[0][0])
        for sn, sv in SLANTS.items():
            o1, R1, T1, J1 = uniform_solve(tv, sv, M_NULL, th, ph)
            leak = float(max(np.max(np.abs(np.delete(R1, i0, axis=1))),
                             np.max(np.abs(np.delete(T1, i0, axis=1)))))
            null_rows.append({
                "tensor": tn, "mount": mn, "slant": sn,
                "dR": float(np.max(np.abs(R1 - R0))),
                "dT": float(np.max(np.abs(T1 - T0))),
                "dJr": float(np.max(np.abs(J1 - J0))),
                "leak": leak})
print("  %d rows at M=%d: worst dR %.2e  dT %.2e  dJr %.2e  leak %.2e"
      % (len(null_rows), M_NULL, max(r["dR"] for r in null_rows),
         max(r["dT"] for r in null_rows), max(r["dJr"] for r in null_rows),
         max(r["leak"] for r in null_rows)))
ob = [r for r in null_rows if r["mount"] != "normal"]
nm_ = [r for r in null_rows if r["mount"] == "normal"]
res["B2_rows"] = null_rows
res["B2_worst"] = {"dR": max(r["dR"] for r in null_rows),
                   "dT": max(r["dT"] for r in null_rows),
                   "dJr": max(r["dJr"] for r in null_rows),
                   "leak": max(r["leak"] for r in null_rows),
                   "dR_oblique": max(r["dR"] for r in ob),
                   "dR_normal": max(r["dR"] for r in nm_),
                   "dJr_normal": max(r["dJr"] for r in nm_),
                   "leak_normal": max(r["leak"] for r in nm_)}
print("  normal-only worst dR %.2e dJr %.2e leak %.2e"
      % (res["B2_worst"]["dR_normal"], res["B2_worst"]["dJr_normal"],
         res["B2_worst"]["leak_normal"]))

print("")
print("B2c  the M-LADDER (worst null row: iso / oblique25 / x35)")
lad = []
for M in (4, 5, 6, 7, 8):
    o0, R0, T0, J0 = uniform_solve(ISO, None, M, np.deg2rad(25.0), 0.0)
    o1, R1, T1, J1 = uniform_solve(ISO, (T35, 0.0), M, np.deg2rad(25.0), 0.0)
    lad.append({"M": M, "dim": 4 * (2 * (M - 1)) ** 2,
                "dR": float(np.max(np.abs(R1 - R0))),
                "dT": float(np.max(np.abs(T1 - T0))),
                "dJr": float(np.max(np.abs(J1 - J0)))})
    print("  M=%d dim=%5d  dR %.2e  dT %.2e  dJr %.2e"
          % (M, lad[-1]["dim"], lad[-1]["dR"], lad[-1]["dT"], lad[-1]["dJr"]))
res["B2_ladder"] = lad

# ------------------------------------------------------------------ B3 phase
print("")
print("B3  FRAME-ANCHOR PHASE (transmission Jones, three arms)")


def _tjones(st):
    a = st._modal
    p0 = a["p0"]
    return np.stack([np.stack([a["tx"][c][p0], a["ty"][c][p0]])
                     for c in (0, 1)], axis=1)


def tj(eps33, sl, M, th, ph, arm):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=3)
    if np.ndim(eps33) == 0:
        st.add_layer(DEP, eps=complex(eps33), slant=sl)
    else:
        st.add_layer(DEP, eps=np.asarray(eps33, dtype=complex), slant=sl)
    st.set_source(WL, theta=th, phi=ph)
    st.solve(jones=True)
    Jt = _tjones(st)
    if arm == "shipped":
        return Jt
    a = st._modal
    p0 = a["p0"]
    shx = -sum(L.get("slant", (0.0, 0.0))[0] * L["thickness"]
               for L in st._layers)
    shy = -sum(L.get("slant", (0.0, 0.0))[1] * L["thickness"]
               for L in st._layers)
    k0 = 2.0 * np.pi / a["wavelength"]
    ph0 = np.exp(-1j * k0 * (a["kx"][p0] * shx + a["ky"][p0] * shy))
    if arm == "none":
        return Jt / ph0
    if arm == "plus":
        return Jt / ph0 / ph0
    raise ValueError(arm)


phase_rows = []
for tn, tv in (("iso", ISO), ("oop", OOPU)):
    for mn, (th, ph) in (("oblique25", (np.deg2rad(25.0), 0.0)),
                         ("conical", (np.deg2rad(25.0), np.deg2rad(40.0)))):
        st0 = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                             n_modes=5, n_orders=3)
        if np.ndim(tv) == 0:
            st0.add_layer(DEP, eps=complex(tv))
        else:
            st0.add_layer(DEP, eps=np.asarray(tv, dtype=complex))
        st0.set_source(WL, theta=th, phi=ph)
        st0.solve(jones=True)
        Jt0 = _tjones(st0)
        for sn, sv in (("x10", (T10, 0.0)), ("x35", (T35, 0.0)),
                       ("diag35", (T35 / np.sqrt(2), T35 / np.sqrt(2)))):
            row = {"tensor": tn, "mount": mn, "slant": sn}
            for arm in ("none", "shipped", "plus"):
                row[arm] = float(np.max(np.abs(
                    tj(tv, sv, 5, th, ph, arm) - Jt0)))
            phase_rows.append(row)
            print("  %-4s %-9s %-7s  none %.2e  shipped %.2e  +i %.2e"
                  % (tn, mn, sn, row["none"], row["shipped"], row["plus"]))
res["B3"] = phase_rows
res["B3_worst_shipped"] = max(r["shipped"] for r in phase_rows)
res["B3_best_none"] = min(r["none"] for r in phase_rows)
res["B3_best_plus"] = min(r["plus"] for r in phase_rows)
print("  worst shipped %.2e | best none %.2e | best +i %.2e"
      % (res["B3_worst_shipped"], res["B3_best_none"], res["B3_best_plus"]))

res["wall_s"] = time.time() - t00
write("g1_hash_null", res)
