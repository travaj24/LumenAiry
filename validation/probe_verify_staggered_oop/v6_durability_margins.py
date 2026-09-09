"""V6 -- the DURABILITY audit's measurements: every bar in
``tests/unit/test_pmm2d_staggered_oop.py`` re-measured on the TEST FILE'S OWN
fixtures, so the margin quoted in the verification report is a measurement and
not a reading of the test's comment.

Run twice, with ``OPENBLAS_NUM_THREADS`` 1 and 4, to get a cross-kernel
envelope for the bars that sit on a roundoff plateau (the partial substitute
for a second LAPACK the Stage-A verification used).

Also measured here and NOT asserted anywhere in the test file:

  * the min ``Re(lam_f)`` / max ``Re(lam_b)`` of the out-of-plane mode split --
    the PHYSICAL content of "the split is 2q^2 / 2q^2", which the shipped
    ``RuntimeError`` guard cannot express because
    ``_select_forward_flux`` rebalances to exactly ``2N`` unconditionally;
  * the G8b M ladder, which the test's docstring calls "two-sided in M" while
    the test itself runs only M=8.

Usage:  PYTHONPATH=<root> python v6_durability_margins.py <root> <out.json>
"""
import json
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
assert os.path.abspath(lumenairy.__file__).startswith(ROOT), (
    f"lumenairy.__file__ = {lumenairy.__file__} is not under {ROOT}")

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402
from lumenairy.elements.pmm import (  # noqa: E402
    PMM2DStackPure,
    pmm_jones_2d,
)
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _region_modes,
    _region_modes_oop,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import rcwa_jones_2d  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402
from lumenairy.elements.rcwa.oned import rcwa_jones_1d_segments  # noqa: E402

OUT = sys.argv[2]

# ---- the TEST FILE's fixtures, restated here so the margins are measured on
# ---- exactly the cells the bars were derived from
_WL, _P, _DEP, _PU, _DEPU = 1.0e-6, 1.2e-6, 0.4e-6, 0.9e-6, 0.35e-6
_NSUB, _NSUP = 1.5, 1.0
_OOP = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
_OOP_LOSSY = uniaxial_tensor(1.5 + 0.02j, 1.7 + 0.02j, np.deg2rad(35.0),
                             phi=np.deg2rad(25.0))
_NONREC = np.array(_OOP, dtype=complex)
_NONREC[0, 2] = _OOP[0, 2] + 0.22j
_NONREC[2, 0] = np.conj(_NONREC[0, 2])
_LC = np.array(uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55), dtype=complex)
_LC[0, 2] = _LC[1, 2] = _LC[2, 0] = _LC[2, 1] = 0.0
_GYRO = np.array([[2.25, 0.5j, 0.0], [-0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                 dtype=complex)
_ISO = np.eye(3, dtype=complex)
_CONICAL = (np.deg2rad(25.0), np.deg2rad(40.0))


def _cell(host, pillar, n=2):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:, :] = host
    c[0, 0] = pillar
    return c


def _uniform(t33, n=2):
    return np.broadcast_to(t33, (n, n, 3, 3)).copy()


def _lcell(pillar, host=_ISO):
    c = np.empty((3, 3, 3, 3), dtype=complex)
    c[:, :] = host
    for i, j in ((0, 0), (1, 0), (0, 1)):
        c[i, j] = pillar
    return c


def _solve(ec, M, theta=0.0, phi=0.0, n_orders=3, px=_P, dep=_DEP):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pmm_jones_2d_staggered(px, px, ec, _NSUB, _NSUP, dep, _WL,
                                      degree=M, n_orders=n_orders,
                                      theta=theta, phi=phi)


def _per_order(o_a, R_a, T_a, o_b, R_b, T_b):
    idx = {tuple(int(v) for v in r): j for j, r in enumerate(np.asarray(o_b))}
    dR = dT = 0.0
    for i, r in enumerate(np.asarray(o_a)):
        j = idx.get(tuple(int(v) for v in r))
        if j is None:
            continue
        dR = max(dR, float(np.max(np.abs(np.asarray(R_a)[:, i]
                                         - np.asarray(R_b)[:, j]))))
        dT = max(dT, float(np.max(np.abs(np.asarray(T_a)[:, i]
                                         - np.asarray(T_b)[:, j]))))
    return dR, dT


R = {"root": ROOT, "lumenairy": lumenairy.__file__,
     "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")}

# ---- G1 reduction + spectral distance ------------------------------------
g1 = []
for name, host, pillar in (("in-plane uniaxial pillar", _ISO * 4.0, _LC),
                           ("gyrotropic pillar", _ISO * 4.0, _GYRO)):
    for theta, phi in ((0.0, 0.0), _CONICAL):
        ec = _cell(host, pillar)
        orig = TS._tile_is_offplane
        try:
            arm_in = _solve(ec, 6, theta=theta, phi=phi)
            TS._tile_is_offplane = lambda t: True
            arm_oop = _solve(ec, 6, theta=theta, phi=phi)
        finally:
            TS._tile_is_offplane = orig
        k0 = 2.0 * np.pi / _WL
        a0x = _NSUP * np.sin(theta) * np.cos(phi) * k0
        a0y = _NSUP * np.sin(theta) * np.sin(phi) * k0
        sol_in = Granet2DTransverseE(_P, _P, 2, 2, 6, ec, alpha0x=a0x,
                                     alpha0y=a0y, k0=k0)
        try:
            TS._tile_is_offplane = lambda t: True
            sol_oop = Granet2DTransverseE(_P, _P, 2, 2, 6, ec, alpha0x=a0x,
                                          alpha0y=a0y, k0=k0)
            _Wf, _Vf, lf, _Wb, _Vb, lb = _region_modes_oop(sol_oop)
        finally:
            TS._tile_is_offplane = orig
        _W, _V, lam_in, _g2 = _region_modes(sol_in)
        ref = np.concatenate([1j * lam_in, -1j * lam_in])
        got = np.concatenate([1j * lf, 1j * lb])
        g1.append({"cell": name, "theta": theta,
                   "dR": float(np.max(np.abs(arm_in[1] - arm_oop[1]))),
                   "dT": float(np.max(np.abs(arm_in[2] - arm_oop[2]))),
                   "dJones": float(np.max(np.abs(arm_in[3] - arm_oop[3]))),
                   "spectral": float(np.max(np.min(
                       np.abs(got[:, None] - ref[None, :]), axis=1)))})
        print(f"[G1] {name} th={theta:.3f} dR={g1[-1]['dR']:.3e} "
              f"dJ={g1[-1]['dJones']:.3e} spec={g1[-1]['spectral']:.3e}",
              flush=True)
R["G1"] = g1

# ---- G3 Berreman: worst over the nine combinations, and the ladder --------
g3 = []
for tname, t33 in (("lossless", _OOP), ("nonrec", _NONREC),
                   ("lossy", _OOP_LOSSY)):
    for mtag, theta, phi in (("normal", 0.0, 0.0),
                             ("oblique25", np.deg2rad(25.0), 0.0),
                             ("conical25_40", *_CONICAL)):
        Rb, Tb, Jrb, _ = berreman_jones_1d([(t33, _DEPU)], _NSUB, _NSUP, _WL,
                                           angle=theta, phi=phi)
        o, Rr, Tt, J = _solve(_uniform(t33), 8, theta=theta, phi=phi, px=_PU,
                              dep=_DEPU)
        p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        g3.append({"tensor": tname, "mount": mtag,
                   "dR": float(np.max(np.abs(Rr.sum(axis=1) - Rb))),
                   "dT": float(np.max(np.abs(Tt.sum(axis=1) - Tb))),
                   "dJones": float(np.max(np.abs(J - Jrb))),
                   "leak": float(np.max(np.abs(np.delete(Rr, p0, axis=1)))
                                 + np.max(np.abs(np.delete(Tt, p0, axis=1))))})
R["G3"] = g3
print(f"[G3] worst dR/dT/dJones over 9 combos = "
      f"{max(x['dR'] for x in g3):.3e} / {max(x['dT'] for x in g3):.3e} / "
      f"{max(x['dJones'] for x in g3):.3e}; worst leak "
      f"{max(x['leak'] for x in g3):.2e}", flush=True)

theta = np.deg2rad(25.0)
Rb, _Tb, Jrb, _ = berreman_jones_1d([(_OOP, _DEPU)], _NSUB, _NSUP, _WL,
                                    angle=theta, phi=0.0)
lad = []
for M in (5, 6, 8):
    _o, Rr, _T, J = _solve(_uniform(_OOP), M, theta=theta, px=_PU, dep=_DEPU)
    lad.append(max(float(np.max(np.abs(Rr.sum(axis=1) - Rb))),
                   float(np.max(np.abs(J - Jrb)))))
R["G3_ladder"] = lad
print(f"[G3 ladder] M5/M6/M8 = {lad[0]:.3e} / {lad[1]:.3e} / {lad[2]:.3e}",
      flush=True)

# ---- G4 stripe ------------------------------------------------------------
ridge, groove = _OOP, _ISO
ec = np.zeros((2, 2, 3, 3), dtype=complex)
ec[0, :] = ridge
ec[1, :] = groove
g4 = []
for mtag, th in (("normal", 0.0), ("oblique25", np.deg2rad(25.0))):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o_r, R_r, T_r, J_r = rcwa_jones_1d_segments(
            _P, [(0.5, ridge), (0.5, groove)], _NSUB, _NSUP, _DEP, _WL,
            n_orders=31, theta=th)
    ir = {int(m): j for j, m in enumerate(np.asarray(o_r))}
    o_s, R_s, T_s, J_s = _solve(ec, 8, theta=th, phi=0.0)
    o_s = np.asarray(o_s)
    dR = dT = 0.0
    for m in [m for m in range(-3, 4) if m in ir]:
        i = int(np.where((o_s[:, 0] == m) & (o_s[:, 1] == 0))[0][0])
        dR = max(dR, float(np.max(np.abs(R_s[:, i] - R_r[:, ir[m]]))))
        dT = max(dT, float(np.max(np.abs(T_s[:, i] - T_r[:, ir[m]]))))
    sel = o_s[:, 1] != 0
    g4.append({"mount": mtag, "dR": dR, "dT": dT,
               "dJones": float(np.max(np.abs(J_s - J_r))),
               "yleak": float(max(np.max(np.abs(R_s[:, sel])),
                                  np.max(np.abs(T_s[:, sel]))))})
    print(f"[G4] {mtag} dR={dR:.3e} dT={dT:.3e} dJ={g4[-1]['dJones']:.3e} "
          f"yleak={g4[-1]['yleak']:.2e}", flush=True)
R["G4"] = g4

# ---- G5 L cell + no floor -------------------------------------------------
ecl = _lcell(_OOP)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fine = np.repeat(np.repeat(ecl, 11, axis=0), 11, axis=1)
    o_r, R_r, T_r, J_r = rcwa_jones_2d(_P, _P, fine, _NSUB, _NSUP, _DEP, _WL,
                                       n_orders_x=7, n_orders_y=7)
o_s, R_s, T_s, J_s = _solve(ecl, 6)
dR, dT = _per_order(o_s, R_s, T_s, o_r, R_r, T_r)
a = _solve(ecl, 6, n_orders=3)
b = _solve(ecl, 6, n_orders=8)
moved = max(_per_order(a[0], a[1], a[2], b[0], b[1], b[2]))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    h7 = pmm_jones_2d(_P, _P, ecl, _NSUB, _NSUP, _DEP, _WL, degree=7,
                      n_orders=7, stabilize=True)
    h9 = pmm_jones_2d(_P, _P, ecl, _NSUB, _NSUP, _DEP, _WL, degree=7,
                      n_orders=9, stabilize=True)
hyb_moved = max(_per_order(h7[0], h7[1], h7[2], h9[0], h9[1], h9[2]))
dR_h, dT_h = _per_order(a[0], a[1], a[2], h9[0], h9[1], h9[2])
R["G5"] = {"dR_vs_rcwa": dR, "dT_vs_rcwa": dT,
           "dJones_vs_rcwa": float(np.max(np.abs(J_s - J_r))),
           "staggered_moved": moved, "hybrid_moved": hyb_moved,
           "ratio": hyb_moved / moved if moved else None,
           "dR_vs_hybrid": dR_h, "dT_vs_hybrid": dT_h}
print(f"[G5] dR={dR:.3e} dT={dT:.3e} dJ={R['G5']['dJones_vs_rcwa']:.3e} | "
      f"no-floor staggered {moved:.3e} vs hybrid {hyb_moved:.3e} | "
      f"vs hybrid dR={dR_h:.3e} dT={dT_h:.3e}", flush=True)

# ---- G6 closure ladder, split, DECAY SIGNS -------------------------------
g6 = []
cell = _uniform(_OOP)
k0 = 2.0 * np.pi / _WL
for theta, phi in ((0.0, 0.0), _CONICAL):
    clos = []
    for dep_lam in (0.25, 1.0, 3.0):
        _o, Rr, Tt, _J = _solve(cell, 7, theta=theta, phi=phi, px=_P,
                                dep=dep_lam * _WL)
        clos.append(float(np.max(np.abs(Rr.sum(axis=1) + Tt.sum(axis=1)
                                        - 1.0))))
    sol = Granet2DTransverseE(_P, _P, 2, 2, 7, cell,
                              alpha0x=_NSUP * np.sin(theta) * np.cos(phi) * k0,
                              alpha0y=_NSUP * np.sin(theta) * np.sin(phi) * k0,
                              k0=k0)
    _Wf, _Vf, lf, _Wb, _Vb, lb = _region_modes_oop(sol)
    g6.append({"theta": theta, "closure": clos,
               "nf": int(lf.size), "nb": int(lb.size), "q2": sol.q ** 2,
               "growth": float(np.max(np.exp(-np.real(lf) * k0 * 3.0 * _WL))),
               "min_Re_lam_f": float(np.min(np.real(lf))),
               "max_Re_lam_b": float(np.max(np.real(lb)))})
    print(f"[G6] th={theta:.3f} closure={clos} split={lf.size}/{lb.size} "
          f"growth={g6[-1]['growth']:.6e} minRe(lam_f)="
          f"{g6[-1]['min_Re_lam_f']:.3e} maxRe(lam_b)="
          f"{g6[-1]['max_Re_lam_b']:.3e}", flush=True)
R["G6"] = g6

# ---- G7 controls + rotation gauge ----------------------------------------
theta, phi = _CONICAL
Rb, _Tb, Jrb, _ = berreman_jones_1d([(_NONREC, _DEPU)], _NSUB, _NSUP, _WL,
                                    angle=theta, phi=phi)
g7 = []
for control in ("drop", "negate", "transpose"):
    t33 = np.array(_NONREC, dtype=complex)
    if control == "drop":
        t33[0, 2] = t33[1, 2] = t33[2, 0] = t33[2, 1] = 0.0
    elif control == "negate":
        t33[0, 2] *= -1
        t33[1, 2] *= -1
        t33[2, 0] *= -1
        t33[2, 1] *= -1
    else:
        t33[0, 2], t33[2, 0] = _NONREC[2, 0], _NONREC[0, 2]
        t33[1, 2], t33[2, 1] = _NONREC[2, 1], _NONREC[1, 2]
    _o, Rr, _T, J = _solve(_uniform(t33), 6, theta=theta, phi=phi, px=_PU,
                           dep=_DEPU)
    g7.append({"control": control,
               "dR": float(np.max(np.abs(Rr.sum(axis=1) - Rb))),
               "dJones": float(np.max(np.abs(J - Jrb)))})
    print(f"[G7] {control:10s} dR={g7[-1]['dR']:.3e} "
          f"dJ={g7[-1]['dJones']:.3e}", flush=True)
R["G7_controls"] = g7

cellN = _uniform(_NONREC)
orig = TS._OOP_ROT_SIGN
try:
    _o, Rr, _T, J = _solve(cellN, 6, theta=theta, phi=phi, px=_PU, dep=_DEPU)
    good = max(float(np.max(np.abs(Rr.sum(axis=1) - Rb))),
               float(np.max(np.abs(J - Jrb))))
    TS._OOP_ROT_SIGN = -orig
    _o, R2, _T2, J2 = _solve(cellN, 6, theta=theta, phi=phi, px=_PU, dep=_DEPU)
    bad = max(float(np.max(np.abs(R2.sum(axis=1) - Rb))),
              float(np.max(np.abs(J2 - Jrb))))
    _o, R3, _T3, J3 = _solve(cellN, 6, px=_PU, dep=_DEPU)
    TS._OOP_ROT_SIGN = orig
    _o, R4, _T4, J4 = _solve(cellN, 6, px=_PU, dep=_DEPU)
finally:
    TS._OOP_ROT_SIGN = orig
R["G7_rot"] = {"good": good, "bad": bad, "ratio": bad / good,
               "normal_dR": float(np.max(np.abs(R3 - R4))),
               "normal_dJ": float(np.max(np.abs(J3 - J4)))}
print(f"[G7 rot] good={good:.3e} bad={bad:.3e} ratio={bad/good:.3e} "
      f"normal_gap dR={R['G7_rot']['normal_dR']:.3e} "
      f"dJ={R['G7_rot']['normal_dJ']:.3e}", flush=True)

# ---- G8a / G8b (WITH the ladder the test omits) / G8c ---------------------
ec8 = _cell(_ISO, _OOP)
g8a = []
for theta, phi in ((0.0, 0.0), (np.deg2rad(20.0), np.deg2rad(35.0))):
    one = PMM2DStackPure(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                         n_modes=6, n_orders=3)
    one.add_layer(_DEP, eps_cell=ec8).set_source(_WL, theta=theta, phi=phi)
    two = PMM2DStackPure(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                         n_modes=6, n_orders=3)
    two.add_layer(_DEP / 2, eps_cell=ec8).add_layer(_DEP / 2, eps_cell=ec8)
    two.set_source(_WL, theta=theta, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aa = one.solve(jones=True)
        bb = two.solve(jones=True)
    g8a.append({"theta": theta,
                "per_order": max(_per_order(aa[0], aa[1], aa[2],
                                            bb[0], bb[1], bb[2])),
                "dJones": float(np.max(np.abs(aa[3] - bb[3])))})
    print(f"[G8a] th={theta:.3f} per_order={g8a[-1]['per_order']:.3e} "
          f"dJ={g8a[-1]['dJones']:.3e}", flush=True)
R["G8a"] = g8a

layers = [(_OOP, 0.30e-6), (_NONREC, 0.22e-6), (_OOP_LOSSY, 0.17e-6)]
g8b = []
for theta, phi in ((np.deg2rad(25.0), 0.0), _CONICAL):
    Rb2, Tb2, Jrb2, _ = berreman_jones_1d(layers, _NSUB, _NSUP, _WL,
                                          angle=theta, phi=phi)
    for M in (4, 5, 6, 7, 8):
        st = PMM2DStackPure(_PU, _PU, n_superstrate=_NSUP, n_substrate=_NSUB,
                            n_modes=M, n_orders=3)
        for t33, d in layers:
            st.add_layer(d, eps=t33)
        st.set_source(_WL, theta=theta, phi=phi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, Rr, Tt, J = st.solve(jones=True)
        g8b.append({"theta": theta, "M": M,
                    "dR": float(np.max(np.abs(Rr.sum(axis=1) - Rb2))),
                    "dT": float(np.max(np.abs(Tt.sum(axis=1) - Tb2))),
                    "dJones": float(np.max(np.abs(J - Jrb2)))})
        print(f"[G8b] th={theta:.3f} M={M} dR={g8b[-1]['dR']:.3e} "
              f"dJ={g8b[-1]['dJones']:.3e}", flush=True)
R["G8b"] = g8b

ec_o = _cell(_ISO, _OOP_LOSSY)
ec_i = _cell(_ISO, _LC)
res, deficits = [], []
for M in (5, 7):
    st = PMM2DStackPure(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                        n_modes=M, n_orders=3)
    st.add_layer(0.30e-6, eps_cell=ec_o).add_layer(0.20e-6, eps_cell=ec_i)
    st.set_source(_WL, theta=np.deg2rad(20.0), phi=np.deg2rad(35.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, Rr, Tt, _J = st.solve(jones=True, retain_internal=True)
        A = st.layer_absorption()
    deficit = 1.0 - Rr.sum(axis=1) - Tt.sum(axis=1)
    deficits.append(float(np.min(deficit)))
    res.append(float(np.max(np.abs(A.sum(axis=0) - deficit))))
R["G8c"] = {"res": res, "ratio": res[1] / res[0], "deficits": deficits}
print(f"[G8c] res={res} ratio={res[1]/res[0]:.3e} deficits={deficits}",
      flush=True)

st = PMM2DStackPure(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                    n_modes=6, n_orders=3)
st.add_layer(0.25e-6, eps_cell=_cell(_ISO, _OOP))
st.add_layer(0.20e-6, eps_cell=_cell(_ISO, _LC))
st.add_layer(0.15e-6, eps=2.25)
st.set_source(_WL, theta=np.deg2rad(20.0), phi=np.deg2rad(35.0))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _o, Rr, Tt, _J = st.solve(jones=True, retain_internal=True)
    A = st.layer_absorption()
R["G8c_lossless"] = {
    "maxA": float(np.max(np.abs(A))),
    "closure": float(np.max(np.abs(Rr.sum(axis=1) + Tt.sum(axis=1) - 1)))}
print(f"[G8c lossless] maxA={R['G8c_lossless']['maxA']:.3e} "
      f"closure={R['G8c_lossless']['closure']:.3e}", flush=True)

json.dump(R, open(OUT, "w"), indent=1)
print("DONE")
