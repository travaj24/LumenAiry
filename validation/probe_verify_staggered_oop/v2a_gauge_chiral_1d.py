"""V2a -- the ROTATION gauge ``_OOP_ROT_SIGN`` on CHIRAL fixtures the build
did not use, against the two INDEPENDENT 1-D engines.

The build arbitrated the sign on a UNIFORM slab against ``berreman_jones_1d``
(its table T2).  A uniform cell is invariant under the 180-degree rotation
``rho`` the constant compensates, so that fixture can only see ``rho``'s action
on the TENSOR, never its action on the PATTERN.  This probe adds the missing
half: a y-uniform 3-segment stripe whose segment sequence is CHIRAL (its
180-degree image is a different cell), carrying OUT-OF-PLANE tensors, compared
PER ORDER against ``pmm_jones_1d_segments`` and ``rcwa_jones_1d_segments`` --
two engines that both accept full ``(3, 3)`` out-of-plane tensors at planar
incidence and share no code with the staggered path.

Four claims are separated:

  A  shipped (rot = -1) reproduces both 1-D engines at OBLIQUE incidence,
     inside the two oracles' own mutual spread;
  B  rot = +1 fails by decades on the same fixture (fail-before);
  C  on a UNIFORM (rho-symmetric) cell at NORMAL incidence the flip is
     INVISIBLE -- the build's claim, which is why the oblique gate is
     necessary;
  D  on a CHIRAL cell at NORMAL incidence the flip IS visible -- so "invisible
     at normal incidence" is a statement about rho-SYMMETRIC cells, not about
     normal incidence.

``_OOP_H_GAUGE`` is walked over the same fixtures in v2c.

Usage:  PYTHONPATH=<root> python v2a_gauge_chiral_1d.py <root> <out.json>
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
assert os.path.abspath(lumenairy.__file__).startswith(ROOT), (
    f"lumenairy.__file__ = {lumenairy.__file__} is not under {ROOT}")

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402
from lumenairy.elements.pmm import pmm_jones_1d_segments  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import (  # noqa: E402
    rcwa_jones_1d_segments,
    uniaxial_tensor,
)

OUT = sys.argv[2]

# ------------------------------------------------------------------ fixture
tA = uniaxial_tensor(1.48, 1.73, 0.62, phi=0.37)      # out-of-plane
tB = 2.10 * np.eye(3, dtype=complex)                  # isotropic groove
tC = uniaxial_tensor(1.55, 1.80, 1.02, phi=2.20)      # out-of-plane, different
SEGS = [(1.0 / 3.0, tA), (1.0 / 3.0, tB), (1.0 / 3.0, tC)]

tU = uniaxial_tensor(1.52, 1.78, 0.55, phi=0.93)      # uniform out-of-plane


def stripe_cell(ts):
    c = np.zeros((3, 3, 3, 3), dtype=complex)
    for i, t in enumerate(ts):
        c[i, :, :, :] = t
    return c


def uniform_cell(t, n=2):
    c = np.zeros((n, n, 3, 3), dtype=complex)
    c[:] = t
    return c


CHIRAL = stripe_cell([tA, tB, tC])
UNIFORM = uniform_cell(tU, 2)

PX = 1.15e-6
WL = 0.70e-6
DEP = 0.31e-6
NSUB = 1.45
NSUP = 1.0
TH_OB = np.deg2rad(25.0)


def n0(orders):
    o = np.asarray(orders)
    if o.ndim == 1:
        return np.arange(o.size), o.astype(int)
    keep = np.where(o[:, 1] == 0)[0]
    return keep, o[keep, 0].astype(int)


def align(o2, A2, o1, A1):
    k2, m2 = n0(o2)
    k1, m1 = n0(o1)
    map1 = {int(m): j for j, m in zip(k1, m1)}
    d, n = 0.0, 0
    for j, m in zip(k2, m2):
        if int(m) in map1:
            d = max(d, float(np.max(np.abs(np.asarray(A2)[:, j]
                                           - np.asarray(A1)[:, map1[int(m)]]))))
            n += 1
    return d, n


def run_stag(cell, theta, phi, M, n_orders=5):
    return pmm_jones_2d_staggered(PX, PX, cell, NSUB, NSUP, DEP, WL,
                                  degree=M, n_orders=n_orders, theta=theta,
                                  phi=phi)


def compare(stag, orc):
    o2, R2, T2, J2 = stag
    o1, R1, T1, J1 = orc
    dR, n = align(o2, R2, o1, R1)
    dT, _ = align(o2, T2, o1, T1)
    dJ = float(np.max(np.abs(np.asarray(J2) - np.asarray(J1))))
    return {"orders_matched": n, "dR": dR, "dT": dT, "dJones": dJ}


out = {"root": ROOT, "lumenairy": lumenairy.__file__,
       "shipped_rot": TS._OOP_ROT_SIGN,
       "shipped_hgauge": str(TS._OOP_H_GAUGE),
       "px_um": PX * 1e6, "wl_um": WL * 1e6, "depth_um": DEP * 1e6,
       "runs": []}


def save():
    json.dump(out, open(OUT, "w"), indent=1)


# ---- A/B: chiral stripe, both gauge signs ---------------------------------
for theta, tag in ((TH_OB, "oblique25"), (0.0, "normal")):
    orc_p = pmm_jones_1d_segments(PX, SEGS, NSUB, NSUP, DEP, WL, angle=theta,
                                  degree=18, far_field_orders=31,
                                  stabilize=False)
    orc_r = rcwa_jones_1d_segments(PX, SEGS, NSUB, NSUP, DEP, WL, angle=theta,
                                   n_orders=41)
    sp = compare(orc_p, orc_r)
    rec = {"fixture": "chiral_oop_stripe", "mount": tag, "oracle_spread": sp,
           "arms": []}
    print(f"[chiral {tag}] ORACLE MUTUAL SPREAD dR={sp['dR']:.3e} "
          f"dT={sp['dT']:.3e} dJ={sp['dJones']:.3e} n={sp['orders_matched']}",
          flush=True)
    for rot, Ms in ((-1.0, (5, 6, 7, 8)), (+1.0, (7,))):
        TS._OOP_ROT_SIGN = rot
        for M in Ms:
            st = run_stag(CHIRAL, theta, 0.0, M)
            a = {"rot": rot, "M": M,
                 "sumRT": [float(np.sum(st[1][r]) + np.sum(st[2][r]))
                           for r in (0, 1)],
                 "vs_pmm1d": compare(st, orc_p),
                 "vs_rcwa1d": compare(st, orc_r)}
            rec["arms"].append(a)
            print(f"  rot={rot:+.0f} M={M} vs pmm1d dR={a['vs_pmm1d']['dR']:.3e}"
                  f" dT={a['vs_pmm1d']['dT']:.3e} dJ="
                  f"{a['vs_pmm1d']['dJones']:.3e} | vs rcwa1d "
                  f"dR={a['vs_rcwa1d']['dR']:.3e} dT={a['vs_rcwa1d']['dT']:.3e}"
                  f" dJ={a['vs_rcwa1d']['dJones']:.3e} | R+T={a['sumRT'][0]:.9f}",
                  flush=True)
    TS._OOP_ROT_SIGN = -1.0
    out["runs"].append(rec)
    save()

# ---- C: UNIFORM cell -- flip invisible at normal, visible off-normal ------
for theta, phi, tag in ((0.0, 0.0, "normal"),
                        (TH_OB, 0.0, "oblique25"),
                        (TH_OB, np.deg2rad(40.0), "conical25_40")):
    Rb, Tb, Jrb, _Jtb = berreman_jones_1d([(tU, DEP)], NSUB, NSUP, WL,
                                          theta=theta, phi=phi)
    rec = {"fixture": "uniform_oop_slab", "mount": tag, "arms": []}
    Js = {}
    for rot in (-1.0, +1.0):
        TS._OOP_ROT_SIGN = rot
        o2, R2, T2, J2 = run_stag(UNIFORM, theta, phi, 8, n_orders=4)
        k2, m2 = n0(o2)
        i0 = [j for j, m in zip(k2, m2) if m == 0][0]
        oo = np.asarray(o2)
        i00 = int(np.where((oo[:, 0] == 0) & (oo[:, 1] == 0))[0][0])
        Js[rot] = np.asarray(J2)
        a = {"rot": rot,
             "dR_vs_berreman": float(np.max(np.abs(R2[:, i00] - Rb))),
             "dT_vs_berreman": float(np.max(np.abs(T2[:, i00] - Tb))),
             "dJones_vs_berreman": float(np.max(np.abs(np.asarray(J2) - Jrb))),
             "order_leak": float(max(np.max(np.abs(np.delete(R2, i00, axis=1))),
                                     np.max(np.abs(np.delete(T2, i00,
                                                             axis=1)))))}
        rec["arms"].append(a)
        print(f"[uniform {tag}] rot={rot:+.0f} dR={a['dR_vs_berreman']:.3e} "
              f"dT={a['dT_vs_berreman']:.3e} dJ={a['dJones_vs_berreman']:.3e} "
              f"leak={a['order_leak']:.2e}", flush=True)
    TS._OOP_ROT_SIGN = -1.0
    rec["arm_gap_dJones"] = float(np.max(np.abs(Js[-1.0] - Js[+1.0])))
    rec["arms_bit_identical"] = bool(np.array_equal(Js[-1.0], Js[+1.0]))
    print(f"[uniform {tag}] ARM GAP dJones={rec['arm_gap_dJones']:.3e} "
          f"bit-identical={rec['arms_bit_identical']}", flush=True)
    out["runs"].append(rec)
    save()

# ---- D: CHIRAL cell at NORMAL incidence -- is the flip visible? -----------
rec = {"fixture": "chiral_normal_flip_visibility", "arms": []}
Js = {}
for rot in (-1.0, +1.0):
    TS._OOP_ROT_SIGN = rot
    st = run_stag(CHIRAL, 0.0, 0.0, 7)
    Js[rot] = np.asarray(st[3])
    rec["arms"].append({"rot": rot,
                        "R": np.asarray(st[1]).tolist()})
TS._OOP_ROT_SIGN = -1.0
rec["arm_gap_dJones"] = float(np.max(np.abs(Js[-1.0] - Js[+1.0])))
rec["arm_gap_dR"] = float(np.max(np.abs(np.asarray(rec["arms"][0]["R"])
                                        - np.asarray(rec["arms"][1]["R"]))))
rec["arms_bit_identical"] = bool(np.array_equal(Js[-1.0], Js[+1.0]))
print(f"[chiral normal] ARM GAP dJones={rec['arm_gap_dJones']:.3e} "
      f"dR={rec['arm_gap_dR']:.3e} bit-identical={rec['arms_bit_identical']}",
      flush=True)
out["runs"].append(rec)
save()
print("DONE")
