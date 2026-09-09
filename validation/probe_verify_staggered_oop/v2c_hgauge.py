"""V2c -- the H gauge ``_OOP_H_GAUGE = -1j``, adjudicated against an EXACT
oracle rather than only against the library's own in-plane path.

The build doc says table T4 (the forced in-plane reduction) is "the ONLY gate"
on this constant, and explains that the constant "cancels in a pure
out-of-plane stack and does not cancel in a mixed one".  Both halves are tested
here:

  1. a PURE single out-of-plane uniform layer between isotropic half-spaces vs
     ``berreman_jones_1d`` -- if the gauge really cancelled there, every value
     of the constant would give the same answer;
  2. a MIXED uniform multilayer (out-of-plane + in-plane + isotropic) vs the
     Berreman multilayer -- the exact oracle for the non-cancelling case;
  3. the forced in-plane REDUCTION (the build's own gate), re-measured on fresh
     cells with the dispatch forced by patching ``_tile_is_offplane``;
  4. whether the constant leaks into ``layer_absorption`` (the docstring claims
     only ratios enter, so an overall scale should cancel there).

Usage:  PYTHONPATH=<root> python v2c_hgauge.py <root> <out.json>
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
from lumenairy.elements.pmm import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.rcwa import uniaxial_tensor  # noqa: E402

OUT = sys.argv[2]
PARTS = sys.argv[3] if len(sys.argv) > 3 else "1234"

PX = 0.87e-6
WL = 0.66e-6
NSUB = 1.55
NSUP = 1.0
M = 6

# fresh tensors, none of the build's
T_OOP = uniaxial_tensor(1.47, 1.71, 0.68, phi=1.10)          # out-of-plane
T_IN = uniaxial_tensor(1.53, 1.79, np.pi / 2, phi=0.80)      # IN-plane
T_ISO = 2.31 * np.eye(3, dtype=complex)
T_LOSSY = uniaxial_tensor(1.50 + 0.05j, 1.74 + 0.05j, 0.90, phi=0.25)

GAUGES = [(-1j, "-1j (shipped)"), (+1j, "+1j"), (1.0, "+1"), (-1.0, "-1"),
          (-2j, "-2j (scale x2)")]


def cell_of(t, n=2):
    c = np.zeros((n, n, 3, 3), dtype=complex)
    c[:] = t
    return c


def stack_solve(layers, theta, phi, m=M, n_orders=4, retain=False):
    st = PMM2DStackPure(PX, PX, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=m, n_orders=n_orders)
    for t, e in layers:
        if np.asarray(e).ndim == 2 and np.asarray(e).shape == (3, 3):
            st.add_layer(t, eps=np.asarray(e))
        else:
            st.add_layer(t, eps_cell=e)
    st.set_source(WL, theta=theta, phi=phi)
    res = st.solve(jones=True, retain_internal=retain)
    return st, res


def i00(orders):
    o = np.asarray(orders)
    return int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])


out = {"root": ROOT, "lumenairy": lumenairy.__file__,
       "shipped_hgauge": str(TS._OOP_H_GAUGE), "cases": []}


def save():
    json.dump(out, open(OUT, "w"), indent=1)


# ---- 1/2: PURE and MIXED uniform stacks vs the Berreman multilayer --------
STACKS = [
    ("pure_single_oop", [(0.29e-6, T_OOP)]),
    ("pure_oop_two_layers", [(0.17e-6, T_OOP), (0.13e-6, T_LOSSY)]),
    ("mixed_oop_inplane_iso", [(0.19e-6, T_OOP), (0.15e-6, T_IN),
                               (0.11e-6, T_ISO)]),
]
MOUNTS = [(0.0, 0.0, "normal"), (np.deg2rad(25.0), 0.0, "oblique25"),
          (np.deg2rad(25.0), np.deg2rad(40.0), "conical25_40")]

for sname, layers in (STACKS if "1" in PARTS else []):
    for th, ph, mtag in MOUNTS:
        Rb, Tb, Jrb, _ = berreman_jones_1d([(e, t) for t, e in layers],
                                           NSUB, NSUP, WL, theta=th, phi=ph)
        rec = {"case": sname, "mount": mtag, "arms": []}
        for g, gname in GAUGES:
            TS._OOP_H_GAUGE = g
            _st, (o, R, T, J) = stack_solve(layers, th, ph)
            k = i00(o)
            a = {"gauge": gname,
                 "dR": float(np.max(np.abs(R[:, k] - Rb))),
                 "dT": float(np.max(np.abs(T[:, k] - Tb))),
                 "dJones": float(np.max(np.abs(np.asarray(J) - Jrb))),
                 "sumRT": [float(R[r].sum() + T[r].sum()) for r in (0, 1)]}
            rec["arms"].append(a)
            print(f"[{sname} {mtag}] gauge={gname:16s} dR={a['dR']:.3e} "
                  f"dT={a['dT']:.3e} dJ={a['dJones']:.3e} "
                  f"R+T={a['sumRT'][0]:.9f}", flush=True)
        TS._OOP_H_GAUGE = -1j
        out["cases"].append(rec)
        save()

# ---- 3: the forced IN-PLANE REDUCTION, fresh cells -------------------------
LC_IN = uniaxial_tensor(1.41, 1.83, np.pi / 2, phi=0.53)
GYRO = np.array([[2.71, 0.29j, 0.0], [-0.29j, 2.71, 0.0], [0.0, 0.0, 2.44]],
                dtype=complex)
LOSSY_D = np.diag([3.6 + 0.21j, 4.4 + 0.13j, 3.0]).astype(complex)


def zero_cross(c):
    """EXACTLY zero the four out-of-plane slots (uniaxial_tensor(pi/2) leaves
    ~1e-16 float noise there; the reduction gate needs them at hard zero)."""
    c = c.copy()
    c[..., 0, 2] = 0.0
    c[..., 1, 2] = 0.0
    c[..., 2, 0] = 0.0
    c[..., 2, 1] = 0.0
    return c


RED = []
for nm, t in (("in_plane_lc", LC_IN), ("gyrotropic", GYRO),
              ("lossy_diagonal", LOSSY_D)):
    pil = cell_of(t, 2)
    pil[0, 0] = T_ISO                          # patterned pillar
    RED.append((nm + "_pillar", zero_cross(pil)))
    RED.append((nm + "_uniform", zero_cross(cell_of(t, 2))))

orig_off = TS._tile_is_offplane
for nm, cell in (RED if "3" in PARTS else []):
    assert np.max(np.abs(cell[..., 0, 2])) == 0.0, nm
    assert np.max(np.abs(cell[..., 2, 0])) == 0.0, nm
    assert np.max(np.abs(cell[..., 1, 2])) == 0.0, nm
    assert np.max(np.abs(cell[..., 2, 1])) == 0.0, nm
    for th, ph, mtag in ((0.0, 0.0, "normal"),
                         (np.deg2rad(20.0), np.deg2rad(35.0), "conical20_35")):
        # in-plane arm (shipped dispatch)
        _st, ref = stack_solve([(0.27e-6, cell)], th, ph)
        rec = {"case": "reduction_" + nm, "mount": mtag, "arms": []}
        for g, gname in GAUGES:
            TS._OOP_H_GAUGE = g
            TS._tile_is_offplane = lambda _t: True
            try:
                _st2, alt = stack_solve([(0.27e-6, cell)], th, ph)
            finally:
                TS._tile_is_offplane = orig_off
            a = {"gauge": gname,
                 "dR": float(np.max(np.abs(alt[1] - ref[1]))),
                 "dT": float(np.max(np.abs(alt[2] - ref[2]))),
                 "dJones": float(np.max(np.abs(np.asarray(alt[3])
                                               - np.asarray(ref[3]))))}
            rec["arms"].append(a)
            print(f"[reduction {nm} {mtag}] gauge={gname:16s} "
                  f"dR={a['dR']:.3e} dT={a['dT']:.3e} dJ={a['dJones']:.3e}",
                  flush=True)
        TS._OOP_H_GAUGE = -1j
        out["cases"].append(rec)
        save()

# ---- 4: does the gauge leak into layer_absorption? ------------------------
LOSSY_OOP = uniaxial_tensor(1.52 + 0.07j, 1.78 + 0.07j, 0.70, phi=0.40)
layers = [(0.21e-6, cell_of(LOSSY_OOP, 2)), (0.13e-6, T_IN), (0.09e-6, T_ISO)]
rec = {"case": "layer_absorption_gauge_sweep", "mount": "conical20_35",
       "arms": []}
for g, gname in (GAUGES if "4" in PARTS else []):
    TS._OOP_H_GAUGE = g
    st, (o, R, T, J) = stack_solve(layers, np.deg2rad(20.0), np.deg2rad(35.0),
                                   retain=True)
    A = np.asarray(st.layer_absorption())
    k = i00(o)
    dev = [float(abs(A[:, r].sum() - (1.0 - R[r].sum() - T[r].sum())))
           for r in (0, 1)]
    a = {"gauge": gname, "A": A.tolist(), "budget_dev": dev,
         "sumRT": [float(R[r].sum() + T[r].sum()) for r in (0, 1)]}
    rec["arms"].append(a)
    print(f"[absorption] gauge={gname:16s} A={np.round(A[:, 0], 8).tolist()} "
          f"budget_dev={dev[0]:.3e}", flush=True)
TS._OOP_H_GAUGE = -1j
out["cases"].append(rec)
save()
print("DONE")
