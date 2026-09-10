"""V4a -- THE NULL TEST.  A shear of a HOMOGENEOUS medium is a coordinate
change, so a uniform layer at ANY slant must reproduce the unslanted answer --
and the residual must be DISCRETIZATION (spectral in ``M``), not a formulation
error that would sit at a fixed level.

Fixtures of this verification's own choosing: an isotropic slab, an IN-PLANE
uniaxial slab, an OUT-OF-PLANE (tilted-director) uniaxial slab, a gyrotropic
slab and a LOSSY out-of-plane slab, at slants of 10 / 35 / 60 degrees (x-only
and diagonal) and at normal / oblique 25 / conical 25-40 incidence.

Observables: per-order R and T, the REFLECTION Jones, the TRANSMISSION Jones
(which is where the frame-anchor phase lives -- so this doubles as an
independent check that the phase is exactly right on a null), and the leak into
the non-(0,0) orders (a uniform layer must diffract nothing).

The M-ladder (4..8) on the worst row is the "is it discretization?" arm.
"""
import numpy as np
from _lib import arm, dump, mx  # noqa: I001

from lumenairy.elements.pmm import PMM2DStackPure
from lumenairy.elements.rcwa._core import uniaxial_tensor

WL = 0.68e-6
PX = PY = 1.10e-6
DEP = 0.34e-6
NSUP, NSUB = 1.0, 1.5

TIL = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
INP = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=np.deg2rad(25.0))
GYR = np.array([[2.25, 0.3j, 0.0], [-0.3j, 2.25, 0.0], [0.0, 0.0, 2.25]],
               dtype=complex)
LOSS = uniaxial_tensor(1.5 + 0.05j, 1.7 + 0.02j, np.deg2rad(35.0),
                       phi=np.deg2rad(25.0))
ISO = np.diag([2.25, 2.25, 2.25]).astype(complex)

TENSORS = {"isotropic": ISO, "inplane_uniaxial": INP,
           "oop_uniaxial": TIL, "gyrotropic": GYR, "lossy_oop": LOSS}
T10 = float(np.tan(np.deg2rad(10.0)))
T35 = float(np.tan(np.deg2rad(35.0)))
T60 = float(np.tan(np.deg2rad(60.0)))
SLANTS = {"x10": (T10, 0.0), "x35": (T35, 0.0), "x60": (T60, 0.0),
          "y35": (0.0, T35), "diag35": (T35 / np.sqrt(2), T35 / np.sqrt(2)),
          "diag60": (T60 / np.sqrt(2), T60 / np.sqrt(2))}
MOUNTS = {"normal": (0.0, 0.0), "oblique25": (np.deg2rad(25.0), 0.0),
          "conical25_40": (np.deg2rad(25.0), np.deg2rad(40.0))}


def run(tens, sl, theta, phi, M=5, nord=3):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=nord)
    st.add_layer(DEP, eps=np.asarray(tens, dtype=complex), slant=sl)
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T, J = st.solve(jones=True)
    o = np.asarray(o)
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    off = [i for i in range(len(o)) if i != p0]
    return dict(R=R, T=T, J=J, JT=st.jones_transmission(),
                leak=float(max(np.max(np.abs(R[:, off])),
                               np.max(np.abs(T[:, off])))))


def main():
    out = {"rows": {}, "ladder": {}, "worst": {}}
    worst = (0.0, None)
    for tname, tens in TENSORS.items():
        for mname, (th, ph) in MOUNTS.items():
            ref = run(tens, None, th, ph)
            for sname, sv in SLANTS.items():
                a = run(tens, sv, th, ph)
                row = dict(dR=mx(a["R"], ref["R"]), dT=mx(a["T"], ref["T"]),
                           dJones=mx(a["J"], ref["J"]),
                           dJonesT=mx(a["JT"], ref["JT"]),
                           leak=a["leak"], leak_ref=ref["leak"])
                key = f"{tname}/{sname}/{mname}"
                out["rows"][key] = row
                w = max(row["dR"], row["dT"], row["dJones"], row["dJonesT"])
                if w > worst[0]:
                    worst = (w, (tname, sv, th, ph, key))
    out["worst"] = dict(value=worst[0], key=worst[1][4])
    print(f"worst null row: {worst[1][4]} = {worst[0]:.3e}")
    # the M-ladder on the worst row
    tname, sv, th, ph, key = worst[1]
    tens = TENSORS[tname]
    for M in (4, 5, 6, 7, 8):
        ref = run(tens, None, th, ph, M=M)
        a = run(tens, sv, th, ph, M=M)
        out["ladder"][M] = dict(dR=mx(a["R"], ref["R"]),
                                dT=mx(a["T"], ref["T"]),
                                dJones=mx(a["J"], ref["J"]),
                                dJonesT=mx(a["JT"], ref["JT"]))
        print(f"  M={M}: dR {out['ladder'][M]['dR']:.3e}  dT "
              f"{out['ladder'][M]['dT']:.3e}  dJ "
              f"{out['ladder'][M]['dJones']:.3e}  dJT "
              f"{out['ladder'][M]['dJonesT']:.3e}")
    # per-mount worst, so the normal-incidence (kt = 0) row is visible
    for mname in MOUNTS:
        vals = [max(v["dR"], v["dT"], v["dJones"], v["dJonesT"])
                for k, v in out["rows"].items() if k.endswith(mname)]
        lk = [v["leak"] for k, v in out["rows"].items() if k.endswith(mname)]
        out["worst"][f"worst_{mname}"] = max(vals)
        out["worst"][f"leak_{mname}"] = max(lk)
        print(f"  worst at {mname}: {max(vals):.3e}   leak {max(lk):.3e}")
    dump("v4a_null", out)
    print("arm", arm())


if __name__ == "__main__":
    main()
