"""V1 -- verifier's OWN byte-identity fixture set (C1), run in BOTH trees.

Different periods (non-square 1.1 x 0.9), wavelength 0.95, indices 1.3 / 1.6,
other patterns, angles and layer sequences than the builder's c1_bytes.py.
SHA-256 of operator blocks, region modes, far projectors and every solve
output (efficiency TE/TM, Jones scalar/lossy/tensor/OOP/magnetic/slant,
conical, stacks with tensor / magnetic / OOP / slant / lossy layers and
absorption, the per-layer mortar, tapered pillars).

  PRE : LUM_TREE=C:/tmp/vcc_pre PYTHONPATH=C:/tmp/vcc_pre python v1_bytes.py pre
  POST: PYTHONPATH=C:/tmp/lum_vcurved_c python v1_bytes.py post [see]
  'see' adds one key with eps * (1 + 1e-15), which must differ from its
  partner (the hash sees a round-off change).
"""
import hashlib
import sys
import warnings

import numpy as np
from _vc import TREE, dump

from lumenairy.elements.pmm import PMM2DStackPure, twod_staggered as TS
from lumenairy.elements.rcwa._core import uniaxial_tensor

LABEL = sys.argv[1]
SEE = len(sys.argv) > 2 and sys.argv[2] == "see"
H = {}


def sha(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        if a is None:
            h.update(b"None")
            continue
        if isinstance(a, (tuple, list)):
            h.update(sha(*a).encode())
            continue
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


PX, PY, WL = 1.1, 0.9, 0.95
K0 = 2 * np.pi / WL
EYE = np.eye(3, dtype=complex)
LC = uniaxial_tensor(1.45, 1.75, np.pi / 2, phi=0.3)          # in-plane
LC_OOP = uniaxial_tensor(1.45, 1.75, 0.8, phi=-0.4)           # out-of-plane
MU_T = np.array([[1.3, 0.2j, 0], [-0.2j, 1.3, 0], [0, 0, 1.1]], complex)
p4 = np.ones((4, 4), complex)
p4[1:3, 1:3] = 3.2 + 0.05j
p4[0, 3] = 1.9
p3 = np.ones((3, 3), complex)
p3[1, 1] = 5.0
p3[2, 0] = 2.4
p2 = np.array([[3.0, 1.0], [1.0, 1.0]], complex)


def tcell(host, incl, n=2, where=((0, 0),)):
    c = np.empty((n, n, 3, 3), complex)
    c[:] = host
    for ij in where:
        c[ij] = incl
    return c


def ops(name, s):
    for attr in ("Rmat", "Lmat", "Stt", "Schur", "Agen", "Bgen"):
        H[f"op.{name}.{attr}"] = sha(getattr(s, attr, None))
    for attr in ("Et_blocks", "Et_offdiag", "Ggram_blocks"):
        v = getattr(s, attr, None)
        H[f"op.{name}.{attr}"] = sha(*(v if v is not None else (None,)))


NWX = np.array([0.0, 0.2, 0.7, 0.95, PX])
NWY = np.array([0.0, 0.15, 0.4, 0.8, PY])
opcases = {
    "s4_M4": dict(wx=4, wy=4, M=4, eps=p4),
    "s4nu_M4": dict(wx=NWX, wy=NWY, M=4, eps=p4),
    "s3_M6_obl": dict(wx=3, wy=3, M=6, eps=p3, a0=(1.3, -0.7)),
    "t2_M4": dict(wx=2, wy=2, M=4, eps=tcell(1.8 * EYE, LC, where=((1, 0),))),
    "oop2_M3": dict(wx=2, wy=2, M=3, eps=tcell(EYE, LC_OOP)),
    "mag2_M4": dict(wx=2, wy=2, M=4, eps=p2,
                    mu=tcell(EYE, MU_T, where=((0, 1),))),
    "slant2_M4": dict(wx=2, wy=2, M=4, eps=p2, slant=(-0.15, 0.25)),
    "slant2_M4_obl": dict(wx=2, wy=2, M=4, eps=p2, slant=(0.1, 0.0),
                          a0=(0.5, 0.2)),
}
for name, c in opcases.items():
    a0 = c.get("a0", (0.0, 0.0))
    s = TS.Granet2DTransverseE(PX, PY, c["wx"], c["wy"], c["M"], c["eps"],
                               alpha0x=a0[0], alpha0y=a0[1], k0=K0,
                               mu_cell=c.get("mu"), slant=c.get("slant"))
    ops(name, s)
    if not s.offplane:
        H[f"modes.{name}"] = sha(*TS._region_modes(s))
    else:
        H[f"modes_oop.{name}"] = sha(*TS._region_modes_oop(s, symmetry=False))

for name, (wx, wy, a0, M) in {
        "far_nu": (NWX, NWY, (0.0, 0.0), 4),
        "far_int_obl": (3, 3, (1.1, 0.6), 5)}.items():
    bx = TS.Basis1D(PX, wx, M, np.exp(-1j * a0[0] * PX))
    by = TS.Basis1D(PY, wy, M, np.exp(-1j * a0[1] * PY))
    o = np.arange(-2, 3)
    H[name] = sha(*TS._far_projector_2d(bx, by, o, o, a0[0], a0[1]))

warnings.simplefilter("ignore")
# ---- efficiency entry --------------------------------------------------------
for pol in ("te", "tm"):
    for th, ph in ((0.0, 0.0), (0.35, 0.0), (0.3, 1.1)):
        r = TS.pmm_efficiency_2d_staggered(
            PX, PY, p3, 1.6, 1.3, 0.42, WL, degree=4, n_orders=2,
            polarization=pol, theta=th, phi=ph)
        H[f"eff.{pol}.{th}.{ph}"] = sha(*r)
for th in (0.0, 0.2):
    r = TS.pmm_efficiency_2d_staggered(PX, PY, p4, 1.6, 1.3, 0.3, WL,
                                       degree=4, n_orders=2, theta=th)
    H[f"eff.lossy4.{th}"] = sha(*r)
# ---- Jones entry -------------------------------------------------------------
jcases = {
    "lossy4": dict(eps_cell=p4),
    "lossy4_conical": dict(eps_cell=p4, theta=0.4, phi=0.9),
    "s3_oblique_x": dict(eps_cell=p3, theta=0.3, phi=0.0),
    "s3_oblique_y": dict(eps_cell=p3, theta=0.3, phi=np.pi / 2),
    "tensor": dict(eps_cell=tcell(1.8 * EYE, LC, where=((1, 0),))),
    "tensor_conical": dict(eps_cell=tcell(1.8 * EYE, LC), theta=0.2, phi=0.5),
    "oop_auto": dict(eps_cell=tcell(EYE, LC_OOP)),
    "oop_nosym": dict(eps_cell=tcell(EYE, LC_OOP), symmetry=False),
    "oop_conical": dict(eps_cell=tcell(EYE, LC_OOP), theta=0.25, phi=0.3),
    "magnetic": dict(eps_cell=p2, mu_cell=tcell(EYE, MU_T, where=((0, 1),))),
    "slant": dict(eps_cell=p2, slant=(-0.15, 0.25)),
    "slant_obl": dict(eps_cell=p2, slant=(0.1, 0.0), theta=0.2),
}
for name, kw in jcases.items():
    out = TS.pmm_jones_2d_staggered(PX, PY, depth=0.37, wavelength=WL,
                                    n_substrate=1.6, n_superstrate=1.3,
                                    degree=4, n_orders=2, **kw)
    H[f"jones.{name}"] = sha(*out)
if SEE:
    out = TS.pmm_jones_2d_staggered(PX, PY, p4 * (1 + 1e-15), 1.6, 1.3, 0.37,
                                    WL, degree=4, n_orders=2)
    H["jones.lossy4@see"] = sha(*out)


# ---- stacks ------------------------------------------------------------------
def stack(name, layers, theta=0.0, phi=0.0, M=4, **kw):
    st = PMM2DStackPure(PX, PY, n_superstrate=1.3, n_substrate=1.6,
                        n_modes=M, n_orders=2, **kw)
    for t, lk in layers:
        st.add_layer(t, **lk)
    st.set_source(WL, theta=theta, phi=phi)
    ret = not any(lk.get("slant") for _t, lk in layers)
    out = st.solve(retain_internal=ret)
    H[f"stack.{name}"] = sha(*out)
    try:
        H[f"stack.{name}.abs"] = sha(st.layer_absorption())
    except Exception as e:                       # noqa: BLE001
        H[f"stack.{name}.abs"] = "raised:" + type(e).__name__


stack("uni_pat_tensor", [(0.12, dict(eps=2.4)), (0.3, dict(eps_cell=p4)),
                         (0.1, dict(eps=LC))], theta=0.2, phi=0.7)
stack("mag_layer", [(0.2, dict(eps_cell=p2, mu=1.4)),
                    (0.15, dict(eps=2.0 + 0.1j))])
stack("mag_tensor_cell", [(0.2, dict(eps_cell=p2,
                                     mu_cell=tcell(EYE, MU_T)))], theta=0.1)
stack("oop_layer", [(0.25, dict(eps_cell=tcell(EYE, LC_OOP))),
                    (0.1, dict(eps=1.9))], M=3)
stack("slant_layer", [(0.3, dict(eps_cell=p2, slant=(0.2, -0.05))),
                      (0.1, dict(eps=2.2))])
stack("lossy3", [(0.2, dict(eps_cell=p3 + 0.2j)), (0.2, dict(eps_cell=p3)),
                 (0.05, dict(eps=1.0))], theta=0.33, phi=0.0)
stack("mortar", [(0.25, dict(eps_cell=np.array([[3.5, 1.0], [1.0, 1.0]]),
                              x_walls=[0.4], y_walls=[0.6])),
                 (0.2, dict(eps_cell=p3)),
                 (0.1, dict(eps=1.7))], layer_grids="per-layer")
stack("mortar_obl", [(0.25, dict(eps_cell=np.array([[3.5, 1.0], [1.0, 1.0]]),
                                  x_walls=[0.4], y_walls=[0.6])),
                     (0.2, dict(eps_cell=p3))], layer_grids="per-layer",
      theta=0.25, phi=0.4)
print(LABEL, TREE, len(H), "hashes")
dump(f"v1_bytes_{LABEL}.json", {"sha": H})
