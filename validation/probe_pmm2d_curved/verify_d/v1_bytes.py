"""V1 -- the verifier's OWN byte-identity fixture set: no map = the bytes of
607d43b0; scalar maps = Phase C's bytes.  SHA-256 of operator blocks and of
every public output, on fixtures chosen independently of the builder's
(different cells, tensors, angles, degrees): every 5.43 / 5.44 tensor and
magnetic class (in-plane tensor, gyrotropic, real-asymmetric, lossy tensor,
scalar mu, gyrotropic mu with a tensor eps, lossy mu), the Li 2003 cell, an
out-of-plane tensor and a slant cell (must take the unmapped path), the
rectangles-only shapes route, multilayer stacks with absorption, and the
mapped SCALAR solves of Phases A-C.

usage (PRE tree = git archive 607d43b0 extracted to C:/tmp/vcd_pre_607d):
  cd /c/tmp/vcd_pre_607d && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/vcd_pre_607d python \
    C:/tmp/lum_vcurved_d/validation/probe_pmm2d_curved/verify_d/v1_bytes.py \
    C:/tmp/vcd_pre_607d pre
  cd /c/tmp/lum_vcurved_d && ... PYTHONPATH=C:/tmp/lum_vcurved_d python \
    validation/probe_pmm2d_curved/verify_d/v1_bytes.py C:/tmp/lum_vcurved_d post
Output v1_bytes_<label>.json.  A 4th arg ``trap`` (post only) makes the
three Phase D kernels RAISE for the whole run: every key must still be
produced and equal (no unmapped or scalar-mapped fixture reaches them)."""
import hashlib
import json
import os
import sys
import warnings

ROOT = os.path.normcase(os.path.abspath(sys.argv[1]))
LABEL = sys.argv[2]
TRAP = len(sys.argv) > 3 and sys.argv[3] == "trap"

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(ROOT), (
    f"lumenairy imported from {lumenairy.__file__}, not {ROOT}")

from lumenairy.elements.pmm import (
    PMM2DStackPure,  # noqa: E402
    _curvemap as CM,  # noqa: E402
    twod_staggered as TS,  # noqa: E402
)
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
H = {}
EXTRA = {}

if TRAP:
    def _boom(*a, **k):
        raise AssertionError("Phase D kernel reached")
    for nm in ("_stag_map_eff_tensor", "_stag_map_weights_tensor",
               "_stag_map_as33"):
        setattr(TS, nm, _boom)


def sha(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        if a is None:
            h.update(b"None")
            continue
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def ops(name, s):
    for attr in ("Rmat", "Lmat", "Stt", "Schur", "Agen", "Bgen"):
        H[f"{name}.{attr}"] = sha(getattr(s, attr, None))
    for attr in ("Et_blocks", "Et_offdiag", "Ggram_blocks"):
        v = getattr(s, attr, None)
        H[f"{name}.{attr}"] = sha(*(v if v is not None else (None,)))


def lc(deg):
    return uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=np.deg2rad(deg))


EYE = np.eye(3, dtype=complex)
BIAX = np.array([[2.3, 0.28, 0], [0.28, 2.9, 0], [0, 0, 2.5]], complex)
GYRO_V = np.array([[2.4, -0.7j, 0], [0.7j, 2.1, 0], [0, 0, 2.2]], complex)
RASYM = np.array([[2.3, 0.45, 0], [-0.15, 2.0, 0], [0, 0, 2.6]], complex)
LOSSY = np.array([[2.2 + 0.3j, 0.2 + 0.05j, 0], [0.2 + 0.05j, 3.0 + 0.1j, 0],
                  [0, 0, 2.4 + 0.2j]], complex)
MU_GYRO = np.array([[1.5, 0.35j, 0], [-0.35j, 1.4, 0], [0, 0, 1.25]], complex)
MU_LOSSY = np.array([[1.5 + 0.2j, 0.1, 0], [0.1, 1.35 + 0.05j, 0],
                     [0, 0, 1.2 + 0.1j]], complex)
OOP = uniaxial_tensor(1.45, 1.75, 0.7, phi=-0.4)
LI_B = np.array([[2.25, -0.5j, 0.0], [0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                dtype=complex)
LI_A = np.conj(LI_B)


def tcell(host, incl, n=2, where=((0, 0),)):
    c = np.empty((n, n, 3, 3), complex)
    c[:] = host
    for ij in where:
        c[ij] = incl
    return c


P, WL = 1.1, 1.0
K0 = 2 * np.pi / WL
rng = np.random.default_rng(7)
sc4 = 1.0 + 2.5 * rng.random((4, 4)) + 0.05j * rng.random((4, 4))
NW = np.array([0.0, 0.2, 0.55, 0.8, P])
NWY = np.array([0.0, 0.35, 0.5, 0.95, P])

cases = {
    "o_scalar4_M4": dict(px=P, py=P, wx=4, wy=4, M=4, eps=sc4),
    "o_scalar_nonuni_M4_obl": dict(px=P, py=0.9, wx=NW, wy=NWY * 0.9 / P,
                                   M=4, eps=sc4, a0=(0.7, -0.5)),
    "o_lc45_M5": dict(px=P, py=P, wx=2, wy=2, M=5,
                      eps=tcell(1.7 * EYE, lc(45))),
    "o_biax3_M4_con": dict(px=P, py=P, wx=3, wy=3, M=4,
                           eps=tcell(EYE, BIAX, 3, ((1, 1), (0, 2))),
                           a0=(0.6, 0.45)),
    "o_li_M6": dict(px=2.4, py=1.4, wx=2, wy=2, M=6, eps=tcell(LI_A, LI_B)),
    "o_gyro_M5": dict(px=P, py=P, wx=2, wy=2, M=5,
                      eps=tcell(2.0 * EYE, GYRO_V)),
    "o_rasym_M4": dict(px=P, py=P, wx=2, wy=2, M=4,
                       eps=tcell(1.5 * EYE, RASYM), a0=(0.3, 0.2)),
    "o_lossyT_M4": dict(px=P, py=P, wx=2, wy=2, M=4,
                        eps=tcell(EYE, LOSSY)),
    "o_mag_scalar_M4": dict(px=P, py=P, wx=2, wy=2, M=4,
                            eps=np.array([[3.0, 1.0], [1.0, 1.0]], complex),
                            mu=np.array([[1.4, 1.0], [1.0, 1.0]], complex)),
    "o_mag_gyro_lc_M4": dict(px=P, py=P, wx=2, wy=2, M=4,
                             eps=tcell(EYE, lc(30)), mu=tcell(EYE, MU_GYRO),
                             a0=(0.4, 0.3)),
    "o_mag_lossy_M4": dict(px=P, py=P, wx=3, wy=3, M=4,
                           eps=np.full((3, 3), 2.0 + 0j),
                           mu=tcell(EYE, MU_LOSSY, 3, ((1, 1),))),
    "o_oop_M4": dict(px=P, py=P, wx=2, wy=2, M=4, eps=tcell(2.0 * EYE, OOP)),
    "o_oop_M3_obl": dict(px=P, py=P, wx=2, wy=2, M=3,
                         eps=tcell(2.0 * EYE, OOP), a0=(0.5, 0.0)),
    "o_slant_M4": dict(px=P, py=P, wx=2, wy=2, M=4,
                       eps=np.array([[3.2, 1.0], [1.0, 1.0]], complex),
                       slant=(0.15, 0.05)),
}
for name, c in cases.items():
    a0 = c.get("a0", (0.0, 0.0))
    s = TS.Granet2DTransverseE(c["px"], c["py"], c["wx"], c["wy"], c["M"],
                               c["eps"], alpha0x=a0[0], alpha0y=a0[1], k0=K0,
                               mu_cell=c.get("mu"), slant=c.get("slant"))
    ops(name, s)
    if not s.offplane:
        H[f"{name}.region_modes"] = sha(*TS._region_modes(s))
    else:
        H[f"{name}.modes_oop"] = sha(*TS._region_modes_oop(s, symmetry=False))

warnings.simplefilter("ignore")
# ---- public entries ---------------------------------------------------------
for pol in ("te", "tm"):
    o, R, T = TS.pmm_efficiency_2d_staggered(
        P, P, sc4, 1.5, 1.0, 0.45, WL, degree=4, n_orders=3,
        polarization=pol, theta=0.3, phi=-0.6)
    H[f"eff_{pol}"] = sha(o, R, T)
J = {
    "j_lc45_con": dict(eps_cell=tcell(1.7 * EYE, lc(45)), theta=0.3, phi=0.5),
    "j_gyro": dict(eps_cell=tcell(2.0 * EYE, GYRO_V), theta=0.2),
    "j_rasym": dict(eps_cell=tcell(1.5 * EYE, RASYM)),
    "j_lossyT_con": dict(eps_cell=tcell(EYE, LOSSY), theta=0.25, phi=1.0),
    "j_mag_gyro": dict(eps_cell=tcell(EYE, lc(30)), mu_cell=tcell(EYE, MU_GYRO),
                       theta=0.2, phi=0.3),
    "j_mag_scalar": dict(eps_cell=np.array([[3.0, 1.0], [1.0, 1.0]], complex),
                         mu_cell=np.array([[1.4, 1.0], [1.0, 1.0]], complex)),
    "j_mag_halfspaces": dict(eps_cell=np.array([[3.0, 1.0], [1.0, 1.0]],
                                               complex),
                             mu_superstrate=1.0, mu_substrate=1.0),
    "j_oop_auto": dict(eps_cell=tcell(2.0 * EYE, OOP)),
    "j_oop_obl": dict(eps_cell=tcell(2.0 * EYE, OOP), theta=0.3),
    "j_slant": dict(eps_cell=np.array([[3.2, 1.0], [1.0, 1.0]], complex),
                    slant=(0.15, 0.05)),
}
for name, kw in J.items():
    out = TS.pmm_jones_2d_staggered(P, P, depth=0.35, wavelength=WL,
                                    n_substrate=1.45, n_superstrate=1.0,
                                    degree=4, n_orders=2, **kw)
    H[name] = sha(*out)
out = TS.pmm_jones_2d_staggered(2.4, 1.4, tcell(LI_A, LI_B), 1.0 + 5.0j, 1.0,
                                1.0, 1.0, degree=6, n_orders=4)
H["j_li2003_M6"] = sha(*out)

# ---- stacks -----------------------------------------------------------------
st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.5, n_modes=4,
                    n_orders=2)
st.add_layer(0.12, eps=BIAX)
st.add_layer(0.25, eps_cell=np.array([[3.0 + 0.2j, 1.0], [1.0, 1.0]]),
             mu_cell=np.array([[1.3, 1.0], [1.0, 1.0]], complex))
st.add_layer(0.1, eps=2.0, mu=MU_LOSSY)
st.add_layer(0.2, eps_cell=tcell(EYE, GYRO_V))
st.set_source(WL, theta=0.2, phi=0.7)
H["stack_tensor_mag"] = sha(*st.solve(retain_internal=True))
H["stack_tensor_mag_absorption"] = sha(st.layer_absorption())

# rectangles-only shapes route (scalar; Phase C byte-identical to unmapped)
from lumenairy.elements.pmm import Rect  # noqa: E402

st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=4,
                    n_orders=2)
st.add_layer(0.4, shapes=[Rect(0.55, 0.55, 0.5, 0.7, 3.0 + 0.1j),
                          Rect(0.3, 0.3, 0.2, 0.2, 1.8)], background_eps=1.0)
st.set_source(WL, theta=0.15)
H["shapes_rects_scalar"] = sha(*st.solve(retain_internal=True))
H["shapes_rects_scalar_absorption"] = sha(st.layer_absorption())

# a TENSOR (and a magnetic) rectangle: PRE has no tensor shapes, so the PRE
# key is the EXPLICIT non-uniform cell; POST records BOTH (the shapes route
# must equal the PRE explicit bytes)
xw = np.array([0.0, 0.3, 0.8, P])
cellT = tcell(EYE, lc(30), 3, ((1, 1),))
cellM = np.ones((3, 3), complex)
cellM[1, 1] = 1.6


def rect_explicit(mag):
    s = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=4,
                       n_orders=2, layer_grids="per-layer")
    if mag:
        s.add_layer(0.4, eps_cell=cellT, mu_cell=cellM, x_walls=xw,
                    y_walls=xw)
    else:
        s.add_layer(0.4, eps_cell=cellT, x_walls=xw, y_walls=xw)
    s.set_source(WL, theta=0.2, phi=0.4)
    return s.solve()


RX = {m: rect_explicit(m) for m in (False, True)}
H["rect_tensor_explicit"] = sha(*RX[False])
H["rect_tensor_mag_explicit"] = sha(*RX[True])
for mag in (False, True):
    if mag and LABEL.endswith("pre"):
        continue                     # PRE has no mu= on a shape
    if True:
        s = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                           n_modes=4, n_orders=2)
        kw = {"mu": 1.6} if mag else {}
        s.add_layer(0.4, shapes=[Rect(0.55, 0.55, 0.5, 0.5, lc(30), **kw)],
                    background_eps=1.0)
        s.set_source(WL, theta=0.2, phi=0.4)
        out = s.solve()
        key = "rect_tensor" + ("_mag" if mag else "") + "@shapes"
        if not mag:
            H[key] = sha(*out)
        EXTRA[key + "_vs_explicit"] = float(max(
            np.max(np.abs(np.asarray(a) - np.asarray(b)))
            for a, b in zip(out[1:], RX[mag][1:])))

# ---- MAPPED SCALAR (Phases A-C bytes) ----------------------------------------
from lumenairy.elements.pmm import Circle, Ellipse  # noqa: E402

cm5, _w = CM._circle_map_5x5(P, 0.33)
eps5 = np.ones((5, 5), complex)
eps5[1:4, 1:4] = 2.8 + 0.05j
s_m = TS.Granet2DTransverseE(P, P, cm5.u_walls, cm5.v_walls, 3, eps5, k0=K0,
                             cmap=cm5, alpha0x=0.3, alpha0y=0.1)
ops("m_circle5_solver", s_m)
for nm, (th, ph) in {"normal": (0.0, 0.0), "conical": (0.25, -0.8)}.items():
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.5, n_modes=4,
                        n_orders=2)
    st.add_layer(0.45, shapes=[Circle(0.5, 0.6, 0.3, 2.6 + 0.02j)],
                 background_eps=1.0)
    st.set_source(WL, theta=th, phi=ph)
    H[f"m_shape_circle_{nm}"] = sha(*st.solve(retain_internal=True))
    H[f"m_shape_circle_{nm}_abs"] = sha(st.layer_absorption())
st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=4,
                    n_orders=2)
st.add_layer(0.3, shapes=[Ellipse(0.55, 0.55, 0.28, 0.18, 3.0, angle=0.3)],
             background_eps=1.0)
st.set_source(WL)
H["m_shape_ellipse"] = sha(*st.solve())
cmv, _w = CM._sine_stripe_map_3x3(P, 0.35, 0.75, 0.06)
epsv = np.ones((3, 3), complex)
epsv[1, :] = 2.5
st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=4,
                    n_orders=2, cmap=cmv)
st.add_layer(0.3, eps_cell=epsv)
st.add_layer(0.2, eps=1.7)
st.set_source(WL, theta=0.1)
H["m_sine_stripe_stack"] = sha(*st.solve(retain_internal=True))
H["m_sine_stripe_stack_abs"] = sha(st.layer_absorption())

env = {"python": sys.version.split()[0], "numpy": np.__version__,
       "lumenairy": lumenairy.__file__, "trap": TRAP,
       "threads": {k: os.environ.get(k) for k in
                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS")}}
with open(os.path.join(HERE, f"v1_bytes_{LABEL}.json"), "w") as f:
    json.dump({"env": env, "sha": H, "extra": EXTRA}, f, indent=1)
print(LABEL, len(H), "hashes", len(EXTRA), "extra")
