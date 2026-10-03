"""V1 (E3-1, verifier's own set): NO JAX CALL = THE eae470d9 NUMPY BYTES.

    python v1_bytes.py TREE LABEL [block-twin|block-jax]

SHA-256 of operators, kernels, layouts and stack outputs over the verifier's
OWN fixtures (different numbers from build_e3/e1_bytes.py), covering every
class the xp= parametrisation touched: the unmapped / mapped / tensor /
magnetic assembly, the region-mode post-processing, the geometric cache, the
unmapped and the cofactor far field, the incident load and decomposition, the
order kz, the branch selector, the curve classes and every shape primitive's
layout, and the stack / entry outputs (incl. OOP, slant, per-layer,
retain_internal / layer_absorption, which the twin refuses).
block-twin installs an import hook that makes importing the twin module
raise (the MUTANT: if any NumPy path touched it, the run fails or a hash
moves); block-jax blocks every jax import as well.
"""
import hashlib
import importlib.abc
import json
import os
import sys
import warnings

ROOT = os.path.normcase(os.path.abspath(sys.argv[1]))
LABEL = sys.argv[2]
BLOCK = sys.argv[3] if len(sys.argv) > 3 else ""


class _Block(importlib.abc.MetaPathFinder):
    def __init__(self, names):
        self.names = names
        self.hits = []

    def find_spec(self, name, path, target=None):
        for n in self.names:
            if name == n or name.startswith(n + "."):
                self.hits.append(name)
                raise ImportError(f"BLOCKED by v1_bytes mutant: {name}")
        return None


blk = None
if BLOCK == "block-twin":
    blk = _Block(["lumenairy.elements.pmm._jax_twod_staggered"])
elif BLOCK == "block-jax":
    # jax "not installed": importlib.util.find_spec returns None and every
    # import raises; the twin module is blocked as well
    blk = _Block(["lumenairy.elements.pmm._jax_twod_staggered"])
    for _n in ("jax", "jaxlib", "jax.numpy"):
        sys.modules[_n] = None
if blk is not None:
    sys.meta_path.insert(0, blk)

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(ROOT), (
    lumenairy.__file__, ROOT)
from lumenairy.elements.pmm import (  # noqa: E402
    Circle,
    Ellipse,
    FilletRect,
    PMM2DStackPure,
    Rect,
    SinusoidalWall,
    _core as PC,  # noqa: E402
    _curvemap as CM,  # noqa: E402
    compile_shapes,
    pmm_jones_2d_staggered,
    twod_staggered as TS,  # noqa: E402
)
from lumenairy.elements.pmm.shapes2d import _merge  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

warnings.simplefilter("ignore")
P, WL = 1.1, 1.0
H = {}


def h(key, *arrs):
    m = hashlib.sha256()
    for a in arrs:
        if a is None:
            m.update(b"None")
            continue
        a = np.ascontiguousarray(np.asarray(a))
        m.update(str((a.shape, a.dtype.str)).encode())
        m.update(a.tobytes())
    H[key] = m.hexdigest()


def ops(key, sol):
    h(key + ".L", sol.Lmat)
    h(key + ".R", sol.Rmat)
    h(key + ".Stt", getattr(sol, "Stt", None))
    h(key + ".Schur", getattr(sol, "Schur", None))


def modes(key, sol):
    W, V, lam, g2 = TS._region_modes(sol)
    h(key + ".modes", W, V, lam, g2)


def guard(key, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 -- the probe records the outcome
        H[key] = "err:" + type(exc).__name__ + ":" + str(exc)[:60]


k0 = 2 * np.pi / WL
LC = uniaxial_tensor(1.52, 1.74, np.pi / 2, phi=0.31)
GY = np.array([[2.9, 0.35j, 0], [-0.35j, 2.9, 0], [0, 0, 2.4]], complex)
MUG = np.array([[1.3, 0.2j, 0], [-0.2j, 1.3, 0], [0, 0, 1.1]], complex)
OOP = uniaxial_tensor(1.5, 1.7, 0.6, phi=0.2)
wx = np.array([0.0, 0.25, 0.7, P])
wy = np.array([0.0, 0.4, 0.8, P])
cs = np.array([[1.0, 3.2, 1.0], [2.6 + 0.05j, 1.0, 1.3], [1.0, 1.9, 1.0]],
              complex)
a0x, a0y = 0.37, -0.21
M = 3


def A():
    s = TS.Granet2DTransverseE(P, P, wx, wy, M, cs, alpha0x=0.0, alpha0y=0.0,
                               k0=k0)
    ops("A1.scalar_nonuni", s)
    modes("A1.scalar_nonuni", s)
    s = TS.Granet2DTransverseE(P, P, wx, wy, M, cs, alpha0x=a0x,
                               alpha0y=a0y, k0=k0)
    ops("A2.scalar_oblique", s)
    modes("A2.scalar_oblique", s)
    ct = np.broadcast_to(np.eye(3), (3, 3, 3, 3)).astype(complex).copy()
    ct[1, 0] = LC
    ct[0, 1] = GY
    s = TS.Granet2DTransverseE(P, P, wx, wy, M, ct, alpha0x=a0x,
                               alpha0y=a0y, k0=k0)
    ops("A3.tensor_lc_gyro_conical", s)
    modes("A3.tensor_lc_gyro_conical", s)
    mc = np.ones((3, 3), complex)
    mc[1, 0] = 1.7 + 0.02j
    s = TS.Granet2DTransverseE(P, P, wx, wy, M, cs, alpha0x=a0x, alpha0y=0.0,
                               k0=k0, mu_cell=mc)
    ops("A4.magnetic_scalar", s)
    modes("A4.magnetic_scalar", s)
    mt = np.broadcast_to(np.eye(3), (3, 3, 3, 3)).astype(complex).copy()
    mt[2, 1] = MUG
    s = TS.Granet2DTransverseE(P, P, wx, wy, M, ct, alpha0x=0.0, alpha0y=a0y,
                               k0=k0, mu_cell=mt)
    ops("A5.magnetic_tensor", s)
    modes("A5.magnetic_tensor", s)
    geo = TS._homog_geom_cache(TS.Granet2DTransverseE(
        P, P, wx, wy, M, np.full((3, 3), 2.1 + 0j), alpha0x=a0x,
        alpha0y=a0y, k0=k0))
    h("A6.geom_cache_unmapped",
      *[g for g in geo if isinstance(g, np.ndarray)])
    for eps in (1.0, 2.25 + 0.1j):
        h(f"A6.homog_region_modes_{eps}", *TS._homog_region_modes(geo, eps))


SH = {
    "circle": [Circle(0.52, 0.57, 0.31, 3.3)],
    "circle_core": [Circle(0.55, 0.55, 0.42, 3.3, core=0.4)],
    "ellipse_axis": [Ellipse(0.55, 0.5, 0.36, 0.22, 2.7)],
    "ellipse_rot": [Ellipse(0.55, 0.55, 0.3, 0.2, 2.7, angle=0.17)],
    "fillet": [FilletRect(0.55, 0.53, 0.58, 0.44, 0.09, 3.6)],
    "sine": [SinusoidalWall("y", 0.5, 0.07, eps=2.4)],
    "sine_ridge": [SinusoidalWall("x", 0.3, 0.06, eps=2.4, width=0.45,
                                  phase=0.4)],
    "rect_circle": [Rect(0.55, 0.55, 0.9, 0.9, 1.8),
                    Circle(0.55, 0.55, 0.3, 3.3)],
    "two_circles_5x5": [Circle(0.3, 0.3, 0.16, 3.0),
                        Circle(0.8, 0.8, 0.18, 2.2)],
}


def B_one(name, shapes):
    e, xw, yw, cm = compile_shapes(P, P, shapes, 1.0)
    h(f"B.{name}.layout", e, xw, yw)
    if cm is None:
        return
    vals = []
    for sx in range(len(xw) - 1):
        for sy in range(len(yw) - 1):
            U = 0.5 * (cm.u_walls[sx] + cm.u_walls[sx + 1]) + 0.3 * (
                cm.u_walls[sx + 1] - cm.u_walls[sx]) * np.linspace(-1, 1, 5)
            V = 0.5 * (cm.v_walls[sy] + cm.v_walls[sy + 1]) + 0.3 * (
                cm.v_walls[sy + 1] - cm.v_walls[sy]) * np.linspace(-1, 1, 5)
            vals.append(np.stack(cm.geom(sx, sy, U, V)))
    h(f"B.{name}.geom", *vals)
    if name not in ("circle", "fillet", "sine", "ellipse_rot",
                    "two_circles_5x5"):
        return
    s = TS.Granet2DTransverseE(P, P, cm.u_walls, cm.v_walls, M, e,
                               alpha0x=a0x, alpha0y=0.0, k0=k0, cmap=cm)
    ops(f"B.{name}.ops", s)
    modes(f"B.{name}", s)
    bx, by = s.bx, s.by
    ox = np.arange(-2, 3)
    h(f"B.{name}.far_mapped",
      *TS._far_projector_mapped(bx, by, ox, ox, a0x, 0.0, cm))
    h(f"B.{name}.inc_load",
      TS._stag_incident_load_mapped(bx, by, cm, a0x, 0.0, (0.6, 0.8)))
    sh = TS.Granet2DTransverseE(P, P, cm.u_walls, cm.v_walls, M,
                                np.full(np.shape(e)[:2], 1.0 + 0j),
                                alpha0x=a0x, alpha0y=0.0, k0=k0, cmap=cm)
    g = TS._homog_geom_cache(sh)
    h(f"B.{name}.geom_cache", *[x for x in g if isinstance(x, np.ndarray)])
    h(f"B.{name}.inc_coeffs",
      TS._stag_incident_coeffs_mapped(g, bx, by, cm, a0x, 0.0))


def B_tensor_magnetic():
    e, xw, yw, cm = compile_shapes(P, P, [Circle(0.52, 0.57, 0.31, LC)], 1.0)
    s = TS.Granet2DTransverseE(P, P, cm.u_walls, cm.v_walls, M, e,
                               alpha0x=a0x, alpha0y=a0y, k0=k0, cmap=cm)
    ops("B.circle_lc.ops", s)
    modes("B.circle_lc", s)
    r = compile_shapes(P, P, [Ellipse(0.55, 0.5, 0.36, 0.22, 2.7 + 0.03j,
                                      mu=MUG)], 1.0, with_mu=True)
    e, cm, mu = r[0], r[3], r[-1]
    s = TS.Granet2DTransverseE(P, P, cm.u_walls, cm.v_walls, M, e,
                               alpha0x=a0x, alpha0y=a0y, k0=k0, cmap=cm,
                               mu_cell=mu)
    ops("B.ellipse_mag.ops", s)
    modes("B.ellipse_mag", s)


def C():
    rng = np.random.default_rng(7)
    q = rng.normal(size=40) + 1j * rng.normal(size=40) * 1e-3
    q[::7] = -q[::7]
    h("C.forward_branch_flip", PC._forward_branch_flip(q))
    h("C.order_kz", *TS._pmm2d_order_kz(1.0 + 0j, 2.1 + 0.01j,
                                          np.linspace(-2, 2, 9),
                                          np.linspace(-1, 1.5, 9), 0.3, -0.1))
    h("C.inv_lam", TS._inv_lam(np.array([1e-14, 0.3 + 0.1j, -2.0])))
    bx = TS.Basis1D(P, wx, M)
    by = TS.Basis1D(P, wy, M)
    h("C.far_unmapped", *TS._far_projector_2d(bx, by, np.arange(-2, 3),
                                             np.arange(-2, 3), a0x, a0y))
    ss = np.linspace(0, 1, 9)
    for nm, mk in (("arc", lambda: CM.Arc((0.5, 0.5), 0.3, 0.2, 1.4)),
                   ("earc", lambda: CM.EllipseArc((0.5, 0.5), (0.3, 0.2), 0.3,
                                                  1.9, 0.2)),
                   ("line", lambda: CM.Line((0.1, 0.2), (0.7, 0.9)))):
        guard(f"C.curve_{nm}",
              lambda mk=mk, nm=nm: h(f"C.curve_{nm}",
                                     *[np.asarray(v) for v in mk()(ss)]))


def stk(layers, M=3, theta=0.0, phi=0.0, **kw):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.52,
                        n_modes=M, n_orders=2, **kw)
    for L in layers:
        st.add_layer(**L)
    st.set_source(WL, theta=theta, phi=phi)
    return st


def sol(key, st, **kw):
    o, R, T, J = st.solve(**kw)
    h(key + ".R", R)
    h(key + ".T", T)
    h(key + ".J", J)
    return st


cell2 = np.array([[1.0, 2.9 + 0.04j], [3.3, 1.0]], complex)


def D():
    sol("D.multilayer_conical", stk([
        dict(thickness=0.2, eps=2.1),
        dict(thickness=0.3, eps_cell=cell2),
        dict(thickness=0.25, eps=LC)], theta=0.31, phi=0.52))
    st = sol("D.multilayer_absorb", stk([
        dict(thickness=0.2, eps=2.1 + 0.05j),
        dict(thickness=0.3, eps_cell=cell2)], theta=0.2),
        retain_internal=True)
    h("D.multilayer_absorb.layer_abs",
      *[np.asarray(v) for v in np.atleast_1d(st.layer_absorption())])
    for name, shapes in SH.items():
        guard(f"D.shapes_{name}", lambda name=name, shapes=shapes: sol(
            f"D.shapes_{name}", stk([dict(thickness=0.4, shapes=shapes,
                                          background_eps=1.0)],
                                    theta=0.25, phi=0.3)))
    sol("D.shapes_ellipse_mag_lossy_conical", stk([dict(
        thickness=0.4, shapes=[Ellipse(0.55, 0.5, 0.36, 0.22, 2.7 + 0.03j,
                                       mu=MUG)], background_eps=1.0)],
        theta=0.35, phi=0.6))
    sol("D.two_layer_merged", stk([
        dict(thickness=0.3, shapes=[Circle(0.52, 0.57, 0.31, 3.3)],
             background_eps=1.0),
        dict(thickness=0.2, shapes=[Rect(0.52, 0.57, 0.8, 0.76, 2.0)],
             background_eps=1.2)], theta=0.15))
    guard("D.oop_tensor", lambda: sol("D.oop_tensor", stk([
        dict(thickness=0.3, eps=OOP), dict(thickness=0.2, eps_cell=cell2)])))
    cell_oop = np.broadcast_to(np.eye(3), (2, 2, 3, 3)).astype(complex).copy()
    cell_oop[0, 1] = OOP
    guard("D.oop_patterned", lambda: sol("D.oop_patterned", stk(
        [dict(thickness=0.3, eps_cell=cell_oop)])))
    guard("D.slant", lambda: sol("D.slant", stk(
        [dict(thickness=0.3, eps_cell=cell2, slant=(0.2, 0.0))])))
    guard("D.per_layer", lambda: sol("D.per_layer", stk(
        [dict(thickness=0.3, eps_cell=cell2),
         dict(thickness=0.2, eps_cell=np.array([[1.0, 2.0, 1.0], [1.0, 1.0, 1.0],
                                      [2.0, 1.0, 1.0]], complex))],
        layer_grids="per-layer", M=4)))
    o = pmm_jones_2d_staggered(P, P, cell2, 1.52, 1.0, 0.33, WL, degree=3,
                               n_orders=2, theta=0.2, phi=0.4)
    h("D.entry_plain", *o[1:])
    o = pmm_jones_2d_staggered(P, P, None, 1.52, 1.0, 0.33, WL, n_modes=3,
                               n_orders=2, shapes=SH["fillet"],
                               background_eps=1.0)
    h("D.entry_shapes", *o[1:])
    o = pmm_jones_2d_staggered(P, P, cell2, 1.52, 1.0, 0.33, WL, degree=3,
                               n_orders=2, mu_cell=np.array(
                                   [[1.0, 1.4], [1.0, 1.0]], complex))
    h("D.entry_magnetic", *o[1:])
    U, V, cmm, _c, ident, _m = _merge(P, P, [("l1", SH["two_circles_5x5"],
                                              1.0, None)])
    h("D.merge_5x5", U, V)


A()
for name, shapes in SH.items():
    guard("B." + name, lambda name=name, shapes=shapes: B_one(name, shapes))
B_tensor_magnetic()
C()
D()
out = {"keys": len(H), "hashes": H, "label": LABEL, "block": BLOCK,
       "jax_imported": sys.modules.get("jax") is not None,
       "twin_imported": "lumenairy.elements.pmm._jax_twod_staggered"
       in sys.modules,
       "blocked_hits": [] if blk is None else blk.hits}
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, f"v1_bytes_{LABEL}.json"), "w") as f:
    json.dump(out, f, indent=1)
print(LABEL, len(H), out["jax_imported"], out["twin_imported"],
      out["blocked_hits"][:5])
