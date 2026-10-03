"""E2 verifier item 1 -- NO MAP = TODAY'S BYTES (the verifier's OWN fixture set).

SHA-256 of R / T / Jones / absorption outputs (and operator blocks where
cheap) on a fixture set independent of build_e2/e2_1_bytes.py: different
periods, permittivities, wall positions and incidences, plus the shipped
per-layer MORTAR suites' OWN builders (loaded by path from the tree being
measured, so the constructions are the shipped tests' verbatim).

Run in BOTH trees, on Windows AND WSL:

  PRE : LUM_TREE=C:/tmp/vcurved_e2_pre PYTHONPATH=C:/tmp/vcurved_e2_pre python v1_bytes.py
  POST: PYTHONPATH=C:/tmp/lum_vcurved_e2 python v1_bytes.py

Output: v1_bytes_<tree>_<build>.json (sections: ``sha`` = byte fixtures,
``declared`` = fixtures the fold-in is DOCUMENTED to move (rotated Ellipse,
near-grid rectangle claims) -- reported, not counted, ``behaviour`` =
rectangles-only per-layer shape stacks (raise in PRE), ``fb`` = fail-before
(a round-off perturbation must change the hash)).  Compare: v1_compare.py.
"""
import hashlib
import importlib.util
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ve import ROOT, TREE, dump  # noqa: E402

from lumenairy.elements.pmm import (  # noqa: E402
    Circle,
    Ellipse,
    FilletRect,
    PMM2DStackPure,
    PMMStack,
    Rect,
    SinusoidalWall,
    _curvemap as CM,  # noqa: E402
    compile_shapes,
    twod_staggered as TS,  # noqa: E402
)
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

warnings.simplefilter("ignore")
H, DECL, BEH, FB, TIME = {}, {}, {}, {}, {}


def _upd(h, v):
    if v is None:
        h.update(b"None")
    elif isinstance(v, (tuple, list)):
        h.update(f"seq{len(v)}".encode())
        for x in v:
            _upd(h, x)
    elif isinstance(v, dict):
        h.update(f"dict{len(v)}".encode())
        for k in sorted(v, key=repr):
            h.update(repr(k).encode())
            _upd(h, v[k])
    elif isinstance(v, str):
        h.update(v.encode())
    else:
        a = np.ascontiguousarray(np.asarray(v))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())


def sha(*v):
    h = hashlib.sha256()
    _upd(h, list(v))
    return h.hexdigest()


def put(key, fn, store=H):
    """store[key] = sha(fn()) or the hash of the raised exception."""
    t0 = time.perf_counter()
    try:
        out = fn()
        store[key] = sha(out)
    except Exception as e:  # noqa: BLE001
        store[key] = "EXC:" + type(e).__name__ + ":" + str(e)[:300]
    TIME[key] = round(time.perf_counter() - t0, 3)


def ops(name, s):
    for attr in ("Rmat", "Lmat", "Stt", "Schur", "Agen", "Bgen"):
        H[f"{name}.{attr}"] = sha(getattr(s, attr, None))
    for attr in ("Et_blocks", "Et_offdiag", "Ggram_blocks"):
        v = getattr(s, attr, None)
        H[f"{name}.{attr}"] = sha(*(v if v is not None else (None,)))


def solve_abs(st, retain=True):
    out = st.solve(retain_internal=retain)
    return out, st.layer_absorption()


def load_test(name):
    path = os.path.join(ROOT, "tests", "unit", name + ".py")
    spec = importlib.util.spec_from_file_location("_v1_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- constants (OWN, deliberately not the builder's) ----------------------
WL, P = 0.93, 1.07
K0 = 2 * np.pi / WL
EYE = np.eye(3, dtype=complex)
LC_IN = uniaxial_tensor(1.52, 1.74, np.pi / 2, phi=0.83)
LC_OOP = uniaxial_tensor(1.52, 1.74, 0.71, phi=-0.42)
MU_G = np.array([[1.4, 0.3j, 0], [-0.3j, 1.4, 0], [0, 0, 1.1]], complex)


def tcell(host, incl, n=2, at=(0, 0)):
    c = np.empty((n, n, 3, 3), complex)
    c[:] = host
    c[at] = incl
    return c


p3 = np.full((3, 3), 1.21 + 0j)
p3[1, 1] = 5.3
p2 = np.array([[3.1, 1.0], [1.0, 1.44]], complex)
lossy3 = p3.copy()
lossy3[1, 1] = 5.3 + 0.41j

# ---- A. operator blocks ---------------------------------------------------
NWX = np.array([0.0, 0.19, 0.77, P])
NWY = np.array([0.0, 0.33, 0.81, P])
cases = {
    "op_scalar_M4": dict(wx=3, wy=3, M=4, eps=p3),
    "op_scalar_nonuni_M5": dict(wx=NWX, wy=NWY, M=5, eps=p3),
    "op_scalar_obl_M4": dict(wx=3, wy=3, M=4, eps=lossy3, a0=(0.7, -0.5)),
    "op_tensor_in_M4": dict(wx=2, wy=2, M=4, eps=tcell(1.9 * EYE, LC_IN)),
    "op_magnetic_M4": dict(wx=2, wy=2, M=4, eps=p2,
                           mu=np.full((2, 2), 1.2 + 0j)),
    "op_magnetic_tensor_M4": dict(wx=2, wy=2, M=4, eps=p2,
                                  mu=tcell(EYE, MU_G)),
    "op_oop_M4": dict(wx=2, wy=2, M=4, eps=tcell(1.9 * EYE, LC_OOP)),
    "op_slant_M4": dict(wx=2, wy=2, M=4, eps=p2, slant=(0.17, 0.06)),
}
for name, c in cases.items():
    a0 = c.get("a0", (0.0, 0.0))

    def _mk(c=c, a0=a0):
        return TS.Granet2DTransverseE(P, P, c["wx"], c["wy"], c["M"],
                                      c["eps"], alpha0x=a0[0], alpha0y=a0[1],
                                      k0=K0, mu_cell=c.get("mu"),
                                      slant=c.get("slant"))
    s = _mk()
    ops(name, s)
    if not s.offplane:
        put(f"{name}.region_modes", lambda s=s: TS._region_modes(s))
    else:
        put(f"{name}.modes_oop_sym",
            lambda s=s: TS._region_modes_oop(s, symmetry=True))

# grid ops and separable cross-mass factors, oblique Bloch phases
for nm, (wa, wb, M, tx, ty) in {
        "cross_nonuni_vs_int3": (NWX, 3, 5, np.exp(-0.41j), np.exp(0.27j)),
        "cross_int2_vs_int4": (2, 4, 4, 1.0 + 0j, 1.0 + 0j),
        "cross_nonuni_pair": (NWX, np.array([0, 0.5, 0.62, P]), 4,
                              np.exp(0.9j), np.exp(-0.2j))}.items():
    ga = TS.StagGridOps(P, P, wa, wa, M, tx, ty)
    gb = TS.StagGridOps(P, P, wb, wb, M, tx, ty)
    cr = TS.StagCrossOps(ga, gb)
    H[f"{nm}.C"] = sha(*cr.C1, *cr.C2)
    H[f"{nm}.V"] = sha(*ga.V1, *ga.V2, *gb.V1, *gb.V2)

# far projector
for nm, (wx, a0) in {"far_nonuni": (NWX, (0.0, 0.0)),
                     "far_int3_obl": (3, (0.6, -0.3))}.items():
    bx = TS.Basis1D(P, wx, 4, np.exp(-1j * a0[0] * P))
    by = TS.Basis1D(P, wx, 4, np.exp(-1j * a0[1] * P))
    ox = np.arange(-2, 3)
    put(nm, lambda bx=bx, by=by, ox=ox, a0=a0:
        TS._far_projector_2d(bx, by, ox, ox, a0[0], a0[1]))

# ---- B. single-layer drivers ---------------------------------------------
for pol in ("te", "tm"):
    for th, ph in ((0.0, 0.0), (0.21, -0.33)):
        put(f"eff_{pol}_{th}", lambda pol=pol, th=th, ph=ph:
            TS.pmm_efficiency_2d_staggered(P, P, lossy3, 1.5, 1.0, 0.43, WL,
                                           degree=4, n_orders=2,
                                           polarization=pol, theta=th,
                                           phi=ph))
for name, kw in {
        "jones_scalar_conical": dict(eps_cell=p2, theta=0.3, phi=-0.5),
        "jones_tensor_in": dict(eps_cell=tcell(1.9 * EYE, LC_IN)),
        "jones_magnetic": dict(eps_cell=p2, mu_cell=tcell(EYE, MU_G)),
        "jones_oop_auto": dict(eps_cell=tcell(1.9 * EYE, LC_OOP)),
        "jones_oop_nosym": dict(eps_cell=tcell(1.9 * EYE, LC_OOP),
                                symmetry=False),
        "jones_slant": dict(eps_cell=p2, slant=(0.17, 0.06)),
}.items():
    put(name, lambda kw=kw: TS.pmm_jones_2d_staggered(
        P, P, depth=0.37, wavelength=WL, n_substrate=1.5,
        n_superstrate=1.0, degree=4, n_orders=2, **kw))


# ---- C. SHARED stacks -----------------------------------------------------
def sh(**kw):
    a = dict(n_superstrate=1.0, n_substrate=1.5, n_modes=4, n_orders=2)
    a.update(kw)
    return PMM2DStackPure(P, P, **a)


def _shared_multilayer(th, ph):
    st = sh()
    st.add_layer(0.17, eps=2.3)
    st.add_layer(0.29, eps_cell=lossy3)
    st.add_layer(0.12, eps=LC_IN)
    st.set_source(WL, theta=th, phi=ph)
    return solve_abs(st)


put("shared_multilayer_normal", lambda: _shared_multilayer(0.0, 0.0))
put("shared_multilayer_conical", lambda: _shared_multilayer(0.24, 0.61))


def _shared_magnetic():
    st = sh()
    st.add_layer(0.2, eps_cell=p2, mu_cell=np.array([[1.3, 1.0], [1.0, 1.0]],
                                                    complex))
    st.add_layer(0.1, eps=2.0)
    st.set_source(WL, theta=0.15, phi=0.2)
    return st.solve()


def _shared_oop():
    st = sh()
    st.add_layer(0.2, eps_cell=tcell(1.9 * EYE, LC_OOP))
    st.add_layer(0.15, eps_cell=p2)
    st.set_source(WL, theta=0.12, phi=0.4)
    return st.solve()


def _shared_slant():
    st = sh()
    st.add_layer(0.2, eps_cell=p2, slant=(0.17, 0.0))
    st.add_layer(0.2, eps_cell=p2, slant=(0.17, 0.0))
    st.set_source(WL, theta=0.1)
    return st.solve()


put("shared_magnetic", _shared_magnetic)
put("shared_oop", _shared_oop)
put("shared_slant", _shared_slant)


# ---- D. PER-LAYER MORTAR stacks (own) -------------------------------------
def pl(M=4, n_orders=2, **kw):
    a = dict(n_superstrate=1.0, n_substrate=1.5, n_modes=M,
             n_orders=n_orders, layer_grids="per-layer")
    a.update(kw)
    return PMM2DStackPure(P, P, **a)


half = np.array([[4.4, 1.0], [1.0, 1.0]], complex)
third = np.full((3, 3), 1.0 + 0j)
third[1, 1] = 2.6
INC = {"normal": (0.0, 0.0), "oblique": (0.19, 0.0),
       "conical": (0.19, 0.47)}
for nm, (th, ph) in INC.items():
    def _nc(th=th, ph=ph):
        st = pl(5)
        st.add_layer(0.27, eps_cell=half * (1 + 0.05j))
        st.add_layer(0.22, eps_cell=third)
        st.set_source(WL, theta=th, phi=ph)
        return solve_abs(st)
    put(f"pl_nonconf_lossy_{nm}", _nc)

    def _nu(th=th, ph=ph):
        st = pl(4)
        st.add_layer(0.2, eps_cell=half, x_walls=[0.41], y_walls=[0.63])
        st.add_layer(0.1, eps=2.0 + 0.12j)
        st.add_layer(0.2, eps_cell=third, x_walls=[0.27, 0.71],
                     y_walls=[0.36, 0.88])
        st.set_source(WL, theta=th, phi=ph)
        return solve_abs(st)
    put(f"pl_nonuni_uniform_mid_{nm}", _nu)

    def _oop(th=th, ph=ph):
        st = pl(4)
        st.add_layer(0.2, eps_cell=tcell(1.9 * EYE, LC_OOP),
                     x_walls=[0.38], y_walls=[0.52])
        st.add_layer(0.2, eps_cell=third)
        st.set_source(WL, theta=th, phi=ph)
        return solve_abs(st, retain=True)
    put(f"pl_oop_tensor_{nm}", _oop)

    def _sl(th=th, ph=ph):
        st = pl(4)
        st.add_layer(0.15, eps_cell=half, slant=(0.12, 0.05), n_modes=5)
        st.add_layer(0.15, eps_cell=third, slant=(0.12, 0.05), n_modes=4)
        st.set_source(WL, theta=th, phi=ph)
        return st.solve()
    put(f"pl_slant_{nm}", _sl)

    def _mag(th=th, ph=ph):
        st = pl(4)
        st.add_layer(0.2, eps_cell=half,
                     mu_cell=np.array([[1.3, 1.0], [1.0, 1.0]], complex))
        st.add_layer(0.2, eps_cell=third)
        st.set_source(WL, theta=th, phi=ph)
        return solve_abs(st)
    put(f"pl_magnetic_{nm}", _mag)


def _pl_tensor_uniform():
    st = pl(4)
    st.add_layer(0.2, eps_cell=half, x_walls=[0.6], y_walls=[0.35])
    st.add_layer(0.1, eps=LC_IN)
    st.add_layer(0.15, eps_cell=third)
    st.set_source(WL, theta=0.1, phi=0.3)
    return solve_abs(st)


def _pl_forced(th, ph):
    st = pl(4)
    st.add_layer(0.2, eps_cell=half)
    st.add_layer(0.2, eps_cell=half * 1.07)
    st.set_source(WL, theta=th, phi=ph)
    return st._solve_per_layer(jones=True, retain_internal=False,
                               force_mortar=True)


def _pl_taper():
    st = pl(4)
    st.add_tapered_pillar(0.35, eps_pillar=4.2, eps_host=1.0,
                          x_bounds_bottom=(0.25, 0.85),
                          y_bounds_bottom=(0.2, 0.9),
                          x_bounds_top=(0.37, 0.71),
                          y_bounds_top=(0.33, 0.77), n_slices=3)
    st.set_source(WL, theta=0.1, phi=0.2)
    return solve_abs(st)


def _pl_tapers():
    st = pl(4, n_orders=1)
    st.add_tapered_pillars(0.2, eps_host=1.1, n_slices=3, pillars=[
        ((0.3 * P, 0.3 * P), (0.2 * P, 0.2 * P), (0.27 * P, 0.27 * P), 7.0)])
    st.set_source(WL)
    return st.solve()


def _pl_floor():
    st = pl(4)
    st.add_layer(0.2, eps_cell=half)
    st.add_layer(0.2, eps_cell=third)
    st.set_source(WL)
    return st.convergence_floor()


def _pl_conforming_bypass():
    st = pl(4)
    st.add_layer(0.2, eps_cell=third)
    st.add_layer(0.15, eps_cell=third * 1.2)
    st.set_source(WL, theta=0.2, phi=0.1)
    return solve_abs(st)


put("pl_tensor_uniform_mid", _pl_tensor_uniform)
put("pl_forced_conforming_normal", lambda: _pl_forced(0.0, 0.0))
put("pl_forced_conforming_conical", lambda: _pl_forced(0.22, 0.31))
put("pl_taper", _pl_taper)
put("pl_tapered_pillars", _pl_tapers)
put("pl_convergence_floor", _pl_floor)
put("pl_conforming_bypass", _pl_conforming_bypass)


# ---- E. the SHIPPED per-layer mortar suites' own builders -----------------
def _solve_full(st):
    return st.solve(retain_internal=True), st.layer_absorption()


ms = load_test("test_pmm2d_staggered_mortar")
put("shipped.mortar._mortar_stripe(4,6)", lambda: ms._mortar_stripe(4, 6))
put("shipped.mortar._mortar_stripe(6,4,n2)",
    lambda: ms._mortar_stripe(6, 4, n_orders=2))

r2 = load_test("test_fix_pmm2d_mortar_round2")
for nm, st in r2._shipped_geometry_battery().items():
    if any(f"n{n}" in nm for n in (16, 32, 64)):
        continue                                  # cost; covered by capture
    def _b(st=st):
        st.set_source(0.85, theta=0.15, phi=0.25)
        return _solve_full(st)
    put(f"shipped.r2.battery.{nm}", _b)
for d, M in ((0.3, 4), (0.05, 5)):
    put(f"shipped.r2._y_uniform_stack({d},{M})",
        lambda d=d, M=M: _solve_full(r2._y_uniform_stack(d, M)))
put("shipped.r2._mortar_pair(0.3,4)",
    lambda: r2._mortar_pair(0.3, 4)[0])

r3 = load_test("test_fix_pmm2d_mortar_round3")
for kind in ("spacer", "spacer_conf", "pattern", "pattern_union",
             "oop_both"):
    put(f"shipped.r3._mixed({kind},4)",
        lambda kind=kind: _solve_full(r3._mixed(kind, 4)))
put("shipped.r3._mixed(slant_both,4)", lambda: r3._mixed("slant_both", 4).solve())
for fr in (0.3, 0.01):
    put(f"shipped.r3._band_stack({fr})",
        lambda fr=fr: _solve_full(r3._band_stack(fr)))

r4 = load_test("test_fix_pmm2d_mortar_round4")
put("shipped.r4._uniform_xy_differ(0.2)",
    lambda: _solve_full(r4._uniform_xy_differ(0.2, M=4)))
put("shipped.r4._closing_taper(4)",
    lambda: _solve_full(r4._closing_taper(4)))
put("shipped.r4._delta_sweep(0.2)",
    lambda: _solve_full(r4._delta_sweep(0.2)))
put("shipped.r4._mixed(0.3,0.1)", lambda: _solve_full(r4._mixed(0.3, 0.1)))
put("shipped.r4._mixed(0.3,0.1,share_y=False)",
    lambda: _solve_full(r4._mixed(0.3, 0.1, share_y=False)))

v4 = load_test("test_verify_pmm2d_mortar_round4")
for kind in ("Xmort", "Xconf_y", "Xnomort", "Ymort", "Yconf_x"):
    put(f"shipped.v4._stack({kind},wide)",
        lambda kind=kind: _solve_full(v4._stack(kind, v4._WIDE)))

v3 = load_test("test_verify_pmm2d_mortar_round3")


def _v3_stack(kind, M=4):                 # the round-3 gate's nested builder
    st = PMM2DStackPure(v3._P, n_modes=M, n_orders=2, n_substrate=1.45,
                        layer_grids="per-layer")
    st.add_layer(0.118e-6, eps_cell=v3._tensor(v3._WA, *v3._WA),
                 x_walls=v3._sc(v3._WA), y_walls=v3._sc(v3._WA))
    if kind == "both_promoted":
        st.add_layer(0.071e-6, eps_cell=v3._scalar(v3._WB, *v3._WB),
                     x_walls=v3._sc(v3._WB), y_walls=v3._sc(v3._WB))
        st.add_layer(0.083e-6, eps_cell=v3._scalar(v3._WD, *v3._WD, e=4.7),
                     x_walls=v3._sc(v3._WD), y_walls=v3._sc(v3._WD))
    else:
        st.add_layer(0.094e-6, eps_cell=v3._tensor(v3._WB, *v3._WB,
                                                    e=v3._E_OOP2),
                     x_walls=v3._sc(v3._WB), y_walls=v3._sc(v3._WB))
    st.set_source(v3._WL, theta=v3._TH, phi=v3._PH)
    return _solve_full(st)


put("shipped.v3.both_promoted", lambda: _v3_stack("both_promoted"))
put("shipped.v3.neither", lambda: _v3_stack("neither"))

vs = load_test("test_verify_pmm2d_perlayer_slant")


def _vs_split(slant, MA, MB, th):
    st = PMM2DStackPure(vs._P, vs._P, n_superstrate=vs._NSUP,
                        n_substrate=vs._NSUB, n_modes=max(MA, MB), n_orders=3,
                        layer_grids="per-layer")
    st.add_layer(vs._DEP / 2, eps_cell=vs._CELL2, slant=slant, n_modes=MA)
    st.add_layer(vs._DEP / 2, eps_cell=vs._CELL4, slant=slant, n_modes=MB)
    st.set_source(vs._WL, theta=th, phi=0.0)
    return st.solve()


t22 = float(np.tan(np.deg2rad(22.0)))
put("shipped.vslant._split(22deg,6,4,0.19)",
    lambda: _vs_split((t22, 0.0), 6, 4, 0.19))
put("shipped.vslant._split(0,5,5,0.0)",
    lambda: _vs_split((0.0, 0.0), 5, 5, 0.0))


def _vs_forced():
    st = PMM2DStackPure(vs._P, vs._P, n_superstrate=vs._NSUP,
                        n_substrate=vs._NSUB, n_modes=5, n_orders=2,
                        layer_grids="per-layer")
    st.add_layer(vs._DEP / 2, eps_cell=vs._CELL4, slant=(t22, 0.0),
                 n_modes=5)
    st.add_layer(vs._DEP / 2, eps_cell=vs._CELL4, slant=(t22, 0.0),
                 n_modes=5)
    st.set_source(vs._WL, theta=0.19, phi=0.0)
    return st._solve_per_layer(jones=True, retain_internal=False,
                               force_mortar=True)


put("shipped.vslant.forced_mortar", _vs_forced)

# the 1-D per-layer surface (test_pmm_per_layer_grids)
g1 = load_test("test_pmm_per_layer_grids")
for nm, lay in (("common", g1.LAY_COMMON), ("two", g1.LAY_TWO)):
    for grids in ("shared", "per-layer"):
        put(f"shipped.1d.{nm}.{grids}",
            lambda lay=lay, grids=grids: g1._solve(g1._stack(lay, grids)))


def _1d_three_absorb():
    st = PMMStack(1.0e-6, n_substrate=1.5, n_superstrate=1.0, degree=6,
                  far_field_orders=5, layer_grids="per-layer")
    st.add_layer(150e-9, segments=[(0.30, 4.0 + 0.2j), (0.70, 1.0 + 0j)])
    st.add_layer(200e-9, segments=[(0.42, 4.0 + 0j), (0.58, 1.0 + 0j)])
    st.add_layer(120e-9, segments=[(0.55, 2.25 + 0j), (0.45, 1.0 + 0j)])
    st.set_source(700e-9, theta=0.12)
    return st.solve(retain_internal=True), st.layer_absorption()


put("shipped.1d.three_lossy", _1d_three_absorb)


# ---- F. Phase C / D mapped SHARED solves ----------------------------------
def _shape_stack(shapes, th=0.0, ph=0.0, retain=False, bg=1.0, M=4):
    st = sh(n_modes=M)
    st.add_layer(0.41, shapes=shapes, background_eps=bg)
    st.set_source(WL, theta=th, phi=ph)
    if retain:
        return solve_abs(st)
    return st.solve()


for nm, (th, ph) in INC.items():
    put(f"map_circle_{nm}", lambda th=th, ph=ph: _shape_stack(
        [Circle(0.5, 0.55, 0.33, 3.9)], th, ph))
put("map_circle_lossy_retain", lambda: _shape_stack(
    [Circle(0.5, 0.55, 0.33, 3.9 + 0.2j)], 0.1, 0.2, retain=True))
put("map_fillet", lambda: _shape_stack(
    [FilletRect(0.53, 0.53, 0.6, 0.5, 0.09, 4.1)], 0.12, 0.0))
put("map_ellipse_unrotated", lambda: _shape_stack(
    [Ellipse(0.5, 0.5, 0.36, 0.22, 3.3)], 0.0, 0.0))
put("map_sinusoid", lambda: _shape_stack(
    [SinusoidalWall("x", 0.55, 0.09, eps=2.2)], 0.15, 0.0))
put("map_circle_core", lambda: _shape_stack(
    [Circle(0.53, 0.53, 0.35, 3.9, core=0.6)], 0.0, 0.0))
put("shape_rects_shared", lambda: _shape_stack(
    [Rect(0.4, 0.5, 0.3, 0.4, 3.5), Rect(0.75, 0.5, 0.2, 0.6, 2.0)],
    0.17, 0.29, retain=True))
put("shape_rects_compile", lambda: compile_shapes(
    P, P, [Rect(0.4, 0.5, 0.3, 0.4, 3.5)], 1.0)[:3])


def _two_layer_shared_shapes():
    st = sh()
    st.add_layer(0.3, shapes=[Circle(0.5, 0.5, 0.3, 4.0)],
                 background_eps=1.0)
    st.add_layer(0.2, shapes=[Circle(0.5, 0.5, 0.3, 2.25 + 0.1j)],
                 background_eps=1.0)
    st.set_source(WL, theta=0.1, phi=0.0)
    return solve_abs(st)


put("map_two_layer_shared_circles", _two_layer_shared_shapes)

cm3, _w = CM._circle_map_3x3(P, 0.33)


def _explicit_cmap(th, ph):
    st = sh(cmap=cm3)
    st.add_layer(0.4, eps_cell=p3)
    st.add_layer(0.1, eps=2.0)
    st.set_source(WL, theta=th, phi=ph)
    return solve_abs(st)


put("map_explicit_circle_cmap_normal", lambda: _explicit_cmap(0.0, 0.0))
put("map_explicit_circle_cmap_conical", lambda: _explicit_cmap(0.2, 0.5))
s_m = TS.Granet2DTransverseE(P, P, cm3.u_walls, cm3.v_walls, 4, p3, k0=K0,
                             cmap=cm3)
ops("op_mapped_circle", s_m)
cms = CM.SeparableStretch(np.array([0.0, 0.31, 0.83, P]),
                          np.array([0.0, 0.42, 0.77, P]),
                          fy=CM.SineStretch(0.04 * P), period_x=P,
                          period_y=P)


def _stretch():
    st = sh(cmap=cms)
    st.add_layer(0.33, eps_cell=lossy3)
    st.set_source(WL, theta=0.1)
    return solve_abs(st)


put("map_stretch_lossy", _stretch)

# ---- G. DECLARED changes (the Phase C verifier fold-in) -------------------
put("rotated_ellipse_30deg", lambda: _shape_stack(
    [Ellipse(0.5, 0.5, 0.3, 0.2, 3.3, angle=np.deg2rad(30.0))]), DECL)
put("rotated_ellipse_10deg_compile", lambda: compile_shapes(
    P, P, [Ellipse(0.5, 0.5, 0.3, 0.2, 3.3, angle=np.deg2rad(10.0))],
    1.0)[:3], DECL)


# ---- H. BEHAVIOUR: rectangles-only shape stacks under per-layer -----------
def _pl_rects(two=True):
    st = pl(4)
    st.add_layer(0.3, shapes=[Rect(0.4, 0.5, 0.3, 0.4, 3.5)],
                 background_eps=1.0)
    if two:
        st.add_layer(0.2, shapes=[Rect(0.6, 0.55, 0.25, 0.3, 2.0)],
                     background_eps=1.0)
    st.set_source(WL, theta=0.1)
    return solve_abs(st)


def _pl_rects_vs_eps_cell():
    """The same two rectangles as eps_cell + walls (the PRE way)."""
    st = pl(4)
    c1 = np.ones((3, 3), complex)
    c1[1, 1] = 3.5
    st.add_layer(0.3, eps_cell=c1, x_walls=[0.25, 0.55], y_walls=[0.3, 0.7])
    c2 = np.ones((3, 3), complex)
    c2[1, 1] = 2.0
    st.add_layer(0.2, eps_cell=c2, x_walls=[0.475, 0.725],
                 y_walls=[0.4, 0.7])
    st.set_source(WL, theta=0.1)
    return solve_abs(st)


put("pl_rects_only_two_layers", _pl_rects, BEH)
put("pl_rects_only_one_layer", lambda: _pl_rects(False), BEH)
put("pl_rects_as_eps_cell", _pl_rects_vs_eps_cell, BEH)


def _pl_circle():
    st = pl(4)
    st.add_layer(0.3, shapes=[Circle(0.5, 0.5, 0.3, 4.0)],
                 background_eps=1.0)
    st.set_source(WL)
    return st.solve()


put("pl_circle_one_layer", _pl_circle, BEH)

BNUM = {}
try:
    _a, _b = _pl_rects(), _pl_rects_vs_eps_cell()
    BNUM["rects_shapes_vs_eps_cell_maxdRT"] = float(max(
        np.max(np.abs(_a[0][1] - _b[0][1])), np.max(np.abs(_a[0][2]
                                                         - _b[0][2]))))
    BNUM["rects_shapes_vs_eps_cell_maxdA"] = float(np.max(np.abs(
        np.asarray(_a[1]) - np.asarray(_b[1]))))
except Exception as e:  # noqa: BLE001
    BNUM["rects_shapes_vs_eps_cell"] = "EXC:" + type(e).__name__ + str(e)[:200]


def _sh_rects():
    st = sh()
    st.add_layer(0.3, shapes=[Rect(0.4, 0.5, 0.3, 0.4, 3.5)],
                 background_eps=1.0)
    st.add_layer(0.2, shapes=[Rect(0.6, 0.55, 0.25, 0.3, 2.0)],
                 background_eps=1.0)
    st.set_source(WL, theta=0.1)
    return solve_abs(st)


try:
    BNUM["rects_perlayer_shapes_vs_shared_shapes_bytes_equal"] = (
        sha(_pl_rects()) == sha(_sh_rects()))
except Exception as e:  # noqa: BLE001
    BNUM["rects_perlayer_shapes_vs_shared_shapes"] = (
        "EXC:" + type(e).__name__ + str(e)[:200])

# ---- FAIL-BEFORE: round-off perturbations the hash must SEE ---------------
ref = _shared_multilayer(0.24, 0.61)
lossy3_bak = lossy3.copy()
lossy3[1, 1] = np.nextafter(lossy3[1, 1].real, 10.0) + 1j * lossy3[1, 1].imag
pert = _shared_multilayer(0.24, 0.61)
lossy3[:] = lossy3_bak
FB["ulp_eps_shared_multilayer"] = {
    "sha_ref": sha(ref), "sha_pert": sha(pert),
    "max_dR": float(np.max(np.abs(ref[0][1] - pert[0][1]))),
    "max_dT": float(np.max(np.abs(ref[0][2] - pert[0][2])))}


def _nc_eps(e):
    st = pl(5)
    st.add_layer(0.27, eps_cell=e)
    st.add_layer(0.22, eps_cell=third)
    st.set_source(WL, theta=0.19, phi=0.47)
    return solve_abs(st)


e0 = half * (1 + 0.05j)
e1 = e0.copy()
e1[0, 0] = complex(np.nextafter(e1[0, 0].real, 10.0), e1[0, 0].imag)
a, b = _nc_eps(e0), _nc_eps(e1)
FB["ulp_eps_pl_nonconf_conical"] = {
    "sha_ref": sha(a), "sha_pert": sha(b),
    "matches_fixture": sha(a) == H["pl_nonconf_lossy_conical"],
    "max_dR": float(np.max(np.abs(a[0][1] - b[0][1])))}
sI = TS.Granet2DTransverseE(P, P, 3, 3, 4, p3, k0=K0,
                            cmap=CM.IdentityMap(3, 3, P, P))
s0 = TS.Granet2DTransverseE(P, P, 3, 3, 4, p3, k0=K0)
FB["identity_map_quadrature_op"] = {
    "Rmat_equal_hash": sha(sI.Rmat) == sha(s0.Rmat),
    "Lmat_equal_hash": sha(sI.Lmat) == sha(s0.Lmat),
    "Lmat_rel": float(np.max(np.abs(sI.Lmat - s0.Lmat))
                      / np.max(np.abs(s0.Lmat)))}

dump(f"v1_bytes_{TREE}", {"sha": H, "declared": DECL, "behaviour": BEH,
                          "fb": FB, "time": TIME, "beh_num": BNUM})
print(TREE, len(H), "byte keys;", sum(v.startswith("EXC") for v in
                                       H.values()), "are exceptions")
