"""V1 -- the verifier's OWN byte-identity fixture set for Phase E1: no map
(and every Phase D mapped solve) = the bytes of eae470d9.

SHA-256 of operator blocks, region modes and every public output, on
fixtures chosen independently of the builder's: unmapped out-of-plane
generators (general director, non-reciprocal, lossy gyrotropic; uniform and
non-uniform walls; normal / oblique / conical Bloch phases), the PARITY
reduction on AND off on centro-symmetric out-of-plane cells, slanted scalar /
in-plane-tensor / out-of-plane cells, slanted stacks (global shear, the
frame-anchor phase on the per-order transmitted amplitudes, a slanted uniform
layer), in-plane magnetic cells (scalar, lossy, gyrotropic mu with a tensor
eps), the Jones entry on every one of those, mixed stacks with
``layer_absorption``, per-layer grids with an out-of-plane layer, and the
Phase D MAPPED tensor and magnetic solves (whose permeability blocks were
refactored into ``_chi_R / _chi_Ktz / _chi_Gw``) plus Phase C mapped scalar
ones and the ``_homog_geom_cache`` tuple (now a NamedTuple).

usage:
  cd /c/tmp/vce1_pre_eae4 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/vce1_pre_eae4 python \
    C:/tmp/lum_vcurved_e1/validation/probe_pmm2d_curved/verify_e1/v1_bytes.py \
    C:/tmp/vce1_pre_eae4 pre
  cd /c/tmp/lum_vcurved_e1 && ... python .../v1_bytes.py C:/tmp/lum_vcurved_e1 post
A 4th arg ``trap`` (post only): the three Phase E1 kernels
(``_assemble_oop_general``, ``_stag_map_slant_weights``,
``_stag_scale_weight``) RAISE for the whole run -- every UNMAPPED key must
still be produced and equal; mapped keys of Phase D must too (none of them
is out of plane).  ``chitrap``: ``_chi_R / _chi_Ktz / _chi_Gw`` raise when
called on a MAPPED solver -- every unmapped key must be produced and equal.
Output v1_bytes_<label>.json."""
import hashlib
import json
import os
import sys
import warnings

ROOT = os.path.normcase(os.path.abspath(sys.argv[1]))
LABEL = sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else ""

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(ROOT), (
    f"lumenairy imported from {lumenairy.__file__}, not {ROOT}")

from lumenairy.elements.pmm import (  # noqa: E402
    Circle,
    Ellipse,
    PMM2DStackPure,
    Rect,
    _curvemap as CM,
    twod_staggered as TS,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
H = {}
ERR = {}

if MODE == "trap":
    def _boom(*a, **k):
        raise AssertionError("Phase E1 kernel reached")
    TS.Granet2DTransverseE._assemble_oop_general = _boom
    TS._stag_map_slant_weights = _boom
    TS._stag_scale_weight = _boom
if MODE == "chitrap":
    for nm in ("_chi_R", "_chi_Ktz", "_chi_Gw"):
        orig = getattr(TS.Granet2DTransverseE, nm)

        def wrap(self, *a, _o=orig, **k):
            if self.cmap is not None:
                raise AssertionError("_chi_* reached under a map")
            return _o(self, *a, **k)
        setattr(TS.Granet2DTransverseE, nm, wrap)


def sha(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        if a is None:
            h.update(b"None")
            continue
        if isinstance(a, (tuple, list)):
            h.update(sha(*a).encode())
            continue
        if isinstance(a, dict):
            for k in sorted(a):
                h.update(str(k).encode())
                h.update(sha(a[k]).encode())
            continue
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def guard(name):
    def deco(fn):
        try:
            fn()
        except Exception as exc:          # recorded, never silently dropped
            ERR[name] = f"{type(exc).__name__}: {exc}"[:300]
        return fn
    return deco


def ops(name, s):
    for attr in ("Rmat", "Lmat", "Stt", "Schur", "Agen", "Bgen"):
        H[f"{name}.{attr}"] = sha(getattr(s, attr, None))
    for attr in ("Et_blocks", "Et_offdiag", "Ggram_blocks"):
        v = getattr(s, attr, None)
        H[f"{name}.{attr}"] = sha(*(v if v is not None else (None,)))


def modes(name, s, symmetry=False):
    if s.offplane:
        H[f"{name}.modes"] = sha(*TS._region_modes_oop(s, symmetry=symmetry))
    else:
        H[f"{name}.modes"] = sha(*TS._region_modes(s))


EYE = np.eye(3, dtype=complex)
DIR = uniaxial_tensor(1.52, 1.78, np.deg2rad(52.0), phi=np.deg2rad(37.0))
NR = np.array(DIR, complex)
NR[0, 2] += 0.18j
NR[2, 0] = np.conj(NR[0, 2])
G0 = np.array([[2.3 + 0.06j, 0.35j + 0.03, 0.0],
               [-0.35j + 0.03, 2.1 + 0.05j, 0.0],
               [0.0, 0.0, 2.6 + 0.04j]], complex)
ca, sa = np.cos(0.7), np.sin(0.7)
RY = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
GYL = RY @ G0 @ RY.T
LC = uniaxial_tensor(1.5, 1.75, np.pi / 2, phi=np.deg2rad(33.0))
MUG = np.array([[1.35, -0.25j, 0], [0.25j, 1.5, 0], [0, 0, 1.1]], complex)
MUL = np.array([[1.45 + 0.04j, 0.28j + 0.02, 0],
                [-0.28j + 0.02, 1.3 + 0.03j, 0], [0, 0, 1.15 + 0.02j]],
               complex)
P = 1.05
K0 = 2 * np.pi


def tcell(n, host, incl, where):
    c = np.empty((n, n, 3, 3), complex)
    c[:] = host
    for ij in where:
        c[ij] = incl
    return c


NW = np.array([0.0, 0.18, 0.62, P])
SOLVERS = {
    "oop_dir_2_M4": dict(wx=2, wy=2, M=4, eps=tcell(2, EYE, DIR, [(0, 0)])),
    "oop_nr_nonuni_M4_obl": dict(wx=NW, wy=NW, M=4,
                                 eps=tcell(3, 1.3 * EYE, NR,
                                           [(0, 0), (1, 2)]),
                                 a0=(0.9, 0.0)),
    "oop_gyl_2_M4_con": dict(wx=2, wy=2, M=4,
                             eps=tcell(2, EYE, GYL, [(1, 0)]),
                             a0=(0.5, 0.7)),
    "oop_dir_cs3_M4": dict(wx=3, wy=3, M=4,
                           eps=tcell(3, EYE, DIR, [(1, 1)])),
    "slant_sc_3_M4": dict(wx=3, wy=3, M=4,
                          eps=np.array([[1, 1, 1], [1, 3.2, 1], [1, 1, 1]],
                                       complex), slant=(0.15, 0.07)),
    "slant_lc_2_M4_con": dict(wx=2, wy=2, M=4,
                              eps=tcell(2, EYE, LC, [(0, 1)]),
                              slant=(-0.12, 0.2), a0=(0.4, -0.6)),
    "slant_oop_2_M4": dict(wx=2, wy=2, M=4, eps=tcell(2, EYE, DIR, [(1, 1)]),
                           slant=(0.1, -0.05)),
    "mag_gyro_lc_2_M4_con": dict(wx=2, wy=2, M=4,
                                 eps=tcell(2, EYE, LC, [(0, 0)]),
                                 mu=tcell(2, EYE, MUG, [(0, 0)]),
                                 a0=(0.3, 0.8)),
    "mag_scalar_lossy_3_M4": dict(wx=3, wy=3, M=4,
                                  eps=np.full((3, 3), 2.0 + 0j),
                                  mu=np.array([[1, 1, 1], [1, 1.6 + 0.1j, 1],
                                               [1, 1, 1]], complex)),
    "mag_lossy_tensor_2_M4": dict(wx=2, wy=2, M=4,
                                  eps=np.array([[2.1, 1], [1, 1]], complex),
                                  mu=tcell(2, EYE, MUL, [(0, 0)])),
}


def run_solvers():
    for name, c in SOLVERS.items():
        @guard(name)
        def _():
            a0 = c.get("a0", (0.0, 0.0))
            s = TS.Granet2DTransverseE(P, P, c["wx"], c["wy"], c["M"],
                                       c["eps"], alpha0x=a0[0],
                                       alpha0y=a0[1], k0=K0,
                                       mu_cell=c.get("mu"),
                                       slant=c.get("slant"))
            ops(name, s)
            modes(name, s)
            if s.offplane:
                modes(name + ".sym", s, symmetry=True)
                g = TS._stag_parity_gauge(s)
                H[name + ".parity"] = sha(*(g if g is not None else (None,)))
    # the homogeneous geometric cache (a NamedTuple since E1)

    @guard("homog")
    def _():
        s = TS.Granet2DTransverseE(P, P, 3, 3, 4, np.full((3, 3), 1.0 + 0j),
                                   alpha0x=0.4, alpha0y=0.2, k0=K0)
        g = TS._homog_geom_cache(s)
        H["homog.tuple"] = sha(*tuple(g))
        H["homog.modes"] = sha(*TS._homog_region_modes(g, 2.25 + 0j))


# --------------------------------------------------------------------------- #
# mapped solvers of Phases C / D (must be unchanged)
# --------------------------------------------------------------------------- #
def run_mapped_solvers():
    cmC = CM._circle_map_3x3(P, 0.3, center=(0.47, 0.55))[0]
    w = np.linspace(0.0, P, 4)
    V = np.stack(np.meshgrid(w, w, indexing="ij"), axis=-1)
    V[1, 1] += (0.04, -0.03)
    V[2, 2] += (-0.05, 0.02)
    cmS = CM.TransfiniteMap(w, w, V, None)
    cmT = CM.SeparableStretch(w, w, fx=CM.SineStretch(0.08),
                              fy=CM.SineStretch(-0.05))
    cm5 = CM._circle_map_5x5(P, 0.33)[0]
    cases = {
        "map_c3off_scalar_M4_obl": (cmC, np.array(
            [[1, 1, 1], [1, 3.0, 1], [1, 1, 1]], complex), None, (0.5, 0.0)),
        "map_th_lc_M4": (cmT, tcell(3, EYE, LC, [(1, 1)]), None, (0, 0)),
        "map_sh_lc_mug_M4_con": (cmS, tcell(3, EYE, LC, [(1, 1), (0, 2)]),
                                 tcell(3, EYE, MUG, [(1, 1)]), (0.3, 0.5)),
        "map_c5_mul_M3": (cm5, tcell(5, EYE, 2.2 * EYE,
                                     [(i, j) for i in (1, 2, 3)
                                      for j in (1, 2, 3)]),
                          tcell(5, EYE, MUL, [(2, 2)]), (0, 0)),
    }
    for name, (cm, eps, mu, a0) in cases.items():
        @guard(name)
        def _():
            M = 3 if "M3" in name else 4
            s = TS.Granet2DTransverseE(P, P, cm.u_walls, cm.v_walls, M, eps,
                                       alpha0x=a0[0], alpha0y=a0[1], k0=K0,
                                       mu_cell=mu, cmap=cm)
            ops(name, s)
            modes(name, s)


# --------------------------------------------------------------------------- #
# public entries and stacks
# --------------------------------------------------------------------------- #
def out(name, res, st=None):
    H[name + ".out"] = sha(*res)
    if st is not None:
        md = getattr(st, "_modal", None)
        if md is not None:
            H[name + ".modal"] = sha(*[md[k] for k in ("rx", "ry", "tx",
                                                       "ty")])
        try:
            H[name + ".abs"] = sha(st.layer_absorption())
        except Exception as exc:
            H[name + ".abs"] = f"raise {type(exc).__name__}"


def run_entries():
    J = TS.pmm_jones_2d_staggered
    cells = {
        "j_oop_auto_n": dict(eps=tcell(2, EYE, DIR, [(0, 0)]), kw={}),
        "j_oop_obl": dict(eps=tcell(2, EYE, NR, [(0, 1)]),
                          kw=dict(theta=0.3, phi=0.0)),
        "j_oop_con_sym0": dict(eps=tcell(3, EYE, DIR, [(1, 1)]),
                               kw=dict(theta=0.25, phi=0.9, symmetry=False)),
        "j_oop_cs_sym_on": dict(eps=tcell(3, EYE, DIR, [(1, 1)]),
                                kw=dict(symmetry=True)),
        "j_oop_cs_sym_off": dict(eps=tcell(3, EYE, DIR, [(1, 1)]),
                                 kw=dict(symmetry=False)),
        "j_gyl_con": dict(eps=tcell(2, 1.2 * EYE, GYL, [(1, 1)]),
                          kw=dict(theta=0.2, phi=-0.5)),
        "j_slant_sc_con": dict(eps=np.array([[3.5, 1], [1, 1]], complex),
                               kw=dict(slant=(0.18, -0.06), theta=0.2,
                                       phi=0.4)),
        "j_slant_oop": dict(eps=tcell(2, EYE, DIR, [(0, 0)]),
                            kw=dict(slant=(-0.1, 0.0))),
        "j_mag_lc_mug": dict(eps=tcell(2, EYE, LC, [(0, 0)]),
                             kw=dict(mu_cell=tcell(2, EYE, MUG, [(0, 0)]),
                                     theta=0.15, phi=0.3)),
        "j_mag_scalar": dict(eps=np.array([[2.0, 1], [1, 1]], complex),
                             kw=dict(mu_cell=np.array([[1.5, 1], [1, 1]],
                                                      complex))),
    }
    for name, c in cells.items():
        @guard(name)
        def _():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = J(P, P, c["eps"], 1.45, 1.0, 0.38, 1.0, degree=4,
                      n_orders=2, **c["kw"])
            out(name, r)


def stack(name, build, src=(0.0, 0.0), **kw):
    @guard(name)
    def _():
        st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                            n_modes=4, n_orders=2, **kw)
        build(st)
        st.set_source(1.0, theta=src[0], phi=src[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = st.solve(jones=True)
        out(name, r, st)


def run_stacks():
    oopc = tcell(3, EYE, DIR, [(1, 1)])
    stack("s_oop_mixed_auto", lambda s: (
        s.add_layer(0.2, eps=DIR), s.add_layer(0.3, eps_cell=oopc),
        s.add_layer(0.1, eps=LC)))
    stack("s_oop_mixed_nosym", lambda s: (
        s.add_layer(0.2, eps=DIR), s.add_layer(0.3, eps_cell=oopc),
        s.add_layer(0.1, eps=LC)), symmetry=False)
    stack("s_oop_lossy_con", lambda s: (
        s.add_layer(0.25, eps_cell=tcell(2, EYE, GYL, [(0, 0)])),
        s.add_layer(0.1, eps=2.1 + 0.05j)), src=(0.3, 0.6))
    stack("s_slant_global_con", lambda s: (
        s.add_layer(0.15, eps=1.8, slant=(0.2, 0.1)),
        s.add_layer(0.3, eps_cell=np.array([[3.3, 1], [1, 1]], complex),
                    slant=(0.2, 0.1))), src=(0.25, 0.5))
    stack("s_slant_under_vertical", lambda s: (
        s.add_layer(0.12, eps=1.7),
        s.add_layer(0.35, eps_cell=tcell(2, EYE, DIR, [(1, 0)]),
                    slant=(-0.15, 0.05)),
        s.add_layer(0.1, eps=2.0)), src=(0.2, 0.0))
    stack("s_slant_uniform", lambda s: s.add_layer(0.3, eps=2.4,
                                                   slant=(0.3, -0.2)),
          src=(0.2, 0.3))
    stack("s_mag_mixed_con", lambda s: (
        s.add_layer(0.2, eps_cell=tcell(2, EYE, LC, [(0, 0)]),
                    mu_cell=tcell(2, EYE, MUG, [(0, 0)])),
        s.add_layer(0.1, eps=2.0, mu=MUL)), src=(0.2, 0.7))
    stack("s_perlayer_oop", lambda s: (
        s.add_layer(0.2, eps_cell=tcell(2, EYE, DIR, [(0, 0)]),
                    x_walls=np.array([0, 0.4, P]),
                    y_walls=np.array([0, 0.5, P])),
        s.add_layer(0.2, eps_cell=np.array([[2.5, 1, 1], [1, 1, 1],
                                            [1, 1, 1]], complex),
                    x_walls=np.linspace(0, P, 4),
                    y_walls=np.linspace(0, P, 4))),
          layer_grids="per-layer", src=(0.15, 0.2))
    # shapes: rectangles only (unmapped) with an out-of-plane tensor
    stack("s_shapes_rect_oop", lambda s: s.add_layer(
        0.3, shapes=[Rect(0.4, 0.5, 0.3, 0.4, DIR)], background_eps=1.0),
        src=(0.2, 0.1))
    # Phase D mapped shapes (curved, block-form / magnetic)
    stack("s_shapes_circle_lc_mu", lambda s: s.add_layer(
        0.3, shapes=[Circle(0.5, 0.55, 0.3, LC, mu=MUG)],
        background_eps=1.0), src=(0.2, 0.5))
    stack("s_shapes_ellipse_scalar_con", lambda s: (s.add_layer(
        0.3, shapes=[Ellipse(0.55, 0.5, 0.33, 0.22, 3.0, angle=0.0)],
        background_eps=1.0), s.add_layer(0.1, eps=2.0)), src=(0.3, 0.9))


run_solvers()
run_mapped_solvers()
run_entries()
run_stacks()
res = {"keys": H, "errors": ERR, "n": len(H),
       "lumenairy": lumenairy.__file__, "mode": MODE}
with open(os.path.join(HERE, f"v1_bytes_{LABEL}.json"), "w") as f:
    json.dump(res, f, indent=1)
print(LABEL, "keys", len(H), "errors", len(ERR), list(ERR)[:6])
