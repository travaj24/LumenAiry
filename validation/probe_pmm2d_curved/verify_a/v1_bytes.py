"""V1 -- A1 byte identity on the VERIFIER's own fixture set (no map).

Runs UNCHANGED against the PRE tree (``git archive 47c1c3a0``, no ``cmap``
anywhere) and the Phase-A tip; every key is a SHA-256 of the raw bytes of an
operator attribute, a region-mode set, a far projector, or a public result
(R / T / Jones / absorption).  The operator keys hash EVERY ndarray attribute
a solver instance holds (generic ``vars()`` walk), so nothing is cherry-picked.

usage: VA_ROOT=<tree> PYTHONPATH=<tree> v1_bytes.py <tag>
writes v1_bytes_<tag>_<build>.json
"""
import hashlib
import sys
import warnings

import _vcommon as C
import numpy as np

from lumenairy.elements.pmm import PMM2DStackPure, twod_staggered as TS
from lumenairy.elements.rcwa._core import uniaxial_tensor

TAG = sys.argv[1]
OUT = {}


def h(*arrs):
    m = hashlib.sha256()
    for a in arrs:
        if a is None:
            m.update(b"None")
        else:
            a = np.ascontiguousarray(a)
            m.update(str(a.dtype).encode() + str(a.shape).encode())
            m.update(a.tobytes())
    return m.hexdigest()


def put(key, *arrs):
    assert key not in OUT, key
    OUT[key] = h(*arrs)


def walk_solver(tag, s):
    """Hash every ndarray (or tuple of ndarrays) attribute of a solver."""
    skip = {"cmap", "_mapw", "_qrule", "_qcache"}   # tip-only, None w/o map
    for name, v in sorted(vars(s).items()):
        if name in skip:
            continue
        if isinstance(v, np.ndarray):
            put(f"op/{tag}/{name}", v)
        elif isinstance(v, tuple) and v and all(
                isinstance(t, np.ndarray) or t is None for t in v):
            put(f"op/{tag}/{name}", *v)


P = C.P
K0 = C.K0
WL = C.WL
eps3 = C.cell("pillar")
eps3[0, 2] = 2.1 + 0.05j
lc = np.empty((2, 2, 3, 3), complex)
lc[:] = 2.25 * np.eye(3)
lc[1, 0] = uniaxial_tensor(1.52, 1.74, np.pi / 2, phi=0.4)    # in-plane
lc_oop = lc.copy()
lc_oop[1, 0] = uniaxial_tensor(1.52, 1.74, 0.7, phi=0.4)       # tilted
mu_t = np.empty((2, 2, 3, 3), complex)
mu_t[:] = np.eye(3)
mu_t[0, 1] = np.diag([1.6, 1.2, 1.4])
mu_t[0, 1, 0, 1] = mu_t[0, 1, 1, 0] = 0.1

solvers = {
    "scal_int2_M4": dict(wx=2, wy=2, M=4, eps=C.cell("pillar", n=2)),
    "scal_nonu3_M5_obl": dict(wx=C.XW, wy=C.YW, M=5, eps=eps3, a0x=0.9,
                              a0y=-0.35),
    "scal_int3_M6": dict(wx=3, wy=3, M=6, eps=eps3),
    "tens_ip_M5": dict(wx=2, wy=2, M=5, eps=lc, a0x=0.4),
    "mag_scal_M4": dict(wx=2, wy=2, M=4, eps=C.cell("pillar", n=2),
                        mu=np.full((2, 2), 1.3 + 0j)),
    "mag_tens_M5": dict(wx=2, wy=2, M=5, eps=C.cell("stripe", n=2),
                        mu=mu_t, a0y=0.3),
    "oop_M4": dict(wx=2, wy=2, M=4, eps=lc_oop),
    "slant_M4": dict(wx=2, wy=2, M=4, eps=C.cell("pillar", n=2),
                     slant=(0.2, -0.1)),
}
built = {}
for tag, d in solvers.items():
    kw = dict(alpha0x=d.get("a0x", 0.0) * K0, alpha0y=d.get("a0y", 0.0) * K0,
              k0=K0)
    if "mu" in d:
        kw["mu_cell"] = d["mu"]
    if "slant" in d:
        kw["slant"] = d["slant"]
    s = TS.Granet2DTransverseE(P, P, d["wx"], d["wy"], d["M"], d["eps"], **kw)
    built[tag] = s
    walk_solver(tag, s)

# region modes
for tag in ("scal_int2_M4", "scal_nonu3_M5_obl", "scal_int3_M6",
            "tens_ip_M5", "mag_scal_M4", "mag_tens_M5"):
    W, V, lam, g2 = TS._region_modes(built[tag])
    put(f"modes/{tag}", W, V, lam, g2)
for sym in (False, True):
    six = TS._region_modes_oop(built["oop_M4"], symmetry=sym)
    put(f"modes/oop_M4_sym{sym}", *six)
six = TS._region_modes_oop(built["slant_M4"], symmetry=False)
put("modes/slant_M4", *six)
# the shared geometric eig + homogeneous modes (normal and oblique)
for tag, wx, M, a0 in (("geo_int2_M4", 2, 4, 0.0), ("geo_nonu_M5", C.XW, 5,
                                                     0.7)):
    wy = 2 if np.ndim(wx) == 0 else C.YW
    n = 2 if np.ndim(wx) == 0 else 3
    s = TS.Granet2DTransverseE(P, P, wx, wy, M, np.full((n, n), 1.0 + 0j),
                               alpha0x=a0 * K0, k0=K0)
    g = TS._homog_geom_cache(s)
    put(f"geom/{tag}", *[x for x in g if isinstance(x, np.ndarray)])
    for e in (1.0, 2.3104, 3.0 + 0.2j):
        put(f"homog/{tag}/eps{e}", *TS._homog_region_modes(g, e))
# far projectors
for tag, wx, M, a0x, a0y in (("int2_M4", 2, 4, 0.0, 0.0),
                             ("nonu_M5_obl", C.XW, 5, 0.9, -0.35),
                             ("int3_M6_con", 3, 6, 0.5, 0.6)):
    wy = 2 if (np.ndim(wx) == 0 and wx == 2) else (3 if np.ndim(wx) == 0
                                                    else C.YW)
    bx = TS.Basis1D(P, wx, M, np.exp(-1j * a0x * K0 * P))
    by = TS.Basis1D(P, wy, M, np.exp(-1j * a0y * K0 * P))
    ox = np.arange(-3, 4)
    put(f"proj/{tag}", *TS._far_projector_2d(bx, by, ox, ox, a0x * K0,
                                             a0y * K0))
    put(f"proj_ordrs/{tag}", *TS._pmm2d_project_orders(
        *TS._far_projector_2d(bx, by, ox, ox, a0x * K0, a0y * K0),
        np.eye(2 * bx.dim * by.dim, dtype=complex)[:, :5], bx.dim * by.dim),
        )


def eff_hash(key, res):
    arrs = []
    for name in ("orders", "R", "T", "r", "t", "jones"):
        v = getattr(res, name, None)
        if isinstance(v, np.ndarray):
            arrs.append(v)
    if not arrs:
        arrs = [np.asarray(x) for x in res if isinstance(x, np.ndarray)]
    put(key, *arrs)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # single-polarization entry
    for pol in ("te", "tm"):
        for th, ph, M in ((0.0, 0.0, 4), (0.3, 0.0, 5), (0.25, 0.6, 6)):
            r = TS.pmm_efficiency_2d_staggered(
                P, P, C.cell("pillar", n=2), C.N_SUB, C.N_SUP, C.DEPTH, WL,
                degree=M, n_orders=2, polarization=pol, theta=th, phi=ph)
            eff_hash(f"eff2d/{pol}/th{th}_ph{ph}_M{M}", r)
    # Jones entry
    jcases = {
        "scal_con_M5": dict(eps=C.cell("pillar", n=2), M=5, th=0.3, ph=0.5),
        "scal_norm_M6": dict(eps=C.cell("stripe", n=2), M=6),
        "tens_M4": dict(eps=lc, M=4, th=0.2, ph=0.3),
        "mag_M4": dict(eps=C.cell("pillar", n=2), M=4,
                       mu=np.full((2, 2), 1.4 + 0j)),
        "magT_M5_obl": dict(eps=C.cell("pillar", n=2), M=5, mu=mu_t, th=0.2),
        "oop_auto_M4": dict(eps=lc_oop, M=4),
        "oop_nosym_M4": dict(eps=lc_oop, M=4, sym=False),
        "oop_obl_M5": dict(eps=lc_oop, M=5, th=0.15, ph=0.2),
        "slant_M4": dict(eps=C.cell("pillar", n=2), M=4, slant=(0.15, 0.05)),
        "slant_obl_M5": dict(eps=C.cell("stripe", n=2), M=5,
                             slant=(-0.1, 0.0), th=0.2),
        "lossy_M5": dict(eps=C.cell("pillar", 4.0 + 0.6j, n=2), M=5,
                         th=0.1, ph=0.9),
    }
    for tag, d in jcases.items():
        kw = dict(degree=d["M"], n_orders=2, theta=d.get("th", 0.0),
                  phi=d.get("ph", 0.0))
        if "mu" in d:
            kw["mu_cell"] = d["mu"]
        if "slant" in d:
            kw["slant"] = d["slant"]
        if "sym" in d:
            kw["symmetry"] = d["sym"]
        out = TS.pmm_jones_2d_staggered(P, P, d["eps"], C.N_SUB, C.N_SUP,
                                        C.DEPTH, WL, **kw)
        put(f"jones/{tag}", *[np.asarray(x) for x in out])

    # stacks: lossy multilayer with absorption, normal + oblique + conical
    def stack_case(tag, grids, layers, th, ph, M, absorb=True):
        st = PMM2DStackPure(P, P, n_superstrate=C.N_SUP, n_substrate=C.N_SUB,
                            n_modes=M, n_orders=2, layer_grids=grids)
        for L in layers:
            st.add_layer(L[0], **L[1])
        st.set_source(WL, theta=th, phi=ph)
        o, R, T, J = st.solve(retain_internal=absorb)
        put(f"stack/{tag}/RTJ", o, R, T, J)
        if absorb:
            put(f"stack/{tag}/absorption", st.layer_absorption())
    lossy = [(0.15, dict(eps=2.1)),
             (0.3, dict(eps_cell=C.cell("pillar", 4.0 + 0.5j, n=2))),
             (0.2, dict(eps=1.8 + 0.1j)),
             (0.25, dict(eps_cell=C.cell("stripe", 3.0 + 0j, n=2)))]
    for th, ph, M in ((0.0, 0.0, 4), (0.3, 0.0, 5), (0.25, 0.7, 4)):
        stack_case(f"lossy4_th{th}_ph{ph}_M{M}", "shared", lossy, th, ph, M)
    dbr = [(0.1, dict(eps_cell=C.cell("pillar", n=2))), (0.12, dict(eps=2.0)),
           (0.1, dict(eps_cell=C.cell("pillar", n=2))), (0.12, dict(eps=2.0))]
    stack_case("dbr_dedupe_M5", "shared", dbr, 0.1, 0.2, 5)
    stack_case("tensor_mag_mix_M4", "shared",
               [(0.2, dict(eps_cell=lc)),
                (0.2, dict(eps=2.0, mu=1.3)),
                (0.2, dict(eps_cell=lc_oop))], 0.0, 0.0, 4, absorb=False)
    # per-layer mortar with absorption
    per = [(0.25, dict(eps_cell=C.cell("pillar", 4.0 + 0.3j, n=2),
                       x_walls=[0.4], y_walls=[0.55])),
           (0.2, dict(eps=2.0)),
           (0.25, dict(eps_cell=C.cell("stripe", n=2), x_walls=[0.3],
                       y_walls=[0.5]))]
    stack_case("perlayer_M4", "per-layer", per, 0.0, 0.0, 4)
    stack_case("perlayer_obl_M5", "per-layer", per, 0.2, 0.4, 5)

print(TAG, len(OUT), "keys")
C.dump(f"v1_bytes_{TAG}", {"tag": TAG, "n_keys": len(OUT), "hashes": OUT})
