"""C1 (Phase C copy of build_b/b2_bytes.py, itself build_a/a1_bytes.py; only
the output name differs: c1_bytes_<label>.json; PRE tree = git archive
91d00288 in C:/tmp/curved_pre_c).

A1 -- NO MAP = TODAY'S BYTES.  SHA-256 of every operator block and of every
R / T / Jones / absorption output on a fixture set that touches every dispatch
branch of the pure staggered 2-D PMM (scalar, non-uniform walls, in-plane
tensor, magnetic, out-of-plane with and without the parity reduction, slant,
oblique and conical incidence, a multilayer stack with retain_internal and a
lossy layer, the per-layer mortar).

Run in BOTH trees with the SAME file, and compare the two JSON files
(a1_compare.py):

  cd /c/tmp/curved_pre_a && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/curved_pre_a \
    python C:/tmp/lum_curved/validation/probe_pmm2d_curved/build_a/a1_bytes.py \
    C:/tmp/curved_pre_a pre
  cd /c/tmp/lum_curved && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved \
    python validation/probe_pmm2d_curved/build_a/a1_bytes.py \
    C:/tmp/lum_curved post idmap

Output: a1_bytes_<label>.json next to this file.  Arg 3 'idmap' (post tree
only) adds the IDENTITY map through the quadrature path as the fail-before arm
(keys suffixed '@idmap'): the hash must SEE that round-off change.
"""
import hashlib
import json
import os
import sys
import warnings

ROOT = os.path.normcase(os.path.abspath(sys.argv[1]))
LABEL = sys.argv[2]
IDMAP = len(sys.argv) > 3 and sys.argv[3] == "idmap"

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(ROOT), (
    f"lumenairy imported from {lumenairy.__file__}, not {ROOT}")

from lumenairy.elements.pmm import (
    PMM2DStackPure,  # noqa: E402
    twod_staggered as TS,  # noqa: E402
)
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
H = {}
DIFF = {}


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


WL, P = 1.0, 1.2
K0 = 2 * np.pi / WL
pil3 = np.ones((3, 3), complex)
pil3[1, 1] = 4.0
pil2 = np.ones((2, 2), complex)
pil2[0, 0] = 2.25 + 0.0j
LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
LC_OOP = uniaxial_tensor(1.5, 1.8, 0.6, phi=0.3)
EYE = np.eye(3, dtype=complex)
MU_G = np.array([[1.6, 0.4j, 0], [-0.4j, 1.6, 0], [0, 0, 1.2]], complex)


def tcell(host, incl, n=2):
    c = np.empty((n, n, 3, 3), complex)
    c[:] = host
    c[0, 0] = incl
    return c


# ---- operator blocks ------------------------------------------------------
NW = np.array([0.0, 0.25, 0.85, P])
cases = {
    "scalar_int3_M5": dict(wx=3, wy=3, M=5, eps=pil3),
    "scalar_nonuni3_M5": dict(wx=NW, wy=np.array([0, 0.3, 0.9, P]), M=5,
                              eps=pil3),
    "scalar_int3_M5_obl": dict(wx=3, wy=3, M=5, eps=pil3, a0=(0.9, 0.4)),
    "tensor_int2_M5": dict(wx=2, wy=2, M=5, eps=tcell(2.0 * EYE, LC)),
    "magnetic_int2_M5": dict(wx=2, wy=2, M=5, eps=pil2,
                             mu=np.ones((2, 2), complex) * 1.3),
    "magnetic_tensor_mu_M4": dict(wx=2, wy=2, M=4, eps=pil2,
                                  mu=tcell(EYE, MU_G)),
    "oop_int2_M4": dict(wx=2, wy=2, M=4, eps=tcell(2.0 * EYE, LC_OOP)),
    "slant_int2_M4": dict(wx=2, wy=2, M=4, eps=pil2, slant=(0.2, -0.1)),
    "uniform_int2_M5": dict(wx=2, wy=2, M=5, eps=np.full((2, 2), 2.1 + 0j)),
}
for name, c in cases.items():
    a0 = c.get("a0", (0.0, 0.0))
    s = TS.Granet2DTransverseE(P, P, c["wx"], c["wy"], c["M"], c["eps"],
                               alpha0x=a0[0], alpha0y=a0[1], k0=K0,
                               mu_cell=c.get("mu"), slant=c.get("slant"))
    ops(name, s)
    if not s.offplane:
        W, V, lam, g2 = TS._region_modes(s)
        H[f"{name}.region_modes"] = sha(W, V, lam, g2)
    if name.startswith("uniform"):
        geom = TS._homog_geom_cache(s)
        H[f"{name}.geom"] = sha(*geom[:5])
        H[f"{name}.homog_modes"] = sha(*TS._homog_region_modes(geom, 2.1))
    if name == "oop_int2_M4":
        H[f"{name}.modes_oop_sym"] = sha(*TS._region_modes_oop(s, symmetry=True))
        H[f"{name}.modes_oop"] = sha(*TS._region_modes_oop(s, symmetry=False))
    if IDMAP and not s.offplane and not s.magnetic and s.eps_cell.ndim == 2:
        from lumenairy.elements.pmm._curvemap import IdentityMap
        sI = TS.Granet2DTransverseE(P, P, c["wx"], c["wy"], c["M"], c["eps"],
                                    alpha0x=a0[0], alpha0y=a0[1], k0=K0,
                                    cmap=IdentityMap(c["wx"], c["wy"], P, P))
        ops(name + "@idmap", sI)
        DIFF[name] = {
            "Rmat": float(np.max(np.abs(sI.Rmat - s.Rmat))),
            "Lmat": float(np.max(np.abs(sI.Lmat - s.Lmat))),
            "Lmat_rel": float(np.max(np.abs(sI.Lmat - s.Lmat))
                              / np.max(np.abs(s.Lmat)))}

# far projectors
for name, (wx, a0) in {"far_int3": (3, (0.0, 0.0)),
                       "far_nonuni": (NW, (0.0, 0.0)),
                       "far_int3_obl": (3, (0.9, 0.4))}.items():
    bx = TS.Basis1D(P, wx, 5, np.exp(-1j * a0[0] * P))
    by = TS.Basis1D(P, wx, 5, np.exp(-1j * a0[1] * P))
    ox = np.arange(-3, 4)
    H[name] = sha(*TS._far_projector_2d(bx, by, ox, ox, a0[0], a0[1]))

# ---- full solves ----------------------------------------------------------
warnings.simplefilter("ignore")
for pol in ("te", "tm"):
    for th, ph in ((0.0, 0.0), (0.2, 0.3)):
        o, R, T = TS.pmm_efficiency_2d_staggered(
            P, P, pil3, 1.45, 1.0, 0.5, WL, degree=5, n_orders=3,
            polarization=pol, theta=th, phi=ph)
        H[f"eff_{pol}_{th}"] = sha(o, R, T)
for name, kw in {
        "jones_scalar_conical": dict(eps_cell=pil2, theta=0.25, phi=0.4),
        "jones_tensor": dict(eps_cell=tcell(2.0 * EYE, LC)),
        "jones_magnetic": dict(eps_cell=pil2, mu_cell=tcell(EYE, MU_G)),
        "jones_oop_auto": dict(eps_cell=tcell(2.0 * EYE, LC_OOP)),
        "jones_oop_nosym": dict(eps_cell=tcell(2.0 * EYE, LC_OOP),
                                symmetry=False),
        "jones_slant": dict(eps_cell=pil2, slant=(0.2, -0.1)),
}.items():
    out = TS.pmm_jones_2d_staggered(P, P, depth=0.4, wavelength=WL,
                                    n_substrate=1.45, n_superstrate=1.0,
                                    degree=4, n_orders=2, **kw)
    H[name] = sha(*out)

# multilayer stack: uniform | patterned (lossy) | uniform tensor, absorption
st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=4,
                    n_orders=2)
st.add_layer(0.2, eps=2.1)
st.add_layer(0.3, eps_cell=np.array([[4.0 + 0.3j, 1.0], [1.0, 1.0]]))
st.add_layer(0.15, eps=LC)
st.set_source(WL, theta=0.15, phi=0.2)
H["stack_multilayer"] = sha(*st.solve(retain_internal=True))
H["stack_multilayer_absorption"] = sha(st.layer_absorption())

# per-layer mortar
st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=4,
                    n_orders=2, layer_grids="per-layer")
st.add_layer(0.3, eps_cell=np.array([[4.0, 1.0], [1.0, 1.0]]),
             x_walls=[0.5], y_walls=[0.45])
st.add_layer(0.2, eps_cell=np.array([[2.25, 1.0], [1.0, 1.0]]))
st.set_source(WL)
H["stack_perlayer"] = sha(*st.solve(retain_internal=True))
H["stack_perlayer_absorption"] = sha(st.layer_absorption())

env = {"python": sys.version.split()[0], "numpy": np.__version__,
       "lumenairy": lumenairy.__file__,
       "threads": {k: os.environ.get(k) for k in
                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS")}}
with open(os.path.join(HERE, f"c1_bytes_{LABEL}.json"), "w") as f:
    json.dump({"env": env, "sha": H, "idmap_diff": DIFF}, f, indent=1)
print(LABEL, len(H), "hashes")
