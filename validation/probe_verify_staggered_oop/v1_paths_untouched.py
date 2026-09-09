"""V1 -- the SCALAR and IN-PLANE-TENSOR paths are untouched by the Stage-B
OUT-OF-PLANE integration, proven ACROSS CODE VERSIONS.

The build doc's T4 shows the out-of-plane generator REDUCES to the in-plane
answer on a cell with zero cross terms.  That is an internal-consistency
statement inside one build.  It cannot show that the shipped scalar / in-plane
answers did not move when ``_require_block_form`` became ``_tile_needs_oop``,
when ``_region_modes`` grew a guard, and when ``PMM2DStackPure.solve`` grew the
``any_oop`` branch.

This probe runs the SAME fixtures under three PYTHONPATHs, one process each:

  (a) this worktree            C:/tmp/lum_aniso           (HEAD, merge 6e49bed)
  (b) the pre-integration head C:/tmp/lum_aniso_pre       (0c3e871, Stage A)
  (c) the read-only main clone D:/.../Lumenairy           (f70628d, pre-Stage-A)

(a) vs (b) must be bit-identical on BOTH the scalar and the in-plane-tensor
fixtures; (c) has no tensor entry, so it runs the scalar ones only.

Usage:  PYTHONPATH=<root> python v1_paths_untouched.py <root> <out.json>
"""
import hashlib
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

from lumenairy.elements.pmm import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _region_modes,
    pmm_efficiency_2d_staggered,
)

HAS_TENSOR = hasattr(
    __import__("lumenairy.elements.pmm.twod_staggered", fromlist=["x"]),
    "pmm_jones_2d_staggered")


def h(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def rot2(e11, e12, e21, e22, e33):
    return np.array([[e11, e12, 0.0], [e21, e22, 0.0], [0.0, 0.0, e33]],
                    dtype=complex)


# ---------------------------------------------------------------- fixtures
# FRESH geometry -- deliberately none of the Stage-A verification's numbers.
P = 0.83e-6
WL = 0.61e-6
DEP = 0.34e-6

S_A = np.array([[7.29, 2.56], [2.56, 2.56]], dtype=complex)          # (2,2)
S_B = np.array([[4.41, 2.10, 9.00],                                  # (3,3)
                [2.10, 6.25, 2.10],
                [3.24, 2.10, 2.10]], dtype=complex)
S_C = np.array([[5.76 + 0.37j, 2.10], [2.10, 3.24]], dtype=complex)  # lossy

SCALAR = [
    ("s1_22_te_normal", S_A, dict(degree=6, n_orders=3, polarization="te")),
    ("s2_33_tm_oblique", S_B,
     dict(degree=5, n_orders=3, polarization="tm", theta=0.27)),
    ("s3_22_lossy_conical", S_C,
     dict(degree=6, n_orders=3, polarization="te", theta=0.19, phi=0.71)),
    ("s4_22_te_conical", S_A,
     dict(degree=7, n_orders=4, polarization="te", theta=0.41, phi=1.13)),
]

out = {"root": ROOT, "lumenairy": lumenairy.__file__,
       "version": lumenairy.__version__, "has_tensor_entry": HAS_TENSOR,
       "python": sys.version.split()[0], "numpy": np.__version__,
       "fixtures": {}}

for name, cell, kw in SCALAR:
    o, R, T = pmm_efficiency_2d_staggered(P, P, cell, 1.45, 1.0, DEP, WL, **kw)
    out["fixtures"][name] = {"orders": h(np.asarray(o)), "R": h(R), "T": h(T),
                             "sumR": float(np.sum(R)), "sumT": float(np.sum(T))}

# scalar pure stack (two layers, conical, Jones)
st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=6,
                    n_orders=3)
st.add_layer(DEP, eps_cell=S_A)
st.add_layer(0.13e-6, eps=2.56)
st.set_source(WL, theta=0.23, phi=0.55)
o, R, T, J = st.solve(jones=True)
out["fixtures"]["s5_stack_scalar_jones"] = {
    "R": h(R), "T": h(T), "jones": h(J),
    "sumRT": [float(R[r].sum() + T[r].sum()) for r in (0, 1)]}

# scalar assembled operators
SOPS = [
    ("sop_22_M6", S_A, 6, dict(alpha0x=0.37, alpha0y=-0.21,
                               k0=2 * np.pi / 0.61)),
    ("sop_33_M5", S_B, 5, dict(alpha0x=0.0, alpha0y=0.0,
                               k0=2 * np.pi / 0.58)),
    ("sop_lossy_M6", S_C, 6, dict(alpha0x=0.13, alpha0y=0.31,
                                  k0=2 * np.pi / 0.61)),
]
for name, cell, M, kw in SOPS:
    s = Granet2DTransverseE(1.3, 1.3, cell.shape[0], cell.shape[1], M, cell,
                            **kw)
    rec = {k: h(getattr(s, k)) for k in ("Lmat", "Rmat", "Stt", "Schur")}
    rec["Et0"] = h(s.Et_blocks[0])
    rec["Et1"] = h(s.Et_blocks[1])
    W, V, lam, g2 = _region_modes(s)
    rec["lam_sorted"] = h(np.sort_complex(np.asarray(lam)))
    rec["W"] = h(W)
    rec["V"] = h(V)
    out["fixtures"][name] = rec

# ------------------------------------------------- IN-PLANE TENSOR fixtures
if HAS_TENSOR:
    from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
        pmm_jones_2d_staggered,
    )
    LC = rot2(2.25, 0.42, 0.42, 2.89, 2.25)          # symmetric in-plane
    GY = rot2(2.60, 0.35j, -0.35j, 2.60, 2.40)       # gyrotropic (Hermitian)
    LOS = rot2(3.61 + 0.22j, 0.11, 0.09, 4.41 + 0.15j, 3.00)
    ISO = 2.56 * np.eye(3, dtype=complex)

    def tile(cells):
        a = np.zeros(cells.shape + (3, 3), dtype=complex)
        return a

    T_A = np.zeros((2, 2, 3, 3), dtype=complex)      # LC host + iso pillar
    T_A[:] = LC
    T_A[0, 0] = ISO
    T_B = np.zeros((3, 3, 3, 3), dtype=complex)      # gyro host, 2 pillars
    T_B[:] = GY
    T_B[0, 1] = ISO
    T_B[2, 2] = LOS
    T_U = np.zeros((2, 2, 3, 3), dtype=complex)      # uniform LC
    T_U[:] = LC

    TENSOR = [
        ("t1_lc_pillar_normal", T_A, dict(degree=6, n_orders=3)),
        ("t2_gyro_33_oblique", T_B, dict(degree=5, n_orders=3, theta=0.27)),
        ("t3_uniform_lc_conical", T_U,
         dict(degree=6, n_orders=3, theta=0.19, phi=0.71)),
        ("t4_lc_pillar_conical", T_A,
         dict(degree=7, n_orders=4, theta=0.41, phi=1.13)),
    ]
    for name, cell, kw in TENSOR:
        o, R, T, J = pmm_jones_2d_staggered(P, P, cell, 1.45, 1.0, DEP, WL,
                                            **kw)
        out["fixtures"][name] = {"orders": h(np.asarray(o)), "R": h(R),
                                 "T": h(T), "jones": h(J),
                                 "sumR": float(np.sum(R)),
                                 "sumT": float(np.sum(T))}

    # mixed tensor + scalar stack
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=6,
                        n_orders=3)
    st.add_layer(DEP, eps_cell=T_A)
    st.add_layer(0.13e-6, eps=LC)
    st.add_layer(0.09e-6, eps=2.56)
    st.set_source(WL, theta=0.23, phi=0.55)
    o, R, T, J = st.solve(jones=True)
    out["fixtures"]["t5_mixed_stack_jones"] = {
        "R": h(R), "T": h(T), "jones": h(J),
        "sumRT": [float(R[r].sum() + T[r].sum()) for r in (0, 1)]}

    # in-plane tensor assembled operators
    TOPS = [
        ("top_lc_22_M6", T_A, 6, dict(alpha0x=0.37, alpha0y=-0.21,
                                      k0=2 * np.pi / 0.61)),
        ("top_gyro_33_M5", T_B, 5, dict(alpha0x=0.0, alpha0y=0.0,
                                        k0=2 * np.pi / 0.58)),
    ]
    for name, cell, M, kw in TOPS:
        s = Granet2DTransverseE(1.3, 1.3, cell.shape[0], cell.shape[1], M,
                                cell, **kw)
        rec = {k: h(getattr(s, k)) for k in ("Lmat", "Rmat", "Stt", "Schur")}
        rec["Et0"] = h(s.Et_blocks[0])
        rec["Et1"] = h(s.Et_blocks[1])
        rec["Etoff0"] = h(s.Et_offdiag[0])
        rec["Etoff1"] = h(s.Et_offdiag[1])
        W, V, lam, g2 = _region_modes(s)
        rec["lam_sorted"] = h(np.sort_complex(np.asarray(lam)))
        rec["W"] = h(W)
        rec["V"] = h(V)
        out["fixtures"][name] = rec

json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out, indent=1))
