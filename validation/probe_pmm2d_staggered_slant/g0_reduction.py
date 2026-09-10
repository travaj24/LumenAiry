"""G0 -- REDUCTION gates for the slant prototype (must pass before any measurement).

G0a: at slant = 0 the prototype generator pencil (A, B) is BIT-IDENTICAL to the
     shipped ``Granet2DTransverseE._assemble_oop`` on an out-of-plane cell.
G0b: at slant = 0 the prototype driver reproduces ``PMM2DStackPure.solve``
     bit-for-bit on (i) a scalar pillar, (ii) an in-plane tensor pillar,
     (iii) an out-of-plane tensor pillar.
G0c: the covariant congruence is an algebraic identity: A^-1 (A eps A^T) A^-T
     == eps, and cov_tensor(eps, 0, 0) == eps.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import (  # noqa: E402
    SlantSolver, assert_worktree, cov_tensor, solve_slant_stack,
    tensor_uniaxial,
)
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
)
from lumenairy.elements.pmm import PMM2DStackPure  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
res = {"lumenairy": assert_worktree()}
t00 = time.time()

# --------------------------------------------------------------- G0c (algebra)
rng = np.random.default_rng(7)
e = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))).astype(complex)
tx, ty = 0.37, -0.21
A = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=complex)
rt = cov_tensor(A @ e @ A.T, tx, ty)
res["G0c_roundtrip"] = float(np.max(np.abs(rt - e)))
res["G0c_t0_identity"] = float(np.max(np.abs(cov_tensor(e, 0.0, 0.0) - e)))
res["G0c_detA"] = float(abs(np.linalg.det(A) - 1.0))
print(f"G0c roundtrip {res['G0c_roundtrip']:.3e}  t0 {res['G0c_t0_identity']:.3e}"
      f"  detA-1 {res['G0c_detA']:.3e}")

# --------------------------------------------------------------- G0a (pencil)
px = py = 0.9
M = 5
Nx = Ny = 2
cell = np.zeros((Nx, Ny, 3, 3), dtype=complex)
tu = tensor_uniaxial(1.5, 1.7, np.deg2rad(35), np.deg2rad(25))
for i in range(Nx):
    for j in range(Ny):
        cell[i, j] = tu if (i == 0 and j == 0) else np.eye(3)
k0 = 2 * np.pi
a0x, a0y = 0.30, 0.17
ship = Granet2DTransverseE(px, py, Nx, Ny, M, cell, alpha0x=a0x, alpha0y=a0y,
                           k0=k0)
proto = SlantSolver(px, py, Nx, Ny, M, cell, slant=(0.0, 0.0),
                    alpha0x=a0x, alpha0y=a0y, k0=k0)
res["G0a_offplane_dispatch"] = bool(ship.offplane)
res["G0a_dA_bits"] = int(np.count_nonzero(ship.Agen != proto.Agen))
res["G0a_dB_bits"] = int(np.count_nonzero(ship.Bgen != proto.Bgen))
res["G0a_dA_max"] = float(np.max(np.abs(ship.Agen - proto.Agen)))
res["G0a_dB_max"] = float(np.max(np.abs(ship.Bgen - proto.Bgen)))
print(f"G0a offplane={ship.offplane} dA bits {res['G0a_dA_bits']} "
      f"max {res['G0a_dA_max']:.3e} | dB bits {res['G0a_dB_bits']}")

# --------------------------------------------------------------- G0b (driver)
def run_lib(cell, depth, wl, theta, phi, M, n_orders, nsup, nsub):
    st = PMM2DStackPure(px, py, n_superstrate=nsup, n_substrate=nsub,
                        n_modes=M, n_orders=n_orders)
    if cell.ndim == 4:
        st.add_layer(depth, eps_cell=cell)
    else:
        st.add_layer(depth, eps_cell=cell)
    st.set_source(wl, theta=theta, phi=phi)
    return st.solve(jones=True)


cases = {}
scal = np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex)
inpl = np.zeros((2, 2, 3, 3), dtype=complex)
tup = tensor_uniaxial(1.5, 1.7, np.pi / 2, np.deg2rad(25))   # in-plane
for i in range(2):
    for j in range(2):
        inpl[i, j] = tup if (i == 0 and j == 0) else np.eye(3)
oop = cell

for name, c in (("scalar", scal), ("inplane", inpl), ("oop", oop)):
    for mount, (th, ph) in (("normal", (0.0, 0.0)),
                            ("conical", (0.25, 0.6))):
        o1, R1, T1, J1 = run_lib(c, 0.35, 0.85, th, ph, 5, 4, 1.0, 1.5)
        o2, R2, T2, J2, _Jt, _inf = solve_slant_stack(
            px, py, [{"thickness": 0.35, "cell": c, "slant": (0.0, 0.0)}],
            1.0, 1.5, 0.85, M=5, n_orders=4, theta=th, phi=ph)
        if R1.ndim == 1:                       # scalar entry returns (Nfo,)
            R1 = np.atleast_2d(R1)
            T1 = np.atleast_2d(T1)
        d = dict(dR=float(np.max(np.abs(R1 - R2))),
                 dT=float(np.max(np.abs(T1 - T2))),
                 dJ=float(np.max(np.abs(J1 - J2))),
                 bits_R=int(np.count_nonzero(R1 != R2)))
        cases[f"{name}/{mount}"] = d
        print(f"G0b {name:8s} {mount:8s} dR {d['dR']:.3e} dT {d['dT']:.3e} "
              f"dJ {d['dJ']:.3e} bits {d['bits_R']}")
res["G0b"] = cases
res["wall_s"] = time.time() - t00
with open(os.path.join(OUT, "g0_reduction.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"\nWROTE results/g0_reduction.json  ({res['wall_s']:.1f} s)")
