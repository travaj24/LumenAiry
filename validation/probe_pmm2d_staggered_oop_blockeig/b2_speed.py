"""B2 -- interleaved A/B timing of the parity-sign block reduction, plus the
eig-vs-whole-solve share it acts on.

Two arms (``symmetry=True`` / ``False``) run as SEPARATE subprocesses,
alternating round-robin, min over N rounds -- the instrument
docs/audits/EXPERIMENT_PMM2D_OOP_BLOCK_EIG_2026_08_17.md S5 uses, so the
ratios here are comparable with the hybrid's 1.61-2.35x.

Run:
  cd /c/tmp/lum_oopfast && PYTHONPATH=/c/tmp/lum_oopfast OMP_NUM_THREADS=1 \\
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python validation/probe_pmm2d_staggered_oop_blockeig/b2_speed.py
"""
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(ROOT), (
    f"lumenairy.__file__ = {lumenairy.__file__} is not under {ROOT}")

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _region_modes_oop,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import uniaxial_tensor  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

PX = PY = 1.10e-6
WL = 0.68e-6
DEP = 0.34e-6
NSUB, NSUP = 1.50, 1.0
K0 = 2.0 * np.pi / WL
TIL = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
AIR = np.eye(3, dtype=complex)
ISO = 2.25 * np.eye(3, dtype=complex)


def cell_for(Nx, kind):
    c = np.zeros((Nx, Nx, 3, 3), dtype=complex)
    c[:, :] = AIR
    if kind == "uniform":
        c[:, :] = TIL
    elif Nx == 2:
        c[0, 0] = TIL
        c[1, 1] = TIL
    else:
        c[1, 1] = TIL
        c[0, 0] = ISO
        c[2, 2] = ISO
    return c


CASES = [(Nx, M, kind, nl)
         for kind in ("pillar",)
         for Nx in (2, 3)
         for M in (6, 7, 8)
         for nl in (1,)]
CASES += [(2, 7, "pillar", 3), (3, 7, "pillar", 3)]
ROUNDS = 4


def run_case(Nx, M, kind, nl, sym):
    from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
    cell = cell_for(Nx, kind)
    # warm-up (BLAS / import), then the timed solve
    stack = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                           n_modes=4, n_orders=3, symmetry=sym)
    stack.add_layer(DEP, eps_cell=cell)
    stack.set_source(WL)
    stack.solve(jones=True)
    t0 = time.perf_counter()
    if nl == 1:
        pmm_jones_2d_staggered(PX, PY, cell, NSUB, NSUP, DEP, WL, degree=M,
                               n_orders=5, symmetry=sym)
    else:
        st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                            n_modes=M, n_orders=5, symmetry=sym)
        for i in range(nl):
            # DISTINCT cells so the eig cache cannot collapse the stack
            cc = cell.copy()
            cc[0, 0] = cc[0, 0] * (1.0 + 0.01 * i)
            if Nx == 3:
                cc[2, 2] = cc[2, 2] * (1.0 + 0.01 * i)
            else:
                cc[1, 1] = cc[1, 1] * (1.0 + 0.01 * i)
            st.add_layer(DEP / nl, eps_cell=cc)
        st.set_source(WL)
        st.solve(jones=True)
    return time.perf_counter() - t0


if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    Nx, M, nl = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[5])
    sym = sys.argv[4] == "1"
    print(f"{run_case(Nx, M, 'pillar', nl, sym):.6f}")
    raise SystemExit(0)

R = {"lumenairy": lumenairy.__file__, "version": lumenairy.__version__,
     "rounds": ROUNDS}

print("\n## B4a  interleaved whole-solve A/B (separate subprocesses, "
      f"min over {ROUNDS} rounds)")
print("   grid  M  layers  4q^2   dense [s]   factored [s]   speedup   "
      "per-round dense / factored")
rows = []
env = dict(os.environ)
for Nx, M, kind, nl in CASES:
    td, tf = [], []
    for _ in range(ROUNDS):
        for sym, acc in ((0, td), (1, tf)):
            out = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--worker",
                 str(Nx), str(M), str(sym), str(nl)],
                capture_output=True, text=True, env=env, cwd=ROOT, check=True)
            acc.append(float(out.stdout.strip().splitlines()[-1]))
    q = Nx * (M - 1)
    sp = min(td) / min(tf)
    print(f"   ({Nx},{Nx})  {M}   {nl}     {4*q*q:5d}  {min(td):8.3f}   "
          f"{min(tf):10.3f}   {sp:6.2f}x   "
          f"[{' '.join(f'{x:.3f}' for x in td)}] / "
          f"[{' '.join(f'{x:.3f}' for x in tf)}]")
    rows.append(dict(Nx=Nx, M=M, layers=nl, dim=4 * q * q, dense=min(td),
                     factored=min(tf), speedup=sp, td=td, tf=tf))
R["B4a"] = rows

print("\n## B4b  where the time goes: region solve vs whole solve, and the "
      "reduction's own share")
print("   grid  M   4q^2   region OFF [s]  region ON [s]  region ratio   "
      "whole OFF [s]  region share")
rows = []
for Nx in (2, 3):
    for M in (6, 7, 8):
        cell = cell_for(Nx, "pillar")
        sol = Granet2DTransverseE(PX, PY, Nx, Nx, M, cell, k0=K0)
        _region_modes_oop(sol, symmetry=True)          # warm
        t0 = time.perf_counter()
        _region_modes_oop(sol, symmetry=False)
        toff = time.perf_counter() - t0
        t0 = time.perf_counter()
        _region_modes_oop(sol, symmetry=True)
        ton = time.perf_counter() - t0
        t0 = time.perf_counter()
        pmm_jones_2d_staggered(PX, PY, cell, NSUB, NSUP, DEP, WL, degree=M,
                               n_orders=5, symmetry=False)
        twhole = time.perf_counter() - t0
        q = Nx * (M - 1)
        print(f"   ({Nx},{Nx})  {M}   {4*q*q:5d}  {toff:12.3f}  {ton:12.3f}  "
              f"{toff/ton:11.2f}x  {twhole:12.3f}  {toff/twhole*100:9.1f}%")
        rows.append(dict(Nx=Nx, M=M, dim=4 * q * q, region_off=toff,
                         region_on=ton, region_ratio=toff / ton,
                         whole_off=twhole, share=toff / twhole))
R["B4b"] = rows

with open(os.path.join(OUT, "b2_speed.json"), "w") as f:
    json.dump(R, f, indent=1, default=str)
print("\nwrote results/b2_speed.json")
