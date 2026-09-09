"""Probe 12 -- cost: peak RSS and wall time of a (3,3) M=8 TENSOR solve vs the
same-size SCALAR solve, and the retained-operator inventory."""
import time
import tracemalloc

import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _region_modes,
    pmm_efficiency_2d_staggered,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

P, WL, DEP = 0.7e-6, 0.55e-6, 0.28e-6
LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
ISO = 4.0 * np.eye(3, dtype=complex)


def scal_cell(n):
    c = np.full((n, n), 2.25 + 0j)
    c[0, 0] = 4.0
    return c


def tens_cell(n):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = LC
    c[0, 0] = ISO
    return c


def bench(fn, label):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{label:42s} wall {dt:7.3f} s   peak {peak / 2**20:8.1f} MiB")
    return out, dt, peak


def retained(sol):
    tot = 0
    inv = []
    for name in ("Lmat", "Rmat", "Stt", "Schur"):
        a = getattr(sol, name)
        tot += a.nbytes
        inv.append((name, a.nbytes))
    for i, a in enumerate(sol.Et_blocks):
        tot += a.nbytes
        inv.append((f"Et_blocks[{i}]", a.nbytes))
    if sol.Et_offdiag is not None:
        for i, a in enumerate(sol.Et_offdiag):
            tot += a.nbytes
            inv.append((f"Et_offdiag[{i}]", a.nbytes))
    for dead in ("Curl", "Kzt", "Ktz", "G3", "Meps33"):
        assert not hasattr(sol, dead), dead
    return tot, inv


for n, M in ((3, 8),):
    print(f"--- grid ({n},{n})  M={M}  (q = {n*(M-1)}, dim = {2*(n*(M-1))**2})")
    bench(lambda: pmm_efficiency_2d_staggered(P, P, scal_cell(n), 1.5, 1.0,
                                              DEP, WL, degree=M, n_orders=4),
          "SCALAR pmm_efficiency_2d_staggered")
    bench(lambda: pmm_jones_2d_staggered(P, P, tens_cell(n), 1.5, 1.0, DEP,
                                         WL, degree=M, n_orders=4),
          "TENSOR pmm_jones_2d_staggered")
    k0 = 2 * np.pi / WL
    ss = Granet2DTransverseE(P, P, n, n, M, scal_cell(n), k0=k0)
    st = Granet2DTransverseE(P, P, n, n, M, tens_cell(n), k0=k0)
    ts, invs = retained(ss)
    tt, invt = retained(st)
    print(f"  retained operator bytes: scalar {ts/2**20:.1f} MiB   "
          f"tensor {tt/2**20:.1f} MiB   ratio {tt/ts:.3f}")
    print("  scalar:", [(k, f"{v/2**20:.1f}M") for k, v in invs])
    print("  tensor:", [(k, f"{v/2**20:.1f}M") for k, v in invt])
    bench(lambda: _region_modes(ss), "  region eig SCALAR")
    bench(lambda: _region_modes(st), "  region eig TENSOR")
