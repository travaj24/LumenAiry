"""V7 -- the COST claims (build doc T10): "1.33 - 2.03x the in-plane region
solve and ~3.0x its peak working set".

Timing is INTERLEAVED (the two arms alternate inside one loop) so a machine
drift cannot be read as a formulation difference, and only RATIOS are reported.
Peak allocation is measured in a separate pass with ``tracemalloc`` (its
bookkeeping perturbs the timings).

In-plane arm = the SAME cell with the four out-of-plane slots zeroed, so the
two arms differ only in the formulation, exactly as the build's T10 does.

Run this ALONE (no other probe in flight).

Usage:  PYTHONPATH=<root> python v7_cost.py <root> <out.json>
"""
import json
import os
import statistics
import sys
import time
import tracemalloc

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
assert os.path.abspath(lumenairy.__file__).startswith(ROOT), (
    f"lumenairy.__file__ = {lumenairy.__file__} is not under {ROOT}")

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _region_modes,
    _region_modes_oop,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

OUT = sys.argv[2]
WL = 1.0e-6
PX = 1.2e-6
K0 = 2.0 * np.pi / WL
T33 = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))


def cells(n):
    oop = np.broadcast_to(T33, (n, n, 3, 3)).copy()
    ip = oop.copy()
    ip[..., 0, 2] = ip[..., 1, 2] = ip[..., 2, 0] = ip[..., 2, 1] = 0.0
    return ip, oop


def solve_inplane(cell, n, M):
    s = Granet2DTransverseE(PX, PX, n, n, M, cell, alpha0x=0.3 * K0,
                            alpha0y=0.2 * K0, k0=K0)
    assert not s.offplane
    W, V, lam, g2 = _region_modes(s)
    return s.dimtot, W.shape


def solve_oop(cell, n, M):
    s = Granet2DTransverseE(PX, PX, n, n, M, cell, alpha0x=0.3 * K0,
                            alpha0y=0.2 * K0, k0=K0)
    assert s.offplane
    six = _region_modes_oop(s)
    return s.dimtot, six[0].shape


out = {"root": ROOT, "lumenairy": lumenairy.__file__,
       "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
       "timing": [], "memory": []}

for n, M, reps in ((2, 8, 5), (3, 8, 3), (3, 6, 5)):
    ip, oop = cells(n)
    ti, to = [], []
    # warm up both arms once (BLAS / cache), then interleave
    solve_inplane(ip, n, M)
    solve_oop(oop, n, M)
    for _ in range(reps):
        t0 = time.perf_counter()
        dim_i, _ = solve_inplane(ip, n, M)
        t1 = time.perf_counter()
        dim_o, _ = solve_oop(oop, n, M)
        t2 = time.perf_counter()
        ti.append(t1 - t0)
        to.append(t2 - t1)
    rec = {"grid": n, "M": M, "reps": reps, "dim_inplane": dim_i,
           "dim_oop": dim_o,
           "t_inplane_min": min(ti), "t_oop_min": min(to),
           "t_inplane_median": statistics.median(ti),
           "t_oop_median": statistics.median(to),
           "ratio_min": min(to) / min(ti),
           "ratio_median": statistics.median(to) / statistics.median(ti)}
    out["timing"].append(rec)
    print(f"[time] ({n},{n}) M={M}  dim {dim_i} -> {dim_o}  "
          f"ratio_min={rec['ratio_min']:.3f} "
          f"ratio_median={rec['ratio_median']:.3f}", flush=True)

for n, M in ((2, 8), (3, 8), (3, 6)):
    ip, oop = cells(n)
    peaks = {}
    for tag, fn, c in (("inplane", solve_inplane, ip), ("oop", solve_oop, oop)):
        tracemalloc.start()
        tracemalloc.reset_peak()
        fn(c, n, M)
        _cur, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks[tag] = pk
    rec = {"grid": n, "M": M,
           "peak_inplane_MB": peaks["inplane"] / 2 ** 20,
           "peak_oop_MB": peaks["oop"] / 2 ** 20,
           "peak_ratio": peaks["oop"] / peaks["inplane"]}
    out["memory"].append(rec)
    print(f"[mem]  ({n},{n}) M={M}  {rec['peak_inplane_MB']:.1f} MB -> "
          f"{rec['peak_oop_MB']:.1f} MB  ratio={rec['peak_ratio']:.3f}",
          flush=True)

json.dump(out, open(OUT, "w"), indent=1)
print("DONE")
