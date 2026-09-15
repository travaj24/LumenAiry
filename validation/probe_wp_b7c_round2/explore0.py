"""Exploratory: the pixel-halving continuity ratio, measured OUTSIDE the
library by calling the branch sum twice (dx, dx/2 with ray_subsample doubled
so the launch pitch and the window are held fixed)."""
# ruff: noqa: E402, I001
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'probe_verify_b7c'))
import fixtures as FX                                     # noqa: E402

import lumenairy                                          # noqa: E402
from lumenairy.elements._lens_traced_multibranch import (  # noqa: E402
    apply_real_lens_traced_multibranch)


def mb(fx, z, dx, N, sub):
    E = FX.gauss(N, dx, fx['w0'])
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        Eo, d = apply_real_lens_traced_multibranch(
            E, prescription=fx['prescription'], wavelength=fx['wavelength'],
            dx=dx, output_plane_distance=z, ray_subsample=sub,
            return_diagnostics=True)
    dt = time.perf_counter() - t0
    p_out = float(np.sum(np.abs(np.asarray(Eo)) ** 2)) * dx * dx
    return p_out, d, dt


def main(argv):
    print('lumenairy', lumenairy.__file__)
    cases = [
        ('F_alt', 1076.0), ('F_alt', 1080.0), ('F_alt', 1056.0),
        ('M', 2201.74), ('M', 2343.0),
        ('S', 3214.78), ('S', 3274.02),
        ('Q', 5400.0), ('Q', 5680.0), ('Q', 5460.0),
    ]
    print(f"{'fx':7s} {'z_um':>9s} {'p1':>11s} {'p2':>11s} {'C':>8s} "
          f"{'br1':>10s} {'nbr1':>5s} {'t1':>6s} {'t2':>6s}")
    for name, zum in cases:
        fx = FX.FIXTURES[name]
        z = zum * 1e-6
        N, dx = fx['N'], fx['dx']
        p1, d1, t1 = mb(fx, z, dx, N, 2)
        p2, d2, t2 = mb(fx, z, dx / 2.0, 2 * N, 4)
        vals = [float(v) for v in (d1.get('power_ratio'),
                                   d1.get('power_ratio_triangles'))
                if v is not None and np.isfinite(v)]
        br1 = min(vals) if vals else float('nan')
        C = p1 / p2 if p2 > 0 else float('inf')
        print(f"{name:7s} {zum:9.2f} {p1:11.4e} {p2:11.4e} {C:8.4f} "
              f"{br1:10.4g} {d1.get('n_branch_max'):5d} {t1:6.2f} {t2:6.2f}",
              flush=True)


if __name__ == '__main__':
    main(sys.argv)
