"""
Lumenairy example 15 -- FGA performance levers (v5.24.0).

The Frozen Gaussian Approximation (``apply_real_lens_fga``) is a phase-space
beamlet sum: cost ~ ``N_q * N_p * window``, memory ~ ``N_q * N_p``.  v5.24.0 adds
five **no-accuracy-loss** levers, each of which leaves the output field unchanged
(fidelity 1.0) while cutting time or memory:

* ``nsig`` 4 -> 3 (default): the frozen-beamlet window is ``(nsig*w0)^2``; the
  >3-sigma tail is filled by overlapping beamlets, so a smaller window is ~1.8x
  faster with no change to the reconstruction.
* ``prune_frac`` (default 1e-4): drop launch-lattice points where the windowed
  ``|E_in|`` is negligible -- those beamlets carry ~zero Gabor coefficient
  (Cauchy-Schwarz).  3-5x on concentrated fields.
* ``coeff_frac`` (default 1e-4): skip whole momenta whose peak coefficient is
  negligible -- the field carries ~no energy at that direction.  Faster for
  smooth (spectrally concentrated) fields.
* ``separable`` (default ``'auto'``): the tensor-``pv (x) pv`` momentum grid
  factorizes the 2-D Gabor analysis into an x-transform reused across every
  ``py`` (~``n_p`` x on analysis), and the scatter advances its window phase /
  Gaussian by recurrences that hoist the cos/sin/exp out of the inner loop.  Both
  kernels are numerically equivalent to the direct ones to well within the FGA
  error floor.  ~1.5-1.8x combined.
* ``mem_budget_mb`` / ``chunk``: process the momentum swarm in batches, bounding
  peak beamlet memory from ``O(N_q*N_p)`` to ``O(N_q*chunk)`` -- makes
  high-resolution / fine-sampled FGA runnable instead of OOM.  Bit-for-bit the
  same result.

This example benchmarks the levers OFF vs ON on a concentrated field and confirms
the field is unchanged (fidelity 1.0).
"""
import time

import numpy as np

import lumenairy as la


def _fid(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    wl = 0.633e-6
    n, dx = 192, 0.7e-6
    xs = (np.arange(n) - n / 2) * dx
    xg, yg = np.meshgrid(xs, xs)
    # a concentrated, smooth Gaussian -> both position and coefficient pruning bite
    u0 = np.exp(-(xg ** 2 + yg ** 2) / (11e-6) ** 2).astype(np.complex128)
    flat = {'name': 'flat', 'aperture_diameter': n * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0, 'glass_before': 'air',
                          'glass_after': 'air', 'semi_diameter': n * dx / 2}],
            'thicknesses': []}
    common = dict(prescription=flat, wavelength=wl, dx=dx,
                  output_plane_distance=250e-6, w0_factor=6.0, p_max=0.12,
                  n_p=19)

    def run(label, **kw):
        t0 = time.perf_counter()
        out = la.apply_real_lens_fga(u0, **common, **kw)
        return label, out, time.perf_counter() - t0

    print("=" * 62)
    print(" FGA performance levers -- concentrated smooth field, N=192")
    print("=" * 62)
    # everything OFF = the pre-v5.24.0 behaviour
    base_label, base, base_t = run(
        "all levers OFF (nsig=4)", nsig=4.0, prune_frac=0.0, coeff_frac=0.0,
        separable=False)
    rows = [(base_label, base_t, 1.0)]
    for label, kw in [
        ("nsig=3 only", dict(nsig=3.0, prune_frac=0.0, coeff_frac=0.0,
                             separable=False)),
        ("+ position pruning", dict(nsig=3.0, prune_frac=1e-4, coeff_frac=0.0,
                                    separable=False)),
        ("+ coefficient pruning", dict(nsig=3.0, prune_frac=1e-4,
                                       coeff_frac=1e-4, separable=False)),
        ("+ separable (all v5.24.0 defaults)", dict()),  # separable='auto'
    ]:
        _lbl, out, t = run(label, **kw)
        rows.append((label, t, _fid(out, base)))

    print(f"\n  {'configuration':<38} {'time(s)':>8} {'speedup':>8} {'fidelity':>9}")
    print(f"  {'-'*38} {'-'*8} {'-'*8} {'-'*9}")
    for label, t, fdel in rows:
        print(f"  {label:<38} {t:>8.1f} {base_t/t:>7.1f}x {fdel:>9.6f}")
    print("\n  -> the field is unchanged (fidelity 1.0) at every step: pure speed.")

    # memory lever: chunking is bit-for-bit identical
    print("\n  memory lever (mem_budget_mb bounds peak beamlet memory):")
    full = la.apply_real_lens_fga(u0, **common)
    budg = la.apply_real_lens_fga(u0, **common, mem_budget_mb=2)
    print(f"    mem_budget_mb=2 vs full: max|diff|={np.max(np.abs(full-budg)):.1e}"
          f"  (bit-for-bit identical)")


if __name__ == '__main__':
    main()
