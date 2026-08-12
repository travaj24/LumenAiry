# SUM-AT-APERTURE probe -- how much of one exact leg is SHAREABLE across
# frames, and how much is irreducibly per-frame.
#
# The architecture's whole cost claim rests on this split: everything the
# readout does BEFORE the final Bluestein inverse (reference the sphere, crop /
# upsample, reconstruct, forward FFT, transfer function) depends only on the
# summed field, so it could in principle be computed ONCE for all K frames;
# the Bluestein inverse onto frame i's window cannot.  The shipped library has
# no entry point that exposes that split -- ``carrier_referenced_exact_focus_
# readout`` does the whole thing per call -- so this script measures what such
# an entry point would be worth.
#
# The timing is on a SYNTHETIC field of the production shape: every operation
# here is data-independent (FFTs, chirps, elementwise phases), and using
# random data keeps the measurement off the probe's own cached arrays.
#
#   python sumap_legsplit_121.py [--n 16384] [--dx 0.6146e-6] [--reps 2]
import argparse
import time

import numpy as np

import lumenairy.propagators._bluestein as BL
import lumenairy.propagators.carrier as CAR

LAM = 1.31e-6
R = -0.0077124254602782
Z = 7.7058e-3
DXO = 0.2e-6
TILE = 1024


def timed_leg(n, dx, seed=0):
    rng = np.random.default_rng(seed)
    # a smooth converging beam of the right radius, so the readout's own
    # sizing arithmetic lands where production lands
    x = (np.arange(n) - n / 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    amp = np.exp(-r2 / (2 * (1.1844e-3) ** 2))
    E = (amp * np.exp(1j * 2 * np.pi / LAM
                      * CAR._exact_sphere_eikonal((n, n), dx, dx, LAM, R))
         ).astype(np.complex128)
    E += (1e-6 * rng.standard_normal((n, n))).astype(np.complex128)
    t = {}
    o_bl = BL._bluestein_centred_2d

    def _bl(*a, **kw):
        t0 = time.perf_counter()
        out = o_bl(*a, **kw)
        t['bluestein'] = t.get('bluestein', 0.0) + time.perf_counter() - t0
        return out

    BL._bluestein_centred_2d = _bl
    try:
        t0 = time.perf_counter()
        CAR.carrier_referenced_exact_focus_readout(
            E, R, Z, LAM, dx, dx_out=DXO, N_out=TILE, N_fine=n,
            window_factor=1e6, centre=(0.0, 0.0), tilt=(0.0, 0.0),
            centre_out=(0.0, 0.0), ram_budget=float('inf'),
            on_replica='warn', on_readout_window='warn')
        t['total'] = time.perf_counter() - t0
    finally:
        BL._bluestein_centred_2d = o_bl
    t['shareable'] = t['total'] - t['bluestein']
    return t


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=16384)
    ap.add_argument('--dx', type=float, default=0.6146e-6)
    ap.add_argument('--reps', type=int, default=2)
    a = ap.parse_args()
    print(f"exact-leg split at N_fine={a.n}, dx_fine={a.dx * 1e6:.4f} um, "
          f"readout {TILE} x {DXO * 1e6:.3f} um")
    for i in range(a.reps):
        t = timed_leg(a.n, a.dx, seed=i)
        print(f"  rep {i}: total {t['total']:7.2f} s = shareable "
              f"{t['shareable']:7.2f} s + per-frame Bluestein "
              f"{t['bluestein']:7.2f} s   "
              f"({t['bluestein'] / t['total'] * 100:.1f} % per-frame)",
              flush=True)
