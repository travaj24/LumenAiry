"""VERIFY-WAVE5-HYGIENE2 round 2 -- the two kernel arms the 342 keys miss.

``vh3_bitid.py``'s key set is built from physically reasonable carrier legs,
and two arms of ``_exact_dispersion_phase`` / ``_tf_phase_to_H`` are simply
not reached by such legs:

 E1  THE EVANESCENT CLAMP.  ``bld.maximum(rad, 0.0, out=rad)`` only bites when
     the grid resolves spatial frequencies beyond ``k``, i.e. when
     ``pi/dx > k``, i.e. ``dx < lambda/2``.  Every fixture in the key set has
     ``dx >> lambda``, so deleting the clamp moves nothing there.  Here the
     pitch is put below half a wavelength on purpose.

 E2  THE complex64 mod-2pi FOLD.  ``_tf_phase_to_H`` folds the phase modulo
     one turn in float64 BEFORE the float32 cast, on the CuPy/JAX branch only.
     It bites when ``k z`` is large enough that float32 cannot hold it; the
     key set's ``z`` are millimetres, so it does not.  Here ``z`` is metres.

Each is reported as a PAIR: the reading, and the reading with the arm's own
precondition removed, so a mutant that deletes the arm is visibly caught by
this file even though it is invisible to the main key set.

    PYTHONPATH=<tree> python vh3_edge.py OUT.json
"""
import json
import sys

import numpy as np

WL = 1.55e-6


def main(out_path):
    import lumenairy
    from lumenairy.propagators import carrier as CA

    res = {'lumenairy_file': lumenairy.__file__,
           'python': sys.version.split()[0], 'platform': sys.platform}
    k = 2.0 * np.pi / WL

    # --- E1: a pitch BELOW half a wavelength -> the band edge is crossed ---
    n = 64
    e1 = {}
    for dx in (0.25 * WL, 0.4 * WL, 2.0 * WL):
        qmax = np.pi / dx
        x = (np.arange(n) - n / 2.0) * dx
        X, Y = np.meshgrid(x, x)
        env = np.exp(-(X ** 2 + Y ** 2) / (8 * dx) ** 2).astype(np.complex128)
        qx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
        for tilt in ((0.0, 0.0), (0.3, -0.2)):
            ph = CA._exact_dispersion_phase(qx, qx, k, tilt, np, 'probe')
            out = CA._exact_envelope_tf_step(env, 1e-5, WL, dx, dx, tilt=tilt)
            e1[f'dx{dx / WL:.2f}lam.t{tilt[0]:g}_{tilt[1]:g}'] = {
                'qmax_over_k': float(qmax / k),
                'phase_has_nan': bool(np.any(~np.isfinite(ph))),
                'phase_min': float(np.min(ph)),
                'n_clamped': int(np.sum(np.asarray(ph) == -float(
                    np.sqrt(max(k * k * (1.0 - tilt[0] ** 2
                                         - tilt[1] ** 2), 0.0))))),
                'out_has_nan': bool(np.any(~np.isfinite(out))),
                'out_norm': float(np.linalg.norm(out)),
            }
    res['E1_evanescent'] = e1

    # --- E2: complex64 on the device branch with a LARGE k z ---------------
    e2 = {}
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        x = (np.arange(n) - n / 2.0) * 2e-6
        X, Y = np.meshgrid(x, x)
        env = np.exp(-(X ** 2 + Y ** 2) / (24e-6) ** 2)
        for z in (1e-3, 1.0, 100.0):
            a64 = np.asarray(CA._exact_tf_2d_xp(
                jnp.asarray(env.astype(np.complex64)), z, WL, 2e-6, 2e-6,
                (0.0, 0.0), jnp, True, np))
            a128 = np.asarray(CA._exact_tf_2d_xp(
                jnp.asarray(env.astype(np.complex128)), z, WL, 2e-6, 2e-6,
                (0.0, 0.0), jnp, True, np))
            num = float(np.linalg.norm(a64 - a128))
            den = float(np.linalg.norm(a128))
            e2[f'z{z:g}'] = {
                'kz': float(k * z),
                'kz_over_float32_eps_limit': float(
                    k * z * np.finfo(np.float32).eps),
                'dtype64': str(a64.dtype),
                'rel_c64_vs_c128': num / den if den else None,
                'c64_has_nan': bool(np.any(~np.isfinite(a64))),
            }
    except ImportError as e:
        e2['UNAVAILABLE'] = str(e)[:60]
    res['E2_complex64_fold'] = e2

    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=str)
    print(f"lumenairy.__file__ = {lumenairy.__file__}")
    for key, v in e1.items():
        print(f"E1 {key:28s} qmax/k={v['qmax_over_k']:.3f}  "
              f"nan(phase)={v['phase_has_nan']!s:5s}  "
              f"nan(out)={v['out_has_nan']!s:5s}  "
              f"|out|={v['out_norm']:.6e}")
    for key, v in e2.items():
        if isinstance(v, dict):
            print(f"E2 {key:8s} kz={v['kz']:.4e}  c64-vs-c128 "
                  f"{v['rel_c64_vs_c128']:.4e}  nan={v['c64_has_nan']}")
    print(f"-> {out_path}")


if __name__ == '__main__':
    main(sys.argv[1])
