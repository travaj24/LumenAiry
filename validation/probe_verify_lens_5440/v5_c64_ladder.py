"""V5 -- the complex64 phasor precision ladder, per helper.

Three builds of the SAME unit phasor, compared against a complex128
float64-argument reference:

  c64      -- the shipped v5.44 path: float64 ARGUMENT, complex64 STORAGE
  f32arg   -- the control the CHANGELOG cites: float32 ARGUMENT
  c128     -- dtype=None / complex128 (must be byte-identical to the ref)

swept over the phase argument from ~1e+02 to ~1e+05 rad.  The claim under
test: the c64 error is FLAT in the argument at ~one float32 rounding
(4.2e-08 claimed, analytic ceiling sqrt(2) * eps32/2 = 8.43e-08) while the
f32arg control grows linearly with it (9.8e-04 at 3.3e+04 rad claimed).

Usage:  python v5_c64_ladder.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fix                                                    # noqa: E402

WL = _fix.WL
K = 2 * np.pi / WL
N, DX = 512, 1.9e-6
RMAX = (N / 2) * DX
TARGETS = [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5]


def main():
    la = _fix.banner()
    from lumenairy.propagators import carrier as C
    out = {'version': la.__version__, 'N': N, 'dx': DX,
           'eps32_ceiling': float(np.sqrt(2) * np.finfo(np.float32).eps / 2),
           'ladder': []}
    x = (np.arange(N) - N / 2) * DX
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    for A in TARGETS:
        R = K * RMAX ** 2 / (2.0 * A)
        L, M = 0.0515, -0.02
        row = {'target_arg': A, 'R': R}
        # --- the four helpers, c64 vs c128 vs a float32-ARGUMENT control ---
        specs = {
            '_radial_carrier_phase': (
                lambda dt: C._radial_carrier_phase((N, N), DX, DX, WL, R, +1,
                                                   dtype=dt),
                lambda: np.exp(1j * (K * r2 / (2.0 * R)).astype(np.float32))),
            '_tilt_ramp': (
                lambda dt: C._tilt_ramp((N, N), DX, WL, L, M, 0.0, 0.0, -1,
                                        dtype=dt),
                lambda: np.exp(-1j * (K * (L * x[None, :] + M * x[:, None])
                                      ).astype(np.float32))),
            '_tilt_exactness_phase': (
                lambda dt: C._tilt_exactness_phase((N, N), DX, DX, WL, R,
                                                   L, M, +1, dtype=dt), None),
            '_sphere_parab_conversion': (
                lambda dt: C._sphere_parab_conversion((N, N), DX, WL, R, +1,
                                                      dtype=dt), None),
        }
        for name, (fn, naive) in specs.items():
            ref = fn(np.complex128)
            if ref is None:
                row[name] = None
                continue
            none_ = fn(None)
            got = fn(np.complex64)
            e_c64 = float(np.abs(ref - got.astype(np.complex128)).max())
            arg_max = float(np.abs(np.angle(ref)).max())   # wrapped; report R
            rec = {'c128_equals_none': bool(np.array_equal(ref, none_)),
                   'c64_dtype': str(got.dtype),
                   'err_c64': e_c64,
                   'arg_wrapped_max': arg_max}
            if naive is not None:
                nv = naive()
                rec['err_f32arg'] = float(
                    np.abs(ref - nv.astype(np.complex128)).max())
                rec['ratio'] = (rec['err_f32arg'] / e_c64 if e_c64 > 0
                                else float('inf'))
            row[name] = rec
        # true (unwrapped) max argument of the radial phase
        row['radial_arg_max_rad'] = float((K * r2 / (2.0 * R)).max())
        out['ladder'].append(row)
        r = row['_radial_carrier_phase']
        print(f"arg~{A:8.3g} rad (true {row['radial_arg_max_rad']:9.4g}): "
              f"c64 {r['err_c64']:.3e}  f32arg {r['err_f32arg']:.3e}  "
              f"ratio {r['ratio']:8.1f}  c128==none {r['c128_equals_none']}",
              flush=True)
        rt = row['_tilt_ramp']
        print(f"{'':30s}tilt_ramp c64 {rt['err_c64']:.3e} "
              f"f32arg {rt['err_f32arg']:.3e}", flush=True)
        for nm in ('_tilt_exactness_phase', '_sphere_parab_conversion'):
            if row[nm]:
                print(f"{'':30s}{nm} c64 {row[nm]['err_c64']:.3e} "
                      f"c128==none {row[nm]['c128_equals_none']}", flush=True)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
