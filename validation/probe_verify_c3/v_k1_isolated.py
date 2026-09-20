"""VERIFY-WP-C3 CLAIM 9d -- is _collins_readout_k1 BLAS-dependent AT ALL,
given a fixed input?

    python v_k1_isolated.py <tree> <out.json>

A deterministic field (no BLAS anywhere in its construction), the same
(R, z, dx) the design-121 chain exits on, and the K1 the library computes --
printed to full precision together with the two pieces.  Run under several
OPENBLAS_CORETYPE / thread settings the reading must not move: the only
floating-point work inside is an FFT (pocketfft / pyFFTW, not BLAS), an
abs-squared, two band reductions, a cumsum and a searchsorted.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import vlib  # noqa: E402
import numpy as np  # noqa: E402

import lumenairy.propagators.carrier as CA  # noqa: E402

vlib.anchor(_TREE)

LAM = 1.31e-6
Z = 7.7058e-3
R = -0.0077124254602782
DX = 3.3211246503896516e-05
N = 1024


def main():
    out = sys.argv[2]
    # a deterministic complex field with a wide angular tail, built from
    # elementary ufuncs only -- no dot, no solve, no BLAS.
    x = (np.arange(N) - N // 2) * DX
    X, Y = np.meshgrid(x, x)
    r2 = X ** 2 + Y ** 2
    env = (np.exp(-r2 / (2.6e-3 ** 2))
           * np.exp(1j * (37.0 * X / 1e-3 + 11.0 * Y / 1e-3))
           + 3e-4 * np.exp(1j * 991.0 * (X + Y) / 1e-3)).astype(np.complex128)
    k1 = CA._collins_readout_k1(env, R, Z, LAM, DX, DX)
    r_x, r_y, th_x, th_y = CA._collins_input_box(
        env, DX, DX, LAM, CA._COLLINS_TAIL_FRAC)
    Ax, B, _, _ = CA._collins_envelope_abcd(R, Z, np.inf)
    rec = {'build': vlib.build_tag(), 'env': vlib.env_tag(), 'tree': _TREE,
           'carrier_file': CA.__file__,
           'k1': repr(float(k1)), 'k1_hex': float(k1).hex(),
           'r_x': repr(float(r_x)), 'r_x_hex': float(r_x).hex(),
           'th_x': repr(float(th_x)), 'th_x_hex': float(th_x).hex(),
           'Ax': repr(float(Ax)), 'Ax_hex': float(Ax).hex(),
           'space_term': repr(2 * DX * abs(Ax) * float(r_x) / abs(B) / LAM),
           'angle_term': repr(2 * DX * float(th_x) / LAM),
           'env_l2_hex': float(np.linalg.norm(env)).hex(),
           'fft_backend': None}
    try:
        from lumenairy.propagators import fft_infra as fi
        rec['fft_backend'] = {'pyfftw': bool(fi.PYFFTW_AVAILABLE),
                              'cupy': bool(fi.CUPY_AVAILABLE)}
    except Exception as exc:                          # noqa: BLE001
        rec['fft_backend'] = str(exc)
    vlib.write_json(rec, out)
    print('[k1iso] %s ct=%s th=%s  K1=%s (%s)  r=%s th=%s Ax=%s'
          % (rec['build'], rec['env'].get('OPENBLAS_CORETYPE'),
             rec['env'].get('OMP_NUM_THREADS'), rec['k1'], rec['k1_hex'],
             rec['r_x_hex'], rec['th_x_hex'], rec['Ax_hex']))


if __name__ == '__main__':
    main()
