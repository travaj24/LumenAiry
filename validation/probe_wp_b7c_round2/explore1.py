"""Is the healthy-plane departure of C from 1 the INPUT resampling or the
quadrature?  Same double call, but the fine grid's input field is the exact
BILINEAR refinement of the coarse one, so the launch nodes see (to rounding)
the identical amplitudes."""
# ruff: noqa: E402, I001
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'probe_verify_b7c'))
import fixtures as FX                                     # noqa: E402

import lumenairy                                          # noqa: E402
from lumenairy.elements._lens_traced_multibranch import (  # noqa: E402
    apply_real_lens_traced_multibranch)


def bilinear_refine(E):
    """(N,N) -> (2N,2N) on the half-pixel grid with the SAME centre
    convention ``x = (i - N/2) dx``: the fine sample i' sits at
    ``(i' - N) dx/2``, i.e. coarse index ``i'/2``."""
    N = E.shape[0]
    fi = np.arange(2 * N) * 0.5
    fi = np.clip(fi, 0.0, N - 1.0 - 1e-12)
    i0 = np.floor(fi).astype(int)
    w = fi - i0
    # rows then cols
    A = (1 - w)[:, None] * E[i0, :] + w[:, None] * E[np.minimum(i0 + 1,
                                                                N - 1), :]
    B = (1 - w)[None, :] * A[:, i0] + w[None, :] * A[:, np.minimum(i0 + 1,
                                                                   N - 1)]
    return B


def mb(fx, z, dx, E, sub):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        Eo, d = apply_real_lens_traced_multibranch(
            E, prescription=fx['prescription'], wavelength=fx['wavelength'],
            dx=dx, output_plane_distance=z, ray_subsample=sub,
            return_diagnostics=True)
    return float(np.sum(np.abs(np.asarray(Eo)) ** 2)) * dx * dx, d


def main(argv):
    print('lumenairy', lumenairy.__file__)
    cases = [('F_alt', 1076.0), ('F_alt', 1080.0), ('F_alt', 1056.0),
             ('M', 2201.74), ('S', 3214.78), ('Q', 5400.0), ('Q', 5680.0)]
    print(f"{'fx':7s} {'z_um':>9s} {'C_analytic':>11s} {'C_bilinear':>11s} "
          f"{'p_in_rel':>10s}")
    for name, zum in cases:
        fx = FX.FIXTURES[name]
        z = zum * 1e-6
        N, dx = fx['N'], fx['dx']
        E1 = FX.gauss(N, dx, fx['w0'])
        p1, d1 = mb(fx, z, dx, E1, 2)
        p2a, d2a = mb(fx, z, dx / 2.0, FX.gauss(2 * N, dx / 2, fx['w0']), 4)
        p2b, d2b = mb(fx, z, dx / 2.0, bilinear_refine(E1), 4)
        print(f"{name:7s} {zum:9.2f} {p1 / p2a:11.5f} {p1 / p2b:11.5f} "
              f"{d2b['launched_power'] / d1['launched_power']:10.7f}",
              flush=True)


if __name__ == '__main__':
    main(sys.argv)
