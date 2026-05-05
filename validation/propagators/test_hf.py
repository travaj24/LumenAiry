"""HF (Van-Vleck-corrected Huygens-Fresnel) module tests."""
from __future__ import annotations

import sys

import numpy as np

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from _harness import Harness

import lumenairy as la
from lumenairy.propagators.hf import (
    propagate_huygens_fresnel_freespace,
    propagate_huygens_fresnel_with_opl_callable,
)


H = Harness('hf')


def t_freespace_synonym():
    N = 32; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    out = propagate_huygens_fresnel_freespace(E, 1e-3, lam, dx)
    return out.shape == E.shape and bool(np.all(np.isfinite(np.abs(out)))), 'ok'


def t_custom_opl_finite():
    N = 16; dx = 5e-6; lam = 633e-9; z = 1e-3
    x = (np.arange(N) - N/2 + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    w0 = 30e-6
    E = np.exp(-(X*X + Y*Y) / (w0*w0)).astype(np.complex128)
    def opl(s1x, s1y, s2x, s2y):
        return np.sqrt((s1x - s2x)**2 + (s1y - s2y)**2 + z*z) / lam
    out_x = (np.arange(8) - 4 + 0.5) * dx * 2
    out_y = (np.arange(8) - 4 + 0.5) * dx * 2
    out = propagate_huygens_fresnel_with_opl_callable(
        E, opl_fn=opl,
        output_grid_x=out_x, output_grid_y=out_y,
        input_grid_dx=dx, wavelength=lam,
        apply_van_vleck=True,
        chunk_output=8,
    )
    return out.shape == (8, 8) and bool(np.all(np.isfinite(np.abs(out)))), 'finite'


def t_van_vleck_changes_amplitude():
    N = 16; dx = 5e-6; lam = 633e-9; z = 1e-3
    x = (np.arange(N) - N/2 + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    w0 = 30e-6
    E = np.exp(-(X*X + Y*Y) / (w0*w0)).astype(np.complex128)
    def opl(s1x, s1y, s2x, s2y):
        return np.sqrt((s1x - s2x)**2 + (s1y - s2y)**2 + z*z) / lam
    out_x = (np.arange(4) - 2 + 0.5) * dx * 2
    out_y = (np.arange(4) - 2 + 0.5) * dx * 2
    out_with = propagate_huygens_fresnel_with_opl_callable(
        E, opl_fn=opl, output_grid_x=out_x, output_grid_y=out_y,
        input_grid_dx=dx, wavelength=lam, apply_van_vleck=True, chunk_output=4)
    out_without = propagate_huygens_fresnel_with_opl_callable(
        E, opl_fn=opl, output_grid_x=out_x, output_grid_y=out_y,
        input_grid_dx=dx, wavelength=lam, apply_van_vleck=False, chunk_output=4)
    return float(np.max(np.abs(out_with - out_without))) > 0, 'differs'


def main():
    H.section('Free-space HF synonym')
    H.run('freespace synonym', t_freespace_synonym)

    H.section('Custom OPL HF')
    H.run('returns finite field', t_custom_opl_finite)
    H.run('Van Vleck factor changes output', t_van_vleck_changes_amplitude)

    sys.exit(H.summary())


if __name__ == '__main__':
    main()
