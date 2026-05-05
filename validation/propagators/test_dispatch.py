"""Dispatcher (top-level propagate) tests."""
from __future__ import annotations

import sys

import numpy as np

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from _harness import Harness

import lumenairy as la
from lumenairy.propagators.dispatch import propagate, VALID_METHODS, _auto_select_method


H = Harness('dispatch')


def t_valid_methods():
    expected = {'auto', 'asm', 'fresnel', 'fraunhofer', 'rs',
                'maslov', 'asymptotic', 'gbd', 'hfpi'}
    return set(VALID_METHODS) == expected, f'methods = {VALID_METHODS}'


def t_auto_freespace_picks_asm():
    N = 64; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    return _auto_select_method(E, z=1e-3, wavelength=lam, dx=dx,
                               prescription=None) == 'asm', 'asm'


def t_auto_far_field_picks_fraunhofer():
    N = 64; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    return _auto_select_method(E, z=1e6, wavelength=lam, dx=dx,
                               prescription=None) == 'fraunhofer', 'fraunhofer'


def t_dispatch_asm():
    N = 64; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    out = propagate(E, z=1e-3, wavelength=lam, dx=dx, method='asm')
    return out.shape == E.shape and np.iscomplexobj(out), 'asm dispatch'


def t_dispatch_gbd():
    N = 32; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    out = propagate(E, z=1e-3, wavelength=lam, dx=dx, method='gbd',
                    sample_step=4, chunk_beamlets=256)
    return out.shape == E.shape and bool(np.all(np.isfinite(np.abs(out)))), 'gbd ok'


def t_dispatch_hfpi():
    N = 16; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    out = propagate(E, wavelength=lam, dx=dx, method='hfpi',
                    z_to_aperture=1e-3, aperture_radius=200e-6,
                    z_aperture_to_output=1e-3, n_paths=2000, rng=42)
    return out.shape == E.shape and bool(np.all(np.isfinite(np.abs(out)))), 'hfpi ok'


def t_invalid_method_raises():
    N = 8; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    try:
        propagate(E, z=1e-3, wavelength=lam, dx=dx, method='nonsense')
        return False, 'should raise'
    except ValueError:
        return True, 'invalid raises'


def main():
    H.section('Method registry')
    H.run('VALID_METHODS expected set', t_valid_methods)

    H.section('Auto selection')
    H.run('free-space near-field picks ASM', t_auto_freespace_picks_asm)
    H.run('far-field picks Fraunhofer', t_auto_far_field_picks_fraunhofer)

    H.section('Per-method dispatch')
    H.run('ASM dispatch', t_dispatch_asm)
    H.run('GBD dispatch', t_dispatch_gbd)
    H.run('HFPI dispatch', t_dispatch_hfpi)

    H.section('Error handling')
    H.run('invalid method raises ValueError', t_invalid_method_raises)

    sys.exit(H.summary())


if __name__ == '__main__':
    main()
