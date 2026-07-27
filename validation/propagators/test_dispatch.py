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
    expected = {'auto', 'asm', 'sas', 'fresnel', 'fraunhofer', 'rs',
                'maslov', 'asymptotic', 'gbd', 'hfpi', 'hf', 'mhs'}
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


def t_dispatch_mhs():
    """Dispatcher routes method='mhs' to MhsPipeline.run."""
    N = 16; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    s0 = la.HuygensSurface(z=0.0, Ny=N, Nx=N, dx=dx, label='in')
    s1 = la.HuygensSurface(z=1e-3, Ny=N, Nx=N, dx=dx, label='out')
    sub = la.asm_subdomain(s0, s1, wavelength=lam)
    out = propagate(E, dx=dx, wavelength=lam, method='mhs',
                    subdomains=[sub])
    return (out.shape == E.shape
            and bool(np.all(np.isfinite(np.abs(out))))), 'mhs dispatch ok'


def t_dispatch_mhs_requires_subdomains():
    """method='mhs' without subdomains/pipeline raises."""
    E = np.ones((8, 8), dtype=np.complex128)
    try:
        propagate(E, dx=5e-6, wavelength=633e-9, method='mhs')
        return False, 'should raise'
    except ValueError:
        return True, 'mhs without subdomains raises'


def t_propagate_return_result():
    """propagate(return_result=True) returns a PropagationResult."""
    N = 16; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    out = propagate(E, z=1e-3, wavelength=lam, dx=dx, method='asm',
                    return_result=True)
    return (isinstance(out, la.PropagationResult)
            and out.shape == E.shape
            and out.method == 'asm'
            and out.wavelength == lam), (
        f'type={type(out).__name__}, method={getattr(out, "method", None)!r}')


def t_propagate_default_returns_result():
    """v5.30 (audit P5 / roadmap F1): propagate()'s DEFAULT return is the
    stable PropagationResult -- for every method, not just this one."""
    N = 16; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    out = propagate(E, z=1e-3, wavelength=lam, dx=dx, method='asm')
    sas = propagate(E, z=1e-3, wavelength=lam, dx=5e-7, method='sas')
    return (isinstance(out, la.PropagationResult)
            and out.shape == E.shape
            and isinstance(sas, la.PropagationResult)), (
        f'asm={type(out).__name__}, sas={type(sas).__name__}')


def t_propagate_false_returns_native_shapes():
    """The permanent escape hatch: return_result=False keeps the kernels'
    native shapes -- bare ndarray for ASM, (E, dx_out, dy_out) for SAS."""
    N = 16; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    bare = propagate(E, z=1e-3, wavelength=lam, dx=dx, method='asm',
                     return_result=False)
    triple = propagate(E, z=1e-3, wavelength=lam, dx=5e-7, method='sas',
                       return_result=False)
    return (isinstance(bare, np.ndarray) and bare.shape == E.shape
            and isinstance(triple, tuple) and len(triple) == 3), (
        f'asm={type(bare).__name__}, sas={type(triple).__name__}')


def t_auto_aspheric_picks_gbd():
    """_auto_select_method routes aspheric prescriptions to GBD."""
    presc = {
        'name': 'asph',
        'aperture_diameter': 4e-3,
        'surfaces': [
            {'radius': 25e-3, 'conic': 0.0,
             'aspheric_coeffs': {4: 1e6},
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': float('inf'), 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [4e-3],
    }
    E = np.ones((16, 16), dtype=np.complex128)
    m = _auto_select_method(E, z=None, wavelength=633e-9, dx=5e-6,
                              prescription=presc)
    return m == 'gbd', f'got {m!r}'


def t_auto_accuracy_accurate_picks_hf():
    """accuracy='accurate' on a plain singlet picks HF (hard aperture)."""
    presc = la.make_singlet(R1=5e-3, R2=float('inf'), d=2e-3,
                              glass='N-BK7', aperture=4e-3)
    E = np.ones((16, 16), dtype=np.complex128)
    m = _auto_select_method(E, z=None, wavelength=633e-9, dx=5e-6,
                              prescription=presc, accuracy='accurate')
    return m == 'hf', f'got {m!r}'


def t_auto_balanced_plain_singlet_picks_maslov():
    """accuracy='balanced' on a plain singlet stays with maslov."""
    presc = la.make_singlet(R1=5e-3, R2=float('inf'), d=2e-3,
                              glass='N-BK7', aperture=4e-3)
    E = np.ones((16, 16), dtype=np.complex128)
    m = _auto_select_method(E, z=None, wavelength=633e-9, dx=5e-6,
                              prescription=presc, accuracy='balanced')
    return m == 'maslov', f'got {m!r}'


def t_propagation_result_array_protocol():
    """np.asarray(result) returns the field, no copy needed."""
    N = 16; dx = 5e-6; lam = 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    out = propagate(E, z=1e-3, wavelength=lam, dx=dx, method='asm',
                    return_result=True)
    arr = np.asarray(out)
    return arr.shape == (N, N) and np.iscomplexobj(arr), (
        f'shape={arr.shape}, dtype={arr.dtype}')


def t_propagation_result_tuple_unpack():
    """`field, intermediates = result` works as a backward-compat
    drop-in for `propagate_through_system`."""
    elements = [{'type': 'propagate', 'z': 1e-3}]
    res = la.propagate_through_system(
        np.ones((16, 16), dtype=np.complex128),
        elements, 633e-9, 5e-6, return_result=True)
    field, intermediates = res
    return (field.shape == (16, 16)
            and isinstance(intermediates, list)
            and len(intermediates) >= 1), (
        f'field.shape={field.shape}, '
        f'len(intermediates)={len(intermediates)}')


def t_propagation_result_to_source_round_trip():
    """PropagationResult.to_source() preserves dx and wavelength."""
    N = 16; dx = 5e-6; lam = 633e-9
    out = propagate(np.ones((N, N), dtype=np.complex128),
                     z=1e-3, wavelength=lam, dx=dx, method='asm',
                     return_result=True)
    src = out.to_source()
    return (src.dx == out.dx
            and src.wavelength == out.wavelength
            and src.shape == out.shape), (
        f'dx: {out.dx} -> {src.dx}, '
        f'wavelength: {out.wavelength} -> {src.wavelength}')


def t_auto_select_empty_surfaces_falls_back():
    """A prescription with empty surfaces should not crash auto-select."""
    presc = {
        'surfaces': [],
        'thicknesses': [],
        'aperture_diameter': 4e-3,
    }
    E = np.ones((16, 16), dtype=np.complex128)
    # Just verify it returns *some* string without raising.
    m = _auto_select_method(E, z=None, wavelength=633e-9, dx=5e-6,
                              prescription=presc)
    return isinstance(m, str), f'got {m!r}'


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
    H.run('MHS dispatch', t_dispatch_mhs)

    H.section('Smarter auto-selection')
    H.run('aspheric prescription picks GBD',
          t_auto_aspheric_picks_gbd)
    H.run('accuracy=accurate + hard aperture picks HF',
          t_auto_accuracy_accurate_picks_hf)
    H.run('plain singlet stays with maslov',
          t_auto_balanced_plain_singlet_picks_maslov)

    H.section('PropagationResult opt-in + interop')
    H.run('propagate(return_result=True) returns PropagationResult',
          t_propagate_return_result)
    H.run('propagate() default returns PropagationResult (v5.30 F1 flip)',
          t_propagate_default_returns_result)
    H.run('propagate(return_result=False) returns the native shapes',
          t_propagate_false_returns_native_shapes)
    H.run('np.asarray(result) returns the field array',
          t_propagation_result_array_protocol)
    H.run('result tuple-unpacks as (field, intermediates)',
          t_propagation_result_tuple_unpack)
    H.run('result.to_source() preserves dx/wavelength/shape',
          t_propagation_result_to_source_round_trip)

    H.section('Auto-select edge cases')
    H.run('empty surfaces list does not crash auto-select',
          t_auto_select_empty_surfaces_falls_back)

    H.section('Error handling')
    H.run('invalid method raises ValueError', t_invalid_method_raises)
    H.run('MHS without subdomains raises', t_dispatch_mhs_requires_subdomains)

    sys.exit(H.summary())


if __name__ == '__main__':
    main()
