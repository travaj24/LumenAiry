"""VERIFY-B14 arm 5b -- the stacklevel sweep moved no VALUES.

Runs a battery of numeric entry points through the three swept modules and
prints an md5 of the returned array's raw bytes plus the array's dtype and
shape.  Run against two ``git archive`` trees with ``PYTHONPATH`` pinned and
diff the JSON: any differing digest is a value change, which a warning-
attribution sweep must not produce.
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings

import numpy as np

import lumenairy as la
from lumenairy.propagators import carrier as CA, carrier_field as CF, system as SY


def _md5(a):
    arr = np.ascontiguousarray(np.asarray(a))
    return hashlib.md5(arr.tobytes()).hexdigest()


def _env(n=64, dx=8e-6, w=60e-6, seed=None):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    E = np.exp(-(X ** 2 + Y ** 2) / (w ** 2)).astype(np.complex128)
    if seed is not None:
        rng = np.random.default_rng(seed)
        E = E * np.exp(1j * 0.3 * rng.normal(size=E.shape))
    return E


def _cf(env, dx, wl, R=-0.05):
    return CF.CarrierField(
        envelope=env.copy(),
        grid=CF.FieldGrid(shape=env.shape, dx=dx, dy=dx, origin=(0.0, 0.0)),
        carrier=CF.CarrierSpec(R=R, centre=(0.0, 0.0), tilt=(0.0, 0.0),
                               piston=0.0),
        wavelength=wl)


def _cases():
    dx, wl = 8e-6, 633e-9
    env = _env()
    envn = _env(seed=7)
    big = _env(n=128, dx=4e-6, w=120e-6)
    return {
        'carrier.propagate_referenced_asm':
            lambda: CA.propagate_carrier_referenced(
                env, -0.05, 5e-3, wavelength=wl, dx=dx),
        'carrier.propagate_referenced_fresnel_tilt':
            lambda: CA.propagate_carrier_referenced(
                env, -0.05, 5e-3, wavelength=wl, dx=dx,
                gap_kernel='fresnel', tilt=(0.12, 0.0)),
        'carrier.propagate_referenced_noisy':
            lambda: CA.propagate_carrier_referenced(
                envn, -0.05, 5e-3, wavelength=wl, dx=dx),
        'carrier.referenced_envelope':
            lambda: CA.carrier_referenced_envelope(
                envn, -0.05, wavelength=wl, dx=dx),
        'carrier.focus_readout':
            lambda: CA.carrier_referenced_focus_readout(
                env, -0.05, 5e-3, wavelength=wl, dx=dx, dx_out=dx,
                N_out=32),
        'carrier.focus_readout_replica_warn':
            lambda: CA.carrier_referenced_focus_readout(
                env, -0.05, 5e-3, wavelength=wl, dx=dx, dx_out=dx * 8.0,
                N_out=256, on_replica='warn', on_focus_containment='warn'),
        'carrier.exact_focus_readout':
            lambda: CA.carrier_referenced_exact_focus_readout(
                big, -0.05, 4.9e-2, wavelength=wl, dx=4e-6,
                dx_out=4e-6 * 40, N_out=256, on_readout_window='warn',
                on_replica='warn', on_n_fine_cap='warn', n_fine_cap=64),
        'carrier.fit_radius':
            lambda: np.asarray(CA.carrier_referenced_fit_radius(
                envn, wavelength=wl, dx=dx, on_aliased='warn')),
        'carrier_field.re_reference':
            lambda: CF.re_reference(
                _cf(envn, dx, wl),
                CF.CarrierSpec(R=-0.06, centre=(0.0, 0.0), tilt=(0.0, 0.0),
                               piston=0.0),
                CF.FieldGrid(shape=(64, 64), dx=dx, dy=dx,
                             origin=(0.0, 0.0)),
                on_nyquist='warn', on_window='warn').envelope,
        'system.asm_leg':
            lambda: SY.propagate_through_system(
                envn, elements=[{'type': 'propagate', 'z': 0.02}],
                wavelength=wl, dx=dx),
        'system.fresnel_leg':
            lambda: SY.propagate_through_system(
                envn, elements=[{'type': 'propagate', 'z': 0.5,
                                 'method': 'fresnel'}],
                wavelength=wl, dx=dx),
        'system.lens_then_leg':
            lambda: SY.propagate_through_system(
                envn, elements=[{'type': 'lens', 'f': 0.05},
                                {'type': 'propagate', 'z': 0.05}],
                wavelength=wl, dx=dx),
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'v5b.json'
    res = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'cases': {}}
    for name, fn in _cases().items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                out = fn()
                if isinstance(out, tuple):
                    out = out[0]
                arr = np.asarray(out)
                res['cases'][name] = {'md5': _md5(arr),
                                      'shape': list(arr.shape),
                                      'dtype': str(arr.dtype)}
            except Exception as exc:
                res['cases'][name] = {
                    'err': type(exc).__name__ + ': ' + str(exc)[:140]}
        print('%-42s %s' % (name, res['cases'][name]), flush=True)
    print('lumenairy:', la.__file__)
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
