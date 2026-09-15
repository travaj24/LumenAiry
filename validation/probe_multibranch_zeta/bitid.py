"""Bit-identity probe for WP-B7c -- run INSIDE a tree, prints one JSON line.

Digests the fields the two changed modules return over a fixture matrix, so the
same script can be run in a ``git archive`` tree of the parent commit and in the
worktree and the digests compared byte for byte.  Asserts ``lumenairy.__file__``
lives under the tree it was pointed at, so a stray site-packages install cannot
answer instead.

Usage (from inside the tree):  python bitid.py <tree_root> [out.json]
"""
# ruff: noqa: E402, I001  (the tree assertion runs before the fixtures import)
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings

import numpy as np

TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, TREE)
import lumenairy as la                                # noqa: E402

assert os.path.abspath(la.__file__).startswith(TREE), (
    f'lumenairy resolved to {la.__file__}, not under {TREE}')
from lumenairy.glass import GLASS_REGISTRY            # noqa: E402

GLASS_REGISTRY['_B7CCAU'] = lambda wl: 1.5168
U = la.elements._lens_traced_uniform
M = la.elements._lens_traced_multibranch


def _dig(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _gauss(N, dx, w0):
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


# --- the fixture matrix ----------------------------------------------------
K4 = {'wavelength': 1.31e-6, 'aperture_diameter': 1.4e-3, 'surfaces': [
    {'radius': 2.7e-3, 'thickness': 1.0e-3, 'glass_before': 'air',
     'glass_after': '_B7CCAU', 'semi_diameter': 0.75e-3},
    {'radius': float('inf'), 'thickness': 0.0, 'glass_before': '_B7CCAU',
     'glass_after': 'air', 'semi_diameter': 0.75e-3}],
    'thicknesses': [1.0e-3], 'stop_index': 0}

V = {'wavelength': 1.064e-6, 'aperture_diameter': 0.90e-3, 'surfaces': [
    {'radius': 2.6e-3, 'thickness': 0.70e-3, 'glass_before': 'air',
     'glass_after': 'N-BAF10', 'semi_diameter': 0.45e-3},
    {'radius': -2.6e-3, 'thickness': 0.0, 'glass_before': 'N-BAF10',
     'glass_after': 'air', 'semi_diameter': 0.45e-3}],
    'thicknesses': [0.70e-3], 'stop_index': 0}

W2 = {'wavelength': 1.55e-6, 'aperture_diameter': 1.10e-3, 'surfaces': [
    {'radius': 6.0e-3, 'thickness': 0.90e-3, 'glass_before': 'air',
     'glass_after': 'N-LAK22', 'semi_diameter': 0.55e-3},
    {'radius': -6.0e-3, 'thickness': 0.0, 'glass_before': 'N-LAK22',
     'glass_after': 'air', 'semi_diameter': 0.55e-3}],
    'thicknesses': [0.90e-3], 'stop_index': 0}

CASES = [
    # (label, fn, presc, wl, N, dx, w0, z, kwargs)
    ('K4 uniform z=4.3704mm (the K4/a3 suites own fixture)',
     'uniform', K4, 1.31e-6, 512, 3.0e-6, 0.55e-3, 4.3704e-3, {}),
    ('K4 uniform N=256 (the a3_verify dark-fill fixture)',
     'uniform', K4, 1.31e-6, 256, 3.0e-6, 0.55e-3, 4.3704e-3, {}),
    ('K4 multibranch z=4.3704mm',
     'multibranch', K4, 1.31e-6, 512, 3.0e-6, 0.55e-3, 4.3704e-3, {}),
    ('K4 uniform z=0 (exit vertex, no fold -> fallback)',
     'uniform', K4, 1.31e-6, 256, 3.0e-6, 0.55e-3, 0.0, {}),
    ('V uniform z=1565.4um (zeta_extrapolation 531)',
     'uniform', V, 1.064e-6, 512, 2.20e-6, 330e-6, 1565.4e-6, {}),
    ('V uniform z=1683.4um (zeta_extrapolation 1.03)',
     'uniform', V, 1.064e-6, 512, 2.20e-6, 330e-6, 1683.4e-6, {}),
    ('V uniform z=1750.0um (the last healthy plane)',
     'uniform', V, 1.064e-6, 512, 2.20e-6, 330e-6, 1750.0e-6, {}),
    ('V uniform z=1764.0um (the blow-up window)',
     'uniform', V, 1.064e-6, 512, 2.20e-6, 330e-6, 1764.0e-6, {}),
    ('V uniform z=1770.0um (blow-up + zeta_nonlinear fallback)',
     'uniform', V, 1.064e-6, 512, 2.20e-6, 330e-6, 1770.0e-6, {}),
    ('V multibranch z=1764.0um (the blow-up, multibranch alone)',
     'multibranch', V, 1.064e-6, 512, 2.20e-6, 330e-6, 1764.0e-6, {}),
    ('V uniform plain band z=1683.4um',
     'uniform', V, 1.064e-6, 512, 2.20e-6, 330e-6, 1683.4e-6,
     {'caustic_band': 'plain'}),
    ('W2 uniform z=4.400mm (WP-B7b fixture F, u* 17.2 < 20 cells)',
     'uniform', W2, 1.55e-6, 320, 3.85e-6, 400e-6, 4.400e-3, {}),
    ('W2 uniform z=4.450mm (u* 14.9 < 20 cells)',
     'uniform', W2, 1.55e-6, 320, 3.85e-6, 400e-6, 4.450e-3, {}),
]


def main():
    out = {'tree': TREE, 'lumenairy_file': la.__file__,
           'version': la.__version__, 'numpy': np.__version__,
           'python': sys.version.split()[0], 'cases': {}}
    for label, fn, presc, wl, N, dx, w0, z, kw in CASES:
        E = _gauss(N, dx, w0)
        call = (U.apply_real_lens_traced_uniform if fn == 'uniform'
                else M.apply_real_lens_traced_multibranch)
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter('always')
                F = call(E, prescription=presc, wavelength=wl, dx=dx,
                         output_plane_distance=z, **kw)
            out['cases'][label] = {
                'digest': _dig(F), 'shape': list(np.shape(F)),
                'power': float(np.sum(np.abs(np.asarray(F)) ** 2)) * dx * dx,
                'n_warnings': len(rec)}
        except Exception as exc:                       # noqa: BLE001
            out['cases'][label] = {
                'raised': type(exc).__name__,
                'message_head': str(exc)[:120]}
    print(json.dumps(out))
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'w', encoding='cp1252') as fh:
            json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
