"""Shared fixture for the P4 / S14.6 determinism probes.

Mirrors ``tests/unit/test_niche_p4_gbd_reexpand.py``'s
``test_frame_completeness_metric_published`` exactly: same grid, same
prescription, same two calls (diagnosed then undiagnosed).
"""
import os
import sys
import warnings

import numpy as np

_ROOT = os.environ.get('LUMENAIRY_ROOT')
if _ROOT:
    sys.path.insert(0, _ROOT)

import lumenairy  # noqa: E402

if _ROOT:
    _want = os.path.normcase(os.path.abspath(os.path.join(_ROOT, 'lumenairy')))
    _got = os.path.normcase(os.path.abspath(os.path.dirname(lumenairy.__file__)))
    if _want != _got:
        raise SystemExit(f"WRONG TREE: lumenairy at {_got}, wanted {_want}")

from lumenairy.elements.lenses_gbd import apply_real_lens_gbd  # noqa: E402
from lumenairy import glass as _glass_mod  # noqa: E402

_WL = 1.31e-6
_N_GLASS = 1.5168
_k = 2 * np.pi / _WL
_N, _DX, _W_L, _SS = 384, 10e-6, 1.0e-3, 3
_R_IN = -35e-3

_glass_mod.GLASS_REGISTRY['_P4_GLASS'] = lambda wl: _N_GLASS


def m5_biconcave():
    return {'wavelength': _WL, 'aperture_diameter': 24e-3,
            'surfaces': [
                {'radius': -51.68e-3, 'thickness': 3e-3, 'glass_before': 'air',
                 'glass_after': '_P4_GLASS', 'semi_diameter': 12e-3},
                {'radius': 51.68e-3, 'thickness': 0.0,
                 'glass_before': '_P4_GLASS', 'glass_after': 'air',
                 'semi_diameter': 12e-3}],
            'thicknesses': [3e-3], 'stop_index': 0}


def conv_input(N=_N, dx=_DX, w_L=_W_L, R_in=_R_IN):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r_sq = X ** 2 + Y ** 2
    env = np.exp(-r_sq / w_L ** 2)
    ph = (np.ones_like(env) if np.isinf(R_in)
          else np.exp(1j * _k * r_sq / (2.0 * R_in)))
    return (env * ph).astype(np.complex128)


def gbd(E, presc, *, dx=_DX, ss=_SS, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(apply_real_lens_gbd(
            E, prescription=presc, wavelength=_WL, dx=dx, sample_step=ss, **kw))


def field_hash(E):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(E).tobytes()).hexdigest()[:16]
