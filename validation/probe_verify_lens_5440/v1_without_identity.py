"""V1 -- WITHOUT-ARM BIT IDENTITY where nothing should move.

Every NON-banded traced call (``sag_chunk_rows=0`` and ``None`` at N < 4096,
which AUTO-resolves to whole-grid) on 12 fixtures, hashed.  Run on the
v5.43.0 worktree and on 50824e9 and diff the JSON: bit-identical is REQUIRED
everywhere except where the CHANGELOG names a difference.

Usage:  python v1_without_identity.py <out.json>
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import _fix  # noqa: E402

N, DX = 320, 14e-6
W = 0.9e-3
SUB = 4


def _base(**over):
    kw = dict(prescription=_fix.presc_singlet(ap=4.0e-3), wavelength=_fix.WL,
              dx=DX, ray_subsample=SUB, n_workers=1, on_undersample='silent',
              on_noncollimated='off', on_aperture_beam='silent',
              parallel_amp=False)
    kw.update(over)
    return kw


def _sph(n, dx, w, R, x0=0.0, y0=0.0):
    """A DIVERGING (or converging) Gaussian with a real spherical carrier --
    the state the carrier/evaluator machinery is actually driven in."""
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    k = 2 * np.pi / _fix.WL
    return (np.exp(-r2 / w ** 2) * np.exp(1j * k * r2 / (2.0 * R))
            ).astype(np.complex128)


def fixtures():
    """(name, E-builder, kwargs).  Both amplitude models, preserve / remap
    (both samplings), aperture, piston (automatic), origin, a decentred
    congruence, a tilt-aware call, both inversion routes, sub=1."""
    E = lambda: _fix.gauss(N, DX, W)                           # noqa: E731
    Edec = lambda: _fix.gauss(N, DX, W, x0=0.35e-3, y0=-0.25e-3)   # noqa: E731
    Esph = lambda: _sph(N, DX, W, 0.060)                       # noqa: E731
    return [
        ('f01_screen_preserve', E, _base()),
        ('f02_screen_carrier', Esph, _base(carrier=0.060)),
        ('f03_rd_carrier', Esph,
         _base(carrier=0.060, amplitude_model='ray_density')),
        ('f04_rd_remap_lattice', Esph,
         _base(carrier=0.060, amplitude_model='ray_density',
               preserve_input_phase='remap', remap_sampling='lattice')),
        ('f05_rd_remap_full', Esph,
         _base(carrier=0.060, amplitude_model='ray_density',
               preserve_input_phase='remap', remap_sampling='full')),
        ('f06_rd_preserve_collim', E,
         _base(amplitude_model='ray_density')),
        ('f07_rd_remap_full_collim', E,
         _base(amplitude_model='ray_density',
               preserve_input_phase='remap', remap_sampling='full')),
        # origin is implemented ONLY for ray_density + remap
        ('f08_rd_origin', Edec,
         _base(amplitude_model='ray_density',
               preserve_input_phase='remap', remap_sampling='full',
               origin=(0.35e-3, -0.25e-3))),
        ('f09_rd_tilt_aware', Edec,
         _base(amplitude_model='ray_density', tilt_aware_rays=True,
               preserve_input_phase='remap', remap_sampling='lattice',
               beam_centre=(0.35e-3, -0.25e-3))),
        ('f10_screen_noinv', Esph, _base(carrier=0.060, inverse_map=False)),
        ('f11_rd_noinv', Esph,
         _base(carrier=0.060, amplitude_model='ray_density',
               inverse_map=False)),
        # caustic-bearing: the evaluator's own G2 guard refuses a folded map,
        # so this exercises the coarse-Newton route WITH the fold warning
        ('f12_rd_caustic', lambda: _fix.gauss(N, DX, 1.6e-3),
         _base(prescription=_fix.presc_strong(ap=4.0e-3),
               amplitude_model='ray_density')),
        ('f13_screen_nosub', E, _base(ray_subsample=1)),
        ('f14_rd_nosub', E, _base(ray_subsample=1,
                                  amplitude_model='ray_density')),
    ]


def main():
    la = _fix.banner()
    out = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'numpy': np.__version__, 'cases': {}}
    for name, mkE, kw in fixtures():
        E = mkE()
        for rows, tag in ((0, 'rows0'), (None, 'rowsNone')):
            t = time.time()
            f, rec, msgs = _fix.run(la, E, kw, rows)
            key = f'{name}.{tag}'
            out['cases'][key] = {
                'hash': _fix.h(f), 'dtype': str(f.dtype),
                'shape': list(f.shape),
                'sum_abs2': float(np.sum(np.abs(f) ** 2)),
                'max_abs': float(np.max(np.abs(f))),
                'engaged': bool(rec.get('engaged', False)),
                'gate_open': bool(rec.get('gate_open', False)),
                'refused': rec.get('refused'),
                'n_out_of_domain': rec.get('n_out_of_domain'),
                'warnings': msgs,
                'secs': round(time.time() - t, 2)}
            print(f"{key:40s} {out['cases'][key]['hash']} "
                  f"eng={out['cases'][key]['engaged']} "
                  f"nw={len(msgs)} {out['cases'][key]['secs']}s", flush=True)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
