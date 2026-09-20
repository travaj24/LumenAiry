"""VERIFY-WAVE5-E / E5: does the O-3 guard (or the O-1 freeze) move a byte of
an AIR-terminated FGA field?

Runs unchanged in the PRE tree (``ede07f30``, no guard, no freeze) and in the
POST tree and prints a digest per fixture.  ``LUMENAIRY_MEM_BUDGET_MB`` is
REQUIRED to be set (VERIFY-WP-B12 D-4: the returned bytes depend on the
momentum/lattice chunk, which the default budget derives from the free RAM at
call time, so an unpinned comparison is not run-reproducible).
"""
import hashlib
import json
import os
import sys
import warnings

import numpy as np

import lumenairy as la
from lumenairy.propagators import fga as _fga

if not os.environ.get('LUMENAIRY_MEM_BUDGET_MB'):
    raise SystemExit('LUMENAIRY_MEM_BUDGET_MB must be pinned (VERIFY-B12 D-4)')

_LAM = 1.03e-6


def _dig(a):
    if isinstance(a, tuple):
        a = a[0]
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:24]


def _E(n, dx, w):
    xs = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


def _presc(last_radius, conic, semi, t, glass):
    s1 = {'radius': np.inf, 'conic': 0.0, 'thickness': t,
          'glass_before': 'air', 'glass_after': glass, 'semi_diameter': semi}
    s2 = {'radius': last_radius, 'conic': conic, 'thickness': 0.0,
          'glass_before': glass, 'glass_after': 'air', 'semi_diameter': semi}
    return {'name': 'v_bytes', 'aperture_diameter': 2 * semi,
            'surfaces': [s1, s2], 'thicknesses': [t], 'stop_index': 0}


_FIX = {
    'conic_air': dict(r=-1.05e-3, k=-0.35, semi=0.18e-3, t=0.6e-3,
                      glass='N-SSK8', n=64, dx=6e-6, w=45e-6, z=0.30e-3),
    'sphere_air': dict(r=-1.40e-3, k=0.0, semi=0.22e-3, t=0.8e-3,
                       glass='N-BK7', n=64, dx=6e-6, w=52e-6, z=0.45e-3),
    'flat_air': dict(r=np.inf, k=0.0, semi=0.22e-3, t=0.8e-3,
                     glass='N-BK7', n=64, dx=6e-6, w=52e-6, z=0.45e-3),
}


def main():
    out = dict(lumenairy_file=la.__file__, version=la.__version__,
               python=sys.version.split()[0], platform=sys.platform,
               numpy=np.__version__,
               mem_budget=os.environ.get('LUMENAIRY_MEM_BUDGET_MB'),
               digests={})
    print('lumenairy.__file__ =', la.__file__, flush=True)
    for name, f in _FIX.items():
        presc = _presc(f['r'], f['k'], f['semi'], f['t'], f['glass'])
        E = _E(f['n'], f['dx'], f['w'])
        kw = dict(prescription=presc, wavelength=_LAM, dx=f['dx'],
                  output_plane_distance=f['z'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out['digests'][name + '/through_lens'] = _dig(
                la.apply_real_lens_fga(E, **kw))
            out['digests'][name + '/coarse'] = _dig(
                la.apply_real_lens_fga(E, coarse_stride=3, **kw))
            out['digests'][name + '/vector'] = _dig(
                la.apply_real_lens_fga_vector(np.stack([E, E * 0.5]), **kw))
            cz = _fga._caustic_zone(E, f['dx'], presc, _LAM)
            out['digests'][name + '/caustic_zone'] = repr(
                cz if np.isscalar(cz) else np.asarray(cz).tolist())
        print('%-24s %s' % (name, out['digests'][name + '/through_lens']),
              flush=True)
    print(json.dumps(out, indent=1))


main()
