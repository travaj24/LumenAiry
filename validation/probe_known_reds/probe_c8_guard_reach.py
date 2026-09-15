"""Probe: does REMAP_STATIONARY_PHASE_FIT_GUARD still reach the traced fit?

The c7 / c8 reds share one premise: with the C6 launch AND the C6 fit guard on,
the _GHOST fixture manufactures a lobe beyond 3 w at ~4.6e-2 of peak.  On this
box it reads 1.5e-4.  This probe separates the two candidate causes:
  (a) the flag no longer reaches the code that consumes it (guard on == off,
      byte-identical) -- a library defect;
  (b) the flag reaches, the numbers moved -- a numerics / build question.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import lumenairy as la  # noqa: E402
from lumenairy.elements import _lens_traced as LT  # noqa: E402

assert 'lum_reds' in la.__file__, la.__file__

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL
_GHOST = dict(n=768, dx=25e-6, w=1.5e-3, rc=-0.15, alpha=5.0,
              r1=150e-3, r2=-150e-3, th=4e-3, z=6e-3, ap=18e-3)
_BASE_KW = dict(wavelength=_WL, amplitude_model='ray_density',
                preserve_input_phase='remap', remap_sampling='full',
                parallel_amp=False, on_undersample='silent',
                on_noncollimated='silent', on_aperture_beam='silent',
                ray_subsample=4, fit_radius_beam_factor=2.0)


def _flat():
    return {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air',
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _surf(r, gb, ga):
    d = _flat()
    d['radius'], d['glass_before'], d['glass_after'] = r, gb, ga
    return d


def _singlet(s, glass='N-BK7'):
    return {'name': 'c8_singlet', 'aperture_diameter': s['ap'],
            'surfaces': [_surf(s['r1'], 'air', glass),
                         _surf(s['r2'], glass, 'air'), _flat()],
            'thicknesses': [s['th'], s['z']]}


def _field(s):
    ax = (np.arange(s['n']) - s['n'] // 2) * s['dx']
    X, Y = np.meshgrid(ax, ax)
    rc = s['rc']
    sg = 1.0 if rc > 0 else -1.0
    rho = np.sqrt(X ** 2 + Y ** 2 + rc * rc)
    Wc = sg * (rho - abs(rc))
    r2 = X ** 2 + Y ** 2
    a = (s['alpha'] / _K0) * (r2 / (s['w'] * s['w'])) ** 2
    return (np.exp(-r2 / (s['w'] * s['w']))
            * np.exp(1j * _K0 * (Wc + a))).astype(np.complex128)


def call(s, bound, guard, launch=True, **over):
    import warnings
    E = _field(s)
    presc = _singlet(s)
    old = (LT.REMAP_STATIONARY_PHASE_LAUNCH,
           LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
           LT.REMAP_INVERSE_SUPPORT_BOUND)
    LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(launch)
    LT.REMAP_STATIONARY_PHASE_FIT_GUARD = bool(guard)
    LT.REMAP_INVERSE_SUPPORT_BOUND = bool(bound)
    try:
        kw = dict(_BASE_KW)
        kw['dx'] = s['dx']
        kw.update(over)
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            F = np.asarray(la.apply_real_lens_traced(
                E, prescription=presc, carrier=s['rc'], **kw))
    finally:
        (LT.REMAP_STATIONARY_PHASE_LAUNCH,
         LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
         LT.REMAP_INVERSE_SUPPORT_BOUND) = old
    return F, [str(w.message) for w in wl]


def halo(F, s):
    ax = (np.arange(s['n']) - s['n'] // 2) * s['dx']
    X, Y = np.meshgrid(ax, ax)
    R = np.hypot(X, Y)
    a = np.abs(F)
    pk = float(a.max())
    m = R > 3.0 * s['w']
    return float(a[m].max()) / pk


def main():
    # module-level warm-up, as the test's _warm fixture does
    for _ in range(2):
        call(dict(_GHOST, n=384, dx=30e-6, w=0.9e-3, rc=-0.20, alpha=3.0,
                  r1=200e-3, r2=-200e-3, th=4e-3, z=5e-3, ap=11e-3),
             bound=False, guard=False, launch=False)

    out = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'numpy': np.__version__,
           'env': {k: os.environ.get(k) for k in
                   ('OPENBLAS_CORETYPE', 'OPENBLAS_NUM_THREADS',
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMBA_DISABLE_JIT')},
           'flag_defaults': {
               'REMAP_STATIONARY_PHASE_LAUNCH':
                   bool(LT.REMAP_STATIONARY_PHASE_LAUNCH),
               'REMAP_STATIONARY_PHASE_FIT_GUARD':
                   bool(LT.REMAP_STATIONARY_PHASE_FIT_GUARD),
               'REMAP_INVERSE_SUPPORT_BOUND':
                   bool(LT.REMAP_INVERSE_SUPPORT_BOUND)},
           'cells': {}}

    fields = {}
    for guard in (False, True):
        for bound in (False, True):
            F, msgs = call(_GHOST, bound=bound, guard=guard)
            key = f'guard={guard},bound={bound}'
            fields[key] = F
            out['cells'][key] = {
                'halo_3w': halo(F, _GHOST),
                'peak': float(np.abs(F).max()),
                'power': float((np.abs(F) ** 2).sum()),
                'warnings': msgs}

    a = fields['guard=False,bound=False']
    b = fields['guard=True,bound=False']
    out['guard_on_equals_guard_off_bitwise'] = bool(np.array_equal(a, b))
    out['guard_max_abs_diff'] = float(np.max(np.abs(a - b)))
    c = fields['guard=True,bound=False']
    d = fields['guard=True,bound=True']
    out['bound_on_equals_off_bitwise'] = bool(np.array_equal(c, d))
    out['bound_max_abs_diff'] = float(np.max(np.abs(c - d)))
    print(json.dumps(out, indent=1))
    tag = os.environ.get('PROBE_TAG', 'default')
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f'c8_guard_reach_{tag}.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', p)


if __name__ == '__main__':
    main()
