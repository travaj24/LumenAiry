"""VERIFY-WP-C2 ROUND 2, item D5 -- the ghost leg's normal route, measured
independently.

Four questions, each answered with bytes rather than with a reading:

1. does the ghost path resolve the route by ASKING the library
   (``trace._library_trace_default('sphere_normal')``) rather than naming
   one?  A spy on ``_refract`` / ``_reflect`` records the ``sphere_normal``
   every ghost step passes, and the helper is then made to return a
   DIFFERENT route to show the ghost leg follows it (an entry point that
   hard-codes ``'analytic'`` would not move).
2. is the refraction step BYTE-IDENTICAL to what ``trace``'s default route
   produces on the same bundle -- and NOT identical to the generic route,
   so the identity is not vacuous?
3. is ``renormalize`` in the ghost path really ``True``?  The spy records
   what each call passed, and the effective value is read from
   ``_refract``'s own signature when the call omits it.
4. what does the route change cost the ANSWER?  The public
   ``retrace_ghost_path`` is run on this verification's own doublet with
   the route forced both ways and the RMS spot radius compared.

Usage:  LUMENAIRY_ROOT=<root> python vr2_ghost.py <out.json>
"""
import hashlib
import inspect
import json
import os
import sys

import numpy as np

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import lumenairy as la  # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

WL = 546.1e-9


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def prescription():
    return {
        'name': 'vr2 ghost doublet (all spherical)',
        'aperture_diameter': 0.0260,
        'surfaces': [
            {'radius': 0.0625, 'conic': 0.0,
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -0.0388, 'conic': 0.0,
             'glass_before': 'N-BK7', 'glass_after': 'N-SF5'},
            {'radius': -0.1550, 'conic': 0.0,
             'glass_before': 'N-SF5', 'glass_after': 'air'},
        ],
        'thicknesses': [0.0075, 0.0028],
    }


def main(out_path):
    from lumenairy.analysis import ghost as G
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.intersection import _transfer
    from lumenairy.raytrace.trace import _library_trace_default, trace

    out = {'lumenairy_file': la.__file__,
           'python': sys.version.split()[0], 'numpy': np.__version__}

    lib_route = _library_trace_default('sphere_normal')
    out['library_trace_default_sphere_normal'] = lib_route
    out['library_trace_default_renormalize'] = _library_trace_default(
        'renormalize')
    out['ghost_source_asks_the_helper'] = (
        "_library_trace_default('sphere_normal')"
        in inspect.getsource(G.retrace_ghost_path))

    pres = prescription()
    paths = [[(0, 'transmit'), (2, 'reflect'), (1, 'reflect'),
              (2, 'transmit')],
             [(0, 'transmit'), (1, 'transmit'), (2, 'reflect'),
              (1, 'reflect'), (2, 'transmit')],
             [(0, 'transmit'), (1, 'reflect'), (0, 'reflect'),
              (1, 'transmit'), (2, 'transmit')]]

    # ---- 1. spy on the two private helpers the ghost loop calls -------
    import lumenairy.raytrace.intersection as I
    seen = []
    real_refract, real_reflect = I._refract, I._reflect

    def spy_refract(rays, surface, n1, n2, **kw):
        seen.append(('_refract', kw.get('sphere_normal', '<omitted>'),
                     kw.get('renormalize', '<omitted>')))
        return real_refract(rays, surface, n1, n2, **kw)

    def spy_reflect(rays, surface, **kw):
        seen.append(('_reflect', kw.get('sphere_normal', '<omitted>'),
                     kw.get('renormalize', '<omitted>')))
        return real_reflect(rays, surface, **kw)

    I._refract, I._reflect = spy_refract, spy_reflect
    try:
        res_spy = G.retrace_ghost_path(pres, paths[0], WL,
                                       semi_aperture=0.0090, n_rays=64)
    finally:
        I._refract, I._reflect = real_refract, real_reflect
    out['ghost_calls_spied'] = len(seen)
    out['ghost_sphere_normal_seen'] = sorted({s[1] for s in seen})
    out['ghost_renormalize_seen'] = sorted({str(s[2]) for s in seen})
    out['refract_default_renormalize'] = (
        inspect.signature(real_refract).parameters['renormalize'].default)
    out['reflect_default_renormalize'] = (
        inspect.signature(real_reflect).parameters['renormalize'].default)
    out['ghost_effective_renormalize'] = (
        out['refract_default_renormalize']
        if out['ghost_renormalize_seen'] == ['<omitted>']
        else out['ghost_renormalize_seen'])
    out['ghost_spy_rms_radius_mm'] = float(res_spy['rms_radius_mm'])

    # ---- 2. the helper is ASKED, not named: make it answer differently
    import lumenairy.raytrace.trace  # noqa: F401
    # the package attribute ``lumenairy.raytrace.trace`` is the FUNCTION,
    # so the module has to come from sys.modules
    T = sys.modules['lumenairy.raytrace.trace']
    T._library_trace_default('sphere_normal')     # prime the cache
    cache = dict(T._LIBRARY_TRACE_DEFAULTS)
    seen.clear()
    try:
        T._LIBRARY_TRACE_DEFAULTS['sphere_normal'] = 'generic'
        I._refract, I._reflect = spy_refract, spy_reflect
        G.retrace_ghost_path(pres, paths[0], WL, semi_aperture=0.0090,
                             n_rays=64)
    finally:
        I._refract, I._reflect = real_refract, real_reflect
        T._LIBRARY_TRACE_DEFAULTS.clear()
        T._LIBRARY_TRACE_DEFAULTS.update(cache)
    out['ghost_follows_the_helper'] = sorted({s[1] for s in seen}) == [
        'generic']

    # ---- 3. byte identity at the refraction step itself ---------------
    surfs = surfaces_from_prescription(pres)
    from lumenairy.raytrace.trace import _make_bundle
    n = 96
    h = np.linspace(-0.0090, 0.0090, n)
    zeros = np.zeros(n)

    def _step(route):
        rays = _make_bundle(h, 0.31 * h, zeros + 0.03, zeros + 0.01, WL)
        _transfer(rays, 0.0, 1.0)
        from lumenairy.raytrace.intersection import _intersect_surface
        _intersect_surface(rays, surfs[0], n_medium=1.0)
        real_refract(rays, surfs[0], 1.0, 1.5168, sphere_normal=route)
        return digest(rays.L, rays.M, rays.N, rays.x, rays.y, rays.z,
                      rays.opd, rays.alive, rays.error_code)

    out['step_digest_ghost_route'] = _step(lib_route)
    out['step_digest_trace_default'] = _step(
        inspect.signature(trace).parameters['sphere_normal'].default)
    out['step_digest_generic'] = _step('generic')
    out['step_digest_analytic'] = _step('analytic')
    out['ghost_step_matches_trace_default'] = (
        out['step_digest_ghost_route'] == out['step_digest_trace_default'])
    out['ghost_step_differs_from_generic'] = (
        out['step_digest_ghost_route'] != out['step_digest_generic'])

    # ---- 4. what the route costs the public ghost answer --------------
    deltas = []
    for p in paths:
        r_lib = G.retrace_ghost_path(pres, p, WL, semi_aperture=0.0090,
                                     n_rays=256)
        try:
            T._LIBRARY_TRACE_DEFAULTS['sphere_normal'] = 'generic'
            r_gen = G.retrace_ghost_path(pres, p, WL, semi_aperture=0.0090,
                                         n_rays=256)
        finally:
            T._LIBRARY_TRACE_DEFAULTS.clear()
            T._LIBRARY_TRACE_DEFAULTS.update(cache)
        row = {'path': [list(map(str, s)) for s in p]}
        ra = np.asarray(r_lib['rays_image_plane'], dtype=float)
        rb = np.asarray(r_gen['rays_image_plane'], dtype=float)
        row['image_rays_max_abs_change_mm'] = float(
            np.max(np.abs(ra - rb))) if ra.shape == rb.shape else None
        for key in ('rms_radius_mm', 'fwhm_mm', 'total_transmittance',
                    'energy_fraction_ppm'):
            a, b = r_lib.get(key), r_gen.get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                row[key + '_lib'] = float(a)
                row[key + '_generic'] = float(b)
                row[key + '_abs_change'] = abs(float(a) - float(b))
        deltas.append(row)
    out['paths'] = deltas
    out['worst_rms_radius_mm_change'] = max(
        (d.get('rms_radius_mm_abs_change', 0.0) for d in deltas),
        default=0.0)
    out['worst_fwhm_mm_change'] = max(
        (d.get('fwhm_mm_abs_change', 0.0) for d in deltas), default=0.0)
    # and the rays themselves, which is the strongest statement
    out['worst_image_ray_change_mm'] = None
    out['worst_transmittance_change'] = max(
        (d.get('total_transmittance_abs_change', 0.0) for d in deltas),
        default=0.0)

    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=1)
    for k, v in out.items():
        if k != 'paths':
            print('%-40s %s' % (k, v))
    for d in out['paths']:
        print('   path', d['path'], 'rms_mm change',
              d.get('rms_radius_mm_abs_change'), 'rays change',
              d.get('image_rays_max_abs_change_mm'))
    print('wrote', out_path)


if __name__ == '__main__':
    main(sys.argv[1])
