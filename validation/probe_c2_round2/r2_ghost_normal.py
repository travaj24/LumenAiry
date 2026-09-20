"""WP-C2 round 2, defect D5 -- ``analysis.ghost`` refracted off a different
sphere normal from ``trace``.

Measures, on the running build:

1. the BEFORE/AFTER difference: the same three 2-bounce ghost paths of a
   spherical doublet traced with the ghost leg forced to the generic
   normal (what shipped) and to the library's public default (what this
   round ships), reporting RMS spot radius, FWHM, ray counts and
   transmittance;
2. the AFTER identity: the normal route the ghost leg asks for is the one
   ``trace`` asks for, read off a spy on ``surface._surface_normal`` in
   BOTH paths, so "the same default" is a measurement and not a reading of
   two source lines;
3. the way back: forcing ``sphere_normal='generic'`` into the ghost leg
   reproduces the pre-round-2 numbers exactly (byte-identical).

Run with the three BLAS variables on the command line and ``--root``
naming the tree under test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

WL = 587.5618e-9


def _paths():
    from lumenairy.analysis.ghost import _path_from_pair
    return [_path_from_pair(3, 0, 1), _path_from_pair(3, 0, 2),
            _path_from_pair(3, 1, 2)]


def _retrace(pres, path, force=None):
    """Retrace one ghost path, optionally forcing the leg's normal route."""
    import lumenairy.analysis.ghost as gh
    from lumenairy.analysis.ghost import retrace_ghost_path

    if force is None:
        return retrace_ghost_path(pres, path, WL, semi_aperture=0.010,
                                  n_rays=256, image_plane_z=0.090)
    # force by pinning the helper the ghost leg reads.  NOTE:
    # ``import lumenairy.raytrace.trace as tr`` binds the re-exported
    # FUNCTION (the package __init__ shadows the submodule name), so the
    # module has to come out of sys.modules.
    import importlib
    tr = importlib.import_module('lumenairy.raytrace.trace')
    saved = dict(tr._LIBRARY_TRACE_DEFAULTS)
    have_helper = hasattr(tr, '_library_trace_default')
    try:
        if have_helper:
            tr._LIBRARY_TRACE_DEFAULTS['sphere_normal'] = force
            return retrace_ghost_path(pres, path, WL, semi_aperture=0.010,
                                      n_rays=256, image_plane_z=0.090)
        raise RuntimeError('no helper on this tree')
    finally:
        tr._LIBRARY_TRACE_DEFAULTS.clear()
        tr._LIBRARY_TRACE_DEFAULTS.update(saved)
        del gh


def _fields(res):
    out = {}
    for k, v in res.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
        elif isinstance(v, bool):
            out[k] = bool(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    import lumenairy
    assert os.path.abspath(lumenairy.__file__).startswith(root)
    import numpy as np
    print('lumenairy.__file__ =', lumenairy.__file__)
    print('numpy', np.__version__, 'python', sys.version.split()[0])

    import warnings

    from lumenairy.io.prescriptions_builders import make_doublet
    from lumenairy.raytrace import surface as surf_mod
    from lumenairy.raytrace.trace import trace

    pres = make_doublet(0.0517, -0.0345, -0.1200, 0.0090, 0.0025,
                        'N-BK7', 'N-SF5', 0.0250)

    payload = {'lumenairy_file': lumenairy.__file__,
               'python': sys.version.split()[0],
               'numpy': np.__version__,
               'paths': []}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i, path in enumerate(_paths()):
            shipped = _retrace(pres, path)            # library default
            old = _retrace(pres, path, force='generic')
            forced_new = _retrace(pres, path, force='analytic')
            row = {'path': [[int(a), str(b)] for a, b in path],
                   'library_default': _fields(shipped),
                   'generic_forced': _fields(old),
                   'analytic_forced': _fields(forced_new)}
            row['delta'] = {
                k: abs(row['library_default'][k] - row['generic_forced'][k])
                for k in row['library_default']
                if k in row['generic_forced']}
            row['default_is_analytic'] = (
                row['library_default'] == row['analytic_forced'])
            payload['paths'].append(row)
            print(f'  path {i}: '
                  f'rms {shipped.get("rms_radius_m")!r} vs '
                  f'{old.get("rms_radius_m")!r}')

    # --- the AFTER identity: what normal route each path actually asks for
    seen = {'ghost': set(), 'trace': set()}
    real = surf_mod._surface_normal

    def spy_factory(tag):
        def spy(x, y, surface, *, analytic_sphere=False):
            seen[tag].add(bool(analytic_sphere))
            return real(x, y, surface, analytic_sphere=analytic_sphere)
        return spy

    import lumenairy.raytrace.intersection as isect
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.trace import _make_bundle

    surfs = surfaces_from_prescription(pres)
    h = np.linspace(-0.008, 0.008, 33)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        isect._surface_normal = spy_factory('trace')
        try:
            trace(_make_bundle(h, 0.3 * h, np.zeros_like(h),
                               np.zeros_like(h), WL), surfs, WL,
                  output_filter='last')
        finally:
            isect._surface_normal = real
        isect._surface_normal = spy_factory('ghost')
        try:
            _retrace(pres, _paths()[0])
        finally:
            isect._surface_normal = real
    payload['analytic_sphere_seen'] = {k: sorted(v) for k, v in seen.items()}
    payload['ghost_asks_what_trace_asks'] = (
        sorted(seen['ghost']) == sorted(seen['trace']) and bool(seen['trace']))
    print('analytic_sphere seen -- trace:', sorted(seen['trace']),
          ' ghost:', sorted(seen['ghost']))

    with open(args.out, 'w', encoding='ascii') as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
