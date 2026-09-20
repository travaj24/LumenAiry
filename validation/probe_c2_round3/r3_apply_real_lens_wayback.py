"""WP-C2 ROUND 3, VR2-D1 -- the seventeenth entry point's way back.

``lumenairy.apply_real_lens`` traces internally whenever
``seidel_correction=True``: ``_apply_real_lens_impl`` imports the tracer
under the local alias ``trace as _rt_trace`` and calls it for the
Seidel-residual fan.  Two censuses missed it (an import alias in series
with a private split-out), so it moved with the two WP-C2 default flips
and carried no keyword back.

This probe measures, in ONE process pinned to ONE tree, a SHA-256 over
``dtype + shape + raw bytes`` of the returned field for each of three
prescriptions and three call shapes:

* ``seidel``      -- ``seidel_correction=True``, no way-back keyword
* ``seidel_old``  -- ``seidel_correction=True``, ``sphere_normal='generic'``
  and ``renormalize='surface'`` (POST tree only; the PRE tree has no such
  parameters)
* ``control``     -- ``seidel_correction=False``, no keyword
* ``control_kw``  -- ``seidel_correction=False`` WITH both keywords set to
  the old routes (POST tree only).  The analytic split-step screen does
  not trace at all, so this must be byte-identical to ``control``.

The comparison the report makes is PRE(``seidel``) == POST(``seidel_old``)
-- the way back reproduces the ``49ddf4bd`` bytes -- and
PRE(``seidel``) != POST(``seidel``), which is the move.

Usage:  python r3_apply_real_lens_wayback.py <out.json> [--tag pre|post]
"""
import hashlib
import inspect
import json
import pathlib
import sys
import warnings

import numpy as np

WL = 587.6e-9


def digest(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def build_field(n=256, dx=80.0e-6, w0=4.0e-3):
    x = (np.arange(n) - (n - 1) / 2.0) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128), dx


def prescriptions():
    """Three prescriptions, all carrying at least one PURE SPHERE so the
    ``sphere_normal`` flip has something to select on, plus an ASPHERE
    whose conic surfaces are the control for that half of the pair (they
    take the generic normal either way and move only through
    ``renormalize``)."""
    from lumenairy.io.prescriptions_builders import make_doublet, make_singlet

    spherical = make_singlet(0.0300, -0.0300, 0.0060, 'N-BK7', 0.0200)
    spherical['aperture_diameter'] = 0.0180

    doublet = make_doublet(0.0517, -0.0345, -0.1200, 0.0090, 0.0025,
                           'N-BK7', 'N-SF5', 0.0250)
    doublet['aperture_diameter'] = 0.0220

    asphere = make_singlet(0.0320, -0.0480, 0.0065, 'N-BK7', 0.0200)
    asphere['aperture_diameter'] = 0.0180
    asphere['surfaces'][0]['conic'] = -0.62
    asphere['surfaces'][0]['aspheric_coeffs'] = {4: 1.4e4, 6: -2.1e8}
    asphere['surfaces'][1]['conic'] = -1.70
    return {'spherical_singlet': spherical,
            'spherical_doublet': doublet,
            'asphere_singlet': asphere}


def main(out_path, tag):
    import lumenairy as la
    from lumenairy.elements import apply_real_lens

    E, dx = build_field()
    params = inspect.signature(apply_real_lens).parameters
    has_kw = all(k in params for k in ('renormalize', 'sphere_normal'))
    old_kw = dict(renormalize='surface', sphere_normal='generic')

    out = {
        'tag': tag,
        'lumenairy_file': la.__file__,
        'version': getattr(la, '__version__', '?'),
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'apply_real_lens_carries_the_pair': has_kw,
        'signature_defaults': {
            k: repr(params[k].default) for k in ('renormalize',
                                                 'sphere_normal')
            if k in params},
        'field_sha256': digest(E),
        'digests': {},
    }

    def run(name, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return digest(apply_real_lens(
                E, prescription=pres, wavelength=WL, dx=dx,
                seidel_poly_order=6, **kw))

    for key, pres in prescriptions().items():
        d = {}
        d['seidel'] = run(key, seidel_correction=True)
        d['control'] = run(key, seidel_correction=False)
        if has_kw:
            d['seidel_old'] = run(key, seidel_correction=True, **old_kw)
            d['seidel_none'] = run(key, seidel_correction=True,
                                   renormalize=None, sphere_normal=None)
            d['control_kw'] = run(key, seidel_correction=False, **old_kw)
        out['digests'][key] = d

    pathlib.Path(out_path).write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({'lumenairy': la.__file__, 'tag': tag,
                      'has_kw': has_kw}, indent=2))


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    tag = 'post'
    if '--tag' in sys.argv:
        tag = sys.argv[sys.argv.index('--tag') + 1]
    main(args[0], tag)
