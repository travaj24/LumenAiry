"""VERIFY-WP-C2 item 5 -- the way back, measured archive to archive.

The single most user-facing claim of the release is that
``sphere_normal='generic', renormalize='surface'`` reproduces the pre-5.49.0
arithmetic EXACTLY.  The shipped test asserts it against the branch's own
``_surface_normal`` with the keyword ignored, which is a self-consistency
check.  This compares against a read-only ``git archive 49ddf4bd`` in a
SEPARATE PROCESS with its own ``PYTHONPATH``, which is the statement the
Migration Guide actually makes.

Run twice by the caller, once per root, writing a dump; the second invocation
diffs the two dumps byte for byte.

Usage:
    VC2_ROLE=dump  LUMENAIRY_ROOT=<root> python vc2_wayback.py OUT.npz [old]
    VC2_ROLE=diff  python vc2_wayback.py A.npz B.npz OUT.json
"""
import json
import os
import sys

import numpy as np


def _dump(out_path, force_old):
    root = os.environ['LUMENAIRY_ROOT']
    sys.path.insert(0, root)
    import lumenairy as la
    want = os.path.realpath(os.path.join(root, 'lumenairy'))
    assert os.path.realpath(os.path.dirname(la.__file__)) == want, (
        la.__file__, root)
    from lumenairy.raytrace.surface import RayBundle, Surface
    from lumenairy.raytrace.trace import trace

    wl = 587.5618e-9

    def s(R, th, gb, ga, sd, conic=0.0, mirror=False):
        return Surface(radius=R, conic=conic, thickness=th, glass_before=gb,
                       glass_after=ga, semi_diameter=sd, is_mirror=mirror)

    def ladder(n):
        out = []
        for j in range(n):
            out.append(s(0.0800 + 0.003 * j, 0.0055, 'air', 'N-BK7', 0.011))
            out.append(s(-0.0900 - 0.003 * j, 0.0090, 'N-BK7', 'air', 0.011))
        out.append(s(0.2500, 0.0300, 'air', 'N-BK7', 0.011))
        return out

    pres = {
        'doublet': [s(0.0517, 0.0090, 'air', 'N-BK7', 0.0125),
                    s(-0.0345, 0.0025, 'N-BK7', 'N-SF5', 0.0125),
                    s(-0.1200, 0.0400, 'N-SF5', 'air', 0.0125)],
        'stack7': ladder(3),
        'ladder13': ladder(6),
        'conic': [s(0.0517, 0.0090, 'air', 'N-BK7', 0.0125, conic=-0.6),
                  s(-0.0345, 0.0025, 'N-BK7', 'N-SF5', 0.0125, conic=-1.2),
                  s(-0.1200, 0.0400, 'N-SF5', 'air', 0.0125, conic=0.4)],
        'mirrors': [s(-0.3000, -0.1000, 'air', 'air', 0.060, mirror=True),
                    s(-0.0900, 0.2000, 'air', 'air', 0.020, mirror=True)],
    }

    def bundle(n, hmax, tilt):
        rng = np.random.default_rng(13579)
        r = hmax * np.sqrt(rng.uniform(0, 1, n))
        th = rng.uniform(0, 2 * np.pi, n)
        L = np.full(n, np.sin(np.radians(tilt)))
        return RayBundle(x=r * np.cos(th), y=r * np.sin(th), z=np.zeros(n),
                         L=L, M=np.zeros(n),
                         N=np.sqrt(np.maximum(1.0 - L ** 2, 0.0)),
                         wavelength=wl, alive=np.ones(n, dtype=bool),
                         opd=np.zeros(n))

    kw = dict(sphere_normal='generic',
              renormalize='surface') if force_old else {}
    out = {}
    for name, surfs in pres.items():
        for tilt in (0.0, 4.0):
            for mode in ('last', 'all'):
                rb = bundle(3000, 0.0090, tilt)
                res = trace(rb, surfs, wl, output_filter=mode, **kw)
                bundles = ([res.image_rays] if mode == 'last'
                           else list(res.ray_history))
                for bi, b in enumerate(bundles):
                    for fld in ('x', 'y', 'z', 'L', 'M', 'N', 'opd'):
                        key = f'{name}.{tilt:g}.{mode}.{bi}.{fld}'
                        out[key] = np.asarray(getattr(b, fld), dtype=float)
                    out[f'{name}.{tilt:g}.{mode}.{bi}.alive'] = np.asarray(
                        b.alive, dtype=np.uint8)
                    out[f'{name}.{tilt:g}.{mode}.{bi}.code'] = np.asarray(
                        b.error_code, dtype=np.uint8)
    np.savez(out_path, **out)
    print(f'wrote {out_path}: {len(out)} arrays, version {la.__version__}, '
          f'force_old={force_old}, file={la.__file__}')


def _diff(a_path, b_path, out_path):
    a = np.load(a_path)
    b = np.load(b_path)
    keys = sorted(set(a.files) & set(b.files))
    assert len(keys) == len(a.files) == len(b.files), (
        len(keys), len(a.files), len(b.files))
    identical, moved = [], []
    for k in keys:
        x, y = a[k], b[k]
        if x.shape == y.shape and np.array_equal(
                x.view(np.uint8) if x.dtype == np.uint8 else x.view(np.int64),
                y.view(np.uint8) if y.dtype == np.uint8 else y.view(np.int64)):
            identical.append(k)
        else:
            d = (float(np.max(np.abs(x.astype(float) - y.astype(float))))
                 if x.shape == y.shape else float('inf'))
            moved.append(dict(key=k, max_abs=d))
    res = dict(n_arrays=len(keys), n_identical=len(identical),
               n_moved=len(moved), moved=moved[:40],
               all_identical=(not moved))
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in res.items() if k != 'moved'},
                     indent=1, sort_keys=True))
    for m in moved[:10]:
        print('  MOVED', m)


if __name__ == '__main__':
    if os.environ.get('VC2_ROLE') == 'diff':
        _diff(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        _dump(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == 'old')
