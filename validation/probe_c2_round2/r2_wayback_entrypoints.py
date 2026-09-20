"""WP-C2 round 2, defect D4 -- the way back through all SIXTEEN internally
tracing entry points, proved archive to archive.

Run twice, in two separate processes with two different trees on
``PYTHONPATH``:

* ``--mode pre``  against a ``git archive 49ddf4bd`` extraction: every entry
  point is called with NO way-back keyword, because at that commit neither
  keyword existed and ``trace``'s own defaults were ``'surface'`` /
  ``'generic'``;
* ``--mode post`` against the round-2 tree: every entry point is called with
  ``renormalize='surface', sphere_normal='generic'``.

A third mode, ``--mode post_default``, calls the same fixtures with no
keyword at all, which is the CONTRAST arm: it shows the entry points really
did move, so a 16/16 identity in the ``post`` arm cannot be a switch that
reaches nothing.  ``--mode post_none`` passes ``None`` for both, which must
be byte-identical to ``post_default`` (the "None stamps nothing" claim).

Every value compared is the raw ``ndarray.tobytes()`` of a result field,
digested with SHA-256 together with its dtype and shape, so "identical"
means identical bytes and not "close".

Usage::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python r2_wayback_entrypoints.py --root <tree> --mode pre \\
        --out r2_wayback_pre_win.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback

WL = 587.5618e-9


# ---------------------------------------------------------------- digests

def _digest(obj, depth=0, seen=None):
    """Recursively reduce a result object to ``{path: sha256}``.

    Arrays digest their dtype, shape and raw bytes.  Python scalars digest
    their ``repr`` (exact for float64 on every build this runs on).  Nested
    dataclasses / namedtuples / plain objects are walked; anything else is
    reported as its type name so a silently changed return TYPE shows up as
    a difference rather than as an empty match.
    """
    import numpy as np

    if seen is None:
        seen = set()
    out = {}
    if isinstance(obj, np.ndarray):
        h = hashlib.sha256()
        h.update(str(obj.dtype).encode())
        h.update(str(obj.shape).encode())
        h.update(np.ascontiguousarray(obj).tobytes())
        return {'': h.hexdigest()}
    if isinstance(obj, (bool, int, float, str, type(None))):
        return {'': hashlib.sha256(repr(obj).encode()).hexdigest()}
    if depth > 3:
        return {'': 'DEPTH:' + type(obj).__name__}
    if id(obj) in seen:
        return {'': 'CYCLE'}
    seen = seen | {id(obj)}
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            for k, d in _digest(v, depth + 1, seen).items():
                out[f'[{i}]{k}'] = d
        return out
    if isinstance(obj, dict):
        for kk in sorted(obj):
            for k, d in _digest(obj[kk], depth + 1, seen).items():
                out[f'[{kk!r}]{k}'] = d
        return out
    fields = getattr(obj, '_fields', None)
    if fields is None:
        fields = [f for f in vars(obj)] if hasattr(obj, '__dict__') else None
    if fields is None:
        slots = getattr(type(obj), '__slots__', None)
        fields = list(slots) if slots else None
    if fields is None:
        return {'': 'TYPE:' + type(obj).__name__}
    for f in sorted(fields):
        if f.startswith('__'):
            continue
        try:
            v = getattr(obj, f)
        except Exception:
            continue
        if callable(v):
            continue
        for k, d in _digest(v, depth + 1, seen).items():
            out[f'.{f}{k}'] = d
    return out


def _trace_result_digest(res):
    """A ``TraceResult`` digested the way the WP-C2 probe does it: the image
    bundle AND every history bundle, nine fields each."""
    import numpy as np

    out = {}
    bundles = [('image', res.image_rays)]
    hist = getattr(res, 'ray_history', None) or []
    for i, b in enumerate(hist):
        if hasattr(b, 'x'):
            bundles.append((f'hist{i}', b))
    for tag, b in bundles:
        for f in ('x', 'y', 'z', 'L', 'M', 'N', 'opd', 'alive', 'error_code'):
            v = getattr(b, f, None)
            if v is None:
                continue
            h = hashlib.sha256()
            v = np.ascontiguousarray(v)
            h.update(str(v.dtype).encode())
            h.update(str(v.shape).encode())
            h.update(v.tobytes())
            out[f'{tag}.{f}'] = h.hexdigest()
    return out


# --------------------------------------------------------------- fixtures

def _fixtures():
    from lumenairy.io.prescriptions_builders import make_doublet, make_singlet

    singlet = make_singlet(0.0515, -0.0515, 0.0060, 'N-BK7', 0.0200)
    doublet = make_doublet(0.0517, -0.0345, -0.1200, 0.0090, 0.0025,
                           'N-BK7', 'N-SF5', 0.0250)
    return singlet, doublet


def _surfaces(prescription):
    from lumenairy.raytrace import surfaces_from_prescription
    return surfaces_from_prescription(prescription)


def _world(prescription):
    from lumenairy.raytrace.world import world_surfaces_from_prescription
    return world_surfaces_from_prescription(prescription)


# ------------------------------------------------------------ entry points

def run_all(kw):
    """Call all sixteen entry points with ``**kw`` and digest each answer.

    ``kw`` is either ``{}`` (the PRE tree, or the POST tree's default arm) or
    ``{'renormalize': ..., 'sphere_normal': ...}``.
    """
    import numpy as np

    import lumenairy  # noqa: F401
    from lumenairy.analysis.aberration import caustic_diagnostic
    from lumenairy.analysis.image_plane_wfe import eval_image_plane_wfe
    from lumenairy.propagators.asymptotic_canonical_fit import (
        fit_canonical_polynomials, fit_hf_polynomials)
    from lumenairy.raytrace.differential import ray_transfer_jacobian
    from lumenairy.raytrace.ray_fan import (
        opd_fan_data, opd_fan_data_world, ray_fan_data, ray_fan_data_world,
        through_focus_rms)
    from lumenairy.raytrace.trace import raytrace_system, trace_prescription
    from lumenairy.raytrace.world import paraxial_focus_world

    singlet, doublet = _fixtures()
    surf_d = _surfaces(doublet)
    world_d = _world(doublet)
    out = {}
    errs = {}

    def _run(name, fn):
        try:
            out[name] = fn()
        except Exception:
            errs[name] = traceback.format_exc(limit=4)

    _run('trace_prescription', lambda: _trace_result_digest(
        trace_prescription(doublet, WL, semi_aperture=0.010,
                           field_angle=0.030, num_rings=5,
                           rays_per_ring=24, **kw)))

    def _raytrace_system():
        elements = [
            {'type': 'real_lens', 'prescription': doublet,
             'aperture_diameter': 0.025},
            {'type': 'propagate', 'z': 0.100},
        ]
        res, _s = raytrace_system(elements, WL, semi_aperture=0.008,
                                  field_angle=0.020, num_rings=4,
                                  rays_per_ring=16, **kw)
        return _trace_result_digest(res)
    _run('raytrace_system', _raytrace_system)

    _run('ray_fan_data', lambda: _digest(
        ray_fan_data(surf_d, WL, 0.010, field_angle=0.030, n_rays=41, **kw)))
    _run('ray_fan_data_world', lambda: _digest(
        ray_fan_data_world(world_d, WL, 0.010, field_angle=0.030,
                           n_rays=41, **kw)))
    _run('opd_fan_data', lambda: _digest(
        opd_fan_data(surf_d, WL, 0.010, field_angle=0.030, n_rays=41, **kw)))
    _run('opd_fan_data_world', lambda: _digest(
        opd_fan_data_world(world_d, WL, 0.010, field_angle=0.030,
                           n_rays=41, **kw)))
    _run('through_focus_rms', lambda: _digest(
        through_focus_rms(surf_d, WL, 0.010,
                          np.linspace(0.085, 0.105, 9),
                          field_angle=0.020, num_rings=4,
                          rays_per_ring=16, **kw)))
    _run('paraxial_focus_world', lambda: _digest(
        paraxial_focus_world(world_d, WL, aperture_radius=0.002, **kw)))

    def _jac():
        n = 24
        r = np.linspace(-0.009, 0.009, n)
        z = np.zeros(n)
        return _digest(ray_transfer_jacobian(r, 0.5 * r, z, z + 0.01,
                                             surf_d, WL, **kw))
    _run('ray_transfer_jacobian', _jac)

    _run('eval_image_plane_wfe', lambda: _digest(
        eval_image_plane_wfe(dict(doublet, object_distance=float('inf')),
                             WL, field=(0.0, 0.5), n_pupil=15,
                             field_max_rad=0.030, **kw)))
    _run('caustic_diagnostic', lambda: _digest(
        caustic_diagnostic(doublet, WL, fan_radius=2e-3, n_z_per_gap=16,
                           z_after_last_surface=0.100, **kw)))

    def _layout():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from lumenairy.analysis.plotting import plot_lens_layout
        fig, ax = plot_lens_layout(doublet, wavelength=WL, show_rays=True,
                                   n_field_angles=3, max_field_deg=2.0,
                                   rays_per_fan=5, **kw)
        data = [np.asarray(ln.get_xydata(), dtype=float)
                for ln in ax.get_lines()]
        plt.close(fig)
        return _digest(data)
    _run('plot_lens_layout', _layout)

    _run('fit_canonical_polynomials', lambda: _digest(
        fit_canonical_polynomials(doublet, WL, n_field=4, n_pupil=6,
                                  poly_order=4, **kw)))
    _run('fit_hf_polynomials', lambda: _digest(
        fit_hf_polynomials(doublet, WL, n_field=4, n_pupil=6,
                           poly_order=4, **kw)))

    def _E_in(n=64, dx=20e-6):
        ax_ = (np.arange(n) - n // 2) * dx
        X, Y = np.meshgrid(ax_, ax_, indexing='xy')
        w0 = 0.30e-3
        return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)

    def _traced():
        from lumenairy.elements import apply_real_lens_traced
        E = _E_in()
        out_E = apply_real_lens_traced(
            E, prescription=singlet, wavelength=WL, dx=20e-6,
            ray_subsample=8, bandlimit=False, on_undersample='silent',
            on_noncollimated='silent', on_aperture_beam='silent',
            on_fit_domain_basis='silent', on_pool_memory='silent',
            n_workers=1, **kw)
        return _digest(np.asarray(out_E))
    _run('apply_real_lens_traced', _traced)

    def _maslov():
        from lumenairy.elements import apply_real_lens_maslov
        E = _E_in()
        out_E = apply_real_lens_maslov(
            E, prescription=singlet, wavelength=WL, dx=20e-6,
            ray_field_samples=6, ray_pupil_samples=6, poly_order=4,
            output_subsample=2, **kw)
        return _digest(np.asarray(out_E))
    _run('apply_real_lens_maslov', _maslov)

    return out, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--mode', required=True,
                    choices=('pre', 'post', 'post_default', 'post_none'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    import lumenairy
    assert os.path.abspath(lumenairy.__file__).startswith(root), (
        lumenairy.__file__, root)
    import numpy as np
    print('lumenairy.__file__ =', lumenairy.__file__)
    print('lumenairy.__version__ =', lumenairy.__version__)
    print('numpy', np.__version__, 'python', sys.version.split()[0])

    kw = {'pre': {},
          'post': {'renormalize': 'surface', 'sphere_normal': 'generic'},
          'post_default': {},
          'post_none': {'renormalize': None, 'sphere_normal': None},
          }[args.mode]
    digests, errs = run_all(kw)
    payload = {
        'mode': args.mode,
        'root': root,
        'lumenairy_file': lumenairy.__file__,
        'version': lumenairy.__version__,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'kwargs': {k: v for k, v in kw.items()},
        'n_entry_points': len(digests),
        'digests': digests,
        'errors': errs,
    }
    with open(args.out, 'w', encoding='ascii') as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print('entry points digested:', len(digests))
    for k in sorted(digests):
        print(f'  {k}: {len(digests[k])} arrays')
    for k in sorted(errs):
        print('  ERROR', k)
        print(errs[k])


if __name__ == '__main__':
    main()
