"""VERIFY-WP-C2 ROUND 2 -- an INDEPENDENT archive-to-archive way-back census
over all sixteen internally-tracing entry points.

Nothing here is shared with ``validation/probe_c2_round2/r2_wayback_entrypoints.py``:
different prescriptions (a spherical doublet the round-2 probe does not use,
an ASPHERIC singlet, and a two-SPHERICAL-MIRROR system), different apertures,
different field angles, different ray counts, and a SPY that records the
keywords every internal tracer call actually received.

Four arms, each in its own process with its own ``sys.path``:

* ``pre``          -- the ``git archive 49ddf4bd`` tree, no keyword (neither
                      existed on the entry points there);
* ``post_oldkw``   -- this tree with ``renormalize='surface',
                      sphere_normal='generic'`` (the way back);
* ``post_default`` -- this tree, no keyword (the contrast arm: the answers
                      must MOVE, or the way back is a switch that reaches
                      nothing);
* ``post_none``    -- this tree with ``None`` for both (must be byte-identical
                      to ``post_default``).

``--spy`` additionally wraps ``raytrace.trace``, ``raytrace.trace_world`` and
``world_trace.trace_world`` and records, per entry point, the
``(sphere_normal, renormalize)`` every call received -- which is how "the
keyword reaches EVERY trace call inside the entry point" is checked rather
than assumed, including for the entry points that trace more than once
(``through_focus_rms`` scans focus, the ``*_world`` fans trace per fan,
``ray_transfer_jacobian`` traces per finite difference).

Usage::

    python vr2_wayback.py --root <tree> --mode pre --out <json> [--spy]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback

WL = 632.8e-9


# ---------------------------------------------------------------- digests

def _digest(obj, depth=0, seen=None):
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
                out['[%d]%s' % (i, k)] = d
        return out
    if isinstance(obj, dict):
        for kk in sorted(obj, key=repr):
            for k, d in _digest(obj[kk], depth + 1, seen).items():
                out['[%r]%s' % (kk, k)] = d
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
        except Exception:                      # noqa: BLE001
            continue
        if callable(v):
            continue
        for k, d in _digest(v, depth + 1, seen).items():
            out['.%s%s' % (f, k)] = d
    return out


def _tr_digest(res):
    """A ``TraceResult``: the image bundle AND every history bundle."""
    import numpy as np

    out = {}
    bundles = [('image', res.image_rays)]
    for i, b in enumerate(getattr(res, 'ray_history', None) or []):
        if hasattr(b, 'x'):
            bundles.append(('hist%d' % i, b))
    for tag, b in bundles:
        for f in ('x', 'y', 'z', 'L', 'M', 'N', 'opd', 'alive', 'error_code'):
            v = getattr(b, f, None)
            if v is None:
                continue
            v = np.ascontiguousarray(v)
            h = hashlib.sha256()
            h.update(str(v.dtype).encode())
            h.update(str(v.shape).encode())
            h.update(v.tobytes())
            out['%s.%s' % (tag, f)] = h.hexdigest()
    return out


# --------------------------------------------------------------- fixtures
# Three prescriptions, none of them the round-2 probe's.  P_SPH is all
# spherical (the analytic route bites at every surface), P_ASPH is aspheric
# and conic (the analytic route CANNOT bite -- a control for sphere_normal,
# still moved by renormalize), P_MIR is two SPHERICAL mirrors (the analytic
# route bites through _reflect rather than _refract).

def _prescriptions():
    P_SPH = {
        'name': 'vr2 spherical doublet',
        'aperture_diameter': 0.0280,
        'surfaces': [
            {'radius': 0.0731, 'conic': 0.0,
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -0.0437, 'conic': 0.0,
             'glass_before': 'N-BK7', 'glass_after': 'N-SF5'},
            {'radius': -0.1013, 'conic': 0.0,
             'glass_before': 'N-SF5', 'glass_after': 'air'},
        ],
        'thicknesses': [0.0082, 0.0031],
    }
    P_ASPH = {
        'name': 'vr2 aspheric singlet',
        'aperture_diameter': 0.0300,
        'surfaces': [
            {'radius': 0.0450, 'conic': -0.62,
             'aspheric_coeffs': {4: 1.2e-6, 6: -3.4e-10},
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -0.3000, 'conic': -1.7,
             'aspheric_coeffs': {4: -8.0e-7, 6: 5.0e-11},
             'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [0.0055],
    }
    P_MIR = {
        'name': 'vr2 two spherical mirrors',
        'aperture_diameter': 0.0400,
        'surfaces': [
            {'radius': -0.2500, 'conic': 0.0, 'is_mirror': True,
             'glass_before': 'air', 'glass_after': 'air'},
            {'radius': -0.0900, 'conic': 0.0, 'is_mirror': True,
             'glass_before': 'air', 'glass_after': 'air'},
        ],
        'thicknesses': [-0.0900],
    }
    return {'sph': P_SPH, 'asph': P_ASPH, 'mir': P_MIR}


# ------------------------------------------------------------ entry points

def run_all(kw, spy=False):
    import numpy as np

    import lumenairy  # noqa: F401
    from lumenairy.analysis.aberration import caustic_diagnostic
    from lumenairy.analysis.image_plane_wfe import eval_image_plane_wfe
    from lumenairy.propagators.asymptotic_canonical_fit import (
        fit_canonical_polynomials,
        fit_hf_polynomials,
    )
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.differential import ray_transfer_jacobian
    from lumenairy.raytrace.ray_fan import (
        opd_fan_data,
        opd_fan_data_world,
        ray_fan_data,
        ray_fan_data_world,
        through_focus_rms,
    )
    from lumenairy.raytrace.trace import raytrace_system, trace_prescription
    from lumenairy.raytrace.world import paraxial_focus_world, world_surfaces_from_prescription

    P = _prescriptions()
    S = {k: surfaces_from_prescription(v) for k, v in P.items()}
    W = {k: world_surfaces_from_prescription(v) for k, v in P.items()}

    out, errs = {}, {}
    spy_log = {}
    current = {'name': None}

    if spy:
        # The entry points bind ``trace`` / ``trace_world`` at MODULE import
        # time (``from .trace import trace``), so patching the defining
        # module alone reaches only the late-binding callers.  Walk every
        # loaded lumenairy module and replace EVERY attribute that IS one of
        # the two real tracer objects -- that is the only way to see the
        # keywords the ray_fan / differential / trace_prescription entry
        # points actually forward.
        import sys as _sys

        import lumenairy.raytrace.trace  # noqa: F401
        import lumenairy.raytrace.world_trace  # noqa: F401
        # ``lumenairy.raytrace.trace`` the ATTRIBUTE is the function, not
        # the submodule (the package re-exports it), so reach the modules
        # through sys.modules.
        _rtt = _sys.modules['lumenairy.raytrace.trace']
        _rtw = _sys.modules['lumenairy.raytrace.world_trace']

        def _make(real, tag):
            def spied(*a, **k):
                spy_log.setdefault(current['name'] or '<none>', []).append(
                    {'tracer': tag,
                     'sphere_normal': k.get('sphere_normal', '<omitted>'),
                     'renormalize': k.get('renormalize', '<omitted>')})
                return real(*a, **k)
            spied.__name__ = getattr(real, '__name__', tag)
            return spied

        _reals = [(_rtt.trace, 'trace'), (_rtw.trace_world, 'trace_world')]
        _spies = {id(r): _make(r, tg) for r, tg in _reals}
        _n_patched = 0
        for _mn, _m in list(_sys.modules.items()):
            if not _mn.startswith('lumenairy') or _m is None:
                continue
            for _an in list(vars(_m)) if hasattr(_m, '__dict__') else []:
                try:
                    _v = getattr(_m, _an)
                except Exception:              # noqa: BLE001
                    continue
                if id(_v) in _spies:
                    setattr(_m, _an, _spies[id(_v)])
                    _n_patched += 1
        spy_log['<patched attributes>'] = [{'tracer': 'n', 'sphere_normal':
                                            _n_patched, 'renormalize': ''}]

    def _run(name, fn):
        current['name'] = name
        try:
            out[name] = fn()
        except Exception:                      # noqa: BLE001
            errs[name] = traceback.format_exc(limit=5)
        finally:
            current['name'] = None

    # --- raytrace.trace ------------------------------------------------
    for tag, pres, ap, fld in (('sph', P['sph'], 0.0120, 0.025),
                               ('asph', P['asph'], 0.0130, 0.010),
                               ('mir', P['mir'], 0.0150, 0.004)):
        _run('trace_prescription[%s]' % tag,
             (lambda p=pres, a=ap, f=fld: _tr_digest(trace_prescription(
                 p, WL, semi_aperture=a, field_angle=f, num_rings=4,
                 rays_per_ring=19, **kw))))

    for tag, pres, ap in (('sph', P['sph'], 0.0090), ('asph', P['asph'], 0.0100)):
        def _rs(p=pres, a=ap):
            elements = [{'type': 'real_lens', 'prescription': p,
                         'aperture_diameter': p['aperture_diameter']},
                        {'type': 'propagate', 'z': 0.0850}]
            res, _ = raytrace_system(elements, WL, semi_aperture=a,
                                     field_angle=0.015, num_rings=3,
                                     rays_per_ring=13, **kw)
            return _tr_digest(res)
        _run('raytrace_system[%s]' % tag, _rs)

    # --- raytrace.ray_fan ----------------------------------------------
    for tag in ('sph', 'asph', 'mir'):
        _run('ray_fan_data[%s]' % tag, (lambda t=tag: _digest(
            ray_fan_data(S[t], WL, 0.0100, field_angle=0.021,
                         n_rays=33, **kw))))
        _run('opd_fan_data[%s]' % tag, (lambda t=tag: _digest(
            opd_fan_data(S[t], WL, 0.0100, field_angle=0.021,
                         n_rays=33, **kw))))
        _run('ray_fan_data_world[%s]' % tag, (lambda t=tag: _digest(
            ray_fan_data_world(W[t], WL, 0.0100, field_angle=0.021,
                               n_rays=33, **kw))))
        _run('opd_fan_data_world[%s]' % tag, (lambda t=tag: _digest(
            opd_fan_data_world(W[t], WL, 0.0100, field_angle=0.021,
                               n_rays=33, **kw))))
    for tag, zs in (('sph', (0.070, 0.095)), ('asph', (0.060, 0.090))):
        _run('through_focus_rms[%s]' % tag, (lambda t=tag, z=zs: _digest(
            through_focus_rms(S[t], WL, 0.0100,
                              np.linspace(z[0], z[1], 7),
                              field_angle=0.012, num_rings=3,
                              rays_per_ring=13, **kw))))

    # --- raytrace.world ------------------------------------------------
    for tag in ('sph', 'asph'):
        _run('paraxial_focus_world[%s]' % tag, (lambda t=tag: _digest(
            paraxial_focus_world(W[t], WL, aperture_radius=0.0018, **kw))))

    # --- raytrace.differential -----------------------------------------
    for tag in ('sph', 'asph'):
        def _jac(t=tag):
            n = 17
            r = np.linspace(-0.0085, 0.0085, n)
            z = np.zeros(n)
            return _digest(ray_transfer_jacobian(
                r, 0.37 * r, z, z + 0.008, S[t], WL, **kw))
        _run('ray_transfer_jacobian[%s]' % tag, _jac)

    # --- analysis ------------------------------------------------------
    for tag in ('sph', 'asph'):
        _run('eval_image_plane_wfe[%s]' % tag, (lambda t=tag: _digest(
            eval_image_plane_wfe(dict(P[t], object_distance=float('inf')),
                                 WL, field=(0.0, 0.4), n_pupil=13,
                                 field_max_rad=0.022, **kw))))
        _run('caustic_diagnostic[%s]' % tag, (lambda t=tag: _digest(
            caustic_diagnostic(P[t], WL, fan_radius=2.5e-3, n_z_per_gap=11,
                               z_after_last_surface=0.0800, **kw))))

        def _layout(t=tag):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            from lumenairy.analysis.plotting import plot_lens_layout
            fig, ax = plot_lens_layout(P[t], wavelength=WL, show_rays=True,
                                       n_field_angles=2, max_field_deg=1.5,
                                       rays_per_fan=7, **kw)
            data = [np.asarray(ln.get_xydata(), dtype=float)
                    for ln in ax.get_lines()]
            plt.close(fig)
            return _digest(data)
        _run('plot_lens_layout[%s]' % tag, _layout)

    # --- propagators ---------------------------------------------------
    for tag in ('sph', 'asph'):
        _run('fit_canonical_polynomials[%s]' % tag, (lambda t=tag: _digest(
            fit_canonical_polynomials(P[t], WL, n_field=3, n_pupil=5,
                                      poly_order=4, **kw))))
        _run('fit_hf_polynomials[%s]' % tag, (lambda t=tag: _digest(
            fit_hf_polynomials(P[t], WL, n_field=3, n_pupil=5,
                               poly_order=4, **kw))))

    # --- elements ------------------------------------------------------
    def _E_in(n=64, dx=25e-6):
        a = (np.arange(n) - n // 2) * dx
        X, Y = np.meshgrid(a, a, indexing='xy')
        return np.exp(-(X ** 2 + Y ** 2) / (0.35e-3) ** 2).astype(
            np.complex128)

    for tag in ('sph', 'asph'):
        def _traced(t=tag):
            from lumenairy.elements import apply_real_lens_traced
            return _digest(np.asarray(apply_real_lens_traced(
                _E_in(), prescription=P[t], wavelength=WL, dx=25e-6,
                ray_subsample=8, bandlimit=False, on_undersample='silent',
                on_noncollimated='silent', on_aperture_beam='silent',
                on_fit_domain_basis='silent', on_pool_memory='silent',
                n_workers=1, **kw)))
        _run('apply_real_lens_traced[%s]' % tag, _traced)

        def _maslov(t=tag):
            from lumenairy.elements import apply_real_lens_maslov
            return _digest(np.asarray(apply_real_lens_maslov(
                _E_in(), prescription=P[t], wavelength=WL, dx=25e-6,
                ray_field_samples=5, ray_pupil_samples=5, poly_order=4,
                output_subsample=2, **kw)))
        _run('apply_real_lens_maslov[%s]' % tag, _maslov)

    return out, errs, spy_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--mode', required=True,
                    choices=('pre', 'post_oldkw', 'post_default',
                             'post_none'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--spy', action='store_true')
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
          'post_oldkw': {'renormalize': 'surface', 'sphere_normal': 'generic'},
          'post_default': {},
          'post_none': {'renormalize': None, 'sphere_normal': None},
          }[args.mode]
    digests, errs, spy_log = run_all(kw, spy=args.spy)
    n_arrays = sum(len(v) for v in digests.values())
    payload = {
        'mode': args.mode, 'root': root,
        'lumenairy_file': lumenairy.__file__,
        'version': lumenairy.__version__,
        'python': sys.version.split()[0], 'numpy': np.__version__,
        'kwargs': kw, 'n_cases': len(digests), 'n_arrays': n_arrays,
        'digests': digests, 'errors': errs, 'spy': spy_log,
    }
    with open(args.out, 'w', encoding='ascii') as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print('cases:', len(digests), ' arrays:', n_arrays)
    for k in sorted(errs):
        print('  ERROR', k)
        print(errs[k])
    if spy_log:
        for k in sorted(spy_log):
            kinds = sorted({(d['sphere_normal'], d['renormalize'])
                            for d in spy_log[k]})
            print('  SPY %-34s calls=%d kinds=%s'
                  % (k, len(spy_log[k]), kinds))


if __name__ == '__main__':
    main()
