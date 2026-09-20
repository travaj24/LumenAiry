"""WP-C2 item 6 -- byte identity, ARCHIVE TO ARCHIVE.

Two claims:

1. with the OLD keywords passed explicitly (``sphere_normal='generic'``,
   ``renormalize='surface'``) every trace-touching fixture is
   byte-identical to 49ddf4bd;
2. every NON-trace fixture is byte-identical with no keyword at all.

The probe runs ONE side at a time.  It is invoked twice with two
different roots -- an extracted ``git archive`` of 49ddf4bd and the
working tree -- dumps every array it produces to an ``.npz``, and a
third invocation (``--compare a.npz b.npz``) diffs them.  cwd AND
``PYTHONPATH`` are the root, and ``lumenairy.__file__`` is asserted
inside it before anything else is imported, so an installed lumenairy
cannot bind.  It never runs under pytest and never touches the working
tree's ``lumenairy`` from the archive side.

Usage:
    LUMENAIRY_ROOT=<root> python byte_identity.py dump <out.npz> <mode>
    python byte_identity.py compare <old.npz> <new.npz> <out.json>

``mode`` is ``old`` (pass the pre-5.49.0 keywords explicitly) or
``none`` (pass nothing -- the defaults).
"""
import json
import math
import os
import sys

import numpy as np


def _dump(out_path, mode):
    root = os.environ['LUMENAIRY_ROOT']
    sys.path.insert(0, root)
    import lumenairy as la
    want = os.path.realpath(os.path.join(root, 'lumenairy'))
    got = os.path.realpath(os.path.dirname(la.__file__))
    assert got == want, f'{got!r} is not inside {want!r}'
    print('lumenairy.__file__ =', la.__file__)

    from lumenairy.raytrace.core import Surface
    from lumenairy.raytrace.trace import _make_bundle, trace
    from lumenairy.raytrace.world_trace import trace_world

    WL = 587.6e-9
    # the pre-5.49.0 spelling, passed explicitly
    OLD = (dict(sphere_normal='generic', renormalize='surface')
           if mode == 'old' else {})

    out = {}

    def rec(name, obj):
        for f in ('x', 'y', 'z', 'L', 'M', 'N', 'opd', 'alive',
                  'error_code'):
            v = getattr(obj, f, None)
            if v is not None:
                out[f'{name}.{f}'] = np.asarray(v)

    def rec_arr(name, v):
        out[name] = np.asarray(v)

    def bundle(n, semi, tilt=0.0, seed=20260920):
        rng = np.random.default_rng(seed)
        r = semi * np.sqrt(rng.random(n))
        th = 2 * np.pi * rng.random(n)
        L = np.full(n, math.sin(math.radians(tilt)))
        return _make_bundle(r * np.cos(th), r * np.sin(th), L,
                            np.zeros(n), WL)

    stacks = {
        'spherical7': [
            Surface(radius=0.0515, thickness=0.008, glass_before='air',
                    glass_after='N-BK7', semi_diameter=0.0127),
            Surface(radius=-0.0345, thickness=0.003, glass_before='N-BK7',
                    glass_after='N-SF5', semi_diameter=0.0127),
            Surface(radius=-0.120, thickness=0.010, glass_before='N-SF5',
                    glass_after='air', semi_diameter=0.0127),
            Surface(radius=0.080, thickness=0.006, glass_before='air',
                    glass_after='N-BK7', semi_diameter=0.0127),
            Surface(radius=-0.080, thickness=0.090, glass_before='N-BK7',
                    glass_after='air', semi_diameter=0.0127),
            Surface(radius=np.inf, thickness=0.010, glass_before='air',
                    glass_after='air', semi_diameter=0.02),
            Surface(radius=np.inf, thickness=0.0, glass_before='air',
                    glass_after='air', semi_diameter=0.02)],
        'conic3': [
            Surface(radius=0.0515, conic=-0.6, thickness=0.008,
                    glass_before='air', glass_after='N-BK7',
                    semi_diameter=0.0127),
            Surface(radius=-0.0345, conic=-1.2, thickness=0.100,
                    glass_before='N-BK7', glass_after='air',
                    semi_diameter=0.0127),
            Surface(radius=np.inf, thickness=0.0, glass_before='air',
                    glass_after='air', semi_diameter=0.030)],
        'aspheric3': [
            Surface(radius=0.0515, aspheric_coeffs={4: 1.0e3, 6: -2.0e5},
                    thickness=0.008, glass_before='air',
                    glass_after='N-BK7', semi_diameter=0.0127),
            Surface(radius=-0.0345, thickness=0.100, glass_before='N-BK7',
                    glass_after='air', semi_diameter=0.0127),
            Surface(radius=np.inf, thickness=0.0, glass_before='air',
                    glass_after='air', semi_diameter=0.030)],
        'cassegrain': [
            Surface(radius=-0.400, thickness=-0.150, glass_before='air',
                    glass_after='air', is_mirror=True,
                    semi_diameter=0.050),
            Surface(radius=-0.120, thickness=0.250, glass_before='air',
                    glass_after='air', is_mirror=True,
                    semi_diameter=0.015),
            Surface(radius=np.inf, thickness=0.0, glass_before='air',
                    glass_after='air', semi_diameter=0.030)],
        'biconic3': [
            Surface(radius=0.0515, radius_y=0.060, thickness=0.008,
                    glass_before='air', glass_after='N-BK7',
                    semi_diameter=0.0127),
            Surface(radius=-0.0345, thickness=0.100, glass_before='N-BK7',
                    glass_after='air', semi_diameter=0.0127),
            Surface(radius=np.inf, thickness=0.0, glass_before='air',
                    glass_after='air', semi_diameter=0.030)],
    }

    # ---- TRACE-TOUCHING fixtures, with the old keywords explicit.
    for name, S in stacks.items():
        for tilt in (0.0, 3.0, 7.0):
            for filt in ('all', 'last'):
                res = trace(bundle(2000, 0.0126, tilt), S, WL,
                            output_filter=filt, **OLD)
                rec(f'trace.{name}.t{tilt}.{filt}', res.image_rays)
                if filt == 'all':
                    for i in range(len(S)):
                        rec(f'trace.{name}.t{tilt}.h{i}', res.rays_at(i))
    # DOE kick
    res = trace(bundle(500, 0.0126), stacks['spherical7'], WL,
                surface_diffraction={1: (1.0, 0.0, 5e-6, np.inf)}, **OLD)
    rec('trace.doe', res.image_rays)
    # world trace
    wsurfs = la.world_surfaces_from_prescription(
        la.make_singlet(51.5e-3, -80e-3, 4.1e-3, 'N-BK7', aperture=12e-3))
    res = trace_world(bundle(800, 0.005), wsurfs, WL, **OLD)
    rec('trace_world.singlet', res.image_rays)

    # Public entry points that trace INTERNALLY.  There is no keyword
    # to pass here, so these are exactly the fixtures the default flip
    # is entitled to move -- and they are the interesting half of this
    # sweep.  With the old keywords explicit they cannot move; without
    # them they can, and `byte_identity_defaults_*.json` records which
    # did.
    pres = la.make_singlet(51.5e-3, -80e-3, 4.1e-3, 'N-BK7',
                           aperture=12e-3)
    pres2 = la.make_doublet(R1=51.7e-3, R2=-34.5e-3, R3=-120.0e-3,
                            d1=8e-3, d2=3e-3, glass1='N-BK7',
                            glass2='N-SF5', aperture=25.4e-3)

    def _rec_any(name, v):
        if isinstance(v, dict):
            for k, vv in v.items():
                _rec_any(f'{name}.{k}', vv)
        elif isinstance(v, (tuple, list)):
            for i, vv in enumerate(v):
                _rec_any(f'{name}.{i}', vv)
        elif hasattr(v, '_fields'):
            for k in v._fields:
                _rec_any(f'{name}.{k}', getattr(v, k))
        elif hasattr(v, 'image_rays'):
            rec(name, v.image_rays)
        else:
            try:
                out[name] = np.asarray(v, dtype=float)
            except Exception:
                out[name] = np.asarray([repr(v)[:200]], dtype=object)

    for tag, p in (('singlet', pres), ('doublet', pres2)):
        S = la.surfaces_from_prescription(p)
        semi = 5e-3
        for fld in (0.0, 3.0):
            for fn in ('ray_fan_data', 'opd_fan_data'):
                try:
                    _rec_any(f'{fn}.{tag}.{fld}',
                             getattr(la, fn)(S, WL, semi,
                                             field_angle=fld))
                except Exception as exc:
                    out[f'{fn}.{tag}.{fld}.ERR'] = np.asarray(
                        [repr(exc)[:200]], dtype=object)
            try:
                res = la.trace_prescription(p, WL, semi_aperture=semi,
                                            field_angle=fld)
                rec(f'trace_prescription.{tag}.{fld}', res.image_rays)
                rms, ctr = la.spot_rms(res)
                out[f'spot_rms.{tag}.{fld}'] = np.asarray(
                    [rms, ctr[0], ctr[1]])
                out[f'spot_geo.{tag}.{fld}'] = np.asarray(
                    [la.spot_geo_radius(res)])
                rf = la.refocus(res, 1e-4)
                rec(f'refocus.{tag}.{fld}', rf.image_rays)
            except Exception as exc:
                out[f'trace_prescription.{tag}.{fld}.ERR'] = np.asarray(
                    [repr(exc)[:200]], dtype=object)
        for fn in ('seidel_coefficients', 'first_order_data',
                   'compute_pupils'):
            try:
                _rec_any(f'{fn}.{tag}', getattr(la, fn)(S, WL))
            except Exception as exc:
                out[f'{fn}.{tag}.ERR'] = np.asarray([repr(exc)[:200]],
                                                    dtype=object)
        try:
            _rec_any(f'system_abcd.{tag}',
                     la.system_abcd_prescription(p, WL))
        except Exception as exc:
            out[f'system_abcd.{tag}.ERR'] = np.asarray(
                [repr(exc)[:200]], dtype=object)
        try:
            out[f'paraxial_focus.{tag}'] = np.asarray(
                [la.find_paraxial_focus(S, WL)])
            _rec_any(f'through_focus.{tag}', la.through_focus_rms(
                S, WL, semi, [-2e-4, 0.0, 2e-4]))
        except Exception as exc:
            out[f'through_focus.{tag}.ERR'] = np.asarray(
                [repr(exc)[:200]], dtype=object)
        try:
            from lumenairy.analysis.ghost import enumerate_ghost_paths, ghost_analysis
            paths = enumerate_ghost_paths(len(S))
            out[f'ghost_npaths.{tag}'] = np.asarray([len(paths)])
            rows = ghost_analysis(p, WL, semi_aperture=semi, n_rays=21,
                                  verbose=False)
            for i, row in enumerate(rows[:6]):
                for k, v in row.items():
                    _rec_any(f'ghost.{tag}.{i}.{k}', v)
        except Exception as exc:
            out[f'ghost.{tag}.ERR'] = np.asarray([repr(exc)[:200]],
                                                 dtype=object)

    # ---- NON-TRACE fixtures, NO keyword at all.
    from lumenairy.propagators.propagation import angular_spectrum_propagate
    n = 128
    ax = np.linspace(-1e-3, 1e-3, n)
    X, Y = np.meshgrid(ax, ax, indexing='xy')
    E = np.exp(-(X ** 2 + Y ** 2) / (3e-4) ** 2).astype(np.complex128)
    for zz in (0.01, 0.05):
        rec_arr(f'asm.{zz}', angular_spectrum_propagate(
            E, dx=float(ax[1] - ax[0]), wavelength=WL, z=zz))
    from lumenairy._math.chebyshev import chebyshev_derivative_vandermonde, chebyshev_vandermonde
    u = np.linspace(-1, 1, 51)
    rec_arr('cheb.T', chebyshev_vandermonde(u, 8))
    rec_arr('cheb.dT', chebyshev_derivative_vandermonde(u, 8))
    for name in ('N-BK7', 'N-SF5', 'N-SK16'):
        rec_arr(f'glass.{name}', np.asarray(
            [la.get_glass_index(name, w)
             for w in (486.1e-9, 587.6e-9, 656.3e-9, 1.064e-6)]))

    np.savez(out_path, **{k: v for k, v in out.items()})
    n_vals = sum(int(np.asarray(v).size) for v in out.values())
    print(f'wrote {out_path}: {len(out)} arrays, {n_vals} values')


def _compare(a_path, b_path, out_path):
    A = np.load(a_path, allow_pickle=True)
    B = np.load(b_path, allow_pickle=True)
    ka, kb = set(A.files), set(B.files)
    common = sorted(ka & kb)
    trace_keys = [k for k in common
                  if k.split('.')[0] not in
                  ('asm', 'cheb', 'glass')]
    other_keys = [k for k in common if k not in trace_keys]
    moved = []
    n_vals = 0
    for k in common:
        x, y = A[k], B[k]
        n_vals += int(np.asarray(x).size)
        if x.dtype == object or y.dtype == object:
            # object arrays can nest arrays; compare by repr, which is
            # byte-exact for the floats these hold.
            same = (x.shape == y.shape
                    and repr(x.tolist()) == repr(y.tolist()))
        else:
            same = (x.shape == y.shape
                    and np.array_equal(x, y, equal_nan=True))
        if not same:
            try:
                xf = np.asarray(x, dtype=float)
                yf = np.asarray(y, dtype=float)
                d = float(np.nanmax(np.abs(xf - yf)))
                scale = float(np.nanmax(np.abs(xf)))
                rel = d / scale if scale > 0 else float('inf')
                n_diff = int(np.sum(xf != yf))
            except Exception:
                d = rel = float('nan')
                n_diff = -1
            moved.append(dict(key=k, max_abs_delta=d, max_rel=rel,
                              scale=scale if 'scale' in dir() else None,
                              n_elements_differing=n_diff))
    res = dict(
        n_common=len(common), n_values=n_vals,
        n_trace_keys=len(trace_keys), n_non_trace_keys=len(other_keys),
        only_in_old=sorted(ka - kb), only_in_new=sorted(kb - ka),
        n_moved=len(moved), moved=moved,
        moved_families={f: sum(1 for m in moved
                               if m['key'].split('.')[0] == f)
                        for f in sorted({m['key'].split('.')[0]
                                         for m in moved})},
        worst_abs=(max(m['max_abs_delta'] for m in moved)
                   if moved else 0.0),
        worst_rel=(max(m['max_rel'] for m in moved) if moved else 0.0),
        worst_rel_key=(max(moved, key=lambda m: m['max_rel'])['key']
                       if moved else None),
        all_identical=(not moved and not (ka - kb) and not (kb - ka)),
    )
    print(f"{res['n_common'] - res['n_moved']} / {res['n_common']} arrays "
          f"byte-identical ({res['n_values']} values); "
          f"{res['n_trace_keys']} trace-touching, "
          f"{res['n_non_trace_keys']} non-trace; {res['n_moved']} moved")
    print('  families:', res['moved_families'])
    if moved:
        print('  worst abs %.3e  worst rel %.3e at %s'
              % (res['worst_abs'], res['worst_rel'],
                 res['worst_rel_key']))
    with open(out_path, 'w') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', out_path)


if __name__ == '__main__':
    if sys.argv[1] == 'dump':
        _dump(sys.argv[2], sys.argv[3])
    else:
        _compare(sys.argv[2], sys.argv[3], sys.argv[4])
