"""WP-B12 probe C -- the caustic route, re-scored with the repair in place,
and the `aberrated` condition MEASURED (not changed).

For every fixture:

* the routing quantities the gate reads -- NA, the sag-screen aberration
  estimate, the tilt dispersion, the caustic zone -- and the route
  ``_universal_route`` returns at the caustic;
* the three members ``'fga'`` / ``'phase_screen'`` / ``'traced'`` scored
  against the probe's own Rayleigh-Sommerfeld oracle at the caustic, with
  their wall-clock cost;
* for the fixtures whose sag-screen estimate is OVER the 2.0 rad budget (the
  class the ``aberrated`` condition keeps on ``'fga'``), the same three
  members -- that is the measurement the condition turns on.

Run with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b12_common import FIXTURES, assert_tree, build_tag, dump, fidelity  # noqa: E402

ROOT = os.environ.get('B12_TREE', os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
ONLY = [k for k in os.environ.get('B12_ONLY', '').split(',') if k]
SKIP_MEMBERS = {k for k in os.environ.get('B12_SKIP_MEMBERS', '').split(',')
                if k}


def main():
    assert_tree(ROOT)
    import lumenairy as la
    from lumenairy.propagators.fga import (
        _ABERRATION_MAX_RAD,
        _SEIDEL_SA_MAX_RAD,
        _caustic_zone,
        _sag_screen_aberration_rad,
        _system_na,
        _tilt_dispersion,
        _universal_route,
    )
    out = {'build': build_tag(), 'version': la.__version__,
           'aberration_max_rad': float(_ABERRATION_MAX_RAD),
           'fixtures': {}}
    for key in (ONLY or list(FIXTURES)):
        fx = FIXTURES[key]
        p = fx.prescription()
        E = fx.beam()
        rec = {'note': fx.note, 'N': fx.N, 'dx': fx.dx, 'w0': fx.w0,
               'lam': fx.lam}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            na = float(_system_na(p, fx.lam))
            ab = float(_sag_screen_aberration_rad(E, fx.dx, fx.dx, p, fx.lam))
            mv = float(_tilt_dispersion(E, fx.dx, fx.dx, fx.lam, na))
            zone = _caustic_zone(E, fx.dx, p, fx.lam)
        rec.update(system_na=na, sag_screen_aberration_rad=ab,
                   tilt_dispersion=mv, over_budget=bool(ab > _ABERRATION_MAX_RAD),
                   caustic_zone=(None if zone is None
                                 else [float(zone[0]), float(zone[1])]))
        zf = fx.best_focus()
        rec['best_focus_m'] = zf
        rec['airy_radius_m'] = fx.airy_radius()
        # the grid a diffraction oracle needs to RESOLVE this caustic, and the
        # grid the fixture actually has -- the ratio is why some fixtures can
        # be routed but not scored.
        rec['dx_over_airy_radius'] = fx.dx / rec['airy_radius_m']
        rec['N_needed_to_resolve'] = float(
            2.0 * (fx.N / 2 * fx.dx) / (rec['airy_radius_m'] / 3.0))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rec['route_at_caustic'] = _universal_route(
                E, p, fx.lam, fx.dx, fx.dx, zf, 0.12, 3.0, None, 0.06,
                _ABERRATION_MAX_RAD, _SEIDEL_SA_MAX_RAD)
            rec['route_at_vertex'] = _universal_route(
                E, p, fx.lam, fx.dx, fx.dx, 0.0, 0.12, 3.0, None, 0.06,
                _ABERRATION_MAX_RAD, _SEIDEL_SA_MAX_RAD)
        print(f'== {key:22s} NA {na:.4f} ab {ab:8.3f} rad  over_budget '
              f'{rec["over_budget"]!s:5s} route(caustic) '
              f'{rec["route_at_caustic"]:12s} dx/airy '
              f'{rec["dx_over_airy_radius"]:.2f}  N_needed '
              f'{rec["N_needed_to_resolve"]:.0f}')
        if key in SKIP_MEMBERS:
            rec['members'] = 'skipped: caustic not resolvable on this grid'
            out['fixtures'][key] = rec
            continue
        orc = fx.oracle_field(zf)
        rec['members'] = {}
        for m in ('fga', 'phase_screen', 'traced'):
            t0 = time.time()
            try:
                # ``traced``'s default ray_subsample=8 refuses these small
                # apertures (fewer than 32 coarse samples across them); drop it
                # to 2, which is the DENSEST setting and therefore cannot be
                # the reason traced scores badly if it does.
                mkw = {'traced': {'ray_subsample': 2}} if m == 'traced' else None
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fld = la.apply_real_lens_universal(
                        E, prescription=p, wavelength=fx.lam, dx=fx.dx,
                        output_plane_distance=zf, method=m,
                        method_kwargs=mkw)
                rec['members'][m] = {
                    'fidelity': fidelity(fld, orc),
                    'power_ratio': float(np.sum(np.abs(fld) ** 2)
                                         / np.sum(np.abs(orc) ** 2)),
                    'rms_um': _rms_radius(fld, fx) * 1e6,
                    'seconds': time.time() - t0,
                }
            except Exception as exc:                          # noqa: BLE001
                rec['members'][m] = {'error': f'{type(exc).__name__}: {exc}'}
        rec['oracle_rms_um'] = _rms_radius(orc, fx) * 1e6
        for m, r in rec['members'].items():
            if 'fidelity' in r:
                print(f'   {m:13s} fid {r["fidelity"]:.4f}  P '
                      f'{r["power_ratio"]:.3f}  rms {r["rms_um"]:7.3f} um  '
                      f'{r["seconds"]:7.2f} s')
            else:
                print(f'   {m:13s} {r["error"]}')
        print(f'   {"oracle":13s} rms {rec["oracle_rms_um"]:7.3f} um')
        out['fixtures'][key] = rec
    dump(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      f'probe_c_route_{sys.platform}_'
                      f'{sys.version_info.major}{sys.version_info.minor}.json'),
         out)


def _rms_radius(fld, fx):
    a2 = np.abs(np.asarray(fld)) ** 2
    X, Y = fx.grid()
    tot = a2.sum()
    if tot <= 0:
        return float('nan')
    cx = (a2 * X).sum() / tot
    cy = (a2 * Y).sum() / tot
    return float(np.sqrt((a2 * ((X - cx) ** 2 + (Y - cy) ** 2)).sum() / tot))


if __name__ == '__main__':
    main()
