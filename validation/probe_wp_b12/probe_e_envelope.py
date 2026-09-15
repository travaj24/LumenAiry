"""WP-B12 probe E -- the `aberrated` condition, measured as a LADDER.

``_universal_route``'s caustic branch sends a single-valued field to
``'phase_screen'`` while the sag-screen aberration estimate is under
``_ABERRATION_MAX_RAD`` (2.0 rad) and to ``'fga'`` above it.  Probe C measures
the three members at a handful of estimates; this probe walks ONE optic across
the whole envelope by the beam radius alone -- every other parameter fixed --
so the screen's loss can be read against the estimate the gate actually tests.

That is the measurement the condition turns on.  This probe does NOT change the
condition; it reports where the screen's error crosses each decade.

Run with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 and
PYTHONPATH pointing at the tree under test.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b12_common import Fixture, assert_tree, build_tag, dump, fidelity  # noqa: E402

ROOT = os.environ.get('B12_TREE', os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# VERIFY-B7b's fixture V, swept in beam radius only.  Its last surface is FLAT,
# so nothing on this ladder moves with WP-B12 -- which is what makes it the
# right optic for a route question: the ladder reads the same before and after
# the repair, and it is about the SCREEN's model error, not about FGA's.
W0_LADDER = (60e-6, 100e-6, 140e-6, 165e-6, 190e-6, 205e-6)


def main():
    assert_tree(ROOT)
    import lumenairy as la
    from lumenairy.propagators.fga import (
        _ABERRATION_MAX_RAD,
        _SEIDEL_SA_MAX_RAD,
        _sag_screen_aberration_rad,
        _system_na,
        _universal_route,
    )
    out = {'build': build_tag(), 'version': la.__version__,
           'aberration_max_rad': float(_ABERRATION_MAX_RAD), 'rungs': []}
    for w0 in W0_LADDER:
        fx = Fixture(f'V_w{int(w0 * 1e6)}', 'N-LASF9', 1.45e-3, float('inf'),
                     0.55e-3, 0.262e-3, 0.850e-6, w0, 384, 1.6e-6,
                     note='fixture V, beam radius swept')
        p = fx.prescription()
        E = fx.beam()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            na = float(_system_na(p, fx.lam))
            ab = float(_sag_screen_aberration_rad(E, fx.dx, fx.dx, p, fx.lam))
        zf = fx.best_focus()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            route = _universal_route(E, p, fx.lam, fx.dx, fx.dx, zf, 0.12, 3.0,
                                     None, 0.06, _ABERRATION_MAX_RAD,
                                     _SEIDEL_SA_MAX_RAD)
        orc = fx.oracle_field(zf)
        row = {'w0_m': w0, 'system_na': na, 'sag_screen_aberration_rad': ab,
               'over_budget': bool(ab > _ABERRATION_MAX_RAD),
               'route_at_caustic': route, 'best_focus_m': zf, 'members': {}}
        for m in ('fga', 'phase_screen', 'traced'):
            t0 = time.time()
            mkw = {'traced': {'ray_subsample': 2}} if m == 'traced' else None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fld = la.apply_real_lens_universal(
                        E, prescription=p, wavelength=fx.lam, dx=fx.dx,
                        output_plane_distance=zf, method=m, method_kwargs=mkw)
                row['members'][m] = {'fidelity': fidelity(fld, orc),
                                     'seconds': time.time() - t0}
            except Exception as exc:                          # noqa: BLE001
                row['members'][m] = {'error': f'{type(exc).__name__}: {exc}'}
        f = row['members']
        print(f'w0 {w0 * 1e6:6.1f} um  ab {ab:7.3f} rad  route '
              f'{route:12s}  fga {_g(f, "fga")}  screen {_g(f, "phase_screen")}'
              f'  traced {_g(f, "traced")}')
        out['rungs'].append(row)
    dump(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      f'probe_e_envelope_{sys.platform}_'
                      f'{sys.version_info.major}{sys.version_info.minor}.json'),
         out)


def _g(f, m):
    r = f.get(m, {})
    return (f'{r["fidelity"]:.4f}' if 'fidelity' in r
            else r.get('error', '?')[:28])


if __name__ == '__main__':
    main()
