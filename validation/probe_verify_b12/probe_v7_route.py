"""VERIFY-WP-B12 probe V7 -- the caustic-route re-score, on MY optic.

WP-B12 section 6 walks the ``aberrated`` gate on ONE optic (VERIFY-B7b's
fixture V) by beam radius alone, and makes four recommendations.  This probe
reproduces that ladder on MY OWN flat-last-surface plano-convex -- a different
glass, a different wavelength, a different focal length -- and scores the three
members against MY OWN angular-spectrum oracle at each rung's own traced best
focus.

The optic's LAST surface is FLAT on purpose, exactly as in the report: then
nothing on the ladder moves with WP-B12 and the reading is about the SCREEN's
model error, not about the repair.  A CURVED-last-surface rung is added at the
end, where the repair does move the ``'fga'`` column, so the two can be
compared.

The route is READ, never changed.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vb12_common as C  # noqa: E402

W0_LADDER = (34e-6, 46e-6, 57e-6, 68e-6, 79e-6, 90e-6, 101e-6)


def score(fx, members=('fga', 'phase_screen', 'traced'), subsample=2):
    from lumenairy.propagators.fga import (
        _ABERRATION_MAX_RAD,
        _sag_screen_aberration_rad,
        _universal_route,
        apply_real_lens_universal,
    )
    zf = fx.best_focus()
    E_in = fx.E_in()
    O, _ = fx.oracle_field(zf, refine=4)
    est = float(_sag_screen_aberration_rad(E_in, fx.dx, fx.dx,
                                           fx.prescription(), fx.lam))
    route = _universal_route(E_in, fx.prescription(), fx.lam, fx.dx, fx.dx,
                             zf, 0.12, 3.0, None, 0.06, _ABERRATION_MAX_RAD)
    rec = dict(w0=fx.w0, sag_screen_est_rad=est, route=route,
               best_focus_m=zf, budget=_ABERRATION_MAX_RAD, members={})
    for m in members:
        kw = {'traced': {'ray_subsample': subsample}} if m == 'traced' else {}
        t0 = time.time()
        try:
            F = apply_real_lens_universal(
                E_in, prescription=fx.prescription(), wavelength=fx.lam,
                dx=fx.dx, output_plane_distance=zf, method=m,
                method_kwargs=kw)
            rec['members'][m] = dict(fidelity=C.fidelity(F, O),
                                     power=C.power_ratio(F, O),
                                     seconds=time.time() - t0)
        except Exception as exc:                            # noqa: BLE001
            rec['members'][m] = dict(refused=f'{type(exc).__name__}: {exc}')
    return rec


def main():
    import copy as _c

    import lumenairy as la
    print('lumenairy.__file__ =', os.path.abspath(la.__file__), flush=True)
    out = {'env': C.env_block(), 'ladder': [], 'curved': []}
    base = C.fixture('flat_planoconvex')
    for w0 in W0_LADDER:
        fx = _c.copy(base)
        fx.w0 = w0
        r = score(fx)
        out['ladder'].append(r)
        ms = r['members']
        print(f"[flat w0={w0 * 1e6:5.0f} um] est={r['sag_screen_est_rad']:7.3f} rad "
              f"route={r['route']:12s} " + '  '.join(
                  f"{k}={v.get('fidelity', float('nan')):.4f}"
                  f"({v.get('seconds', 0):5.1f}s)" for k, v in ms.items()),
              flush=True)
    for key in ('asph', 'menisc'):
        fx = C.fixture(key)
        r = score(fx)
        r['fixture'] = key
        out['curved'].append(r)
        ms = r['members']
        print(f"[curved {key:8s}] est={r['sag_screen_est_rad']:7.3f} rad "
              f"route={r['route']:12s} " + '  '.join(
                  f"{k}={v.get('fidelity', float('nan')):.4f}"
                  f"({v.get('seconds', 0):5.1f}s)" for k, v in ms.items()),
              flush=True)
    C.dump(out, 'probe_v7_route')


if __name__ == '__main__':
    main()
