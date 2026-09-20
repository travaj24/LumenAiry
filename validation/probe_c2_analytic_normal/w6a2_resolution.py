"""WP-C2 item 1 -- the DERIVED resolution of the w6_a2 symmetry pin.

``test_w6_a2_v2_star_is_untouched_by_the_verdict_fix`` asserts
``|v2*| < 1e-15`` at the fit centre on a symmetry argument (the exact
answer is 0).  1e-15 is an S4 floor bar: VERIFY-B9 measured 1.150e-15
under ``sphere_normal='analytic'``.  The build-free quantity is the
SOLVER'S OWN RESOLUTION on the root -- measured here as the spread of
the returned ``v2*`` over a ladder of independent Newton starts spread
across the basin, against the basin's own size.

Usage:  LUMENAIRY_ROOT=<root> python w6a2_resolution.py <out.json>
"""
import importlib
import importlib.util
import json
import os
import sys

import numpy as np

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import lumenairy as la  # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

trace_mod = importlib.import_module('lumenairy.raytrace.trace')
wtrace_mod = importlib.import_module('lumenairy.raytrace.world_trace')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _set_defaults(renorm, sph):
    for fn in (trace_mod.trace, wtrace_mod.trace_world):
        d = list(fn.__defaults__)
        d[-2], d[-1] = renorm, sph
        fn.__defaults__ = tuple(d)


def _measure(w6):
    from lumenairy.propagators.asymptotic import solve_envelope_stationary
    from lumenairy.propagators.asymptotic_maslov import _phi_v2_hessian_batch
    w6._fit.cache_clear()
    fit = w6._fit()
    sx, sy = float(fit.s2x_centre), float(fit.s2y_centre)
    vcx, vcy = float(fit.v2x_centre), float(fit.v2y_centre)
    vx, vy, _ = w6._solve_envelope_stationary_batch(
        fit, np.array([sx]), np.array([sy]), 0.0, 0.0,
        w_s=20e-6, w_p=0.02, v_cx=vcx, v_cy=vcy)
    vbx, vby = float(vx[0]), float(vy[0])
    hr = float(fit.v2x_halfrange)
    # ladder of independent Newton starts across the basin
    starts, got = [], []
    for f in (0.0, 1e-6, -1e-6, 1e-4, -1e-4, 1e-2, -1e-2, 0.1, -0.1):
        for g in (0.0, 1e-4, -1e-4, 0.1, -0.1):
            v0 = (vcx + f * hr, vcy + g * hr)
            try:
                v, _, _ = solve_envelope_stationary(
                    fit, (sx, sy), (0.0, 0.0), w_s=20e-6, w_p=0.02,
                    v2_centre=(vcx, vcy), v2_initial=v0)
            except Exception:
                continue
            starts.append(v0)
            got.append((float(v[0]), float(v[1])))
    arr = np.asarray(got)
    spread = float(np.max(np.hypot(
        arr[:, 0] - arr[:, 0].mean(), arr[:, 1] - arr[:, 1].mean())))
    span = float(max(np.ptp(arr[:, 0]), np.ptp(arr[:, 1])))
    worst_scalar = float(np.max(np.abs(arr)))
    H = np.asarray(_phi_v2_hessian_batch(
        fit, np.array([sx]), np.array([sy]),
        np.array([vbx]), np.array([vby])))[0]
    sv = np.linalg.svd(np.asarray(H, dtype=complex), compute_uv=False)
    return dict(
        v_batch=(vbx, vby), worst_batch=max(abs(vbx), abs(vby)),
        n_starts=len(got), start_spread=spread, start_span=span,
        worst_scalar=worst_scalar,
        batch_vs_scalar_max=float(np.max(np.abs(arr - np.array([vbx, vby])))),
        halfrange=hr, svmin=float(np.min(np.abs(sv))),
        svmax=float(np.max(np.abs(sv))),
    )


def main():
    out = sys.argv[1]
    w6 = _load('_w6', os.path.join(_ROOT, 'tests', 'unit',
                                   'test_niche_audit_w6_asymptotic.py'))
    shipped = (trace_mod.trace.__defaults__[-2],
               trace_mod.trace.__defaults__[-1])
    res = {}
    for renorm in ('surface', 'exit'):
        for sph in ('generic', 'analytic'):
            _set_defaults(renorm, sph)
            k = f'{renorm}/{sph}'
            res[k] = _measure(w6)
            print(f'[{k}] |v|batch={res[k]["worst_batch"]:.4e} '
                  f'spread={res[k]["start_spread"]:.4e} '
                  f'span={res[k]["start_span"]:.4e} '
                  f'scalar_worst={res[k]["worst_scalar"]:.4e} '
                  f'hr={res[k]["halfrange"]:.4e}', flush=True)
    _set_defaults(*shipped)
    meta = dict(python=sys.version, numpy=np.__version__,
                lumenairy=la.__version__, platform=sys.platform)
    with open(out, 'w') as fh:
        json.dump(dict(meta=meta, results=res), fh, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
