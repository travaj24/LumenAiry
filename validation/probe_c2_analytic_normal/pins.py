"""WP-C2 item 1 -- the structure of the two knife-edge pins, under all
four (renormalize, sphere_normal) default combinations.

Measures, on the running build, for the two
``TestAuditFixesV4_14_0_agent_1_1APropagateModalAsymptoticStillBitEqual``
arms and for ``test_w6_a2_v2_star_is_untouched_by_the_verdict_fix``:

* the reading each arm's bar sees,
* the per-pixel |new - cold_ref| population (bulk floor vs outliers),
* the WARM-start divergence ``max|new - warm_ref|`` -- the wrong-saddle
  signal the v4.15 cold-start change exists to remove, i.e. the smallest
  REAL signal the bar has to stay under,
* for w6_a2, the Newton residual at the returned point and the local
  Hessian conditioning, from which a root-isolation bound follows.

Defaults are flipped exactly as VERIFY-WP-B9 section 4 did: on the
function objects' ``__defaults__``, which is what "flip the default"
means and reaches every module that already imported them.

Usage:  LUMENAIRY_ROOT=<root> python pins.py <out.json>
"""
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

import importlib  # noqa: E402

trace_mod = importlib.import_module('lumenairy.raytrace.trace')
wtrace_mod = importlib.import_module('lumenairy.raytrace.world_trace')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _set_defaults(renorm, sphnorm):
    """Rewrite the two public tracers' keyword defaults in place."""
    for fn in (trace_mod.trace, wtrace_mod.trace_world):
        d = list(fn.__defaults__)
        # signature tail is (..., renormalize, sphere_normal)
        d[-2] = renorm
        d[-1] = sphnorm
        fn.__defaults__ = tuple(d)


def _read_defaults():
    return (trace_mod.trace.__defaults__[-2],
            trace_mod.trace.__defaults__[-1])


def _modal_arm(tp, which):
    """Return the arm's readings: bar reading, populations, warm signal."""
    inst = tp.TestAuditFixesV4_14_0_agent_1_1APropagateModalAsymptoticStillBitEqual()
    fit = tp._build_singlet_fit()
    N = 32
    s2x = np.linspace(-5e-6, 5e-6, N)
    s2y = np.linspace(-5e-6, 5e-6, N)
    S2X, S2Y = np.meshgrid(s2x, s2y, indexing='xy')
    if which == 'lg00':
        src = {(0, 0): 1.0 + 0.0j}
        pup = {(0, 0): 1.0 + 0.0j}
    else:
        src = {(0, 0): 1.0 + 0.0j, (1, 0): 0.3 - 0.1j, (2, 0): 0.05 + 0.02j}
        pup = {(0, 0): 1.0 + 0.0j, (1, 0): 0.2 + 0.0j}
    new = tp.propagate_modal_asymptotic(
        fit, source_point=(0.0, 0.0), source_amplitudes=src,
        pupil_amplitudes=pup, w_s=50e-6, w_p=0.02, v2_centre=(0.0, 0.0),
        s2_grid_x=S2X, s2_grid_y=S2Y)
    cold = inst._cold_start_reference_propagate_modal_asymptotic(
        fit, src, pup, 50e-6, 0.02, S2X, S2Y)
    warm = inst._reference_propagate_modal_asymptotic(
        fit, src, pup, 50e-6, 0.02, S2X, S2Y)
    peak = float(np.max(np.abs(cold)))
    d = np.abs(new - cold)
    rel = d / max(peak, 1.0)
    srt = np.sort(rel.ravel())[::-1]
    warm_peak = float(np.max(np.abs(warm)))
    warm_rel = float(np.max(np.abs(new - warm))) / max(warm_peak, 1.0)
    return dict(
        peak=peak,
        max_rel=float(srt[0]),
        top10=[float(v) for v in srt[:10]],
        n_above_1e_12=int(np.sum(rel > 1e-12)),
        n_above_1e_10=int(np.sum(rel > 1e-10)),
        n_above_1e_9=int(np.sum(rel > 1e-9)),
        n_pixels=int(rel.size),
        bulk_median=float(np.median(rel)),
        bulk_p99=float(np.percentile(rel, 99)),
        warm_signal_rel=warm_rel,
        warm_peak=warm_peak,
        new_energy=float(np.sum(np.abs(new) ** 2)),
        warm_energy=float(np.sum(np.abs(warm) ** 2)),
        new_nz=int(np.sum(new != 0)), warm_nz=int(np.sum(warm != 0)),
    )


def _w6_a2(w6):
    w6._fit.cache_clear()          # lru_cache: the fit is TRACED, so it
    fit = w6._fit()               # must be rebuilt under each default
    vx, vy, _ = w6._solve_envelope_stationary_batch(
        fit, np.array([fit.s2x_centre]), np.array([fit.s2y_centre]),
        0.0, 0.0, w_s=20e-6, w_p=0.02,
        v_cx=fit.v2x_centre, v_cy=fit.v2y_centre)
    vx0, vy0 = float(vx[0]), float(vy[0])
    out = dict(vx=vx0, vy=vy0, worst=max(abs(vx0), abs(vy0)),
               v2x_halfrange=float(fit.v2x_halfrange),
               v2y_halfrange=float(fit.v2y_halfrange),
               v2x_centre=float(fit.v2x_centre),
               v2y_centre=float(fit.v2y_centre))
    # Root-isolation bound: |v* - v_true| <= ||H^-1 g(v*)||, the Newton
    # step that would be taken FROM the returned point.  g = grad_v2 phi,
    # H = its Hessian -- both already available batched.
    from lumenairy.propagators.asymptotic import _phi_v2_hessian_batch
    try:
        H = _phi_v2_hessian_batch(
            fit, np.array([fit.s2x_centre]), np.array([fit.s2y_centre]),
            np.array([vx0]), np.array([vy0]))
        H0 = np.asarray(H)[0]
        out['hess'] = [[complex(H0[i, j]).real for j in range(2)]
                       for i in range(2)]
        sv = np.linalg.svd(np.asarray(H0, dtype=complex), compute_uv=False)
        out['hess_svmin'] = float(np.min(np.abs(sv)))
        out['hess_svmax'] = float(np.max(np.abs(sv)))
    except Exception as exc:          # pragma: no cover - diagnostic
        out['hess_error'] = repr(exc)
    return out


def main():
    out_path = sys.argv[1]
    tp = _load('_tp', os.path.join(_ROOT, 'tests', 'unit',
                                   'test_audit_propagation.py'))
    w6 = _load('_w6', os.path.join(_ROOT, 'tests', 'unit',
                                   'test_niche_audit_w6_asymptotic.py'))
    shipped = _read_defaults()
    results = {}
    for renorm in ('surface', 'exit'):
        for sph in ('generic', 'analytic'):
            _set_defaults(renorm, sph)
            key = f'{renorm}/{sph}'
            results[key] = dict(
                defaults=_read_defaults(),
                lg00=_modal_arm(tp, 'lg00'),
                mode4=_modal_arm(tp, '4mode'),
                w6_a2=_w6_a2(w6),
            )
            print(f'[{key}] lg00 {results[key]["lg00"]["max_rel"]:.4e} '
                  f'4mode {results[key]["mode4"]["max_rel"]:.4e} '
                  f'w6 {results[key]["w6_a2"]["worst"]:.4e}', flush=True)
    _set_defaults(*shipped)
    meta = dict(
        python=sys.version, numpy=np.__version__,
        lumenairy=la.__version__, lumenairy_file=la.__file__,
        platform=sys.platform, shipped_defaults=list(shipped),
    )
    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, results=results), fh, indent=1)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
