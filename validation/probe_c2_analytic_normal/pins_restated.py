"""WP-C2 item 1 -- the RESTATED pins, run under all four
``(renormalize, sphere_normal)`` ray-tracer default combinations.

Calls the test bodies directly (the fixture is a plain function call)
with the two public tracers' ``__defaults__`` rewritten, which is what
"flip the default" means and reaches every module that already imported
them.  Reports pass / fail plus the reading and the bar each arm
derived, so the margins are visible rather than merely green.

Usage:  LUMENAIRY_ROOT=<root> python pins_restated.py <out.json>
"""
import importlib
import importlib.util
import json
import os
import sys
import traceback

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


def _run(fn, *a, **k):
    try:
        fn(*a, **k)
        return dict(status='PASS')
    except AssertionError as exc:
        return dict(status='FAIL', detail=str(exc)[:900])
    except Exception:                       # pragma: no cover - diagnostic
        return dict(status='ERROR', detail=traceback.format_exc()[-900:])


def _margins(tp, fit, which):
    """The reading and the derived bar each modal arm works with."""
    inst = tp.TestAuditFixesV4_14_0_agent_1_1APropagateModalAsymptoticStillBitEqual()
    N = 32
    ax = np.linspace(-5e-6, 5e-6, N)
    S2X, S2Y = np.meshgrid(ax, ax, indexing='xy')
    if which == 'lg00':
        src = {(0, 0): 1.0 + 0.0j}
        pup = {(0, 0): 1.0 + 0.0j}
    else:
        src = {(0, 0): 1.0 + 0.0j, (1, 0): 0.3 - 0.1j, (2, 0): 0.05 + 0.02j}
        pup = {(0, 0): 1.0 + 0.0j, (1, 0): 0.2 + 0.0j}
    kw = dict(source_point=(0.0, 0.0), source_amplitudes=src,
              pupil_amplitudes=pup, w_s=50e-6, w_p=0.02,
              v2_centre=(0.0, 0.0), s2_grid_x=S2X, s2_grid_y=S2Y)
    new = tp.propagate_modal_asymptotic(fit, **kw)
    cold = inst._cold_start_reference_propagate_modal_asymptotic(
        fit, src, pup, 50e-6, 0.02, S2X, S2Y)
    peak = float(np.max(np.abs(cold)))
    reading = float(np.max(np.abs(new - cold))) / peak
    # WP-C2 round 2 (D2): ``_conditioning_bar`` returns a fifth value (the
    # BELOW-bar probe) and its bar is 10x the floor, not 100x, because
    # ``kappa`` is now the true worst case over directions rather than one
    # random draw.
    bar, kappa, probe, pdelta, probe_below = inst._conditioning_bar(
        fit, kw, new, peak)
    injected = float(np.max(np.abs(probe - cold))) / peak
    under = float(np.max(np.abs(probe_below - cold))) / peak
    return dict(reading=reading, bar=bar, kappa=kappa,
                floor=bar / 10.0, margin=bar / reading,
                reading_over_floor=reading / (bar / 10.0),
                injected=injected, injected_over_bar=injected / bar,
                under=under, under_over_bar=under / bar,
                probe_delta=pdelta)


def _w6_margins(w6):
    w6._fit.cache_clear()
    fit = w6._fit()
    w_s, w_p = 20e-6, 0.02
    v_c = np.array([float(fit.v2x_centre), float(fit.v2y_centre)])
    vx, vy, _ = w6._solve_envelope_stationary_batch(
        fit, np.array([fit.s2x_centre]), np.array([fit.s2y_centre]),
        0.0, 0.0, w_s=w_s, w_p=w_p, v_cx=v_c[0], v_cy=v_c[1])
    offset = np.array([float(vx[0]), float(vy[0])]) - v_c
    s1x, s1y, a, b, c, d = fit.eval_s1_with_v2_grad(
        np.asarray(float(fit.s2x_centre)).reshape(()),
        np.asarray(float(fit.s2y_centre)).reshape(()),
        np.asarray(v_c[0]).reshape(()), np.asarray(v_c[1]).reshape(()))
    J = np.array([[float(a), float(b)], [float(c), float(d)]])
    ds1 = np.array([float(s1x), float(s1y)])
    r_c = (J.T @ ds1) / w_s ** 2
    H = (J.T @ J) / w_s ** 2 + np.eye(2) / w_p ** 2
    pred = -np.linalg.solve(H, r_c)
    svmin = float(np.min(np.linalg.svd(H, compute_uv=False)))
    hr = float(fit.v2x_halfrange)
    return dict(
        offset_worst=float(np.max(np.abs(offset))),
        step_residual=float(np.max(np.abs(offset - pred))),
        step_bar=4.0 * float(np.finfo(np.float64).eps) * hr,
        rn=float(np.linalg.norm(r_c)), svmin=svmin,
        norm_offset=float(np.max(np.abs(offset)) / hr),
        derived_bound=(float(np.linalg.norm(r_c)) / svmin) / hr,
    )


def main():
    out = sys.argv[1]
    tp = _load('_tp', os.path.join(_ROOT, 'tests', 'unit',
                                   'test_audit_propagation.py'))
    w6 = _load('_w6', os.path.join(_ROOT, 'tests', 'unit',
                                   'test_niche_audit_w6_asymptotic.py'))
    inst = tp.TestAuditFixesV4_14_0_agent_1_1APropagateModalAsymptoticStillBitEqual()
    shipped = (trace_mod.trace.__defaults__[-2],
               trace_mod.trace.__defaults__[-1])
    res = {}
    for renorm in ('surface', 'exit'):
        for sph in ('generic', 'analytic'):
            _set_defaults(renorm, sph)
            k = renorm + '/' + sph
            fit = tp._build_singlet_fit()
            w6._fit.cache_clear()
            entry = dict(
                lg00=_run(inst.test_lg00_single_mode_bit_equal, fit),
                mode4=_run(inst.test_lg_p0_4mode_prescription_bit_equal,
                           fit),
                w6_a2=_run(
                    w6.test_w6_a2_v2_star_is_untouched_by_the_verdict_fix),
                lg00_margins=_margins(tp, fit, 'lg00'),
                mode4_margins=_margins(tp, fit, '4mode'),
                w6_margins=_w6_margins(w6),
            )
            res[k] = entry
            m = entry['lg00_margins']
            wm = entry['w6_margins']
            print('[%-17s] lg00 %s mode4 %s w6 %s | reading %.4e bar %.4e '
                  '(margin %.1fx, kappa %.3e, injected %.1fx bar) | '
                  'w6 offset %.4e step_resid %.2e bound %.3e'
                  % (k, entry['lg00']['status'], entry['mode4']['status'],
                     entry['w6_a2']['status'], m['reading'], m['bar'],
                     m['margin'], m['kappa'], m['injected_over_bar'],
                     wm['offset_worst'], wm['step_residual'],
                     wm['derived_bound']), flush=True)
            for name in ('lg00', 'mode4', 'w6_a2'):
                if entry[name]['status'] != 'PASS':
                    print('   ' + name + ': ' + entry[name]['detail'])
    _set_defaults(*shipped)
    meta = dict(python=sys.version, numpy=np.__version__,
                lumenairy=la.__version__, lumenairy_file=la.__file__,
                platform=sys.platform, shipped_defaults=list(shipped))
    with open(out, 'w') as fh:
        json.dump(dict(meta=meta, results=res), fh, indent=1)
    allpass = all(res[k][n]['status'] == 'PASS'
                  for k in res for n in ('lg00', 'mode4', 'w6_a2'))
    print('ALL FOUR COMBINATIONS PASS:', allpass)
    print('wrote', out)


if __name__ == '__main__':
    main()
