"""VERIFY-WP-C2 item 7 -- the two restated knife-edge pins, re-derived.

Three independent questions:

1. **Is the ModalAsymptotic disagreement BIMODAL?**  Both B9 reports
   explain the pin as "one knife-edge pixel changing saddle basin".  That
   predicts a two-mode histogram -- one pixel at the maximum, the rest at
   zero.  This measures the whole per-pixel distribution and reports the
   shape (fraction of pixels above 1 %, 10 %, 50 % of the maximum; the
   median-to-maximum ratio; a dip statistic).
2. **Is the conditioning bar the right bar?**  ``kappa`` is re-measured
   here with a DIFFERENT random direction, a DIFFERENT seed and a
   DIFFERENT ladder from the one the restated test uses, and the linearity
   of the response is reported.
3. **Is ``w6_a2``'s root genuinely off centre?**  The library's offset is
   compared against (a) a 60-digit ``mpmath`` solve of the SAME Gauss-
   Newton system -- an independent oracle of the LINEAR half -- and (b) a
   SECOND Newton step taken from the returned point, which tests the
   nonlinear half: if one step is the root, the second step is negligible
   beside the first.

Also re-measures both pins' margins under all four
``(renormalize, sphere_normal)`` default combinations.

Usage: ``LUMENAIRY_ROOT=<root> python vc2_pins.py OUT.json``
"""
import dataclasses
import importlib.util
import json
import os
import sys

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import numpy as np                                            # noqa: E402
import lumenairy as la                                        # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

import lumenairy.raytrace.trace                              # noqa: E402,F401
import lumenairy.raytrace.world_trace                        # noqa: E402,F401

# the package re-exports the FUNCTIONS under these names, which shadow the
# submodules on attribute access -- reach the modules through sys.modules.
_trace_mod = sys.modules['lumenairy.raytrace.trace']
_world_mod = sys.modules['lumenairy.raytrace.world_trace']

COMBOS = [('surface', 'generic'), ('surface', 'analytic'),
          ('exit', 'generic'), ('exit', 'analytic')]


def _load(rel, name):
    path = os.path.join(_ROOT, *rel)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _set_defaults(renorm, sphere):
    """Flip the two public tracers' defaults on the function objects."""
    for fn in (_trace_mod.trace, _world_mod.trace_world):
        d = list(fn.__defaults__)
        d[-2], d[-1] = renorm, sphere
        fn.__defaults__ = tuple(d)


def _modal(combo):
    ap = _load(('tests', 'unit', 'test_audit_propagation.py'), '_ap')
    from lumenairy.propagators.asymptotic import (
        propagate_modal_asymptotic)
    fit = ap._build_singlet_fit()
    N = 32
    s2x = np.linspace(-5e-6, 5e-6, N)
    s2y = np.linspace(-5e-6, 5e-6, N)
    S2X, S2Y = np.meshgrid(s2x, s2y, indexing='xy')
    amps = {(0, 0): 1.0 + 0.0j}
    kwargs = dict(source_point=(0.0, 0.0), source_amplitudes=amps,
                  pupil_amplitudes=amps, w_s=50e-6, w_p=0.02,
                  v2_centre=(0.0, 0.0), s2_grid_x=S2X, s2_grid_y=S2Y)
    base = propagate_modal_asymptotic(fit, **kwargs)

    cls = ap.TestAuditFixesV4_14_0_agent_1_1APropagateModalAsymptoticStillBitEqual  # noqa: E501
    inst = cls()
    ref = inst._cold_start_reference_propagate_modal_asymptotic(
        fit, amps, amps, 50e-6, 0.02, S2X, S2Y)
    peak = float(np.max(np.abs(ref)))
    d = np.abs(np.asarray(base) - np.asarray(ref)).ravel() / max(peak, 1.0)
    mx = float(np.max(d))

    # --- the SHAPE of the disagreement: bimodal or dense? ----------
    shape = dict(
        n_pixels=int(d.size), max=mx,
        median=float(np.median(d)), p99=float(np.percentile(d, 99)),
        median_over_max=float(np.median(d) / mx) if mx else 0.0,
        frac_above_1pct_of_max=float(np.mean(d > 0.01 * mx)),
        frac_above_10pct_of_max=float(np.mean(d > 0.10 * mx)),
        frac_above_50pct_of_max=float(np.mean(d > 0.50 * mx)),
        n_at_the_max_within_1pct=int(np.sum(d > 0.99 * mx)),
        frac_below_1e_12_relative=float(np.mean(d < 1e-12)))
    # a "one knife-edge pixel" story predicts frac_above_10pct ~ 1/1024
    shape['knife_edge_prediction_frac_above_10pct'] = 1.0 / d.size
    shape['dense_not_bimodal'] = shape['frac_above_10pct_of_max'] > 0.05

    # --- kappa, re-measured with a DIFFERENT direction and ladder --
    rng = np.random.default_rng(7770001)
    direction = rng.normal(size=np.asarray(fit.coef_phi).shape)
    direction /= np.linalg.norm(direction)
    ladder = {}
    for delta in (3e-13, 3e-12, 3e-11, 3e-10):
        alt = dataclasses.replace(
            fit, coef_phi=np.asarray(fit.coef_phi) * (1.0 + delta
                                                      * direction))
        moved = propagate_modal_asymptotic(alt, **kwargs)
        ladder[f'{delta:g}'] = (float(np.max(np.abs(moved - base)))
                                / peak / delta)
    ks = list(ladder.values())
    kappa = float(np.median(ks))
    eps = float(np.finfo(np.float64).eps)
    return dict(combo=f'{combo[0]}/{combo[1]}', peak=peak,
                max_disagreement=mx, shape=shape,
                kappa_ladder=ladder, kappa=kappa,
                kappa_linearity_spread=max(ks) / min(ks),
                eps_kappa=eps * kappa, reading_over_floor=mx / (eps * kappa),
                bar_100x_floor=100.0 * eps * kappa,
                margin_to_bar=(100.0 * eps * kappa) / mx)


def _w6(combo):
    w6 = _load(('tests', 'unit', 'test_niche_audit_w6_asymptotic.py'), '_w6')
    from lumenairy.propagators.asymptotic import (
        _solve_envelope_stationary_batch)
    w6._fit.cache_clear() if hasattr(w6._fit, 'cache_clear') else None
    fit = w6._fit()
    w_s, w_p = 20e-6, 0.02
    v_c = np.array([float(fit.v2x_centre), float(fit.v2y_centre)])
    vx, vy, _ = _solve_envelope_stationary_batch(
        fit, np.array([fit.s2x_centre]), np.array([fit.s2y_centre]),
        0.0, 0.0, w_s=w_s, w_p=w_p, v_cx=v_c[0], v_cy=v_c[1])
    v_star = np.array([float(vx[0]), float(vy[0])])
    offset = v_star - v_c

    def _sys(v):
        s1x, s1y, jxx, jxy, jyx, jyy = fit.eval_s1_with_v2_grad(
            np.asarray(float(fit.s2x_centre)).reshape(()),
            np.asarray(float(fit.s2y_centre)).reshape(()),
            np.asarray(v[0]).reshape(()), np.asarray(v[1]).reshape(()))
        J = np.array([[float(jxx), float(jxy)], [float(jyx), float(jyy)]])
        ds1 = np.array([float(s1x), float(s1y)])
        r = (J.T @ ds1) / w_s ** 2 + (v - v_c) / w_p ** 2
        H = (J.T @ J) / w_s ** 2 + np.eye(2) / w_p ** 2
        return r, H, J, ds1

    r_c, H, J, ds1 = _sys(v_c)
    predicted = -np.linalg.solve(H, r_c)
    svmin = float(np.min(np.linalg.svd(H, compute_uv=False)))
    rn = float(np.linalg.norm(r_c))

    # --- independent 60-digit solve of the SAME linear system ------
    try:
        import mpmath as mp
        mp.mp.dps = 60
        Hm = mp.matrix([[mp.mpf(float(H[i, j])) for j in range(2)]
                        for i in range(2)])
        rm = mp.matrix([mp.mpf(float(r_c[0])), mp.mpf(float(r_c[1]))])
        xm = mp.lu_solve(Hm, -rm)
        oracle = np.array([float(xm[0]), float(xm[1])])
        oracle_err = float(np.max(np.abs(predicted - oracle)))
        have_mp = True
    except Exception as e:
        oracle, oracle_err, have_mp = predicted, float('nan'), str(e)

    # --- the NONLINEAR half: a second Newton step from v_star ------
    r2, H2, _J2, _d2 = _sys(v_star)
    step2 = -np.linalg.solve(H2, r2)
    halfrange = float(fit.v2x_halfrange)
    return dict(
        combo=f'{combo[0]}/{combo[1]}',
        v_centre=v_c.tolist(), v_star=v_star.tolist(),
        offset=offset.tolist(), predicted_step=predicted.tolist(),
        offset_minus_predicted=float(np.max(np.abs(offset - predicted))),
        step_floor_4eps_halfrange=(4.0 * float(np.finfo(np.float64).eps)
                                   * halfrange),
        mpmath_available=have_mp,
        mpmath_60digit_step=oracle.tolist(),
        float64_step_vs_60digit=oracle_err,
        second_newton_step=float(np.max(np.abs(step2))),
        second_over_first=(float(np.max(np.abs(step2)))
                           / max(float(np.max(np.abs(offset))), 1e-300)),
        residual_norm_at_centre=rn,
        residual_norm_at_v_star=float(np.linalg.norm(r2)),
        sigma_min_H=svmin,
        derived_bound_r_over_svmin=(rn / svmin) / halfrange,
        normalised_offset=float(np.max(np.abs(offset)) / halfrange),
        abs_v_star=float(np.max(np.abs(v_star))),
        old_bar_1e_15_would_pass=bool(np.max(np.abs(v_star)) < 1e-15))


def main(out_path):
    res = dict(meta=dict(python=sys.version.split()[0],
                         numpy=np.__version__, lumenairy=la.__version__,
                         file=la.__file__))
    saved = (_trace_mod.trace.__defaults__,
             _world_mod.trace_world.__defaults__)
    modal, w6 = {}, {}
    try:
        for combo in COMBOS:
            _set_defaults(*combo)
            key = f'{combo[0]}/{combo[1]}'
            modal[key] = _modal(combo)
            w6[key] = _w6(combo)
    finally:
        _trace_mod.trace.__defaults__ = saved[0]
        _world_mod.trace_world.__defaults__ = saved[1]
    res['modal'] = modal
    res['w6_a2'] = w6
    res['summary'] = dict(
        modal_reading_over_floor={k: v['reading_over_floor']
                                  for k, v in modal.items()},
        modal_margin_to_100x_bar={k: v['margin_to_bar']
                                  for k, v in modal.items()},
        modal_kappa={k: v['kappa'] for k, v in modal.items()},
        modal_frac_above_10pct_of_max={
            k: v['shape']['frac_above_10pct_of_max']
            for k, v in modal.items()},
        modal_dense_not_bimodal=all(v['shape']['dense_not_bimodal']
                                    for v in modal.values()),
        w6_offset_minus_predicted={k: v['offset_minus_predicted']
                                   for k, v in w6.items()},
        w6_float64_step_vs_60digit={k: v['float64_step_vs_60digit']
                                    for k, v in w6.items()},
        w6_second_over_first={k: v['second_over_first']
                              for k, v in w6.items()},
        w6_abs_v_star={k: v['abs_v_star'] for k, v in w6.items()},
        w6_old_1e_15_bar_would_pass={k: v['old_bar_1e_15_would_pass']
                                     for k, v in w6.items()})
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps(res['summary'], indent=1, sort_keys=True))
    print()
    print('modal shape (surface/generic):',
          json.dumps(modal['surface/generic']['shape'], sort_keys=True))
    print('modal kappa ladder:',
          json.dumps(modal['surface/generic']['kappa_ladder']),
          'spread', modal['surface/generic']['kappa_linearity_spread'])
    print('w6 (surface/generic):',
          json.dumps({k: v for k, v in w6['surface/generic'].items()
                      if k != 'combo'}, sort_keys=True))


if __name__ == '__main__':
    main(sys.argv[1])
