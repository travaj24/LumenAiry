"""Tests for the design-optimization stack (merits + global optimizers).

Merges the former ``test_merits.py`` (merit-function unit tests) with
the optimizer-convergence tests pulled from ``physics_deep_test.py``,
``physics_extended_test.py``, ``new_features_deep_test.py``, and
``deep_audit.py``.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from _harness import Harness

import lumenairy as op
from lumenairy.optimize import EvaluationContext


H = Harness('optimize')


def _make_test_pres(aperture=4e-3):
    return op.make_singlet(50e-3, float('inf'), 4e-3, 'N-BK7',
                           aperture=aperture)


def _ctx_with_opd(opd_map, dx, prescription, wavelength=1.31e-6):
    return EvaluationContext(
        prescription=prescription, wavelength=wavelength,
        N=opd_map.shape[0], dx=dx,
        opd_map=opd_map.astype(np.float64))


# ---------------------------------------------------------------------
# Merit-function unit tests (from the former test_merits.py)
# ---------------------------------------------------------------------

H.section('Merit-function behaviour')


def t_match_ideal_zero_on_ideal():
    N = 256; dx = 16e-6; ap = 3e-3; f = 100e-3
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    opd_ideal = -(X**2 + Y**2) / (2.0 * f)
    pres = _make_test_pres(aperture=ap)
    ctx = _ctx_with_opd(opd_ideal, dx, pres)
    m = op.MatchIdealThinLensMerit(target_focal_length=f, weight=1.0,
                                   exclude_low_order=4, n_modes=15)
    val = m.evaluate(ctx)
    return val < 1e-4, f'merit on ideal = {val:.3e} (expect ~0)'


H.run('MatchIdealThinLens: ~0 on ideal wavefront',
      t_match_ideal_zero_on_ideal)


def t_match_ideal_nonzero_on_aberrated():
    N = 256; dx = 16e-6; ap = 3e-3; f = 100e-3
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    opd_aberr = (-r2 / (2.0 * f)
                 + 100e-9 * (r2 / (ap/2)**2)**2)
    pres = _make_test_pres(aperture=ap)
    ctx = _ctx_with_opd(opd_aberr, dx, pres)
    m = op.MatchIdealThinLensMerit(target_focal_length=f, weight=1.0,
                                   exclude_low_order=4, n_modes=15)
    val = m.evaluate(ctx)
    return 1e-5 < val < 1.0, f'merit on aberrated = {val:.3e}'


H.run('MatchIdealThinLens: > 0 on aberrated wavefront',
      t_match_ideal_nonzero_on_aberrated)


def t_match_ideal_scales_with_aberration():
    N = 256; dx = 16e-6; ap = 3e-3; f = 100e-3
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    pres = _make_test_pres(aperture=ap)
    m = op.MatchIdealThinLensMerit(target_focal_length=f, weight=1.0,
                                   exclude_low_order=4, n_modes=15)
    vals = []
    for amp in [50e-9, 100e-9, 200e-9]:
        opd = -r2 / (2.0 * f) + amp * (r2 / (ap/2)**2)**2
        val = m.evaluate(_ctx_with_opd(opd, dx, pres))
        vals.append(val)
    ratio_low = vals[1] / vals[0]
    ratio_high = vals[2] / vals[1]
    ok = (3.5 < ratio_low < 4.5) and (3.5 < ratio_high < 4.5)
    return ok, (f'ratios = {ratio_low:.2f}, {ratio_high:.2f} '
                f'(expect ~4 each)')


H.run('MatchIdealThinLens: ~quadratic scaling with amplitude',
      t_match_ideal_scales_with_aberration)


def t_match_target_zero_when_equal():
    N = 256; dx = 16e-6; ap = 3e-3
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    target = -(X**2 + Y**2) / 0.2 + 1e-9 * X**3
    pres = _make_test_pres(aperture=ap)
    m = op.MatchTargetOPDMerit(target_opd=target, weight=1.0,
                               exclude_low_order=4, n_modes=15)
    val = m.evaluate(_ctx_with_opd(target.copy(), dx, pres))
    return val < 1e-6, f'merit when wavefronts equal = {val:.3e}'


H.run('MatchTargetOPD: ~0 when wavefront == target',
      t_match_target_zero_when_equal)


def t_match_target_callable():
    N = 256; dx = 16e-6; ap = 3e-3
    pres = _make_test_pres(aperture=ap)
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    actual = -(X**2 + Y**2) / 0.2

    def target_fn(X_, Y_, prescription):
        return -(X_**2 + Y_**2) / 0.2

    m = op.MatchTargetOPDMerit(target_opd=target_fn, weight=1.0,
                               exclude_low_order=4, n_modes=15)
    val = m.evaluate(_ctx_with_opd(actual, dx, pres))
    return val < 1e-6, f'merit with callable target = {val:.3e}'


H.run('MatchTargetOPD: callable target works', t_match_target_callable)


def t_zernike_coeff_target():
    N = 256; dx = 16e-6; ap = 3e-3
    pres = _make_test_pres(aperture=ap)
    coeffs_in = np.zeros(15)
    coeffs_in[4] = 30e-9
    coeffs_in[12] = 50e-9
    opd = op.zernike_reconstruct(coeffs_in, dx, (N, N), ap)
    m = op.ZernikeCoefficientMerit(targets={12: 0.0}, weight=1.0,
                                   n_modes=15)
    val = m.evaluate(_ctx_with_opd(opd, dx, pres))
    expected = (50e-9 / 1.31e-6) ** 2
    rel_err = abs(val - expected) / expected
    return rel_err < 0.05, (f'merit = {val:.3e}, expected = '
                            f'{expected:.3e}, rel err {rel_err:.2%}')


H.run('ZernikeCoefficient: hits target value with known input',
      t_zernike_coeff_target)


def t_zernike_coeff_zero_when_match():
    N = 256; dx = 16e-6; ap = 3e-3
    pres = _make_test_pres(aperture=ap)
    coeffs_in = np.zeros(15)
    coeffs_in[12] = 75e-9
    opd = op.zernike_reconstruct(coeffs_in, dx, (N, N), ap)
    m = op.ZernikeCoefficientMerit(targets={12: 75e-9}, weight=1.0,
                                   n_modes=15)
    val = m.evaluate(_ctx_with_opd(opd, dx, pres))
    return val < 1e-6, f'merit when target met = {val:.3e}'


H.run('ZernikeCoefficient: ~0 when target == actual',
      t_zernike_coeff_zero_when_match)


def t_composite_sum():
    N = 64
    pres = _make_test_pres(aperture=3e-3)
    ctx = EvaluationContext(
        prescription=pres, wavelength=1.31e-6, N=N, dx=32e-6,
        efl=99e-3, bfl=80e-3,
        seidel=np.array([1e-5, 0, 0, 0, 0]))
    a = op.FocalLengthMerit(target=100e-3, weight=1.0)
    b = op.SphericalSeidelMerit(weight=1.0)
    composite = op.CompositeMerit([a, b], weight=1.0)
    val_composite = composite.evaluate(ctx)
    val_sum = a.evaluate(ctx) + b.evaluate(ctx)
    return abs(val_composite - val_sum) < 1e-15, \
        f'composite={val_composite:.3e}, sum={val_sum:.3e}'


H.run('CompositeMerit: sums components correctly', t_composite_sum)


def t_composite_weight():
    N = 64
    pres = _make_test_pres(aperture=3e-3)
    ctx = EvaluationContext(
        prescription=pres, wavelength=1.31e-6, N=N, dx=32e-6,
        efl=99e-3, bfl=80e-3,
        seidel=np.array([1e-5, 0, 0, 0, 0]))
    a = op.FocalLengthMerit(target=100e-3, weight=1.0)
    b = op.SphericalSeidelMerit(weight=1.0)
    base = op.CompositeMerit([a, b], weight=1.0).evaluate(ctx)
    scaled = op.CompositeMerit([a, b], weight=3.5).evaluate(ctx)
    return abs(scaled - 3.5 * base) < 1e-12 * max(abs(base), 1e-12), \
        f'scaled={scaled:.3e} vs base*3.5={3.5*base:.3e}'


H.run('CompositeMerit: weight scales the total', t_composite_weight)


def t_callable_merit_passes_ctx():
    pres = _make_test_pres(aperture=3e-3)
    ctx = EvaluationContext(
        prescription=pres, wavelength=1.31e-6, N=64, dx=32e-6,
        efl=99e-3, bfl=80e-3, strehl_best=0.5)
    def fn(c):
        return c.efl + c.strehl_best
    m = op.CallableMerit(fn, weight=2.0)
    val = m.evaluate(ctx)
    expected = 2.0 * (99e-3 + 0.5)
    return abs(val - expected) < 1e-15, \
        f'val={val:.6e}, expected={expected:.6e}'


H.run('CallableMerit: passes ctx and applies weight',
      t_callable_merit_passes_ctx)


def t_multifield_merit():
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=3e-3)
    from lumenairy.raytrace import (
        surfaces_from_prescription, system_abcd)
    surfs = surfaces_from_prescription(pres)
    _, efl, bfl, _ = system_abcd(surfs, 1.31e-6)
    ctx = EvaluationContext(
        prescription=pres, wavelength=1.31e-6, N=64, dx=32e-6,
        efl=float(efl), bfl=float(bfl))
    sub = op.StrehlMerit(min_strehl=0.9, weight=1.0)
    mf = op.MultiFieldMerit(field_angles=[0.0, 0.005, 0.01],
                            sub_merit=sub)
    val = mf.evaluate(ctx)
    return np.isfinite(val), f'multi-field merit = {val:.4e}'


H.run('MultiFieldMerit: evaluates at 3 field angles', t_multifield_merit)


# ---------------------------------------------------------------------
# Optimizer-convergence tests
# ---------------------------------------------------------------------

H.section('Optimizer convergence')


def t_optimizer_converges_to_efl():
    template = op.make_singlet(60e-3, np.inf, 4e-3, 'N-BK7',
                               aperture=10e-3)
    param = op.DesignParameterization(
        template=template,
        free_vars=[('surfaces', 0, 'radius')],
        bounds=[(30e-3, 100e-3)])
    merit = [op.FocalLengthMerit(target=100e-3, weight=1.0)]
    result = op.design_optimize(param, merit, wavelength=1.31e-6,
                                N=64, dx=64e-6, max_iter=30,
                                verbose=False)
    err_pct = abs(result.context_final.efl - 100e-3) / 100e-3 * 100
    return err_pct < 0.1, \
        f'EFL={result.context_final.efl*1e3:.4f}mm, err={err_pct:.4f}%'


H.run('Optimizer: converges to target EFL',
      t_optimizer_converges_to_efl)


def t_de_optimizer_converges():
    template = op.make_singlet(60e-3, np.inf, 4e-3, 'N-BK7',
                               aperture=10e-3)
    p = op.DesignParameterization(template,
        [('surfaces', 0, 'radius')], [(30e-3, 100e-3)])
    r = op.design_optimize(p, [op.FocalLengthMerit(100e-3)], 1.31e-6,
                           N=64, dx=64e-6, method='de', max_iter=30,
                           verbose=False)
    err = abs(r.context_final.efl - 100e-3)
    return err < 0.5e-3, f'EFL err = {err*1e3:.4f} mm'


H.run('DE optimizer: converges to EFL target', t_de_optimizer_converges)


def t_basin_hopping_converges():
    template = op.make_singlet(60e-3, np.inf, 4e-3, 'N-BK7',
                               aperture=10e-3)
    p = op.DesignParameterization(template,
        [('surfaces', 0, 'radius')], [(30e-3, 100e-3)])
    r = op.design_optimize(p, [op.FocalLengthMerit(100e-3)], 1.31e-6,
                           N=64, dx=64e-6, method='basin_hopping',
                           max_iter=10, verbose=False)
    return r.merit < 0.01, f'merit = {r.merit:.4e}'


H.run('Basin-hopping: finds local minimum', t_basin_hopping_converges)


def t_optimizer_match_ideal():
    template = op.make_doublet(
        R1=80e-3, R2=-40e-3, R3=-200e-3, d1=6e-3, d2=2.5e-3,
        glass1='N-BAF10', glass2='N-SF6HT', aperture=3e-3)
    param = op.DesignParameterization(
        template=template,
        free_vars=[('surfaces', 0, 'radius'),
                   ('surfaces', 1, 'radius'),
                   ('surfaces', 2, 'radius')],
        bounds=[(40e-3, 100e-3), (-80e-3, -25e-3),
                (-300e-3, -120e-3)])
    merit_terms = [
        op.FocalLengthMerit(target=85e-3, weight=1.0),
        op.MatchIdealThinLensMerit(target_focal_length=85e-3,
                                   weight=10.0, exclude_low_order=4),
    ]
    result = op.design_optimize(param, merit_terms,
                                wavelength=1.31e-6, N=64, dx=32e-6,
                                max_iter=30, verbose=False)
    return result.merit < 1.0, f'final merit = {result.merit:.3e}'


H.run('Optimizer: MatchIdealThinLens converges', t_optimizer_match_ideal)


def t_match_ideal_in_optimizer():
    template = op.make_doublet(
        R1=80e-3, R2=-40e-3, R3=-200e-3, d1=6e-3, d2=2.5e-3,
        glass1='N-BAF10', glass2='N-SF6HT', aperture=3e-3)
    param = op.DesignParameterization(
        template=template,
        free_vars=[('surfaces', 0, 'radius'),
                   ('surfaces', 1, 'radius'),
                   ('surfaces', 2, 'radius')],
        bounds=[(40e-3, 100e-3), (-80e-3, -25e-3),
                (-300e-3, -120e-3)])
    merit_terms = [
        op.FocalLengthMerit(target=85e-3, weight=1.0),
        op.MatchIdealThinLensMerit(target_focal_length=85e-3,
                                   weight=10.0,
                                   exclude_low_order=4, n_modes=15),
    ]
    res = op.design_optimize(parameterization=param,
                             merit_terms=merit_terms,
                             wavelength=1.31e-6, N=64, dx=32e-6,
                             method='L-BFGS-B', max_iter=30,
                             verbose=False)
    init_pres = template
    init_ctx = EvaluationContext(prescription=init_pres,
                                 wavelength=1.31e-6, N=64, dx=32e-6)
    E_in = np.ones((64, 64), dtype=np.complex128)
    E_init = op.apply_real_lens(E_in, init_pres, 1.31e-6, 32e-6)
    from lumenairy.analysis import wave_opd_2d
    try:
        _, _, opd_init = wave_opd_2d(E_init, 32e-6, 1.31e-6,
                                     aperture=3e-3,
                                     focal_length=85e-3, f_ref=85e-3)
        init_ctx.opd_map = opd_init
    except Exception:
        return False, 'could not get initial opd map'
    init_merit = sum(m.evaluate(init_ctx) for m in merit_terms
                     if not m.needs_wave or init_ctx.opd_map is not None)
    final_merit = res.merit
    return final_merit < init_merit, \
        f'initial={init_merit:.3e}, final={final_merit:.3e}'


H.run('MatchIdealThinLens: optimizer reduces it',
      t_match_ideal_in_optimizer)


def t_design_optimize_focal_length_target():
    template = op.make_singlet(60e-3, float('inf'), 4e-3, 'N-BK7',
                               aperture=10e-3)
    param = op.DesignParameterization(
        template=template,
        free_vars=[('surfaces', 0, 'radius')],
        bounds=[(30e-3, 100e-3)])
    merit = [op.FocalLengthMerit(target=100e-3, weight=1.0)]
    res = op.design_optimize(parameterization=param, merit_terms=merit,
                             wavelength=1.31e-6, N=64, dx=64e-6,
                             method='L-BFGS-B', max_iter=30,
                             verbose=False)
    err = abs(res.context_final.efl - 100e-3) / 100e-3
    return err < 1e-4, f'rel err = {err:.2e}'


H.run('design_optimize focal-length target',
      t_design_optimize_focal_length_target)


if __name__ == '__main__':
    sys.exit(H.summary())
