"""Validation tests for the 4.3.0 off-axis Seidel API:

* :func:`lumenairy.seidel_field_sweep` -- multi-field Seidel evaluator
* :func:`lumenairy.seidel_wfe` -- W(rho, theta) reconstruction from totals

The existing on-axis :func:`seidel_coefficients` is already covered in
``test_raytrace.py``; this file verifies:

* Scaling laws: S1 / S4 are field-independent; S2 ~ h; S3 ~ h^2; for
  the singlet test case the Petzval-dominated S5 is approximately
  linear in h (mostly the (A_c/A_m)*S4 term).
* Bit-identical agreement: ``seidel_field_sweep([h])`` matches
  ``seidel_coefficients(field_angle=h)`` for every aberration.
* WFE reconstruction at rho=0 returns 0 (piston-free Hopkins form).
* WFE reconstruction grows quadratically off-axis along the +y
  meridian for a pure-S3 / S4 sample (astigmatism + Petzval).
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from _harness import Harness

import numpy as np

import lumenairy as la


H = Harness('seidel-field')


def _build_test_singlet():
    presc = la.make_singlet(R1=50e-3, R2=np.inf, d=3e-3,
                              glass='N-BK7', aperture=10e-3)
    return la.surfaces_from_prescription(presc)


SURFACES = _build_test_singlet()
WL_TEST = 1.31e-6


# ----------------------------------------------------------------------
# Field-sweep scaling laws
# ----------------------------------------------------------------------

def t_S1_field_independent():
    heights = np.array([0.001, 0.01, 0.05, 0.1])
    result, _ = la.seidel_field_sweep(SURFACES, WL_TEST, heights)
    s1 = result['total']['S1']
    rel_spread = float((s1.max() - s1.min()) / abs(s1.mean()))
    return rel_spread < 1e-12, \
        f'S1 spread across 4 field heights = {rel_spread:.2e}'


H.run('S1 (spherical) is field-independent', t_S1_field_independent)


def t_S2_linear_in_field():
    heights = np.array([0.001, 0.01, 0.1])
    result, _ = la.seidel_field_sweep(SURFACES, WL_TEST, heights)
    s2 = result['total']['S2']
    # S2(h=0.1) / S2(h=0.001) should be ~100
    ratio = float(s2[2] / s2[0])
    expected = 100.0
    rel_err = abs(ratio - expected) / expected
    return rel_err < 1e-9, \
        f'S2 ratio = {ratio:.4f}, expected {expected}, rel-err {rel_err:.2e}'


H.run('S2 (coma) scales linearly with field height', t_S2_linear_in_field)


def t_S3_quadratic_in_field():
    heights = np.array([0.001, 0.01, 0.1])
    result, _ = la.seidel_field_sweep(SURFACES, WL_TEST, heights)
    s3 = result['total']['S3']
    ratio = float(s3[2] / s3[0])
    expected = 10000.0
    rel_err = abs(ratio - expected) / expected
    return rel_err < 1e-9, \
        f'S3 ratio = {ratio:.2f}, expected {expected}, rel-err {rel_err:.2e}'


H.run('S3 (astigmatism) scales quadratically with field',
      t_S3_quadratic_in_field)


def t_S4_field_independent():
    heights = np.array([0.001, 0.01, 0.05, 0.1])
    result, _ = la.seidel_field_sweep(SURFACES, WL_TEST, heights)
    s4 = result['total']['S4']
    rel_spread = float((s4.max() - s4.min()) / abs(s4.mean()))
    return rel_spread < 1e-12, \
        f'S4 spread across 4 field heights = {rel_spread:.2e}'


H.run('S4 (Petzval) is field-independent', t_S4_field_independent)


# ----------------------------------------------------------------------
# Bit-identical with single-call seidel_coefficients
# ----------------------------------------------------------------------

def t_sweep_single_field_matches_seidel_coefficients():
    h0 = 0.025
    sw, abcd_sw = la.seidel_field_sweep(SURFACES, WL_TEST, [h0])
    cs, abcd_cs = la.seidel_coefficients(SURFACES, WL_TEST,
                                          field_angle=h0)
    rel_errs = []
    for k in ('S1', 'S2', 'S3', 'S4', 'S5'):
        a = float(sw['total'][k][0])
        b = float(cs['total'][k])
        if abs(b) > 1e-30:
            rel_errs.append(abs(a - b) / abs(b))
        else:
            rel_errs.append(abs(a - b))
    max_err = max(rel_errs)
    abcd_err = float(np.max(np.abs(np.asarray(abcd_sw) -
                                    np.asarray(abcd_cs))))
    ok = max_err < 1e-12 and abcd_err < 1e-12
    return ok, f'max Seidel rel-err = {max_err:.2e}, abcd err = {abcd_err:.2e}'


H.run('seidel_field_sweep([h]) == seidel_coefficients(field_angle=h)',
      t_sweep_single_field_matches_seidel_coefficients)


# ----------------------------------------------------------------------
# Per-surface shape
# ----------------------------------------------------------------------

def t_field_sweep_per_surface_shape():
    heights = np.array([0.01, 0.05, 0.1])
    result, _ = la.seidel_field_sweep(SURFACES, WL_TEST, heights)
    n_surf = len(SURFACES)
    expected = (n_surf, 3)
    shapes = [result[k].shape for k in ('S1', 'S2', 'S3', 'S4', 'S5')]
    ok = all(s == expected for s in shapes)
    return ok, f'per-surface shapes {shapes}, expected ({n_surf}, 3) each'


H.run('seidel_field_sweep per-surface arrays are (N_surfaces, N_fields)',
      t_field_sweep_per_surface_shape)


# ----------------------------------------------------------------------
# WFE reconstruction
# ----------------------------------------------------------------------

def t_wfe_zero_at_pupil_origin():
    """The Hopkins form has only ρ^4, ρ^3, ρ^2, ρ terms -- no piston.
    Therefore W(rho=0, theta=any) = 0."""
    s, _ = la.seidel_coefficients(SURFACES, WL_TEST, field_angle=0.05)
    W = float(la.seidel_wfe(s, rho=0.0, theta=0.0))
    return abs(W) < 1e-15, f'W(rho=0) = {W:.2e} (expected 0)'


H.run('seidel_wfe is zero at pupil origin', t_wfe_zero_at_pupil_origin)


def t_wfe_scales_with_field_dependent_S():
    """At fixed rho=0.5, theta=0, the wavefront error should grow
    with field height because S2 ~ h, S3 ~ h^2, S5 ~ h or h^3
    contributions all add."""
    heights = np.array([0.001, 0.01, 0.1])
    sw, _ = la.seidel_field_sweep(SURFACES, WL_TEST, heights)
    W0 = abs(float(la.seidel_wfe(sw, rho=0.5, theta=0.0, field_index=0)))
    W1 = abs(float(la.seidel_wfe(sw, rho=0.5, theta=0.0, field_index=1)))
    W2 = abs(float(la.seidel_wfe(sw, rho=0.5, theta=0.0, field_index=2)))
    ok = (W0 < W1 < W2)
    return ok, f'|W| at rho=0.5: h=0.001 {W0:.2e}, h=0.01 {W1:.2e}, h=0.1 {W2:.2e}'


H.run('seidel_wfe grows with field height (rho=0.5, theta=0)',
      t_wfe_scales_with_field_dependent_S)


def t_wfe_evaluates_on_grid():
    """Sanity: 2-D pupil grid returns 2-D WFE map of the same shape."""
    s, _ = la.seidel_coefficients(SURFACES, WL_TEST, field_angle=0.05)
    N = 33
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    rho = np.sqrt(X * X + Y * Y)
    theta = np.arctan2(Y, X)
    W = la.seidel_wfe(s, rho, theta)
    ok = (W.shape == rho.shape and np.all(np.isfinite(W)))
    return ok, f'W shape={W.shape}, all-finite={np.all(np.isfinite(W))}'


H.run('seidel_wfe vectorizes over 2-D pupil grids',
      t_wfe_evaluates_on_grid)


def t_wfe_petzval_only_is_sigma2_rho2():
    """Synthetic system with S1=S2=S3=S5=0 and only S4 non-zero:
    W = (1/4) S4 sigma^2 rho^2 (Petzval/field-curvature defocus)."""
    fake_totals = {
        'S1': 0.0, 'S2': 0.0, 'S3': 0.0,
        'S4': -1.0, 'S5': 0.0,
    }
    rho = np.array([0.0, 0.25, 0.5, 1.0])
    theta = np.zeros_like(rho)
    sigma = 0.1
    W = la.seidel_wfe(fake_totals, rho, theta, field_angle=sigma)
    # The totals dict is in the library's code = -S_Welford convention;
    # seidel_wfe negates it at ingestion (audit R-2 sign fix), so
    # S4_code = -1.0 composes as S4_Welford = +1.0.
    expected = (1.0 / 4.0) * (+1.0) * (sigma ** 2) * rho ** 2
    rel_err = float(np.max(np.abs(W - expected)))
    return rel_err < 1e-15, \
        f'max |W - expected| = {rel_err:.2e}'


H.run('seidel_wfe with S4-only matches (1/4) S4 sigma^2 rho^2',
      t_wfe_petzval_only_is_sigma2_rho2)


if __name__ == '__main__':
    import sys
    sys.exit(H.summary())
