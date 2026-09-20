"""VERIFY-A8 -- independent re-verification pins for WP-A8 (audit
``AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`` section 5, rows E1-E3, E5-E7).

Written by the adversarial verifier, not by the WP.  Every oracle here is
built in this file from a published formula or from first principles, on
fixtures the WP's own tests do NOT use, so that a fix that merely agrees
with itself cannot pass:

* **E1** -- the SCHOTT dispersion formula evaluated here from the
  coefficients as published in the Zemax catalogue (``schott_2017-01-20b``,
  reproduced verbatim below), plus the data-sheet ``n_d`` / ``V_d`` and the
  six-digit glass code, which are independent of the coefficients.
* **E1 gate** -- a deliberate perturbation of a glass WP-A8 never touched,
  swept across the bar so the gate is shown to be two-sided.
* **E3** -- the exact discrete structure function of the code's own lattice
  at grids and parities the WP's tests do not use.
* **E5** -- the rod's paraxial ABCD obtained by RK4 integration of
  ``y'' = -g**2 y``, i.e. WITHOUT the ``sin(g d)`` closed form the library
  now hard-codes; the integrator's own error floor is measured by
  Richardson.
* **E6** -- the hemisphere integral on a composite Gauss-Legendre rule
  built here, and the inverse-transform property of the sampler.
* **E7** -- the analytic disc area at anamorphic ``dy != dx`` and sub-pixel
  centre offsets.

No ``pytest.skip`` anywhere: where an optional package is required the
absence is asserted as a fact instead (TESTING_STANDARDS rule 4).
"""
from __future__ import annotations

import importlib.util
import inspect
import warnings

import numpy as np
import pytest

from lumenairy.elements import bsdf as bsdf_mod
from lumenairy.elements import doe as doe_mod  # noqa: F401  (import health)
from lumenairy.elements import elements as elem_mod
from lumenairy.elements._lens_thin import apply_grin_lens
from lumenairy.glass import get_glass_index
import lumenairy.glass as glass_mod


# ===========================================================================
# E1 -- the three repaired rows against the published SCHOTT catalogue
# ===========================================================================

# SCHOTT Zemax catalog 2017-01-20b, "formula 2" (Sellmeier 1) coefficients
# exactly as published, transcribed from the catalogue entry for each glass:
#   n^2 - 1 = sum_i B_i lam^2 / (lam^2 - C_i),  lam in um, C_i in um^2.
# The ``nd`` / ``Vd`` / ``glass_code`` values are the catalogue's own
# PROPERTIES block -- published numbers that do not come from the
# coefficients, which is what makes them a second, independent oracle.
_SCHOTT_PUBLISHED = {
    # glass:        (B1, B2, B3), (C1, C2, C3),                nd,       Vd
    'N-BAF52': ((1.43903433, 0.0967046052, 1.09875818),
                (0.00907800128, 0.050821208, 105.691856), 1.60863, 46.60),
    'N-LAK33A': ((1.44116999, 0.571749501, 1.16605226),
                 (0.00680933877, 0.0222291824, 80.9379555), 1.75393, 52.27),
    'N-LAK33B': ((1.42288601, 0.593661336, 1.1613526),
                 (0.00670283452, 0.021941621, 80.7407701), 1.75500, 52.30),
    # the two the audit demanded stay exact
    'N-BK7': ((1.03961212, 0.231792344, 1.01046945),
              (0.00600069867, 0.0200179144, 103.560653), 1.51680, 64.17),
    'N-SF11': ((1.73759695, 0.313747346, 1.89878101),
               (0.013188707, 0.0623068142, 155.23629), 1.78472, 25.68),
}

# The coefficient rows as they stood BEFORE the fix (commit 658e6142), so the
# fail-before arm does not depend on a stashed working tree.
_PRE_FIX_ROWS = {
    'N-BAF52': ((1.43903433, 0.179827671, 1.13174268),
                (9.07800726e-3, 4.39222348e-2, 1.06317650e2)),
    'N-LAK33A': ((1.44116999, 0.571749501, 1.16605226),
                 (6.80933877e-3, 2.22291824e-2, 1.07097324e2)),
    'N-LAK33B': ((1.42288601, 0.593661336, 1.16135260),
                 (6.70283452e-3, 2.19416210e-2, 1.01736644e2)),
}

_LINE_D, _LINE_F, _LINE_C = 587.5618e-9, 486.1327e-9, 656.2725e-9


def _sellmeier_here(coeffs, wavelength_m):
    """The SCHOTT dispersion formula, evaluated in this file."""
    B, C = coeffs
    lam_sq = (float(wavelength_m) * 1e6) ** 2
    n_sq = 1.0
    for b, c in zip(B, C):
        n_sq += b * lam_sq / (lam_sq - c)
    return float(np.sqrt(n_sq))


def _nd_vd_here(coeffs):
    nd = _sellmeier_here(coeffs, _LINE_D)
    nF = _sellmeier_here(coeffs, _LINE_F)
    nC = _sellmeier_here(coeffs, _LINE_C)
    return nd, (nd - 1.0) / (nF - nC)


@pytest.mark.parametrize('name', sorted(_SCHOTT_PUBLISHED))
def test_verify_a8_e1_bundled_row_is_the_published_schott_row(name):
    """``get_glass_index`` must reproduce the published coefficients
    evaluated here, at nine wavelengths spanning the catalogue range.

    Bar 1e-12 relative: the two evaluations differ only by float
    reassociation of the same three-term sum, measured max |delta| = 2.2e-16
    over the 5 x 9 grid (2026-09-12), so the bar is ~4 decades above that
    floor and ~10 decades below the 2.85e-2 (N-BAF52) / 3.5e-4 (N-LAK33A)
    the pre-fix rows were off by.
    """
    B, C, _nd, _vd = _SCHOTT_PUBLISHED[name]
    for wl in (0.4e-6, 0.4861327e-6, 0.5875618e-6, 0.6562725e-6, 0.85e-6,
               1.064e-6, 1.31e-6, 1.55e-6, 2.0e-6):
        want = _sellmeier_here((B, C), wl)
        got = float(get_glass_index(name, wl))
        assert abs(got - want) <= 1e-12 * want, (name, wl, got, want)


@pytest.mark.parametrize('name', sorted(_SCHOTT_PUBLISHED))
def test_verify_a8_e1_row_reproduces_the_data_sheet_nd_and_vd(name):
    """Second oracle: the catalogue's published ``nd`` / ``Vd``, which are
    measured quantities, not outputs of the dispersion fit.

    Bars: |d n_d| < 1e-5 and |d V_d| < 0.05.  The data sheet quotes n_d to
    5 decimals and V_d to 2, so half an ULP of the quote is 5e-6 / 0.005;
    measured residuals (2026-09-12) are 1.01e-6 / 0.0026 (N-BAF52),
    8.7e-8 / 0.0008 (N-LAK33A), 1.6e-7 / 0.0001 (N-LAK33B), 3.5e-8 / 0.0027
    (N-BK7), 5.8e-8 / 0.00004 (N-SF11) -- so the bars sit ~1 decade above
    the quote floor and 3 (n_d) / 2 (V_d) decades below the smallest defect
    they must catch (N-LAK33A pre-fix: 3.5e-4 and 0.76).
    """
    B, C, nd_sheet, vd_sheet = _SCHOTT_PUBLISHED[name]
    nd = float(get_glass_index(name, _LINE_D))
    nF = float(get_glass_index(name, _LINE_F))
    nC = float(get_glass_index(name, _LINE_C))
    vd = (nd - 1.0) / (nF - nC)
    assert abs(nd - nd_sheet) < 1e-5, (name, nd, nd_sheet)
    assert abs(vd - vd_sheet) < 0.05, (name, vd, vd_sheet)


@pytest.mark.parametrize('name', sorted(_PRE_FIX_ROWS))
def test_verify_a8_e1_pre_fix_rows_would_fail_both_oracles(name):
    """Fail-before, oracle-side: the rows as they stood at 658e6142 miss the
    data sheet by 2.85e-2 / 3.5e-4 / 2.9e-4 in n_d and by 4.13 / 0.76 / 0.64
    in V_d -- i.e. by 3 to 4 decades more than the bars above allow, so the
    two tests above are genuinely discriminating and not merely loose.
    """
    nd_pre, vd_pre = _nd_vd_here(_PRE_FIX_ROWS[name])
    _B, _C, nd_sheet, vd_sheet = _SCHOTT_PUBLISHED[name]
    assert abs(nd_pre - nd_sheet) > 1e-4, (name, nd_pre, nd_sheet)
    assert abs(vd_pre - vd_sheet) > 0.5, (name, vd_pre, vd_sheet)
    # ... and the shipped row is NOT the pre-fix row.
    assert abs(float(get_glass_index(name, _LINE_D)) - nd_pre) > 1e-4


def test_verify_a8_e1_value_gate_is_two_sided_on_a_glass_the_wp_never_touched():
    """The E1 cross-check must fire on ANY drifted row, not just the three
    that were repaired -- and must stay silent just below its bar.

    Perturbing ``N-SK16``'s B1 (a row WP-A8 did not touch, and one whose own
    residual against the catalogue is 0.0) by successive decades moves n_d
    by 4.23e-7 / 4.23e-6 / 4.23e-5 / 4.23e-4.  With the shipped bar
    ``_BUNDLED_VALUE_TOL_ND = 5e-5`` the gate must be silent on the first
    three and fire on the last; measured exactly that, 2026-09-12.  The same
    ladder is run on a formula-3 POLYNOMIAL row, the table the partition
    report called its biggest untested gap.
    """
    if importlib.util.find_spec('refractiveindex') is None:
        # No skip: assert the fact so the gate can never silently drop out.
        assert glass_mod._cross_check_bundled_values()[0] == 0
        return

    def _sweep(table, name, perturb):
        orig = table[name]
        seen = []
        try:
            for scale in (1e-6, 1e-5, 1e-4, 1e-3):
                table[name] = perturb(orig, scale)
                _n, problems = glass_mod._cross_check_bundled_values()
                seen.append(any(p.startswith(name + ':') for p in problems))
        finally:
            table[name] = orig
        return seen

    def _bump_sellmeier(orig, scale):
        B, C = orig
        return ((B[0] * (1.0 + scale), B[1], B[2]), C)

    def _bump_poly(orig, scale):
        new = list(orig)
        new[0] = new[0] + scale
        return tuple(new)

    seen = _sweep(glass_mod.SELLMEIER_COEFFICIENTS, 'N-SK16', _bump_sellmeier)
    assert seen == [False, False, False, True], seen

    poly_name = sorted(glass_mod.POLYNOMIAL_COEFFICIENTS)[0]
    seen = _sweep(glass_mod.POLYNOMIAL_COEFFICIENTS, poly_name, _bump_poly)
    assert seen == [False, False, False, True], (poly_name, seen)

    # and the clean table passes, with every row actually resolved
    n_checked, problems = glass_mod._cross_check_bundled_values()
    assert problems == []
    assert n_checked == (len(glass_mod.SELLMEIER_COEFFICIENTS)
                         + len(glass_mod.POLYNOMIAL_COEFFICIENTS)), (
        f"the gate resolved only {n_checked} rows; a row it cannot resolve "
        f"is a row it does not gate")


def test_verify_a8_e1_catalogue_guards_are_narrow_and_still_reject_a_k_only_page():
    """The two catalogue guards must name their exceptions, and the one case
    that cannot be named must be tested for rather than caught.

    ``refractiveindex`` raises a BARE ``Exception`` from
    ``get_refractive_index`` for a page carrying only tabulated k, so a
    narrow except clause cannot cover it -- ``_catalogue_index_fn_from_entry``
    checks the page's index model instead.  ``main/BaF2/Bosomworth-5K`` and
    ``-200K`` are such pages (verified 2026-09-12: ``_n_func is None``, forced
    evaluation raises ``Exception: No refractive index specified...``), while
    ``-80K`` and ``-300K`` carry n and must still resolve.  Structural, not
    numeric: no bar applies.
    """
    assert 'except Exception' not in inspect.getsource(
        glass_mod._catalogue_index_fn_from_entry)
    assert 'except Exception' not in inspect.getsource(
        glass_mod._cross_check_bundled_values)
    if importlib.util.find_spec('refractiveindex') is None:
        assert glass_mod._catalogue_index_fn_from_entry(
            ('specs', 'SCHOTT-optical', 'N-BK7')) is None
        return
    tup = glass_mod._catalogue_lookup_exceptions()
    assert KeyError in tup and ValueError in tup
    assert any(c.__name__ == 'NoExtinctionCoefficient' for c in tup), (
        [c.__name__ for c in tup])
    assert Exception not in tup, 'the tuple must not smuggle in bare Exception'
    for page in ('Bosomworth-5K', 'Bosomworth-200K'):       # k only
        assert glass_mod._catalogue_index_fn_from_entry(
            ('main', 'BaF2', page)) is None, page
    for page in ('Bosomworth-80K', 'Bosomworth-300K'):      # carry n
        assert glass_mod._catalogue_index_fn_from_entry(
            ('main', 'BaF2', page)) is not None, page
    # an absent page resolves to None rather than propagating
    assert glass_mod._catalogue_index_fn_from_entry(
        ('specs', 'SCHOTT-optical', 'NO-SUCH-GLASS')) is None


# ===========================================================================
# E2 (second arm) -- a catalogue page that does not span the request
# ===========================================================================

def test_verify_a8_e2_out_of_range_catalogue_lookup_refuses_instead_of_nan():
    """``SILICON`` is registered ``('main', 'Si', 'Li-293K')``, whose data
    starts at 1.2 um.  Below that the package's interpolator returns NaN
    rather than raising, so ``get_glass_index`` handed back ``nan`` and
    ``get_glass_index_complex`` ``nan + 0j`` -- with only a validity WARNING,
    and one multiply later the NaN is across the field.  Measured before the
    fix: `nan` at 0.35 / 0.4 / 0.5876 / 0.633 / 1.064 um, 5 of the 8
    wavelengths swept.  It is now a named ``ValueError`` carrying the page's
    range, per CONVENTIONS Section 2.

    Two-sided by construction: the same glass INSIDE the range must still
    return its ordinary value (3.5003 at 1.31 um, 3.4757 at 1.55 um,
    3.4150 at 10 um), and the other catalogue-dispatched glasses must be
    untouched.  Exact-value assertions where the quantity is a lookup; no
    bar is meaningful.

    Without the optional package this guard is unreachable by construction --
    WP-A8's changelog: "Only the live-catalogue (tuple-registered) path can
    produce it; the bundled Sellmeier and polynomial evaluators are closed
    forms with their own guards and are untouched" -- and ``SILICON`` is the
    one tuple entry with no bundled row, so it takes ``get_glass_index``'s
    documented ``ImportError`` first.  Per this module's own rule (no
    ``pytest.skip``; TESTING_STANDARDS rule 4) the absence is asserted as a
    fact instead: the refusal is the actionable ImportError, the guard is
    still wired into the tuple path, and the other catalogue names still
    resolve from their bundled rows.
    """
    if importlib.util.find_spec('refractiveindex') is None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for wl in (0.633e-6, 1.064e-6, 20e-6):
                with pytest.raises(
                        ImportError,
                        match=r"requires the 'refractiveindex' package"):
                    get_glass_index('SILICON', wl)
                with pytest.raises(ImportError, match=r"'SILICON'"):
                    glass_mod.get_glass_index_complex('SILICON', wl)
            # the guard itself is still wired into the tuple path, so this
            # arm is not zero-coverage of the fix under test
            assert '_require_finite_catalogue_index(' in inspect.getsource(
                glass_mod.get_glass_index)
            # every other catalogue-dispatched glass resolves from its
            # bundled row and is unaffected
            for name, wl in (('N-BK7', 587.5618e-9), ('CaF2', 1.064e-6),
                             ('MgF2', 1.55e-6), ('SiO2', 633e-9)):
                assert np.isfinite(float(get_glass_index(name, wl))), name
        return
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for wl in (0.633e-6, 1.064e-6, 20e-6):
            with pytest.raises(ValueError, match=r'get_glass_index: '):
                get_glass_index('SILICON', wl)
            with pytest.raises(ValueError, match=r"'SILICON'"):
                glass_mod.get_glass_index_complex('SILICON', wl)
        # the message must name the range that would work
        try:
            get_glass_index('SILICON', 0.633e-6)
        except ValueError as exc:
            assert '1.200e-06' in str(exc) and '1.400e-05' in str(exc), str(exc)
        # in range: unchanged, and finite
        for wl, want in ((1.31e-6, 3.5003), (1.55e-6, 3.4757), (10e-6, 3.4150)):
            got = float(get_glass_index('SILICON', wl))
            assert abs(got - want) < 5e-4, (wl, got, want)
        assert np.isfinite(
            glass_mod.get_glass_index_complex('SILICON', 1.55e-6).real)
        # an array request that straddles the edge is refused as a whole
        with pytest.raises(ValueError, match=r'get_glass_index: '):
            get_glass_index('SILICON', np.array([1.3e-6, 0.633e-6]))
        # every other catalogue-dispatched glass is unaffected
        for name, wl in (('N-BK7', 587.5618e-9), ('CaF2', 1.064e-6),
                         ('MgF2', 1.55e-6), ('SiO2', 633e-9)):
            assert np.isfinite(float(get_glass_index(name, wl)))


# ===========================================================================
# E3 -- turbulence variance at grids and parities the WP's tests do not use
# ===========================================================================

def _lattice_psd(N, dx, r0, L0=np.inf, l0=0.0):
    df = 1.0 / (N * dx)
    fx = (np.arange(N) - N // 2) * df
    FX, FY = np.meshgrid(fx, fx)
    f_sq = FX ** 2 + FY ** 2
    psd = 0.023 * r0 ** (-5.0 / 3.0) * (
        np.where(f_sq > 0, f_sq, 1.0) + 1.0 / L0 ** 2) ** (-11.0 / 6.0)
    if l0 > 0:
        psd = psd * np.exp(-(np.sqrt(f_sq) * l0 * 2 * np.pi / 5.92) ** 2)
    psd[N // 2, N // 2] = 0.0
    return FX, psd, df


@pytest.mark.parametrize('N,dx,r0,sep', [(127, 2.0e-3, 0.05, 1),
                                         (200, 1.0e-2, 0.30, 2),
                                         (65, 5.0e-3, 0.10, 1)])
def test_verify_a8_e3_structure_function_at_other_grids_and_parities(
        N, dx, r0, sep):
    """``E[(phi(r+d)-phi(r))**2] = sum_k PSD_k df^2 * 2(1-cos(2 pi f_k.d))``
    exactly for this construction; the ratio must be 1, not 2.

    Measured 2026-09-12 over 60 seeds: 0.991 +- 0.012 (N = 127, odd),
    1.011 +- 0.014 (N = 200), 1.029 +- 0.013 (N = 65, odd).  Bar
    [0.80, 1.20] is >= 12 standard errors from the measurements on both
    sides at the 48 seeds used here, and excludes the pre-fix 2.0 by four
    times the bar's own half-width.
    """
    n_seeds = 48
    FX, psd, df = _lattice_psd(N, dx, r0)
    d_exact = float(np.sum(psd * df ** 2 * 2
                           * (1 - np.cos(2 * np.pi * FX * sep * dx))))
    acc = 0.0
    for s in range(n_seeds):
        ph = elem_mod.generate_turbulence_screen(N, dx, r0, seed=51000 + s)
        acc += float(np.mean((ph[:, sep:] - ph[:, :-sep]) ** 2))
    ratio = (acc / n_seeds) / d_exact
    assert 0.80 < ratio < 1.20, (
        f"N={N} sep={sep}: D_meas/D_lattice = {ratio:.4f} "
        f"(1.0 expected, 2.0 = the pre-fix sqrt(2))")


def test_verify_a8_e3_subharmonics_keep_the_correction_zero_mean():
    """A screen's piston is unobservable, so the Lane correction must not
    add one; and ``subharmonics=0`` must stay bit-identical to the default.

    Bar 1e-12 on the mean of the correction relative to its own rms:
    measured |mean| <= 1.3e-14 against an rms of 1.9-6.8 rad at p = 1..5
    (2026-09-12), i.e. the float64 cancellation floor of an N**2 sum.
    """
    a = elem_mod.generate_turbulence_screen(128, 5e-3, 0.1, seed=11)
    assert np.array_equal(
        a, elem_mod.generate_turbulence_screen(128, 5e-3, 0.1, seed=11,
                                               subharmonics=0))
    for p in (1, 3, 5):
        b = elem_mod.generate_turbulence_screen(128, 5e-3, 0.1, seed=11,
                                                subharmonics=p)
        corr = b - a
        assert abs(float(corr.mean())) < 1e-12 * float(corr.std()), p
        assert float(corr.std()) > 0.5, p       # the level actually acts


# ===========================================================================
# E5 -- the rod ABCD by RK4, not by the sin(g d) closed form
# ===========================================================================

def _rod_C_rk4(g, d, n0, steps):
    """Integrate ``dy/dz = u/n0``, ``du/dz = -n0 g^2 y`` (paraxial rays in a
    parabolic-index rod) and return the ABCD element ``C``, whose negative
    is the rod's paraxial power."""
    h = d / steps

    def deriv(y, u):
        return u / n0, -n0 * g * g * y

    y, u = 1.0, 0.0
    for _ in range(steps):
        k1 = deriv(y, u)
        k2 = deriv(y + 0.5 * h * k1[0], u + 0.5 * h * k1[1])
        k3 = deriv(y + 0.5 * h * k2[0], u + 0.5 * h * k2[1])
        k4 = deriv(y + h * k3[0], u + h * k3[1])
        y, u = (y + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                u + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))
    return u


@pytest.mark.parametrize('n0,g,gd', [(1.52, 180.0, 0.30),
                                     (1.75, 90.0, 1.20),
                                     (2.00, 1200.0, np.pi / 2)])
def test_verify_a8_e5_screen_power_equals_the_integrated_rod_abcd(n0, g, gd):
    """The screen's quadratic coefficient must equal ``-C`` of the rod's
    ray-transfer matrix, obtained here by RK4 rather than from ``sin(g d)``.

    The oracle's own floor is measured in-test by Richardson (steps vs
    2 x steps); measured |C(2h) - C(h)| / |C| <= 4e-13 at steps = 1500, and
    the screen agrees with C(h) to <= 4e-14 relative (2026-09-12).  Bar 1e-9
    relative: ~4 decades above the oracle floor and 7 decades below the
    smallest deviation it must catch -- the short-rod form is low by
    1 - sin(gd)/(gd) = 1.5 % at g*d = 0.30, 22 % at 1.20 and 36 % at the
    quarter pitch.
    """
    d = gd / g
    C1 = _rod_C_rk4(g, d, n0, 1500)
    C2 = _rod_C_rk4(g, d, n0, 3000)
    assert abs(C2 - C1) < 1e-9 * abs(C1), (C1, C2)   # oracle floor
    power_exact = -C2

    N, dx, wl = 48, 2e-6, 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = apply_grin_lens(np.ones((N, N), dtype=np.complex128), n0=n0,
                              g=g, d=d, wavelength=wl, dx=dx)
    x1 = dx                      # one pixel off centre: |phase| << pi there
    phase = float(np.angle(out[N // 2, N // 2 + 1]))
    power_screen = -2.0 * phase / ((2 * np.pi / wl) * x1 ** 2)
    assert abs(power_screen / power_exact - 1.0) < 1e-9, (
        power_screen, power_exact,
        f"thin form would give {n0 * g * g * d:.6f}")


# ===========================================================================
# E6 -- TIS against a quadrature built here; the sampler's inverse-transform
# ===========================================================================

def _tis_gauss_legendre(bfun):
    """``2 pi int_0^1 B(u) u du`` on a 400-cell geometric composite
    24-point Gauss-Legendre rule -- 9600 nodes, unrelated to the library's
    own 256-node grid."""
    edges = np.concatenate([[0.0], np.geomspace(1e-9, 1.0, 400)])
    xg, wg = np.polynomial.legendre.leggauss(24)
    tot = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        u = 0.5 * (b - a) * xg + 0.5 * (a + b)
        tot += float(np.sum(0.5 * (b - a) * wg * bfun(u) * u))
    return 2 * np.pi * tot


@pytest.mark.parametrize('b0,l,s', [(0.025, 3e-1, 1.2), (7.3, 5e-2, 3.1),
                                    (1.0, 2e-3, 2.3), (0.5, 4e-4, 1.7)])
def test_verify_a8_e6_harvey_shack_tis_at_parameters_the_wp_did_not_use(
        b0, l, s):
    """The closed form vs an independent composite Gauss-Legendre integral,
    at ``b0 != 1`` and ``s`` off the 1.5 / 2 / 2.5 grid the WP tested.

    Bar 1e-9 relative: the 9600-node rule's own residual on these smooth
    integrands is below 1e-13 (it reproduces the s = 2 analytic form to
    3e-15), and the measured agreement over a 75-triple sweep
    (b0 in {1, 0.025, 7.3} x l in {3e-1..1e-5} x s in {1.2..3.1}) was
    worst 1.8e-14 (2026-09-12).  The pre-fix inherited quadrature read
    -18 % at l = 1e-3 and -32 % at l = 1e-4, so the bar has ~5 decades of
    gap above and 7 below.
    """
    model = bsdf_mod.HarveyShackBSDF(b0=b0, l=l, s=s)
    ref = _tis_gauss_legendre(
        lambda u: b0 / (1 + (u / l) ** 2) ** (s / 2))
    assert abs(model.total_integrated_scatter() / ref - 1.0) < 1e-9


def test_verify_a8_e6_sampler_is_an_exact_inverse_transform():
    """A sampler that is EXACTLY ``F^-1(xi)`` maps the same uniform stream
    onto every parameter set, so ``F(u_i) = xi_i`` and the KS distance
    against ``F`` is identical for all parameters -- a property no rejection
    sampler (whose draw length depends on acceptance) can have.

    Measured 2026-09-12 at n = 2e4 with one shared seed: KS = 0.0075243 for
    all five (l, s) in {(1e-1,2), (1e-2,1.5), (1e-3,2.5), (5e-4,2), (2e-2,3)},
    identical to 16 significant figures.  Bars: the spread across parameter
    sets < 1e-12 (float64 round-trip of F o F^-1), and the common value
    below the 99.9 % two-sided critical value 1.949/sqrt(n) = 0.0138 so the
    arm also fails if the transform is exact but of the WRONG law.
    """
    n = 20_000
    cases = [(1e-1, 2.0), (1e-2, 1.5), (1e-3, 2.5), (5e-4, 2.0), (2e-2, 3.0)]
    ks_values = []
    for l, s in cases:
        model = bsdf_mod.HarveyShackBSDF(b0=1.0, l=l, s=s)
        d = model.sample(np.array([0.0, 0.0, -1.0]), n,
                         rng=np.random.default_rng(4242))
        u = np.sort(np.hypot(d[:, 0], d[:, 1]))
        t = 1.0 + (u / l) ** 2
        big_t = 1.0 + 1.0 / l ** 2
        if abs(s - 2.0) < 1e-12:
            cdf = np.log(t) / np.log(big_t)
        else:
            p = 1.0 - s / 2.0
            cdf = (t ** p - 1.0) / (big_t ** p - 1.0)
        emp = np.arange(1, n + 1) / n
        ks_values.append(float(np.max(np.abs(emp - cdf))))
    spread = max(ks_values) - min(ks_values)
    assert spread < 1e-12, (cases, ks_values)
    assert max(ks_values) < 1.949 / np.sqrt(n), ks_values


def test_verify_a8_sample_scatter_rays_still_serves_a_sample_only_subclass():
    """``BSDFModel`` is a public export and the README documents it as the
    subclassing entry point; its only abstract draw method is ``sample``.

    The vectorised ``sample_scatter_rays`` reaches for the new
    ``_sample_local`` hook, which such a subclass does not have -- as first
    written it raised ``NotImplementedError`` where the pre-vectorisation
    code worked.  VERIFY-A8 added the per-ray compatibility path; this pins
    it, and pins that the shipped models still take the batched path.
    """
    import lumenairy.raytrace as rt

    class _SampleOnly(bsdf_mod.BSDFModel):
        def evaluate(self, incident_dir, scattered_dir):
            sd = np.asarray(scattered_dir, dtype=float)
            return np.where(sd[..., 2] > 0, 1.0 / np.pi, 0.0)

        def sample(self, incident_dir, n_samples, rng=None):
            rng = np.random.default_rng() if rng is None else rng
            v = rng.standard_normal((n_samples, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            v[v[:, 2] < 0] *= -1
            return v

    n = 64
    rr = np.random.default_rng(2)
    L = rr.normal(0, 0.05, n)
    M = rr.normal(0, 0.05, n)
    Nc = -np.sqrt(1 - L ** 2 - M ** 2)
    bundle = rt.RayBundle(x=np.zeros(n), y=np.zeros(n), z=np.zeros(n),
                          L=L, M=M, N=Nc, wavelength=1e-6,
                          alive=np.ones(n, bool), opd=np.zeros(n))

    class _Surface:
        bsdf = _SampleOnly()

    out = bsdf_mod.sample_scatter_rays(_Surface(), bundle, n_per_ray=3, rng=9)
    dirs = np.stack([out.L, out.M, out.N], axis=1)
    assert dirs.shape == (3 * n, 3)
    assert np.max(np.abs(np.linalg.norm(dirs, axis=1) - 1.0)) < 1e-12
    assert np.all(out.N > 0)
    # the three shipped models must NOT fall back
    for cls in (bsdf_mod.LambertianBSDF, bsdf_mod.GaussianBSDF,
                bsdf_mod.HarveyShackBSDF):
        assert cls._sample_local is not bsdf_mod.BSDFModel._sample_local, cls


# ===========================================================================
# E7 -- grey aperture edge at anamorphic and offset fixtures
# ===========================================================================

@pytest.mark.parametrize('d_px,dy_ratio,offset', [(37, 1.0, 0.37),
                                                  (63, 2.5, 0.13),
                                                  (145, 0.4, 0.29)])
def test_verify_a8_e7_gray_edge_beats_hard_at_anamorphic_and_offset_rims(
        d_px, dy_ratio, offset):
    """Transmitted area against the analytic disc area ``pi (D/2)**2``, on
    rectangular pixels (``dy != dx``) and with the centre off the pixel
    lattice -- fixtures the WP's own aperture test does not use.

    Measured 2026-09-12 (|relative area error|):

        D/dx = 37,  dy/dx = 1.0, offset 0.37 px : 4.85e-3 hard -> 8.35e-4 gray(4) -> 8.1e-5 gray(16)
        D/dx = 63,  dy/dx = 2.5, offset 0.13 px : 7.20e-4 hard -> 8.17e-5 gray(4) -> 1.5e-5 gray(16)
        D/dx = 145, dy/dx = 0.4, offset 0.29 px : 6.08e-5 hard -> 3.05e-5 gray(4) -> 1.0e-6 gray(16)

    Bars: gray(16) < 3e-4 absolute relative error (the worst measurement is
    8.1e-5, so 3.7x of headroom, and the coarsest hard reading it must beat
    is 4.85e-3); ``e_hard / e_g4 >= 1.5``, derived below; and gray(16)
    better than gray(4), a strict inequality that needs no bar at all.

    The ratio bar, and why it is not ``e_g4 <= e_hard`` (VERIFY-C1 defect
    D2).  ``<=`` is satisfied by EQUALITY, so the id passed when the two arms
    were the SAME array: the verification mutated ``edge='hard'`` to return
    the grey mask, watched fifteen other ids go red, and found this one still
    green on both builds.  ``1.0`` is exactly where "grey IS hard" lives, so
    the bar has to sit strictly above it.  Re-measured 2026-09-20 on this
    id's own three fixtures, Windows py3.14 / numpy 2.4.4 and WSL py3.12 /
    numpy 2.4.6, IDENTICAL to sixteen significant figures on both (there is
    no BLAS in any of it -- a mask sum is an integer count over
    ``n_sub**2``), raw JSON in ``validation/probe_verify_c1/d2_ratio_*.json``:

        D/dx = 37,  dy/dx = 1.0, offset 0.37 px : 5.804555
        D/dx = 63,  dy/dx = 2.5, offset 0.13 px : 8.815875
        D/dx = 145, dy/dx = 0.4, offset 0.29 px : 1.992823   <- the worst

    so 1.5 sits 1.33x below the smallest real reading and 1.5x above the
    degenerate 1.0, which is a gap on both sides.

    The 145-px fixture is the binding one, but NOT because a bigger rim is a
    milder staircase (VERIFY-C1 ROUND2 defect R4, which refuted that reading
    of it).  The ratio is set by where each arm's SIGNED area error falls
    relative to ZERO, and at ``D/dx = 145`` the hard arm happens to be near a
    crossing.  A 30-point scan of that fixture's neighbourhood -- the same
    ``D``, ``dy/dx`` in {0.4, 0.5, 1.0} and ten offsets -- finds FOUR ratios
    below 1.0, the worst **0.003602** at ``dy/dx = 1.0, offset 0.23``, where

        e_hard = +2.188924e-07   against   e_g4 = +6.077725e-05

    i.e. the hard arm is ~278x BETTER there, purely because its staircase
    error is passing through zero; the three shipped fixtures' own signed
    readings are -4.845644e-03 / -8.348003e-04, -7.202868e-04 / +8.170338e-05
    and +6.077725e-05 / +3.049807e-05.  So the ratio is **not monotone in the
    rim width**: it inverts wherever the hard arm's signed error crosses zero,
    which on this fixture family happens inside ``D/dx = 145`` itself as the
    offset moves from 0.29 (ratio 1.9928) to 0.23 (ratio 0.0036) -- one
    neighbouring offset away.  The grey arm cannot compensate, because its own
    sub-sample quantisation residual (about 3e-05 to 8e-05 here) does not
    shrink with ``D``; see VERIFY_WP-C1.md sec. 2.5, and note that the
    second-order convergence claim is a FIELD property, not an AREA one.

    Consequence for a future re-pinner: the bar is a pin on THESE THREE
    fixtures and must be RE-MEASURED, not extrapolated, if they change.
    "Pick a bigger D, the staircase will be milder" is the opposite of what
    the data does.  Re-measured 2026-09-20, character-identical on Windows
    py3.14 / numpy 2.4.4 and WSL py3.12 / numpy 2.4.6; raw JSON in
    ``validation/probe_verify_c1_round2/d2_ratio_R2_*.json`` and
    ``validation/probe_wpc1_round3/d2_ratio_R3_*.json``.
    """
    N, dx = 512, 1e-6
    dy = dx * dy_ratio
    D = d_px * dx
    E = np.ones((N, N), dtype=np.complex128)
    analytic = np.pi * (D / 2) ** 2

    def _area(**kw):
        out = elem_mod.apply_aperture(E, dx, 'circular', {'diameter': D},
                                      xc=offset * dx, yc=-offset * dy, dy=dy,
                                      **kw)
        return float(np.sum(np.real(out))) * dx * dy

    # v5.49.0 (WP-C1): the hard arm is named explicitly -- the default moved
    # to 'gray', and an unnamed ``_area()`` here would have turned
    # ``e_g4 <= e_hard`` into a tautology instead of a comparison.  The
    # readings and the bars are unchanged.
    e_hard = abs(_area(edge='hard') / analytic - 1.0)
    e_g4 = abs(_area(edge='gray') / analytic - 1.0)
    e_g16 = abs(_area(edge='gray', edge_samples=16) / analytic - 1.0)
    assert e_g16 < 3e-4, (d_px, dy_ratio, offset, e_g16)
    # VERIFY-C1 D2: a RATIO, not ``<=``.  ``e_g4 <= e_hard`` is satisfied by
    # EQUALITY, so it passed while the two arms were the same array -- with
    # ``edge='hard'`` mutated to return the grey mask, 15 ids went red on
    # both builds and this was not one of them.  Derivation and the three
    # re-measured ratios are in the docstring's "Bars" paragraph.
    assert e_hard / e_g4 >= 1.5, (d_px, dy_ratio, offset, e_hard, e_g4,
                                  e_hard / e_g4)
    assert e_g16 < e_g4, (e_g16, e_g4)


@pytest.mark.parametrize('fill', [np.nan + 0j, np.inf + 0j, -np.inf + 0j])
def test_verify_a8_e7_gray_edge_zeroes_a_blocked_pixel_even_if_it_is_not_finite(
        fill):
    """A blocked pixel must come out exactly 0, as the hard edge gives, for
    ANY input value.

    As first written the grey path returned ``E_in * frac``; ``frac`` is
    exactly 0 outside the opening, and ``0.0 * nan`` is ``nan``, so a field
    carrying NaN or inf outside the stop came back with those values intact
    (measured: the whole blocked region read ``nan+nanj``) and the next FFT
    would smear them over the entire plane.  Non-finite values outside a
    stop are ordinary here -- ``surface_sag_general`` returns NaN outside
    the conic domain and ``apply_thin_lens(model='aplanatic')`` leaves a
    sentinel -- so VERIFY-A8 changed the grey return to select rather than
    scale outside the opening.  Exact-zero assertion; no bar is meaningful.
    """
    E = np.full((32, 32), fill, dtype=np.complex128)
    # v5.49.0 (WP-C1): both arms named, so this stays a hard-vs-gray claim
    # after the default moved to 'gray'.
    hard = elem_mod.apply_aperture(E, 1e-6, 'circular', {'diameter': 1e-5},
                                   edge='hard')
    gray = elem_mod.apply_aperture(E, 1e-6, 'circular', {'diameter': 1e-5},
                                   edge='gray')
    outside = np.zeros((32, 32), dtype=bool)
    outside[0, 0] = outside[0, -1] = outside[-1, 0] = outside[-1, -1] = True
    assert np.all(hard[outside] == 0), hard[outside]
    assert np.all(gray[outside] == 0), gray[outside]
    # ... and the open centre still carries the (non-finite) input.
    assert not np.isfinite(gray[16, 16]) or np.isfinite(fill)


def test_verify_a8_e7_gray_edge_is_not_advertised_for_axis_aligned_rims():
    """Scope note, pinned: for a RECTANGULAR aperture the rim is axis
    aligned, so box supersampling only quantises each edge to ``1/n_sub`` of
    a pixel and the area error falls like ``1/n_sub``, not like the
    ``~1/n_sub**1.5`` the docstring quotes for a circular rim.

    Measured 2026-09-12 on Wx = 40.3 px, Wy = 17.7 px: 2.29e-2 hard,
    6.39e-3 gray(4), 1.58e-3 gray(16) -- a factor 4.0 from 4 to 16
    sub-samples, i.e. exactly linear.  This is correct behaviour for the
    algorithm, not a defect; the pin exists so the docstring's circular-rim
    numbers are never read as a guarantee for rectangles.
    """
    N, dx = 512, 1e-6
    Wx, Wy = 40.3 * dx, 17.7 * dx
    E = np.ones((N, N), dtype=np.complex128)
    analytic = Wx * Wy

    def _area(**kw):
        out = elem_mod.apply_aperture(E, dx, 'rectangular',
                                      {'width_x': Wx, 'width_y': Wy}, **kw)
        return float(np.sum(np.real(out))) * dx * dx

    e_g4 = abs(_area(edge='gray') / analytic - 1.0)
    e_g16 = abs(_area(edge='gray', edge_samples=16) / analytic - 1.0)
    # v5.49.0 (WP-C1): the hard arm named explicitly (the default moved).
    assert e_g4 < abs(_area(edge='hard') / analytic - 1.0)
    # linear in 1/n_sub: the ratio is 4.0 +- 25 %, not the 8 of n**1.5
    assert 3.0 < e_g4 / e_g16 < 5.0, (e_g4, e_g16)
