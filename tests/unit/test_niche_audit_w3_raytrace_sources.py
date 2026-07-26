"""Territory-R pins for AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 (wave 3).

MEDIUM / LOW findings + dead-code verification across
``raytrace/{surface,_conic_core,trace,paraxial,differential}.py``,
``glass.py``, ``sources/core.py`` and ``optimize/{driver,merit_terms}.py``.

Every test whose name starts ``test_r`` FAILS on a pre-fix worktree of
7ea2eb9 unless its class docstring says it is a CONTRACT LOCK (R-14), a
CONVENTION LOCK (R-15) or a DEAD-CODE COUNTER-PIN (R-18).  Those three
classes pin behaviour the audit proposed changing and this wave
DECLINED, with the measurement that justifies each declination inline;
they pass pre-fix by design.

Findings pinned here
--------------------
* **R-8 / E-L7** -- odd aspheric powers are sag/normal-inconsistent, and
  differently per backend (measured 100x).  Now rejected at BOTH shared
  entry points: the ``Surface`` dataclass and ``_conic_core``'s
  sag/derivative pair (which is the JAX backend's only sag route, and the
  JAX path never builds a ``Surface``).
* **R-11** -- ``paraxial.f_number`` returned a SIGNED f/# (-9.97
  measured) where all three siblings use ``abs()``.
* **R-12** -- ``glass._sellmeier_index`` was scalar-only; an array input
  died with numpy's opaque ambiguity error while the ``_polynomial_index``
  sibling's docstring claimed the two were at parity.
* **R-13** -- the NumPy DOE kick had no zero / non-finite period guard
  (``ZeroDivisionError``; ``nan`` silently NaN-poisoned L/M) where the
  JAX twin returns a zero kick.
* **R-14** -- aperture-clip order differs NumPy-vs-JAX.  Pinned as
  ALIVE-row exactness (the observable) + the alive-mask parity.
* **R-15** -- ``make_fan(axis='x')`` tilts in ``L`` where
  ``make_ring`` / ``make_grid`` tilt in ``M``.  CONVENTION LOCK: the
  per-axis tilt is load-bearing (see the test docstring's measurement).
* **R-16** -- the numba ``_dsqrtq`` clamped a NaN radicand to 0.0 where
  the ``_dual_sqrt`` NumPy twin propagates NaN.
* **R-9 / R-10** -- HG/LG take ``w0`` in the slot every other factory
  uses for ``wavelength``; annular siblings disagree on radius vs
  diameter.  Documented + a zero-false-positive runtime swap warning.
* **R-17** -- ``wave_traced`` / ``use_traced_lens`` / ``focus_search``
  are documented public flags with live branches and zero callers:
  deprecated (removal v5.32), not deleted.
* **R-18 / overdue shims** -- ``version_removed='5.0'`` was still
  shipping at v5.29; re-scheduled to v5.32 through one constant.
* **R-18 dead code** -- counter-pins for the four claims this wave
  verified to be WRONG (the symbols have live or grep-invisible
  consumers).

Author: audit wave 3, Territory R.
"""
from __future__ import annotations

import importlib
import inspect
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.raytrace._conic_core import conic_sag, conic_sag_derivs
from lumenairy.raytrace.surface import Surface, _surface_sag_derivatives_xy, _surface_sag_xy
from lumenairy.raytrace.trace import (
    _make_bundle,
    make_fan,
    make_grid,
    make_ring,
    surfaces_from_prescription,
    trace,
    validate_prescription,
)

_WL = 587.56e-9


def _singlet(R1=51.5e-3):
    return la.make_singlet(R1=R1, R2=float('inf'), d=4e-3, glass='N-BK7',
                           aperture=10e-3)


# ==========================================================================
# R-8 / E-L7 -- odd aspheric powers rejected at both shared entry points
# ==========================================================================

class TestR8OddAsphericPowerRejected:
    """The sag evaluates ``coeff * h_sq ** (power // 2)`` -- ``h ** power``
    only for EVEN ``power``.  For an ODD power the ``//`` floors silently, so
    the surface is the next-lower EVEN one in the sag while the NumPy normal
    uses ``power * h ** (power - 1)`` and the JAX normal
    ``coeff * power * h_sq ** ((power - 2) // 2) * x``.

    Pre-fix measurement (``{5: 1e6}``, flat base, h = 10 mm): sag = 0.01 m
    (i.e. ``h**4``, the floored power), NumPy dz/dx = 0.05, JAX dz/dx = 5.0,
    sag-consistent truth = 4.0 -- a 100x CROSS-BACKEND divergence on a
    silently-accepted surface.  ``validate_prescription`` rejected it, but a
    hand-built ``Surface`` and the JAX prescription path (which never builds a
    ``Surface``) both sailed through.
    """

    def test_r8_surface_dataclass_rejects_odd_power(self):
        with pytest.raises(ValueError, match=r'ODD aspheric power'):
            Surface(radius=np.inf, aspheric_coeffs={5: 1.0e6})

    def test_r8_surface_dataclass_rejects_odd_power_y(self):
        with pytest.raises(ValueError, match=r'ODD aspheric power'):
            Surface(radius=np.inf, radius_y=60e-3,
                    aspheric_coeffs_y={3: 1.0e6})

    def test_r8_surface_message_matches_validate_prescription(self):
        """Same diagnosis text at both entry points (single-sourced in
        ``_conic_core.check_even_aspheric_powers``) so a user who hits one
        can grep the other."""
        pres = {'surfaces': [{'radius': 50e-3, 'glass_before': 'air',
                              'glass_after': 'N-BK7',
                              'aspheric_coeffs': {5: 1.0e6}},
                             {'radius': np.inf, 'glass_before': 'N-BK7',
                              'glass_after': 'air'}],
                'thicknesses': [3e-3], 'aperture_diameter': 20e-3}
        with pytest.raises(ValueError) as vp:
            validate_prescription(pres)
        with pytest.raises(ValueError) as sf:
            Surface(radius=50e-3, aspheric_coeffs={5: 1.0e6})
        tail = ('only EVEN powers are supported (sag uses h**(p//2) while '
                'the normal uses p*h**(p-1), so an odd power is '
                'sag/normal-inconsistent).')
        assert tail in str(vp.value)
        assert tail in str(sf.value)

    @pytest.mark.parametrize('fn', [conic_sag, conic_sag_derivs])
    def test_r8_conic_core_pair_rejects_odd_power(self, fn):
        """The shared core is the JAX backend's ONLY sag route, so guarding
        it covers ``trace_jax`` on a raw prescription dict."""
        with pytest.raises(ValueError, match=r'ODD aspheric power'):
            fn(np.array([1e-2]), np.array([0.0]), np.inf, 0.0,
               ((5, 1.0e6),), xp=np)

    def test_r8_trace_jax_prescription_path_rejects_odd_power(self):
        """Pre-fix this returned a finite ray height (0.004502378897875953 m)
        from a sag/normal-inconsistent surface: the one backend that bypasses
        BOTH ``validate_prescription`` and the ``Surface`` dataclass."""
        pytest.importorskip('jax')
        from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax
        pres = {'surfaces': [{'radius': 50e-3, 'glass_before': 'air',
                              'glass_after': 'N-BK7',
                              'aspheric_coeffs': {5: 1.0e6}},
                             {'radius': np.inf, 'glass_before': 'N-BK7',
                              'glass_after': 'air'}],
                'thicknesses': [3e-3], 'aperture_diameter': 20e-3}
        st = make_jax_ray_state(
            x=np.array([0.0]), y=np.array([5e-3]), z=np.array([0.0]),
            L=np.array([0.0]), M=np.array([0.0]), N=np.array([1.0]),
            opd=np.array([0.0]), alive=np.array([True]))
        with pytest.raises(ValueError, match=r'ODD aspheric power'):
            trace_jax(st, pres, _WL)

    def test_r8_non_integral_power_rejected(self):
        """``4.5`` floors to ``h_sq ** 2.0`` in the sag -- the same silent
        mis-evaluation as an odd integer, so it is rejected too."""
        with pytest.raises(ValueError, match=r'ODD aspheric power'):
            Surface(radius=np.inf, aspheric_coeffs={4.5: 1.0})

    def test_r8_guard_is_not_vacuous_even_powers_still_trace(self):
        """Non-vacuity: EVEN powers construct, trace, and keep their
        sag/normal consistency (FD-vs-analytic on the shared pair)."""
        A4 = 1.0e5
        s = Surface(radius=np.inf, aspheric_coeffs={4: A4},
                    glass_before='air', glass_after='N-BK7', thickness=3e-3)
        h = 1.0e-2
        sag = float(_surface_sag_xy(np.array([h]), np.array([0.0]), s)[0])
        dzdx, _ = _surface_sag_derivatives_xy(np.array([h]), np.array([0.0]),
                                              s)
        assert sag == pytest.approx(A4 * h ** 4, rel=1e-12)
        assert float(np.asarray(dzdx)[0]) == pytest.approx(
            4.0 * A4 * h ** 3, rel=1e-9)
        # and the JAX-side twin agrees with the NumPy side
        zx, _ = conic_sag_derivs(np.array([h]), np.array([0.0]), np.inf, 0.0,
                                 ((4, A4),), xp=np)
        assert float(np.asarray(zx)[0]) == pytest.approx(
            4.0 * A4 * h ** 3, rel=1e-9)

    def test_r8_helper_accepts_empty_and_even(self):
        # local import: the helper does not exist on a pre-fix tree, and a
        # module-level import would collapse this whole file into one
        # collection error instead of per-finding failures.
        from lumenairy.raytrace._conic_core import check_even_aspheric_powers
        check_even_aspheric_powers((), fn_label='probe')
        check_even_aspheric_powers((2, 4, 6, 8, 10), fn_label='probe')
        check_even_aspheric_powers({4: 1.0, 6: 2.0}.keys(), fn_label='probe')


# ==========================================================================
# R-11 -- f_number magnitude, matching all three siblings
# ==========================================================================

class TestR11FNumberIsUnsigned:
    """``paraxial.f_number`` returned ``efl / ap`` -- SIGNED.  A diverging
    singlet measured ``f/-9.965`` while ``raytrace.layout`` (``abs(efl) /
    ap``) and ``merit_terms.MaxFNumberMerit`` (``abs(ctx.efl) / ap``)
    reported ``+9.965`` for the identical prescription.  f/# is a cone-angle
    magnitude (``1 / (2 NA)``) by definition."""

    def test_r11_diverging_prescription_gives_positive_f_number(self):
        fn = la.f_number(_singlet(R1=-51.5e-3), _WL)
        assert fn > 0.0, f'f_number returned a signed f/#: {fn}'
        assert np.isfinite(fn)

    def test_r11_matches_the_abs_siblings_bitwise(self):
        from lumenairy.raytrace.core import system_abcd
        pres = _singlet(R1=-51.5e-3)
        _, efl, _, _ = system_abcd(surfaces_from_prescription(pres), _WL)
        assert efl < 0.0, 'probe must be a diverging system to be meaningful'
        sibling = abs(efl) / float(pres['aperture_diameter'])
        assert la.f_number(pres, _WL) == float(sibling)

    def test_r11_converging_case_unchanged(self):
        """The (only previously-exercised) converging case is untouched."""
        assert la.f_number(_singlet(R1=51.5e-3), _WL) == pytest.approx(
            9.96516816258418, rel=1e-12)

    def test_r11_degenerate_still_inf(self):
        """No ``aperture_diameter`` key at all -> the documented ``inf``.
        (An explicit ``0.0`` is rejected earlier, by
        ``validate_prescription``'s positive-and-finite check.)"""
        pres = _singlet()
        pres.pop('aperture_diameter', None)
        assert la.f_number(pres, _WL) == float('inf')


# ==========================================================================
# R-12 -- _sellmeier_index accepts arrays, at parity with its sibling
# ==========================================================================

_BK7_SELLMEIER = ((1.03961212, 0.231792344, 1.01046945),
                  (0.00600069867, 0.0200179144, 103.560653))


class TestR12SellmeierVectorised:
    """Pre-fix: ``_sellmeier_index(np.array([...]))`` raised "The truth value
    of an array with more than one element is ambiguous" out of the resonance
    guard, and a list raised "can't multiply sequence by non-int of type
    'float'" -- while ``_polynomial_index``'s docstring advertised scalar/array
    parity with "the ``_sellmeier_index`` sibling"."""

    @staticmethod
    def _fn():
        return importlib.import_module('lumenairy.glass')._sellmeier_index

    def test_r12_ndarray_input_matches_scalar_loop_bitwise(self):
        f = self._fn()
        lams = np.array([420e-9, 500e-9, 587.6e-9, 700e-9, 1.55e-6])
        vec = np.asarray(f(lams, _BK7_SELLMEIER, 'N-BK7'), dtype=float)
        ref = np.array([f(float(L), _BK7_SELLMEIER, 'N-BK7') for L in lams])
        assert vec.shape == lams.shape
        assert np.array_equal(vec, ref), (vec - ref)

    def test_r12_list_and_tuple_inputs_work(self):
        f = self._fn()
        lams = [500e-9, 587.6e-9]
        assert np.allclose(np.asarray(f(lams, _BK7_SELLMEIER)),
                           np.asarray(f(tuple(lams), _BK7_SELLMEIER)),
                           rtol=0, atol=0)

    def test_r12_scalar_path_still_returns_a_python_float(self):
        f = self._fn()
        v = f(587.6e-9, _BK7_SELLMEIER, 'N-BK7')
        assert type(v) is float, type(v)
        assert v == pytest.approx(1.5167984379050088, rel=1e-15)

    def test_r12_shape_preserved_for_2d(self):
        f = self._fn()
        lams = np.array([[500e-9, 587.6e-9], [700e-9, 1.0e-6]])
        assert np.asarray(f(lams, _BK7_SELLMEIER)).shape == (2, 2)

    def test_r12_vector_resonance_guard_names_the_offender(self):
        """The vector path must keep the 4.10 resonance diagnosis, not fall
        back to an opaque ``math domain error`` / ambiguity error."""
        f = self._fn()
        lam_res = np.sqrt(_BK7_SELLMEIER[1][1]) * 1e-6  # lam^2 == C2
        with pytest.raises(ValueError, match=r'Sellmeier resonance'):
            f(np.array([587.6e-9, lam_res]), _BK7_SELLMEIER, 'N-BK7')

    def test_r12_public_get_glass_index_still_scalar_exact(self):
        """End-to-end non-regression on the hot scalar catalogue path."""
        from lumenairy.glass import get_glass_index
        assert get_glass_index('N-BK7', 587.56e-9) == pytest.approx(
            1.5168, abs=1e-4)


# ==========================================================================
# R-13 -- NumPy DOE kick zero / non-finite period guard (JAX-twin parity)
# ==========================================================================

class TestR13DoeZeroPeriodGuard:
    """``trace.py``'s DOE kick divided by the period unguarded: ``period=0``
    raised ``ZeroDivisionError`` mid-trace and ``period=nan`` silently
    NaN-poisoned (L, M) -- measured NumPy ``(nan, nan)`` vs JAX
    ``(0.0, 0.0)``.  The JAX twin documents the contract: "Returns 0.0 when
    ``period`` is non-finite or zero"."""

    @staticmethod
    def _numpy_kick(period):
        surfs = [Surface(radius=np.inf, thickness=10e-3),
                 Surface(radius=np.inf, thickness=0.0)]
        rays = la.make_ray(0.0, 1e-3, 0.0, 0.0, wavelength=_WL)
        out = trace(rays, surfs, _WL,
                    surface_diffraction={0: (1, 0, period, np.inf)}
                    ).image_rays
        return float(out.L[0]), float(out.M[0])

    @pytest.mark.parametrize('period', [0.0, np.nan, -0.0])
    def test_r13_degenerate_period_gives_zero_kick(self, period):
        assert self._numpy_kick(period) == (0.0, 0.0)

    def test_r13_inf_period_unchanged(self):
        assert self._numpy_kick(np.inf) == (0.0, 0.0)

    def test_r13_real_period_still_kicks(self):
        """Non-vacuity: a real grating still diffracts by m*lam/Lambda."""
        L, M = self._numpy_kick(1e-6)
        assert L == pytest.approx(_WL / 1e-6, rel=1e-12)
        assert M == 0.0

    @pytest.mark.parametrize('period', [0.0, np.nan, np.inf])
    def test_r13_matches_the_jax_twin(self, period):
        pytest.importorskip('jax')
        from lumenairy.raytrace.jax_trace import _apply_doe_kick_jax, make_jax_ray_state
        st = make_jax_ray_state(
            x=np.array([0.0]), y=np.array([1e-3]), z=np.array([0.0]),
            L=np.array([0.0]), M=np.array([0.0]), N=np.array([1.0]),
            opd=np.array([0.0]), alive=np.array([True]))
        out = _apply_doe_kick_jax(st, 1, 0, period, np.inf, _WL)
        jx = (float(np.asarray(out.L)[0]), float(np.asarray(out.M)[0]))
        assert self._numpy_kick(period) == jx


# ==========================================================================
# R-14 -- aperture-clip order: ALIVE rows are exactly equal across backends
# ==========================================================================

class TestR14ApertureClipOrderParity:
    """NumPy clips INSIDE ``_intersect_surface`` (before ``_refract``), the
    JAX backends clip AFTER refracting.  Measured consequence: the alive
    masks agree exactly and every ALIVE row's ``L`` / ``x`` is
    BIT-IDENTICAL (0.0 difference), because the clip reads only ``(x, y)``
    -- which refraction never touches -- and ``alive`` is the same
    conjunction either way.  Only DEAD rows differ (measured
    ``|dL| = 0.187``): NumPy leaves a clipped ray's incident direction,
    JAX leaves its refracted one.

    The reorder is DECLINED for this wave: the two sites live in
    ``intersection.py`` / ``jax_trace.py`` (outside this territory), the
    difference is unobservable for live rays, and moving the NumPy clip
    after refraction would relabel a ray that is BOTH vignetted and
    TIR-ing from ``RAY_APERTURE`` to ``RAY_TIR``.  This pin locks the
    observable contract so a future reorder cannot silently change it.
    """

    _PRES = {'surfaces': [{'radius': 50e-3, 'glass_before': 'air',
                           'glass_after': 'N-BK7', 'semi_diameter': 4e-3},
                          {'radius': -50e-3, 'glass_before': 'N-BK7',
                           'glass_after': 'air', 'semi_diameter': 4e-3}],
             'thicknesses': [4e-3], 'aperture_diameter': 20e-3}

    def test_r14_alive_rows_agree_across_backends(self):
        """Alive rows must agree to the JAX backend's own floating-point
        resolution -- BIT-IDENTICAL under ``JAX_ENABLE_X64=1`` (measured 0.0
        for ``L`` and ``x``), and within float32 eps otherwise (measured
        1.9e-07 relative).  A clip-order change that touched live rays would
        blow past both."""
        jax = pytest.importorskip('jax')
        from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax
        x64 = bool(jax.config.read('jax_enable_x64'))
        xs = np.array([0.0, 2e-3, 6e-3, 9e-3])
        ys = np.zeros(4)
        surfs = surfaces_from_prescription(self._PRES)
        npo = trace(_make_bundle(xs, ys, np.zeros(4), np.zeros(4), _WL),
                    surfs, _WL).image_rays
        st = make_jax_ray_state(x=xs, y=ys, z=np.zeros(4), L=np.zeros(4),
                                M=np.zeros(4), N=np.ones(4),
                                opd=np.zeros(4), alive=np.ones(4, bool))
        jxo = trace_jax(st, self._PRES, _WL)

        alive_np = np.asarray(npo.alive, bool)
        alive_jx = np.asarray(jxo.alive, bool)
        assert alive_np.tolist() == alive_jx.tolist()
        assert alive_np.any() and (~alive_np).any(), (
            'probe must straddle the clip to be meaningful')
        for field in ('x', 'y', 'L', 'M'):
            a = np.asarray(getattr(npo, field), float)[alive_np]
            b = np.asarray(getattr(jxo, field), float)[alive_np]
            if x64:
                assert np.array_equal(a, b), (field, a, b)
            else:
                assert np.allclose(a, b, rtol=1e-6, atol=1e-12), (
                    field, a, b, a - b)

    def test_r14_numpy_clip_is_attributed_to_the_aperture(self):
        """NumPy's clip-before-refract order is what makes the diagnosis
        ``RAY_APERTURE`` rather than ``RAY_TIR`` for a doubly-failing ray;
        that attribution is the reason the reorder was declined."""
        from lumenairy.raytrace.surface import RAY_APERTURE, RAY_OK
        xs = np.array([0.0, 9e-3])
        surfs = surfaces_from_prescription(self._PRES)
        out = trace(_make_bundle(xs, np.zeros(2), np.zeros(2), np.zeros(2),
                                 _WL), surfs, _WL).image_rays
        assert out.error_code.tolist() == [RAY_OK, RAY_APERTURE]


# ==========================================================================
# R-15 -- CONVENTION LOCK (declined behaviour change)
# ==========================================================================

class TestR15MakeFanFieldAxisConvention:
    """CONVENTION LOCK -- passes pre-fix by design.

    The audit proposed making ``make_fan(axis='x')`` tilt the field in ``M``
    "like every sibling".  DECLINED, by measurement: ``ray_fan.ray_fan_data``
    / ``ray_fan_data_world`` / ``opd_fan_data`` / ``opd_fan_data_world`` all
    reference each fan against a chief of the SAME orientation (``chief_y``
    in ``M``, ``chief_x`` in ``L``) precisely so ``ey(0) == ex(0) == 0``
    (the RT-5 invariant).  Monkeypatching ``make_fan`` to the proposed
    "always M" form on a BK7 singlet at a 3 deg field moved ``ex(0)`` from
    exactly 0.0 to -1.381e-04 m, i.e. the sagittal fan no longer passes
    through its own chief -- the very bug RT-5 fixed.  ``analysis.field``
    already documents the pitfall (its 4.10 comment) and builds the true
    sagittal fan by hand instead.

    So: ``make_fan`` builds the MERIDIONAL fan of the plane containing
    ``axis``; ``make_ring`` / ``make_grid`` sample 2-D patterns at a +y
    field.  Both are correct and both are now documented.  This locks them.
    """

    _FA = 0.1

    def test_r15_make_fan_tilts_along_its_own_axis(self):
        fy = make_fan('y', 10e-3, 5, self._FA, _WL)
        fx = make_fan('x', 10e-3, 5, self._FA, _WL)
        s = float(np.sin(self._FA))
        assert np.allclose(fy.M, s) and np.allclose(fy.L, 0.0)
        assert np.allclose(fx.L, s) and np.allclose(fx.M, 0.0)

    def test_r15_ring_and_grid_always_tilt_in_m(self):
        s = float(np.sin(self._FA))
        for b in (make_ring(10e-3, 8, self._FA, _WL),
                  make_grid(10e-3, 3, self._FA, _WL)):
            assert np.allclose(b.M, s) and np.allclose(b.L, 0.0)

    def test_r15_convention_buys_the_rt5_zero_crossing(self):
        """The invariant that the per-axis convention exists to preserve:
        both fans pass through their own chief at the pupil centre."""
        rf = importlib.import_module('lumenairy.raytrace.ray_fan')
        surfs = surfaces_from_prescription(_singlet())
        _, ey, px, ex = rf.ray_fan_data(surfs, _WL, 4e-3,
                                        field_angle=np.radians(3.0),
                                        n_rays=21)
        mid = len(px) // 2
        assert abs(float(ey[mid])) < 1e-12, float(ey[mid])
        assert abs(float(ex[mid])) < 1e-12, float(ex[mid])

    def test_r15_docstring_documents_the_divergence(self):
        doc = inspect.getdoc(make_fan) or ''
        assert 'make_ring' in doc and 'make_grid' in doc, (
            'make_fan must document that its field convention differs from '
            'its siblings, so the next reader does not "fix" it')


# ==========================================================================
# R-16 -- numba _dsqrtq propagates NaN like its NumPy twin
# ==========================================================================

class TestR16NumbaDualSqrtNanParity:
    """``_dsqrtq`` clamped with ``a[0] if a[0] > 0.0 else 0.0``; ``nan > 0``
    is False, so a NaN radicand became a perfectly finite ``0.0`` with a
    huge-but-finite tangent -- while ``_dual_sqrt``'s
    ``np.maximum(nan, 0.0)`` is ``nan`` (numpy's maximum propagates NaN),
    giving nan value AND nan tangent.  Measured pre-fix: numpy ``nan`` vs
    numba ``0.0``.

    The divergence is INTERNAL: ``_adrt_numba`` scrubs its outputs through
    ``np.nan_to_num`` and every alive test is ``< 0.0`` (False for NaN on
    both sides), so it is invisible at the public boundary -- verified by
    sweeping 7 NaN/inf/extreme input placements x refract/mirror stacks on a
    pre-fix worktree: all IDENTICAL.  That is exactly why the primitive is
    published as ``_ADRT_NUMBA_PRIMS`` and compared directly here: no
    end-to-end pin can see this class of bug.
    """

    @staticmethod
    def _prims():
        pytest.importorskip('numba')
        dif = importlib.import_module('lumenairy.raytrace.differential')
        assert dif._adrt_numba_kernel() is not None, 'numba kernel unavailable'
        prims = dif._ADRT_NUMBA_PRIMS
        assert prims is not None, '_ADRT_NUMBA_PRIMS not published'
        return dif, prims

    def test_r16_nan_radicand_propagates(self):
        dif, prims = self._prims()
        v, *tang = prims['dsqrt']((np.nan, 1.0, 2.0, 3.0, 4.0))
        assert np.isnan(v), f'numba _dsqrtq clamped a NaN radicand to {v!r}'
        assert all(np.isnan(t) for t in tang), tang

    @pytest.mark.parametrize('radicand', [4.0, 1.0, 0.0, -1.0, -1e-9,
                                          np.inf, np.nan])
    def test_r16_matches_dual_sqrt_bitwise(self, radicand):
        """Full parity against ``_dual_sqrt`` on normalized radicands.

        (Subnormal radicands ~1e-320 diverge for an unrelated, PRE-EXISTING
        reason -- numba's ``math.sqrt`` flushes them to 0.0 where
        ``np.sqrt`` does not -- and are deliberately out of this pin's
        scope.)"""
        dif, prims = self._prims()
        d = (1.0, 2.0, 3.0, 4.0)
        got = prims['dsqrt']((radicand,) + d)
        ref = dif._dual_sqrt(dif._AdrtDual(np.array([radicand]),
                                           np.array([list(d)])))
        assert np.array_equal(np.array([got[0]]), ref.v, equal_nan=True)
        assert np.array_equal(np.array(got[1:]), ref.d[0], equal_nan=True)

    def test_r16_end_to_end_backends_still_agree(self):
        """Non-regression: healthy rays remain bit-identical between the
        numba and NumPy-dual ADRT backends after the change."""
        dif, _ = self._prims()
        surfs = [Surface(radius=50e-3, glass_before='air',
                         glass_after='N-BK7', thickness=4e-3),
                 Surface(radius=-50e-3, glass_before='N-BK7',
                         glass_after='air', thickness=0.0)]
        x = np.array([0.0, 1e-3, -2e-3])
        z = np.zeros(3)
        nb = dif._adrt_numba(x, z, z, z, surfs, _WL)
        npd = dif._adrt_numpy(x, z, z, z, surfs, _WL, False)
        assert np.array_equal(np.asarray(nb.jacobian),
                              np.asarray(npd.jacobian))
        for f in ('x', 'y', 'ux', 'uy', 'opd'):
            assert np.array_equal(np.asarray(getattr(nb, f)),
                                  np.asarray(getattr(npd, f))), f


# ==========================================================================
# R-9 / R-10 -- API hazards: documented + a runtime swap warning
# ==========================================================================

class TestR9HgLgPositionalSwap:
    """``create_hermite_gauss`` / ``create_laguerre_gauss`` take ``w0`` in
    positional slot 3 -- the slot ``create_gaussian_beam`` /
    ``create_annular_beam`` / ``create_bessel_beam`` / ``create_top_hat_beam``
    all use for ``wavelength``.  A swapped call was silently accepted.  The
    discriminator is physical (``w0 >= wavelength`` for any real paraxial
    mode) and false-positive-free on the corpus: the six in-repo HG/LG call
    sites run at ``w0 / wavelength`` = 6.45, 15.8, 19.1, 38.2, 45.8 and 2000,
    so the tightest one clears the threshold by 6.4x."""

    @pytest.mark.parametrize('name', ['create_hermite_gauss',
                                      'create_laguerre_gauss'])
    def test_r9_swapped_call_warns(self, name):
        fn = getattr(la, name)
        with pytest.warns(UserWarning, match=r'SMALLER than wavelength'):
            fn(32, 4e-6, 1.31e-6, 50e-6)   # w0 <-> wavelength swapped

    @pytest.mark.parametrize('name', ['create_hermite_gauss',
                                      'create_laguerre_gauss'])
    def test_r9_correct_call_is_silent(self, name):
        fn = getattr(la, name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fn(32, 4e-6, 50e-6, 1.31e-6)
        assert not [w for w in caught
                    if 'SMALLER than wavelength' in str(w.message)], (
            [str(w.message) for w in caught])

    def test_r9_zero_w0_still_raises_without_a_spurious_warning_first(self):
        """The guard runs AFTER the ``w0 > 0`` validation, so the deliberate
        ``w0 = 0`` / ``w0 < 0`` probes elsewhere in the suite keep raising
        ``ValueError`` cleanly."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(ValueError, match=r'w0 must be a positive'):
                la.create_hermite_gauss(32, 1e-6, 0.0, 0.5e-6)
        assert not [w for w in caught
                    if 'SMALLER than wavelength' in str(w.message)]

    @pytest.mark.parametrize('name', ['create_hermite_gauss',
                                      'create_laguerre_gauss'])
    def test_r9_docstring_carries_the_hazard_warning(self, name):
        doc = inspect.getdoc(getattr(la, name)) or ''
        assert 'ARGUMENT-ORDER HAZARD' in doc, name


class TestR10AnnularRadiusVsDiameter:
    """``create_annular_beam`` takes DIAMETERS, ``create_annular_incoherent_
    source`` takes RADII, for the same geometry.  Documented at both sites
    (no signature change: both sets of kwargs are keyword-ONLY, so a silent
    positional mix-up is impossible -- the hazard is copy-and-rename)."""

    def test_r10_both_sites_warn_about_the_other_convention(self):
        core = importlib.import_module('lumenairy.sources.core')
        d_doc = inspect.getdoc(core.create_annular_beam) or ''
        r_doc = inspect.getdoc(core.create_annular_incoherent_source) or ''
        assert 'RADIUS-vs-DIAMETER HAZARD' in d_doc
        assert 'RADIUS-vs-DIAMETER HAZARD' in r_doc
        assert 'create_annular_incoherent_source' in d_doc
        assert 'create_annular_beam' in r_doc

    def test_r10_kwargs_are_keyword_only_at_both_sites(self):
        core = importlib.import_module('lumenairy.sources.core')
        for fn, names in ((core.create_annular_beam,
                           ('outer_diameter', 'inner_diameter')),
                          (core.create_annular_incoherent_source,
                           ('outer_radius', 'inner_radius'))):
            params = inspect.signature(fn).parameters
            for n in names:
                assert params[n].kind is inspect.Parameter.KEYWORD_ONLY, \
                    (fn.__name__, n)


# ==========================================================================
# R-17 -- zero-caller optimize flags are deprecated, not deleted
# ==========================================================================

class TestR17DeadOptimizeFlagsDeprecated:
    """``design_optimize(wave_traced=)`` and
    ``MatchIdealSystemMerit(use_traced_lens=, focus_search=)`` are
    documented public flags with live branches and ZERO callers anywhere in
    the repo (grep-verified twice: library, tests, validation, examples,
    UI).  Deprecated with removal v5.32 -- NOT deleted, since out-of-repo
    callers cannot be ruled out.  Warnings fire only on a non-default
    value, so the whole existing corpus stays silent."""

    def test_r17_use_traced_lens_warns(self):
        from lumenairy.optimize.merit_terms import MatchIdealSystemMerit
        with pytest.warns(DeprecationWarning, match=r"'use_traced_lens'"):
            MatchIdealSystemMerit([{'type': 'lens', 'f': 50e-3}],
                                   use_traced_lens=True)

    def test_r17_focus_search_warns(self):
        from lumenairy.optimize.merit_terms import MatchIdealSystemMerit
        with pytest.warns(DeprecationWarning, match=r"'focus_search'"):
            MatchIdealSystemMerit([{'type': 'lens', 'f': 50e-3}],
                                   focus_search=True)

    def test_r17_warnings_name_the_v5_32_removal(self):
        from lumenairy.optimize.merit_terms import MatchIdealSystemMerit
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            MatchIdealSystemMerit([{'type': 'lens', 'f': 50e-3}],
                                   use_traced_lens=True, focus_search=True)
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, DeprecationWarning)]
        assert len(msgs) == 2, msgs
        assert all('v5.32' in m for m in msgs), msgs

    def test_r17_default_construction_is_silent(self):
        from lumenairy.optimize.merit_terms import MatchIdealSystemMerit
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            MatchIdealSystemMerit([{'type': 'lens', 'f': 50e-3}])
        assert not [w for w in caught
                    if issubclass(w.category, DeprecationWarning)], (
            [str(w.message) for w in caught])

    def test_r17_flags_and_branches_still_exist(self):
        """Deprecated, not removed: the kwargs and the four unexercised
        penalty helpers must all still be there."""
        from lumenairy.optimize.driver import design_optimize
        from lumenairy.optimize.merit_terms import MatchIdealSystemMerit
        assert 'wave_traced' in inspect.signature(design_optimize).parameters
        mp = inspect.signature(MatchIdealSystemMerit.__init__).parameters
        assert {'use_traced_lens', 'focus_search', 'match'} <= set(mp)
        for helper in ('_focus_search_penalty', '_field_mse_penalty',
                       '_intensity_mse_penalty', '_intensity_overlap_penalty'):
            assert callable(getattr(MatchIdealSystemMerit, helper)), helper

    def test_r17_unexercised_helpers_say_so_in_their_docstrings(self):
        from lumenairy.optimize.merit_terms import MatchIdealSystemMerit
        for helper in ('_focus_search_penalty', '_field_mse_penalty',
                       '_intensity_mse_penalty', '_intensity_overlap_penalty'):
            doc = inspect.getdoc(getattr(MatchIdealSystemMerit, helper)) or ''
            assert 'R-17' in doc, helper

    def test_r17_design_optimize_wave_traced_warns(self):
        """The flag's warning must fire from the PUBLIC entry point, not
        only from an internal helper."""
        from lumenairy.optimize.driver import design_optimize
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                design_optimize(object(), [], _WL, wave_traced=True)
            except Exception:
                pass          # any downstream failure is fine; we want the warn
        dep = [str(w.message) for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'wave_traced' in str(w.message)]
        assert dep, [str(w.message) for w in caught]
        assert 'v5.32' in dep[0], dep


# ==========================================================================
# Overdue shims -- the blown v5.0 horizon is re-scheduled, and still fires
# ==========================================================================

class TestOverdueSourceShims:
    """``sources/core.py`` shipped ``version_removed='5.0'`` at v5.29 -- 29
    minor releases past its own removal date -- across the
    ``create_led_source`` positional shim, the five ``Source.*`` legacy
    positional shims and the Schell ``return_kind`` sentinel helper.  A
    removal version the library has demonstrably blown through trains
    callers to ignore the message.  Re-scheduled ONCE, to v5.32, through the
    single ``_OVERDUE_SHIM_VERSION_REMOVED`` constant."""

    def test_shims_no_longer_advertise_the_blown_v5_0_horizon(self):
        core = importlib.import_module('lumenairy.sources.core')
        src = inspect.getsource(core)
        # match the CALL form (trailing comma) so this module's own
        # explanatory comment about the old value doesn't self-trip.
        assert "version_removed='5.0'," not in src, (
            'a shim still hard-codes the blown v5.0 removal horizon')
        assert core._OVERDUE_SHIM_VERSION_REMOVED == '5.32'

    def test_led_positional_shim_fires_and_names_v5_32(self):
        core = importlib.import_module('lumenairy.sources.core')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            la.create_led_source(64, 16e-6, 100e-6, 0.3, 1.31e-6)
        dep = [str(w.message) for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'create_led_source' in str(w.message)]
        assert dep, [str(w.message) for w in caught]
        assert f'v{core._OVERDUE_SHIM_VERSION_REMOVED}' in dep[0], dep[0]

    @pytest.mark.parametrize('factory, args', [
        ('gaussian', (50e-6, 32, 4e-6, 1.31e-6)),
        ('plane_wave', (32, 4e-6, 1.31e-6)),
        ('point_source', (32, 4e-6, 1.31e-6)),
        ('top_hat', (50e-6, 32, 4e-6, 1.31e-6)),
        ('fiber_mode', (50e-6, 32, 4e-6, 1.31e-6)),
    ])
    def test_source_legacy_positional_shims_fire_and_name_v5_32(
            self, factory, args):
        """Measured, not assumed: each shim must actually emit from the
        production classmethod, with the re-scheduled version in the text."""
        core = importlib.import_module('lumenairy.sources.core')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            getattr(la.Source, factory)(*args)
        dep = [str(w.message) for w in caught
               if issubclass(w.category, DeprecationWarning)
               and f'Source.{factory}' in str(w.message)]
        assert dep, (factory, [str(w.message) for w in caught])
        assert f'v{core._OVERDUE_SHIM_VERSION_REMOVED}' in dep[0], dep[0]

    def test_schell_return_kind_helper_names_v5_32(self):
        core = importlib.import_module('lumenairy.sources.core')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            core._warn_schell_return_kind_default('probe_factory')
        dep = [str(w.message) for w in caught
               if issubclass(w.category, DeprecationWarning)]
        assert dep, [str(w.message) for w in caught]
        assert f'v{core._OVERDUE_SHIM_VERSION_REMOVED}' in dep[0], dep[0]

    def test_canonical_kwarg_calls_stay_silent(self):
        """Non-vacuity in the other direction: the shims must NOT fire on
        the canonical form."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            la.Source.plane_wave(N=32, dx=4e-6, wavelength=1.31e-6)
        assert not [w for w in caught
                    if issubclass(w.category, DeprecationWarning)], (
            [str(w.message) for w in caught])


# ==========================================================================
# R-18 dead code -- COUNTER-PINS for the claims that verification refuted
# ==========================================================================

class TestR18DeadCodeCounterPins:
    """DEAD-CODE COUNTER-PINS -- pass pre-fix by design.

    Four of the audit's "grep-verified dead, delete" entries have LIVE or
    GREP-INVISIBLE consumers and were NOT deleted.  These pins record the
    consumer so the next hygiene pass does not repeat the mistake.
    """

    def test_r18_invalidate_glass_name_has_a_live_caller(self):
        """Claimed dead; actually called by
        ``raytrace.trace._register_fixed_index`` (the GL-2 lock-correct
        cache purge) -- deleting it breaks every thin-lens prescription."""
        src = inspect.getsource(
            importlib.import_module('lumenairy.raytrace.trace'))
        assert '_invalidate_glass_name(name)' in src
        import lumenairy.glass as g
        assert callable(g._invalidate_glass_name)

    def test_r18_glass_cache_alias_is_required_by_the_lock_walker(self):
        """Claimed a dead alias; actually the companion name the v4.14.2
        cache<->lock dispatcher pin DISCOVERS BY REFLECTION (candidates for
        ``_GLASS_CACHE_LOCK`` are ``_GLASS_CACHE_CACHE`` / ``_GLASS_CACHE``
        -- the lower-case ``_glass_cache`` does NOT match), so no grep can
        see the consumer and deleting it reds that pin."""
        import lumenairy.glass as g
        assert hasattr(g, '_GLASS_CACHE')
        assert g._GLASS_CACHE is g._glass_cache

    def test_r18_polynomial_stub_names_is_a_live_invariant_hook(self):
        """Claimed a dead empty frozenset; the module comment states the
        retention intent (future formula-3 additions land here as a stub
        before their coefficient row, keeping the migration-message
        dispatch arm reachable) AND a load-time well-formedness loop reads
        it.  Emptiness is the INVARIANT (stubbed and ingested are mutually
        exclusive), not evidence of death -- and
        ``test_v5_2_glass_formula3.py`` already carries two tests that
        SKIP with exactly that reasoning ("_POLYNOMIAL_STUB_NAMES is empty
        (v5.2.3 finished the ingestion); the NotImplementedError dispatch
        arm has no triggering input until a future catalogue addition")."""
        import lumenairy.glass as g
        assert isinstance(g._POLYNOMIAL_STUB_NAMES, frozenset)
        assert not (g._POLYNOMIAL_STUB_NAMES
                    & set(g.POLYNOMIAL_COEFFICIENTS)), (
            'stubbed and ingested must stay disjoint')
        src = inspect.getsource(g)
        assert 'for name in _POLYNOMIAL_STUB_NAMES:' in src

    def test_r18_trace_result_rays_at_is_public_documented_api(self):
        """Claimed dead (zero in-repo callers); it is a documented public
        accessor whose behaviour ``trace()``'s own ``output_filter``
        contract describes ("every ``rays_at(i)`` for ``i <
        len(surfaces)-1`` raises IndexError").  Deprecate-then-remove, not
        delete-now."""
        from lumenairy.raytrace.surface import TraceResult
        assert callable(TraceResult.rays_at)
        assert 'rays_at' in (inspect.getdoc(trace) or '')
        surfs = surfaces_from_prescription(_singlet())
        res = trace(la.make_ray(0.0, 1e-3, 0.0, 0.0, wavelength=_WL),
                    surfs, _WL)
        assert res.rays_at(-1) is res.image_rays

    def test_r18_public_return_types_stay_exported(self):
        """``DifferentialTransfer`` and ``ParetoResult`` were listed as
        "exported but unused".  Both are the documented RETURN TYPES of
        public functions (constructed at 6 and 1 sites respectively), so
        the export is what lets callers annotate / isinstance them."""
        assert la.DifferentialTransfer is not None
        assert la.ParetoResult is not None
        from lumenairy.raytrace.differential import ray_transfer_jacobian
        surfs = surfaces_from_prescription(_singlet())
        out = ray_transfer_jacobian(np.array([0.0]), np.array([1e-3]),
                                    np.array([0.0]), np.array([0.0]),
                                    surfs, _WL)
        assert isinstance(out, la.DifferentialTransfer)
