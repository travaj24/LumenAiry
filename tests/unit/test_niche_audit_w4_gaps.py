"""Wave-4 recorded-leftover closures (v5.31).

Four independent gaps, each measured pre-fix and pinned here:

* **W4-1 / R-8 residual** -- the WAVE-OPTICS aspheric sag sites evaluated
  ``coeff * h_sq ** (power // 2)`` straight from a prescription dict without
  ever building a :class:`~lumenairy.raytrace.Surface`, so an ODD power floored
  silently to the next-lower EVEN one.  ``lenses.surface_sag_general`` (both its
  numba kernel and its NumPy fallback), ``lenses.surface_sag_biconic`` (both
  per-axis coefficient dicts) and ``_lens_thin.apply_aspheric_lens`` (both
  surfaces, both the flat and the curved branch) now route through the SAME
  shared ``check_even_aspheric_powers`` the ``Surface`` dataclass and the JAX
  prescription path already used.

* **W4-2** -- ``apply_thin_lens(f=non-finite)``, and therefore
  ``JonesField.apply_thin_lens``, silently poisoned or silently no-op'd.

* **W4-3** -- a 1-D ``RCWAStack`` handed a y-VARYING 2-D cell silently solves
  the cell's y-AVERAGE.  Closed as a DIAGNOSTIC (``RCWAYAverageWarning``), not a
  rejection: the y-average is the documented 1-D contract and is EXACT for a
  y-INVARIANT 2-D cell, which is what the existing
  ``test_audit_s1_2_rcwa_lossless_tripwire`` pin and the 2-D-shaped-cell-on-1-D
  -stack idiom rely on.

* **W4-4 / R-18** -- the dead ``surface_diffraction`` parameters on the two
  jax trace bodies and the dead ``jp_aux`` on ``_make_jit_kernel``.

Every measured number quoted below was taken on a pre-fix worktree of 865e922.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from lumenairy.elements import (
    apply_aspheric_lens,
    apply_real_lens,
    apply_thin_lens,
)
from lumenairy.elements.lenses import surface_sag_biconic, surface_sag_general
from lumenairy.elements.polarization import (
    JonesField,
    degree_of_polarization,
)
from lumenairy.elements.rcwa import RCWAStack, RCWAYAverageWarning
from lumenairy.elements.rcwa._core import _validate_cell_sampling
from lumenairy.elements.rcwa.stack import _warn_if_y_averaged

_WL = 1.55e-6
_DX = 5e-6


# ===========================================================================
# W4-1 -- odd aspheric powers at the WAVE-OPTICS sag sites
# ===========================================================================

class TestW41OddAsphericPowerWaveOptics:
    """The five ``power // 2`` wave-optics sites reject ODD powers.

    Pre-fix measurement (h = 10 mm, flat base, ``{5: 1e6}`` through
    ``surface_sag_general``): sag ``0.01`` m -- BIT-identical to the
    ``{4: 1e6}`` sag, i.e. the floored EVEN twin -- against the true
    ``1.0000000000000002e-04``, a factor 100; the returned sag's ``dz/dh``
    was ``4.0`` against the true ``0.05`` (80x).  ``surface_sag_biconic``
    with ``{3: 1e4}`` returned ``1.0`` m against the true ``0.01`` m (100x),
    on the x AND the y coefficient dict.  ``apply_aspheric_lens(A1={5: 1e6})``
    returned a field bit-identical to ``A1={4: 1e6}`` on both the flat and the
    curved branch, and ``apply_real_lens`` accepted ``{5: 1e6}`` and returned a
    field bit-identical to the ``{4: 1e6}`` lens.
    """

    # The message every site shares (single-sourced in
    # raytrace._conic_core.check_even_aspheric_powers).
    _MSG = r"ODD aspheric power\(s\).*only EVEN powers are supported"

    def test_surface_sag_general_rejects_odd_power(self):
        h_sq = np.array([1e-4])
        with pytest.raises(ValueError, match=self._MSG) as ei:
            surface_sag_general(h_sq, np.inf, 0.0, {5: 1e6})
        assert 'surface_sag_general' in str(ei.value), (
            "the error must name the rejecting site so the user can find it; "
            f"got {str(ei.value)!r}")

    @pytest.mark.parametrize('R', [np.inf, 0.05, -0.05])
    @pytest.mark.parametrize('asph', [{5: 1e6}, {3: 1.0}, {4: 1e6, 7: 1.0},
                                      {1: 1.0}])
    def test_surface_sag_general_rejects_odd_on_every_branch(self, R, asph):
        """Flat AND curved base, odd-only AND mixed even/odd dicts."""
        with pytest.raises(ValueError, match=self._MSG):
            surface_sag_general(np.array([1e-4]), R, -0.5, asph)

    def test_surface_sag_general_odd_power_grid_hits_both_kernels(self):
        """A float64 grid takes the numba kernel (``powers[j] // 2``) when
        numba is present and the NumPy fallback (``power // 2``) otherwise;
        the guard runs BEFORE the dispatch so both are covered."""
        xx = np.linspace(-0.01, 0.01, 33)
        X, Y = np.meshgrid(xx, xx)
        h_sq = np.ascontiguousarray(X ** 2 + Y ** 2)
        with pytest.raises(ValueError, match=self._MSG):
            surface_sag_general(h_sq, np.inf, 0.0, {5: 1e6})

    def test_surface_sag_biconic_rejects_odd_x_coeffs(self):
        X, Y = np.meshgrid(np.linspace(-0.01, 0.01, 5),
                           np.linspace(-0.01, 0.01, 5))
        with pytest.raises(ValueError, match=self._MSG) as ei:
            surface_sag_biconic(X, Y, 0.05, -0.08, 0.0, 0.0, {3: 1e4})
        assert 'surface_sag_biconic' in str(ei.value)

    def test_surface_sag_biconic_rejects_odd_y_coeffs(self):
        """``aspheric_coeffs_y`` is a SEPARATE dict feeding the same
        ``h_sq ** (power // 2)``; the guard must cover it and say which one."""
        X, Y = np.meshgrid(np.linspace(-0.01, 0.01, 5),
                           np.linspace(-0.01, 0.01, 5))
        with pytest.raises(ValueError, match=self._MSG) as ei:
            surface_sag_biconic(X, Y, 0.05, -0.08, 0.0, 0.0,
                                {4: 1.0}, {3: 1e4})
        assert 'aspheric_coeffs_y' in str(ei.value), (
            "with two independent coefficient dicts the message must say WHICH "
            f"one is odd; got {str(ei.value)!r}")

    @pytest.mark.parametrize('R1', [np.inf, 0.05])
    @pytest.mark.parametrize('which', ['A1', 'A2'])
    def test_apply_aspheric_lens_rejects_odd(self, R1, which):
        """Both surfaces, and both ``_aspheric_sag`` branches (``R1=inf`` is
        the flat branch, ``R1=0.05`` the curved one)."""
        E = np.ones((16, 16), dtype=complex)
        kw = {which: {5: 1e6}}
        with pytest.raises(ValueError, match=self._MSG) as ei:
            apply_aspheric_lens(E, R1=R1, R2=-0.06, d=2e-3, n_lens=1.5,
                                wavelength=_WL, dx=_DX, **kw)
        assert which in str(ei.value), (
            f"the message must name the offending surface; got {str(ei.value)!r}")

    def test_apply_real_lens_rejects_odd_via_shared_sag(self):
        """``apply_real_lens`` never builds a ``Surface`` -- it calls
        ``surface_sag_general`` directly -- so the sag-site guard is what
        protects it.  Pre-fix it accepted ``{5: 1e6}`` and returned a field
        bit-identical to the ``{4: 1e6}`` lens (measured: identical
        ``sum|E|^2 = 1828.477593041`` and identical off-axis phase)."""
        N, dxr = 64, 3e-3 / 64
        xr = (np.arange(N) - N / 2) * dxr
        Xr, Yr = np.meshgrid(xr, xr)
        E = np.exp(-(Xr ** 2 + Yr ** 2) / 0.8e-3 ** 2).astype(complex)
        presc = {
            'name': 's', 'aperture_diameter': 3e-3, 'thicknesses': [3e-3],
            'surfaces': [
                {'radius': 9e-3, 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'conic': 0.0, 'aspheric_coeffs': {5: 1e6}},
                {'radius': -9e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'conic': 0.0,
                 'aspheric_coeffs': None}]}
        with pytest.raises(ValueError, match=self._MSG):
            apply_real_lens(E, prescription=presc, wavelength=_WL, dx=dxr)

    def test_non_integer_power_also_rejected(self):
        """``2.5`` is not an even integer either; the shared checker has a
        dedicated arm for it."""
        with pytest.raises(ValueError, match='aspheric power'):
            surface_sag_general(np.array([1e-4]), np.inf, 0.0, {2.5: 1.0})

    # ---- the guard must not disturb EVEN powers -------------------------

    @pytest.mark.parametrize('asph', [None, {}, {2: 1e2}, {4: 1e6},
                                      {4: 1e6, 6: -1e9},
                                      {2: 1.0, 4: 1e5, 6: 1e8, 8: -1e11}])
    def test_even_powers_unchanged(self, asph):
        """Even (and empty / absent) coefficient sets must survive the guard
        with the sag they always had.  Checked against an independent
        hand-evaluated oracle rather than a frozen constant, so the pin is
        cross-platform (the guard is pure control flow -- it adds no
        arithmetic -- and the full 149-array byte-capture confirming
        bit-identity is in the closure record)."""
        xx = np.linspace(-0.012, 0.012, 41)
        X, Y = np.meshgrid(xx, xx)
        h_sq = np.ascontiguousarray(X ** 2 + Y ** 2)
        R, kc = 0.05, -0.5
        got = surface_sag_general(h_sq, R, kc, asph)
        norm = (1 + kc) * h_sq / R ** 2
        want = np.where(norm < 0.9999,
                        h_sq / (R * (1 + np.sqrt(np.where(norm < 0.9999,
                                                          1 - norm, 0.01)))),
                        np.nan)
        for p, c in (asph or {}).items():
            want = want + c * h_sq ** (p // 2)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
        m = ~np.isnan(want)
        assert np.allclose(got[m], want[m], rtol=1e-13, atol=0.0), (
            f"even-power sag must be unchanged by the R-8 guard; asph={asph}")

    def test_even_power_biconic_and_thin_screen_still_work(self):
        X, Y = np.meshgrid(np.linspace(-0.01, 0.01, 9),
                           np.linspace(-0.01, 0.01, 9))
        s = surface_sag_biconic(X, Y, 0.05, -0.08, -0.5, 0.3,
                                {4: 1e6}, {6: -1e9})
        assert np.isfinite(s).all() and np.any(s != 0.0)
        E = np.ones((16, 16), dtype=complex)
        out = apply_aspheric_lens(E, R1=0.05, R2=-0.06, d=2e-3, n_lens=1.5,
                                  wavelength=_WL, dx=_DX,
                                  A1={4: 1e6}, A2={6: -1e9})
        assert np.isfinite(out).all() and np.allclose(np.abs(out), 1.0)

    def test_guard_is_the_shared_checker_not_a_copy(self):
        """All five sites must delegate to
        ``raytrace._conic_core.check_even_aspheric_powers`` so the wave-optics
        and ray-trace paths can never disagree about which surfaces exist."""
        import lumenairy.elements._lens_thin as _thin
        import lumenairy.elements.lenses as _lenses
        from lumenairy.raytrace._conic_core import check_even_aspheric_powers
        for mod, fn in ((_lenses, 'surface_sag_general'),
                        (_lenses, 'surface_sag_biconic'),
                        (_thin, 'apply_aspheric_lens')):
            src = inspect.getsource(getattr(mod, fn))
            assert 'check_even_aspheric_powers' in src, (
                f"{mod.__name__}.{fn} must call the SHARED checker, not "
                f"re-implement the even-power test")
        # and the shared checker itself still rejects what we rely on
        with pytest.raises(ValueError):
            check_even_aspheric_powers([5], fn_label='probe')


# ===========================================================================
# W4-2 -- non-finite focal length in apply_thin_lens / JonesField
# ===========================================================================

class TestW42NonFiniteFocalLength:
    """``f`` must be finite.

    Pre-fix measurement (N = 32, dx = 5 um, lambda = 1.55 um; fraction of
    non-finite output pixels)::

        f       paraxial  nonparaxial  aplanatic
        nan     1.000     1.000        0.000   <- silent NO-OP, no lens applied
        +inf    0.000     1.000        1.000
        -inf    0.000     1.000        1.000

    i.e. the SAME bad input was an all-NaN field, a no-op, or nothing at all
    depending on ``lens_model``, and nothing was raised in any case.
    ``JonesField.apply_thin_lens(f=nan)`` returned ``self`` with every pixel of
    Ex AND Ey ``nan+nanj``, and ``degree_of_polarization`` /
    ``stokes_parameters`` read all-NaN.

    W4-2 scoped itself to NON-FINITE ``f`` and recorded ``f = 0`` as a
    measured-but-unfixed sibling inconsistency.  v5.32 (audit W5-2) took
    that decision and extended this same guard to ``f == 0``; see
    :meth:`test_f_zero_recorded_contract_is_SUPERSEDED_by_w5_2` for the
    supersession record and :class:`TestW52ZeroFocalLength` for the live
    contract.  The non-finite behaviour pinned in this class is unchanged.
    """

    _E = np.ones((16, 16), dtype=complex)

    @pytest.mark.parametrize('f', [np.nan, np.inf, -np.inf,
                                   float('nan'), float('inf')])
    @pytest.mark.parametrize('lens_model', ['paraxial', 'nonparaxial',
                                            'aplanatic'])
    def test_scalar_rejects_non_finite_f(self, f, lens_model):
        with pytest.raises(ValueError, match='must be a finite focal length'):
            apply_thin_lens(self._E, f=f, wavelength=_WL, dx=_DX,
                            lens_model=lens_model)

    def test_error_names_the_per_model_divergence(self):
        """The message must explain that the damage DIFFERED per model --
        that inconsistency is the reason this was undetectable."""
        with pytest.raises(ValueError) as ei:
            apply_thin_lens(self._E, f=np.nan, wavelength=_WL, dx=_DX)
        msg = str(ei.value)
        assert 'apply_thin_lens' in msg
        assert 'aplanatic' in msg and 'NO lens' in msg, (
            "the silent-no-op arm is the surprising one and must be named; "
            f"got {msg!r}")

    def test_jonesfield_rejects_non_finite_f_without_mutating(self):
        """``JonesField.apply_thin_lens`` mutates in place and returns
        ``self``, so a poisoned call used to corrupt the caller's own object.
        The guard must fire BEFORE either component is written."""
        jf = JonesField(self._E.copy(), self._E.copy(), _DX)
        with pytest.raises(ValueError, match='must be a finite focal length'):
            jf.apply_thin_lens(f=np.nan, wavelength=_WL)
        assert np.isfinite(jf.Ex).all() and np.isfinite(jf.Ey).all(), (
            "the field must be untouched when the guard rejects f")
        assert np.isfinite(degree_of_polarization(jf)).all()

    @pytest.mark.parametrize('f', [0.03, -0.03, 1e-3, 1e6])
    @pytest.mark.parametrize('lens_model', ['paraxial', 'nonparaxial',
                                            'aplanatic'])
    def test_finite_f_unaffected(self, f, lens_model):
        out = apply_thin_lens(self._E, f=f, wavelength=_WL, dx=_DX,
                              lens_model=lens_model)
        assert np.isfinite(out).all(), (
            f"finite f={f} under {lens_model} must still produce a finite "
            f"field")
        # pure phase screen on a unit field
        assert np.allclose(np.abs(out), 1.0, atol=1e-12)

    def test_finite_f_paraxial_matches_closed_form(self):
        """Guard-only change: the paraxial phase must still be exactly
        ``exp(-i k r^2 / 2f)``.  Oracle-based (not a frozen constant) so the
        pin is cross-platform."""
        N, f = 16, 0.03
        E = np.ones((N, N), dtype=complex)
        out = apply_thin_lens(E, f=f, wavelength=_WL, dx=_DX)
        x = (np.arange(N) - N / 2) * _DX
        X, Y = np.meshgrid(x, x)
        want = np.exp(-1j * (2 * np.pi / _WL) / (2 * f) * (X ** 2 + Y ** 2))
        assert np.allclose(out, want, rtol=0.0, atol=1e-12)

    @pytest.mark.parametrize('f', [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize('axis', ['x', 'y'])
    def test_cylindrical_sibling_rejects_non_finite_f(self, f, axis):
        """``apply_cylindrical_lens`` is the only other ``f``-taking entry
        point in ``_lens_thin`` and had the IDENTICAL gap (measured pre-fix,
        N = 16: ``f=nan`` -> every pixel ``nan+nanj``; ``f=+-inf`` -> silent
        no-op).  Closed in the same change so the two cannot drift apart --
        the W3-T4 polarization sweep exists because an earlier guard did not
        do this."""
        from lumenairy.elements import apply_cylindrical_lens
        with pytest.raises(ValueError, match='must be a finite focal length'):
            apply_cylindrical_lens(self._E, f=f, wavelength=_WL, dx=_DX,
                                   axis=axis)

    def test_cylindrical_finite_f_unaffected(self):
        from lumenairy.elements import apply_cylindrical_lens
        N, f = 16, 0.03
        E = np.ones((N, N), dtype=complex)
        out = apply_cylindrical_lens(E, f=f, wavelength=_WL, dx=_DX, axis='x')
        x = (np.arange(N) - N / 2) * _DX
        want = np.exp(-1j * (2 * np.pi / _WL) / (2 * f) * x ** 2)[None, :]
        assert np.allclose(out, np.broadcast_to(want, (N, N)),
                           rtol=0.0, atol=1e-12)

    def test_both_f_guards_share_one_message_shape(self):
        """A user who hits one and greps for the other should find the same
        wording."""
        from lumenairy.elements import apply_cylindrical_lens
        msgs = []
        for fn, kw in ((apply_thin_lens, {}),
                       (apply_cylindrical_lens, {})):
            with pytest.raises(ValueError) as ei:
                fn(self._E, f=np.nan, wavelength=_WL, dx=_DX, **kw)
            msgs.append(str(ei.value))
        for m in msgs:
            assert 'must be a finite focal length in metres' in m
            assert 'omit the call' in m

    def test_f_zero_recorded_contract_is_SUPERSEDED_by_w5_2(self):
        """**SUPERSEDED (v5.32, audit W5-2) -- do not restore.**

        W4-2 measured ``f = 0`` and deliberately left it alone, recording
        the split contract here so the next reader would not have to
        re-derive it:

            "``f = 0`` raises ``ZeroDivisionError`` under 'paraxial' but is a
            SILENT no-op under 'nonparaxial' / 'aplanatic'.  Deliberately
            left alone -- W4-2 is scoped to NON-FINITE f, and no caller in
            the repo passes ``f=0``.  If this contract is unified later,
            this test is the one to update."

        This is that update.  W5-2 took the decision the recorded note
        asked for: ``f = 0`` is physically meaningless for a thin lens (an
        infinite-power surface with no realisation), so it now raises
        ``ValueError`` from the same guard as non-finite ``f``, above the
        ``lens_model`` dispatch.

        The full pre-W5-2 measurement, for the record -- W4-2 named three
        models but ``apply_thin_lens`` has FIVE, and they split THREE ways
        (N = 16, dx = 5 um, lambda = 1.55 um, f = 0.0)::

            lens_model    pre-W5-2 behaviour on f = 0
            paraxial      ZeroDivisionError('division by zero')   <- k/(2f)
            nonparaxial   silent NO-OP, exactly E_in back, no warning
            aplanatic     silent NO-OP, exactly E_in back, + 2 bare
                          numpy RuntimeWarnings
            stigmatic     ZeroDivisionError                       <- 1.0/f
            local_only    ZeroDivisionError

        ``apply_cylindrical_lens`` has no ``lens_model`` knob and raised
        ``ZeroDivisionError`` on every path, so it had no silent arm to
        unify away -- only a bare exception naming neither the function
        nor the argument.  Both now raise ``ValueError``.

        The live contract is pinned in
        :class:`TestW52ZeroFocalLength` below; this test remains only as
        the supersession record, and asserts the OLD behaviour is gone.
        """
        # The two silent-no-op arms are the ones that mattered: a
        # meaningless f used to return the caller's field untouched.
        for lm in ('nonparaxial', 'aplanatic'):
            with warnings.catch_warnings():
                warnings.simplefilter('error')     # no bare numpy warnings
                with pytest.raises(ValueError,
                                   match='must not be zero'):
                    apply_thin_lens(self._E, f=0.0, wavelength=_WL, dx=_DX,
                                    lens_model=lm)
        # ...and the ZeroDivisionError arms no longer leak a bare builtin.
        for lm in ('paraxial', 'stigmatic', 'local_only'):
            with pytest.raises(ValueError):
                apply_thin_lens(self._E, f=0.0, wavelength=_WL, dx=_DX,
                                lens_model=lm)
            try:
                apply_thin_lens(self._E, f=0.0, wavelength=_WL, dx=_DX,
                                lens_model=lm)
            except ZeroDivisionError:               # pragma: no cover
                pytest.fail(f"f=0 under {lm} still raises the bare "
                            f"ZeroDivisionError W5-2 replaced")
            except ValueError:
                pass


# ===========================================================================
# W5-2 -- f == 0 in apply_thin_lens / apply_cylindrical_lens
#         (the decision W4-2 recorded but did not take)
# ===========================================================================

class TestW52ZeroFocalLength:
    """``f`` must not be zero, and all five ``lens_model`` branches must
    say so identically.

    Decision (v5.32): ``f = 0`` is physically meaningless for a thin or
    cylindrical lens -- zero focal length is an infinite-power surface with
    no physical realisation -- so it is rejected rather than given a
    consistent *value* semantics.  There is no defensible value: 'no lens'
    is ``f = inf`` (expressed by omitting the call), and the two sqrt
    models' accidental no-op was the ``f = inf`` answer returned for the
    ``f = 0`` question.

    Pre-fix the five branches split THREE ways (bare ZeroDivisionError /
    silent no-op / silent no-op plus two anonymous numpy RuntimeWarnings);
    see ``TestW42NonFiniteFocalLength::
    test_f_zero_recorded_contract_is_SUPERSEDED_by_w5_2`` for the table.
    """

    _E = np.ones((16, 16), dtype=complex)
    _MODELS = ('paraxial', 'nonparaxial', 'aplanatic', 'stigmatic',
               'local_only')

    @pytest.mark.parametrize('lens_model', _MODELS)
    @pytest.mark.parametrize('f', [0.0, -0.0, 0])
    def test_every_model_raises_valueerror_on_zero_f(self, lens_model, f):
        """The headline: no branch may quietly accept ``f = 0`` any more.

        ``-0.0`` and the ``int`` ``0`` are covered too -- ``float(f) == 0.0``
        is True for all three, and a caller who computes ``f`` from a
        difference can easily land on negative zero.
        """
        with pytest.raises(ValueError, match='must not be zero'):
            apply_thin_lens(self._E, f=f, wavelength=_WL, dx=_DX,
                            lens_model=lens_model)

    def test_all_five_models_give_the_IDENTICAL_message(self):
        """Consistency is the actual fix, so it is pinned directly: the
        guard sits above the ``lens_model`` dispatch, therefore the message
        cannot vary by model.  A future refactor that pushed the check down
        into each branch would still pass the test above while re-opening
        the divergence this closed."""
        msgs = set()
        for lm in self._MODELS:
            with pytest.raises(ValueError) as ei:
                apply_thin_lens(self._E, f=0.0, wavelength=_WL, dx=_DX,
                                lens_model=lm)
            msgs.add(str(ei.value))
        assert len(msgs) == 1, (
            f"f=0 must produce ONE message across all five lens_model "
            f"branches; got {len(msgs)}:\n" + "\n---\n".join(sorted(msgs)))

    def test_no_bare_numpy_warning_survives(self):
        """'aplanatic' used to emit two anonymous numpy RuntimeWarnings
        (invalid value / divide-by-zero from ``r_sq / f**2``) on its way to
        returning the field unchanged.  The guard fires first, so nothing
        reaches numpy."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with pytest.raises(ValueError):
                apply_thin_lens(self._E, f=0.0, wavelength=_WL, dx=_DX,
                                lens_model='aplanatic')

    def test_message_names_the_function_argument_and_the_old_split(self):
        """CONVENTIONS.md: the message carries the ``fn_name: `` prefix and
        the offending value.  It also names the pre-fix divergence, because
        a bare ZeroDivisionError naming neither the function nor the
        argument is what made this hard to attribute."""
        with pytest.raises(ValueError) as ei:
            apply_thin_lens(self._E, f=0.0, wavelength=_WL, dx=_DX)
        msg = str(ei.value)
        assert msg.startswith('apply_thin_lens: ')
        assert 'f=0.0' in msg
        assert 'NO lens' in msg and 'ZeroDivisionError' in msg
        assert 'omit the call' in msg

    @pytest.mark.parametrize('axis', ['x', 'y'])
    @pytest.mark.parametrize('f', [0.0, -0.0, 0])
    def test_cylindrical_sibling_rejects_zero_f(self, axis, f):
        """``apply_cylindrical_lens`` gets the matching guard in the same
        change -- the W3-T4 sweep exists precisely because an earlier guard
        fixed one of a pair and left the sibling to drift."""
        from lumenairy.elements import apply_cylindrical_lens
        with pytest.raises(ValueError, match='must not be zero'):
            apply_cylindrical_lens(self._E, f=f, wavelength=_WL, dx=_DX,
                                   axis=axis)

    def test_both_zero_f_guards_share_one_message_shape(self):
        """Same house-style pin the non-finite guards carry: a user who
        hits one and greps for the other finds the same wording."""
        from lumenairy.elements import apply_cylindrical_lens
        for fn in (apply_thin_lens, apply_cylindrical_lens):
            with pytest.raises(ValueError) as ei:
                fn(self._E, f=0.0, wavelength=_WL, dx=_DX)
            m = str(ei.value)
            assert 'must be a finite focal length in metres' in m
            assert 'must not be zero' in m
            assert 'f=0 is not a lens' in m
            assert 'omit the call' in m

    def test_jonesfield_rejects_zero_f_without_mutating(self):
        """``JonesField.apply_thin_lens`` routes both components through
        the guarded scalar entry point and mutates itself in place, so the
        f=0 guard must protect the caller's object exactly as the
        non-finite guard does."""
        jf = JonesField(self._E.copy(), self._E.copy(), _DX)
        with pytest.raises(ValueError, match='must not be zero'):
            jf.apply_thin_lens(f=0.0, wavelength=_WL)
        assert np.array_equal(jf.Ex, self._E)
        assert np.array_equal(jf.Ey, self._E)
        assert np.isfinite(degree_of_polarization(jf)).all()

    @pytest.mark.parametrize('f', [1e-9, -1e-9, 1e-3, 0.03, -0.03])
    @pytest.mark.parametrize('lens_model', _MODELS)
    def test_nearby_small_f_is_untouched(self, f, lens_model):
        """Scope pin: ONLY exact zero is rejected.

        The guard is ``float(f) == 0.0``, not a tolerance, so focal lengths
        arbitrarily close to zero still compute.  ``1e-9`` is nine orders
        below the grid pitch -- far into "absurd but legal" territory --
        and must come back a finite unit-modulus phase screen.
        """
        out = apply_thin_lens(self._E, f=f, wavelength=_WL, dx=_DX,
                              lens_model=lens_model)
        assert out.shape == self._E.shape
        assert np.isfinite(out).all(), (
            f"f={f} under {lens_model} must still produce a finite field")
        assert np.allclose(np.abs(out), 1.0, atol=1e-12), (
            "a thin lens is a pure phase screen on a unit field")

    @pytest.mark.parametrize('lens_model', _MODELS)
    def test_smallest_denormal_f_is_accepted_not_rejected(self, lens_model):
        """The boundary case, kept separate because its OUTPUT is garbage
        while its ACCEPTANCE is the point.

        ``5e-324`` is the smallest positive double.  ``k / (2f)`` overflows
        and the returned field is NaN-poisoned -- but that is pre-existing
        floating-point behaviour shared with every other absurd-but-finite
        input, and it is emphatically NOT this guard's business.  Pinning
        acceptance here stops a future reader from "hardening" the guard
        into ``abs(f) < eps``, which would start silently rejecting inputs
        the library has always accepted.  Warnings are suppressed because
        the overflow is expected, not because it is hidden.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            out = apply_thin_lens(self._E, f=5e-324, wavelength=_WL, dx=_DX,
                                  lens_model=lens_model)
        assert out.shape == self._E.shape        # accepted, did not raise

    def test_small_f_paraxial_still_matches_closed_form_exactly(self):
        """Guard-only change: the accepted-path phase is bit-for-bit the
        closed form, right down to a focal length nine orders below the
        grid pitch."""
        N, f = 16, 1e-9
        E = np.ones((N, N), dtype=complex)
        out = apply_thin_lens(E, f=f, wavelength=_WL, dx=_DX)
        x = (np.arange(N) - N / 2) * _DX
        X, Y = np.meshgrid(x, x)
        want = np.exp(-1j * (2 * np.pi / _WL) / (2 * f) * (X ** 2 + Y ** 2))
        assert np.array_equal(out, want)

    @pytest.mark.parametrize('f', [np.nan, np.inf, -np.inf])
    def test_non_finite_f_contract_unchanged(self, f):
        """W4-2's guard is extended, not replaced: ``f = nan / +-inf`` still
        raises, still names the per-model divergence, and still carries the
        substrings the W4-2 pins match on."""
        from lumenairy.elements import apply_cylindrical_lens
        for fn in (apply_thin_lens, apply_cylindrical_lens):
            with pytest.raises(ValueError) as ei:
                fn(self._E, f=f, wavelength=_WL, dx=_DX)
            m = str(ei.value)
            assert 'must be a finite focal length in metres' in m
            assert 'omit the call' in m
            # the f=0-specific explanation must NOT leak onto this path
            assert 'f=0 is not a lens' not in m


# ===========================================================================
# W4-3 -- 1-D RCWAStack y-averaging a y-VARYING 2-D cell
# ===========================================================================

def _pillar(size=24, n_lo=1.5, n_hi=2.5):
    """A y-VARYING 2-D cell (square pillar) -- the trap case.  Same geometry
    as ``test_audit_s1_2_rcwa_lossless_tripwire._iso_cell``."""
    c = np.full((size, size), n_lo ** 2 + 0j)
    q = size // 4
    c[q:-q, q:-q] = n_hi ** 2
    return c


def _stripe(size=24, n_lo=1.5, n_hi=2.5):
    """A y-INVARIANT 2-D cell (x-only grating on a 2-D grid) -- the LEGITIMATE
    case, for which the y-average is the structure itself."""
    c = np.full((size, size), n_lo ** 2 + 0j)
    q = size // 4
    c[q:-q, :] = n_hi ** 2
    return c


def _uniform_tensor(size=24, eps=2.25):
    return (eps + 0j) * np.broadcast_to(
        np.eye(3, dtype=complex), (size, size, 3, 3)).copy()


def _stack(*, noy=None, n_orders=5):
    kw = dict(period=1.0e-6, n_superstrate=1.0, n_substrate=1.5,
              n_orders=n_orders)
    if noy is not None:
        kw.update(period_y=1.0e-6, n_orders_y=noy)
    return RCWAStack(**kw)


def _categories(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        fn()
    return [w.category for w in rec]


class TestW43OneDStackYAverage:
    """A 1-D stack fed a y-VARYING 2-D cell WARNS; a y-INVARIANT one does not.

    Pre-fix measurement (24x24 pillar cell, period 1 um, lambda = 633 nm,
    theta = 0.1 rad, n_orders = 5): the 1-D solve returned
    ``sum(R) = 0.244615594382``, BIT-identical (``array_equal`` on both R and
    T) to solving the explicitly y-averaged cell, against the y-resolved 2-D
    stack's ``0.183045443696`` -- a 33.64% error -- with the energy closure
    equally clean in both, so no tripwire could see it.
    """

    def test_y_varying_cell_on_1d_stack_warns(self):
        cats = _categories(
            lambda: _stack().add_layer(0.10e-6, eps_cell=_pillar()))
        assert RCWAYAverageWarning in cats, (
            f"a y-VARYING 2-D cell on a 1-D stack must warn; got {cats}")

    def test_warning_names_the_averaging_and_the_fix(self):
        with pytest.warns(RCWAYAverageWarning) as rec:
            _stack().add_layer(0.10e-6, eps_cell=_pillar())
        msg = str(rec[0].message)
        assert 'y-AVERAGED' in msg, (
            f"the warning must name the averaging; got {msg!r}")
        assert 'n_orders_y' in msg and 'period_y' in msg, (
            f"the warning must name the fix (build a 2-D stack); got {msg!r}")

    def test_y_invariant_2d_cell_on_1d_stack_is_silent(self):
        """The LEGITIMATE idiom: an x-only grating written on a 2-D grid.  The
        y-average IS the structure, so warning here would be noise -- and
        would punish exactly the pattern the existing pins use."""
        cats = _categories(
            lambda: _stack().add_layer(0.10e-6, eps_cell=_stripe()))
        assert RCWAYAverageWarning not in cats, (
            f"a y-INVARIANT 2-D cell loses nothing to the y-average; got "
            f"{cats}")

    def test_true_1d_cell_is_silent(self):
        cats = _categories(
            lambda: _stack().add_layer(0.10e-6, eps_cell=_stripe()[:, :1]))
        assert RCWAYAverageWarning not in cats
        cats = _categories(
            lambda: _stack().add_layer(0.10e-6, eps_cell=_stripe()[:, 0]))
        assert RCWAYAverageWarning not in cats, (
            "a 1-D cell is promoted to (S, 1) internally -- still nothing "
            f"averaged away; got {cats}")

    def test_explicit_2d_stack_does_not_warn(self):
        """A 2-D stack RESOLVES y, so there is no averaging to report."""
        cats = _categories(
            lambda: _stack(noy=5).add_layer(0.10e-6, eps_cell=_pillar()))
        assert RCWAYAverageWarning not in cats, (
            f"an n_orders_y>=1 stack resolves y; got {cats}")

    def test_tensor_cell_path_covered(self):
        """The ``eps_tensor_cell`` branch has its own validation call and its
        own y-average; both must behave like the iso branch."""
        uniform = _uniform_tensor()
        cats = _categories(
            lambda: _stack().add_layer(0.08e-6, eps_tensor_cell=uniform))
        assert RCWAYAverageWarning not in cats, (
            f"a uniform tensor cell is y-invariant; got {cats}")
        varying = uniform.copy()
        varying[:, :12, 0, 0] = 4.0 + 0j
        cats = _categories(
            lambda: _stack().add_layer(0.08e-6, eps_tensor_cell=varying))
        assert RCWAYAverageWarning in cats, (
            f"a y-VARYING tensor cell on a 1-D stack must warn; got {cats}")

    def test_imaginary_only_variation_detected(self):
        """The variance test sums the real AND imaginary spreads, so a cell
        that varies along y only in its LOSS still warns."""
        c = np.full((24, 24), 2.25 + 0j)
        c[:, :12] += 0.1j
        cats = _categories(lambda: _stack().add_layer(0.10e-6, eps_cell=c))
        assert RCWAYAverageWarning in cats, (
            f"y-varying Im(eps) is still a y-varying structure; got {cats}")

    def test_warning_is_a_diagnostic_not_a_rejection(self):
        """The 1-D + 2-D-cell contract is UNCHANGED -- the solve still runs and
        still returns the y-averaged answer, bit-identical to pre-averaging the
        cell by hand.  That is what keeps
        ``test_audit_s1_2_rcwa_lossless_tripwire`` (which feeds 2-D pillar
        cells to 1-D stacks) green."""
        cell = _pillar()
        yavg = np.repeat(cell.mean(axis=1, keepdims=True), cell.shape[1],
                         axis=1)

        def _solve(c):
            st = _stack()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RCWAYAverageWarning)
                st.add_layer(0.10e-6, eps_cell=c)
                r = st.set_source(0.633e-6, theta=0.1).solve()
            return np.asarray(r._R), np.asarray(r._T)

        R_raw, T_raw = _solve(cell)
        R_avg, T_avg = _solve(yavg)
        np.testing.assert_array_equal(R_raw, R_avg)
        np.testing.assert_array_equal(T_raw, T_avg)

    def test_warning_can_be_silenced_and_promoted_by_category(self):
        """Its own category (not bare ``UserWarning``) so a caller can filter
        exactly this without muting the sibling physics diagnostics."""
        assert issubclass(RCWAYAverageWarning, UserWarning)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RCWAYAverageWarning)
            with pytest.raises(RCWAYAverageWarning):
                _stack().add_layer(0.10e-6, eps_cell=_pillar())
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            warnings.simplefilter('ignore', RCWAYAverageWarning)
            _stack().add_layer(0.10e-6, eps_cell=_pillar())
        assert not [w for w in rec
                    if w.category is RCWAYAverageWarning]

    def test_solve_vs_wavelength_dispersive_path_covered(self):
        """A DISPERSIVE (callable) cell is materialised per wavelength inside
        ``_materialized_layers``, a second validation site that also needed the
        diagnostic."""
        st = _stack()
        st.add_layer(0.10e-6, eps_cell=lambda wl: _pillar())
        st.set_source(0.633e-6, theta=0.1)
        with pytest.warns(RCWAYAverageWarning):
            st.solve_vs_wavelength([0.633e-6, 0.640e-6])

    def test_solve_vs_wavelength_y_invariant_dispersive_is_silent(self):
        st = _stack()
        st.add_layer(0.10e-6, eps_cell=lambda wl: _stripe())
        st.set_source(0.633e-6, theta=0.1)
        cats = _categories(
            lambda: st.solve_vs_wavelength([0.633e-6, 0.640e-6]))
        assert RCWAYAverageWarning not in cats, (
            f"a y-invariant dispersive cell must stay silent; got {cats}")

    # ---- no drift between the warning and the M8 strict_y raise ----------

    @pytest.mark.parametrize('cell,y_varying', [
        (_pillar(), True),
        (_stripe(), False),
        (np.full((24, 24), 2.25 + 0j), False),
        (_uniform_tensor(), False),
    ])
    def test_warning_verdict_matches_the_strict_y_raise(self, cell, y_varying):
        """``_warn_if_y_averaged`` re-implements the M8 ``strict_y`` variance
        test (per-component ptp along y, real + imag).  Pin that the two agree,
        so a future edit to one cannot silently diverge from the other: the
        stack WARNS exactly when the 2-D entry points would RAISE."""
        raised = False
        try:
            _validate_cell_sampling('probe', cell, 5, 0, strict_y=True)
        except ValueError as e:
            raised = 'y-INVARIANT' in str(e)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            warned = _warn_if_y_averaged('probe', cell, 0, strict_y=False)
        assert warned is bool(warned)
        assert warned == y_varying, (
            f"expected warn={y_varying} for this cell; got {warned}")
        assert raised == warned, (
            "the 1-D warning and the 2-D strict_y raise must reach the SAME "
            f"verdict about y-variance; raise={raised} warn={warned}")
        assert len([w for w in rec
                    if w.category is RCWAYAverageWarning]) == int(warned)

    def test_helper_noops_when_strict_y(self):
        """When ``strict_y`` is on, ``_validate_cell_sampling`` already raised;
        the helper must not double-report."""
        assert _warn_if_y_averaged('probe', _pillar(), 0,
                                   strict_y=True) is False
        assert _warn_if_y_averaged('probe', _pillar(), 5,
                                   strict_y=False) is False


# ===========================================================================
# W4-4 / R-18 -- dead jax_trace parameters (VERIFIED, removed)
# ===========================================================================

class TestW44JaxTraceDeadParams:
    """The R-18 dead-code claim about ``jax_trace`` is CONFIRMED and closed.

    Verified before removal by AST (every ``ast.Name`` load inside each
    function body, nested closures included) plus a repo-wide pure-Python
    grep for callers across ``lumenairy/``, ``tests/`` and ``validation/``:

    * ``_trace_body_static(state, jp, wavelength, surface_diffraction)`` --
      ``surface_diffraction`` UNUSED (the DOE kicks come from ``jp.aux[-1]``);
    * ``_trace_body_traced(...)`` -- likewise UNUSED;
    * ``_make_jit_kernel(jp_aux, wavelength_float, surface_diffraction)`` --
      ``jp_aux`` UNUSED (the cache key is built by ``trace_jax``), and its
      ``surface_diffraction`` only fed the ignoring body.

    The only readers were the four internal call sites in ``jax_trace`` itself;
    ``raytrace/trace.py`` and ``tests/unit/test_audit_raytrace.py`` mention the
    body names in PROSE only.  All three parameters are gone.  This class pins
    the new signatures so the vestige cannot creep back, and pins that DOE
    kicks still work (they were never carried by those parameters).
    """

    def test_trace_body_signatures_have_no_dead_params(self):
        from lumenairy.raytrace import jax_trace as jt
        for name, want in (('_trace_body_static', ['state', 'jp',
                                                   'wavelength']),
                           ('_trace_body_traced', ['state', 'jp',
                                                   'wavelength']),
                           ('_make_jit_kernel', ['wavelength_float'])):
            got = list(inspect.signature(getattr(jt, name)).parameters)
            assert got == want, (
                f"{name} signature drifted: expected {want}, got {got}.  The "
                f"R-18 dead parameters (surface_diffraction / jp_aux) must "
                f"stay removed -- a parameter the body ignores is a lie about "
                f"what the function honours.")

    def test_no_stale_call_sites(self):
        """A stale 4-argument call would only TypeError on the branch that
        reaches it, so assert statically over the whole module.  ``jp.aux`` must
        no longer be handed to ``_make_jit_kernel``, and neither trace body may
        be called with a 4th argument.  (``_build_jax_prescription`` still takes
        ``surface_diffraction`` -- that one is LIVE, it builds ``diff_aux`` --
        so the patterns below are anchored on the body names.)"""
        import re

        import lumenairy.raytrace.jax_trace as jt
        src = inspect.getsource(jt)
        for body in ('_trace_body_static', '_trace_body_traced'):
            for call in re.finditer(
                    body + r'\(\s*([^)]*)\)', src):
                args = [a.strip() for a in call.group(1).split(',')]
                assert len(args) == 3, (
                    f"{body} must be called with exactly 3 arguments; found "
                    f"{args} -- a resurrected surface_diffraction argument "
                    f"would be silently ignored by the body.")
        for call in re.finditer(r'_make_jit_kernel\(\s*([^)]*\))', src):
            inner = call.group(1)
            assert 'jp.aux' not in inner and 'surface_diffraction' not in inner, (
                f"_make_jit_kernel must be called with the wavelength only; "
                f"got {inner!r}")

    def test_doe_kick_still_applied_through_jp_aux(self):
        """The kicks were always read from ``jp.aux``; prove the removal did
        not take the feature with it."""
        pytest.importorskip('jax')
        import jax.numpy as jnp

        from lumenairy.raytrace.jax_trace import JaxRayState, clear_trace_jax_cache, trace_jax
        presc = {'thicknesses': [],
                 'surfaces': [{'radius': float('inf'), 'glass_before': 'air',
                               'glass_after': 'air'}]}
        n = 5
        st = JaxRayState(
            x=jnp.linspace(-1e-4, 1e-4, n), y=jnp.zeros(n), z=jnp.zeros(n),
            L=jnp.zeros(n), M=jnp.zeros(n), N=jnp.ones(n),
            opd=jnp.zeros(n), alive=jnp.ones(n, dtype=bool))
        period, m, wl = 2e-6, 1.0, 633e-9
        clear_trace_jax_cache()
        out = trace_jax(st, presc, wl,
                        surface_diffraction={0: (m, 0.0, period,
                                                 float('inf'))})
        # grating equation: the x direction cosine picks up m*lambda/period
        want = m * wl / period
        assert np.allclose(np.asarray(out.L), want, rtol=1e-9, atol=1e-12), (
            f"the DOE kick must still land; expected L={want}, got "
            f"{np.asarray(out.L)}")
        clear_trace_jax_cache()
        plain = trace_jax(st, presc, wl)
        assert np.allclose(np.asarray(plain.L), 0.0), (
            "no surface_diffraction -> no kick")

    def test_prebuilt_prescription_still_rejects_surface_diffraction(self):
        """RT-8's raise is the reason the parameter could be dropped: this path
        CANNOT honour a late spec, so it refuses instead of ignoring."""
        pytest.importorskip('jax')
        import jax.numpy as jnp

        from lumenairy.raytrace.jax_trace import JaxRayState, _build_jax_prescription, trace_jax
        presc = {'thicknesses': [],
                 'surfaces': [{'radius': float('inf'), 'glass_before': 'air',
                               'glass_after': 'air'}]}
        jp = _build_jax_prescription(presc, 633e-9)
        n = 3
        st = JaxRayState(
            x=jnp.zeros(n), y=jnp.zeros(n), z=jnp.zeros(n),
            L=jnp.zeros(n), M=jnp.zeros(n), N=jnp.ones(n),
            opd=jnp.zeros(n), alive=jnp.ones(n, dtype=bool))
        with pytest.raises(ValueError, match='surface_diffraction'):
            trace_jax(st, jp, 633e-9,
                      surface_diffraction={0: (1.0, 0.0, 2e-6,
                                                float('inf'))})
