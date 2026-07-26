"""Regression pins for the Tier-2 wave of
``docs/audits/AUDIT_ADVERSARIAL_CODEBASE_2026_07_25.md`` (Territories P,
A and the two elements findings fixed with them).  Every pin below was
verified to FAIL on the pre-fix tree (git worktree at ``5c9f7c3``).

* **P3** ``hf.py`` -- Van-Vleck cross-Hessian ``finite_diff_step``
  default was 1e-9 m, essentially pure round-off: 9.05% low density
  amplitude at the origin, up to 1.56e-2 end-to-end amplitude error vs
  exact Fresnel quadrature.  Default is now 1e-6 m.
* **P4** ``vector_diffraction.py`` -- no guard that the pupil ARRAY
  spans the ``f*NA`` rim; when it does not the exit pupil silently
  degenerates to the square array boundary at the array-limited NA
  (measured 5.5x PSF-width error at NA_eff=0.16 vs a requested 0.9,
  zero warnings).  Now a loud, actionable ``RuntimeWarning``;
  values unchanged.
* **P6** ``system.py`` -- ``propagate_through_system_jax(method=...)``
  was never read: every value (including junk) returned the ASM field
  while the NumPy twin honours ``'fresnel'`` / ``'sas'``.  Now raises.
* **P7** ``hf.py`` -- ``…with_opl_callable(wavelength=)`` was a REQUIRED
  keyword the body never read, and the ``opl_fn`` units contract (WAVES,
  not metres -- a factor ~1e6) was documented nowhere.  Now optional +
  deprecated, contract documented.
* **E-H6** ``coatings.py`` -- ``broadband_ar_v_coat(n_substrate, ...)``
  never read ``n_substrate``: the fixed n_H=2.3 / n_L=1.38 stack is
  matched only to a substrate of 2.778 and measured WORSE THAN BARE
  GLASS on every common substrate.  Now the quarter-wave admittance
  match ``n_H = n_L*sqrt(n_substrate)``.
* **E-H11** ``doe.py`` -- ``makedammann2d(_legacy_units='auto')`` default
  silently multiplied any period/wavelength above 1 mm by 1e-6 (SI THz
  design -> 5e-10 m cells) behind a default-suppressed
  ``DeprecationWarning``.  Default is now ``'SI'``; the shim is retired
  behind an explicit opt-in with a loud ``UserWarning``.
* **A-3** ``__init__.py`` -- the four top-level ``DEFAULT_*`` constants
  were import-time snapshots that the setters did not move.  Now
  PEP-562 live-forwarded (``propagation.py:296`` precedent).
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators.hf import (
    propagate_huygens_fresnel_with_opl_callable as _hf_opl,
)

# ==========================================================================
# P3 -- Van-Vleck finite-difference step default
# ==========================================================================

_LAM = 1.0e-6
_Z = 0.05


def _fresnel_opl_waves(s1x, s1y, s2x, s2y):
    """Exact-quadratic (Fresnel) OPL in WAVES -- the oracle: its
    cross-Hessian is the constant ``diag(-1/(lam z))`` so
    ``|det| = 1/(lam z)**2`` exactly, and the HF integral collapses to
    the ``1/(i lam z)`` Fresnel quadrature."""
    return (_Z + ((s1x - s2x) ** 2 + (s1y - s2y) ** 2) / (2 * _Z)) / _LAM


def _vv_amplitude_error(h):
    """Relative error of the finite-difference Van Vleck density
    amplitude ``sqrt|det d2Phi/ds1ds2|`` at the origin, for step ``h``."""
    x1 = np.array([[0.0]])
    y1 = np.array([[0.0]])
    s2x = s2y = 0.0
    f = _fresnel_opl_waves
    pxx = (f(x1 + h, y1, s2x + h, s2y) - f(x1 + h, y1, s2x - h, s2y)
           - f(x1 - h, y1, s2x + h, s2y) + f(x1 - h, y1, s2x - h, s2y)
           ) / (4 * h * h)
    pyy = (f(x1, y1 + h, s2x, s2y + h) - f(x1, y1 + h, s2x, s2y - h)
           - f(x1, y1 - h, s2x, s2y + h) + f(x1, y1 - h, s2x, s2y - h)
           ) / (4 * h * h)
    pxy = (f(x1 + h, y1, s2x, s2y + h) - f(x1 + h, y1, s2x, s2y - h)
           - f(x1 - h, y1, s2x, s2y + h) + f(x1 - h, y1, s2x, s2y - h)
           ) / (4 * h * h)
    pyx = (f(x1, y1 + h, s2x + h, s2y) - f(x1, y1 + h, s2x - h, s2y)
           - f(x1, y1 - h, s2x + h, s2y) + f(x1, y1 - h, s2x - h, s2y)
           ) / (4 * h * h)
    det = float((pxx * pyy - pxy * pyx).ravel()[0])
    exact = (1.0 / (_LAM * _Z)) ** 2
    return np.sqrt(abs(det)) / np.sqrt(exact) - 1.0


def _hf_vs_exact_fresnel(**kwargs):
    """|out/ref| for the HF OPL-callable against exact Fresnel
    quadrature on the SAME discretisation (so the only error source is
    the Van Vleck density factor)."""
    N, dx = 32, 4e-6
    xi = (np.arange(N) - N / 2) * dx
    E_in = np.ones((N, N), dtype=np.complex128)
    ox = np.array([0.0, 5e-6, 20e-6])
    oy = np.array([0.0])
    X1, Y1 = np.meshgrid(xi, xi, indexing='xy')
    k = 2 * np.pi / _LAM
    ref = np.array([
        np.sum(E_in * np.exp(1j * k * (_Z + ((X1 - sx) ** 2 + Y1 ** 2)
                                       / (2 * _Z)))) * dx ** 2
        / (1j * _LAM * _Z)
        for sx in ox])
    out = _hf_opl(E_in, opl_fn=_fresnel_opl_waves,
                  output_grid_x=ox, output_grid_y=oy,
                  input_grid_dx=dx, apply_van_vleck=True, **kwargs)
    return np.abs(np.asarray(out).ravel() / ref)


class TestP3VanVleckFiniteDiffStep:
    """P3: the shipped default step must resolve the cross-Hessian, not
    the round-off floor."""

    def test_default_step_is_not_the_roundoff_floor(self):
        h = inspect.signature(_hf_opl).parameters['finite_diff_step'].default
        assert h == pytest.approx(1e-6, rel=1e-12), (
            f"finite_diff_step default is {h!r}; the Van Vleck "
            f"cross-Hessian stencil needs ~1e-6 m.  The pre-v5.30 1e-9 "
            f"default was almost pure round-off (9.05% amplitude error).")

    def test_default_step_amplitude_error_below_1e6_on_exact_oracle(self):
        err = abs(_vv_amplitude_error(
            inspect.signature(_hf_opl)
            .parameters['finite_diff_step'].default))
        assert err < 1e-6, (
            f"Van Vleck density amplitude is off by {err:.3e} at the "
            f"DEFAULT finite_diff_step on an exact-quadratic OPL whose "
            f"cross-Hessian determinant is analytic; expected < 1e-6.")

    def test_old_default_step_is_rejected_as_all_roundoff(self):
        """Counter-pin: the pre-fix default must still measure ~9% wrong,
        so this file is measuring the step (not an unrelated change)."""
        err = abs(_vv_amplitude_error(1e-9))
        assert err > 1e-2, (
            f"h=1e-9 amplitude error is {err:.3e}; the P3 measurement "
            f"(9.05e-2) no longer reproduces -- if the stencil changed, "
            f"re-derive the default rather than loosening this pin.")

    def test_end_to_end_matches_exact_fresnel_quadrature_at_default(self):
        r = _hf_vs_exact_fresnel()
        assert np.max(np.abs(r - 1.0)) < 1e-6, (
            f"HF OPL-callable amplitude vs exact Fresnel quadrature: "
            f"max|err|={np.max(np.abs(r - 1.0)):.3e} at the default step "
            f"(pre-fix 1.56e-2).")

    def test_end_to_end_old_default_still_reproduces_the_defect(self):
        r = _hf_vs_exact_fresnel(finite_diff_step=1e-9)
        assert np.max(np.abs(r - 1.0)) > 1e-3, (
            "Explicit finite_diff_step=1e-9 no longer reproduces the "
            "measured 1.56e-2 end-to-end error; the counter-pin has "
            "stopped discriminating.")


# ==========================================================================
# P7 -- OPL-callable units contract + inert ``wavelength``
# ==========================================================================

class TestP7OplCallableUnitsContract:

    def test_docstring_states_the_waves_units_contract(self):
        doc = inspect.getdoc(_hf_opl) or ''
        assert 'WAVES' in doc, (
            "propagate_huygens_fresnel_with_opl_callable must document "
            "IN THE DOCSTRING that opl_fn returns the optical path in "
            "WAVES -- a metres-valued callable is silently wrong by "
            "~1/wavelength (measured 1.0e-6 of the correct field).")
        assert 'metres' in doc or 'metre' in doc, (
            "The units contract must contrast waves with metres so the "
            "~1e6 error mode is discoverable.")

    def test_wavelength_is_optional_and_call_succeeds_without_it(self):
        p = inspect.signature(_hf_opl).parameters['wavelength']
        assert p.default is None, (
            f"wavelength default is {p.default!r}; it must be optional "
            f"(the body never read it -- audit P7).")
        with warnings.catch_warnings():
            warnings.simplefilter('error')          # no warning at all
            out = _hf_opl(
                np.ones((8, 8), dtype=np.complex128),
                opl_fn=_fresnel_opl_waves,
                output_grid_x=np.array([0.0]),
                output_grid_y=np.array([0.0]),
                input_grid_dx=4e-6)
        assert np.isfinite(np.asarray(out)).all()

    def test_passing_wavelength_warns_and_does_not_change_the_result(self):
        kw = dict(opl_fn=_fresnel_opl_waves,
                  output_grid_x=np.array([0.0, 5e-6]),
                  output_grid_y=np.array([0.0]),
                  input_grid_dx=4e-6)
        E = np.ones((8, 8), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out_bare = np.asarray(_hf_opl(E, **kw))
        with pytest.warns(DeprecationWarning, match='wavelength'):
            out_wl = np.asarray(_hf_opl(E, wavelength=_LAM, **kw))
        assert np.array_equal(out_bare, out_wl), (
            "Deprecating ``wavelength`` must not change the numbers: the "
            "parameter was inert and stays inert.")

    def test_metres_valued_callable_is_the_documented_error_mode(self):
        """Discriminator for the contract: a metres-returning callable
        lands ~1e-6 of the correct field, which is exactly why the
        docstring must state WAVES."""
        kw = dict(output_grid_x=np.array([0.0]),
                  output_grid_y=np.array([0.0]),
                  input_grid_dx=4e-6)
        E = np.ones((16, 16), dtype=np.complex128)
        good = np.asarray(_hf_opl(E, opl_fn=_fresnel_opl_waves, **kw))
        bad = np.asarray(_hf_opl(
            E, opl_fn=lambda a, b, c, d: _fresnel_opl_waves(a, b, c, d) * _LAM,
            **kw))
        ratio = abs(complex(bad.ravel()[0]) / complex(good.ravel()[0]))
        assert ratio < 1e-4, (
            f"metres-vs-waves ratio {ratio:.3e}: the ~1e-6 error mode the "
            f"units contract warns about is not reproducing.")


# ==========================================================================
# P4 -- Richards-Wolf pupil array must span the f*NA rim
# ==========================================================================

def _fwhm(prof, dx):
    p = np.asarray(prof, dtype=float)
    p = p / p.max()
    c = int(np.argmax(p))

    def cross(idx, step):
        i = idx
        while 0 <= i + step < len(p) and p[i + step] >= 0.5:
            i += step
        if not (0 <= i + step < len(p)):
            return None
        f0, f1 = p[i], p[i + step]
        return i + step * (f0 - 0.5) / (f0 - f1)

    a, b = cross(c, -1), cross(c, +1)
    return None if a is None or b is None else (b - a) * dx


class TestP4RichardsWolfPupilSpanGuard:
    """P4: an under-spanning pupil array must say so, loudly, with the
    numbers needed to fix the call."""

    _LAM = 633e-9
    _NA = 0.9
    _F = 2e-3
    _NP = 128

    def _psf(self, dx_pupil, record=True):
        from lumenairy.propagators.vector_diffraction import debye_wolf_psf
        P = np.ones((self._NP, self._NP), dtype=complex)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            psf, xf, yf = debye_wolf_psf(
                P, self._LAM, self._NA, self._F, dx_pupil)
        return psf, xf, w

    def test_under_spanning_pupil_warns_with_actionable_numbers(self):
        # half-extent 0.320 mm vs rim f*NA = 1.800 mm -> NA_eff = 0.160.
        psf, xf, w = self._psf(5e-6)
        msgs = [str(r.message) for r in w
                if issubclass(r.category, RuntimeWarning)]
        assert len(msgs) >= 1, (
            "richards_wolf_focus silently delivered an array-limited "
            "SQUARE pupil at NA_eff=0.160 for a requested NA=0.9 "
            "(measured 5.5x PSF-width error) -- it must warn.")
        msg = next(m for m in msgs if 'does not span' in m)
        # requested NA, delivered NA_eff, and BOTH remedies by number.
        assert '0.9000' in msg, f"requested NA missing from: {msg!r}"
        assert '0.1600' in msg, f"delivered NA_eff missing from: {msg!r}"
        assert 'dx_pupil >=' in msg and 'Np >=' in msg, (
            f"the warning must state the dx_pupil / Np needed to span "
            f"the rim; got: {msg!r}")

    def test_spanning_pupil_is_silent(self):
        # half-extent 2.560 mm > rim 1.800 mm: the rim mask bites, the
        # delivered NA is the requested one, nothing to report.
        psf, xf, w = self._psf(40e-6)
        assert [str(r.message) for r in w
                if issubclass(r.category, RuntimeWarning)] == [], (
            "A pupil array that DOES span f*NA must not warn (false "
            "positives would train users to filter the guard).")

    def test_zero_padded_pupil_that_under_spans_is_silent(self):
        """No false positives: if the caller's own amplitude mask ends
        INSIDE the array, the array boundary is not the limiting aperture,
        so the guard must stay quiet even though Np*dx_pupil/2 < f*NA.
        Same gating rule as the S9-VD2 crop warning (which fires only when
        the crop discards non-zero content) -- pinned there too."""
        from lumenairy.propagators.vector_diffraction import debye_wolf_psf
        i = (np.arange(self._NP) - self._NP / 2) * 5e-6
        R = np.hypot(*np.meshgrid(i, i))
        # support radius 0.4 * (Np/2) * dx_pupil: well inside the array.
        P = (R <= 0.4 * (self._NP / 2) * 5e-6).astype(np.complex128)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            debye_wolf_psf(P, self._LAM, self._NA, self._F, 5e-6)
        assert [str(r.message) for r in w
                if issubclass(r.category, RuntimeWarning)
                and 'does not span' in str(r.message)] == [], (
            "The span guard fired on a zero-padded pupil whose support "
            "ends inside the array -- that aperture is set by the "
            "caller's mask, not by the grid boundary.")

    def test_warned_case_delivers_the_array_limited_na_it_reports(self):
        """The NA_eff number in the message is load-bearing: the focal
        FWHM must follow 0.51*lam/NA_eff, not 0.51*lam/NA."""
        psf, xf, w = self._psf(5e-6)
        dxf = float(xf[1] - xf[0])
        fw = _fwhm(psf[self._NP // 2], dxf)
        na_eff = (self._NP * 5e-6 / 2.0) / self._F
        airy_eff = 0.51 * self._LAM / na_eff
        airy_req = 0.51 * self._LAM / self._NA
        assert abs(fw / airy_eff - 1.0) < 0.15, (
            f"FWHM={fw*1e6:.4f} um vs 0.51*lam/NA_eff={airy_eff*1e6:.4f} "
            f"um: the reported NA_eff does not describe the delivered "
            f"aperture.")
        assert fw > 4.0 * airy_req, (
            f"FWHM={fw*1e6:.4f} um is not the ~5.5x-too-wide PSF the P4 "
            f"measurement recorded (0.51*lam/NA={airy_req*1e6:.4f} um).")

    def test_guard_is_diagnostic_only_and_bit_for_bit(self):
        """Same call with the warning raised vs suppressed must return
        bit-identical arrays -- the guard touches no value."""
        from lumenairy.propagators.vector_diffraction import richards_wolf_focus
        P = np.ones((self._NP, self._NP), dtype=complex)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = richards_wolf_focus(P, self._LAM, self._NA, self._F, 5e-6)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            b = richards_wolf_focus(P, self._LAM, self._NA, self._F, 5e-6)
        for ca, cb in zip(a, b):
            assert np.array_equal(np.asarray(ca), np.asarray(cb))


# ==========================================================================
# P6 -- propagate_through_system_jax must honour or reject ``method``
# ==========================================================================

jax = pytest.importorskip('jax', reason='P6 pins the JAX system walker')


class TestP6SystemJaxMethod:

    _LAM = 1.0e-6
    _DX = 1.0e-6
    _N = 64

    def _field(self):
        x = (np.arange(self._N) - self._N / 2) * self._DX
        X, Y = np.meshgrid(x, x, indexing='xy')
        return np.exp(-(X ** 2 + Y ** 2) / (12e-6) ** 2).astype(np.complex128)

    _ELEMS = [{'type': 'propagate', 'z': 2e-3}]

    def test_asm_is_supported_and_matches_the_numpy_twin(self):
        from lumenairy.propagators.system import (
            propagate_through_system,
            propagate_through_system_jax,
        )
        E0 = self._field()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E_np, _ = propagate_through_system(
                E0, self._ELEMS, self._LAM, self._DX, method='asm')
            E_jx = np.asarray(propagate_through_system_jax(
                E0, self._ELEMS, self._LAM, self._DX, method='asm'))
        rel = np.linalg.norm(E_jx - E_np) / np.linalg.norm(E_np)
        assert rel < 1e-6, f"JAX/NumPy ASM disagree at {rel:.3e}"

    @pytest.mark.parametrize('method', ['fresnel', 'sas', 'rs',
                                        'rayleigh_sommerfeld'])
    def test_numpy_only_methods_raise_not_implemented(self, method):
        """Pre-fix EVERY one of these silently returned the ASM field --
        5.0e-2 relative L2 from the NumPy twin's Fresnel answer."""
        from lumenairy.propagators.system import propagate_through_system_jax
        with pytest.raises(NotImplementedError, match=method):
            propagate_through_system_jax(
                self._field(), self._ELEMS, self._LAM, self._DX,
                method=method)

    @pytest.mark.parametrize('method', ['totally-bogus', 'ASM', 'angular',
                                        ''])
    def test_junk_method_raises_value_error(self, method):
        from lumenairy.propagators.system import propagate_through_system_jax
        with pytest.raises(ValueError, match='method'):
            propagate_through_system_jax(
                self._field(), self._ELEMS, self._LAM, self._DX,
                method=method)

    def test_docstring_matches_the_implemented_set(self):
        from lumenairy.propagators.system import propagate_through_system_jax
        doc = inspect.getdoc(propagate_through_system_jax) or ''
        assert 'ASM only' in doc or 'asm' in doc, (
            "The docstring must name the free-space method(s) this entry "
            "point actually implements.")
        assert 'NotImplementedError' in doc, (
            "The docstring must document that unsupported ``method`` "
            "values raise (silent fall-through was the defect).")


# ==========================================================================
# E-H6 -- broadband_ar_v_coat must read n_substrate
# ==========================================================================

class TestEH6VCoatReadsSubstrate:

    _WL = 550e-9
    _SUBSTRATES = (1.45, 1.52, 1.75, 2.0, 2.35, 2.78, 3.42, 4.0)

    def test_stack_depends_on_n_substrate(self):
        from lumenairy.elements.coatings import broadband_ar_v_coat
        stacks = [broadband_ar_v_coat(ns, self._WL) for ns in self._SUBSTRATES]
        assert len({tuple(s) for s in stacks}) == len(stacks), (
            "broadband_ar_v_coat returned the same stack for substrates "
            "1.45 through 4.00 -- ``n_substrate`` is inert (audit E-H6).")

    @pytest.mark.parametrize('ns', _SUBSTRATES)
    def test_reflectance_is_nulled_at_the_design_wavelength(self, ns):
        """The design condition, measured with this module's own TMM."""
        from lumenairy.elements.coatings import (
            broadband_ar_v_coat,
            coating_reflectance,
        )
        R = float(coating_reflectance(
            broadband_ar_v_coat(ns, self._WL), self._WL,
            n_substrate=ns, polarization='avg')[0])
        assert R < 1e-12, (
            f"R={R:.6e} at the design wavelength on n_substrate={ns}; the "
            f"quarter-wave admittance match n_H = n_L*sqrt(n_s) must null "
            f"it exactly.")

    @pytest.mark.parametrize('ns', _SUBSTRATES)
    def test_coating_beats_bare_substrate(self, ns):
        """Counter-pin for the pre-fix defect: the fixed n_H=2.3 stack
        measured R=0.0856 on N-BK7 against 0.0426 BARE -- an AR function
        that doubled the reflectance."""
        from lumenairy.elements.coatings import (
            broadband_ar_v_coat,
            coating_reflectance,
        )
        wl = np.linspace(500e-9, 600e-9, 21)
        R = np.asarray(coating_reflectance(
            broadband_ar_v_coat(ns, self._WL), wl,
            n_substrate=ns, polarization='avg')[0], dtype=float)
        R_bare = float(coating_reflectance(
            [], self._WL, n_substrate=ns, polarization='avg')[0])
        assert float(R.max()) < R_bare, (
            f"n_substrate={ns}: coated max R over 500-600 nm "
            f"{float(R.max()):.4f} is not below the bare-substrate "
            f"{R_bare:.4f} -- this is an AR design.")

    def test_high_index_layer_follows_the_admittance_match(self):
        from lumenairy.elements.coatings import broadband_ar_v_coat
        for ns in self._SUBSTRATES:
            (n_L, d_L), (n_H, d_H) = broadband_ar_v_coat(ns, self._WL)
            assert n_H == pytest.approx(n_L * np.sqrt(ns), rel=1e-12)
            # both layers stay quarter-wave at the design wavelength
            assert d_L == pytest.approx(self._WL / (4 * n_L), rel=1e-12)
            assert d_H == pytest.approx(self._WL / (4 * n_H), rel=1e-12)
            # ambient-side-first order preserved (v5.4.6 P3-6)
            assert n_L < n_H

    def test_non_physical_substrate_rejected(self):
        from lumenairy.elements.coatings import broadband_ar_v_coat
        for bad in (0.0, -1.52, float('nan'), float('inf')):
            with pytest.raises(ValueError, match='n_substrate'):
                broadband_ar_v_coat(bad, self._WL)


# ==========================================================================
# E-H11 -- makedammann2d default unit system
# ==========================================================================

class TestEH11DammannUnitDefault:

    _KW = dict(diforders=np.ones((3, 3)), itr=3, plot=False, seed=0)
    # SI THz / MMW design: 8 mm grating period at 1.1 mm wavelength.
    _THZ = dict(periodx=8.0e-3, periody=8.0e-3, waveln=1.1e-3)

    def test_default_is_si(self):
        from lumenairy.elements.doe import makedammann2d
        got = (inspect.signature(makedammann2d)
               .parameters['_legacy_units'].default)
        assert got == 'SI', (
            f"_legacy_units default is {got!r}; 'auto' silently rescaled "
            f"SI THz/MMW designs by 1e-6 (audit E-H11).")

    def test_si_thz_design_is_not_rescaled_and_is_silent(self):
        from lumenairy.elements.doe import makedammann2d
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _nf, _ff, cell = makedammann2d(**self._THZ, **self._KW)
        # ndifordersx = ceil(8e-3/(0.5*1.1e-3)*0.5)*2 = 16 -> 5e-4 m
        ref = 8.0e-3 / (int(np.ceil(8.0e-3 / (0.5 * 1.1e-3) * 0.5)) * 2)
        assert cell[0] == pytest.approx(ref, rel=1e-12), (
            f"cell_pixel_size={cell[0]} for an SI 8 mm / 1.1 mm design; "
            f"expected {ref} (pre-fix: 5e-10, rescaled by 1e-6).")
        assert cell[0] > 1e-5, (
            f"cell_pixel_size={cell[0]} m is sub-micron -- the silent "
            f"auto-rescale is back.")
        assert [str(r.message) for r in w
                if 'makedammann2d' in str(r.message)] == [], (
            "A correct SI call must not warn.")

    def test_explicit_si_and_default_agree_bit_for_bit(self):
        from lumenairy.elements.doe import makedammann2d
        a = makedammann2d(**self._THZ, **self._KW)
        b = makedammann2d(**self._THZ, _legacy_units='SI', **self._KW)
        assert np.array_equal(a[0], b[0]) and a[2] == b[2]

    def test_legacy_auto_still_converts_but_warns_loudly(self):
        from lumenairy.elements.doe import makedammann2d
        with pytest.warns(UserWarning, match='RETIRED') as rec:
            _nf, _ff, cell = makedammann2d(
                **self._THZ, _legacy_units='auto', **self._KW)
        assert cell[0] == pytest.approx(5e-10, rel=1e-9), (
            f"the explicit legacy path must still convert (cell={cell[0]})")
        msg = str(rec[0].message)
        assert '5.32' in msg and 'um' in msg, (
            f"the retired-shim warning must name the removal version and "
            f"the ``_legacy_units='um'`` migration; got {msg!r}")

    def test_legacy_warning_is_visible_under_the_default_filter(self):
        """A ``DeprecationWarning`` is hidden outside ``__main__``; that is
        how the 1e-6 rescale stayed silent for five releases."""
        from lumenairy.elements.doe import makedammann2d
        seen = []
        old = warnings.showwarning

        def hook(message, category, filename, lineno, file=None, line=None):
            seen.append(category)

        warnings.showwarning = hook
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('default')
                makedammann2d(**self._THZ, _legacy_units='auto', **self._KW)
        finally:
            warnings.showwarning = old
        assert any(issubclass(c, UserWarning)
                   and not issubclass(c, DeprecationWarning) for c in seen), (
            f"the retired legacy path must surface under the DEFAULT "
            f"warning filter; surfaced categories: {seen}")

    def test_explicit_um_path_unchanged(self):
        from lumenairy.elements.doe import makedammann2d
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _nf, _ff, cell = makedammann2d(
                periodx=61.0, periody=61.0, waveln=1.31,
                _legacy_units='um', **self._KW)
        # Rescaled to 61 um / 1.31 um, then the usual wavsamp sizing.
        n_ord = int(np.ceil(61e-6 / (0.5 * 1.31e-6) * 0.5)) * 2
        assert cell[0] == pytest.approx(61e-6 / n_ord, rel=1e-9)

    def test_meter_scale_still_rejected_under_the_new_default(self):
        from lumenairy.elements.doe import makedammann2d
        with pytest.raises(ValueError, match='periodx'):
            makedammann2d(periodx=2.0, periody=61e-6, waveln=1.31e-6,
                          itr=1, plot=False)


# ==========================================================================
# A-3 -- top-level DEFAULT_* constants must be live
# ==========================================================================

@pytest.fixture
def restore_defaults():
    """Save / restore all four config knobs (fixture-safe)."""
    saved = (la.get_default_complex_dtype(),
             la.get_default_real_dtype(),
             la.get_default_wave_propagator(),
             la.get_default_dy())
    yield
    la.set_default_complex_dtype(saved[0])
    la.set_default_real_dtype(saved[1])
    la.set_default_wave_propagator(saved[2])
    la.set_default_dy(saved[3])


class TestA3TopLevelDefaultsAreLive:

    def test_complex_dtype_constant_follows_the_setter(self, restore_defaults):
        la.set_default_complex_dtype('complex64')
        assert np.dtype(la.DEFAULT_COMPLEX_DTYPE) == np.dtype(np.complex64), (
            f"la.DEFAULT_COMPLEX_DTYPE reads "
            f"{la.DEFAULT_COMPLEX_DTYPE!r} after "
            f"set_default_complex_dtype('complex64') -- it is an "
            f"import-time snapshot (audit A-3).")
        assert (np.dtype(la.DEFAULT_COMPLEX_DTYPE)
                == np.dtype(la.get_default_complex_dtype())), (
            "constant and getter disagree")

    def test_all_four_constants_follow_their_setters(self, restore_defaults):
        la.set_default_real_dtype('float32')
        la.set_default_wave_propagator('fresnel')
        la.set_default_dy(2.5e-6)
        assert np.dtype(la.DEFAULT_REAL_DTYPE) == np.dtype(np.float32)
        assert la.DEFAULT_WAVE_PROPAGATOR == 'fresnel'
        assert la.DEFAULT_DY == pytest.approx(2.5e-6)

    def test_constants_agree_with_the_submodule_live_forward(
            self, restore_defaults):
        """Sibling parity with the ``propagation.py:296`` precedent."""
        from lumenairy.propagators import propagation as _prop
        la.set_default_complex_dtype('complex64')
        la.set_default_wave_propagator('sas')
        assert (np.dtype(la.DEFAULT_COMPLEX_DTYPE)
                == np.dtype(_prop.DEFAULT_COMPLEX_DTYPE))
        assert la.DEFAULT_WAVE_PROPAGATOR == _prop.DEFAULT_WAVE_PROPAGATOR

    def test_restored_after_the_fixture(self):
        """Runs after the mutating tests in file order: the fixture must
        have put the pristine defaults back."""
        assert np.dtype(la.DEFAULT_COMPLEX_DTYPE) == np.dtype(np.complex128)
        assert la.DEFAULT_WAVE_PROPAGATOR == 'asm'
        assert la.DEFAULT_DY is None

    def test_from_import_form_also_sees_the_live_value(self, restore_defaults):
        la.set_default_wave_propagator('fresnel')
        import importlib
        mod = importlib.import_module('lumenairy')
        assert getattr(mod, 'DEFAULT_WAVE_PROPAGATOR') == 'fresnel'

    def test_export_integrity_and_attribute_error_preserved(self):
        unresolved = [n for n in la.__all__ if not hasattr(la, n)]
        assert unresolved == [], (
            f"__all__ entries no longer resolvable after the PEP-562 "
            f"forwarding: {unresolved}")
        for name in ('DEFAULT_COMPLEX_DTYPE', 'DEFAULT_REAL_DTYPE',
                     'DEFAULT_WAVE_PROPAGATOR', 'DEFAULT_DY'):
            assert name in la.__all__
            assert name in dir(la), (
                f"{name} vanished from dir(lumenairy) -- __dir__ must "
                f"keep the forwarded names discoverable.")
        with pytest.raises(AttributeError):
            la.DEFINITELY_NOT_A_REAL_LUMENAIRY_NAME
