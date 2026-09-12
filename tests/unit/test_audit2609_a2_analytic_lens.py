"""WP-A2 regression pins for the 2026-09-11 adversarial audit --
``apply_real_lens`` (``elements/_lens_real.py``) and the sag helper it shares
with the mirror path (``elements/lenses.py``).

Findings covered here: L3, L4, L5, L6, L7, L8, L11, L12, L13, L14, L15, L16,
L17, L19, L20 and E4.  L1 (``seidel_correction``) and L2 / L19's tilt
convention are pinned where their defective predecessors lived, in
``test_audit_glass.py`` and ``test_v5_2_off_axis_conic_surface_frame.py``.

Every numeric bar below carries its derivation: what the oracle is, the
oracle's own error floor, the measured value on the fixed code, and the
measured value on the defect it guards.  No test here asserts a wall clock or
a speed-up (``docs/TESTING_STANDARDS.md`` S1), skips on a resource
precondition (S2), or scores the code against another implementation in this
library (the self-referential-oracle trap that made the predecessor of the
Seidel test unfalsifiable).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy.glass as _glass
from lumenairy.elements._lens_real import (
    _screen_exp,
    apply_real_lens,
    prepare_real_lens,
)
from lumenairy.elements.lenses import surface_sag_general
from lumenairy.glass import get_glass_index

LAM = 632.8e-9
K0 = 2.0 * np.pi / LAM


# ===========================================================================
# Shared oracles -- written from the surface equation, not from this library's
# models.  A meridional Newton intersection + vector Snell + the signed
# transfer to the exit vertex plane.
# ===========================================================================
def _sag_oracle(h_sq, R, k=0.0, asph=None):
    h_sq = np.asarray(h_sq, dtype=np.float64)
    if not np.isfinite(R) or R == 0:
        s = np.zeros_like(h_sq)
    else:
        c = 1.0 / R
        arg = 1.0 - (1.0 + k) * c * c * h_sq
        s = c * h_sq / (1.0 + np.sqrt(np.where(arg < 0, np.nan, arg)))
    if asph:
        h = np.sqrt(h_sq)
        for p, a in asph.items():
            s = s + a * h ** p
    return s


def _dsag_oracle(h, R, k=0.0, asph=None):
    h = np.asarray(h, dtype=np.float64)
    if not np.isfinite(R) or R == 0:
        d = np.zeros_like(h)
    else:
        c = 1.0 / R
        d = c * h / np.sqrt(1.0 - (1.0 + k) * c * c * h * h)
    if asph:
        for p, a in asph.items():
            d = d + a * p * h ** (p - 1)
    return d


def _single_surface_eikonal(R, n1, n2, h):
    """EXACT vertex-plane eikonal of one refracting conic face for a collimated
    input, at the entrance height ``h``.

    Closed form, no iteration: the facet tilt is ``theta_i = atan(sag')``, the
    refracted ray leaves at ``theta_i - theta_t`` to the z axis, and the
    axial-translation identity gives the screen OPD of a facet a height ``sag``
    above the vertex plane as ``(n2 cos(theta_i - theta_t) - n1) * sag``.  The
    OPL the wave carries is minus that (the screen is ``exp(-i k0 OPD)`` under
    ``phase = exp(+i k0 OPL)``).  Error: none -- this IS the identity, verified
    elsewhere against an explicit 3-D trace to 1.14e-12 relative.
    """
    sag = _sag_oracle(h * h, R)
    th_i = np.arctan(_dsag_oracle(h, R))
    th_t = np.arcsin(np.sin(th_i) * n1 / n2)
    return -(n2 * np.cos(th_i - th_t) - n1) * sag


def _exit_phase_opl(E_out, dx):
    """Unwrapped OPL [m] along the central row of an exit field."""
    N = E_out.shape[0]
    return np.unwrap(np.angle(E_out[N // 2])) / K0


# ===========================================================================
# L12 -- slant_correction uses the Z-AXIS-referenced identity
# ===========================================================================
class TestL12SlantCorrectionAxialTranslationIdentity:
    def test_slant_beats_the_paraxial_screen_on_one_face(self):
        """DERIVATION OF THE BAR.  Oracle: ``_single_surface_eikonal`` above --
        the exact axial-translation identity for one facet, error floor zero
        (it is the closed form, not a discretisation of one).  Fixture: one
        N-BK7 face, R = 50 mm, over +-1.6 mm, collimated.

        Measured, piston-free rms of the exit OPL against that oracle:
        paraxial screen 3.83 nm, slant screen 0.0041 nm -- a factor 940.
        Pre-fix the slant screen used NORMAL-referenced cosines,
        ``n2 cos(theta_t) - n1 cos(theta_i)``, whose leading obliquity term has
        the WRONG SIGN and is ``1/(n2-1) = 1.94x`` too large, so its total
        error was ``n2/(n2-1) = 2.94x`` the paraxial screen's -- i.e. the
        ratio below was 0.34, not 940.

        The bar is 50x: two orders below the measured margin and two orders
        above 1.0, so it fails on any return of the defect (which puts the
        ratio under 1) and on any milder regression too.
        """
        R, ap = 50e-3, 3.2e-3
        n2 = float(get_glass_index('N-BK7', LAM))
        rx = dict(surfaces=[dict(radius=R, glass_before='AIR',
                                 glass_after='N-BK7')],
                  thicknesses=[])
        N, dx = 1024, 1.0e-6
        E = np.ones((N, N), dtype=np.complex128)
        x = (np.arange(N) - N / 2) * dx
        m = np.abs(x) <= ap / 2
        out = {}
        for label, kw in (('paraxial', {}),
                          ('slant', dict(slant_correction=True))):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                Eo = apply_real_lens(E, prescription=rx, wavelength=LAM,
                                     dx=dx, **kw)
            d = (_exit_phase_opl(Eo, dx)[m]
                 - _single_surface_eikonal(R, 1.0, n2, x[m]))
            d = d - d.mean()
            out[label] = float(np.sqrt(np.mean(d ** 2)))
        assert out['slant'] < out['paraxial'] / 50.0, (
            f"slant_correction gives {out['slant'] * 1e9:.4f} nm rms against "
            f"the exact one-facet eikonal, where the paraxial screen gives "
            f"{out['paraxial'] * 1e9:.4f} nm -- a {out['paraxial'] / out['slant']:.1f}x "
            f"ratio where >= 50x is required.  Pre-fix it was 0.34x (2.94x "
            f"WORSE than the screen it is meant to improve).")

    def test_slant_is_bit_identical_between_banded_and_whole_grid(self):
        """The banded ``_slant_narrow_chunk`` arm carries a verbatim copy of
        the screen, and the defect this pins was duplicated in it.  Byte
        identity is a two-sided statement with no tolerance to derive."""
        rx = dict(surfaces=[dict(radius=40e-3, glass_before='AIR',
                                 glass_after='N-BK7'),
                            dict(radius=-40e-3, glass_before='N-BK7',
                                 glass_after='AIR')],
                  thicknesses=[3e-3], aperture_diameter=1.6e-3)
        N, dx = 256, 4e-6
        E = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = apply_real_lens(E, prescription=rx, wavelength=LAM, dx=dx,
                                slant_correction=True, sag_chunk_rows=0)
            b = apply_real_lens(E, prescription=rx, wavelength=LAM, dx=dx,
                                slant_correction=True, sag_chunk_rows=37)
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8))


# ===========================================================================
# L13 -- fresnel=True applies the POWER transmittance
# ===========================================================================
class TestL13FresnelPowerTransmittance:
    """Oracle: the Fresnel power transmittance at normal incidence,
    ``T = 4 n1 n2 / (n1 + n2)**2``.  Closed form, error floor zero.  The
    library's ``sum |E|**2`` is its power everywhere (the ASM is
    Parseval-unitary), so this is what crossing an index step must cost.

    Bars are 1e-6 relative: the measurement is one multiply per pixel of an
    exactly flat face, so the only error is float64 rounding (~1e-16), and the
    defect it guards is 34 % (a single air-glass face returned ``|t|**2`` =
    0.632344 against the correct 0.958057).
    """

    N, DX = 128, 4e-6

    def _power_ratio(self, rx):
        E = np.ones((self.N, self.N), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Eo = apply_real_lens(E, prescription=rx, wavelength=LAM,
                                 dx=self.DX, fresnel=True)
        return float(np.sum(np.abs(Eo) ** 2) / np.sum(np.abs(E) ** 2))

    def test_single_air_to_glass_face(self):
        n = float(get_glass_index('N-BK7', LAM))
        expect = 4.0 * n / (1.0 + n) ** 2
        got = self._power_ratio(dict(
            surfaces=[dict(radius=float('inf'), glass_before='AIR',
                           glass_after='N-BK7')], thicknesses=[]))
        assert got == pytest.approx(expect, rel=1e-6), (
            f"a single AIR->N-BK7 face transmits {got:.6f} of the power "
            f"against the Fresnel value {expect:.6f}.  Pre-fix it returned "
            f"|t|**2 = {(2.0 / (1.0 + n)) ** 2:.6f}, 34 % low.")

    def test_bare_cemented_interface(self):
        n1 = float(get_glass_index('N-BK7', LAM))
        n2 = float(get_glass_index('N-SF11', LAM))
        expect = 4.0 * n1 * n2 / (n1 + n2) ** 2
        got = self._power_ratio(dict(
            surfaces=[dict(radius=float('inf'), glass_before='N-BK7',
                           glass_after='N-SF11')], thicknesses=[]))
        assert got == pytest.approx(expect, rel=1e-6), (
            f"a bare N-BK7 -> N-SF11 interface transmits {got:.6f} against "
            f"{expect:.6f}; pre-fix 0.846390.")

    def test_air_glass_air_plate_is_unchanged(self):
        """The impedance factors telescope to ``n_last / n_first = 1`` for an
        element that starts and ends in air, so a PLATE must be bit-for-bit
        what it always was -- this is the backward-compatibility half of the
        change and the reason the defect survived every plate fixture."""
        n = float(get_glass_index('N-BK7', LAM))
        expect = (4.0 * n / (1.0 + n) ** 2) ** 2
        got = self._power_ratio(dict(
            surfaces=[dict(radius=float('inf'), glass_before='AIR',
                           glass_after='N-BK7'),
                      dict(radius=float('inf'), glass_before='N-BK7',
                           glass_after='AIR')], thicknesses=[3e-3]))
        assert got == pytest.approx(expect, rel=1e-9)

    def test_banded_fresnel_is_bit_identical_to_whole_grid(self):
        rx = dict(surfaces=[dict(radius=40e-3, glass_before='AIR',
                                 glass_after='N-BK7'),
                            dict(radius=-40e-3, glass_before='N-BK7',
                                 glass_after='AIR')],
                  thicknesses=[3e-3], aperture_diameter=1.6e-3)
        E = np.ones((256, 256), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = apply_real_lens(E, prescription=rx, wavelength=LAM, dx=4e-6,
                                fresnel=True, sag_chunk_rows=0)
            b = apply_real_lens(E, prescription=rx, wavelength=LAM, dx=4e-6,
                                fresnel=True, sag_chunk_rows=33)
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8))


# ===========================================================================
# L14 / L11 -- prescription-key validation
# ===========================================================================
class TestL14StopIndexAndL11Contracts:
    N, DX = 128, 20e-6

    @staticmethod
    def _rx(stop=None):
        d = dict(surfaces=[dict(radius=50e-3, glass_before='AIR',
                                glass_after='N-BK7'),
                           dict(radius=-50e-3, glass_before='N-BK7',
                                glass_after='AIR')],
                 thicknesses=[3e-3], aperture_diameter=1.5e-3)
        if stop is not None:
            d['stop_index'] = stop
        return d

    def _power(self, stop):
        E = np.ones((self.N, self.N), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Eo = apply_real_lens(E, prescription=self._rx(stop),
                                 wavelength=LAM, dx=self.DX)
        return float(np.sum(np.abs(Eo) ** 2) / np.sum(np.abs(E) ** 2))

    def test_negative_stop_index_selects_from_the_end(self):
        """``stop_index=-1`` must mean the LAST surface, as everywhere else in
        Python.  Two-sided and tolerance-free: it must transmit exactly what
        ``stop_index = len(surfaces) - 1`` transmits.

        Pre-fix ``-1`` matched no surface AND suppressed the entrance
        aperture, so the element transmitted 1.00000 of the input power where
        the 1.5 mm stop passes ~0.07 -- 14x the energy, with no warning.
        """
        assert self._power(-1) == self._power(1)
        assert self._power(-2) == self._power(0)
        assert self._power(-1) < 0.5, (
            "the stop stopped stopping: an aperture that transmits more than "
            "half the flooded grid is not clipping")

    @pytest.mark.parametrize('stop', [2, 5, -3, 1.5, 'first', True])
    def test_out_of_range_or_mistyped_stop_index_raises(self, stop):
        with pytest.raises(ValueError, match=r"apply_real_lens: prescription"):
            apply_real_lens(np.ones((self.N, self.N), dtype=np.complex128),
                            prescription=self._rx(stop), wavelength=LAM,
                            dx=self.DX)

    def test_prepare_real_lens_reads_the_key_the_same_way(self):
        """The two entry points disagreed about ``stop_index``: one accepted
        anything, the other refused everything.  They must at least DIAGNOSE
        the same malformed value the same way."""
        with pytest.raises(ValueError, match=r"out of range"):
            prepare_real_lens(prescription=self._rx(5), wavelength=LAM,
                              dx=self.DX, N=self.N)

    def test_thicknesses_contract_is_a_valueerror_not_an_assert(self):
        """``assert`` is stripped under ``python -O``, where a short
        ``thicknesses`` surfaced as a bare ``IndexError`` from inside the loop
        and an over-long one was accepted silently.  CONVENTIONS SS2 requires
        the ``f"{fn_name}: ..."`` form, which ``prepare_real_lens`` already
        used for the same condition."""
        bad = dict(self._rx())
        bad['thicknesses'] = []
        with pytest.raises(ValueError, match=r"apply_real_lens: prescription "
                                             r"needs 1 thickness"):
            apply_real_lens(np.ones((self.N, self.N), dtype=np.complex128),
                            prescription=bad, wavelength=LAM, dx=self.DX)

    def test_slant_and_seidel_together_are_refused(self):
        """L20.  The Seidel reference is built from the screen the split step
        applies, and both flags replace the SAME per-surface coefficient, so
        stacking them double-counts the facet obliquity: measured 173.5 ->
        1488.6 nm rms exit OPD on an 8 mm cemented doublet."""
        with pytest.raises(ValueError, match=r"mutually exclusive"):
            apply_real_lens(np.ones((self.N, self.N), dtype=np.complex128),
                            prescription=self._rx(), wavelength=LAM,
                            dx=self.DX, slant_correction=True,
                            seidel_correction=True)


# ===========================================================================
# L15 -- form_error validation
# ===========================================================================
class TestL15FormErrorValidation:
    N, DX = 64, 8e-6

    def _rx(self, fe):
        return dict(surfaces=[dict(radius=50e-3, glass_before='AIR',
                                   glass_after='N-BK7', form_error=fe),
                              dict(radius=float('inf'),
                                   glass_before='N-BK7', glass_after='AIR')],
                    thicknesses=[1e-3])

    def _call(self, fe):
        return apply_real_lens(
            np.ones((self.N, self.N), dtype=np.complex128),
            prescription=self._rx(fe), wavelength=LAM, dx=self.DX)

    def test_a_well_shaped_map_is_applied_with_the_right_value_and_sign(self):
        """DERIVATION OF THE BAR.  Oracle: a uniform ``dz`` of sag through an
        index step ``n2-n1`` imprints ``exp(-i k0 (n2-n1) dz)``, i.e. a phase
        of ``-k0 (n2-n1) dz``.  Closed form, error floor zero.  For
        dz = 100 nm and N-BK7 at 632.8 nm that is -0.511441 rad.  Bar 1e-9 rad:
        seven decades above the float64 floor and nine below the value."""
        dz = 100e-9
        n = float(get_glass_index('N-BK7', LAM))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = self._call(None)
            b = self._call(np.full((self.N, self.N), dz))
        got = float(np.angle(b[self.N // 2, self.N // 2]
                             * np.conj(a[self.N // 2, self.N // 2])))
        assert got == pytest.approx(-K0 * (n - 1.0) * dz, abs=1e-9)

    @pytest.mark.parametrize('shape', [(64,), (32, 32), (64, 64, 1), (1, 64)])
    def test_wrongly_shaped_maps_raise_with_the_module_prefix(self, shape):
        """A ``(N,)`` map used to be BROADCAST silently across every row -- one
        row of a figure map replicated N times, applied with no diagnostic; a
        mismatched 2-D map died with a raw numpy broadcast error, and an
        ``(N, N, 1)`` one survived the lens and died inside the ASM."""
        with pytest.raises(ValueError, match=r"apply_real_lens: surfaces\[0\]"
                                             r"\['form_error'\]"):
            self._call(np.zeros(shape))

    def test_complex_map_is_refused(self):
        with pytest.raises(ValueError, match=r"complex"):
            self._call(np.zeros((self.N, self.N), dtype=np.complex128))


# ===========================================================================
# L16 + E4 -- surface_sag_general
# ===========================================================================
def _legacy_surface_sag_general(h_sq, R, conic=0.0):
    """The pre-fix expression-per-line conic chain, verbatim.  The comparison
    target for the in-place rewrite's bit-identity claim."""
    norm = (1 + conic) * h_sq / R ** 2
    valid = norm < 0.9999
    denom_arg = np.where(valid, 1 - norm, 0.01)
    return np.where(valid, h_sq / (R * (1 + np.sqrt(denom_arg))), np.nan)


class TestL16InPlaceSagIsBitIdentical:
    """The in-place rewrite must be BIT-identical, not merely close: every
    caller's output is compared byte-for-byte somewhere in this suite (the
    banded/whole-grid matrix, the prepared-lens pin), so an ULP here surfaces
    as a failure there.  Two-sided, no tolerance to derive."""

    AX = np.linspace(-11e-3, 11e-3, 257)

    @pytest.mark.parametrize('R,k', [(50e-3, 0.0), (30e-3, -1.0),
                                     (12e-3, -2.0), (12e-3, 0.7),
                                     (51.68e-3, -0.6)])
    def test_conic_branch_bit_identical(self, R, k):
        X, Y = np.meshgrid(self.AX, self.AX)
        h = X * X + Y * Y
        a = surface_sag_general(h, R, k, None)
        b = _legacy_surface_sag_general(h, R, k)
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8))

    def test_float32_geometry_bit_identical_and_dtype_preserved(self):
        X, Y = np.meshgrid(self.AX, self.AX)
        h = (X * X + Y * Y).astype(np.float32)
        a = surface_sag_general(h, 51.68e-3, -0.6, None)
        b = _legacy_surface_sag_general(h, 51.68e-3, -0.6)
        assert a.dtype == np.float32
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8))

    def test_flat_surface_returns_zeros(self):
        X, Y = np.meshgrid(self.AX, self.AX)
        h = X * X + Y * Y
        for flat in (float('inf'), -float('inf'), None):
            assert np.array_equal(surface_sag_general(h, flat),
                                  np.zeros_like(h)), flat

    def test_zero_radius_raises_instead_of_returning_all_nan(self):
        """``R = 0`` is not a surface: the conic sag divides by ``R**2`` and
        then by ``R``, so it returned an all-NaN grid behind four ANONYMOUS
        numpy RuntimeWarnings ("divide by zero", "invalid value") that name
        neither this function nor the offending prescription key -- and the
        NaN then propagated into the OPD screen, where the module's sentinel
        zeroing quietly turned the whole surface into a no-op.  The two
        spellings of a FLAT surface (``inf``, ``None``) are unaffected, and
        are checked above so this cannot be met by refusing them too.
        """
        X, Y = np.meshgrid(self.AX, self.AX)
        with warnings.catch_warnings():
            warnings.simplefilter('error')      # no anonymous numpy warnings
            with pytest.raises(ValueError,
                               match=r"surface_sag_general: radius R = 0"):
                surface_sag_general(X * X + Y * Y, 0.0)


class TestE4AsphericKernelNonContiguousInput:
    """E4.  The numba aspheric kernel accumulates through ``sag.ravel()``,
    which is a VIEW only when ``sag`` is C-contiguous.  ``sag`` inherits its
    memory order from ``h_sq``, so for an F-ordered or transposed ``h_sq`` the
    kernel added the whole polynomial into a temporary copy and the copy was
    discarded -- the aspheric term vanished completely and silently (measured
    9.41e-6 m = 100 % of the term).  Two-sided: the answer must be BIT-equal to
    the C-ordered one, which is a statement no tolerance can soften.
    """

    R, K = 51.68e-3, -0.6
    ASPH = {4: 1.0e3, 6: -2.0e5}

    @staticmethod
    def _h():
        ax = np.linspace(-10e-3, 10e-3, 129)
        X, Y = np.meshgrid(ax, ax)
        return np.ascontiguousarray(X * X + Y * Y)

    def test_aspheric_term_is_present_at_all(self):
        h = self._h()
        d = np.nanmax(np.abs(surface_sag_general(h, self.R, self.K, self.ASPH)
                             - surface_sag_general(h, self.R, self.K, None)))
        assert d > 1e-6, "fixture has no aspheric departure to lose"

    @pytest.mark.parametrize('order', ['F', 'T', 'strided'])
    def test_non_c_contiguous_h_sq_keeps_the_polynomial(self, order):
        h = self._h()
        ref = surface_sag_general(h, self.R, self.K, self.ASPH)
        if order == 'F':
            hq = np.asfortranarray(h)
        elif order == 'T':
            hq = np.ascontiguousarray(h.T).T
        else:
            hq = np.ascontiguousarray(np.repeat(h, 2, axis=1))[:, ::2]
        assert not hq.flags['C_CONTIGUOUS']
        got = surface_sag_general(hq, self.R, self.K, self.ASPH)
        assert np.array_equal(np.ascontiguousarray(got).view(np.uint8),
                              ref.view(np.uint8))


# ===========================================================================
# L17 -- the phase screen
# ===========================================================================
class TestL17PhaseScreen:
    @pytest.mark.parametrize('dtype', [np.float64, np.float32])
    def test_the_screen_is_bit_identical_to_the_complex_exp(self, dtype):
        """DERIVATION.  numpy's complex ``exp`` of a pure-imaginary argument is
        ``exp(0) * (cos b + i sin b)`` with ``exp(0) == 1.0`` exactly, so at
        float64 the cos/sin build is the same arithmetic on the same inputs --
        the claim is exactness, not closeness, and is asserted as such.

        At float32 it is NOT: numpy's ``complex64`` exponential carries more
        than float32 through its own sine/cosine while ``np.cos(float32_arg,
        out=<float32 view>)`` does not, and the two diverge by ~8.4e-08 of unit
        modulus.  ``_screen_exp`` therefore falls back to ``xp.exp`` for a
        narrower geometry dtype, so THIS assertion -- one bar, both dtypes --
        covers the fallback as well as the fast path.  Without it,
        ``set_lens_sag_dtype(np.float32)`` broke ``PreparedAnalyticLens``'s
        byte-identity with ``apply_real_lens`` by 1.2e-07 of peak field.

        Checked over five decades of OPD, including the many-wave regime where
        the argument reduction runs.
        """
        rng = np.random.default_rng(20260911)
        for scale in (1e-9, 1e-7, 1e-5, 1e-3, 1e-1):
            opd = (rng.standard_normal((64, 64)) * scale).astype(dtype)
            a = _screen_exp(opd, K0, np)
            b = np.exp(-1j * K0 * opd)
            assert a.dtype == b.dtype
            assert np.array_equal(a.view(np.uint8), b.view(np.uint8)), (
                f"screen differs from np.exp at {dtype.__name__}, scale "
                f"{scale:g} by {np.abs(a - b).max():.3e}")

    def test_float32_geometry_gives_a_complex64_screen(self):
        opd = np.zeros((8, 8), dtype=np.float32)
        assert _screen_exp(opd, K0, np).dtype == np.complex64

    def test_prepared_lens_stays_byte_identical_at_float32_geometry(self):
        """The process-wide ``set_lens_sag_dtype(np.float32)`` knob must keep
        ``PreparedAnalyticLens`` byte-identical to ``apply_real_lens`` -- the
        contract ``test_niche_audit_e_prepared_and_enums.py`` pins, and the one
        a float32-only screen optimisation would have broken."""
        from lumenairy.elements._lens_real import (
            get_lens_sag_dtype,
            prepare_real_lens as _prep,
            set_lens_sag_dtype,
        )
        rx = dict(surfaces=[dict(radius=50e-3, glass_before='AIR',
                                 glass_after='N-BK7'),
                            dict(radius=-50e-3, glass_before='N-BK7',
                                 glass_after='AIR')],
                  thicknesses=[4e-3], aperture_diameter=5e-3)
        N, dx = 128, 40e-6
        x = (np.arange(N) - N / 2) * dx
        Xg, Yg = np.meshgrid(x, x)
        E = np.exp(-(Xg ** 2 + Yg ** 2) / (1.2e-3) ** 2).astype(np.complex128)
        keep = get_lens_sag_dtype()
        try:
            for dt in (np.float64, np.float32):
                set_lens_sag_dtype(dt)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    prep = _prep(prescription=rx, wavelength=LAM, dx=dx, N=N)
                    direct = apply_real_lens(E, prescription=rx,
                                             wavelength=LAM, dx=dx)
                assert np.array_equal(prep(E).view(np.uint8),
                                      direct.view(np.uint8)), (
                    f"sag_dtype={dt.__name__}: prepared vs apply differ by "
                    f"{np.abs(prep(E) - direct).max():.3e}")
        finally:
            set_lens_sag_dtype(keep)

    def test_a_flat_face_screen_is_skipped_not_computed(self, monkeypatch):
        """The flat-face early-out is an OPERATION-COUNT property, not a
        numerical one: ``exp(-i k0 * 0) == 1 + 0j`` and multiplying a finite
        field by it is already exact, so nothing about the OUTPUT can pin it
        (``docs/TESTING_STANDARDS.md`` S1 forbids asserting the wall clock it
        actually saves -- 566 ms and 134 MB per plano face at N = 2048).  What
        IS pinnable is that the screen is not built at all.

        Counted here by wrapping the module's own screen builder.  The fixture
        is one CURVED face and one FLAT face, so the count must be exactly 1,
        not 2.  The grid is kept under the numexpr gate (``E.size < 2**20``) so
        the numpy branch runs on any build.
        """
        import lumenairy.elements._lens_real as _lr

        calls = {'n': 0}
        real = _lr._screen_exp

        def counting(opd, k0, xp):
            calls['n'] += 1
            return real(opd, k0, xp)

        monkeypatch.setattr(_lr, '_screen_exp', counting)
        rx = dict(surfaces=[dict(radius=40e-3, glass_before='AIR',
                                 glass_after='N-BK7'),
                            dict(radius=float('inf'),
                                 glass_before='N-BK7', glass_after='AIR')],
                  thicknesses=[2e-3])
        E = np.ones((128, 128), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = _lr.apply_real_lens(E, prescription=rx, wavelength=LAM,
                                      dx=4e-6, bandlimit=False)
        assert calls['n'] == 1, (
            f"the phase screen was built {calls['n']} times for an element "
            f"with ONE powered face; the flat face must be skipped")
        assert np.all(np.isfinite(out))

    def test_a_flat_refracting_face_leaves_the_field_untouched(self):
        """And the skip must be exactly that -- a skip.  A single FLAT
        refracting face with no gap behind it imprints no OPD and moves
        nothing, so the output is the input byte for byte.  Two-sided, no
        tolerance."""
        rx = dict(surfaces=[dict(radius=float('inf'), glass_before='AIR',
                                 glass_after='N-BK7')], thicknesses=[])
        rng = np.random.default_rng(3)
        E = (rng.standard_normal((64, 64))
             + 1j * rng.standard_normal((64, 64))).astype(np.complex128)
        out = apply_real_lens(E, prescription=rx, wavelength=LAM, dx=6e-6)
        assert np.array_equal(out.view(np.uint8), E.view(np.uint8))

    def test_complex64_input_stays_complex64_through_the_screen(self):
        """The in-place ``E *= ph`` must not promote.  The screen is built at
        the geometry dtype and narrowed BEFORE the multiply, exactly as the
        rebinding form did."""
        rx = dict(surfaces=[dict(radius=40e-3, glass_before='AIR',
                                 glass_after='N-BK7'),
                            dict(radius=float('inf'), glass_before='N-BK7',
                                 glass_after='AIR')],
                  thicknesses=[2e-3])
        E = np.ones((64, 64), dtype=np.complex64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = apply_real_lens(E, prescription=rx, wavelength=LAM, dx=6e-6)
        assert out.dtype == np.complex64

    def test_the_input_array_is_never_mutated(self):
        """In-place screens and in-place masks operate on this function's OWN
        copy.  Pinned two-sided: the caller's array must come back byte-for-byte
        as it went in."""
        rx = dict(surfaces=[dict(radius=40e-3, glass_before='AIR',
                                 glass_after='N-BK7'),
                            dict(radius=-40e-3, glass_before='N-BK7',
                                 glass_after='AIR')],
                  thicknesses=[2e-3], aperture_diameter=0.4e-3)
        rng = np.random.default_rng(7)
        E = (rng.standard_normal((64, 64))
             + 1j * rng.standard_normal((64, 64))).astype(np.complex128)
        before = E.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens(E, prescription=rx, wavelength=LAM, dx=6e-6,
                            fresnel=True, slant_correction=True)
        assert np.array_equal(E.view(np.uint8), before.view(np.uint8))

    def test_tir_and_aperture_masks_keep_the_nan_safe_sense(self):
        """The TIR and aperture masks became boolean assignments.  Their sense
        must stay ``not (sin2_tt < 1.0)`` rather than ``sin2_tt >= 1.0``, so a
        NaN lands on the ZEROED side exactly where ``xp.where(sin2_tt < 1, E,
        0)`` put it -- a steep conic is undefined past its rim and returns NaN
        there, and the flipped sense would leave those pixels LIT with a NaN
        amplitude that poisons the whole downstream ASM.

        Two-sided and tolerance-free: the field outside the entrance aperture
        is exactly zero, and the whole field is finite.  The prescription is a
        SINGLE surface with no gap so no diffraction can refill the mask
        between the assertion and the masking step.
        """
        rx = dict(surfaces=[dict(radius=6e-3, conic=-3.0, glass_before='AIR',
                                 glass_after='N-BK7')],
                  thicknesses=[], aperture_diameter=1.0e-3)
        N, dx = 128, 8e-6
        E = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = apply_real_lens(E, prescription=rx, wavelength=LAM, dx=dx,
                                  slant_correction=True)
        assert np.all(np.isfinite(out)), (
            "a NaN survived the TIR / conic-domain masking")
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        outside = (X * X + Y * Y) > (0.5e-3) ** 2
        assert np.all(out[outside] == 0)
        assert np.any(np.abs(out) > 0), "the whole field was masked away"


# ===========================================================================
# L19 -- absorption uses the LOCAL glass path
# ===========================================================================
class TestL19AbsorptionLocalPath:
    KAPPA = 1e-4
    LAM_A = 1.0e-6

    @pytest.fixture(autouse=True)
    def _abs_glass(self):
        k = self.KAPPA
        _glass.GLASS_REGISTRY['WPA2ABS'] = lambda wl: complex(1.5, k)
        _glass._clear_glass_caches()
        yield
        _glass.GLASS_REGISTRY.pop('WPA2ABS', None)
        _glass._clear_glass_caches()

    def test_a_flat_plate_keeps_the_axial_factor_exactly(self):
        """The sag halves are zero on a plate, so the shipped
        ``exp(-k0 kappa t)`` must be reproduced to float64.  Oracle:
        Beer-Lambert on the axial thickness, closed form."""
        t = 2e-3
        rx = dict(surfaces=[dict(radius=float('inf'), glass_before='AIR',
                                 glass_after='WPA2ABS'),
                            dict(radius=float('inf'),
                                 glass_before='WPA2ABS', glass_after='AIR')],
                  thicknesses=[t])
        E = np.ones((64, 64), dtype=np.complex128)
        out = apply_real_lens(E, prescription=rx, wavelength=self.LAM_A,
                              dx=5e-6, absorption=True)
        k0a = 2 * np.pi / self.LAM_A
        assert float(np.abs(out).mean()) == pytest.approx(
            np.exp(-k0a * self.KAPPA * t), rel=1e-12)

    def test_no_attenuation_after_the_last_surface(self):
        """There is no gap behind the last surface, so its outgoing half must
        be dropped.  Two-sided: exactly 1.0."""
        rx = dict(surfaces=[dict(radius=float('inf'), glass_before='AIR',
                                 glass_after='WPA2ABS')], thicknesses=[])
        E = np.ones((64, 64), dtype=np.complex128)
        out = apply_real_lens(E, prescription=rx, wavelength=self.LAM_A,
                              dx=5e-6, absorption=True)
        assert float(np.abs(out).max()) == pytest.approx(1.0, abs=1e-15)

    def test_biconvex_apodisation_follows_the_local_thickness(self):
        """DERIVATION OF THE BAR.  Oracle: Beer-Lambert along the (near-axial)
        ray column, ``exp(-k0 kappa (t + sag_2 - sag_1))``, evaluated from the
        surface equation.  Its error is the near-axial approximation, O(NA**2)
        ~ 1e-3 relative on this f/1.25 fixture, and the propagation smear of a
        6 mm gap.

        Measured max deviation over the scored disc: 7.0e-5 with the local
        path against 5.9e-4 for the shipped AXIAL-only factor -- 8.5x, i.e.
        88 % of the apodisation depth recovered.  (On the audit's larger
        8 mm / 1024 fixture the same comparison reads 36x and 97 %; this one is
        trimmed for suite runtime, which costs margin, not correctness.)  The
        bar is a 5x improvement: below the measured 8.5x and above the
        oracle's own near-axial floor.
        """
        t, R, ap = 6e-3, 20e-3, 2.0e-3
        rx = dict(surfaces=[dict(radius=R, glass_before='AIR',
                                 glass_after='WPA2ABS'),
                            dict(radius=-R, glass_before='WPA2ABS',
                                 glass_after='AIR')],
                  thicknesses=[t], aperture_diameter=ap)
        N, dx = 256, ap / 0.8 / 256
        E = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            on = apply_real_lens(E, prescription=rx, wavelength=self.LAM_A,
                                 dx=dx, absorption=True)
            off = apply_real_lens(E, prescription=rx, wavelength=self.LAM_A,
                                  dx=dx, absorption=False)
        x = (np.arange(N) - N / 2) * dx
        k0a = 2 * np.pi / self.LAM_A
        got = (np.abs(on[N // 2])
               / np.maximum(np.abs(off[N // 2]), 1e-300))
        path = t + _sag_oracle(x * x, -R) - _sag_oracle(x * x, R)
        oracle = np.exp(-k0a * self.KAPPA * path)
        axial = np.exp(-k0a * self.KAPPA * t)
        m = np.abs(x) <= 0.45 * ap
        err_now = float(np.max(np.abs(got[m] - oracle[m])))
        err_axial = float(np.max(np.abs(axial - oracle[m])))
        assert err_now < err_axial / 5.0, (
            f"local-path absorption is {err_now:.3e} from the Beer-Lambert "
            f"oracle against {err_axial:.3e} for the axial-only factor -- a "
            f"{err_axial / err_now:.1f}x gain where >= 5x is required.")


# ===========================================================================
# L20 -- the anamorphic aperture guard
# ===========================================================================
class TestSasRefusesAnAnamorphicPitch:
    """``scalable_angular_spectrum_propagate`` takes a SINGLE pitch and
    assumes a square grid, while every other leg of ``apply_real_lens``
    threads ``dy`` correctly.  An anamorphic in-glass gap was therefore
    propagated as if ``dy == dx`` -- wrong physics on the y axis with no
    diagnostic.  ``propagate_through_system`` already refuses the same
    combination on its own sas branch (``_require_square_pitch``); this is the
    matching refusal here.

    Two-sided: the square-pitch call on the SAME prescription must still work,
    so the guard cannot be satisfied by refusing sas outright.
    """

    RX = dict(surfaces=[dict(radius=50e-3, glass_before='AIR',
                             glass_after='N-BK7'),
                        dict(radius=float('inf'), glass_before='N-BK7',
                             glass_after='AIR')],
              thicknesses=[2e-3])

    def test_anamorphic_sas_raises(self):
        E = np.ones((64, 64), dtype=np.complex128)
        with pytest.raises(ValueError,
                           match=r"apply_real_lens: wave_propagator='sas' "
                                 r"assumes a square grid pitch"):
            apply_real_lens(E, prescription=self.RX, wavelength=LAM,
                            dx=6e-6, dy=9e-6, wave_propagator='sas')

    def test_square_sas_still_runs(self):
        E = np.ones((64, 64), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = apply_real_lens(E, prescription=self.RX, wavelength=LAM,
                                  dx=6e-6, dy=6e-6, wave_propagator='sas')
        assert out.shape == E.shape and np.all(np.isfinite(out))


class TestL20AnamorphicApertureGuard:
    def test_guard_uses_the_smaller_semi_extent(self):
        """``shape[0]`` is Ny; pairing it with ``dx`` described a semi-extent
        that exists on NEITHER axis of an anamorphic grid.  Fixture: a 512x64
        grid at dx = 2 um, dy = 40 um -- x semi 0.512 mm, y semi 1.28 mm -- and
        a 1.2 mm aperture, which fits in y and does NOT fit in x.  The guard
        must fire.  Pre-fix it was handed (Ny=64, dx=2 um) = 0.064 mm, which
        is neither."""
        rx = dict(surfaces=[dict(radius=50e-3, glass_before='AIR',
                                 glass_after='N-BK7'),
                            dict(radius=float('inf'), glass_before='N-BK7',
                                 glass_after='AIR')],
                  thicknesses=[1e-3], aperture_diameter=1.2e-3)
        E = np.ones((64, 512), dtype=np.complex128)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            apply_real_lens(E, prescription=rx, wavelength=LAM,
                            dx=2e-6, dy=40e-6)
        msgs = [str(x.message) for x in w
                if 'exceed the simulation grid' in str(x.message)]
        assert msgs, ("the anamorphic aperture guard did not fire for a "
                      "1.2 mm aperture on a 1.024 mm-wide grid")
        assert 'semi=0.512' in msgs[0], (
            f"the guard reported the wrong semi-extent: {msgs[0][:160]}")
