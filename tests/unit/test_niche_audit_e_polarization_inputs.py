"""AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 Territory E, ``elements/polarization.py``:
input-validation / weak-field / convention-label pins.

The audit verified this module's element ALGEBRA textbook-exact (Malus to
1e-16, unitarity, realizability over 400 random Stokes, cross-family
retardance agreement 9e-15).  Everything below is about the surfaces around
that algebra -- the silent-wrong-input paths and the two numerically dead
zones -- plus one doc-only convention label.  Nothing here changes a valid
numeric result.

* E-H8  ``create_circular_polarized`` parsed handedness as
  ``.lower().startswith('r')`` with no else-branch, so ``'cw'``, ``'ccw'``,
  ``'clockwise'``, ``'linear'``, ``''`` and every typo not starting with 'r'
  silently returned LEFT circular (and the typo ``'rihgt'`` silently
  returned RIGHT).  Now a documented alias set, everything else raises.
* E-H9  ``apply_polarizing_beam_splitter`` rejected only
  ``extinction_ratio <= 0``; ``ER = 0.1`` was accepted and SWAPPED the two
  ports (0.909 of an x-polarized input's power came out of the "reflected"
  port), power conserved so nothing flagged it.  ER is a wanted:unwanted
  power ratio, hence >= 1.
* E-H10 ``jones_pupil_to_stokes_unpolarized`` had no shape guard: handed the
  ``(2, 2, Ny, Nx)`` layout this same module calls canonical at
  ``apply_jones_matrix``, it silently returned ``(2, 2)``-shaped Stokes maps
  wrong by O(0.5).  Both layouts are now accepted (as
  ``apply_jones_matrix`` does); shapes with no 2x2 Jones block raise.
* E-L13/E-L17 ``degree_of_polarization`` used an ABSOLUTE ``S0 > 1e-30``
  background cut, so a perfectly polarized 1e-15 V/m field reported DOP 0.0
  and a NaN field reported 0.0; and the docstring promised a [0, 1] range
  the raw ratio overshot (1 + 4.4e-16) while describing sub-1 values this
  container cannot produce.
* E-L14 ``_order_power_scale`` substituted ``kz = 1.0`` when
  ``|kz_m| < 1e-12`` on a reachable grazing branch: measured 7.1e12x too
  small an amplitude scale at ``|kz| = 1e-13``.  Now 0.0, like the
  evanescent branch.
* E-L15 ``JonesField`` read ``Ex.shape`` before ``np.asarray``, so list
  input died with ``AttributeError`` instead of the documented ``ValueError``.
* E-L16 ``create_elliptical_polarized`` accepted ``|chi| > pi/4``, outside
  the domain where the (chi, psi) parameterisation is one-to-one: the axes
  silently swap (``chi = pi/2`` came back LINEAR at ``psi = pi/2``).
* E-L12 a comment advertised a ``set_jones_batch_threshold`` setter that
  exists nowhere in the library.
* E-M13 (doc-only) the module implements the IEEE / right-hand-rule ``S3``
  sign but the convention table labelled it "Born-Wolf"; measured
  ``S3_lib / S3_BW = -1`` exactly.  No code sign changed.
"""
from __future__ import annotations

import io

import numpy as np
import pytest

from lumenairy.elements import polarization as P
from lumenairy.elements.polarization import (
    JonesField,
    _order_power_scale,
    apply_polarizing_beam_splitter,
    create_circular_polarized,
    create_elliptical_polarized,
    degree_of_polarization,
    jones_pupil_to_stokes_unpolarized,
    polarization_ellipse,
    stokes_parameters,
    stokes_to_dop,
)

DX = 1e-6
N = 4


def _jf(ex, ey, n=N):
    return JonesField(np.full((n, n), ex, complex),
                      np.full((n, n), ey, complex), DX)


def _s3_over_s0(f):
    S = stokes_parameters(f)
    return float(S['S3'][0, 0] / S['S0'][0, 0])


def _power(f):
    return float((np.abs(f.Ex) ** 2 + np.abs(f.Ey) ** 2)[0, 0])


# =====================================================================
# E-H8 -- handedness is a closed, documented alias set
# =====================================================================

class TestEH8CircularHandednessStrict:

    @pytest.mark.parametrize("spelling", ['right', 'r', 'rcp', 'rhc', 'rhcp',
                                          'RIGHT', 'Right', '  right  '])
    def test_right_spellings_give_s3_plus_one(self, spelling):
        f = create_circular_polarized(np.ones((2, 2), complex), DX, spelling)
        assert _s3_over_s0(f) == pytest.approx(+1.0, abs=1e-14), (
            f"handedness={spelling!r} must be RIGHT circular (S3 = +1 under "
            f"this module's S3 = -2 Im(Ex conj(Ey)) convention)")

    @pytest.mark.parametrize("spelling", ['left', 'l', 'lcp', 'lhc', 'lhcp',
                                          'LEFT', 'Left', '  left  '])
    def test_left_spellings_give_s3_minus_one(self, spelling):
        f = create_circular_polarized(np.ones((2, 2), complex), DX, spelling)
        assert _s3_over_s0(f) == pytest.approx(-1.0, abs=1e-14), (
            f"handedness={spelling!r} must be LEFT circular (S3 = -1)")

    @pytest.mark.parametrize("bad", ['cw', 'ccw', 'clockwise',
                                      'counterclockwise', 'CW', 'linear', '',
                                      'typo', 'rihgt', 'lieft', 'x', 'none'])
    def test_unknown_spelling_raises_naming_value_and_allowed_set(self, bad):
        # Pre-fix: every one of these silently produced a circular field --
        # LEFT for anything not starting with 'r', RIGHT for 'rihgt'.
        with pytest.raises(ValueError) as ei:
            create_circular_polarized(np.ones((2, 2), complex), DX, bad)
        msg = str(ei.value)
        assert repr(bad) in msg, "error must name the offending value"
        assert "'right'" in msg and "'left'" in msg, (
            "error must name the allowed set")

    @pytest.mark.parametrize("bad", [None, 3, 1.5, ('right',)])
    def test_non_string_raises_valueerror_not_attributeerror(self, bad):
        # Pre-fix these raised AttributeError from ``.lower()``.
        with pytest.raises(ValueError):
            create_circular_polarized(np.ones((2, 2), complex), DX, bad)

    def test_rotation_sense_names_explain_why_they_are_rejected(self):
        with pytest.raises(ValueError, match="viewing direction"):
            create_circular_polarized(np.ones((2, 2), complex), DX, 'cw')

    def test_default_is_right(self):
        f = create_circular_polarized(np.ones((2, 2), complex), DX)
        assert _s3_over_s0(f) == pytest.approx(+1.0, abs=1e-14)


# =====================================================================
# E-H9 -- PBS extinction ratio domain
# =====================================================================

class TestEH9PbsExtinctionRatioDomain:

    @pytest.mark.parametrize("er", [0.5, 0.1, 1e-6, 0.999999])
    def test_er_below_one_raises(self, er):
        # Pre-fix: accepted, and the two ports came out INVERTED.
        with pytest.raises(ValueError) as ei:
            apply_polarizing_beam_splitter(_jf(1.0, 0.0), 0.0,
                                           extinction_ratio=er)
        assert repr(er) in str(ei.value) or str(er) in str(ei.value)
        assert ">= 1" in str(ei.value)

    @pytest.mark.parametrize("er", [0.0, -5.0, float('nan')])
    def test_nonpositive_and_nan_still_raise(self, er):
        with pytest.raises(ValueError):
            apply_polarizing_beam_splitter(_jf(1.0, 0.0), 0.0,
                                           extinction_ratio=er)

    @pytest.mark.parametrize("er", [1.0, 10.0, 1000.0, 1e6, float('inf')])
    def test_er_at_or_above_one_behaves_and_conserves_power(self, er):
        t, r = apply_polarizing_beam_splitter(_jf(1.0, 0.0), 0.0,
                                              extinction_ratio=er)
        leak = 1.0 / (1.0 + er)
        assert _power(t) == pytest.approx(1.0 - leak, abs=1e-12)
        assert _power(r) == pytest.approx(leak, abs=1e-12)
        assert _power(t) + _power(r) == pytest.approx(1.0, abs=1e-15)
        # the "transmitted" port is the strong one for every legal ER
        assert _power(t) >= _power(r) - 1e-15

    def test_er_1e6_transmitted_port_carries_the_p_component(self):
        t, r = apply_polarizing_beam_splitter(_jf(1.0, 0.0), 0.0,
                                              extinction_ratio=1e6)
        assert abs(t.Ex[0, 0]) > abs(t.Ey[0, 0])
        assert _power(t) == pytest.approx(1.0 - 1e-6 / (1 + 1e-6), abs=1e-12)

    def test_ideal_and_infinite_er_agree(self):
        t0, r0 = apply_polarizing_beam_splitter(_jf(1.0, 1.0), 0.0)
        ti, ri = apply_polarizing_beam_splitter(_jf(1.0, 1.0), 0.0,
                                                extinction_ratio=np.inf)
        assert np.allclose(t0.Ex, ti.Ex) and np.allclose(t0.Ey, ti.Ey)
        assert np.allclose(r0.Ex, ri.Ex) and np.allclose(r0.Ey, ri.Ey)


# =====================================================================
# E-H10 -- Jones-pupil layout guard
# =====================================================================

def _stokes_first_principles(J):
    """Unpolarized-input output Stokes of a 2x2 Jones, from rho = J (I/2) J^H
    under this module's S3 = -2 Im(Ex conj(Ey)) convention."""
    rho = J @ (0.5 * np.eye(2)) @ J.conj().T
    return np.array([(rho[0, 0] + rho[1, 1]).real,
                     (rho[0, 0] - rho[1, 1]).real,
                     2 * rho[0, 1].real,
                     -2 * rho[0, 1].imag])


class TestEH10JonesPupilLayout:

    @staticmethod
    def _pupil(ny=4, nx=3, seed=7):
        rng = np.random.default_rng(seed)
        return (rng.normal(size=(ny, nx, 2, 2))
                + 1j * rng.normal(size=(ny, nx, 2, 2))) * 0.4

    def test_documented_layout_matches_first_principles(self):
        J = self._pupil()
        S = jones_pupil_to_stokes_unpolarized(J)
        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                ref = _stokes_first_principles(J[i, j])
                got = np.array([S[k][i, j] for k in ('S0', 'S1', 'S2', 'S3')])
                assert np.abs(got - ref).max() < 1e-14

    def test_canonical_2_2_ny_nx_layout_gives_identical_stokes(self):
        # Pre-fix this returned (2, 2)-shaped maps wrong by O(0.5).
        J = self._pupil()
        S_doc = jones_pupil_to_stokes_unpolarized(J)
        S_can = jones_pupil_to_stokes_unpolarized(
            np.moveaxis(J, (-2, -1), (0, 1)))
        for k in ('S0', 'S1', 'S2', 'S3'):
            assert S_can[k].shape == S_doc[k].shape == J.shape[:2]
            assert np.array_equal(S_can[k], S_doc[k]), (
                f"{k}: the (2,2,Ny,Nx) layout must give the same Stokes as "
                f"the documented (Ny,Nx,2,2) one")

    @pytest.mark.parametrize("shape", [(3, 3), (2, 2, 3), (4, 4, 3, 3),
                                        (2, 3, 2, 3), (5,)])
    def test_shapes_without_a_2x2_jones_block_raise(self, shape):
        rng = np.random.default_rng(3)
        A = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        with pytest.raises(ValueError, match="2x2"):
            jones_pupil_to_stokes_unpolarized(A)

    def test_square_ambiguous_input_is_read_as_the_documented_layout(self):
        # (2, 2, 2, 2) satisfies both layouts; the documented trailing-2x2
        # reading wins (and is what compute_jones_pupil produces).
        rng = np.random.default_rng(5)
        J = rng.normal(size=(2, 2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2, 2))
        S = jones_pupil_to_stokes_unpolarized(J)
        for i in range(2):
            for j in range(2):
                ref = _stokes_first_principles(J[i, j])
                got = np.array([S[k][i, j] for k in ('S0', 'S1', 'S2', 'S3')])
                assert np.abs(got - ref).max() < 1e-14

    def test_identity_pupil_normalisation_unchanged(self):
        # guards the v5.4.6 P3-23 pin's numbers through the new shape branch
        J = np.zeros((4, 4, 2, 2), dtype=complex)
        J[..., 0, 0] = 1.0
        J[..., 1, 1] = 1.0
        s = jones_pupil_to_stokes_unpolarized(J)
        assert np.allclose(s['S0'], 1.0)
        assert np.allclose(s['S1'], 0.0)


# =====================================================================
# E-L13 / E-L17 -- DOP floor, NaN, documented range
# =====================================================================

class TestEL13WeakFieldDop:

    @pytest.mark.parametrize("amp", [1.0, 1e-7, 1e-15, 1e-16, 1e-20, 1e-140])
    def test_fully_polarized_weak_field_is_dop_one(self, amp):
        # Pre-fix: 0.0 for every amp with amp**2 <= 1e-30.
        d = degree_of_polarization(_jf(amp, 0.0))
        assert float(d[0, 0]) == pytest.approx(1.0, abs=1e-12), (
            f"|E| = {amp:g} is FULLY polarized; DOP must not depend on the "
            f"absolute field scale")

    def test_zero_field_is_dop_zero(self):
        z = JonesField(np.zeros((N, N), complex), np.zeros((N, N), complex), DX)
        assert float(degree_of_polarization(z)[0, 0]) == 0.0

    def test_nan_field_propagates_nan(self):
        # Pre-fix: 0.0 -- a NaN field was laundered into "dark background".
        d = degree_of_polarization(_jf(np.nan, 0.0))
        assert np.isnan(d).all()

    def test_background_floor_is_relative_to_the_brightest_pixel(self):
        Ex = np.zeros((N, N), complex)
        Ex[0, 0] = 1.0
        Ex[1, 1] = 1e-140          # 280 decades below the peak intensity
        d = degree_of_polarization(JonesField(Ex, np.zeros((N, N), complex), DX))
        assert float(d[0, 0]) == pytest.approx(1.0, abs=1e-12)
        assert float(d[1, 1]) == 0.0, "relative background must read as dark"
        assert float(d[2, 2]) == 0.0

    def test_dop_never_exceeds_one(self):
        # Pre-fix worst over 4000 random pure states: 1.0000000000000004.
        rng = np.random.default_rng(11)
        worst = 0.0
        for _ in range(400):
            v = rng.normal(size=2) + 1j * rng.normal(size=2)
            worst = max(worst,
                        float(degree_of_polarization(_jf(v[0], v[1]))[0, 0]))
        assert worst <= 1.0, f"DOP must stay inside [0, 1]; got {worst!r}"
        assert worst == pytest.approx(1.0, abs=1e-12)

    def test_stokes_to_dop_sibling_floor_is_relative_too(self):
        for s0 in (1e-29, 1e-30, 9.9e-31, 1e-40):
            dd = stokes_to_dop({'S0': np.array([[s0]]), 'S1': np.array([[s0]]),
                                'S2': np.array([[0.0]]),
                                'S3': np.array([[0.0]])})
            assert float(dd['DOP'][0, 0]) == pytest.approx(1.0, abs=1e-12), (
                f"stokes_to_dop must not use an absolute S0 cut (S0={s0:g})")
        dd = stokes_to_dop({k: np.array([[np.nan]])
                            for k in ('S0', 'S1', 'S2', 'S3')})
        assert np.isnan(dd['DOP']).all()


# =====================================================================
# E-L14 -- grazing order carries no reconstructable power
# =====================================================================

class TestEL14GrazingOrderScale:

    @pytest.mark.parametrize("kz", [1e-12, 1e-13, 1e-14, 1e-20])
    def test_grazing_kz_returns_zero_not_a_substituted_unit_kz(self, kz):
        # Pre-fix ``az`` was divided by 1.0 instead of kz: at kz = 1e-13 the
        # returned scale was 4.472e-07 where the honest continuation is
        # 3.162e+06 -- a factor 7.1e12.
        s = _order_power_scale(1.0 + 0j, 0.0 + 0j, complex(kz),
                               complex(0.9999999999999), 0.0 + 0j,
                               1.0, 0.0, 0.0, (1.0, 0.0))
        assert s == 0.0, (
            f"a grazing order (|kz| = {kz:g}) has no finite efficiency limit; "
            f"it must be dropped like an evanescent one, got {s!r}")

    @pytest.mark.parametrize("kz", [1e-11, 1e-6, 0.1, 0.5, 1.0])
    def test_propagating_orders_unchanged(self, kz):
        ax, kx = 1.0 + 0j, 0.9999999999999
        s = _order_power_scale(ax, 0.0 + 0j, complex(kz), complex(kx),
                               0.0 + 0j, 1.0, 0.0, 0.0, (1.0, 0.0))
        az = -(kx * ax) / kz
        ref = float(np.sqrt(kz * (1.0 + abs(az) ** 2)))
        assert s == pytest.approx(ref, rel=1e-13)

    def test_evanescent_branch_still_zero(self):
        s = _order_power_scale(1.0 + 0j, 0.0 + 0j, complex(-0.5), 0.5 + 0j,
                               0.0 + 0j, 1.0, 0.0, 0.0, (1.0, 0.0))
        assert s == 0.0


# =====================================================================
# E-L15 -- JonesField coerces before it validates
# =====================================================================

class TestEL15JonesFieldListInput:

    def test_shape_mismatch_on_lists_raises_valueerror(self):
        # Pre-fix: AttributeError: 'list' object has no attribute 'shape'.
        with pytest.raises(ValueError, match="same shape"):
            JonesField([[1, 2], [3, 4]], [[1, 2, 5], [3, 4, 6]], DX)

    def test_one_dimensional_list_raises_the_documented_2d_valueerror(self):
        with pytest.raises(ValueError, match="2-D"):
            JonesField([1, 2, 3], [1, 2, 3], DX)

    def test_wellformed_2d_list_is_coerced_to_a_complex_field(self):
        f = JonesField([[1, 2], [3, 4]], [[0, 0], [0, 0]], DX)
        assert f.shape == (2, 2)
        assert np.iscomplexobj(f.Ex) and np.iscomplexobj(f.Ey)
        assert f.Ex[1, 1] == 4.0 + 0j

    def test_ndarray_inputs_are_not_copied(self):
        Ex = np.ones((2, 2), dtype=np.complex128)
        Ey = np.zeros((2, 2), dtype=np.complex128)
        f = JonesField(Ex, Ey, DX)
        assert f.Ex is Ex and f.Ey is Ey, (
            "asarray must stay a no-op for arrays (dtype/aliasing contract "
            "from 4.11.2)")


# =====================================================================
# E-L16 -- elliptical ellipticity domain
# =====================================================================

class TestEL16EllipticityDomain:

    @pytest.mark.parametrize("chi", [0.0, 0.1, -0.3, np.pi / 8,
                                      np.pi / 4, -np.pi / 4])
    def test_in_domain_chi_round_trips(self, chi):
        f = create_elliptical_polarized(np.ones((2, 2)), DX, chi, 0.0)
        _, e = polarization_ellipse(f)
        assert float(e[0, 0]) == pytest.approx(chi, abs=1e-12)

    @pytest.mark.parametrize("chi", [0.9, np.pi / 2, -np.pi / 2, 3.0,
                                      np.nan, np.inf])
    def test_out_of_domain_chi_raises(self, chi):
        # Pre-fix: silently accepted; chi = 0.9 came back as 0.6708 with the
        # orientation rotated by pi/2, chi = pi/2 came back LINEAR.
        with pytest.raises(ValueError, match="pi/4|chi"):
            create_elliptical_polarized(np.ones((2, 2)), DX, chi, 0.0)

    def test_orientation_stays_unrestricted(self):
        # psi is pi-periodic, so no domain limit is imposed on it.
        for psi in (0.0, np.pi / 3, 10.0, -7.5):
            create_elliptical_polarized(np.ones((2, 2)), DX, 0.1, psi)


# =====================================================================
# E-L12 / E-M13 -- documentation pins
# =====================================================================

class TestDocumentationPins:

    @staticmethod
    def _source():
        with io.open(P.__file__, encoding='utf-8') as fh:
            return fh.read()

    def test_no_phantom_batch_threshold_setter_reference(self):
        # E-L12: the comment advertised a setter that exists nowhere.
        assert 'set_jones_batch_threshold' not in self._source()
        assert not hasattr(P, 'set_jones_batch_threshold')
        assert P.JonesField._BATCH_PROPAGATE_MIN_N == 512

    def test_module_docstring_labels_s3_ieee_not_born_wolf(self):
        # E-M13 is doc-only: the module implements S3 = -2 Im(Ex conj(Ey)),
        # which is the IEEE / right-hand-rule sign, and used to be labelled
        # "Born-Wolf" in the convention table.
        doc = P.__doc__ or ''
        assert 'IEEE' in doc and 'right-hand-rule' in doc
        assert 'Born & Wolf' in doc, (
            "the docstring must say which convention this is NOT")
        assert 'IEEE' in (stokes_parameters.__doc__ or '')

    def test_s3_sign_is_the_negative_of_the_born_wolf_one(self):
        # The numeric half of E-M13: no code sign was changed, so this must
        # hold both before and after the doc fix.
        f = create_circular_polarized(np.ones((2, 2), complex), DX, 'right')
        Ex, Ey = f.Ex[0, 0], f.Ey[0, 0]
        s3_lib = float(-2 * np.imag(Ex * np.conj(Ey)))
        s3_bw = float(+2 * np.imag(Ex * np.conj(Ey)))
        assert s3_lib == pytest.approx(+1.0, abs=1e-14)
        assert s3_lib / s3_bw == pytest.approx(-1.0, abs=1e-15)
        assert stokes_parameters(f)['S3'][0, 0] == pytest.approx(s3_lib,
                                                                 abs=1e-15)


# =====================================================================
# guard-does-not-move-the-physics
# =====================================================================

class TestValidationDidNotChangeValidResults:
    """The audit measured the element algebra textbook-exact; the new input
    guards must be pure gatekeeping."""

    def test_malus_through_the_pbs_is_still_exact(self):
        for deg, expect in ((0.0, 1.0), (30.0, 0.75), (45.0, 0.5), (90.0, 0.0)):
            t, r = apply_polarizing_beam_splitter(_jf(1.0, 0.0),
                                                  angle_deg=deg)
            assert _power(t) == pytest.approx(expect, abs=1e-15)
            assert _power(r) == pytest.approx(1.0 - expect, abs=1e-15)

    def test_circular_jones_vectors_are_bit_exact(self):
        one = np.ones((2, 2), dtype=np.complex128)
        fr = create_circular_polarized(one, DX, 'right')
        fl = create_circular_polarized(one, DX, 'left')
        assert np.array_equal(fr.Ex, one / np.sqrt(2))
        assert np.array_equal(fr.Ey, one * 1j / np.sqrt(2))
        assert np.array_equal(fl.Ey, one * (-1j) / np.sqrt(2))

    def test_elliptical_in_domain_values_are_bit_exact(self):
        one = np.ones((2, 2), dtype=np.complex128)
        chi, psi = 0.23, 0.41
        f = create_elliptical_polarized(one, DX, chi, psi)
        cp, sp = np.cos(psi), np.sin(psi)
        cc, sc = np.cos(chi), np.sin(chi)
        assert np.array_equal(f.Ex, one * (cp * cc - 1j * sp * sc))
        assert np.array_equal(f.Ey, one * (sp * cc + 1j * cp * sc))

    def test_dop_of_ordinary_fields_is_still_one(self):
        for f in (_jf(1.0, 0.0), _jf(1.0, 1.0),
                  create_circular_polarized(np.ones((N, N), complex), DX),
                  create_elliptical_polarized(np.ones((N, N)), DX, 0.2, 0.4)):
            assert np.allclose(degree_of_polarization(f), 1.0, atol=1e-12)
