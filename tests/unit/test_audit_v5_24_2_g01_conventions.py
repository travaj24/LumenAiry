"""Regression pins for AUDIT_V5_24_2 group G01 (conventions / docs).

Each test pins the ACTUAL library behaviour that the group's doc/convention
fixes now describe, using an INDEPENDENT oracle (hand Jones math, an analytic
power balance, or a signature/inspection check) rather than the code's own
formula.  Two of these guard a real code change (fail-before / pass-after):

* ``strehl_marechal`` now returns a Python float for scalar input (was a
  0-d ndarray) -- S3-19.
* the deprecated Zemax-loader aliases now announce removal in v6.0, not the
  already-passed v5.0 -- S4-17.

The remainder pin conventions the audit found mis-documented so a future
drift re-breaks the test rather than silently contradicting the docs.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# S1-5 -- waveplate slow-axis phase exp(+i*retardance) (CONVENTIONS.md sec 7)
# ---------------------------------------------------------------------------
def _x_pol_field():
    from lumenairy.elements.polarization import JonesField
    return JonesField(np.ones((1, 1), complex), np.zeros((1, 1), complex),
                      dx=1e-6)


def test_s1_5_qwp_fast_axis_plus45_gives_left_circular_s3_minus1():
    """QWP (retardance pi/2) with fast axis at +45 deg on x-pol yields
    S3 = -1 ('left'), and -45 deg yields S3 = +1 ('right').

    Independent oracle: with the library convention the slow axis picks up
    exp(+i*retardance).  For a QWP at +45 deg on [1, 0] that maps to
    Ey/Ex = -i (S3 = -1); at -45 deg to Ey/Ex = +i (S3 = +1).  The
    CONVENTIONS.md row was corrected from exp(-i*retardance) to
    exp(+i*retardance) to match this.
    """
    from lumenairy.elements.polarization import apply_waveplate, stokes_parameters

    g_plus = apply_waveplate(_x_pol_field(), np.pi / 2, np.pi / 4)
    s_plus = stokes_parameters(g_plus)
    assert float(np.asarray(s_plus['S3']).ravel()[0]) == pytest.approx(
        -1.0, abs=1e-9)
    assert complex(g_plus.Ey.ravel()[0] / g_plus.Ex.ravel()[0]) == \
        pytest.approx(-1j, abs=1e-9)

    g_minus = apply_waveplate(_x_pol_field(), np.pi / 2, -np.pi / 4)
    s_minus = stokes_parameters(g_minus)
    assert float(np.asarray(s_minus['S3']).ravel()[0]) == pytest.approx(
        +1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# S1-6 -- Berreman _solve_core takes RAW public eps (no internal conjugation)
# ---------------------------------------------------------------------------
def test_s1_6_berreman_public_path_raw_eps_absorbs_with_positive_imag():
    """A single absorbing isotropic slab (Im(eps) > 0) must LOSE power
    (R + T < 1 per incident polarization); a gain slab (Im(eps) < 0) must
    exceed 1.

    This pins the corrected ``_solve_core`` docstring: the public cascade
    receives RAW eps.  Had it silently conjugated (the old docstring's
    claim), absorption would flip to gain and this oracle would invert.
    """
    from lumenairy.elements.berreman import berreman_jones_1d

    wl = 0.633e-6
    eps_abs = complex((1.5 + 0.05j) ** 2)   # kappa > 0 -> absorbing
    R, T, _Jr, _Jt = berreman_jones_1d([(eps_abs, 0.5e-6)], 1.0, 1.0, wl,
                                        angle=0.0)
    assert float(R[0] + T[0]) < 1.0
    assert float(R[1] + T[1]) < 1.0

    eps_gain = complex((1.5 - 0.05j) ** 2)  # kappa < 0 -> gain
    Rg, Tg, _a, _b = berreman_jones_1d([(eps_gain, 0.5e-6)], 1.0, 1.0, wl,
                                       angle=0.0)
    assert float(Rg[0] + Tg[0]) > 1.0


# ---------------------------------------------------------------------------
# S1-19 -- _tensor_convolutions isotropic reduction is the Li-1996 mixed rule
# ---------------------------------------------------------------------------
def test_s1_19_tensor_convolutions_scalar_reduction_is_mixed_rule():
    """For a PATTERNED scalar cell the corrected docstring says
    Cxx = [[1/eps]]^-1 (inverse rule, wall-normal x) and Cyy = [[eps]]
    (direct rule, tangential y), so Cxx != Cyy; they coincide only for a
    UNIFORM cell.  (The old docstring wrongly said Cxx = Cyy = [[eps]].)
    """
    from lumenairy.elements.rcwa._core import _tensor_convolutions

    n_orders = 3
    dim = 2 * n_orders + 1

    patt = np.array([2.0] * 8 + [5.0] * 8, dtype=float)
    prof = {k: patt.copy() for k in ('xx', 'yy', 'zz')}
    prof['xy'] = np.zeros_like(patt)
    prof['yx'] = np.zeros_like(patt)
    Cxx, Cxy, Cyx, Cyy, _EZZ = _tensor_convolutions(prof, n_orders)
    # off-diagonal vanishes for a scalar cell
    assert np.max(np.abs(Cxy)) == 0.0
    assert np.max(np.abs(Cyx)) == 0.0
    # wall-normal (inverse) and tangential (direct) rules DIFFER when patterned
    assert np.max(np.abs(Cxx - Cyy)) > 1e-3
    # independent inverse-rule oracle: Cxx == inv([[1/eps]])
    from lumenairy.elements.rcwa._core import _inv_toeplitz_of_profile, _toeplitz_of_profile
    assert np.allclose(Cxx, _inv_toeplitz_of_profile(patt, n_orders),
                       atol=1e-12)
    assert np.allclose(Cyy, _toeplitz_of_profile(patt, n_orders), atol=1e-12)

    uni = np.full(16, 3.0)
    profu = {k: uni.copy() for k in ('xx', 'yy', 'zz')}
    profu['xy'] = np.zeros_like(uni)
    profu['yx'] = np.zeros_like(uni)
    Cxx2, _a, _b, Cyy2, _c = _tensor_convolutions(profu, n_orders)
    assert np.allclose(Cxx2, Cyy2, atol=1e-12)
    assert np.allclose(Cyy2, 3.0 * np.eye(dim), atol=1e-12)


def test_s1_19_rcwa_jones_2d_uses_out_of_plane_components():
    """rcwa_jones_2d docstring previously said only the in-plane block +
    zz are used; OOP tensors have been supported since v5.14.1.  Zeroing
    the off-plane couplings of a tilted-uniaxial cell must CHANGE the
    zeroth-order Jones -- proof they are consumed, not ignored.
    """
    from lumenairy.elements.rcwa._core import uniaxial_tensor
    from lumenairy.elements.rcwa.twod import rcwa_jones_2d

    wl = 0.633e-6
    tens = uniaxial_tensor(1.5, 1.8, np.deg2rad(40.0), phi=0.0)
    assert abs(tens[0, 2]) > 0.1  # real off-plane coupling present
    Sx = Sy = 24
    cell = np.tile(tens[None, None, :, :], (Sx, Sy, 1, 1)).astype(complex)
    cell[:, Sy // 2:, :, :] = np.eye(3) * 2.25   # make it a real 2-D grating

    _o, _Re, _Te, Jr = rcwa_jones_2d(
        1e-6, 1e-6, cell, 1.0, 1.0, 0.3e-6, wl,
        theta=0.2, phi=0.1, n_orders_x=2, n_orders_y=2)

    cell_ip = cell.copy()
    for a, b in ((0, 2), (2, 0), (1, 2), (2, 1)):
        cell_ip[:, :, a, b] = 0.0
    _o2, _Re2, _Te2, Jr2 = rcwa_jones_2d(
        1e-6, 1e-6, cell_ip, 1.0, 1.0, 0.3e-6, wl,
        theta=0.2, phi=0.1, n_orders_x=2, n_orders_y=2)

    assert np.max(np.abs(Jr - Jr2)) > 1e-4


# ---------------------------------------------------------------------------
# S2-19 -- ray_subsample default is 8; GBD adaptive keeps ALL coarse cells
# ---------------------------------------------------------------------------
def test_s2_19_ray_subsample_signature_default_is_8():
    from lumenairy.elements._lens_traced import apply_real_lens_traced
    sig = inspect.signature(apply_real_lens_traced)
    assert sig.parameters['ray_subsample'].default == 8


def test_s2_19_gbd_adaptive_keeps_all_coarse_beamlets():
    """decompose_field_adaptive summary was corrected: coarse cells are
    KEPT and fine residual beamlets ADD to them.  Pin that the reported
    n_coarse equals the full uniform coarse count and n_total = coarse +
    fine (no cells dropped).
    """
    from lumenairy.propagators.gbd import decompose_field_adaptive, decompose_field_to_beamlets

    N, dx = 64, 2e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (20e-6) ** 2).astype(complex)
    E[np.abs(X) > 40e-6] = 0.0    # hard edge -> triggers refinement

    base = 4
    _bundle, stats = decompose_field_adaptive(
        E, dx, wavelength=1.55e-6, base_step=base, refine_step=1,
        return_stats=True)
    full_coarse = decompose_field_to_beamlets(
        E, dx, wavelength=1.55e-6, waist_factor=1.5 * base, sample_step=base)

    assert stats['n_coarse'] == len(full_coarse)
    assert stats['n_total'] == stats['n_coarse'] + stats['n_fine']
    assert stats['n_fine'] > 0     # the hard edge did refine


# ---------------------------------------------------------------------------
# S3-19 -- strehl_marechal scalar -> float; compute_psf default is 'power'
# ---------------------------------------------------------------------------
def test_s3_19_strehl_marechal_scalar_returns_python_float():
    """Fail-before / pass-after: a scalar input previously leaked a numpy
    scalar (0-d / np.float64) rather than the documented Python 'float'.
    ``type(out) is float`` is strict -- np.float64 is a float SUBCLASS so
    isinstance would not catch the regression, but the exact-type check
    distinguishes the pre-fix numpy scalar from the post-fix pure float.
    """
    from lumenairy.analysis.strehl import strehl_marechal

    out = strehl_marechal(1.0 / 14.0)
    assert type(out) is float
    # independent Marechal oracle
    assert out == pytest.approx(np.exp(-(2 * np.pi / 14.0) ** 2), rel=1e-12)

    arr = strehl_marechal(np.array([0.05, 0.1]))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2,)


def test_s3_19_compute_psf_default_normalize_is_power():
    """plot_psf docstring was corrected: compute_psf defaults to power-,
    not peak-, normalization.  Pin the actual default.
    """
    from lumenairy.analysis.psf_mtf_otf import compute_psf
    assert inspect.signature(compute_psf).parameters['normalize'].default \
        == 'power'


# ---------------------------------------------------------------------------
# S4-17 -- deprecated Zemax aliases now target v6.0 (not the passed v5.0)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('alias_name', ['load_zmx_prescription',
                                        'load_zemax_prescription_txt'])
def test_s4_17_zemax_alias_deprecation_targets_v6_not_v5(alias_name):
    import lumenairy as la

    alias = getattr(la, alias_name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            alias('this_file_does_not_exist_zzz.zmx')
        except Exception:
            pass   # the shim warns BEFORE the (failing) real load
    msgs = [str(w.message) for w in caught
            if 'deprecated alias' in str(w.message)]
    assert msgs, 'expected a deprecation warning from the alias'
    assert 'v6.0' in msgs[0]
    assert 'v5.0' not in msgs[0]


# ---------------------------------------------------------------------------
# S5-10 -- 2-D entries reject angle=; 1-D entries accept angle/theta alias
# ---------------------------------------------------------------------------
def test_s5_10_rcwa_jones_2d_rejects_angle_keyword():
    from lumenairy.elements.rcwa.twod import rcwa_jones_2d
    cell = np.tile(np.eye(3)[None, None] * 2.25, (16, 16, 1, 1)).astype(complex)
    with pytest.raises(TypeError):
        rcwa_jones_2d(1e-6, 1e-6, cell, 1.0, 1.0, 0.2e-6, 0.5e-6, angle=0.1)


def test_s5_10_rcwa_jones_1d_theta_is_angle_alias():
    """1-D entries accept both angle and theta; theta is the same number
    and wins when both are given (documented in CONVENTIONS.md sec 7.1).
    """
    from lumenairy.elements.rcwa.oned import rcwa_jones_1d

    args = (1.0e-6,
            np.diag([2.5, 2.5, 2.5]).astype(complex),
            np.diag([1.0, 1.0, 1.0]).astype(complex),
            1.0, 1.0, 0.3e-6, 0.5, 0.633e-6)
    out_angle = rcwa_jones_1d(*args, angle=0.25, n_orders=5)
    out_theta = rcwa_jones_1d(*args, theta=0.25, n_orders=5)
    # theta == angle: identical result
    for a, b in zip(out_angle, out_theta):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-12)
    # theta overrides a conflicting angle (theta wins -> equals theta-only)
    out_both = rcwa_jones_1d(*args, angle=0.0, theta=0.25, n_orders=5)
    for a, b in zip(out_both, out_theta):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-12)
