"""v5.12.0 -- COVARIANT oblique-coordinate slant path (Li 1999 JOSA A 16:2521).

`pmm_jones_1d_slanted(..., factorization='covariant')` routes the slanted layer
through the Li covariant oblique-coordinate generator: the slanted wall becomes a
coordinate surface, so the TM/p-pol channel converges SPECTRALLY (vertical-grade)
instead of the convection path's ALGEBRAIC ~1e-4 floor -- the SAME physical answer
(cross-validated), ~100-2400x fewer degrees for matched accuracy.

Validated in these tests:
  - slant=0 reduces to pmm_jones_1d (diagonal + coupled, R/T + Jones incl cross-pol)
  - slant!=0 matches the convection path (same truth), energy-conserving
  - SPECTRAL convergence: covariant self-conv << convection self-conv
  - lossy + oblique
  - default ('convection') byte-identical; out-of-plane / bad-arg guards
"""
import numpy as np
import pytest

from lumenairy.elements.pmm import (
    pmm_jones_1d,
    pmm_jones_1d_segments,
    pmm_jones_1d_slanted,
    pmm_jones_1d_slanted_segments,
)

_GEOM = dict(period=1e-6, n_substrate=1.5, n_superstrate=1.0, depth=0.5e-6,
             duty_cycle=0.5, wavelength=0.633e-6)


def _diag(exx, eyy, ezz):
    return np.diag([exx, eyy, ezz]).astype(np.complex128)


def _coupled(exx, eyy, ezz, off):
    m = np.diag([exx, eyy, ezz]).astype(np.complex128)
    m[0, 1] = m[1, 0] = off
    return m


def _slant(er, eg, slant, deg, fac, ang=0.0, ff=21):
    return pmm_jones_1d_slanted(
        _GEOM["period"], er, eg, _GEOM["n_substrate"], _GEOM["n_superstrate"],
        _GEOM["depth"], _GEOM["duty_cycle"], _GEOM["wavelength"], slant,
        angle=ang, degree=deg, far_field_orders=ff, stabilize=False,
        factorization=fac)


def _perorder(o1, A1, ch, o2, A2):
    i1 = {int(m): k for k, m in enumerate(o1.tolist())}
    i2 = {int(m): k for k, m in enumerate(o2.tolist())}
    return max(abs(float(A1[ch][i1[m]]) - float(A2[ch][i2[m]]))
               for m in i1 if m in i2)


def test_covariant_slant0_reduces_to_jones_1d_diagonal():
    er, eg = _diag(4.0, 2.25, 2.0), _diag(2.0, 2.0, 2.0)
    o, R, T, J = _slant(er, eg, 0.0, 22, "covariant")
    oR, ReR, TeR, JR = pmm_jones_1d(
        _GEOM["period"], er, eg, _GEOM["n_substrate"], _GEOM["n_superstrate"],
        _GEOM["depth"], _GEOM["duty_cycle"], _GEOM["wavelength"], degree=22)
    for ch in (0, 1):
        assert _perorder(o, R, ch, oR, ReR) < 1e-5
        assert _perorder(o, T, ch, oR, TeR) < 1e-5
    assert np.max(np.abs(J - JR)) < 1e-5


def test_covariant_slant0_reduces_to_jones_1d_coupled_crosspol():
    er, eg = _coupled(4.0, 2.25, 2.0, 0.5), _diag(2.0, 2.0, 2.0)
    o, R, T, J = _slant(er, eg, 0.0, 22, "covariant")
    oR, ReR, TeR, JR = pmm_jones_1d(
        _GEOM["period"], er, eg, _GEOM["n_substrate"], _GEOM["n_superstrate"],
        _GEOM["depth"], _GEOM["duty_cycle"], _GEOM["wavelength"], degree=22)
    # cross-pol (off-diagonal Jones) must match in the PUBLIC convention
    assert np.max(np.abs(J - JR)) < 1e-5
    for ch in (0, 1):
        assert _perorder(o, R, ch, oR, ReR) < 1e-5


@pytest.mark.parametrize("slant_deg", [15.0, 30.0, 45.0])
def test_covariant_matches_convection_same_truth(slant_deg):
    """Covariant and convection are independent formulations -> same physical
    answer (converged convection at high degree)."""
    er, eg = _coupled(4.0, 2.25, 2.0, 0.4), _diag(2.0, 2.0, 2.0)
    sl = np.deg2rad(slant_deg)
    o_v, R_v, T_v, _ = _slant(er, eg, sl, 30, "covariant")
    o_c, R_c, T_c, _ = _slant(er, eg, sl, 38, "convection")  # high-deg convection
    for ch in (0, 1):
        assert _perorder(o_v, R_v, ch, o_c, R_c) < 5e-4
        assert _perorder(o_v, T_v, ch, o_c, T_c) < 5e-4


@pytest.mark.parametrize("slant_deg", [30.0, 45.0])
def test_covariant_energy_conserved(slant_deg):
    er, eg = _coupled(4.0, 2.25, 2.0, 0.4), _diag(2.0, 2.0, 2.0)
    o, R, T, _ = _slant(er, eg, np.deg2rad(slant_deg), 26, "covariant")
    for ch in (0, 1):
        assert abs(np.sum(R[ch]) + np.sum(T[ch]) - 1.0) < 1e-4


def test_covariant_is_spectral_vs_convection_algebraic():
    """The headline result: at matched (low) degree the covariant TM error is
    >=20x below convection, and its self-convergence is spectral."""
    er, eg = _diag(4.0, 2.25, 2.0), _diag(2.0, 2.0, 2.0)
    sl = np.deg2rad(30.0)
    o_ref, R_ref, _, _ = _slant(er, eg, sl, 38, "covariant")     # converged
    o_v, R_v, _, _ = _slant(er, eg, sl, 22, "covariant")
    o_c, R_c, _, _ = _slant(er, eg, sl, 22, "convection")
    cov_err = _perorder(o_v, R_v, 0, o_ref, R_ref)               # TM channel
    con_err = _perorder(o_c, R_c, 0, o_ref, R_ref)
    assert cov_err < 1e-5                       # covariant ~converged at deg22
    assert con_err > 20 * cov_err               # convection far from converged


def test_covariant_lossy_and_oblique():
    er = _coupled(4.0 + 0.3j, 2.25 + 0.1j, 2.0 + 0.2j, 0.3)
    eg = _diag(2.0, 2.0, 2.0)
    o, R, T, J = _slant(er, eg, np.deg2rad(20.0), 26, "covariant",
                        ang=np.deg2rad(15.0))
    assert np.all(np.isfinite(J))
    for ch in (0, 1):                           # lossy -> R+T < 1, all finite >=0
        s = np.sum(R[ch]) + np.sum(T[ch])
        assert 0.0 <= s <= 1.0 + 1e-9
        assert np.all(R[ch] >= -1e-12) and np.all(T[ch] >= -1e-12)


def test_default_is_convection_byte_identical():
    er, eg = _diag(4.0, 2.25, 2.0), _diag(2.0, 2.0, 2.0)
    a = _slant(er, eg, np.deg2rad(30.0), 20, "convection")
    b = pmm_jones_1d_slanted(
        _GEOM["period"], er, eg, _GEOM["n_substrate"], _GEOM["n_superstrate"],
        _GEOM["depth"], _GEOM["duty_cycle"], _GEOM["wavelength"],
        np.deg2rad(30.0), degree=20, stabilize=False)   # no factorization kwarg
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_covariant_rejects_out_of_plane():
    er = _diag(4.0, 2.25, 2.0)
    er[0, 2] = er[2, 0] = 0.3                    # out-of-plane
    eg = _diag(2.0, 2.0, 2.0)
    with pytest.raises(NotImplementedError, match="OUT-OF-PLANE"):
        _slant(er, eg, np.deg2rad(30.0), 16, "covariant")


def test_invalid_factorization_raises():
    er, eg = _diag(4.0, 2.25, 2.0), _diag(2.0, 2.0, 2.0)
    with pytest.raises(ValueError, match="factorization"):
        _slant(er, eg, np.deg2rad(30.0), 16, "bogus")


def test_auto_picks_covariant_for_inplane_slant():
    """'auto' routes an in-plane slanted cell to the SPECTRAL covariant path."""
    er, eg = _diag(4.0, 2.25, 2.0), _diag(2.0, 2.0, 2.0)
    a = _slant(er, eg, np.deg2rad(30.0), 24, "auto")
    cov = _slant(er, eg, np.deg2rad(30.0), 24, "covariant")
    for x, y in zip(a, cov):
        assert np.array_equal(x, y)


def test_auto_picks_convection_for_out_of_plane():
    """'auto' falls back to convection for an out-of-plane cell (no raise)."""
    er = _diag(4.0, 2.25, 2.0)
    er[0, 2] = er[2, 0] = 0.3
    eg = _diag(2.0, 2.0, 2.0)
    a = _slant(er, eg, np.deg2rad(30.0), 20, "auto")
    con = _slant(er, eg, np.deg2rad(30.0), 20, "convection")
    for x, y in zip(a, con):
        assert np.array_equal(x, y)


def test_auto_picks_convection_for_vertical():
    """'auto' uses convection (the existing routing) at slant=0."""
    er, eg = _diag(4.0, 2.25, 2.0), _diag(2.0, 2.0, 2.0)
    a = _slant(er, eg, 0.0, 20, "auto")
    con = _slant(er, eg, 0.0, 20, "convection")
    for x, y in zip(a, con):
        assert np.array_equal(x, y)


# ---------------------------------------------------------------------------
# SEGMENTS (multi-region) covariant path
# ---------------------------------------------------------------------------
_SEG3 = [(0.3, _diag(4.0, 2.25, 2.0)), (0.3, _coupled(3.0, 2.5, 2.2, 0.3)),
         (0.4, _diag(2.0, 2.0, 2.0))]


def _seg(segments, slant, deg, fac, ang=0.0):
    return pmm_jones_1d_slanted_segments(
        _GEOM["period"], segments, _GEOM["n_substrate"], _GEOM["n_superstrate"],
        _GEOM["depth"], _GEOM["wavelength"], slant, angle=ang, degree=deg,
        stabilize=False, factorization=fac)


def test_covariant_segments_slant0_reduces_to_jones_1d_segments():
    """3-region covariant at slant=0 == pmm_jones_1d_segments (the pre-reverse
    fix cancels the internal _segment_elem_bnds [::-1] mirror)."""
    o, R, T, J = _seg(_SEG3, 0.0, 26, "covariant")
    oR, ReR, TeR, JR = pmm_jones_1d_segments(
        _GEOM["period"], _SEG3, _GEOM["n_substrate"], _GEOM["n_superstrate"],
        _GEOM["depth"], _GEOM["wavelength"], degree=26)
    for ch in (0, 1):
        assert _perorder(o, R, ch, oR, ReR) < 1e-5
    assert np.max(np.abs(J - JR)) < 1e-5


def test_covariant_segments_2seg_equals_binary():
    """2 equal segments via the covariant segments path == the binary covariant
    path, bit-for-bit (the multi-region assembly is exact)."""
    er, eg = _diag(4.0, 2.25, 2.0), _diag(2.0, 2.0, 2.0)
    o_s, R_s, T_s, J_s = _seg([(0.5, er), (0.5, eg)], np.deg2rad(30.0), 22,
                              "covariant")
    o_b, R_b, T_b, J_b = _slant(er, eg, np.deg2rad(30.0), 22, "covariant")
    assert np.array_equal(R_s, R_b)
    assert np.array_equal(T_s, T_b)
    assert np.array_equal(J_s, J_b)


@pytest.mark.parametrize("slant_deg,ang_deg", [(30.0, 0.0), (45.0, 12.0)])
def test_covariant_segments_matches_convection(slant_deg, ang_deg):
    o_v, R_v, T_v, _ = _seg(_SEG3, np.deg2rad(slant_deg), 30, "covariant",
                            ang=np.deg2rad(ang_deg))
    o_c, R_c, T_c, _ = _seg(_SEG3, np.deg2rad(slant_deg), 42, "convection",
                            ang=np.deg2rad(ang_deg))
    for ch in (0, 1):
        assert _perorder(o_v, R_v, ch, o_c, R_c) < 5e-4
    for ch in (0, 1):
        assert abs(np.sum(R_v[ch]) + np.sum(T_v[ch]) - 1.0) < 1e-4


def test_covariant_segments_rejects_out_of_plane():
    seg = list(_SEG3)
    oop = _diag(3.0, 2.5, 2.2)
    oop[0, 2] = oop[2, 0] = 0.3
    seg[1] = (0.3, oop)
    with pytest.raises(NotImplementedError, match="OUT-OF-PLANE"):
        _seg(seg, np.deg2rad(30.0), 16, "covariant")


def test_covariant_segments_invalid_factorization_raises():
    with pytest.raises(ValueError, match="factorization"):
        _seg(_SEG3, np.deg2rad(30.0), 16, "bogus")
