"""Multi-region (segmented) PMM -- ``pmm_efficiency_1d_segments`` (scalar) and
``pmm_jones_1d_segments`` (anisotropic Jones): arbitrary piecewise-constant 1-D
profiles, the spectral-element analogue of ``rcwa_jones_1d_segments``.

Oracles: ``rcwa_jones_1d_segments`` (the FMM) at matched angle.  The segmented
solver uses the noise-robust / z-Poynting-flux forward selector even at normal
incidence (the many-element multi-region cell has dense isolated-degree
resonances the legacy ``Im(q)`` branch cannot handle), so the binary path stays
bit-identical while the multi-region path is clean.  Normal AND oblique; the
2-segment case reproduces the binary ridge/groove special case.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.pmm import (
    pmm_efficiency_1d_segments,
    pmm_jones_1d,
    pmm_jones_1d_segments,
    pmm_jones_1d_slanted_segments,
)
from lumenairy.elements.rcwa import (
    binary_grating_segments,
    rcwa_jones_1d_segments,
    uniaxial_tensor,
)

_C = np.complex128
GR = np.eye(3, dtype=_C)
PERIOD, DEPTH, WL = 1.2e-6, 0.4e-6, 0.633e-6
NSUB, NSUP = 1.5, 1.0


def _jmax(J1, J2):
    return float(np.max(np.abs(J1 - J2)))


# --------------------------------------------------------------------------- #
# anisotropic Jones segments vs the FMM oracle
# --------------------------------------------------------------------------- #
def _three_region():
    return [(0.25, GR), (0.5, 2.25 * GR), (0.25, GR)]


def _four_region_lc():
    lc = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=0.3)
    return [(0.25, 2.0 * GR), (0.25, lc), (0.25, 2.0 * GR), (0.25, lc)]


@pytest.mark.parametrize("ang_deg", [0.0, 20.0, 45.0])
def test_three_region_jones_matches_fmm(ang_deg):
    segs = _three_region()
    ang = np.radians(ang_deg)
    o, R, T, J = pmm_jones_1d_segments(PERIOD, segs, NSUB, NSUP, DEPTH, WL,
                                       angle=ang, degree=20)
    oo, Ro, To, Jo = rcwa_jones_1d_segments(PERIOD, segs, NSUB, NSUP, DEPTH, WL,
                                            angle=ang, n_orders=41)
    assert _jmax(J, Jo) < 5e-4
    assert np.allclose(R.sum(axis=1) + T.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("ang_deg", [0.0, 30.0])
def test_four_region_lc_device_matches_fmm(ang_deg):
    # the tunable-LC device class (metal-tooth | LC | metal-tooth | LC)
    segs = _four_region_lc()
    ang = np.radians(ang_deg)
    o, R, T, J = pmm_jones_1d_segments(PERIOD, segs, NSUB, NSUP, DEPTH, WL,
                                       angle=ang, degree=20)
    oo, Ro, To, Jo = rcwa_jones_1d_segments(PERIOD, segs, NSUB, NSUP, DEPTH, WL,
                                            angle=ang, n_orders=41)
    assert _jmax(J, Jo) < 5e-4
    assert np.allclose(R.sum(axis=1) + T.sum(axis=1), 1.0, atol=1e-6)


def test_lossy_anisotropic_segments_matches_fmm():
    lc = uniaxial_tensor(1.5 + 0.02j, 1.8 + 0.05j, np.pi / 2, phi=0.25)
    segs = [(0.3, GR), (0.4, lc), (0.3, 2.0 * GR)]
    o, R, T, J = pmm_jones_1d_segments(PERIOD, segs, NSUB, NSUP, DEPTH, WL,
                                       angle=np.radians(25.0), degree=20)
    oo, Ro, To, Jo = rcwa_jones_1d_segments(PERIOD, segs, NSUB, NSUP, DEPTH, WL,
                                            angle=np.radians(25.0), n_orders=41)
    assert _jmax(J, Jo) < 5e-4
    # PMM and FMM agree on the absorbed power
    assert np.max(np.abs((R.sum(1) + T.sum(1)) - (Ro.sum(1) + To.sum(1)))) < 5e-4


def test_asymmetric_oblique_per_order_matches_fmm():
    # The mirror-handedness regression guard: an ASYMMETRIC profile at OBLIQUE
    # incidence (a 3-level monotone staircase) must match the FMM ORDER-BY-ORDER
    # -- a reversed x-layout would give the mirror-image spectrum (the binary
    # 2-region cell is always mirror-symmetric, so only 3+ regions catch this).
    segs = [(1 / 3, (1.4 ** 2) * GR), (1 / 3, (1.6 ** 2) * GR),
            (1 / 3, (1.8 ** 2) * GR)]
    ang = np.radians(25.0)
    o, R, T, J = pmm_jones_1d_segments(1.5e-6, segs, NSUB, NSUP, DEPTH, WL,
                                       angle=ang, degree=22)
    oo, Ro, To, Jo = rcwa_jones_1d_segments(1.5e-6, segs, NSUB, NSUP, DEPTH, WL,
                                            angle=ang, n_orders=81)

    def _pe(o1, A1, o2, A2):
        e = 0.0
        for row in (0, 1):
            d1 = {int(a): float(b) for a, b in zip(o1, A1[row])}
            d2 = {int(a): float(b) for a, b in zip(o2, A2[row])}
            for k in set(d1) | set(d2):
                e = max(e, abs(d1.get(k, 0.0) - d2.get(k, 0.0)))
        return e
    assert _pe(o, R, oo, Ro) < 1e-3
    assert _pe(o, T, oo, To) < 1e-3
    # and not the mirror: reversing the profile must give a DIFFERENT spectrum
    _o2, R2, _t2, _j2 = pmm_jones_1d_segments(
        1.5e-6, list(reversed(segs)), NSUB, NSUP, DEPTH, WL, angle=ang,
        degree=22)
    assert _pe(o, R, o, R2) > 1e-3                  # asymmetry is resolved


def test_scalar_eps_segment_promoted_to_isotropic():
    # a scalar eps in a segment is promoted to an isotropic tensor
    segs = [(0.5, 2.25), (0.5, 1.0)]                # scalar entries
    o, R, T, J = pmm_jones_1d_segments(PERIOD, segs, NSUB, NSUP, DEPTH, WL,
                                       angle=np.radians(15.0), degree=18)
    oo, Ro, To, Jo = rcwa_jones_1d_segments(PERIOD, segs, NSUB, NSUP, DEPTH, WL,
                                            angle=np.radians(15.0), n_orders=41)
    assert _jmax(J, Jo) < 5e-4


def test_two_segment_matches_binary_via_builder():
    # binary_grating_segments output must solve like the binary pmm_jones_1d
    # (both converged) -- match to the FMM tolerance.
    lc = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=0.3)
    segs = binary_grating_segments(0.5, lc, GR)
    o, R, T, J = pmm_jones_1d_segments(0.8e-6, segs, NSUB, NSUP, 0.5e-6, 0.55e-6,
                                       angle=np.radians(20.0), degree=20)
    ob, Rb, Tb, Jb = pmm_jones_1d(0.8e-6, lc, GR, NSUB, NSUP, 0.5e-6, 0.5,
                                  0.55e-6, angle=np.radians(20.0), degree=20)
    assert _jmax(J, Jb) < 1e-3


# --------------------------------------------------------------------------- #
# scalar efficiency segments vs the FMM (the s/p channels of the Jones oracle)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pol,row", [("te", 1), ("tm", 0)])
def test_scalar_segments_matches_fmm(pol, row):
    # 3-level isotropic staircase; the scalar TE/TM result must equal the
    # s/p channel of rcwa_jones_1d_segments.
    ns = [1.4, 1.6, 1.8]
    segs_n = [(1 / 3, n) for n in ns]
    segs_eps = [(1 / 3, (n ** 2) * GR) for n in ns]
    ang = np.radians(20.0)
    o, R, T = pmm_efficiency_1d_segments(1.5e-6, segs_n, NSUB, NSUP, DEPTH, WL,
                                         angle=ang, polarization=pol, degree=20)
    oo, Ro, To, Jo = rcwa_jones_1d_segments(1.5e-6, segs_eps, NSUB, NSUP, DEPTH,
                                            WL, angle=ang, n_orders=41)
    assert abs(float(R.sum()) - float(Ro[row].sum())) < 1e-3
    assert abs(float(T.sum()) - float(To[row].sum())) < 1e-3
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-4    # lossless


def test_uniform_segments_are_transparent():
    # N identical vacuum regions -> R=0, T=1 for any N (a regression guard on the
    # periodic multi-element assembly)
    for N in (2, 3, 4, 5):
        segs = [(1.0 / N, GR)] * N
        o, R, T, J = pmm_jones_1d_segments(1.0e-6, segs, 1.0, 1.0, DEPTH, WL,
                                           angle=np.radians(20.0), degree=14)
        assert np.allclose(R.sum(axis=1) + T.sum(axis=1), 1.0, atol=1e-9)
        assert np.max(np.abs(R)) < 1e-9


# --------------------------------------------------------------------------- #
# validation / guards
# --------------------------------------------------------------------------- #
def test_widths_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        pmm_jones_1d_segments(PERIOD, [(0.3, GR), (0.3, 2.0 * GR)], NSUB, NSUP,
                              DEPTH, WL)


def test_rejects_slanted_out_of_plane_segment():
    # VERTICAL out-of-plane multi-region is now supported (see
    # test_segments_out_of_plane_matches_rcwa); only a SLANTED out-of-plane
    # grating is not yet per-order-validated and raises.
    op = uniaxial_tensor(1.5, 1.7, np.pi / 4, phi=0.3)     # tilted -> off-plane
    with pytest.raises(ValueError, match="out-of-plane"):
        pmm_jones_1d_slanted_segments(PERIOD, [(0.5, op), (0.5, GR)], NSUB, NSUP,
                                      DEPTH, WL, np.deg2rad(30.0))


def test_rejects_bad_tensor_shape():
    with pytest.raises(ValueError, match="scalar or a"):
        pmm_jones_1d_segments(PERIOD, [(0.5, np.eye(2)), (0.5, GR)], NSUB, NSUP,
                              DEPTH, WL)


def test_segments_exported_top_level():
    import lumenairy as la
    for name in ("pmm_efficiency_1d_segments", "pmm_jones_1d_segments"):
        assert hasattr(la, name) and name in la.__all__


def test_segments_out_of_plane_matches_rcwa():
    """A MULTI-REGION grating with OUT-OF-PLANE coupling (eps_xz/eyz != 0) now
    routes through the native full-3x3 metric generator (region-count-agnostic)
    and matches rcwa_jones_1d_segments per-order to <1.5e-3; conserves energy."""
    e1 = np.diag([2.25, 2.25, 2.25]).astype(_C)
    e1[0, 2] = e1[2, 0] = 0.3
    e2 = np.diag([4.0, 3.6, 4.0]).astype(_C)
    e2[0, 2] = e2[2, 0] = 0.4
    e2[1, 2] = e2[2, 1] = 0.2
    segs = [(0.2, e1), (0.5, e2), (0.3, GR)]
    o, R, T, J = pmm_jones_1d_segments(PERIOD, segs, 1.5, 1.0, DEPTH, WL,
                                       degree=16, elements_per_region=6)
    oR, RR, TR, JR = rcwa_jones_1d_segments(PERIOD, segs, 1.5, 1.0, DEPTH, WL,
                                            n_orders=61)
    c = np.intersect1d(o, oR)
    i1 = {int(x): k for k, x in enumerate(o)}
    i2 = {int(x): k for k, x in enumerate(oR)}
    d = max(max(np.max(np.abs(R[:, i1[int(m)]] - RR[:, i2[int(m)]])),
                np.max(np.abs(T[:, i1[int(m)]] - TR[:, i2[int(m)]]))) for m in c)
    assert d < 1.5e-3
    assert abs(R[0].sum() + T[0].sum() - 1.0) < 1e-3
    assert abs(R[1].sum() + T[1].sum() - 1.0) < 1e-3
