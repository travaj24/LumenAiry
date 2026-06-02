"""v5.11.0 RCWA device-modeling ergonomics (audit W2 + W3).

W3 -- 1-D grating builders that emit the ``segments`` list for
``rcwa_jones_1d_segments`` (so users don't hand-roll region masks).
W2 -- reflective-Jones device helpers (metasurface-as-Jones-element): the
PBS->QWP@45->grating->QWP@45->PBS out-coupling FOM, and retardance/diattenuation
extraction from a 2x2 Jones reflection.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.rcwa import (
    binary_grating_segments,
    grating_segments,
    interdigitated_grating_segments,
    jones_retardance_diattenuation,
    rcwa_jones_1d,
    rcwa_jones_1d_segments,
    reflective_outcoupling,
)

WL = 0.8e-6
PERIOD = 1.0e-6
DEPTH = 0.3e-6


def _eye(eps):
    return eps * np.eye(3, dtype=complex)


# ---------------------------------------------------------------------------
# W3 -- segment builders
# ---------------------------------------------------------------------------

def test_binary_grating_segments_matches_rcwa_jones_1d():
    # the builder must emit the 2-region list equivalent to rcwa_jones_1d.
    duty, e_r, e_g = 0.4, 2.5 ** 2, 1.0
    segs = binary_grating_segments(duty, e_r, e_g)
    assert segs == [(duty, e_r), (1.0 - duty, e_g)]
    o1, R1, T1, J1 = rcwa_jones_1d(PERIOD, _eye(e_r), _eye(e_g), 1.5, 1.0, DEPTH,
                                   duty, WL, n_orders=9)
    o2, R2, T2, J2 = rcwa_jones_1d_segments(PERIOD, segs, 1.5, 1.0, DEPTH, WL,
                                            n_orders=9)
    assert np.allclose(R1, R2, atol=1e-12)
    assert np.allclose(T1, T2, atol=1e-12)
    assert np.allclose(J1, J2, atol=1e-12)


def test_grating_segments_normalizes_relative_widths():
    # relative widths are normalized to sum to 1.
    segs = grating_segments([2.0, 3.0], [4.0, 1.0])
    assert np.isclose(sum(w for w, _ in segs), 1.0)
    assert np.isclose(segs[0][0], 0.4) and np.isclose(segs[1][0], 0.6)
    # and it round-trips through the solver identically to the binary builder.
    o1, R1, T1, J1 = rcwa_jones_1d_segments(PERIOD, segs, 1.0, 1.0, DEPTH, WL,
                                            n_orders=7)
    o2, R2, T2, J2 = rcwa_jones_1d_segments(
        PERIOD, binary_grating_segments(0.4, 4.0, 1.0), 1.0, 1.0, DEPTH, WL,
        n_orders=7)
    assert np.allclose(J1, J2, atol=1e-12)


def test_interdigitated_grating_segments_pattern_and_energy():
    # 'grounded tooth | gap | floating tooth | gap' = 4 regions, normalized.
    segs = interdigitated_grating_segments(
        tooth_widths=[0.2, 0.3], gap_width=0.1,
        tooth_materials=[5.0, 6.0], gap_material=2.0)
    assert [m for _, m in segs] == [5.0, 2.0, 6.0, 2.0]          # tooth|gap|tooth|gap
    assert np.isclose(sum(w for w, _ in segs), 1.0)
    # raw widths [0.2,0.1,0.3,0.1] -> /0.7
    assert np.allclose([w for w, _ in segs], np.array([0.2, 0.1, 0.3, 0.1]) / 0.7)
    # lossless -> energy conserved through the solver.
    o, R, T, J = rcwa_jones_1d_segments(PERIOD, segs, 1.0, 1.0, DEPTH, WL,
                                        n_orders=11)
    assert abs(float(R.sum(axis=1)[0] + T.sum(axis=1)[0]) - 1.0) < 1e-9


def test_grating_segment_builders_validate():
    with pytest.raises(ValueError, match="same length"):
        grating_segments([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="positive"):
        grating_segments([1.0, -1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="duty_cycle"):
        binary_grating_segments(1.5, 2.0, 1.0)
    with pytest.raises(ValueError, match="same length"):
        interdigitated_grating_segments([0.2], 0.1, [5.0, 6.0], 2.0)


# ---------------------------------------------------------------------------
# W2 -- reflective-Jones device helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gamma", [0.0, np.pi / 4, np.pi / 3, np.pi / 2, np.pi])
def test_reflective_outcoupling_lossless_retarder(gamma):
    # a lossless TE/TM-aligned retarder with retardance gamma: the device
    # cross-port power is cos^2(gamma/2), and out + co = 1 (lossless).
    J = np.diag([1.0, np.exp(1j * gamma)]).astype(complex)
    oc = reflective_outcoupling(J)
    assert np.isclose(oc, np.cos(gamma / 2) ** 2, atol=1e-12)
    # the co-polarized (through) port is the complementary sin^2(gamma/2)
    Q = np.array([[(1 - 1j) / 2, (1 + 1j) / 2],
                  [(1 + 1j) / 2, (1 - 1j) / 2]], dtype=complex)
    co = abs((Q @ J @ Q)[0, 0]) ** 2
    assert np.isclose(oc + co, 1.0, atol=1e-12)


def test_reflective_outcoupling_on_real_grating_is_a_fraction():
    # a real lossy grating: the out-coupled power is a valid power fraction.
    o, R, T, J = rcwa_jones_1d(PERIOD, _eye((0.2 + 6j) ** 2), _eye(2.1), 1.5, 1.0,
                               DEPTH, 0.5, WL, n_orders=15)
    oc = reflective_outcoupling(J)
    assert 0.0 <= oc <= 1.0 + 1e-12


def test_jones_retardance_diattenuation_diagonal():
    a, b = np.exp(1j * 0.3), 0.5 * np.exp(1j * 1.1)             # |a|=1, |b|=0.5
    gamma, diatt, _axis = jones_retardance_diattenuation(np.diag([a, b]))
    assert np.isclose(abs(gamma), abs(1.1 - 0.3), atol=1e-9)    # retardance (sign-free)
    assert np.isclose(diatt, (1.0 - 0.25) / (1.0 + 0.25), atol=1e-9)  # 0.6


def test_jones_helpers_reject_bad_shape():
    with pytest.raises(ValueError, match="2, 2"):
        reflective_outcoupling(np.ones((3, 3), dtype=complex))
    with pytest.raises(ValueError, match="2, 2"):
        jones_retardance_diattenuation(np.ones(2, dtype=complex))
