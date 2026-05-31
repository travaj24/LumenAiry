"""Regression pins for the polarizing beam splitter
(:func:`lumenairy.apply_polarizing_beam_splitter`).

A PBS separates a field into a transmitted port (the polarization along the
transmission axis) and a reflected port (the orthogonal polarization).  The
pins cover energy conservation, ideal separation, Malus's law, the finite
extinction ratio, and that the input field is NOT mutated (the two output
ports operate on independent copies -- ``apply_jones_matrix`` is in-place).
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.polarization import JonesField


def _field(ex, ey, n=4):
    return JonesField(ex * np.ones((n, n), complex),
                      ey * np.ones((n, n), complex), dx=1e-6)


def _power(f):
    return float((np.abs(f.Ex) ** 2 + np.abs(f.Ey) ** 2)[0, 0])


def test_pbs_energy_conservation_and_no_mutation():
    f = _field(1.0, 1.0)            # 45 deg linear
    t, r = la.apply_polarizing_beam_splitter(f, angle=0.0)
    # input is unchanged (each port worked on an independent copy)
    assert f.Ex[0, 0] == 1.0 and f.Ey[0, 0] == 1.0
    # energy conserved exactly, per pixel
    Pin = np.abs(f.Ex) ** 2 + np.abs(f.Ey) ** 2
    Pt = np.abs(t.Ex) ** 2 + np.abs(t.Ey) ** 2
    Pr = np.abs(r.Ex) ** 2 + np.abs(r.Ey) ** 2
    assert np.max(np.abs(Pt + Pr - Pin)) < 1e-12


def test_pbs_ideal_separation():
    # 45 deg input -> transmitted = pure p (x), reflected = pure s (y), 50/50.
    t, r = la.apply_polarizing_beam_splitter(_field(1.0, 1.0), angle=0.0)
    assert np.allclose(t.Ex, 1.0) and np.allclose(t.Ey, 0.0)
    assert np.allclose(r.Ex, 0.0) and np.allclose(r.Ey, 1.0)
    # pure-p input -> all transmitted, nothing reflected.
    tp, rp = la.apply_polarizing_beam_splitter(_field(1.0, 0.0), angle=0.0)
    assert _power(tp) > 1 - 1e-12 and _power(rp) < 1e-12


@pytest.mark.parametrize("axis_deg,expect_t", [(0.0, 1.0), (30.0, 0.75),
                                               (45.0, 0.5), (90.0, 0.0)])
def test_pbs_malus_law(axis_deg, expect_t):
    # x-polarized input through a PBS with transmission axis at `axis_deg`:
    # transmitted power = cos^2(axis).
    t, r = la.apply_polarizing_beam_splitter(_field(1.0, 0.0),
                                             angle_deg=axis_deg)
    assert abs(_power(t) - expect_t) < 1e-12
    assert abs(_power(r) - (1.0 - expect_t)) < 1e-12


def test_pbs_extinction_ratio():
    # pure-s input, transmission axis = x; a finite extinction ratio lets
    # 1/(1+ER) of the s power leak into the transmitted port.
    ER = 1000.0
    t, r = la.apply_polarizing_beam_splitter(_field(0.0, 1.0), angle=0.0,
                                             extinction_ratio=ER)
    assert abs(_power(t) - 1.0 / (1.0 + ER)) < 1e-12
    assert abs(_power(r) - ER / (1.0 + ER)) < 1e-12
    assert abs(_power(t) + _power(r) - 1.0) < 1e-12   # still energy-conserving


def test_pbs_input_validation():
    f = _field(1.0, 0.0)
    with pytest.raises(ValueError):
        la.apply_polarizing_beam_splitter(f, extinction_ratio=-5.0)
    with pytest.raises(ValueError):
        la.apply_polarizing_beam_splitter(f, angle=0.1, angle_deg=45.0)


def test_pbs_is_exposed():
    assert "apply_polarizing_beam_splitter" in la.__all__
    assert hasattr(la, "apply_polarizing_beam_splitter")
