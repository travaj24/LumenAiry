"""v5.7.0 Adaptive Spatial Resolution (ASR / Granet 1999) + matched coordinates
for the 1-D RCWA solver (items E & F).

``rcwa_efficiency_1d(..., asr_eta>0)`` applies a matched coordinate stretch
``f(u) = 1 - eta*cos(...)`` that clusters Fourier harmonics at the grating
walls, converging much faster for metals / high-contrast TM at LOW order
counts.  Item F (matched coordinates + Li/FFF inverse rule) is the same path
with ``formulation='li'`` (or a metallic index, which auto-selects it).

Validation strategy: the existing uniform solver AT HIGH ORDER is the
convergence oracle (ASR must reach the same efficiencies at FEWER orders).
``asr_eta=0`` must be bit-identical to the uniform path.  ASR is a
low-to-moderate-order accelerator -- its bridge is ill-conditioned at high
order, where a warning is emitted -- so the gains are asserted at LOW order.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.elements.rcwa import (
    _asr_metric_profile,
    rcwa_efficiency_1d,
)

# gold-like lossy metal (PUBLIC Im > 0): the canonical fast-ASR case
GOLD = dict(period=1.0, n_ridge=0.2 + 6j, n_groove=1.0, n_substrate=1.0,
            n_superstrate=1.0, depth=0.3, duty_cycle=0.5, wavelength=0.8)


def _sumT(**kw):
    o, R, T = rcwa_efficiency_1d(**kw)
    return float(T.sum())


# ---------------------------------------------------------------------------
# asr_eta = 0 is the exact uniform path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pol", ["te", "tm"])
def test_asr_eta0_bit_identical_to_uniform(pol):
    o0, R0, T0 = rcwa_efficiency_1d(**GOLD, polarization=pol, n_orders=15)
    o1, R1, T1 = rcwa_efficiency_1d(**GOLD, polarization=pol, n_orders=15,
                                    asr_eta=0.0)
    assert np.array_equal(R0, R1) and np.array_equal(T0, T1)


# ---------------------------------------------------------------------------
# ASR converges faster at low order (the whole point) -- and to the SAME value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pol", ["tm", "te"])
def test_asr_beats_uniform_at_low_order(pol):
    ref = _sumT(**GOLD, polarization=pol, n_orders=80)          # oracle
    err_uniform = abs(_sumT(**GOLD, polarization=pol, n_orders=12) - ref)
    err_asr = abs(_sumT(**GOLD, polarization=pol, n_orders=12, asr_eta=0.7)
                  - ref)
    assert err_asr < err_uniform                                # faster
    assert err_asr < 5e-3                                       # and accurate


def test_asr_reduces_required_order_count():
    # ASR with HALF the orders is more accurate than the uniform method --
    # i.e. it reaches a target accuracy at fewer harmonics (the practical win).
    # (ASR error is non-monotonic in order count: it has a low-order sweet spot
    # and plateaus, since the u<->x bridge degrades at high order -- so the
    # claim is order REDUCTION, not machine-precision convergence.)
    ref = _sumT(**GOLD, polarization="tm", n_orders=80)
    err_asr_12 = abs(_sumT(**GOLD, polarization="tm", n_orders=12, asr_eta=0.7)
                     - ref)
    err_uniform_24 = abs(_sumT(**GOLD, polarization="tm", n_orders=24) - ref)
    assert err_asr_12 < err_uniform_24                    # 12 ASR < 24 uniform


def test_matched_coordinate_fff_metal_tm():
    # item F: matched coords + Li inverse rule (auto-selected for the metal TM)
    ref = _sumT(**GOLD, polarization="tm", n_orders=80, formulation="li")
    err_uniform = abs(_sumT(**GOLD, polarization="tm", n_orders=12,
                            formulation="li") - ref)
    err_asr = abs(_sumT(**GOLD, polarization="tm", n_orders=12, asr_eta=0.7,
                        formulation="li") - ref)
    assert err_asr < err_uniform


def test_asr_energy_conserved_lossless():
    lossless = dict(GOLD, n_ridge=2.5 + 0j)
    o, R, T = rcwa_efficiency_1d(**lossless, polarization="te", n_orders=20,
                                 asr_eta=0.6)
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# gates + guards
# ---------------------------------------------------------------------------

def test_asr_rejects_oblique_incidence():
    with pytest.raises(ValueError, match="normal incidence only"):
        rcwa_efficiency_1d(**GOLD, polarization="tm", n_orders=10,
                           asr_eta=0.7, angle=0.2)


@pytest.mark.parametrize("bad_eta", [1.0, 1.5, -0.1])
def test_asr_rejects_eta_outside_unit_interval(bad_eta):
    with pytest.raises(ValueError, match=r"asr_eta must be in"):
        rcwa_efficiency_1d(**GOLD, polarization="tm", n_orders=10,
                           asr_eta=bad_eta)


def test_asr_warns_when_bridge_ill_conditioned_high_order():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rcwa_efficiency_1d(**GOLD, polarization="tm", n_orders=80, asr_eta=0.7)
    assert any("conditioning" in str(x.message) for x in w)


def test_asr_no_warning_in_useful_low_order_regime():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rcwa_efficiency_1d(**GOLD, polarization="tm", n_orders=20, asr_eta=0.7)
    assert not any("conditioning" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# coordinate-map invariants
# ---------------------------------------------------------------------------

def test_metric_profile_invariants():
    u, f, x, in_ridge = _asr_metric_profile(0.5, 0.7, 4096)
    assert np.all(f > 0)                                  # Jacobian positive
    assert abs(f.mean() - 1.0) < 1e-6                     # <f> = 1 (period kept)
    # walls (u=0, u=duty, u=1) are the fine-resolution points f = 1 - eta
    assert f.min() == pytest.approx(1.0 - 0.7, abs=2e-3)


def test_metric_profile_eta0_is_identity():
    u, f, x, in_ridge = _asr_metric_profile(0.5, 0.0, 1024)
    assert np.allclose(f, 1.0)
    assert np.allclose(x, u)
