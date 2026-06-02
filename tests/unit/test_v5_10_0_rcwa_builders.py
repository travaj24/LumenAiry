"""v5.10.0 RCWA layer-builder conveniences (audit GAP4 / P4).

* ``RCWAStack.add_graded_layer(thickness, profile, n_slices, rule)`` --
  auto-slice a continuous ``eps(z)`` profile (carrier/ENZ, thermo-optic, etch
  gradient) into a staircase of thin layers.
* ``RCWAStack.add_tapered_grating(...)`` -- a 1-D grating with slanted
  (trapezoidal) sidewalls as an auto-sliced z-staircase (fab realism).
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la


def _stack(n_orders=5):
    return la.RCWAStack(1e-6, period_y=1e-6, n_substrate=1.5,
                        n_orders=n_orders, n_orders_y=1)


# ---------------------------------------------------------------------------
# add_graded_layer
# ---------------------------------------------------------------------------

def test_graded_constant_profile_equals_single_uniform_layer():
    g = _stack().add_graded_layer(0.3e-6, lambda z: 6.0 + 0.2j, n_slices=6)
    o, R, T = g.set_source(0.8e-6).solve().efficiencies()
    u = _stack().add_layer(0.3e-6, eps=6.0 + 0.2j)
    o2, R2, T2 = u.set_source(0.8e-6).solve().efficiencies()
    assert np.allclose(R, R2, atol=1e-12) and np.allclose(T, T2, atol=1e-12)


def test_graded_layer_conserves_energy_and_slices():
    S = 48
    xx = (np.arange(S) + 0.5) / S

    def enz(z):                                  # ridge index sweeps with depth
        col = np.where(xx < 0.5, 1.0 + (-0.5 + 2.0 * z), 2.1).astype(complex)
        return np.broadcast_to(col[:, None], (S, S)).copy()

    g = _stack(n_orders=5).add_graded_layer(0.2e-6, enz, n_slices=10)
    res = g.set_source(0.6e-6, theta=0.1).solve()
    o, R, T = res.efficiencies()
    assert len(g._layers) == 10
    assert np.allclose(R.sum(1) + T.sum(1) + res.absorptance(), 1.0, atol=1e-6)


def test_graded_layer_tensor_profile_dispatch():
    # a (Sx, Sy, 3, 3) profile -> anisotropic slices
    S = 48
    ten = la.uniaxial_tensor(1.5, 1.7, np.pi / 2)
    g = _stack().add_graded_layer(
        0.2e-6, lambda z: np.broadcast_to(ten, (S, S, 3, 3)).copy(), n_slices=4)
    res = g.set_source(0.8e-6).solve()
    o, R, T = res.efficiencies()
    assert np.allclose(R.sum(1) + T.sum(1) + res.absorptance(), 1.0, atol=1e-6)


@pytest.mark.parametrize("rule", ["midpoint", "trapezoid"])
def test_graded_layer_rules(rule):
    g = _stack().add_graded_layer(0.2e-6, lambda z: 2.0 + z, n_slices=5,
                                  rule=rule)
    res = g.set_source(0.8e-6).solve()
    o, R, T = res.efficiencies()
    assert np.allclose(R.sum(1) + T.sum(1) + res.absorptance(), 1.0, atol=1e-6)


def test_graded_layer_bad_args():
    with pytest.raises(ValueError, match="n_slices"):
        _stack().add_graded_layer(0.2e-6, lambda z: 2.0, n_slices=0)
    with pytest.raises(ValueError, match="rule"):
        _stack().add_graded_layer(0.2e-6, lambda z: 2.0, rule="simpson")


# ---------------------------------------------------------------------------
# add_tapered_grating
# ---------------------------------------------------------------------------

def test_tapered_vertical_equals_single_binary_grating():
    x = (np.arange(256) + 0.5) / 256
    t = _stack(n_orders=8).add_tapered_grating(
        0.3e-6, eps_ridge=6.0, eps_groove=1.0, duty_bottom=0.5, duty_top=0.5,
        n_slices=4, n_x=256)
    o, R, T = t.set_source(0.8e-6).solve().efficiencies()
    col = np.where(np.abs(x - 0.5) < 0.25, 6.0 + 0j, 1.0).astype(complex)
    b = _stack(n_orders=8).add_layer(
        0.3e-6, eps_cell=np.broadcast_to(col[:, None], (256, 5)).copy())
    o2, R2, T2 = b.set_source(0.8e-6).solve().efficiencies()
    assert np.allclose(R, R2, atol=1e-12) and np.allclose(T, T2, atol=1e-12)


def test_tapered_grating_conserves_energy():
    t = _stack(n_orders=8).add_tapered_grating(
        0.3e-6, eps_ridge=2.5, eps_groove=1.0, duty_bottom=0.6, duty_top=0.4,
        n_slices=12)
    res = t.set_source(0.8e-6).solve()
    o, R, T = res.efficiencies()
    assert np.allclose(R.sum(1) + T.sum(1), 1.0, atol=1e-6)        # lossless


def test_taper_changes_result_vs_vertical():
    common = dict(eps_ridge=2.5, eps_groove=1.0, n_slices=12)
    o, R, T = (_stack(n_orders=8)
               .add_tapered_grating(0.3e-6, duty_bottom=0.6, duty_top=0.4,
                                    **common)
               .set_source(0.8e-6).solve().efficiencies())
    o2, Rv, Tv = (_stack(n_orders=8)
                  .add_tapered_grating(0.3e-6, duty_bottom=0.5, duty_top=0.5,
                                       **common)
                  .set_source(0.8e-6).solve().efficiencies())
    assert np.max(np.abs(T - Tv)) > 1e-3          # the taper matters


def test_tapered_grating_bad_duty():
    with pytest.raises(ValueError, match="duty"):
        _stack().add_tapered_grating(0.3e-6, eps_ridge=2.5, eps_groove=1.0,
                                     duty_bottom=1.5)
