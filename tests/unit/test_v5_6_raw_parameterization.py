"""v5.6 prescription-free optimization: RawParameterization.

``RawParameterization`` lets ``design_optimize`` run a wave-only /
rigorous-element (RCWA, coating, metasurface) design from a bare parameter
vector -- no lens-prescription template.  ``build(x)`` returns a minimal dict
(no ``'surfaces'``); merits read ``ctx.x``.  A ``needs_ray=True`` merit paired
with a template-free build raises a clear, actionable error.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy import RawParameterization
from lumenairy.optimize import MeritTerm, design_optimize


class _ParamMerit(MeritTerm):
    """A pure-parameter merit: ||x - target||^2 (no ray, no wave)."""
    needs_ray = False
    needs_wave = False

    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)

    def evaluate(self, ctx):
        return float(np.sum((np.asarray(ctx.x) - self.target) ** 2))


def test_raw_parameterization_interface():
    p = RawParameterization(x0=[0.1, 0.2, 0.3], bounds=[(0, 1)] * 3)
    assert p.n_params == 3
    assert np.array_equal(p.initial_values(), [0.1, 0.2, 0.3])
    built = p.build([0.4, 0.5, 0.6])
    assert "surfaces" not in built              # template-free
    assert built["aperture_diameter"] is None   # wave-leg falls back to default
    assert np.array_equal(built["_raw_params"], [0.4, 0.5, 0.6])
    assert p.scale_floor.shape == (3,)


def test_raw_parameterization_bounds_length_check():
    with pytest.raises(ValueError, match="bounds length"):
        RawParameterization(x0=[0.1, 0.2], bounds=[(0, 1)])


def test_raw_parameterization_scale_floor_broadcast():
    assert np.allclose(RawParameterization(x0=[1, 2, 3], scale_floor=1e-4)
                       .scale_floor, 1e-4)
    with pytest.raises(ValueError, match="scale_floor length"):
        RawParameterization(x0=[1, 2], scale_floor=[1e-4, 1e-4, 1e-4])


def test_prescription_free_optimization_converges():
    """design_optimize drives a bare parameter vector to a target with no lens
    prescription (the wave-only / rigorous-element design path)."""
    target = [0.7, -0.3, 0.45]
    param = RawParameterization(x0=[0.0, 0.0, 0.0], bounds=[(-1, 1)] * 3,
                                scale_floor=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = design_optimize(param, [_ParamMerit(target)],
                              wavelength=0.633e-6, N=16, dx=1e-6, max_iter=300)
    xf = np.array(res.x if hasattr(res, "x") else res.params)
    assert np.allclose(xf, target, atol=1e-3)


def test_needs_ray_with_raw_parameterization_raises_clear_error():
    class _RayMerit(MeritTerm):
        needs_ray = True
        needs_wave = False

        def evaluate(self, ctx):
            return float(ctx.efl)

    with pytest.raises(ValueError, match="needs a prescription with 'surfaces'"):
        design_optimize(RawParameterization(x0=[0.1]), [_RayMerit()],
                        wavelength=0.633e-6, N=16, dx=1e-6, max_iter=2)
