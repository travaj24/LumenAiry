"""Audit P1-02: bruggeman() linear coefficient + physical-root selection.

The v5.17.0 deep audit found the Bruggeman quadratic ``g e^2 + b e - ea eb``
was built with the WRONG linear coefficient ``b = (g f - 1) ea +
(g (1-f) - 1) eb`` instead of the correct ``b = ((1-f) - g f) ea +
(f - g (1-f)) eb``, so the returned eps violated the defining self-consistency
equation everywhere except accidental points (e.g. ``bruggeman(4, 4, 0.5)``
returned 2.828 instead of 4.0).  The pre-existing check in
``test_mixing_rules_properties`` (phase-swap symmetry at f = 0.5) is satisfied
by the wrong coefficient by construction, so every test here is chosen to
DISCRIMINATE: on the pre-fix code the residual check, the ``ea == eb``
identity, the dilute limits and the metallic passive-branch checks all fail.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.emt import bruggeman

_G = {"sphere": 2.0, "cylinder": 1.0}


def _residual(e: complex, ea: complex, eb: complex, f: float, g: float):
    """The defining Bruggeman self-consistency equation (must be ~0)."""
    return f * (ea - e) / (ea + g * e) + (1 - f) * (eb - e) / (eb + g * e)


@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
@pytest.mark.parametrize("ea,eb,f", [
    (9.0, 2.0, 0.3),
    (2.0, 9.0, 0.75),
    (1.0, 12.0, 0.5),
    (9.0 + 0.5j, 2.0, 0.3),            # lossy phase a
    (9.0 + 0.5j, 2.0 + 0.1j, 0.7),     # both lossy
    (-10.0 + 1.0j, 2.25, 0.4),         # lossy metal / dielectric
])
def test_self_consistency_residual(geometry, ea, eb, f):
    e = bruggeman(ea, eb, f, geometry=geometry)
    r = _residual(e, complex(ea), complex(eb), f, _G[geometry])
    assert abs(r) < 1e-12


@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
def test_equal_phases_identity(geometry):
    # ea == eb must return that value for ANY fill (pre-fix: 2.828 / 6.472)
    for f in (0.0, 0.25, 0.5, 0.9, 1.0):
        assert abs(bruggeman(4.0, 4.0, f, geometry=geometry) - 4.0) < 1e-12
    e = bruggeman(3.0 + 0.2j, 3.0 + 0.2j, 0.5, geometry=geometry)
    assert abs(e - (3.0 + 0.2j)) < 1e-12


@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
def test_endpoint_and_dilute_limits(geometry):
    # exact end points (pre-fix code was wrong even at f = 0 / f = 1)
    assert abs(bruggeman(9.0, 2.0, 0.0, geometry=geometry) - 2.0) < 1e-12
    assert abs(bruggeman(9.0, 2.0, 1.0, geometry=geometry) - 9.0) < 1e-12
    # dilute limits approach the majority phase (pre-fix: 5.218 at f=0.001)
    assert abs(bruggeman(9.0, 2.0, 0.001, geometry=geometry) - 2.0) < 0.02
    assert abs(bruggeman(9.0, 2.0, 0.999, geometry=geometry) - 9.0) < 0.02


@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
def test_monotonic_and_bounded_real_mix(geometry):
    fs = np.linspace(0.0, 1.0, 41)
    es = np.array([bruggeman(9.0, 2.0, float(f), geometry=geometry).real
                   for f in fs])
    assert np.all(np.diff(es) > 0)                    # monotone in fill_a
    assert es.min() >= 2.0 - 1e-12 and es.max() <= 9.0 + 1e-12


def test_sphere_matches_textbook_closed_form():
    # 3-D Bruggeman (depolarization 1/3):
    #   e = [(3f-1) ea + (2-3f) eb + sqrt(((3f-1) ea + (2-3f) eb)^2
    #                                     + 8 ea eb)] / 4
    ea, eb, f = 9.0, 2.0, 0.3
    q = (3 * f - 1) * ea + (2 - 3 * f) * eb
    ref = (q + np.sqrt(q * q + 8 * ea * eb)) / 4.0
    assert abs(bruggeman(ea, eb, f, geometry="sphere") - ref) < 1e-12


@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
def test_metal_dielectric_passive_branch(geometry):
    # Lossless metal / dielectric: the physical (passive) root must have
    # Im(e) >= 0 across the whole fill range, reduce exactly at the end
    # points, and track the infinitesimal-loss limit (branch continuity).
    ea, eb = -10.0, 2.25
    assert abs(bruggeman(ea, eb, 0.0, geometry=geometry) - eb) < 1e-12
    assert abs(bruggeman(ea, eb, 1.0, geometry=geometry) - ea) < 1e-12
    for f in np.linspace(0.0, 1.0, 41):
        e = bruggeman(ea, eb, float(f), geometry=geometry)
        assert e.imag >= -1e-12
        e_lossy = bruggeman(ea + 1e-6j, eb, float(f), geometry=geometry)
        assert abs(e - e_lossy) < 1e-4                # lossless = lossy limit
    # lossy metal: passive everywhere, residual already covered above
    for f in np.linspace(0.0, 1.0, 21):
        e = bruggeman(-10.0 + 1.0j, 2.25, float(f), geometry=geometry)
        assert e.imag >= -1e-12
