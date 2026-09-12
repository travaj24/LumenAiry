"""v5.4.6 Wave 4 regression pins: polarization cluster.

- P2-3 : JonesField.apply_real_lens(fresnel=True) warns about s/p collapse.
- P2-4 : JonesField.apply_real_lens / apply_mirror / apply_aperture forward dy.
- P3-23: jones_pupil_dock unpolarized Stokes -- 1/2 norm + correct S1/S3.
- P3-24: apply_vector_aperture_diffraction carries vector_projection
  (opt-IN at v5.4.6; the default flipped to True at v5.46 -- see the test).
"""
from __future__ import annotations

import importlib.util
import inspect

import numpy as np
import pytest

from lumenairy.elements.elements import apply_aperture
from lumenairy.elements.polarization import JonesField


def _anamorphic_field(N=64, dx=2e-6, dy=5e-6):
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = np.meshgrid(x, y)
    E = np.exp(-(X * X + Y * Y) / (40e-6) ** 2).astype(np.complex128)
    return JonesField(E.copy(), E.copy(), dx=dx, dy=dy), E


# ---- P2-4: dy threading -----------------------------------------------

def test_jonesfield_apply_aperture_threads_dy():
    """On an anamorphic JonesField (dx != dy), apply_aperture must use
    self.dy -- bit-identical to the scalar call with dy, and DIFFERENT
    from the dy-dropped (dy defaults to dx) call."""
    jf, E = _anamorphic_field()
    jf.apply_aperture(shape='circular', params={'diameter': 110e-6})
    scalar_dy = apply_aperture(E.copy(), 2e-6, shape='circular',
                               params={'diameter': 110e-6}, dy=5e-6)
    scalar_nodY = apply_aperture(E.copy(), 2e-6, shape='circular',
                                 params={'diameter': 110e-6})
    assert np.array_equal(jf.Ex, scalar_dy), "apply_aperture must thread self.dy"
    assert not np.array_equal(scalar_dy, scalar_nodY), (
        "anamorphic aperture must depend on dy (else the test is vacuous)")


# ---- P2-3: fresnel warning --------------------------------------------

def test_jonesfield_apply_real_lens_fresnel_warns():
    jf, _ = _anamorphic_field(dx=2e-6, dy=2e-6)
    presc = {'surfaces': [{'radius': 0.02, 'conic': 0.0, 'glass_before': 'air',
                           'glass_after': 'N-BK7', 'semi_diameter': 1e-3},
                          {'radius': -0.02, 'conic': 0.0, 'glass_before': 'N-BK7',
                           'glass_after': 'air', 'semi_diameter': 1e-3}],
             'thicknesses': [2e-3], 'aperture_diameter': 2e-3}
    with pytest.warns(UserWarning, match='Fresnel|s/p|polariz'):
        jf.apply_real_lens(presc, wavelength=633e-9, fresnel=True)


# ---- P3-23: unpolarized Stokes 1/2 + signs ----------------------------

def test_jones_pupil_unpolarized_stokes_normalisation():
    # v5.4.7 (audit AUDIT_V5_4_6 #10): the helper now lives in the non-Qt
    # ``elements.polarization`` module, so this runs in CI without PySide6
    # (no skipif needed) -- closing the coverage gap the v5.4.6 skip masked.
    from lumenairy.elements.polarization import (
        jones_pupil_to_stokes_unpolarized as _j2s,
    )
    # Identity Jones pupil -> unpolarized output: S0=1, S1=S2=S3=0.
    J = np.zeros((4, 4, 2, 2), dtype=complex)
    J[..., 0, 0] = 1.0
    J[..., 1, 1] = 1.0
    s = _j2s(J)
    assert np.allclose(s['S0'], 1.0), "S0 must carry the 1/2 (identity -> 1)"
    assert np.allclose(s['S1'], 0.0)
    assert np.allclose(s['S2'], 0.0)
    assert np.allclose(s['S3'], 0.0)
    # Horizontal polarizer diag(1, 0): S0=0.5, S1=+0.5 (DOLP=1, +x linear).
    Jp = np.zeros((4, 4, 2, 2), dtype=complex)
    Jp[..., 0, 0] = 1.0
    sp = _j2s(Jp)
    assert np.allclose(sp['S0'], 0.5)
    assert np.allclose(sp['S1'], 0.5), "S1 sign/pattern: +x polarizer -> +S1"


# ---- P3-24: the vector-projection kwarg (default flipped at v5.46) ----

def test_vector_aperture_diffraction_has_projection_kwarg():
    """``vector_projection`` exists, is keyword-only, and defaults to
    ``True``.

    v5.4.6 (P3-24) added it as an OPT-IN (default ``False``) and this test
    pinned that default.  v5.46 (audit K17, WP-A5) flipped it to ``True``
    DELIBERATELY, with a migration note: the opt-in path was the only one
    that produced any vector behaviour at all, so with it off the
    "vectorial" propagator was measurably identical to two scalar HFPI
    runs -- ``max|Ex_vec - Ex_scalar|`` 1.9e-23 and, on a 45-degree linear
    input, ``max|Ey/Ex - 1|`` <= 1.1e-16 over the whole grid, i.e. zero
    depolarisation anywhere, at twice the cost.  With it on (and rewritten
    as a rigid rotation) the same fixture gives ``|Ez|^2`` 1.40e-2 of the
    incident power, cross-polarised ``|Ey|^2`` 9.3e-5, 45-degree
    ``max|Ey/Ex - 1|`` 0.143, and the projection now conserves
    ``|E|^2`` to 1 - 1.1e-15 where the old opt-in dropped 15.9 %.

    So this pin is restated to the documented default rather than kept at
    the old one.  ``vector_projection=False`` is retained and documented
    as reproducing the pre-v5.46 behaviour exactly, so the counter-pin
    below (the kwarg still EXISTS) is the part that protects migrating
    callers.  The physics behind the flip is measured in
    ``test_audit2609_a5_propagators.py::TestK17VectorialHfpi``; this file
    only pins the signature.
    """
    from lumenairy.propagators.vectorial_hfpi import (
        apply_vector_aperture_diffraction,
    )
    params = inspect.signature(apply_vector_aperture_diffraction).parameters
    assert 'vector_projection' in params
    assert (params['vector_projection'].kind
            is inspect.Parameter.KEYWORD_ONLY)
    assert params['vector_projection'].default is True, (
        "v5.46 (audit K17) flipped this default to True; if it is False "
        "again the vectorial propagator is back to being two scalar HFPI "
        "runs at twice the cost")
