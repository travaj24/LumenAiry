"""Pinning tests for the v4.14.2 audit (Agent A scope --
``lumenairy.glass``, ``lumenairy.elements.freeform``,
``lumenairy.elements.polarization``).

Four audit items are pinned:

* **P0-NEW-1** -- ``GLASS_REGISTRY['S-LAH64']`` /
  ``GLASS_REGISTRY['S-LAH79']`` were stranded with the
  ``'__sellmeier__'`` sentinel after v4.11.2 removed their Sellmeier
  rows from :data:`SELLMEIER_COEFFICIENTS`.  Every lookup for these
  two glasses raised ``ValueError`` from the dispatcher's
  consistency-check branch -- the v4.11.2 fix forgot to re-route them
  to a refractiveindex.info tuple.  v4.14.2 re-routes to
  ``('specs', 'OHARA-optical', 'S-LAH64')`` /
  ``('specs', 'OHARA-optical', 'S-LAH79')``, restoring the Ohara
  catalogue n_d values (1.788 / 2.003).  A new module-load
  consistency check (``_check_glass_registry_consistency``) converts
  the same class-of-bug into a fail-fast at import time so a future
  drift cannot re-surface as a silent ``ValueError`` at first call.

* **P1-NEW-5** -- ``surface_sag_xy_polynomial`` evaluated
  ``c * X**i * Y**j`` for every pixel of the input grid, with no
  out-of-domain guard.  A high-order coefficient on a large grid
  produced a discontinuous step at the aperture rim (e.g.
  ``(2, 0): 1e3`` on a 50 mm half-grid produces 2.5 m of corner sag
  applied to pixels outside the physical aperture).  v4.14.2 adds
  ``norm_x``/``norm_y`` kwargs (default 1.0 = unit box) matching the
  Chebyshev branch's ``np.where(outside, 0.0, departure)`` pattern.

* **P1-NEW-7** -- ``apply_rotator`` was the only angle-taking
  polarization helper without an ``angle_deg=`` kwarg.  Idiomatic
  ``apply_rotator(field, angle_deg=45)`` calls raised TypeError.
  v4.14.2 closes the v4.7 sibling-gap; passing both with
  conflicting values raises ``ValueError`` (consistent-value pairs
  are accepted).

* **P1-NEW-8** -- ``JonesField.__init__`` validated
  ``Ex.shape == Ey.shape`` but not ``Ex.ndim == 2``, ``dx > 0``,
  ``dy > 0``.  Invalid inputs propagated all the way to the FFT in
  :meth:`propagate` where an opaque shape / value error was raised
  far from the construction site.  v4.14.2 validates at construction.

Author:  Agent A -- v4.14.2.
"""
from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

import lumenairy as lm
from lumenairy import glass as _glass
from lumenairy.elements.freeform import surface_sag_xy_polynomial
from lumenairy.elements.polarization import (
    JonesField,
    apply_rotator,
    create_linear_polarized,
)


# ===========================================================================
# A.1 (P0-NEW-1) -- S-LAH64 / S-LAH79 dispatch
# ===========================================================================


class TestSLahDispatch:
    """Pin that ``GLASS_REGISTRY['S-LAH64']`` /
    ``GLASS_REGISTRY['S-LAH79']`` resolve to a real refractive index.

    The pre-v4.14.2 bug:  both glasses were flagged
    ``'__sellmeier__'`` but their Sellmeier rows were removed in
    v4.11.2, so the dispatcher's consistency-check raised
    ``ValueError`` on every lookup.
    """

    def test_s_lah64_dispatch_does_not_raise(self):
        """``get_glass_index('S-LAH64', d-line)`` returns a finite
        refractive index in the [1.5, 2.0] range characteristic of
        Ohara lanthanum-flint glasses.  Pre-v4.14.2 raised
        ``ValueError``.
        """
        n = _glass.get_glass_index('S-LAH64', 587.6e-9)
        assert np.isfinite(n), 'S-LAH64 must return a finite n'
        assert 1.5 < n < 2.0, (
            f'S-LAH64 d-line n = {n:.4f} outside the expected '
            f'lanthanum-flint range [1.5, 2.0]; Ohara catalogue '
            f'value is 1.788.')

    def test_s_lah79_dispatch_does_not_raise(self):
        """``get_glass_index('S-LAH79', d-line)`` returns a finite
        refractive index near the Ohara catalogue value 2.003.
        Pre-v4.14.2 raised ``ValueError``.
        """
        n = _glass.get_glass_index('S-LAH79', 587.6e-9)
        assert np.isfinite(n), 'S-LAH79 must return a finite n'
        # S-LAH79 n_d = 2.00330 per Ohara catalogue
        assert 1.5 < n < 2.1, (
            f'S-LAH79 d-line n = {n:.4f} outside the expected '
            f'high-index lanthanum-flint range [1.5, 2.1]; Ohara '
            f'catalogue value is 2.003.')

    def test_glass_registry_consistency_check_present(self):
        """The new module-load consistency check function exists
        and contains the drift-detection logic.  Its source must
        reference both ``GLASS_REGISTRY`` and
        ``SELLMEIER_COEFFICIENTS`` and contain a ``RuntimeError``
        path -- this is the structural counter-measure that prevents
        the same class-of-bug from re-surfacing silently.
        """
        assert hasattr(_glass, '_check_glass_registry_consistency'), (
            'v4.14.2 module-load consistency check '
            '`_check_glass_registry_consistency` is missing from '
            'lumenairy.glass.')
        src = inspect.getsource(_glass._check_glass_registry_consistency)
        assert 'GLASS_REGISTRY' in src, (
            'consistency check must iterate GLASS_REGISTRY')
        assert 'SELLMEIER_COEFFICIENTS' in src, (
            'consistency check must verify membership in '
            'SELLMEIER_COEFFICIENTS')
        assert 'RuntimeError' in src, (
            'consistency check must fail-fast with RuntimeError on '
            'detected drift')
        assert "'__sellmeier__'" in src or '"__sellmeier__"' in src, (
            'consistency check must filter on the __sellmeier__ '
            'sentinel')

    def test_glass_registry_consistency_check_rejects_drift(self):
        """Inject a synthetic ``'__test_sentinel_v4_14_2__'`` entry
        flagged ``'__sellmeier__'`` but absent from
        ``SELLMEIER_COEFFICIENTS``, then re-invoke the check.  Must
        raise ``RuntimeError`` naming the drift.  Restore state on
        exit so subsequent tests are unaffected.
        """
        sentinel_name = '__test_sentinel_v4_14_2__'
        assert sentinel_name not in _glass.GLASS_REGISTRY
        assert sentinel_name not in _glass.SELLMEIER_COEFFICIENTS
        _glass.GLASS_REGISTRY[sentinel_name] = '__sellmeier__'
        try:
            with pytest.raises(RuntimeError, match='GLASS_REGISTRY drift'):
                _glass._check_glass_registry_consistency()
        finally:
            del _glass.GLASS_REGISTRY[sentinel_name]
        # Re-run the real check to confirm no real-entry drift was
        # introduced.
        _glass._check_glass_registry_consistency()


# ===========================================================================
# A.2 (P1-NEW-5) -- surface_sag_xy_polynomial out-of-domain guard
# ===========================================================================


class TestXyPolynomialDomainGuard:
    """Pin that ``surface_sag_xy_polynomial`` zeros the polynomial
    departure outside the ``(norm_x, norm_y)`` rectangular box.

    The pre-v4.14.2 bug:  no guard, so a high-order coefficient on a
    large grid produced a discontinuous step at the aperture rim
    where the polynomial diverged but the raytracer saw no
    aperture-aware clip.
    """

    def test_xy_polynomial_zero_outside_unit_box_nonzero_inside(self):
        """Build a (2, 0): 1.0 (pure X^2) XY-polynomial surface,
        evaluate on a 64x64 grid spanning [-2, +2] (twice the
        unit-box half-extent in each axis).  The freeform departure
        must be zero outside the unit box and equal to X^2 inside.
        """
        N = 64
        half = 2.0  # twice the default unit-box half-extent
        x = np.linspace(-half, half, N)
        y = np.linspace(-half, half, N)
        X, Y = np.meshgrid(x, y, indexing='xy')
        # Pure X^2 freeform with flat base (R = inf, no conic).
        sag = surface_sag_xy_polynomial(
            X, Y, R=np.inf, conic=0.0,
            xy_coeffs={(2, 0): 1.0},
            norm_x=1.0, norm_y=1.0)
        # Outside the unit box: sag must be 0 (flat base + zeroed
        # freeform).
        outside = (np.abs(X) > 1.0) | (np.abs(Y) > 1.0)
        np.testing.assert_allclose(
            sag[outside], 0.0, atol=1e-15,
            err_msg='sag must be zero outside the unit box')
        # Inside the unit box: sag must equal X^2 (the pure
        # polynomial term, no base sag).
        inside = ~outside
        np.testing.assert_allclose(
            sag[inside], (X ** 2)[inside], atol=1e-15,
            err_msg='sag must equal X^2 inside the unit box')
        # Sanity: must have BOTH inside and outside pixels.
        assert outside.any(), 'test grid must extend beyond unit box'
        assert inside.any(), 'test grid must include unit-box pixels'


# ===========================================================================
# A.3 (P1-NEW-7) -- apply_rotator angle_deg kwarg
# ===========================================================================


def _x_polarized_field(N: int = 32, dx: float = 1e-6) -> JonesField:
    """A unit-amplitude x-polarized JonesField for rotator tests."""
    scalar = np.ones((N, N), dtype=complex)
    return create_linear_polarized(scalar, dx, angle=0.0)


class TestApplyRotatorAngleDeg:
    """Pin that ``apply_rotator`` accepts ``angle_deg=`` matching the
    v4.7 convention used by ``apply_polarizer``, ``apply_waveplate``,
    and the half/quarter-wave-plate convenience wrappers.
    """

    def test_apply_rotator_angle_deg_equivalent_to_angle_rad(self):
        """``apply_rotator(field, angle_deg=45)`` must produce
        bit-equal output to ``apply_rotator(field, angle=pi/4)``.
        """
        f_rad = _x_polarized_field()
        f_deg = _x_polarized_field()
        apply_rotator(f_rad, angle=np.pi / 4)
        apply_rotator(f_deg, angle_deg=45.0)
        np.testing.assert_allclose(
            f_rad.Ex, f_deg.Ex, atol=1e-15,
            err_msg='Ex from angle_deg=45 must match angle=pi/4')
        np.testing.assert_allclose(
            f_rad.Ey, f_deg.Ey, atol=1e-15,
            err_msg='Ey from angle_deg=45 must match angle=pi/4')

    def test_apply_rotator_consistent_angle_and_angle_deg_accepted(self):
        """Supplying both ``angle`` and ``angle_deg`` with consistent
        values must be accepted (silently use ``angle_deg``)."""
        f = _x_polarized_field()
        # pi/4 == radians(45) -- consistent
        apply_rotator(f, angle=np.pi / 4, angle_deg=45.0)
        # Should not raise; Ey should be sin(pi/4) = 1/sqrt(2)
        np.testing.assert_allclose(
            f.Ey[0, 0], 1.0 / np.sqrt(2), atol=1e-12)

    def test_apply_rotator_conflicting_angle_raises(self):
        """Supplying ``angle`` and ``angle_deg`` with disagreeing
        values must raise ``ValueError``."""
        f = _x_polarized_field()
        with pytest.raises(ValueError, match='conflicting'):
            apply_rotator(f, angle=np.pi / 3, angle_deg=45.0)


# ===========================================================================
# A.4 (P1-NEW-8) -- JonesField input validation
# ===========================================================================


class TestJonesFieldInputValidation:
    """Pin that ``JonesField.__init__`` validates ``dx > 0``,
    ``dy > 0``, and ``Ex.ndim == 2`` at construction time, rather
    than letting invalid inputs propagate to the FFT in
    :meth:`JonesField.propagate`.
    """

    def test_jones_field_rejects_zero_dx(self):
        """``dx = 0`` must raise ``ValueError`` at construction."""
        N = 16
        Ex = np.ones((N, N), dtype=complex)
        Ey = np.ones((N, N), dtype=complex)
        with pytest.raises(ValueError, match='dx must be'):
            JonesField(Ex, Ey, dx=0.0)

    def test_jones_field_rejects_negative_dx(self):
        """``dx < 0`` must raise ``ValueError`` at construction."""
        N = 16
        Ex = np.ones((N, N), dtype=complex)
        Ey = np.ones((N, N), dtype=complex)
        with pytest.raises(ValueError, match='dx must be'):
            JonesField(Ex, Ey, dx=-1e-6)

    def test_jones_field_rejects_negative_dy(self):
        """``dy < 0`` must raise ``ValueError`` at construction."""
        N = 16
        Ex = np.ones((N, N), dtype=complex)
        Ey = np.ones((N, N), dtype=complex)
        with pytest.raises(ValueError, match='dy must be'):
            JonesField(Ex, Ey, dx=1e-6, dy=-1e-6)

    def test_jones_field_rejects_1d_input(self):
        """A 1-D ``Ex`` / ``Ey`` must raise ``ValueError`` at
        construction.  Pre-v4.14.2 this propagated to the FFT in
        ``propagate`` and raised an opaque error far from the
        construction site.
        """
        Ex = np.ones(64, dtype=complex)
        Ey = np.ones(64, dtype=complex)
        with pytest.raises(ValueError, match='2-D'):
            JonesField(Ex, Ey, dx=1e-6)

    def test_jones_field_valid_inputs_construct_cleanly(self):
        """Valid 2-D inputs with positive ``dx``, ``dy`` construct
        without error and preserve the supplied pitch."""
        N = 16
        Ex = np.ones((N, N), dtype=complex)
        Ey = np.ones((N, N), dtype=complex)
        f = JonesField(Ex, Ey, dx=1e-6, dy=2e-6)
        assert f.dx == 1e-6
        assert f.dy == 2e-6
        assert f.Ex.shape == (N, N)
        # Default dy -> dx
        f2 = JonesField(Ex, Ey, dx=1e-6)
        assert f2.dy == 1e-6
