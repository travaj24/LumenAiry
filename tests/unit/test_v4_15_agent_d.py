"""Unit tests for v4.15 Agent D scope: Forbes Q-type freeform surfaces
(Q-bfs, Q-con) and the ``make_off_axis_parabola`` prescription factory.

Scope
-----

* **D.1** -- ``lumenairy.surface_sag_q_bfs`` + ``surface_sag_q_con``:
  Forbes 2007 (Q-bfs, Eq. 13) and Forbes 2010 (Q-con, Eq. 6)
  orthonormal polynomial bases for axially-symmetric optical
  freeforms.  See ``lumenairy/elements/freeform.py``.
* **D.2** -- ``lumenairy.make_off_axis_parabola``: builder for an
  off-axis-parabola (OAP) prescription dict.

References
----------

* G. W. Forbes, "Shape specification for axially symmetric optical
  surfaces," Opt. Express 15(8), 5218-5226 (2007).
* G. W. Forbes, "Robust, efficient computational methods for axially
  symmetric optical aspheres," Opt. Express 18(13), 13851-13862
  (2010).

Author: Andrew Traverso -- v4.15 / Agent D
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from lumenairy.elements.freeform import (
    _q_bfs_eval,
    _q_con_eval,
    surface_sag_freeform,
    surface_sag_q_bfs,
    surface_sag_q_con,
)
from lumenairy.elements.lenses import surface_sag_general
from lumenairy.io.prescriptions import make_off_axis_parabola

# ============================================================================
# Shared fixtures
# ============================================================================

def _ortho_quadrature_grid(N=200001):
    """Dense ``u`` grid in [0, 1] for trapezoidal integration of the
    Forbes orthonormality identities.  ``N`` chosen so the integrals
    are well below the 1e-6 tolerance demanded by the audit task."""
    u = np.linspace(0.0, 1.0, N)
    return u, u ** 2


def _build_2d_grid(N=64, half_extent=5e-3):
    """Square aperture grid for the 2-D sag evaluation tests."""
    x = np.linspace(-half_extent, half_extent, N)
    y = np.linspace(-half_extent, half_extent, N)
    X, Y = np.meshgrid(x, y)
    return X, Y


# ============================================================================
# D.1 -- Forbes Q-bfs tests
# ============================================================================

class TestForbesQbfsOrthonormality:
    """Pin the orthonormality identity
    ``int_0^1 Q_m(u^2) Q_n(u^2) u^2 (1 - u^2) d(u^2) = delta_{mn}``
    using a high-resolution trapezoidal quadrature."""

    def test_q_bfs_orthonormality_first_5_orders(self):
        """For m, n in 0..4 the matrix ``<Q_m, Q_n>`` must be the
        5x5 identity to within 1e-6."""
        u, u_sq = _ortho_quadrature_grid()
        # Change of variable: d(u^2) = 2u du, so weight u^2(1-u^2)
        # d(u^2) becomes (u^2 (1-u^2) * 2u) du in the trapezoidal
        # integral over u in [0, 1].
        weight = u_sq * (1.0 - u_sq) * 2.0 * u
        for m in range(5):
            Q_m = _q_bfs_eval(u_sq, m)
            for n in range(5):
                Q_n = _q_bfs_eval(u_sq, n)
                val = np.trapezoid(Q_m * Q_n * weight, u)
                expected = 1.0 if m == n else 0.0
                assert abs(val - expected) < 1e-6, (
                    f"Q-bfs <Q_{m}, Q_{n}> = {val:.6e}; expected "
                    f"{expected:.0f} (orthonormality violated).")


class TestForbesQbfsSagShape:
    """Pin the Eq. 13 surface-sag definition end-to-end."""

    def test_q_bfs_zero_coefficients_returns_best_fit_sphere(self):
        """With all-zero coefficients the returned sag must be the
        best-fit sphere ``z_bfs(r)`` to 1e-12 absolute."""
        X, Y = _build_2d_grid()
        R = 50e-3
        sag = surface_sag_q_bfs(
            X, Y,
            radius=R, coefficients=[0.0, 0.0, 0.0, 0.0],
            r_max=4e-3, dx=1e-4, dy=1e-4,
            norm_x=5e-3, norm_y=5e-3,
        )
        # z_bfs = surface_sag_general(h^2, R, conic=0).
        sag_bfs = surface_sag_general(X ** 2 + Y ** 2, R, 0.0)
        diff = np.max(np.abs(sag - sag_bfs))
        assert diff < 1e-12, (
            f"Zero-coefficient Q-bfs sag deviates from pure best-fit "
            f"sphere by {diff:.3e} (expected < 1e-12).")

    def test_q_bfs_pure_a0_known_shape(self):
        """Single-coefficient case ``a_0 != 0``: the departure must
        match the direct formula ``u^2 (1 - u^2) * a_0 * Q_0(u^2)``
        exactly, with the v4.15.1 P1-F1-1 radial clip
        ``r > r_max`` applied (departure zeroed there)."""
        X, Y = _build_2d_grid()
        r_max = 4e-3
        a0 = 2.5e-7
        sag = surface_sag_q_bfs(
            X, Y, radius=np.inf,
            coefficients=[a0],
            r_max=r_max, dx=1e-4, dy=1e-4,
            norm_x=1.0, norm_y=1.0,  # rectangular clip non-restrictive
        )
        u_sq = (X ** 2 + Y ** 2) / (r_max * r_max)
        departure = u_sq * (1.0 - u_sq) * a0 * _q_bfs_eval(u_sq, 0)
        # v4.15.1 P1-F1-1: radial clip zeros pixels outside the
        # Forbes unit disc ``u <= 1``.
        expected = np.where(u_sq > 1.0, 0.0, departure)
        diff = np.max(np.abs(sag - expected))
        assert diff < 1e-15, (
            f"Pure-a0 Q-bfs sag mismatch: {diff:.3e}.")

    def test_q_bfs_negative_norm_x_rejected(self):
        """v4.14.3 P1-NEW-11 convention: negative ``norm_x`` /
        ``norm_y`` must raise ``ValueError`` at the call site."""
        X, Y = _build_2d_grid()
        with pytest.raises(ValueError, match="norm_x"):
            surface_sag_q_bfs(
                X, Y, radius=np.inf,
                coefficients=[1e-7], r_max=4e-3,
                dx=1e-4, dy=1e-4,
                norm_x=-0.05, norm_y=5e-3,
            )
        with pytest.raises(ValueError, match="norm_y"):
            surface_sag_q_bfs(
                X, Y, radius=np.inf,
                coefficients=[1e-7], r_max=4e-3,
                dx=1e-4, dy=1e-4,
                norm_x=5e-3, norm_y=0.0,
            )

    def test_q_bfs_off_aperture_clipped_to_zero(self):
        """Outside the ``[-norm_x, norm_x] x [-norm_y, norm_y]`` box,
        the freeform departure must be zero (matching the v4.14.2
        Chebyshev / XY-polynomial out-of-domain guard)."""
        X, Y = _build_2d_grid(N=64, half_extent=5e-3)
        norm = 1e-3  # tight; most of the grid lies outside
        sag = surface_sag_q_bfs(
            X, Y, radius=np.inf,
            coefficients=[3e-7, -1e-7, 2e-8],
            r_max=norm, dx=1e-4, dy=1e-4,
            norm_x=norm, norm_y=norm,
        )
        outside = (np.abs(X) > norm) | (np.abs(Y) > norm)
        # Base sag is zero (R=inf, conic=0); so outside region must
        # be all-zero.
        assert np.allclose(sag[outside], 0.0, atol=1e-15), (
            f"Off-aperture sag not zeroed: max abs = "
            f"{float(np.max(np.abs(sag[outside]))):.3e}.")

    def test_q_bfs_dy_default_equals_dx(self):
        """``dy=None`` (default) must produce the same sag as
        ``dy=dx`` -- the anamorphic kwarg behaviour."""
        X, Y = _build_2d_grid()
        coeffs = [3e-7, -1e-7]
        kw = dict(
            radius=np.inf, coefficients=coeffs,
            r_max=4e-3, dx=1e-4, norm_x=5e-3, norm_y=5e-3,
        )
        sag_default = surface_sag_q_bfs(X, Y, **kw)
        sag_explicit = surface_sag_q_bfs(X, Y, dy=1e-4, **kw)
        assert np.array_equal(sag_default, sag_explicit)

    def test_q_bfs_negative_r_max_rejected(self):
        """Edge-case extension of P1-NEW-11 to the ``r_max``
        normalisation: zero or negative ``r_max`` is rejected so a
        zero-divide does not silently produce NaN."""
        X, Y = _build_2d_grid()
        with pytest.raises(ValueError, match="r_max"):
            surface_sag_q_bfs(
                X, Y, radius=np.inf,
                coefficients=[1e-7], r_max=0.0,
                dx=1e-4, dy=1e-4, norm_x=5e-3, norm_y=5e-3,
            )


# ============================================================================
# D.1 -- Forbes Q-con tests
# ============================================================================

class TestForbesQconOrthonormality:
    """Pin the Q-con orthonormality identity
    ``int_0^1 Q_m(u^2) Q_n(u^2) u^4 d(u^2) = delta_{mn}``."""

    def test_q_con_orthonormality_first_5_orders(self):
        u, u_sq = _ortho_quadrature_grid()
        # d(u^2) = 2u du; weight u^4 becomes (u^4 * 2u) du.
        weight = u_sq * u_sq * 2.0 * u
        for m in range(5):
            Q_m = _q_con_eval(u_sq, m)
            for n in range(5):
                Q_n = _q_con_eval(u_sq, n)
                val = np.trapezoid(Q_m * Q_n * weight, u)
                expected = 1.0 if m == n else 0.0
                assert abs(val - expected) < 1e-6, (
                    f"Q-con <Q_{m}, Q_{n}> = {val:.6e}; expected "
                    f"{expected:.0f}.")


class TestForbesQconSagShape:

    def test_q_con_zero_coefficients_returns_pure_conic(self):
        """With all-zero coefficients the sag is the base conic
        ``z_conic(r) = c r^2 / (1 + sqrt(1 - (1+k) c^2 r^2))`` to
        1e-12."""
        X, Y = _build_2d_grid()
        R = 50e-3
        k = -0.7
        sag = surface_sag_q_con(
            X, Y, radius=R, conic=k,
            coefficients=[0.0, 0.0, 0.0],
            r_max=4e-3, dx=1e-4, dy=1e-4,
            norm_x=5e-3, norm_y=5e-3,
        )
        sag_conic = surface_sag_general(X ** 2 + Y ** 2, R, k)
        diff = np.max(np.abs(sag - sag_conic))
        assert diff < 1e-12, (
            f"Zero-coefficient Q-con sag deviates from pure base "
            f"conic by {diff:.3e}.")

    def test_q_con_pure_a0_known_shape(self):
        """Single-coefficient case: departure must equal
        ``u^4 * a_0 * Q_0^con(u^2)`` directly, with v4.15.1 P1-F1-1
        radial clip applied outside ``u <= 1``."""
        X, Y = _build_2d_grid()
        r_max = 4e-3
        a0 = 5e-7
        sag = surface_sag_q_con(
            X, Y, radius=np.inf, conic=-1.0,
            coefficients=[a0],
            r_max=r_max, dx=1e-4, dy=1e-4,
            norm_x=1.0, norm_y=1.0,
        )
        u_sq = (X ** 2 + Y ** 2) / (r_max * r_max)
        departure = u_sq * u_sq * a0 * _q_con_eval(u_sq, 0)
        # v4.15.1 P1-F1-1 radial clip.
        expected = np.where(u_sq > 1.0, 0.0, departure)
        # Base sag for R=inf is identically zero, so the comparison is
        # against the freeform departure alone.
        diff = np.max(np.abs(sag - expected))
        assert diff < 1e-15, (
            f"Pure-a0 Q-con sag mismatch: {diff:.3e}.")

    def test_q_con_off_aperture_clipped_to_zero(self):
        X, Y = _build_2d_grid(N=64, half_extent=5e-3)
        norm = 1e-3
        sag = surface_sag_q_con(
            X, Y, radius=np.inf, conic=0.0,
            coefficients=[3e-7, -1e-7],
            r_max=norm, dx=1e-4, dy=1e-4,
            norm_x=norm, norm_y=norm,
        )
        outside = (np.abs(X) > norm) | (np.abs(Y) > norm)
        assert np.allclose(sag[outside], 0.0, atol=1e-15)

    def test_q_con_negative_r_max_rejected(self):
        X, Y = _build_2d_grid()
        with pytest.raises(ValueError, match="r_max"):
            surface_sag_q_con(
                X, Y, radius=np.inf, conic=0.0,
                coefficients=[1e-7], r_max=-1.0,
                dx=1e-4, dy=1e-4, norm_x=5e-3, norm_y=5e-3,
            )

    def test_q_con_independent_of_q_bfs_basis(self):
        """Sanity check: the Q-con basis is *not* the Q-bfs basis.
        For the same coefficient vector and ``r_max``, the freeform
        departures differ both in radial prefactor (``u^4`` vs.
        ``u^2(1-u^2)``) and polynomial values (Q-con has weight
        ``x^2`` orthonormal; Q-bfs has weight ``x(1-x)``).  Pin that
        the two sag surfaces are not numerically equal -- protects
        against accidentally pointing Q-con at the Q-bfs evaluator
        in a future refactor."""
        X, Y = _build_2d_grid()
        coeffs = [2e-7, -1e-7, 5e-8]
        sag_bfs = surface_sag_q_bfs(
            X, Y, radius=np.inf, coefficients=coeffs,
            r_max=4e-3, dx=1e-4, dy=1e-4,
            norm_x=5e-3, norm_y=5e-3,
        )
        sag_con = surface_sag_q_con(
            X, Y, radius=np.inf, conic=0.0,
            coefficients=coeffs,
            r_max=4e-3, dx=1e-4, dy=1e-4,
            norm_x=5e-3, norm_y=5e-3,
        )
        # Distinct prefactors and basis -> max diff must be
        # non-trivial (well above floating-point noise).
        assert float(np.max(np.abs(sag_bfs - sag_con))) > 1e-9


# ============================================================================
# D.1 -- Dispatch via ``surface_sag_freeform``
# ============================================================================

class TestFreeformDispatch:
    """The legacy dispatcher must recognise the new ``q_bfs`` /
    ``q_con`` ``freeform_type`` strings."""

    def test_dispatch_q_bfs(self):
        X, Y = _build_2d_grid()
        a0 = 1e-7
        surf_dict = {
            'freeform_type': 'q_bfs',
            'radius': float('inf'),
            'q_bfs_coeffs': [a0],
            'r_max': 4e-3,
            'dx': 1e-4,
            'norm_x': 5e-3,
            'norm_y': 5e-3,
        }
        sag_disp = surface_sag_freeform(X, Y, surf_dict)
        sag_direct = surface_sag_q_bfs(
            X, Y, radius=np.inf, coefficients=[a0],
            r_max=4e-3, dx=1e-4, norm_x=5e-3, norm_y=5e-3,
        )
        assert np.array_equal(sag_disp, sag_direct)

    def test_dispatch_q_con(self):
        X, Y = _build_2d_grid()
        a0 = 1e-7
        surf_dict = {
            'freeform_type': 'q_con',
            'radius': 50e-3,
            'conic': -1.0,
            'q_con_coeffs': [a0],
            'r_max': 4e-3,
            'dx': 1e-4,
            'norm_x': 5e-3,
            'norm_y': 5e-3,
        }
        sag_disp = surface_sag_freeform(X, Y, surf_dict)
        sag_direct = surface_sag_q_con(
            X, Y, radius=50e-3, conic=-1.0, coefficients=[a0],
            r_max=4e-3, dx=1e-4, norm_x=5e-3, norm_y=5e-3,
        )
        assert np.array_equal(sag_disp, sag_direct)


# ============================================================================
# D.2 -- Off-axis parabola factory tests
# ============================================================================

class TestOffAxisParabolaFactory:

    def test_valid_inputs_return_prescription_with_expected_fields(self):
        """A typical OAP (f=150 mm, theta=15 deg, D=50 mm) must yield
        a single-surface dict with conic=-1, R=2f, decenter=(f tan
        theta, 0), tilt=(theta, 0, 0), aperture=clear_aperture."""
        f = 150e-3
        theta = math.radians(15.0)
        D = 50e-3
        rx = make_off_axis_parabola(
            focal_length=f, off_axis_angle=theta, clear_aperture=D)
        # Top-level structure.
        assert 'surfaces' in rx
        assert len(rx['surfaces']) == 1
        assert rx['aperture_diameter'] == pytest.approx(D)
        surf = rx['surfaces'][0]
        # Conic = -1 (paraboloid).
        assert surf['conic'] == pytest.approx(-1.0)
        # Vertex radius = 2 f.
        assert surf['radius'] == pytest.approx(2.0 * f)
        # Decenter -- v4.15.1 (P0-NEW-1 Bug 2): the chief-ray launch
        # radius on the parent paraboloid is 2*f*tan(alpha), where
        # alpha is the surface-normal off-axis angle.  Pre-v4.15.1
        # this assertion pinned the buggy ``f*tan(theta)`` form.
        dx_expected = 2.0 * f * math.tan(theta)
        assert surf['decenter'][0] == pytest.approx(dx_expected)
        assert surf['decenter'][1] == pytest.approx(0.0)
        # Tilt (3-tuple per task spec).
        assert len(surf['tilt']) == 3
        assert surf['tilt'][0] == pytest.approx(theta)
        assert surf['tilt'][1] == pytest.approx(0.0)
        assert surf['tilt'][2] == pytest.approx(0.0)
        # Aperture.
        assert surf['clear_aperture'] == pytest.approx(D)

    def test_zero_or_negative_focal_length_raises(self):
        """``focal_length <= 0`` is rejected."""
        with pytest.raises(ValueError, match="focal_length"):
            make_off_axis_parabola(
                focal_length=0.0,
                off_axis_angle=math.radians(15.0),
                clear_aperture=50e-3,
            )
        with pytest.raises(ValueError, match="focal_length"):
            make_off_axis_parabola(
                focal_length=-1e-2,
                off_axis_angle=math.radians(15.0),
                clear_aperture=50e-3,
            )

    def test_off_axis_angle_at_or_beyond_pi_over_2_raises(self):
        """``off_axis_angle >= pi/2`` is rejected (would force the
        OAP centre to infinity).  Also ``<= 0`` is rejected (would
        degenerate to an on-axis parabola)."""
        with pytest.raises(ValueError, match="off_axis_angle"):
            make_off_axis_parabola(
                focal_length=150e-3,
                off_axis_angle=math.pi / 2,
                clear_aperture=50e-3,
            )
        with pytest.raises(ValueError, match="off_axis_angle"):
            make_off_axis_parabola(
                focal_length=150e-3,
                off_axis_angle=0.0,
                clear_aperture=50e-3,
            )

    def test_zero_or_negative_clear_aperture_raises(self):
        with pytest.raises(ValueError, match="clear_aperture"):
            make_off_axis_parabola(
                focal_length=150e-3,
                off_axis_angle=math.radians(15.0),
                clear_aperture=0.0,
            )

    def test_glass_mirror_sentinel_produces_reflective_surface(self):
        """The default ``glass='__MIRROR__'`` must mark the surface
        as reflective via the standard ``is_mirror`` flag elsewhere
        in this module."""
        rx = make_off_axis_parabola(
            focal_length=150e-3,
            off_axis_angle=math.radians(15.0),
            clear_aperture=50e-3,
        )
        surf = rx['surfaces'][0]
        assert surf.get('is_mirror') is True, (
            "Default __MIRROR__ glass must mark the surface as a "
            "mirror via the is_mirror flag.")

    def test_refractive_glass_produces_non_mirror_surface(self):
        """Passing a glass-catalog name (e.g. ``N-BK7``) yields a
        refractive parabolic element instead of a mirror, exercising
        the alternate code path of the factory."""
        rx = make_off_axis_parabola(
            focal_length=150e-3,
            off_axis_angle=math.radians(15.0),
            clear_aperture=50e-3,
            glass='N-BK7',
        )
        surf = rx['surfaces'][0]
        assert surf.get('is_mirror') is False
        assert surf['glass_after'] == 'N-BK7'

    def test_vertex_radius_override(self):
        """``vertex_radius`` kwarg overrides the geometric
        ``R = 2 f`` default."""
        rx = make_off_axis_parabola(
            focal_length=150e-3,
            off_axis_angle=math.radians(15.0),
            clear_aperture=50e-3,
            vertex_radius=400e-3,
        )
        assert rx['surfaces'][0]['radius'] == pytest.approx(400e-3)


# ============================================================================
# Public-name resolution pin (ensures __init__.py exports the new
# v4.15 Agent D names)
# ============================================================================

def test_public_names_resolvable():
    """The three new public names must be importable from the
    top-level ``lumenairy`` namespace."""
    import lumenairy
    for name in ('surface_sag_q_bfs', 'surface_sag_q_con',
                 'make_off_axis_parabola'):
        assert hasattr(lumenairy, name), (
            f"lumenairy.{name} is not exported.")
        assert name in lumenairy.__all__, (
            f"{name!r} is missing from lumenairy.__all__.")
