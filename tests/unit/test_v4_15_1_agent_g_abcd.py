"""v4.15.1 Agent G -- ABCD pin tests for the operator algebra layer.

This module pins the closed-form ABCD readout of every primitive in
:mod:`lumenairy.algebra` plus composition correctness
(``A * B`` -> ``A.abcd @ B.abcd``) and the anamorphic / isotropic
guard semantics.

Corresponds to ``CLUSTER_B_SPEC.md`` §3.7 ``test_algebra_abcd.py``.

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.algebra import (
    Aperture,
    CompositeOperator,
    CylindricalLens,
    FourierTransform,
    FreeSpace,
    GaussianAperture,
    Magnify,
    Operator,
    ThinLens,
)

# ============================================================================
# Individual primitives
# ============================================================================


class TestPrimitiveABCDs:
    """Closed-form ABCD of each primitive must match its spec entry."""

    def test_freespace_abcd(self) -> None:
        op = FreeSpace(0.1)
        assert np.allclose(op.abcd, [[1.0, 0.1], [0.0, 1.0]])
        # afocal -- C == 0 so efl == +inf
        assert np.isinf(op.efl)

    def test_freespace_zero_distance(self) -> None:
        op = FreeSpace(0.0)
        assert np.allclose(op.abcd, np.eye(2))
        assert np.isinf(op.efl)

    def test_freespace_negative_distance(self) -> None:
        """Back-propagation -- ABCD honours signed distance."""
        op = FreeSpace(-0.05)
        assert np.allclose(op.abcd, [[1.0, -0.05], [0.0, 1.0]])

    def test_thinlens_abcd(self) -> None:
        op = ThinLens(0.05)
        assert np.allclose(op.abcd, [[1.0, 0.0], [-20.0, 1.0]])
        assert np.isclose(op.efl, 0.05)

    def test_thinlens_infinite_focal_length_is_identity(self) -> None:
        op = ThinLens(np.inf)
        assert np.allclose(op.abcd, np.eye(2))
        assert np.isinf(op.efl)

    def test_thinlens_negative_focal_length(self) -> None:
        """Diverging lens -- C > 0, efl < 0."""
        op = ThinLens(-0.1)
        assert np.allclose(op.abcd, [[1.0, 0.0], [10.0, 1.0]])
        assert np.isclose(op.efl, -0.1)

    def test_thinlens_zero_focal_length_raises(self) -> None:
        with pytest.raises(ValueError):
            ThinLens(0.0)

    def test_cylindrical_lens_isotropic_when_equal(self) -> None:
        """f_x == f_y reduces to an isotropic operator (no guard
        violation)."""
        op = CylindricalLens(f_x=0.05, f_y=0.05)
        assert not op.is_anamorphic
        assert np.allclose(op.abcd, [[1.0, 0.0], [-20.0, 1.0]])

    def test_cylindrical_lens_x_only(self) -> None:
        op = CylindricalLens(f_x=0.05, f_y=np.inf)
        assert op.is_anamorphic
        assert np.allclose(op.abcd_x, [[1.0, 0.0], [-20.0, 1.0]])
        assert np.allclose(op.abcd_y, np.eye(2))

    def test_cylindrical_lens_y_only(self) -> None:
        op = CylindricalLens(f_x=np.inf, f_y=0.05)
        assert op.is_anamorphic
        assert np.allclose(op.abcd_x, np.eye(2))
        assert np.allclose(op.abcd_y, [[1.0, 0.0], [-20.0, 1.0]])

    def test_magnify_isotropic(self) -> None:
        op = Magnify(2.0)
        # diag(1/a, a)
        assert np.allclose(op.abcd, [[0.5, 0.0], [0.0, 2.0]])
        assert not op.is_anamorphic

    def test_magnify_anamorphic(self) -> None:
        op = Magnify(a_x=2.0, a_y=3.0)
        assert op.is_anamorphic
        assert np.allclose(op.abcd_x, [[0.5, 0.0], [0.0, 2.0]])
        assert np.allclose(op.abcd_y, [[1.0 / 3.0, 0.0], [0.0, 3.0]])

    def test_magnify_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            Magnify(-1.0)

    def test_fourier_transform_abcd(self) -> None:
        f = 0.1
        op = FourierTransform(f)
        assert np.allclose(op.abcd, [[0.0, f], [-1.0 / f, 0.0]])

    def test_fourier_transform_nonpositive_raises(self) -> None:
        with pytest.raises(ValueError):
            FourierTransform(-0.1)
        with pytest.raises(ValueError):
            FourierTransform(0.0)

    def test_aperture_abcd_is_identity(self) -> None:
        op = Aperture(diameter=5e-3)
        assert np.allclose(op.abcd, np.eye(2))

    def test_aperture_shape_validation(self) -> None:
        with pytest.raises(ValueError):
            Aperture(diameter=5e-3, shape='hexagonal')

    def test_gaussian_aperture_abcd_is_identity(self) -> None:
        op = GaussianAperture(sigma=1e-3)
        assert np.allclose(op.abcd, np.eye(2))

    def test_gaussian_aperture_nonpositive_raises(self) -> None:
        with pytest.raises(ValueError):
            GaussianAperture(sigma=-1e-3)


# ============================================================================
# Composition
# ============================================================================


class TestComposition:
    """``A * B`` -> ``A.abcd @ B.abcd``; chains compose associatively."""

    def test_composition_matches_matmul(self) -> None:
        """4f imaging system ABCD matches the matmul of its pieces and
        equals the [[-1, 0], [0, -1]] inverter."""
        f = 0.1
        sys = (
            FreeSpace(f) * ThinLens(f) * FreeSpace(2 * f)
            * ThinLens(f) * FreeSpace(f)
        )
        expected = (
            np.array([[1, f], [0, 1]])
            @ np.array([[1, 0], [-1 / f, 1]])
            @ np.array([[1, 2 * f], [0, 1]])
            @ np.array([[1, 0], [-1 / f, 1]])
            @ np.array([[1, f], [0, 1]])
        )
        assert np.allclose(sys.abcd, expected, atol=1e-12)
        # 4f inverter
        assert np.allclose(
            sys.abcd, [[-1.0, 0.0], [0.0, -1.0]], atol=1e-10,
        )

    def test_two_lens_composition_combines_powers(self) -> None:
        """Two thin lenses in contact: 1/f_total = 1/f1 + 1/f2."""
        f1, f2 = 0.1, 0.2
        sys = ThinLens(f1) * ThinLens(f2)
        # Composite ABCD via direct matmul
        M1 = np.array([[1.0, 0.0], [-1.0 / f1, 1.0]])
        M2 = np.array([[1.0, 0.0], [-1.0 / f2, 1.0]])
        assert np.allclose(sys.abcd, M1 @ M2)
        # EFL of the pair (in-contact): 1/f = 1/f1 + 1/f2
        f_total = 1.0 / (1.0 / f1 + 1.0 / f2)
        assert np.isclose(sys.efl, f_total)

    def test_composition_associativity(self) -> None:
        """(A * B) * C == A * (B * C) for ABCDs."""
        a = FreeSpace(0.01)
        b = ThinLens(0.05)
        c = FreeSpace(0.02)
        left = (a * b) * c
        right = a * (b * c)
        assert np.allclose(left.abcd, right.abcd)

    def test_composition_flattens_chains(self) -> None:
        """Nested CompositeOperators flatten; the resulting chain
        has all five pieces."""
        f = 0.1
        a = FreeSpace(f) * ThinLens(f)
        b = FreeSpace(2 * f)
        c = ThinLens(f) * FreeSpace(f)
        sys = a * b * c
        assert isinstance(sys, CompositeOperator)
        # 5 operators total
        assert len(sys.stages()) == 5

    def test_composition_with_aperture_preserves_abcd(self) -> None:
        """Apertures have identity ABCD -- they should not change the
        composite ABCD."""
        f = 0.1
        base = FreeSpace(f) * ThinLens(f) * FreeSpace(f)
        with_ap = FreeSpace(f) * Aperture(5e-3) * ThinLens(f) * FreeSpace(f)
        assert np.allclose(base.abcd, with_ap.abcd)

    def test_anamorphic_cylindrical_lens_through_freespace(self) -> None:
        """Anamorphic system retains per-axis ABCDs through composition
        with isotropic FreeSpace."""
        op = CylindricalLens(f_x=np.inf, f_y=0.05)
        sys = FreeSpace(0.02) * op * FreeSpace(0.01)
        assert sys.is_anamorphic
        # x-leg should be free-space (no power)
        assert np.allclose(
            sys.abcd_x, [[1.0, 0.03], [0.0, 1.0]], atol=1e-12,
        )
        # y-leg includes the cylindrical lens
        M_fs_post = np.array([[1.0, 0.02], [0.0, 1.0]])
        M_lens = np.array([[1.0, 0.0], [-20.0, 1.0]])
        M_fs_pre = np.array([[1.0, 0.01], [0.0, 1.0]])
        assert np.allclose(sys.abcd_y, M_fs_post @ M_lens @ M_fs_pre)


# ============================================================================
# Anamorphic / isotropic guards
# ============================================================================


class TestAnamorphicGuards:
    """``.abcd`` / ``.A`` / ``.B`` / ``.C`` / ``.D`` / ``.efl`` must
    raise on anamorphic systems."""

    def test_abcd_raises_when_anamorphic(self) -> None:
        op = CylindricalLens(f_x=np.inf, f_y=0.05)
        with pytest.raises(ValueError, match='anamorphic'):
            _ = op.abcd

    def test_efl_raises_when_anamorphic(self) -> None:
        op = CylindricalLens(f_x=np.inf, f_y=0.05)
        with pytest.raises(ValueError, match='anamorphic'):
            _ = op.efl

    def test_scalar_A_raises_when_anamorphic(self) -> None:
        op = CylindricalLens(f_x=np.inf, f_y=0.05)
        for prop_name in ('A', 'B', 'C', 'D'):
            with pytest.raises(ValueError, match='anamorphic'):
                _ = getattr(op, prop_name)

    def test_abcd_x_and_abcd_y_always_available(self) -> None:
        op = CylindricalLens(f_x=0.02, f_y=0.05)
        # These never raise -- they're the per-axis accessors.
        assert op.abcd_x.shape == (2, 2)
        assert op.abcd_y.shape == (2, 2)

    def test_isotropic_operator_passes_guards(self) -> None:
        op = ThinLens(0.1)
        # All scalars accessible without raising
        assert op.A == 1.0
        assert op.B == 0.0
        assert op.C == -10.0
        assert op.D == 1.0
        assert np.isclose(op.efl, 0.1)


# ============================================================================
# Top-level export sanity
# ============================================================================


class TestTopLevelExports:
    """Symbols must be reachable from the top-level lumenairy namespace
    (the operator-algebra public API contract)."""

    @pytest.mark.parametrize('sym', [
        'Operator',
        'CompositeOperator',
        'FreeSpace',
        'ThinLens',
        'CylindricalLens',
        'Magnify',
        'FourierTransform',
        'Aperture',
        'GaussianAperture',
    ])
    def test_symbol_exported(self, sym: str) -> None:
        assert hasattr(la, sym), f"lumenairy.{sym} missing from top-level"
        assert sym in la.__all__, f"'{sym}' missing from lumenairy.__all__"

    def test_from_prescription_is_classmethod(self) -> None:
        """``Operator.from_prescription(...)`` is the canonical factory
        for prescription -> CompositeOperator."""
        assert hasattr(Operator, 'from_prescription')
        assert callable(Operator.from_prescription)
