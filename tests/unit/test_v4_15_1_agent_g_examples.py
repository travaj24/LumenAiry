"""v4.15.1 Agent G -- end-to-end example tests.

This module pins a handful of canonical operator-algebra recipes a
user would actually type at the REPL: build a system algebraically,
read off ABCD / EFL, apply it to a Source, compose with apertures,
combine with prescriptions, etc.

Corresponds to ``CLUSTER_B_SPEC.md`` §3.7 ``test_algebra_examples.py``.

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
    FreeSpace,
    GaussianAperture,
    Magnify,
    Operator,
    ThinLens,
)

# ============================================================================
# Mini-example 1: 4f imaging system
# ============================================================================


class TestExample4fImaging:

    def test_build_4f_inspect_and_apply(self) -> None:
        """The canonical user journey: build the 4f system, inspect
        its ABCD, propagate a Gaussian through it."""
        f = 100e-3
        sys = (
            la.FreeSpace(f) * la.ThinLens(f) * la.FreeSpace(2 * f)
            * la.ThinLens(f) * la.FreeSpace(f)
        )
        # Read off properties without ever applying to a field
        assert np.allclose(
            sys.abcd, [[-1.0, 0.0], [0.0, -1.0]], atol=1e-10,
        )
        assert np.isinf(sys.efl)
        assert np.isclose(sys.A, -1.0)
        # Apply to a Source
        src = la.Source.gaussian(
            N=256, dx=2e-6, wavelength=633e-9, w0=80e-6,
        )
        out = sys(src)
        # Output should have ~the same energy as the input (no apertures)
        e_in = np.sum(np.abs(src.E) ** 2) * src.dx ** 2
        e_out = np.sum(np.abs(out.E) ** 2) * out.dx ** 2
        assert np.isclose(e_out, e_in, rtol=0.2)


# ============================================================================
# Mini-example 2: 2f Fourier transform setup
# ============================================================================


class TestExample2fFourierTransform:

    def test_2f_setup_matches_fourier_transform(self) -> None:
        """Composing ``FreeSpace(f) * ThinLens(f) * FreeSpace(f)`` gives
        a system whose ABCD equals :class:`FourierTransform(f)`."""
        from lumenairy.algebra import FourierTransform
        f = 0.1
        manual_2f = la.FreeSpace(f) * la.ThinLens(f) * la.FreeSpace(f)
        op_ft = FourierTransform(f)
        assert np.allclose(manual_2f.abcd, op_ft.abcd, atol=1e-12)


# ============================================================================
# Mini-example 3: anamorphic / cylindrical beam cleanup
# ============================================================================


class TestExampleAnamorphic:

    def test_anamorphic_system_is_flagged(self) -> None:
        """An astigmatic system built with a cylindrical lens reports
        as anamorphic."""
        # Pair of cylindrical lenses (x focus + y collimation) is
        # the classic anamorphic clean-up arrangement.
        sys = la.CylindricalLens(f_x=0.05, f_y=np.inf)
        assert sys.is_anamorphic
        # Cannot extract a scalar EFL
        with pytest.raises(ValueError):
            _ = sys.efl
        # But per-axis ABCDs are always available
        assert sys.abcd_x.shape == (2, 2)
        assert sys.abcd_y.shape == (2, 2)


# ============================================================================
# Mini-example 4: prescription -> operator -> field application
# ============================================================================


class TestExampleFromPrescription:

    def test_singlet_from_prescription_roundtrip(self) -> None:
        """Build an operator from a singlet prescription; read its EFL;
        apply it to a Source.  The EFL should be in the ballpark of
        what you'd expect from the lensmaker's equation."""
        rx = la.make_singlet(
            R1=50e-3, R2=-50e-3, d=5e-3,
            glass='N-BK7', aperture=25.4e-3,
        )
        wl = 633e-9
        op = la.Operator.from_prescription(rx, wl, method='asm')
        # Lensmaker's equation:
        # 1/f ~ (n - 1) (1/R1 - 1/R2) = 0.515 * (1/0.05 - 1/-0.05) ~ 20.6
        # so f ~ 0.0486 m.  The thin-lens-paraxial collapse gives a
        # slightly different number (~48 mm) -- pin in a reasonable band.
        assert 40e-3 < op.efl < 60e-3, f"singlet EFL out of range: {op.efl}"
        # Apply to a small Source.  Should produce a finite output.
        src = la.Source.gaussian(
            N=128, dx=4e-6, wavelength=wl, w0=200e-6,
        )
        out = op(src)
        assert np.all(np.isfinite(out.E))


# ============================================================================
# Mini-example 5: aperture + lens cascade
# ============================================================================


class TestExampleApertureCascade:

    def test_aperture_then_lens_then_propagate(self) -> None:
        """A common building block: aperture stop in front of a lens,
        followed by propagation to its back focal plane."""
        f = 0.05
        D = 5e-3
        sys = (
            la.FreeSpace(f, method='asm')
            * la.ThinLens(f)
            * la.Aperture(diameter=D, shape='circular')
        )
        src = la.Source.gaussian(
            N=256, dx=2e-6, wavelength=633e-9, w0=2e-3,
        )
        out = sys(src)
        # The aperture should have removed energy outside D/2.
        # The post-lens, post-propagate output should be a bright
        # central spot.  Pin the peak intensity ratio: peak / mean
        # should be substantially > 1 (focused).
        I = np.abs(out.E) ** 2
        ratio = float(I.max() / I.mean())
        assert ratio > 50.0, f"Output not focused (peak/mean = {ratio:.1f})"


# ============================================================================
# Mini-example 6: composing with Magnify
# ============================================================================


class TestExampleMagnify:

    def test_magnify_then_freespace(self) -> None:
        """``Magnify(2)`` halves the output pitch.  A subsequent
        FreeSpace propagation should run end-to-end on the rescaled
        grid."""
        src = la.Source.gaussian(
            N=128, dx=4e-6, wavelength=633e-9, w0=80e-6,
        )
        sys = la.FreeSpace(0.01, method='asm') * la.Magnify(2.0)
        out = sys(src)
        # Output pitch == input pitch / 2 (Magnify alters dx,
        # FreeSpace does not).
        assert np.isclose(out.dx, src.dx / 2.0, rtol=1e-6)
        assert np.all(np.isfinite(out.E))


# ============================================================================
# Mini-example 7: repr / readability
# ============================================================================


class TestExampleRepr:

    def test_primitives_have_useful_repr(self) -> None:
        assert 'FreeSpace' in repr(la.FreeSpace(0.1))
        assert '0.05' in repr(la.ThinLens(0.05))
        assert 'CylindricalLens' in repr(la.CylindricalLens(0.05))
        assert 'Magnify' in repr(la.Magnify(2.0))
        assert 'Aperture' in repr(la.Aperture(5e-3))

    def test_composite_repr_shows_chain(self) -> None:
        sys = la.FreeSpace(0.1) * la.ThinLens(0.05)
        r = repr(sys)
        assert 'FreeSpace' in r
        assert 'ThinLens' in r
        assert 'CompositeOperator' in r


# ============================================================================
# Mini-example 8: free-space cascade
# ============================================================================


class TestExampleFreeSpaceCascade:

    def test_two_freespaces_compose_to_sum_distance_abcd(self) -> None:
        """``FreeSpace(a) * FreeSpace(b)`` produces an ABCD equivalent
        to ``FreeSpace(a + b)``.  Useful sanity check for ABCD
        composition correctness."""
        a, b = 0.05, 0.07
        sys = la.FreeSpace(a) * la.FreeSpace(b)
        equivalent = la.FreeSpace(a + b)
        assert np.allclose(sys.abcd, equivalent.abcd)


# ============================================================================
# Mini-example 9: identity / no-op handling
# ============================================================================


class TestExampleIdentityHandling:

    def test_chain_of_identity_lenses_is_identity_abcd(self) -> None:
        sys = la.ThinLens(np.inf) * la.ThinLens(np.inf) * la.ThinLens(np.inf)
        assert np.allclose(sys.abcd, np.eye(2))

    def test_composite_with_apertures_only_is_identity_abcd(self) -> None:
        sys = (
            la.Aperture(5e-3) * la.GaussianAperture(1e-3)
            * la.Aperture(10e-3, shape='rectangular')
        )
        assert np.allclose(sys.abcd, np.eye(2))
        # But it still vignettes the field
        src = la.Source.gaussian(
            N=128, dx=2e-6, wavelength=633e-9, w0=80e-6,
        )
        out = sys(src)
        # Output should have less energy than the input
        e_in = np.sum(np.abs(src.E) ** 2)
        e_out = np.sum(np.abs(out.E) ** 2)
        assert e_out < e_in


# ============================================================================
# Mini-example 10: apply_with_intermediates for through-stage viz
# ============================================================================


class TestExampleIntermediates:

    def test_intermediates_chain_advances_distance(self) -> None:
        """For a free-space chain, every intermediate stage has the
        same N, and the field shape is preserved."""
        src = la.Source.gaussian(
            N=128, dx=2e-6, wavelength=633e-9, w0=80e-6,
        )
        sys = (la.FreeSpace(0.01, method='asm')
               * la.FreeSpace(0.02, method='asm')
               * la.FreeSpace(0.03, method='asm'))
        # Note: stages[0] is the rightmost in the algebra
        # expression -- the first to be applied.
        intermediates = sys.apply_with_intermediates(
            src.E, dx=src.dx, wavelength=src.wavelength,
        )
        assert len(intermediates) == 3
        for E_i, dx_i, dy_i in intermediates:
            assert E_i.shape == src.E.shape
            assert dx_i > 0
            assert dy_i > 0
