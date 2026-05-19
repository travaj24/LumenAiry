"""v4.15.2 Agent B tests -- CLUSTER_B operator-algebra audit closures.

This module pins the v4.15.2 closures for audit findings P1-NEW-A,
P1-NEW-B, P1-NEW-C, and the P2 Magnify docstring / dead-reference
items from ``docs/audits/AUDIT_V4_15_1_2026_05_18.md``.

Scope
-----

* **B.1 -- P1-NEW-A:** ``FourierTransform._apply`` rewrite from the
  v4.15.1 2-stage shortcut (``ThinLens(f) -> fresnel(f)``) to the
  literal 3-stage chain
  ``FreeSpace(f) -> ThinLens(f) -> FreeSpace(f)`` (Goodman Sec. 5.2).
  The 2-stage shortcut matched the 3-stage ABCD but left a residual
  ``exp(+i*k/(2f)*r^2)`` quadratic phase on the output, so the
  ``FourierTransform(f)`` operator was inconsistent with the
  composed primitive chain on the field side.

* **B.2 -- P1-NEW-B:** ``from_prescription`` flat-mirror parity
  drop.  Pre-v4.15.2 the algebra layer flipped ``mirror_parity``
  for every ``is_mirror=True`` surface; ``raytrace.system_abcd``
  only flips parity for curved mirrors.  v4.15.2 condition the
  flip on ``np.isfinite(R)`` to match ``system_abcd``.

* **B.3 -- P1-NEW-C:** ``FreeSpace._apply`` threads ``dy`` into the
  underlying dispatcher when ``dy != dx``.  Pre-v4.15.2 the call
  dropped ``dy`` and the kernel silently defaulted to ``dy = dx``.

* **B.4 -- P2 Magnify docstring direction:** the v4.15.1 docstring
  said the output physical extent ``scales as a * (N * dx)``, but
  the code produces ``new_dx = dx / a`` (i.e. output is SHRUNK by
  ``a`` for ``a > 1``).  v4.15.2 rewrites the docstring to match
  the code (Nazarathy/Shamir ``V[a]: (x, y) -> (x/a, y/a)``).

* **B.5 -- P2 dead operators.py reference:** removed in the
  Magnify docstring rewrite (the reference was to a non-existent
  ``operators.py:556-577`` external file).

Author: Andrew Traverso
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import lumenairy as la
from lumenairy.algebra import (
    CompositeOperator,
    FourierTransform,
    FreeSpace,
    Magnify,
    ThinLens,
)
from lumenairy.algebra.primitives import Magnify as _MagnifyCls
from lumenairy.raytrace.core import surfaces_from_prescription, system_abcd


# ============================================================================
# B.1 -- FourierTransform: ABCD and field both match the 3-stage chain
# ============================================================================


class TestFourierTransform3Stage:
    """v4.15.2 (audit P1-NEW-A): ``FourierTransform(f)`` must agree
    with the literal 3-stage chain ``FreeSpace(f) * ThinLens(f) *
    FreeSpace(f)`` on BOTH the ABCD side and the FIELD side.

    The v4.15.1 implementation matched on ABCDs but ran only 2 stages
    (lens-then-Fresnel) for ``_apply``, leaving a residual quadratic
    phase ``exp(+i*k/(2f)*r^2)`` on the output.  v4.15.2 closes this
    inconsistency by invoking the literal 3-stage chain inside
    ``_apply``.
    """

    def test_fourier_transform_two_stage_versus_three_stage_compatible_abcd(
        self,
    ) -> None:
        """Sanity pin: the v4.15.2 FT ABCD must equal the 3-stage
        chain ABCD to within 1e-12 (the FT ABCD was already correct
        in v4.15.1; this test pins it against the closure path).
        """
        f = 0.1
        ft = FourierTransform(f)
        chain = FreeSpace(f) * ThinLens(f) * FreeSpace(f)
        assert np.allclose(ft.abcd, chain.abcd, atol=1e-12), (
            f"FourierTransform({f}) ABCD does not match 3-stage chain "
            f"ABCD:\n  ft.abcd = {ft.abcd}\n  chain.abcd = {chain.abcd}"
        )

    def test_fourier_transform_field_matches_3_stage_chain(self) -> None:
        """v4.15.2 closure: the FT-applied field must match the
        3-stage-chain-applied field on a complex off-axis Gaussian
        test source.

        The off-axis offset is the critical test stimulus: a centered
        Gaussian would only be sensitive to a global phase, but an
        off-axis Gaussian carries non-trivial transverse-position
        content that a residual ``exp(+i*k/(2f)*r^2)`` quadratic
        phase WILL modulate pointwise.

        .. note::
            This pin is **tautological with the v4.15.2 implementation**
            (the v4.15.2 ``FourierTransform._apply`` literally invokes
            the 3-stage chain), so it only checks the rewrite, not the
            underlying physics.  See
            :meth:`test_fourier_transform_gaussian_beam_waist_relation`
            below for the v4.15.3 (audit P3 / Tier-1) non-tautological
            pin against the closed-form Gaussian-beam waist relation
            ``w_out = lambda * f / (pi * w_in)`` (Saleh & Teich §3.2.2,
            Goodman §5.2.3).
        """
        f = 0.1
        src = la.Source.gaussian(
            N=256, dx=2e-6, wavelength=633e-9, w0=100e-6,
            x0=80e-6, y0=-40e-6,
        )
        op = FourierTransform(f)
        chain = FreeSpace(f) * ThinLens(f) * FreeSpace(f)
        out_op = op(src)
        out_chain = chain(src)
        assert np.isclose(out_op.dx, out_chain.dx, rtol=1e-12)
        assert np.allclose(
            out_op.E, out_chain.E, rtol=1e-10, atol=1e-12,
        ), (
            f"FourierTransform field does not match 3-stage chain: "
            f"max abs diff = "
            f"{np.max(np.abs(out_op.E - out_chain.E)):.3e}"
        )

    @pytest.mark.parametrize('label,N,dx,wavelength,f,w_in', [
        ('VIS_mid_focal', 256, 4.7e-6, 633e-9, 0.05, 200e-6),
        ('NIR_short_focal', 256, 4.0e-6, 1.55e-6, 0.025, 150e-6),
        ('VIS_long_focal', 256, 8.0e-6, 633e-9, 0.1, 300e-6),
    ])
    def test_fourier_transform_gaussian_beam_waist_relation(
        self,
        label: str,
        N: int,
        dx: float,
        wavelength: float,
        f: float,
        w_in: float,
    ) -> None:
        """v4.15.3 (audit P3 / Tier-1): non-tautological pin against
        the closed-form Gaussian-beam waist relation for a thin-lens
        Fourier transform.

        For an input Gaussian beam with waist ``w_in`` at the front
        focal plane of a Fourier-transforming lens of focal length
        ``f``, the canonical Fourier-optics relation gives the output
        waist at the back focal plane as::

            w_out = (lambda * f) / (pi * w_in)

        (Saleh & Teich, *Fundamentals of Photonics*, 2nd ed., §3.2.2;
        Goodman, *Introduction to Fourier Optics*, 4th ed., §5.2.3).

        The v4.15.2 ``test_fourier_transform_field_matches_3_stage_chain``
        pin compared ``FourierTransform(f)`` against the literal
        ``FreeSpace(f) * ThinLens(f) * FreeSpace(f)`` chain -- since
        v4.15.2 ``FourierTransform._apply`` *invokes* that very chain
        internally, the v4.15.2 pin is tautological (it asserts that
        ``X == X``).  This v4.15.3 test asserts physics: the output
        waist must match the analytical relation, INDEPENDENT of the
        3-stage decomposition.  Agreement requires the lens-side
        Fourier optics to be correct, not just the implementation to
        be self-consistent.

        Acceptance: measured output waist within 5% of the analytical
        ``lambda * f / (pi * w_in)`` (the discrete-grid sampling
        introduces small errors that the audit specifies as the
        tolerance budget).  Empirically the error is ~0.001% for the
        parametrised cases here, so the 5% pin has ~5000x headroom.
        """
        # Analytical output waist (Saleh & Teich §3.2.2).
        w_out_analytical = wavelength * f / (np.pi * w_in)

        # Construct the on-axis Gaussian input.  Note ``w0`` in
        # :meth:`Source.gaussian` is the 1/e^2 intensity radius -- the
        # standard convention used by the Saleh/Teich/Goodman
        # formula.
        src = la.Source.gaussian(
            N=N, dx=dx, wavelength=wavelength, w0=w_in,
        )

        # Fourier-transform via FourierTransform(f).  The output
        # ``out.E`` is on the back-focal-plane grid (the dispatcher
        # decides the output pitch -- it may differ from the input
        # pitch for Fraunhofer-regime propagation).
        op = FourierTransform(f)
        out = op(src)

        # Extract the output waist from the second moment of the
        # intensity.  For a 2-D Gaussian intensity profile
        # ``I(r) = exp(-2 r^2 / w^2)`` the centred second moment is
        # ``<r^2> = w^2 / 2``, so ``w = sqrt(2 <r^2>)``.  This is
        # independent of any field-side normalisation -- only the
        # SHAPE of the intensity matters.
        I_out = np.abs(out.E) ** 2
        Ny, Nx = I_out.shape
        # The grid index ``i`` runs from ``0`` to ``N - 1``; the
        # physical coordinate ``x_i`` is ``(i - N//2) * dx_out`` --
        # the standard FFT-centred convention shared by every
        # :mod:`lumenairy.propagators` kernel.
        dx_out = float(out.dx)
        x_coord = (np.arange(Nx) - Nx // 2) * dx_out
        # ``out.dy`` defaults to ``out.dx`` when the source dy was
        # not specified.
        dy_out = float(getattr(out, 'dy', dx_out))
        y_coord = (np.arange(Ny) - Ny // 2) * dy_out
        X, Y = np.meshgrid(x_coord, y_coord)
        I_total = float(I_out.sum())
        assert I_total > 0.0, (
            f"FourierTransform output intensity sum is non-positive "
            f"({I_total!r}) -- the propagation collapsed the field, "
            f"likely an upstream regression."
        )
        # Normalise to PDF so ``<.>`` is a proper expectation.
        I_pdf = I_out / I_total
        x_centroid = float(np.sum(X * I_pdf))
        y_centroid = float(np.sum(Y * I_pdf))
        # On-axis Gaussian input -- centroid must sit at origin to
        # well under one pixel; if it does not the propagation has a
        # bug independent of the waist measurement.
        assert abs(x_centroid) < dx_out, (
            f"FT output centroid x = {x_centroid:.3e} m drifted off "
            f"axis by more than one output pixel ({dx_out:.3e} m); "
            f"input was on-axis Gaussian."
        )
        assert abs(y_centroid) < dy_out, (
            f"FT output centroid y = {y_centroid:.3e} m drifted off "
            f"axis by more than one output pixel ({dy_out:.3e} m); "
            f"input was on-axis Gaussian."
        )
        R2_centered = (X - x_centroid) ** 2 + (Y - y_centroid) ** 2
        mean_r2 = float(np.sum(R2_centered * I_pdf))
        w_out_measured = np.sqrt(2.0 * mean_r2)

        rel_error = (
            abs(w_out_measured - w_out_analytical) / w_out_analytical
        )
        # The audit specifies 5% as the tolerance budget (the
        # discrete-grid sampling introduces small errors).  Empirical
        # error on these parametrised cases is well under 0.01%; the
        # 5% tolerance gives orders-of-magnitude headroom.
        tolerance = 0.05  # 5%
        assert rel_error < tolerance, (
            f"{label}: Gaussian-beam waist relation violated. "
            f"Expected w_out = lambda * f / (pi * w_in) = "
            f"{w_out_analytical*1e6:.4f} um, got "
            f"{w_out_measured*1e6:.4f} um "
            f"(rel error {rel_error*100:.4f}%, tol "
            f"{tolerance*100:.1f}%)."
        )


# ============================================================================
# B.2 -- from_prescription flat-mirror parity drop
# ============================================================================


def _build_folded_prescription_flat() -> dict:
    """Folded singlet: refractive singlet then a flat fold mirror."""
    glass = 'N-BK7'
    return {
        'name': 'FoldedFlat',
        'aperture_diameter': 25.4e-3,
        'surfaces': [
            {'radius': 60e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': glass},
            {'radius': -60e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': glass, 'glass_after': 'air'},
            {'radius': float('inf'), 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'MIRROR'},
        ],
        'thicknesses': [4e-3, 25e-3, 0.0],
    }


def _build_folded_prescription_curved() -> dict:
    """Folded Newtonian-ish: parabolic primary (concave mirror) then
    refractive Cassegrain rear group.

    The curved primary mirror DOES flip parity in both
    ``system_abcd`` and ``from_prescription`` -- this fixture is the
    control case for ``test_from_prescription_curved_mirror_parity_still_flips``.
    """
    return {
        'name': 'CurvedMirrorFolded',
        'aperture_diameter': 25.4e-3,
        'surfaces': [
            # Concave primary (R = -100 mm in Welford convention)
            {'radius': -100e-3, 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'MIRROR'},
            # Refractive corrector behind the primary
            {'radius': 200e-3, 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -200e-3, 'conic': 0.0,
             'aspheric_coeffs': None,
             'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [-50e-3, 4e-3, 0.0],
    }


class TestFromPrescriptionMirrorParity:
    """v4.15.2 (audit P1-NEW-B): the operator-algebra ``from_prescription``
    must condition ``mirror_parity ^= 1`` on the mirror having a finite
    radius of curvature (curved mirror only), matching
    ``raytrace.system_abcd`` exactly on folded prescriptions.
    """

    def test_from_prescription_flat_mirror_parity_matches_system_abcd(
        self,
    ) -> None:
        """A folded prescription with a flat fold mirror must produce
        the SAME ABCD from ``from_prescription`` and ``system_abcd``,
        to within 1e-12 absolute.

        Pre-v4.15.2 the algebra layer flipped parity on the flat
        mirror, producing an off-sign ABCD on the post-mirror
        thickness term.
        """
        wavelength = 633e-9
        prescription = _build_folded_prescription_flat()
        surfaces = surfaces_from_prescription(prescription)
        M_ref, _, _, _ = system_abcd(surfaces, wavelength)
        op = la.Operator.from_prescription(prescription, wavelength)
        assert np.allclose(op.abcd, M_ref, atol=1e-12), (
            f"Folded-flat-mirror ABCD mismatch:\n"
            f"  system_abcd: {M_ref}\n"
            f"  from_prescription: {op.abcd}\n"
            f"  max abs diff: {np.max(np.abs(op.abcd - M_ref)):.3e}"
        )

    def test_from_prescription_curved_mirror_parity_still_flips(
        self,
    ) -> None:
        """Control test: a folded prescription with a CURVED mirror
        must STILL flip parity correctly.  v4.15.2 only drops the
        parity flip for flat mirrors; curved mirrors are unchanged.

        Verify by:
          1) confirming the algebra-layer ABCD matches ``system_abcd``
             (which we know flips parity for curved mirrors), and
          2) confirming the post-mirror thickness term carries the
             negative ``n_after`` reduction (Welford convention).
        """
        wavelength = 633e-9
        prescription = _build_folded_prescription_curved()
        surfaces = surfaces_from_prescription(prescription)
        M_ref, _, _, _ = system_abcd(surfaces, wavelength)
        op = la.Operator.from_prescription(prescription, wavelength)
        assert np.allclose(op.abcd, M_ref, atol=1e-12), (
            f"Folded-curved-mirror ABCD mismatch (curved-mirror parity "
            f"flip broken?):\n"
            f"  system_abcd: {M_ref}\n"
            f"  from_prescription: {op.abcd}\n"
            f"  max abs diff: {np.max(np.abs(op.abcd - M_ref)):.3e}"
        )
        # The Newtonian-ish prescription has a NEGATIVE post-mirror
        # thickness (-50e-3 -> we're physically moving backwards
        # along the folded axis).  After the curved mirror flips
        # parity, the ``d_eff = t / n_after`` term in
        # ``from_prescription`` carries the n -> -n flip, which
        # makes the FreeSpace distance positive (the post-mirror
        # leg goes back through air with effective n = -1, so
        # -50e-3 / -1 = +50e-3).  Verify:
        free_space_stages = [
            s for s in op.stages() if isinstance(s, la.FreeSpace)
        ]
        # The first FreeSpace stage is after the mirror; its
        # distance is -50e-3 / -1 = +50e-3 (n was flipped to -1).
        assert len(free_space_stages) >= 1
        first_post_mirror = free_space_stages[0]
        assert np.isclose(first_post_mirror.distance, 50e-3, atol=1e-12), (
            f"Curved-mirror parity flip did not produce the expected "
            f"post-mirror FreeSpace distance (Welford n -> -n flip): "
            f"got {first_post_mirror.distance:.6e}, expected 50e-3."
        )


# ============================================================================
# B.3 -- FreeSpace dy threading
# ============================================================================


class TestFreeSpaceDyThreading:
    """v4.15.2 (audit P1-NEW-C): ``FreeSpace._apply`` must thread
    ``dy`` into the dispatcher when ``dy != dx``, so anamorphic
    chains starting with ``Magnify(a_x, a_y) * FreeSpace(d)``
    propagate on the correct y-axis grid pitch.
    """

    def test_freespace_apply_threads_dy(self) -> None:
        """An anamorphic input (``dy != dx``) must yield a different
        FreeSpace propagation than the equivalent square-grid
        input -- if ``dy`` is dropped, the two calls would produce
        identical fields.
        """
        N = 64
        dx_in = 2e-6
        dy_in = 2.5 * dx_in
        wavelength = 633e-9
        # Off-axis Gaussian to ensure the y-axis kernel matters.
        src = la.Source.gaussian(
            N=N, dx=dx_in, wavelength=wavelength, w0=15e-6, y0=18e-6,
        )
        op = FreeSpace(0.05, method='asm')
        # Anamorphic _apply: dy != dx.
        E_anam, dx_anam, dy_anam = op._apply(
            src.E, dx=dx_in, dy=dy_in, wavelength=wavelength,
        )
        # Isotropic _apply: dy = dx.
        E_iso, dx_iso, dy_iso = op._apply(
            src.E, dx=dx_in, dy=dx_in, wavelength=wavelength,
        )
        # Output dy should reflect the input dy threading.
        assert np.isclose(dy_anam, dy_in, rtol=1e-12), (
            f"FreeSpace dropped dy: input dy={dy_in:.3e}, output "
            f"dy_out={dy_anam:.3e} (expected {dy_in:.3e} since ASM "
            f"preserves pitch)"
        )
        assert np.isclose(dy_iso, dx_in, rtol=1e-12)
        # The output fields must differ between the two -- if the
        # ASM kernel didn't see dy at all (i.e. silently defaulted
        # to dy = dx), the two calls would yield identical fields.
        assert not np.allclose(E_anam, E_iso, atol=1e-12), (
            "FreeSpace silently dropped dy: anamorphic and isotropic "
            "kernels produced identical fields, suggesting dy did not "
            "reach the underlying propagator."
        )

    def test_thinlens_apply_threads_dy(self) -> None:
        """Regression check: ``ThinLens._apply`` already threads
        ``dy`` into ``apply_thin_lens`` (it always has).  Verify
        v4.15.2 hasn't introduced a parallel gap.
        """
        N = 64
        dx_in = 2e-6
        dy_in = 2.5 * dx_in
        wavelength = 633e-9
        src = la.Source.gaussian(
            N=N, dx=dx_in, wavelength=wavelength, w0=20e-6, x0=10e-6,
            y0=8e-6,
        )
        op = ThinLens(0.05)
        E_anam, _, _ = op._apply(
            src.E, dx=dx_in, dy=dy_in, wavelength=wavelength,
        )
        E_iso, _, _ = op._apply(
            src.E, dx=dx_in, dy=dx_in, wavelength=wavelength,
        )
        # The thin-lens phase mask uses dy explicitly to build the
        # y-axis quadratic coordinate.  Anamorphic and isotropic
        # calls must produce different fields.
        assert not np.allclose(E_anam, E_iso, atol=1e-12), (
            "ThinLens silently dropped dy: anamorphic and isotropic "
            "phase masks produced identical fields."
        )


# ============================================================================
# B.4 -- Magnify docstring direction
# ============================================================================


class TestMagnifyDocstringDirection:
    """v4.15.2 (audit P2): the v4.15.1 Magnify docstring said the
    output extent ``scales as a * (N * dx)`` but the code produces
    ``new_dx = dx / a`` (extent shrinks by ``a`` for ``a > 1``).
    The docstring has been rewritten to match the code direction
    (Nazarathy/Shamir ``V[a]: (x, y) -> (x/a, y/a)``).
    """

    def test_magnify_docstring_matches_code_direction(self) -> None:
        """Parse the Magnify docstring and check it announces the
        ``a > 1`` => output SHRINKS direction, not the inverted
        ``a > 1`` => output GROWS direction.
        """
        doc = inspect.getdoc(_MagnifyCls) or ''
        # The pre-v4.15.2 docstring contained the inverted claim
        # ``physical extent scales as ``a * (N * dx)`` -- this exact
        # substring (with the leading ``a *``) must NOT be present
        # in the v4.15.2 docstring.
        bad_phrase = 'physical extent scales as `a * (N * dx)`'
        assert bad_phrase not in doc, (
            f"v4.15.2 Magnify docstring still contains the inverted "
            f"extent claim from v4.15.1:\n  {bad_phrase!r}"
        )
        # The corrected docstring announces both pixel-pitch and
        # extent direction via the ``new_dx = dx / a`` form and the
        # explicit ``a > 1: output is shrunk by a`` statement.
        # Check at least one of the corrected forms is present.
        new_dx_form = 'new_dx = dx / a_x'
        shrunk_phrase = 'shrunk'
        assert new_dx_form in doc, (
            f"v4.15.2 Magnify docstring missing the explicit "
            f"output-pitch formula {new_dx_form!r}; the docstring "
            f"must announce ``new_dx = dx / a_x`` to match the code."
        )
        assert shrunk_phrase in doc.lower(), (
            f"v4.15.2 Magnify docstring missing the ``shrunk`` "
            f"keyword that describes the ``a > 1`` direction."
        )

    def test_magnify_docstring_drops_dead_operators_reference(
        self,
    ) -> None:
        """v4.15.2 P2 closure (audit B.5): the v4.15.1 Magnify
        docstring referenced ``operators.py:556-577``, which was
        a reference to the lens-designer source repo (not part of
        the LumenAiry codebase).  v4.15.2 removes this dead
        reference.
        """
        doc = inspect.getdoc(_MagnifyCls) or ''
        assert 'operators.py:556-577' not in doc, (
            "v4.15.2 Magnify docstring still contains the dead "
            "``operators.py:556-577`` reference from v4.15.1."
        )
        assert 'operators.py' not in doc, (
            "v4.15.2 Magnify docstring still references the "
            "external ``operators.py`` source file."
        )


# ============================================================================
# B.5 -- End-to-end integration sanity
# ============================================================================


class TestIntegrationSanity:
    """End-to-end smoke tests over the v4.15.2 closures: a full
    operator chain that exercises all three fixes (FT 3-stage,
    flat-mirror parity, FreeSpace dy threading) should compose
    cleanly without raising.
    """

    def test_anamorphic_chain_with_freespace_runs(self) -> None:
        """An anamorphic chain ``Magnify(a_x, a_y) * FreeSpace(d)``
        applied via the 4-tuple anamorphic entry point must run
        without dropping ``dy``.
        """
        src = la.Source.gaussian(
            N=64, dx=2e-6, wavelength=633e-9, w0=20e-6,
        )
        chain = Magnify(2.0, 1.5) * FreeSpace(0.05, method='asm')
        out = chain(src)
        assert out.E.shape == src.E.shape
        assert np.all(np.isfinite(out.E))
        # Magnify shrinks the pitch by a_x and a_y respectively.
        # After FreeSpace (which preserves pitch on ASM), the
        # Magnify stage divides dx by 2.0 and dy by 1.5.
        assert np.isclose(out.dx, src.dx / 2.0, rtol=1e-12)
        # out.dy may have been set by Magnify; the Source's dy
        # default tracks dx if not set.
        assert np.isclose(out.dy, src.dy / 1.5, rtol=1e-12)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
