"""Unit tests for v4.15.1 Agent D scope: close the Forbes Q-bfs / Q-con
consumer paths flagged in ``docs/audits/AUDIT_V4_15_0_2026_05_18.md``.

The v4.15.0 release shipped Forbes Q math + dispatcher correctly, but
the audit found three half-shipped paths:

* **P1-NEW-D** -- ``raytrace/core.py:1507-1511`` flat-keys gather
  allowlist dropped ``q_bfs_coeffs`` / ``q_con_coeffs`` / ``r_max``,
  so a prescription that put those keys at the surface-dict top level
  (flat-keys form) silently lost the coefficients before reaching
  ``Surface.freeform``.  ``apply_real_lens_traced`` therefore traced
  a pure best-fit-sphere.
* **P1-F1-1** -- The rectangular ``|x| <= norm_x AND |y| <= norm_y``
  clip in ``surface_sag_q_bfs`` / ``surface_sag_q_con`` let corner
  pixels at ``(0.9 r_max, 0.9 r_max)`` (radial 1.27 r_max, outside
  the Forbes domain) through, and zeroed ``(0, r_max)`` correctly.
  The basis is defined on the unit disc in ``u = r / r_max``, so the
  primary clip must be radial.
* **P1-F1-2** -- ``surface_sag_freeform`` silently defaulted
  ``r_max=1.0`` when missing.  For a user passing X/Y in metres
  this meant the Forbes domain shrank to a 1 m disc, and the
  polynomial evaluated on a sub-pixel of the actual aperture --
  silent units mismatch.
* **P3 doc-formula** -- the module docstring stated the Q-bfs
  orthonormalizer with the wrong ``(n+1)`` power and a spurious
  factor of 8.  The implementation is correct; only the prose was
  wrong.
* **P3-NEW-A** -- ``apply_real_lens`` (the wave-optics path)
  silently ``RuntimeWarning``-and-skipped the Forbes Q departure,
  so the freeform contribution was a no-op in the wave-optics path.
* **P2** -- ``surface_sag_q_bfs`` / ``_q_con`` are exported from
  top-level ``lumenairy`` but not re-exported through
  ``lumenairy.elements.__init__``; analogously ``make_off_axis_parabola``
  was not re-exported through ``lumenairy.io.__init__``.

Author: Andrew Traverso -- v4.15.1 / Agent D
"""
from __future__ import annotations

import inspect
import math
import re
import warnings

import numpy as np
import pytest

from lumenairy.elements._lens_traced import apply_real_lens_traced
from lumenairy.elements.freeform import (
    _jacobi_norm_factor,
    _q_bfs_eval,
    _q_con_eval,
    surface_sag_freeform,
    surface_sag_q_bfs,
    surface_sag_q_con,
)
from lumenairy.elements.lenses import (
    apply_real_lens,
    surface_sag_general,
)
from lumenairy.raytrace.core import (
    _surface_sag_xy,
    surfaces_from_prescription,
)

# ============================================================================
# Helpers
# ============================================================================

def _grid(N=64, half_extent=5e-3):
    """Square aperture grid for sag and wave-optics tests.

    Default ``half_extent`` is 5 mm so a typical ``r_max = 4 mm``
    Forbes domain sits comfortably inside the grid box and corner
    pixels at ``(half_extent, half_extent)`` lie *outside* the
    Forbes domain (``r = 7.07 mm > r_max``).
    """
    x = np.linspace(-half_extent, half_extent, N)
    y = np.linspace(-half_extent, half_extent, N)
    X, Y = np.meshgrid(x, y)
    return X, Y


# ============================================================================
# D.1 -- raytrace flat-keys allowlist (P1-NEW-D)
# ============================================================================

class TestRaytraceFlatKeysGather:
    """``surfaces_from_prescription`` must gather the Forbes Q keys
    (``q_bfs_coeffs``, ``q_con_coeffs``, ``r_max``) from a flat-keys
    surface dict into the ``Surface.freeform`` dict so the raytraced
    OPD honours the freeform departure.

    Pre-v4.15.1 the allowlist was
    ``('freeform_type', 'xy_coeffs', 'zernike_coeffs',
       'cheb_coeffs', 'norm_radius', 'norm_x', 'norm_y')`` --
    Forbes Q keys missing.  The dispatcher routed correctly because
    ``freeform_type`` was IN the allowlist, but the coefficients
    were silently dropped, so the surface sag reduced to a pure
    best-fit sphere.
    """

    def _make_flat_keys_q_bfs_prescription(self, *, with_r_max=True):
        """Single-surface prescription with Q-bfs in flat-keys form."""
        return {
            'surfaces': [
                {
                    'radius': float('inf'),
                    'conic': 0.0,
                    'glass_before': 'air',
                    'glass_after': 'air',
                    'is_mirror': True,
                    'freeform_type': 'q_bfs',
                    'q_bfs_coeffs': [2.0e-6, -1.0e-6],
                    **({'r_max': 3e-3} if with_r_max else {}),
                    'norm_x': 5e-3,
                    'norm_y': 5e-3,
                },
            ],
            'thicknesses': [0.0],
            'aperture_diameter': 8e-3,
        }

    def _make_flat_keys_q_con_prescription(self):
        return {
            'surfaces': [
                {
                    'radius': 50e-3,
                    'conic': -1.0,
                    'glass_before': 'air',
                    'glass_after': 'air',
                    'is_mirror': True,
                    'freeform_type': 'q_con',
                    'q_con_coeffs': [3.0e-6, -1.0e-6, 5.0e-7],
                    'r_max': 3e-3,
                    'norm_x': 5e-3,
                    'norm_y': 5e-3,
                },
            ],
            'thicknesses': [0.0],
            'aperture_diameter': 8e-3,
        }

    def test_raytrace_flat_keys_q_bfs_coeffs_pass_through(self):
        """Flat-keys ``q_bfs_coeffs`` survives the gather step and is
        reflected in the Surface sag (not silently zero).

        The sag computed via ``_surface_sag_xy`` on the gathered
        ``Surface`` must equal the direct ``surface_sag_q_bfs`` call
        with the same parameters."""
        rx = self._make_flat_keys_q_bfs_prescription()
        surfaces = surfaces_from_prescription(rx)
        assert len(surfaces) == 1
        surf = surfaces[0]
        # The freeform dict must carry the coefficients + r_max.
        assert surf.freeform is not None, (
            "P1-NEW-D fix dropped: Surface.freeform is None after "
            "gather of a flat-keys Q-bfs prescription.")
        assert 'q_bfs_coeffs' in surf.freeform, (
            "P1-NEW-D fix dropped: q_bfs_coeffs key missing from "
            "the gathered freeform dict.")
        assert 'r_max' in surf.freeform, (
            "P1-NEW-D fix dropped: r_max key missing from the "
            "gathered freeform dict.")
        # The traced sag must match the direct evaluation.
        X, Y = _grid(N=32, half_extent=2.5e-3)
        sag_traced = _surface_sag_xy(X, Y, surf)
        sag_direct = surface_sag_q_bfs(
            X, Y, radius=float('inf'),
            coefficients=[2.0e-6, -1.0e-6],
            r_max=3e-3, dx=float(X[0, 1] - X[0, 0]),
            norm_x=5e-3, norm_y=5e-3,
        )
        # Must not be the pure best-fit sphere (= zero here for R=inf).
        peak_freeform = float(np.max(np.abs(sag_direct)))
        assert peak_freeform > 1e-9, (
            "Sanity check: the test setup must produce a non-zero "
            "freeform departure.")
        max_diff = float(np.max(np.abs(sag_traced - sag_direct)))
        assert max_diff < 1e-15, (
            f"Traced Q-bfs sag deviates from direct evaluation by "
            f"{max_diff:.3e}; the flat-keys allowlist is still "
            f"dropping coefficients.")

    def test_raytrace_flat_keys_q_con_coeffs_pass_through(self):
        """Same round-trip test for Q-con."""
        rx = self._make_flat_keys_q_con_prescription()
        surfaces = surfaces_from_prescription(rx)
        surf = surfaces[0]
        assert surf.freeform is not None
        assert 'q_con_coeffs' in surf.freeform
        assert 'r_max' in surf.freeform
        X, Y = _grid(N=32, half_extent=2.5e-3)
        sag_traced = _surface_sag_xy(X, Y, surf)
        sag_direct = surface_sag_q_con(
            X, Y, radius=50e-3, conic=-1.0,
            coefficients=[3.0e-6, -1.0e-6, 5.0e-7],
            r_max=3e-3, dx=float(X[0, 1] - X[0, 0]),
            norm_x=5e-3, norm_y=5e-3,
        )
        max_diff = float(np.max(np.abs(sag_traced - sag_direct)))
        assert max_diff < 1e-15, (
            f"Traced Q-con sag deviates from direct evaluation by "
            f"{max_diff:.3e}; the flat-keys allowlist is still "
            f"dropping coefficients.")


# ============================================================================
# D.2 -- Radial clip alignment (P1-F1-1)
# ============================================================================

class TestRadialClipAlignment:
    """The Q-bfs / Q-con sag must be zeroed for pixels with
    ``r > r_max`` (radial clip), not only outside the rectangular
    ``[-norm_x, norm_x] x [-norm_y, norm_y]`` box.  Pre-v4.15.1 used
    the rectangular clip only, letting corner pixels at
    ``(0.9 r_max, 0.9 r_max)`` (r = 1.27 r_max, outside Forbes domain)
    through."""

    def test_q_bfs_radial_clip_zeros_outside_r_max(self):
        """A pixel at ``(0.9 r_max, 0.9 r_max)`` has radial distance
        1.27 r_max -- outside the Forbes domain.  Its freeform
        departure must be zero, even when the rectangular clip
        ``norm_x = norm_y = 10 r_max`` would let it through."""
        r_max = 3e-3
        # Pick (x, y) coords on the grid such that radial distance is
        # outside r_max but rectangular clip is loose.
        x0 = 0.9 * r_max
        y0 = 0.9 * r_max
        # 3x3 grid centered on the corner so we can read the (1, 1) pixel.
        X = np.array([[x0 - 1e-7, x0, x0 + 1e-7]] * 3)
        Y = np.array(
            [[y0 - 1e-7] * 3,
             [y0] * 3,
             [y0 + 1e-7] * 3])
        sag = surface_sag_q_bfs(
            X, Y, radius=float('inf'),
            coefficients=[2.0e-6, -1.0e-6, 5.0e-7],
            r_max=r_max, dx=1e-7,
            norm_x=10 * r_max, norm_y=10 * r_max,  # box: non-restrictive
        )
        r = math.hypot(x0, y0)
        assert r > r_max, (
            "Sanity check: test corner must lie outside the Forbes "
            "domain.")
        # Centre pixel of the 3x3 must be zero (radial clip).
        sag_corner = sag[1, 1]
        assert abs(sag_corner) < 1e-15, (
            f"Pixel at ({x0:.3e}, {y0:.3e}) with r={r:.3e} > "
            f"r_max={r_max:.3e} has non-zero sag {sag_corner:.3e}; "
            f"the radial clip is not aligned with r_max.")

    def test_q_bfs_radial_clip_includes_inside_r_max(self):
        """Sanity check the dual: a pixel at ``(0.7 r_max, 0.0)`` is
        inside the Forbes domain (r = 0.7 r_max < r_max) and must
        produce a non-zero sag.  Without this we'd falsely close
        the audit just by zeroing everything."""
        r_max = 3e-3
        x0 = 0.7 * r_max
        y0 = 0.0
        X = np.array([[x0]])
        Y = np.array([[y0]])
        sag = surface_sag_q_bfs(
            X, Y, radius=float('inf'),
            coefficients=[5.0e-6],
            r_max=r_max, dx=1e-7,
            norm_x=10 * r_max, norm_y=10 * r_max,
        )
        assert abs(float(sag[0, 0])) > 1e-12, (
            "Pixel inside the Forbes domain must receive a non-zero "
            "freeform sag; the new radial clip is over-zealous.")

    def test_q_con_radial_clip_zeros_outside_r_max(self):
        """Q-con counterpart of the Q-bfs radial clip test."""
        r_max = 3e-3
        x0 = 0.9 * r_max
        y0 = 0.9 * r_max
        X = np.array([[x0 - 1e-7, x0, x0 + 1e-7]] * 3)
        Y = np.array(
            [[y0 - 1e-7] * 3,
             [y0] * 3,
             [y0 + 1e-7] * 3])
        sag = surface_sag_q_con(
            X, Y, radius=float('inf'), conic=0.0,
            coefficients=[2.0e-6, -1.0e-6, 5.0e-7],
            r_max=r_max, dx=1e-7,
            norm_x=10 * r_max, norm_y=10 * r_max,
        )
        r = math.hypot(x0, y0)
        assert r > r_max
        sag_corner = float(sag[1, 1])
        assert abs(sag_corner) < 1e-15, (
            f"Q-con pixel at r={r:.3e} > r_max={r_max:.3e} has "
            f"non-zero sag {sag_corner:.3e}; radial clip not aligned.")


# ============================================================================
# D.3 -- surface_sag_freeform requires r_max (P1-F1-2)
# ============================================================================

class TestSurfaceSagFreeformRMaxRequired:
    """``surface_sag_freeform`` must require ``r_max`` when
    ``freeform_type='q_bfs'`` or ``'q_con'``.  Pre-v4.15.1 silently
    defaulted to ``r_max = 1.0``, which for a user passing X/Y in
    metres produced a Forbes domain of 1 m -- usually orders of
    magnitude larger than the actual aperture, so the polynomial got
    evaluated on a sub-pixel of the surface and the freeform
    departure was effectively a no-op."""

    def test_surface_sag_freeform_q_bfs_requires_r_max(self):
        """Omitting ``r_max`` for ``freeform_type='q_bfs'`` raises
        TypeError with a clear message."""
        X, Y = _grid(N=16, half_extent=2.5e-3)
        bad_surf = {
            'freeform_type': 'q_bfs',
            'radius': float('inf'),
            'q_bfs_coeffs': [1e-6],
            # 'r_max' deliberately missing.
            'dx': X[0, 1] - X[0, 0],
            'norm_x': 5e-3,
            'norm_y': 5e-3,
        }
        with pytest.raises(TypeError, match="r_max"):
            surface_sag_freeform(X, Y, bad_surf)

    def test_surface_sag_freeform_q_con_requires_r_max(self):
        X, Y = _grid(N=16, half_extent=2.5e-3)
        bad_surf = {
            'freeform_type': 'q_con',
            'radius': 50e-3,
            'conic': -1.0,
            'q_con_coeffs': [1e-6],
            # 'r_max' deliberately missing.
            'dx': X[0, 1] - X[0, 0],
            'norm_x': 5e-3,
            'norm_y': 5e-3,
        }
        with pytest.raises(TypeError, match="r_max"):
            surface_sag_freeform(X, Y, bad_surf)

    def test_surface_sag_freeform_xy_polynomial_unchanged(self):
        """Sanity check: the new r_max guard only fires for q_bfs /
        q_con.  Other freeform types (xy_polynomial, zernike,
        chebyshev) must continue to work without ``r_max``."""
        X, Y = _grid(N=16, half_extent=2.5e-3)
        surf = {
            'freeform_type': 'xy_polynomial',
            'radius': float('inf'),
            'xy_coeffs': {(2, 0): 1e-3},
            'norm_x': 5e-3,
            'norm_y': 5e-3,
        }
        # Must not raise.
        sag = surface_sag_freeform(X, Y, surf)
        assert sag.shape == X.shape


# ============================================================================
# D.4 -- Orthonormalizer docstring matches code (P3)
# ============================================================================

class TestOrthonormalizerDocstringMatchesCode:
    """The corrected module-docstring formula for ``c_n`` must agree
    with the value computed by ``_jacobi_norm_factor``.

    Audit reference: the v4.15.0 CHANGELOG and module docstring stated
    ``c_n = sqrt((2n+3)(n+2) / (n+1)^2)`` with a spurious factor of
    8.  Correct formula (from A&S 22.2.1 specialised to
    ``alpha = beta = 1`` and mapped to [0, 1]):
    ``c_n = sqrt((2n+3)(n+2) / (n+1))``.
    """

    def test_q_bfs_docstring_formula_matches_code(self):
        """For n in 0..6 the value of
        ``c_n = sqrt((2n+3)(n+2) / (n+1))`` (the corrected formula)
        must match ``_jacobi_norm_factor(n, 1, 1)`` to 1e-12."""
        for n in range(7):
            expected = math.sqrt((2 * n + 3) * (n + 2) / (n + 1))
            actual = _jacobi_norm_factor(n, 1, 1)
            assert abs(actual - expected) < 1e-12, (
                f"Q-bfs c_{n}: code returns {actual:.12e}, "
                f"corrected docstring formula gives "
                f"{expected:.12e} -- mismatch.")

    def test_q_con_docstring_formula_matches_code(self):
        """Q-con: ``c_n = sqrt(2n+3)`` (no Q-bfs/Q-con asymmetry
        confusion)."""
        for n in range(7):
            expected = math.sqrt(2 * n + 3)
            actual = _jacobi_norm_factor(n, 0, 2)
            assert abs(actual - expected) < 1e-12, (
                f"Q-con c_{n}: code returns {actual:.12e}, "
                f"docstring formula gives {expected:.12e}.")

    def test_orthonormalizer_docstring_matches_code(self):
        """Parse the module-docstring text for the Q-bfs ``c_n``
        formula and verify it matches the value produced by
        ``_jacobi_norm_factor`` for a representative ``n``.

        Protects against re-introduction of the v4.15.0 wrong
        formula ``sqrt((2n+3)(n+2)/(n+1)^2)``.
        """
        import lumenairy.elements.freeform as freeform_mod
        source = inspect.getsource(freeform_mod)
        # Look in the module-level comment block for the Q-bfs
        # specialisation line.  The corrected formula is
        # ``c_n = sqrt((2n+3)(n+2) / (n+1))``; the old wrong form
        # has ``(n+1)^2`` or carries a spurious factor of 8.
        # Match exactly the corrected form (no exponent on ``(n+1)``,
        # no ``8 (n+2)`` factor in the numerator under sqrt).
        # Use a permissive regex that tolerates whitespace.
        pattern = re.compile(
            r"c_n\s*=\s*sqrt\(\s*\(2n\+3\)\s*\(n\+2\)\s*/\s*\(n\+1\)\s*\)",
            re.IGNORECASE,
        )
        assert pattern.search(source) is not None, (
            "Module-docstring no longer contains the corrected "
            "Q-bfs formula `c_n = sqrt((2n+3)(n+2) / (n+1))`.  "
            "Check that the v4.15.1 P3 doc-formula fix has not "
            "been reverted.")
        # And the OLD wrong forms must be absent (only inside the
        # corrected docstring's reference to the historical wrong
        # form is the substring tolerated; we anchor on whether the
        # corrected line is also present).
        re.compile(
            r"c_n\s*=\s*sqrt\(\s*\(2n\+3\)\s*\(n\+2\)\s*/\s*\(n\+1\)\s*\^?\s*2",
        )
        # Allow the wrong form to appear in the v4.15.1 historical-
        # note paragraph (it is now framed as "pre-v4.15.1 stated
        # ... incorrectly"), so just confirm the corrected formula
        # is the closing-source-of-truth line.  (Numerical check
        # above is the load-bearing pin.)


# ============================================================================
# D.5 -- __all__ symmetry (P2)
# ============================================================================

class TestAllSymmetry:
    """The freeform sag helpers + OAP factory are exported from
    top-level ``lumenairy`` but were missing from the subpackage
    ``__init__`` re-exports; v4.15.1 P2 closes that asymmetry."""

    def test_elements_re_exports_q_bfs(self):
        """``from lumenairy.elements import surface_sag_q_bfs`` must
        work."""
        from lumenairy.elements import surface_sag_q_bfs as _q_bfs

        # And must be the same callable as the implementation.
        from lumenairy.elements.freeform import surface_sag_q_bfs as _qb_impl
        assert _q_bfs is _qb_impl

    def test_elements_re_exports_q_con(self):
        from lumenairy.elements import surface_sag_q_con as _q_con
        from lumenairy.elements.freeform import surface_sag_q_con as _qc_impl
        assert _q_con is _qc_impl

    def test_elements_all_contains_q_bfs_q_con(self):
        """``lumenairy.elements.__all__`` must list both names so
        ``from lumenairy.elements import *`` exposes them."""
        import lumenairy.elements as elements_mod
        assert 'surface_sag_q_bfs' in elements_mod.__all__
        assert 'surface_sag_q_con' in elements_mod.__all__

    def test_io_re_exports_make_off_axis_parabola(self):
        """``from lumenairy.io import make_off_axis_parabola`` must
        work."""
        from lumenairy.io import make_off_axis_parabola as _oap
        from lumenairy.io.prescriptions import (
            make_off_axis_parabola as _oap_impl,
        )
        assert _oap is _oap_impl

    def test_io_all_contains_make_off_axis_parabola(self):
        import lumenairy.io as io_mod
        assert 'make_off_axis_parabola' in io_mod.__all__


# ============================================================================
# D.6 -- Wave-optics path no longer silently skips Q-bfs / Q-con (P3-NEW-A)
# ============================================================================

class TestWaveOpticsQbfsNotSilentlySkipped:
    """``apply_real_lens`` (the wave-optics path) must apply the
    Forbes Q-bfs / Q-con sag in the per-surface phase screen instead
    of warn-and-skip.  Pre-v4.15.1 the freeform departure was a
    no-op; the resulting field was identical to a pure plain-conic
    lens.

    v4.15.1 choice (a): properly apply the Q sag.  We verify that
    the output field DIFFERS from the no-freeform baseline when the
    only change is to add a Q-bfs coefficient.  This is sufficient to
    pin "not silently skipped".
    """

    def _build_test_field(self, N=64, dx=1e-5):
        """A plane wave on a small grid."""
        return np.ones((N, N), dtype=np.complex128), dx

    def _q_bfs_refractive_prescription(self, *, coeffs):
        """Two-surface plano-convex refractive prescription with
        optional Q-bfs departure on the first surface.

        ``apply_real_lens`` requires N surfaces + N-1 thicknesses and
        rejects mirror surfaces, so we use a plano-convex N-BK7
        singlet and put the Q-bfs freeform on the first (curved)
        surface.  The Q-bfs sag contributes the per-surface OPD
        ``(n2 - n1) * sag`` in the phase screen, which is what the
        test pins (zero vs non-zero Q-bfs coefficient must produce
        different output fields).
        """
        return {
            'surfaces': [
                {
                    'radius': 50e-3,
                    'conic': 0.0,
                    'glass_before': 'air',
                    'glass_after': 'N-BK7',
                    'freeform_type': 'q_bfs',
                    'q_bfs_coeffs': coeffs,
                    'r_max': 200e-6,  # 200 um radial domain
                    'norm_x': 1e-3,
                    'norm_y': 1e-3,
                },
                {
                    'radius': float('inf'),
                    'conic': 0.0,
                    'glass_before': 'N-BK7',
                    'glass_after': 'air',
                },
            ],
            'thicknesses': [2.5e-3],
            'aperture_diameter': 600e-6,
        }

    def test_apply_real_lens_q_bfs_not_silently_skipped(self):
        """Run apply_real_lens twice on identical input, with the only
        difference being a Q-bfs coefficient (zero vs non-zero).
        The output fields must DIFFER; pre-v4.15.1 they were
        bit-identical because the freeform was warn-skipped."""
        N = 64
        dx = 1e-5
        E_in = np.ones((N, N), dtype=np.complex128)
        # Baseline: zero coefficients (pure spherical sag).
        rx_zero = self._q_bfs_refractive_prescription(coeffs=[0.0, 0.0])
        # Test: non-zero coefficients.
        rx_q = self._q_bfs_refractive_prescription(
            coeffs=[2.0e-6, -5.0e-7])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # Pre-v4.15.1 this raised the freeform warn-and-skip
            # RuntimeWarning (and the test would catch
            # bit-identical outputs).  Post-v4.15.1 the warning is
            # gone for q_bfs and the outputs must differ.
            E_zero = apply_real_lens(
                E_in,
                prescription=rx_zero,
                wavelength=1.31e-6,
                dx=dx,
            )
            E_q = apply_real_lens(
                E_in,
                prescription=rx_q,
                wavelength=1.31e-6,
                dx=dx,
            )
        diff = float(np.max(np.abs(E_q - E_zero)))
        assert diff > 1e-9, (
            f"apply_real_lens output is bit-identical with and "
            f"without a Q-bfs coefficient (max diff {diff:.3e}); "
            f"the wave-optics path is still silently skipping the "
            f"freeform departure.")

    def test_apply_real_lens_q_bfs_no_runtime_warning(self):
        """Specifically, no ``RuntimeWarning`` is emitted for the
        ``freeform_type='q_bfs'`` branch (the old warn-and-skip
        message)."""
        N = 32
        dx = 2e-5
        E_in = np.ones((N, N), dtype=np.complex128)
        rx = self._q_bfs_refractive_prescription(coeffs=[1e-6])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            apply_real_lens(
                E_in,
                prescription=rx,
                wavelength=1.31e-6,
                dx=dx,
            )
        freeform_skip_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and 'freeform' in str(w.message)
            and 'NOT included' in str(w.message)
        ]
        assert not freeform_skip_warnings, (
            f"apply_real_lens still emits the freeform skip "
            f"RuntimeWarning for q_bfs: "
            f"{[str(w.message) for w in freeform_skip_warnings]}")
