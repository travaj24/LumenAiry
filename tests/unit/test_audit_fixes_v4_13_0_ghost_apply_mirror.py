"""Pinning tests for the v4.13.0 Track-B closure of two v4.12.2 known-
limitations carried forward from ``AUDIT_V4_12_1_2026_05_16.md``:

* **S3** ``analysis/ghost.py`` R_i/R_j convention conflict.
* **L6** ``elements.elements.apply_mirror`` array-namespace dispatch +
  missing ``dy`` parameter.

S3 -- ghost.py R convention
---------------------------

Pre-v4.13.0 the ``ghost_analysis`` body docstring used ``|R_i|`` and
``|R_j|`` to refer to *curvature radii* in the harmonic-mean
heuristic, while elsewhere in the same module ``R_i`` / ``R_j``
denoted the *Fresnel reflectance*.  Two different conventions for
the same letter in the same module.

v4.13.0 (this fix) adopts a strict upper/lowercase split:

* Uppercase ``R_i`` / ``R_j`` / ``R_k`` -- Fresnel reflectance
  (dimensionless, ratio in [0, 1]).
* Lowercase ``r_i`` / ``r_j`` -- curvature radius (metres, signed
  Welford convention).

The body code renames the local heuristic variables from
``R_i_val`` / ``R_j_val`` to ``r_i`` / ``r_j``; the docstrings (both
module and function) explicitly disambiguate the two
interpretations; a new top-of-module "Naming convention" block
documents the split.

L6 -- apply_mirror array-namespace dispatch + dy
------------------------------------------------

Pre-v4.13.0 ``apply_mirror`` used ``np.exp``, ``np.where``,
``np.meshgrid`` directly -- the lone holdout in ``elements.py``
that didn't go through ``array_namespace(E_in)`` dispatch.  CuPy
and JAX inputs silently fell through to the NumPy code path
(round-tripping arrays through the host).  Also still missing the
``dy: Optional[float] = None`` parameter that every other ``apply_*``
function in the module accepts (for anamorphic / rectangular grids).

v4.13.0 closes both gaps:

* ``apply_mirror(jax_E_in, ...)`` returns a JAX array (the namespace
  is ``jax.numpy``, not ``numpy``).
* ``apply_mirror(..., dy=dy)`` builds the meshgrid + aperture mask
  with the supplied y-pitch, producing the expected anamorphic
  geometry (more y-pixels inside the aperture when ``dy < dx``).

The wave-side ``R > 0 = concave`` convention is preserved (this is
the user-facing API on the wave-optics side; ``system_abcd`` /
``seidel_coefficients`` continue to use the OPPOSITE Welford
signed-R convention -- see
``validation/elements/test_elements.py::t_curved_mirror_focus``).

Author: Andrew Traverso
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import lumenairy as la
from lumenairy.analysis import ghost as ghost_mod
from lumenairy.elements.elements import apply_mirror


# ============================================================================
# S3 -- ghost.py R convention disambiguation
# ============================================================================


class TestGhostRConventionDisambiguated:
    """Pin the v4.13.0 ghost.py docstring + variable-name disambiguation.

    Pre-fix: ``|R_i|`` / ``|R_j|`` appeared in the body docstring of
    :func:`ghost_analysis` referring to curvature radii while everywhere
    else in the module the same letters meant Fresnel reflectance.

    Post-fix:

    * Top-of-module ``"Naming convention"`` block explicitly documents
      the upper/lower ``R``/``r`` split.
    * Body docstring of :func:`ghost_analysis` uses the prose "Fresnel
      reflectance at surface i" when referring to ``R_i`` and the
      prose "curvature radii ``|r_i|`` and ``|r_j|``" (lowercase) when
      referring to radii.
    * Body code locals renamed from ``R_i_val`` / ``R_j_val`` to
      ``r_i`` / ``r_j``.
    """

    def test_module_docstring_has_naming_convention_block(self):
        """Top-of-module convention block exists and names both
        interpretations explicitly."""
        doc = ghost_mod.__doc__ or ""
        assert "Naming convention" in doc, (
            "v4.13.0 ghost.py must declare a Naming-convention block at "
            "module level to disambiguate R (Fresnel reflectance) from r "
            "(curvature radius).  Block missing.")
        # Both interpretations explicitly named in the block.
        assert "Fresnel reflectance" in doc, (
            "Naming convention must explicitly name 'Fresnel reflectance' "
            "for the uppercase-R interpretation.")
        assert ("curvature" in doc.lower()), (
            "Naming convention must explicitly name 'curvature' radius "
            "for the lowercase-r interpretation.")

    def test_module_docstring_mentions_both_uppercase_R_and_lowercase_r(self):
        """The convention block uses both ``R_i`` AND ``r_i`` (or ``r_j``)
        to label the two distinct quantities."""
        doc = ghost_mod.__doc__ or ""
        assert "R_i" in doc, "Module docstring should still mention R_i (Fresnel)."
        # Either lowercase ``r_i``/``r_j`` literal or the prose
        # "lowercase r" / "lowercase ``r``" is acceptable.
        has_lower = ("r_i" in doc and "R_i" in doc) and (
            "r_i`" in doc or "r_j`" in doc or "lowercase" in doc.lower()
        )
        assert has_lower, (
            "Module docstring must use lowercase r_i / r_j (or call out "
            "'lowercase r') for curvature radii.  Found: "
            f"{doc[:400]!r}")

    def test_ghost_analysis_docstring_disambiguates_R_vs_r(self):
        """:func:`ghost_analysis` body docstring labels each ``R`` /
        ``r`` it mentions with explicit prose ('Fresnel reflectance' vs
        'curvature radii')."""
        doc = ghost_mod.ghost_analysis.__doc__ or ""
        # 'R_i' / 'R_j' appear in the Returns block (they're the
        # public dictionary keys; we MUST keep them).  Pin that the
        # docstring explicitly identifies them as Fresnel reflectance
        # near where they appear, AND mentions curvature radii with
        # lowercase r for the focus_z_estimate.
        assert "Fresnel reflectance" in doc, (
            "ghost_analysis docstring must explicitly identify R_i/R_j "
            "as 'Fresnel reflectance' to disambiguate from curvature "
            "radii.")
        assert ("curvature radii" in doc.lower()
                or "curvature radius" in doc.lower()), (
            "ghost_analysis docstring must explicitly call the focus_z_"
            "estimate inputs 'curvature radii' (or 'curvature radius') "
            "to disambiguate from Fresnel reflectance.")
        # And the lowercase r_i / r_j must appear (in the focus_z_estimate
        # description).
        assert ("r_i" in doc or "r_j" in doc), (
            "ghost_analysis docstring must use lowercase r_i / r_j for "
            "the curvature-radii heuristic.")

    def test_body_code_uses_lowercase_r_for_curvature(self):
        """The function body's local variables for curvature radii are
        named ``r_i`` / ``r_j`` (lowercase), not ``R_i_val`` /
        ``R_j_val`` (pre-fix) or ``R_i`` / ``R_j`` (the Fresnel form)."""
        src = inspect.getsource(ghost_mod.ghost_analysis)
        # Renamed locals present.
        assert "r_i = surfs[i].radius" in src, (
            "Body should use lowercase r_i = surfs[i].radius for the "
            "curvature-radius local (v4.13.0 rename).")
        assert "r_j = surfs[j].radius" in src, (
            "Body should use lowercase r_j = surfs[j].radius for the "
            "curvature-radius local (v4.13.0 rename).")
        # Pre-fix naming gone.
        assert "R_i_val" not in src, (
            "Pre-fix R_i_val local should be renamed to r_i in v4.13.0.")
        assert "R_j_val" not in src, (
            "Pre-fix R_j_val local should be renamed to r_j in v4.13.0.")

    def test_ghost_analysis_still_returns_R_i_and_R_j_keys(self):
        """The dictionary keys ``'R_i'`` / ``'R_j'`` remain the
        Fresnel-reflectance public API (no rename of these -- they're
        the keys downstream code reads)."""
        # Build a valid prescription via the library's own factory so
        # we don't depend on the prescription-validator schema (which
        # has changed across releases).  ``make_singlet`` is the
        # canonical 2-surface lens factory used in the validation
        # suite (validation/analysis/test_features.py:241).
        prescription = la.make_singlet(
            50.0e-3, np.inf, 4e-3, 'N-BK7', aperture=10e-3)

        ghosts = ghost_mod.ghost_analysis(
            prescription, wavelength=587.6e-9, verbose=False)
        assert len(ghosts) >= 1
        g0 = ghosts[0]
        # Public Fresnel-reflectance keys preserved (uppercase R).
        assert 'R_i' in g0
        assert 'R_j' in g0
        # Values are dimensionless reflectance in [0, 1] (Fresnel),
        # NOT metres (curvature radius).  A glass-air interface with
        # n ~ 1.52 gives R ~ 0.04, so anything < 0.5 distinguishes
        # Fresnel from any conceivable radius-in-metres value.
        assert 0.0 <= g0['R_i'] < 0.5, (
            f"R_i={g0['R_i']} is not a Fresnel reflectance (should be "
            f"in [0, ~0.5] for typical glass surfaces).")
        assert 0.0 <= g0['R_j'] < 0.5, (
            f"R_j={g0['R_j']} is not a Fresnel reflectance.")
        # The intensity is the product R_i * R_j by construction --
        # this also pins that we didn't accidentally rename one of
        # the dictionary keys when disambiguating from curvature radii.
        expected_I = g0['R_i'] * g0['R_j']
        assert abs(g0['intensity'] - expected_I) < 1e-12, (
            f"intensity={g0['intensity']} != R_i*R_j={expected_I} -- "
            f"the public Fresnel-reflectance contract was broken.")


# ============================================================================
# L6.a -- apply_mirror array_namespace dispatch: JAX input -> JAX output
# ============================================================================


class TestApplyMirrorBackendDispatch:
    """Pin that ``apply_mirror`` now goes through ``array_namespace``
    dispatch instead of hard-coding ``np.exp`` / ``np.where`` /
    ``np.meshgrid``.

    Pre-fix: any non-NumPy input was silently downcast through the host
    (JAX -> NumPy -> JAX round-trip) without the user knowing -- defeated
    the point of running on JAX.

    Post-fix: JAX input stays on JAX, CuPy input stays on CuPy.
    """

    def test_jax_input_returns_jax_output(self):
        """``apply_mirror(jax_E_in, ...)`` returns a JAX array (the
        output namespace is ``jax.numpy``, not ``numpy``)."""
        jax = pytest.importorskip('jax')
        jnp = pytest.importorskip('jax.numpy')

        N = 64
        dx = 8e-6
        lam = 1.31e-6
        R = 50e-3

        E_in = jnp.ones((N, N), dtype=jnp.complex64)
        # Pre-condition: input really is a JAX array.
        assert isinstance(E_in, jax.Array), (
            "Test setup error: jnp.ones should produce a jax.Array.")

        E_out = apply_mirror(E_in, lam, dx, radius=R,
                             aperture_diameter=3e-3)

        # Post-condition: output is a JAX array (not a NumPy fallback).
        assert isinstance(E_out, jax.Array), (
            f"apply_mirror with JAX input must return a JAX array, "
            f"got {type(E_out).__module__}.{type(E_out).__name__}.  "
            f"Indicates apply_mirror still uses np.* directly instead "
            f"of array_namespace(E_in) dispatch (L6).")
        # And the dtype is preserved on the JAX side.
        assert E_out.dtype == jnp.complex64, (
            f"Output dtype {E_out.dtype} != complex64 (input dtype).  "
            f"Backend dispatch should preserve dtype.")

    def test_jax_flat_mirror_returns_jax_output(self):
        """Flat-mirror branch (radius=None) also goes through the
        array-namespace dispatch (only the aperture mask is applied)."""
        jax = pytest.importorskip('jax')
        jnp = pytest.importorskip('jax.numpy')

        N = 64
        dx = 8e-6
        lam = 1.31e-6
        # JAX defaults to 32-bit; use complex64 to avoid the truncation
        # warning and to keep tests independent of the user's
        # JAX_ENABLE_X64 env setting.
        E_in = jnp.ones((N, N), dtype=jnp.complex64)
        E_out = apply_mirror(E_in, lam, dx, radius=None,
                             aperture_diameter=3e-3)
        assert isinstance(E_out, jax.Array), (
            "Flat-mirror branch must also preserve the JAX backend.")

    def test_numpy_input_still_returns_numpy(self):
        """Regression guard: the NumPy fast-path (legacy code path
        using ``_surface_sag_general`` with numba aspherics) is not
        broken by the dispatch change."""
        N = 64
        dx = 8e-6
        lam = 1.31e-6
        E_in = np.ones((N, N), dtype=np.complex128)
        E_out = apply_mirror(E_in, lam, dx, radius=50e-3,
                             aperture_diameter=3e-3)
        assert isinstance(E_out, np.ndarray), (
            "NumPy input must still return a NumPy ndarray (regression "
            "guard).")
        assert E_out.dtype == np.complex128

    def test_jax_vs_numpy_close(self):
        """Cross-backend numeric agreement: JAX and NumPy paths produce
        equivalent results for a small mirror up to JAX's default
        single-precision rounding (``jax_enable_x64`` is off by default,
        so JAX silently demotes complex128 -> complex64)."""
        jax = pytest.importorskip('jax')
        jnp = pytest.importorskip('jax.numpy')

        N = 64
        dx = 8e-6
        lam = 1.31e-6
        R = 50e-3
        aperture = 3e-3

        # Match JAX's default precision (complex64) on the NumPy side
        # so the two compute paths are at the same dtype.
        E_in_np = np.ones((N, N), dtype=np.complex64)
        E_out_np = apply_mirror(E_in_np, lam, dx, radius=R,
                                aperture_diameter=aperture)

        E_in_jx = jnp.ones((N, N), dtype=jnp.complex64)
        E_out_jx = apply_mirror(E_in_jx, lam, dx, radius=R,
                                aperture_diameter=aperture)
        E_out_jx_host = np.asarray(E_out_jx)

        # Single-precision phase tolerance: 2 * k * sag with sag ~ 1e-5 m
        # and k = 2*pi/1.31e-6 m^-1 -> phase ~ 100 rad.  At fp32 each
        # operation has ~1e-7 relative error, accumulating to ~1e-5
        # absolute on the exp() output.  1e-4 is comfortable headroom.
        max_err = float(np.max(np.abs(E_out_np - E_out_jx_host)))
        assert max_err < 1e-4, (
            f"JAX vs NumPy apply_mirror disagreement |dE|_max = "
            f"{max_err:.3e}, expected < 1e-4 at fp32 precision.  "
            f"Indicates the JAX inline-sag branch is computing something "
            f"different from the NumPy helper.")


# ============================================================================
# L6.b -- apply_mirror dy parameter: anamorphic geometry
# ============================================================================


class TestApplyMirrorDyAnamorphic:
    """Pin that ``apply_mirror`` now accepts a ``dy`` parameter and the
    aperture mask correctly samples a circular *physical* aperture on
    an anamorphic grid (``dx != dy``).

    With ``dx > dy`` (coarser x-pitch, finer y-pitch) more y-pixels fit
    inside a fixed-diameter circular aperture than x-pixels.  Pre-fix
    (no dy, both axes used dx) the mask was a degenerate ellipse in
    physical space.
    """

    def test_signature_accepts_dy_kwarg(self):
        """``apply_mirror`` exposes a ``dy`` keyword argument with a
        default of ``None`` (interpreted as ``dx``)."""
        sig = inspect.signature(apply_mirror)
        assert 'dy' in sig.parameters, (
            "apply_mirror must accept a 'dy' keyword argument (v4.13.0 "
            "L6).  Current signature: " + str(sig))
        assert sig.parameters['dy'].default is None, (
            "apply_mirror 'dy' parameter must default to None (interpreted "
            "as dx for backward compatibility).")

    def test_dy_default_equals_dx_backwards_compat(self):
        """When ``dy`` is omitted (or explicitly ``None``) the output
        is identical to passing ``dy=dx`` -- i.e. the v4.12.x square-
        grid behaviour."""
        N = 32
        dx = 1e-5
        lam = 1.31e-6
        E_in = np.ones((N, N), dtype=np.complex128)
        E_default = apply_mirror(E_in, lam, dx, radius=20e-3,
                                 aperture_diameter=8 * dx)
        E_explicit = apply_mirror(E_in, lam, dx, dy=dx, radius=20e-3,
                                  aperture_diameter=8 * dx)
        assert np.allclose(E_default, E_explicit, atol=0.0), (
            "dy=None must reproduce dy=dx exactly (backwards-compat).")

    def test_anamorphic_aperture_geometry(self):
        """With ``dy = dx / 2`` (finer y-pitch) more y-pixels fit inside
        a fixed circular aperture than x-pixels.

        Synthetic asymmetric grid:
          * dx = 1e-5 m (10 micron x-pitch)
          * dy = 5e-6 m (5 micron y-pitch)
          * aperture_diameter = 10 * dx = 1e-4 m
        Along x: half-aperture span = 5*dx, so the mask is +-5 pixels
        wide -> ~11 pixels through the centre row.
        Along y: half-aperture span = 5*dx / dy = 10 pixels, so the
        mask is +-10 pixels wide -> ~21 pixels through the centre col.
        Ratio ny/nx -> ~ dx/dy = 2.
        """
        N = 64
        dx = 1e-5
        dy = 5e-6
        lam = 1.31e-6
        aperture_diameter = 10 * dx

        E_in = np.ones((N, N), dtype=np.complex128)
        E_out = apply_mirror(E_in, lam, dx, dy=dy,
                             aperture_diameter=aperture_diameter)

        mask = np.abs(E_out) > 0
        nx_in = int(mask[N // 2, :].sum())  # along x at y=0
        ny_in = int(mask[:, N // 2].sum())  # along y at x=0

        # Expect ny ~ 2 nx (dy half the size of dx, so twice as many
        # y-pixels fit in the same physical aperture).
        assert nx_in > 0 and ny_in > 0, (
            "Aperture mask should leave SOME pixels through; got "
            f"nx_in={nx_in}, ny_in={ny_in}.")
        ratio = ny_in / nx_in
        assert 1.5 < ratio < 2.5, (
            f"Anamorphic ratio ny/nx = {ratio:.3f} expected ~2 "
            f"(dx={dx}, dy={dy} -> dx/dy=2).  Got nx_in={nx_in}, "
            f"ny_in={ny_in}.  Indicates dy is not flowing into the "
            f"meshgrid.")

    def test_anamorphic_circular_mask_matches_physical_ellipse(self):
        """For a circular *physical* aperture on an anamorphic grid the
        binary mask in pixel-space is the set of (i, j) satisfying
        ``((i - N/2)*dx)^2 + ((j - N/2)*dy)^2 <= (D/2)^2``.

        We compute the expected mask from first principles and verify
        apply_mirror's output matches it exactly.
        """
        N = 64
        dx = 1e-5
        dy = 7e-6  # anamorphic, not a simple 2:1 ratio
        lam = 1.31e-6
        D = 0.6 * N * min(dx, dy)  # fits comfortably inside

        E_in = np.ones((N, N), dtype=np.complex128)
        E_out = apply_mirror(E_in, lam, dx, dy=dy,
                             aperture_diameter=D)

        # Build expected mask in physical (m) coordinates.
        x = (np.arange(N) - N / 2) * dx
        y = (np.arange(N) - N / 2) * dy
        X, Y = np.meshgrid(x, y)
        h_sq = X ** 2 + Y ** 2
        expected_mask = h_sq <= (D / 2) ** 2

        actual_mask = np.abs(E_out) > 0
        assert np.array_equal(actual_mask, expected_mask), (
            f"apply_mirror anamorphic mask doesn't match physical-ellipse "
            f"prediction.  diff = "
            f"{int((actual_mask != expected_mask).sum())} pixels.")
