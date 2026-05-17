"""Pinning tests for the v4.14.1 audit closures handled by Agent D.

Audit reference
---------------

``AUDIT_V4_14_0_2026_05_17.md`` flagged 1 P0 + 6 P1 follow-ups.  Agent
D closed:

* **P1-NEW-4** -- ``clear_lg_mode_stack_cache`` was present in
  ``lumenairy/propagators/asymptotic.__all__`` and the v4.14.0
  CHANGELOG claimed it was a public top-level helper, but it was
  never imported into ``lumenairy/__init__.py``.  The 7 sibling
  ``clear_*_cache`` helpers were all re-exported at top level; this
  one slipped through.  v4.14.1 wires it in to match the documented
  contract.  (The companion file
  ``test_v4_14_1_dispatcher_pin_cache_clears.py`` adds a
  parametrized sibling-gap pin so a future regression of the same
  class fails at CI.)
* **P1-NEW-6** -- ``encircled_energy_radius``'s in-line comment
  (``"The first sample (radius 0) is always ee = 0"``) was wrong:
  ``encircled_energy_curve`` returns ``ee[0] = p_cum[0]`` (the
  centre-pixel intensity contribution) when ``radii_out[0] = 0``
  collides with ``r_sorted[0] = 0``, which happens whenever the
  centroid lands on a pixel and the default radii grid is used.  The
  ``idx <= 0`` short-circuit at the bottom of
  ``encircled_energy_radius`` then returns ``radii[0]`` (= 0 m) when
  ``threshold <= ee[0]`` -- physically reasonable for a delta-like
  hot-centre input, but the documented invariant was off.  v4.14.1
  rewrites both the docstring and the comment to describe the
  observed behaviour and pins the hot-centre case here.

Author: Andrew Traverso -- v4.14.1 / Agent D
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la


# ============================================================================
# P1-NEW-4 -- clear_lg_mode_stack_cache promoted to top-level public API
# ============================================================================

class TestP1New4ClearLgModeStackCacheTopLevel:
    """``clear_lg_mode_stack_cache`` is re-exported at the top level.

    Pre-fix the v4.14.0 CHANGELOG (line 52) advertised the helper as a
    public top-level utility ("Public ``clear_lg_mode_stack_cache()``
    for explicit flushes") but ``lumenairy/__init__.py`` neither
    imported nor re-exported it; a user following the CHANGELOG would
    hit ``AttributeError`` on ``lumenairy.clear_lg_mode_stack_cache``.
    """

    def test_clear_lg_mode_stack_cache_is_callable_at_top_level(self):
        """The helper is accessible as ``la.clear_lg_mode_stack_cache``
        and is callable.
        """
        assert hasattr(la, 'clear_lg_mode_stack_cache')
        assert callable(la.clear_lg_mode_stack_cache)

    def test_clear_lg_mode_stack_cache_in_all(self):
        """The helper is listed in ``lumenairy.__all__`` alongside the
        7 sibling ``clear_*_cache`` helpers.
        """
        assert 'clear_lg_mode_stack_cache' in la.__all__

    def test_clear_lg_mode_stack_cache_no_error_on_empty(self):
        """The helper is safe to call when the caches are already
        empty -- subsequent decompose calls rebuild and re-cache.
        """
        # Should not raise.
        la.clear_lg_mode_stack_cache()
        # Idempotent.
        la.clear_lg_mode_stack_cache()

    def test_clear_lg_mode_stack_cache_matches_submodule_export(self):
        """The top-level re-export is the same object as the
        ``propagators.asymptotic`` definition.  Pre-fix the audit
        flagged the cross-module identity could drift if the import
        were retyped at the top level.
        """
        from lumenairy.propagators.asymptotic import (
            clear_lg_mode_stack_cache as submod_helper,
        )
        assert la.clear_lg_mode_stack_cache is submod_helper


# ============================================================================
# P1-NEW-6 -- encircled_energy_radius hot-centre behaviour
# ============================================================================

class TestP1New6EncircledEnergyHotCentre:
    """``encircled_energy_radius`` returns 0 m for a delta-like input
    whose entire intensity is concentrated in the centre pixel.

    Pre-fix the docstring + comment claimed ``ee[0] = 0 always``; in
    reality ``encircled_energy_curve`` returns ``ee[0] = p_cum[0]``
    when ``radii_out[0] = 0`` collides with ``r_sorted[0] = 0``.  For
    a perfect-centre delta, ``p_cum[0] == 1`` and any
    ``threshold < 1`` short-circuits at the ``idx <= 0`` branch,
    returning ``radii[0] = 0``.

    The v4.14.1 docstring fix documents this as the physically
    reasonable hot-centre answer.  This test pins that contract.
    """

    def _build_delta_at_centre(self, N=128, dx=1e-6):
        """Construct a complex field that is zero everywhere except
        the central pixel.

        Even-N convention: the central pixel index is ``(N//2, N//2)``
        which corresponds to ``(x, y) = (0, 0)`` under the
        ``(np.arange(N) - N/2) * dx`` axis convention used by
        :func:`encircled_energy_curve`.  ``beam_centroid`` should
        return exactly the origin for this field, so ``R[N//2, N//2]
        = 0`` and the centre pixel lands at ``r_sorted[0] = 0``.
        """
        E = np.zeros((N, N), dtype=np.complex128)
        E[N // 2, N // 2] = 1.0 + 0.0j
        return E

    def test_hot_centre_small_threshold_returns_zero(self):
        """``encircled_energy_radius(threshold=0.5)`` returns 0 m for
        a delta at the centre pixel: 100 % of the power lives at
        ``r = 0`` so any threshold in ``(0, 1]`` is reached at the
        first sample of the curve.
        """
        E = self._build_delta_at_centre(N=128, dx=1e-6)
        # threshold well below the centre-pixel's contribution
        # (which is 1.0 for a single-pixel delta).
        r = la.encircled_energy_radius(E, dx=1e-6, threshold=0.5)
        assert r == 0.0, (
            f"Hot-centre delta should yield r_threshold = 0 m for "
            f"any threshold below the centre-pixel contribution; "
            f"got r = {r!r}.")

    def test_hot_centre_high_threshold_also_zero(self):
        """Same delta at the centre pixel, threshold near 1: the
        centre pixel already accounts for 100 % of the in-grid power
        so the curve hits 1 at ``r = 0`` and any threshold <= 1
        short-circuits.
        """
        E = self._build_delta_at_centre(N=128, dx=1e-6)
        r = la.encircled_energy_radius(E, dx=1e-6, threshold=0.99)
        assert r == 0.0, (
            f"Hot-centre delta with threshold=0.99 should still "
            f"short-circuit at r = 0 because p_cum[0] = 1; got "
            f"r = {r!r}.")

    def test_encircled_curve_starts_at_centre_pixel_value(self):
        """Cross-check the underlying curve: ``ee[0]`` is the
        centre-pixel cumulative-power fraction, not zero.  This is
        the observation the v4.14.1 docstring + comment fix now
        accurately describes.
        """
        E = self._build_delta_at_centre(N=128, dx=1e-6)
        radii, ee = la.encircled_energy_curve(
            E, dx=1e-6, n_radii=8)
        # First sampled radius is 0 by construction.
        assert radii[0] == 0.0
        # For a single-pixel delta at the centre, ee[0] is 1.0 (the
        # entire in-grid power lives in the centre pixel).
        assert ee[0] == pytest.approx(1.0, abs=1e-12), (
            f"Hot-centre delta should give ee[0] = 1 (not 0): the "
            f"centre pixel holds all the power.  Observed "
            f"ee[0] = {ee[0]!r}.")

    def test_smooth_gaussian_ee_radius_nonzero(self):
        """Sanity: for a well-resolved smooth Gaussian the
        encircled-energy radius is NOT zero -- the hot-centre
        short-circuit only fires for delta-like inputs.  This guards
        against an over-broad fix that would return 0 for all
        inputs.
        """
        N, dx = 256, 1e-6
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        w0 = 20e-6
        E = np.exp(-(X * X + Y * Y) / w0 ** 2).astype(np.complex128)
        r84 = la.encircled_energy_radius(E, dx, threshold=0.84)
        # A 1/e^2 = w0 = 20 um Gaussian has 84% encircled near w0;
        # certainly not 0.
        assert r84 > 5e-6, (
            f"Smooth Gaussian (w0=20um) at 84% threshold should "
            f"give r > 5um; got {r84!r}.  The hot-centre fix must "
            f"NOT fire for non-delta inputs.")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
