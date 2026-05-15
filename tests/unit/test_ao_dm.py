"""Unit tests for DeformableMirror memory behaviour and fit_phase.

Regression coverage for the deferred 4.0.1-era audit item that
flagged ``DeformableMirror._IF_basis`` as an 8-GB foot-gun on a
32 x 32 actuator grid x 1024 x 1024 pupil and made worked examples
reach into the private attribute to do modal-to-zonal projection.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.analysis.ao import DeformableMirror


class TestCacheBasisAuto:
    """The default ``cache_basis='auto'`` policy should pre-compute
    the basis for small configurations but go lazy for large ones."""

    def test_small_dm_eagerly_caches(self):
        dm = DeformableMirror(n_actuators=8, pitch=2e-3, dx=20e-6, N=64)
        assert dm._cache_active is True
        assert dm._IF_basis is not None
        # 8*8*64*64*8 = 2 MB, well under the 512 MB ceiling.
        assert dm._IF_basis.shape == (8, 8, 64, 64)

    def test_large_dm_skips_eager_cache(self):
        """The 8 GB foot-gun: a 32x32 DM on a 1024x1024 grid.  Auto
        mode must NOT allocate it eagerly."""
        dm = DeformableMirror(n_actuators=32, pitch=2e-3, dx=2e-6, N=1024)
        assert dm._cache_active is False
        assert dm._IF_basis is None


class TestPhaseEquivalence:
    """The lazy path must produce the same phase map as the eager
    path for any command vector."""

    def test_lazy_matches_eager(self):
        kw = dict(n_actuators=6, pitch=2e-3, dx=20e-6, N=64,
                  inter_actuator_coupling=0.15)
        dm_eager = DeformableMirror(cache_basis=True, **kw)
        dm_lazy = DeformableMirror(cache_basis=False, **kw)
        # Drive both with the same random command.
        rng = np.random.default_rng(0)
        cmd = rng.standard_normal((6, 6))
        dm_eager.set_command(cmd)
        dm_lazy.set_command(cmd)
        np.testing.assert_allclose(
            dm_eager.phase(), dm_lazy.phase(), rtol=1e-12, atol=1e-12
        )

    def test_zero_command_returns_zero_phase(self):
        dm = DeformableMirror(n_actuators=4, pitch=1e-3, dx=10e-6, N=32,
                                cache_basis=False)
        np.testing.assert_array_equal(dm.phase(), np.zeros((32, 32)))


class TestFitPhase:
    """The new public ``fit_phase`` is the replacement for reaching
    into ``_IF_basis`` for modal-to-zonal projection."""

    def test_fit_round_trip(self):
        """Fitting an IF-basis phase back through fit_phase must
        recover its original command vector to within float64 lstsq
        precision (gaussians overlap heavily so the design matrix is
        moderately conditioned, ~1e-6 typical for a 25-actuator DM)."""
        dm = DeformableMirror(n_actuators=5, pitch=2e-3, dx=20e-6, N=32)
        rng = np.random.default_rng(42)
        target_cmd = rng.standard_normal((5, 5))
        dm.set_command(target_cmd)
        target_phase = dm.phase()
        dm.set_command(np.zeros((5, 5)))   # reset
        fitted = dm.fit_phase(target_phase)
        np.testing.assert_allclose(fitted, target_cmd, rtol=1e-5, atol=1e-5)

    def test_fit_works_without_cache(self):
        """fit_phase must work in the lazy / no-cache regime too, and
        must agree with the cached path to lstsq precision."""
        dm = DeformableMirror(n_actuators=4, pitch=1e-3, dx=10e-6, N=32,
                                cache_basis=False)
        rng = np.random.default_rng(123)
        cmd = rng.standard_normal((4, 4))
        dm.set_command(cmd)
        target = dm.phase()
        dm.set_command(np.zeros((4, 4)))
        fitted = dm.fit_phase(target)
        np.testing.assert_allclose(fitted, cmd, rtol=1e-5, atol=1e-5)


class TestApplyDM:
    """The high-level apply path must keep working under both cache
    strategies."""

    @pytest.mark.parametrize("cache_basis", [True, False, 'auto'])
    def test_apply_preserves_amplitude(self, cache_basis):
        dm = DeformableMirror(n_actuators=4, pitch=1e-3, dx=10e-6, N=32,
                                cache_basis=cache_basis)
        rng = np.random.default_rng(7)
        dm.set_command(rng.standard_normal((4, 4)))
        E_in = np.ones((32, 32), dtype=complex)
        E_out = la.apply_dm(E_in, dm)
        # A pure phase mask preserves the amplitude of a unit field.
        np.testing.assert_allclose(np.abs(E_out), 1.0, atol=1e-12)
