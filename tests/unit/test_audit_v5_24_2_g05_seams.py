"""Regression pins for the v5.24.2 exhaustive-audit G05 seam group.

Each test targets one finding and is written to FAIL on the pre-fix code
and PASS after.  Where practical the oracle is INDEPENDENT of the code
under test (a hand formula, a cross-implementation reference, or the sibling
NumPy path) rather than the code's own expression.

Findings covered:
  * S1-12  PMM off-plane tile detection relative to tensor scale
  * S1-13  Berreman numpy/JAX forward-backward split ordering aligned
  * S1-16  BOR mode classifiers share the index-ceiling leg (lockstep)
  * S2-12  fft_backend_for gates pyFFTW on iscomplexobj like _fft2
  * S2-13  JAX system kernel threads dy on propagate steps
  * S3-13  JAX Newton iteration count aligned to the NumPy reference
  * S3-14  JAX merit x64 requirement: warn-and-enable / require-and-raise
  * S3-15  trace_jax docstring signposts trace_jax_with_params
"""
from __future__ import annotations

import inspect
import re
import warnings

import numpy as np
import pytest

# ============================================================================
# S1-12 -- PMM off-plane tile detection uses a scale-relative threshold
# ============================================================================

class TestS1_12OffplaneRelativeThreshold:
    def test_rotation_roundoff_stays_in_plane(self):
        """A physically in-plane cell carrying ~1e-17 rounding noise in the
        xz/yz/zx/zy slots must be classified IN-PLANE (was routed to the
        ~8x-slower 4Nf generator by the strict > 0.0 test)."""
        from lumenairy.elements.pmm.twod_jones import _tile_is_offplane

        tile = np.zeros((2, 3, 3), dtype=complex)
        # two regions, diagonal (in-plane) tensors of O(1) scale
        for r, (exx, eyy, ezz) in enumerate([(2.1, 2.4, 2.2), (1.9, 2.0, 2.05)]):
            tile[r] = np.diag([exx, eyy, ezz]).astype(complex)
        # inject float roundoff into every off-plane slot
        for a, b in [(0, 2), (1, 2), (2, 0), (2, 1)]:
            tile[:, a, b] = 1e-17
        assert _tile_is_offplane(tile) is False

    def test_genuine_offplane_still_detected(self):
        """A real out-of-plane coupling (O(tensor scale)) must be detected."""
        from lumenairy.elements.pmm.twod_jones import _tile_is_offplane

        tile = np.diag([2.1, 2.4, 2.2]).astype(complex)[None, :, :].copy()
        tile[0, 0, 2] = 0.1          # genuine xz coupling
        tile[0, 2, 0] = 0.1
        assert _tile_is_offplane(tile) is True

    def test_threshold_is_relative_not_absolute(self):
        """A coupling many decades above roundoff but small in absolute terms
        (e.g. a nano-radian tilt of a birefringent tensor) is still off-plane;
        the discriminator is the tensor SCALE, not a fixed absolute floor."""
        from lumenairy.elements.pmm.twod_jones import _tile_is_offplane

        scale = 3.0
        tile = np.diag([scale, scale * 0.9, scale]).astype(complex)[None].copy()
        # off ~ 1e-9 * scale: >> 1e-16 roundoff, >> 1e-12 * scale floor
        tile[0, 1, 2] = 1e-9 * scale
        tile[0, 2, 1] = 1e-9 * scale
        assert _tile_is_offplane(tile) is True


# ============================================================================
# S1-13 -- Berreman fwd/bwd split matches the JAX stable-flag rule
# ============================================================================

def _jax_partition_oracle(gam):
    """Independent reimplementation of _berreman_jax._layer_modes_jax's
    partition: stable argsort of the per-mode forward flag."""
    g = np.asarray(gam)
    tol = 1e-9 * max(1.0, float(np.max(np.abs(g))))
    is_fwd = np.where(g.real < -tol, True,
                      np.where(g.real > tol, False, g.imag > 0.0))
    order = np.argsort(np.where(is_fwd, 0, 1), kind='stable')
    return list(order[:2]), list(order[2:])


class TestS1_13BerremanSplitAligned:
    def test_physical_two_forward_unchanged(self):
        """Two clearly-forward + two clearly-backward modes: the partition is
        the physical one and identical to the JAX rule (no regression)."""
        from lumenairy.elements.berreman import _split_fwd_bwd

        gam = np.array([-2.0, +3.0, -1.0, +4.0], dtype=complex)
        fwd, bwd = _split_fwd_bwd(gam)
        assert list(fwd) == [0, 2]
        assert list(bwd) == [1, 3]
        of, ob = _jax_partition_oracle(gam)
        assert list(fwd) == of and list(bwd) == ob

    def test_degenerate_branch_matches_jax(self):
        """Three forward flags (a degenerate/bianisotropic case): the pre-fix
        numpy fallback ranked by decay (argsort Re gam) and returned fwd=[0,2];
        the aligned rule returns the JAX flag-then-index fwd=[0,1]."""
        from lumenairy.elements.berreman import _split_fwd_bwd

        gam = np.array([-3.0, -1.0, -2.0, +5.0], dtype=complex)
        fwd, bwd = _split_fwd_bwd(gam)
        of, ob = _jax_partition_oracle(gam)
        assert list(fwd) == of == [0, 1]
        assert list(bwd) == ob == [2, 3]
        # explicit fail-before witness: the old decay-ranked fallback gave [0, 2]
        assert list(fwd) != [0, 2]


# ============================================================================
# S1-16 -- BOR mode classifiers are INTENTIONALLY basis-specific
# ============================================================================

def _bor_nodal_core_oracle(qn, reldiv, reldiv_tol=0.5):
    """Independent physical criterion for the NODAL classifier: propagating
    (imag small), above the q~0 floor, and div-clean (reldiv).  The
    index-ceiling leg the staggered twins carry is DELIBERATELY absent --
    applying it to the nodal FD basis over-filters the reldiv-screened set and
    degrades the documented ~4% nodal energy floor (regression guarded by
    test_bor_solve::test_structured_stack_energy_floor_nodal, which the forced
    index-ceiling drove to ~10.7%)."""
    return ((np.abs(qn.imag) < 5e-5) & (qn.real > 1e-6)
            & (reldiv < reldiv_tol))


class TestS1_16BorClassifierBasisSpecific:
    """S1-16 resolution: the three BOR classifiers share the {imag, real-floor}
    core and each carries ONE basis-specific leg -- the NODAL classifier's is
    reldiv (screens the FD spurious sea); the staggered twins' is the index
    ceiling (they are div-conforming, spurious-free, so carry no reldiv).  The
    finding was that the 'keep all three in lockstep' comment was FALSE; the fix
    is to make the comment accurate, NOT to force the index ceiling onto the
    nodal path (which over-filters and breaks its energy floor)."""

    def test_nodal_classifier_does_not_apply_index_ceiling(self):
        """The nodal ``_physical_propagating`` must NOT reject a mode purely
        because q/k0 exceeds sqrt(eps): a propagating, div-clean 'super-index'
        mode is KEPT (the index-ceiling leg belongs to the staggered twins)."""
        from lumenairy.elements.bor.bor_solve import _physical_propagating

        k0 = 3.0
        eps = 2.25 + 0j            # sqrt(eps).real = 1.5
        qn = np.array([0.8, 1.7, 5e-7, 2e-3, 0.6], dtype=complex)
        reldiv = np.array([0.0, 0.0, 0.0, 0.0, 0.9])
        L = {"q": qn * k0, "reldiv": reldiv, "eps_ceiling": eps}
        keep = np.asarray(_physical_propagating(L, k0))
        oracle = _bor_nodal_core_oracle(qn, reldiv)
        assert list(keep) == list(oracle)
        # the super-index mode (qn=1.7 > 1.5) is div-clean + propagating and is
        # KEPT (NOT rejected by an index ceiling the nodal basis must not apply)
        assert keep[1] == True    # noqa: E712  (numpy bool identity)

    def test_nodal_unique_leg_is_reldiv_not_index_ceiling(self):
        """The nodal classifier reduces EXACTLY to {imag, real-floor, reldiv}:
        a reldiv-dirty mode is dropped (the nodal-unique leg), while a
        super-index reldiv-clean mode is kept (no ceiling applied)."""
        from lumenairy.elements.bor.bor_solve import _physical_propagating

        k0 = 5.0
        eps = (1.41 ** 2) + 0j
        qn = np.array([1.40, 1.42, 0.7, 1e-3, 5e-7, 1.0], dtype=complex)
        reldiv = np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0])
        L = {"q": qn * k0, "reldiv": reldiv, "eps_ceiling": eps}
        keep = np.asarray(_physical_propagating(L, k0))
        assert list(keep) == list(_bor_nodal_core_oracle(qn, reldiv))
        # qn=1.42 > sqrt(eps)=1.41 but reldiv-clean + propagating -> KEPT
        assert keep[1] == True    # noqa: E712
        # the reldiv-dirty mode (index 2) IS dropped by the nodal-unique leg
        assert keep[2] == False   # noqa: E712

    def test_build_layer_stores_index_ceiling_reference(self):
        """build_layer records the medium eps ceiling as a per-layer REFERENCE
        (the staggered twins' leg value), but the nodal ``_physical_propagating``
        does not apply it -- so no existing nodal result changes."""
        from lumenairy.elements.bor.bor_solve import (
            _physical_propagating,
            build_layer,
        )

        m, Rbig, N, k0 = 1, 6.0, 60, 2.0
        n_med = 1.7
        L = build_layer(m, Rbig, N, lambda r: np.full_like(r, n_med ** 2,
                                                            dtype=complex), k0)
        assert "eps_ceiling" in L
        assert abs(np.sqrt(L["eps_ceiling"]).real - n_med) < 1e-9
        # nodal criterion (imag, real-floor, reldiv only) -- ceiling NOT applied
        qn = L["q"] / k0
        crit = ((np.abs(qn.imag) < 5e-5) & (qn.real > 1e-6)
                & (L["reldiv"] < 0.5))
        post = np.asarray(_physical_propagating(L, k0))
        assert np.array_equal(crit, post)


# ============================================================================
# S2-12 -- fft_backend_for mirrors the _fft2 iscomplexobj gate
# ============================================================================

class TestS2_12FftBackendForRealDtype:
    def test_real_dtype_not_reported_as_pyfftw(self, monkeypatch):
        """Force the pyFFTW branch active and verify a real-dtype array is NOT
        reported as 'pyfftw' (it routes to scipy/numpy in _fft2), while its
        complex sibling still is -- the iscomplexobj gate _fft2 already has."""
        from lumenairy.backend.fft import fft_backend_for
        from lumenairy.propagators import fft_infra as prop

        monkeypatch.setattr(prop, 'USE_PYFFTW', True, raising=False)
        monkeypatch.setattr(prop, 'PYFFTW_AVAILABLE', True, raising=False)
        monkeypatch.setattr(prop, 'FFTW_MIN_SIZE', 64, raising=False)
        monkeypatch.setattr(prop, '_PYFFTW_BAD_SHAPES', set(), raising=False)

        cplx = np.ones((128, 128), dtype=np.complex128)
        real = np.ones((128, 128), dtype=np.float64)
        assert fft_backend_for(cplx) == 'pyfftw'      # sanity: gate is live
        assert fft_backend_for(real) != 'pyfftw'      # the fix


# ============================================================================
# S2-13 -- JAX system kernel threads dy on propagate steps
# ============================================================================

class TestS2_13JaxSystemThreadsDy:
    def _field(self, jnp, Ny, Nx, dx, dy):
        y = (np.arange(Ny) - Ny / 2) * dy
        x = (np.arange(Nx) - Nx / 2) * dx
        X, Y = np.meshgrid(x, y)
        w = 40e-6
        E = np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)
        return E

    def test_anamorphic_matches_numpy_asm(self):
        pytest.importorskip('jax')
        import jax

        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp

        from lumenairy.propagators.asm import angular_spectrum_propagate
        from lumenairy.propagators.system import propagate_through_system_jax

        wl = 1.31e-6
        dx, dy = 8e-6, 16e-6          # strongly anamorphic (dy = 2 dx)
        z = 5e-3
        E = self._field(jnp, 96, 96, dx, dy)
        elements = [{'type': 'propagate', 'z': z}]

        out = np.asarray(propagate_through_system_jax(
            jnp.asarray(E), elements, wl, dx, dy=dy))
        # INDEPENDENT oracle: the NumPy ASM with the SAME anamorphic pitch
        ref = angular_spectrum_propagate(E, z, wl, dx, dy=dy)

        num = np.abs(out - ref).max()
        den = np.abs(ref).max()
        assert num / den < 1e-6, f"anamorphic JAX vs NumPy rel err {num/den:.2e}"

    def test_dy_actually_matters(self):
        """Guard: propagating with dy != dx differs materially from dy == dx, so
        the previous drop-dy behaviour was a real bug, not a harmless no-op."""
        pytest.importorskip('jax')
        import jax

        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp

        from lumenairy.propagators.system import propagate_through_system_jax

        wl = 1.31e-6
        dx, dy = 8e-6, 16e-6
        z = 5e-3
        E = self._field(jnp, 96, 96, dx, dy)
        elements = [{'type': 'propagate', 'z': z}]
        with_dy = np.asarray(propagate_through_system_jax(
            jnp.asarray(E), elements, wl, dx, dy=dy))
        wrong = np.asarray(propagate_through_system_jax(
            jnp.asarray(E), elements, wl, dx, dy=dx))   # the pre-fix behaviour
        assert np.abs(with_dy - wrong).max() > 1e-3 * np.abs(with_dy).max()


# ============================================================================
# S3-13 -- JAX Newton iteration count aligned to the NumPy reference
# ============================================================================

class TestS3_13NewtonIterCount:
    def test_jax_iters_at_least_numpy_reference(self):
        """The JAX fixed Newton count must be >= the NumPy sag solver's max
        iteration bound.  The bound is read from intersection.py's source
        (independent of the JAX constant)."""
        from lumenairy.raytrace import intersection
        from lumenairy.raytrace.jax_trace import _ASPHERIC_NEWTON_ITERS

        src = inspect.getsource(intersection)
        bounds = [int(n) for n in re.findall(r'for _ in range\((\d+)\)', src)]
        assert bounds, "could not locate the NumPy Newton range() bound"
        numpy_max = max(bounds)
        assert numpy_max == 10        # pin the reference we aligned to
        assert _ASPHERIC_NEWTON_ITERS >= numpy_max


# ============================================================================
# S3-14 -- JAX merit x64 requirement is explicit (warn / raise), not silent
# ============================================================================

class _FakeJaxConfig:
    def __init__(self, enabled):
        self._enabled = enabled
        self.updates = []

    def read(self, key):
        assert key == 'jax_enable_x64'
        return self._enabled

    def update(self, key, value):
        assert key == 'jax_enable_x64'
        self.updates.append((key, value))
        self._enabled = value


class TestS3_14X64Explicit:
    def test_require_and_raise_when_off(self, monkeypatch):
        pytest.importorskip('jax')
        import jax

        from lumenairy.optimize.jax_merits import _ensure_jax_x64

        fake = _FakeJaxConfig(enabled=False)
        monkeypatch.setattr(jax, 'config', fake)
        with pytest.raises(RuntimeError, match='jax_enable_x64'):
            _ensure_jax_x64('unit', enable_x64=False)
        assert fake.updates == []          # no global mutation on the raise path

    def test_warn_and_enable_when_off(self, monkeypatch):
        pytest.importorskip('jax')
        import jax

        from lumenairy.optimize.jax_merits import _ensure_jax_x64

        fake = _FakeJaxConfig(enabled=False)
        monkeypatch.setattr(jax, 'config', fake)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _ensure_jax_x64('unit', enable_x64=True)
        assert fake.updates == [('jax_enable_x64', True)]
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)

    def test_noop_when_already_on(self, monkeypatch):
        pytest.importorskip('jax')
        import jax

        from lumenairy.optimize.jax_merits import _ensure_jax_x64

        fake = _FakeJaxConfig(enabled=True)
        monkeypatch.setattr(jax, 'config', fake)
        # already on: neither raises (enable_x64=False) nor mutates
        _ensure_jax_x64('unit', enable_x64=False)
        assert fake.updates == []


# ============================================================================
# S3-15 -- trace_jax docstring signposts trace_jax_with_params
# ============================================================================

class TestS3_15Signpost:
    def test_docstring_points_to_params_variant(self):
        from lumenairy.raytrace.jax_trace import trace_jax

        doc = trace_jax.__doc__ or ""
        assert 'trace_jax_with_params' in doc
        low = doc.lower()
        assert ('re-jit' in low) or ('sweep' in low) or ('finite-difference' in low)
