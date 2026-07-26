"""Unit tests for the 4.8.1 scoped runtime-environment APIs:
``lumenairy_context`` and the ``dtype=`` kwarg on source factories.
"""

from __future__ import annotations

import numpy as np
import pytest

import lumenairy as lm
from lumenairy import (
    apply_globals,
    get_asm_cache_size,
    get_default_complex_dtype,
    get_fft_threads,
    get_max_ram,
    get_pyfftw_planner,
    lumenairy_context,
    set_asm_cache_size,
    set_default_complex_dtype,
    set_fft_threads,
    set_max_ram,
    set_pyfftw_planner,
    snapshot_globals,
)

# ============================================================================
# Snapshot / apply round-trip
# ============================================================================

class TestSnapshotApplyRoundTrip:
    """``apply_globals(snapshot_globals())`` should be a no-op."""

    def test_snapshot_keys(self):
        snap = snapshot_globals()
        assert set(snap.keys()) == {
            'complex_dtype', 'pyfftw_planner', 'fft_threads',
            'max_ram', 'asm_cache_size',
        }

    def test_round_trip_no_op(self):
        before = snapshot_globals()
        apply_globals(before)
        after = snapshot_globals()
        assert before == after

    def test_round_trip_after_modification(self):
        before = snapshot_globals()
        set_default_complex_dtype(np.complex64)
        set_pyfftw_planner('FFTW_MEASURE')
        # Restore
        apply_globals(before)
        after = snapshot_globals()
        assert after == before


# ============================================================================
# Context manager: restores on normal exit
# ============================================================================

class TestContextRestoreOnNormalExit:
    """Each kwarg should restore on the with-exit even if the body
    further mutates global state."""

    def test_complex_dtype_restore(self):
        prior = get_default_complex_dtype()
        with lumenairy_context(complex_dtype=np.complex64):
            assert get_default_complex_dtype() == np.dtype(np.complex64)
        assert get_default_complex_dtype() == prior

    def test_pyfftw_planner_restore(self):
        prior = get_pyfftw_planner()
        with lumenairy_context(pyfftw_planner='FFTW_MEASURE'):
            assert get_pyfftw_planner() == 'FFTW_MEASURE'
        assert get_pyfftw_planner() == prior

    def test_fft_threads_restore(self):
        prior = get_fft_threads()
        with lumenairy_context(fft_threads=2):
            assert get_fft_threads() == 2
        assert get_fft_threads() == prior

    def test_max_ram_restore(self):
        prior = get_max_ram()
        with lumenairy_context(max_ram=4):   # 4 GB
            inside = get_max_ram()
            assert inside == 4 * 1024**3
        assert get_max_ram() == prior

    def test_asm_cache_total_bytes_restore(self):
        prior = get_asm_cache_size()
        with lumenairy_context(asm_cache_max_total_bytes=512 * 1024**2):
            assert get_asm_cache_size()['h_max_total_bytes'] == 512 * 1024**2
            # Other caps unchanged
            assert get_asm_cache_size()['h_cache'] == prior['h_cache']
        assert get_asm_cache_size() == prior

    def test_asm_cache_per_entry_restore(self):
        prior = get_asm_cache_size()
        with lumenairy_context(asm_cache_max_bytes_per_entry=256 * 1024**2):
            assert get_asm_cache_size()['h_max_bytes_per_entry'] == 256 * 1024**2
        assert get_asm_cache_size() == prior

    def test_omitted_kwargs_unchanged(self):
        prior = snapshot_globals()
        with lumenairy_context(complex_dtype=np.complex64):
            inside = snapshot_globals()
            # Only complex_dtype should differ
            assert inside['complex_dtype'] == np.dtype(np.complex64)
            for k in ('pyfftw_planner', 'fft_threads', 'max_ram',
                      'asm_cache_size'):
                assert inside[k] == prior[k]
        assert snapshot_globals() == prior


# ============================================================================
# Context manager: restores on exception
# ============================================================================

class TestContextRestoreOnException:
    """Exceptions inside the with block must still restore prior state."""

    def test_complex_dtype_restored_after_exception(self):
        prior = get_default_complex_dtype()
        with pytest.raises(RuntimeError):
            with lumenairy_context(complex_dtype=np.complex64):
                assert get_default_complex_dtype() == np.dtype(np.complex64)
                raise RuntimeError("boom")
        assert get_default_complex_dtype() == prior

    def test_pyfftw_planner_restored_after_exception(self):
        prior = get_pyfftw_planner()
        with pytest.raises(ValueError):
            with lumenairy_context(pyfftw_planner='FFTW_MEASURE'):
                raise ValueError("boom")
        assert get_pyfftw_planner() == prior


# ============================================================================
# Context manager: nesting
# ============================================================================

class TestContextNesting:
    """Inner context overrides outer; on inner exit, outer's state
    (which may itself differ from the outermost default) is restored."""

    def test_two_level_nesting_dtype_and_planner(self):
        prior_dtype = get_default_complex_dtype()
        prior_planner = get_pyfftw_planner()

        with lumenairy_context(complex_dtype=np.complex64):
            assert get_default_complex_dtype() == np.dtype(np.complex64)
            assert get_pyfftw_planner() == prior_planner

            with lumenairy_context(pyfftw_planner='FFTW_MEASURE'):
                assert get_default_complex_dtype() == np.dtype(np.complex64)
                assert get_pyfftw_planner() == 'FFTW_MEASURE'

            # Inner exited: planner restored to outer's value (= prior)
            assert get_pyfftw_planner() == prior_planner
            # Outer's dtype still active
            assert get_default_complex_dtype() == np.dtype(np.complex64)

        # All restored
        assert get_default_complex_dtype() == prior_dtype
        assert get_pyfftw_planner() == prior_planner

    def test_inner_overrides_same_knob(self):
        prior = get_default_complex_dtype()
        with lumenairy_context(complex_dtype=np.complex64):
            with lumenairy_context(complex_dtype=np.complex128):
                assert get_default_complex_dtype() == np.dtype(np.complex128)
            # Inner exited -> outer's value (complex64)
            assert get_default_complex_dtype() == np.dtype(np.complex64)
        assert get_default_complex_dtype() == prior


# ============================================================================
# Context manager: clear_caches_on_exit flag
# ============================================================================

class TestClearCachesOnExit:
    """The ``clear_caches_on_exit=True`` flag drops the ASM caches when
    the with block exits."""

    def test_caches_not_cleared_by_default(self):
        # Populate the H cache by running a small propagation.
        E = np.ones((64, 64), dtype=np.complex128)
        lm.angular_spectrum_propagate(E, z=1e-3,
                                       wavelength=1.55e-6, dx=5e-6)
        from lumenairy.propagators.propagation import _H_CACHE
        n_before = len(_H_CACHE)
        assert n_before >= 1

        with lumenairy_context(complex_dtype=np.complex128):
            # set_default_complex_dtype clears the H cache.  But we
            # set it to the same value so the wipe is real but the
            # subsequent population restores it.
            lm.angular_spectrum_propagate(
                E, z=2e-3, wavelength=1.55e-6, dx=5e-6)

        # No clear_caches_on_exit -> H cache retains its entries
        assert len(_H_CACHE) >= 1

    def test_caches_cleared_when_flagged(self):
        E = np.ones((64, 64), dtype=np.complex128)
        lm.angular_spectrum_propagate(E, z=1e-3,
                                       wavelength=1.55e-6, dx=5e-6)
        from lumenairy.propagators.propagation import _H_CACHE

        with lumenairy_context(clear_caches_on_exit=True):
            lm.angular_spectrum_propagate(
                E, z=2e-3, wavelength=1.55e-6, dx=5e-6)
            assert len(_H_CACHE) >= 1

        # With clear_caches_on_exit=True the H cache is empty after exit
        assert len(_H_CACHE) == 0


# ============================================================================
# Source-factory dtype kwarg
# ============================================================================

class TestFactoryDtypeKwarg:
    """Every source factory should honour ``dtype=`` and fall back to
    the library default when not given."""

    def test_gaussian_default_follows_global(self):
        prior = get_default_complex_dtype()
        try:
            set_default_complex_dtype(np.complex64)
            E, _, _ = lm.create_gaussian_beam(
                N=64, dx=5e-6, wavelength=1.55e-6, w0=(10e-6) * np.sqrt(2))
            assert E.dtype == np.dtype(np.complex64)
        finally:
            set_default_complex_dtype(prior)

    def test_gaussian_explicit_overrides_global(self):
        prior = get_default_complex_dtype()
        try:
            set_default_complex_dtype(np.complex64)
            E, _, _ = lm.create_gaussian_beam(
                N=64, dx=5e-6, wavelength=1.55e-6, w0=(10e-6) * np.sqrt(2),
                dtype=np.complex128)
            assert E.dtype == np.dtype(np.complex128)
        finally:
            set_default_complex_dtype(prior)

    def test_hermite_gauss_dtype_kwarg(self):
        E, _, _ = lm.create_hermite_gauss(
            N=64, dx=5e-6, w0=10e-6, wavelength=1.55e-6,
            m=1, n=0, dtype=np.complex64)
        assert E.dtype == np.dtype(np.complex64)

    def test_laguerre_gauss_dtype_kwarg(self):
        E, _, _ = lm.create_laguerre_gauss(
            N=64, dx=5e-6, w0=10e-6, wavelength=1.55e-6,
            p=0, l=1, dtype=np.complex64)
        assert E.dtype == np.dtype(np.complex64)

    def test_tilted_plane_wave_dtype_kwarg(self):
        E, _, _ = lm.create_tilted_plane_wave(
            N=64, dx=5e-6, wavelength=1.55e-6,
            angle_x=1e-3, dtype=np.complex64)
        assert E.dtype == np.dtype(np.complex64)

    def test_point_source_dtype_kwarg(self):
        E, _, _ = lm.create_point_source(
            N=64, dx=5e-6, wavelength=1.55e-6,
            z0=-0.1, dtype=np.complex64)
        assert E.dtype == np.dtype(np.complex64)

    def test_top_hat_dtype_kwarg(self):
        E, _, _ = lm.create_top_hat_beam(
            N=64, dx=5e-6, wavelength=1.55e-6,
            diameter=100e-6, dtype=np.complex64)
        assert E.dtype == np.dtype(np.complex64)

    def test_annular_dtype_kwarg(self):
        E, _, _ = lm.create_annular_beam(
            N=64, dx=5e-6, wavelength=1.55e-6,
            outer_diameter=200e-6, inner_diameter=100e-6,
            dtype=np.complex64)
        assert E.dtype == np.dtype(np.complex64)

    def test_fiber_mode_dtype_kwarg(self):
        E, _, _ = lm.create_fiber_mode(
            N=64, dx=5e-6, wavelength=1.55e-6,
            mode_field_diameter=10e-6, dtype=np.complex64)
        assert E.dtype == np.dtype(np.complex64)

    def test_bessel_dtype_kwarg(self):
        E, _, _ = lm.create_bessel_beam(
            N=64, dx=5e-6, wavelength=1.55e-6,
            cone_angle=np.deg2rad(0.5), dtype=np.complex64)
        assert E.dtype == np.dtype(np.complex64)

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="complex64 or np.complex128"):
            lm.create_gaussian_beam(
                N=64, dx=5e-6, wavelength=1.55e-6, w0=(10e-6) * np.sqrt(2),
                dtype=np.float32)

    def test_factory_inside_context_manager(self):
        """A factory called inside ``lumenairy_context(complex_dtype=...)``
        with no explicit ``dtype=`` should pick up the contextual default."""
        prior = get_default_complex_dtype()
        with lumenairy_context(complex_dtype=np.complex64):
            E, _, _ = lm.create_gaussian_beam(
                N=64, dx=5e-6, wavelength=1.55e-6, w0=(10e-6) * np.sqrt(2))
            assert E.dtype == np.dtype(np.complex64)
        # After the with: factory falls back to whatever was outside
        E_after, _, _ = lm.create_gaussian_beam(
            N=64, dx=5e-6, wavelength=1.55e-6, w0=(10e-6) * np.sqrt(2))
        assert E_after.dtype == np.dtype(prior)


# ============================================================================
# Atexit hook
# ============================================================================

class TestAtexitHook:
    """The atexit handler should be installed and idempotent."""

    def test_install_is_idempotent(self):
        from lumenairy._context import (
            _ATEXIT_INSTALLED,
            install_atexit_restore,
        )
        # Already installed at import time
        assert _ATEXIT_INSTALLED
        # Calling again is a no-op
        install_atexit_restore()
        from lumenairy._context import _ATEXIT_INSTALLED as still_installed
        assert still_installed
