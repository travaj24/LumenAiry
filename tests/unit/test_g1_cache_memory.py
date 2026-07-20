"""G1 item 4: cache-memory audit regression guards.

Covers the caches the real-lens capability batch added or newly exercises:
  * PreparedTracedLens.screen -- per-object N^2*16-byte payload + the new
    .release()/.clear() explicit-free API (G1);
  * fga _COARSE_CACHE -- FIFO bound (2) + eviction + cache-registry enrollment
    (clear_asm_caches / clear_all_registered_caches drain it, keeping the
    v4.16.1 walker green).
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements._lens_traced import (
    PreparedTracedLens,
    prepare_real_lens_traced,
)
from lumenairy.glass import GLASS_REGISTRY
from lumenairy.propagators import fga as _fga

_WL = 1.31e-6
GLASS_REGISTRY['_G1CACHE'] = lambda wl: 1.5168


def _singlet(semi=6e-3):
    return {'wavelength': _WL, 'aperture_diameter': 2 * semi,
            'surfaces': [
                {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
                 'glass_after': '_G1CACHE', 'semi_diameter': semi},
                {'radius': -51.68e-3, 'thickness': 0.0,
                 'glass_before': '_G1CACHE', 'glass_after': 'air',
                 'semi_diameter': semi}],
            'thicknesses': [5e-3], 'stop_index': 0}


def _gauss(N, dx, w0):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


# ---------------------------------------------------------------------------
# PreparedTracedLens: footprint + explicit-release API (G1 addition).
# ---------------------------------------------------------------------------
def test_prepared_screen_footprint_is_n2_complex128():
    """The retained payload is exactly one (N, N) complex128 screen = N^2*16
    bytes (1 MB at N=256; 64 MB at N=2048, 256 MB at N=4096); no hidden
    per-call retention."""
    N = 256
    dx = 16e-3 / N
    prep = prepare_real_lens_traced(prescription=_singlet(), wavelength=_WL,
                                    dx=dx, N=N, carrier=None,
                                    min_coarse_samples_per_aperture=0)
    assert prep.screen.dtype == np.complex128
    assert prep.screen.shape == (N, N)
    assert prep.screen.nbytes == N * N * 16


def test_prepared_release_frees_screen_and_blocks_calls():
    """.release() drops the screen and a subsequent call raises; .clear() is an
    alias; release is idempotent."""
    N = 256
    dx = 16e-3 / N
    prep = prepare_real_lens_traced(prescription=_singlet(), wavelength=_WL,
                                    dx=dx, N=N, carrier=None,
                                    min_coarse_samples_per_aperture=0)
    E = _gauss(N, dx, 1e-3)
    out = prep(E)                       # works before release
    assert out.shape == (N, N)
    prep.release()
    assert prep.screen is None
    with pytest.raises(RuntimeError):
        prep(E)
    prep.release()                      # idempotent
    # .clear is the same operation
    prep2 = prepare_real_lens_traced(prescription=_singlet(), wavelength=_WL,
                                     dx=dx, N=N, carrier=None,
                                     min_coarse_samples_per_aperture=0)
    assert PreparedTracedLens.clear is PreparedTracedLens.release
    prep2.clear()
    with pytest.raises(RuntimeError):
        prep2(E)


# ---------------------------------------------------------------------------
# fga _COARSE_CACHE: FIFO bound + eviction + registry drain.
# ---------------------------------------------------------------------------
def test_coarse_cache_fifo_bound_and_eviction():
    """The opt-in coarse-trace cache is FIFO-bounded at 2: pushing >2 distinct
    geometry keys never grows it past the bound."""
    import warnings
    _fga.clear_coarse_trace_cache()
    N = 256
    dx = 12e-3 / N
    E = _gauss(N, dx, 1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for th in (5.0e-3, 5.5e-3, 6.0e-3, 6.5e-3):     # 4 distinct keys
            p = _singlet()
            p['surfaces'][0]['thickness'] = th
            p['thicknesses'] = [th]
            _fga.apply_real_lens_fga(E, prescription=p, wavelength=_WL, dx=dx,
                                     coarse_stride=3, cache_trace=True,
                                     mem_budget_mb=2000.0)
            assert len(_fga._COARSE_CACHE) <= _fga._COARSE_CACHE_MAX
    assert _fga._COARSE_CACHE_MAX == 2
    assert len(_fga._COARSE_CACHE) == 2
    _fga.clear_coarse_trace_cache()
    assert len(_fga._COARSE_CACHE) == 0


def test_coarse_cache_drained_by_registry_walker():
    """clear_asm_caches (the v4.16.1 registry walker) drains the coarse-trace
    cache -- the enrollment must stay green."""
    import warnings

    from lumenairy._cache_registry import list_registered_cache_clearers
    assert 'fga_coarse_trace' in list_registered_cache_clearers()
    _fga.clear_coarse_trace_cache()
    N = 256
    dx = 12e-3 / N
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _fga.apply_real_lens_fga(_gauss(N, dx, 1e-3), prescription=_singlet(),
                                 wavelength=_WL, dx=dx, coarse_stride=3,
                                 cache_trace=True, mem_budget_mb=2000.0)
    assert len(_fga._COARSE_CACHE) >= 1
    la.clear_asm_caches()                          # walks the registry
    assert len(_fga._COARSE_CACHE) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
