"""Roadmap R1 (B1 + B5): pointwise cos-grid cache + structured-grid interp.

The pointwise 2-D obliquity path (``surface_model='displaced'`` with
``displaced_obliquity='pointwise'`` on a decentered / tilted / freeform
element) traces a 2-D ray grid and interpolates the per-surface obliquity
cosines onto the field grid.  Two deferred perf/memory items land here:

* **B5 (structured-grid interp):** the launch fan is a STRUCTURED grid, so the
  smooth launch->crossing map is inverted by a few Newton steps (evaluating the
  crossing grids + their gradients with ``map_coordinates``) and the cos grids
  are sampled there -- O(N^2) direct, replacing the scattered QHull-Delaunay
  ``LinearNDInterpolator``.  The legacy Delaunay path (``interp_method=
  'delaunay'``) is byte-identical to v5.27.0 and is the **oracle** the
  structured backend is validated against.

* **B1 (cos-grid cache):** the cos-grid is FIELD-INDEPENDENT given the
  prescription + conjugate + wavelength + grid, so a design-iteration loop that
  only moves the field re-uses it.  It is an N^2-scale cache (~16 MB @ N=1024)
  and therefore ships on the shared :class:`ByteBudgetedLRU` **off by default**
  (``max_bytes=0``); the caller opts in with
  :func:`set_pointwise_cos_grid_cache_budget`.  Pure memoization -> a warm hit
  returns the SAME arrays the cold trace produced (byte-identical output).

Validation (all lumenairy-only, no Zemax):
  (B5) structured matches the Delaunay oracle to tolerance on the symmetric
       limit and on M2/M3-class decentered / tilted singlets; measured cold
       speedup; bad-method error.
  (B1) default-off byte-identical; opt-in hit reproduces the cold result;
       byte-budgeted (never exceeds; LRU); complete key (no stale reuse across
       N / wavelength / decenter / conjugate); registered (clear_asm_caches +
       clear_pointwise_cos_grid_cache drain it); introspectable (cache_report);
       budget get/set API.

Author: Andrew Traverso -- roadmap R1
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.cache import cache_report, clear_all_byte_budgeted_caches, set_cache_budget
from lumenairy.elements._lens_real import (
    _DISPLACED_COS_GRID_CACHE,
    _build_displaced_cos_grid,
    _displaced_cos_grid_key,
    _get_displaced_cos_grid,
)
from lumenairy.memory import available_memory_bytes

# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_R1A': lambda wl: 1.5168}
_WL = 1.31e-6
_MB = 1024 * 1024
_RMAX = 5e-3
_THICK = [5e-3]


# ---------------------------------------------------------------------------
# Prescriptions
# ---------------------------------------------------------------------------
def _surfaces(dec=(0.0, 0.0), tilt=(0.0, 0.0), sag_callable=None):
    s0 = {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
          'glass_after': '_R1A', 'semi_diameter': 6e-3,
          'decenter': dec, 'tilt': tilt}
    if sag_callable is not None:
        s0['sag_callable'] = sag_callable
        s0['radius'] = float('inf')
    return [s0,
            {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': '_R1A',
             'glass_after': 'air', 'semi_diameter': 6e-3}]


def _presc(dec=(0.0, 0.0), tilt=(0.0, 0.0), sag_callable=None):
    return {'wavelength': _WL, 'aperture_diameter': 10e-3,
            'surfaces': _surfaces(dec, tilt, sag_callable),
            'thicknesses': [5e-3], 'stop_index': 0}


def _gauss(N, dx, w0=3e-3):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _disp_pointwise(E0, presc, dx, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens(
            E0, prescription=presc, wavelength=_WL, dx=dx,
            surface_model='displaced', displaced_obliquity='pointwise', **kw))


def _cos_relerr(g_a, g_b):
    """Max over surfaces of the relative-L2 diff between two cos-grid lists."""
    out = 0.0
    for (ci_a, co_a), (ci_b, co_b) in zip(g_a, g_b):
        out = max(out, np.linalg.norm(ci_a - ci_b) / np.linalg.norm(ci_b))
        out = max(out, np.linalg.norm(co_a - co_b) / np.linalg.norm(co_b))
    return out


def _cos_absdiff(g_a, g_b):
    return max(max(float(np.max(np.abs(ci_a - ci_b))),
                   float(np.max(np.abs(co_a - co_b))))
               for (ci_a, co_a), (ci_b, co_b) in zip(g_a, g_b))


@pytest.fixture(autouse=True)
def _isolate_cache():
    """Every test starts and ends with the cos-grid cache OFF + drained and
    the global budget at its default, so the default-off invariant is never
    leaked to another test file."""
    la.set_pointwise_cos_grid_cache_budget(0)
    clear_all_byte_budgeted_caches()
    set_cache_budget(None)
    yield
    la.set_pointwise_cos_grid_cache_budget(0)
    clear_all_byte_budgeted_caches()
    set_cache_budget(None)


# ===========================================================================
# B5 -- structured-grid interp matches the Delaunay oracle (byte-id to tol)
# ===========================================================================
@pytest.mark.parametrize('label,dec,tilt', [
    ('symmetric', (0.0, 0.0), (0.0, 0.0)),
    ('decenter_0.5mm', (0.5e-3, 0.0), (0.0, 0.0)),
    ('decenter_0.8mm', (0.8e-3, 0.0), (0.0, 0.0)),
    ('decenter_xy', (0.4e-3, -0.3e-3), (0.0, 0.0)),
    ('tilt_3.5mrad', (0.0, 0.0), (3.491e-3, 0.0)),
])
def test_structured_matches_delaunay_oracle(label, dec, tilt):
    """The B5 structured backend reproduces the legacy Delaunay result (the
    v5.27.0 golden) to tolerance on the symmetric limit AND on M2/M3-class
    decentered / tilted singlets -- the cos grids are near-unity with an
    aperture-scale variation of ~3e-4, and the two backends agree on that
    variation to ~1e-8 absolute (well under the model's obliquity tol)."""
    surf = _surfaces(dec, tilt)
    for N, dx in [(384, 10e-6), (768, 8e-6)]:
        d = _build_displaced_cos_grid(surf, _THICK, _WL, _RMAX, N, N, dx, dx,
                                      dir_fn=None, interp_method='delaunay')
        s = _build_displaced_cos_grid(surf, _THICK, _WL, _RMAX, N, N, dx, dx,
                                      dir_fn=None, interp_method='structured')
        assert _cos_absdiff(s, d) < 1e-6, (label, N, _cos_absdiff(s, d))
        assert _cos_relerr(s, d) < 1e-5, (label, N, _cos_relerr(s, d))


def test_structured_matches_delaunay_with_conjugate():
    """The structured/Delaunay agreement holds for a DIVERGING-source congruence
    (scalar conjugate -> a non-axial launch fan), not just collimated."""
    from lumenairy.elements._lens_real import _displaced_carrier_dir_fn
    surf = _surfaces(dec=(0.3e-3, 0.0))
    N, dx = 512, 8e-6
    E0 = _gauss(N, dx)
    dir_fn = _displaced_carrier_dir_fn(300e-3, E0, _WL, dx, dx, N, N)
    d = _build_displaced_cos_grid(surf, _THICK, _WL, _RMAX, N, N, dx, dx,
                                  dir_fn=dir_fn, interp_method='delaunay')
    s = _build_displaced_cos_grid(surf, _THICK, _WL, _RMAX, N, N, dx, dx,
                                  dir_fn=dir_fn, interp_method='structured')
    assert _cos_absdiff(s, d) < 1e-6, _cos_absdiff(s, d)


def test_structured_cold_speedup():
    """The structured cold path is measurably faster than the Delaunay cold
    path (the QHull triangulation build is the ~3.9 s K3 cost).  Best-of-N,
    generous bound -- not a brittle absolute threshold.  Measured ~5-6x."""
    if available_memory_bytes() < 2 * (1 << 30):
        pytest.skip("insufficient free RAM for the timed interp comparison")
    surf = _surfaces(dec=(0.5e-3, 0.0))
    N, dx = 512, 8e-6

    def _t(method, reps=3):
        fn = lambda: _build_displaced_cos_grid(  # noqa: E731
            surf, _THICK, _WL, _RMAX, N, N, dx, dx, dir_fn=None,
            interp_method=method)
        fn()  # warm import/JIT
        best = np.inf
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    t_struct = _t('structured')
    t_delau = _t('delaunay')
    assert t_struct < 0.7 * t_delau, (
        f"structured not faster: {t_struct*1e3:.0f} ms vs delaunay "
        f"{t_delau*1e3:.0f} ms (ratio {t_struct / t_delau:.2f})")


def test_bad_interp_method_raises():
    surf = _surfaces()
    with pytest.raises(ValueError, match='interp_method'):
        _build_displaced_cos_grid(surf, _THICK, _WL, _RMAX, 128, 128, 1e-5,
                                  1e-5, dir_fn=None, interp_method='bogus')


# ===========================================================================
# B1 -- cache defaults OFF (opt-in) and is byte-identical when enabled
# ===========================================================================
def test_cache_off_by_default():
    """The N^2-scale cos-grid cache ships DISABLED (Section 0 opt-in): a
    pointwise call stores nothing and the budget getter reports 0."""
    assert la.get_pointwise_cos_grid_cache_budget() == 0
    surf = _surfaces(dec=(0.4e-3, 0.0))
    N, dx = 256, 10e-6
    before = _DISPLACED_COS_GRID_CACHE.report()['misses']
    _get_displaced_cos_grid(surf, _THICK, _WL, _RMAX, N, N, dx, dx,
                            dir_fn=None, conjugate=None, n_launch=65)
    _get_displaced_cos_grid(surf, _THICK, _WL, _RMAX, N, N, dx, dx,
                            dir_fn=None, conjugate=None, n_launch=65)
    assert len(_DISPLACED_COS_GRID_CACHE) == 0     # nothing retained
    # both calls missed (no hit served) -- the disabled cache never stores
    assert _DISPLACED_COS_GRID_CACHE.report()['misses'] >= before + 2


def test_opt_in_hit_reproduces_cold_result_exactly():
    """Enabled, a repeat call HITS and the served grids are the SAME arrays the
    cold trace produced -- pure memoization (byte-identical output)."""
    la.set_pointwise_cos_grid_cache_budget(None)   # bound only by global budget
    surf = _surfaces(dec=(0.4e-3, 0.0))
    N, dx = 256, 10e-6
    kw = dict(dir_fn=None, conjugate=None, n_launch=65)
    r0 = _DISPLACED_COS_GRID_CACHE.report()
    cold = _get_displaced_cos_grid(surf, _THICK, _WL, _RMAX, N, N, dx, dx, **kw)
    assert len(_DISPLACED_COS_GRID_CACHE) == 1
    warm = _get_displaced_cos_grid(surf, _THICK, _WL, _RMAX, N, N, dx, dx, **kw)
    r1 = _DISPLACED_COS_GRID_CACHE.report()
    assert r1['hits'] == r0['hits'] + 1            # served from cache
    # SAME objects (memoization), hence bit-identical downstream.
    for (ci_c, co_c), (ci_w, co_w) in zip(cold, warm):
        assert ci_w is ci_c and co_w is co_c


def test_apply_real_lens_cache_on_equals_off():
    """apply_real_lens pointwise output with the cache ON == with it OFF, to
    tolerance (memoization does not perturb the field)."""
    N, dx = 384, 10e-6
    E0 = _gauss(N, dx)
    p = _presc(dec=(0.5e-3, 0.0))
    off = _disp_pointwise(E0, p, dx)               # cache disabled (default)
    la.set_pointwise_cos_grid_cache_budget(None)
    cold = _disp_pointwise(E0, p, dx)              # populates
    warm = _disp_pointwise(E0, p, dx)              # served from cache
    scale = float(np.max(np.abs(off)))
    assert np.max(np.abs(cold - off)) <= 1e-10 * scale
    assert np.max(np.abs(warm - off)) <= 1e-10 * scale
    # the second apply hit the cache
    assert _DISPLACED_COS_GRID_CACHE.report()['hits'] >= 1


# ===========================================================================
# B1 -- byte-budgeted (never exceeds; LRU) + registered + introspectable
# ===========================================================================
def test_cache_byte_budgeted_never_exceeds_and_evicts_lru():
    """With a local budget that fits ONE cos-grid entry, feeding distinct
    configs never exceeds the budget and evicts least-recently-used."""
    N, dx = 384, 10e-6
    # one entry = 2 surfaces x (cin+cout) x N^2 x 8 B ~= 4.7 MB at N=384.
    one = None
    la.set_pointwise_cos_grid_cache_budget(None)
    g = _get_displaced_cos_grid(_surfaces(dec=(0.1e-3, 0.0)), _THICK, _WL,
                                _RMAX, N, N, dx, dx, dir_fn=None,
                                conjugate=None, n_launch=65)
    one = sum(a.nbytes for pair in g for a in pair)
    assert one > 3 * _MB                            # genuinely N^2-scale
    budget_bytes = int(one * 1.5)                   # room for exactly one
    la.set_pointwise_cos_grid_cache_budget(budget_bytes / _MB)
    clear_all_byte_budgeted_caches()
    for wl_scale in (1.0, 1.05, 1.10):              # 3 distinct keys (wavelength)
        _get_displaced_cos_grid(_surfaces(dec=(0.1e-3, 0.0)), _THICK,
                                _WL * wl_scale, _RMAX, N, N, dx, dx,
                                dir_fn=None, conjugate=None, n_launch=65)
        assert _DISPLACED_COS_GRID_CACHE.retained_bytes <= budget_bytes
    assert len(_DISPLACED_COS_GRID_CACHE) == 1      # only the hot one survives
    assert _DISPLACED_COS_GRID_CACHE.evictions > 0


def test_cache_lru_keeps_reused_entry():
    """A design loop that re-uses one config keeps its hot entry (LRU, not
    FIFO): with a TWO-entry budget, touching the hot entry before the next
    insert evicts the OTHER (least-recently-used) entry, not the hot one."""
    N, dx = 384, 10e-6
    la.set_pointwise_cos_grid_cache_budget(None)
    hot_surf = _surfaces(dec=(0.2e-3, 0.0))

    def _key(wl_scale):
        return _displaced_cos_grid_key(hot_surf, _THICK, _WL * wl_scale, _RMAX,
                                       None, N, N, dx, dx, 65, 384, 'structured')

    def _build(wl_scale):
        return _get_displaced_cos_grid(hot_surf, _THICK, _WL * wl_scale, _RMAX,
                                       N, N, dx, dx, dir_fn=None,
                                       conjugate=None, n_launch=65)

    g = _build(1.0)
    one = sum(a.nbytes for pair in g for a in pair)
    la.set_pointwise_cos_grid_cache_budget(int(one * 2.5) / _MB)   # room for 2
    clear_all_byte_budgeted_caches()
    _build(1.0)                                     # hot (entry 1)
    _build(1.05)                                    # coldA (entry 2) -- 2 fit
    assert _DISPLACED_COS_GRID_CACHE.get(_key(1.0)) is not None    # touch hot -> MRU
    _build(1.10)                                    # coldB -> evict LRU (coldA)
    assert _key(1.0) in _DISPLACED_COS_GRID_CACHE   # hot survived the churn
    assert _key(1.05) not in _DISPLACED_COS_GRID_CACHE   # LRU evicted
    assert _DISPLACED_COS_GRID_CACHE.retained_bytes <= int(one * 2.5)


def test_cache_registered_and_introspectable():
    """cache_report() shows the cos-grid cache footprint; clear_asm_caches()
    and clear_pointwise_cos_grid_cache() both drain it."""
    la.set_pointwise_cos_grid_cache_budget(None)
    N, dx = 256, 10e-6
    _get_displaced_cos_grid(_surfaces(dec=(0.3e-3, 0.0)), _THICK, _WL, _RMAX,
                            N, N, dx, dx, dir_fn=None, conjugate=None,
                            n_launch=65)
    rep = cache_report()
    assert 'displaced_cos_grid' in rep['caches']
    assert rep['caches']['displaced_cos_grid']['entries'] == 1
    assert rep['caches']['displaced_cos_grid']['retained_bytes'] > 0
    # clear_asm_caches walks the byte-budgeted registry -> drains us
    la.clear_asm_caches()
    assert len(_DISPLACED_COS_GRID_CACHE) == 0
    assert cache_report()['caches']['displaced_cos_grid']['retained_bytes'] == 0
    # the dedicated clearer also drains (and does not change the budget)
    _get_displaced_cos_grid(_surfaces(dec=(0.3e-3, 0.0)), _THICK, _WL, _RMAX,
                            N, N, dx, dx, dir_fn=None, conjugate=None,
                            n_launch=65)
    assert len(_DISPLACED_COS_GRID_CACHE) == 1
    la.clear_pointwise_cos_grid_cache()
    assert len(_DISPLACED_COS_GRID_CACHE) == 0
    assert la.get_pointwise_cos_grid_cache_budget() is None    # still enabled


# ===========================================================================
# B1 -- complete key: no stale reuse across the value's determinants
# ===========================================================================
def test_complete_key_no_stale_reuse():
    """Distinct N / wavelength / decenter / conjugate all key distinctly (the
    v5.26.0 stale-key lesson) -- each variation misses, none returns a stale
    entry from a different config."""
    N, dx = 256, 10e-6
    base = dict(surfaces=_surfaces(dec=(0.4e-3, 0.0)), thicknesses=_THICK,
                wavelength=_WL, r_max=_RMAX, conjugate=None,
                Nx=N, Ny=N, dx=dx, dy=dx, n_launch=65, n_coarse=384,
                interp_method='structured')
    k0 = _displaced_cos_grid_key(**base)
    # each single-field change must produce a DIFFERENT key
    assert _displaced_cos_grid_key(**{**base, 'Nx': N + 2, 'Ny': N + 2}) != k0
    assert _displaced_cos_grid_key(**{**base, 'wavelength': _WL * 1.01}) != k0
    assert _displaced_cos_grid_key(**{**base, 'dx': dx * 1.01, 'dy': dx * 1.01}) != k0
    assert _displaced_cos_grid_key(**{**base, 'conjugate': 0.3}) != k0
    assert _displaced_cos_grid_key(
        **{**base, 'surfaces': _surfaces(dec=(0.5e-3, 0.0))}) != k0
    assert _displaced_cos_grid_key(
        **{**base, 'surfaces': _surfaces(tilt=(1e-3, 0.0))}) != k0
    assert _displaced_cos_grid_key(**{**base, 'interp_method': 'delaunay'}) != k0
    # the same config keys identically (a real hit, not a stale-key collision)
    assert _displaced_cos_grid_key(**base) == k0
    # field-dependent congruences ('auto' / ndarray) are NOT cacheable
    assert _displaced_cos_grid_key(**{**base, 'conjugate': 'auto'}) is None
    assert _displaced_cos_grid_key(
        **{**base, 'conjugate': np.zeros((N, N))}) is None
    # +-inf conjugate folds to the collimated (None) key
    assert _displaced_cos_grid_key(**{**base, 'conjugate': np.inf}) == k0


def test_distinct_configs_distinct_values():
    """End-to-end: two different decenters cached simultaneously return their
    OWN grids, never each other's (no stale reuse)."""
    la.set_pointwise_cos_grid_cache_budget(None)
    N, dx = 256, 10e-6
    kw = dict(dir_fn=None, conjugate=None, n_launch=65)
    ga = _get_displaced_cos_grid(_surfaces(dec=(0.2e-3, 0.0)), _THICK, _WL,
                                 _RMAX, N, N, dx, dx, **kw)
    gb = _get_displaced_cos_grid(_surfaces(dec=(-0.6e-3, 0.0)), _THICK, _WL,
                                 _RMAX, N, N, dx, dx, **kw)
    assert len(_DISPLACED_COS_GRID_CACHE) == 2
    ga2 = _get_displaced_cos_grid(_surfaces(dec=(0.2e-3, 0.0)), _THICK, _WL,
                                  _RMAX, N, N, dx, dx, **kw)
    for (a, _), (a2, _) in zip(ga, ga2):
        assert a2 is a                             # served the RIGHT entry
    assert _cos_absdiff(ga, gb) > 1e-6             # genuinely different values


# ===========================================================================
# B1 -- budget get/set API
# ===========================================================================
def test_budget_api_roundtrip_and_validation():
    la.set_pointwise_cos_grid_cache_budget(64)
    assert la.get_pointwise_cos_grid_cache_budget() == 64 * _MB
    la.set_pointwise_cos_grid_cache_budget(None)
    assert la.get_pointwise_cos_grid_cache_budget() is None
    la.set_pointwise_cos_grid_cache_budget(0)
    assert la.get_pointwise_cos_grid_cache_budget() == 0
    with pytest.raises(ValueError):
        la.set_pointwise_cos_grid_cache_budget(-1)


def test_disable_clears_the_cache():
    """Setting the budget to 0 both disables AND drains the cache."""
    la.set_pointwise_cos_grid_cache_budget(None)
    N, dx = 256, 10e-6
    _get_displaced_cos_grid(_surfaces(dec=(0.3e-3, 0.0)), _THICK, _WL, _RMAX,
                            N, N, dx, dx, dir_fn=None, conjugate=None,
                            n_launch=65)
    assert len(_DISPLACED_COS_GRID_CACHE) == 1
    la.set_pointwise_cos_grid_cache_budget(0)      # disable
    assert len(_DISPLACED_COS_GRID_CACHE) == 0
    # and a subsequent call stores nothing while disabled
    _get_displaced_cos_grid(_surfaces(dec=(0.3e-3, 0.0)), _THICK, _WL, _RMAX,
                            N, N, dx, dx, dir_fn=None, conjugate=None,
                            n_launch=65)
    assert len(_DISPLACED_COS_GRID_CACHE) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
