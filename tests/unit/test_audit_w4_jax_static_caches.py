"""Wave-4 audit fixes -- JAX/prepared static-cache lifecycle (v5.17.0 deep
audit P3-16, P3-28, P3-32).

Pins, for each of the three formerly-unbounded caches:

* **growth bounded** -- sweeping 3x the LRU bound leaves ``len == bound``
  (pre-fix these grew 1:1 with the sweep forever);
* **clearer drains** -- the module caches via their enrolled registry names
  (``eme_jax_frozen_ops`` / ``pmm_jax_twod_static`` ->
  ``clear_all_registered_caches`` / ``clear_asm_caches``), the instance
  caches via the new ``_PreparedPMMStack.clear_cache()``;
* **cached-value correctness** -- a cache hit returns the SAME object, a
  post-eviction (and post-clear) recompute is BYTE-identical, and eviction
  never mutates arrays a caller still holds (values are returned by
  reference and only the dict's reference is dropped).

The prepared-stack caches (P3-32) are INSTANCE attributes: they are
deliberately NOT enrolled in ``lumenairy._cache_registry`` (the v4.16.1
enrollment meta-pin walker discovers only module-level ``_CACHE``
assignments, and the registry contract covers process-global caches);
``clear_cache()`` / dropping the prepared object is the release path.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy._cache_registry import (
    clear_all_registered_caches,
    list_registered_cache_clearers,
)
from lumenairy.elements.eme import _jax_modes as jm
from lumenairy.elements.pmm import _jax_twod as jt
from lumenairy.elements.pmm.stack import PMMStack, _PreparedPMMStack

_P, _WL = 0.8e-6, 0.633e-6


@pytest.fixture(autouse=True)
def _isolate_module_caches():
    """Snapshot + restore the two module caches so this file neither sees
    nor leaks other tests' entries."""
    saved_frozen = dict(jm._FROZEN_CACHE)
    saved_static = dict(jt._STATIC_CACHE)
    jm._FROZEN_CACHE.clear()
    jt._STATIC_CACHE.clear()
    yield
    jm._FROZEN_CACHE.clear()
    jm._FROZEN_CACHE.update(saved_frozen)
    jt._STATIC_CACHE.clear()
    jt._STATIC_CACHE.update(saved_static)


# =========================================================================== #
# P3-16 -- eme _FROZEN_CACHE
# =========================================================================== #

def test_eme_frozen_cache_bounded_and_byte_identical():
    L0 = jm._frozen_helmholtz_L(12, 12, 1.0, 1.0, 0.0, 0.0)
    b0 = L0.tobytes()
    # hit: same object, no growth
    assert jm._frozen_helmholtz_L(12, 12, 1.0, 1.0, 0.0, 0.0) is L0
    assert len(jm._FROZEN_CACHE) == 1
    # 3x the bound of distinct k-points -> len == bound (pre-fix: 25 entries)
    for kx0 in np.linspace(0.1, 3.0, 3 * jm._FROZEN_CACHE_SIZE):
        jm._frozen_helmholtz_L(12, 12, 1.0, 1.0, float(kx0), 0.0)
    assert len(jm._FROZEN_CACHE) == jm._FROZEN_CACHE_SIZE
    # eviction dropped the reference only -- the caller-held array is intact
    assert L0.tobytes() == b0
    # post-eviction recompute: a NEW array, byte-identical
    L1 = jm._frozen_helmholtz_L(12, 12, 1.0, 1.0, 0.0, 0.0)
    assert L1 is not L0
    assert L1.tobytes() == b0


def test_eme_frozen_cache_yee_branch_bounded_and_byte_identical():
    y0 = jm._frozen_yee_dense(8, 8, 1.0, 1.0, 0.0, 0.0)
    yb = tuple(a.tobytes() for a in y0)
    assert jm._frozen_yee_dense(8, 8, 1.0, 1.0, 0.0, 0.0) is y0
    for kx0 in np.linspace(0.1, 2.0, 3 * jm._FROZEN_CACHE_SIZE):
        jm._frozen_yee_dense(8, 8, 1.0, 1.0, float(kx0), 0.0)
    assert len(jm._FROZEN_CACHE) == jm._FROZEN_CACHE_SIZE
    y1 = jm._frozen_yee_dense(8, 8, 1.0, 1.0, 0.0, 0.0)
    assert y1 is not y0
    assert tuple(a.tobytes() for a in y1) == yb


def test_eme_frozen_cache_lru_recency_preserves_hot_entry():
    """A re-touched key survives an insert burst that evicts colder keys."""
    hot = jm._frozen_helmholtz_L(12, 12, 1.0, 1.0, 0.0, 0.0)
    for kx0 in np.linspace(0.1, 1.0, jm._FROZEN_CACHE_SIZE - 1):
        jm._frozen_helmholtz_L(12, 12, 1.0, 1.0, float(kx0), 0.0)
    # cache is now full; touch the hot key, then insert one more
    assert jm._frozen_helmholtz_L(12, 12, 1.0, 1.0, 0.0, 0.0) is hot
    jm._frozen_helmholtz_L(12, 12, 1.0, 1.0, 2.5, 0.0)
    # the hot key survived (still the SAME object => it was a hit, not a
    # recompute)
    assert jm._frozen_helmholtz_L(12, 12, 1.0, 1.0, 0.0, 0.0) is hot


def test_eme_frozen_cache_registry_drain():
    assert "eme_jax_frozen_ops" in list_registered_cache_clearers()
    jm._frozen_helmholtz_L(12, 12, 1.0, 1.0, 0.3, 0.0)
    assert len(jm._FROZEN_CACHE) > 0
    clear_all_registered_caches()
    assert len(jm._FROZEN_CACHE) == 0
    # the public aggregate entry drains it too
    jm._frozen_helmholtz_L(12, 12, 1.0, 1.0, 0.3, 0.0)
    la.clear_asm_caches()
    assert len(jm._FROZEN_CACHE) == 0


# =========================================================================== #
# P3-28 -- pmm 2-D JAX _STATIC_CACHE
# =========================================================================== #

_PREP_KEYS = ("Tp", "Tpinv", "Gx0F", "Gy0F", "IprojF", "w")


def test_pmm_jax_twod_static_cache_bounded_and_byte_identical():
    e0 = jt._static_prep(1.0, 1.0, 0.2, 0.6, 0.2, 0.8, 3, 1, 1.0, 2)
    eb = {k: e0[k].tobytes() for k in _PREP_KEYS}
    assert jt._static_prep(1.0, 1.0, 0.2, 0.6, 0.2, 0.8, 3, 1, 1.0, 2) is e0
    # 3x the bound of distinct geometries (an x0 sweep -- the audit's
    # geometry-optimization pattern) -> len == bound (pre-fix: 1:1 growth)
    for x0 in np.linspace(0.05, 0.35, 3 * jt._STATIC_CACHE_SIZE):
        jt._static_prep(1.0, 1.0, float(x0), 0.6, 0.2, 0.8, 3, 1, 1.0, 2)
    assert len(jt._STATIC_CACHE) == jt._STATIC_CACHE_SIZE
    # eviction never mutates caller-held arrays
    assert all(e0[k].tobytes() == eb[k] for k in _PREP_KEYS)
    # post-eviction recompute: new dict, byte-identical operators
    e1 = jt._static_prep(1.0, 1.0, 0.2, 0.6, 0.2, 0.8, 3, 1, 1.0, 2)
    assert e1 is not e0
    assert all(e1[k].tobytes() == eb[k] for k in _PREP_KEYS)


def test_pmm_jax_twod_static_cache_cell_branch_bounded():
    lay = np.zeros((8, 8), dtype=np.int64)
    lay[2:6, 2:6] = 1
    c0 = jt._static_prep_cell(1.0, 1.0, lay, 3, 1, 1.0, 2)
    cb = c0["Tp"].tobytes()
    assert jt._static_prep_cell(1.0, 1.0, lay, 3, 1, 1.0, 2) is c0
    # distinct layouts (shifting pillar) blow through 3x the bound
    for s in range(3 * jt._STATIC_CACHE_SIZE):
        lay_s = np.zeros((8 + s // 8, 8), dtype=np.int64)
        lay_s[1:4, 1 + s % 4:5 + s % 4] = 1
        jt._static_prep_cell(1.0, 1.0, lay_s, 3, 1, 1.0, 2)
    assert len(jt._STATIC_CACHE) <= jt._STATIC_CACHE_SIZE
    c1 = jt._static_prep_cell(1.0, 1.0, lay, 3, 1, 1.0, 2)
    assert c1["Tp"].tobytes() == cb


def test_pmm_jax_twod_static_cache_registry_drain():
    assert "pmm_jax_twod_static" in list_registered_cache_clearers()
    jt._static_prep(1.0, 1.0, 0.2, 0.6, 0.2, 0.8, 3, 1, 1.0, 2)
    assert len(jt._STATIC_CACHE) > 0
    clear_all_registered_caches()
    assert len(jt._STATIC_CACHE) == 0


# =========================================================================== #
# P3-32 -- _PreparedPMMStack._eig_cache / _mats_cache (instance caches)
# =========================================================================== #

def _lc(phi):
    no2, ne2 = 1.5 ** 2, 1.7 ** 2
    c, s = np.cos(phi), np.sin(phi)
    M = np.diag([no2, no2, no2]).astype(complex)
    M[0, 0] = ne2 * c * c + no2 * s * s
    M[1, 1] = ne2 * s * s + no2 * c * c
    M[0, 1] = M[1, 0] = (ne2 - no2) * c * s
    return M


def _lc_stack():
    st = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=8)
    st.add_layer(0.10e-6, eps=2.25)
    st.add_layer(0.15e-6, segments=[(0.4, 4.0 + 0.5j), (0.6, "LC")])
    return st


class _TinyPrepared(_PreparedPMMStack):
    """Bound-shrunk subclass so forcing eviction stays seconds-fast; the
    LRU mechanism under test is identical at the default 64/8 bounds."""
    _EIG_CACHE_SIZE = 4
    _MATS_CACHE_SIZE = 2


def test_prepared_stack_default_bounds_and_clear_api():
    prep = _lc_stack().prepare()
    # the audit-fix contract: bounded LRU + a public drain
    assert _PreparedPMMStack._EIG_CACHE_SIZE == 64
    assert _PreparedPMMStack._MATS_CACHE_SIZE == 8
    assert callable(prep.clear_cache)


def test_prepared_stack_caches_bounded_lru():
    st = _lc_stack()
    prep = st.prepare()
    prep.__class__ = _TinyPrepared
    # 3x the eig bound of distinct tensors -> len == bound (pre-fix: 1:1)
    for f in np.linspace(0.05, 1.5, 3 * _TinyPrepared._EIG_CACHE_SIZE):
        prep.solve(wavelength=_WL, materials={"LC": _lc(f)})
    assert len(prep._eig_cache) == _TinyPrepared._EIG_CACHE_SIZE
    # 3x the mats bound of distinct wavelengths -> len == bound
    for wl in np.linspace(0.60e-6, 0.68e-6, 3 * _TinyPrepared._MATS_CACHE_SIZE):
        prep.solve(wavelength=float(wl), materials={"LC": _lc(0.3)})
    assert len(prep._mats_cache) == _TinyPrepared._MATS_CACHE_SIZE


def test_prepared_stack_hit_eviction_and_clear_byte_identical():
    st = _lc_stack()
    ref = st.prepare()
    o0, R0, T0, J0 = ref.solve(wavelength=_WL, materials={"LC": _lc(0.0)})
    bR, bT, bJ = R0.tobytes(), T0.tobytes(), J0.tobytes()
    # cache HIT solve is byte-identical
    _, R1, T1, J1 = ref.solve(wavelength=_WL, materials={"LC": _lc(0.0)})
    assert (R1.tobytes(), T1.tobytes(), J1.tobytes()) == (bR, bT, bJ)

    prep = st.prepare()
    prep.__class__ = _TinyPrepared
    _, Ra, Ta, Ja = prep.solve(wavelength=_WL, materials={"LC": _lc(0.0)})
    # the tiny-bound prepared object matches the default-bound one exactly
    assert (Ra.tobytes(), Ta.tobytes(), Ja.tobytes()) == (bR, bT, bJ)
    # evict the lc(0.0) entry by blowing through the tiny bound...
    for f in np.linspace(0.05, 1.5, 2 * _TinyPrepared._EIG_CACHE_SIZE):
        prep.solve(wavelength=_WL, materials={"LC": _lc(f)})
    # ...post-EVICTION recompute is byte-identical
    _, Rb, Tb, Jb = prep.solve(wavelength=_WL, materials={"LC": _lc(0.0)})
    assert (Rb.tobytes(), Tb.tobytes(), Jb.tobytes()) == (bR, bT, bJ)
    # clear_cache drains BOTH caches; post-CLEAR recompute is byte-identical
    prep.clear_cache()
    assert len(prep._eig_cache) == 0 and len(prep._mats_cache) == 0
    _, Rc, Tc, Jc = prep.solve(wavelength=_WL, materials={"LC": _lc(0.0)})
    assert (Rc.tobytes(), Tc.tobytes(), Jc.tobytes()) == (bR, bT, bJ)


def test_prepared_stack_instance_caches_not_registry_scoped():
    """Documented decision (P3-32): the prepared object's caches are
    INSTANCE attributes -- clear_all_registered_caches must NOT touch a
    live prepared object (its lifetime is the object's), and clear_cache()
    is the supported drain."""
    prep = _lc_stack().prepare()
    prep.solve(wavelength=_WL, materials={"LC": _lc(0.2)})
    n_eig, n_mats = len(prep._eig_cache), len(prep._mats_cache)
    assert n_eig > 0 and n_mats > 0
    clear_all_registered_caches()
    assert len(prep._eig_cache) == n_eig
    assert len(prep._mats_cache) == n_mats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
