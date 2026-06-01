"""v5.6 glass index value memoization (safe immutable-branch cache).

``get_glass_index`` memoizes the result of the IMMUTABLE-catalogue dispatch
branches (the ``__sellmeier__`` / ``__polynomial__`` sentinels and the
refractiveindex-unavailable fallback) keyed on ``(glass_name, wavelength)``.
The cache is consulted only inside those branches (after the entry sentinel is
confirmed), so a name re-registered under a different dispatch can never serve
a stale value, and ``register_fixed_glass`` clears it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy.glass as G
from lumenairy.glass import (
    GLASS_REGISTRY,
    _glass_value_cache,
    get_glass_index,
)
from lumenairy.user_library import register_fixed_glass

# a bundled __sellmeier__-sentinel glass: always routes through the cached
# branch regardless of whether refractiveindex is installed.
_SENTINEL = next(k for k, v in GLASS_REGISTRY.items() if v == "__sellmeier__")


def test_sentinel_glasses_exist():
    sentinels = [k for k, v in GLASS_REGISTRY.items() if v == "__sellmeier__"]
    assert len(sentinels) >= 1


def test_repeated_calls_evaluate_sellmeier_once(monkeypatch):
    _glass_value_cache.clear()
    calls = {"n": 0}
    orig = G._sellmeier_index

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(G, "_sellmeier_index", spy)
    vals = [get_glass_index(_SENTINEL, 600e-9) for _ in range(5)]
    assert calls["n"] == 1                 # evaluated once, then cached
    assert len(set(vals)) == 1             # all identical
    assert vals[0] == orig(600e-9,
                           G.SELLMEIER_COEFFICIENTS[_SENTINEL])  # bit-identical


def test_distinct_wavelengths_recompute(monkeypatch):
    _glass_value_cache.clear()
    calls = {"n": 0}
    orig = G._sellmeier_index
    monkeypatch.setattr(G, "_sellmeier_index",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1)
                                         or orig(*a, **k)))
    get_glass_index(_SENTINEL, 600e-9)     # miss -> compute
    get_glass_index(_SENTINEL, 600e-9)     # hit
    get_glass_index(_SENTINEL, 700e-9)     # different wl -> compute
    assert calls["n"] == 2


def test_value_is_bit_identical_to_uncached():
    _glass_value_cache.clear()
    direct = G._sellmeier_index(587.6e-9, G.SELLMEIER_COEFFICIENTS[_SENTINEL])
    cached = get_glass_index(_SENTINEL, 587.6e-9)
    assert cached == direct


def test_register_fixed_glass_clears_cache():
    """Re-registering a cached sentinel name as a fixed index must NOT serve
    the stale catalogue value."""
    _glass_value_cache.clear()
    name = "_V56_MEMO_TESTGLASS_"
    G.SELLMEIER_COEFFICIENTS[name] = G.SELLMEIER_COEFFICIENTS[_SENTINEL]
    GLASS_REGISTRY[name] = "__sellmeier__"
    try:
        catalogue = get_glass_index(name, 550e-9)
        assert catalogue != 2.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            register_fixed_glass(name, 2.0)        # clears the value cache
        assert get_glass_index(name, 550e-9) == 2.0
    finally:
        GLASS_REGISTRY.pop(name, None)
        G.SELLMEIER_COEFFICIENTS.pop(name, None)
        G._glass_cache.pop(name, None)
        _glass_value_cache.clear()


def test_callable_entry_is_never_cached():
    """A user dispersion callable is invoked on every call (handled before the
    cache, so it is never memoized)."""
    name = "_V56_MEMO_CALLABLE_"
    calls = {"n": 0}

    def disp(wl):
        calls["n"] += 1
        return 1.5

    GLASS_REGISTRY[name] = disp
    try:
        get_glass_index(name, 500e-9)
        get_glass_index(name, 500e-9)
        assert calls["n"] == 2
    finally:
        GLASS_REGISTRY.pop(name, None)
        _glass_value_cache.clear()
