"""Audit v5.17.0 P1-07 / P2-36 / P3-61: fixed-index glass registration.

surfaces_from_elements named spherical/aspheric pseudo-glasses by id(elem);
CPython recycles ids after GC, so two successive builds with different n_lens
shared a glass name and _register_fixed_index's unconditional overwrite
silently retargeted previously built surface lists to the LATER index (trace()
resolves glass at trace time).  Fixed with content-derived names
('__spherical_' + repr(float(n_lens))), which are also idempotent, bounding
the previously unbounded GLASS_REGISTRY/_glass_cache growth (P3-61).
_register_fixed_index also wrote the sentinel
('__fixed__', '__fixed__', '__fixed__') which glass.get_glass_index does NOT
match in its user-fixed branch (that requires entry[0] == '__user__'), so on
installs without the optional refractiveindex package every thin-lens /
spherical / aspheric trace raised ImportError (P2-36); the sentinel is now
('__user__', '__fixed__', '__fixed__').

Pre-fix (verified via git-stash A/B): test_distinct_n_lens_distinct_names,
test_retained_surface_list_resolves_original_index,
test_sweep_resolutions_all_correct, test_sentinel_routes_user_fixed_branch,
test_minimal_install_resolution and test_registry_growth_bounded all FAIL.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from lumenairy import glass
from lumenairy.glass import GLASS_REGISTRY, _glass_cache, get_glass_index
from lumenairy.raytrace.trace import (
    _register_fixed_index,
    raytrace_system,
    surfaces_from_elements,
)

WV = 550e-9


def _build(n_lens, etype='spherical_lens'):
    elem = {'type': etype, 'n_lens': n_lens,
            'R1': 0.05, 'R2': -0.05, 'd': 0.005}
    return surfaces_from_elements([elem], WV)


# ---------------------------------------------------------------------------
# P1-07: id-recycling collision
# ---------------------------------------------------------------------------

def test_distinct_n_lens_distinct_names():
    """Back-to-back builds with different n_lens must yield DIFFERENT glass
    names (pre-fix the elem dicts recycled the same id -> same name)."""
    name_a = _build(1.7)[0].glass_after
    name_b = _build(1.9)[0].glass_after
    assert name_a != name_b


def test_retained_surface_list_resolves_original_index():
    """A surface list built first must keep resolving its ORIGINAL index
    after a later build registers a different n_lens (pre-fix: 1.9)."""
    list_a = _build(1.7)
    _build(1.9)
    assert get_glass_index(list_a[0].glass_after, WV) == 1.7


def test_sweep_resolutions_all_correct():
    """200-point n_lens sweep keeping every surface list: all 200 must
    resolve to their own index (pre-fix: 198+/200 wrong, ~1 distinct name)."""
    ns = np.linspace(1.4, 2.0, 200)
    lists = [_build(float(n)) for n in ns]
    wrong = sum(
        1 for n, lst in zip(ns, lists)
        if abs(get_glass_index(lst[0].glass_after, WV) - n) > 1e-12)
    assert wrong == 0
    assert len({lst[0].glass_after for lst in lists}) == 200


def test_aspheric_names_content_derived():
    """Aspheric branch shares the fix: same content -> same name; the
    prefix keeps spherical/aspheric namespaces distinct."""
    elem = {'type': 'aspheric_lens', 'n_lens': 1.6,
            'R1': 0.04, 'R2': -0.04, 'd': 0.004}
    name1 = surfaces_from_elements([elem], WV)[0].glass_after
    name2 = surfaces_from_elements([dict(elem)], WV)[0].glass_after
    assert name1 == name2
    assert name1.startswith('__aspheric_')


# ---------------------------------------------------------------------------
# P2-36: sentinel must route to get_glass_index's user-fixed branch
# ---------------------------------------------------------------------------

def test_sentinel_routes_user_fixed_branch():
    """The registered sentinel must be the ('__user__', ...) tuple that
    glass.get_glass_index's user-fixed branch matches (pre-fix it was
    ('__fixed__', '__fixed__', '__fixed__') and fell through to the
    refractiveindex.info tuple branch)."""
    _register_fixed_index('__test_p236__', 1.23, WV)
    try:
        entry = GLASS_REGISTRY['__test_p236__']
        assert entry == ('__user__', '__fixed__', '__fixed__')
        # Same sentinel as the canonical user_library.register_fixed_glass.
        assert entry[0] == '__user__'
    finally:
        GLASS_REGISTRY.pop('__test_p236__', None)
        _glass_cache.pop('__test_p236__', None)


def test_minimal_install_resolution():
    """With the refractiveindex package unavailable, thin-lens and
    spherical pseudo-glasses must still resolve (pre-fix: ImportError
    telling the user to pip install refractiveindex)."""
    name = _build(1.9)[0].glass_after
    avail = glass._REFRACTIVEINDEX_AVAILABLE
    glass._REFRACTIVEINDEX_AVAILABLE = False
    try:
        assert get_glass_index('__thin_lens__', WV) == 1.5
        assert get_glass_index(name, WV) == 1.9
    finally:
        glass._REFRACTIVEINDEX_AVAILABLE = avail


def test_minimal_install_end_to_end_trace():
    """End-to-end raytrace_system with a thin lens + spherical lens on a
    simulated minimal install must trace, not raise ImportError."""
    avail = glass._REFRACTIVEINDEX_AVAILABLE
    glass._REFRACTIVEINDEX_AVAILABLE = False
    try:
        result, _ = raytrace_system(
            [{'type': 'lens', 'f': 0.1, 'aperture_diameter': 0.02},
             {'type': 'propagation', 'distance': 0.02},
             {'type': 'spherical_lens', 'n_lens': 1.7, 'R1': 0.05,
              'R2': -0.05, 'd': 0.005, 'aperture_diameter': 0.02}],
            WV, semi_aperture=0.008, num_rings=3, rays_per_ring=8)
        assert np.any(result.image_rays.alive)
    finally:
        glass._REFRACTIVEINDEX_AVAILABLE = avail


# ---------------------------------------------------------------------------
# P3-61: registry growth bounded by distinct content
# ---------------------------------------------------------------------------

def test_registry_growth_bounded():
    """Rebuilding the SAME element many times must not grow the global
    registry/cache beyond one entry (pre-fix: one entry per recycled id,
    unbounded across distinct ids)."""
    _build(1.5432)  # ensure the single content-derived entry exists
    before_reg = len(GLASS_REGISTRY)
    before_cache = len(_glass_cache)
    for _ in range(500):
        _build(1.5432)
    assert len(GLASS_REGISTRY) == before_reg
    assert len(_glass_cache) == before_cache


def test_reregistration_invalidates_value_cache():
    """_register_fixed_index must drop stale _glass_value_cache entries
    for a name it overwrites."""
    from lumenairy.glass import _glass_value_cache
    key = ('__test_p361__', round(WV * 1e12))
    _glass_value_cache[key] = 99.0  # plant a stale value
    _register_fixed_index('__test_p361__', 1.44, WV)
    try:
        assert key not in _glass_value_cache
        assert get_glass_index('__test_p361__', WV) == 1.44
    finally:
        GLASS_REGISTRY.pop('__test_p361__', None)
        _glass_cache.pop('__test_p361__', None)
