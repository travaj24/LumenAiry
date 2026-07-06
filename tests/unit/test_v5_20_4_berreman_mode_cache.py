"""Berreman per-layer modal eig cache (speed, byte-identical).

The Berreman 4x4 layer eig depends only on ``eps`` and ``Kx/Ky`` (angle), NOT
on wavelength, so a fixed-angle wavelength sweep and a periodic ABAB... stack
recompute the SAME eig repeatedly.  A module-level bounded LRU
(``_layer_modes_cached``) returns byte-identical modes.  These tests pin that
the cache (a) changes nothing numerically, (b) dedups repeated layers within a
solve, and (c) reuses the eig across a fixed-angle wavelength sweep.
"""
from __future__ import annotations

import numpy as np

from lumenairy.elements import berreman as B
from lumenairy.elements.berreman import (
    _MODE_CACHE,
    _clear_berreman_mode_cache,
)

_WL = 0.55e-6
_ANG = np.deg2rad(20.0)
_PHI = 0.2
_LC = np.diag([1.6 ** 2, 1.5 ** 2, 1.5 ** 2]).astype(complex)   # uniaxial
_HI = 2.1 ** 2                                                   # isotropic
_DBR = [(_LC, 0.30e-6), (_HI, 0.20e-6)] * 4                      # ABAB... x4


def _solve():
    return B.berreman_jones_1d(_DBR, 1.5, 1.0, _WL, angle=_ANG, phi=_PHI)


def test_cache_is_byte_identical_and_clearable():
    _clear_berreman_mode_cache()
    o1 = _solve()
    o2 = _solve()                                    # 2nd call hits the cache
    assert all(np.array_equal(a, b) for a, b in zip(o1, o2))
    _clear_berreman_mode_cache()                     # cleared -> recompute
    o3 = _solve()
    assert all(np.array_equal(a, b) for a, b in zip(o1, o3))


def test_cache_dedups_repeated_layers_in_one_solve():
    _clear_berreman_mode_cache()
    _solve()
    # ABAB...x4 has only 2 distinct layer tensors + 2 half-spaces = 4 unique
    # eigs, even though the stack lists 8 layers.
    assert len(_MODE_CACHE) == 4


def test_cache_reused_across_fixed_angle_wavelength_sweep():
    _clear_berreman_mode_cache()
    for wl in np.linspace(0.50e-6, 0.60e-6, 25):
        B.berreman_jones_1d(_DBR, 1.5, 1.0, wl, angle=_ANG, phi=_PHI)
    # modes are wavelength-independent -> the sweep adds NO new eigs.
    assert len(_MODE_CACHE) == 4


def test_cached_dbr_is_energy_conserving():
    """The cached path stays physically correct (a lossless stack: R+T=1).
    ``berreman_jones_1d`` returns ``(R, T, Jr, Jt)`` -- R/T are the per-incident
    -polarization power vectors."""
    _clear_berreman_mode_cache()
    R, T, _Jr, _Jt = _solve()
    assert np.allclose(R + T, 1.0, atol=1e-9)
