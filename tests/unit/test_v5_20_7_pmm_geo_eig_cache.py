"""PMM geometric-eig sweep cache (1-D uniform half-space eig reuse).

The eps-free geometric eig of a 1-D PMM uniform half-space -- the standard
``_uniform_geo_eig`` (tensor path) and the generalized ``_scalar_uniform_geo_eig``
(scalar path) -- depends only on the nodal GEOMETRY and ``kx0`` (angle), NOT on
``k0`` (wavelength), which enters purely as the ``1/k0^2`` spectrum scale.  So a
fixed-angle wavelength sweep re-eigs the SAME pencil every point (the audited
51-64% of 1-D eig time).  A bounded LRU (`_cached_geo_eig`) eigs it once and
scales -- verified: the cache does not grow across a sweep, results stay
energy-conserving, and a cleared vs warm solve is byte-identical.
"""
from __future__ import annotations

import numpy as np

from lumenairy.elements.pmm import pmm_efficiency_1d
from lumenairy.elements.pmm._core import _GEO_EIG_CACHE, _clear_geo_eig_cache

_P = 0.8e-6
_WL = 0.55e-6
_DEP = 0.30e-6
_DUTY = 0.5
_KW = dict(polarization="tm", degree=14, n_orders=12)


def _solve(wl):
    return pmm_efficiency_1d(_P, 2.5, 1.0, 1.5, 1.0, _DEP, _DUTY, wl, **_KW)


def test_geo_eig_reused_across_fixed_angle_sweep():
    _clear_geo_eig_cache()
    for wl in np.linspace(0.50e-6, 0.60e-6, 30):
        _solve(wl)
    # the geometric eig is wavelength-independent -> a fixed-angle sweep adds no
    # new eigs beyond the (few) distinct half-space pencils.
    assert len(_GEO_EIG_CACHE) <= 4


def test_cached_solve_is_byte_identical_and_energy_conserving():
    _clear_geo_eig_cache()
    o1, R1, T1 = _solve(_WL)                 # cold: computes + caches the eig
    o2, R2, T2 = _solve(_WL)                 # warm: hits the cache
    assert np.array_equal(R1, R2) and np.array_equal(T1, T2)
    _clear_geo_eig_cache()
    o3, R3, T3 = _solve(_WL)                 # cleared: recomputes -> identical
    assert np.array_equal(R1, R3) and np.array_equal(T1, T3)
    # lossless dielectric grating -> energy conserved
    assert abs(float(np.sum(R1)) + float(np.sum(T1)) - 1.0) < 1e-9
