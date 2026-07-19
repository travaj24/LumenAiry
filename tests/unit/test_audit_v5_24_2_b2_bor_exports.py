"""Audit v5.24.2 B2 / S5-10 -- BOR result/terminology/export normalization.

Non-breaking parts shipped this pass (the breaking return-shape change is
intentionally deferred to the S5-6 solve-result-container unification):

* ``BORStack`` -- the headline axisymmetric stack solver, the cylindrical
  peer of ``RCWAStack`` / ``PMMStack`` / ``BerremanStack`` -- is now a
  top-level re-export (``lumenairy.BORStack``) and listed in
  ``lumenairy.__all__``.
* The dict return shape of ``BORStack.solve`` is UNCHANGED (pinned here).
* mode/order terminology + the S5-6 forward-compat alias path are
  documented on the class.

Author: audit remediation -- v5.25 / B2
"""
from __future__ import annotations

import numpy as np

import lumenairy as la


def test_borstack_is_top_level_exported():
    """``lumenairy.BORStack`` resolves and IS the submodule class."""
    assert hasattr(la, 'BORStack')
    assert la.BORStack is la.elements.bor.BORStack


def test_borstack_in_dunder_all():
    assert 'BORStack' in la.__all__


def test_borstack_no_longer_in_symmetry_exemptions():
    """The walker exemption for BORStack must be RETIRED now that it is a
    genuine top-level re-export (else it would be a stale exemption)."""
    from tests.unit.test_v4_16_0_walker_all_symmetry import (
        _KNOWN_ALL_SYMMETRY_EXEMPTIONS,
    )
    assert ('lumenairy.elements.bor', 'BORStack') not in \
        _KNOWN_ALL_SYMMETRY_EXEMPTIONS
    # The lower-level BOR helpers stay namespaced (still exempt).
    assert ('lumenairy.elements.bor', 'radial_spectrum') in \
        _KNOWN_ALL_SYMMETRY_EXEMPTIONS


def test_borstack_solve_return_shape_unchanged():
    """S5-10/B2 explicitly ships WITHOUT changing return shapes.  Pin the
    documented dict contract so the deferred S5-6 container work is a
    deliberate, separate change."""
    s = la.BORStack(Rbig=4.0, m=1, N=64,
                    n_superstrate=1.4142, n_substrate=1.4142)
    s.add_layer(0.5, eps=2.0)
    s.set_source(k0=2.0)
    res = s.solve()
    assert isinstance(res, dict), (
        "BOR solve still returns a plain dict this pass (S5-6 container "
        "unification deferred)")
    for key in ('q', 'gamma', 'angles', 'R', 'T', 'energy', 'S'):
        assert key in res, f"documented solve() key {key!r} missing"
    # Energy conservation sanity (R + T ~ 1 for a lossless dielectric stack).
    assert np.all(np.isfinite(res['energy']))
    assert np.allclose(res['energy'], 1.0, atol=1e-6)


def test_borstack_terminology_documented():
    """The mode/order terminology + S5-6 alias path must be documented on
    the class (guards against the docstring silently dropping the
    convention a future reader relies on)."""
    doc = la.BORStack.__doc__ or ''
    assert 'azimuthal order' in doc
    assert 'mode' in doc and 'order' in doc
    assert 'S5-6' in doc  # the forward-compat container note
