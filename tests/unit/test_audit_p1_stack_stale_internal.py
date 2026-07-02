"""Audit P1-04: stale ``self._internal`` must NOT survive a re-solve.

Pre-fix, ``PMMStack.solve``/``PMM2DStack.solve`` never invalidated
``self._internal``, so ``solve(retain_internal=True)`` followed by
``set_source(new_wl)`` + ``solve()`` (no retain) left ``internal_field()``
and ``layer_absorption()`` silently serving the PREVIOUS wavelength's
fields (probe: max|stale - true 750 nm field| ~ 0.8 while R changed).
The fix invalidates the retained internals at the START of every solve
(above the JAX/covariant dispatches), so only the MOST RECENT
``retain_internal=True`` solve can serve internals -- everything else
lands on the documented ValueError.

The staleness tests FAIL on the pre-fix code (the calls return stale
data instead of raising).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pytest

from lumenairy.elements.pmm import PMM2DStack, PMMStack

P, WL_A, WL_B = 0.8e-6, 550e-9, 750e-9


def _stack_1d():
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=10)
    st.add_layer(0.30e-6, segments=[(0.45, 4.0 + 0.8j), (0.55, 1.0)])
    return st


def _stack_2d():
    st = PMM2DStack(P, n_substrate=1.5, n_superstrate=1.0, degree=7,
                    n_orders=3)
    st.add_layer(0.2e-6, eps=4.0 + 0.8j)
    st.add_layer(0.08e-6, eps=2.25)
    return st


def test_1d_resolve_without_retain_invalidates_internals():
    st = _stack_1d()
    st.set_source(WL_A).solve(retain_internal=True)
    st.internal_field(np.array([0.15e-6]))          # valid while retained
    st.layer_absorption()
    st.set_source(WL_B)
    st.solve()                                       # NO retain
    with pytest.raises(ValueError, match="MOST RECENT"):
        st.internal_field(np.array([0.15e-6]))
    with pytest.raises(ValueError, match="retain_internal"):
        st.layer_absorption()


def test_2d_resolve_without_retain_invalidates_internals():
    st = _stack_2d()
    st.set_source(WL_A).solve(retain_internal=True)
    st.internal_field(np.array([0.05e-6]), nx=8, ny=8)
    st.layer_absorption()
    st.set_source(WL_B)
    st.solve()                                       # NO retain
    with pytest.raises(ValueError, match="MOST RECENT"):
        st.internal_field(np.array([0.05e-6]), nx=8, ny=8)
    with pytest.raises(ValueError, match="retain_internal"):
        st.layer_absorption()


def test_1d_retained_resolve_serves_the_new_solve():
    """Re-solving WITH retain at a new wavelength serves the NEW field --
    identical to a fresh stack solved once at that wavelength."""
    st = _stack_1d()
    st.set_source(WL_A).solve(retain_internal=True)
    f_a = st.internal_field(np.array([0.15e-6]))
    st.set_source(WL_B).solve(retain_internal=True)
    f_b = st.internal_field(np.array([0.15e-6]))
    fresh = _stack_1d()
    fresh.set_source(WL_B).solve(retain_internal=True)
    f_ref = fresh.internal_field(np.array([0.15e-6]))
    assert np.array_equal(f_b["Ex"], f_ref["Ex"])    # deterministic path
    assert np.max(np.abs(f_b["Ex"] - f_a["Ex"])) > 1e-2   # really moved


def test_2d_retained_resolve_serves_the_new_solve():
    st = _stack_2d()
    st.set_source(WL_A).solve(retain_internal=True)
    a_a = st.layer_absorption()
    st.set_source(WL_B).solve(retain_internal=True)
    a_b = st.layer_absorption()
    fresh = _stack_2d()
    fresh.set_source(WL_B).solve(retain_internal=True)
    assert np.array_equal(a_b, fresh.layer_absorption())
    assert np.max(np.abs(a_b - a_a)) > 1e-3


def test_never_retained_still_raises():
    st = _stack_1d()
    st.set_source(WL_A).solve()
    with pytest.raises(ValueError, match="retain_internal"):
        st.internal_field(np.array([0.15e-6]))
    st2 = _stack_2d()
    st2.set_source(WL_A).solve()
    with pytest.raises(ValueError, match="retain_internal"):
        st2.internal_field(np.array([0.05e-6]))


def test_1d_superseding_paths_invalidate_retained():
    """Wave-1 adversarial follow-up: solve_vs_wavelength, prepare().solve,
    set_source and add_layer all supersede retained internals (they bypass or
    precede solve(), which was the only invalidation site at first)."""
    # solve_vs_wavelength bypasses solve()
    st = _stack_1d()
    st.set_source(WL_A).solve(retain_internal=True)
    st.internal_field(np.array([0.15e-6]))              # sanity: works
    st.solve_vs_wavelength(np.array([WL_A, WL_B]))
    with pytest.raises(ValueError, match="retain_internal"):
        st.internal_field(np.array([0.15e-6]))
    # set_source alone supersedes (fields belong to the OLD source)
    st = _stack_1d()
    st.set_source(WL_A).solve(retain_internal=True)
    st.set_source(WL_B)
    with pytest.raises(ValueError, match="retain_internal"):
        st.internal_field(np.array([0.15e-6]))
    # add_layer alone supersedes (fields belong to the OLD geometry)
    st = _stack_1d()
    st.set_source(WL_A).solve(retain_internal=True)
    st.add_layer(0.05e-6, eps=2.25)
    with pytest.raises(ValueError, match="retain_internal"):
        st.internal_field(np.array([0.15e-6]))
    # the prepared assemble-once path supersedes the parent stack's internals
    st = _stack_1d()
    prepared = st.prepare()
    st.set_source(WL_A).solve(retain_internal=True)
    prepared.solve(wavelength=WL_B)
    with pytest.raises(ValueError, match="retain_internal"):
        st.internal_field(np.array([0.15e-6]))


def test_2d_superseding_paths_invalidate_retained():
    st = _stack_2d()
    st.set_source(WL_A).solve(retain_internal=True)
    st.internal_field(np.array([0.05e-6]))              # sanity: works
    st.solve_vs_wavelength(np.array([WL_A, WL_B]))
    with pytest.raises(ValueError, match="retain_internal"):
        st.internal_field(np.array([0.05e-6]))
    st = _stack_2d()
    st.set_source(WL_A).solve(retain_internal=True)
    st.set_source(WL_B)
    with pytest.raises(ValueError, match="retain_internal"):
        st.internal_field(np.array([0.05e-6]))
    st = _stack_2d()
    st.set_source(WL_A).solve(retain_internal=True)
    st.add_layer(0.03e-6, eps=2.25)
    with pytest.raises(ValueError, match="retain_internal"):
        st.internal_field(np.array([0.05e-6]))
