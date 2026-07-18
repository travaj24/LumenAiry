"""Audit S4-7 (v5.24.2 exhaustive audit) [P2]: two independent defects in
the GUI geometric optimizer (``SystemModel.run_optimization`` +
``OptimizeWorker``).

Part 1 -- data race.  The background ``OptimizeWorker`` ran
``model.run_optimization(...)`` on its worker thread, and that method
mutated the SHARED live model on every scipy probe (``merit_function`` ->
``set_variable_values``) and then wrote the final solution back +
``system_changed.emit()`` -- all off the GUI thread, while the main thread
could be rendering the same ``self.elements``.  Fix: the worker now calls
``apply_result=False``, so ``run_optimization`` restores the live model to
its pre-run state and hands the solution vector back via
``model._last_optimization_x`` / ``worker.result_x``; the dock's
finished-handler applies it on the MAIN thread.

Part 2 -- bounds.  The docstring claimed bounded methods used a
per-variable bound list, but no ``bounds=`` was EVER passed to
``scipy.minimize`` -- so a user selecting L-BFGS-B / TNC / SLSQP /
trust-constr ran effectively unconstrained and could drive a thickness /
air-gap negative.  Fix: build a lightweight physical default box for those
bounded methods only (``distance`` / ``thickness`` >= 0, other fields
free); the default Nelder-Mead path stays unbounded (byte-identical).

Independent oracles (not a re-run of the fix formula):
* Part 1 uses a monkeypatched ``minimize`` that returns ``x0 + 1`` as the
  "solution".  The live model ending at x0 (apply_result=False) vs at
  x0+1 (apply_result=True) is a direct, deterministic witness of WHERE the
  write-back happens -- independent of any real optimization.
* Part 2 captures the exact ``bounds`` argument the method hands to
  ``minimize``; pre-fix it was absent (None) for every method, so the
  bounded-method assertion fails pre-fix and passes post-fix, while the
  Nelder-Mead-stays-None assertion guards the byte-identical default.
"""
import os
import types

os.environ.setdefault('OMP_NUM_THREADS', '4')
# Force offscreen Qt BEFORE importing PySide6 in any path below.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

# Probe GUI deps: on CI the [gui] extra may be absent -> skip cleanly
# (same guard the sibling dock tests use).
try:
    import scipy.optimize as _sciopt

    from lumenairy.ui import optimizer_dock as _dock_mod
    from lumenairy.ui.model import (
        Element,
        SourceDefinition,
        SurfaceRow,
        SystemModel,
    )
    _GUI_OK = True
    _SKIP_REASON = ''
except ImportError as _e:  # pragma: no cover - exercised only on [gui]-less CI
    _GUI_OK = False
    _SKIP_REASON = f'GUI deps unavailable: {_e}'


def _build_singlet_model():
    """Source + one singlet + detector.  Free vars: the front radius (a
    surface field, must stay UNBOUNDED) and the air gap preceding the
    singlet (a ``distance`` field, must get a non-negative box)."""
    sm = SystemModel()
    singlet = Element(
        1, 'A', 'Singlet', distance_mm=30.0, surfaces=[
            SurfaceRow(radius=50.0, thickness=5.0, glass='N-BK7',
                       semi_diameter=10.0),
            SurfaceRow(radius=-50.0, thickness=0.0, glass='',
                       semi_diameter=10.0),
        ])
    sm.elements = [
        Element(0, 'Source', 'Source', distance_mm=0.0,
                source=SourceDefinition()),
        singlet,
        Element(2, 'Detector', 'Detector', distance_mm=100.0),
    ]
    sm.opt_variables = [
        (1, 0, 'radius'),     # surface field -> unbounded
        (1, 0, 'distance'),   # air gap       -> (0, None)
    ]
    sm._invalidate()
    return sm


def _patch_minimize_return_x0_plus_one(monkeypatch, captured):
    """Replace scipy.optimize.minimize with a stub that records the
    ``bounds`` / ``method`` it was handed and returns ``x0 + 1`` as the
    solution WITHOUT ever calling the objective or the callback (so the
    test needs no working ray-trace and is deterministic)."""
    def _fake_minimize(fun, x0, **kwargs):
        captured['bounds'] = kwargs.get('bounds')
        captured['method'] = kwargs.get('method')
        return types.SimpleNamespace(
            x=np.asarray(x0, dtype=float) + 1.0,
            fun=0.123, nit=3, success=True)
    monkeypatch.setattr(_sciopt, 'minimize', _fake_minimize)


# ---------------------------------------------------------------------------
# Part 1 -- apply_result gate (the data-race fix)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _GUI_OK, reason=_SKIP_REASON)
def test_worker_path_leaves_live_model_at_x0(monkeypatch):
    """``apply_result=False`` (the worker path) must NOT mutate the live
    model: it restores x0 and only stashes the solution."""
    sm = _build_singlet_model()
    x0 = sm.get_variable_values().copy()
    _patch_minimize_return_x0_plus_one(monkeypatch, {})

    ok, _msg = sm.run_optimization(max_iter=5, method='Nelder-Mead',
                                   apply_result=False)
    assert ok
    # Live model unchanged (restored to x0) -- the worker thread wrote
    # nothing the GUI thread could see torn.
    np.testing.assert_allclose(sm.get_variable_values(), x0)
    # Solution captured for the main thread to apply.
    np.testing.assert_allclose(sm._last_optimization_x, x0 + 1.0)


@pytest.mark.skipif(not _GUI_OK, reason=_SKIP_REASON)
def test_synchronous_path_applies_solution(monkeypatch):
    """``apply_result=True`` (the default, synchronous callers) applies the
    solution to the live model -- the pre-v5.24.4 write-back is preserved."""
    sm = _build_singlet_model()
    x0 = sm.get_variable_values().copy()
    _patch_minimize_return_x0_plus_one(monkeypatch, {})

    ok, _msg = sm.run_optimization(max_iter=5, method='Nelder-Mead',
                                   apply_result=True)
    assert ok
    np.testing.assert_allclose(sm.get_variable_values(), x0 + 1.0)


@pytest.mark.skipif(not _GUI_OK, reason=_SKIP_REASON)
def test_optimize_worker_carries_solution_and_leaves_model_clean(monkeypatch):
    """End-to-end at the worker layer: ``OptimizeWorker.run()`` (driven
    synchronously) must restore the live model to x0 AND expose the
    solution on ``worker.result_x`` for the main-thread handler."""
    sm = _build_singlet_model()
    x0 = sm.get_variable_values().copy()
    _patch_minimize_return_x0_plus_one(monkeypatch, {})

    worker = _dock_mod.OptimizeWorker(sm, max_iter=5,
                                      advanced_kwargs={'method': 'Nelder-Mead'})
    worker.run()   # run the body in-thread; no QThread.start()

    np.testing.assert_allclose(sm.get_variable_values(), x0)   # not mutated
    assert worker.result_x is not None
    np.testing.assert_allclose(worker.result_x, x0 + 1.0)      # solution carried


# ---------------------------------------------------------------------------
# Part 2 -- bounds are actually passed for bounded methods only
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _GUI_OK, reason=_SKIP_REASON)
def test_bounded_method_passes_nonneg_box_for_length_vars(monkeypatch):
    """A user-selected bounded method (L-BFGS-B) now receives a real
    ``bounds`` list: ``distance``/``thickness`` variables get (0, None),
    other fields stay free (None, None)."""
    sm = _build_singlet_model()
    captured = {}
    _patch_minimize_return_x0_plus_one(monkeypatch, captured)

    sm.run_optimization(max_iter=5, method='L-BFGS-B', apply_result=False)
    bounds = captured['bounds']
    assert bounds is not None, 'bounded method must receive a bounds list'
    # opt_variables order: [radius (surface), distance (air gap)].
    assert bounds[0] == (None, None)   # radius: free
    assert bounds[1] == (0.0, None)    # distance: non-negative


@pytest.mark.skipif(not _GUI_OK, reason=_SKIP_REASON)
def test_default_nelder_mead_stays_unbounded(monkeypatch):
    """The default Nelder-Mead path must keep passing ``bounds=None`` so the
    pre-v5.4 default behaviour is byte-identical (no silent box clipping)."""
    sm = _build_singlet_model()
    captured = {}
    _patch_minimize_return_x0_plus_one(monkeypatch, captured)

    sm.run_optimization(max_iter=5, method='Nelder-Mead', apply_result=False)
    assert captured['bounds'] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
