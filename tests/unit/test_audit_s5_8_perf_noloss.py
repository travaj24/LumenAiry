"""Audit S5-8 [P2][perf]: measured no-loss (bit-identical) wins.

This finding bundles several profiler-measured speedups that change NO numbers.
Two of them are implemented here (the two cleanest, surgical, bit-identical
ones); the rest are deliberately skipped per the recipe (see the module-level
notes in the PR / caveats):

* Win (1) -- ``elements/rcwa/_core.py``: ``threadpool_limits(...)`` rebuilt a
  fresh ``ThreadpoolController`` on EVERY RCWA solve, re-enumerating every
  loaded BLAS DLL (~9 ms on Windows; a 20-wavelength sweep measured
  283 -> 105 ms once cached).  We now enumerate ONCE into a process-wide
  cached controller and reuse ``controller.limit(...)``.  Only the opt-in
  ``set_blas_threads`` / ``rcwa_blas_threads`` path is affected; the numbers
  are unchanged (the limiter save/restore is identical, only DLL discovery is
  amortised).

* Win (6) -- ``propagators/fft_infra.py``: ``pyfftw.interfaces.cache.enable()``
  spun up an idle ``PyFFTWCacheThread`` keep-alive daemon.  lumenairy drives
  every FFT through raw ``pyfftw.FFTW`` plans in ``_PYFFTW_PLAN_CACHE``, never
  the ``pyfftw.interfaces.*`` wrapper API, so that cache was ALWAYS empty for
  us -- the daemon (and the disable/enable toggles in the reset / failure
  paths) did nothing but pollute every profile.  Dropped.

The tests below are INDEPENDENT PROBES, not tautologies:

* The controller test COUNTS ``ThreadpoolController`` constructions across many
  ``_blas_limit()`` entries via an injected stub module and asserts it is
  built AT MOST ONCE -- i.e. it directly measures "enumerate once, reuse", the
  actual win.  (A stub is used because ``threadpoolctl`` is optional and may be
  absent from the CI venv; the caching logic is what we are validating, not the
  native library.)
* The pyfftw tests SPY on ``pyfftw.interfaces.cache.enable`` / ``.disable`` and
  assert the loader / reset path never touches them, while ``reset_fft_backend``
  still clears the REAL plan cache -- proving the daemon is gone without
  weakening the reset's genuine job.
"""
from __future__ import annotations

import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Win (1): cached ThreadpoolController (RCWA BLAS-cap opt-in path)
# ---------------------------------------------------------------------------

def _install_stub_threadpoolctl(monkeypatch):
    """Inject a fake ``threadpoolctl`` whose ``ThreadpoolController`` counts
    its own constructions, and reset the module-level controller cache so the
    fake is the one that gets built.  Returns (build_count, limit_calls)."""
    from lumenairy.elements.rcwa import _core

    build_count = {"n": 0}
    limit_calls = []

    class _FakeLimiter:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _FakeController:
        def __init__(self):
            build_count["n"] += 1

        def limit(self, *, limits, user_api):
            limit_calls.append((limits, user_api))
            return _FakeLimiter()

    fake_mod = types.ModuleType("threadpoolctl")
    fake_mod.ThreadpoolController = _FakeController
    # Present but unused: _blas_limit only reaches this on the legacy fallback,
    # which the controller path avoids.
    fake_mod.threadpool_limits = None
    monkeypatch.setitem(sys.modules, "threadpoolctl", fake_mod)

    # Clear the process-wide cache so the first _get_blas_controller() builds
    # our fake (monkeypatch restores the originals after the test).
    monkeypatch.setattr(_core, "_BLAS_CONTROLLER", None)
    monkeypatch.setattr(_core, "_BLAS_CONTROLLER_UNAVAILABLE", False)
    return _core, build_count, limit_calls


def test_blas_controller_built_once_and_reused(monkeypatch):
    """The whole S5-8 win: the BLAS-thread cap must enumerate the loaded BLAS
    libraries ONCE and reuse the controller, not rebuild it (re-scanning the
    DLLs) on every solve."""
    _core, build_count, limit_calls = _install_stub_threadpoolctl(monkeypatch)

    n_solves = 25
    with _core.rcwa_blas_threads(2):
        for _ in range(n_solves):
            with _core._blas_limit():
                pass

    assert build_count["n"] == 1, (
        f"ThreadpoolController rebuilt {build_count['n']}x across {n_solves} "
        "capped solves -- the point of S5-8 is to enumerate the BLAS DLLs "
        "once and reuse the controller (per-call rebuild is the ~9 ms/solve "
        "regression the fix removes).")
    # The cap is still applied on every entry (bit-identical behaviour).
    assert len(limit_calls) == n_solves
    assert all(call == (2, "blas") for call in limit_calls)


def test_blas_limit_noop_without_cap_builds_no_controller(monkeypatch):
    """Default path (no cap set) must be a zero-overhead no-op that never even
    constructs the controller -- untouched threading for the common case."""
    _core, build_count, limit_calls = _install_stub_threadpoolctl(monkeypatch)

    # No rcwa_blas_threads(...) scope -> _get_blas_threads() is None.
    assert _core._get_blas_threads() is None
    for _ in range(10):
        with _core._blas_limit():
            pass

    assert build_count["n"] == 0, (
        "the no-cap default path must not construct a ThreadpoolController "
        "(no DLL enumeration when the user never opted into a cap).")
    assert limit_calls == []


def test_blas_controller_unavailable_falls_back_cleanly(monkeypatch):
    """When threadpoolctl lacks ThreadpoolController (< 3.0) the helper must
    latch unavailable and _blas_limit must still yield a usable context (the
    legacy fallback), never raise."""
    from lumenairy.elements.rcwa import _core

    # A threadpoolctl with neither symbol -> ImportError on both paths.
    empty_mod = types.ModuleType("threadpoolctl")
    monkeypatch.setitem(sys.modules, "threadpoolctl", empty_mod)
    monkeypatch.setattr(_core, "_BLAS_CONTROLLER", None)
    monkeypatch.setattr(_core, "_BLAS_CONTROLLER_UNAVAILABLE", False)

    assert _core._get_blas_controller() is None
    assert _core._BLAS_CONTROLLER_UNAVAILABLE is True  # latched, no re-probe
    with _core.rcwa_blas_threads(2):
        with _core._blas_limit():  # must not raise
            pass


# ---------------------------------------------------------------------------
# Win (6): no idle pyfftw.interfaces.cache daemon
# ---------------------------------------------------------------------------

def test_pyfftw_loader_does_not_enable_interfaces_cache(monkeypatch):
    """_ensure_pyfftw_loaded must not enable the interfaces cache (which would
    spin up an idle PyFFTWCacheThread); lumenairy uses raw FFTW plans only."""
    pytest.importorskip("pyfftw")
    cache = pytest.importorskip("pyfftw.interfaces.cache")
    from lumenairy.propagators import fft_infra

    calls = []
    monkeypatch.setattr(cache, "enable",
                        lambda *a, **k: calls.append("enable"))
    monkeypatch.setattr(cache, "set_keepalive_time",
                        lambda *a, **k: calls.append("set_keepalive_time"))
    # Force the fresh-load branch (loader short-circuits once pyfftw is bound).
    monkeypatch.setattr(fft_infra, "pyfftw", None)

    assert fft_infra._ensure_pyfftw_loaded() is True
    assert calls == [], (
        "loading pyFFTW must not touch pyfftw.interfaces.cache -- the "
        "interfaces API is never used, so enabling its cache only leaks an "
        "idle keep-alive daemon thread.")


def test_reset_fft_backend_clears_plan_cache_without_interfaces_toggle(monkeypatch):
    """reset_fft_backend must still flush the REAL plan cache, but must no
    longer disable/enable the (unused) interfaces cache daemon."""
    pytest.importorskip("pyfftw")
    cache = pytest.importorskip("pyfftw.interfaces.cache")
    from lumenairy.propagators import fft_infra

    toggles = []
    monkeypatch.setattr(cache, "disable",
                        lambda *a, **k: toggles.append("disable"))
    monkeypatch.setattr(cache, "enable",
                        lambda *a, **k: toggles.append("enable"))
    monkeypatch.setattr(cache, "set_keepalive_time",
                        lambda *a, **k: toggles.append("set_keepalive_time"))

    # Seed the real plan cache with a sentinel so we can prove the reset still
    # does its genuine job.
    fft_infra._PYFFTW_PLAN_CACHE[("sentinel",)] = {"marker": True}
    fft_infra.reset_fft_backend()

    assert ("sentinel",) not in fft_infra._PYFFTW_PLAN_CACHE, (
        "reset_fft_backend must still clear the real _PYFFTW_PLAN_CACHE.")
    assert toggles == [], (
        "reset_fft_backend must no longer toggle pyfftw.interfaces.cache -- "
        "lumenairy never populates it, so the disable/enable freed nothing.")
