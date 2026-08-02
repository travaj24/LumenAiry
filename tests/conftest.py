"""Shared pytest fixtures for the lumenairy unit-test suite.

Provides small grids, standard wavelengths, and minimal optical
elements so individual unit-test modules don't repeat the same setup.

Design goals
------------
* **Fast**: every fixture targets N=64 by default so a unit-test
  module finishes in under a second per file.
* **No external deps**: nothing here requires Zemax, rayoptics,
  Optiland, h5py, or matplotlib.  Unit tests should run on a fresh
  checkout with only the base library dependencies.
* **Standard SI units everywhere**: meters, radians, real Hz/m
  spatial frequencies.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la

# ---------------------------------------------------------------------------
# Grid / wavelength / k0
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def N_small() -> int:
    """Small grid dimension for fast unit tests (64x64)."""
    return 64


@pytest.fixture(scope='session')
def N_med() -> int:
    """Medium grid dimension when 64x64 is too coarse (128x128)."""
    return 128


@pytest.fixture(scope='session')
def dx_m() -> float:
    """Standard test grid spacing (5 microns)."""
    return 5e-6


@pytest.fixture(scope='session')
def wavelength_m() -> float:
    """Standard test wavelength (1.31 microns, telecom O-band)."""
    return 1.31e-6


@pytest.fixture(scope='session')
def k0(wavelength_m) -> float:
    """Wave-number magnitude k = 2*pi / lambda."""
    return 2.0 * np.pi / wavelength_m


# ---------------------------------------------------------------------------
# Field fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def plane_wave(N_small) -> np.ndarray:
    """Unit-amplitude plane wave on the small grid."""
    return np.ones((N_small, N_small), dtype=np.complex128)


@pytest.fixture
def gaussian_beam(N_small, dx_m, wavelength_m) -> np.ndarray:
    """Tiny Gaussian beam: sigma = 30 microns, peak-normalized.

    ``create_gaussian_beam`` returns ``(E, x, y)``; the fixture
    unpacks the field for tests that only need the array.
    """
    E, _x, _y = la.create_gaussian_beam(N_small, dx_m, wavelength_m,
                                          w0=(30e-6) * np.sqrt(2))
    return E


# ---------------------------------------------------------------------------
# FFT dispatch isolation
# ---------------------------------------------------------------------------

@pytest.fixture
def shipped_fft_dispatch():
    """Run a test from the SHIPPED FFT-dispatch configuration, then restore
    whatever the process had before.

    Opt-in (NOT autouse): request it from a module's own autouse fixture.

    WHY (release verification 2026-08-01).  The pyFFTW plan-cache pins --
    ``test_perf_v4_12_0_fft_infra.py::TestAutoPromote`` and
    ``test_niche_audit_w9_traced_determinism.py::
    test_auto_promote_still_promotes_when_opted_in`` -- all assert that the
    FIRST plan built at a key is ``FFTW_ESTIMATE`` and that a cache entry
    exists at all.  Both of those depend on process-global dispatch state
    that neither module's own fixture owned:

      * ``USE_PYFFTW`` -- ``False`` (e.g. left behind by any consumer of the
        UI dock's backend selector, which sets it unconditionally and only
        re-raises it for ``backend == 'pyfftw'``) makes ``_fft2`` skip the
        plan cache entirely, so the probed entry is ``None``;
      * ``FFTW_MIN_SIZE`` -- raised above the test's N has the same effect,
        and it is the ONE dispatch global ``snapshot_fft_state`` does not
        carry;
      * ``_PYFFTW_PLAN_FLAGS`` -- left at ``FFTW_MEASURE`` makes the first
        plan MEASURE, so "starts at ESTIMATE" fails;
      * ``PYFFTW_FALLBACK_ON_ERROR`` / ``_PYFFTW_DOUBLE_BUFFER`` /
        ``_PYFFTW_PLAN_CACHE_SIZE`` -- reachable via ``set_low_memory``.

    Each of those makes the pins pass alone and fail in a full sweep.  This
    fixture removes the coupling by CONSTRUCTION rather than by chasing the
    polluter: it forces every one of them to its shipped value, clears the
    plan cache and the bad-shape blacklist, and restores the caller's state
    (including ``FFTW_MIN_SIZE``, by hand) on the way out.

    It deliberately does NOT touch libfftw3 *wisdom* -- that is
    process-global inside the C library, affects bits rather than the plan
    flags asserted here, and the w9 module already snapshots it.
    """
    from lumenairy.propagators import fft_infra as _fi

    state = _fi.snapshot_fft_state()
    prev_min_size = _fi.FFTW_MIN_SIZE
    prev_planner = _fi.get_pyfftw_planner()
    _fi.USE_PYFFTW = _fi.PYFFTW_AVAILABLE     # shipped: on iff importable
    _fi.USE_SCIPY_FFT = True                  # shipped
    _fi.PYFFTW_FALLBACK_ON_ERROR = True       # shipped
    _fi.FFTW_MIN_SIZE = 256                   # shipped
    if not _fi._PYFFTW_DOUBLE_BUFFER:         # shipped (clears plans on flip)
        _fi.set_fft_double_buffer(True)
    _fi.set_fft_plan_cache_size(8)            # shipped
    _fi.set_pyfftw_planner('FFTW_ESTIMATE')   # shipped; clears the plan cache
    _fi.reset_fft_backend()                   # + bad shapes, + call counters
    try:
        yield
    finally:
        _fi.FFTW_MIN_SIZE = prev_min_size
        _fi.restore_fft_state(state)
        _fi.set_pyfftw_planner(prev_planner)
        _fi.reset_fft_backend()


# ---------------------------------------------------------------------------
# Prescription fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def singlet_prescription():
    """Minimal singlet prescription dict (N-BK7 plano-convex,
    R1=25mm/R2=inf, 2.5mm thick, 5mm aperture)."""
    return la.make_singlet(
        R1=25e-3, R2=np.inf, d=2.5e-3,
        glass='N-BK7', aperture=5e-3,
    )


@pytest.fixture(scope='session')
def doublet_prescription():
    """Minimal cemented N-BK7 + N-SF2 doublet."""
    return la.make_doublet(
        R1=25e-3, R2=-20e-3, R3=-50e-3,
        d1=2.5e-3, d2=1.5e-3,
        glass1='N-BK7', glass2='N-SF2',
        aperture=5e-3,
    )
