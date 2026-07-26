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
