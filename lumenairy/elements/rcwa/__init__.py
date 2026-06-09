# ruff: noqa: F401, F403, F405  -- this package facade re-exports every submodule name
"""Rigorous Coupled-Wave Analysis (RCWA) -- 1-D + 2-D Fourier Modal Method.

Split into submodules for readability (the public API is unchanged):
``_core`` (shared S-matrix / eigenmode machinery), ``oned`` (1-D gratings),
``twod`` (2-D crossed gratings), ``stack`` (the unified RCWAStack API).
Everything below is re-exported so ``lumenairy.elements.rcwa.<name>`` keeps
resolving for every public name AND the test-imported / monkeypatched
privates (the dispatch + caches call through these module globals)."""
from ._core import *
from .oned import *
from .stack import *
from .twod import *

__all__ = [
    "rcwa_efficiency_1d",
    "rcwa_efficiency_vs_wavelength",
    "rcwa_efficiency_2d",
    "rcwa_efficiency_2d_shapes",
    "rcwa_extrapolate",
    "rcwa_convergence",
    "rcwa_jones_1d",
    "rcwa_jones_1d_segments",
    "grating_segments",
    "binary_grating_segments",
    "interdigitated_grating_segments",
    "reflective_outcoupling",
    "jones_retardance_diattenuation",
    "rcwa_jones_2d",
    "rcwa_jones_vs_wavelength",
    "rcwa_jones_vs_wavelength_segments",
    "rcwa_efficiency_1d_jax",
    "uniaxial_tensor",
    "RCWAStack",
    "RCWAResult",
    "set_blas_threads",
    "rcwa_blas_threads",
]
