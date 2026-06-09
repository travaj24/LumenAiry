# ruff: noqa: F401, F403, F405  -- package facade re-exports every submodule name
"""Polynomial Modal Method (PMM) -- non-Fourier spectral-element modal solver.

Split into submodules for readability (the public API is unchanged):
``_core`` (basis + slant generators + S-matrix + far-field), ``oned`` (the
public 1-D entry points), ``stack`` (PMMStack).  Re-exported so
``lumenairy.elements.pmm.<name>`` resolves for every public name AND the
test-imported / monkeypatched privates (the slant dispatch spies these)."""
from ._core import *
from .oned import *
from .stack import *

__all__ = ["pmm_efficiency_1d", "pmm_efficiency_1d_jax",
           "pmm_efficiency_1d_segments",
           "pmm_jones_1d", "pmm_jones_1d_segments", "PMMStack",
           "pmm_efficiency_1d_slanted", "pmm_jones_1d_slanted",
           "pmm_jones_1d_slanted_segments", "pmm_1d",
           "grating_convergence_class", "classify_from_grating"]
