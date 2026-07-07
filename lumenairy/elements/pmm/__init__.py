# ruff: noqa: F401, F403, F405  -- package facade re-exports every submodule name
"""Polynomial Modal Method (PMM) -- non-Fourier spectral-element modal solver.

Split into submodules for readability (the public API is unchanged):
``_core`` (basis + slant generators + S-matrix + far-field), ``oned`` (the
public 1-D entry points), ``stack`` (PMMStack), ``twod`` (the hybrid 2-D PMM,
``pmm_efficiency_2d`` / ``pmm_efficiency_2d_cell``), ``twod_jones`` (the
anisotropic 2-D Jones solver ``pmm_jones_2d``), ``twod_staggered`` (the
no-floor staggered 2-D PMM, ``pmm_efficiency_2d_staggered``).  Re-exported so
``lumenairy.elements.pmm.<name>`` resolves for every public name AND the
test-imported / monkeypatched privates (the slant dispatch spies these)."""
from ._core import *
from .conical import pmm_jones_1d_conical, pmm_jones_1d_conical_tensor
from .oned import *
from .stack import *
from .stack2d import *
from .stack2d_pure import *
from .twod import *
from .twod_jones import *
from .twod_staggered import *

__all__ = ["pmm_efficiency_1d", "pmm_efficiency_1d_jax",
           "pmm_efficiency_1d_segments",
           "pmm_jones_1d", "pmm_jones_1d_conical",
           "pmm_jones_1d_conical_tensor",
           "pmm_jones_1d_segments", "PMMStack",
           "pmm_efficiency_1d_slanted", "pmm_jones_1d_slanted",
           "pmm_jones_1d_slanted_segments", "pmm_1d",
           "pmm_efficiency_1d_vs_wavelength", "pmm_jones_1d_vs_wavelength",
           "pmm_jones_1d_segments_vs_wavelength",
           "pmm_graded_segments",
           "grating_convergence_class", "classify_from_grating",
           "pmm_efficiency_2d", "pmm_efficiency_2d_cell", "pmm_jones_2d",
           "pmm_efficiency_2d_staggered",
           "PMM2DStack", "PMM2DStackHybrid", "PMM2DStack_hybrid",
           "PMM2DStackPure",
           "PreparedPMM2D", "prepare_pmm_2d", "prepare_pmm_2d_cell",
           "pmm_efficiency_2d_vs_wavelength",
           "pmm_efficiency_2d_cell_vs_wavelength"]
