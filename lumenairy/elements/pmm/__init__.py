# ruff: noqa: F401, F403, F405  -- package facade re-exports every submodule name
"""Polynomial Modal Method (PMM) -- non-Fourier spectral-element modal solver.

Split into submodules for readability (the public API is unchanged):
``_core`` (basis + slant generators + S-matrix + far-field), ``oned`` (the
public 1-D entry points), ``stack`` (PMMStack), ``twod`` (the hybrid 2-D PMM,
``pmm_efficiency_2d`` / ``pmm_efficiency_2d_cell``), ``twod_jones`` (the
anisotropic 2-D Jones solver ``pmm_jones_2d``), ``twod_staggered`` (the
no-floor staggered 2-D PMM, ``pmm_efficiency_2d_staggered`` /
``pmm_jones_2d_staggered``).  Re-exported so
``lumenairy.elements.pmm.<name>`` resolves for every public name AND the
test-imported / monkeypatched privates (the slant dispatch spies these).

REGISTRATION: THE CELL IS SOLVED WHERE IT IS DECLARED (v5.50.1).  A segment sits
on its ``[lo, hi)`` from ``x = 0``, an ``eps_cell`` pixel / segment at its
index, a shape at its centre, a slanted wall at ``x_top + t z`` (``z`` down
from the layer top) -- in the frame of ``theta`` / ``phi``, the order labels
``kx_m = kx0 + m lambda / P`` and every field output.  From v5.8.0 to v5.50.0
every path that takes lateral structure solved the POINT MIRROR
``eps(-x, -y)`` instead (the PMM side of the RCWA fault), reached three ways:
the 1-D nodal cores carried calibrations against the then-mirrored RCWA (a
``[::-1]`` in ``_core._segment_elem_bnds`` and the slant-convection sign); the
2-D hybrid ran public-form operators on conjugated (internal-gauge) cells; the
staggered family glued its Bloch basis with ``exp(-i alpha0 x)``, a
180-degree-rotated solve.  The bridges now live in ONE place per core family:
``_core._internal_mirror_1d`` (the internal-gauge covariant 1-D core),
``twod._build_axis`` / ``twod._internal_slant`` (the hybrid's internal frame)
and ``twod_staggered._stag_bloch_tau`` (the staggered carrier), with the
frame-walk anchor of every sheared cascade in ``_core._frame_anchor_phase``.
Cells symmetric about the origin (or ``P/2``), uniform layers, the binary 1-D
entries (whose mirror is a translation) and BOR are unaffected.  See
``tests/unit/test_pmm_point_mirror_fix.py``."""
from ._core import *
from .conical import pmm_jones_1d_conical, pmm_jones_1d_conical_tensor
from .oned import *
from .shapes2d import *
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
           "pmm_efficiency_2d_staggered", "pmm_jones_2d_staggered",
           "PMM2DStack", "PMM2DStackHybrid", "PMM2DStack_hybrid",
           "PMM2DStackPure",
           "Rect", "FilletRect", "Circle", "Ellipse", "SinusoidalWall",
           "Shape2D", "compile_shapes",
           "PreparedPMM2D", "prepare_pmm_2d", "prepare_pmm_2d_cell",
           "pmm_efficiency_2d_vs_wavelength",
           "pmm_efficiency_2d_cell_vs_wavelength",
           "pmm_2d_order_drift"]
