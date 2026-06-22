"""Eigenmode-expansion (EME) lateral-cascade solvers — 2-D Bloch modes / band
structure from 1-D-x strip eigenmodes.

EME is the **mode / band-structure** peer of the diffraction solvers
(:mod:`lumenairy.elements.pmm`, :mod:`lumenairy.elements.rcwa`): given a layer
that is periodic in ``y`` and invariant in ``z`` (an ``eps(x, y)`` cell), it
returns the propagation eigenvalues ``qz^2`` and modal fields of the 2-D Bloch
modes, built by cascading 1-D-x strip eigenmodes laterally.  It is a *mode*
solver, not a diffraction-efficiency solver — for crossed-grating efficiencies
use ``rcwa_efficiency_2d`` / ``pmm_efficiency_2d`` (the EME diffraction route is a
documented dead end; see ``eme_diffraction``).

Two physics tiers:

* **Scalar** (:mod:`eme_2d`): the Helmholtz lateral cascade — ``layer_modes`` finds
  ``qz^2`` via a stable S-matrix cascade + ``sigma_min`` root-find; ``ref_2d_modes``
  is the spurious-free 2-D-FD oracle.
* **Vector / full-Maxwell** (:mod:`eme_2d_vector`): TE/TM-hybridized
  ``layer_vector_modes`` from a Berreman-in-y strip generator + a well-conditioned
  global block-``G`` singularity test.  Universality features: arbitrary geometry
  (``eps_xy_to_strips``), full out-of-plane 3x3 anisotropic strips (diagonal +
  ``exz`` + ``eyz``, validated vs Berreman / the Christoffel determinant), scalar
  magnetic ``mu``, a banded O(S) ``sigma_min`` (``solver="banded"``), and a seeded
  Beyn refiner (``beyn_refine_complex``) for complex / lossy / leaky ``qz^2``.

See ``README.md`` for scope, validation, and the documented research-grade limits
(the eyz *layer* finder is gated; high-degeneracy / autonomous-complex finding use
the oracle).  A differentiable (JAX) twin of the eig-based mode solvers lives in
:mod:`_jax_modes`.
"""
from __future__ import annotations

from .eme_2d import (
    cell_smatrix,
    dispersion,
    layer_modes,
    mode_field,
    ref_2d_modes,
    strip_x_modes,
)
from .eme_2d_vector import (
    beyn_refine_complex,
    dispersion_vec,
    eps_xy_to_strips,
    layer_vector_modes,
    layer_vector_modes_complex,
    mode_field_vec,
    ref_2d_modes_vector,
    strip_vector_modes,
    strips_to_eps_xy,  # the general (tensor-aware) rasterizer
)
from .eme_diffraction import (
    diffraction_eme,
    diffraction_fd,
    mode_match,
    plane_wave_orders,
    pw_matrix,
)

__all__ = [
    # scalar Helmholtz cascade
    "strip_x_modes",
    "cell_smatrix",
    "dispersion",
    "layer_modes",
    "mode_field",
    "ref_2d_modes",
    # vector / full-Maxwell cascade
    "strip_vector_modes",
    "dispersion_vec",
    "layer_vector_modes",
    "mode_field_vec",
    "beyn_refine_complex",
    "layer_vector_modes_complex",
    "ref_2d_modes_vector",
    # geometry helpers
    "strips_to_eps_xy",
    "eps_xy_to_strips",
    # mode-matching diffraction (documented dead end for structured layers)
    "plane_wave_orders",
    "pw_matrix",
    "mode_match",
    "diffraction_fd",
    "diffraction_eme",
]
