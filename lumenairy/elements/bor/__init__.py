"""Body-of-revolution (axisymmetric / cylindrical) Polynomial Modal Method.

The BOR-PMM is the **cylindrical-coordinate** peer of the Cartesian
:mod:`lumenairy.elements.pmm` / :mod:`lumenairy.elements.rcwa` solvers: for a
structure that is invariant under rotation about an axis (a concentric-ring
grating, a fiber, an axisymmetric diffractive element), the fields separate as
``exp(i m phi + i q z)`` and the problem reduces to a 1-D *radial* eigenproblem at
each azimuthal order ``m``, cascaded in ``z`` by a Redheffer S-matrix -- the
direct analog of the 1-D PMM lateral solve.

Headline API -- :class:`BORStack` (mirrors ``PMMStack`` in cylindrical
coordinates)::

    s = BORStack(Rbig=48.0, m=1, N=256, n_superstrate=1.41, n_substrate=1.41)
    s.add_layer(0.5, rings=(3.0, 0.5, 2.45, 1.41))   # (period, duty, n_ridge, n_groove)
    s.set_source(k0=2.0)
    res = s.solve()        # res['R'], res['T'] per propagating order, res['angles']

Built on the **Yee div-conforming (staggered) radial discretization** (``E_r`` on
faces, ``E_phi``/``E_z`` on nodes), which makes the discrete ``curl.grad == 0`` to
machine precision and so eliminates the curl-curl gradient (spurious) mode sea --
the cascade then conserves energy to machine precision.

Validation: the radial operator matches Bessel zeros to ~1e-13 (M1); the coupled
vector eigensolve matches an exact open-cladding fiber dispersion oracle (M2,
``fiber_modes``); the radial PML keeps bound modes invariant (M3); the z-cascade
reproduces the analytic slab Fabry-Perot (M4); and **GATE 4** -- a concentric ring
grating's per-order diffraction efficiency matches the rigorous planar
``pmm_efficiency_1d`` at each mode's local oblique angle, to well under the
documented few-percent bar on the staggered basis (the residual is the 2nd-order
FD floor).  See ``README.md`` for the full milestone ledger.

Lower-level building blocks (``solve`` / ``build_layer`` / ``cascade`` /
``interface_smatrix`` ...) live in the submodules (``bor_solve``, ``zcascade``);
the validation oracles are ``fiber_modes`` (open fiber) / ``stepindex_modes``
(closed PEC cavity).

TWO REFUSALS ARE ARMED (5.45.1), both on measured two-sided bars and both behind
a fail-before switch; each is exported here so a caller can catch it by name:

* :class:`BORSemMeshError` -- the ``+-1`` enrichment window MANUFACTURED a radial
  element no single layer asked for, and the affected layer's spectrum shows the
  spurious axial wavenumbers that proves it.  Switch:
  ``lumenairy.elements.bor._sem_contract.BOR_SEM_MESH_GUARD``.
* :class:`BORNodalPassivityError` -- the LEGACY nodal basis
  (``bor_solve.build_layer(basis='nodal')``) returned a non-physical ``R + T``
  on a provably passive lossless stack.  Switch:
  ``lumenairy.elements.bor.bor_solve.BOR_NODAL_PASSIVITY_GUARD``.

See ``docs/audits/BUILD_BOR_MULTILAYER_GUARDS_2026_09_12.md`` for the
populations both bars were derived on.
"""
from __future__ import annotations

# The two armed refusals, imported here so a caller can catch them by name --
# ``from lumenairy.elements.bor import BORSemMeshError``.  DELIBERATELY NOT in
# ``__all__``: no error class in this library is a top-level public name (the
# Cartesian peers are ``rcwa/_core._EnergyError`` and ``_ConditioningError``),
# and the v4.16.0 ``__all__``-symmetry walker requires every submodule
# ``__all__`` entry to be re-exported from ``lumenairy/__init__.py``.  Keeping
# them out of ``__all__`` keeps that convention while leaving the names
# importable and catchable.
from ._sem_contract import BORSemMeshError  # noqa: F401
from .bor_solve import BORNodalPassivityError  # noqa: F401
from .bor_stack import BORStack
from .coupled_radial_eigensolver import guided_modes, radial_coupled_modes
from .farfield import far_field_angles, fourier_bessel, order_power_fractions
from .fiber_oracle import fiber_modes
from .radial_eigensolver import radial_spectrum
from .sem_radial import (
    SemRadialMesh,
    equalize_meshes,
    sem_interface_smatrix,
    sem_layer_modes,
)
from .stepindex_oracle import stepindex_modes
from .zcascade import layer_modes

__all__ = [
    # headline axisymmetric stack solver
    "BORStack",
    # radial eigensolvers (scalar M1 / coupled-vector M2 / guided subset)
    "radial_spectrum",
    "radial_coupled_modes",
    "guided_modes",
    # per-layer modal basis
    "layer_modes",
    # SEM radial basis (BORStack basis="sem" building blocks)
    "SemRadialMesh",
    "sem_layer_modes",
    "sem_interface_smatrix",
    "equalize_meshes",
    # cylindrical far-field (Fourier-Bessel / Hankel orders)
    "fourier_bessel",
    "far_field_angles",
    "order_power_fractions",
    # analytic validation oracles
    "fiber_modes",
    "stepindex_modes",
]
