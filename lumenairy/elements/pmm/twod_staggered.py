"""Canonical no-floor 2-D crossed-grating PMM -- Granet 2023 staggered
modified-Legendre basis.
================================================================================

A NON-Fourier rigorous modal solver for a 2-D (doubly periodic) crossed grating
of axis-aligned rectangular pillars -- the 2-D analogue of
:func:`~lumenairy.elements.pmm.pmm_efficiency_1d`, and the *no-floor* counterpart
of the FMM-floored hybrid :func:`~lumenairy.elements.pmm.twod.pmm_efficiency_2d`.

This implements the FAITHFUL staggered modified-Legendre basis of Granet,
"Modal spectral element method with modified Legendre polynomials to analyze
binary crossed gratings," J. Opt. Soc. Am. A 40, 652 (2023) (Eqs. 23-34) -- NOT
a DG-broken-space + flux-penalty substitute.  Continuity is EMBEDDED in the
basis via shared hats (Eq. 32) and the Bloch periodic hat (Eq. 33); there is NO
stabilization parameter, so there is NO mechanism to inject spurious modes.  The
two staggered 1-D sets ``Btilde`` (continuous, C0) and ``B`` (its discontinuous
partner, equal cardinality) realize the 2-D Li inverse-rule placement BY
CONSTRUCTION: each transverse field component is continuous across the wall it
crosses and reduced (bubble) in the other direction.  The longitudinal field is
slaved by ``div(D) = 0`` (Eq. 16-18; the ``-K_tz (eps33)^-1 K_zt`` Schur term in
``L``), which removes the longitudinal spurious sea.

Why this over the FMM-floored hybrid
-------------------------------------
* **No Fourier floor.**  Every region (cover, film, substrate) is solved in the
  SAME staggered modal basis at the same dimension, so every interface is a
  SQUARE modal match (the 1-D PMM architecture lifted to 2-D); the
  Rayleigh-order projection is applied ONCE, FORWARD only, at the far field.
  The energy balance is therefore ``n_orders``-INDEPENDENT and tracks ONLY the
  modal degree ``M`` (verified to ~1e-13 machine precision).  The hybrid, by
  contrast, projects the layer into a truncated Fourier basis before the
  eigensolve and inherits the FMM ``n_orders`` floor.
* **Exact sidewalls + position invariance.**  The pillar walls land on element
  boundaries (Eq. 26), so ``eps`` is exact per element (no Gibbs); the total
  efficiencies are invariant to the pillar's position in the cell.

Segment boundaries -- UNIFORM or ARBITRARY (Eq. 31)
----------------------------------------------------
Granet Eq. 31 maps EACH segment individually,
``x = 0.5 (x_{n+1} - x_n) u + 0.5 (x_{n+1} + x_n)``, so nothing in the
formulation requires the segments to be equal.  :class:`Basis1D` and
:class:`Granet2DTransverseE` therefore take their per-axis segmentation as
EITHER an ``int N`` -- the uniform lattice, and then every matrix they build is
BIT-IDENTICAL to the pre-2026-09-11 library -- or an increasing ``(N + 1,)``
array of wall positions.  The generalization is the scalar jacobian ``J``
becoming a per-segment ``J_n``, at exactly four sites
(:meth:`Basis1D._global_matrix`, :func:`_global_pair_segmat`,
:meth:`Granet2DTransverseE._eps_dir` and :func:`_stag_fourier_projection`);
everything else -- the reference-interval elementary matrices, the hat glue of
Eqs. 32-33 including the Bloch ``tau`` hat, and the de Rham property
``d(Btilde) subset span(B)`` that makes this basis spurious-free -- is
per-segment and scale-free.

This is what makes an arbitrary TAPER representable: a wall that moves 1.8 nm
per z-slice on a 700 nm period needs ``N ~ 390`` uniform segments (``q >= 1170``,
an eigenproblem above 2.7e+06) and THREE non-uniform ones.

Per-layer element grids (the L2 mortar)
----------------------------------------
:class:`StagGridOps` / :class:`StagCrossOps`, :func:`_stag_cross_mass_1d` and
:func:`_stag_kron_apply` are the basis-level ingredients of
``PMM2DStackPure(..., layer_grids='per-layer')``, where each layer keeps its own
segmentation and adjacent grids are coupled weakly.  The cross-mass between two
partitions is EXACT by Gauss-Legendre on their UNION (on each union
sub-interval both sides are polynomials of degree ``<= M-1``), it is COMPLEX
(the Bloch ``tau`` glue lives in the basis), and its 2-D form factors exactly
as a Kronecker product -- which is why it is never materialised (900x memory at
``N = (6, 12), M = 6``).  The interface algebra itself lives beside its 1-D
sibling in :mod:`lumenairy.elements.pmm._core`.

Anisotropy -- FULL ``(3, 3)`` tensors, in-plane AND out-of-plane
----------------------------------------------------------------
``eps_cell`` may be a scalar ``(Nx, Ny)`` map (the isotropic solver
:func:`pmm_efficiency_2d_staggered`, unchanged bit-for-bit) OR a ``(Nx, Ny,
3, 3)`` tensor map driven through :func:`pmm_jones_2d_staggered`.  Two
formulations sit behind one entry, dispatched on the cell itself:

* **IN-PLANE (block-form, Granet Eq. 7** ``[[e11, e12, 0], [e21, e22, 0],
  [0, 0, e33]]`` **).**  The paper's GENERAL case, of which the shipped
  isotropic solver was the reduction: the SAME ``2 q^2`` second-order
  eigenproblem, the same ``[W; -V] <-> -lam`` symmetry, the same square
  Redheffer cascade and the same far field.  What it adds is the two MIXED
  ``[eps_t]`` masses (Eq. 40), the ``e33``-weighted ``Meps33`` (Eq. 41) and
  the second term in each ``K_zt`` column (Eq. 44, the divergence of
  ``D_t = eps_t E_t``); the Eq. 25 H-partner picks up the same two mixed
  blocks.  Rotated-uniaxial (liquid-crystal) and GYROTROPIC
  (``e12 = -e21 = i b``, Hermitian, lossless) media are both in scope.
* **OUT-OF-PLANE (``e_xz``/``e_yz``/``e_zx``/``e_zy`` above a RELATIVE
  ``1e-12`` floor)** -- a tilted-director liquid crystal.  This breaks the
  paper's Eq. 16 (``div D = 0`` no longer slaves ``E3`` algebraically: ``E3``
  appears under ``d_t`` through ``e13``/``e23`` while ``gamma`` multiplies
  ``e31``/``e32``), so it is not a second-order problem.  It routes instead to
  the FIRST-ORDER staggered generator on ``[E1; E2; G1; G2]``
  (:meth:`Granet2DTransverseE._assemble_oop`, dimension ``4 q^2``), whose
  forward and backward modes are genuinely distinct
  (:func:`_region_modes_oop`) and which therefore cascades through the
  GENERALIZED S-matrix.  The de Rham placement, the far field, the union grid
  and the ``layer_absorption`` budget all carry over unchanged.  COST:
  1.3-2.0x the in-plane region solve in wall time and ~3x its peak working
  set at equal ``M`` (measured; the dimension doubles but the pencil is
  Cholesky-whitened to a standard eig while the in-plane path pays a QZ).
  At NORMAL incidence on a cell that is its own PARITY image, ``symmetry``
  (on by default) takes that ``4 q^2`` eig down to ONE ``2 q^2`` eig
  (:func:`_stag_block_eig`): MEASURED 3.3-4.2x on the region solve and
  1.5x (single layer) to 1.9x (a three-layer stack) end to end, with the
  structure verified on the assembled pencil every call and a BIT-IDENTICAL
  dense fallback everywhere it does not hold.

Both routes keep the ISOTROPIC half-spaces (the Rayleigh match is scalar), as
the hybrid does, and a cell whose out-of-plane entries are float noise stays
BIT-IDENTICAL to the in-plane path (the dispatch floor is relative).

MAGNETIC media -- a block-form PERMEABILITY tensor
--------------------------------------------------
``mu_cell`` (``None`` = nonmagnetic, and then every operator below is
BIT-IDENTICAL to the nonmagnetic path) carries a scalar ``(Nx, Ny)`` or
BLOCK-FORM ``(Nx, Ny, 3, 3)`` relative permeability, Granet Eq. 6
``[[m11, m12, 0], [m21, m22, 0], [0, 0, m33]]``.  The paper's equations are
ALREADY the magnetic ones: with ``[chi_t] = [mu_t]^-1`` (the POINTWISE 2x2
inverse -- exact for the piecewise-constant cells this basis is built on) and
``chi33 = 1/m33``,

  * ``R = C[chi_t]C`` (Eq. 24, Appendix-A Eq. 39) -- the C-rotation SWAPS the
    transverse indices, so ``R11`` carries ``chi22`` and ``R22`` ``chi11``,
    and the two MIXED blocks (``chi21`` in V1xV2, ``chi12`` in V2xV1) are
    kron'd from the UNLIKE-set 1-D masses exactly like the Eq. 40 eps blocks;
  * ``K_tz = C[chi_t][d2; -d1]`` (Eq. 21, A43) -- row 1 becomes
    ``-chi22 d1 + chi21 d2`` and row 2 ``-chi11 d2 + chi12 d1``;
  * ``S_tt`` is the ``chi33``-weighted curl-curl (Eq. 20, A42): the Vw-space
    inner product between the two curls becomes ``chi33``-weighted, i.e. the
    mimetic middle operator ``Gw^-1 -> Gw^-1 Gw_chi Gw^-1``.

``[eps_t]``, ``Meps33`` and ``K_zt`` are untouched (they carry permittivity
only), the pencil keeps its ``2 q^2`` dimension, and ``G = -R`` stays Hermitian
positive definite for a Hermitian positive-definite ``[mu_t]``.

THE ONE TRAP.  With ``chi_t = I`` the shipped code uses a SINGLE object
(``-Rmat``) for two roles: the pencil's right-hand matrix AND the block field
Gram that recovers the Eq.-25 H partners.  With ``chi_t != I`` they are
DIFFERENT operators -- Eq. 25 (``gamma C [H1;H2] = [k^2 eps_t + S_tt][E1;E2]``)
carries no ``chi_t`` at all -- so the plain Gram is retained separately
(``Ggram_blocks``, the two ``q^2`` diagonal blocks) and :func:`_region_modes`
projects with it.  Collapsing the two applies ``[chi_t]^-1`` to every H partner:
an interface error invisible to the eigenvalues and to any renormalised energy
check, and measured at 2.1e-01 against the analytic oracle (build doc
``BUILD_PMM2D_STAGGERED_MAGNETIC_2026_09_10.md`` M1b) where the correct
separation reads 1.9e-14.

Scope of the magnetic route: IN-PLANE (block-form) only -- an out-of-plane
``mu``, or a ``mu`` together with an out-of-plane ``eps``, raises
``NotImplementedError`` (the first-order out-of-plane generator has no
permeability blocks).  Half-spaces stay NONMAGNETIC: the Rayleigh flux
normalisation and the incident-amplitude overlap both assume the vacuum wave
impedance, so ``mu_superstrate`` / ``mu_substrate`` exist only to RAISE.  A
uniform magnetic layer cannot ride the shared eps-free geometric eig
(:func:`_homog_geom_cache` raises) and takes its own region eig, deduped by
``(eps bytes, mu bytes)``.  Losslessness now means Hermitian eps AND Hermitian
mu (a gyrotropic ``m12 = -m21 = i b`` absorbs nothing).

ACCURACY NOTE for out-of-plane cells with a RE-ENTRANT (270-degree) corner:
the uniform, straight-walled and convex regimes are machine-exact to
oracle-limited, and a (3, 3) L-shaped feature -- the hardest cell measured --
lands at the two Fourier oracles' OWN mutual spread (build doc
``BUILD_PMM2D_STAGGERED_OOP_2026_09_09.md`` T6), but neither Fourier arm is
converged there, so no engine in the suite currently pins that regime better
than ~1e-4 on ``R`` per order.

Scope / limitations
-------------------
* **Axis-aligned RECTANGULAR pillars only** -- the walls must coincide with the
  segment boundaries of the ``(Nx, Ny)`` ``eps_cell`` grid (Eq. 26).  CURVED
  boundaries need Granet's transfinite curved-quad mapping (not implemented).
  A constant TILT of the walls IS supported: see SLANT below.
* **Corner-capped.**  A right-angle dielectric pillar has field singularities at
  its four corners, so the bound-mode (and hence efficiency) convergence is
  ALGEBRAIC, not spectral -- monotone with NO floor, but at-best RCWA-parity
  per DOF on vertical pillars (the universal corner cap; see
  :func:`~lumenairy.elements.pmm.grating_convergence_class`).  The genuine win is
  accuracy QUALITY (no-floor, exact sidewalls, position invariance, pinning the
  value RCWA converges toward), not raw per-DOF speed.  A smooth-field region
  (vacuum / homogeneous) converges spectrally.
* Single layer per entry (cascade with
  :class:`~lumenairy.elements.pmm.PMM2DStackPure`); scalar TE/TM through
  :func:`pmm_efficiency_2d_staggered`, full ``(3, 3)`` tensors through
  :func:`pmm_jones_2d_staggered`; NumPy/SciPy dense eig (not
  JAX-differentiable).
* A UNIFORM TENSOR region is NOT eps-free-separable (``K_zt`` mixes e11/e21
  while ``Meps33`` carries e33 alone), so it takes a full region eig like a
  patterned cell -- the shared geometric eig :func:`_homog_geom_cache` is
  scalar-only and raises on a tensor assembly.
* ANISOTROPIC HALF-SPACES and the JAX twin stay out of scope.

SLANT (roadmap Phase D, 2026-09-10)
-----------------------------------
``slant=(t_x, t_y)`` on :func:`pmm_jones_2d_staggered` and on
:meth:`~lumenairy.elements.pmm.PMM2DStackPure.add_layer` makes a layer ONE
EXACT SLANTED region instead of a z-staircase: the whole cross-section
translates laterally by ``t * depth`` from the layer's TOP face to its bottom,
``eps_cell`` being the cross-section at the top.  ``t`` is a TANGENT
(``t_x = tan(wall_tilt_x)``) -- the SAME public convention as
:meth:`~lumenairy.elements.pmm.PMM2DStackHybrid.add_layer` and the 1-D
``slant_angle`` entries, so a layer moves between the engines unchanged --
and ``0`` / ``None`` is BIT-IDENTICAL to the pre-slant library.

It is EXACT at any slant magnitude and costs ONE eigensolve for the whole
layer.  In the sheared frame (``u = x - t_x w``, ``v = y - t_y w``, ``w = z``)
``det J = 1``, so ``sqrt(g) = 1``, ``mu^33 = 1`` and ``eps^33 = eps_zz``; the
shear is then exactly a POINTWISE congruence ``eps -> A^-1 eps A^-T`` on the
cell tensor plus SIX extra Galerkin blocks on the first-order out-of-plane
generator (:meth:`Granet2DTransverseE._assemble_oop`).  The COVARIANT field
components are used, not the lab-Cartesian ones with a chain-rule convection:
across a slanted wall the continuous combination is ``t . E_t + E_z``, not
``E_z``, so the covariant components have exactly the vertical continuity
structure in the frame and the shipped staggered de Rham placement is conformal
for them at any slant.  A slanted cell therefore always runs the ``4 q^2``
generator and the generalized cascade -- its covariant tensor has out-of-plane
entries even when the cell is scalar -- and slant x ANISOTROPY (including
out-of-plane) is the same line of code, a combination no other 2-D engine in
this suite covers.

The one piece of bookkeeping a shear adds is the FRAME-ANCHOR PHASE: the frame
is anchored at each slanted layer's TOP, so the transmitted amplitudes carry
one unimodular phase per order, ``exp(-i alpha_m . t d)``.  R, T and the
REFLECTION Jones are exact without it.

A SHEAR IS NOT A TAPER.  A shear is a tilted axis with a CONSTANT
cross-section; no shear absorbs a dilation (a taper's ``sqrt(g)`` is
z-dependent, which brings back a dilation generator, a non-normal pencil with
no valid mode selector and a distorted far field).  A shrinking cross-section
still needs a z-staircase.

Out of scope for the slant, all raising: MIXED slants between PATTERNED layers,
a mix of vertical and slanted layers ABOVE a pattern, ``mu`` together with a
slant, ``retain_internal`` on a slanted stack, and
:func:`pmm_efficiency_2d_staggered` (single-polarization efficiencies are not
well-posed for a cell that is out-of-plane in the frame).  Derivation and every
measured number: ``docs/audits/EXPERIMENT_PMM2D_STAGGERED_SLANT_2026_09_10.md``
and ``docs/audits/BUILD_PMM2D_STAGGERED_SLANT_2026_09_10.md``.

Conventions match the rest of the library: PUBLIC ``exp(-i w t)`` (``n = n + i
kappa``, ``Im eps > 0`` for loss), forward ``exp(+i kz z)``, ``Im(kz) >= 0``.
Eigenvalue ``gamma^2/k0^2 = n_eff^2``; derivatives are ``(1/k0) d/dx``.

Equations implemented (Granet 2023, verified against the paper):
  Eq.23-24 :  -gamma^2 R [E1;E2] = L [E1;E2];  R = C[chi_t]C (nonmagnetic
              chi_t=I -> R=C@C=-I; magnetic -> the four chi-weighted blocks
              of Appendix-A Eq.39);  L = k^2[eps_t] + S_tt - K_tz(eps33)^-1 K_zt
  Eq.20-22 :  S_tt = [d2;-d1] chi33 [d2,-d1];  K_tz = C[chi_t][d2;-d1]
              (nonmagnetic C[d2;-d1]);
              K_zt = [d1 e11 + d2 e21, d2 e22 + d1 e12]  (isotropic
              reduction: [d1 eps, d2 eps])
  Eq.16-18 :  E3 slaved by div(D)=0 (the Schur term)
  Eq.30-33 :  modified Legendre Ltilde_m; continuity hats; Bloch periodic hat
  Eq.34    :  staggered tensor expansion E1=B(x)Btilde, E2=Btilde(x)B,
              E3=Btilde(x)Btilde
  Eq.39-44 :  Appendix-A bra-ket matrix elements (real inner product INT x* y).

Out-of-plane (Stage B) -- the first-order generator that replaces Eq. 23-25
when ``e13/e23/e31/e32 != 0``; derivation, spurious census and the GO verdict
in ``docs/audits/EXPERIMENT_PMM2D_STAGGERED_OOP_2026_09_09.md`` (candidate
(a)), integration measurements in
``docs/audits/BUILD_PMM2D_STAGGERED_OOP_2026_09_09.md``.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as sla
from numpy.polynomial.legendre import leggauss

from ...cache import ByteBudgetedLRU as _ByteBudgetedLRU
from ..rcwa import Efficiency2D  # cross-suite 2-D result (unpacks (o,R,T), carries .dof)
from ..rcwa._core import (  # shared flux projection + the OOP mode selector
    _norm_slant_pair,
    _project_efficiency,
    _select_forward_flux,
    _slant_is_zero,
)

# UNCHANGED S-matrix algebra from the shipped 1-D PMM -- every region's modal
# matrix W is square (same staggered dimension), so these are plain square
# solves (no pseudo-inverse, no weighting).
from ._core import (
    _forward_branch_flip,
    _guarded_lstsq,
    _interface_smatrix,
    _propagation_smatrix,
    _redheffer_star,
)

# Tensor-entry guards shared with the hybrid 2-D Jones solver (the RELATIVE
# out-of-plane floor + the e_zz != 0 precondition are one contract across the
# two engines -- reused, not copied, so the floor cannot drift apart).
from .twod_jones import _require_nonzero_ezz, _tile_is_offplane

__all__ = ["pmm_efficiency_2d_staggered", "pmm_jones_2d_staggered"]

_C = np.complex128

#: Sign applied to the OUT-OF-PLANE tensor entries (``e13, e23, e31, e32``) in
#: :meth:`Granet2DTransverseE._assemble_oop`.  It is the 180-degree rotation
#: about ``z`` that the shipped gauge carries: the ``Basis1D`` glue
#: ``tau = exp(-i alpha0 p)`` makes the basis run as ``exp(-i alpha0 x)``
#: (MEASURED, ``validation/probe_pmm2d_staggered_oop/g0f_basis_phase_slope.py``)
#: while the far-field kernel and the ``eps_cell`` indexing run the other way.
#: In-plane operators are invariant under that rotation -- which is exactly why
#: it was unobservable for the isotropic and Stage-A paths -- and it is a pure
#: SIGN on the out-of-plane block, so it is applied once, in the assembly.
#: DERIVED BY MEASUREMENT 2026-09-09 (build doc
#: docs/audits/BUILD_PMM2D_STAGGERED_OOP_2026_09_09.md, tables T2/T3): with
#: -1.0 a uniform out-of-plane slab reproduces ``berreman_jones_1d`` to ~1e-15
#: at oblique and conical incidence; with +1.0 the same solve sits at ~1e-03.
#: It is a module constant so a test can walk BOTH arms of that two-sided
#: claim through the shipped code rather than re-deriving it.
_OOP_ROT_SIGN = -1.0

#: Constant relating the out-of-plane generator's magnetic state ``G = i Z0 H``
#: to the Eq.-25 tangential-H partner ``[H1; H2]`` that :func:`_region_modes`
#: and :func:`_homog_region_modes` build for the in-plane regions and the
#: half-spaces; applied in :func:`_region_modes_oop`.  It does NOT cancel
#: anywhere -- not even in a single uniform out-of-plane slab between
#: isotropic half-spaces, because the half-space partners are built by
#: :func:`_homog_region_modes` in the Eq.-25 gauge and the interface match
#: compares the two directly.  MEASURED 2026-09-09 (verify doc
#: docs/audits/VERIFY_PMM2D_STAGGERED_OOP_2026_09_09.md): a uniform
#: out-of-plane slab vs ``berreman_jones_1d`` reads dJones 1.5e-14 with -1j,
#: 3.9e-02 with +1j, and O(1) with +/-1 (R+T = 3.69 / 11.94 -- the pure
#: lossless-trap shape: on every +/-i arm R+T stays exactly 1).  Two gates
#: hold it: that Berreman comparison and the in-plane-reduction gate (build
#: doc table T4: cross terms exactly zero, shipped 3e-14 vs wrong gauge
#: 2.2e-02..3.0e+02).  (An earlier note here claimed the constant cancels in
#: a pure out-of-plane stack; that was refuted by the measurement above.)
_OOP_H_GAUGE = -1j

#: Structural gate for the OUT-OF-PLANE generator's PARITY-sign block
#: reduction (:func:`_stag_block_eig`).  The quantity is exactly what that
#: function computes: ``max|R A R + A| / max|A|`` (and ``max|R B R - B| /
#: max|B|``) on the ASSEMBLED pencil.  MEASURED 2026-09-09 (py3.14.6 /
#: numpy 2.4.4 / scipy 1.17.1 / scipy-openblas, tesla-ryzen), see
#: docs/audits/BUILD_PMM2D_STAGGERED_OOP_BLOCK_EIG_2026_09_10.md table B2:
#: cells that carry the structure (uniform tilted uniaxial, centro-symmetric
#: pillar, non-reciprocal, lossy; (2,2) and (3,3) grids, M 5..8) read
#: 5.8e-16 .. 1.5e-14 on ``A`` and 5.8e-17 .. 2.0e-16 on ``B``; cells that do
#: not (an off-centre pillar, a parity-breaking tensor -- oblique and conical
#: incidence never reach here, the gauge itself refuses) read
#: 6.2e-02 .. 6.9e-01.  1e-10 sits 3.8 decades above the satisfied envelope
#: and 8.8 below the smallest real violation -- the same bar the hybrid's
#: :data:`~lumenairy.elements.rcwa._core._OOP_BLOCK_TOL` uses for the same
#: structure in the Fourier basis.
_STAG_BLOCK_TOL = 1e-10

#: Reconstruction floor for :func:`_stag_block_eig`: the factored eigenvector
#: is ``[up; +/- Yh up / q]``, so a ``q`` at the scale of the spectrum's own
#: roundoff is not reconstructible.  MEASURED 2026-09-09 over the same fixture
#: set: ``min|q| / max|q|`` reads 4.1e-03 .. 9.5e-02 (a staggered region
#: spectrum has no null mode away from a Rayleigh cutoff, which the entry
#: points already warn about), so 1e-13 fires only on a genuinely null mode
#: -> dense fallback.  Mirrors
#: :data:`~lumenairy.elements.rcwa._core._OOP_GAM_FLOOR`.
_STAG_GAM_FLOOR = 1e-13


def _tile_needs_oop(fn_name, tile33):
    """Dispatch gate for a ``(..., 3, 3)`` tensor cell: ``True`` when the cell
    carries OUT-OF-PLANE coupling and must run the first-order ``4 q^2``
    generator (Stage B), ``False`` when it is Granet BLOCK-FORM (Eq. 7,
    ``[[e11, e12, 0], [e21, e22, 0], [0, 0, e33]]``) and stays on the
    second-order ``2 q^2`` pencil.  Raises when ``e33 == 0`` either way (both
    eliminations divide by it).

    The out-of-plane test is RELATIVE (``1e-12 * scale``), shared verbatim with
    the hybrid via :func:`~lumenairy.elements.pmm.twod_jones._tile_is_offplane`:
    a physically in-plane cell built by ROTATING a diagonal tensor carries
    ~1e-16 float noise in the xz/yz/zx/zy slots (``uniaxial_tensor(no, ne,
    pi/2, phi)`` has ``cos(pi/2) = 6.1e-17``), and a strict ``> 0`` would send
    every real liquid-crystal cell down the 1.7-2.4x costlier generator instead
    of the bit-identical in-plane path.
    """
    tile33 = np.asarray(tile33, dtype=_C)
    _require_nonzero_ezz(fn_name, tile33)
    return bool(_tile_is_offplane(tile33))


def _validate_stag_cell(fn_name, eps_cell):
    """Shape + SQUARE-grid + block-form validation shared by every anisotropic
    staggered entry (:func:`pmm_jones_2d_staggered` and
    :meth:`~lumenairy.elements.pmm.PMM2DStackPure.add_layer`), so each names
    ITSELF in its message while enforcing one contract.  A ``(Nx, Ny)`` SCALAR
    map is returned unchanged (it stays on the isotropic assembly); a
    ``(Nx, Ny, 3, 3)`` cell is gated by :func:`_tile_needs_oop`, which
    raises on ``e33 == 0`` and otherwise selects the in-plane or the
    out-of-plane assembly."""
    cell = np.asarray(eps_cell, dtype=_C)
    if cell.ndim == 4:
        if cell.shape[2:] != (3, 3):
            raise ValueError(
                f"{fn_name}: a tensor eps_cell must be (Nx, Ny, 3, 3), got "
                f"shape {cell.shape}.")
        _tile_needs_oop(fn_name, cell)
    elif cell.ndim != 2:
        raise ValueError(
            f"{fn_name}: eps_cell must be a 2-D (Nx, Ny) scalar grid or a "
            f"(Nx, Ny, 3, 3) block-form tensor grid, got shape {cell.shape}.")
    if cell.shape[0] != cell.shape[1]:
        raise ValueError(
            f"{fn_name}: eps_cell must be SQUARE (Nx == Ny; the staggered "
            f"tensor-product basis requires Nx*(M-1) == Ny*(M-1)), got "
            f"{cell.shape}.  Pad the uniform axis into equal segments (e.g. "
            f"tile a (2, 1) cell to (2, 2)).")
    return cell


def _wood_eps_reals(*eps_arrays):
    """The DISTINCT real permittivities a Rayleigh cut-off can sit on, for the
    Wood-anomaly nudge list of :func:`~lumenairy.elements.rcwa._core._grazing_safe_wavelength`.

    ONE rule for both staggered paths (scalar and tensor, 2026-09-10).  The
    nudge exists because the staggered solver degrades like
    ``~1/sqrt(cut-off distance)`` and an EXACT grazing mode crashes the
    interface S-matrix -- and that is true of a cut-off inside a LAYER, not
    just in a half-space, so every region's permittivity belongs on the list.
    Callers pass already-scalar quantities: half-space ``eps``, a scalar
    ``(Nx, Ny)`` cell, or a tensor's principal DIAGONAL (the off-diagonals are
    not cut-offs) -- this helper never sniffs a shape, because a scalar
    ``(3, 3)`` cell is a legal 3x3 segmentation grid.

    Deduplication is numerically inert -- ``_grazing_safe_wavelength`` takes a
    MIN over the list -- and it is what keeps the list O(materials) instead of
    O(Nx*Ny*3) on a fine cell, where the min is evaluated per candidate
    wavelength in a Python loop.
    """
    if not eps_arrays:
        return []
    vals = np.concatenate([np.real(np.asarray(a, dtype=_C)).ravel()
                           for a in eps_arrays])
    return [float(v) for v in np.unique(vals)]


def _slant_congruence(eps33, tx, ty):
    """The pointwise COVARIANT congruence ``eps^{lm} = A^-1 eps_lab A^-T`` of a
    constant x-z / y-z SHEAR, applied to the trailing ``(3, 3)`` axes.

    With the frame anchored at the layer TOP (the shipped 1-D / hybrid
    convention -- ``u = x - t_x w``, ``v = y - t_y w``, ``w = z``)::

        A = d(x, y, z) / d(u, v, w) = [[1, 0, t_x], [0, 1, t_y], [0, 0, 1]]

    ``A`` is unit upper-triangular, so ``det A = 1`` EXACTLY at any slant and
    ``sqrt(g) = 1``: the mass matrices are untouched, there is no dilation
    generator, and ``eps^{33} = eps_zz`` is UNCHANGED by the congruence -- which
    is what lets :meth:`Granet2DTransverseE._assemble_oop` keep its pointwise
    ``e33``-Schur and its strong ``G3`` elimination verbatim.  ``mu^{lm} =
    g^{lm}`` is likewise never assembled: the derivation
    (``docs/audits/EXPERIMENT_PMM2D_STAGGERED_SLANT_2026_09_10.md`` S1.3) shows
    its whole content is the two ``t G^3`` terms in the E rows -- where ``G^3``
    is already eliminated strongly -- and the ``t_x G_1 + t_y G_2`` inside
    ``G_3cov`` in the G rows, i.e. the six extra Galerkin blocks.

    An ISOTROPIC slanted cell becomes an OUT-OF-PLANE tensor cell in the frame
    (``eps^{13} = -t_x eps``), which is why a slant always takes the
    first-order ``4 q^2`` generator and never the ``2 q^2`` in-plane pencil.

    ``tx = ty = 0`` returns the input unchanged bit-for-bit (``A = I``).
    """
    e = np.asarray(eps33, dtype=_C)
    if tx == 0.0 and ty == 0.0:
        return e
    Ai = np.array([[1.0, 0.0, -tx], [0.0, 1.0, -ty], [0.0, 0.0, 1.0]],
                  dtype=_C)
    return np.einsum("mp,...pq,nq->...mn", Ai, e, Ai)


def _slant_rot_gauge(eps33, tx, ty):
    """``(eps_cov, t_x_rot, t_y_rot)`` -- the congruence taken in the ROTATED
    gauge :data:`_OOP_ROT_SIGN` runs the out-of-plane assembly in.

    That constant is a 180-degree rotation about ``z`` between the basis and the
    ``eps_cell`` / far-field indexing.  Under it the four out-of-plane tensor
    entries flip sign AND so does the slant vector, and the two are consistent:
    ``eps^{lm}(R eps R, -t) = R eps^{lm}(eps, t) R``.  So BOTH flips are applied
    once, here, and :meth:`Granet2DTransverseE._assemble_oop` then builds in the
    rotated gauge with ``rot = 1`` (it must not apply the rotation twice).

    Getting the ``t`` half wrong is not silent, and the gap is measured on two
    builds: the sheared-frame dispersion gate reads ``3.53e-02 .. 1.13e-01`` on
    the un-rotated (``a-``) arms against ``1.4e-14 .. 6.2e-14`` on the physical
    one (``docs/audits/BUILD_PMM2D_STAGGERED_SLANT_2026_09_10.md`` table B4a;
    ``tests/unit/test_pmm2d_staggered_slant.py`` walks both).
    """
    e = np.array(eps33, dtype=_C, copy=True)
    rot = _OOP_ROT_SIGN
    e[..., 0, 2] *= rot
    e[..., 1, 2] *= rot
    e[..., 2, 0] *= rot
    e[..., 2, 1] *= rot
    txr, tyr = rot * float(tx), rot * float(ty)
    return _slant_congruence(e, txr, tyr), txr, tyr


def _require_inplane_mu(fn_name, tile33):
    """Block-form gate for a ``(..., 3, 3)`` PERMEABILITY tile (Granet Eq. 6,
    ``[[m11, m12, 0], [m21, m22, 0], [0, 0, m33]]``).

    Magnetic anisotropy is IN-PLANE only in this engine: the paper's
    ``R = C[chi_t]C`` / ``K_tz = C[chi_t][d2;-d1]`` / ``S_tt(chi33)`` route
    generalizes the SECOND-ORDER pencil, while the out-of-plane FIRST-ORDER
    generator (:meth:`Granet2DTransverseE._assemble_oop`) carries no ``mu``
    blocks at all (its ``G3``/``E3`` eliminations assume ``mu = 1``).  So an
    out-of-plane ``mu`` raises, at the SAME RELATIVE ``1e-12 * scale`` floor
    the permittivity uses (:func:`~lumenairy.elements.pmm.twod_jones._tile_is_offplane`):
    a physically in-plane tensor built by ROTATING a diagonal one carries
    ~1e-16 float noise in the xz/yz/zx/zy slots and must NOT be rejected.

    ``m33 != 0`` (``chi33 = 1/m33`` enters ``S_tt``) and an INVERTIBLE ``[mu_t]``
    (``chi_t = [mu_t]^-1``, the 2x2 inverse taken pointwise per cell) are the
    two preconditions; both are checked relative to the tile scale."""
    tile33 = np.asarray(tile33, dtype=_C)
    if float(np.min(np.abs(tile33[..., 2, 2]))) < 1e-300:
        raise ValueError(
            f"{fn_name}: m_zz must be nonzero in every region (chi33 = 1/m33 "
            f"weights the S_tt curl-curl operator).")
    if _tile_is_offplane(tile33):
        raise NotImplementedError(
            f"{fn_name}: OUT-OF-PLANE permeability (m_xz / m_yz / m_zx / "
            f"m_zy above the relative 1e-12 floor) is not implemented -- the "
            f"magnetic route generalizes the SECOND-ORDER (2 q^2) block-form "
            f"pencil (Granet Eq. 6: [[m11, m12, 0], [m21, m22, 0], "
            f"[0, 0, m33]]), and the out-of-plane first-order generator has "
            f"no mu blocks.  Pass a BLOCK-FORM mu.")
    det = (tile33[..., 0, 0] * tile33[..., 1, 1]
           - tile33[..., 0, 1] * tile33[..., 1, 0])
    scale = max(float(np.max(np.abs(tile33))), 1.0)
    if float(np.min(np.abs(det))) <= 1e-14 * scale ** 2:
        raise ValueError(
            f"{fn_name}: the transverse block [mu_t] = [[m11, m12], "
            f"[m21, m22]] is SINGULAR in at least one cell "
            f"(min |det| = {float(np.min(np.abs(det))):.3g} vs the relative "
            f"floor {1e-14 * scale ** 2:.3g}); chi_t = [mu_t]^-1 does not "
            f"exist there.")


def _validate_stag_mu(fn_name, mu_cell):
    """Shape + SQUARE-grid + block-form validation for a PERMEABILITY cell,
    the mirror of :func:`_validate_stag_cell`.  Accepts a ``(Nx, Ny)`` SCALAR
    map (isotropic magnetic: ``chi_t = (1/mu) I``, ``chi33 = 1/mu``) or a
    ``(Nx, Ny, 3, 3)`` BLOCK-FORM tensor map, and returns it as complex."""
    cell = np.asarray(mu_cell, dtype=_C)
    if cell.ndim == 4:
        if cell.shape[2:] != (3, 3):
            raise ValueError(
                f"{fn_name}: a tensor mu_cell must be (Nx, Ny, 3, 3), got "
                f"shape {cell.shape}.")
        _require_inplane_mu(fn_name, cell)
    elif cell.ndim == 2:
        if float(np.min(np.abs(cell))) < 1e-300:
            raise ValueError(
                f"{fn_name}: a scalar mu_cell must be nonzero in every cell "
                f"(chi = 1/mu).")
    else:
        raise ValueError(
            f"{fn_name}: mu_cell must be a 2-D (Nx, Ny) scalar grid or a "
            f"(Nx, Ny, 3, 3) block-form tensor grid, got shape {cell.shape}.")
    if cell.shape[0] != cell.shape[1]:
        raise ValueError(
            f"{fn_name}: mu_cell must be SQUARE (Nx == Ny; the staggered "
            f"tensor-product basis requires Nx*(M-1) == Ny*(M-1)), got "
            f"{cell.shape}.")
    return cell


def _require_nonmagnetic_halfspace(fn_name, mu_sup, mu_sub):
    """The half-spaces of this engine are NONMAGNETIC (mu = 1) and isotropic.

    The Rayleigh far field normalises each order's power with the ELECTRIC
    flux factor ``Re(kz)`` and reconstructs ``E_z`` from ``k . E = 0``; a
    magnetic half-space changes the wave impedance ``Z = sqrt(mu/eps)``, hence
    both the flux normalisation and the incident-amplitude overlap.  Rather
    than silently returning efficiencies normalised for vacuum, raise."""
    for name, val in (("mu_superstrate", mu_sup), ("mu_substrate", mu_sub)):
        if val is None:
            continue
        v = np.asarray(val, dtype=_C)
        if v.ndim == 0 and v == 1.0:
            continue
        raise NotImplementedError(
            f"{fn_name}: {name}={val!r} -- MAGNETIC HALF-SPACES are not "
            f"implemented.  The half-spaces are nonmagnetic and isotropic "
            f"(mu = 1): the Rayleigh flux normalisation and the incident "
            f"overlap both assume the vacuum wave impedance.  A magnetic "
            f"medium must be a LAYER (add_layer(..., mu=...) / mu_cell=).")


# === 1-D modified-Legendre staggered basis + 2-D transverse-E eigensolver ===
# (Granet 2023 Eqs.30-34 / 23-24; faithful no-shortcuts staggered basis)
def _legendre_value_deriv(maxdeg, u):
    """L_k(u) and L_k'(u) for k=0..maxdeg at scalar/array u, via recurrence.
    Returns arrays of shape (maxdeg+1,) + u.shape."""
    u = np.asarray(u, dtype=float)
    P = np.zeros((maxdeg + 1,) + u.shape)
    dP = np.zeros((maxdeg + 1,) + u.shape)
    P[0] = 1.0
    if maxdeg >= 1:
        P[1] = u
        dP[1] = 1.0
    for k in range(1, maxdeg):
        P[k + 1] = ((2 * k + 1) * u * P[k] - k * P[k - 1]) / (k + 1)
        dP[k + 1] = ((2 * k + 1) * (P[k] + u * dP[k]) - k * dP[k - 1]) / (k + 1)
    return P, dP


def _modleg_value_deriv(M, u):
    """Modified-Legendre Ltilde_m(u), m=1..M, and derivative, on u in [-1,1].
    Eq.30:  m=1 -> (L0-L1)/2,  m=2 -> (L0+L1)/2,  m>2 -> L_{m-1...} L_m - L_{m-2}.
    We index the returned array 0..M-1 for m=1..M.  For m>2 (array index >=2)
    the polynomial is L_{idx} - L_{idx-2} where idx = m-1 ... careful indexing:
    Granet's m runs 1,2,3,...  L_m is the Legendre poly of degree m.  So
    m=1 uses L0,L1; m=2 uses L0,L1; m=3 -> L_3 - L_1? NO.  Re-read Eq.30:
        Ltilde_m = L_m - L_{m-2} for m>2.
    With m the Legendre DEGREE label, m=3 -> L_3 - L_1, m=4 -> L_4 - L_2, etc.
    But that makes Ltilde_3, Ltilde_4 NOT vanish at the endpoints?  L_m(+/-1) =
    (+/-1)^m, so L_m(1)-L_{m-2}(1) = 1-1 = 0 and L_m(-1)-L_{m-2}(-1) =
    (-1)^m-(-1)^{m-2} = 0.  YES -- they vanish at both ends (bubbles).  Good.
    So array index a=0 -> m=1, a=1 -> m=2, a>=2 -> m=a+1 (bubble L_{a+1}-L_{a-1}).
    """
    maxdeg = max(2, M)        # need up to L_M
    Pp, dPp = _legendre_value_deriv(maxdeg, u)  # values+derivs
    vals = []
    ders = []
    for a in range(M):
        if a == 0:                       # Ltilde_1 = (L0 - L1)/2  (left half-hat)
            vals.append(0.5 * (Pp[0] - Pp[1]))
            ders.append(0.5 * (dPp[0] - dPp[1]))
        elif a == 1:                     # Ltilde_2 = (L0 + L1)/2  (right half-hat)
            vals.append(0.5 * (Pp[0] + Pp[1]))
            ders.append(0.5 * (dPp[0] + dPp[1]))
        else:                            # bubble: L_a - L_{a-2} (degrees 2,3,...,M-1)
            # array index a (>=2) -> degree-a bubble L_a - L_{a-2}, vanishing at
            # both endpoints (L_a(+/-1)=(+/-1)^a, so L_a-L_{a-2}=0 at +/-1).
            # This INCLUDES the degree-2 bubble (a=2: L_2-L_0) -- the previous
            # L_{a+1}-L_{a-1} reading SKIPPED degree 2 and left the C0 space
            # incomplete (wrong plane-wave eigenvalues for N>2).
            vals.append(Pp[a] - Pp[a - 2])
            ders.append(dPp[a] - dPp[a - 2])
    return np.array(vals), np.array(ders)


#: FAIL-BEFORE switch for the MINIMUM-SEGMENT contract (round-2 D1).
#: ``False`` restores the pre-2026-09-11 acceptance -- any strictly increasing
#: wall array -- bit for bit.  It exists so a test can demonstrate the
#: behaviour it prevents; it is not a supported user knob.
PMM2D_STAG_MIN_SEG_GUARD = True

#: MINIMUM SEGMENT WIDTH of a NON-UNIFORM element grid, as a fraction of the
#: period.  This is a CONTRACT of the ``x_walls`` / ``y_walls`` /
#: ``add_tapered_pillar(s)`` surface, not a tolerance.
#:
#: WHY THERE IS ONE.  The staggered stiffness carries the per-segment ``1/J_n``
#: (Granet Eq. 31), so a segment of width ``w`` acquires spurious modal
#: wavenumbers ``|gamma|/k0 = c(M) M (M+1) / (4 k0 J)`` with ``J = w/2`` -- the
#: same ``1/w`` spectrum the 1-D nodal SEM has
#: (``docs/audits/FIX_PMMSTACK_SLIVER_WALLS_2026_09_11.md`` S3).  MEASURED here
#: 2026-09-11 (``validation/probe_pmm2d_mortar_round2/r1_onset.py``): the
#: constant reads **0.9165 (M = 4) / 0.9536 (M = 6)** and is stable to four
#: digits over ``delta`` from 3e-01 to 1e-06, i.e. the spectrum is a pure
#: function of the WALL ARRAY and ``M`` and carries no information the walls do
#: not -- which is why the guard is a WIDTH bar and not a spectral one (a
#: spectral bar would additionally refuse legitimately fine UNIFORM lattices,
#: whose ``|gamma|max`` is just as large and whose mortars are healthy: the
#: same overlap that defeated the 1-D ``|q|max`` bar, S3.3 there).
#:
#: Inside ONE grid those modes are harmless -- they are evanescent to machine
#: zero and the plain square modal match sees the SAME set on both sides
#: (measured ``delta``-independent to 4.1e-13).  Across a MORTAR they are not:
#: the cross-grid projection conditions as ``1/w^2`` on the E row and FASTER
#: than that on the H row (fitted exponents on two fixtures and two builds:
#: E row 1.995 / 1.995 and 1.994 / 1.994; H row 3.218 / 3.003 and 3.214 /
#: 3.103, the spread there being the fit's last rung sitting at the float64
#: ceiling), and the damage is ENERGY-INVISIBLE -- the lossless closure
#: stays pinned at 8.0e-08 across the whole ladder, so ``_warn_stag_closure``
#: and the 1-D fix's ``R+T`` screen have nothing to see.
#:
#: WHAT THE DAMAGE IS, measured against an EXACT oracle (a y-uniform 3-layer
#: stack whose middle layer is ALL HOST, so the device cannot depend on the
#: wall separation at all, scored per order against ``PMMStack`` at degree 14
#: whose own self-gap is 1.7e-06; probe ``r5_conv.py``).  There is no onset and
#: no wandering: the sliver puts a FLOOR under the ``M`` ladder.
#:
#:   ``delta``   err at ``M`` = 4 / 5 / 6 / 7 / 8            7->8
#:   3e-01   2.37e-02 2.42e-02 4.94e-03 1.38e-03 1.18e-04   11.71x
#:   1e-02   2.41e-02 3.73e-02 8.09e-03 1.54e-03 4.05e-04    3.81x
#:   1e-03   2.44e-02 4.00e-02 9.13e-03 1.65e-03 5.31e-04    3.11x
#:   1e-04   2.45e-02 4.32e-02 1.01e-02 1.69e-03 6.36e-04    2.66x
#:
#: Raising ``M`` still helps in absolute terms but does NOT remove the floor
#: (the spurious spectrum grows as ``M (M + 1)``), so "raise the modal count"
#: is not a remedy for a sliver -- that is a measurement, and it is why the
#: message does not offer it.
#:
#: THE BAR, with the gap on both sides measured.  ``1e-3`` of the period sits
#:
#:   * **2.10 decades BELOW** the narrowest segment any ORDINARY shipped
#:     geometry asks for.  Census over every per-layer geometry class the
#:     library builds: 4.000e-01 (a single interior wall), 3.333e-01
#:     (duty-1/3), 2.371e-01 (conforming and non-conforming), 2.100e-01 (axes
#:     carrying different walls), 2.004e-01 .. 1.881e-01 (the mortar suite's
#:     taper at 4 .. 64 slices), 1.733e-01 (``add_tapered_pillars``) and
#:     **1.250e-01** (a nested refinement -- the worst).
#:     The ONE surface that walks toward the bar is a taper whose tip CLOSES:
#:     the midpoint rule's narrowest sampled segment is
#:     ``~ w_bottom / (2 n_slices)``, measured 3.14e-02 / 8.01e-03 / 4.11e-03
#:     at ``n_slices`` = 8 / 32 / 64, so a fully closing taper crosses the bar
#:     at about 250 slices.  Both halves are re-measured on the running build
#:     by ``tests/unit/test_fix_pmm2d_mortar_round2.py`` rather than pinned
#:     here.
#:   * **AT** the width where the mortar operator runs out of float64, and that
#:     is a consistency check rather than a coincidence: the H-row operator
#:     with the sliver in the ``A`` slot reads LAPACK ``rcond`` = **7.34e-13**
#:     at exactly ``w/d`` = 1e-03, ``M`` = 6, against the 1e-12 refusal of
#:     :data:`~lumenairy.elements.pmm._core._MORTAR_RCOND_REFUSE`.  The two
#:     bars were derived INDEPENDENTLY -- one from accuracy against an exact
#:     oracle, the other from how many digits the operator can carry -- and
#:     they land on the same width.  Below it the operator collapses fast
#:     (2.06e-14 at 3e-04, 2.15e-17 at 3e-05) and the unguarded ``solve``
#:     raised outright at 1e-07.
#:
#: THIS BAR IS ``M``-INDEPENDENT AND THE CONDITIONING IS NOT, and the layering
#: is deliberate.  ``rcond`` falls about 3x per modal rung and then flattens
#: (measured on the shipped taper: 1.62e-05 / 5.98e-07 / 2.64e-07 / 4.67e-08 /
#: 3.89e-08 / 2.81e-08 at ``M`` = 4 / 6 / 7 / 8 / 9 / 10), so above ``M ~ 6``
#: the conditioning backstop is the one that fires first on a marginal grid.
#: A fixed WIDTH contract cannot know ``M``, the wavelength or the contrast;
#: it is the simple, documented, cheap half, and the backstop names the same
#: remedies.
#:
#: THE COMPARISON CARRIES A 1e-9 RELATIVE SLACK, because a caller who asks for
#: EXACTLY this minimum computes it in floating point and can land 1.8e-16
#: BELOW it (measured: ``(0.28572 - 0.28452) / 1.2`` = 9.999999999999824e-04,
#: which is a SHIPPED fixture).  Without the slack the documented boundary
#: would be decided by the caller's own arithmetic -- the at-threshold shape
#: ``docs/TESTING_STANDARDS.md`` calls S4.  See ``Basis1D.__init__``.
#:
#: THE INTEGER PATH IS EXEMPT, and that costs nothing: a uniform lattice's
#: segments are all ``d/N``, so reaching this bar needs ``N > 1000``, i.e.
#: ``q >= 2000`` and a ``2 q^2 = 8e+06``-dimension region eigenproblem.  The
#: exemption is what keeps gate N1 (integer walls BIT-IDENTICAL to the
#: pre-2026-09-11 library) unconditional.
#:
#: **CORRECTIONS, ROUND 3 (2026-09-11**,
#: ``docs/audits/FIX_PMM2D_MORTAR_ROUND3_2026_09_11.md`` **).**
#:
#: * this bar is a pure PERIOD FRACTION -- there is NO WAVELENGTH in it -- so
#:   on a large-period cell it refuses features that are physically ordinary
#:   (40 nm on a 40 um period).  That is DEFENSIBLE, because the conditioning
#:   it guards is a cross-grid projection on ONE period and is governed by
#:   ``w/d`` alone, but it was unstated (VERIFY round 2, DEFECT V5);
#: * the contract fires at ``PMM2DStackPure.solve()``, NOT at ``add_layer`` --
#:   ``add_layer`` only RECORDS the wall array, and :class:`Basis1D` is built
#:   when the stack is solved, so a 400-slice taper is built in full before it
#:   is refused (DEFECT V6).  "At the grid's entry point" above is true of
#:   :class:`Basis1D`, not of the user's call site;
#: * the E-row exponent quoted above as "``1/w^2``" is the TWO-AXIS value.  It
#:   is **1.0 per axis carrying the sliver** (measured 1.01 / 1.006 at
#:   ``M`` = 4 / 6 on a ONE-axis fixture against 1.995 / 1.994 on a two-axis
#:   one), so a sliver on one axis conditions as ``1/w`` and on both as
#:   ``1/w^2`` (DEFECT V4).  The H row is faster than the E row on the same
#:   slot on both fixtures, which is the claim that matters, but its exponent
#:   is BOUNDED, not pinned;
#: * the CROSS-MASS exponents (``C1_x`` / ``C2_x``) quoted in the round-2 fix
#:   doc are SATURATED at the float64 ceiling on both fixtures (they reach
#:   3.6e+18 and disagree between ``M`` = 4 and 6 by 2.4x).  Nothing rests on
#:   them; read them as "saturated", not as measurements;
#: * the CONDITIONING BACKSTOP
#:   (:data:`~lumenairy.elements.pmm._core._MORTAR_RCOND_REFUSE`) is NOT an
#:   independent second line.  Its ``MassH_A V_A`` operator is built from the
#:   FIRST grid, so on a sliver that occupies ONE axis in the LAST layer it is
#:   ``delta``-INDEPENDENT (measured 1.384e+05 at every ``delta`` from 3e-1 to
#:   1e-6) and does not fire down to ``delta`` = 1e-7 (DEFECT V3).  At the
#:   GENERALIZED site it does not screen conditioning at all any more
#:   (:data:`~lumenairy.elements.pmm._core._MORTAR_RESID_REFUSE`).  THIS
#:   CONTRACT IS THE ONLY LINE AGAINST A SLIVER on the per-layer path.
_STAG_MIN_SEG_FRAC = 1.0e-3

#: Switch for the DEGRADATION-BAND warning (round-3, VERIFY S5.4).  ``False``
#: restores the round-2 SILENCE in the band above the width contract, bit for
#: bit -- it exists so a test can demonstrate the silence the warning replaces,
#: and so a user who has read the measurement can turn it off.  Same style and
#: same status as :data:`PMM2D_STAG_MIN_SEG_GUARD`: not a supported user knob.
PMM2D_STAG_SLIVER_BAND_WARN = True

#: UPPER EDGE of the SILENTLY-DEGRADED band, as a fraction of the period.  A
#: per-layer stack whose narrowest segment lands in
#: ``[_STAG_MIN_SEG_FRAC, _STAG_SLIVER_BAND_FRAC)`` -- ACCEPTED by the width
#: contract, and MEASURABLY degraded by it -- gets a :class:`UserWarning` from
#: :meth:`~lumenairy.elements.pmm.PMM2DStackPure.solve`.  ROUND 3, 2026-09-11.
#:
#: WHY A BAND AND NOT A SECOND REFUSAL.  Above :data:`_STAG_MIN_SEG_FRAC` the
#: answer is a real answer -- it converges, it conserves energy, and it is
#: build-stable.  It is simply LESS ACCURATE than the same device on an
#: ordinary partition, by a factor that grows smoothly as the segment narrows
#: and that ``n_modes`` does NOT remove.  Refusing it would be wrong; leaving
#: it SILENT is what the VERIFY audit's S5.4 called the last silent-wrong-
#: answer surface on this path, because the ``n_modes`` ladder FLATTENS in the
#: band, so a user converging in ``n_modes`` reads the flattening as
#: convergence while sitting on a floor.
#:
#: THE MEASUREMENT.  Scored against the exact 1-D ``PMMStack`` (degree 14,
#: whose own 12 -> 14 self-gap is 1.49e-05) on a y-uniform 3-layer stack whose
#: MIDDLE layer is ALL HOST, so the DEVICE cannot depend on the wall separation
#: at all and every deviation is numerical damage.  Ratio to the ordinary
#: (3e-01) end of the SAME ladder, re-measured 2026-09-11 on a fixture
#: independent of the VERIFY audit's -- different period, wavelength, angle,
#: contrast, wall positions and sliver centre
#: (``validation/probe_fix_mortar_round3/r5_degradation_band.py``, identical on
#: WIN and WSL):
#:
#:   narrowest/period  3e-1  2e-1  1e-1  5e-2  3e-2  1e-2  3e-3  1e-3
#:   this fixture M=6  1.00  1.03  1.13  1.25  1.32  1.44  1.55  1.66
#:   this fixture M=7  1.00  1.06  1.31  1.49  1.56  1.62  1.64  1.64
#:   this fixture M=8  1.00  2.28  4.04  4.55  4.65  4.74  4.81  4.87
#:   VERIFY fixture M=8  1.00   --   2.09   --   4.02  5.31  5.76  5.86
#:
#: Every rung of every row above is IDENTICAL on WIN and WSL to the three
#: decimals printed, and the 1-D oracle's own self-gap agrees to 11 significant
#: figures -- it is a pure discretisation quantity with no meaningful
#: cross-build spread.
#:
#: The floor is only VISIBLE at rungs where the ordinary arm has converged
#: below it, which is why the ``M`` = 6 and 7 rows are mild and the ``M`` = 8
#: row is not: at ``M`` = 8 the ordinary arm reads 4.68e-04 while the band arm
#: saturates at ~2.2e-03.
#:
#: THE UPPER EDGE, with BOTH constraints measured -- and they pull in OPPOSITE
#: directions, which is the whole difficulty:
#:
#:   * ACCURACY says as WIDE as possible.  The bar is "the measured cost
#:     exceeds 2x", and on the ``M`` = 8 ladder that is first true at
#:     **2e-01** on this fixture (2.28x) and at **1e-01** on the VERIFY
#:     audit's (2.09x).
#:   * FALSE POSITIVES say as NARROW as possible.  The narrowest segment any
#:     ORDINARY geometry the library builds asks for is **1.2500e-01** in the
#:     shipped battery (a nested refinement) and **1.0937e-01** in the VERIFY
#:     audit's wider census (``add_tapered_pillars`` at 16 slices).
#:
#: So the ACCURACY criterion alone would put the edge ON TOP OF the ordinary
#: population -- 2e-01 is WIDER than every per-layer geometry except a single
#: interior wall and a duty-1/3 pair.  **THE CENSUS IS THE BINDING
#: CONSTRAINT**, and 3e-02 is the largest round decade-third that keeps a
#: stated factor on both sides:
#:
#:   * **4.65x** measured cost at the edge on this fixture (4.02x on the VERIFY
#:     audit's), i.e. the 2x bar is cleared by **2.3x**;
#:   * **3.6x below** the narrowest ordinary geometry (4.2x below the shipped
#:     battery's), so NO ordinary stack warns.  The census is re-run on the
#:     running build by ``tests/unit/test_fix_pmm2d_mortar_round3.py``.
#:
#: The ONE surface that DOES land in the band is the one the round-2 census
#: identified as walking toward the contract: a taper whose tip CLOSES, whose
#: narrowest sampled width is ``~ w_bottom / (2 n_slices)`` -- measured
#: 8.0094e-03 at 32 slices and 4.1047e-03 at 64.  That is the intent, not a
#: false positive.
#:
#: The band's LOWER edge is :data:`_STAG_MIN_SEG_FRAC` itself: below it the
#: stack is REFUSED, so the warning would never be read.  Both edges carry the
#: same 1e-9 RELATIVE slack the contract does, so the two rules are exactly
#: complementary -- see :func:`_warn_stag_sliver_band`.
_STAG_SLIVER_BAND_FRAC = 3.0e-2

#: Census hook for the minimum-segment contract.  When set to a list, every
#: NON-UNIFORM :class:`Basis1D` appends
#: ``(d, N, M, min_segment_fraction, refused)``.  ``None`` (the default) costs
#: one ``is None`` test per basis.  This is the instrument the false-positive
#: census is measured with; it is NOT a behaviour switch.
_STAG_SEG_CENSUS = None


def _raise_stag_sliver(xb, w, d, frac, M):
    """The minimum-segment refusal, naming the width, the bar, what the width
    does to the solve, and the remedies."""
    n = int(np.argmin(w))
    # |gamma| ~ 0.93 M (M + 1) / (4 J), MEASURED (see _STAG_MIN_SEG_FRAC),
    # quoted in units of the reciprocal lattice vector G = 2 pi / d so the
    # number is dimensionless without knowing k0: it is the number of
    # oscillations per period the spurious modes carry.
    spur = 0.93 * M * (M + 1) * d / (8.0 * np.pi * (0.5 * float(w[n])))
    raise ValueError(
        f"Basis1D: segment {n} of this NON-UNIFORM element grid is "
        f"{float(w[n]):.6g} wide -- {frac:.3e} of the period {d:.6g}, below "
        f"the minimum {_STAG_MIN_SEG_FRAC:.0e} this basis contracts for "
        f"(walls {np.array2string(np.asarray(xb), precision=6, threshold=12)}"
        f").\n"
        f"  WHY.  The staggered stiffness carries the per-segment 1/J_n "
        f"(Granet Eq. 31), so a segment this narrow carries spurious modal "
        f"wavenumbers |gamma| ~ {spur:.2e} x G (G = 2 pi / period) -- modes "
        f"oscillating that many times per period, against the handful a "
        f"physical mode of this cell can.  Inside one grid they are "
        f"harmless; across an L2 MORTAR (layer_grids='per-layer' with "
        f"neighbours on other grids) the cross-grid projection conditions as "
        f"1/w^2 on the E row and 1/w^3.2 on the H row, and the damage is "
        f"ENERGY-INVISIBLE -- the lossless closure stays pinned, so no "
        f"tripwire fires.  MEASURED: a device that cannot depend on this wall "
        f"separation at all reads a 3.4-5.4x FLOOR under its n_modes ladder "
        f"(docs/audits/FIX_PMM2D_MORTAR_ROUND2_2026_09_11.md).\n"
        f"  REMEDIES.  (1) MERGE the two walls -- a feature this fine is "
        f"below what the method resolves; (2) carry the fine feature on the "
        f"SHARED lattice (PMM2DStackPure(..., layer_grids='shared') with an N "
        f"that resolves it), where every cell is period/N and this cannot "
        f"form; (3) use PMM2DStackHybrid, which is Fourier-projected and has "
        f"no element grid; (4) on a TAPER, lower n_slices or stop the taper "
        f"before its tip closes (the midpoint rule's narrowest sampled width "
        f"is ~w_bottom / (2 n_slices)).  Raising n_modes is NOT a remedy: the "
        f"spurious spectrum grows as M (M + 1).")


def _stag_band_narrowest(grids):
    """Narrowest segment over a list of :class:`StagGridOps`, as a fraction of
    that axis's period, with the grid and axis that owns it.  Costs no solve --
    it reads the wall arrays the grids already hold."""
    worst, where = 1.0, None
    for gi, g in enumerate(grids):
        for ax, b in (("x", g.bx), ("y", g.by)):
            if b.uniform:
                f = 1.0 / float(b.N)
            else:
                f = float(np.min(np.diff(np.asarray(b.xb)))) / float(b.d)
            if f < worst:
                worst, where = f, (gi, ax, b)
    return worst, where


def _warn_stag_sliver_band(grids, mortared, fn="PMM2DStackPure.solve"):
    """ROUND-3 band warning: a per-layer segment ACCEPTED by the width contract
    but measurably degraded by its own narrowness (VERIFY round 2, S5.4).

    ``grids`` are the per-layer :class:`StagGridOps`; ``mortared`` says whether
    this stack actually builds a CROSS-GRID interface.  It does not warn when
    it does not: on a stack whose layers all share ONE grid every interface is
    the plain square modal match, and such a stack is MEASURED
    ``delta``-independent to ~1e-04 over three decades of wall separation
    against the mortared arm's 5.9x -- warning there would be the same false
    positive as DEFECT V2.

    WARNS, never raises, and returns whether it warned."""
    if not (PMM2D_STAG_SLIVER_BAND_WARN and mortared):
        return False
    frac, where = _stag_band_narrowest(grids)
    # BOTH edges carry the SAME 1e-9 RELATIVE slack the width contract does,
    # and for the same reason: a caller who asks for EXACTLY an edge computes
    # it in floating point and lands either side of it (measured:
    # ``0.47 - 0.44`` is 2.99999999999999970e-02, 3e-17 UNDER 3e-2).  With the
    # slack the documented boundary is deterministic instead of a coin flip on
    # the caller's own arithmetic, and the two rules line up EXACTLY: a grid
    # the contract accepts is either in the band or above it, never neither.
    if not (_STAG_MIN_SEG_FRAC * (1.0 - 1e-9) <= frac
            < _STAG_SLIVER_BAND_FRAC * (1.0 - 1e-9)):
        return False
    gi, ax, b = where
    # the measured degradation CLASS, read off the band the width lands in --
    # the ladder in _STAG_SLIVER_BAND_FRAC, quoted as a range because it is
    # fixture-dependent and grows toward the contract
    if frac >= 1.0e-2:
        cls = "about 4-5x"
    elif frac >= 3.0e-3:
        cls = "about 5-6x"
    else:
        cls = "about 6x, its floor"
    warnings.warn(
        f"{fn}: layer {gi}'s element grid has a segment "
        f"{frac:.3e} of the period wide on the {ax} axis (walls "
        f"{np.array2string(np.asarray(b.xb), precision=6, threshold=10)}).  "
        f"That is ABOVE the {_STAG_MIN_SEG_FRAC:.0e} minimum this basis "
        f"contracts for, so the solve proceeds -- but it is inside the "
        f"MEASURED degradation band {_STAG_MIN_SEG_FRAC:.0e} .. "
        f"{_STAG_SLIVER_BAND_FRAC:.0e}, where the per-segment 1/J_n stiffness "
        f"puts spurious modal wavenumbers into the L2 MORTAR that couples this "
        f"layer to neighbours on other grids.  MEASURED cost on a device that "
        f"cannot depend on the wall separation at all: {cls} the error of the "
        f"same device on an ordinary partition, and it is a FLOOR -- raising "
        f"n_modes does NOT remove it, and the n_modes ladder FLATTENS here, so "
        f"a convergence study will read the floor as convergence.  The "
        f"lossless closure stays pinned, so no energy tripwire can see this.\n"
        f"  REMEDIES.  (1) MERGE the two walls if the feature is finer than "
        f"the device needs; (2) carry the fine feature on the SHARED lattice "
        f"(PMM2DStackPure(..., layer_grids='shared') with an N that resolves "
        f"it), where every cell is period/N and no mortar forms; (3) put the "
        f"NEIGHBOURS on this layer's wall array too -- a CONFORMING per-layer "
        f"stack takes the plain square modal match and no mortar at all; "
        f"(4) use PMM2DStackHybrid, which is Fourier-projected and has no "
        f"element grid; (5) on a TAPER, lower n_slices.  To silence this "
        f"warning after reading the measurement, set "
        f"lumenairy.elements.pmm.twod_staggered."
        f"PMM2D_STAG_SLIVER_BAND_WARN = False.", stacklevel=3)
    return True


class Basis1D:
    """One-period 1-D staggered modified-Legendre basis on [0, d], N segments,
    M modified-Legendre functions per segment.  Builds BOTH global sets:

      Btilde : continuous C0 set (hats glued across nodes + Bloch periodic hat).
               dimension  Ntil = N*(M-1).
      B      : staggered partner with one bubble dropped per interval, SAME dim.

    Each global function is represented by its expansion in the per-segment local
    modified-Legendre functions: a global function g has, on segment n, a vector
    of M local coefficients g_local[n, 0:M] (coeff of Ltilde_a^n).  This local-
    coefficient representation lets us assemble ALL Galerkin integrals exactly
    from the per-segment elementary matrices of the modified-Legendre functions
    (computed once by Gauss-Legendre quadrature of high order -> exact for
    polynomials).

    SEGMENT BOUNDARIES (Granet Eq. 31) may be UNIFORM or ARBITRARY.  ``walls``
    is either an ``int`` ``N`` -- the uniform lattice, and then every matrix
    this class builds is BIT-IDENTICAL to the pre-2026-09-11 library -- or an
    increasing ``(N + 1,)`` array of boundaries running ``0 .. d``.  Eq. 31
    maps EACH segment individually,
    ``x = 0.5 (x_{n+1} - x_n) u + 0.5 (x_{n+1} + x_n)``, so nothing in the
    formulation requires the segments to be equal; the uniform lattice was an
    implementation choice, and lifting it is what makes arbitrary tapers (walls
    that move a few nm per z-slice) representable at all -- on a uniform
    lattice a 1.8 nm wall offset on a 700 nm period needs ``N ~ 390``.

    MINIMUM SEGMENT WIDTH -- a CONTRACT, not a tolerance.  A wall array whose
    narrowest segment is below :data:`_STAG_MIN_SEG_FRAC` (1e-3 of the period)
    is REFUSED, naming the width and the remedies.  The reason is the
    per-segment ``1/J_n`` stiffness: a narrow segment carries spurious modal
    wavenumbers ``~ 0.93 M (M + 1) / (4 k0 J)``, which are harmless inside one
    grid and corrupt the L2 MORTAR that couples per-layer grids -- with the
    lossless closure PINNED, so nothing downstream can see it.  See
    :data:`_STAG_MIN_SEG_FRAC` for the derivation, both gaps and the measured
    ladder.  The INTEGER path is exempt (all segments are ``d/N``; reaching the
    bar needs ``N > 1000``), which is what keeps the bit-identity claim below
    unconditional.

    ``self.J`` (the ONE scalar jacobian) survives only on the uniform path and
    is ``None`` on a non-uniform basis ON PURPOSE, so that any un-migrated
    reader raises a ``TypeError`` immediately instead of silently applying one
    segment's scaling to all of them.  The per-segment jacobians
    ``self.Jn = 0.5 * diff(xb)`` are the general quantity, and the four sites
    that consume them are :meth:`_global_matrix`, :func:`_global_pair_segmat`,
    :meth:`Granet2DTransverseE._eps_dir` and
    :func:`_stag_fourier_projection`.  Everything else is invariant:
    ``m_ref``/``s_ref``/``c_ref`` live on the reference interval, the hat glue
    (Eqs. 32-33, including the Bloch ``tau`` hat) is a statement about the
    reference interval alone, and the de Rham property
    ``d(Btilde) subset span(B)`` is per-segment and scale-free.

    The INTEGER path is kept DISTINCT from an explicitly-passed uniform array,
    and that is a measurement, not fastidiousness: ``np.linspace`` computes
    ``start + i*step`` and pins the last element, so ``linspace[i+1] -
    linspace[i]`` is not always the same double as ``d/N`` (measured 1.96e-16
    relative at ``d = 0.9, N = 4``; exactly 0 at ``d = 1.2``).  Routing the int
    through the array path would make the bit-identity claim conditional on the
    period.
    """

    def __init__(self, d, walls, M, tau=1.0 + 0.0j):
        assert M >= 3, "Basis1D needs M>=3 (M=2 gives a degenerate cardinality)"
        self.d = float(d)
        self.M = int(M)
        self.tau = _C(tau)
        if np.ndim(walls) == 0:
            # UNIFORM lattice -- the shipped path, bit for bit.
            self.N = int(walls)
            if self.N < 1:
                raise ValueError(f"Basis1D: N must be >= 1, got {walls!r}.")
            self.h = self.d / self.N             # segment length
            self.J = 0.5 * self.h                # dx/du jacobian
            # segment boundaries on the eps walls (Eq.31, uniform)
            self.xb = np.linspace(0.0, self.d, self.N + 1)
            self.Jn = np.full(self.N, self.J, dtype=float)
            self.uniform = True
        else:
            xb = np.asarray(walls, dtype=float).ravel()
            if xb.ndim != 1 or xb.size < 2:
                raise ValueError(
                    f"Basis1D: walls must be an int N or an increasing "
                    f"(N + 1,) boundary array, got shape "
                    f"{np.shape(walls)!r}.")
            if abs(xb[0]) > 1e-13 * self.d or abs(xb[-1] - self.d) > 1e-13 * self.d:
                raise ValueError(
                    f"Basis1D: walls must run 0 .. d = {self.d!r}, got "
                    f"{xb[0]!r} .. {xb[-1]!r}.")
            w = np.diff(xb)
            if np.any(w <= 0.0):
                raise ValueError(
                    "Basis1D: walls must be STRICTLY increasing (a zero-width "
                    f"segment has no affine map), got {xb!r}.")
            frac = float(np.min(w)) / self.d
            # The comparison carries a RELATIVE slack, and it is not
            # fastidiousness: a caller who asks for EXACTLY the documented
            # minimum computes it in floating point, and
            # ``(0.28572 - 0.28452) / 1.2`` is 9.999999999999824e-04 -- 1.8e-16
            # BELOW 1e-3.  Without the slack the documented boundary is a coin
            # flip on the caller's own arithmetic, which is exactly the
            # at-threshold shape ``docs/TESTING_STANDARDS.md`` calls S4.  The
            # slack is 1e-9 relative: nine decades tighter than any real
            # feature and nine decades looser than a rounding of the bar.
            refuse = bool(PMM2D_STAG_MIN_SEG_GUARD
                          and frac < _STAG_MIN_SEG_FRAC * (1.0 - 1e-9))
            if _STAG_SEG_CENSUS is not None:
                _STAG_SEG_CENSUS.append((self.d, int(xb.size - 1), self.M,
                                         frac, refuse))
            if refuse:
                _raise_stag_sliver(xb, w, self.d, frac, self.M)
            self.N = int(xb.size - 1)
            self.xb = xb
            self.Jn = 0.5 * np.diff(xb)
            self.h = None
            self.J = None                        # loud, not silent
            self.uniform = False
        self._build_elementary()
        self._build_sets()

    # ----- per-segment elementary modified-Legendre matrices (reference u) ---
    def _build_elementary(self):
        M = self.M
        # high-order Gauss-Legendre: exact for products up to degree ~2M
        gx, gw = leggauss(2 * M + 4)
        V, Vp = _modleg_value_deriv(M, gx)       # (M, Q)
        # mass  m_ab = INT_{-1}^{1} Ltilde_a Ltilde_b du
        self.m_ref = (V * gw) @ V.T               # (M, M)
        # stiffness s_ab = INT Ltilde_a' Ltilde_b' du
        self.s_ref = (Vp * gw) @ Vp.T
        # mixed  c_ab = INT Ltilde_a' Ltilde_b du   (for derivative-bra terms)
        self.c_ref = (Vp * gw) @ V.T
        # endpoint values (for hat continuity / inspection)
        Ve, _ = _modleg_value_deriv(M, np.array([-1.0, 1.0]))
        self.val_m1 = Ve[:, 0]                    # Ltilde_a(-1)
        self.val_p1 = Ve[:, 1]                    # Ltilde_a(+1)

    # ----- assemble the two global sets as local-coefficient stencils --------
    def _build_sets(self):
        """Each global function -> a list of (segment, M-vector) contributions.
        We store, for each global dof j, a dense (N, M) complex coefficient
        array S[j] giving its local modified-Legendre coefficients on every
        segment.  Sparse in practice (few nonzero segments) but small here."""
        N, M, tau = self.N, self.M, self.tau

        # local modified-Legendre indices:  0 -> Ltilde_1 (left half-hat),
        # 1 -> Ltilde_2 (right half-hat),  2..M-1 -> bubbles.
        # ---- Btilde : N hats (periodic) + bubbles 2..M-1 on every segment ----
        til = []          # list of (N,M) arrays
        # hats T^n, n=1..N (Eq.32 / Eq.33).  T^n glues Ltilde_2 of seg n-1 to
        # Ltilde_1 of seg n (0-based: node between seg n-1 and seg n).  The
        # periodic hat T (the one spanning the period seam) uses tau.
        for node in range(N):                 # node index 0..N-1 (N nodes, periodic)
            S = np.zeros((N, M), dtype=_C)
            left_seg = (node - 1) % N          # segment ending at this node
            right_seg = node                   # segment starting at this node
            if node == 0:
                # Bloch periodic hat (Eq.33): Ltilde_1 on seg 0, tau*Ltilde_2 on seg N-1
                S[right_seg, 0] += 1.0
                S[left_seg, 1] += tau
            else:
                S[left_seg, 1] += 1.0          # Ltilde_2 on the left segment
                S[right_seg, 0] += 1.0         # Ltilde_1 on the right segment
            til.append(S)
        # interior bubbles: one global dof per (segment, bubble index 2..M-1)
        for seg in range(N):
            for a in range(2, M):
                S = np.zeros((N, M), dtype=_C)
                S[seg, a] = 1.0
                til.append(S)
        self.Btilde = til                       # dimension N + N*(M-2) = N*(M-1)

        # ---- B : staggered partner, SAME dimension, one bubble dropped/interval
        # We build B as the DISCONTINUOUS set: per segment, BOTH half-hats as
        # INDEPENDENT functions (NOT glued) + bubbles 2..M-2 (drop the LAST
        # bubble a=M-1 per segment).  Per segment that is 2 + (M-3) = M-1
        # functions -> total N*(M-1), matching |Btilde| exactly.
        b = []
        for seg in range(N):
            # two independent half-hats on this segment
            for a in (0, 1):
                S = np.zeros((N, M), dtype=_C)
                S[seg, a] = 1.0
                b.append(S)
            for a in range(2, M - 1):           # bubbles, drop last (a=M-1)
                S = np.zeros((N, M), dtype=_C)
                S[seg, a] = 1.0
                b.append(S)
        self.B = b
        assert len(self.Btilde) == len(self.B) == N * (M - 1), \
            (len(self.Btilde), len(self.B), N * (M - 1))
        self.dim = N * (M - 1)

    # ----- global Galerkin matrices between two sets -------------------------
    def _global_matrix(self, ref, setL, setR, eps_seg=None):
        """<setL_i | (ref) | setR_j> on the period, assembled from per-segment
        elementary matrix `ref` (one of m_ref/s_ref/c_ref on reference u).
        Physical scaling: mass-type (m_ref) -> *J_n ; stiffness s_ref ->
        *(1/J_n); mixed c_ref (one derivative) -> *1 (the du cancels the 1/J_n
        of d/dx times the J_n of du, on EVERY segment whatever its length --
        which is why the mixed matrix was already non-uniform-correct and needs
        no edit).  `eps_seg` (length N) multiplies the per-segment integral
        (piecewise-constant eps; element walls on eps steps -> exact).
        Real inner product INT x* y -> conjugate the LEFT coefficients.

        SITE 1 of 4 for non-uniform segments (Granet Eq. 31): the scale is a
        per-segment VECTOR ``J_n``.  On the uniform path ``Jn`` is
        ``np.full(N, 0.5 * d / N)``, i.e. exactly the ``np.ones(N) * J`` this
        line used to build, so the assembled matrices are bit-identical."""
        _N, _M = self.N, self.M
        if ref is self.m_ref:
            scale = self.Jn
        elif ref is self.s_ref:
            scale = 1.0 / self.Jn
        else:                                    # c_ref : one derivative
            scale = np.ones(self.N)
        # Stack each set into a (dim, N, M) tensor for vectorized contraction.
        L_ten = np.array(setL)                    # (dimL, N, M)
        R_ten = np.array(setR)                    # (dimR, N, M)
        w_seg = np.asarray(scale, dtype=_C)
        if eps_seg is not None:
            w_seg = w_seg * np.asarray(eps_seg, dtype=_C)
        # out[i,j] = sum_seg w_seg[seg] * conj(L[i,seg,:]) @ ref @ R[j,seg,:]
        # = sum_seg w_seg[seg] * (conj(L[i,seg]) . (ref @ R[j,seg]))
        RR = np.einsum("ab,jsb->jsa", ref, R_ten)         # ref @ R[j,seg]  (dimR,N,M)
        out = np.einsum("isa,s,jsa->ij",
                        np.conj(L_ten), w_seg, RR)
        return out

    # convenience builders (mass=m_ref, stiff=s_ref, mixed=c_ref) ------------
    def mass(self, setL, setR, eps_seg=None):
        return self._global_matrix(self.m_ref, setL, setR, eps_seg)

    def stiff(self, setL, setR, eps_seg=None):
        return self._global_matrix(self.s_ref, setL, setR, eps_seg)

    def mixed(self, setL, setR, eps_seg=None):
        """<setL | d/dx | setR> : INT setL* (d setR/dx).  c_ref[a,b] =
        INT Ltilde_a'(u) Ltilde_b(u) du has the derivative on the LEFT index, so
        for INT L_i*(setR)' we need c_ref with derivative on the RIGHT operand ->
        use c_ref.T contracted appropriately: build <L | R'> = conj(ci) @ c_ref.T
        @ cj.  We pass ref=c_ref.T-like by transposing in the contraction."""
        L_ten = np.array(setL)
        R_ten = np.array(setR)
        # INT setL* d(setR)/dx du-physical: d/dx = (1/J) d/du, du = ... the J and
        # 1/J cancel -> scale 1.  c_ref[a,b]=INT Ltilde_a' Ltilde_b; we want
        # INT (setL)_a (setR)_b' = c_ref[b,a] -> use c_ref.T.
        cT = self.c_ref.T
        RR = np.einsum("ab,jsb->jsa", cT, R_ten)
        out = np.einsum("isa,jsa->ij", np.conj(L_ten), RR)
        return out


# =========================================================================== #
# 2-D STAGGERED TENSOR ASSEMBLY  (Eq.34) + transverse-E eigenproblem (Eq.23-24)
# --------------------------------------------------------------------------- #
# Field spaces (tensor products of the two 1-D sets):
#   V1 (E1) = B(x1)      (x) Btilde(x2)
#   V2 (E2) = Btilde(x1) (x) B(x2)
#   V3 (E3) = Btilde(x1) (x) Btilde(x2)
# Each per-axis set has dimension  q = N*(M-1).  So dim V1=dim V2=dim V3 = q^2.
#
# Operators (k = 2*pi numerically; coords in units of wavelength; we report
# g2 = gamma^2/k0^2 = -lam/k0^2 where lam is the raw eigenvalue of the
# generalized pencil).  We build everything with PHYSICAL d/dx (units 1/lambda).
#
# Per-axis elementary global matrices we need, between the relevant set pairs:
#   Mtt   = <Btilde | Btilde>            (mass, x-axis or y-axis)
#   Mtt_e = <Btilde | eps | Btilde>      (eps-weighted mass)
#   Mbb   = <B | B>,   Mbb_e = <B|eps|B>
#   Cbt   = <B | d/dx Btilde>            (mixed: bubble-test, hat-trial deriv)
#   Ctb   = <Btilde | d/dx B>
#   Ctt   = <Btilde | d/dx Btilde>
#   Stt   = <Btilde'| Btilde'> etc as needed
# With separable eps(x1,x2)=eps1(x1)*... NO: eps is a 2-D map (pillar).  For a
# rectangular pillar eps(x1,x2) is SEPARABLE on the element grid ONLY if the
# pillar is the full-width product of an x-interval and a y-interval -> YES, a
# rectangular pillar IS separable: eps = eps_p inside [xa,xb]x[ya,yb].  We handle
# the general (separable-in-segments) case by per-(seg_x,seg_y) constant eps and
# a Kronecker assembly: the 2-D eps-weighted mass is sum over (sx,sy) of
# eps[sx,sy] * (x-seg-mass kron y-seg-mass).  For a rectangular pillar this is
# exact (walls on element boundaries, Eq.26).
# =========================================================================== #


def _global_pair_segmat(basis: Basis1D, ref, setL, setR):
    """Like _global_matrix but returns the PER-SEGMENT contributions stacked:
    G[seg] (dimL,dimR) so the 2-D eps-weighted assembly can weight each
    (segx,segy) cell.  ref in {m_ref}; scale handled here.

    SITE 2 of 4 for non-uniform segments: the scale is the per-segment
    ``J_n`` broadcast over the segment axis.  On the uniform path every entry
    of ``Jn`` is the scalar ``J`` this used to multiply by, so the result is
    bit-identical (IEEE multiplication by the same double)."""
    _N, _M = basis.N, basis.M
    if ref is basis.m_ref:
        scale = basis.Jn
    elif ref is basis.s_ref:
        scale = 1.0 / basis.Jn
    else:
        scale = np.ones(basis.N)
    L_ten = np.array(setL)        # (dimL,N,M)
    R_ten = np.array(setR)        # (dimR,N,M)
    RR = np.einsum("ab,jsb->jsa", ref, R_ten)            # (dimR,N,M)
    # per-segment matrix G[s,i,j] = J_n[s] * conj(L[i,s]) . RR[j,s]
    G = np.asarray(scale)[:, None, None] * np.einsum(
        "isa,jsa->sij", np.conj(L_ten), RR)
    return G                       # (N, dimL, dimR)


def _stag_axis_masses(bx: Basis1D, by: Basis1D):
    """The four 1-D masses that FACTOR the V1 / V2 block field Grams:
    ``(Mtt_x, Mbb_x, Mtt_y, Mbb_y)``.

    ``G1 = kron(Mtt_y, Mbb_x)`` and ``G2 = kron(Mbb_y, Mtt_x)`` -- and these
    are not merely LIKE the eigensolver's ``-Rmat`` blocks, they ARE them, bit
    for bit (measured 0.0 in the probe's algebra smoke), which is why the
    mortar reads its E-row mass operator off the same factors the assembly
    already builds instead of rebuilding a second one.  Factored out of
    :meth:`Granet2DTransverseE._axis_mats` so the per-layer-grid cascade
    (:class:`~lumenairy.elements.pmm.stack2d_pure.PMM2DStackPure` with
    ``layer_grids='per-layer'``) shares one definition with the assembly."""
    return (bx.mass(bx.Btilde, bx.Btilde), bx.mass(bx.B, bx.B),
            by.mass(by.Btilde, by.Btilde), by.mass(by.B, by.B))


# --------------------------------------------------------------------------- #
# L2 MORTAR ingredients for PER-LAYER element grids (2-D pure staggered PMM).
# Design + measurements: docs/audits/EXPERIMENT_PMM2D_STAGGERED_MORTAR_2026_09_10.md
# --------------------------------------------------------------------------- #
def _stag_basis_fingerprint(b: Basis1D):
    """Exact content fingerprint of a :class:`Basis1D` -- the geometry the
    cross-mass consumes and nothing else.

    The WALL ARRAY is part of the key (not an integer ``N``): with non-uniform
    segments two bases can share ``N`` and ``M`` and be different grids.  So is
    ``tau``: the Bloch glue (Eq. 33) is IN the basis, so an angle sweep is a
    different basis and must not hit a cached cross-mass (open item O-8 in the
    experiment doc proposes splitting the tau-free part; not done here)."""
    return (float(b.d), int(b.M), complex(b.tau),
            np.asarray(b.xb, dtype=float).tobytes())


def _stag_cross_mass_1d(ba: Basis1D, bb: Basis1D, which: str):
    """``C[i, j] = INT_0^d conj(phi^a_i(x)) phi^b_j(x) dx`` between the SAME
    named global set (``'B'`` or ``'Btilde'``) on two :class:`Basis1D` objects
    covering the same period but on DIFFERENT segmentations.

    EXACT by Gauss-Legendre on the UNION of the two segment partitions: on
    every union sub-interval both sides are polynomials of degree ``<= M-1``,
    so ``M_a + M_b + 2`` points integrate the product exactly.  Near-coincident
    walls of the two lattices therefore appear only in this INTEGRATION mesh
    and NEVER as spectral elements -- the property the 1-D
    :func:`~lumenairy.elements.pmm._core._sem_cross_mass` docstring calls the
    decisive difference from a shared union grid, and the reason the shipped
    1-D sliver defect at near-coincident LAYER walls has no analogue here (the
    probe measured the pure arm smooth and monotone in the wall separation all
    the way to zero while the 1-D oracle's own self-gap blew up).

    The LEFT set is CONJUGATED, and unlike the 1-D nodal SEM's real cross-mass
    this one is COMPLEX: :class:`Basis1D` glues its periodic hat with
    ``tau = exp(-i alpha0 d)`` (Eq. 33), so the basis itself carries the Bloch
    phase and both the mass and the cross-mass are complex.  ``Cab^T`` in the
    1-D mortar algebra is therefore ``Cab^H`` here -- using the transpose is a
    silent error at normal incidence (``tau = 1``) and O(1) at oblique.

    Reduces to ``ba.mass(set, set)`` to quadrature round-off when
    ``ba is bb`` (measured 9.8e-16)."""
    if abs(ba.d - bb.d) > 1e-13 * max(ba.d, 1.0):
        raise ValueError(
            f"_stag_cross_mass_1d: the two bases must span ONE period, got "
            f"{ba.d!r} and {bb.d!r}.")
    Sa = np.asarray(getattr(ba, which))          # (dimA, Na, Ma)
    Sb = np.asarray(getattr(bb, which))          # (dimB, Nb, Mb)
    Ma, Mb = ba.M, bb.M
    xg, wg = leggauss(Ma + Mb + 2)
    cuts = np.unique(np.concatenate([ba.xb, bb.xb]))
    tol = 1e-12 * ba.d
    merged = [float(cuts[0])]
    for x in cuts[1:]:
        if float(x) - merged[-1] > tol:
            merged.append(float(x))
    C = np.zeros((Sa.shape[0], Sb.shape[0]), dtype=_C)
    for u0, u1 in zip(merged[:-1], merged[1:]):
        mid = 0.5 * (u0 + u1)
        ea = min(max(int(np.searchsorted(ba.xb, mid, side="right") - 1), 0),
                 ba.N - 1)
        eb = min(max(int(np.searchsorted(bb.xb, mid, side="right") - 1), 0),
                 bb.N - 1)
        axl, axr = ba.xb[ea], ba.xb[ea + 1]
        bxl, bxr = bb.xb[eb], bb.xb[eb + 1]
        J = 0.5 * (u1 - u0)
        xphys = mid + J * xg
        Va, _ = _modleg_value_deriv(Ma, (2.0 * xphys - (axl + axr)) / (axr - axl))
        Vb, _ = _modleg_value_deriv(Mb, (2.0 * xphys - (bxl + bxr)) / (bxr - bxl))
        Ce = (Va * (wg * J)) @ Vb.T              # (Ma, Mb)
        C += np.conj(Sa[:, ea, :]) @ Ce @ Sb[:, eb, :].T
    return C


#: Retained staggered 1-D cross-masses.  An entry is ``q_a x q_b`` complex128,
#: i.e. KILOBYTES (0.014 MB at ``N = (3, 6), M = 6``) against the 3.09 MB the
#: dense 2-D Kronecker product it factors would take -- see
#: :func:`_stag_kron_apply`.  ``max_bytes=None`` = bounded by the collective
#: ``LUMENAIRY_CACHE_BUDGET_MB`` ceiling only; ``clear_asm_caches`` drains it
#: through the registry and ``cache_report()`` shows it by name.
_STAG_GEO_CACHE = _ByteBudgetedLRU("pmm2d_staggered_geometry")


def _stag_cross_mass_1d_cached(ba: Basis1D, bb: Basis1D, which: str):
    """:func:`_stag_cross_mass_1d`, memoized on the two bases' fingerprints.

    A hit returns the stored (read-only) array by identity; a miss computes the
    identical bytes the uncached function computes."""
    key = ("stag_cross", which, _stag_basis_fingerprint(ba),
           _stag_basis_fingerprint(bb))
    hit = _STAG_GEO_CACHE.get(key)
    if hit is not None:
        return hit
    C = _stag_cross_mass_1d(ba, bb, which)
    C.setflags(write=False)
    _STAG_GEO_CACHE.put(key, C)
    return C


def _stag_kron_apply(Ky, Kx, X):
    """``kron(Ky, Kx) @ X`` WITHOUT materialising the Kronecker product.

    The eigensolver's index convention is ``I = jx + qx*jy`` (y slow, x fast),
    i.e. ``np.kron(Ky, Kx)``.  ``X`` has ``qyB * qxB`` rows
    (``qxB = Kx.shape[1]``, ``qyB = Ky.shape[1]``); the result has
    ``qyA * qxA``.

    This is an IDENTITY, not an approximation -- both 2-D field spaces are
    tensor products and both segment partitions are rectangular, so every 2-D
    mass and cross-mass factors exactly (measured 3.2e-16 .. 4.3e-16, i.e. BLAS
    reassociation only).  Materialising the dense operator is a REJECTED
    design and the measurement is not close: at ``N = (6, 12), M = 6`` the
    dense ``C1`` is 49.4 MB against 0.055 MB for its two factors (900x) and the
    separable apply is 49x faster -- per COMPONENT per INTERFACE, of which a
    staircase has two and ``nlay + 1``."""
    qxA, qxB = Kx.shape
    qyA, qyB = Ky.shape
    n = X.shape[1] if X.ndim == 2 else 1
    Xr = X.reshape(qyB, qxB, n)
    T = np.einsum("bx,yxn->ybn", Kx, Xr, optimize=True)      # (qyB, qxA, n)
    Y = np.einsum("ay,ybn->abn", Ky, T, optimize=True)       # (qyA, qxA, n)
    return Y.reshape(qyA * qxA, n)


class StagGridOps:
    """One layer's own element grid: the two :class:`Basis1D` axes and the
    FACTORS of its V1 / V2 block field Grams.

    ``V1 (E1 = Ex) = B(x) (x) Btilde(y)``  with Gram ``G1 = kron(Mtt_y, Mbb_x)``
    ``V2 (E2 = Ey) = Btilde(x) (x) B(y)``  with Gram ``G2 = kron(Mbb_y, Mtt_x)``

    and ``G = -Rmat = blkdiag(G1, G2)`` is exactly what the eigensolver
    assembles, so ``.V1`` / ``.V2`` are the same operators bit for bit
    (:func:`_stag_axis_masses`).  Each is held as its ``(Ky, Kx)`` FACTOR PAIR
    and applied through :func:`_stag_kron_apply`; the dense form is never
    built.

    ``key()`` fingerprints the grid CONTENT (wall arrays, ``M``, ``tau``), so
    two layers whose walls coincide share one grid object and their interface
    takes the plain square modal match rather than a mortar."""

    __slots__ = ("bx", "by", "M", "q", "qq", "Mtt_x", "Mbb_x", "Mtt_y",
                 "Mbb_y", "V1", "V2", "_key")

    def __init__(self, period_x, period_y, wx, wy, M, taux, tauy):
        self.M = int(M)
        self.bx = Basis1D(period_x, wx, M, taux)
        self.by = Basis1D(period_y, wy, M, tauy)
        if self.bx.dim != self.by.dim:
            raise ValueError(
                f"StagGridOps: the staggered tensor basis needs equal SEGMENT "
                f"COUNTS per axis (Nx == Ny); got Nx = {self.bx.N}, "
                f"Ny = {self.by.N}.  The wall POSITIONS may differ freely.")
        self.q = self.bx.dim
        self.qq = self.q * self.q
        (self.Mtt_x, self.Mbb_x,
         self.Mtt_y, self.Mbb_y) = _stag_axis_masses(self.bx, self.by)
        self.V1 = (self.Mtt_y, self.Mbb_x)      # (Ky, Kx)
        self.V2 = (self.Mbb_y, self.Mtt_x)
        self._key = (_stag_basis_fingerprint(self.bx),
                     _stag_basis_fingerprint(self.by))

    def key(self):
        return self._key

    @property
    def N(self):
        return self.bx.N


class StagCrossOps:
    """The cross-mass between two :class:`StagGridOps`, as EXACT Kronecker
    FACTORS::

        C1 = kron( Ctt_y(A,B), Cbb_x(A,B) )      (the V1 = E1 / H2 space)
        C2 = kron( Cbb_y(A,B), Ctt_x(A,B) )      (the V2 = E2 / H1 space)

    Both factor exactly because both spaces are tensor products and both
    partitions are rectangular.  ``C1H()`` / ``C2H()`` are the CONJUGATE
    transposes -- the staggered basis is complex (the Bloch ``tau`` glue lives
    IN it), so the 1-D mortar's ``Cab^T`` is ``Cab^H`` here."""

    __slots__ = ("C1", "C2")

    def __init__(self, ga: "StagGridOps", gb: "StagGridOps"):
        Cbb_x = _stag_cross_mass_1d_cached(ga.bx, gb.bx, "B")
        Ctt_x = _stag_cross_mass_1d_cached(ga.bx, gb.bx, "Btilde")
        Cbb_y = _stag_cross_mass_1d_cached(ga.by, gb.by, "B")
        Ctt_y = _stag_cross_mass_1d_cached(ga.by, gb.by, "Btilde")
        self.C1 = (Ctt_y, Cbb_x)
        self.C2 = (Cbb_y, Ctt_x)

    @staticmethod
    def _H(pair):
        return (pair[0].conj().T, pair[1].conj().T)

    def C1H(self):
        return self._H(self.C1)

    def C2H(self):
        return self._H(self.C2)


class Granet2DTransverseE:
    """Faithful Granet staggered transverse-E eigensolver for a rectangular
    (or separable) 2-D unit cell, isotropic nonmagnetic media.

    Parameters
    ----------
    px, py    : period (units of wavelength)
    wx, wy    : per-axis segmentation -- either an ``int`` (``Nx`` / ``Ny``
                UNIFORM segments, the shipped spelling, BIT-IDENTICAL) or an
                increasing ``(N + 1,)`` array of SEGMENT BOUNDARIES running
                ``0 .. period`` (Granet Eq. 31 maps each segment individually,
                so the walls need not be equally spaced).  Arbitrary walls are
                what makes a TAPER representable: a wall that moves 1.8 nm per
                z-slice needs ``N ~ 390`` on a uniform lattice and 3 segments
                on its own.  The two axes may carry DIFFERENT wall positions;
                only the segment COUNTS must match (``bx.dim == by.dim``).
    M         : modified-Legendre functions per segment per axis (degree knob)
    eps_cell  : (Nx, Ny) array of constant SCALAR eps per segment-cell, OR
                (Nx, Ny, 3, 3) BLOCK-FORM permittivity tensors (Granet Eq. 7,
                ``[[e11, e12, 0], [e21, e22, 0], [0, 0, e33]]``) -- see
                :meth:`_assemble`.  Scalar input runs the shipped isotropic
                assembly BIT-FOR-BIT (a dispatch, not a rewrite); the caller is
                responsible for the out-of-plane / e33 gate
                (:func:`_tile_needs_oop`).
    alpha0x, alpha0y : Bloch wavenumbers (units 1/lambda) -> tau_x, tau_y
    k0        : 2*pi / wavelength (numerically 2*pi if coords in wavelengths)
    mu_cell   : (Nx, Ny) or (Nx, Ny, 3, 3) BLOCK-FORM relative permeability, or
                ``None`` (nonmagnetic; every operator stays BIT-IDENTICAL).
    slant     : ``(t_x, t_y)`` TANGENT pair (or a bare scalar ``t_x``), or
                ``None``/``0`` for a vertical cell.  A constant x-z / y-z SHEAR:
                the cell translates laterally by ``t * depth`` from the layer's
                TOP face to its bottom, ``eps_cell`` being the cross-section at
                the TOP.  Same PUBLIC convention as
                :meth:`~lumenairy.elements.pmm.PMM2DStackHybrid.add_layer` and
                the 1-D ``slant_angle`` entries (``t = tan(wall_tilt)``), so a
                cell moves between the engines unchanged.  Realized as the
                POINTWISE covariant congruence :func:`_slant_congruence` plus
                six extra Galerkin blocks in :meth:`_assemble_oop`; a slanted
                cell ALWAYS takes the first-order out-of-plane generator (its
                covariant tensor has out-of-plane entries).  ``slant = 0`` is
                BIT-IDENTICAL to no slant.  A shear is NOT a taper -- no shear
                absorbs a dilation; a tapered feature still needs a
                z-staircase.
    """

    def __init__(self, px, py, wx, wy, M, eps_cell,
                 alpha0x=0.0, alpha0y=0.0, k0=2.0 * np.pi, mu_cell=None,
                 slant=None):
        self.k0 = float(k0)
        self.alpha0x = float(alpha0x)
        self.alpha0y = float(alpha0y)
        taux = np.exp(-1j * alpha0x * px)
        tauy = np.exp(-1j * alpha0y * py)
        self.bx = Basis1D(px, wx, M, taux)
        self.by = Basis1D(py, wy, M, tauy)
        self.eps_cell = np.asarray(eps_cell, dtype=_C)   # (Nx,Ny) or (Nx,Ny,3,3)
        if self.eps_cell.ndim not in (2, 4) or (
                self.eps_cell.ndim == 4 and self.eps_cell.shape[2:] != (3, 3)):
            raise ValueError(
                f"Granet2DTransverseE: eps_cell must be (Nx, Ny) scalar or "
                f"(Nx, Ny, 3, 3) block-form tensor, got shape "
                f"{self.eps_cell.shape}.")
        if self.eps_cell.shape[:2] != (self.bx.N, self.by.N):
            # With explicit WALLS the cell grid is no longer implied by the
            # constructor arguments, so the pairing has to be checked (on the
            # integer path it is automatic and this can never fire).
            raise ValueError(
                f"Granet2DTransverseE: eps_cell grid "
                f"{self.eps_cell.shape[:2]} does not match the segmentation "
                f"({self.bx.N}, {self.by.N}) implied by wx / wy.")
        self.q = self.bx.dim                              # = Nx*(M-1)
        assert self.bx.dim == self.by.dim, "use square (Nx*(M-1)==Ny*(M-1))"
        # SLANT (constant x-z / y-z SHEAR; roadmap Phase D, build doc
        # docs/audits/BUILD_PMM2D_STAGGERED_SLANT_2026_09_10.md).  ``slant``
        # is the PUBLIC (t_x, t_y) TANGENT pair of
        # ``PMM2DStackHybrid.add_layer`` -- the lateral walk per unit depth,
        # cross-section taken at the layer's TOP -- normalised by the SAME
        # helper both 2-D engines and the 1-D entries use, so a scalar
        # ``slant=0.0`` and ``slant=None`` are one (vertical, byte-identical)
        # path.  The INTERNAL shear of the frame ``x = u + t w`` is the
        # NEGATIVE of it, pinned FOUR ways in the build doc's tables, each
        # against an independently validated engine: B5 against
        # ``pmm_efficiency_1d_slanted`` per order (1.22e-03 vs 4.15e-01), M4b
        # against the hybrid slant metric (1.23e-02 vs 2.79e-01, and only the
        # right arm improves with truncation), M4c against a pure-solver
        # z-staircase (converges only marching WITH the slant), and B4a
        # against the EXACT quartic roots (1.4e-14 vs 3.5e-02).  The rotation
        # gauge flips it once more (:func:`_slant_rot_gauge`).
        self.slant = _norm_slant_pair(slant, "Granet2DTransverseE")
        self.slanted = not _slant_is_zero(self.slant)
        self._slant_rot = (0.0, 0.0)
        self._eps_pre_rotated = False
        self.eps_lab = None
        if self.slanted:
            if mu_cell is not None:
                raise NotImplementedError(
                    "Granet2DTransverseE: slant= together with mu_cell is not "
                    "implemented -- a sheared cell runs the OUT-OF-PLANE "
                    "first-order generator, which carries no permeability "
                    "blocks (it eliminates G3 assuming mu = 1).  The shear's "
                    "OWN magnetic anisotropy mu^{lm} = g^{lm} is absorbed "
                    "analytically by the six slant blocks; a MATERIAL mu is "
                    "not.")
            cell33 = self.eps_cell
            if cell33.ndim == 2:            # scalar map -> isotropic tensor
                cell33 = cell33[..., None, None] * np.eye(3, dtype=_C)
            _require_nonzero_ezz("Granet2DTransverseE", cell33)
            # A sheared cell IS an out-of-plane cell in the frame, ALWAYS.
            self.eps_lab = cell33               # the cell the CALLER passed
            self.eps_cell, txr, tyr = _slant_rot_gauge(
                cell33, -self.slant[0], -self.slant[1])
            self._slant_rot = (txr, tyr)
            self._eps_pre_rotated = True    # _assemble_oop must not re-rotate
        # OUT-OF-PLANE dispatch (Stage B).  The test is the RELATIVE floor the
        # hybrid uses (:func:`_tile_needs_oop`), so a cell whose xz/yz/zx/zy
        # entries are float noise stays BIT-IDENTICAL to the in-plane path --
        # the dispatch is the Stage-A ``NotImplementedError`` guard turned into
        # a branch, at exactly the same floor.
        self.offplane = self.slanted or (self.eps_cell.ndim == 4
                                         and _tile_is_offplane(self.eps_cell))
        # MAGNETIC dispatch (chi_t = [mu_t]^-1 != I).  ``mu_cell=None`` -- the
        # nonmagnetic default -- leaves every operator below untouched, so the
        # shipped isotropic and Stage-A tensor paths stay BIT-IDENTICAL (gate
        # G1); a mu present routes the SAME second-order pencil through the
        # paper's general R / K_tz / S_tt weights (Eqs. 24, 21, 20).
        self.mu_cell = None
        self.magnetic = False
        self.Ggram_blocks = None
        if mu_cell is not None:
            mu = np.asarray(mu_cell, dtype=_C)
            if mu.ndim not in (2, 4) or (mu.ndim == 4
                                         and mu.shape[2:] != (3, 3)):
                raise ValueError(
                    f"Granet2DTransverseE: mu_cell must be (Nx, Ny) scalar or "
                    f"(Nx, Ny, 3, 3) block-form tensor, got shape {mu.shape}.")
            if mu.shape[:2] != self.eps_cell.shape[:2]:
                raise ValueError(
                    f"Granet2DTransverseE: mu_cell grid {mu.shape[:2]} must "
                    f"match the eps_cell grid {self.eps_cell.shape[:2]}.")
            if mu.ndim == 4:
                _require_inplane_mu("Granet2DTransverseE", mu)
            elif float(np.min(np.abs(mu))) < 1e-300:
                raise ValueError(
                    "Granet2DTransverseE: a scalar mu_cell must be nonzero in "
                    "every cell (chi = 1/mu).")
            if self.offplane:
                raise NotImplementedError(
                    "Granet2DTransverseE: mu_cell together with an "
                    "OUT-OF-PLANE eps_cell (e_xz / e_yz / e_zx / e_zy above "
                    "the relative 1e-12 floor) is not implemented -- the "
                    "out-of-plane FIRST-ORDER generator carries no "
                    "permeability blocks (it eliminates G3 assuming mu = 1).  "
                    "Magnetic anisotropy is available on the IN-PLANE "
                    "(block-form) second-order pencil only.")
            self.mu_cell = mu
            self.magnetic = True
        if self.offplane:
            self._assemble_oop()
        else:
            self._assemble()

    # --- per-axis 1-D ingredient matrices between set pairs (no eps) ---------
    def _axis_mats(self):
        bx, by = self.bx, self.by
        # the four MASSES -- shared with the per-layer-grid mortar, which needs
        # exactly these as the FACTORS of the V1 / V2 block field Grams
        # (:func:`_stag_axis_masses`; ``-Rmat``'s blocks bit for bit)
        (self.Mtt_x, self.Mbb_x,
         self.Mtt_y, self.Mbb_y) = _stag_axis_masses(bx, by)
        # x-axis
        self.Ctb_x = bx.mixed(bx.Btilde, bx.B)            # <til| d B>
        self.Cbt_x = bx.mixed(bx.B, bx.Btilde)            # <B  | d til>
        self.Ctt_x = bx.mixed(bx.Btilde, bx.Btilde)       # <til| d til>
        # y-axis
        self.Ctb_y = by.mixed(by.Btilde, by.B)
        self.Cbt_y = by.mixed(by.B, by.Btilde)
        self.Ctt_y = by.mixed(by.Btilde, by.Btilde)

    # --- eps-weighted per-axis-pair segment tensors (for 2-D eps assembly) ---
    def _eps_weighted(self, refx_pair, refy_pair, wmap=None):
        """Assemble a 2-D eps-weighted Galerkin matrix:
          sum_{sx,sy} eps[sx,sy] * kron( Gy[sy], Gx[sx] )
        where Gx = per-segment x-matrix between the x set-pair, Gy similarly.
        refx_pair = (basis_x, refmat_x, setLx, setRx); same for y.

        ``wmap`` is the per-cell scalar weight map ``(Nx, Ny)``; ``None`` (the
        isotropic default) uses ``self.eps_cell``.  The TENSOR assembly passes
        one COMPONENT map (e11 / e12 / e21 / e22 / e33) per block, so the
        arithmetic is identical to the scalar path when that component IS
        ``eps_cell`` (the G1 bit-identity reduction)."""
        bx, refx, sLx, sRx = refx_pair
        by, refy, sLy, sRy = refy_pair
        Gx = _global_pair_segmat(bx, refx, sLx, sRx)      # (Nx, dLx, dRx)
        Gy = _global_pair_segmat(by, refy, sLy, sRy)      # (Ny, dLy, dRy)
        eps = self.eps_cell if wmap is None else wmap     # (Nx, Ny)
        dLx, dRx = Gx.shape[1], Gx.shape[2]
        dLy, dRy = Gy.shape[1], Gy.shape[2]
        out = np.zeros((dLy * dLx, dRy * dRx), dtype=_C)
        for sx in range(bx.N):
            # accumulate over sy with eps weights -> weighted y-matrix
            Wy = np.einsum("y,yij->ij", eps[sx, :], Gy)   # (dLy,dRy)
            out += np.kron(Wy, Gx[sx])
        return out

    def _chi_maps(self):
        """Per-cell inverse-permeability maps ``(chi11, chi12, chi21, chi22,
        chi33)`` -- Granet Eqs. 11-12 -- or ``None`` when nonmagnetic.

        ``[chi_t] = [mu_t]^-1`` is the POINTWISE 2x2 inverse (exact for the
        piecewise-constant cells this basis is built on: the walls are element
        boundaries, so inverting per cell and discretizing commute) and
        ``chi33 = 1/m33``.  A SCALAR ``mu`` gives ``chi_t = (1/mu) I``."""
        mu = self.mu_cell
        if mu is None:
            return None
        if mu.ndim == 2:
            inv = 1.0 / mu
            zero = np.zeros_like(inv)
            return inv, zero, zero, inv, inv
        m11, m12 = mu[..., 0, 0], mu[..., 0, 1]
        m21, m22 = mu[..., 1, 0], mu[..., 1, 1]
        det = m11 * m22 - m12 * m21
        return (m22 / det, -m12 / det, -m21 / det, m11 / det,
                1.0 / mu[..., 2, 2])

    def _assemble(self):
        """Build R (Eq. 24) and L = k^2[eps_t] + S_tt - K_tz eps33^-1 K_zt.

        ISOTROPIC (scalar ``(Nx, Ny)`` eps_cell) and BLOCK-FORM ANISOTROPIC
        ((Nx, Ny, 3, 3)) share ONE body: the component maps ``e11/e12/e21/
        e22/e33`` are ``None`` for the scalar case, which makes every
        eps-weighted call fall back to ``self.eps_cell`` -- so the scalar
        arithmetic is EXACTLY the shipped isotropic assembly (gate G1 pins
        the bit identity against the tensor ``e*I`` arm).

        The anisotropic terms (Granet 2023 Appendix A, conjugated into the
        module's PUBLIC ``exp(-i w t)`` convention -- the operators carry no
        explicit ``i``, so the bridge is exactly "use the public eps"):

        * Eq. 40 -- FOUR ``[eps_t]`` blocks.  ``e11`` in V1xV1 and ``e22`` in
          V2xV2 as before, PLUS the MIXED masses ``<V1| e12 |V2>`` and
          ``<V2| e21 |V1>`` (kron of the UNLIKE-set 1-D masses).
        * Eq. 41 -- ``Meps33`` weighted by ``e33`` (not the scalar eps).
        * Eq. 42/43 -- ``S_tt`` and ``K_tz`` unchanged (chi_t = I, chi33 = 1):
          pure geometry.
        * Eq. 44 -- ``K_zt`` gains a SECOND term per column: it is the
          divergence of ``D_t = eps_t E_t``, so column 1 pairs ``d2`` with
          ``e21`` and column 2 pairs ``d1`` with ``e12``.
        * Eq. 25 -- the H-partner ``Lhh = k^2[eps_t] + S_tt`` inherits the two
          mixed blocks; :func:`_region_modes` folds them in from
          ``self.Et_offdiag``.
        """
        bx, by = self.bx, self.by
        k0 = self.k0
        self._axis_mats()
        q = self.q
        qq = q * q
        # component maps: None -> the eps-weighted helpers use self.eps_cell
        tensor = self.eps_cell.ndim == 4
        if tensor:
            e11 = self.eps_cell[..., 0, 0]
            e12 = self.eps_cell[..., 0, 1]
            e21 = self.eps_cell[..., 1, 0]
            e22 = self.eps_cell[..., 1, 1]
            e33 = self.eps_cell[..., 2, 2]
        else:
            e11 = e12 = e21 = e22 = e33 = None
        # MAGNETIC (chi_t = [mu_t]^-1 != I).  None on the nonmagnetic path, and
        # every branch below is skipped -> the shipped arithmetic, unchanged.
        chi = self._chi_maps()
        magnetic = chi is not None
        if magnetic:
            chi11, chi12, chi21, chi22, chi33 = chi

        # ----- per-axis 1-D primitive matrices (real inner product) -----
        # Mass (no deriv) between set pairs:
        # Only the like-set masses enter the staggered Gram (G1, G2 below).
        Mtt_x = self.Mtt_x      # <til|til>_x
        Mtt_y = self.Mtt_y
        Mbb_x = self.Mbb_x      # <B|B>_x
        Mbb_y = self.Mbb_y
        # Directed weak derivative (deriv on the TRIAL/right operand), /k0 ->
        # dimensionless.  d[L,R] = INT L* (d R/dx) /k0.  Only <B | d til> enters
        # the K_zt / K_tz coupling terms below.
        dbt_x = bx.mixed(bx.B, bx.Btilde) / k0        # <B  | d til>_x
        dbt_y = by.mixed(by.B, by.Btilde) / k0

        # ============ R = C[chi_t]C = -I  -> -block Gram (mass) ===============
        G1 = np.kron(Mtt_y, Mbb_x)                  # <V1|V1>, V1=B(x1)til(x2)
        G2 = np.kron(Mbb_y, Mtt_x)                  # <V2|V2>, V2=til(x1)B(x2)
        Rmat = np.zeros((2 * qq, 2 * qq), dtype=_C)
        if magnetic:
            # Eq. 24 / Appendix-A Eq. 39: R = C[chi_t]C with C = [[0,1],[-1,0]],
            # i.e. R = [[-chi22, chi21], [chi12, -chi11]] -- the C-rotation
            # SWAPS the transverse indices (R11 carries chi22, R22 chi11) and
            # the two MIXED blocks (chi21 in V1 x V2, chi12 in V2 x V1) are
            # kron'd from the UNLIKE-set 1-D masses exactly like the Eq. 40
            # eps12 / eps21 blocks.  For a Hermitian positive-definite [mu_t]
            # this R is Hermitian NEGATIVE definite, so G = -R stays the
            # Hermitian PD right-hand matrix of the pencil (chi_t = I gives
            # back -blockdiag(G1, G2) to ~1e-16 -- summation order only).
            Rmat[:qq, :qq] = -self._eps_weighted(
                (bx, bx.m_ref, bx.B, bx.B),
                (by, by.m_ref, by.Btilde, by.Btilde), chi22)
            Rmat[:qq, qq:] = self._eps_weighted(
                (bx, bx.m_ref, bx.B, bx.Btilde),
                (by, by.m_ref, by.Btilde, by.B), chi21)
            Rmat[qq:, :qq] = self._eps_weighted(
                (bx, bx.m_ref, bx.Btilde, bx.B),
                (by, by.m_ref, by.B, by.Btilde), chi12)
            Rmat[qq:, qq:] = -self._eps_weighted(
                (bx, bx.m_ref, bx.Btilde, bx.Btilde),
                (by, by.m_ref, by.B, by.B), chi11)
            # THE PLAIN BLOCK GRAM, kept SEPARATELY.  On the nonmagnetic path
            # -R IS blockdiag(G1, G2) and the shipped code uses the one object
            # for both roles; with chi_t != I they are DIFFERENT operators.
            # The Eq.-25 H recovery (gamma C [H1;H2] = [k^2 eps_t + S_tt] E)
            # carries NO chi_t, so it must project with THIS Gram, not with
            # the pencil's R (see :func:`_region_modes`).  Stored as the two
            # q^2 diagonal blocks (the off-diagonal blocks are zero), so the
            # magnetic path retains q^2 x 2, not 4 q^2.
            self.Ggram_blocks = (G1, G2)
        else:
            Rmat[:qq, :qq] = -G1
            Rmat[qq:, qq:] = -G2

        # ============ L = [eps_t] + S_tt - K_tz eps33^-1 K_zt  (/k0^2) ========
        # --- [eps_t] : eps-weighted component masses (k^2[eps]/k0^2 = [eps]) ---
        Et_11 = self._eps_weighted(
            (bx, bx.m_ref, bx.B, bx.B),
            (by, by.m_ref, by.Btilde, by.Btilde), e11)   # E1 space, V1
        Et_22 = self._eps_weighted(
            (bx, bx.m_ref, bx.Btilde, bx.Btilde),
            (by, by.m_ref, by.B, by.B), e22)             # E2 space, V2
        # Eq.40 MIXED masses (tensor only): <V1| e12 |V2> and <V2| e21 |V1>.
        # V1 = B(x1) (x) Btilde(x2), V2 = Btilde(x1) (x) B(x2), so each mixed
        # block is a kron of the two UNLIKE-set 1-D masses -- <B|Btil>_x with
        # <Btil|B>_y for the (1,2) block, and the transposed pairing for (2,1).
        Et_12 = Et_21 = None
        if tensor:
            Et_12 = self._eps_weighted(
                (bx, bx.m_ref, bx.B, bx.Btilde),
                (by, by.m_ref, by.Btilde, by.B), e12)    # V1 test, V2 trial
            Et_21 = self._eps_weighted(
                (bx, bx.m_ref, bx.Btilde, bx.B),
                (by, by.m_ref, by.B, by.Btilde), e21)    # V2 test, V1 trial

        # --- DIRECTED DERIVATIVE OPERATORS into the V3 (Btil x Btil) space ---
        # We discretize the STRONG symbol exactly as in the analytic check:
        #   curl-like  w = d2 E1 - d1 E2   (V3 coeffs via the V3 Gram)
        #   div-like   s = d1(eps E1) + d2(eps E2)
        # and form S_tt, Schur as G3-mediated products of these directed ops.
        #
        # V3 Gram (no eps), the E3=Btil(x)Btil space, would be
        # kron(Mtt_y, Mtt_x) -- never consumed (the curl is tested in its own
        # Vw space below, and the Schur solve runs against Meps33), so it is
        # not built (audit P3-37; already flagged 2026-06-08).

        # === S_tt : the curl chi33 curl operator ===
        # MIMETIC PLACEMENT (the crux of the staggered basis).  The de Rham
        # property d(Btilde) subset span(B) (verified ~1e-15) means the CURL
        #   w = d2 E1 - d1 E2,  E1=B(x1)Btil(x2), E2=Btil(x1)B(x2)
        # lands EXACTLY in the space  Vw = B(x1) (x) B(x2)  (NOT V3=Btil(x)Btil):
        #   d2 E1 = B(x1) Btil'(x2) ,  Btil'(x2) in span(B) -> B(x1)(x)B(x2)
        #   d1 E2 = Btil'(x1) B(x2) ,  Btil'(x1) in span(B) -> B(x1)(x)B(x2)
        # Testing the curl in its OWN natural space Vw=B(x)B (Gram Gw) is what
        # makes the discrete curl FULL-RANK on the transverse modes and renders
        # the de Rham complex EXACT -> spurious-free.  (Projecting the curl into
        # V3=Btil(x)Btil instead -- the previous attempt -- loses rank because
        # B' is NOT in span(Btil), leaving a longitudinal residue = spurious sea.)
        Gw = np.kron(Mbb_y, Mbb_x)                  # <Vw|Vw>, Vw=B(x1)(x)B(x2)
        # Curl (Vw <- [E1;E2]):
        #   <w| d2 E1> : x1 <B|B>=Mbb_x ; x2 <B| d Btil>=dbt_y -> kron(dbt_y, Mbb_x)
        Cw_E1 = np.kron(dbt_y, Mbb_x)
        #   <w| d1 E2> : x1 <B| d Btil>=dbt_x ; x2 <B|B>=Mbb_y -> kron(Mbb_y, dbt_x)
        Cw_E2 = np.kron(Mbb_y, dbt_x)
        Curl = np.concatenate([Cw_E1, -Cw_E2], axis=1)     # (q^2, 2q^2)
        Gw_inv = np.linalg.inv(Gw)
        # IBP: <t,[d2;-d1] w> = -<Curl(t), w>  -> S_tt = -Curl^dag Gw^{-1} Curl.
        #
        # Eq. 20/42 with chi33 != 1.  ``Curl`` is the WEAK matrix <Vw | curl E>,
        # so ``Gw^-1 Curl`` are the EXACT Vw coefficients of curl E (the de Rham
        # property puts curl E in Vw strongly).  The bilinear form is then
        # -<curl v, chi33 curl E> = -(Gw^-1 Curl v)^dag Gw_chi (Gw^-1 Curl E),
        # i.e. the middle operator Gw^-1 -> Gw^-1 Gw_chi Gw^-1 with
        # Gw_chi = <Vw| chi33 |Vw>.  chi33 = 1 gives Gw_chi = Gw and the middle
        # collapses back to Gw^-1 -- the expression below is then the shipped
        # matmul chain object-for-object (bit-identical, gate G1).
        Sw = Gw_inv
        if magnetic:
            Gw_chi = self._eps_weighted((bx, bx.m_ref, bx.B, bx.B),
                                        (by, by.m_ref, by.B, by.B), chi33)
            Sw = Gw_inv @ Gw_chi @ Gw_inv
        Stt = -Curl.conj().T @ Sw @ Curl            # (2q^2,2q^2), neg-semidef

        # === K_tz eps33^-1 K_zt : the div(D)=0 Schur term (Eq.16-18) ===
        # MIMETIC GRADIENT (the dual de Rham map).  K_tz = C[chi_t][d2;-d1] acts
        # V3 -> [E1;E2]; with chi_t=I, C[d2;-d1]=[-d1;-d2], i.e. K_tz V3 =
        #   [ -d1 V3 ; -d2 V3 ].  Since d Btil in span(B):
        #     d1 V3 = (d1 Btil)(x1) Btil(x2) in B(x1)(x)Btil(x2) = V1
        #     d2 V3 = Btil(x1) (d2 Btil)(x2) in Btil(x1)(x)B(x2) = V2
        #   so the GRADIENT maps V3 cleanly INTO [V1;V2] -- FULL RANK (the dual of
        #   the curl's exactness).  We build K_tz as the Galerkin gradient and
        #   K_zt as its adjoint through the component masses, with eps inserted in
        #   the middle (eps33^-1 on the V3 side; [eps_t] on the field side is
        #   already in L's first term -- here K_zt carries the eps of the D-field
        #   divergence d(eps E)).
        #
        # Build the eps-free mimetic gradient blocks (V1<-V3, V2<-V3):
        #   <t1 | -d1 V3> : t1 in V1=B(x1)Btil(x2), V3=Btil(x1)Btil(x2)
        #     x1: <B | d Btil> = dbt_x ; x2: <Btil|Btil> = Mtt_y
        if not magnetic:
            Grad1 = -np.kron(Mtt_y, dbt_x)          # V1 <- V3   (=<t1|-d1 V3>)
            Grad2 = -np.kron(dbt_y, Mtt_x)          # V2 <- V3   (=<t2|-d2 V3>)
            Ktz = np.concatenate([Grad1, Grad2], axis=0)   # (2q^2, q^2)
        else:
            # Eq. 21 / Appendix-A Eq. 43: K_tz = C[chi_t][d2; -d1], i.e.
            #   row 1 (tested in V1) = -chi22 d1 + chi21 d2
            #   row 2 (tested in V2) = -chi11 d2 + chi12 d1
            # -- the same C-rotation index swap as R (row 1 carries chi22 /
            # chi21, row 2 chi11 / chi12), and chi_t = I reproduces Grad1 /
            # Grad2 above exactly (to summation order).  The derivative sits on
            # the V3 TRIAL function ("d"), as the eps-free gradient has it.
            Ktz = np.concatenate([
                (-self._eps_dir(bx, "B", "d", "Btilde",
                                by, "Btilde", "m", "Btilde", wmap=chi22)
                 + self._eps_dir(bx, "B", "m", "Btilde",
                                 by, "Btilde", "d", "Btilde", wmap=chi21)) / k0,
                (-self._eps_dir(bx, "Btilde", "m", "Btilde",
                                by, "B", "d", "Btilde", wmap=chi11)
                 + self._eps_dir(bx, "Btilde", "d", "Btilde",
                                 by, "B", "m", "Btilde", wmap=chi12)) / k0,
            ], axis=0)

        # eps33 mass in V3 (Eq.41): <V3| eps | V3>
        Meps33 = self._eps_weighted(
            (bx, bx.m_ref, bx.Btilde, bx.Btilde),
            (by, by.m_ref, by.Btilde, by.Btilde), e33)

        # K_zt = [d1 eps, d2 eps] (V3 <- [E1;E2]).  This is the divergence of the
        # D-field d(eps E).  The CONSISTENT (mimetic) discretization is the eps-
        # weighted negative adjoint of the gradient: <V3| d1(eps E1)> assembled so
        # that, with eps piecewise-constant on the cells (walls on element bnds),
        # it equals  -<(d1 V3) , eps E1>  with the SAME component masses as the
        # gradient.  We assemble it via the eps-weighted component overlap of the
        # gradient blocks: K_zt = -(eps-weighted Ktz-partner)^dag is NOT generally
        # right when eps varies, so we assemble K_zt DIRECTLY in V3 with eps:
        #   <V3| d1(eps E1)> = -<d1 V3 | eps E1> = -eps-weighted <(d Btil)B>_x ...
        #   per cell: x1 -< (d Btil) | eps | B >_x ; x2 <Btil|eps|Btil>_y
        #   (deriv-on-LEFT, i.e. on the V3 test) -- realized by _eps_dir with the
        #   derivative on the TEST set Btilde of the x-axis.
        Kzt_E1 = -self._eps_dir(bx, "Btilde", "dL", "B",
                                by, "Btilde", "m", "Btilde",
                                wmap=e11) / k0             # <V3|d1(e11 .)|V1>
        Kzt_E2 = -self._eps_dir(bx, "Btilde", "m", "Btilde",
                                by, "Btilde", "dL", "B",
                                wmap=e22) / k0             # <V3|d2(e22 .)|V2>
        if tensor:
            # Eq.44 second terms: div(D_t) also picks up d2(e21 E1) in column 1
            # and d1(e12 E2) in column 2.  The derivative again sits on the V3
            # TEST function ("dL"), now on the OTHER axis than the shipped term.
            Kzt_E1 = Kzt_E1 - self._eps_dir(
                bx, "Btilde", "m", "B",
                by, "Btilde", "dL", "Btilde", wmap=e21) / k0   # <V3|d2(e21 .)|V1>
            Kzt_E2 = Kzt_E2 - self._eps_dir(
                bx, "Btilde", "dL", "Btilde",
                by, "Btilde", "m", "B", wmap=e12) / k0         # <V3|d1(e12 .)|V2>
        Kzt = np.concatenate([Kzt_E1, Kzt_E2], axis=1)     # (q^2, 2q^2)

        # Schur term  K_tz @ Meps33^{-1} @ K_zt  (eps33^-1 = solve vs Meps33).
        Schur = Ktz @ np.linalg.solve(Meps33, Kzt)

        # ----- assemble L -----
        Lmat = np.zeros((2 * qq, 2 * qq), dtype=_C)
        Lmat[:qq, :qq] = Et_11
        Lmat[qq:, qq:] = Et_22
        if tensor:
            Lmat[:qq, qq:] = Et_12
            Lmat[qq:, :qq] = Et_21
        Lmat += Stt
        Lmat -= Schur

        # Only the operators DOWNSTREAM consumers read are retained
        # (_region_modes: Lmat/Rmat/Et_blocks/Stt; _homog_geom_cache adds
        # Schur).  Curl/Kzt/Ktz/Meps33 stay assembly locals -- storing them
        # retained ~44% more dead operator memory per solver at convergence-
        # grade sizes (audit P3-37; e.g. 1.66 GB per solver at Nx=Ny=4, M=16).
        self.Rmat = Rmat
        self.Lmat = Lmat
        self.Et_blocks = (Et_11, Et_22)
        # Eq.40 mixed masses, needed ONLY by the Eq.25 H-partner recovery
        # (Lhh).  None on the scalar path -> the isotropic solver retains not
        # one byte more than before (audit P3-37).
        self.Et_offdiag = (Et_12, Et_21) if tensor else None
        self.Stt = Stt
        self.Schur = Schur
        self.dimtot = 2 * qq

    # --- OUT-OF-PLANE: the first-order staggered generator (Stage B) ---------
    def _assemble_oop(self):
        """Build the FIRST-ORDER staggered generator pencil ``A x = q B x`` on
        ``x = [E1; E2; G1; G2]`` (dimension ``4 q^2``) for a cell carrying
        OUT-OF-PLANE coupling (``e13``/``e23``/``e31``/``e32``).

        Formulation: ``docs/audits/EXPERIMENT_PMM2D_STAGGERED_OOP_2026_09_09.md``
        S2, candidate (a) -- the GO route.  Out-of-plane coupling breaks
        Granet's Eq. 16 (``div D = 0`` no longer slaves ``E3`` algebraically:
        ``E3`` appears under ``d_t`` through ``e13``/``e23`` while ``gamma``
        multiplies ``e31``/``e32``), so the second-order ``2 q^2`` pencil this
        class solves for a block-form cell does not exist here.  The de Rham
        placement is UNCHANGED (``E1`` in ``V1``, ``E2`` in ``V2``, ``E3`` in
        ``V3``, ``G1`` in ``V2``, ``G2`` in ``V1``, ``G3`` in ``Vw``), and the
        two longitudinal unknowns are eliminated in the two ways that placement
        makes exact::

            G3 = Gw^-1 (CwE2 E2 - CwE1 E1)
                 STRONG -- the curl of a transverse E lands exactly in Vw
            E3 = A33^-1 (P23^H G1 - P13^H G2 - A31 E1 - A32 E2)
                 WEAK -- the longitudinal curl-H row tested in V3 with the
                 derivative moved onto the continuous V3 test (the
                 :meth:`_eps_dir` ``"dL"`` device)

        leaving four rows, each tested in its own space, against
        ``B = blkdiag(Ggram1, Ggram2, Ggram2, Ggram1)``::

            q Ggram1 e1 = -i Ggram1 g2 - i P13 e3
            q Ggram2 e2 = +i Ggram2 g1 - i P23 e3
            q Ggram2 g1 = -i [A21 e1 + A22 e2 + A23 e3 - CwE2^H g3]
            q Ggram1 g2 = +i [A11 e1 + A12 e2 + A13 e3 + CwE1^H g3]

        Every ``i`` comes from ``D_3 = i q`` and nothing else; the transverse
        derivative blocks are plain Galerkin brackets carrying no explicit
        ``i``.  ``G = i Z0 H`` is the normalized magnetic field that makes
        Maxwell real-coefficient (``D x E = G``, ``D x G = eps E``) -- the same
        state the 1-D ``_build_generator_metric`` uses and the one
        ``rcwa._core._select_forward_flux`` reads.  ``A33`` is invertible
        exactly when ``e33 != 0`` (:func:`_tile_needs_oop` enforces it), and the
        ``e33``-Schur is POINTWISE per cell -- never a product of separately
        discretized factors, the ordering the 1-D ``gen2`` prototype got wrong.

        THE ROTATION GAUGE (:data:`_OOP_ROT_SIGN`).  The shipped ``Basis1D``
        glue is ``tau = exp(-i alpha0 p)``, so the basis carries
        ``exp(-i alpha0 x)`` -- MEASURED, not asserted:
        ``validation/probe_pmm2d_staggered_oop/g0f_basis_phase_slope.py`` fits
        the reconstructed order-m field's phase slope at -2.65539 against
        ``alpha0`` = +2.65539 (and order +/-1 at -(alpha0 +/- G) likewise) --
        while the far-field kernel and the ``eps_cell`` indexing run the other
        way.  The composition is a 180-degree rotation about ``z`` of the whole
        solve.  It is INVISIBLE to every in-plane operator (``e11, e12, e21,
        e22, e33`` are invariant under that rotation, which is why the
        isotropic and Stage-A paths never had to know about it), and it is
        exactly a SIGN on the out-of-plane block -- so the assembly applies it
        once, here.

        The discrete identity behind that equivalence: negating the four
        single-derivative blocks (``P13, P23, CwE1, CwE2``) AND the four
        out-of-plane entries leaves the pencil above invariant term for term
        (``e3`` and ``g3`` both flip sign, and every row then picks up two
        flips), so "the basis' transverse derivative runs backwards" and "the
        tensor is rotated by 180 degrees about z" are the SAME statement.  Both
        directions are measured in
        ``docs/audits/BUILD_PMM2D_STAGGERED_OOP_2026_09_09.md``: with the
        rotation a uniform out-of-plane slab reproduces ``berreman_jones_1d`` to
        ~1e-15 at conical incidence; with ``_OOP_ROT_SIGN`` flipped to +1 the
        same solve sits at ~1e-03, the size of the prototype's own
        negate-the-out-of-plane-block negative control.

        THE SLANT (roadmap Phase D, 2026-09-10).  A constant x-z / y-z shear
        adds exactly two things to the pencil below and NOTHING else: the
        POINTWISE congruence ``eps -> A^-1 eps A^-T`` (applied in ``__init__``,
        :func:`_slant_congruence`) and SIX extra Galerkin blocks at the end of
        this method, guarded by ``t != 0``.  ``det A = 1`` is what keeps the
        rest untouched -- ``mu^{33} = g^{33} = 1`` exactly, so the STRONG
        ``G3`` elimination survives, and ``eps^{33} = eps_zz`` is unchanged by
        the congruence, so the POINTWISE ``e33``-Schur survives.  The retained
        state ``[E_1; E_2; G_1; G_2]`` is the COVARIANT tangential state, which
        EQUALS the lab-Cartesian one (a shear alters only the NORMAL
        component), so :func:`_region_modes_oop`, the flux split, the H gauge,
        the generalized cascade and the far-field projector all keep their
        meanings verbatim.  Derivation and every measured number:
        ``docs/audits/EXPERIMENT_PMM2D_STAGGERED_SLANT_2026_09_10.md`` S1 and
        ``docs/audits/BUILD_PMM2D_STAGGERED_SLANT_2026_09_10.md``.
        """
        bx, by = self.bx, self.by
        k0 = self.k0
        self._axis_mats()
        qq = self.q * self.q
        e = self.eps_cell
        # A SLANTED cell arrives already in the rotated gauge -- ``__init__``
        # folded _OOP_ROT_SIGN into BOTH the out-of-plane entries and the slant
        # vector before taking the congruence (:func:`_slant_rot_gauge`), which
        # is the only way the two stay consistent.  Applying it a second time
        # here would undo it.
        rot = 1.0 if self._eps_pre_rotated else _OOP_ROT_SIGN
        tx, ty = self._slant_rot
        e11, e12, e13 = e[..., 0, 0], e[..., 0, 1], rot * e[..., 0, 2]
        e21, e22, e23 = e[..., 1, 0], e[..., 1, 1], rot * e[..., 1, 2]
        e31, e32 = rot * e[..., 2, 0], rot * e[..., 2, 1]
        e33 = e[..., 2, 2]

        # ----- eps-free geometry: block Grams + the mimetic derivatives -----
        Mtt_x, Mtt_y = self.Mtt_x, self.Mtt_y
        Mbb_x, Mbb_y = self.Mbb_x, self.Mbb_y
        dbt_x = bx.mixed(bx.B, bx.Btilde) / k0        # <B | d til>_x
        dbt_y = by.mixed(by.B, by.Btilde) / k0
        Ggram1 = np.kron(Mtt_y, Mbb_x)                # <V1|V1>
        Ggram2 = np.kron(Mbb_y, Mtt_x)                # <V2|V2>
        Gw = np.kron(Mbb_y, Mbb_x)                    # <Vw|Vw>
        CwE1 = np.kron(dbt_y, Mbb_x)                  # <Vw| D2 |V1>
        CwE2 = np.kron(Mbb_y, dbt_x)                  # <Vw| D1 |V2>
        P13 = np.kron(Mtt_y, dbt_x)                   # <V1| D1 |V3>
        P23 = np.kron(dbt_y, Mtt_x)                   # <V2| D2 |V3>

        # ----- the NINE eps-weighted component masses: Appendix-A Eq. 40/41
        #       (A11/A22 shipped, A12/A21 Stage A) plus the FOUR NEW
        #       out-of-plane blocks A13/A23/A31/A32.  Same kron assembly, one
        #       component map each -- no new basis code.
        ew = self._eps_weighted
        xB, xT = bx.B, bx.Btilde
        yB, yT = by.B, by.Btilde
        mx, my = bx.m_ref, by.m_ref
        A11 = ew((bx, mx, xB, xB), (by, my, yT, yT), e11)
        A12 = ew((bx, mx, xB, xT), (by, my, yT, yB), e12)
        A13 = ew((bx, mx, xB, xT), (by, my, yT, yT), e13)
        A21 = ew((bx, mx, xT, xB), (by, my, yB, yT), e21)
        A22 = ew((bx, mx, xT, xT), (by, my, yB, yB), e22)
        A23 = ew((bx, mx, xT, xT), (by, my, yB, yT), e23)
        A31 = ew((bx, mx, xT, xB), (by, my, yT, yT), e31)
        A32 = ew((bx, mx, xT, xT), (by, my, yT, yB), e32)
        A33 = ew((bx, mx, xT, xT), (by, my, yT, yT), e33)

        # ----- eliminate E3 (weak, in V3) and G3 (strong, in Vw) ------------
        Z = np.zeros((qq, qq), dtype=_C)
        E3S = np.linalg.solve(
            A33, np.concatenate([-A31, -A32, P23.conj().T, -P13.conj().T],
                                axis=1))
        G3S = np.linalg.solve(Gw, np.concatenate([-CwE1, CwE2, Z, Z], axis=1))

        row0 = (-1j * np.concatenate([Z, Z, Z, Ggram1], axis=1)
                - 1j * (P13 @ E3S))
        row1 = (1j * np.concatenate([Z, Z, Ggram2, Z], axis=1)
                - 1j * (P23 @ E3S))
        row2 = (-1j * np.concatenate([A21, A22, Z, Z], axis=1)
                - 1j * (A23 @ E3S) + 1j * (CwE2.conj().T @ G3S))
        row3 = (1j * np.concatenate([A11, A12, Z, Z], axis=1)
                + 1j * (A13 @ E3S) + 1j * (CwE1.conj().T @ G3S))

        # ----- THE SIX SLANT BLOCKS (roadmap Phase D) -----------------------
        # Guarded: a VERTICAL cell never enters here, which is what makes
        # ``slant = 0`` BIT-IDENTICAL to the pre-slant assembly (gate B1).
        #
        # In the sheared frame Maxwell is the vertical system with exactly two
        # rigid substitutions (EXPERIMENT doc S1.3):
        #
        #   (2) q E1 = -i G2 - i D1 E3 + i t_y G^3           [V1]
        #   (1) q E2 = +i G1 - i D2 E3 - i t_x G^3           [V2]
        #   (5) q G1 = -i (eps E)_2 - i D1 (G^3 + t_x G1 + t_y G2)   [V2]
        #   (4) q G2 = +i (eps E)_1 - i D2 (G^3 + t_x G1 + t_y G2)   [V1]
        #
        # i.e. the two E rows pick up ``t G^3`` (and ``G^3`` is ALREADY the
        # strongly eliminated ``G3S``, so this is one cross mass times it), and
        # the two G rows carry the COVARIANT ``G_3cov = G^3 + t.G_t`` under the
        # existing single-derivative bracket.  Every matrix below is a
        # Kronecker product of per-axis matrices the basis already supplies --
        # no new basis code, no new quadrature.
        #
        # ``dtb = -(dbt)^H`` is the SAME integration-by-parts identity the
        # shipped assembly uses for CwE1/CwE2, and it is the distributionally
        # exact ``<Btilde | d B>`` -- deltas included.  The ELEMENT-WISE
        # ``b.mixed(b.Btilde, b.B)`` (i.e. ``self.Ctb_*``) is NOT the same
        # matrix: it silently drops the jump deltas of the discontinuous set.
        if tx != 0.0 or ty != 0.0:
            Mtb_x = bx.mass(bx.Btilde, bx.B)          # <til|B>_x
            Mtb_y = by.mass(by.Btilde, by.B)
            Mbt_x = bx.mass(bx.B, bx.Btilde)          # <B|til>_x
            Mbt_y = by.mass(by.B, by.Btilde)
            dtb_x = -dbt_x.conj().T                   # <til| D1 |B>_x
            dtb_y = -dbt_y.conj().T
            ctt_x = self.Ctt_x / k0                   # <til| D1 |til>_x
            ctt_y = self.Ctt_y / k0
            MwV1 = np.kron(Mtb_y, Mbb_x)              # <V1|Vw>
            MwV2 = np.kron(Mbb_y, Mtb_x)              # <V2|Vw>
            D1_22 = np.kron(Mbb_y, ctt_x)             # <V2| D1 |V2>
            D1_21 = np.kron(Mbt_y, dtb_x)             # <V2| D1 |V1>
            D2_11 = np.kron(ctt_y, Mbb_x)             # <V1| D2 |V1>
            D2_12 = np.kron(dtb_y, Mbt_x)             # <V1| D2 |V2>
            # E rows: +i t_y G^3 (row0, tested in V1) / -i t_x G^3 (row1, V2)
            row0 = row0 + 1j * ty * (MwV1 @ G3S)
            row1 = row1 - 1j * tx * (MwV2 @ G3S)
            # G rows: the t.G_t half of G_3cov under the D-bracket
            row2 = row2 - 1j * np.concatenate(
                [Z, Z, tx * D1_22, ty * D1_21], axis=1)
            row3 = row3 - 1j * np.concatenate(
                [Z, Z, tx * D2_12, ty * D2_11], axis=1)

        Bgen = np.zeros((4 * qq, 4 * qq), dtype=_C)
        Bgen[:qq, :qq] = Ggram1
        Bgen[qq:2 * qq, qq:2 * qq] = Ggram2
        Bgen[2 * qq:3 * qq, 2 * qq:3 * qq] = Ggram2
        Bgen[3 * qq:, 3 * qq:] = Ggram1

        # Only the pencil is retained (audit P3-37): the fifteen assembly
        # blocks above die with this frame, exactly as Curl/Kzt/Ktz/Meps33 do
        # on the in-plane path.  The second-order operators do not EXIST for an
        # out-of-plane cell, so they are None rather than stale -- every
        # consumer dispatches on ``self.offplane``.
        self.Agen = np.concatenate([row0, row1, row2, row3], axis=0)
        self.Bgen = Bgen
        self.Rmat = None
        self.Lmat = None
        self.Et_blocks = None
        self.Et_offdiag = None
        self.Stt = None
        self.Schur = None
        self.dimtot = 4 * qq

    # --- eps-weighted directed (deriv) 2-D operator into V3 ------------------
    def _eps_dir(self, bx, lx, opx, rx, by, ly, opy, ry, wmap=None):
        """Assemble  sum_{sx,sy} eps[sx,sy] * kron(Gy, Gx)  where Gx is the
        per-segment x-matrix between sets (lx-test, rx-trial) with op opx in
        {'m'(mass), 'd'(deriv-on-trial)}, similarly Gy.  Realizes the eps-
        weighted <V3 | d_k (eps .) | Vt> matrix exactly (walls on cell bnds).

        ``wmap`` is the per-cell weight map; ``None`` uses ``self.eps_cell``
        (the isotropic default).  The tensor K_zt columns (Eq. 44) pass one
        component map per term."""
        def segmat(basis, lset, op, rset):
            sL = getattr(basis, lset)
            sR = getattr(basis, rset)
            Lt = np.array(sL)
            Rt = np.array(sR)
            # SITE 3 of 4 for non-uniform segments: per-segment ``J_n``.  On
            # the uniform path ``Jn`` is the constant ``J`` this multiplied
            # by, so the assembly is bit-identical.
            if op == "m":                           # mass (no deriv)
                RR = np.einsum("ab,jsb->jsa", basis.m_ref, Rt)
                scale = basis.Jn
            elif op == "dL":                        # deriv on LEFT (test): INT (dL) R
                # c_ref[a,b]=INT La' Lb -> INT (dL_a) R_b = conj(L)@c_ref@R
                RR = np.einsum("ab,jsb->jsa", basis.c_ref, Rt)
                scale = np.ones(basis.N)
            else:                                   # 'd' deriv on trial (right): INT L (dR)
                RR = np.einsum("ab,jsb->jsa", basis.c_ref.T, Rt)
                scale = np.ones(basis.N)
            return np.asarray(scale)[:, None, None] * np.einsum(
                "isa,jsa->sij", np.conj(Lt), RR)
        Gx = segmat(bx, lx, opx, rx)
        Gy = segmat(by, ly, opy, ry)
        eps = self.eps_cell if wmap is None else wmap
        out = np.zeros((Gy.shape[1] * Gx.shape[1],
                        Gy.shape[2] * Gx.shape[2]), dtype=_C)
        for sx in range(bx.N):
            Wy = np.einsum("y,yij->ij", eps[sx, :], Gy)
            out += np.kron(Wy, Gx[sx])
        return out



# === full canonical solve: square all-SEM match + Li S-matrix + far field ===
# =========================================================================== #
# decay-branch helpers (public exp(-i w t))
# =========================================================================== #
def _sqrt_decay(x):
    r = np.sqrt(np.asarray(x, dtype=_C))
    on_cut = r.real == 0
    return np.where(on_cut & (r.imag < 0), -r, r)


def _inv_lam(lam):
    safe = np.where(np.abs(lam) < 1e-12, 1e-12, lam)
    return 1.0 / safe


def _kz_forward2(eps, kx, ky):
    val = np.sqrt(np.asarray(eps - kx ** 2 - ky ** 2, dtype=_C))
    return np.where(val.imag < 0.0, -val, val)


# =========================================================================== #
# Fourier projection of the staggered modified-Legendre 1-D set onto Rayleigh
# orders:   T[m, j] = (1/d) INT_0^d  phi_j(x) exp(-i m G x) dx,   G = 2pi/d.
# A global staggered dof j has local modified-Legendre coefficients S[j][seg, a]
# (the same (N,M) stencil the eigensolver stores).  We integrate each local
# modified-Legendre function exactly by Gauss-Legendre quadrature on its segment.
# This is the staggered-basis analogue of pmm._sem_fourier_projection.
# =========================================================================== #
#: MEASURED cost of the OSCILLATORY factor in the projection kernel, in Gauss
#: nodes per unit of a segment's own HALF-PHASE
#: ``omega_n = |m G + alpha0| J_n`` -- the slope of the smallest ``nq`` that
#: reproduces a refined rule to the refined rule's OWN floor.
#:
#: MEASURED 2026-09-11 (``validation/probe_pmm2d_mortar_round2/r2_quad.py``,
#: section ``need``; identical on both builds) over ``M = 3..12`` x
#: ``omega = 0..128``, with the bar derived at each point from the reference
#: rule's own 37-node self-drift rather than pinned: the slope reads
#: **0.6341 .. 0.6455** across all eight ``M`` (a spread of 1.8 %, which is
#: what makes it a predictor and not a fit) and the intercept
#: **9.38 + 0.47 (M - 3)**.  The shipped constants below are an UPPER envelope
#: of that measurement -- ``0.72`` against a measured 0.645, and
#: ``0.5 M + 10`` against a measured ``0.47 M + 8.0`` -- so the rule clears
#: every measured requirement (worst margin re-measured in the probe) while
#: still fitting UNDER the ``2 M + 8`` reserve on every lattice the uniform
#: path can build.
_STAG_QUAD_OMEGA = 0.72
_STAG_QUAD_M = 0.5
_STAG_QUAD_CONST = 10.0


def _stag_quad_order(M, omega):
    """Gauss nodes for ONE segment carrying half-phase ``omega`` (D3).

    ``2 M + 8`` -- the historical rule, sized for a segment of length ``d/N``
    -- is kept as a FLOOR, and topped up only when the segment's own phase
    outruns it.  On an INTEGER-``N`` lattice the caller does not reach this
    function at all (see :func:`_stag_fourier_projection`), so the uniform path
    is bit-identical by construction; on an explicitly-passed uniform ARRAY the
    formula returns the same ``2 M + 8`` for every ``(M, N)`` the shipped order
    cap allows with ``|alpha0| <= G/2``, which is what keeps the two spellings
    ULP-close (gate N2)."""
    base = 2 * int(M) + 8
    need = (_STAG_QUAD_OMEGA * float(omega) + _STAG_QUAD_M * int(M)
            + _STAG_QUAD_CONST)
    if need <= base:
        return base
    return int(np.ceil(need))


def _stag_fourier_projection(basis: Basis1D, orders, alpha0=0.0):
    d, N, M = basis.d, basis.N, basis.M
    G = 2.0 * np.pi / d
    xb = basis.xb
    orders = np.asarray(orders)
    # D3 (VERIFY_PMM2D_STAGGERED_MORTAR_2026_09_11 S2.6 / S11): the rule is
    # sized PER SEGMENT from that segment's own half-phase.  ``2 M + 8`` was
    # sized for a segment of length ``d/N``, on which the phase a segment
    # carries is bounded by the far-field order cap; with arbitrary walls one
    # segment can be almost the whole period and the kernel is then
    # under-resolved (measured 7.5e-04 relative at longest segment 0.96 d,
    # ``M = 4``, orders to 7, against 6-8e-15 on a uniform ``N = 3`` lattice).
    # The INTEGER path keeps the historical single rule EXACTLY -- one
    # ``leggauss`` call, one ``_modleg_value_deriv``, the same doubles -- so
    # every integer-``N`` projector is bit-identical to the pre-2026-09-11
    # library (hashed in the probe and in the shipped gate).
    kvec = orders * G + alpha0
    if basis.uniform:
        nq_of = None
        nq0 = 2 * M + 8
    else:
        kmax = float(np.max(np.abs(kvec))) if np.size(kvec) else 0.0
        nq_of = [_stag_quad_order(M, kmax * basis.Jn[s]) for s in range(N)]
        nq0 = nq_of[0]
    nq = nq0
    xg, wg = leggauss(nq)                       # on [-1,1]
    Vref, _ = _modleg_value_deriv(M, xg)        # (M, nq) modified-Legendre values
    _rules = {nq: (xg, wg, Vref)}
    # per-segment contribution of local function a to Rayleigh order m.  The
    # modal field is a BLOCH mode: the tau-glued basis carries the transverse
    # momentum (tau = exp(-i alpha0 d)), so the physical field is
    # ~ exp(-i alpha0 x) p(x), and the library's Rayleigh order m (far-field
    # kxv = kx0 + m*wl/period) has transverse dependence e^{-i(alpha0 + mG)x}.
    # Its amplitude is recovered by projecting on the CONJUGATE kernel
    # e^{+i(mG + alpha0) x}.  Both pieces of the kernel matter and each failure
    # is invisible in a subset of tests: omitting ``alpha0`` breaks the
    # one-period orthogonality at oblique (energy loss even in the specular
    # order); flipping the sign of ``m`` deposits physical order -m into slot m
    # (a MIRROR) -- exact at normal incidence (kz even in kx) and for
    # reflection-symmetric cells, but at oblique every |m|>0 order gets the
    # WRONG kz flux factor (kz(+m) applied to order -m's amplitude), which
    # leaked/gained several % energy for patterned cells (asymmetric-cell
    # per-order oracle test vs pmm_efficiency_2d_cell pinned the form
    # stag[m] = oracle[-m]*kz(m)/kz(-m) before this fix).
    #   c[m, seg, a] = (1/d) INT_seg Ltilde_a(u(x)) e^{+i(mG + alpha0)x} dx
    # SITE 4 of 4 for non-uniform segments: the quadrature points and the
    # ``J/d`` weight come from segment ``n``'s OWN affine map (Eq. 31).  On the
    # uniform path ``Jn[seg]`` is the constant ``J``, so the projector is
    # bit-identical.
    T_local = np.zeros((len(orders), N, M), dtype=_C)
    for seg in range(N):
        J = basis.Jn[seg]
        if nq_of is not None and nq_of[seg] != nq:
            nq = nq_of[seg]
            hit = _rules.get(nq)
            if hit is None:
                xg, wg = leggauss(nq)
                Vref, _ = _modleg_value_deriv(M, xg)
                _rules[nq] = (xg, wg, Vref)
            else:
                xg, wg, Vref = hit
        xphys = 0.5 * (xb[seg] + xb[seg + 1]) + J * xg     # physical x at quad pts
        phase = np.exp(1j * np.outer(kvec, xphys))         # (nO, nq)
        # contribution[m,a] = (J/d) sum_q phase[m,q] wg[q] Vref[a,q]
        T_local[:, seg, :] = (J / d) * (phase * wg) @ Vref.T
    # assemble onto the global dofs of a chosen set (list of (N,M) stencils)
    def _assemble(global_set):
        S = np.array(global_set)               # (dim, N, M)
        # T[m, j] = sum_{seg,a} T_local[m,seg,a] * S[j,seg,a]
        return np.einsum("msa,jsa->mj", T_local, S)
    return _assemble


def _far_projector_2d(bx: Basis1D, by: Basis1D, ox, oy, alpha0x=0.0, alpha0y=0.0):
    """Forward Fourier->Rayleigh projectors for the 2-D staggered field
    components.  Returns the per-component (E1,E2) projection operators that map
    a region's [E1;E2] modal coefficient vector onto the Rayleigh orders.

    E1 = sum B(x1) Btilde(x2)  -> project x1 with B-set, x2 with Btilde-set.
    E2 = sum Btilde(x1) B(x2)  -> project x1 with Btilde-set, x2 with B-set.
    Tensor ordering matches the eigensolver's kron(y, x): index I = jx + qx*jy.

    ``alpha0x``/``alpha0y`` are the per-axis Bloch wavenumbers (``kx0*k0`` /
    ``ky0*k0``); the projection kernel carries them so the once-only far-field
    Rayleigh projection stays ORTHOGONAL at oblique incidence (see
    :func:`_stag_fourier_projection`).  Default 0 = normal incidence (byte-
    identical to the un-shifted projection).
    """
    asmx = _stag_fourier_projection(bx, ox, alpha0x)
    asmy = _stag_fourier_projection(by, oy, alpha0y)
    Tx_B = asmx(bx.B)            # (Mx, qx)
    Tx_til = asmx(bx.Btilde)    # (Mx, qx)
    Ty_B = asmy(by.B)            # (My, qy)
    Ty_til = asmy(by.Btilde)    # (My, qy)
    # E1 in V1 = B(x1) (x) Btilde(x2):  P1 = kron(Ty_til, Tx_B)
    P1 = np.kron(Ty_til, Tx_B)   # (Mx*My, qx*qy)
    # E2 in V2 = Btilde(x1) (x) B(x2):  P2 = kron(Ty_B, Tx_til)
    P2 = np.kron(Ty_B, Tx_til)
    return P1, P2


def _pmm2d_project_orders(P1, P2, Wmodes, qq):
    """Project a PMM-2D ``[E1; E2]`` modal matrix onto the Rayleigh orders --
    the ``_proj`` closure formerly duplicated in :mod:`.stack2d_pure` and this
    module (audit S1-10).  ``P1``/``P2`` map the E1 (Ex) / E2 (Ey) nodal blocks
    (``qq`` rows each) onto the orders; returns the stacked
    ``[Ex_orders; Ey_orders]``.  Reproduces the former closure operation-for-
    operation."""
    top = P1 @ Wmodes[:qq, :]
    bot = P2 @ Wmodes[qq:, :]
    return np.concatenate([top, bot], axis=0)


def _pmm2d_order_kz(eps_sup, eps_sub, kxv, kyv, kx0, ky0):
    """Per-order forward ``kz`` for the two half-spaces, the incident ``kz``, and
    the safe-divide ``kz`` used by the longitudinal-field reconstruction -- the
    ``kz_ref``/``kz_trn``/``kz_inc``/``safe_r``/``safe_t`` block formerly
    duplicated in :mod:`.stack2d_pure` and this module (audit S1-10).  Returns
    ``(kz_ref, kz_trn, kz_inc, safe_r, safe_t)``, reproducing the former inline
    block byte-for-byte."""
    kz_ref = _kz_forward2(eps_sup, kxv, kyv)
    kz_trn = _kz_forward2(eps_sub, kxv, kyv)
    kz_inc = float(np.real(_kz_forward2(eps_sup, kx0, ky0)))
    safe_r = np.where(np.abs(kz_ref) < 1e-12, 1.0, kz_ref)
    safe_t = np.where(np.abs(kz_trn) < 1e-12, 1.0, kz_trn)
    return kz_ref, kz_trn, kz_inc, safe_r, safe_t


# =========================================================================== #
# Region modes from the staggered eigensolver, with the Eq.25 H-partner.
# =========================================================================== #
def _region_modes(solver: Granet2DTransverseE):
    """Solve  L v = g2 G v  (G = -R, block field Gram), return:
        W    : tangential-E modal matrix [E1;E2] coefficients (2 q^2, 2 q^2)
        V    : tangential-H modal partner via Eq.25
        lam  : z-decay constants  sqrt(-g2)  (forward Im>=0)
        g2   : eigenvalues (gamma/k0)^2
    """
    if solver.offplane:
        raise ValueError(
            "_region_modes: this solver assembled the OUT-OF-PLANE "
            "(first-order, 4 q^2) generator, which has no second-order pencil "
            "and DISTINCT forward/backward modes -- call _region_modes_oop, "
            "whose 6-tuple feeds the generalized S-matrix cascade.")
    L = solver.Lmat
    # The pencil's right-hand matrix is -R = -C[chi_t]C (Hermitian PD).  For a
    # NONMAGNETIC region chi_t = I, so -R IS the block field Gram
    # blockdiag(G1, G2) and the one object serves both roles below; for a
    # MAGNETIC region the two are DIFFERENT operators and the Gram is carried
    # separately on ``solver.Ggram_blocks`` (see the H recovery below).
    G = -solver.Rmat
    g2, W = sla.eig(L, G)
    # q = kz/k0 = gamma/k0 = sqrt(g2).  FORWARD branch chosen ROBUSTLY (the
    # naive _sqrt_decay flips degenerate real-g2 pairs inconsistently on QZ
    # noise -> the H-partner sign flips and the S-matrix loses passivity).  We
    # pick Im(q) >= 0 (evanescent decay) with a noise tolerance, and for the
    # (near-)real propagating modes Re(q) > 0 (outgoing).  lam = -i q forward.
    q = np.sqrt(np.asarray(g2, dtype=_C))
    q = _forward_branch_flip(q)       # shared scalar-vertical selector (S1-8)
    lam = -1j * q                     # forward propagator exp(-lam k0 z) decays
    gamma_over_k0 = q

    # H recovery (Eq.25):  gamma C [H1;H2] = Lhh [E1;E2]
    #   Lhh = Et + Stt  (the L WITHOUT the div(D)=0 Schur term).
    #   [H1;H2] = (1/gamma) C^{-1} Lhh [E]  = (k0/(gamma/k0)) (-C) G^{-1} Lhh [E].
    # For a BLOCK-FORM TENSOR cell [eps_t] carries the Eq.40 MIXED blocks too,
    # so Lhh is full 2x2-block (H1 still lives in V2 and H2 in V1, so the
    # interface match stays a SQUARE modal match -- the cascade is untouched).
    Et11, Et22 = solver.Et_blocks
    qq = solver.q * solver.q
    Lhh = np.zeros_like(L)
    Lhh[:qq, :qq] = Et11
    Lhh[qq:, qq:] = Et22
    if solver.Et_offdiag is not None:
        Et12, Et21 = solver.Et_offdiag
        Lhh[:qq, qq:] = Et12
        Lhh[qq:, :qq] = Et21
    Lhh = Lhh + solver.Stt
    # block C^{-1} = -C = [[0,-1],[1,0]] acting on the 2-block coeff vector:
    #   (-C) [a;b] = [-b; a]  (a = top block, b = bottom block).
    #
    # THE R-vs-GRAM SEPARATION.  Eq. 25 is a WEAK statement tested in V1 / V2:
    # <v, gamma C [H1;H2]> = <v, (k^2[eps_t] + S_tt)[E1;E2]>, and its left side
    # is gamma times the PLAIN block Gram blockdiag(G1, G2) applied to the
    # C-rotated H coefficients -- NO chi_t appears in Eq. 25 (only chi33, which
    # is already inside S_tt).  On the nonmagnetic path -R is that Gram, so the
    # shipped single inverse is right; on the magnetic path -R = -C[chi_t]C is
    # a DIFFERENT operator and using it here would silently apply [chi_t]^-1 to
    # every H partner (an interface-match error invisible to the eigenvalues
    # and to any energy check that renormalises).  Solve blockwise against the
    # two retained Gram blocks instead.
    if solver.Ggram_blocks is None:
        Ginv = np.linalg.inv(G)
        Dual = Ginv @ (Lhh @ W)      # G^{-1} Lhh W  (back to coefficients)
    else:
        G1g, G2g = solver.Ggram_blocks
        LW = Lhh @ W
        Dual = np.concatenate([np.linalg.solve(G1g, LW[:qq, :]),
                               np.linalg.solve(G2g, LW[qq:, :])], axis=0)
    top = Dual[:qq, :]
    bot = Dual[qq:, :]
    rot = np.concatenate([-bot, top], axis=0)     # (-C) Dual
    inv_g = _inv_lam(gamma_over_k0)               # 1/(gamma/k0)
    V = rot * inv_g[None, :]
    return W, V, lam, g2


def _stag_parity_1d(basis: Basis1D):
    """The exact PARITY ``x -> d - x`` of one axis' two staggered global sets,
    as SIGNED PERMUTATIONS ``(perm_t, sign_t, perm_b, sign_b)`` -- or ``None``
    when the Bloch glue ``tau != 1`` (oblique incidence), where the map is not
    a signed permutation of the set at all.

    Derivation (all three pieces are properties of the shipped
    :class:`Basis1D`, not new discretization).  Segment ``n`` of the uniform
    wall grid maps to ``N-1-n`` and the reference coordinate to ``-u``, so the
    local modified-Legendre functions permute::

        Ltilde_1(-u) = (1+u)/2 = Ltilde_2(u)      (the two HALF-HATS swap)
        Ltilde_2(-u) = Ltilde_1(u)
        (L_a - L_{a-2})(-u) = (-1)^a (L_a - L_{a-2})(u)   (BUBBLES: a sign)

    Lifting that through the two global stencils of
    :meth:`Basis1D._build_sets`:

    * ``Btilde`` -- the continuous set.  Its hat at node ``j`` glues
      ``Ltilde_2`` of segment ``j-1`` to ``Ltilde_1`` of segment ``j``, and the
      half-hat swap turns that into the hat at node ``(N - j) mod N``, sign
      ``+1``.  The seam hat ``j = 0`` carries ``tau`` on the ``Ltilde_2`` leg
      while its image carries ``tau`` on the other leg, so it is fixed ONLY
      when ``tau = 1`` -- which is why the gauge is normal-incidence-only.  Its
      bubbles ``(seg, a)`` go to ``(N-1-seg, a)`` with sign ``(-1)^a``.
    * ``B`` -- the discontinuous partner.  Its two per-segment half-hats are
      INDEPENDENT dofs, so ``(seg, 0) <-> (N-1-seg, 1)`` with sign ``+1``, and
      its bubbles ``(seg, a)``, ``a = 2 .. M-2``, go to ``(N-1-seg, a)`` with
      sign ``(-1)^a`` exactly as above.

    Both maps are involutions (``J^2 = I`` EXACTLY -- a permutation composed
    with itself, signs squared) and the sign is constant on every orbit, which
    is what lets :func:`_stag_block_eig` use the same ``(perm, sign)`` algebra
    :func:`~lumenairy.elements.rcwa._core._generator_block_eig` uses for the
    Fourier order flip.
    """
    if basis.tau != 1.0:
        return None
    if not basis.uniform:
        # NON-UNIFORM walls: ``x -> d - x`` sends segment ``n`` to ``N-1-n``,
        # which is a signed permutation of the LOCAL functions only when the
        # two segments have the same length (their affine maps must agree).  A
        # mirror-symmetric wall set qualifies; anything else does not, and the
        # honest answer is "no parity structure" -- the caller then runs the
        # dense 4 q^2 eig, which is the correct-by-construction path.
        seglen = np.diff(basis.xb)
        if not np.allclose(seglen, seglen[::-1], rtol=0.0,
                           atol=1e-13 * basis.d):
            return None
    N, M = basis.N, basis.M
    seg = np.arange(N)
    # --- Btilde: N hats (node j) then bubbles at N + seg*(M-2) + (a-2)
    perm_t = np.empty(N + N * (M - 2), dtype=np.intp)
    sign_t = np.empty(perm_t.size)
    perm_t[:N] = (N - seg) % N
    sign_t[:N] = 1.0
    a_t = np.arange(2, M)
    perm_t[N:] = (N + (N - 1 - seg)[:, None] * (M - 2)
                  + (a_t - 2)[None, :]).ravel()
    sign_t[N:] = np.broadcast_to((-1.0) ** a_t, (N, M - 2)).ravel()
    # --- B: per segment [half-hat a=0, half-hat a=1, bubbles a=2..M-2];
    #     local slot l carries degree a = l for l >= 2, and 0 <-> 1 swap
    loc = np.arange(M - 1)
    swap = loc.copy()
    swap[0], swap[1] = 1, 0
    s_loc = np.where(loc >= 2, (-1.0) ** loc, 1.0)
    perm_b = ((N - 1 - seg)[:, None] * (M - 1) + swap[None, :]).ravel()
    sign_b = np.broadcast_to(s_loc, (N, M - 1)).ravel().copy()
    return perm_t, sign_t, perm_b.astype(np.intp), sign_b


def _stag_parity_gauge(solver: Granet2DTransverseE):
    """``(perm, r)`` -- the signed permutation ``R = S . blkdiag(P1, P2, P2,
    P1)`` on the out-of-plane state ``[E1; E2; G1; G2]``, or ``None`` when the
    necessary conditions fail.

    ``P1`` and ``P2`` are the 2-D parities of ``V1 = B(x) (x) Btilde(y)`` and
    ``V2 = Btilde(x) (x) B(y)`` (krons of :func:`_stag_parity_1d`, in the
    module's ``kron(y, x)`` index order), and ``S = diag(I, I, -I, -I)`` is the
    E/H SIGN flip -- the staggered analogue of the Fourier
    ``R = S (I4 (x) F)`` of
    :func:`~lumenairy.elements.rcwa._core._generator_block_eig`.

    WHY THAT SIGN PATTERN (derived on THIS state ordering, not inherited).
    With every component map ``e11 .. e33`` parity-EVEN on the grid, every
    eps-weighted mass of :meth:`Granet2DTransverseE._assemble_oop` is
    parity-EVEN (``P_i A_ij P_j = A_ij``) while each of the four
    SINGLE-DERIVATIVE blocks is parity-ODD (``P13, P23, CwE1, CwE2 -> -``,
    because ``d/dx -> -d/dx``).  Feeding ``e1 -> P1 e1``, ``e2 -> P2 e2``,
    ``g1 -> -P2 g1``, ``g2 -> -P1 g2`` through the two eliminations then gives
    ``e3 -> +P3 e3`` (its eps terms and its derivative terms each pick up two
    flips) and ``g3 -> -Pw g3``, after which every one of the four pencil rows
    has its ``B`` side EVEN and its ``A`` side ODD: ``R A R = -A`` and
    ``R B R = B``, i.e. ``(q, x)`` a solution implies ``(-q, R x)``.  Neither
    factor works alone -- the parity alone is broken by the derivative blocks
    and the sign alone by the eps blocks, exactly as in the Fourier case.

    Necessary conditions only, and all three are free of the assembly: NORMAL
    incidence (``tau = 1`` on both axes, else the hats do not permute), a
    VERTICAL cell (a SHEAR is refused here BY CONSTRUCTION -- measured, the
    structural residual does NOT catch it, so this is the only gate there is;
    see the comment on that check), and matching per-axis dimensions.
    Everything else -- a cell whose eps grid is not its own parity image, a wall
    layout that is not mirror-symmetric, a tensor that breaks the symmetry -- is
    decided by :func:`_stag_block_eig` on the ASSEMBLED pencil, which is where
    the condition actually lives.
    """
    if solver.alpha0x != 0.0 or solver.alpha0y != 0.0:
        return None
    # A SLANTED cell is refused HERE, unconditionally -- and the reason is
    # MEASURED, not assumed (build doc
    # docs/audits/BUILD_PMM2D_STAGGERED_SLANT_2026_09_10.md table B11a).
    #
    # The expectation going in was that a shear breaks the symmetry and
    # :func:`_stag_block_eig`'s structural residual would catch it.  IT DOES
    # NOT.  ``R`` here is a 180-degree ROTATION about z (both axes flip), not a
    # mirror, and a rotation carries the sheared cell's covariant tensor AND
    # its slant vector consistently -- so on a centro-symmetric slanted cell at
    # normal incidence ``max|R A R + A| / max|A|`` reads 1.3e-15 .. 2.3e-15,
    # FIVE DECADES BELOW :data:`_STAG_BLOCK_TOL`, i.e. the structural gate
    # ACCEPTS.  (Forced onto that pencil the reduction also reproduces the dense
    # spectrum to 1.0e-12 and satisfies the original pencil to 2.8e-14, so no
    # wrong answer is known here -- see the build doc's open item.)
    #
    # The refusal is therefore the ONLY thing standing between a slanted cell
    # and this accelerator, and it is deliberate: the shear is a new geometry
    # whose forward/backward split, gauge and reconstruction have been
    # validated ONLY on the dense branch, and the precedent for not letting a
    # slanted answer ride on a tolerance is the hybrid's NORMAL-INCIDENCE
    # SILENT-WRONG -- its even-parity fold was eligible at normal incidence,
    # bypassed the convection entirely, and returned the VERTICAL answer with
    # energy conserved and nothing warned
    # (docs/audits/BUILD_PMM2D_SLANT_METRIC_2026_08_16.md S8: wrong by 2.5e-01
    # at 35 degrees, while every OBLIQUE test passed with the bug present).
    # Re-enabling the reduction for slanted cells is a follow-up that needs its
    # own two-sided gate, not a tolerance.
    if not _slant_is_zero(getattr(solver, "slant", None)):
        return None
    px = _stag_parity_1d(solver.bx)
    py = _stag_parity_1d(solver.by)
    if px is None or py is None:
        return None
    ptx, stx, pbx, sbx = px
    pty, sty, pby, sby = py
    q = solver.q
    if ptx.size != q or pbx.size != q or pty.size != q or pby.size != q:
        return None
    qq = q * q
    # V1 = B(x) (x) Btilde(y): flat index iy*q + ix (the module's kron order)
    p1 = (pty[:, None] * q + pbx[None, :]).ravel()
    s1 = (sty[:, None] * sbx[None, :]).ravel()
    # V2 = Btilde(x) (x) B(y)
    p2 = (pby[:, None] * q + ptx[None, :]).ravel()
    s2 = (sby[:, None] * stx[None, :]).ravel()
    perm = np.concatenate([p1, p2 + qq, p2 + 2 * qq, p1 + 3 * qq])
    r = np.concatenate([s1, s2, -s2, -s1])
    return perm.astype(np.intp), r


def _stag_block_eig(Amat, Bmat, qq, parity, *, tol=None):
    """All ``4 q^2`` eigenpairs of the OUT-OF-PLANE staggered PENCIL from ONE
    ``2 q^2`` eig, or ``None`` when the structural precondition fails (-> the
    dense Cholesky-whitened ``4 q^2`` solve, bit-for-bit).

    THE STRUCTURE.  ``R`` (:func:`_stag_parity_gauge`) is a real signed
    permutation with ``R^2 = I`` that ANTI-commutes with the generator and
    COMMUTES with the block Gram::

        R A R = -A        R B R = +B

    so in the orthogonal eigenbasis ``U = [U+ | U-]`` of ``R`` (each sector
    exactly ``2 q^2``-dimensional: ``tr R = 0``, because on a square grid the
    two ``E`` blocks and the two ``G`` blocks contribute equal and opposite
    parity traces) the pencil is block-ANTI-diagonal against a block-DIAGONAL
    Gram::

        U^T A U = [[0, X], [Y, 0]]        U^T B U = blkdiag(Bp, Bm)

    Whitening each sector by its own Cholesky (``Bp = Lp Lp^H``,
    ``Bm = Lm Lm^H``) and writing ``Xh = Lp^-1 X Lm^-H``,
    ``Yh = Lm^-1 Y Lp^-H`` reduces the pencil to ONE standard ``2 q^2`` eig::

        Xh Yh up = q^2 up ,    um = Yh up / q ,
        x = U+ Lp^-H up  +/-  U- Lm^-H um     for the +/- q pair

    -- the out-of-plane analogue of what ``eig(P Q)`` does for an in-plane
    layer, and the staggered twin of the Fourier reduction in
    :func:`~lumenairy.elements.rcwa._core._generator_block_eig`.  ``U`` is a
    real orthogonal signed pairing, so forming ``X, Y, Bp, Bm`` and expanding
    the ``4 q^2`` vectors are ``O(n^2)``; the only cubic work is at ``2 q^2``.

    VERIFY THEN USE.  The condition is on the ASSEMBLED pencil, never on
    ``eps``: a cell whose permittivity is its own parity image but whose
    spectral-element WALLS are not mirror-symmetric breaks it at the
    discretisation level.  Both residuals are measured here, row-blocked so no
    second ``4 q^2 x 4 q^2`` transient is allocated, and anything above
    :data:`_STAG_BLOCK_TOL` returns ``None``.  ``tol`` is read at CALL time
    (never bound as a default) so a test can walk the bar's own two-sided gap
    through the shipped code.
    """
    perm, r = parity
    tol = _STAG_BLOCK_TOL if tol is None else float(tol)
    n2, n4 = 2 * qq, 4 * qq
    if Amat.shape != (n4, n4) or Bmat.shape != (n4, n4) or perm.size != n4:
        return None
    sA = float(np.max(np.abs(Amat)))
    sB = float(np.max(np.abs(Bmat)))
    if not (np.isfinite(sA) and np.isfinite(sB)) or sA == 0.0 or sB == 0.0:
        return None
    # ---- structural test.  r is real +/-1 and constant on every orbit, so
    # (R M R)[i, j] = r_i M[perm_i, perm_j] r_j and no gauge division arises.
    for i0 in range(0, n4, 256):
        i1 = min(i0 + 256, n4)
        rr = r[i0:i1, None] * r[None, :]
        pi = perm[i0:i1]
        if float(np.max(np.abs(
                rr * Amat[np.ix_(pi, perm)] + Amat[i0:i1]))) > tol * sA:
            return None
        if float(np.max(np.abs(
                rr * Bmat[np.ix_(pi, perm)] - Bmat[i0:i1]))) > tol * sB:
            return None

    # ---- the R eigenbasis as (index, index, coeff, coeff) columns
    plus, minus = [], []
    seen = np.zeros(n4, dtype=bool)
    inv2 = 1.0 / np.sqrt(2.0)
    for i in range(n4):
        if seen[i]:
            continue
        j = int(perm[i])
        seen[i] = True
        if j == i:                                   # self-paired dof
            (plus if r[i] > 0 else minus).append((i, i, 1.0, 0.0))
            continue
        seen[j] = True
        # R e_i = r_i e_j, so R(e_i +/- e_j) = r_i (e_j +/- e_i)
        if r[i] > 0:
            plus.append((i, j, inv2, inv2))
            minus.append((i, j, inv2, -inv2))
        else:
            plus.append((i, j, inv2, -inv2))
            minus.append((i, j, inv2, inv2))
    if len(plus) != n2 or len(minus) != n2:
        return None

    def _desc(cols):
        return (np.array([c[0] for c in cols], dtype=np.intp),
                np.array([c[1] for c in cols], dtype=np.intp),
                np.array([c[2] for c in cols], dtype=_C),
                np.array([c[3] for c in cols], dtype=_C))

    dp, dm = _desc(plus), _desc(minus)

    def _cols(desc, Mm):                              # Mm @ U
        i, j, ci, cj = desc
        return Mm[:, i] * ci[None, :] + Mm[:, j] * cj[None, :]

    def _rows(desc, Mm):                              # U^T @ Mm  (U is REAL)
        i, j, ci, cj = desc
        return ci[:, None] * Mm[i, :] + cj[:, None] * Mm[j, :]

    def _expand(desc, Cc):                            # U @ Cc
        # i and j are all-distinct and disjoint apart from the self-paired
        # dofs, where i == j and cj == 0 -- so assign-then-add is exact and
        # avoids np.add.at's unbuffered slow path.
        i, j, ci, cj = desc
        out = np.zeros((n4, Cc.shape[1]), dtype=_C)
        out[i] = ci[:, None] * Cc
        out[j] += cj[:, None] * Cc
        return out

    Xb = _rows(dp, _cols(dm, Amat))                   # U+^T A U-
    Yb = _rows(dm, _cols(dp, Amat))                   # U-^T A U+
    Bp = _rows(dp, _cols(dp, Bmat))                   # U+^T B U+  (HPD)
    Bm = _rows(dm, _cols(dm, Bmat))                   # U-^T B U-  (HPD)
    try:
        Lp = np.linalg.cholesky(Bp)
        Lm = np.linalg.cholesky(Bm)
    except np.linalg.LinAlgError:                     # not PD -> dense path
        return None
    Xh = sla.solve_triangular(Lp, Xb, lower=True)
    Xh = sla.solve_triangular(Lm, Xh.conj().T, lower=True).conj().T
    Yh = sla.solve_triangular(Lm, Yb, lower=True)
    Yh = sla.solve_triangular(Lp, Yh.conj().T, lower=True).conj().T
    mu, up = np.linalg.eig(Xh @ Yh)
    qv = np.sqrt(np.asarray(mu, dtype=_C))
    gmax = float(np.max(np.abs(qv)))
    if not np.isfinite(gmax) or gmax == 0.0:
        return None
    if float(np.min(np.abs(qv))) <= _STAG_GAM_FLOOR * gmax:
        return None                                   # null mode: 1/q
    um = (Yh @ up) / qv[None, :]
    # np.linalg.eig returns unit-norm ``up``, and ``[up; +/- um]`` is the
    # whitened vector in the (U, Cholesky) factorization of B -- unitarily
    # equivalent to the dense path's, so normalising it here reproduces that
    # path's scaling convention, which _select_forward_flux's RELATIVE noise
    # ceilings read.
    nrm = np.sqrt(1.0 + np.sum(np.abs(um) ** 2, axis=0))
    Cp = sla.solve_triangular(Lp.conj().T, up, lower=False)
    Cm = sla.solve_triangular(Lm.conj().T, um, lower=False)
    Xp = _expand(dp, Cp)
    Xm = _expand(dm, Cm)
    Xfull = np.concatenate([Xp + Xm, Xp - Xm], axis=1)
    Xfull = Xfull / np.concatenate([nrm, nrm])[None, :]
    if not np.all(np.isfinite(Xfull)):
        return None
    return np.concatenate([qv, -qv]), Xfull


def _region_modes_oop(solver: Granet2DTransverseE, *, symmetry=False):
    """Forward AND backward modes of an OUT-OF-PLANE region, as the 6-tuple
    ``(Wf, Vf, lam_f, Wb, Vb, lam_b)`` -- the shape
    ``rcwa._core._layer_eigenmodes_tensor`` returns on its generator branch and
    the shape the GENERALIZED S-matrix cascade
    (:func:`~lumenairy.elements.rcwa._core._interface_smatrix_general`) consumes.

    ``W`` holds ``[E1; E2]`` coefficients and ``V`` the tangential-H partner in
    the SAME coefficient bases the in-plane :func:`_region_modes` uses (``H1``
    in ``V2``, ``H2`` in ``V1``), so a MIXED cascade -- isotropic half-spaces,
    in-plane layers and out-of-plane layers in one stack -- is a square modal
    match at every interface, with no re-projection anywhere.

    Three things are load-bearing and each is measured, not assumed:

    1. **The eig.**  ``B = blkdiag(Ggram1, Ggram2, Ggram2, Ggram1)`` is
       Hermitian positive definite (it is a block Gram), so the pencil is
       Cholesky-WHITENED to a standard eig rather than handed to a QZ.  That is
       why the ``4 q^2`` out-of-plane path costs only 1.7-2.4x the ``2 q^2``
       in-plane path, which pays ``scipy.linalg.eig(L, G)``.

    2. **The forward/backward split MUST be flux-based with the deep-decay
       override.**  The out-of-plane generator breaks the in-plane
       ``[W; -V] <-> -lam`` symmetry, so the two half-spectra are genuinely
       distinct and a bare ``Re(gam)`` split will not do: the off-branch modes
       (unresolved harmonics of a polynomial basis) carry ~1e-14 RELATIVE flux
       of RANDOM SIGN, and one growing mode classified forward blows the
       cascade up by ``exp(+|Re gam| k0 L)``.  Those modes are deeply
       evanescent (``Re(lam) >= 5.75`` measured on the prototype, ten times
       :func:`~lumenairy.elements.rcwa._core._select_forward_flux`'s
       ``|Re gam| > 0.5`` bar), which is exactly what that override classifies.
       The selector is fed the CHOLESKY-WHITENED blocks ``[L1 e1; L2 e2;
       L2 g1; L1 g2]`` with ``Ggram1 = L1^H L1``, ``Ggram2 = L2^H L2``, so its
       plain harmonic sums reproduce the Gram-weighted modal flux
       ``Sz = Im(e1^H Ggram1 g2 - e2^H Ggram2 g1)`` EXACTLY -- the PMM_ROADMAP
       C-FLUX rule.  The ``2 q^2 / 2 q^2`` count is pinned by a ``RuntimeError`` that the
       selector's unconditional rebalance makes unreachable (see the note
       at the end of this docstring).

    3. **The H gauge.**  The generator state carries ``G = i Z0 H`` while the
       Eq.-25 partner :func:`_region_modes` builds for the in-plane regions and
       the half-spaces is ``[H1; H2]`` itself, so the two differ by ONE global
       constant (:data:`_OOP_H_GAUGE`).  It does not cancel in ANY stack --
       the isotropic half-spaces always enter in the Eq.-25 gauge -- so it is
       applied here and gated twice: a uniform out-of-plane slab against
       ``berreman_jones_1d`` (dJones 1.5e-14 vs 3.9e-02 for the conjugate
       gauge; see the constant's note) and the in-plane-reduction test, which
       drives this path on a cell whose cross terms are exactly zero and
       compares against the Stage-A path.

    On the ``2 q^2 / 2 q^2`` check below: :func:`_select_forward_flux`
    rebalances to EXACTLY ``2N`` unconditionally (its defensive tail ranks
    every mode by a signed forwardness score), so a misclassified mode is
    silently REBALANCED, never raised.  The ``RuntimeError`` is therefore a
    contract pin on that selector's behaviour, unreachable while it holds
    (verified 2026-09-09: 16 stressors -- a lossy metal in an out-of-plane
    host, an on-cutoff walk, high contrast at M=8, 60-degree incidence -- all
    split ``2 q^2 / 2 q^2`` BEFORE the rebalance, ``min Re(lam_f) >= -1.5e-14``).

    ``symmetry`` opts into the PARITY-sign block reduction
    (:func:`_stag_block_eig`) -- ONE ``2 q^2`` eig instead of the ``4 q^2``
    one, measured 1.5-1.9x on the whole out-of-plane solve.  It is a pure
    accelerator: the structure is verified on the ASSEMBLED pencil every call
    and any failure (oblique incidence, a SLANTED cell, an off-centre or
    unmirrored cell, a tensor whose component grid is not its own parity image)
    falls back to the dense branch below, which is then executed BIT-FOR-BIT as
    if the keyword had never been passed.
    """
    if not solver.offplane:
        raise ValueError(
            "_region_modes_oop: this solver assembled the IN-PLANE (second-"
            "order) pencil -- call _region_modes.  The dispatch is "
            "Granet2DTransverseE.offplane.")
    Amat, Bmat = solver.Agen, solver.Bgen
    qq = solver.q * solver.q
    fac = None
    if symmetry:
        gauge = _stag_parity_gauge(solver)
        if gauge is not None:
            fac = _stag_block_eig(Amat, Bmat, qq, gauge)
    if fac is None:
        Lc = np.linalg.cholesky(Bmat)             # Bmat HPD (block Gram)
        Ah = sla.solve_triangular(Lc, Amat, lower=True)
        Ah = sla.solve_triangular(Lc, Ah.conj().T, lower=True).conj().T
        qv, Y = np.linalg.eig(Ah)
        X = sla.solve_triangular(Lc.conj().T, Y, lower=False)
    else:
        qv, X = fac
    W = X[:2 * qq, :]                              # [E1; E2]
    Gst = X[2 * qq:, :]                            # [G1; G2] = i Z0 [H1; H2]
    # flux split on the whitened blocks (see 2. above)
    L1 = np.linalg.cholesky(Bmat[:qq, :qq]).conj().T
    L2 = np.linalg.cholesky(Bmat[qq:2 * qq, qq:2 * qq]).conj().T
    Vfull = np.concatenate([L1 @ W[:qq], L2 @ W[qq:],
                            L2 @ Gst[:qq], L1 @ Gst[qq:]], axis=0)
    nrm = np.linalg.norm(Vfull, axis=0)
    Vfull = Vfull / np.where(nrm == 0.0, 1.0, nrm)[None, :]
    lam = -1j * qv                                 # forward exp(-lam k0 z)
    fidx = np.asarray(_select_forward_flux(lam, Vfull, qq))
    bidx = np.array(sorted(set(range(qv.size)) - set(fidx.tolist())), dtype=int)
    if fidx.size != 2 * qq or bidx.size != 2 * qq:
        raise RuntimeError(
            f"_region_modes_oop: the flux selector split the {4 * qq} "
            f"out-of-plane modes {fidx.size}/{bidx.size} instead of "
            f"{2 * qq}/{2 * qq}.  A cascade built on an unbalanced set is "
            f"rank-deficient; raise n_modes (M) or move off the Rayleigh "
            f"cutoff (a |gamma| -> 0 mode is what defeats the split).")
    V = _OOP_H_GAUGE * Gst
    return (W[:, fidx], V[:, fidx], lam[fidx],
            W[:, bidx], V[:, bidx], lam[bidx])


def _modes_as_general(W, V, lam):
    """Express a SYMMETRIC (in-plane / isotropic) region's modes in the
    generalized 6-tuple form ``(W, V, lam, W, -V, -lam)``.

    The in-plane pencil has the exact symmetry ``[W; -V] <-> -lam``, so the
    backward set is the forward one with ``V`` and ``lam`` negated; writing it
    out is what lets a stack that contains ONE out-of-plane layer run the
    generalized cascade throughout.  Measured cost: the generalized interface
    reproduces the shipped square :func:`_interface_smatrix` to 7.7e-14 (S11) /
    8.1e-14 (S21) on an isotropic pair at normal incidence and 1.0e-13 /
    9.3e-14 at conical, so a mixed stack pays nothing in accuracy for it."""
    return W, V, lam, W, -V, -lam


def _homog_geom_cache(solver: Granet2DTransverseE):
    """Pre-solve the eps-FREE geometric eig shared by EVERY homogeneous region.

    For a uniform-eps region the modal operator splits EXACTLY into an eps-scaled
    field-Gram plus an eps-free geometric part::

        L(eps) = eps * G + L0_geom ,   G = -Rmat = blockdiag(G1, G2)

    because (i) the component masses are ``Et_jj = eps * G_jj`` and (ii) the
    div(D)=0 Schur term is eps-INVARIANT -- its ``Meps33 = eps*G3`` and
    ``Kzt = eps*Kzt0`` cancel, so ``Schur = Ktz @ solve(G3, Kzt0)`` carries no eps
    (verified eps-independent to ~5e-29).  Hence ``L0_geom = Stt - Schur`` is purely
    geometric, and ONE generalized eig ``L0_geom W0 = g2_geo G W0`` serves all
    half-spaces: each region's modes are the SAME eigenvectors ``W0`` with the
    scalar-shifted spectrum ``g2 = g2_geo + eps`` (see :func:`_homog_region_modes`).
    This replaces the two homogeneous-region eigs (superstrate + substrate) with a
    single shared one -- 3 region eigs -> 2, the dominant cost (the eig is ~97% of a
    region solve).  ``solver`` must be a HOMOGENEOUS assembly (uniform ``eps_cell``)
    so its ``Stt``/``Schur``/``Rmat`` are the geometric operators.

    SCALAR ONLY.  The eps-free split needs ``Meps33 = eps*G3`` and
    ``Kzt = eps*Kzt0`` to cancel; for a TENSOR cell ``Kzt`` mixes e11/e21 (and
    e22/e12) while ``Meps33`` carries e33 alone, so ``Schur`` is NOT tensor-free
    and a uniform anisotropic region takes its own :func:`_region_modes` eig
    (that is what :class:`~lumenairy.elements.pmm.PMM2DStackPure` does with a
    ``(3, 3)`` uniform layer).  A tensor assembly here RAISES rather than
    silently returning a wrong geometric basis.
    """
    if solver.eps_cell.ndim == 4 or solver.offplane or solver.magnetic:
        raise ValueError(
            "_homog_geom_cache: the shared eps-free geometric eig is defined "
            "for a uniform SCALAR region only, and never for a MAGNETIC "
            "region -- a uniform TENSOR region's div(D)=0 Schur term is not "
            "eps-free (K_zt mixes e11/e21 while Meps33 carries e33), and a "
            "MAGNETIC region's L0_geom is not geometric at all (chi_t and "
            "chi33 weight R, K_tz and S_tt), so either needs its own "
            "_region_modes eig.")
    G = -solver.Rmat                       # block field Gram (Hermitian PD)
    Stt = solver.Stt
    L0_geom = Stt - solver.Schur           # = Lmat - eps*G, manifestly eps-free
    g2_geo, W0 = sla.eig(L0_geom, G)
    Ginv = np.linalg.inv(G)
    qq = solver.q * solver.q
    # Pre-fold the eps-free pieces of the H-partner recovery (Eq.25): for a
    # homogeneous region Lhh = Et + Stt = eps*G + Stt, so Lhh @ W0 = eps*(G W0) +
    # (Stt W0) -- both terms eps-free and reusable across regions.
    return W0, g2_geo, G @ W0, Stt @ W0, Ginv, qq


def _homog_region_modes(geom, eps):
    """Modes of a homogeneous region (uniform permittivity ``eps``) from the shared
    eps-free geometric eig -- NO per-region eig.

    Mirrors :func:`_region_modes` EXACTLY (same forward-branch selection and Eq.25
    H-partner), but the eigenvectors are the cached ``W0`` and the spectrum is the
    scalar shift ``g2 = g2_geo + eps`` (the field-Gram term contributes ``+eps`` to
    every eigenvalue; the eigenvectors are unchanged).  The H-partner uses the
    pre-folded ``Lhh @ W0 = eps*(G W0) + (Stt W0)`` -- a cheap scalar combination, no
    matmul against a fresh operator.  Returns ``(W, V, lam)`` (g2 is not needed
    downstream for the half-spaces)."""
    W0, g2_geo, GW0, SttW0, Ginv, qq = geom
    g2 = g2_geo + eps
    q = np.sqrt(np.asarray(g2, dtype=_C))
    q = _forward_branch_flip(q)            # shared scalar-vertical selector (S1-8)
    lam = -1j * q                          # forward propagator exp(-lam k0 z) decays
    # H recovery (Eq.25) with Lhh = eps*G + Stt (homogeneous -> no Schur term):
    Dual = Ginv @ (eps * GW0 + SttW0)      # G^{-1} Lhh W0  (back to coefficients)
    top = Dual[:qq, :]
    bot = Dual[qq:, :]
    rot = np.concatenate([-bot, top], axis=0)     # (-C) Dual
    V = rot * _inv_lam(q)[None, :]
    return W0, V, lam


# =========================================================================== #
# THE FULL CANONICAL 2-D PMM SOLVE.
# =========================================================================== #
def pmm_efficiency_2d_staggered(
    period_x: float,
    period_y: float,
    eps_cell,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    degree: int = 8,
    n_modes: int | None = None,
    n_orders: int = 7,
    polarization: str = "te",
    theta: float = 0.0,
    phi: float = 0.0,
    slant=None,
) -> Efficiency2D:
    """Rigorous diffraction efficiencies of a 2-D crossed grating of axis-aligned
    rectangular pillars by the canonical no-floor Polynomial Modal Method (Granet
    2023 staggered modified-Legendre basis).

    The NO-FLOOR 2-D counterpart of :func:`pmm_efficiency_1d` and of the
    FMM-floored hybrid :func:`~lumenairy.elements.pmm.twod.pmm_efficiency_2d`: the
    energy balance is ``n_orders``-INDEPENDENT (no Fourier floor) and tracks only
    the modal ``degree``.  See the module docstring for the method and its scope.

    Parameters
    ----------
    period_x, period_y : float
        Unit-cell periods along x and y (metres).
    eps_cell : (Nx, Ny) array_like of complex
        The layer permittivity as a constant value PER SEGMENT of the
        ``Nx`` x ``Ny`` rectangular grid.  The pillar walls are the segment
        boundaries, so ``eps`` is exact per element (Eq. 26) -- the grid must
        resolve the rectangular pillar exactly (e.g. a centred half-fill pillar
        is ``Nx = Ny = 2``).  The grid MUST be SQUARE (``Nx == Ny``; the
        staggered tensor-product basis requires ``Nx*(M-1) == Ny*(M-1)``) --
        tile a uniform axis into equal segments, or use
        :func:`pmm_efficiency_2d_cell`, which handles non-square cells.
        PUBLIC convention ``Im(eps) > 0`` for loss.
    n_substrate, n_superstrate : complex
        Transmission / incidence half-space indices (``n = n + i kappa``).
    depth : float
        Grating layer thickness (metres).
    wavelength : float
        Vacuum wavelength (metres).
    degree : int, optional
        Number ``M`` of modified-Legendre functions per segment per axis -- the
        modal convergence knob (raise for accuracy; the per-component DOF is
        ``(Nx*(M-1)) * (Ny*(M-1))``).  Default 8.  NOTE (naming divergence): unlike
        the GLL POLYNOMIAL ``degree`` of :func:`pmm_efficiency_2d` and the 1-D
        ``pmm_*`` solvers, this ``degree`` is a BASIS-FUNCTION COUNT ``M`` -- so a
        cross-solver ``degree`` sweep compares unlike quantities.  Prefer the clearer
        alias ``n_modes`` (below); ``degree`` is kept for backward compatibility.
    n_modes : int, optional
        Clearer alias for ``degree`` (the modified-Legendre mode count ``M``).  If
        given, it overrides ``degree``.  Use this in new code.
    n_orders : int, optional
        Half-width of the retained Rayleigh diffraction-order set for the
        once-only forward far-field projection (the result is independent of
        this as long as it covers the propagating orders).  Default 7.
    polarization : {'te', 'tm'}, optional
        Incident polarization.  Default ``'te'``.
    theta, phi : float, optional
        Incidence polar / azimuthal angles (radians).  Default normal incidence.
    slant : (t_x, t_y) or float, optional
        Accepted only as ``None`` / ``0`` (vertical).  A nonzero slant RAISES:
        a sheared scalar cell is an OUT-OF-PLANE tensor cell in the frame
        (``eps^{13} = -t_x eps``), so the two incident polarizations mix and a
        SINGLE-polarization efficiency is not what this entry would be
        returning -- the same reason it already refuses a ``(Nx, Ny, 3, 3)``
        cell.  Use :func:`pmm_jones_2d_staggered`, which drives both
        polarizations.  The keyword exists to raise rather than to be silently
        dropped by a branch that never reads it.

    Returns
    -------
    result : :class:`~lumenairy.elements.rcwa.Efficiency2D`
        A ``tuple`` subclass that unpacks as ``orders, R, T`` (NOT a 4-tuple) and
        carries the modal problem size on ``.dof``:

        * ``orders`` -- ``(Nfo, 2)`` retained ``(m, n)`` diffraction-order pairs;
        * ``R, T`` -- ``(Nfo,)`` reflected / transmitted efficiency per order
          (real power fractions; evanescent orders 0; lossless ``sum(R)+sum(T)
          == 1``);
        * ``result.dof`` -- per-region modal problem size
          (``2 * Nx*(M-1) * Ny*(M-1)``).

        .. note:: **API change (v5.11 -> v5.12).**  Formerly a bare 4-tuple
           ``(orders, R, T, dof)``; now the cross-suite :class:`Efficiency2D`
           that unpacks to ``orders, R, T`` with ``dof`` as an attribute.  See
           :func:`pmm_efficiency_2d`.

    Notes
    -----
    Axis-aligned rectangular pillars only (walls on the ``eps_cell`` grid);
    corner-capped (algebraic, no-floor) convergence -- at-best RCWA parity per
    DOF on vertical pillars, the win being accuracy quality (no floor, exact
    sidewalls, position invariance).  NumPy/SciPy dense generalized eig.
    """
    if not _slant_is_zero(_norm_slant_pair(
            slant, "pmm_efficiency_2d_staggered")):
        raise NotImplementedError(
            f"pmm_efficiency_2d_staggered: slant={slant!r} -- a SLANTED cell "
            f"is an OUT-OF-PLANE tensor cell in the sheared frame "
            f"(eps^13 = -t_x eps), so the two incident polarizations MIX and "
            f"this entry's single-polarization efficiencies are not "
            f"well-posed -- exactly the reason it already refuses a "
            f"(Nx, Ny, 3, 3) cell.  Use pmm_jones_2d_staggered(..., "
            f"slant=...), which drives BOTH polarizations and returns "
            f"(orders, R, T, jones).")
    pol = polarization.lower()
    if pol not in ("te", "tm"):
        raise ValueError(
            f"pmm_efficiency_2d_staggered: polarization must be 'te' or 'tm', "
            f"got {polarization!r}.")
    if n_modes is not None:
        # 'n_modes' is the clearer alias for the modified-Legendre mode count M --
        # 'degree' here is NOT a GLL polynomial degree (see the docstring); 'degree'
        # stays the default for backward compatibility.
        degree = int(n_modes)
    if int(degree) < 3:
        # M=2 passes the old >=2 guard but yields |Btilde|=N, |B|=2N and trips the
        # Basis1D cardinality assert deep in the build (audit P2); require M>=3.
        raise ValueError("pmm_efficiency_2d_staggered: degree (modified-Legendre "
                         "count M) must be >= 3.")
    eps_cell = np.asarray(eps_cell, dtype=_C)
    if eps_cell.ndim == 4:
        # ANISOTROPIC entry point separation (mirrors the scalar
        # pmm_efficiency_2d_cell vs tensor pmm_jones_2d split): this entry is
        # scalar-only and returns SINGLE-polarization efficiencies, which a
        # tensor cell does not admit (the two incident polarizations mix).
        raise ValueError(
            f"pmm_efficiency_2d_staggered: eps_cell must be a 2-D (Nx, Ny) "
            f"SCALAR array; got a (Nx, Ny, 3, 3) tensor cell of shape "
            f"{eps_cell.shape}.  Use pmm_jones_2d_staggered (the anisotropic "
            f"no-floor entry: it drives BOTH incident polarizations and "
            f"returns (orders, R, T, jones)).")
    if eps_cell.ndim != 2:
        raise ValueError(
            f"pmm_efficiency_2d_staggered: eps_cell must be a 2-D (Nx, Ny) "
            f"array, got shape {eps_cell.shape}.")
    # SQUARE-grid restriction (audit P3-36): the staggered tensor-product
    # basis requires Nx*(M-1) == Ny*(M-1), i.e. Nx == Ny -- previously enforced
    # only by a deep assert (stripped under ``python -O`` into a cryptic
    # kron/broadcast shape error).  Fail loudly at the entry point instead.
    if eps_cell.shape[0] != eps_cell.shape[1]:
        raise ValueError(
            f"pmm_efficiency_2d_staggered: eps_cell must be SQUARE (Nx == Ny; "
            f"the staggered tensor-product basis requires "
            f"Nx*(M-1) == Ny*(M-1)), got shape {eps_cell.shape}.  Pad the "
            f"uniform axis into equal segments (e.g. tile a (2, 1) cell to "
            f"(2, 2)), or use pmm_efficiency_2d_cell, which handles "
            f"non-square cells.")
    polarization = pol
    Nx, Ny = eps_cell.shape
    M = int(degree)
    eps_sup = _C(n_superstrate) ** 2
    eps_sub = _C(n_substrate) ** 2
    # Wood-anomaly guard (v5.14 robustness audit P1): the staggered solver's
    # H-partner ``V ~ 1/gamma`` amplifies basis-truncation error for a
    # JUST-PROPAGATING order, and the lossless total DIVERGES like
    # ~1/sqrt(cutoff distance) as the wavelength approaches any Rayleigh
    # cutoff FROM BELOW (measured tot = 1.2 at wl = P*(1 - 1e-6), 7.5 at
    # -1e-9, 205 at -1e-12).  Nudge off an EXACT coincidence, and WARN loudly
    # inside the divergence band (the hybrid pmm_efficiency_2d_cell is clean
    # there -- use it near cutoffs).
    from ..rcwa._core import (
        _grazing_safe_wavelength,
        _require_propagating_incidence,
    )
    nre0 = float(np.real(np.sqrt(eps_sup)))
    _mo = np.arange(-int(n_orders), int(n_orders) + 1)
    _mx = np.tile(_mo, len(_mo))
    _my = np.repeat(_mo, len(_mo))
    _kx0n = nre0 * np.sin(theta) * np.cos(phi)
    _ky0n = nre0 * np.sin(theta) * np.sin(phi)
    # Incidence guard (mirrors pmm_efficiency_2d / rcwa_efficiency_2d, v5.14.1
    # suite-wide fix): reject a gain superstrate (kz_inc forward root flips
    # negative -> every T silently negated) and an evanescent incident wave
    # (kz_inc ~ 0 divides the flux normalization).  This module keeps eps in
    # the PUBLIC exp(-iwt) convention (Im eps > 0 for loss), so conjugate into
    # the guard's INTERNAL convention.
    _require_propagating_incidence("pmm_efficiency_2d_staggered",
                                   np.conj(eps_sup),
                                   _kx0n ** 2 + _ky0n ** 2)
    # The nudge list carries the LAYER's permittivities as well as the two
    # half-spaces' (unified with the tensor path 2026-09-10; before that this
    # scalar entry listed only the half-spaces, so a scalar cell and its
    # ``e * I`` promotion through pmm_jones_2d_staggered took DIFFERENT nudges
    # -- and therefore different answers, by a measured 4.59e-08 -- when an
    # order sat exactly on a layer's own cut-off).  A cut-off INSIDE the layer
    # degrades this solver just as a half-space one does, so listing it is the
    # more robust convention, and it moves nothing off an EXACT coincidence:
    # the guard's trigger band is |eps - kt^2| <= 1e-9, i.e. a relative
    # wavelength window of ~1.2e-10 around the cut-off.
    wl = _grazing_safe_wavelength(float(wavelength), _kx0n, _ky0n, _mx, _my,
                                  period_x, period_y,
                                  _wood_eps_reals(eps_sup, eps_sub, eps_cell))
    _kt2 = ((_kx0n + _mx * (wl / period_x)) ** 2
            + (_ky0n + _my * (wl / period_y)) ** 2)
    _gap = min(float(np.min(np.abs(float(np.real(e)) - _kt2)))
               for e in (eps_sup, eps_sub))
    if _gap < 1e-4:
        warnings.warn(
            f"pmm_efficiency_2d_staggered: a diffraction order is within "
            f"{_gap:.2g} (kt^2 units) of a Rayleigh cutoff; the staggered "
            f"solver's accuracy DEGRADES like ~1/sqrt(distance) near cutoffs "
            f"(energy errors of several % to >100% measured inside 1e-6).  "
            f"Use pmm_efficiency_2d_cell (clean there) or detune the "
            f"wavelength.", stacklevel=2)
    k0 = 2.0 * np.pi / wl
    nre = float(np.real(np.sqrt(eps_sup)))
    alpha0x = nre * np.sin(theta) * np.cos(phi) * k0
    alpha0y = nre * np.sin(theta) * np.sin(phi) * k0
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)

    # ---- region eigensolvers (all on the SAME grid/basis) ----
    sol_l = Granet2DTransverseE(period_x, period_y, Nx, Ny, M, eps_cell,
                                alpha0x=alpha0x, alpha0y=alpha0y, k0=k0)
    Wl, Vl, lam_l, _g2l = _region_modes(sol_l)

    # The two HALF-SPACES are HOMOGENEOUS (uniform eps), where L(eps) = eps*G +
    # L0_geom with G, L0_geom eps-FREE.  So ONE eps-free generalized eig of the
    # geometric operator serves BOTH -- each region's modes are the SAME eigenvectors
    # with a scalar-shifted spectrum g2 = g2_geo + eps (no per-region eig).  This
    # turns the 3 region eigs into 2 (the eig is ~97% of a region solve) at machine-
    # identical R/T.  We assemble ONE homogeneous solver to source the geometric
    # G/Stt/Schur, then reconstruct both half-spaces.  See _homog_geom_cache.
    sol_h = Granet2DTransverseE(period_x, period_y, Nx, Ny, M,
                                np.full((Nx, Ny), _C(eps_sup)),
                                alpha0x=alpha0x, alpha0y=alpha0y, k0=k0)
    geom = _homog_geom_cache(sol_h)
    # The geom tuple carries everything the half-spaces need; release the
    # homogeneous solver's large operator attributes before the S-matrix /
    # far-field stages (audit P3-37 -- keeping BOTH solver instances alive
    # through the far field retained several GB at convergence-grade M).
    del sol_h
    Wsup, Vsup, _ls = _homog_region_modes(geom, eps_sup)
    Wsub, Vsub, _lb = _homog_region_modes(geom, eps_sub)

    # ---- SQUARE Redheffer recursion (every W is 2 q^2) ----
    S = _interface_smatrix(Wsup, Vsup, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam_l, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wsub, Vsub))
    S11, _S12, S21, _S22 = S

    # ---- FORWARD-only far-field Fourier->Rayleigh projection (once) ----
    ox = np.arange(-n_orders, n_orders + 1)
    oy = np.arange(-n_orders, n_orders + 1)
    order_x = np.tile(ox, len(oy))
    order_y = np.repeat(oy, len(ox))
    Nfo = len(order_x)
    P1, P2 = _far_projector_2d(sol_l.bx, sol_l.by, ox, oy,
                               alpha0x, alpha0y)             # (Nfo, q^2) each
    qq = sol_l.q * sol_l.q
    # H_E[m]-projector on a [E1;E2] modal matrix -> [Ex_orders; Ey_orders]:
    #   note: E1 = E_x component, E2 = E_y component (Granet transverse comps).
    Hsup = _pmm2d_project_orders(P1, P2, Wsup, qq)     # (2 Nfo, 2 q^2)
    Hsub = _pmm2d_project_orders(P1, P2, Wsub, qq)

    kxv = kx0 + order_x * (wl / period_x)
    kyv = ky0 + order_y * (wl / period_y)

    # incident (0,0) plane wave -> superstrate mode amplitudes by FORWARD overlap
    kt = float(np.hypot(kx0, ky0))
    if kt < 1e-12:
        ex0, ey0 = (0.0, 1.0) if polarization == "te" else (1.0, 0.0)
        einc_sq = 1.0
    else:
        axu, ayu = kx0 / kt, ky0 / kt
        if polarization == "te":
            ex0, ey0 = -ayu, axu
            einc_sq = 1.0
        else:
            ex0, ey0 = axu, ayu
            kz_inc0 = float(np.real(_kz_forward2(eps_sup, kx0, ky0)))
            einc_sq = 1.0 + (kt / kz_inc0) ** 2
    delta = ((order_x == 0) & (order_y == 0)).astype(_C)
    rhs = np.concatenate([ex0 * delta, ey0 * delta])         # (2 Nfo,)
    cinc = _guarded_lstsq(                                   # (2 q^2,)
        Hsup, rhs, "pmm 2-D staggered far-field Rayleigh projection")

    r_ord = Hsup @ (S11 @ cinc)                              # (2 Nfo,)
    t_ord = Hsub @ (S21 @ cinc)
    rx, ry = r_ord[:Nfo], r_ord[Nfo:]
    tx, ty = t_ord[:Nfo], t_ord[Nfo:]

    kz_ref, kz_trn, kz_inc, safe_r, safe_t = _pmm2d_order_kz(
        eps_sup, eps_sub, kxv, kyv, kx0, ky0)
    rz = -(kxv * rx + kyv * ry) / safe_r
    tz = -(kxv * tx + kyv * ty) / safe_t
    R, T = _project_efficiency(np, kz_ref, kz_trn, kz_inc,
                               rx, ry, rz, tx, ty, tz, einc_sq)
    orders2d = np.stack([order_x, order_y], axis=1)
    # cross-suite return shape: unpacks as (orders, R, T); .dof = 2*q^2 (the modal
    # eigenproblem dimension).  Was a bare 4-tuple (orders, R, T, dof) pre-v5.12.
    return Efficiency2D(orders2d, R, T, 2 * qq)


# =========================================================================== #
# ANISOTROPIC (block-form tensor) Jones entry -- the no-floor mirror of
# pmm_jones_2d.
# =========================================================================== #
def pmm_jones_2d_staggered(
    period_x: float,
    period_y: float,
    eps_cell,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    mu_cell=None,
    degree: int = 8,
    n_modes: int | None = None,
    n_orders: int = 7,
    theta: float = 0.0,
    phi: float = 0.0,
    mu_superstrate=None,
    mu_substrate=None,
    symmetry="auto",
    slant=None,
):
    """Rigorous 2-D crossed grating with a FULL ``(3, 3)`` ANISOTROPIC cell --
    in-plane OR out-of-plane -- by the canonical NO-FLOOR staggered PMM: the
    anisotropic entry of :func:`pmm_efficiency_2d_staggered` and the no-floor
    mirror of the FMM-floored hybrid
    :func:`~lumenairy.elements.pmm.pmm_jones_2d`.

    Both incident polarizations are driven, so the return is the
    :func:`~lumenairy.elements.pmm.pmm_jones_2d` /
    :meth:`~lumenairy.elements.pmm.PMM2DStackPure.solve` /
    :func:`~lumenairy.elements.rcwa.rcwa_jones_2d` shape.

    Parameters
    ----------
    period_x, period_y : float
        Unit-cell periods (metres).
    eps_cell : (Nx, Ny, 3, 3) or (Nx, Ny) array_like of complex
        Per-segment permittivity over one unit cell (PUBLIC convention
        ``Im(eps) > 0`` for loss).  A ``(Nx, Ny)`` SCALAR map is promoted to
        ``e * I`` per cell.  A Granet BLOCK-FORM tensor (Eq. 7,
        ``[[e11, e12, 0], [e21, e22, 0], [0, 0, e33]]``) runs the ``2 q^2``
        second-order pencil; OUT-OF-PLANE coupling
        (``e_xz``/``e_yz``/``e_zx``/``e_zy`` above a RELATIVE ``1e-12`` floor
        -- a tilted-director liquid crystal) routes to the ``4 q^2``
        first-order staggered generator and the generalized cascade, at
        1.3-2.0x the region-solve time.  ``e33 != 0`` is required either way
        (both eliminations divide by it).  The grid MUST be SQUARE
        (``Nx == Ny``) and the walls are the segment boundaries (exact ``eps``
        per element, Eq. 26).
    n_substrate, n_superstrate : complex
        Half-space refractive indices.  The half-spaces are ISOTROPIC and
        NONMAGNETIC (the Rayleigh match is scalar and the flux normalisation
        assumes the vacuum wave impedance); an anisotropic or magnetic
        half-space is out of scope (``mu_superstrate`` / ``mu_substrate``
        exist only to RAISE on one).
    depth, wavelength : float
        Layer thickness / vacuum wavelength (metres).
    mu_cell : (Nx, Ny, 3, 3) or (Nx, Ny) array_like of complex, optional
        Per-segment RELATIVE PERMEABILITY over the same unit cell (default
        ``None`` = nonmagnetic, ``mu = 1``, which leaves every operator and
        every result BIT-IDENTICAL to the nonmagnetic path).  A ``(Nx, Ny)``
        scalar map is the isotropic magnetic case; a ``(Nx, Ny, 3, 3)`` map
        must be BLOCK-FORM (Granet Eq. 6, ``[[m11, m12, 0], [m21, m22, 0],
        [0, 0, m33]]``) with ``m33 != 0`` and an invertible ``[mu_t]``.  The
        paper's ``chi_t = [mu_t]^-1`` then weights ``R = C[chi_t]C`` (Eq. 24),
        ``K_tz = C[chi_t][d2; -d1]`` (Eq. 21) and -- through ``chi33`` --
        ``S_tt`` (Eq. 20), on the SAME ``2 q^2`` second-order pencil.
        OUT-OF-PLANE ``mu``, and ``mu`` together with an out-of-plane
        ``eps_cell``, raise ``NotImplementedError``.
    degree : int, optional
        Modified-Legendre function count ``M`` per segment per axis (the modal
        convergence knob).  Default 8.  ``n_modes`` is the clearer alias.
    n_modes : int, optional
        Alias for ``degree``; overrides it when given.
    n_orders : int, optional
        Half-width of the retained Rayleigh order set for the once-only forward
        far-field projection.  The result is n_orders-INDEPENDENT (no Fourier
        floor) as long as it covers the propagating orders.  Default 7.
    theta, phi : float, optional
        Conical incidence polar / azimuth angles (radians).
    symmetry : {'auto', True, False}, optional
        Opt into the PARITY-sign block reduction of the OUT-OF-PLANE region
        solve (:func:`_stag_block_eig`): one ``2 q^2`` eig instead of the
        ``4 q^2`` one, measured 1.5-1.9x on the whole out-of-plane solve.  It
        engages ONLY at NORMAL incidence on an out-of-plane cell whose
        ASSEMBLED pencil carries the structure (the cell is its own parity
        image on a mirror-symmetric wall layout); every other case -- oblique
        or conical incidence, an off-centre or unmirrored cell, a
        parity-breaking tensor, and every in-plane or scalar cell -- runs the
        dense path BIT-FOR-BIT, which is what ``symmetry=False`` forces
        everywhere.  A SLANTED cell is refused outright (a shear breaks the
        mirror symmetry), so ``symmetry='auto'`` on a slanted cell is
        bit-identical to ``symmetry=False``.  Default ``'auto'`` (equivalent to
        ``True``).
    slant : (t_x, t_y) or float, optional
        Constant x-z / y-z SHEAR of the layer: the whole cross-section
        translates laterally by ``(t_x, t_y) * depth`` from the layer's TOP
        face to its bottom, and ``eps_cell`` is the cross-section at the TOP.
        ``t`` is a TANGENT (``t_x = tan(wall_tilt_x)``) -- the SAME public
        convention as :meth:`~lumenairy.elements.pmm.PMM2DStackHybrid.add_layer`
        and the 1-D ``slant_angle`` entries, so a cell moves between the
        engines unchanged.  ``None`` / ``0`` (the default) is a vertical layer
        and stays BIT-IDENTICAL to the pre-slant library.  Exact at any slant
        magnitude and ONE eigensolve for the whole layer (``det J = 1``), but a
        slanted cell always takes the ``4 q^2`` first-order generator and the
        generalized cascade -- its covariant tensor has out-of-plane entries
        even when the cell is scalar.  Refused together with ``mu_cell``.  A
        shear is NOT a taper: a shrinking cross-section still needs a
        z-staircase.

    Returns
    -------
    orders : (Nfo, 2) int ndarray
        Retained ``(m, n)`` diffraction-order pairs.
    R_eff, T_eff : (2, Nfo) float ndarray
        Reflected / transmitted efficiency per order; row 0 = incident ``E_x``,
        row 1 = incident ``E_y``.
    jones_reflection : (2, 2) complex ndarray
        Zeroth-order REFLECTION Jones in the lab ``(x, y)`` basis, PUBLIC
        ``exp(-i w t)``, columns = response to incident ``E_x`` / ``E_y``.
        Unlike :func:`~lumenairy.elements.pmm.pmm_jones_2d` (which solves in an
        internal conjugated gauge and conjugates back), this cascade is PUBLIC
        end to end -- there is NO conjugation bridge anywhere in this module.

    Notes
    -----
    Formulation: Granet, J. Opt. Soc. Am. A 40, 652 (2023), Eqs. 23-25 with the
    general block-form ``[eps_t]`` (Appendix A Eqs. 40, 41, 44).  In-plane
    anisotropy keeps the SAME ``2q^2`` second-order eigenproblem, the same
    ``[W; -V] <-> -lam`` symmetry and the same square Redheffer cascade as the
    isotropic solver; an OUT-OF-PLANE cell keeps the same basis, de Rham
    placement, far field and union grid but takes the ``4 q^2`` first-order
    generator (:meth:`Granet2DTransverseE._assemble_oop`) and the GENERALIZED
    S-matrix cascade, because ``div D = 0`` no longer slaves ``E3``.  Either
    way the whole no-floor architecture is inherited -- and so are the corner
    cap, the near-cutoff caveat and the union-grid rule.  The paper uses
    ``exp(+i w t)``; a tensor quoted from it must be CONJUGATED before it is
    passed here.

    A GYROTROPIC cell (``e12 = -e21 = i b``, Hermitian) is lossless and fully
    supported; its ``+/-`` order asymmetry is the observable that a sign error
    in ``e12``/``e21`` would break (energy checks cannot see it).  Likewise a
    NON-RECIPROCAL out-of-plane tensor (``e13 = conj(e31)``, Hermitian, still
    lossless): the dispersion relation is invariant under ``eps -> eps^T``, so
    only the FIELDS -- not any energy or eigenvalue check -- can see an
    ``e13``/``e31`` swap.
    """
    from .stack2d_pure import PMM2DStackPure  # (cycle-free: lazy)
    # Validate HERE so the message names this entry, then hand the checked cell
    # to the single-layer pure cascade (one implementation of the physics).
    cell = _validate_stag_cell("pmm_jones_2d_staggered", eps_cell)
    _require_nonmagnetic_halfspace("pmm_jones_2d_staggered", mu_superstrate,
                                   mu_substrate)
    mu = None
    if mu_cell is not None:
        mu = _validate_stag_mu("pmm_jones_2d_staggered", mu_cell)
        if mu.shape[:2] != cell.shape[:2]:
            raise ValueError(
                f"pmm_jones_2d_staggered: mu_cell grid {mu.shape[:2]} must "
                f"match the eps_cell grid {cell.shape[:2]}.")
        if cell.ndim == 4 and _tile_needs_oop("pmm_jones_2d_staggered", cell):
            raise NotImplementedError(
                "pmm_jones_2d_staggered: mu_cell together with an "
                "OUT-OF-PLANE eps_cell is not implemented -- the out-of-plane "
                "first-order generator carries no permeability blocks.")
    M = int(degree if n_modes is None else n_modes)
    if M < 3:
        raise ValueError("pmm_jones_2d_staggered: degree / n_modes (the "
                         "modified-Legendre count M) must be >= 3.")
    stack = PMM2DStackPure(period_x, period_y, n_superstrate=n_superstrate,
                           n_substrate=n_substrate, n_modes=M,
                           n_orders=int(n_orders), symmetry=symmetry)
    if mu is None:
        stack.add_layer(float(depth), eps_cell=cell, slant=slant)
    else:
        stack.add_layer(float(depth), eps_cell=cell, mu_cell=mu, slant=slant)
    stack.set_source(float(wavelength), theta=float(theta), phi=float(phi))
    return stack.solve(jones=True)
