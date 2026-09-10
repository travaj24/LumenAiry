"""
lumenairy.elements.pmm.stack2d_pure -- multilayer 2-D PURE (no-floor) PMM.
==========================================================================

:class:`PMM2DStackPure` cascades multiple doubly-periodic layers using the
canonical NO-FLOOR staggered modified-Legendre PMM (Granet 2023; see
:mod:`lumenairy.elements.pmm.twod_staggered`) -- the no-floor sibling of the
Fourier-projected :class:`~lumenairy.elements.pmm.stack2d.PMM2DStackHybrid`, and
the 2-D analogue of the 1-D :class:`~lumenairy.elements.pmm.PMMStack`.

Method
------
Every region (both half-spaces AND every layer) is solved in the SAME staggered
modal basis at the same modal degree ``M``; each interface is therefore a SQUARE
modal match (the 1-D PMM architecture lifted to 2-D) cascaded by the Redheffer
S-matrix, and the Rayleigh-order projection is applied ONCE, FORWARD only, at the
two half-spaces.  The energy balance is consequently ``n_orders``-INDEPENDENT (no
Fourier floor) and tracks only the modal degree ``M`` -- unlike the hybrid, which
projects each layer into a truncated Fourier basis before the eigensolve and
inherits the FMM ``n_orders`` floor.

vs PMM2DStackHybrid
-------------------
* **No Fourier floor.**  Patterned-layer energy/accuracy is ``n_orders``-
  independent; raise the modal degree ``n_modes`` (``M``) to converge.
* **Exact sidewalls + position invariance** (walls land on the ``eps_cell`` grid).
* **Union grid by DEFAULT, not by necessity.**  ``layer_grids='shared'`` (the
  default) puts all patterned layers on ONE common SQUARE ``(Nx, Ny)``
  segmentation -- the 1-D :class:`PMMStack` union grid lifted to 2-D -- so the
  modal matrices are conformable across every interface.
  ``layer_grids='per-layer'`` lifts that: each layer keeps its OWN grid and its
  own modal count, and adjacent grids are coupled by an L2 MORTAR (see
  "Per-layer element grids" below).  The hybrid decouples layers through its
  Fourier projection and so never had the constraint at all.

Scope
-----
Any mix of UNIFORM and PATTERNED layers -- including DIRECT
patterned<->patterned (A|B) interfaces -- with axis-aligned rectangular
patterns, on one shared square grid, carrying either ISOTROPIC scalar
permittivity or an IN-PLANE (Granet BLOCK-FORM) permittivity TENSOR:
``add_layer(eps=(3,3))`` for a uniform anisotropic layer and
``add_layer(eps_cell=(Nx,Ny,3,3))`` for a patterned one (see
:mod:`lumenairy.elements.pmm.twod_staggered`, "Anisotropy").  OUT-OF-PLANE
tensors (``e_xz``/``e_yz``/``e_zx``/``e_zy`` -- a tilted-director liquid
crystal) are supported: such a layer takes the first-order ``4 q^2`` staggered
generator, whose forward and backward modes are distinct, and a stack
containing ANY out-of-plane layer runs the GENERALIZED S-matrix cascade
throughout (uniform and in-plane layers and the half-spaces entering as
``[[W, W], [V, -V]]``).  ``retain_internal`` / :meth:`layer_absorption` work
on that path too.  The half-spaces stay isotropic.

A layer may also be MAGNETIC: ``add_layer(..., mu=scalar | (3,3))`` or
``add_layer(..., mu_cell=(Nx,Ny) | (Nx,Ny,3,3))`` gives it a BLOCK-FORM
relative permeability, which enters the SAME second-order pencil through
Granet's ``chi_t = [mu_t]^-1`` weights (see
:mod:`lumenairy.elements.pmm.twod_staggered`, "MAGNETIC media").  Any
combination with the ``eps`` side is allowed -- a uniform eps with a patterned
mu included -- and the uniform side is broadcast onto the union grid.  A
magnetic layer takes its own region eig (it cannot ride the shared eps-free
geometric one), deduped by ``(eps bytes, mu bytes)``.  Out-of-plane mu, mu with
an out-of-plane eps, and MAGNETIC HALF-SPACES (``mu_superstrate`` /
``mu_substrate``, which exist only to raise) are out of scope.  Losslessness --
the closure tripwire's precondition -- means Hermitian eps AND Hermitian mu.
(The historical 'A|B blows up energy' defect that once limited this class to a
single patterned layer was NOT an interface/mode-sorting problem: it was the
far-field projection-kernel order MIRROR in ``_stag_fourier_projection`` --
physical order ``-m`` deposited into slot ``m``, hence the wrong per-order
``kz`` flux factor whenever higher orders propagate.  With the kernel fixed,
the direct A|B Redheffer cascade conserves energy exactly and matches the
exact 1-D multilayer :class:`PMMStack` per-order at oblique incidence; see
``tests/unit/test_v5_21_pmm2d_staggered_oblique.py``.  Li's prescription that
the real-eigenvalue up/down split is non-critical for internal layers --
Li 2003 J. Opt. A 5:345; *Gratings: Theory and Numeric Applications* ch. 13,
2014, 13.2.3.3 -- is what makes the 2nd-order ``(W, +/-V, +/-lam)`` cascade
sound as-is.)
A layer may also be SLANTED: ``add_layer(..., slant=(t_x, t_y))`` makes it ONE
EXACT slanted region rather than a z-staircase -- the whole cross-section
translates by ``t * thickness`` from the layer's TOP face to its bottom, ``t``
being a TANGENT in the same PUBLIC convention as
:class:`PMM2DStackHybrid` and the 1-D ``slant_angle`` entries.  It is exact at
any slant magnitude and costs ONE eigensolve (``det J = 1``: the sheared
frame's metric is z-invariant), and it promotes the whole stack to the
generalized cascade, as an out-of-plane layer does -- a sheared cell IS an
out-of-plane cell in the frame.  Slant on a UNIFORM layer is accepted and is a
physical no-op.  The bookkeeping a shear adds is ONE unimodular phase per order
on the TRANSMITTED amplitudes (the frame anchor); R and the reflection Jones
need nothing.  Out of scope, all raising: MIXED slants between PATTERNED
layers, a mix of vertical and slanted layers ABOVE a pattern, ``mu`` with a
slant, and ``retain_internal`` on a slanted stack.  See
:mod:`lumenairy.elements.pmm.twod_staggered`, "SLANT".
Two ACCURACY notes on that scope, measured 2026-09-10
(``docs/audits/VERIFY_PMM2D_STAGGERED_SLANT_2026_09_10.md`` D3/D4): the
uniform-null residual is spectral but not free -- the ``< 1e-04`` bar at
``n_modes = 5`` is a statement about slants ``<= 35 deg`` (a 60-degree slant
reads ``1.471e-04`` at that rung and falls to ``7.0e-11`` by ``n_modes = 8``),
so steep tilt costs ``M``; and a slanted UNIFORM layer is MATERIALISED as a
constant cell on the union grid and takes its OWN ``4 q^2`` solve, while a
vertical one rides the shared eps-free geometric eig -- two different
discretizations of the same homogeneous medium, so slanting a uniform spacer
costs about TWO DECADES of accuracy at fixed ``M`` (``3.4e-04`` against the
single-layer null's ``4.1e-06`` at ``M = 5``) and converges only algebraically.
Leave a spacer vertical unless the shear is physical.

A shear is NOT a TAPER -- a taper shrinks the cross-section and no shear
absorbs a dilation, so a tapered feature still needs a z-staircase.  That
staircase is now available on THIS engine too:
:meth:`PMM2DStackPure.add_tapered_pillar` / :meth:`add_tapered_pillars` with
``layer_grids='per-layer'`` put each slice on its own NON-UNIFORM grid at
exact walls (see below).

Per-layer element grids (``layer_grids='per-layer'``)
-----------------------------------------------------
Every layer carries its OWN segmentation and its own ``n_modes``, and adjacent
grids are coupled WEAKLY: tangential E is tested against the lower layer's
trace space and tangential H against the upper layer's (the classic
mode-matching pairing, which keeps the interface system square for unequal
mode counts), with the resulting rectangular S-matrices cascaded by
``_redheffer_star_rect``.  Two layers whose grids COINCIDE bypass the mortar
entirely and reproduce ``layer_grids='shared'`` BIT-EXACTLY.

* ``add_layer(..., x_walls=, y_walls=)`` gives a PATTERNED layer arbitrary wall
  POSITIONS (``eps_cell`` is then the strip TILE, exactly the hybrid's
  geometry description); its segment COUNT always comes from that cell and is
  never a free parameter, because a pillar 1/2 of the period wide exists on
  ``N in {2,4,...}`` and one 1/3 wide on ``N in {3,6,...}`` -- accepting a free
  ``N`` would silently change the DEVICE.
* ``add_layer(..., grid=)`` applies to UNIFORM layers, which have no walls of
  their own; it defaults to ``1``, the cheapest region the engine can express.
* ``add_layer(..., n_modes=)`` is the per-layer modal count and is the lever
  that makes the mode pay.  A uniform layer's default is NOT the stack's ``M``
  but the measured neighbour rule ``M_u = max(q_prev, q_next) / N_u + 1``:
  ``N = 1`` is cheap, ``N = 1`` at the stack's ``M`` is a trap.
* The far-field order capacity is set by the END grids (the half-spaces ride
  them), ``n_orders <= (q - 1) // 2`` with ``q = N (M - 1)``; asking for more
  RAISES rather than silently retaining aliased order slots.
* ``window_halfwidth`` RAISES.  On a segment partition the only partition
  carrying two layers' walls is their common refinement -- the union grid
  itself -- so there is no local enrichment to widen.  Per-layer grids here are
  own-walls-only, and own-walls-only works.

**A per-layer solve can be STATIONARY IN ONE KNOB AND WRONG.**  Measured on a
two-layer pillar pair: walking one layer's ``n_modes`` across four rungs gave
an answer stationary to 4 % and wrong by 27 %, because the OTHER layer was the
limiting error the whole time -- and it was not even monotone.  The stopping
criterion is therefore stationarity in EVERY ``n_modes``, screened first by
:meth:`convergence_floor`, whose per-layer own-residual is a measured LOWER
BOUND on the stack's error (15 of 16 surface points) and costs one region eig
per layer instead of the stack's.  The union grid's single ``M`` is
*convenient* precisely because it cannot be mis-set per layer.
Uniform SCALAR layers route through the shared eps-free geometric eig
(:func:`~lumenairy.elements.pmm.twod_staggered._homog_region_modes`) -- all
uniform regions share the SAME eigenvectors, so a uniform<->uniform interface is
``a = W0^-1 W0 = I`` (perfectly conditioned) and costs no eig.  A uniform
TENSOR layer cannot: its div(D)=0 Schur term is not eps-free, so it takes its
own region eig (deduped by bytes, like a patterned cell).

Caveats (inherited from :func:`pmm_efficiency_2d_staggered`)
-----------------------------------------------------------
* Corner-capped ALGEBRAIC convergence on right-angle dielectric pillars (raise
  ``M``); a smooth-field region converges spectrally.
* A UNIFORM layer at OBLIQUE incidence is degree-limited (its transverse Bloch
  phase ``exp(i k_t.r)`` must be resolved in the modified-Legendre basis) -- for
  planar-dominant oblique stacks prefer the hybrid or Berreman.
* ``~1/sqrt(distance)`` accuracy loss within a Rayleigh cutoff (detune the
  wavelength; the hybrid is clean there).

Conventions match the suite: PUBLIC ``exp(-i w t)`` (``Im eps > 0`` for loss),
forward ``exp(+i kz z)``, ``Im(kz) >= 0``.  ``solve()`` drives BOTH incident
polarizations and returns ``(orders, R(2, N), T(2, N), jones(2, 2))`` -- the
:class:`PMM2DStackHybrid` / :func:`rcwa_jones_2d` shape.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..rcwa._core import (  # shared flux projection + the generalized cascade
    _interface_smatrix_general,
    _modes_to_M,
    _norm_slant_pair,
    _project_efficiency,
    _propagation_smatrix_general,
    _slant_is_zero,
    _symmetry_on,
)
from ._core import (
    PerOrderAmplitudesMixin,
    _guarded_lstsq,
    _interface_smatrix,
    _interface_smatrix_general_mortar_2d,
    _interface_smatrix_mortar_2d,
    _propagation_smatrix,
    _redheffer_star,
    _redheffer_star_rect,
)
from .twod_staggered import (
    _C,
    Granet2DTransverseE,
    StagCrossOps,
    StagGridOps,
    _far_projector_2d,
    _homog_geom_cache,
    _homog_region_modes,
    _modes_as_general,
    _pmm2d_order_kz,
    _pmm2d_project_orders,
    _region_modes,
    _region_modes_oop,
    _require_inplane_mu,
    _require_nonmagnetic_halfspace,
    _stag_kron_apply,
    _stag_mortared_axes,
    _tile_needs_oop,
    _validate_stag_cell,
    _validate_stag_mu,
    _warn_stag_sliver_band,
    _wood_eps_reals,
)

__all__ = ["PMM2DStackPure"]

#: Lossless-closure tripwire window for the pure staggered cascade.  DERIVED
#: 2026-09-09 by MEASUREMENT over 40 lossless SCALAR configurations of this
#: engine -- every fixture the shipped staggered suites exercise plus a
#: deliberately adverse set (build doc
#: docs/audits/BUILD_PMM2D_STAGGERED_ANISOTROPIC_2026_09_09.md, table T11;
#: probe validation/probe_pmm2d_staggered_aniso/p11_tol_derivation.py).
#: The WORST |sum R + sum T - 1| measured there is 9.01e-03 -- a UNIFORM layer
#: at theta = 0.25 and M = 6, i.e. exactly the degree-limited oblique-uniform
#: regime this module's docstring already warns about.  Every shipped-suite
#: fixture is at or below 1.63e-03.  This 5e-2 window therefore sits 5.5x
#: above the worst adverse configuration and ~31x above every shipped one,
#: while an under-resolved tensor solve at the minimum modal count M=3 leaves
#: it by 1.8x (9.1e-02 measured) -- a deterministic discretization number
#: whose cross-build spread is ~1e-6, so the decision is not near a knife
#: edge even though the ratio is modest.  It is the same value as the hybrid's
#: ``twod._PASSIVE_TOL_2D`` (5e-2), arrived at independently, which keeps the
#: two 2-D engines' closure contracts comparable.
_STAG_CLOSURE_TOL = 5.0e-2


def _as_layer_cell(spec, uniform, Nx, Ny):
    """A layer's ``eps`` / ``mu`` specification as a ``(Nx, Ny[, 3, 3])`` CELL
    on the stack's union grid.

    ``uniform`` distinguishes a ``(3, 3)`` UNIFORM tensor from a ``(3, 3)``
    scalar GRID -- the two are shape-identical, so the flag (set by
    :meth:`PMM2DStackPure.add_layer` from which keyword the caller used) is
    what disambiguates them.  A uniform spec is broadcast to a constant cell,
    which is what a magnetic or tensor region needs anyway: it takes its own
    region eig rather than riding the shared eps-free geometric one."""
    spec = np.asarray(spec, dtype=_C)
    if not uniform:
        return spec
    if spec.ndim == 0:
        return np.full((Nx, Ny), spec, dtype=_C)
    return np.ascontiguousarray(np.broadcast_to(spec, (Nx, Ny, 3, 3)))


def _stag_walls_spec(period, walls, n_default, axis, fn):
    """A layer's per-axis segmentation as either an ``int`` (the UNIFORM
    lattice) or a full ``(N + 1,)`` boundary array on ``[0, period]``.

    ``walls=None`` returns the INT ``n_default`` -- and returning the int
    rather than ``np.linspace`` is load-bearing: ``Basis1D`` keeps a distinct
    integer path because ``linspace`` computes ``start + i*step`` and pins its
    last element, so ``linspace[i+1] - linspace[i]`` is not always the same
    double as ``d/N`` (measured 1.96e-16 relative at ``d = 0.9, N = 4``).  The
    int is what makes "``x_walls=None`` is bit-identical to today" true for
    every period.

    A given ``walls`` is accepted in EITHER of the two spellings the library
    already uses: the hybrid's INTERIOR wall list (``add_tapered_pillar``'s
    ``xw``, which excludes ``0`` and ``period``) or a full boundary array."""
    if walls is None:
        return int(n_default)
    w = np.asarray(walls, dtype=float).ravel()
    if w.size == 0:
        raise ValueError(
            f"{fn}: {axis} must name at least one wall (or be None for the "
            f"uniform lattice).")
    if np.any(np.diff(w) <= 0.0):
        raise ValueError(
            f"{fn}: {axis} must be STRICTLY increasing, got {w!r}.")
    tol = 1e-12 * period
    if abs(w[0]) <= tol and abs(w[-1] - period) <= tol:
        full = w.copy()
        full[0] = 0.0
        full[-1] = period
    else:
        if not (0.0 < w[0] and w[-1] < period):
            raise ValueError(
                f"{fn}: {axis} interior walls must satisfy "
                f"0 < w < period = {period!r}, got {w!r}.  (A FULL boundary "
                f"array starting at 0 and ending at the period is also "
                f"accepted.)")
        full = np.concatenate([[0.0], w, [period]])
    if full.size < 2:
        raise ValueError(f"{fn}: {axis} yields fewer than one segment.")
    return full


def _stag_walls_n(spec):
    """Segment count of a :func:`_stag_walls_spec` result."""
    return int(spec) if np.ndim(spec) == 0 else int(np.size(spec) - 1)


def _stag_interior(spec):
    """The INTERIOR walls of a :func:`_stag_walls_spec` result, or ``None`` on
    the uniform (integer) path -- the form :meth:`add_layer` takes back."""
    if np.ndim(spec) == 0:
        return None
    return np.asarray(spec, dtype=float)[1:-1].copy()


def _spec_is_lossless(spec, uniform):
    """True when a layer's ``eps`` / ``mu`` specification absorbs nothing:
    exactly real if scalar, Hermitian if a ``(3, 3)`` tensor (per cell)."""
    spec = np.asarray(spec, dtype=_C)
    tensor = spec.ndim == 4 or (uniform and spec.ndim == 2)
    if tensor:
        return _tensor_is_hermitian(spec)
    return not bool(np.any(np.imag(spec) != 0.0))


def _principal_diag(spec, uniform):
    """A layer's ``eps`` / ``mu`` specification as its PRINCIPAL DIAGONAL with
    the component on the LAST axis -- ``(3,)`` for a uniform spec, ``(Nx, Ny,
    3)`` for a patterned one -- so that an eps side and a mu side of any two
    kinds broadcast against each other componentwise.

    A scalar is repeated over the three components (it IS isotropic), and a
    tensor's OFF-diagonals are dropped: only the principal indices set a
    Rayleigh cut-off.  Used only by the Wood-anomaly nudge list."""
    a = np.asarray(spec, dtype=_C)
    if uniform:
        if a.ndim == 0:
            return np.full(3, a, dtype=_C)
        return np.ascontiguousarray(np.diag(a))
    if a.ndim == 4:
        return a[..., [0, 1, 2], [0, 1, 2]]
    return np.repeat(a[..., None], 3, axis=-1)


def _wood_cutoff_products(layer):
    """The permittivities a Rayleigh cut-off can sit on INSIDE a MAGNETIC
    layer: the per-cell, per-component PRODUCT ``eps_ii * mu_ii``.

    A layer mode goes grazing where its longitudinal wavenumber vanishes, i.e.
    at ``kt^2 = Re(eps mu)`` -- the layer index is ``n = sqrt(eps mu)``, not
    ``sqrt(eps)``.  Listing the permittivity alone (as this branch did until
    2026-09-10) therefore misses a magnetic layer's own cut-offs entirely; a
    ``mu = 1`` layer is UNAFFECTED, because multiplying by exactly ``1.0`` is
    exact in float64 and the product is the permittivity bit for bit.

    The componentwise product is a HEURISTIC for an anisotropic pair (the true
    cut-offs of a biaxial magnetic layer are the roots of its own dispersion
    relation, not products of principal values); the guard is a warning /
    nudge heuristic, and over-listing is numerically inert off an EXACT
    coincidence -- ``_grazing_safe_wavelength`` takes a MIN over the list and
    only fires inside a ``|eps - kt^2| <= 1e-9`` band."""
    return (_principal_diag(layer["eps"], layer["eps_uniform"])
            * _principal_diag(layer["mu"], layer["mu_uniform"]))


def _tensor_is_hermitian(t):
    """True if every ``(3, 3)`` tensor in ``t`` is Hermitian, i.e. LOSSLESS.

    RELATIVE floor (``1e-12 * scale``), the same shape as
    :func:`~lumenairy.elements.pmm.twod_jones._tile_is_offplane` and for the
    same reason: a physically lossless tensor built by ROTATING a real diagonal
    one (``R @ diag @ R.T``) computes its ``(0,1)`` and ``(1,0)`` entries by
    different dot products, so exact ``t == t^H`` is not float-attainable,
    while a genuine anti-Hermitian (absorbing) part is O(kappa * n) -- decades
    above roundoff."""
    t = np.asarray(t, dtype=_C)
    dev = float(np.max(np.abs(t - np.conj(np.swapaxes(t, -1, -2)))))
    scale = max(float(np.max(np.abs(t))), 1.0)
    return dev <= 1e-12 * scale


def _stack_is_lossless(layers, eps_sup, eps_sub):
    """True when the whole pure-staggered stack is PROVABLY lossless: real
    half-spaces (exactly, as they come from real indices) and every layer
    either exactly-real scalar or Hermitian tensor.  Mirrors the predicate of
    :func:`~lumenairy.elements.pmm.twod._warn_lossless_energy_2d`, extended to
    the tensor case (a Hermitian permittivity absorbs nothing)."""
    for e in (eps_sup, eps_sub):
        if np.imag(np.asarray(e, dtype=_C)) != 0.0:
            return False
    for L in layers:
        if L["kind"] == "uniform":
            if np.imag(L["eps"]) != 0.0:
                return False
        elif L["kind"] == "uniform_tensor":
            if not _tensor_is_hermitian(L["eps33"]):
                return False
        elif L["kind"] == "magnetic":
            # A HERMITIAN permeability absorbs nothing either: losslessness of
            # a magnetic medium is "eps Hermitian AND mu Hermitian" (the
            # Poynting theorem's dissipation term carries both anti-Hermitian
            # parts).  A real scalar mu != 1 is the isotropic case of that.
            if not (_spec_is_lossless(L["eps"], L["eps_uniform"])
                    and _spec_is_lossless(L["mu"], L["mu_uniform"])):
                return False
        elif L["eps_cell"].ndim == 4:
            if not _tensor_is_hermitian(L["eps_cell"]):
                return False
        elif np.any(np.imag(L["eps_cell"]) != 0.0):
            return False
    return True


def _layer_is_patterned(L):
    """``True`` when the layer carries a PATTERN (a cell that is not constant
    across the union grid), which is what makes a lateral frame offset
    observable.  A ``uniform`` / ``uniform_tensor`` layer is homogeneous: a
    translation maps it to itself, so its frame offset is a pure gauge."""
    if L["kind"] in ("uniform", "uniform_tensor"):
        return False
    if L["kind"] == "magnetic":
        return not (L["eps_uniform"] and L["mu_uniform"])
    return True


def _check_stack_slant(layers, fn):
    """Refuse the slanted stacks whose FRAME ANCHOR is ambiguous.

    Each slanted region is solved in its OWN frame, anchored at that layer's
    TOP face (``u = x - t (z - z_top)``); the interface match between regions
    is the IDENTITY (``det g = 1``, ``w = z``, and the covariant TANGENTIAL
    components equal the lab-Cartesian ones), so what a stack actually
    cascades is each layer's cell displaced laterally by the ACCUMULATED offset
    of everything above it, ``Sh_i = sum_{j<i} t_j d_j``.  Two readings of that
    are exact and each is measured:

    * ``Sh_i = 0`` -- every layer above the pattern is vertical, so the cell
      sits where the caller put it (the workhorse: one slanted patterned layer
      among uniform films);
    * ``Sh_i = t_i Z_i`` with every layer above at the SAME slant -- one global
      shear of the whole stack, the frame simply continuing (the layer-split
      identity: one slanted layer of depth ``d`` equals two of ``d/2``, measured
      at 1.3e-15).

    Anything else -- PATTERNED layers at different slants (a VERTICAL patterned
    layer included), or a mix of vertical and slanted layers above a pattern --
    puts one nodal grid at a real lateral translation relative to another, which
    a nodal SEM basis cannot represent exactly unless the offset is a whole
    number of grid cells.  Refuse it loudly rather than return the shifted
    structure silently.
    """
    if all(_slant_is_zero(L.get("slant")) for L in layers):
        return
    pats = [i for i, L in enumerate(layers) if _layer_is_patterned(L)]
    if pats:
        sl0 = layers[pats[0]].get("slant", (0.0, 0.0))
        bad = [i for i in pats if layers[i].get("slant", (0.0, 0.0)) != sl0]
        if bad:
            raise NotImplementedError(
                f"{fn}: MIXED SLANTS between PATTERNED layers are not "
                f"supported -- layer {pats[0]} is slanted {sl0} while layer "
                f"{bad[0]} is slanted "
                f"{layers[bad[0]].get('slant', (0.0, 0.0))} (a VERTICAL "
                f"patterned layer counts as slant (0.0, 0.0)).  The frame "
                f"offset between two differently sheared PATTERNED regions is "
                f"a real lateral translation of one nodal grid relative to the "
                f"other, exact only when it is a whole number of grid cells.  "
                f"Give every patterned layer ONE slant, or z-staircase the "
                f"odd one out, or use PMM2DStackHybrid (its Fourier "
                f"projection decouples the layers).")
        for i in pats:
            above = [layers[j].get("slant", (0.0, 0.0)) for j in range(i)]
            if not above:
                continue
            ti = layers[i].get("slant", (0.0, 0.0))
            if all(_slant_is_zero(t) for t in above):
                continue                      # Sh_i = 0: the cell is in place
            if all(t == ti for t in above):
                continue                      # one global shear of the stack
            raise NotImplementedError(
                f"{fn}: the layers ABOVE patterned layer {i} carry a MIX of "
                f"vertical and slanted regions ({above}), so the accumulated "
                f"lateral frame offset above that pattern is neither zero nor "
                f"the one global shear {ti} -- the pattern would be silently "
                f"displaced by sum_j t_j d_j.  Slant the whole stack "
                f"uniformly, keep every layer above the pattern vertical, or "
                f"use PMM2DStackHybrid.")


def _warn_stag_closure(R_eff, T_eff, layers, eps_sup, eps_sub):
    """Lossless-closure tripwire for the PURE staggered cascade -- the no-floor
    sibling of :func:`~lumenairy.elements.pmm.twod._warn_lossless_energy_2d`
    and of ``PMMStack._warn_stack_energy``.

    When the stack is provably lossless (Hermitian tensors INCLUDED --
    gyrotropic ``e12 = -e21 = i b`` is lossless), ``sum R + sum T = 1`` is
    EXACT, so the gate is TWO-SIDED about 1.0: a deficit is as much a defect as
    an excess (the 2026-08-17 lesson from the hybrid).  A NON-Hermitian tensor
    is silent -- ``R + T < 1`` is then physical and no unity claim exists to
    violate.  WARNS, never raises: a working solve stays byte-unchanged."""
    if not _stack_is_lossless(layers, eps_sup, eps_sub):
        return
    for row, (Rr, Tr) in enumerate(zip(np.atleast_2d(R_eff),
                                       np.atleast_2d(T_eff))):
        tot = float(np.real(np.sum(Rr)) + np.real(np.sum(Tr)))
        if abs(tot - 1.0) > _STAG_CLOSURE_TOL:
            warnings.warn(
                f"PMM2DStackPure.solve: lossless energy closure violated for "
                f"incident E_{'xy'[row]} (sum R+T = {tot:.4g}, off by "
                f"{tot - 1.0:+.3g}; every permittivity in this stack is real "
                f"or Hermitian, so R+T = 1 is exact).  The staggered basis is "
                f"under-resolved or the solve sits inside a Rayleigh cutoff: "
                f"raise n_modes (M), or detune the wavelength / use "
                f"PMM2DStackHybrid near a cutoff.", stacklevel=3)


class PMM2DStackPure(PerOrderAmplitudesMixin):
    """Builder for a multilayer doubly-periodic stack solved by the PURE
    (no-floor) staggered 2-D PMM.  See the module docstring for the method,
    the union-grid constraint, and the scope/caveats.

    Parameters
    ----------
    period_x, period_y : float
        Lattice periods (metres); ``period_y`` defaults to ``period_x``.
    n_superstrate, n_substrate : complex, optional
        Half-space refractive indices (isotropic).
    n_modes : int, optional
        Modified-Legendre function count ``M`` per segment per axis -- the modal
        convergence knob (raise for accuracy).  Default 8.  ``degree`` is an
        accepted alias.  NB this is a BASIS-FUNCTION COUNT, not a GLL polynomial
        degree (see :func:`pmm_efficiency_2d_staggered`).
    n_orders : int, optional
        Half-width of the retained Rayleigh order set for the once-only forward
        far-field projection.  The result is independent of this (no floor) as
        long as it covers the propagating orders.  Default 7.
    layer_grids : {'shared', 'per-layer'}, optional
        ``'shared'`` (default) solves ONE union grid at ONE modal count and is
        the pre-2026-09-11 path unchanged.  ``'per-layer'`` gives every layer
        its own element grid and its own ``n_modes``, coupled by an L2 mortar;
        see "Per-layer element grids" in the module docstring, and note that
        it makes ``n_modes`` a PER-LAYER knob whose convergence must be checked
        per layer (:meth:`convergence_floor`).  A per-layer stack whose grids
        happen to coincide is BIT-EXACT against ``'shared'``.
    window_halfwidth : optional
        Accepted only to RAISE.  The 1-D per-layer surface enriches each
        layer's grid with its neighbours' walls; in this basis that means their
        common refinement, i.e. the union grid, so there is nothing to widen.
    symmetry : {'auto', True, False}, optional
        Opt into the PARITY-sign block reduction of the OUT-OF-PLANE region
        solve (``lumenairy.elements.pmm.twod_staggered._stag_block_eig``): one
        ``2 q^2`` eig instead of the ``4 q^2`` one, measured 1.5-1.9x on the
        whole out-of-plane solve.  It applies ONLY to out-of-plane layers at
        NORMAL incidence whose ASSEMBLED pencil carries the structure (a cell
        that is its own parity image on a mirror-symmetric wall layout); every
        other case -- oblique incidence, an off-centre or unmirrored cell, a
        parity-breaking tensor, and every in-plane or scalar layer -- runs the
        dense path BIT-FOR-BIT, which is what ``symmetry=False`` forces
        everywhere.  Default ``'auto'`` (equivalent to ``True``).
    """

    def __init__(self, period_x, period_y=None, *, n_superstrate=1.0,
                 n_substrate=1.0, n_modes=8, degree=None, n_orders=7,
                 mu_superstrate=None, mu_substrate=None, symmetry="auto",
                 layer_grids="shared", window_halfwidth=None):
        # The half-spaces are NONMAGNETIC (mu = 1) and isotropic: the Rayleigh
        # far field normalises with the vacuum wave impedance.  Accepting the
        # keyword and RAISING is the loud form of that restriction (a silently
        # ignored mu would return efficiencies normalised for the wrong
        # medium).
        _require_nonmagnetic_halfspace("PMM2DStackPure", mu_superstrate,
                                       mu_substrate)
        self.period_x = float(period_x)
        self.period_y = float(period_x if period_y is None else period_y)
        self.n_sup = complex(n_superstrate)
        self.n_sub = complex(n_substrate)
        M = int(n_modes if degree is None else degree)
        if M < 3:
            raise ValueError(
                "PMM2DStackPure: n_modes (modified-Legendre count M) must be "
                ">= 3.")
        self.M = M
        self.n_orders = int(n_orders)
        # Parity-sign block reduction for OUT-OF-PLANE region solves.  'auto'
        # and True both REQUEST it; the request is honoured only where the
        # structure is verified on the assembled pencil (normal incidence, a
        # cell that is its own parity image) and falls back to the dense
        # 4 q^2 eig bit-for-bit otherwise.  False forces the dense path.
        self.symmetry = _symmetry_on(symmetry)
        if layer_grids not in ("shared", "per-layer"):
            raise ValueError(
                "PMM2DStackPure: layer_grids must be 'shared' or 'per-layer' "
                f"(the 1-D PMMStack spelling, hyphen included), got "
                f"{layer_grids!r}.")
        self.layer_grids = layer_grids
        if window_halfwidth is not None:
            # The 1-D per-layer surface enriches each layer's grid with its
            # NEIGHBOURS' walls (a "window"), because own-walls-only was
            # MEASURED there to leave a 75-83 % degree spread.  In THIS basis a
            # window is unrepresentable: Granet's segmentation is one
            # partition of the period, so the only partition containing two
            # layers' walls is their common refinement -- which IS the union
            # grid this mode exists to avoid.  Per-layer grids here are
            # necessarily own-walls-only, and own-walls-only WORKS (the 1-D
            # failure was a nodal-SEM boundary-layer defect at arbitrary wall
            # positions and does not transfer: measured 7.6x / 7.3x MORE
            # accurate than the union grid at equal DOF on a stripe pair, 1.9x
            # / 5.5x on a corner-dominated 2-D pillar pair).  So there is
            # nothing to widen, and a silently ignored keyword would be worse
            # than a refusal.
            raise ValueError(
                "PMM2DStackPure: window_halfwidth has no meaning in the "
                "staggered basis.  A per-layer grid here is a SEGMENT "
                "PARTITION of the period, so enriching a layer's grid with "
                "its neighbours' walls means their common refinement -- the "
                "union grid itself.  Per-layer grids are own-walls-only "
                "(measured MORE accurate than the union grid at equal degrees "
                "of freedom); drop the keyword.")
        self._layers = []          # dicts: kind, thickness, eps | eps_cell
        self._grid = None          # common (Nx, Ny) set by the first patterned layer
        self._src = None
        self._modal = None         # per-order amplitudes of the last solve (B)
        self._internal = None      # partial cascades for layer_absorption (C3)

    # ------------------------------------------------------------------ build
    def add_layer(self, thickness, *, eps=None, eps_cell=None, mu=None,
                  mu_cell=None, slant=None, x_walls=None, y_walls=None,
                  grid=None, n_modes=None):
        """Append a layer.  Pass exactly ONE of ``eps`` or ``eps_cell``, and
        at most one of ``mu`` (uniform) or ``mu_cell`` (patterned).

        ``eps`` is a UNIFORM layer: a scalar (isotropic) or a ``(3, 3)``
        BLOCK-FORM permittivity tensor (in-plane anisotropic --
        ``[[e11, e12, 0], [e21, e22, 0], [0, 0, e33]]``, PUBLIC ``Im > 0`` for
        loss).  A uniform TENSOR layer is NOT eps-free-separable (its div(D)=0
        Schur term mixes e11/e21), so it costs its own region eig on the common
        grid instead of riding the shared geometric eig -- deduped by bytes,
        exactly like a patterned cell.

        ``eps_cell`` is a PATTERNED layer: a SQUARE ``(Nx, Ny)`` scalar grid, or
        a ``(Nx, Ny, 3, 3)`` block-form tensor grid (walls on the segment
        boundaries).  With ``layer_grids='shared'`` all patterned layers must
        share one common ``(Nx, Ny)`` grid (the union-grid constraint); with
        ``'per-layer'`` each keeps its own.

        ``x_walls`` / ``y_walls`` (``layer_grids='per-layer'`` only) place a
        patterned layer's walls FREELY -- the INTERIOR wall positions in metres
        (a full ``0 .. period`` boundary array is also accepted), so
        ``eps_cell`` becomes the STRIP TILE of shape
        ``(len(x_walls) + 1, len(y_walls) + 1)``: literally the hybrid's
        ``tile``, which is what lets a taper staircase move between the two
        2-D engines unchanged.  ``None`` is the uniform lattice implied by
        ``eps_cell.shape`` and is BIT-IDENTICAL to the pre-2026-09-11 library.
        The two axes may carry DIFFERENT wall positions; only the segment
        COUNTS must match.

        **MINIMUM SEGMENT WIDTH -- a CONTRACT (round 2, 2026-09-11).**  Every
        segment a non-uniform grid asks for must be at least
        :data:`~lumenairy.elements.pmm.twod_staggered._STAG_MIN_SEG_FRAC`
        (1e-3) of the period; a narrower one is REFUSED, naming the width and
        the remedies.  The staggered stiffness carries the per-segment
        ``1/J_n``, so a narrow segment carries spurious modal wavenumbers
        ``~ 0.93 M (M + 1) / (4 k0 J)``.  Those are harmless inside one grid
        and corrupt the L2 MORTAR that couples this layer to neighbours on
        other grids -- ENERGY-INVISIBLY, with the lossless closure pinned, so
        no tripwire downstream can see it.  See that constant for the
        derivation and both measured gaps.  The contract is reached at
        :meth:`solve`, not here -- ``add_layer`` only RECORDS the wall array.

        **THE BAND JUST ABOVE IT WARNS (round 3, 2026-09-11).**  A narrowest
        segment between that contract and
        :data:`~lumenairy.elements.pmm.twod_staggered._STAG_SLIVER_BAND_FRAC`
        (3e-2 of the period) is ACCEPTED and measurably less accurate -- 4.65x
        on a device that cannot depend on the wall separation at all, and it is
        a FLOOR that ``n_modes`` does not remove -- so :meth:`solve` raises a
        :class:`UserWarning` naming the width, the cost and the remedies.  It
        fires only when the stack actually builds a cross-grid MORTAR; a fully
        CONFORMING per-layer stack is silent.

        ``grid`` (``'per-layer'`` only) is a UNIFORM layer's segment count --
        uniform layers have no walls of their own -- and defaults to 1.
        ``n_modes`` (``'per-layer'`` only) overrides the stack's modal count
        for this layer; for a UNIFORM layer it defaults to the measured
        neighbour rule ``max(q_prev, q_next) / N_u + 1`` rather than to the
        stack's ``M``.

        OUT-OF-PLANE tensor coupling (``e_xz``/``e_yz``/``e_zx``/``e_zy``
        above a RELATIVE ``1e-12`` floor) routes that layer to the first-order
        ``4 q^2`` staggered generator and puts the WHOLE stack on the
        generalized cascade; ``e33 == 0`` raises (both ``E_z`` eliminations
        divide by it).

        ``mu`` / ``mu_cell`` make the layer MAGNETIC (Granet's ``chi_t =
        [mu_t]^-1`` weights): a scalar, a ``(3, 3)`` BLOCK-FORM tensor
        (``[[m11, m12, 0], [m21, m22, 0], [0, 0, m33]]``), a ``(Nx, Ny)``
        scalar grid or a ``(Nx, Ny, 3, 3)`` tensor grid.  Any combination with
        the ``eps`` side is allowed (uniform eps + patterned mu included; the
        uniform side is broadcast onto the union grid).  A magnetic layer takes
        its own region eig -- it cannot ride the shared eps-free geometric one
        -- and is deduped by ``(eps bytes, mu bytes)``.  OUT-OF-PLANE mu, and
        mu together with an out-of-plane eps, raise ``NotImplementedError``
        (the first-order generator has no permeability blocks).

        ``slant=(t_x, t_y)`` (or a bare scalar ``t_x``) makes this ONE EXACT
        SLANTED layer instead of a z-staircase: the whole cross-section
        translates linearly with depth, by ``(t_x, t_y) * thickness`` from the
        layer's TOP face to its bottom, and the cell you pass is the
        cross-section at the **top**.  ``t`` is a TANGENT (lateral walk per
        unit depth, ``t_x = tan(wall_tilt_x)``), the SAME public convention as
        :meth:`~lumenairy.elements.pmm.PMM2DStackHybrid.add_layer` and the 1-D
        ``slant_angle`` entries, so a layer moves between the engines
        unchanged.  ``0`` / ``None`` (the default) is a plain vertical layer
        and stays BIT-IDENTICAL to the pre-slant library.

        It is EXACT at any slant magnitude and costs ONE eigensolve for the
        whole layer: ``det J = 1``, so the sheared frame's metric is
        z-invariant and the shear is a pointwise congruence on the cell tensor
        plus six extra Galerkin blocks on the first-order generator
        (``docs/audits/BUILD_PMM2D_STAGGERED_SLANT_2026_09_10.md``).  A slanted
        layer therefore runs the ``4 q^2`` OUT-OF-PLANE generator -- its
        covariant tensor has out-of-plane entries even for a scalar cell -- and
        promotes the whole stack to the generalized cascade, exactly as an
        out-of-plane tensor layer does.  A slant on a UNIFORM layer is accepted
        and is a physical no-op (a shear of a homogeneous medium is a
        coordinate change).

        RESTRICTIONS, all raising: ``slant`` with ``mu`` / ``mu_cell`` (the
        first-order generator has no permeability blocks); a stack whose
        PATTERNED layers do not all share ONE slant, or in which the layers
        ABOVE a patterned layer carry a mix of that slant and vertical -- both
        raise from :meth:`solve`, where the whole stack is visible, because the
        accumulated frame offset between two differently-sheared PATTERNED
        regions is a real lateral translation of one nodal grid relative to the
        other; and ``solve(retain_internal=True)`` on a slanted stack.

        NOTE this models a slanted (tilted-axis, CONSTANT cross-section)
        feature.  It does NOT model a TAPER (shrinking cross-section) -- no
        shear absorbs a dilation.  A tapered feature still needs a
        z-staircase."""
        self._modal = None      # geometry change supersedes retained amplitudes
        self._internal = None
        if (eps is None) == (eps_cell is None):
            raise ValueError(
                "PMM2DStackPure.add_layer: pass exactly ONE of eps (uniform) or "
                "eps_cell (patterned).")
        _pl = self._perlayer_spec(eps, eps_cell, x_walls, y_walls, grid,
                                  n_modes)
        t = float(thickness)
        if not t > 0:
            raise ValueError("PMM2DStackPure.add_layer: thickness must be > 0.")
        sl = _norm_slant_pair(slant, "PMM2DStackPure.add_layer")
        if mu is not None or mu_cell is not None:
            if not _slant_is_zero(sl):
                raise NotImplementedError(
                    "PMM2DStackPure.add_layer: slant= together with mu / "
                    "mu_cell is not implemented -- a SLANTED layer runs the "
                    "OUT-OF-PLANE first-order generator, which carries no "
                    "permeability blocks (it eliminates G3 assuming mu = 1).  "
                    "The shear's own metric anisotropy is absorbed "
                    "analytically; a MATERIAL mu is not.  Drop mu, or "
                    "z-staircase the slanted magnetic layer.")
            self._add_magnetic_layer(t, eps, eps_cell, mu, mu_cell)
            return self._finish_layer(_pl)
        if eps is not None:
            e = np.asarray(eps, dtype=_C)
            if e.ndim == 0:
                self._layers.append(dict(kind="uniform", thickness=t,
                                         eps=_C(eps), slant=sl))
                return self._finish_layer(_pl)
            if e.shape != (3, 3):
                raise ValueError(
                    f"PMM2DStackPure.add_layer: a uniform eps must be a scalar "
                    f"or a (3, 3) block-form tensor, got shape {e.shape}.  A "
                    f"PATTERNED tensor layer goes through eps_cell "
                    f"((Nx, Ny, 3, 3)).")
            _tile_needs_oop("PMM2DStackPure.add_layer", e[None, None])
            self._layers.append(dict(kind="uniform_tensor", thickness=t,
                                     eps33=e, slant=sl))
            return self._finish_layer(_pl)
        cell = _validate_stag_cell("PMM2DStackPure.add_layer", eps_cell)
        cgrid = cell.shape[:2]
        if self.layer_grids == "shared":
            if self._grid is None:
                self._grid = cgrid
            elif cgrid != self._grid:
                raise ValueError(
                    f"PMM2DStackPure.add_layer: all patterned layers must share "
                    f"ONE common (Nx, Ny) grid (the union-grid constraint of the "
                    f"pure staggered cascade); got {cgrid} after {self._grid}.  "
                    f"Re-express every pattern on a common grid, pass "
                    f"layer_grids='per-layer' (each layer keeps its own grid, "
                    f"coupled by an L2 mortar), or use PMM2DStackHybrid (no "
                    f"union-grid constraint).")
        self._layers.append(dict(kind="patterned", thickness=t, eps_cell=cell,
                                 slant=sl))
        return self._finish_layer(_pl)

    def _perlayer_spec(self, eps, eps_cell, x_walls, y_walls, grid, n_modes):
        """Validate the four PER-LAYER-GRID keywords and turn them into the
        record fields ``(wx, wy, M)`` -- or ``{}`` on the shared path, where
        all four are REFUSED.

        The rules are measurements, not taste:

        * a PATTERNED layer's segment COUNT comes from its own ``eps_cell``
          (and its wall POSITIONS from ``x_walls`` / ``y_walls`` when given),
          never from ``grid=``.  Which grids are admissible is a GEOMETRY
          question: a pillar 1/2 of the period wide exists on ``N in {2,4,..}``
          and one 1/3 wide on ``N in {3,6,..}``, so accepting a free ``N`` for
          a patterned layer silently changes the DEVICE.  (The probe made
          exactly that mistake once and had to withdraw the arm.)
        * a UNIFORM layer has no walls of its own, so ``grid=`` is legal there
          and defaults to 1 -- the cheapest region the engine can express, and
          measured 1-5 DECADES better than ``N = 2`` or ``3`` per degree of
          freedom at every angle, including conical, because the field is ONE
          plane wave and degree is the right currency.
        * ``n_modes=`` is the per-layer modal count and it is the lever that
          makes this mode pay.  For a UNIFORM layer it must NOT default to the
          stack's ``M``: inside a cascade the uniform layer must also carry the
          NEIGHBOURS' traces, and at the stack default a ``grid = 1`` layer
          measured 3-6x worse than ``grid = 2/3``.  The MEASURED rule is
          ``M_u = max(q_prev, q_next) / grid + 1`` (at matched ``q_u`` the grid
          choice does not matter at all -- three columns agreeing to 3 %), and
          :meth:`solve` applies it when ``n_modes`` is not given for a uniform
          layer.  ``N = 1`` is cheap; ``N = 1`` at the stack's ``M`` is a trap.
        """
        given = {"x_walls": x_walls, "y_walls": y_walls, "grid": grid,
                 "n_modes": n_modes}
        if self.layer_grids == "shared":
            bad = sorted(k for k, v in given.items() if v is not None)
            if bad:
                raise ValueError(
                    f"PMM2DStackPure.add_layer: {', '.join(bad)} "
                    f"{'is' if len(bad) == 1 else 'are'} only meaningful with "
                    f"layer_grids='per-layer' -- the shared path solves ONE "
                    f"union grid at ONE modal count, which is exactly what "
                    f"makes it impossible to mis-set per layer.  Construct the "
                    f"stack with PMM2DStackPure(..., layer_grids='per-layer').")
            return {}
        fn = "PMM2DStackPure.add_layer"
        M = self.M if n_modes is None else int(n_modes)
        if M < 3:
            raise ValueError(
                f"{fn}: n_modes (modified-Legendre count M) must be >= 3, got "
                f"{n_modes!r}.")
        if eps_cell is not None:
            if grid is not None:
                raise ValueError(
                    f"{fn}: grid= is not accepted for a PATTERNED layer -- its "
                    f"segment count comes from its own eps_cell, which is what "
                    f"names the walls.  Asking for a 1/2-wide pillar 'on N=3' "
                    f"would silently make it 2/3 wide.  Pass the cell you "
                    f"want (and x_walls / y_walls to place its walls freely).")
            cell = np.asarray(eps_cell)
            nx, ny = cell.shape[0], cell.shape[1]
            wx = _stag_walls_spec(self.period_x, x_walls, nx, "x_walls", fn)
            wy = _stag_walls_spec(self.period_y, y_walls, ny, "y_walls", fn)
            for nm, spec, n_cell in (("x_walls", wx, nx), ("y_walls", wy, ny)):
                if _stag_walls_n(spec) != n_cell:
                    raise ValueError(
                        f"{fn}: {nm} yields {_stag_walls_n(spec)} segments but "
                        f"eps_cell has {n_cell} strips on that axis.  With "
                        f"walls given, eps_cell is the STRIP TILE -- shape "
                        f"(len(x_walls) + 1, len(y_walls) + 1) for interior "
                        f"wall lists, exactly the hybrid's `tile`.")
            return {"wx": wx, "wy": wy, "M": M, "n_modes_given": True}
        # UNIFORM (scalar / tensor / magnetic-uniform): no walls of its own.
        if grid is not None and (x_walls is not None or y_walls is not None):
            raise ValueError(
                f"{fn}: pass grid= OR x_walls/y_walls for a uniform layer, not "
                f"both (grid= IS the uniform-lattice spelling).")
        n_def = 1 if grid is None else int(grid)
        if n_def < 1:
            raise ValueError(f"{fn}: grid must be >= 1, got {grid!r}.")
        wx = _stag_walls_spec(self.period_x, x_walls, n_def, "x_walls", fn)
        wy = _stag_walls_spec(self.period_y, y_walls, n_def, "y_walls", fn)
        if _stag_walls_n(wx) != _stag_walls_n(wy):
            raise ValueError(
                f"{fn}: a layer needs EQUAL segment counts per axis "
                f"(Nx == Ny), got {_stag_walls_n(wx)} and "
                f"{_stag_walls_n(wy)}.  The wall POSITIONS may differ freely.")
        return {"wx": wx, "wy": wy, "M": M,
                "n_modes_given": n_modes is not None}

    def _finish_layer(self, pl):
        """Attach the per-layer grid record produced by :meth:`_perlayer_spec`
        to the layer just appended (a no-op on the shared path)."""
        if pl:
            self._layers[-1].update(pl)
        return self

    def _add_magnetic_layer(self, t, eps, eps_cell, mu, mu_cell):
        """The ``mu`` / ``mu_cell`` branch of :meth:`add_layer` (kept apart so
        the NONMAGNETIC code path above is untouched, byte for byte).

        Stores ONE layer record carrying both specifications plus the two
        ``uniform`` flags that disambiguate a ``(3, 3)`` UNIFORM tensor from a
        ``(3, 3)`` scalar GRID; :meth:`solve` broadcasts each onto the union
        grid and hands the pair to :class:`Granet2DTransverseE`."""
        if mu is not None and mu_cell is not None:
            raise ValueError(
                "PMM2DStackPure.add_layer: pass at most ONE of mu (uniform) "
                "or mu_cell (patterned).")
        fn = "PMM2DStackPure.add_layer"
        oop_msg = (
            "{0}: a MAGNETIC layer with an OUT-OF-PLANE eps is not "
            "implemented -- the out-of-plane first-order generator carries no "
            "permeability blocks.  Use a BLOCK-FORM eps with mu, or drop mu."
        ).format(fn)
        if eps is not None:
            e = np.asarray(eps, dtype=_C)
            if e.ndim == 0:
                eps_spec, eps_uni = _C(eps), True
            elif e.shape == (3, 3):
                if _tile_needs_oop(fn, e[None, None]):
                    raise NotImplementedError(oop_msg)
                eps_spec, eps_uni = e, True
            else:
                raise ValueError(
                    f"{fn}: a uniform eps must be a scalar or a (3, 3) "
                    f"block-form tensor, got shape {e.shape}.")
        else:
            eps_spec = _validate_stag_cell(fn, eps_cell)
            if eps_spec.ndim == 4 and _tile_needs_oop(fn, eps_spec):
                raise NotImplementedError(oop_msg)
            eps_uni = False
        if mu is not None:
            m = np.asarray(mu, dtype=_C)
            if m.ndim == 0:
                if m == 0:
                    raise ValueError(f"{fn}: a uniform scalar mu must be "
                                     f"nonzero (chi = 1/mu).")
                mu_spec, mu_uni = _C(mu), True
            elif m.shape == (3, 3):
                _require_inplane_mu(fn, m[None, None])
                mu_spec, mu_uni = m, True
            else:
                raise ValueError(
                    f"{fn}: a uniform mu must be a scalar or a (3, 3) "
                    f"block-form tensor, got shape {m.shape}.  A PATTERNED "
                    f"magnetic layer goes through mu_cell.")
        else:
            mu_spec = _validate_stag_mu(fn, mu_cell)
            mu_uni = False
        # union grid: any PATTERNED side (eps_cell or mu_cell) registers it
        for spec, uni in ((eps_spec, eps_uni), (mu_spec, mu_uni)):
            if uni or self.layer_grids != "shared":
                continue
            grid = tuple(np.shape(spec)[:2])
            if self._grid is None:
                self._grid = grid
            elif grid != self._grid:
                raise ValueError(
                    f"{fn}: all patterned layers must share ONE common "
                    f"(Nx, Ny) grid (the union-grid constraint of the pure "
                    f"staggered cascade); got {grid} after {self._grid}.")
        self._layers.append(dict(kind="magnetic", thickness=t, eps=eps_spec,
                                 eps_uniform=eps_uni, mu=mu_spec,
                                 mu_uniform=mu_uni, slant=(0.0, 0.0)))
        return self

    # ------------------------------------------------------------- tapers
    def _require_per_layer_taper(self, fn):
        if self.layer_grids != "per-layer":
            raise ValueError(
                f"PMM2DStackPure.{fn}: a taper is a z-staircase whose slices "
                f"have DIFFERENT wall positions, so it needs "
                f"layer_grids='per-layer' (each slice on its own non-uniform "
                f"grid, adjacent slices coupled by an L2 mortar).  On the "
                f"shared union grid the walls would have to land on ONE "
                f"lattice: a 2-degree sidewall over 310 nm at 6 slices moves a "
                f"wall ~1.8 nm per slice, which on a 700 nm period needs "
                f"N ~ 390 -- q = 390 (M - 1), i.e. unreachable at any M.  "
                f"Construct with PMM2DStackPure(..., layer_grids='per-layer'), "
                f"or use PMM2DStackHybrid (its Fourier projection decouples "
                f"the layers).")

    def add_tapered_pillar(self, thickness, *, eps_pillar, eps_host,
                           x_bounds_bottom, y_bounds_bottom,
                           x_bounds_top=None, y_bounds_top=None,
                           n_slices=8, rule="midpoint"):
        """Append a TAPERED rectangular pillar (sloped sidewalls) as a
        z-staircase of ``n_slices`` EXACT-WALL scalar layers -- the
        :meth:`PMM2DStackHybrid.add_tapered_pillar` surface on the PURE
        (no-floor) engine, and the reason non-uniform segments were built.

        Each slice sits on its OWN 3-segment NON-UNIFORM grid whose two
        interior walls are the interpolated pillar bounds, so NO TWO SLICES
        SHARE A WALL and the walls are exact (no lattice snapping, no pixel
        rounding).  Adjacent slices are coupled by the L2 mortar.  Requires
        ``layer_grids='per-layer'``.

        The pillar bounds interpolate linearly from ``*_bounds_bottom`` to
        ``*_bounds_top`` (default: equal -> a straight pillar);
        ``rule='midpoint'`` samples at slice midpoints (O(1/n_slices^2)),
        ``'bottom'`` at the slice bottoms.  Slice order is TOP-down, as the
        stack is built superstrate-first.

        Cost, and it is the whole point: 3 segments per slice, ``q = 3 (M-1)``,
        so a production ``M = 9`` slice is a ``2 * 24^2 = 1152`` eigenproblem
        where the uniform lattice would need ``q >= 1170`` and an eig above
        2.7e+06.

        A NOTE ON `M`: this is a per-layer solve, and a per-layer solve can be
        stationary in ONE knob and wrong (see :meth:`convergence_floor`).
        Converge in ``n_modes`` for the taper AND in ``n_slices``.

        A NOTE ON `n_slices` AND A CLOSING TIP (round 2, 2026-09-11): the
        midpoint rule's narrowest SAMPLED width is about
        ``w_bottom / (2 n_slices)``, so a pillar that tapers to a point walks
        toward the minimum-segment contract of
        :data:`~lumenairy.elements.pmm.twod_staggered._STAG_MIN_SEG_FRAC`.
        Measured on a pillar closing from half the period: 3.14e-02 / 8.01e-03
        / 4.11e-03 of the period at ``n_slices`` = 8 / 32 / 64, crossing the
        1e-3 contract at about 250 slices, where the slice is refused.  Stop
        the taper before its tip closes, or lower ``n_slices``."""
        self._require_per_layer_taper("add_tapered_pillar")
        if rule not in ("midpoint", "bottom"):
            raise ValueError(
                f"PMM2DStackPure.add_tapered_pillar: rule must be 'midpoint' "
                f"or 'bottom', got {rule!r}")
        n_slices = int(n_slices)
        if n_slices < 1:
            raise ValueError(
                "PMM2DStackPure.add_tapered_pillar: n_slices must be >= 1")
        xb0 = tuple(map(float, x_bounds_bottom))
        yb0 = tuple(map(float, y_bounds_bottom))
        xb1 = xb0 if x_bounds_top is None else tuple(map(float, x_bounds_top))
        yb1 = yb0 if y_bounds_top is None else tuple(map(float, y_bounds_top))
        dz = float(thickness) / n_slices
        for s in range(n_slices):
            zfrac = (1.0 - (s + 0.5) / n_slices if rule == "midpoint"
                     else 1.0 - (s + 1.0) / n_slices)
            xw = [xb0[0] + (xb1[0] - xb0[0]) * zfrac,
                  xb0[1] + (xb1[1] - xb0[1]) * zfrac]
            yw = [yb0[0] + (yb1[0] - yb0[0]) * zfrac,
                  yb0[1] + (yb1[1] - yb0[1]) * zfrac]
            if not (0.0 < xw[0] < xw[1] < self.period_x
                    and 0.0 < yw[0] < yw[1] < self.period_y):
                raise ValueError(
                    "PMM2DStackPure.add_tapered_pillar: interpolated pillar "
                    f"bounds {xw} x {yw} must satisfy 0 < lo < hi < period at "
                    "every slice.")
            tile = np.full((3, 3), _C(eps_host), dtype=_C)
            tile[1, 1] = _C(eps_pillar)
            self.add_layer(dz, eps_cell=tile, x_walls=xw, y_walls=yw)
        return self

    def add_tapered_pillars(self, thickness, *, pillars, eps_host,
                            n_slices=8):
        """Append MULTI-PILLAR tapered layers as an auto-sliced z-staircase --
        the N-feature, center-anchored generalization of
        :meth:`add_tapered_pillar`, transplanted from
        :meth:`PMM2DStackHybrid.add_tapered_pillars`.

        ``pillars`` is a list of
        ``((cx, cy), (wx_top, wy_top), (wx_bottom, wy_bottom), eps)`` in
        ABSOLUTE metres; each pillar tapers linearly ABOUT ITS OWN FIXED
        CENTER.  ``eps_host`` fills the remainder.  Every slice's walls are
        EXACT spectral-element walls on that slice's OWN non-uniform grid, so
        the whole staircase is representable without a common lattice.
        Pillars must lie strictly inside the cell (no wrap) and may not
        overlap.  Requires ``layer_grids='per-layer'``.

        The staggered tensor basis needs EQUAL SEGMENT COUNTS per axis, so a
        slice whose pillars produce a different number of x- and y-walls is
        refused; center-anchored square-ish pillar sets satisfy it naturally.
        """
        self._require_per_layer_taper("add_tapered_pillars")
        n = int(n_slices)
        if n < 1:
            raise ValueError(
                f"PMM2DStackPure.add_tapered_pillars: n_slices must be >= 1, "
                f"got {n_slices}.")
        pil = [((float(c[0]), float(c[1])), (float(wt[0]), float(wt[1])),
                (float(wb[0]), float(wb[1])), _C(e))
               for c, wt, wb, e in pillars]
        eh = _C(eps_host)
        dz = float(thickness) / n
        for k in range(n):
            zeta = (k + 0.5) / n
            rects = []
            for (cx, cy), (wxt, wyt), (wxb, wyb), e in pil:
                wxz = wxt + (wxb - wxt) * zeta
                wyz = wyt + (wyb - wyt) * zeta
                if wxz <= 0.0 or wyz <= 0.0:
                    continue
                x0, x1 = cx - 0.5 * wxz, cx + 0.5 * wxz
                y0, y1 = cy - 0.5 * wyz, cy + 0.5 * wyz
                if not (0.0 < x0 < x1 < self.period_x
                        and 0.0 < y0 < y1 < self.period_y):
                    raise ValueError(
                        "PMM2DStackPure.add_tapered_pillars: every pillar must "
                        "lie strictly inside the cell at every slice (no "
                        f"wrap); got x [{x0:.3e}, {x1:.3e}], "
                        f"y [{y0:.3e}, {y1:.3e}].")
                rects.append((x0, x1, y0, y1, e))
            for i, (ax0, ax1, ay0, ay1, _e) in enumerate(rects):
                for bx0, bx1, by0, by1, _e2 in rects[i + 1:]:
                    if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                        raise ValueError(
                            "PMM2DStackPure.add_tapered_pillars: pillars "
                            "overlap; merge or separate them explicitly.")
            xw = sorted({v for r in rects for v in (r[0], r[1])})
            yw = sorted({v for r in rects for v in (r[2], r[3])})
            if len(xw) != len(yw):
                raise ValueError(
                    f"PMM2DStackPure.add_tapered_pillars: slice {k} yields "
                    f"{len(xw)} x-walls and {len(yw)} y-walls; the staggered "
                    f"tensor basis needs EQUAL SEGMENT COUNTS per axis "
                    f"(Nx == Ny -- the wall POSITIONS may differ freely).  "
                    f"Use PMM2DStackHybrid for an unequal wall layout.")
            bx = [0.0] + xw + [self.period_x]
            by = [0.0] + yw + [self.period_y]
            tile = np.full((len(xw) + 1, len(yw) + 1), eh, dtype=_C)
            for ix in range(len(xw) + 1):
                mx = 0.5 * (bx[ix] + bx[ix + 1])
                for iy in range(len(yw) + 1):
                    my = 0.5 * (by[iy] + by[iy + 1])
                    for x0, x1, y0, y1, e in rects:
                        if x0 < mx < x1 and y0 < my < y1:
                            tile[ix, iy] = e
                            break
            self.add_layer(dz, eps_cell=tile, x_walls=xw, y_walls=yw)
        return self

    def convergence_floor(self, *, n_modes=None):
        """The CHEAP per-layer convergence SCREEN: each layer's own
        single-layer residual, a MEASURED lower bound on the stack's error.

        Returns ``(floor, per_layer)`` -- ``floor`` the max over layers and
        ``per_layer`` the list.  Each entry is that layer ALONE between the
        stack's half-spaces on its own grid, scored against the SAME layer
        alone one modal rung up (``n_modes`` overrides the rung, default
        ``M_i + 2``).  Cost: one single-layer solve per layer, i.e. ONE region
        eig instead of the stack's.

        WHY THIS EXISTS, and it is a measurement.  A per-layer solve **can be
        stationary in one knob and wrong**.  Walking ``M_A`` alone across four
        rungs on a measured two-layer pillar pair gave 3.42e-01, 2.82e-01,
        2.68e-01, 2.79e-01 -- stationary to 4 %, NOT monotone, and 27 % wrong,
        because layer B's own ``M_B`` was the limiting error the whole time;
        the mirror experiment fails the same way.  Only the DIAGONAL descends.
        So:

        1. **The stopping criterion is stationarity in EVERY ``n_modes``, never
           in one.**  Stationarity in one knob is not evidence.
        2. **Screen with this floor first.**  ``pair_error >= max_i
           own_residual_i`` held at 15 of 16 surface points (the single
           exception missed by 1.2x), so a layer whose own residual is 5e-02
           makes a 1e-03 stack answer impossible -- and that is knowable before
           the stack is ever assembled.
        3. A greedy "raise the layer with the larger residual" rule was tested
           and reads 6/9.  It is a HINT, not a rule; the floor is the gate.
        """
        if self.layer_grids != "per-layer":
            raise ValueError(
                "PMM2DStackPure.convergence_floor: only meaningful with "
                "layer_grids='per-layer' -- the shared path carries ONE modal "
                "count for the whole stack, which is exactly what makes it "
                "impossible to mis-set per layer.")
        if self._src is None:
            raise ValueError(
                "PMM2DStackPure.convergence_floor: call set_source(...) first.")
        Ms = self._perlayer_modal_counts()
        out = []
        for L, Mi in zip(self._layers, Ms):
            vals = []
            # the ISOLATED layer's end grids are its OWN, so its Rayleigh
            # capacity is q = N (M - 1) and the stack's n_orders may exceed it.
            # CLAMP rather than raise -- this is an internal residual probe --
            # and clamp on the LOWER rung so both rungs retain the same order
            # set and the two vectors are comparable.
            _nq = _stag_walls_n(L["wx"]) * (Mi - 1)
            _nord = max(0, min(self.n_orders, (_nq - 1) // 2))
            for M in (Mi, Mi + 2 if n_modes is None else int(n_modes)):
                st = PMM2DStackPure(
                    self.period_x, self.period_y,
                    n_superstrate=self.n_sup, n_substrate=self.n_sub,
                    n_modes=M, n_orders=_nord,
                    symmetry=self.symmetry, layer_grids="per-layer")
                kw = dict(thickness=L["thickness"], n_modes=M)
                if L["kind"] == "patterned":
                    kw.update(eps_cell=L["eps_cell"], x_walls=_stag_interior(
                        L["wx"]), y_walls=_stag_interior(L["wy"]))
                elif L["kind"] == "uniform":
                    kw.update(eps=L["eps"], grid=_stag_walls_n(L["wx"]))
                elif L["kind"] == "uniform_tensor":
                    kw.update(eps=L["eps33"], grid=_stag_walls_n(L["wx"]))
                else:                       # magnetic
                    kw.update(
                        **({"eps": L["eps"]} if L["eps_uniform"]
                           else {"eps_cell": L["eps"]}),
                        **({"mu": L["mu"]} if L["mu_uniform"]
                           else {"mu_cell": L["mu"]}))
                    if L["eps_uniform"] and L["mu_uniform"]:
                        kw["grid"] = _stag_walls_n(L["wx"])
                t = kw.pop("thickness")
                st.add_layer(t, **kw)
                st.set_source(self._src["wl"], theta=self._src["theta"],
                              phi=self._src["phi"])
                o, R, T = st.solve(jones=False)
                vals.append(np.concatenate([np.asarray(R).ravel(),
                                            np.asarray(T).ravel()]))
            out.append(float(np.max(np.abs(vals[0] - vals[1]))))
        return (max(out) if out else 0.0), out

    def set_source(self, wavelength, *, theta=0.0, phi=0.0):
        """Set the incident plane wave: vacuum ``wavelength`` (m), polar
        ``theta`` and azimuth ``phi`` (radians)."""
        self._modal = None      # source change supersedes retained amplitudes
        self._internal = None
        self._src = dict(wl=float(wavelength), theta=float(theta),
                         phi=float(phi))
        return self

    def _source_prep(self):
        """Source/wavelength preamble shared by the shared-grid and per-layer
        cascades.

        Moved VERBATIM out of :meth:`solve` when ``layer_grids='per-layer'``
        was added (2026-09-11) so the two paths cannot drift apart: the
        propagating-incidence precondition, the Wood-anomaly wavelength nudge
        over EVERY region's real permittivities, and the Rayleigh-cutoff
        proximity warning.  The arithmetic is unchanged, so the shared path
        stays bit-identical.

        NOTE the layer loop reads the LAYER RECORDS, never a union cell, so it
        carries over to per-layer grids untouched (open item O-7).

        Returns ``(wl, k0, kx0, ky0, a0x, a0y, eps_sup, eps_sub)`` -- ``wl``
        being the possibly NUDGED wavelength every downstream quantity uses.
        """
        wl = self._src["wl"]
        theta, phi = self._src["theta"], self._src["phi"]
        px, py = self.period_x, self.period_y
        eps_sup = _C(self.n_sup) ** 2
        eps_sub = _C(self.n_sub) ** 2
        n_orders = self.n_orders
        from ..rcwa._core import (
            _grazing_safe_wavelength,
            _require_propagating_incidence,
        )
        nre0 = float(np.real(np.sqrt(eps_sup)))
        _mo = np.arange(-n_orders, n_orders + 1)
        _mx = np.tile(_mo, len(_mo))
        _my = np.repeat(_mo, len(_mo))
        _kx0n = nre0 * np.sin(theta) * np.cos(phi)
        _ky0n = nre0 * np.sin(theta) * np.sin(phi)
        _require_propagating_incidence("PMM2DStackPure.solve", np.conj(eps_sup),
                                       _kx0n ** 2 + _ky0n ** 2)
        # Wood-anomaly nudge.  EVERY region contributes its real
        # permittivities -- both half-spaces, every scalar layer (uniform or
        # patterned, each distinct cell value) and every TENSOR layer's
        # principal DIAGONALS: a grazing LAYER mode is what crashes the
        # interface S-matrix, and an anisotropic layer has several principal
        # indices.  ONE rule for both paths (2026-09-10): until then SCALAR
        # layers were left out here and in `pmm_efficiency_2d_staggered`, so a
        # scalar cell and its `e * I` promotion -- the same discretization
        # everywhere else -- took different nudges on a LAYER cut-off and
        # differed there by a measured 4.59e-08.  Off an EXACT coincidence
        # nothing moves: the guard's trigger band is |eps - kt^2| <= 1e-9.
        _eps_src = [eps_sup, eps_sub]
        for _L in self._layers:
            if _L["kind"] == "uniform":
                _eps_src.append(_L["eps"])
            elif _L["kind"] == "uniform_tensor":
                _eps_src.append(np.diag(_L["eps33"]))
            elif _L["kind"] == "magnetic":
                # A MAGNETIC layer's cut-offs sit at Re(eps*mu), not Re(eps):
                # its index is sqrt(eps mu).  The magnetic record carries
                # ``eps`` / ``mu`` (scalar / (3,3) uniform / (Nx,Ny) /
                # (Nx,Ny,3,3)), not ``eps_cell``.  ``mu = 1`` reproduces the
                # permittivity BIT FOR BIT, so a nonmagnetic-equivalent layer
                # takes exactly the nudge it took before.
                _eps_src.append(_wood_cutoff_products(_L))
            elif _L["eps_cell"].ndim == 4:
                _eps_src.append(_L["eps_cell"][..., [0, 1, 2], [0, 1, 2]])
            else:
                _eps_src.append(_L["eps_cell"])
        wl = _grazing_safe_wavelength(wl, _kx0n, _ky0n, _mx, _my, px, py,
                                      _wood_eps_reals(*_eps_src))
        _kt2 = ((_kx0n + _mx * (wl / px)) ** 2 + (_ky0n + _my * (wl / py)) ** 2)
        _gap = min(float(np.min(np.abs(float(np.real(e)) - _kt2)))
                   for e in (eps_sup, eps_sub))
        if _gap < 1e-4:
            warnings.warn(
                f"PMM2DStackPure.solve: a diffraction order is within {_gap:.2g} "
                f"(kt^2 units) of a Rayleigh cutoff; the staggered basis "
                f"accuracy degrades like ~1/sqrt(distance) near cutoffs.  "
                f"Detune the wavelength or use PMM2DStackHybrid.", stacklevel=2)

        k0 = 2.0 * np.pi / wl
        nre = float(np.real(np.sqrt(eps_sup)))
        kx0 = nre * np.sin(theta) * np.cos(phi)
        ky0 = nre * np.sin(theta) * np.sin(phi)
        a0x, a0y = kx0 * k0, ky0 * k0
        return wl, k0, kx0, ky0, a0x, a0y, eps_sup, eps_sub

    # ------------------------------------------------------------------ solve
    def solve(self, *, jones=True, retain_internal=False):
        """Cascade the stack and return the diffraction efficiencies.

        Returns ``(orders, R, T, jones)`` (default) or ``(orders, R, T)`` when
        ``jones=False``.  ``orders`` is ``(Nfo, 2)`` ``(m, n)`` pairs; ``R``/``T``
        are ``(2, Nfo)`` real efficiencies (row 0 = incident ``Ex``, row 1 =
        incident ``Ey``); ``jones`` is the ``(2, 2)`` order-0 reflection Jones.

        ``retain_internal=True`` (AUDIT_DYNAMETA_CONSUMER_API_GAPS C3)
        additionally retains the per-layer partial cascades + the staggered
        block field Gram for :meth:`layer_absorption` -- absorption budgets
        for the engine of record on lossy-metal cells."""
        # every solve supersedes the retained per-order amplitudes AND the
        # retained internals (the audit-P1-04 invalidation contract)
        self._modal = None
        self._internal = None
        if self._src is None:
            raise ValueError("PMM2DStackPure.solve: call set_source(...) first.")
        if not self._layers:
            raise ValueError("PMM2DStackPure.solve: add at least one layer.")
        _check_stack_slant(self._layers, "PMM2DStackPure.solve")
        _slanted_stack = any(not _slant_is_zero(L.get("slant"))
                             for L in self._layers)
        if retain_internal and _slanted_stack:
            # The frame-anchor bookkeeping below is a FAR-FIELD correction on
            # the transmitted orders.  An internal probe evaluated at a plane
            # inside or below a sheared layer lives in the FRAME, not the lab,
            # and needs the same treatment -- which _flux_at has not been
            # taught.  Refuse rather than return frame-referenced fluxes.
            raise NotImplementedError(
                "PMM2DStackPure.solve: retain_internal=True is not supported "
                "on a stack containing a SLANTED layer -- the internal-field "
                "probe (_flux_at / layer_absorption) evaluates at a plane in "
                "the SHEARED frame, where the lateral frame offset "
                "sum_j t_j d_j has not been undone, so the retained "
                "amplitudes are not lab-referenced.  Solve without "
                "retain_internal, or z-staircase the slanted layer.")
        if self.layer_grids == "per-layer":
            return self._solve_per_layer(jones=jones,
                                         retain_internal=retain_internal)
        # Multi-patterned (A|B) cascades are fully supported: the historical
        # A|B energy blow-up was the far-field projection-kernel order MIRROR
        # (fixed in twod_staggered._stag_fourier_projection), not an
        # interface/mode-sorting defect -- see the module docstring (Scope) and
        # test_v5_21_pmm2d_staggered_oblique.test_stack_pure_multilayer_ab_vs_1d.
        px, py = self.period_x, self.period_y
        Nx, Ny = self._grid if self._grid is not None else (2, 2)
        M = self.M
        n_orders = self.n_orders
        (wl, k0, kx0, ky0, a0x, a0y,
         eps_sup, eps_sub) = self._source_prep()

        # Shared eps-free geometric eig -> both half-spaces AND every uniform
        # layer (degeneracy-safe; all share the eigenvectors W0).
        sol_h = Granet2DTransverseE(px, py, Nx, Ny, M,
                                    np.full((Nx, Ny), eps_sup),
                                    alpha0x=a0x, alpha0y=a0y, k0=k0)
        geom = _homog_geom_cache(sol_h)
        bx, by = sol_h.bx, sol_h.by
        # C3: the block field Gram (geometry-only -- eps-free, shared by every
        # region on the union grid) is the flux bilinear form layer_absorption
        # integrates with; retain it before dropping the assembly.
        G_gram = (-sol_h.Rmat).copy() if retain_internal else None
        del sol_h
        Wsup, Vsup, _ls = _homog_region_modes(geom, eps_sup)
        Wsub, Vsub, _lb = _homog_region_modes(geom, eps_sub)

        # Per-layer modes: uniform SCALAR -> shared geom (cheap); patterned or
        # tensor -> its own staggered eig, deduped across byte-identical cells
        # (a DBR eigs each distinct layer once).  A layer whose tensor carries
        # OUT-OF-PLANE coupling routes through the first-order 4 q^2 generator
        # and returns DISTINCT forward/backward sets; every entry is stored in
        # the 6-tuple form so one cascade serves both kinds.
        modes = []
        eig_cache = {}
        any_oop = False
        for L in self._layers:
            sl = L.get("slant", (0.0, 0.0))
            slanted = not _slant_is_zero(sl)
            if L["kind"] == "uniform" and not slanted:
                W, V, lam = _homog_region_modes(geom, L["eps"])
                six = _modes_as_general(W, V, lam)
            else:
                mcell = None
                if L["kind"] == "uniform" and slanted:
                    # A SLANTED uniform layer cannot ride the shared eps-free
                    # geometric eig: in the frame its covariant tensor carries
                    # out-of-plane entries, so it is an out-of-plane region and
                    # takes its own 4 q^2 solve on the union grid.  (Physically
                    # it is still a no-op -- a shear of a homogeneous medium is
                    # a coordinate change -- which is exactly the null test.)
                    cell = np.ascontiguousarray(
                        np.full((Nx, Ny), _C(L["eps"])))
                elif L["kind"] == "uniform_tensor":
                    # A uniform TENSOR region is NOT eps-free-separable (its
                    # div(D)=0 Schur term mixes e11/e21 while Meps33 carries
                    # e33 alone), so it cannot ride the shared geometric eig --
                    # it is assembled as a constant cell on the common grid and
                    # takes a full region eig, deduped like a patterned cell.
                    cell = np.ascontiguousarray(
                        np.broadcast_to(L["eps33"], (Nx, Ny, 3, 3)))
                elif L["kind"] == "magnetic":
                    # A MAGNETIC region is not eps-free-separable either (chi_t
                    # and chi33 weight R, K_tz and S_tt), so it too takes its
                    # own region eig on the union grid -- deduped on BOTH cells.
                    cell = _as_layer_cell(L["eps"], L["eps_uniform"], Nx, Ny)
                    mcell = _as_layer_cell(L["mu"], L["mu_uniform"], Nx, Ny)
                else:
                    cell = L["eps_cell"]
                # The SLANT is part of the key: two layers with the same
                # cell and different slant vectors have different modes and
                # must not share a cached solve.
                key = ((cell.shape, cell.tobytes(), sl) if mcell is None else
                       (cell.shape, cell.tobytes(), mcell.shape,
                        mcell.tobytes(), sl))
                cached = eig_cache.get(key)
                if cached is None:
                    sol = Granet2DTransverseE(px, py, Nx, Ny, M, cell,
                                              alpha0x=a0x, alpha0y=a0y, k0=k0,
                                              mu_cell=mcell,
                                              slant=sl if slanted else None)
                    if sol.offplane:
                        cached = _region_modes_oop(sol,
                                                   symmetry=self.symmetry)
                    else:
                        Wl, Vl, lam_l, _g2 = _region_modes(sol)
                        cached = _modes_as_general(Wl, Vl, lam_l)
                    eig_cache[key] = cached
                six = cached
                # A SLANTED layer is an OUT-OF-PLANE region in the frame, so it
                # promotes the whole stack to the generalized cascade exactly
                # as an out-of-plane tensor layer does.
                any_oop = any_oop or slanted or (
                    len(cell.shape) == 4
                    and _tile_needs_oop("PMM2DStackPure.solve", cell))
            modes.append(six + (L["thickness"],))

        # SQUARE Redheffer cascade: sup | (interface, propagate)* | sub.  The
        # interface list is precomputed (same matrices, same star sequence --
        # bit-identical to the inline build) so retain_internal can reuse it
        # for the bracketing partial cascades.
        nlay = len(modes)
        if any_oop:
            # GENERALIZED cascade: an out-of-plane layer breaks the in-plane
            # ``[W; -V] <-> -lam`` symmetry, so forward and backward modes are
            # genuinely distinct and every region -- half-spaces, uniform and
            # in-plane layers included -- enters through the full field-mode
            # matrix ``[[Wf, Wb], [Vf, Vb]]``.  The isotropic half-spaces are
            # ``[[W, W], [V, -V]]``; measured against the shipped SQUARE
            # interface on an isotropic pair, the two agree to 7.7e-14 (S11) /
            # 8.1e-14 (S21) at normal and 1.0e-13 / 9.3e-14 at conical, so the
            # mixed route costs nothing in accuracy.
            Msup = _modes_to_M(Wsup, Vsup, Wsup, -Vsup)
            Msub = _modes_to_M(Wsub, Vsub, Wsub, -Vsub)
            Mlay = [_modes_to_M(m[0], m[1], m[3], m[4]) for m in modes]
            ifc = [_interface_smatrix_general(Msup, Mlay[0])]
            for i in range(1, nlay):
                ifc.append(_interface_smatrix_general(Mlay[i - 1], Mlay[i]))
            ifc.append(_interface_smatrix_general(Mlay[-1], Msub))
            prop = [_propagation_smatrix_general(m[2], m[5], k0 * m[6])
                    for m in modes]
        else:
            ifc = [_interface_smatrix(Wsup, Vsup, modes[0][0], modes[0][1])]
            for i in range(1, nlay):
                ifc.append(_interface_smatrix(modes[i - 1][0], modes[i - 1][1],
                                              modes[i][0], modes[i][1]))
            ifc.append(_interface_smatrix(modes[-1][0], modes[-1][1],
                                          Wsub, Vsub))
            prop = [_propagation_smatrix(m[2], k0 * m[6]) for m in modes]
        S = ifc[0]
        for i in range(nlay):
            S = _redheffer_star(S, prop[i])
            S = _redheffer_star(S, ifc[i + 1])
        S11, _S12, S21, _S22 = S
        if retain_internal:
            # partial cascades (the RCWAStack / PMMStack / Hybrid pattern):
            # S_above[i] = sup -> TOP of layer i; S_below_bot[i] = BOTTOM of
            # layer i -> sub (no own propagation -> both exponentials decay).
            S_above = [None] * nlay
            S_above[0] = ifc[0]
            for i in range(1, nlay):
                S_above[i] = _redheffer_star(
                    _redheffer_star(S_above[i - 1], prop[i - 1]), ifc[i])
            S_below_bot = [None] * nlay
            S_below_bot[nlay - 1] = ifc[nlay]
            for i in range(nlay - 2, -1, -1):
                S_below_bot[i] = _redheffer_star(
                    _redheffer_star(ifc[i + 1], prop[i + 1]),
                    S_below_bot[i + 1])

        # FORWARD-only far-field Fourier->Rayleigh projection (once).
        ox = np.arange(-n_orders, n_orders + 1)
        oy = np.arange(-n_orders, n_orders + 1)
        order_x = np.tile(ox, len(oy))
        order_y = np.repeat(oy, len(ox))
        Nfo = len(order_x)
        P1, P2 = _far_projector_2d(bx, by, ox, oy, a0x, a0y)
        qq = (Nx * (M - 1)) * (Ny * (M - 1))

        Hsup = _pmm2d_project_orders(P1, P2, Wsup, qq)
        Hsub = _pmm2d_project_orders(P1, P2, Wsub, qq)
        kxv = kx0 + order_x * (wl / px)
        kyv = ky0 + order_y * (wl / py)
        kz_ref, kz_trn, kz_inc, safe_r, safe_t = _pmm2d_order_kz(
            eps_sup, eps_sub, kxv, kyv, kx0, ky0)
        delta = ((order_x == 0) & (order_y == 0)).astype(_C)
        p0 = int(np.where((order_x == 0) & (order_y == 0))[0][0])

        # ---- THE FRAME-ANCHOR PHASE (the one piece of bookkeeping a shear
        # adds).  Every slanted region is solved in a frame anchored at ITS OWN
        # TOP face, so the SUPERSTRATE side coincides with the lab and the
        # SUBSTRATE plane sits at ``u = x - sum_j t_j d_j``.  Both bounding
        # half-spaces are HOMOGENEOUS, so that lateral offset is a gauge -- a
        # translation maps a homogeneous half-space to itself -- and it appears
        # as ONE unimodular diagonal phase per order on the TRANSMITTED
        # amplitudes alone:
        #
        #     T_lab(m) = exp(-i alpha_m . t d) T_frame(m)
        #
        # ``alpha_m`` is real for every order, propagating or evanescent, so
        # the factor is always unimodular: R, T and the REFLECTION Jones are
        # EXACT without it and the efficiencies are untouched -- only the
        # TRANSMISSION amplitudes carry it.  That is precisely why it is a
        # silent-wrong if omitted (the two papers this method follows report
        # efficiencies, where a unimodular per-order factor is invisible; this
        # library returns Jones matrices, where it is not).  MEASURED on a
        # uniform null at slants of 10-35 deg and 25 deg incidence: the
        # shipped arm reads 9.86e-08 .. 2.64e-05, omitting the factor leaves
        # the transmission Jones wrong by 1.43e-01 .. 7.42e-01, and the ``+i``
        # arm (2.86e-01 .. 1.33e+00) is about TWICE as wrong as no correction
        # at all -- so it is not a fudge that could absorb an arbitrary
        # residual (docs/audits/BUILD_PMM2D_STAGGERED_SLANT_2026_09_10.md
        # table B3; the two builds agree there to every printed digit).
        #
        # ``t`` in that formula is the INTERNAL shear of ``x = u + t w``, which
        # is the NEGATIVE of the public ``slant`` (the same relation
        # ``Granet2DTransverseE`` applies to the congruence).  Written with the
        # public vector the factor is therefore exp(+i alpha_m . slant d) --
        # and the sign is not a matter of taste: on the uniform null it is the
        # difference between 1e-07 and 1.3e+00 (table B3), with the wrong arm
        # TWICE as wrong as applying no correction at all.
        tphase = None
        if _slanted_stack:
            _shx = -sum(L.get("slant", (0.0, 0.0))[0] * L["thickness"]
                        for L in self._layers)
            _shy = -sum(L.get("slant", (0.0, 0.0))[1] * L["thickness"]
                        for L in self._layers)
            if _shx != 0.0 or _shy != 0.0:
                tphase = np.exp(-1j * k0 * (kxv * _shx + kyv * _shy))

        R_rows, T_rows, j_cols, cinc_cols = [], [], [], []
        amp = {k: np.zeros((2, Nfo), dtype=_C)
               for k in ("rx", "ry", "tx", "ty")}
        for col, (ex0, ey0) in enumerate(((1.0, 0.0), (0.0, 1.0))):
            long_inc = kx0 * ex0 + ky0 * ey0
            einc_sq = 1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0 else 1.0
            rhs = np.concatenate([ex0 * delta, ey0 * delta])
            cinc = _guarded_lstsq(
                Hsup, rhs, "PMM2DStackPure far-field Rayleigh projection")
            cinc_cols.append(cinc)
            r_ord = Hsup @ (S11 @ cinc)
            t_ord = Hsub @ (S21 @ cinc)
            rx, ry = r_ord[:Nfo], r_ord[Nfo:]
            tx, ty = t_ord[:Nfo], t_ord[Nfo:]
            if tphase is not None:
                tx, ty = tx * tphase, ty * tphase
            rz = -(kxv * rx + kyv * ry) / safe_r
            tz = -(kxv * tx + kyv * ty) / safe_t
            Re, Te = _project_efficiency(np, kz_ref, kz_trn, kz_inc,
                                         rx, ry, rz, tx, ty, tz, einc_sq)
            R_rows.append(Re)
            T_rows.append(Te)
            j_cols.append(np.stack([rx[p0], ry[p0]]))
            # this cascade is PUBLIC gauge end-to-end -> no conj
            amp["rx"][col], amp["ry"][col] = rx, ry
            amp["tx"][col], amp["ty"][col] = tx, ty
        R_eff = np.stack(R_rows)
        T_eff = np.stack(T_rows)
        jmat = np.stack(j_cols, axis=1)           # jmat[out_comp, in_pol]
        orders2d = np.stack([order_x, order_y], axis=1)
        # AUDIT_DYNAMETA_CONSUMER_API_GAPS B (PerOrderAmplitudesMixin):
        # kz_ref/kz_trn are evaluated on the PUBLIC eps -> decaying branch.
        self._modal = dict(
            orders=orders2d.copy(), p0=p0,
            kx=kxv.copy(), ky=kyv.copy(),
            kz_ref=kz_ref.copy(), kz_trn=kz_trn.copy(),
            kz_inc=kz_inc, kx0=float(kx0), ky0=float(ky0),
            wavelength=float(wl), **amp)
        if retain_internal:
            self._internal = dict(
                modes=modes, S_above=S_above, S_below_bot=S_below_bot,
                G=G_gram, qq=qq, k0=k0, any_oop=any_oop,
                cinc=np.stack(cinc_cols, axis=1),
                R_tot=R_eff.sum(axis=1), T_tot=T_eff.sum(axis=1))
        _warn_stag_closure(R_eff, T_eff, self._layers, eps_sup, eps_sub)
        if not jones:
            return orders2d, R_eff, T_eff
        return orders2d, R_eff, T_eff, jmat

    # ------------------------------------------------- per-layer grid cascade
    def _perlayer_modal_counts(self):
        """Each layer's effective modal count ``M_i``, applying the MEASURED
        default for a UNIFORM layer that did not name one.

        F4 (open item O-6) measured both halves of this.  ISOLATED, a uniform
        region wants the whole ``q`` budget on ONE element: the field is a
        single plane wave, so the error is spectral in the polynomial degree
        and ``N = 1`` beats ``N = 2`` / ``N = 3`` by 1-5 DECADES at matched
        ``q`` at every angle measured, conical included.  IN A CASCADE the
        uniform layer must ALSO represent its neighbours' traces, which have
        kinks at THEIR walls -- and there ``N = 1`` at the stack's ``M`` reads
        3-6x worse than ``N = 2`` or ``3``.  The fix is not the grid but the
        DEGREE: at matched ``q_u = max(q_prev, q_next)`` the three grids agree
        to 3 % and all sit on the neighbours' floor, so

            M_u = max(q_prev, q_next) / N_u + 1

        is the default, and an explicit ``n_modes=`` overrides it.  ``N = 1``
        is cheap; ``N = 1`` at the stack's ``M`` is a trap.
        """
        Ls = self._layers
        n = len(Ls)
        base = [None] * n
        for i, L in enumerate(Ls):
            fixed = (L["kind"] not in ("uniform", "uniform_tensor")
                     or L.get("n_modes_given", False))
            if fixed:
                base[i] = int(L["M"])
        out = []
        for i, L in enumerate(Ls):
            if base[i] is not None:
                out.append(base[i])
                continue
            nb = [j for j in (i - 1, i + 1) if 0 <= j < n and base[j] is not None]
            q_nb = max((_stag_walls_n(Ls[j]["wx"]) * (base[j] - 1)
                        for j in nb), default=None)
            Nu = _stag_walls_n(L["wx"])
            if q_nb is None:
                out.append(int(L["M"]))
            else:
                out.append(max(3, int(-(-q_nb // Nu)) + 1))
        return out

    def _solve_per_layer(self, *, jones, retain_internal, force_mortar=False):
        """``layer_grids='per-layer'``: every layer on its OWN element grid,
        adjacent grids coupled by an L2 MORTAR.

        Design and every number quoted in the comments:
        ``docs/audits/EXPERIMENT_PMM2D_STAGGERED_MORTAR_2026_09_10.md`` and the
        build doc ``BUILD_PMM2D_STAGGERED_MORTAR_2026_09_11.md``.

        ``force_mortar=True`` is a TEST INSTRUMENT: it disables the
        identical-grid bypass so a CONFORMING stack is driven through the
        mortar algebra, which is the only way to score the conforming identity.
        Library code never sets it.

        ROUND 2 (2026-09-11, ``FIX_PMM2D_MORTAR_ROUND2_2026_09_11.md``).  This
        path carries TWO guards that the shared path does not need, and they
        are layered on purpose:

        * the MINIMUM SEGMENT WIDTH contract, enforced where the grid is BUILT
          (:class:`~lumenairy.elements.pmm.twod_staggered.Basis1D`), because
          it is pure geometry and costs no solve.  It is what stops an
          intra-layer sliver from putting spurious ``1/J_n`` wavenumbers into
          the cross-grid projection ENERGY-INVISIBLY;
        * the CONDITIONING backstop on the mortar's own solves
          (:func:`~lumenairy.elements.pmm._core._guarded_mortar_solve`), which
          catches what a fixed WIDTH bar cannot know -- the modal count, the
          wavelength and the contrast all move the conditioning, and it
          degrades about 3x per modal rung.

        The mortar's ALGEBRA is not what either guards: three ALL-HOST layers
        on three different non-uniform grids reproduce the ANALYTIC slab to
        1.2e-10 at ``M`` = 5 and 3.3e-13 at ``M`` = 6, at every wall
        separation.  What a degenerate grid loses is a PATTERNED neighbour's
        structured trace.
        """
        px, py = self.period_x, self.period_y
        (wl, k0, kx0, ky0, a0x, a0y,
         eps_sup, eps_sub) = self._source_prep()
        taux = np.exp(-1j * a0x * px)
        tauy = np.exp(-1j * a0y * py)
        Ms = self._perlayer_modal_counts()

        # ---- per-layer grids, deduped on CONTENT (walls + tau + M) ----------
        def _wkey(w):
            if np.ndim(w) == 0:
                return ("N", int(w))
            return ("W", np.asarray(w, dtype=float).tobytes())

        grids_pre, grids_by_key = {}, {}

        def _grid(wx, wy, M):
            pre = (_wkey(wx), _wkey(wy), int(M))
            g = grids_pre.get(pre)
            if g is None:
                g = StagGridOps(px, py, wx, wy, M, taux, tauy)
                g = grids_by_key.setdefault(g.key(), g)
                grids_pre[pre] = g
            return g

        gof = [_grid(L["wx"], L["wy"], M) for L, M in zip(self._layers, Ms)]
        # ROUND 3 (VERIFY S5.4): the band ABOVE the width contract is accepted
        # and measurably degraded, and until now SILENTLY.  This is the one
        # place where the NEIGHBOURS are known, so the warning is conditioned
        # on the stack actually building a cross-grid interface -- a fully
        # CONFORMING per-layer stack takes the plain square match everywhere
        # and is measured delta-independent, so it must not warn.
        # ROUND 4 (VERIFY round 3, DEFECT 2): that test is PER AXIS.  Round 3
        # asked it of the STACK and then scanned both axes, so layers differing
        # on x while sharing the y wall array warned about a narrow y segment
        # on which the mortar is the identity.
        _warn_stag_sliver_band(gof, _stag_mortared_axes(gof, force_mortar))
        # HALF-SPACES ride the grid of the layer they TOUCH -- the 1-D
        # convention, and more strongly motivated here: both end interfaces
        # become PLAIN square matches, so no mortar ever sits where the far
        # field is projected, and the eps-free geometric eig that makes the two
        # half-spaces free is per grid and already built for that layer.
        g_sup, g_sub = gof[0], gof[-1]

        # ---- the far-field order CAP, derived from the END grids ------------
        # T3-3 applied before it bites: the per-axis Rayleigh projection has
        # q = N (M - 1) columns, so it can carry at most q order slots; asking
        # for more makes the retained orders linearly DEPENDENT (slots aliasing
        # one another) and the least-squares draw build-dependent -- while
        # conserving energy exactly, i.e. energy-invisible.  The probe CLAMPED;
        # a shipped surface RAISES (open item O-9), because a user who asks for
        # orders the end grids cannot carry should be told, not quietly served
        # fewer.
        cap = min((g_sup.q - 1) // 2, (g_sub.q - 1) // 2)
        if self.n_orders > cap:
            raise ValueError(
                f"PMM2DStackPure.solve: n_orders={self.n_orders} exceeds what "
                f"this stack's END grids can carry.  The forward Rayleigh "
                f"projection has q = N (M - 1) columns per axis -- {g_sup.q} "
                f"at the top (N={g_sup.N}, M={g_sup.M}) and {g_sub.q} at the "
                f"bottom (N={g_sub.N}, M={g_sub.M}) -- so "
                f"n_orders <= (q - 1) // 2 = {cap}.  Beyond it the retained "
                f"order slots alias one another and the least-squares draw is "
                f"build-dependent while conserving energy exactly.  Lower "
                f"n_orders to {cap}, or raise the FIRST and LAST layers' "
                f"n_modes / grid (the half-spaces ride their grids).")
        n_orders = self.n_orders

        # ---- per-grid eps-free geometric eig (half-spaces + uniform scalar) -
        geo_cache = {}

        def _walls_of(g):
            return (g.bx.N if g.bx.uniform else g.bx.xb,
                    g.by.N if g.by.uniform else g.by.xb)

        def _geo(g):
            hit = geo_cache.get(g.key())
            if hit is None:
                wxg, wyg = _walls_of(g)
                sol = Granet2DTransverseE(
                    px, py, wxg, wyg, g.M,
                    np.full((g.bx.N, g.by.N), eps_sup),
                    alpha0x=a0x, alpha0y=a0y, k0=k0)
                hit = _homog_geom_cache(sol)
                geo_cache[g.key()] = hit
            return hit

        Wsup, Vsup, lam_sup = _homog_region_modes(_geo(g_sup), eps_sup)
        Wsub, Vsub, lam_sub = _homog_region_modes(_geo(g_sub), eps_sub)

        # ---- per-layer modes ------------------------------------------------
        modes, any_oop, eig_cache = [], False, {}
        for L, g in zip(self._layers, gof):
            sl = L.get("slant", (0.0, 0.0))
            slanted = not _slant_is_zero(sl)
            Nx, Ny = g.bx.N, g.by.N
            if L["kind"] == "uniform" and not slanted:
                W, V, lam = _homog_region_modes(_geo(g), L["eps"])
                six = _modes_as_general(W, V, lam)
            else:
                mcell = None
                if L["kind"] == "uniform" and slanted:
                    cell = np.ascontiguousarray(
                        np.full((Nx, Ny), _C(L["eps"])))
                elif L["kind"] == "uniform_tensor":
                    cell = np.ascontiguousarray(
                        np.broadcast_to(L["eps33"], (Nx, Ny, 3, 3)))
                elif L["kind"] == "magnetic":
                    cell = _as_layer_cell(L["eps"], L["eps_uniform"], Nx, Ny)
                    mcell = _as_layer_cell(L["mu"], L["mu_uniform"], Nx, Ny)
                else:
                    cell = L["eps_cell"]
                # O-8: the eig key carries the GRID (wall arrays + tau + M),
                # not an integer N -- two layers can share N and M and still be
                # different grids, and the Bloch glue lives in the basis.
                key = ((g.key(), cell.shape, cell.tobytes(), sl)
                       if mcell is None else
                       (g.key(), cell.shape, cell.tobytes(), mcell.shape,
                        mcell.tobytes(), sl))
                cached = eig_cache.get(key)
                if cached is None:
                    wxg, wyg = _walls_of(g)
                    sol = Granet2DTransverseE(
                        px, py, wxg, wyg, g.M, cell,
                        alpha0x=a0x, alpha0y=a0y, k0=k0,
                        mu_cell=mcell, slant=sl if slanted else None)
                    if sol.offplane:
                        cached = _region_modes_oop(sol, symmetry=self.symmetry)
                    else:
                        Wl, Vl, lam_l, _g2 = _region_modes(sol)
                        cached = _modes_as_general(Wl, Vl, lam_l)
                    eig_cache[key] = cached
                six = cached
                any_oop = any_oop or slanted or (
                    len(cell.shape) == 4
                    and _tile_needs_oop("PMM2DStackPure.solve", cell))
            modes.append(six + (L["thickness"],))

        # ---- interfaces: PLAIN when the two grids coincide, mortar otherwise -
        cross_cache = {}

        def _cross(ga, gb):
            key = (ga.key(), gb.key())
            hit = cross_cache.get(key)
            if hit is None:
                hit = StagCrossOps(ga, gb)
                cross_cache[key] = hit
            return hit

        nlay = len(modes)

        def _ifc(ia, ib):
            if ia is None:
                ga, gb = g_sup, gof[0]
                sa = (Wsup, Vsup, lam_sup, Wsup, -Vsup, -lam_sup)
                sb = modes[0][:6]
            elif ib is None:
                ga, gb = gof[-1], g_sub
                sa = modes[-1][:6]
                sb = (Wsub, Vsub, lam_sub, Wsub, -Vsub, -lam_sub)
            else:
                ga, gb = gof[ia], gof[ib]
                sa, sb = modes[ia][:6], modes[ib][:6]
            # The identical-grid BYPASS is what makes this mode safe to expose:
            # a per-layer stack whose grids happen to coincide takes the
            # shipped square interface and reproduces layer_grids='shared'
            # BIT-EXACTLY (the 1-D contract, lifted to 2-D).
            same = (ga.key() == gb.key()) and not force_mortar
            if any_oop:
                if same:
                    return _interface_smatrix_general(
                        _modes_to_M(sa[0], sa[1], sa[3], sa[4]),
                        _modes_to_M(sb[0], sb[1], sb[3], sb[4]))
                return _interface_smatrix_general_mortar_2d(
                    sa, sb, ga, gb, _cross(ga, gb), _stag_kron_apply)
            if same:
                return _interface_smatrix(sa[0], sa[1], sb[0], sb[1])
            return _interface_smatrix_mortar_2d(
                sa[0], sa[1], sb[0], sb[1], ga, gb, _cross(ga, gb),
                _stag_kron_apply)

        ifc = [_ifc(None, 0)]
        for i in range(1, nlay):
            ifc.append(_ifc(i - 1, i))
        ifc.append(_ifc(nlay - 1, None))
        if any_oop:
            prop = [_propagation_smatrix_general(m[2], m[5], k0 * m[6])
                    for m in modes]
        else:
            prop = [_propagation_smatrix(m[2], k0 * m[6]) for m in modes]
        # RECTANGULAR Redheffer: adjacent per-layer grids carry different mode
        # counts, so the off-diagonal blocks are rectangular.  The star is
        # REUSED UNCHANGED from the 1-D per-layer path (dimension-agnostic, and
        # already carrying M1's star-denominator guards).
        S = ifc[0]
        for i in range(nlay):
            S = _redheffer_star_rect(S, prop[i])
            S = _redheffer_star_rect(S, ifc[i + 1])
        S11, _S12, S21, _S22 = S
        S_above = S_below_bot = None
        if retain_internal:
            S_above = [None] * nlay
            S_above[0] = ifc[0]
            for i in range(1, nlay):
                S_above[i] = _redheffer_star_rect(
                    _redheffer_star_rect(S_above[i - 1], prop[i - 1]), ifc[i])
            S_below_bot = [None] * nlay
            S_below_bot[nlay - 1] = ifc[nlay]
            for i in range(nlay - 2, -1, -1):
                S_below_bot[i] = _redheffer_star_rect(
                    _redheffer_star_rect(ifc[i + 1], prop[i + 1]),
                    S_below_bot[i + 1])

        # ---- far field: TWO projectors, one per END grid --------------------
        ox = np.arange(-n_orders, n_orders + 1)
        oy = np.arange(-n_orders, n_orders + 1)
        order_x = np.tile(ox, len(oy))
        order_y = np.repeat(oy, len(ox))
        Nfo = len(order_x)
        P1s, P2s = _far_projector_2d(g_sup.bx, g_sup.by, ox, oy, a0x, a0y)
        P1t, P2t = _far_projector_2d(g_sub.bx, g_sub.by, ox, oy, a0x, a0y)
        Hsup = _pmm2d_project_orders(P1s, P2s, Wsup, g_sup.qq)
        Hsub = _pmm2d_project_orders(P1t, P2t, Wsub, g_sub.qq)
        kxv = kx0 + order_x * (wl / px)
        kyv = ky0 + order_y * (wl / py)
        kz_ref, kz_trn, kz_inc, safe_r, safe_t = _pmm2d_order_kz(
            eps_sup, eps_sub, kxv, kyv, kx0, ky0)
        delta = ((order_x == 0) & (order_y == 0)).astype(_C)
        p0 = int(np.where((order_x == 0) & (order_y == 0))[0][0])

        # the frame-anchor phase a SHEAR adds -- identical to the shared path
        # (it is a far-field correction on the TRANSMITTED orders and knows
        # nothing about the element grids)
        tphase = None
        if any(not _slant_is_zero(L.get("slant")) for L in self._layers):
            _shx = -sum(L.get("slant", (0.0, 0.0))[0] * L["thickness"]
                        for L in self._layers)
            _shy = -sum(L.get("slant", (0.0, 0.0))[1] * L["thickness"]
                        for L in self._layers)
            if _shx != 0.0 or _shy != 0.0:
                tphase = np.exp(-1j * k0 * (kxv * _shx + kyv * _shy))

        R_rows, T_rows, j_cols, cinc_cols = [], [], [], []
        amp = {k: np.zeros((2, Nfo), dtype=_C)
               for k in ("rx", "ry", "tx", "ty")}
        for col, (ex0, ey0) in enumerate(((1.0, 0.0), (0.0, 1.0))):
            long_inc = kx0 * ex0 + ky0 * ey0
            einc_sq = 1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0 else 1.0
            rhs = np.concatenate([ex0 * delta, ey0 * delta])
            cinc = _guarded_lstsq(
                Hsup, rhs, "PMM2DStackPure far-field Rayleigh projection")
            cinc_cols.append(cinc)
            r_ord = Hsup @ (S11 @ cinc)
            t_ord = Hsub @ (S21 @ cinc)
            rx, ry = r_ord[:Nfo], r_ord[Nfo:]
            tx, ty = t_ord[:Nfo], t_ord[Nfo:]
            if tphase is not None:
                tx, ty = tx * tphase, ty * tphase
            rz = -(kxv * rx + kyv * ry) / safe_r
            tz = -(kxv * tx + kyv * ty) / safe_t
            Re, Te = _project_efficiency(np, kz_ref, kz_trn, kz_inc,
                                         rx, ry, rz, tx, ty, tz, einc_sq)
            R_rows.append(Re)
            T_rows.append(Te)
            j_cols.append(np.stack([rx[p0], ry[p0]]))
            amp["rx"][col], amp["ry"][col] = rx, ry
            amp["tx"][col], amp["ty"][col] = tx, ty
        R_eff = np.stack(R_rows)
        T_eff = np.stack(T_rows)
        jmat = np.stack(j_cols, axis=1)
        orders2d = np.stack([order_x, order_y], axis=1)
        self._modal = dict(
            orders=orders2d.copy(), p0=p0,
            kx=kxv.copy(), ky=kyv.copy(),
            kz_ref=kz_ref.copy(), kz_trn=kz_trn.copy(),
            kz_inc=kz_inc, kx0=float(kx0), ky0=float(ky0),
            wavelength=float(wl), **amp)
        if retain_internal:
            # PER-LAYER field Grams: _flux_at integrates with the layer's OWN
            # blkdiag(G1_i, G2_i), and there is NO new assembly -- G_i IS
            # -Rmat_i and its FACTORS are already in the layer's StagGridOps,
            # so the flux quadrature is separable too.
            self._internal = dict(
                modes=modes, S_above=S_above, S_below_bot=S_below_bot,
                G=None, qq=None,
                G_of=[(g.V1, g.V2) for g in gof],
                qq_of=[g.qq for g in gof],
                k0=k0, any_oop=any_oop,
                cinc=np.stack(cinc_cols, axis=1),
                R_tot=R_eff.sum(axis=1), T_tot=T_eff.sum(axis=1))
        _warn_stag_closure(R_eff, T_eff, self._layers, eps_sup, eps_sub)
        if not jones:
            return orders2d, R_eff, T_eff
        return orders2d, R_eff, T_eff, jmat

    # -------------------------------------------------- internal observables
    def _internal_amplitudes(self):
        """Per-layer ``(c_fwd_top, c_bwd_bot)`` modal amplitudes from the
        retained partial cascades (both incident polarizations) -- the
        Hybrid/PMMStack recovery on the pure staggered cascade."""
        d = self._internal
        out = []
        k0 = d["k0"]
        for i, m in enumerate(d["modes"]):
            lam_f, lam_b, t = m[2], m[5], m[6]
            Sa = d["S_above"][i]
            Sb = d["S_below_bot"][i]
            # forward decay top -> bottom, backward decay bottom -> top.  For a
            # SYMMETRIC region lam_b = -lam_f and both reduce to the single
            # exp(-lam k0 t) the shipped in-plane path used.
            Xf = np.exp(-lam_f * k0 * t)
            Xb = np.exp(lam_b * k0 * t)
            n = lam_f.shape[0]
            A22, A21 = Sa[3], Sa[2]
            B11 = Sb[0]
            M = np.eye(n, dtype=_C) - (A22 * Xb[None, :]) @ (B11 * Xf[None, :])
            c_fwd = np.linalg.solve(M, A21 @ d["cinc"])
            c_bwd = B11 @ (Xf[:, None] * c_fwd)
            out.append((c_fwd, c_bwd))
        return out

    def _flux_at(self, i, z_frac, amps):
        """z-Poynting flux through a plane inside layer ``i``, integrated
        over the cell via the staggered block field Gram: with the modal
        coefficients ``E = [e1; e2]``, ``H = [h1; h2]`` (the Eq.25 dual
        pairing puts ``H2`` in the ``E1`` placement and ``H1`` in the ``E2``
        placement),

            P(z) ~ Re( h2^H G1 e1  -  h1^H G2 e2 )

        (probe-verified on the homogeneous-region modes: this form is equal
        for both propagating polarizations and ~1e-29 for evanescent modes,
        i.e. the physical z-flux up to a positive scale -- unlike the
        Hybrid's Rayleigh-basis convention, the Eq.25 dual carries no extra
        ``-i``).  Only ratios enter :meth:`layer_absorption`, so the overall
        scale cancels."""
        d = self._internal
        Wf, Vf, lam_f, Wb, Vb, lam_b, t = d["modes"][i]
        c_fwd, c_bwd = amps[i]
        k0 = d["k0"]
        # PER-LAYER grids: the flux bilinear form is the LAYER's own block
        # field Gram blkdiag(G1_i, G2_i) and its own qq_i, not one shared pair.
        # No new assembly -- G_i IS -Rmat_i, and the layer's StagGridOps
        # already holds its Kronecker FACTORS, so the quadrature is separable.
        per_layer = d.get("G_of") is not None
        qq = d["qq_of"][i] if per_layer else d["qq"]
        # ``c_fwd`` is referenced to the layer TOP and ``c_bwd`` to its BOTTOM.
        # For a SYMMETRIC region (Wb = Wf, Vb = -Vf, lam_b = -lam_f) this is
        # the shipped expression term for term.
        P = np.exp(-lam_f * k0 * (z_frac * t))[:, None]
        Q = np.exp(lam_b * k0 * ((1.0 - z_frac) * t))[:, None]
        E = Wf @ (P * c_fwd) + Wb @ (Q * c_bwd)
        H = Vf @ (P * c_fwd) + Vb @ (Q * c_bwd)
        if per_layer:
            V1, V2 = d["G_of"][i]
            G1E = _stag_kron_apply(V1[0], V1[1], E[:qq])
            G2E = _stag_kron_apply(V2[0], V2[1], E[qq:])
        else:
            G = d["G"]
            G1E = G[:qq, :qq] @ E[:qq]
            G2E = G[qq:, qq:] @ E[qq:]
        val = (np.sum(np.conj(H[qq:]) * G1E, axis=0)
               - np.sum(np.conj(H[:qq]) * G2E, axis=0))
        return np.real(val)

    def layer_absorption(self):
        """Per-layer absorbed power fraction ``(n_layers, 2)`` (one column
        per incident lab polarization ``E_x``/``E_y``) -- the
        :meth:`PMM2DStackHybrid.layer_absorption` contract on the PURE
        (no-floor) engine (AUDIT_DYNAMETA_CONSUMER_API_GAPS C3), from the
        z-flux difference across each layer in the staggered modal basis
        (block-field-Gram quadrature).  Requires that the MOST RECENT
        :meth:`solve` used ``retain_internal=True``.  The cross-machinery
        closure ``sum_i A_i == 1 - sum R - sum T`` is the honest check
        (internal Gram flux vs the Rayleigh far field)."""
        d = self._internal
        if d is None:
            raise ValueError(
                "PMM2DStackPure.layer_absorption: no internal data retained; "
                "the MOST RECENT solve must use solve(retain_internal=True) "
                "(any re-solve invalidates previously retained internals).")
        amps = self._internal_amplitudes()
        nlay = len(d["modes"])
        F_top = np.array([self._flux_at(i, 0.0, amps) for i in range(nlay)])
        F_bot = np.array([self._flux_at(i, 1.0, amps) for i in range(nlay)])
        # normalize by the incident flux WITHOUT an absolute flux gauge: the
        # net downward flux at the top of layer 0 is (1 - R) * F_inc (the
        # Hybrid's calibration -- exact by flux conservation in the lossless
        # superstrate half-space).
        one_minus_R = 1.0 - d["R_tot"]
        F_inc = np.where(np.abs(one_minus_R) > 1e-15,
                         F_top[0] / one_minus_R, np.inf)
        return (F_top - F_bot) / F_inc[None, :]
