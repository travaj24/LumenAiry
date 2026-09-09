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
* **UNION-GRID constraint.**  All patterned layers share ONE common SQUARE
  ``(Nx, Ny)`` segmentation (the 1-D :class:`PMMStack` union grid lifted to 2-D):
  the modal matrices must be conformable across interfaces.  The hybrid decouples
  layers through the Fourier projection, so it has NO union-grid constraint (walls
  may differ per layer).  Re-express each layer's pattern on a common grid.

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
Tapered (z-staircase) helpers remain hybrid-only: use :class:`PMM2DStackHybrid`.
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
    _project_efficiency,
    _propagation_smatrix_general,
    _symmetry_on,
)
from ._core import (
    PerOrderAmplitudesMixin,
    _guarded_lstsq,
    _interface_smatrix,
    _propagation_smatrix,
    _redheffer_star,
)
from .twod_staggered import (
    _C,
    Granet2DTransverseE,
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
    _tile_needs_oop,
    _validate_stag_cell,
    _validate_stag_mu,
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


def _spec_is_lossless(spec, uniform):
    """True when a layer's ``eps`` / ``mu`` specification absorbs nothing:
    exactly real if scalar, Hermitian if a ``(3, 3)`` tensor (per cell)."""
    spec = np.asarray(spec, dtype=_C)
    tensor = spec.ndim == 4 or (uniform and spec.ndim == 2)
    if tensor:
        return _tensor_is_hermitian(spec)
    return not bool(np.any(np.imag(spec) != 0.0))


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
                 mu_superstrate=None, mu_substrate=None, symmetry="auto"):
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
        self._layers = []          # dicts: kind, thickness, eps | eps_cell
        self._grid = None          # common (Nx, Ny) set by the first patterned layer
        self._src = None
        self._modal = None         # per-order amplitudes of the last solve (B)
        self._internal = None      # partial cascades for layer_absorption (C3)

    # ------------------------------------------------------------------ build
    def add_layer(self, thickness, *, eps=None, eps_cell=None, mu=None,
                  mu_cell=None):
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
        boundaries).  All patterned layers must share one common ``(Nx, Ny)``
        grid (union-grid constraint).

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
        (the first-order generator has no permeability blocks)."""
        self._modal = None      # geometry change supersedes retained amplitudes
        self._internal = None
        if (eps is None) == (eps_cell is None):
            raise ValueError(
                "PMM2DStackPure.add_layer: pass exactly ONE of eps (uniform) or "
                "eps_cell (patterned).")
        t = float(thickness)
        if not t > 0:
            raise ValueError("PMM2DStackPure.add_layer: thickness must be > 0.")
        if mu is not None or mu_cell is not None:
            return self._add_magnetic_layer(t, eps, eps_cell, mu, mu_cell)
        if eps is not None:
            e = np.asarray(eps, dtype=_C)
            if e.ndim == 0:
                self._layers.append(dict(kind="uniform", thickness=t,
                                         eps=_C(eps)))
                return self
            if e.shape != (3, 3):
                raise ValueError(
                    f"PMM2DStackPure.add_layer: a uniform eps must be a scalar "
                    f"or a (3, 3) block-form tensor, got shape {e.shape}.  A "
                    f"PATTERNED tensor layer goes through eps_cell "
                    f"((Nx, Ny, 3, 3)).")
            _tile_needs_oop("PMM2DStackPure.add_layer", e[None, None])
            self._layers.append(dict(kind="uniform_tensor", thickness=t,
                                     eps33=e))
            return self
        cell = _validate_stag_cell("PMM2DStackPure.add_layer", eps_cell)
        grid = cell.shape[:2]
        if self._grid is None:
            self._grid = grid
        elif grid != self._grid:
            raise ValueError(
                f"PMM2DStackPure.add_layer: all patterned layers must share ONE "
                f"common (Nx, Ny) grid (the union-grid constraint of the pure "
                f"staggered cascade); got {grid} after {self._grid}.  "
                f"Re-express every pattern on a common grid, or use "
                f"PMM2DStackHybrid (no union-grid constraint).")
        self._layers.append(dict(kind="patterned", thickness=t, eps_cell=cell))
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
            if uni:
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
                                 mu_uniform=mu_uni))
        return self

    def set_source(self, wavelength, *, theta=0.0, phi=0.0):
        """Set the incident plane wave: vacuum ``wavelength`` (m), polar
        ``theta`` and azimuth ``phi`` (radians)."""
        self._modal = None      # source change supersedes retained amplitudes
        self._internal = None
        self._src = dict(wl=float(wavelength), theta=float(theta),
                         phi=float(phi))
        return self

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
        # Multi-patterned (A|B) cascades are fully supported: the historical
        # A|B energy blow-up was the far-field projection-kernel order MIRROR
        # (fixed in twod_staggered._stag_fourier_projection), not an
        # interface/mode-sorting defect -- see the module docstring (Scope) and
        # test_v5_21_pmm2d_staggered_oblique.test_stack_pure_multilayer_ab_vs_1d.
        wl = self._src["wl"]
        theta, phi = self._src["theta"], self._src["phi"]
        px, py = self.period_x, self.period_y
        eps_sup = _C(self.n_sup) ** 2
        eps_sub = _C(self.n_sub) ** 2
        Nx, Ny = self._grid if self._grid is not None else (2, 2)
        M = self.M
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
                # merge repair 2026-09-10: the magnetic layer record carries
                # ``eps`` (scalar / (3,3) uniform / (Nx,Ny) / (Nx,Ny,3,3)),
                # not ``eps_cell``.  Permittivity only for now -- a magnetic
                # cut-off sits at Re(eps*mu); the eps*mu products are the
                # documented follow-up (BUILD_PMM2D_STAGGERED_MAGNETIC open 3).
                _e = np.asarray(_L["eps"], dtype=_C)
                if _L["eps_uniform"]:
                    _eps_src.append(np.diag(_e) if _e.ndim == 2 else _e)
                elif _e.ndim == 4:
                    _eps_src.append(_e[..., [0, 1, 2], [0, 1, 2]])
                else:
                    _eps_src.append(_e)
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
            if L["kind"] == "uniform":
                W, V, lam = _homog_region_modes(geom, L["eps"])
                six = _modes_as_general(W, V, lam)
            else:
                mcell = None
                if L["kind"] == "uniform_tensor":
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
                key = ((cell.shape, cell.tobytes()) if mcell is None else
                       (cell.shape, cell.tobytes(), mcell.shape,
                        mcell.tobytes()))
                cached = eig_cache.get(key)
                if cached is None:
                    sol = Granet2DTransverseE(px, py, Nx, Ny, M, cell,
                                              alpha0x=a0x, alpha0y=a0y, k0=k0,
                                              mu_cell=mcell)
                    if sol.offplane:
                        cached = _region_modes_oop(sol,
                                                   symmetry=self.symmetry)
                    else:
                        Wl, Vl, lam_l, _g2 = _region_modes(sol)
                        cached = _modes_as_general(Wl, Vl, lam_l)
                    eig_cache[key] = cached
                six = cached
                any_oop = any_oop or (len(cell.shape) == 4
                                      and _tile_needs_oop(
                                          "PMM2DStackPure.solve", cell))
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
        k0, qq = d["k0"], d["qq"]
        # ``c_fwd`` is referenced to the layer TOP and ``c_bwd`` to its BOTTOM.
        # For a SYMMETRIC region (Wb = Wf, Vb = -Vf, lam_b = -lam_f) this is
        # the shipped expression term for term.
        P = np.exp(-lam_f * k0 * (z_frac * t))[:, None]
        Q = np.exp(lam_b * k0 * ((1.0 - z_frac) * t))[:, None]
        E = Wf @ (P * c_fwd) + Wb @ (Q * c_bwd)
        H = Vf @ (P * c_fwd) + Vb @ (Q * c_bwd)
        G = d["G"]
        G1, G2 = G[:qq, :qq], G[qq:, qq:]
        val = (np.sum(np.conj(H[qq:]) * (G1 @ E[:qq]), axis=0)
               - np.sum(np.conj(H[:qq]) * (G2 @ E[qq:]), axis=0))
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
