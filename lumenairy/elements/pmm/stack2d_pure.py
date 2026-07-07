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
patterns and ISOTROPIC scalar permittivity on one shared square grid.
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
Full anisotropic tensors and out-of-plane coupling are staged extensions (the
staggered assembly already carries the div-D ``E_z`` Schur term).
Uniform layers route through the shared eps-free geometric eig
(:func:`~lumenairy.elements.pmm.twod_staggered._homog_region_modes`) -- all
uniform regions share the SAME eigenvectors, so a uniform<->uniform interface is
``a = W0^-1 W0 = I`` (perfectly conditioned) and costs no eig.

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

from ._core import _interface_smatrix, _propagation_smatrix, _redheffer_star
from .twod_staggered import (
    _C,
    Granet2DTransverseE,
    _far_projector_2d,
    _homog_geom_cache,
    _homog_region_modes,
    _kz_forward2,
    _region_modes,
)

__all__ = ["PMM2DStackPure"]


class PMM2DStackPure:
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
    """

    def __init__(self, period_x, period_y=None, *, n_superstrate=1.0,
                 n_substrate=1.0, n_modes=8, degree=None, n_orders=7):
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
        self._layers = []          # dicts: kind, thickness, eps | eps_cell
        self._grid = None          # common (Nx, Ny) set by the first patterned layer
        self._src = None

    # ------------------------------------------------------------------ build
    def add_layer(self, thickness, *, eps=None, eps_cell=None):
        """Append a layer.  Pass exactly ONE of ``eps`` (uniform isotropic) or
        ``eps_cell`` (a SQUARE ``(Nx, Ny)`` scalar permittivity grid, walls on
        the segment boundaries).  All patterned layers must share one common
        ``(Nx, Ny)`` grid (union-grid constraint)."""
        if (eps is None) == (eps_cell is None):
            raise ValueError(
                "PMM2DStackPure.add_layer: pass exactly ONE of eps (uniform) or "
                "eps_cell (patterned).")
        t = float(thickness)
        if not t > 0:
            raise ValueError("PMM2DStackPure.add_layer: thickness must be > 0.")
        if eps is not None:
            self._layers.append(dict(kind="uniform", thickness=t,
                                     eps=_C(eps)))
            return self
        cell = np.asarray(eps_cell, dtype=_C)
        if cell.ndim != 2:
            raise ValueError(
                f"PMM2DStackPure.add_layer: eps_cell must be a 2-D (Nx, Ny) "
                f"array, got shape {cell.shape}.")
        if cell.shape[0] != cell.shape[1]:
            raise ValueError(
                f"PMM2DStackPure.add_layer: eps_cell must be SQUARE (Nx == Ny; "
                f"the staggered tensor-product basis requires "
                f"Nx*(M-1) == Ny*(M-1)), got {cell.shape}.  Pad the uniform axis "
                f"into equal segments (e.g. tile a (2, 1) cell to (2, 2)).")
        if self._grid is None:
            self._grid = cell.shape
        elif cell.shape != self._grid:
            raise ValueError(
                f"PMM2DStackPure.add_layer: all patterned layers must share ONE "
                f"common (Nx, Ny) grid (the union-grid constraint of the pure "
                f"staggered cascade); got {cell.shape} after {self._grid}.  "
                f"Re-express every pattern on a common grid, or use "
                f"PMM2DStackHybrid (no union-grid constraint).")
        self._layers.append(dict(kind="patterned", thickness=t, eps_cell=cell))
        return self

    def set_source(self, wavelength, *, theta=0.0, phi=0.0):
        """Set the incident plane wave: vacuum ``wavelength`` (m), polar
        ``theta`` and azimuth ``phi`` (radians)."""
        self._src = dict(wl=float(wavelength), theta=float(theta),
                         phi=float(phi))
        return self

    # ------------------------------------------------------------------ solve
    def solve(self, *, jones=True):
        """Cascade the stack and return the diffraction efficiencies.

        Returns ``(orders, R, T, jones)`` (default) or ``(orders, R, T)`` when
        ``jones=False``.  ``orders`` is ``(Nfo, 2)`` ``(m, n)`` pairs; ``R``/``T``
        are ``(2, Nfo)`` real efficiencies (row 0 = incident ``Ex``, row 1 =
        incident ``Ey``); ``jones`` is the ``(2, 2)`` order-0 reflection Jones."""
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
        wl = _grazing_safe_wavelength(wl, _kx0n, _ky0n, _mx, _my, px, py,
                                      [eps_sup, eps_sub])
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
        del sol_h
        Wsup, Vsup, _ls = _homog_region_modes(geom, eps_sup)
        Wsub, Vsub, _lb = _homog_region_modes(geom, eps_sub)

        # Per-layer modes: uniform -> shared geom (cheap); patterned -> its own
        # staggered eig, deduped across byte-identical cells (a DBR eigs each
        # distinct layer once).
        modes = []
        eig_cache = {}
        for L in self._layers:
            if L["kind"] == "uniform":
                W, V, lam = _homog_region_modes(geom, L["eps"])
            else:
                key = L["eps_cell"].tobytes()
                cached = eig_cache.get(key)
                if cached is None:
                    sol = Granet2DTransverseE(px, py, Nx, Ny, M, L["eps_cell"],
                                              alpha0x=a0x, alpha0y=a0y, k0=k0)
                    Wl, Vl, lam_l, _g2 = _region_modes(sol)
                    cached = (Wl, Vl, lam_l)
                    eig_cache[key] = cached
                W, V, lam = cached
            modes.append((W, V, lam, L["thickness"]))

        # SQUARE Redheffer cascade: sup | (interface, propagate)* | sub.
        S = _interface_smatrix(Wsup, Vsup, modes[0][0], modes[0][1])
        nlay = len(modes)
        for i, (W, V, lam, t) in enumerate(modes):
            S = _redheffer_star(S, _propagation_smatrix(lam, k0 * t))
            if i < nlay - 1:
                S = _redheffer_star(
                    S, _interface_smatrix(W, V, modes[i + 1][0], modes[i + 1][1]))
        S = _redheffer_star(
            S, _interface_smatrix(modes[-1][0], modes[-1][1], Wsub, Vsub))
        S11, _S12, S21, _S22 = S

        # FORWARD-only far-field Fourier->Rayleigh projection (once).
        ox = np.arange(-n_orders, n_orders + 1)
        oy = np.arange(-n_orders, n_orders + 1)
        order_x = np.tile(ox, len(oy))
        order_y = np.repeat(oy, len(ox))
        Nfo = len(order_x)
        P1, P2 = _far_projector_2d(bx, by, ox, oy, a0x, a0y)
        qq = (Nx * (M - 1)) * (Ny * (M - 1))

        def _proj(Wmodes):
            top = P1 @ Wmodes[:qq, :]
            bot = P2 @ Wmodes[qq:, :]
            return np.concatenate([top, bot], axis=0)

        Hsup, Hsub = _proj(Wsup), _proj(Wsub)
        kxv = kx0 + order_x * (wl / px)
        kyv = ky0 + order_y * (wl / py)
        kz_ref = _kz_forward2(eps_sup, kxv, kyv)
        kz_trn = _kz_forward2(eps_sub, kxv, kyv)
        kz_inc = float(np.real(_kz_forward2(eps_sup, kx0, ky0)))
        safe_r = np.where(np.abs(kz_ref) < 1e-12, 1.0, kz_ref)
        safe_t = np.where(np.abs(kz_trn) < 1e-12, 1.0, kz_trn)
        delta = ((order_x == 0) & (order_y == 0)).astype(_C)
        p0 = int(np.where((order_x == 0) & (order_y == 0))[0][0])

        R_rows, T_rows, j_cols = [], [], []
        for (ex0, ey0) in ((1.0, 0.0), (0.0, 1.0)):
            long_inc = kx0 * ex0 + ky0 * ey0
            einc_sq = 1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0 else 1.0
            rhs = np.concatenate([ex0 * delta, ey0 * delta])
            cinc, *_ = np.linalg.lstsq(Hsup, rhs, rcond=None)
            r_ord = Hsup @ (S11 @ cinc)
            t_ord = Hsub @ (S21 @ cinc)
            rx, ry = r_ord[:Nfo], r_ord[Nfo:]
            tx, ty = t_ord[:Nfo], t_ord[Nfo:]
            rz = -(kxv * rx + kyv * ry) / safe_r
            tz = -(kxv * tx + kyv * ty) / safe_t
            Re = np.real(kz_ref / kz_inc) * (
                np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2) / einc_sq
            Te = np.real(kz_trn / kz_inc) * (
                np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2) / einc_sq
            R_rows.append(np.where(np.real(kz_ref) > 0, np.real(Re), 0.0))
            T_rows.append(np.where(np.real(kz_trn) > 0, np.real(Te), 0.0))
            j_cols.append(np.stack([rx[p0], ry[p0]]))
        R_eff = np.stack(R_rows)
        T_eff = np.stack(T_rows)
        jmat = np.stack(j_cols, axis=1)           # jmat[out_comp, in_pol]
        orders2d = np.stack([order_x, order_y], axis=1)
        if not jones:
            return orders2d, R_eff, T_eff
        return orders2d, R_eff, T_eff, jmat
