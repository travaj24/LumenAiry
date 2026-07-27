"""RCWA unified multi-layer API: RCWAStack / RCWAResult (+ caching,
Jones bridge).  Handles both 1-D and 2-D layers."""
from __future__ import annotations

import copy as _copy
import os
import warnings
from typing import List, Optional

import numpy as np

from ...backend import (
    array_namespace,
    backend_name,
    is_jax_array,
    to_numpy,
)
from ._core import (
    _C,
    _blas_threads_quiet,
    _cached_homogeneous_eigenmodes,
    _cell_lossless,
    _check_energy,
    _EnergyError,
    _forward_flux_kz,
    _grazing_safe_wavelength,
    _homogeneous_eigenmodes,
    _interface_smatrix,
    _interface_smatrix_general,
    _layer_eigenmodes,
    _layer_eigenmodes_tensor,
    _max_aligned_delta,
    _modes_to_M,
    _order_key,
    _project_efficiency,
    _propagation_smatrix_general,
    _propagation_star,
    _propagation_star_general,
    _rcwa_convergence_stack,
    _rcwa_xp,
    _readonly,
    _recentering_phase,
    _redheffer_star,
    _require_jax_x64,
    _require_propagating_incidence,
    _scalar_PQ,
    _sqrt_decay,
    _sqrt_forward,
    _symmetric_cascade_rt,
    _symmetry_on,
    _tensor_offplane_present,
    _tensor_PQ,
    _validate_cell_sampling,
    _validate_geometry,
    _validate_shapes,
    _with_blas_limit,
)
from .oned import (
    _resolve_incidence,
)
from .twod import (
    _analytic_convolutions_2d,
    _eps_convolution_2d,
    _harmonic_orders_2d,
    _li_convolutions_2d,
)


class RCWAYAverageWarning(UserWarning):
    """A 1-D :class:`RCWAStack` is solving the y-AVERAGE of a y-VARYING cell.

    v5.31 (audit W4-3, closing the deferred 1-D-stack half of audit M8).  A 1-D
    (mono-periodic) stack carries only the ``n = 0`` y-harmonic, so a 2-D cell
    handed to it contributes only its ``c_{k,0}`` Fourier coefficients -- the
    y-AVERAGED permittivity.  That is the stack's long-standing, deliberate
    contract for a y-INVARIANT 2-D cell (an x-only grating written on a 2-D
    grid: the average IS the structure, and the answer is exact).  For a
    genuinely y-VARYING cell it silently solves a DIFFERENT structure.

    Measured (24x24 square-pillar cell, n_lo/n_hi = 1.5/2.5, period 1 um,
    lambda = 633 nm, theta = 0.1 rad, n_orders = 5): the 1-D stack returned
    ``sum(R) = 0.244615594382`` -- BIT-identical to explicitly pre-averaging the
    cell along y -- against the y-resolved 2-D stack's ``0.183045443696``, a
    33.6% error, with the energy closure equally clean in both solves so no
    tripwire can see it.

    This is a diagnostic, NOT a rejection: the y-average is the documented
    behaviour and an existing pin relies on feeding 2-D-shaped cells to 1-D
    stacks.  Its own category is provided so callers can silence exactly this
    (``filterwarnings('ignore', category=RCWAYAverageWarning)``) or promote
    exactly this to an error, without touching the surrounding
    ``UserWarning``-based diagnostics.
    """


def _warn_if_y_averaged(fn_name, cell, n_orders_y, *, strict_y, stacklevel=3):
    """Emit :class:`RCWAYAverageWarning` when a 1-D stack is about to y-average
    a genuinely y-VARYING 2-D cell.

    Complements :func:`~lumenairy.elements.rcwa._core._validate_cell_sampling`,
    whose ``strict_y=True`` branch RAISES on this input for the explicitly-2-D
    entry points.  The stack passes ``strict_y=not self.is_1d``, so a 1-D stack
    (``noy = 0`` sentinel) skips that check entirely -- the y-average is its
    contract.  This is the missing diagnostic for that skipped case.

    The y-variance test is deliberately the SAME formulation as the
    ``strict_y`` branch's -- per-component peak-to-peak along the y axis (real
    plus imaginary), on concrete arrays only, since a traced JAX cell cannot be
    inspected -- so the two can never disagree about whether a cell is
    y-varying.  ``tests/unit/test_niche_audit_w4_gaps.py`` pins that agreement.

    No-ops (returns False) unless ``strict_y`` is False AND ``n_orders_y == 0``
    AND the cell has more than one y sample AND the cell actually varies along
    y.  Returns True when it warned.

    ``stacklevel`` is the number of frames from this function's ``warn`` call
    out to the USER's line, so the diagnostic points at the ``add_layer`` /
    ``solve_vs_wavelength`` call the user wrote rather than at library
    internals: 3 from ``add_layer`` (which calls this directly), 5 from
    ``_materialized_layers`` (measured chain: user -> ``solve_vs_wavelength``
    -> ``_solve_one`` -> ``_materialized_layers``).  A too-large value simply
    clamps at the outermost frame, so the threaded sweep degrades gracefully.
    """
    if strict_y or int(n_orders_y) != 0:
        return False
    if is_jax_array(cell):
        return False                          # tracer: not inspectable
    Sx, Sy = int(cell.shape[0]), int(cell.shape[1])
    if Sy <= 1:
        return False                          # already a 1-D cell, nothing lost
    xpc = array_namespace(cell)
    arr = xpc.asarray(cell)
    spatial = arr.reshape(Sx, Sy, -1)
    yspread = xpc.ptp(spatial.real, axis=1)
    if xpc.iscomplexobj(spatial):
        yspread = yspread + xpc.ptp(spatial.imag, axis=1)
    ysp = float(xpc.max(yspread))
    if ysp == 0.0:
        return False                          # y-INVARIANT: the average is exact
    warnings.warn(
        f"{fn_name}: this stack is 1-D (mono-periodic -- no period_y / "
        f"n_orders_y), so the {(Sx, Sy)} cell enters only through its "
        f"y-AVERAGED permittivity, but the cell VARIES along y (max per-row "
        f"spread {ysp:.6g}).  The solve returns the y-averaged structure's "
        f"answer, which is bit-identical to pre-averaging the cell yourself "
        f"and can be tens of percent from the y-resolved result while the "
        f"energy closure stays clean (so no tripwire fires).  This is exact "
        f"and intended for a y-INVARIANT 2-D cell; if the y structure is "
        f"real, build a 2-D stack (pass period_y AND n_orders_y >= 1).",
        RCWAYAverageWarning, stacklevel=stacklevel)
    return True


def _layer_offplane_or_traced(data):
    """True if a ``(3, 3)`` tensor cell has OUT-OF-PLANE coupling.  A concrete
    array is inspected; a TRACED jax array (under ``jax.grad`` / ``jit``) cannot
    be materialised and is treated as out-of-plane -- routed to the general
    cascade, which is EXACT for in-plane tensors too.  Keeping a traced layer on
    the same (general) branch its concrete forward would take is what makes the
    differentiable out-of-plane stack gradient correct (a traced OOP layer left
    on the in-plane branch silently drops the z-coupling)."""
    if _tensor_offplane_present(data):
        return True
    if is_jax_array(data):
        try:
            return _tensor_offplane_present(np.asarray(to_numpy(data)))
        except Exception:
            return True                       # tracer -> general path
    return False


def rcwa_convergence(solver, *, order_params=("n_orders",), bump=4, atol=1e-3,
                     warn=True, **kwargs):
    """Run an RCWA ``solver`` at the requested harmonic count AND a higher one,
    and report whether the answer has converged in the retained order count.

    A too-low truncation can manufacture a *plausible but wrong* result -- most
    dangerously a spurious deep reflection null at a sharp / high-Q resonance --
    so this solves twice (each name in ``order_params`` bumped by ``bump``) and
    compares the per-order efficiencies, warning if the largest change exceeds
    ``atol``.  Cheap insurance (one extra solve) against silently
    under-resolved physics; for a converged *value* (not just a check) see
    :func:`rcwa_extrapolate`.

    Parameters
    ----------
    solver : callable or RCWAStack
        An RCWA efficiency entry point returning ``(orders, R, T, ...)`` --
        e.g. :func:`rcwa_efficiency_1d`, :func:`rcwa_efficiency_2d`,
        :func:`rcwa_jones_1d` -- OR a configured :class:`RCWAStack` (with its
        source + layers already set).  For a stack the truncation it is bumped
        is its ``n_orders`` (and ``n_orders_y`` for a 2-D stack); ``kwargs`` /
        ``order_params`` are ignored and the two ``RCWAResult`` solves are
        compared on their per-order efficiencies (the HIGHER-resolution
        ``RCWAResult`` is returned).
    order_params : tuple of str, optional
        The harmonic-count keyword(s) to bump (callable ``solver`` only).
        ``("n_orders",)`` (default) for the 1-D / Jones-1-D solvers;
        ``("n_orders_x", "n_orders_y")`` for 2-D.
    bump : int, optional
        Increment added to each order parameter for the high-resolution solve
        (default 4).
    atol : float, optional
        Convergence tolerance on the largest per-order efficiency change
        (default ``1e-3``).
    warn : bool, optional
        Emit a ``UserWarning`` when not converged (default ``True``).
    **kwargs
        Passed verbatim to a callable ``solver`` (must include the
        ``order_params`` and the geometry); ignored for an :class:`RCWAStack`.

    Returns
    -------
    result : tuple or RCWAResult
        The HIGHER-resolution solve (the more trustworthy of the two) -- the
        ``solver(**kwargs_bumped)`` tuple for a callable, or the bumped
        :class:`RCWAResult` for a stack.
    report : dict
        ``converged`` (bool), ``delta`` (max per-order efficiency change),
        ``delta_sum_R`` / ``delta_sum_T`` (change in total R / T),
        ``n_orders_low`` / ``n_orders_high`` (the two truncations).
    """
    if isinstance(solver, RCWAStack):
        return _rcwa_convergence_stack(solver, bump=bump, atol=atol, warn=warn)
    for p in order_params:
        if p not in kwargs:
            raise ValueError(
                f"rcwa_convergence: order parameter {p!r} not found in kwargs; "
                f"pass it (and set order_params to the solver's harmonic-count "
                f"argument names).")
    low = solver(**kwargs)
    hi_kwargs = dict(kwargs)
    for p in order_params:
        hi_kwargs[p] = int(kwargs[p]) + int(bump)
    high = solver(**hi_kwargs)

    o_lo, R_lo, T_lo = low[0], low[1], low[2]
    o_hi, R_hi, T_hi = high[0], high[1], high[2]
    delta = max(_max_aligned_delta(o_lo, R_lo, o_hi, R_hi),
                _max_aligned_delta(o_lo, T_lo, o_hi, T_hi))
    dsR = abs(float(np.sum(to_numpy(R_lo))) - float(np.sum(to_numpy(R_hi))))
    dsT = abs(float(np.sum(to_numpy(T_lo))) - float(np.sum(to_numpy(T_hi))))
    converged = delta <= atol
    report = dict(converged=converged, delta=delta, delta_sum_R=dsR,
                  delta_sum_T=dsT,
                  n_orders_low={p: int(kwargs[p]) for p in order_params},
                  n_orders_high={p: int(hi_kwargs[p]) for p in order_params})
    if warn and not converged:
        import warnings
        warnings.warn(
            f"rcwa_convergence: NOT converged at {report['n_orders_low']} -- the "
            f"per-order efficiency changed by {delta:.2e} (> atol={atol:.1e}) "
            f"going to {report['n_orders_high']}; the lower-order result may be "
            f"unreliable (a too-low truncation can fabricate a spurious "
            f"resonance/null). Increase the order count.", stacklevel=2)
    return high, report



class RCWAResult:
    """Result of an :class:`RCWAStack` solve.

    Accessors
    ---------
    efficiencies() -> (orders, R, T)
        ``R``, ``T`` are ``(2, N)`` real arrays: row 0 is the response to an
        incident ``E_x`` plane wave, row 1 to incident ``E_y``; ``orders`` is
        ``(N, 2)`` (2-D) or ``(N,)`` (1-D).
    absorptance() -> (2,) ndarray
        ``1 - sum(R) - sum(T)`` per incident polarization (``>= 0`` for
        passive media -- the loss-sign bridge guarantees the sign).
    jones_reflection() / jones_transmission() -> (2, 2) complex
        Zeroth-order Jones matrices (columns = incident ``E_x`` / ``E_y``,
        rows = ``[E_x; E_y]``).
    apply_reflection(jones_field) -> JonesField
        Apply the zeroth-order Jones reflection to a
        :class:`~lumenairy.elements.polarization.JonesField` -- the bridge
        from a rigorous metasurface reflection into the polarization
        pipeline.
    to_jones_field(nx, ny, dx, ..., order=None) -> JonesField
        Build a JonesField from one diffraction order (``order=None`` is the
        uniform specular field; a non-zero order is a tilted carrier).
    per_order_amplitudes(port) -> dict
        Per-order complex tangential amplitudes ``(2, N)`` + transverse
        k-vectors -- the modal data the solver computes.
    to_multiorder_field(nx, ny, dx, ...) -> JonesField
        Reconstruct the full diffracted field as a superposition of the
        propagating orders (the metasurface-deflector / metalens bridge).
    """

    def __init__(self, orders, R, T, jones_reflection, jones_transmission,
                 modal=None):
        self.orders = orders
        self._R = R
        self._T = T
        self._Jr = jones_reflection
        self._Jt = jones_transmission
        self._modal = modal   # per-order amplitudes + k-vectors (or None)

    def efficiencies(self):
        return self.orders, self._R, self._T

    # -- per-order modal access + multi-order field reconstruction ---------

    def _require_modal(self):
        if self._modal is None:
            raise ValueError(
                "RCWAResult: per-order modal data is unavailable on this "
                "result (it is retained only by RCWAStack.solve()).")
        return self._modal

    def _order_index(self, order):
        """Flat index of a diffraction order: ``m`` (1-D) or ``(m, n)``
        (2-D)."""
        o2 = self._require_modal()["orders2d"]
        from ...backend import to_numpy
        o2 = to_numpy(o2)
        if np.ndim(order) == 0:                      # 1-D: integer order m
            hit = np.where((o2[:, 0] == int(order)) & (o2[:, 1] == 0))[0]
        else:
            m, n = order
            hit = np.where((o2[:, 0] == int(m)) & (o2[:, 1] == int(n)))[0]
        if hit.size == 0:
            raise ValueError(
                f"RCWAResult: order {order!r} is outside the retained range; "
                f"increase n_orders.")
        return int(hit[0])

    def per_order_amplitudes(self, port="reflection"):
        """Per-order complex tangential field amplitudes (PUBLIC ``exp(-iwt)``
        convention) and transverse k-vectors -- the data the multi-order field
        bridge is built on.

        Returns a dict with ``Ex`` / ``Ey`` each ``(2, N)`` (row 0 = response
        to incident ``E_x``, row 1 to incident ``E_y``), the per-order
        ``kx`` / ``ky`` / ``kz`` normalised by ``k0 = 2*pi/wavelength``, the
        ``orders``, and the ``wavelength``.  ``port`` selects the reflection or
        transmission side.

        Note: these are the raw TANGENTIAL field amplitudes, so
        ``|Ex_m|^2 + |Ey_m|^2`` is NOT the order's efficiency -- recovering the
        efficiency needs the Poynting flux weight ``Re(kz_m/kz_inc)``, the
        longitudinal component ``|az_m|^2`` with ``az = -(kx Ex + ky Ey)/kz``,
        and the incident ``|E|^2``.  :meth:`to_multiorder_field` (with the
        default ``normalize='power'``) applies all three for you.
        """
        if port not in ("reflection", "transmission"):
            raise ValueError(
                f"RCWAResult.per_order_amplitudes: port must be 'reflection' "
                f"or 'transmission', got {port!r}.")
        from ...backend import to_numpy
        m = self._require_modal()
        ex, ey = ("rx", "ry") if port == "reflection" else ("tx", "ty")
        kz = m["kz_ref"] if port == "reflection" else m["kz_trn"]
        # ``kz`` comes straight out of the module-level homogeneous-mode cache,
        # which now hands its values out READ-ONLY (audit W7-B).  Copy it here
        # so the public dict keeps its writable-array contract while a caller's
        # in-place edit can no longer poison the cache for later solves.
        return dict(orders=self.orders, Ex=to_numpy(m[ex]), Ey=to_numpy(m[ey]),
                    kx=to_numpy(m["kx"]), ky=to_numpy(m["ky"]),
                    kz=np.array(to_numpy(kz), copy=True),
                    wavelength=m["wavelength"],
                    # incidence terms for the power-normalized field bridge
                    # (jones_field_from_orders): the transmission-port kz is the
                    # SUBSTRATE kz, so the flux weight Re(kz_m/kz_inc) needs the
                    # INCIDENT-medium specular kz carried separately (S5-5).
                    kz_inc=m["kz_inc"], kx0=m["kx0"], ky0=m["ky0"])

    def absorptance(self):
        return 1.0 - self._R.sum(axis=1) - self._T.sum(axis=1)

    def jones_reflection(self):
        return self._Jr

    def jones_transmission(self):
        return self._Jt

    def apply_reflection(self, jones_field):
        # apply_jones_matrix transforms its field IN PLACE; operate on a copy
        # so the caller's incident field is preserved.  Note: this carries
        # only the zeroth-order (specular) 2x2 Jones -- for a strongly
        # diffracting cell most power is in non-zero orders, so the returned
        # field is the specular component only (a single plane wave).
        #
        # This is the REFLECTION-side convenience; a TRANSMISSIVE metasurface
        # (the majority use) wants :meth:`apply_transmission` instead (audit
        # S5-11 -- the reflection port is the wrong observable for a transmit
        # element).
        from ..polarization import apply_jones_matrix
        return apply_jones_matrix(jones_field.copy(), self._Jr)

    def apply_transmission(self, jones_field):
        """Apply the zeroth-order (specular) TRANSMISSION Jones to an incident
        :class:`~lumenairy.elements.polarization.JonesField` -- the transmissive
        counterpart of :meth:`apply_reflection` (audit S5-11: a transmissive
        metasurface's observable is the transmitted field, but the RCWAResult
        field helpers historically defaulted to / only exposed the reflection
        port).  Operates on a COPY so the caller's incident field is preserved.
        Carries only the specular 2x2 Jones -- for a strongly diffracting cell
        most power is in non-zero orders, so use :meth:`to_multiorder_field`
        (``port='transmission'``) to superpose them."""
        from ..polarization import apply_jones_matrix
        return apply_jones_matrix(jones_field.copy(), self._Jt)

    def _order_amplitude(self, idx, incident, port):
        """Incident-weighted PUBLIC [Ex, Ey] tangential amplitude of one order
        (flat index ``idx``)."""
        from ...backend import to_numpy
        m = self._require_modal()
        ex_a, ey_a = ("rx", "ry") if port == "reflection" else ("tx", "ty")
        Ex = to_numpy(m[ex_a])[:, idx]      # (2,) responses to incident Ex/Ey
        Ey = to_numpy(m[ey_a])[:, idx]
        inc = np.asarray(incident, dtype=_C).reshape(2)
        return inc @ Ex, inc @ Ey           # ex*resp_Ex + ey*resp_Ey

    def _order_power_scale(self, idx, ax, ay, incident, port):
        """Scale ``s`` so the deposited tangential power ``|s ax|^2 + |s ay|^2``
        equals the order's TRUE diffraction efficiency.

        A diffracted order carries power ``flux*(|ax|^2+|ay|^2+|az|^2)/einc_sq``
        with the Poynting flux weight ``flux = Re(kz_m/kz_inc)``, the
        longitudinal field ``az = -(kx ax + ky ay)/kz`` (large for steep
        orders), and the incident ``|E|^2 = einc_sq``.  Depositing the raw
        tangential ``|ax|^2+|ay|^2`` drops all three -> the reconstructed field
        violates energy conservation and can show the wrong dominant order.
        Scaling each carrier by ``s = sqrt(efficiency / (|ax|^2+|ay|^2))``
        restores per-order power (and the right dominant order).

        The scale math itself lives in
        :func:`lumenairy.elements.polarization._order_power_scale` (the single
        source shared with the engine-agnostic :func:`jones_field_from_orders`
        bridge, S5-5); this wrapper only pulls the per-order k-vectors out of
        the modal dict."""
        from ...backend import to_numpy
        from ..polarization import _order_power_scale
        m = self._require_modal()
        kz = complex(to_numpy(m["kz_ref"] if port == "reflection"
                              else m["kz_trn"])[idx])
        kx = complex(to_numpy(m["kx"])[idx])
        ky = complex(to_numpy(m["ky"])[idx])
        return _order_power_scale(ax, ay, kz, kx, ky,
                                  m["kz_inc"], m["kx0"], m["ky0"], incident)

    def to_jones_field(self, nx, ny, dx, *, incident=(1.0, 0.0),
                       port="reflection", order=None, normalize="power",
                       dy=None):
        """Build a :class:`~lumenairy.elements.polarization.JonesField` from a
        single diffraction order's plane-wave response, for the polarization /
        propagation pipeline.

        Parameters
        ----------
        nx, ny : int
            Output grid shape ``(ny, nx)``.
        dx : float
            Grid pitch [m] (``dy`` defaults to ``dx``).
        incident : (2,) complex, optional
            Incident Jones vector ``(E_x, E_y)``.  Default ``(1, 0)``.
        port : {'reflection', 'transmission'}, optional
            Reflection or transmission side.  Default ``'reflection'`` (kept for
            back-compatibility); pass ``port='transmission'`` for a TRANSMISSIVE
            metasurface -- the default is the wrong observable there (audit
            S5-11).
        order : int | (int, int) | None, optional
            Which diffraction order to reconstruct: ``None`` (default) is the
            specular ``(0, 0)`` order, returned as a UNIFORM field (the literal
            specular Jones response); a non-zero order is a TILTED carrier
            ``exp(i (kx_m x + ky_m y))`` ready for :func:`propagate_tilted`.
        normalize : {'power', 'field'}, optional
            For an explicit ``order``: ``'power'`` (default) scales the carrier
            so ``|Ex|^2+|Ey|^2`` equals the order's diffraction EFFICIENCY (the
            energy-correct field that propagates with the right per-order
            power); ``'field'`` deposits the raw tangential boundary amplitude
            (whose ``|.|^2`` is NOT power -- it drops the Poynting flux weight
            and the longitudinal component).  Ignored when ``order is None``.

        Notes
        -----
        Carries one order only; use :meth:`to_multiorder_field` to superpose
        several (a strongly diffracting cell spreads power across orders).
        """
        if port not in ("reflection", "transmission"):
            raise ValueError(
                f"RCWAResult.to_jones_field: port must be 'reflection' or "
                f"'transmission', got {port!r}.")
        if normalize not in ("power", "field"):
            raise ValueError(
                f"RCWAResult.to_jones_field: normalize must be 'power' or "
                f"'field', got {normalize!r}.")
        from ..polarization import JonesField
        ny_i, nx_i = int(ny), int(nx)
        if order is None:
            # Specular order as a UNIFORM field (backward-compatible; the
            # literal specular Jones response applied to the incident vector).
            from ...backend import to_numpy
            J = to_numpy(self._Jr if port == "reflection" else self._Jt)
            inc = np.asarray(incident, dtype=_C).reshape(2)
            out = J @ inc
            ex = np.full((ny_i, nx_i), out[0], dtype=_C)
            ey = np.full((ny_i, nx_i), out[1], dtype=_C)
            return JonesField(ex, ey, dx=dx, dy=dy)
        idx = self._order_index(order)
        from ...backend import to_numpy
        kz_o = to_numpy(self._modal["kz_ref"] if port == "reflection"
                        else self._modal["kz_trn"])[idx]
        if np.real(kz_o) <= 1e-12:
            import warnings
            warnings.warn(
                f"RCWAResult.to_jones_field: order {order!r} is evanescent "
                f"(Re(kz) <= 0); its carrier is a non-physical fast-oscillating "
                f"plane wave (and carries no propagating power, so the "
                f"power-normalized field is zero).", stacklevel=2)
        ax, ay = self._order_amplitude(idx, incident, port)
        if normalize == "power":
            s = self._order_power_scale(idx, ax, ay, incident, port)
            ax, ay = s * ax, s * ay
        carrier = self._order_carrier(idx, nx_i, ny_i, float(dx),
                                      float(dx if dy is None else dy))
        return JonesField(ax * carrier, ay * carrier, dx=dx, dy=dy)

    def _order_carrier(self, idx, nx, ny, dx, dy):
        """Unit plane-wave carrier ``exp(i (kx_m x + ky_m y))`` of one order on
        a centred ``(ny, nx)`` grid (k-vectors are stored normalised by k0).
        Delegates to the shared :func:`lumenairy.elements.polarization.
        _plane_wave_carrier` kernel (S5-5)."""
        from ...backend import to_numpy
        from ..polarization import _plane_wave_carrier
        m = self._require_modal()
        return _plane_wave_carrier(to_numpy(m["kx"])[idx],
                                   to_numpy(m["ky"])[idx],
                                   m["wavelength"], nx, ny, dx, dy)

    def to_multiorder_field(self, nx, ny, dx, *, incident=(1.0, 0.0),
                            port="reflection", orders=None, normalize="power",
                            filter="none", dy=None):
        """Reconstruct the full diffracted field as a superposition of the
        PROPAGATING diffraction orders -- the bridge a strongly diffracting
        metasurface (deflector / grating coupler / metalens cell) needs, where
        most power is in non-zero orders that :meth:`to_jones_field` drops.

        ``E(x, y) = sum_m A_m exp(i (kx_m x + ky_m y))`` over the requested
        ``orders`` (default: every propagating order, ``Re(kz) > 0``).  Returns
        a :class:`~lumenairy.elements.polarization.JonesField` on a centred
        ``(ny, nx)`` grid of pitch ``dx`` (``dy``).

        ``normalize='power'`` (default) scales each order so the field carries
        the correct per-order diffraction efficiency (energy-conserving:
        ``sum |A_m|^2 == sum efficiencies``, and the dominant order in the
        field matches the dominant efficiency); ``'field'`` deposits the raw
        tangential boundary amplitudes (NOT power-conserving -- see
        :meth:`to_jones_field`).

        ``filter='lanczos'`` multiplies each order by the Lanczos sigma factor
        ``sinc(m/(Mx+1)) sinc(n/(My+1))`` before deposition, suppressing the
        Gibbs ringing a truncated-order reconstruction shows at sharp
        permittivity steps (the high orders are damped smoothly rather than
        cut off).  ``'none'`` (default) leaves the orders unweighted.  Note
        the filter trades a little total power for a much smoother field, so
        it is a visualisation / post-processing aid, not energy-exact.

        Note: the reconstruction is exact only over one unit cell (the field is
        quasi-periodic); evanescent orders are excluded.
        """
        if port not in ("reflection", "transmission"):
            raise ValueError(
                f"RCWAResult.to_multiorder_field: port must be 'reflection' "
                f"or 'transmission', got {port!r}.")
        # The superposition + power normalization is the engine-agnostic
        # bridge shared with PMM / 2-D PMM (S5-5): delegate to it on the
        # public per_order_amplitudes(port) dict so the carrier / flux math
        # lives in exactly one place.
        from ..polarization import jones_field_from_orders
        return jones_field_from_orders(
            self.per_order_amplitudes(port), nx, ny, dx, incident=incident,
            normalize=normalize, orders=orders, filter=filter, dy=dy)

    def _lanczos_sigma(self):
        """Per-order Lanczos sigma factors ``sinc(m/(Mx+1)) sinc(n/(My+1))``
        (1-D: the y-factor is 1), indexed by flat order index.  Damps the
        high orders smoothly to suppress Gibbs ringing in the reconstructed
        real-space field.  Delegates to the shared
        :func:`lumenairy.elements.polarization._order_lanczos_sigma` kernel
        (S5-5)."""
        from ..polarization import _order_lanczos_sigma
        return _order_lanczos_sigma(self.orders)

    # -- in-structure (internal) E/H field reconstruction -----------------

    def _internal_cpm(self, info, i, cinc):
        """Forward amplitude ``c+`` at the TOP of layer ``i`` and the backward
        amplitude ``c-_bot`` at its BOTTOM (the numerically-stable reference) for
        the internal-convention source ``cinc`` (the validated gap-free S-matrix
        recovery).

        The backward field is then ``c-_bot exp(-lam k0 (L - z))`` -- a DECAYING
        exponential -- so a deep, lossy layer never forms the overflowing
        ``exp(+lam k0 z)`` (the old top-referenced ``c- exp(+lam k0 z)`` blew up
        to NaN through high-loss metal layers, silently zeroing
        :meth:`layer_absorption`).  Math-identical: ``c- = X c-_bot`` with
        ``X = exp(-lam k0 L)``."""
        N = info["N"]
        Sa = info["S_above"][i]
        Sb = info["S_below"][i]
        Sa22 = np.asarray(to_numpy(Sa[3]))
        Sa21 = np.asarray(to_numpy(Sa[2]))
        Sb11 = np.asarray(to_numpy(Sb[0]))
        denom = np.linalg.inv(np.eye(2 * N, dtype=_C) - Sa22 @ Sb11)
        cplus = denom @ (Sa21 @ cinc)              # forward amp at TOP of layer i
        # backward amp at the BOTTOM = (reflection below the bottom) @ (forward
        # propagated to the bottom).  X = exp(-lam k0 L) decays (Re(lam) >= 0
        # forward branch), so every exponential here is bounded.  On the
        # GENERALIZED retention (info['general']) the forward eigenvalues are
        # the explicit lamf set (same decay convention).
        lam = np.asarray(to_numpy(info["lamf"][i] if info.get("general")
                                  else info["lam"][i]))
        X = np.exp(-lam * info["k0"] * float(info["thick"][i]))
        Sbb11 = np.asarray(to_numpy(info["S_below_bot"][i][0]))
        cminus_bot = Sbb11 @ (X * cplus)
        return cplus, cminus_bot

    def internal_field(self, z, *, component="all", nx=64, ny=None, dx=None,
                       dy=None, layer=None, incident=(1.0, 0.0), filter="none"):
        """Reconstruct the real-space E and/or H field INSIDE the structure
        (audit GAP1) -- the in-layer near field that :meth:`to_multiorder_field`
        (a far-field superposition) cannot show.

        Requires the stack to have been solved with
        ``RCWAStack.solve(retain_internal=True)``.

        Parameters
        ----------
        z : float or array-like
            Depth(s) [m] measured from the TOP of the stack (the superstrate
            interface).  With ``layer=i`` given, ``z`` is instead the LOCAL depth
            inside layer ``i`` (``0`` = its top).
        component : {'E', 'H', 'all'}, optional
            Which field(s) to return (default ``'all'`` -> all six components).
        nx, ny : int, optional
            Real-space grid (``ny`` defaults to 1 for a 1-D stack, else ``nx``).
        dx, dy : float, optional
            Grid pitch [m] (default tiles one unit cell: ``period / n``);
            co-registers with :meth:`to_multiorder_field` when matched.
        layer : int, optional
            Force a specific layer index (then ``z`` is local to that layer);
            default maps each ``z`` to its layer via the cumulative thicknesses.
        incident : (complex, complex), optional
            Incident Jones vector ``(E_x, E_y)`` (default x-polarized).
        filter : {'none', 'lanczos'}, optional
            ``'lanczos'`` damps the high orders (Gibbs suppression at sharp
            permittivity steps) -- a smoothing aid, not energy-exact.

        Returns
        -------
        dict
            ``{'Ex','Ey','Ez','Hx','Hy','Hz'}`` (per ``component``) each a
            ``(nz, ny, nx)`` complex array (``ny`` axis dropped when 1), plus
            ``'z'`` (the depth samples), ``'x'``, ``'y'`` (the grid axes).
            Fields are in the PUBLIC ``exp(-i w t)`` convention.
        """
        info = self._require_modal().get("internal")
        if info is None:
            raise ValueError(
                "RCWAResult.internal_field: the stack was solved without the "
                "internal-field data; call RCWAStack.solve(retain_internal="
                "True).")
        names = {"E": ("Ex", "Ey", "Ez"), "H": ("Hx", "Hy", "Hz"),
                 "all": ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        if component not in names:
            raise ValueError(
                f"RCWAResult.internal_field: component must be 'E', 'H' or "
                f"'all', got {component!r}.")
        if filter not in ("none", "lanczos"):
            raise ValueError(
                f"RCWAResult.internal_field: filter must be 'none' or "
                f"'lanczos', got {filter!r}.")
        want = names[component]
        N = info["N"]
        Kx = np.asarray(to_numpy(info["Kx"]))
        Ky = np.asarray(to_numpy(info["Ky"]))
        k0 = info["k0"]
        delta = np.asarray(to_numpy(info["delta"]))
        thick = info["thick"]
        z_top = np.concatenate([[0.0], np.cumsum(thick)])
        # The cascade runs in the INTERNAL (conjugate) gauge and the output
        # is conjugated back, so a PUBLIC incident Jones (ex0, ey0) must enter
        # CONJUGATED: conj(S(conj(inc))) = ex0 F_pub_x + ey0 F_pub_y -- the
        # public-linear superposition.  Real incidents are unchanged; complex
        # (circular/elliptical) drives previously returned the field of the
        # conjugated incident, i.e. the OPPOSITE handedness (bug found by the
        # PMM internal-field co-registration oracle, 2026-06-11).
        ex0, ey0 = np.conj(incident[0]), np.conj(incident[1])
        cinc = np.concatenate([ex0 * delta, ey0 * delta]).astype(_C)
        sigma = self._lanczos_sigma() if filter == "lanczos" else None

        ny_i = (1 if info["is_1d"] else int(nx)) if ny is None else int(ny)
        nx_i = int(nx)
        dx_f = float(info["period_x"] / nx_i if dx is None else dx)
        dy_f = float(info["period_y"] / ny_i if dy is None else dy)
        xg = (np.arange(nx_i) - nx_i // 2) * dx_f
        yg = (np.arange(ny_i) - ny_i // 2) * dy_f
        # all-order carriers (internal field INCLUDES evanescent orders, unlike
        # the propagating-only far field) -- precompute once and reuse.
        carriers = np.stack([np.asarray(self._order_carrier(idx, nx_i, ny_i,
                                                            dx_f, dy_f))
                             for idx in range(N)])           # (N, ny, nx)

        # cache c+/c- per layer for this incident vector
        cpm = {}

        def _harm(i, zloc):
            if i not in cpm:
                cpm[i] = self._internal_cpm(info, i, cinc)
            cplus, cminus_bot = cpm[i]
            EPS = np.asarray(to_numpy(info["EPS"][i]))
            # forward referenced to the layer TOP (depth z), backward to the
            # BOTTOM (L - z): BOTH exponents are <= 0 (Re(lam) >= 0 on the forward
            # branch), so a deep/lossy layer never overflows.  Same field as the
            # old c- exp(+lam k0 z), evaluated in the numerically-stable order.
            Lz = float(thick[i]) - zloc
            if info.get("general"):
                # GENERALIZED retention: explicit asymmetric mode sets; the
                # backward branch has Re(lamb) <= 0, so exp(+lamb k0 (L-z))
                # decays referenced from the layer BOTTOM.
                Wf = np.asarray(to_numpy(info["Wf"][i]))
                Vf = np.asarray(to_numpy(info["Vf"][i]))
                Wb = np.asarray(to_numpy(info["Wb"][i]))
                Vb = np.asarray(to_numpy(info["Vb"][i]))
                lamf = np.asarray(to_numpy(info["lamf"][i]))
                lamb = np.asarray(to_numpy(info["lamb"][i]))
                fwd = cplus * np.exp(-lamf * k0 * zloc)
                bwd = cminus_bot * np.exp(lamb * k0 * Lz)
                s = Wf @ fwd + Wb @ bwd                      # E tangential
                u = Vf @ fwd + Vb @ bwd                      # H partner
            else:
                W = np.asarray(to_numpy(info["W"][i]))
                V = np.asarray(to_numpy(info["V"][i]))
                lam = np.asarray(to_numpy(info["lam"][i]))
                fwd = cplus * np.exp(-lam * k0 * zloc)
                bwd = cminus_bot * np.exp(-lam * k0 * Lz)
                s = W @ (fwd + bwd)                          # E tangential
                u = V @ (fwd - bwd)                          # H partner
            Sx, Sy = s[:N], s[N:]
            Hx, Hy = -1j * u[:N], -1j * u[N:]                # -i*eta0 scale
            rhs = -(Kx @ Hy - Ky @ Hx)
            if info.get("general") and info["EZX"][i] is not None:
                # OOP tensor: D_z = EZX Ex + EZY Ey + EZZ Ez (internal-gauge
                # convolutions, same as the mode build) -> solve for Ez.
                EZX = np.asarray(to_numpy(info["EZX"][i]))
                EZY = np.asarray(to_numpy(info["EZY"][i]))
                rhs = rhs - EZX @ Sx - EZY @ Sy
            Sz = np.linalg.solve(EPS, rhs)                   # curl-H z-comp
            Hz = Kx @ Sy - Ky @ Sx                           # curl-E z-comp
            return dict(Ex=Sx, Ey=Sy, Ez=Sz, Hx=Hx, Hy=Hy, Hz=Hz)

        zs = np.atleast_1d(np.asarray(z, dtype=float))
        out = {c: np.empty((zs.shape[0], ny_i, nx_i), dtype=_C) for c in want}
        for zi, zz in enumerate(zs):
            if layer is None:
                i = int(np.clip(np.searchsorted(z_top, zz, side="right") - 1,
                                0, len(thick) - 1))
                zloc = zz - z_top[i]
            else:
                i, zloc = int(layer), zz
            h = _harm(i, zloc)
            for c in want:
                amp = np.conj(h[c])                          # internal -> public
                if sigma is not None:
                    amp = amp * sigma
                out[c][zi] = np.tensordot(amp, carriers, axes=([0], [0]))
        result = {c: (out[c][:, 0, :] if ny_i == 1 else out[c]) for c in want}
        result["z"] = zs
        result["x"] = xg
        result["y"] = yg
        return result

    @staticmethod
    def _cell_grid_index(xg, yg, period_x, period_y, Sx, Sy):
        """Nearest-cell-pixel ``(ix, iy)`` index grids mapping the real-space
        field samples ``(xg, yg)`` onto a ``(Sx, Sy)`` unit-cell sampling.

        The solver's permittivity convention is set by
        :func:`~lumenairy.elements.rcwa.twod._eps_convolution_2d`, whose
        ``fft2(eps_cell)/(Sx Sy)`` makes the band-limited ``eps(x, y)`` pass
        EXACTLY through ``eps_cell[j, i]`` at the NODE ``(j Px/Sx, i Py/Sy)``.
        The nearest node to a field sample is therefore ``round``, not
        ``floor``: ``floor`` reads the pixel at or below, a systematic
        HALF-CELL-PIXEL bias of ``P/(2 Sx)`` that does NOT shrink with the
        field grid ``nx`` (it is fixed by ``Sx``).

        .. versionchanged:: 5.29
           Was ``floor`` despite this docstring's "Nearest" (audit W7-D).
           Measured on a half-metal / half-air 64-pixel cell over a lossy
           spacer: :meth:`layer_absorption` converged to 0.16548922 (floor)
           vs 0.16541527 (nearest) for the metal layer -- a 7.4e-5 absolute /
           4.5e-4 relative offset in the per-layer split that persisted at
           ``nx = 4096``.
        """
        ix = (np.round((xg % period_x) / period_x * Sx).astype(int)) % Sx
        iy = (np.round((yg % period_y) / period_y * Sy).astype(int)) % Sy
        return ix, iy

    @staticmethod
    def _layer_im_eps(layer, xg, yg, period_x, period_y):
        """``Im(eps_public(x, y)) >= 0`` of a SCALAR (uniform / isotropic) layer
        on the real-space field grid (the local loss density weight for
        :meth:`layer_absorption`)."""
        ny, nx = len(yg), len(xg)
        if layer.kind == "uniform":
            return np.full((ny, nx), float(np.imag(complex(layer.data))))
        if layer.kind == "iso":
            cell = np.asarray(layer.data)                 # (Sx, Sy), public
            Sx, Sy = cell.shape
            ix, iy = RCWAResult._cell_grid_index(xg, yg, period_x, period_y,
                                                 Sx, Sy)
            return np.imag(cell[np.ix_(ix, iy)]).T          # (ny, nx)
        raise NotImplementedError(
            "RCWAResult.layer_absorption: _layer_im_eps handles only uniform / "
            "isotropic-cell layers; a tensor layer uses _layer_eps_tensor_grid.")

    @staticmethod
    def _layer_eps_tensor_grid(layer, xg, yg, period_x, period_y):
        """Full ``(ny, nx, 3, 3)`` PUBLIC permittivity tensor of a TENSOR layer
        on the real-space field grid (nearest-pixel sampling of the
        ``(Sx, Sy, 3, 3)`` cell) -- the loss operator for the tensor branch of
        :meth:`layer_absorption`."""
        cell = np.asarray(layer.data)                     # (Sx, Sy, 3, 3) public
        Sx, Sy = cell.shape[0], cell.shape[1]
        ix, iy = RCWAResult._cell_grid_index(xg, yg, period_x, period_y, Sx, Sy)
        # gather (nx, ny, 3, 3) then move to (ny, nx, 3, 3)
        g = cell[np.ix_(ix, iy)]                           # (nx, ny, 3, 3)
        return np.transpose(g, (1, 0, 2, 3))               # (ny, nx, 3, 3)

    def layer_absorption(self, *, nx=64, ny=None, nz_per_layer=8):
        """Absorbed power fraction broken down PER LAYER (audit GAP6).

        The total absorptance ``A = 1 - sum(R) - sum(T)`` tells you how much
        power is lost but not WHERE; this attributes it to each layer by
        integrating the local loss density over each layer, normalised so the
        layers sum to the total absorptance (energy-conserving by construction).
        For a scalar (uniform / isotropic) layer the density is
        ``Im(eps) |E|^2``; for an ANISOTROPIC (tensor) layer it is the full
        quadratic form ``Im(E* . eps . E)`` (the rigorous power-loss operator,
        which reduces to ``Im(eps) |E|^2`` for a diagonal medium) -- the
        reconstructed ``(Ex, Ey, Ez)`` is contracted against the local
        ``Im(eps_tensor)``.

        Requires ``RCWAStack.solve(retain_internal=True)``.  Uniform,
        isotropic-cell, and TENSOR (in-plane) layers are supported (raises for
        analytic-shape layers, whose local map is not reconstructed here).

        Returns
        -------
        (2, n_layers) ndarray
            Absorbed fraction in each layer; row 0 for incident ``E_x``, row 1
            for incident ``E_y``.  ``row.sum() == absorptance()[row]``.
        """
        info = self._require_modal().get("internal")
        if info is None:
            raise ValueError(
                "RCWAResult.layer_absorption: the stack was solved without the "
                "internal-field data; call RCWAStack.solve(retain_internal="
                "True).")
        layers = info["layers"]
        thick = info["thick"]
        nlay = len(layers)
        A_tot = self.absorptance()                          # (2,)
        px, py = info["period_x"], info["period_y"]
        ny_i = (1 if info["is_1d"] else int(nx)) if ny is None else int(ny)
        nx_i = int(nx)
        dx, dy = px / nx_i, py / ny_i
        xg = (np.arange(nx_i) - nx_i // 2) * dx
        yg = (np.arange(ny_i) - ny_i // 2) * dy
        # Per-layer local loss operator: a scalar Im(eps) map for uniform /
        # isotropic layers, the full (ny, nx, 3, 3) tensor for anisotropic
        # layers (the loss density is then the quadratic form Im(E* . eps . E)).
        if any(L.kind == "shapes" for L in layers):
            raise NotImplementedError(
                "RCWAResult.layer_absorption: analytic-shape layers are not "
                "supported (their local Im(eps) map is not reconstructed here); "
                "uniform, isotropic-cell, and tensor layers are.")
        im_eps = [None if L.kind == "tensor"
                  else self._layer_im_eps(L, xg, yg, px, py) for L in layers]
        eps_t = [self._layer_eps_tensor_grid(L, xg, yg, px, py)
                 if L.kind == "tensor" else None for L in layers]
        out = np.zeros((2, nlay))
        for p, inc in enumerate(((1.0, 0.0), (0.0, 1.0))):
            raw = np.zeros(nlay)
            for i in range(nlay):
                zl = (np.arange(nz_per_layer) + 0.5) / nz_per_layer * thick[i]
                f = self.internal_field(zl, component="E", nx=nx_i, ny=ny_i,
                                        dx=dx, dy=dy, layer=i, incident=inc)
                ex, ey, ez = f["Ex"], f["Ey"], f["Ez"]
                if ex.ndim == 2:                            # 1-D: (nz, nx)
                    ex, ey, ez = ex[:, None, :], ey[:, None, :], ez[:, None, :]
                if layers[i].kind == "tensor":
                    # density = Im(E* . eps . E) with E = (Ex, Ey, Ez), the
                    # local public tensor T = eps_t[i] broadcast over (nz):
                    # sum_ab conj(E_a) T_ab E_b, then take Im (>= 0 passive).
                    E = np.stack([ex, ey, ez], axis=-1)     # (nz, ny, nx, 3)
                    T = eps_t[i][None]                      # (1, ny, nx, 3, 3)
                    TE = np.einsum("...ab,...b->...a", T, E)  # eps . E
                    dens = np.imag(np.sum(np.conj(E) * TE, axis=-1))
                else:
                    e2 = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
                    dens = im_eps[i][None] * e2
                # thickness-weighted cell+depth average of the loss density
                raw[i] = thick[i] * float(np.mean(dens))
            total = raw.sum()
            out[p] = A_tot[p] * raw / total if total > 0 else np.zeros(nlay)
        return out



class _RCWALayer:
    __slots__ = ("thickness", "kind", "data", "formulation", "dispersive")

    def __init__(self, thickness, kind, data, formulation="laurent",
                 dispersive=False):
        # keep a traced (JAX) thickness native so layer DEPTH is differentiable
        self.thickness = thickness if is_jax_array(thickness) else float(thickness)
        self.kind = kind          # 'uniform' | 'iso' | 'shapes' | 'tensor'
        self.data = data
        self.formulation = formulation   # 'laurent' | 'li' (iso layers)
        self.dispersive = dispersive     # data holds wl -> value callable(s)



# ---------------------------------------------------------------------------
# RCWAStack.solve(stabilize=True) consensus selector
# (audit AUDIT_RCWA_STACK_RESONANT_CONVERGENCE_2026_06_02).  A sharp-resonant
# metal STACK biases a SINGLE diffraction order (a reflection null)
# non-monotonically at isolated n_orders while total power stays bounded -- the
# isolated-resonance pathology fixed for pmm_efficiency_1d in v5.10.6, here in
# the multilayer S-matrix (near-singular layer<->layer / layer<->region
# mode-match at isolated truncation counts).  Scan a short upward n_orders
# window and keep the solve whose ZERO-ORDER R/T is in the consensus cluster --
# PER-ORDER, not total power, which does not move.
# ---------------------------------------------------------------------------
_STACK_STABILIZE_WINDOW = 5
_STACK_STABILIZE_TOL = 0.02          # 0-order efficiency agreement (spikes >~0.1)


def _zero_order_efficiency(result):
    """The zeroth-order ``(R, T)`` efficiencies for both incident polarizations
    of an ``RCWAResult``, as a length-4 vector -- the per-order quantity the
    stack resonance biases (the reflection null)."""
    o, R, T = result.efficiencies()
    o = np.asarray(to_numpy(o))
    R = np.asarray(to_numpy(R))
    T = np.asarray(to_numpy(T))
    if o.ndim == 1:
        p0 = int(np.where(o == 0)[0][0])
    else:
        p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    return np.concatenate([R[:, p0], T[:, p0]])



def _largest_feature_cluster(feats, tol):
    """Indices of the largest group of feature vectors mutually within ``tol``
    (L-inf) -- the resonance-free consensus; greedy around each anchor."""
    best = []
    for i, fi in enumerate(feats):
        grp = [j for j, fj in enumerate(feats)
               if float(np.max(np.abs(fi - fj))) <= tol]
        if len(grp) > len(best):
            best = grp
    return best



def _cluster_is_trending(feats, cluster, tol):
    """True if the consensus ``cluster`` is a MONOTONE trend rather than a flat
    convergence plateau (audit P1-B).  A genuine plateau is flat; an UNDER-RESOLVED
    cell biases all the low-order solves the SAME direction, so they cluster within
    ``tol`` yet drift monotonically in ``n_orders``.  ``feats`` are ordered by the
    DESCENDING-n window, so a cluster index list sorted ascending is descending-n.
    Returns True iff some feature component drifts monotonically across the cluster
    with a spread that is a real trend (> 0.4*tol), not flat-band noise."""
    idx = sorted(cluster)                   # ascending index == descending n_orders
    if len(idx) < 3:                        # 2 points can't distinguish trend / flat
        return False
    series = np.asarray([np.asarray(feats[i], dtype=float) for i in idx])  # (k, F)
    for c in range(series.shape[1]):
        col = series[:, c]
        diffs = np.diff(col)
        spread = float(col.max() - col.min())
        if spread > 0.4 * tol and (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
            return True
    return False



class RCWAStack:
    """Builder + solver for a MULTI-LAYER RCWA stack (1-D or 2-D periodic).

    Compose a stack of uniform spacers and patterned layers (isotropic or
    full-tensor / liquid-crystal) between a superstrate and substrate, set
    the incident plane wave, and solve once for the diffraction efficiencies
    of both incident polarizations plus the zeroth-order Jones reflection.

    Example
    -------
    >>> stack = RCWAStack(period=1.0e-6, n_superstrate=1.0, n_substrate=1.5,
    ...                   n_orders=11)
    >>> stack.add_layer(0.1e-6, eps=2.1 ** 2)          # uniform spacer
    >>> stack.add_layer(0.2e-6, eps_cell=cell)         # 2-D patterned layer
    >>> res = stack.set_source(0.633e-6, theta=0.2).solve()
    >>> orders, R, T = res.efficiencies()

    Parameters
    ----------
    period : float
        Period along x (metres).
    period_y : float, optional
        Period along y for a 2-D (crossed) stack.  If omitted (and
        ``n_orders_y`` is omitted) the stack is 1-D (mono-periodic).
    n_superstrate, n_substrate : complex
        Incidence / transmission half-space indices.
    n_orders : int
        Retained orders per side along x.
    n_orders_y : int, optional
        Retained orders per side along y (2-D only; default = ``n_orders``).
    """

    def __init__(self, period, *, period_y=None, n_superstrate=1.0,
                 n_substrate=1.0, n_orders=11, n_orders_y=None, use_gpu=False,
                 truncation="rectangular"):
        _validate_geometry("RCWAStack", period=period, period_y=period_y,
                           n_orders=n_orders, n_orders_y=n_orders_y)
        if truncation not in ("rectangular", "circular"):
            raise ValueError(
                f"RCWAStack: truncation must be 'rectangular' or 'circular', "
                f"got {truncation!r}")
        self.period_x = float(period)
        self.is_1d = period_y is None and n_orders_y is None
        if truncation == "circular" and self.is_1d:
            # The inscribed-circle radius is min(Mx/px, My/py); with My = 0 (a
            # 1-D stack) it collapses to 0 and keeps ONLY the (0, 0) order.
            # Circular truncation is a 2-D concept -- reject it rather than
            # silently drop every diffracted order.
            raise ValueError(
                "RCWAStack: truncation='circular' applies to 2-D (crossed) "
                "stacks only; a 1-D stack (period_y / n_orders_y unset) keeps "
                "its full 1-D order range -- use truncation='rectangular'.")
        self.period_y = float(period if period_y is None else period_y)
        self.n_superstrate = n_superstrate
        self.n_substrate = n_substrate
        self.nox = int(n_orders)
        self.noy = 0 if self.is_1d else int(
            n_orders if n_orders_y is None else n_orders_y)
        # Reciprocal-lattice order set (audit S1-20 -- exposed here to match the
        # scalar 2-D entries): 'rectangular' (default) keeps the full Cartesian
        # box; 'circular' (Lalanne 1997) keeps the inscribed reciprocal circle
        # (isotropic resolution, fewer harmonics).  2-D stacks only (rejected
        # above for 1-D, where the circle radius would collapse to (0, 0)).
        self.truncation = truncation
        self.use_gpu = bool(use_gpu)
        self._layers: List[_RCWALayer] = []
        self._source: Optional[dict] = None

    @property
    def n_orders_x(self) -> int:
        """Retained Fourier orders per side along x -- the ``n_orders``
        constructor argument.  Read-only public mirror of the internal ``nox``
        field (which ``rcwa_convergence`` bumps in place during a sweep)."""
        return self.nox

    @property
    def n_orders_y(self) -> int:
        """Retained Fourier orders per side along y (0 for a 1-D stack; the
        ``n_orders_y`` constructor argument otherwise).  Read-only public mirror
        of the internal ``noy`` field."""
        return self.noy

    @property
    def layers(self) -> tuple:
        """Read-only view (tuple) of the per-layer records, in stack order
        (top / superstrate side first) -- the PUBLIC accessor for what
        :meth:`add_layer` stored (AUDIT_DYNAMETA_CONSUMER_API_GAPS A2;
        reverse translators previously had to read the private ``_layers``
        slot under a version ceiling).

        Each record carries the PUBLIC fields

        * ``thickness`` -- layer depth [m] (a raw JAX scalar if traced);
        * ``kind`` -- ``'uniform' | 'iso' | 'shapes' | 'tensor'``;
        * ``data`` -- the layer specification exactly as ``add_layer``
          normalized it (``kind``-dependent: the scalar eps, the
          ``(Sx, Sy)`` cell, the ``(shapes, eps_background)`` pair, or the
          ``(Sx, Sy, 3, 3)`` tensor cell; wl-callables when dispersive);
        * ``formulation`` -- ``'laurent' | 'li'`` (iso layers);
        * ``dispersive`` -- True when ``data`` holds ``wl ->`` callables
          (solve via :meth:`solve_vs_wavelength`).

        The tuple is a snapshot; mutating the stack requires the builder
        methods."""
        return tuple(self._layers)

    def _uniform_tensor_cell(self, fn_name, tensor):
        """Expand a uniform ``(3, 3)`` permittivity tensor to a validated
        uniform tensor CELL (audit S1-14 / S5-11): every pixel carries the same
        tensor, so its only Fourier harmonic is DC (a spatially-uniform
        anisotropic slab), but as a proper ``(Sx, Sy, 3, 3)`` cell it routes
        through the tested tensor eigenmode path instead of the scalar
        ``"uniform"`` slot (which stored a ``(3, 3)`` and crashed opaquely at
        solve).  Sampled at ``4*n_orders + 1`` per axis so it clears the strict
        patterned Fourier-aliasing bound even on the JAX backend (where the
        cell's uniformity is not inspectable); cell size drives only the FFT, not
        the ``O(N^3)`` eigensolve, so the oversampling is free."""
        need_x = max(1, 4 * self.nox + 1)
        need_y = max(1, 4 * self.noy + 1)
        if is_jax_array(tensor):
            import jax.numpy as jnp
            t = jnp.asarray(tensor)
            if tuple(t.shape) != (3, 3):
                raise ValueError(
                    f"{fn_name}: eps as an array must be a single (3, 3) "
                    f"permittivity tensor (a spatially-uniform anisotropic "
                    f"layer); got shape {tuple(t.shape)}.  For a PATTERNED layer "
                    f"use eps_cell / eps_tensor_cell.")
            return jnp.broadcast_to(t.astype(_C), (need_x, need_y, 3, 3))
        t = np.asarray(tensor, dtype=_C)
        if t.shape != (3, 3):
            raise ValueError(
                f"{fn_name}: eps as an array must be a single (3, 3) "
                f"permittivity tensor (a spatially-uniform anisotropic layer); "
                f"got shape {t.shape}.  For a PATTERNED layer use eps_cell / "
                f"eps_tensor_cell, and for an isotropic region pass a scalar "
                f"eps.")
        return np.array(np.broadcast_to(t, (need_x, need_y, 3, 3)))

    def add_layer(self, thickness, *, eps=None, eps_cell=None,
                  eps_tensor_cell=None, shapes=None, eps_background=None,
                  formulation="laurent"):
        """Append a layer.  Provide exactly one layer specification:

        * ``eps`` (scalar) -- uniform isotropic spacer; a ``(3, 3)`` array is a
          spatially-uniform ANISOTROPIC slab (audit S1-14 / S5-11: expanded to a
          uniform tensor cell and solved on the tensor path -- the clean
          uniform-tensor entry; a bare ``(3, 3)`` formerly crashed at solve);
        * ``eps_cell`` (``(Sx, Sy)``) -- isotropic patterned, FFT-sampled;
        * ``eps_tensor_cell`` (``(Sx, Sy, 3, 3)``) -- anisotropic patterned;
        * ``shapes`` (with ``eps_background``) -- isotropic patterned using
          ANALYTIC shape Fourier transforms (exact form factors, direct-rule
          ``E_z``; see :func:`rcwa_efficiency_2d_shapes`).

        Permittivities are in the PUBLIC convention (``Im(eps) > 0`` lossy).

        DISPERSIVE materials (audit GAP5, v5.14.1): ``eps``, ``eps_cell``,
        ``eps_tensor_cell``, ``eps_background``, and each shape's ``eps`` may
        be a CALLABLE ``wl -> value`` instead of a concrete value; such a
        stack is solved with :meth:`solve_vs_wavelength` (a plain
        :meth:`solve` raises, since no single wavelength is defined).

        ``formulation`` (audit GAP3, v5.14.1): ``'laurent'`` (default,
        unchanged) or ``'li'``/``'fff'`` for an ISOTROPIC patterned layer --
        the Li-1997 sequential rule (inverse along each E-component's own
        axis, direct along the other, direct-rule ``E_z``), the
        fast-converging factorization for metal / high-contrast layers.
        For 1-D stacks it reduces to the rigorous 1-D ``'li'`` placement.
        Uniform / shapes / tensor layers accept only the default.
        """
        # Uniform ANISOTROPIC entry (audit S1-14 / S5-11): a (3, 3) permittivity
        # tensor passed as ``eps`` is a spatially-uniform tensor layer.  A bare
        # (3, 3) was formerly accepted into the scalar "uniform" slot and then
        # crashed OPAQUELY at solve (``complex()`` of a (3, 3) array).  Expand it
        # to a minimally-sampled uniform tensor CELL and route it through the
        # validated tensor eigenmode path (its only Fourier harmonic is DC, so it
        # is exactly a homogeneous anisotropic slab).  Kept BEFORE the one-of
        # count so the layer registers as an ``eps_tensor_cell``.
        if eps is not None and not callable(eps):
            _ea = eps if is_jax_array(eps) else np.asarray(eps)
            if getattr(_ea, "ndim", 0) >= 2:
                eps_tensor_cell = self._uniform_tensor_cell("add_layer", eps)
                eps = None
        n = sum(x is not None for x in (eps, eps_cell, eps_tensor_cell, shapes))
        if n != 1:
            raise ValueError(
                "add_layer: provide exactly one of eps, eps_cell, "
                "eps_tensor_cell, shapes.")
        if not is_jax_array(thickness):       # traced depth -> skip the guard
            _validate_geometry("add_layer", depth=thickness)
        f = str(formulation).lower()
        if f == "fff":
            f = "li"
        if f not in ("laurent", "li"):
            raise ValueError(
                f"add_layer: formulation must be 'laurent' or 'li'/'fff', "
                f"got {formulation!r}.")
        if f != "laurent" and eps_cell is None:
            raise ValueError(
                "add_layer: formulation='li' applies to isotropic patterned "
                "layers (eps_cell) only; uniform / shapes / tensor layers "
                "use their own factorizations.")
        if eps is not None:
            if callable(eps):
                self._layers.append(_RCWALayer(thickness, "uniform", eps,
                                               dispersive=True))
            elif is_jax_array(eps):
                # D1 (AUDIT_DYNAMETA_CONSUMER_API_GAPS): a traced uniform eps
                # stays RAW so its gradient flows (complex() severed it,
                # forcing consumers to lift a physically-uniform slab into a
                # constant patterned cell and pay a full eigensolve).
                self._layers.append(_RCWALayer(thickness, "uniform", eps))
            else:
                self._layers.append(_RCWALayer(thickness, "uniform", _C(eps)))
        elif eps_cell is not None:
            if callable(eps_cell):
                self._layers.append(_RCWALayer(thickness, "iso", eps_cell,
                                               formulation=f, dispersive=True))
                return self
            # Keep a JAX cell native (a np.asarray would materialise the tracer
            # and break the differentiable RCWAStack path); NumPy/list inputs
            # are still normalised to a contiguous complex array.
            cell = (eps_cell.astype(_C) if is_jax_array(eps_cell)
                    else np.asarray(eps_cell, dtype=_C))
            if cell.ndim == 1:
                cell = cell[:, None]
            _validate_cell_sampling("add_layer", cell, self.nox, self.noy,
                                    strict_y=not self.is_1d)
            _warn_if_y_averaged("add_layer", cell, self.noy,
                                strict_y=not self.is_1d)
            self._layers.append(_RCWALayer(thickness, "iso", cell,
                                           formulation=f))
        elif shapes is not None:
            if eps_background is None:
                raise ValueError(
                    "add_layer: shapes requires eps_background.")
            disp = callable(eps_background) or any(
                callable(sh.get("eps")) for sh in shapes)
            if not disp:
                _validate_shapes("add_layer", shapes, self.period_x,
                                 self.period_y)
                self._layers.append(
                    _RCWALayer(thickness, "shapes",
                               (_C(eps_background), shapes)))
            else:
                self._layers.append(
                    _RCWALayer(thickness, "shapes", (eps_background, shapes),
                               dispersive=True))
        else:
            if callable(eps_tensor_cell):
                self._layers.append(_RCWALayer(thickness, "tensor",
                                               eps_tensor_cell,
                                               dispersive=True))
                return self
            tcell = (eps_tensor_cell.astype(_C) if is_jax_array(eps_tensor_cell)
                     else np.asarray(eps_tensor_cell, dtype=_C))
            _validate_cell_sampling("add_layer", tcell, self.nox, self.noy,
                                    strict_y=not self.is_1d)
            _warn_if_y_averaged("add_layer", tcell, self.noy,
                                strict_y=not self.is_1d)
            # OUT-OF-PLANE tensor layers are SUPPORTED since v5.14.1 (the
            # rcwa_jones_2d GAP2 machinery promoted to the stack via the
            # PMM2DStack any_oop pattern), including the DIFFERENTIABLE (JAX)
            # solve (a concrete cell is inspected for out-of-plane coupling; a
            # traced cell routes to the general cascade -- exact for in-plane
            # too -- so a differentiable out-of-plane stack gets a correct
            # gradient).  The remaining contract is a nonzero e_zz (the pointwise
            # ezz-Schur fold divides by it), checkable only for a concrete cell.
            if _tensor_offplane_present(tcell):
                ezz_min = float(np.min(np.abs(
                    to_numpy(tcell)[..., 2, 2])))
                if ezz_min < 1e-12:
                    raise ValueError(
                        "add_layer: an out-of-plane tensor layer requires "
                        "|e_zz| > 0 everywhere (the E_z elimination divides "
                        f"by it); min |e_zz| = {ezz_min:.3e}.")
            self._layers.append(_RCWALayer(thickness, "tensor", tcell))
        return self

    def add_graded_layer(self, thickness, profile, *, n_slices=8,
                         rule="midpoint"):
        """Append a continuously-graded ``eps(z)`` layer as an auto-sliced
        z-staircase of ``n_slices`` thin layers (audit GAP4 / P4).

        A continuous depth profile (a carrier-accumulation / ENZ layer, a
        thermo-optic or field gradient, a tapered etch) is approximated by a
        staircase; this centralises that slicing so the caller does not
        hand-roll many :meth:`add_layer` calls.

        Parameters
        ----------
        thickness : float
            Total layer thickness (metres).
        profile : callable
            ``profile(zeta) -> permittivity`` at fractional depth
            ``zeta in [0, 1]`` (``0`` = top, ``1`` = bottom).  The return shape
            selects the layer kind: a SCALAR -> uniform spacer; an ``(Sx, Sy)``
            array -> isotropic ``eps_cell``; an ``(Sx, Sy, 3, 3)`` array ->
            anisotropic ``eps_tensor_cell``.
        n_slices : int, optional
            Number of staircase slices (the convergence knob; default 8).
        rule : {'midpoint', 'trapezoid'}, optional
            Sample each slice at its centre (``'midpoint'``, default) or average
            its two edges (``'trapezoid'``).
        """
        n = int(n_slices)
        if n < 1:
            raise ValueError(
                f"add_graded_layer: n_slices must be >= 1, got {n_slices}.")
        if rule not in ("midpoint", "trapezoid"):
            raise ValueError(
                f"add_graded_layer: rule must be 'midpoint' or 'trapezoid', "
                f"got {rule!r}.")
        dz = float(thickness) / n
        for k in range(n):
            if rule == "midpoint":
                eps_k = np.asarray(profile((k + 0.5) / n), dtype=_C)
            else:
                eps_k = 0.5 * (np.asarray(profile(k / n), dtype=_C)
                               + np.asarray(profile((k + 1) / n), dtype=_C))
            if eps_k.ndim == 0:
                self.add_layer(dz, eps=complex(eps_k))
            elif eps_k.ndim == 2:
                self.add_layer(dz, eps_cell=eps_k)
            elif eps_k.ndim == 4:
                self.add_layer(dz, eps_tensor_cell=eps_k)
            else:
                raise ValueError(
                    f"add_graded_layer: profile must return a scalar, an "
                    f"(Sx, Sy) cell, or an (Sx, Sy, 3, 3) tensor cell; got "
                    f"shape {eps_k.shape}.")
        return self

    def add_tapered_grating(self, thickness, *, eps_ridge, eps_groove,
                            duty_bottom, duty_top=None, n_slices=12, n_x=256,
                            shear=0.0):
        """Append a 1-D grating with SLANTED (trapezoidal) sidewalls as an
        auto-sliced z-staircase (audit GAP4 -- fab realism).

        The centred ridge's duty cycle varies linearly with depth from
        ``duty_top`` (at the top, ``zeta = 0``) to ``duty_bottom`` (at the
        bottom, ``zeta = 1``); ``duty_top == duty_bottom`` is the usual vertical
        binary grating.  A small sidewall taper can materially change a device,
        so this makes the staircase a one-liner with a documented ``n_slices``
        convergence knob.

        Parameters
        ----------
        thickness : float
            Grating thickness (metres).
        eps_ridge, eps_groove : complex
            Ridge / groove permittivities (PUBLIC ``Im(eps) > 0``).
        duty_bottom : float
            Ridge fraction at the bottom of the grating, in ``[0, 1]``.
        duty_top : float, optional
            Ridge fraction at the top; defaults to ``duty_bottom`` (vertical).
        n_slices : int, optional
            Staircase slice count (default 12).
        n_x : int, optional
            Cell samples along x for each slice's ``eps_cell`` (default 256).
        shear : float, optional
            SHEARED (parallelogram) sidewalls (audit GAP1, v5.14.1): the
            ridge CENTRE shifts laterally by ``shear`` periods from top to
            bottom (e.g. ``shear = depth * tan(wall_angle) / period``), so a
            30-deg slanted-wall binary grating is
            ``add_tapered_grating(d, ..., duty_top=duty_bottom,
            shear=d*tan(0.524)/period)``.  Wrap-aware (the ridge may cross
            the cell edge).  Default 0 (centred, unchanged).  Staircase
            accuracy class: ~1e-3 absolute at 16-32 slices.
        """
        dt = float(duty_bottom if duty_top is None else duty_top)
        db = float(duty_bottom)
        for d in (db, dt):
            if not (0.0 <= d <= 1.0):
                raise ValueError(
                    f"add_tapered_grating: duty cycles must be in [0, 1], got "
                    f"duty_top={dt}, duty_bottom={db}.")
        er, eg = _C(eps_ridge), _C(eps_groove)
        sh = float(shear)
        x = (np.arange(int(n_x)) + 0.5) / int(n_x)
        n_y = max(1, 4 * self.noy + 1)              # grating is uniform in y

        def _profile(zeta):
            duty = dt + (db - dt) * zeta            # top (0) -> bottom (1)
            half = 0.5 * duty
            centre = 0.5 + sh * (zeta - 0.5)        # sheared ridge centre
            dist = np.abs((x - centre + 0.5) % 1.0 - 0.5)   # wrap-aware
            col = np.where(dist < half, er, eg).astype(_C)
            return np.broadcast_to(col[:, None], (int(n_x), n_y)).copy()

        return self.add_graded_layer(thickness, _profile, n_slices=n_slices,
                                     rule="midpoint")

    def add_tapered_ridges(self, thickness, *, ridges, eps_groove,
                           n_slices=12, n_x=256, n_y=None):
        """Append a MULTI-RIDGE tapered grating as an auto-sliced z-staircase
        (device-geometry roadmap item 1, 2026-06-10) -- the N-feature,
        center-anchored generalization of :meth:`add_tapered_grating`.

        ``ridges`` is a list of ``(center, w_top, w_bottom, eps)`` in ABSOLUTE
        metres; each ridge tapers linearly ABOUT ITS OWN FIXED CENTER (a
        width-sequence construction that left-anchors a ridge drifts its
        center -- the audited geometry bug).  Wrap-aware; later ridges may
        NOT overlap earlier ones (raises).  ``eps_groove`` fills the rest.
        """
        n = int(n_slices)
        if n < 1:
            raise ValueError(
                f"add_tapered_ridges: n_slices must be >= 1, got {n_slices}.")
        rid = [(float(c), float(wt), float(wb), _C(e))
               for c, wt, wb, e in ridges]
        eg = _C(eps_groove)
        nx = int(n_x)
        ny = int(n_y) if n_y is not None else max(1, 4 * self.noy + 1)
        x = (np.arange(nx) + 0.5) / nx          # period fractions

        def _profile(zeta):
            col = np.full(nx, eg, dtype=_C)
            covered = np.zeros(nx, dtype=bool)
            for c, wt, wb, e in rid:
                w = wt + (wb - wt) * zeta
                if w <= 0.0:
                    continue
                half = 0.5 * w / self.period_x
                dist = np.abs((x - c / self.period_x + 0.5) % 1.0 - 0.5)
                m = dist < half
                if np.any(m & covered):
                    raise ValueError(
                        "add_tapered_ridges: ridges overlap; merge or "
                        "separate them explicitly.")
                covered |= m
                col[m] = e
            return np.broadcast_to(col[:, None], (nx, ny)).copy()

        return self.add_graded_layer(thickness, _profile, n_slices=n,
                                     rule="midpoint")

    def add_tapered_pillars(self, thickness, *, pillars, eps_host,
                            n_slices=8, n_x=None, n_y=None):
        """Append MULTI-PILLAR tapered 2-D layers as an auto-sliced
        z-staircase (device-geometry roadmap item 1; the 2-D RCWA stack had
        NO tapered builder at all).

        ``pillars`` is a list of
        ``((cx, cy), (wx_top, wy_top), (wx_bottom, wy_bottom), eps)`` in
        ABSOLUTE metres; each pillar tapers linearly about its own fixed
        center.  ``eps_host`` fills the rest.  Wrap-aware in both axes;
        overlapping pillars raise.
        """
        if self.is_1d:
            raise ValueError(
                "add_tapered_pillars: this is a 1-D stack; use "
                "add_tapered_ridges (or construct the stack with period_y).")
        n = int(n_slices)
        if n < 1:
            raise ValueError(
                f"add_tapered_pillars: n_slices must be >= 1, got {n_slices}.")
        nx = int(n_x) if n_x is not None else max(33, 4 * self.nox + 1)
        ny = int(n_y) if n_y is not None else max(33, 4 * self.noy + 1)
        eh = _C(eps_host)
        xs = (np.arange(nx) + 0.5) / nx
        ys = (np.arange(ny) + 0.5) / ny
        pil = [((float(c[0]), float(c[1])), (float(wt[0]), float(wt[1])),
                (float(wb[0]), float(wb[1])), _C(e))
               for c, wt, wb, e in pillars]

        def _profile(zeta):
            cell = np.full((nx, ny), eh, dtype=_C)
            covered = np.zeros((nx, ny), dtype=bool)
            for (cx, cy), (wxt, wyt), (wxb, wyb), e in pil:
                wxz = wxt + (wxb - wxt) * zeta
                wyz = wyt + (wyb - wyt) * zeta
                if wxz <= 0.0 or wyz <= 0.0:
                    continue
                dx = np.abs((xs - cx / self.period_x + 0.5) % 1.0 - 0.5)
                dy = np.abs((ys - cy / self.period_y + 0.5) % 1.0 - 0.5)
                m = (dx[:, None] < 0.5 * wxz / self.period_x) & \
                    (dy[None, :] < 0.5 * wyz / self.period_y)
                if np.any(m & covered):
                    raise ValueError(
                        "add_tapered_pillars: pillars overlap; merge or "
                        "separate them explicitly.")
                covered |= m
                cell[m] = e
            return cell

        return self.add_graded_layer(thickness, _profile, n_slices=n,
                                     rule="midpoint")

    def plot_geometry(self, ax=None, material_names=None):
        """Draw the stack cross-section as the solver sees it (device-geometry
        roadmap item 7): per-layer Re(eps) maps -- a 1-D stack renders an
        ``(x, z)`` cross-section; a 2-D stack renders the x-z cut at the
        cell's y midline (each layer's own pixel grid, no resampling).
        Dispersive layers render as hatched bands (no single-wavelength
        value).

        Returns
        -------
        matplotlib Axes -- use ``ax.figure.savefig(...)`` to save the figure.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        if ax is None:
            _fig, ax = plt.subplots(figsize=(6, 4))
        total = sum(float(L.thickness) for L in self._layers)
        z = 0.0
        nx = 256
        for L in self._layers:
            t = float(L.thickness)
            if getattr(L, "dispersive", False):
                ax.add_patch(Rectangle((0.0, -z - t), self.period_x, t,
                                       facecolor="none", edgecolor="0.4",
                                       hatch="//"))
                ax.text(0.02 * self.period_x, -z - 0.5 * t,
                        "dispersive (wl -> eps)", fontsize=7, va="center")
            else:
                if L.kind == "uniform":
                    row = np.full(nx, np.real(complex(L.data)))
                elif L.kind == "iso":
                    cell = np.asarray(L.data)
                    col = cell[:, cell.shape[1] // 2]
                    row = np.real(col[(np.arange(nx) * cell.shape[0])
                                      // nx])
                elif L.kind == "tensor":
                    cell = np.asarray(L.data)
                    col = cell[:, cell.shape[1] // 2, 0, 0]
                    row = np.real(col[(np.arange(nx) * cell.shape[0])
                                      // nx])
                else:                       # shapes: rasterize via the cut
                    eps_bg, shapes = L.data
                    xs = (np.arange(nx) + 0.5) / nx * self.period_x
                    ym = 0.5 * self.period_y
                    row = np.full(nx, np.real(complex(eps_bg)))
                    for sh in shapes:
                        cx, cy = sh.get("center", (self.period_x / 2,
                                                   self.period_y / 2))
                        if sh["shape"] == "rectangle":
                            wx, wy = sh["size"]
                            m = (np.abs(xs - cx) < 0.5 * wx) & \
                                (abs(ym - cy) < 0.5 * wy)
                        else:               # disk / ellipse
                            rx = sh.get("radius", sh.get("size", (0, 0))[0])
                            ry = sh.get("radius", sh.get("size", (0, 0))[-1])
                            m = ((xs - cx) / rx) ** 2 + \
                                ((ym - cy) / max(ry, 1e-300)) ** 2 < 1.0
                        row = np.where(m, np.real(complex(sh["eps"])), row)
                ax.imshow(row[None, :], aspect="auto", origin="upper",
                          extent=(0.0, self.period_x, -z - t, -z),
                          cmap="viridis")
            z += t
        ax.set_xlim(0.0, self.period_x)
        ax.set_ylim(-total, 0.0)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m] (0 = top)")
        ax.set_title("RCWAStack geometry (Re eps, y-midline cut)")
        return ax

    def set_source(self, wavelength, *, theta=None, phi=0.0, angle=None):
        """Set the incident plane wave (vacuum ``wavelength`` [m], polar
        ``theta`` and azimuth ``phi`` [rad]).  ``angle`` is accepted as a
        cross-suite alias for the polar ``theta`` (the 1-D ``angle`` /
        ``PMMStack.set_source`` spelling), with ``phi`` unaffected (the classical
        planar mount is ``phi = 0``).  To keep cross-suite substitution
        unambiguous, ``theta`` WINS when both are supplied -- the SAME rule as the
        1-D entry points and ``PMMStack.set_source`` (so ``set_source(angle=A,
        theta=T)`` resolves to ``T`` in every suite); pass just one in practice.

        The stack solver always returns the full zeroth-order Jones response
        (the reaction to both incident ``E_x`` and ``E_y``), so no incident
        polarization is selected here."""
        theta = _resolve_incidence(angle, theta)      # theta wins; falls back to angle
        if theta is None:
            theta = 0.0
        if any(is_jax_array(v) for v in (wavelength, theta, phi)):
            # D1 (AUDIT_DYNAMETA_CONSUMER_API_GAPS): traced source parameters
            # stay RAW so wavelength / angle gradients flow through the stack
            # twin (float() severed them; dispersion-engineering objectives
            # had to route to the PMM twin).  Validation needs concrete
            # values -- the traced solve documents its skipped guards.
            self._source = dict(wavelength=wavelength, theta=theta, phi=phi)
            return self
        _validate_geometry("RCWAStack.set_source", wavelength=wavelength)
        self._source = dict(wavelength=float(wavelength), theta=float(theta),
                            phi=float(phi))
        return self

    def _layer_eps_reals(self):
        """Real permittivities present in the layers (for the Wood-anomaly
        grazing nudge)."""
        vals = []
        for L in self._layers:
            if L.kind == "uniform":
                vals.append(float(np.real(_C(L.data))))
            elif L.kind == "iso":
                r = np.real(L.data)
                vals += [float(r.min()), float(r.max())]
            elif L.kind == "shapes":
                bg, shapes = L.data
                vals.append(float(np.real(_C(bg))))
                vals += [float(np.real(_C(s["eps"]))) for s in shapes]
            else:  # tensor
                d = np.real(L.data[:, :, [0, 1, 2], [0, 1, 2]])
                vals += [float(d.min()), float(d.max())]
        return vals

    def _stack_lossless(self):
        """True when every region index and layer permittivity is exactly real
        (provably lossless -> the closure ``sum(R)+sum(T) = 1`` is exact here,
        so :func:`_check_energy` can police the silent 1e-6..0.05 window).
        Mirrors :meth:`_layer_eps_reals`' per-kind layer enumeration but tests
        losslessness (``Im == 0``) rather than keeping the real parts, then
        delegates to the shared :func:`_cell_lossless` predicate.  Only reached
        on the concrete (non-JAX) solve path, after the dispersive-callable
        guard, so ``n_superstrate / n_substrate`` are concrete here."""
        eps_sup = complex(np.conj(_C(self.n_superstrate) ** 2))
        eps_sub = complex(np.conj(_C(self.n_substrate) ** 2))
        arrs = []
        for L in self._layers:
            if L.kind == "shapes":
                bg, shapes = L.data
                arrs.append(bg)
                arrs += [s["eps"] for s in shapes]
            else:  # uniform / iso / tensor: L.data is the eps scalar / array
                arrs.append(L.data)
        return _cell_lossless(eps_sup, eps_sub, *arrs)

    def _materialized_layers(self, wl, layers=None):
        """Concrete layer list at one wavelength: every DISPERSIVE
        (``wl -> value``) layer spec is resolved and validated; non-dispersive
        layers pass through by reference (their eig-dedup keys are unchanged
        across the sweep).  ``layers`` defaults to the stack's own list; the
        sweep passes its saved BASE list explicitly (inside the sweep loop
        ``self._layers`` already holds the previous wavelength's concrete
        materialisation)."""
        out = []
        for L in (self._layers if layers is None else layers):
            if not getattr(L, "dispersive", False):
                out.append(L)
                continue
            if L.kind == "uniform":
                out.append(_RCWALayer(L.thickness, "uniform", _C(L.data(wl))))
            elif L.kind == "iso":
                cell = np.asarray(L.data(wl), dtype=_C)
                if cell.ndim == 1:
                    cell = cell[:, None]
                _validate_cell_sampling("solve_vs_wavelength", cell, self.nox,
                                        self.noy,
                                        strict_y=not self.is_1d)
                _warn_if_y_averaged("solve_vs_wavelength", cell, self.noy,
                                    strict_y=not self.is_1d, stacklevel=5)
                out.append(_RCWALayer(L.thickness, "iso", cell,
                                      formulation=L.formulation))
            elif L.kind == "shapes":
                bg, shapes = L.data
                bgv = _C(bg(wl)) if callable(bg) else _C(bg)
                sh = [dict(d, eps=(_C(d["eps"](wl)) if callable(d["eps"])
                                   else _C(d["eps"]))) for d in shapes]
                _validate_shapes("solve_vs_wavelength", sh, self.period_x,
                                 self.period_y)
                out.append(_RCWALayer(L.thickness, "shapes", (bgv, sh)))
            else:
                tcell = np.asarray(L.data(wl), dtype=_C)
                _validate_cell_sampling("solve_vs_wavelength", tcell, self.nox,
                                        self.noy,
                                        strict_y=not self.is_1d)
                _warn_if_y_averaged("solve_vs_wavelength", tcell, self.noy,
                                    strict_y=not self.is_1d, stacklevel=5)
                # Mirror the STATIC add_layer tensor contract (audit P3-38):
                # out-of-plane tensors are supported since v5.14.1, so the
                # materialised dispersive cell must not re-impose the old
                # in-plane-only restriction -- only the nonzero-e_zz guard
                # (the pointwise ezz-Schur fold divides by it) applies.
                if _tensor_offplane_present(tcell):
                    ezz_min = float(np.min(np.abs(tcell[..., 2, 2])))
                    if ezz_min < 1e-12:
                        raise ValueError(
                            "solve_vs_wavelength: an out-of-plane dispersive "
                            "tensor layer requires |e_zz| > 0 everywhere "
                            "(the E_z elimination divides by it); min "
                            f"|e_zz| = {ezz_min:.3e} at wavelength "
                            f"{wl:.6g} m.")
                out.append(_RCWALayer(L.thickness, "tensor", tcell))
        return out

    def solve_vs_wavelength(self, wavelengths, *, stabilize=False,
                            max_workers=None, blas_per_worker=1):
        """Solve the stack across a wavelength sweep (audit GAP5, v5.14.1).

        Calls :meth:`set_source`'s stored ``(theta, phi)`` at each wavelength,
        materialising any DISPERSIVE (``wl -> value``) layer or region-index
        callables per wavelength.  An unstable wavelength (the loud
        ``_EnergyError`` guard) is returned as a NaN row and counted in a
        single summary warning -- one bad point must not abort the sweep.

        The per-wavelength solves are INDEPENDENT (each on a private clone of
        the stack) and NumPy releases the GIL inside LAPACK, so they run on a
        bounded thread pool: ``max_workers=None`` (default) picks
        ``min(cpu_count, n_wl)`` on the NumPy backend and 1 on GPU/JAX; each
        worker pins its BLAS pool to ``blas_per_worker`` threads so the product
        stays within the core count.  Results are stored by index, so the output
        is BYTE-IDENTICAL to a serial sweep regardless of worker count.  Pass
        ``max_workers=1`` to force the serial path.

        Returns
        -------
        orders : (N,) or (N, 2) int ndarray
            The requested order set (fixed across the sweep).
        R, T : (n_wl, 2, N) float ndarray
            Reflected / transmitted efficiencies per wavelength, per incident
            polarization (TE row 0, TM row 1 -- the :meth:`solve` layout).
        jones : (n_wl, 2, 2) complex ndarray
            Zeroth-order Jones reflection per wavelength.

        With ``stabilize=True`` a per-wavelength consensus solve may return a
        DIFFERENT order count; its efficiencies are aligned onto the requested
        order set (orders it did not compute stay NaN).
        """
        if self._source is None:
            raise ValueError(
                "RCWAStack.solve_vs_wavelength: call set_source first (the "
                "wavelength passed there is ignored in favour of the sweep).")
        if not self._layers:
            raise ValueError(
                "RCWAStack.solve_vs_wavelength: add at least one layer.")
        wl_arr = np.atleast_1d(np.asarray(wavelengths, dtype=float))
        orders_ref, N = _harmonic_orders_2d(
            self.nox, self.noy, truncation=self.truncation,
            period_x=self.period_x, period_y=self.period_y)
        # key in the SAME form the per-wavelength results report their orders
        # (1-D stacks return a flat (N,) order vector)
        ret_orders = (orders_ref[:, 0] if self.is_1d else orders_ref)
        key_to_col = {_order_key(o): j
                      for j, o in enumerate(np.asarray(ret_orders))}
        R_out = np.full((wl_arr.size, 2, N), np.nan)
        T_out = np.full((wl_arr.size, 2, N), np.nan)
        J_out = np.full((wl_arr.size, 2, 2), np.nan, dtype=_C)
        base_layers = self._layers
        base_ns, base_nb = self.n_superstrate, self.n_substrate
        base_src = self._source
        n_unstable = 0

        def _solve_one(i, w):
            # Independent solve on a PRIVATE shallow clone: shares read-only
            # config but gets its own _layers/_source/half-space indices, so no
            # worker mutates shared self.  RCWAStack holds no instance cache
            # (the only shared cache, _HOMOG_CACHE, is lock-guarded and its
            # recompute is byte-identical across any interleaving), so copy.copy
            # is safe here.
            w = float(w)
            sub = _copy.copy(self)
            sub._layers = self._materialized_layers(w, base_layers)
            sub.n_superstrate = base_ns(w) if callable(base_ns) else base_ns
            sub.n_substrate = base_nb(w) if callable(base_nb) else base_nb
            sub._source = dict(base_src, wavelength=w)
            try:
                # the BLAS cap is THREAD-LOCAL, so each worker sets its own.
                with _blas_threads_quiet(blas_per_worker):
                    return i, sub.solve(stabilize=stabilize)
            except _EnergyError:
                return i, None

        def _store(i, res):
            nonlocal n_unstable
            if res is None:
                n_unstable += 1
                return
            o_i, R_i, T_i = res.efficiencies()
            o_i = np.asarray(to_numpy(o_i))
            R_i = np.asarray(to_numpy(R_i))
            T_i = np.asarray(to_numpy(T_i))
            cols = [key_to_col.get(_order_key(o)) for o in o_i]
            for j_src, j_dst in enumerate(cols):
                if j_dst is not None:
                    R_out[i, :, j_dst] = R_i[:, j_src]
                    T_out[i, :, j_dst] = T_i[:, j_src]
            J_out[i] = np.asarray(to_numpy(res.jones_reflection()))

        # completion order is irrelevant (results stored by index i in the main
        # thread) -> byte-identical to serial.  GPU (shared CUDA stream) and JAX
        # (tracing) stay serial.
        _jax_any = any(is_jax_array(getattr(L, "data", None))
                       for L in base_layers)
        if max_workers is None:
            max_workers = (1 if (self.use_gpu or _jax_any)
                           else min(os.cpu_count() or 1, int(wl_arr.size)))
        max_workers = max(1, int(max_workers))
        if max_workers == 1 or wl_arr.size == 1:
            for i, w in enumerate(wl_arr):
                _store(*_solve_one(i, w))
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as _ex:
                for i, res in _ex.map(lambda iw: _solve_one(*iw),
                                      list(enumerate(wl_arr))):
                    _store(i, res)
        if n_unstable:
            warnings.warn(
                f"RCWAStack.solve_vs_wavelength: {n_unstable} of "
                f"{wl_arr.size} wavelengths were numerically unstable "
                f"(energy guard) and returned NaN; pass stabilize=True or "
                f"adjust n_orders.", stacklevel=2)
        return ret_orders, R_out, T_out, J_out

    def _layer_even_spec(self, layer, Kx, Ky, orders, xp):
        """One layer's ``("uniform", eps)`` / ``("PQ", P, Q, probe)`` spec for
        the even-parity cascade (backlog A1, 2026-06-10), or ``None`` if the
        layer kind cannot fold (out-of-plane / dispersive)."""
        if getattr(layer, "dispersive", False):
            return None
        if layer.kind == "uniform":
            return ("uniform", complex(np.conj(layer.data)))
        if layer.kind == "iso":
            cell_c = xp.conj(xp.asarray(layer.data))
            EPS = _eps_convolution_2d(cell_c, orders, self.nox, self.noy)
            if getattr(layer, "formulation", "laurent") == "li":
                e_np = to_numpy(cell_c)
                scale = max(1.0, float(np.max(np.abs(e_np))))
                if float(np.max(np.abs(e_np - e_np.flat[0]))) > 1e-12 * scale:
                    Cxx, Cyy = _li_convolutions_2d(cell_c, orders, self.nox,
                                                   self.noy, xp)
                    Z = xp.zeros_like(Cxx)
                    P, Q = _tensor_PQ(Kx, Ky, Cxx, Z, Z, Cyy, EPS, xp)
                    return ("PQ", P, Q, EPS)
            P, Q = _scalar_PQ(Kx, Ky, EPS, EPS, None, xp)
            return ("PQ", P, Q, EPS)
        if layer.kind == "shapes":
            eps_bg, shapes = layer.data
            shapes_c = [dict(sh, eps=complex(np.conj(_C(sh["eps"]))))
                        for sh in shapes]
            EPS_np, _ = _analytic_convolutions_2d(   # [[1/eps]] unused (M10)
                complex(np.conj(_C(eps_bg))), shapes_c, orders, self.nox,
                self.noy, self.period_x, self.period_y)
            EPS = xp.asarray(EPS_np)
            P, Q = _scalar_PQ(Kx, Ky, EPS, EPS, None, xp)
            return ("PQ", P, Q, EPS)
        # tensor: in-plane only (OOP cannot fold)
        if _layer_offplane_or_traced(layer.data):
            return None
        et = xp.conj(xp.asarray(layer.data))

        def cv(comp):
            return _eps_convolution_2d(comp, orders, self.nox, self.noy)
        P, Q = _tensor_PQ(Kx, Ky, cv(et[:, :, 0, 0]), cv(et[:, :, 0, 1]),
                          cv(et[:, :, 1, 0]), cv(et[:, :, 1, 1]),
                          cv(et[:, :, 2, 2]), xp)
        return ("PQ", P, Q, cv(et[:, :, 0, 0]))

    def _layer_modes(self, layer, Kx, Ky, orders):
        """Layer eigenmodes ``(W, V, lam, EPS)``.  ``EPS`` is the wall-tangential
        ``[[eps]]`` convolution (``EZZ`` for a tensor layer) in the INTERNAL
        convention -- retained for the curl-H ``E_z`` solve in the internal-field
        reconstruction; ignored on the (default) far-field path."""
        xp = array_namespace(Kx)
        if layer.kind == "uniform":
            # D1: a TRACED uniform eps stays raw (complex() would sever the
            # gradient); _homogeneous_eigenmodes is backend-generic, so the
            # analytic uniform modes trace -- no lifted-cell eigensolve.
            eps_u = (xp.conj(layer.data) if is_jax_array(layer.data)
                     else complex(np.conj(layer.data)))
            W, V, kz = _homogeneous_eigenmodes(Kx, Ky, eps_u)
            lam = _sqrt_decay(-xp.concatenate([kz, kz]) ** 2)
            EPS = eps_u * xp.eye(Kx.shape[0], dtype=_C)
            return W, V, lam, EPS
        if layer.kind == "iso":
            cell_c = xp.conj(xp.asarray(layer.data))
            EPS = _eps_convolution_2d(cell_c, orders, self.nox, self.noy)
            if getattr(layer, "formulation", "laurent") == "li":
                # Li-1997 sequential rule (audit GAP3, v5.14.1): rigorous
                # fast-TM/metal factorization per layer; a 1-D stack cell
                # (Sy = 1) reduces exactly to the 1-D 'li' placement.  A
                # uniform cell falls through to the scalar path (all rules
                # coincide; its analytic uniform modes avoid degenerate eig).
                e_np = to_numpy(cell_c)
                scale = max(1.0, float(np.max(np.abs(e_np))))
                if float(np.max(np.abs(e_np - e_np.flat[0]))) > 1e-12 * scale:
                    Cxx, Cyy = _li_convolutions_2d(cell_c, orders, self.nox,
                                                   self.noy, xp)
                    Zli = xp.zeros_like(Cxx)
                    W, V, lam = _layer_eigenmodes_tensor(Kx, Ky, Cxx, Zli,
                                                         Zli, Cyy, EPS)
                    return W, V, lam, EPS
            W, V, lam = _layer_eigenmodes(Kx, Ky, EPS, EPS)
            return W, V, lam, EPS
        if layer.kind == "shapes":
            # Analytic (host) form factors -> move the convolution to backend.
            eps_bg, shapes = layer.data
            shapes_c = [dict(s, eps=complex(np.conj(_C(s["eps"])))) for s in shapes]
            EPS_np, _ = _analytic_convolutions_2d(   # [[1/eps]] unused (M10)
                complex(np.conj(_C(eps_bg))), shapes_c, orders, self.nox,
                self.noy, self.period_x, self.period_y)
            EPS = xp.asarray(EPS_np)
            # Direct-rule E_z elimination (v5.14.1 audit F1: the dual-Laurent
            # [[1/eps]] z-rule is the wrong factorization for a z-invariant
            # layer -- E_z is wall-tangential; see rcwa_efficiency_2d_shapes).
            W, V, lam = _layer_eigenmodes(Kx, Ky, EPS, EPS)
            return W, V, lam, EPS
        et = xp.conj(xp.asarray(layer.data))

        def cv(comp):
            return _eps_convolution_2d(comp, orders, self.nox, self.noy)
        EZZ = cv(et[:, :, 2, 2])
        if _layer_offplane_or_traced(layer.data):
            # full-3x3 layer (v5.14.1): pointwise ezz-Schur fold on the CELL
            # before the direct-rule convolution + the 6-tuple generator --
            # the rcwa_jones_2d GAP2 machinery, per stack layer.
            inv_ezz = 1.0 / et[:, :, 2, 2]
            exz = et[:, :, 0, 2]
            eyz = et[:, :, 1, 2]
            ezx = et[:, :, 2, 0]
            ezy = et[:, :, 2, 1]
            Wf, Vf, lf, Wb, Vb, lb = _layer_eigenmodes_tensor(
                Kx, Ky,
                cv(et[:, :, 0, 0] - exz * ezx * inv_ezz),
                cv(et[:, :, 0, 1] - exz * ezy * inv_ezz),
                cv(et[:, :, 1, 0] - eyz * ezx * inv_ezz),
                cv(et[:, :, 1, 1] - eyz * ezy * inv_ezz), EZZ,
                EZX=cv(ezx), EZY=cv(ezy), EXZ=cv(exz), EYZ=cv(eyz))
            return ("gen", Wf, Vf, lf, Wb, Vb, lb, EZZ)
        W, V, lam = _layer_eigenmodes_tensor(
            Kx, Ky, cv(et[:, :, 0, 0]), cv(et[:, :, 0, 1]),
            cv(et[:, :, 1, 0]), cv(et[:, :, 1, 1]), EZZ)
        return W, V, lam, EZZ

    @staticmethod
    def _layer_eig_key(layer):
        """Content key identifying a layer's eigenproblem (the layer's
        permittivity + kind; the layer eig is THICKNESS-INDEPENDENT and, within
        one solve, ``Kx`` / ``Ky`` are shared).  Two layers with the same key
        have identical modes, so a repeated layer (a DBR / Bragg period) is
        solved once.  ``None`` -> not dedupable (e.g. a traced JAX array)."""
        kind = layer.kind
        data = layer.data
        if kind == "uniform":
            if is_jax_array(data):
                return None                    # traced uniform eps: no dedup
            return ("uniform", complex(data))
        if kind == "shapes":
            eps_bg, shapes = data
            # PER-SHAPE and ORDER-PRESERVING (audit M1, 2026-07-25).  Flattening
            # EVERY shape's (key, repr) pairs into ONE sorted multiset made
            # structurally DIFFERENT shape lists COLLIDE: two disks of different
            # radii with their centres EXCHANGED carry the same multiset of
            # pairs, so the second layer silently reused the first layer's
            # eigenmodes -- measured, an (A, B) stack returned bit-identically
            # the (A, A) answer (3.7x relative error on the zeroth-order R
            # against a rasterised eps_cell oracle) while conserving energy to
            # 6e-15, i.e. invisible to the closure guard.  Each shape now
            # contributes its OWN sorted tuple (so the same shape dict written
            # with its keys in a different ORDER still dedupes) and the shape
            # LIST order is kept.  A PERMUTED list is physically identical (the
            # analytic form-factor sum is commutative) and now re-solves --
            # conservative, and deliberately so: the accumulation order of the
            # form factors is not commutative in floating point, so a
            # permutation-invariant key would hand back modes that differ in the
            # last bits from that layer's own solve.  ``repr`` keeps the key
            # hashable (a 'size' / 'center' may be a list) and is faithful for
            # the scalar geometry / eps fields the form factors read.
            return ("shapes", complex(eps_bg),
                    tuple(tuple(sorted((k, repr(v)) for k, v in s.items()))
                          for s in shapes))
        try:                                   # iso / tensor: hash the cell
            arr = np.ascontiguousarray(to_numpy(data))
        except Exception:                      # traced array -> no dedup
            return None
        return (kind, getattr(layer, "formulation", "laurent"), arr.shape,
                str(arr.dtype), arr.tobytes())

    @_with_blas_limit
    def solve(self, *, retain_internal=False, stabilize=False,
              symmetry="auto") -> RCWAResult:
        """Solve the stack -> :class:`RCWAResult`.

        ``stabilize=True`` (opt-in; default ``False``) guards against the
        ISOLATED-RESONANCE spikes a sharp-resonant metal MULTILAYER can show: a
        near-singular layer<->layer / layer<->region mode-match at isolated
        ``n_orders`` biases a SINGLE diffraction order (e.g. a reflection null)
        non-monotonically while total power stays bounded -- the same pathology
        fixed for :func:`pmm_efficiency_1d` (v5.10.6).  It re-solves over a short
        ``n_orders`` window at or below the requested count (so the cell sampling,
        sized for ``n_orders``, never aliases) and returns the solve whose
        ZERO-ORDER R/T is in the consensus cluster (the spikes are the outliers;
        the consensus is PER-ORDER because total power does not move).  If no
        plateau forms (genuinely under-resolved, not an isolated spike) it warns
        and returns the requested ``n_orders``.  It costs
        ``_STACK_STABILIZE_WINDOW`` full stack solves, so it is opt-in; clean
        (non-resonant) stacks are unaffected and ``stabilize=False`` is the exact
        single solve.

        ``retain_internal=True`` additionally retains the per-layer modes and the
        cumulative partial S-matrices needed to reconstruct the in-structure E/H
        field via :meth:`RCWAResult.internal_field` (and the per-layer
        absorption via :meth:`RCWAResult.layer_absorption`).  It costs an extra
        ``O(n_layers)`` star-product sweep and keeps the per-layer ``2N x 2N``
        matrices, so it is off by default for hot efficiency sweeps.  NumPy /
        CuPy only.

        ``symmetry='auto'`` (the DEFAULT; ``True`` is equivalent) attempts the
        EVEN-PARITY fast path at NORMAL incidence: when every layer is
        centro-symmetric about one common centre (verified numerically per layer
        on the recentred operators), the whole cascade is solved in the
        ``(N+1)``-dimensional even sector instead of ``2N`` -- measured ~x4.5 on
        a 4-layer mixed stack.  Pass ``symmetry=False`` for the exact full-solve
        bits (the even-adapted basis matches to ~1e-12, not bit-for-bit).  Uniform, pixel-cell
        (``'laurent'`` and ``'li'``), analytic-shapes and IN-PLANE tensor layers
        all fold; ANY failed precondition (oblique incidence, an out-of-plane
        tensor or dispersive layer, layers symmetric about DIFFERENT centres,
        ``retain_internal=True``, JAX backend) falls back to the full solve
        BIT-IDENTICALLY.  NB a pixel cell's feature centre lies on the
        half-pixel grid -- mixing it with an analytic shape at exactly
        ``period/2`` is a genuine centre mismatch and falls back."""
        if not stabilize:
            return self._solve_once(retain_internal=retain_internal,
                                    symmetry=symmetry)
        import warnings
        base_nox, base_noy = self.nox, self.noy

        def _set(nox):
            self.nox = nox
            self.noy = (base_noy if self.is_1d
                        else max(1, base_noy - (base_nox - nox)))

        # DOWNWARD window: distinct n_orders from the requested count down, so the
        # cell sampling (sized for the requested n_orders) NEVER aliases.  The
        # spikes are isolated, so the remaining counts form the consensus cluster.
        window = sorted({max(1, base_nox - b)
                         for b in range(_STACK_STABILIZE_WINDOW)}, reverse=True)
        results, feats = [], []
        try:
            for nox in window:
                _set(nox)
                try:
                    res = self._solve_once(retain_internal=False,
                                           symmetry=symmetry)
                except _EnergyError:
                    continue          # a windowed low-order count that trips the
                    #                   energy guard is SKIPPED, not fatal (audit P2)
                results.append(res)
                feats.append(_zero_order_efficiency(res))
        finally:
            self.nox, self.noy = base_nox, base_noy
        if not results:
            raise _EnergyError(
                f"RCWAStack.solve(stabilize=True): every n_orders in the window "
                f"{window} tripped the energy guard; the stack is numerically "
                f"unstable here -- adjust n_orders / period / index contrast.")
        cluster = _largest_feature_cluster(feats, _STACK_STABILIZE_TOL)
        # A genuine convergence plateau is FLAT.  An under-resolved cell biases the
        # low-order solves the SAME direction, so they still cluster within TOL but
        # TREND monotonically -- that is NOT a consensus (audit P1-B: this silently
        # returned min(cluster), a confidently-wrong value).  Flag it too.
        trending = _cluster_is_trending(feats, cluster, _STACK_STABILIZE_TOL)
        if len(cluster) < 2 or trending:
            reason = ("shows a MONOTONE trend (no flat convergence plateau -- the "
                      "low-order solves are all biased the same way and merely "
                      "cluster)" if trending else
                      "shows no consensus (no convergence plateau, not just an "
                      "isolated spike)")
            warnings.warn(
                f"RCWAStack.solve(stabilize=True): the n_orders window {window} "
                f"{reason}; the stack is likely UNDER-RESOLVED -- raise n_orders / "
                "n_orders_y (and the cell sampling).  Returning the requested "
                "n_orders.", stacklevel=2)
            chosen = 0                        # results[0] = requested (highest) n
        else:
            chosen = min(cluster)             # highest-n_orders consensus solve
        if not retain_internal:
            return results[chosen]
        # re-solve the consensus n_orders WITH the internal-field data retained
        try:
            _set(window[chosen])
            return self._solve_once(retain_internal=True)
        finally:
            self.nox, self.noy = base_nox, base_noy

    def _solve_once(self, *, retain_internal=False,
                    symmetry="auto") -> RCWAResult:
        """Inner single-``n_orders`` stack solve (the public :meth:`solve` body;
        ``stabilize`` scans a window of these).

        ``retain_internal=True`` additionally retains the per-layer modes and the
        cumulative partial S-matrices needed to reconstruct the in-structure E/H
        field via :meth:`RCWAResult.internal_field` (and the per-layer
        absorption via :meth:`RCWAResult.layer_absorption`).  It costs an extra
        ``O(n_layers)`` star-product sweep and keeps the per-layer ``2N x 2N``
        matrices, so it is off by default for hot efficiency sweeps.  NumPy /
        CuPy only."""
        if self._source is None:
            raise ValueError("RCWAStack.solve: call set_source first.")
        if not self._layers:
            raise ValueError("RCWAStack.solve: add at least one layer.")
        if any(getattr(L, "dispersive", False) for L in self._layers) or \
                callable(self.n_superstrate) or callable(self.n_substrate):
            raise ValueError(
                "RCWAStack.solve: the stack holds DISPERSIVE (wl -> value) "
                "materials, which have no value at a single set_source "
                "wavelength alone; use solve_vs_wavelength(wavelengths) "
                "(it materialises every callable per wavelength).")
        src = self._source
        wl, theta, phi = src["wavelength"], src["theta"], src["phi"]
        # Dispatch the backend off the patterned-layer arrays, the traced
        # UNIFORM eps slots, and the (possibly traced) source geometry (D1:
        # a traced wavelength / angle / uniform eps makes the whole stack
        # solve differentiable, not just the patterned-cell values).
        layer_arrays = [L.data for L in self._layers if L.kind in ("iso",
                                                                    "tensor")]
        layer_arrays += [L.data for L in self._layers
                         if L.kind == "uniform" and is_jax_array(L.data)]
        layer_arrays += [v for v in (wl, theta, phi) if is_jax_array(v)]
        src_traced = any(is_jax_array(v) for v in (wl, theta, phi))
        xp = _rcwa_xp("RCWAStack.solve", self.use_gpu, *layer_arrays)
        bname = backend_name(xp)
        is_jax = bname == "jax"
        if is_jax:
            _require_jax_x64("RCWAStack.solve")
        orders, N = _harmonic_orders_2d(
            self.nox, self.noy, truncation=self.truncation,
            period_x=self.period_x, period_y=self.period_y)
        eps_sup = complex(np.conj(_C(self.n_superstrate) ** 2))
        eps_sub = complex(np.conj(_C(self.n_substrate) ** 2))
        nre = float(np.real(np.sqrt(eps_sup)))
        if src_traced:
            # np.sin materialises a tracer; xp == jnp here by dispatch (D1)
            kx0 = nre * xp.sin(theta) * xp.cos(phi)
            ky0 = nre * xp.sin(theta) * xp.sin(phi)
        else:
            kx0 = nre * np.sin(theta) * np.cos(phi)
            ky0 = nre * np.sin(theta) * np.sin(phi)
        if not src_traced:
            # (a traced theta/phi cannot feed the concrete guard; the traced
            # solve documents its skipped guards -- D1)
            _require_propagating_incidence("RCWAStack.solve", eps_sup,
                                           kx0 ** 2 + ky0 ** 2,
                                           warn_lossy=True)
            # The traced (JAX) layer permittivities cannot feed the concrete
            # grazing nudge, so on the differentiable path the nudge sees only
            # the region indices (the dominant region Rayleigh anomaly is
            # still caught).  A traced WAVELENGTH skips the nudge entirely
            # (data-dependent control flow; keep the sweep off exact Rayleigh
            # cutoffs yourself when differentiating through wl).
            eps_reals = ([eps_sup, eps_sub] if is_jax
                         else [eps_sup, eps_sub] + self._layer_eps_reals())
            wl = _grazing_safe_wavelength(
                wl, kx0, ky0, orders[:, 0], orders[:, 1], self.period_x,
                self.period_y, eps_reals)
        k0 = 2.0 * np.pi / wl
        kx = kx0 + orders[:, 0] * (wl / self.period_x)
        ky = ky0 + orders[:, 1] * (wl / self.period_y)
        if src_traced:
            kxv = xp.asarray(kx).astype(_C)
            kyv = xp.asarray(ky).astype(_C)
            Kx = xp.diag(kxv)
            Ky = xp.diag(kyv)
        else:
            Kx = xp.asarray(np.diag(kx.astype(_C)))
            Ky = xp.asarray(np.diag(ky.astype(_C)))
            kxv = xp.asarray(kx.astype(_C))
            kyv = xp.asarray(ky.astype(_C))
        # n_superstrate MUST be part of the key for BOTH region caches: the
        # substrate modes depend on Kx/Ky whose kx0 = Re(sqrt(eps_sup))*... ,
        # so two stacks with the same n_substrate but different n_superstrate
        # have DIFFERENT substrate modes.  Omitting it caused a cache
        # collision (33-40% spurious energy gain on oblique sweeps).  The
        # backend name is in the key too so a NumPy and a CuPy solve of the
        # same geometry never alias to each other's (wrong-device) modes.
        # A TRACED source has no hashable key -> compute uncached (D1).
        if src_traced:
            Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
            Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)
        else:
            # truncation is in the key (audit S1-20): a 'rectangular' and a
            # 'circular' stack with the same nox/noy have DIFFERENT order sets
            # (hence different Kx/Ky and homogeneous modes) -- omitting it would
            # collide them in the module-level mode cache.
            geom = (self.nox, self.noy, wl, theta, phi, self.period_x,
                    self.period_y, self.n_superstrate, bname, self.truncation)
            Wref, Vref, kz_ref = _cached_homogeneous_eigenmodes(
                eps_sup, Kx, Ky, ("sup", self.n_superstrate) + geom)
            Wtrn, Vtrn, kz_trn = _cached_homogeneous_eigenmodes(
                eps_sub, Kx, Ky, ("sub", self.n_substrate) + geom)

        # Eig reuse (v5.6): the per-layer modal eig is the dominant cost and is
        # THICKNESS-INDEPENDENT, so two layers with identical permittivity (a
        # repeated DBR / Bragg period, a metamaterial supercell) share one eig
        # instead of recomputing it.  Bit-exact -- it memoises a pure function;
        # a None key (traced JAX array) falls back to per-layer solves.
        # even-parity fast path (backlog A1): a NORMAL-INCIDENCE stack of
        # jointly centro-symmetric in-plane layers runs the WHOLE cascade in
        # the (N+1)-d even sector (~x3-4); any failed precondition falls
        # through to the full solve bit-identically.
        sym_rt = None
        if (_symmetry_on(symmetry) and not retain_internal and bname != "jax"
                and abs(kx0) < 1e-12 and abs(ky0) < 1e-12):
            specs = [self._layer_even_spec(L, Kx, Ky, orders, xp)
                     for L in self._layers]
            if all(sp is not None for sp in specs):
                delta_s = xp.asarray(((orders[:, 0] == 0)
                                      & (orders[:, 1] == 0)).astype(_C))
                cincs = [xp.concatenate([1.0 * delta_s, 0.0 * delta_s]),
                         xp.concatenate([0.0 * delta_s, 1.0 * delta_s])]
                depths = [float(L.thickness) for L in self._layers]
                sym_rt = _symmetric_cascade_rt(
                    Vref, Vtrn, Kx, Ky, specs, depths, k0, cincs, orders, xp)
                if sym_rt is not None:
                    # UNDO the recentering gauge (audit P2-18) so the stored
                    # per-order AMPLITUDES match the full-solve phases.  The
                    # cascade conjugates every layer operator by the diagonal
                    # gauge D2 = diag(d2) (the half-space blocks are diagonal
                    # and commute), so it returns r' = D2^{-1} S11 D2 cinc;
                    # cinc lives on the (0, 0) order where d = 1, hence the
                    # true amplitudes are r = d2 * r' (|d2| = 1: every
                    # efficiency and the zeroth-order Jones are unchanged).
                    # Same probe -> same deterministic d as the cascade used.
                    probe0 = next(sp[3] for sp in specs if sp[0] == "PQ")
                    d_g = _recentering_phase(probe0, orders, xp)
                    if d_g is not None:
                        d2_g = xp.concatenate([d_g, d_g])
                        sym_rt = [(d2_g * r, d2_g * t) for r, t in sym_rt]

        _mode_cache = {}
        modes = []
        for L in ([] if sym_rt is not None else self._layers):
            key = self._layer_eig_key(L)
            cached = _mode_cache.get(key) if key is not None else None
            if cached is None:
                cached = self._layer_modes(L, Kx, Ky, orders)
                if key is not None:
                    # Freeze before sharing (audit W7-B, the M9 class): the
                    # SAME tuple object is appended for every layer with an
                    # identical eig key, so with retain_internal=True a
                    # 3-period DBR has info['W'][0] IS info['W'][2] IS
                    # info['W'][4] (measured).  A write through one layer's
                    # view silently rewrote the others.
                    cached = tuple(_readonly(a) for a in cached)
                    _mode_cache[key] = cached
            modes.append(cached)
        any_oop = any(isinstance(m, tuple) and len(m) == 8 and m[0] == "gen"
                      for m in modes)
        if sym_rt is not None:
            S = None                          # even sector already solved
        elif not any_oop:
            W0, V0, lam0, _e0 = modes[0]
            S = _interface_smatrix(Wref, Vref, W0, V0)
            S = _propagation_star(S, lam0, k0 * self._layers[0].thickness)
            for i in range(1, len(modes)):
                Wp, Vp, _lp, _ = modes[i - 1]
                Wc, Vc, lamc, _ = modes[i]
                S = _redheffer_star(S, _interface_smatrix(Wp, Vp, Wc, Vc))
                S = _propagation_star(S, lamc, k0 * self._layers[i].thickness)
            Wl, Vl, _ll, _el = modes[-1]
            S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
        else:
            # GENERALIZED cascade (v5.14.1): any out-of-plane tensor layer
            # breaks the [W; -V] <-> -lam symmetry, so the whole stack is
            # promoted -- symmetric layers/half-spaces enter as
            # [W, W; V, -V] blocks with (lam, -lam), the generator layers
            # with their explicit forward/backward sets (the PMM2DStack
            # any_oop pattern; the single-layer machinery is rcwa_jones_2d's
            # GAP2 path).  retain_internal=True (loose-ends round 2026-07-14,
            # the Berreman-C2 port): the bracketing partial cascades are
            # built with the SAME generalized interface/propagation
            # convention and the per-layer ASYMMETRIC mode sets are retained
            # explicitly -- RCWAResult's recovery/field/absorption consumers
            # branch on info['general'].

            def _gblocks(m):
                if isinstance(m, tuple) and len(m) == 8 and m[0] == "gen":
                    _k, Wf, Vf, lf, Wb, Vb, lb, _e = m
                    return _modes_to_M(Wf, Vf, Wb, Vb), lf, lb
                W, V, lam, _e = m
                return _modes_to_M(W, V, W, -V), lam, -lam
            M_prev = _modes_to_M(Wref, Vref, Wref, -Vref)
            S = None
            gen_ifc, gen_prop = [], []
            for i, m in enumerate(modes):
                Ml, lf, lb = _gblocks(m)
                Si = _interface_smatrix_general(M_prev, Ml)
                S = Si if S is None else _redheffer_star(S, Si)
                S = _propagation_star_general(
                    S, lf, lb, k0 * self._layers[i].thickness)
                M_prev = Ml
                if retain_internal:
                    gen_ifc.append(Si)
                    gen_prop.append(_propagation_smatrix_general(
                        lf, lb, k0 * self._layers[i].thickness))
            Msub = _modes_to_M(Wtrn, Vtrn, Wtrn, -Vtrn)
            ifc_sub = _interface_smatrix_general(M_prev, Msub)
            S = _redheffer_star(S, ifc_sub)
            if retain_internal:
                gen_ifc.append(ifc_sub)
        if S is not None:
            S11, _S12, S21, _S22 = S

        p0 = int(np.where((orders[:, 0] == 0) & (orders[:, 1] == 0))[0][0])
        delta = xp.asarray(((orders[:, 0] == 0) & (orders[:, 1] == 0)).astype(_C))
        kz_inc = (xp.real(_sqrt_forward(np.conj(eps_sup) - kx0 ** 2 - ky0 ** 2))
                  if src_traced else
                  float(np.real(_sqrt_forward(np.conj(eps_sup)
                                              - kx0 ** 2 - ky0 ** 2))))
        # PUBLIC-convention forward kz for the z-flux + mask + Ez (see
        # rcwa._core._forward_flux_kz): a lossy exit substrate would otherwise
        # have its transmittance silently zeroed by the Re(kz) > 0 mask.
        kz_ref_f = _forward_flux_kz(eps_sup, kxv, kyv)
        kz_trn_f = _forward_flux_kz(eps_sub, kxv, kyv)
        safe_r = xp.where(xp.abs(kz_ref_f) < 1e-12, 1.0, kz_ref_f)
        safe_t = xp.where(xp.abs(kz_trn_f) < 1e-12, 1.0, kz_trn_f)
        R_rows, T_rows, jr_cols, jt_cols = [], [], [], []
        # Per-order tangential field amplitudes (PUBLIC exp(-iwt) convention =
        # conjugate of the internal), kept for the multi-order field bridge.
        rx_rows, ry_rows, tx_rows, ty_rows = [], [], [], []
        for _col, (ex0, ey0) in enumerate(((1.0, 0.0), (0.0, 1.0))):
            long_inc = kx0 * ex0 + ky0 * ey0
            # traced source: kz_inc is a tracer (no Python truth-value) -- the
            # kz_inc == 0 degenerate point is excluded by the concrete-path
            # guard and unreachable for a physically-propagating source.
            einc_sq = (1.0 + (long_inc / kz_inc) ** 2 if src_traced
                       else (1.0 + (long_inc / kz_inc) ** 2
                             if kz_inc != 0 else 1.0))
            if sym_rt is not None:
                r, t = sym_rt[_col]
            else:
                cinc = xp.concatenate([ex0 * delta, ey0 * delta])
                r = S11 @ cinc
                t = S21 @ cinc
            rx, ry = r[:N], r[N:]
            tx, ty = t[:N], t[N:]
            rz = -(kxv * rx + kyv * ry) / safe_r
            tz = -(kxv * tx + kyv * ty) / safe_t
            Re, Te = _project_efficiency(xp, kz_ref_f, kz_trn_f, kz_inc,
                                         rx, ry, rz, tx, ty, tz, einc_sq)
            R_rows.append(Re)
            T_rows.append(Te)
            rx_rows.append(xp.conj(rx))
            ry_rows.append(xp.conj(ry))
            tx_rows.append(xp.conj(tx))
            ty_rows.append(xp.conj(ty))
            jr_cols.append(xp.stack([xp.conj(rx[p0]), xp.conj(ry[p0])]))
            jt_cols.append(xp.stack([xp.conj(tx[p0]), xp.conj(ty[p0])]))
        R = xp.stack(R_rows)
        T = xp.stack(T_rows)
        Jr = xp.stack(jr_cols, axis=1)
        Jt = xp.stack(jt_cols, axis=1)
        out_orders = orders[:, 0].copy() if self.is_1d else orders
        # Modal data for RCWAResult's multi-order field bridge: per-order
        # amplitudes (2, N) for the two incident pols, the transverse
        # wavevectors normalised by k0 (kxv = kx/k0), and the per-order kz/k0
        # in each region (for propagation + the propagating-order mask).
        modal = dict(
            orders2d=orders, p0=p0,
            wavelength=(wl if src_traced else float(wl)),
            rx=xp.stack(rx_rows), ry=xp.stack(ry_rows),
            tx=xp.stack(tx_rows), ty=xp.stack(ty_rows),
            kx=kxv, ky=kyv, kz_ref=kz_ref, kz_trn=kz_trn,
            period_x=self.period_x, period_y=self.period_y,
            # incidence k-vectors (normalised by k0) for the per-order flux
            # weight Re(kz_m/kz_inc) and the incident |E|^2 (einc_sq) that the
            # power-calibrated field reconstruction needs.  Traced-source
            # values stay raw (D1).
            kx0=(kx0 if src_traced else float(kx0)),
            ky0=(ky0 if src_traced else float(ky0)),
            kz_inc=(kz_inc if src_traced else float(kz_inc)))
        if retain_internal and bname == "jax":
            warnings.warn(
                "RCWAStack.solve: retain_internal=True is unavailable on the "
                "JAX path (the per-layer modal retention is host-side); the "
                "efficiencies are returned WITHOUT internal-field data -- "
                "re-solve with NumPy inputs for field maps.", stacklevel=2)
        if retain_internal and bname != "jax" and not any_oop:
            # Per-layer cumulative partial S-matrices that bracket the TOP plane
            # of each layer -> the c+/c- modal amplitudes for the in-structure
            # field (see RCWAResult.internal_field).  Same gap-free
            # _interface/_propagation/_redheffer sequence as the global solve;
            # star(S_above[0], S_below[0]) reproduces the global S used for r/t.
            S_above, S_below, S_below_bot = self._internal_partials(
                modes, Wref, Vref, Wtrn, Vtrn, k0)
            modal["internal"] = dict(
                W=[m[0] for m in modes], V=[m[1] for m in modes],
                lam=[m[2] for m in modes], EPS=[m[3] for m in modes],
                thick=[float(L.thickness) for L in self._layers],
                S_above=S_above, S_below=S_below, S_below_bot=S_below_bot,
                Kx=Kx, Ky=Ky, k0=float(k0),
                N=N, delta=delta, layers=list(self._layers),
                period_x=self.period_x, period_y=self.period_y,
                is_1d=self.is_1d, nox=self.nox, noy=self.noy)
        elif retain_internal and bname != "jax":
            # GENERALIZED retention (loose-ends 2026-07-14, the Berreman-C2
            # port): explicit asymmetric mode sets + partial cascades built
            # from the SAME generalized interface/propagation matrices the
            # global solve used (gen_ifc/gen_prop collected above).
            nlay = len(modes)
            S_above = [None] * nlay
            S_above[0] = gen_ifc[0]
            for i in range(1, nlay):
                S_above[i] = _redheffer_star(
                    _redheffer_star(S_above[i - 1], gen_prop[i - 1]),
                    gen_ifc[i])
            S_below = [None] * nlay
            S_below_bot = [None] * nlay
            for i in range(nlay - 1, -1, -1):
                S_below_bot[i] = (gen_ifc[nlay] if i == nlay - 1
                                  else _redheffer_star(gen_ifc[i + 1],
                                                       S_below[i + 1]))
                S_below[i] = _redheffer_star(gen_prop[i], S_below_bot[i])
            # explicit (Wf, Vf, lamf, Wb, Vb, lamb) per layer; a symmetric
            # layer degenerates to (W, V, lam, W, -V, -lam).  E_z recovery
            # for an OOP layer needs the EZX/EZY convolutions (same internal
            # conj gauge as _layer_modes builds them); None for the rest.
            Wf_l, Vf_l, lf_l, Wb_l, Vb_l, lb_l = [], [], [], [], [], []
            EPS_l, EZX_l, EZY_l = [], [], []
            for m, L in zip(modes, self._layers):
                if isinstance(m, tuple) and len(m) == 8 and m[0] == "gen":
                    _k, Wf, Vf, lf, Wb, Vb, lb, EZZ = m
                    Wf_l.append(Wf), Vf_l.append(Vf), lf_l.append(lf)
                    Wb_l.append(Wb), Vb_l.append(Vb), lb_l.append(lb)
                    EPS_l.append(EZZ)
                    et = xp.conj(xp.asarray(L.data))
                    EZX_l.append(_eps_convolution_2d(
                        et[:, :, 2, 0], orders, self.nox, self.noy))
                    EZY_l.append(_eps_convolution_2d(
                        et[:, :, 2, 1], orders, self.nox, self.noy))
                else:
                    W, V, lam, EPS = m
                    Wf_l.append(W), Vf_l.append(V), lf_l.append(lam)
                    Wb_l.append(W), Vb_l.append(-V), lb_l.append(-lam)
                    EPS_l.append(EPS)
                    EZX_l.append(None)
                    EZY_l.append(None)
            modal["internal"] = dict(
                general=True,
                Wf=Wf_l, Vf=Vf_l, lamf=lf_l, Wb=Wb_l, Vb=Vb_l, lamb=lb_l,
                EPS=EPS_l, EZX=EZX_l, EZY=EZY_l,
                thick=[float(L.thickness) for L in self._layers],
                S_above=[(None, None, Sa[2], Sa[3]) for Sa in S_above],
                S_below=[(Sb[0], None, None, None) for Sb in S_below],
                S_below_bot=[(Sbb[0], None, None, None)
                             for Sbb in S_below_bot],
                Kx=Kx, Ky=Ky, k0=float(k0),
                N=N, delta=delta, layers=list(self._layers),
                period_x=self.period_x, period_y=self.period_y,
                is_1d=self.is_1d, nox=self.nox, noy=self.noy)
        if not is_jax:                       # the guard needs concrete R/T
            _check_energy("RCWAStack.solve", R, T,
                          lossless=self._stack_lossless())
        return RCWAResult(out_orders, R, T, Jr, Jt, modal=modal)

    def _internal_partials(self, modes, Wref, Vref, Wtrn, Vtrn, k0):
        """Cumulative ``(S_above, S_below)`` partial S-matrices bracketing the
        TOP of each layer (the recovery basis for the internal field).

        ``S_above[i]`` = star product superstrate -> TOP of layer ``i`` (through
        the interface INTO layer ``i``, before its own propagation);
        ``S_below[i]`` = star product TOP of layer ``i`` -> substrate (its own
        propagation first).  ``star(S_above[0], S_below[0])`` reproduces the
        global S-matrix."""
        nlay = len(modes)
        ifc = [_interface_smatrix(Wref, Vref, modes[0][0], modes[0][1])]
        for i in range(1, nlay):
            ifc.append(_interface_smatrix(modes[i - 1][0], modes[i - 1][1],
                                          modes[i][0], modes[i][1]))
        ifc.append(_interface_smatrix(modes[-1][0], modes[-1][1], Wtrn, Vtrn))
        lam_L = [(modes[i][2], k0 * self._layers[i].thickness)
                 for i in range(nlay)]
        S_above = [None] * nlay
        S_above[0] = ifc[0]
        for i in range(1, nlay):
            S_above[i] = _redheffer_star(
                _propagation_star(S_above[i - 1], *lam_L[i - 1]), ifc[i])
        S_below = [None] * nlay
        # S_below_bot[i] = star product from the BOTTOM of layer i to the
        # substrate (i.e. S_below[i] with layer i's own propagation removed from
        # the top).  Its S11 is the reflection seen looking DOWN from the bottom
        # of layer i; it lets the internal-field recovery reference the backward
        # mode to the layer BOTTOM with a DECAYING exponential (no exp(+lam k0 z)
        # overflow through a deep/lossy layer -- see RCWAResult._internal_cpm).
        S_below_bot = [None] * nlay
        for i in range(nlay - 1, -1, -1):
            below_bot = (ifc[nlay] if i == nlay - 1
                         else _redheffer_star(ifc[i + 1], S_below[i + 1]))
            S_below_bot[i] = below_bot
            # left-star against a pure-propagation factor collapses the same
            # way as _propagation_star (P11 = P22 = 0), mirrored:
            B11, B12, B21, B22 = below_bot
            xpb = array_namespace(B11)
            X = xpb.exp(-lam_L[i][0] * lam_L[i][1])
            S_below[i] = ((X[:, None] * B11) * X[None, :], X[:, None] * B12,
                          B21 * X[None, :], B22)
        # MEMORY: the recurrences above are complete, so drop the (2N,2N) blocks
        # no reader touches -- RCWAResult._internal_cpm reads only S_above[.][2:4],
        # S_below[.][0], S_below_bot[.][0].  12 -> 4 retained blocks/layer
        # (byte-identical; the retained arrays are unchanged).
        S_above = [(None, None, Sa[2], Sa[3]) for Sa in S_above]
        S_below = [(Sb[0], None, None, None) for Sb in S_below]
        S_below_bot = [(Sbb[0], None, None, None) for Sbb in S_below_bot]
        return S_above, S_below, S_below_bot


__all__ = [
    "rcwa_convergence",
    "RCWAResult",
    # v5.31 (audit W4-3): exported so callers can silence / promote exactly the
    # 1-D-stack y-averaging diagnostic without touching sibling UserWarnings.
    "RCWAYAverageWarning",
    "_RCWALayer",
    "_STACK_STABILIZE_WINDOW",
    "_STACK_STABILIZE_TOL",
    "_zero_order_efficiency",
    "_largest_feature_cluster",
    "_cluster_is_trending",
    "RCWAStack",
]
