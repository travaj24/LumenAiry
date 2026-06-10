"""RCWA unified multi-layer API: RCWAStack / RCWAResult (+ caching,
Jones bridge).  Handles both 1-D and 2-D layers."""
from __future__ import annotations

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
    _cached_homogeneous_eigenmodes,
    _check_energy,
    _EnergyError,
    _forward_flux_kz,
    _grazing_safe_wavelength,
    _homogeneous_eigenmodes,
    _interface_smatrix,
    _layer_eigenmodes,
    _layer_eigenmodes_tensor,
    _max_aligned_delta,
    _propagation_smatrix,
    _rcwa_convergence_stack,
    _rcwa_xp,
    _redheffer_star,
    _require_inplane_tensor,
    _require_jax_x64,
    _require_propagating_incidence,
    _sqrt_decay,
    _sqrt_forward,
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
)


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
        return dict(orders=self.orders, Ex=to_numpy(m[ex]), Ey=to_numpy(m[ey]),
                    kx=to_numpy(m["kx"]), ky=to_numpy(m["ky"]),
                    kz=to_numpy(kz), wavelength=m["wavelength"])

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
        from ..polarization import apply_jones_matrix
        return apply_jones_matrix(jones_field.copy(), self._Jr)

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
        restores per-order power (and the right dominant order)."""
        from ...backend import to_numpy
        m = self._require_modal()
        kz = complex(to_numpy(m["kz_ref"] if port == "reflection"
                              else m["kz_trn"])[idx])
        kx = complex(to_numpy(m["kx"])[idx])
        ky = complex(to_numpy(m["ky"])[idx])
        kz_inc, kx0, ky0 = m["kz_inc"], m["kx0"], m["ky0"]
        tang = float(abs(ax) ** 2 + abs(ay) ** 2)
        flux = float(np.real(kz / kz_inc)) if kz_inc != 0 else 0.0
        if tang < 1e-300 or flux <= 0.0:     # evanescent / no tangential power
            return 0.0
        az = -(kx * ax + ky * ay) / (kz if abs(kz) > 1e-12 else 1.0)
        inc = np.asarray(incident, dtype=_C).reshape(2)
        ez_inc = (-(kx0 * inc[0] + ky0 * inc[1]) / kz_inc) if kz_inc != 0 else 0.0
        einc_sq = float(abs(inc[0]) ** 2 + abs(inc[1]) ** 2 + abs(ez_inc) ** 2)
        eff = flux * (tang + float(abs(az) ** 2)) / (einc_sq if einc_sq else 1.0)
        return float(np.sqrt(eff / tang)) if eff > 0.0 else 0.0

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
            Reflection or transmission side.  Default reflection.
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
        a centred ``(ny, nx)`` grid (k-vectors are stored normalised by k0)."""
        from ...backend import to_numpy
        m = self._require_modal()
        k0 = 2.0 * np.pi / m["wavelength"]
        kx = k0 * float(np.real(to_numpy(m["kx"])[idx]))   # physical [1/m]
        ky = k0 * float(np.real(to_numpy(m["ky"])[idx]))
        xg = (np.arange(nx) - nx // 2) * dx
        yg = (np.arange(ny) - ny // 2) * dy
        X, Y = np.meshgrid(xg, yg)                         # (ny, nx)
        return np.exp(1j * (kx * X + ky * Y)).astype(_C)

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
        if normalize not in ("power", "field"):
            raise ValueError(
                f"RCWAResult.to_multiorder_field: normalize must be 'power' "
                f"or 'field', got {normalize!r}.")
        if filter not in ("none", "lanczos"):
            raise ValueError(
                f"RCWAResult.to_multiorder_field: filter must be 'none' or "
                f"'lanczos', got {filter!r}.")
        sigma = self._lanczos_sigma() if filter == "lanczos" else None
        from ...backend import to_numpy
        from ..polarization import JonesField
        m = self._require_modal()
        kz = to_numpy(m["kz_ref"] if port == "reflection" else m["kz_trn"])
        if orders is None:
            idxs = [i for i in range(kz.shape[0]) if np.real(kz[i]) > 1e-12]
        else:
            # explicit orders: skip evanescent ones (their carrier would be a
            # bogus fast-oscillating plane wave) with a warning rather than
            # silently depositing garbage.
            idxs = []
            for o in orders:
                i = self._order_index(o)
                if np.real(kz[i]) <= 1e-12:
                    import warnings
                    warnings.warn(
                        f"RCWAResult.to_multiorder_field: order {o!r} is "
                        f"evanescent (Re(kz) <= 0) and is skipped; it carries "
                        f"no propagating power.", stacklevel=2)
                    continue
                idxs.append(i)
        ny_i, nx_i = int(ny), int(nx)
        dy_f = float(dx if dy is None else dy)
        ex = np.zeros((ny_i, nx_i), dtype=_C)
        ey = np.zeros((ny_i, nx_i), dtype=_C)
        for idx in idxs:
            ax, ay = self._order_amplitude(idx, incident, port)
            if normalize == "power":
                s = self._order_power_scale(idx, ax, ay, incident, port)
                ax, ay = s * ax, s * ay
            if sigma is not None:
                w = sigma[idx]
                ax, ay = w * ax, w * ay
            carrier = self._order_carrier(idx, nx_i, ny_i, float(dx), dy_f)
            ex += ax * carrier
            ey += ay * carrier
        return JonesField(ex, ey, dx=dx, dy=dy)

    def _lanczos_sigma(self):
        """Per-order Lanczos sigma factors ``sinc(m/(Mx+1)) sinc(n/(My+1))``
        (1-D: the y-factor is 1), indexed by flat order index.  Damps the
        high orders smoothly to suppress Gibbs ringing in the reconstructed
        real-space field."""
        o = np.asarray(self.orders)
        if o.ndim == 2:
            mx = max(1, int(np.abs(o[:, 0]).max()))
            my = max(1, int(np.abs(o[:, 1]).max()))
            return (np.sinc(o[:, 0] / (mx + 1.0))
                    * np.sinc(o[:, 1] / (my + 1.0)))
        mx = max(1, int(np.abs(o).max()))
        return np.sinc(o / (mx + 1.0))

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
        # forward branch), so every exponential here is bounded.
        lam = np.asarray(to_numpy(info["lam"][i]))
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
        ex0, ey0 = incident
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
            W = np.asarray(to_numpy(info["W"][i]))
            V = np.asarray(to_numpy(info["V"][i]))
            lam = np.asarray(to_numpy(info["lam"][i]))
            EPS = np.asarray(to_numpy(info["EPS"][i]))
            # forward referenced to the layer TOP (depth z), backward to the
            # BOTTOM (L - z): BOTH exponents are <= 0 (Re(lam) >= 0 on the forward
            # branch), so a deep/lossy layer never overflows.  Same field as the
            # old c- exp(+lam k0 z), evaluated in the numerically-stable order.
            Lz = float(thick[i]) - zloc
            fwd = cplus * np.exp(-lam * k0 * zloc)
            bwd = cminus_bot * np.exp(-lam * k0 * Lz)
            s = W @ (fwd + bwd)                              # E tangential
            u = V @ (fwd - bwd)                              # H partner
            Sx, Sy = s[:N], s[N:]
            Hx, Hy = -1j * u[:N], -1j * u[N:]                # -i*eta0 scale
            Sz = np.linalg.solve(EPS, -(Kx @ Hy - Ky @ Hx))  # curl-H z-comp
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
        field samples ``(xg, yg)`` onto a ``(Sx, Sy)`` unit-cell sampling."""
        ix = (np.floor((xg % period_x) / period_x * Sx).astype(int)) % Sx
        iy = (np.floor((yg % period_y) / period_y * Sy).astype(int)) % Sy
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
    __slots__ = ("thickness", "kind", "data")

    def __init__(self, thickness, kind, data):
        # keep a traced (JAX) thickness native so layer DEPTH is differentiable
        self.thickness = thickness if is_jax_array(thickness) else float(thickness)
        self.kind = kind          # 'uniform' | 'iso' | 'tensor'
        self.data = data



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
                 n_substrate=1.0, n_orders=11, n_orders_y=None, use_gpu=False):
        _validate_geometry("RCWAStack", period=period, period_y=period_y,
                           n_orders=n_orders, n_orders_y=n_orders_y)
        self.period_x = float(period)
        self.is_1d = period_y is None and n_orders_y is None
        self.period_y = float(period if period_y is None else period_y)
        self.n_superstrate = n_superstrate
        self.n_substrate = n_substrate
        self.nox = int(n_orders)
        self.noy = 0 if self.is_1d else int(
            n_orders if n_orders_y is None else n_orders_y)
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

    def add_layer(self, thickness, *, eps=None, eps_cell=None,
                  eps_tensor_cell=None, shapes=None, eps_background=None):
        """Append a layer.  Provide exactly one layer specification:

        * ``eps`` (scalar) -- uniform spacer;
        * ``eps_cell`` (``(Sx, Sy)``) -- isotropic patterned, FFT-sampled;
        * ``eps_tensor_cell`` (``(Sx, Sy, 3, 3)``) -- anisotropic patterned;
        * ``shapes`` (with ``eps_background``) -- isotropic patterned using
          ANALYTIC shape Fourier transforms + the dual-Laurent factorization
          (exact, no pixelation; see :func:`rcwa_efficiency_2d_shapes`).

        Permittivities are in the PUBLIC convention (``Im(eps) > 0`` lossy).
        """
        n = sum(x is not None for x in (eps, eps_cell, eps_tensor_cell, shapes))
        if n != 1:
            raise ValueError(
                "add_layer: provide exactly one of eps, eps_cell, "
                "eps_tensor_cell, shapes.")
        if not is_jax_array(thickness):       # traced depth -> skip the guard
            _validate_geometry("add_layer", depth=thickness)
        if eps is not None:
            self._layers.append(_RCWALayer(thickness, "uniform", _C(eps)))
        elif eps_cell is not None:
            # Keep a JAX cell native (a np.asarray would materialise the tracer
            # and break the differentiable RCWAStack path); NumPy/list inputs
            # are still normalised to a contiguous complex array.
            cell = (eps_cell.astype(_C) if is_jax_array(eps_cell)
                    else np.asarray(eps_cell, dtype=_C))
            if cell.ndim == 1:
                cell = cell[:, None]
            _validate_cell_sampling("add_layer", cell, self.nox, self.noy)
            self._layers.append(_RCWALayer(thickness, "iso", cell))
        elif shapes is not None:
            if eps_background is None:
                raise ValueError(
                    "add_layer: shapes requires eps_background.")
            _validate_shapes("add_layer", shapes, self.period_x, self.period_y)
            self._layers.append(
                _RCWALayer(thickness, "shapes", (_C(eps_background), shapes)))
        else:
            tcell = (eps_tensor_cell.astype(_C) if is_jax_array(eps_tensor_cell)
                     else np.asarray(eps_tensor_cell, dtype=_C))
            _validate_cell_sampling("add_layer", tcell, self.nox, self.noy)
            _require_inplane_tensor("add_layer", tcell)
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
                            duty_bottom, duty_top=None, n_slices=12, n_x=256):
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
        """
        dt = float(duty_bottom if duty_top is None else duty_top)
        db = float(duty_bottom)
        for d in (db, dt):
            if not (0.0 <= d <= 1.0):
                raise ValueError(
                    f"add_tapered_grating: duty cycles must be in [0, 1], got "
                    f"duty_top={dt}, duty_bottom={db}.")
        er, eg = _C(eps_ridge), _C(eps_groove)
        x = (np.arange(int(n_x)) + 0.5) / int(n_x)
        n_y = max(1, 4 * self.noy + 1)              # grating is uniform in y

        def _profile(zeta):
            duty = dt + (db - dt) * zeta            # top (0) -> bottom (1)
            half = 0.5 * duty
            ridge = np.abs(x - 0.5) < half          # centred ridge
            col = np.where(ridge, er, eg).astype(_C)
            return np.broadcast_to(col[:, None], (int(n_x), n_y)).copy()

        return self.add_graded_layer(thickness, _profile, n_slices=n_slices,
                                     rule="midpoint")

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
        _validate_geometry("RCWAStack.set_source", wavelength=wavelength)
        theta = _resolve_incidence(angle, theta)      # theta wins; falls back to angle
        if theta is None:
            theta = 0.0
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

    def _layer_modes(self, layer, Kx, Ky, orders):
        """Layer eigenmodes ``(W, V, lam, EPS)``.  ``EPS`` is the wall-tangential
        ``[[eps]]`` convolution (``EZZ`` for a tensor layer) in the INTERNAL
        convention -- retained for the curl-H ``E_z`` solve in the internal-field
        reconstruction; ignored on the (default) far-field path."""
        xp = array_namespace(Kx)
        if layer.kind == "uniform":
            W, V, kz = _homogeneous_eigenmodes(Kx, Ky, complex(np.conj(layer.data)))
            lam = _sqrt_decay(-xp.concatenate([kz, kz]) ** 2)
            EPS = complex(np.conj(layer.data)) * xp.eye(Kx.shape[0], dtype=_C)
            return W, V, lam, EPS
        if layer.kind == "iso":
            EPS = _eps_convolution_2d(xp.conj(xp.asarray(layer.data)), orders,
                                      self.nox, self.noy)
            W, V, lam = _layer_eigenmodes(Kx, Ky, EPS, EPS)
            return W, V, lam, EPS
        if layer.kind == "shapes":
            # Analytic (host) form factors -> move the convolution to backend.
            eps_bg, shapes = layer.data
            shapes_c = [dict(s, eps=complex(np.conj(_C(s["eps"])))) for s in shapes]
            EPS_np, _EPS_inv_np = _analytic_convolutions_2d(
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
            return ("uniform", complex(data))
        if kind == "shapes":
            eps_bg, shapes = data
            return ("shapes", complex(eps_bg),
                    tuple(sorted((k, repr(v)) for s in shapes
                                 for k, v in s.items())))
        from ...backend import to_numpy
        try:                                   # iso / tensor: hash the cell
            arr = np.ascontiguousarray(to_numpy(data))
        except Exception:                      # traced array -> no dedup
            return None
        return (kind, arr.shape, str(arr.dtype), arr.tobytes())

    @_with_blas_limit
    def solve(self, *, retain_internal=False, stabilize=False) -> RCWAResult:
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
        CuPy only."""
        if not stabilize:
            return self._solve_once(retain_internal=retain_internal)
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
                    res = self._solve_once(retain_internal=False)
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

    def _solve_once(self, *, retain_internal=False) -> RCWAResult:
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
        src = self._source
        wl, theta, phi = src["wavelength"], src["theta"], src["phi"]
        # Dispatch the backend off the patterned-layer arrays so a JAX cell
        # makes the whole stack solve differentiable (the source geometry --
        # wavelength/angle -- is always concrete here, so only the layer
        # permittivities are traced).
        layer_arrays = [L.data for L in self._layers if L.kind in ("iso",
                                                                    "tensor")]
        xp = _rcwa_xp("RCWAStack.solve", self.use_gpu, *layer_arrays)
        bname = backend_name(xp)
        is_jax = bname == "jax"
        if is_jax:
            _require_jax_x64("RCWAStack.solve")
        orders, N = _harmonic_orders_2d(self.nox, self.noy)
        eps_sup = complex(np.conj(_C(self.n_superstrate) ** 2))
        eps_sub = complex(np.conj(_C(self.n_substrate) ** 2))
        nre = float(np.real(np.sqrt(eps_sup)))
        kx0 = nre * np.sin(theta) * np.cos(phi)
        ky0 = nre * np.sin(theta) * np.sin(phi)
        _require_propagating_incidence("RCWAStack.solve", eps_sup,
                                       kx0 ** 2 + ky0 ** 2)
        # The traced (JAX) layer permittivities cannot feed the concrete grazing
        # nudge, so on the differentiable path the nudge sees only the region
        # indices (the dominant region Rayleigh anomaly is still caught).
        eps_reals = ([eps_sup, eps_sub] if is_jax
                     else [eps_sup, eps_sub] + self._layer_eps_reals())
        wl = _grazing_safe_wavelength(
            wl, kx0, ky0, orders[:, 0], orders[:, 1], self.period_x,
            self.period_y, eps_reals)
        k0 = 2.0 * np.pi / wl
        kx = kx0 + orders[:, 0] * (wl / self.period_x)
        ky = ky0 + orders[:, 1] * (wl / self.period_y)
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
        geom = (self.nox, self.noy, wl, theta, phi, self.period_x,
                self.period_y, self.n_superstrate, bname)
        Wref, Vref, kz_ref = _cached_homogeneous_eigenmodes(
            eps_sup, Kx, Ky, ("sup", self.n_superstrate) + geom)
        Wtrn, Vtrn, kz_trn = _cached_homogeneous_eigenmodes(
            eps_sub, Kx, Ky, ("sub", self.n_substrate) + geom)

        # Eig reuse (v5.6): the per-layer modal eig is the dominant cost and is
        # THICKNESS-INDEPENDENT, so two layers with identical permittivity (a
        # repeated DBR / Bragg period, a metamaterial supercell) share one eig
        # instead of recomputing it.  Bit-exact -- it memoises a pure function;
        # a None key (traced JAX array) falls back to per-layer solves.
        _mode_cache = {}
        modes = []
        for L in self._layers:
            key = self._layer_eig_key(L)
            cached = _mode_cache.get(key) if key is not None else None
            if cached is None:
                cached = self._layer_modes(L, Kx, Ky, orders)
                if key is not None:
                    _mode_cache[key] = cached
            modes.append(cached)
        W0, V0, lam0, _e0 = modes[0]
        S = _interface_smatrix(Wref, Vref, W0, V0)
        S = _redheffer_star(S, _propagation_smatrix(lam0, k0 * self._layers[0].thickness))
        for i in range(1, len(modes)):
            Wp, Vp, _lp, _ = modes[i - 1]
            Wc, Vc, lamc, _ = modes[i]
            S = _redheffer_star(S, _interface_smatrix(Wp, Vp, Wc, Vc))
            S = _redheffer_star(S, _propagation_smatrix(lamc, k0 * self._layers[i].thickness))
        Wl, Vl, _ll, _el = modes[-1]
        S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
        S11, S12, S21, S22 = S

        p0 = int(np.where((orders[:, 0] == 0) & (orders[:, 1] == 0))[0][0])
        delta = xp.asarray(((orders[:, 0] == 0) & (orders[:, 1] == 0)).astype(_C))
        kz_inc = float(np.real(_sqrt_forward(np.conj(eps_sup) - kx0 ** 2 - ky0 ** 2)))
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
        for ex0, ey0 in ((1.0, 0.0), (0.0, 1.0)):
            long_inc = kx0 * ex0 + ky0 * ey0
            einc_sq = 1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0 else 1.0
            cinc = xp.concatenate([ex0 * delta, ey0 * delta])
            r = S11 @ cinc
            t = S21 @ cinc
            rx, ry = r[:N], r[N:]
            tx, ty = t[:N], t[N:]
            rz = -(kxv * rx + kyv * ry) / safe_r
            tz = -(kxv * tx + kyv * ty) / safe_t
            Re = xp.real(kz_ref_f / kz_inc) * (xp.abs(rx) ** 2 + xp.abs(ry) ** 2
                                               + xp.abs(rz) ** 2) / einc_sq
            Te = xp.real(kz_trn_f / kz_inc) * (xp.abs(tx) ** 2 + xp.abs(ty) ** 2
                                               + xp.abs(tz) ** 2) / einc_sq
            R_rows.append(xp.where(xp.real(kz_ref_f) > 0, xp.real(Re), 0.0))
            T_rows.append(xp.where(xp.real(kz_trn_f) > 0, xp.real(Te), 0.0))
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
            orders2d=orders, p0=p0, wavelength=float(wl),
            rx=xp.stack(rx_rows), ry=xp.stack(ry_rows),
            tx=xp.stack(tx_rows), ty=xp.stack(ty_rows),
            kx=kxv, ky=kyv, kz_ref=kz_ref, kz_trn=kz_trn,
            period_x=self.period_x, period_y=self.period_y,
            # incidence k-vectors (normalised by k0) for the per-order flux
            # weight Re(kz_m/kz_inc) and the incident |E|^2 (einc_sq) that the
            # power-calibrated field reconstruction needs.
            kx0=float(kx0), ky0=float(ky0), kz_inc=float(kz_inc))
        if retain_internal and bname != "jax":
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
        if not is_jax:                       # the guard needs concrete R/T
            _check_energy("RCWAStack.solve", R, T)
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
        prop = [_propagation_smatrix(modes[i][2], k0 * self._layers[i].thickness)
                for i in range(nlay)]
        S_above = [None] * nlay
        S_above[0] = ifc[0]
        for i in range(1, nlay):
            S_above[i] = _redheffer_star(
                _redheffer_star(S_above[i - 1], prop[i - 1]), ifc[i])
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
            S_below[i] = _redheffer_star(prop[i], below_bot)
        return S_above, S_below, S_below_bot


__all__ = [
    "rcwa_convergence",
    "RCWAResult",
    "_RCWALayer",
    "_STACK_STABILIZE_WINDOW",
    "_STACK_STABILIZE_TOL",
    "_zero_order_efficiency",
    "_largest_feature_cluster",
    "_cluster_is_trending",
    "RCWAStack",
]
