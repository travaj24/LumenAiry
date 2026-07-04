"""Differentiable (JAX) ``PMM2DStack.solve`` -- the multilayer twin of the
2-D hybrid scalar surface (``_jax_twod``), composed with the same backend-
generic Redheffer cascade the NumPy stack uses.

Scope (the SCALAR in-plane vertical surface, mirroring ``_jax_twod``):

* UNIFORM layers: traced eps (analytic Rayleigh modes -- no eig at all).
* CONCRETE patterned scalar layers: the k0-free projected operators
  ``Gx0F / EpsF / EinvF / EpnF / IpxF / IpyF`` are frozen NumPy constants
  (built by the SAME ``_scalar_projected_ops`` the NumPy stack uses); the
  traced ``k0 / kx0 / ky0`` enter through ``GxF = Gx0F/k0 + kx0 IpxF`` and
  the eig is the gauge-stable custom-VJP one.  Gradients flow to thickness,
  wavelength, theta/phi and the half-space indices -- NOT to the frozen
  pixel values.
* TRACED patterned scalar layers: ``add_layer(eps_cell=<jax>,
  region_layout=<int grid>)`` -- the CONCRETE layout defines the walls (a
  traced cell cannot: wall extraction is data-dependent control flow) and
  the traced cell provides the region VALUES, read at each region's first
  pixel (the ``_pmm_efficiency_2d_cell_jax`` contract).  Requires walls on
  BOTH axes.
* Tensor cells (in-plane or OOP), ``retain_internal``, dispersive layers
  raise upstream (the dispatch in :meth:`PMM2DStack.solve`).

Host-side guards that need concrete values degrade gracefully: the grazing-
safe wavelength nudge and the propagating-incidence check run only when the
wavelength / angles are concrete (the documented Wood-anomaly caveat:
gradients are valid only BETWEEN order cutoffs).
"""
from __future__ import annotations

import numpy as np

_C = np.complex128

__all__ = ["_pmm_stack2d_solve_jax"]


def _concrete(v):
    """float(v) when concrete, else None (tracers have no concrete value)."""
    try:
        return float(np.real(np.asarray(v)))
    except Exception:
        return None


def _layer_static_scalar(stack, L):
    """Frozen k0-free projected operators for a CONCRETE scalar layer --
    verbatim the NumPy stack's per-layer build (INTERNAL conj convention)."""
    from .twod import _build_axis, _scalar_projected_ops
    ax = _build_axis(stack.period_x, L["xw"], stack.degree, L["el_x"],
                     stack.grade)
    ay = _build_axis(stack.period_y, L["yw"], stack.degree, L["el_y"],
                     stack.grade)
    ox = np.arange(-stack.n_orders, stack.n_orders + 1)
    oy = np.arange(-stack.n_orders, stack.n_orders + 1)
    return _scalar_projected_ops(ax, ay, np.conj(L["tile"]), ox, oy,
                                 stack.period_x, stack.period_y)


def _layer_static_traced(stack, L):
    """Geometry constants for a TRACED scalar layer: per-region partition-of-
    unity weights + the kron-factorized projectors (the ``_static_prep_cell``
    construction on the layer's stored CONCRETE region layout)."""
    from .twod import _axis_projection, _build_axis
    ax = _build_axis(stack.period_x, L["xw"], stack.degree, L["el_x"],
                     stack.grade)
    ay = _build_axis(stack.period_y, L["yw"], stack.degree, L["el_y"],
                     stack.grade)
    nsx, nsy = len(ax["strips"]), len(ay["strips"])
    if nsx < 2 or nsy < 2:
        raise NotImplementedError(
            "PMM2DStack: a TRACED eps_cell layer needs walls on BOTH axes "
            "(the kron-factorized traced assembly); axis-uniform traced "
            "cells are not supported -- use a uniform `eps` layer or the "
            "1-D PMMStack.")
    ox = np.arange(-stack.n_orders, stack.n_orders + 1)
    oy = np.arange(-stack.n_orders, stack.n_orders + 1)
    Tx = _axis_projection(ax, ox)
    Txp = np.linalg.pinv(Tx)
    Ty = _axis_projection(ay, oy)
    Typ = np.linalg.pinv(Ty)
    mdx = np.diag(ax["M"])
    mdy = np.diag(ay["M"])
    Mdiag = np.kron(mdy, mdx)
    gx1 = Tx @ ((1.0 / mdx)[:, None] * ax["D"]) @ Txp
    gy1 = Ty @ ((1.0 / mdy)[:, None] * ay["D"]) @ Typ
    NxO, NyO = len(ox), len(oy)
    Gx0F = -1j * np.kron(np.eye(NyO, dtype=_C), gx1)
    Gy0F = -1j * np.kron(gy1, np.eye(NxO, dtype=_C))
    Ip = np.kron(Ty @ Typ, Tx @ Txp)
    # per-region weight rows: eps_nodal = eps_regions @ Wreg (linear)
    region_ids = L["layout_tile"]                      # (nsx, nsy) int
    uniq = np.unique(region_ids)
    Wreg = np.zeros((uniq.size, Mdiag.size))
    for sx in range(nsx):
        for sy in range(nsy):
            r = int(np.where(uniq == region_ids[sx, sy])[0][0])
            Wreg[r] += np.real(
                np.kron(np.diag(ay["Mtile"][sy]),
                        np.diag(ax["Mtile"][sx])) / Mdiag)
    return dict(Tp=np.kron(Ty, Tx), Tpinv=np.kron(Typ, Txp), Wreg=Wreg,
                Gx0F=Gx0F, Gy0F=Gy0F, IpxF=Ip, IpyF=Ip, uniq=uniq)


def _pmm_stack2d_solve_jax(stack):
    """The JAX twin of the symmetric (in-plane vertical) branch of
    :meth:`PMM2DStack.solve`.  Returns ``(orders(N,2), R(2,N), T(2,N),
    jones(2,2))`` with jnp R/T/jones."""
    import jax.numpy as jnp

    from ..rcwa import _jax_eig_stable, _require_jax_x64
    from ..rcwa._core import (
        _interface_smatrix,
        _propagation_star,
        _redheffer_star,
    )
    _require_jax_x64("PMM2DStack.solve")
    cj = jnp.complex128

    wavelength = stack._src["wavelength"]
    theta, phi = stack._src["theta"], stack._src["phi"]
    n_sup = jnp.asarray(stack.n_sup).astype(cj)
    n_sub = jnp.asarray(stack.n_sub).astype(cj)
    eps_sup = jnp.conj(n_sup ** 2)                 # INTERNAL exp(+iwt) gauge
    eps_sub = jnp.conj(n_sub ** 2)

    n_orders = stack.n_orders
    ox = np.arange(-n_orders, n_orders + 1)
    oy = np.arange(-n_orders, n_orders + 1)
    order_x = np.tile(ox, len(oy))
    order_y = np.repeat(oy, len(ox))
    Nf = len(order_x)

    # ---- host-side guards, ONLY when the inputs are concrete --------------
    wl_c = _concrete(wavelength)
    th_c = _concrete(theta)
    ph_c = _concrete(phi)
    nsup_c = _concrete(stack.n_sup)
    wl_use = wavelength
    if None not in (wl_c, th_c, ph_c, nsup_c):
        from .twod import (
            _grazing_safe_wavelength,
            _require_propagating_incidence,
        )
        kx0_c = nsup_c * np.sin(th_c) * np.cos(ph_c)
        ky0_c = nsup_c * np.sin(th_c) * np.sin(ph_c)
        eps_reals = []
        for v in (stack.n_sup, stack.n_sub):
            vc = _concrete(v)
            if vc is not None:
                eps_reals.append(complex(np.conj(complex(np.asarray(v)) ** 2)))
        for L in stack._layers:
            if L["kind"] == "uniform":
                ec = _concrete(L["eps"])
                if ec is not None:
                    eps_reals.append(complex(np.asarray(L["eps"])))
            elif L["kind"] == "scalar" and not L.get("traced"):
                eps_reals += [complex(e) for e in
                              np.asarray(L["tile"]).ravel()]
        _require_propagating_incidence(
            "PMM2DStack.solve", complex(np.conj(complex(nsup_c) ** 2)),
            kx0_c ** 2 + ky0_c ** 2)
        wl_use = _grazing_safe_wavelength(
            wl_c, kx0_c, ky0_c, order_x, order_y, stack.period_x,
            stack.period_y, eps_reals)

    wl_t = jnp.asarray(wl_use)
    k0 = 2.0 * jnp.pi / wl_t
    nre = jnp.real(jnp.sqrt(jnp.conj(eps_sup)))
    kx0 = nre * jnp.sin(jnp.asarray(theta)) * jnp.cos(jnp.asarray(phi))
    ky0 = nre * jnp.sin(jnp.asarray(theta)) * jnp.sin(jnp.asarray(phi))
    kxv = kx0 + jnp.asarray(order_x) * (wl_t / stack.period_x)
    kyv = ky0 + jnp.asarray(order_y) * (wl_t / stack.period_y)

    def _sqrt_decay(x):
        r = jnp.sqrt(x.astype(cj))
        on_cut = r.real == 0
        return jnp.where(on_cut & (r.imag < 0), -r, r)

    def _inv_lam(lam):
        safe = jnp.where(jnp.abs(lam) < 1e-12, 1e-12, lam)
        return 1.0 / safe

    def _kz_fwd(eps, kx, ky):
        val = jnp.sqrt((eps - kx ** 2 - ky ** 2).astype(cj))
        return jnp.where(val.imag < 0.0, -val, val)

    def _homog(eps):
        Kx = jnp.diag(kxv.astype(cj))
        Ky = jnp.diag(kyv.astype(cj))
        kz = _kz_fwd(eps, kxv, kyv)
        lam = _sqrt_decay(-jnp.concatenate([kz, kz]) ** 2)
        eps_eye = eps * jnp.eye(Nf, dtype=cj)
        Q = jnp.block([[Kx @ Ky, eps_eye - Kx @ Kx],
                       [Ky @ Ky - eps_eye, -Ky @ Kx]])
        W = jnp.eye(2 * Nf, dtype=cj)
        V = Q @ jnp.diag(_inv_lam(lam))
        return W, V, lam

    def _modes_projected(GxF, GyF, EpsF, EinvF, EpnF):
        eye_F = jnp.eye(Nf, dtype=cj)
        EPS_normal = EpnF if stack.formulation == "li" else EpsF
        Q = jnp.block([[GxF @ GyF, EpsF - GxF @ GxF],
                       [GyF @ GyF - EPS_normal, -GyF @ GxF]])
        EPS_inv = (EinvF if stack.formulation == "li"
                   else jnp.linalg.inv(EpsF))
        P = jnp.block([[GxF @ EPS_inv @ GyF, eye_F - GxF @ EPS_inv @ GxF],
                       [GyF @ EPS_inv @ GyF - eye_F, -GyF @ EPS_inv @ GxF]])
        lam2, Wl = eig(P @ Q)
        lam = _sqrt_decay(lam2)
        Vl = Q @ Wl @ jnp.diag(_inv_lam(lam))
        return Wl, Vl, lam

    eig = _jax_eig_stable()
    Wsup, Vsup, _ls = _homog(eps_sup)
    Wsub, Vsub, _lb = _homog(eps_sub)

    modes = []
    for L in stack._layers:
        thk = jnp.asarray(L["t"])
        if L["kind"] == "uniform":
            eps_l = jnp.conj(jnp.asarray(L["eps"]).astype(cj))
            Wl, Vl, lam = _homog(eps_l)
        elif L.get("traced"):
            st = _layer_static_traced(stack, L)
            cell = jnp.asarray(L["cell"]).astype(cj)
            eps_regions = jnp.conj(jnp.stack(
                [cell[i, j] for (i, j) in L["first_idx"]]))
            Wreg = jnp.asarray(st["Wreg"])
            eps_nodal = eps_regions @ Wreg
            inv_nodal = (1.0 / eps_regions) @ Wreg
            Tp = jnp.asarray(st["Tp"], cj)
            Tpinv = jnp.asarray(st["Tpinv"], cj)
            EpsF = (Tp * eps_nodal[None, :]) @ Tpinv
            EinvF = (Tp * inv_nodal[None, :]) @ Tpinv
            EpnF = (Tp * (1.0 / inv_nodal)[None, :]) @ Tpinv
            GxF = jnp.asarray(st["Gx0F"], cj) / k0 + kx0 * jnp.asarray(
                st["IpxF"], cj)
            GyF = jnp.asarray(st["Gy0F"], cj) / k0 + ky0 * jnp.asarray(
                st["IpyF"], cj)
            Wl, Vl, lam = _modes_projected(GxF, GyF, EpsF, EinvF, EpnF)
        else:                                       # concrete scalar layer
            tile_i = np.conj(L["tile"])
            eps0 = tile_i.flat[0]
            if bool(np.all(np.abs(tile_i - eps0) < 1e-12)):
                Wl, Vl, lam = _homog(jnp.asarray(eps0))
            else:
                lops = _layer_static_scalar(stack, L)
                GxF = (jnp.asarray(lops["Gx0F"], cj) / k0
                       + kx0 * jnp.asarray(lops["IpxF"], cj))
                GyF = (jnp.asarray(lops["Gy0F"], cj) / k0
                       + ky0 * jnp.asarray(lops["IpyF"], cj))
                Wl, Vl, lam = _modes_projected(
                    GxF, GyF, jnp.asarray(lops["EpsF"], cj),
                    jnp.asarray(lops["EinvF"], cj),
                    jnp.asarray(lops["EpnF"], cj))
        modes.append((Wl, Vl, lam, thk))

    # ---- symmetric Redheffer cascade (backend-generic helpers) ------------
    nlay = len(modes)
    ifc = [_interface_smatrix(Wsup, Vsup, modes[0][0], modes[0][1])]
    for ii in range(1, nlay):
        ifc.append(_interface_smatrix(modes[ii - 1][0], modes[ii - 1][1],
                                      modes[ii][0], modes[ii][1]))
    ifc.append(_interface_smatrix(modes[-1][0], modes[-1][1], Wsub, Vsub))
    S = ifc[0]
    for ii, (Wl, Vl, lam, thk) in enumerate(modes):
        S = _propagation_star(S, lam, k0 * thk)
        S = _redheffer_star(S, ifc[ii + 1])
    S11, _S12, S21, _S22 = S

    # ---- Jones far field (both incident polarizations; verbatim the numpy
    # stack's tail with tracer-safe guards) ---------------------------------
    p0 = int(np.where((order_x == 0) & (order_y == 0))[0][0])
    delta = jnp.asarray(((order_x == 0) & (order_y == 0)).astype(_C))
    kz_inc = jnp.real(_kz_fwd(jnp.conj(eps_sup), kx0, ky0))
    kz_ref_f = _kz_fwd(jnp.conj(eps_sup), kxv, kyv)
    kz_trn_f = _kz_fwd(jnp.conj(eps_sub), kxv, kyv)
    safe_r = jnp.where(jnp.abs(kz_ref_f) < 1e-12, 1.0, kz_ref_f)
    safe_t = jnp.where(jnp.abs(kz_trn_f) < 1e-12, 1.0, kz_trn_f)
    safe_kzi = jnp.where(kz_inc == 0, 1.0, kz_inc)
    R_rows, T_rows, j_cols = [], [], []
    for ex0, ey0 in ((1.0, 0.0), (0.0, 1.0)):
        long_inc = kx0 * ex0 + ky0 * ey0
        einc_sq = 1.0 + (long_inc / safe_kzi) ** 2
        cinc = jnp.concatenate([ex0 * delta, ey0 * delta])
        r = S11 @ cinc
        t_ = S21 @ cinc
        rx, ry = r[:Nf], r[Nf:]
        tx, ty = t_[:Nf], t_[Nf:]
        rz = -(kxv * rx + kyv * ry) / safe_r
        tz = -(kxv * tx + kyv * ty) / safe_t
        Re = jnp.real(kz_ref_f / kz_inc) * (
            jnp.abs(rx) ** 2 + jnp.abs(ry) ** 2 + jnp.abs(rz) ** 2) / einc_sq
        Te = jnp.real(kz_trn_f / kz_inc) * (
            jnp.abs(tx) ** 2 + jnp.abs(ty) ** 2 + jnp.abs(tz) ** 2) / einc_sq
        R_rows.append(jnp.where(jnp.real(kz_ref_f) > 0, jnp.real(Re), 0.0))
        T_rows.append(jnp.where(jnp.real(kz_trn_f) > 0, jnp.real(Te), 0.0))
        j_cols.append(jnp.stack([jnp.conj(rx[p0]), jnp.conj(ry[p0])]))
    R_eff = jnp.stack(R_rows)
    T_eff = jnp.stack(T_rows)
    jones = jnp.stack(j_cols, axis=1)
    orders2d = np.stack([order_x, order_y], axis=1)
    return orders2d, R_eff, T_eff, jones
