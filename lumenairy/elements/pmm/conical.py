"""Native 1-D conical (out-of-plane, ``phi != 0``) PMM -- the ``O(N)`` reduction
of the 2-D coupled build (AUDIT_PMM_CONICAL_OUT_OF_PLANE, Path B, isotropic).

Under conical incidence the invariant-axis wavenumber ``ky0`` is nonzero, so the
1-D grating's TE and TM problems no longer separate: a single incident
polarization excites BOTH tangential field components and the solve must go
through the coupled ``2N`` operator everywhere.  Rather than pay the 2-D bridge's
``O(N^2)`` order set (Path A), this native path keeps the y-axis DEGENERATE --
one Fourier order ``n_y = 0``, so ``GyF = ky0 * I`` is a scalar shift and every
operator is ``Nf = 2*n_orders+1`` wide -- and routes it through the SAME coupled
eigenmode + generalized S-matrix + Jones machinery the 2-D hybrid uses
(``twod._layer_modes_projected`` etc.).  It is the 2-D conical build with the
y-order sweep collapsed to a single scalar.

Scope: ISOTROPIC scalar ridge/groove (the exp12 metal / dielectric grating
case).  The full-tensor (LC) conical layer and the ``PMMStack`` wiring are
follow-ups (Path B phases 2 and 4); this module is phases 0/1/3.  Validated
against the analytic Berreman 4x4 conical oracle (uniform slab) and the
same-family Path A (PMM2DStack y-invariant) bridge.
"""
from __future__ import annotations

import numpy as np

from .twod import (
    _build_axis,
    _homogeneous_modes,
    _interface_smatrix,
    _kz_forward2,
    _layer_modes_projected,
    _propagation_smatrix,
    _redheffer_star,
    _scalar_projected_ops,
)

_C = complex

__all__ = ["pmm_jones_1d_conical"]


def pmm_jones_1d_conical(period, eps_ridge, eps_groove, n_substrate,
                         n_superstrate, depth, duty_cycle, wavelength, *,
                         theta, phi, degree=16, elements_per_region=1,
                         grade=True, n_orders=None, formulation="li"):
    r"""Conical (out-of-plane) diffraction of an ISOTROPIC 1-D binary grating.

    Parameters mirror :func:`~lumenairy.elements.pmm.pmm_efficiency_1d` plus the
    azimuth ``phi``.  ``eps_ridge``/``eps_groove`` are PERMITTIVITIES (PUBLIC
    convention ``Im(eps) > 0`` for loss).  The x-period is ``period``; the cell
    is uniform in y.

    Returns ``(orders, R, T, jones)`` with ``orders`` an ``(Nf, 2)`` int array of
    the ``(m, 0)`` diffraction orders, ``R``/``T`` shape ``(2, Nf)`` (rows =
    incident lab ``E_x`` / ``E_y``), and ``jones`` the ``(2, 2)`` zeroth-order
    reflection Jones (columns = incident ``[E_x; E_y]``), exactly as
    :func:`~lumenairy.elements.pmm.pmm_jones_2d`.
    """
    if n_orders is None:
        n_orders = max(8, degree // 2)
    # ridge occupies [0, duty*period); one wall at duty*period -> 2 x-strips
    x_walls = [float(duty_cycle) * period]
    ax = _build_axis(period, x_walls, degree, elements_per_region, grade)
    ay = _build_axis(period, [], degree, elements_per_region, grade)  # y uniform
    # INTERNAL (conjugated, exp(+iwt)) convention for the solve
    eps_sup = np.conj(_C(n_superstrate) ** 2)
    eps_sub = np.conj(_C(n_substrate) ** 2)
    eps_tile = np.conj(np.array([[eps_ridge], [eps_groove]], dtype=complex))

    nre = float(np.real(np.sqrt(eps_sup)))
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)

    ox = np.arange(-n_orders, n_orders + 1)
    oy = np.array([0])
    order_x = np.tile(ox, len(oy))
    order_y = np.repeat(oy, len(ox))
    Nf = len(order_x)
    k0 = 2.0 * np.pi / wavelength
    kxv = kx0 + order_x * (wavelength / period)
    kyv = ky0 + order_y * (wavelength / period)          # == ky0 (constant)

    # ---- conical half-spaces (kz = sqrt(eps - kx^2 - ky^2)) ----
    Wsup, Vsup, _ls, _kzr = _homogeneous_modes(kxv, kyv, eps_sup)
    Wsub, Vsub, _lb, _kzt = _homogeneous_modes(kxv, kyv, eps_sub)

    # ---- coupled layer modes (GyF = ky0*I; the 2-D machinery, y degenerate) ----
    lops = _scalar_projected_ops(ax, ay, eps_tile, ox, oy, period, period)
    GxF = lops["Gx0F"] / k0 + kx0 * lops["IpxF"]
    GyF = lops["Gy0F"] / k0 + ky0 * lops["IpyF"]
    Wl, Vl, lam_l = _layer_modes_projected(
        GxF, GyF, lops["EpsF"], lops["EinvF"], lops["EpnF"],
        formulation=formulation, EpnxF=lops["EpnxF"], EpnyF=lops["EpnyF"])

    # ---- Redheffer cascade ----
    S = _interface_smatrix(Wsup, Vsup, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam_l, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wsub, Vsub))
    S11, _S12, S21, _S22 = S

    # ---- far field: drive unit Ex / Ey, build R/T + 2x2 Jones (jones_2d tail) --
    p0 = int(np.where((order_x == 0) & (order_y == 0))[0][0])
    delta = ((order_x == 0) & (order_y == 0)).astype(_C)
    kz_inc = float(np.real(_kz_forward2(np.conj(eps_sup), kx0, ky0)))
    kz_ref_f = _kz_forward2(np.conj(eps_sup), kxv, kyv)
    kz_trn_f = _kz_forward2(np.conj(eps_sub), kxv, kyv)
    safe_r = np.where(np.abs(kz_ref_f) < 1e-12, 1.0, kz_ref_f)
    safe_t = np.where(np.abs(kz_trn_f) < 1e-12, 1.0, kz_trn_f)
    R_rows, T_rows, j_cols = [], [], []
    for ex0, ey0 in ((1.0, 0.0), (0.0, 1.0)):
        long_inc = (kx0 * ex0 + ky0 * ey0)
        einc_sq = 1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0 else 1.0
        cinc = np.concatenate([ex0 * delta, ey0 * delta])
        r = S11 @ cinc
        t = S21 @ cinc
        rx, ry = r[:Nf], r[Nf:]
        tx, ty = t[:Nf], t[Nf:]
        rz = -(kxv * rx + kyv * ry) / safe_r
        tz = -(kxv * tx + kyv * ty) / safe_t
        Re = np.real(kz_ref_f / kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2
                                           + np.abs(rz) ** 2) / einc_sq
        Te = np.real(kz_trn_f / kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                           + np.abs(tz) ** 2) / einc_sq
        R_rows.append(np.where(np.real(kz_ref_f) > 0, np.real(Re), 0.0))
        T_rows.append(np.where(np.real(kz_trn_f) > 0, np.real(Te), 0.0))
        j_cols.append(np.stack([np.conj(rx[p0]), np.conj(ry[p0])]))
    R_eff = np.stack(R_rows)
    T_eff = np.stack(T_rows)
    jones = np.stack(j_cols, axis=1)
    orders = np.stack([order_x, order_y], axis=1)
    return orders, R_eff, T_eff, jones
