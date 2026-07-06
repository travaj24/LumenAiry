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

Scope: ISOTROPIC scalar ridge/groove (:func:`pmm_jones_1d_conical`, the exp12
metal / dielectric grating case) AND the full-tensor (LC) conical layer
(:func:`pmm_jones_1d_conical_tensor`, Path B phase 2 -- the exp10/exp11 director
path).  The tensor entry routes the y-uniform (3, 3) profile -- including the
out-of-plane ``xz/yz`` director coupling -- through the SAME 2-D tensor
machinery (:func:`~lumenairy.elements.pmm.twod_jones._tensor_layer_modes`) with
``oy = [0]``, so an off-plane tensor picks up the generalized S-matrix generator
automatically.  MULTILAYER conical is wired into ``PMMStack`` (its
``_solve_conical`` Path-B-phase-4 cascade; ``PMMStack.set_source(..., phi)`` +
``solve()`` route there for ``phi != 0``, all-vertical NumPy stacks).
Validated against the analytic Berreman 4x4 conical oracle (uniform slab) and
the same-family Path A (PMM2DStack y-invariant) bridge.
"""
from __future__ import annotations

import numpy as np

from ..rcwa._core import (
    _interface_smatrix_general,
    _modes_to_M,
    _propagation_smatrix_general,
)
from .twod import (
    _build_axis,
    _cell_to_walls_tile,
    _homogeneous_modes,
    _interface_smatrix,
    _kz_forward2,
    _layer_modes_projected,
    _propagation_smatrix,
    _redheffer_star,
    _scalar_projected_ops,
)
from .twod_jones import _require_nonzero_ezz, _tensor_layer_modes

_C = complex

__all__ = ["pmm_jones_1d_conical", "pmm_jones_1d_conical_tensor"]


def _conical_jones_farfield(S11, S21, order_x, order_y, kxv, kyv, kx0, ky0,
                            eps_sup, eps_sub):
    """The shared conical far-field tail (drive unit ``Ex``/``Ey``; build
    ``R``/``T`` per incident polarization + the ``(2, 2)`` zeroth-order
    reflection Jones) -- the :func:`pmm_jones_2d` convention verbatim, reused by
    the isotropic and tensor native-conical entry points."""
    Nf = len(order_x)
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
    orders = np.stack([order_x, order_y], axis=1)
    return orders, np.stack(R_rows), np.stack(T_rows), np.stack(j_cols, axis=1)


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
    k0 = 2.0 * np.pi / wavelength
    kxv = kx0 + order_x * (wavelength / period)
    kyv = ky0 + order_y * (wavelength / period)          # == ky0 (constant)

    # ---- conical half-spaces (kz = sqrt(eps - kx^2 - ky^2)) ----
    Wsup, Vsup, _ls, _kzr = _homogeneous_modes(kxv, kyv, eps_sup)
    Wsub, Vsub, _lb, _kzt = _homogeneous_modes(kxv, kyv, eps_sub)

    # ---- coupled layer modes (GyF = ky0*I; the 2-D machinery, y degenerate) ----
    # A UNIFORM tile (eps_ridge == eps_groove) has DOUBLY-DEGENERATE modes (TE/TM
    # share kz at conical too), so the general eig of _layer_modes_projected
    # returns an arbitrary, BLAS-build-dependent eigenvector basis whose interface
    # solve corrupts the reflected Jones.  Detect it and use the analytic Rayleigh
    # modes (the _homogeneous_modes uniform path the half-spaces already use) --
    # exact, non-degenerate and deterministic.
    eps0 = eps_tile.flat[0]
    if bool(np.all(np.abs(eps_tile - eps0) < 1e-14)):
        Wl, Vl, lam_l, _ = _homogeneous_modes(kxv, kyv, eps0)
    else:
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

    return _conical_jones_farfield(S11, S21, order_x, order_y, kxv, kyv,
                                   kx0, ky0, eps_sup, eps_sub)


def pmm_jones_1d_conical_tensor(period, eps_tensor_cell, n_substrate,
                                n_superstrate, depth, wavelength, *,
                                theta, phi, degree=16, elements_per_region=1,
                                grade=True, n_orders=None, formulation="laurent"):
    r"""Conical (out-of-plane) diffraction of a FULL-TENSOR 1-D grating -- the
    native ``O(N)`` reduction for a y-uniform anisotropic (LC director) profile
    (Path B phase 2).

    ``eps_tensor_cell`` is the per-x-pixel permittivity tensor, shape
    ``(Sx, 3, 3)`` (or ``(Sx, 1, 3, 3)``), uniform along y (PUBLIC convention
    ``Im(eps) > 0`` for loss).  FULL ``(3, 3)`` tensors are supported, including
    the out-of-plane ``xz/yz/zx/zy`` coupling a tilted director carries: such a
    layer routes through the generalized S-matrix generator automatically (the
    ``[W; -V] <-> -lam`` symmetry is broken), exactly as :func:`pmm_jones_2d`.

    ``theta``/``phi`` are the conical incidence angles (radians); the remaining
    parameters mirror :func:`pmm_jones_1d_conical`.  Returns
    ``(orders, R(2, Nf), T(2, Nf), jones(2, 2))`` in the :func:`pmm_jones_2d`
    convention (the ``(m, 0)`` orders; rows = incident lab ``E_x``/``E_y``).

    Validation status.  This is a FAITHFUL ``O(N)`` reduction of the 2-D tensor
    path -- an IN-PLANE tensor at conical incidence matches the analytic
    Berreman 4x4 oracle to Berreman grade (singular values), an isotropic tensor
    reduces byte-exactly to :func:`pmm_jones_1d_conical`, and an OUT-OF-PLANE
    (tilted-director) tensor matches Berreman to MACHINE PRECISION at every
    incidence -- normal, planar-oblique, AND conical.  (Historical note: the
    docstring here previously reported a "few-percent OOP-at-conical residual vs
    Berreman"; that was an artifact of a BUG in the ``berreman_jones_1d`` S-matrix
    oracle it was graded against -- fixed 2026-07-05 -- NOT of this generator.
    With the corrected oracle the singular-value agreement is ``~1e-15``; this
    solver, :func:`pmm_jones_2d`, and ``rcwa_jones_2d`` were all correct.)
    """
    if formulation not in ("laurent", "li"):
        raise ValueError(
            f"pmm_jones_1d_conical_tensor: formulation must be 'laurent' or "
            f"'li', got {formulation!r}")
    cell = np.asarray(eps_tensor_cell, dtype=_C)
    if cell.ndim == 3:                             # (Sx, 3, 3) -> (Sx, 1, 3, 3)
        cell = cell[:, None, :, :]
    if cell.ndim != 4 or cell.shape[2:] != (3, 3):
        raise ValueError(
            f"pmm_jones_1d_conical_tensor: eps_tensor_cell must be (Sx, 3, 3) "
            f"or (Sx, 1, 3, 3), got shape {np.asarray(eps_tensor_cell).shape}.")
    if cell.shape[1] != 1:
        raise ValueError(
            "pmm_jones_1d_conical_tensor: the grating is uniform along y -- the "
            "cell's second axis must have length 1 (use pmm_jones_2d for a "
            "genuinely doubly-periodic tensor cell).")
    if n_orders is None:
        n_orders = max(8, degree // 2)
    x_walls, y_walls, tile = _cell_to_walls_tile(
        cell, period, period, "pmm_jones_1d_conical_tensor")
    _require_nonzero_ezz("pmm_jones_1d_conical_tensor", tile)
    tile_i = np.conj(tile)                          # INTERNAL exp(+iwt) gauge

    eps_sup = np.conj(_C(n_superstrate) ** 2)
    eps_sub = np.conj(_C(n_substrate) ** 2)
    ax = _build_axis(period, x_walls, degree, elements_per_region, grade)
    ay = _build_axis(period, [], degree, elements_per_region, grade)  # y uniform

    nre = float(np.real(np.sqrt(eps_sup)))
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)

    ox = np.arange(-n_orders, n_orders + 1)
    oy = np.array([0])
    order_x = np.tile(ox, len(oy))
    order_y = np.repeat(oy, len(ox))
    k0 = 2.0 * np.pi / wavelength
    kxv = kx0 + order_x * (wavelength / period)
    kyv = ky0 + order_y * (wavelength / period)          # == ky0 (constant)

    Wsup, Vsup, _ls, _kzr = _homogeneous_modes(kxv, kyv, eps_sup)
    Wsub, Vsub, _lb, _kzt = _homogeneous_modes(kxv, kyv, eps_sub)

    # coupled tensor layer modes (GyF = ky0*I; the 2-D tensor machinery, y
    # degenerate).  An out-of-plane tile returns the generator 6-tuple.
    modes = _tensor_layer_modes(
        ax, ay, x_walls, y_walls, tile_i, k0, kx0, ky0, ox, oy, kxv, kyv,
        formulation)
    if len(modes) == 3:
        Wl, Vl, lam_l = modes
        S = _interface_smatrix(Wsup, Vsup, Wl, Vl)
        S = _redheffer_star(S, _propagation_smatrix(lam_l, k0 * depth))
        S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wsub, Vsub))
    else:
        Wf, Vf, lam_f, Wb, Vb, lam_b = modes
        Msup = _modes_to_M(Wsup, Vsup, Wsup, -Vsup)
        Msub = _modes_to_M(Wsub, Vsub, Wsub, -Vsub)
        Ml = _modes_to_M(Wf, Vf, Wb, Vb)
        S = _interface_smatrix_general(Msup, Ml)
        S = _redheffer_star(
            S, _propagation_smatrix_general(lam_f, lam_b, k0 * depth))
        S = _redheffer_star(S, _interface_smatrix_general(Ml, Msub))
    S11, _S12, S21, _S22 = S

    return _conical_jones_farfield(S11, S21, order_x, order_y, kxv, kyv,
                                   kx0, ky0, eps_sup, eps_sub)
