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
    _grazing_safe_wavelength,
    _interface_smatrix_general,
    _modes_to_M,
    _propagation_smatrix_general,
    _require_propagating_incidence,
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

#: T3-3 fail-before switch (M1, 2026-08-04).  ``False`` restores the pre-M1
#: arithmetic bit for bit: the conical far-field order cap computed from the
#: FULL-UNION cell count on BOTH grid paths, including the per-layer one whose
#: half-spaces live on the (much smaller) three-layer window grids.
#:
#: Measured on the audit staircase (6 slices, 4 nm wall shift, degree 6,
#: ``elements_per_region`` 1): the union grid has 13 cells and carries
#: ``n_glob`` = 78, while the END WINDOW grids have 5 cells and carry 30 -- so
#: the shipped cap allowed 77 far-field orders on a projector built over 30
#: nodes, which supports 29.  ``False`` here is how the audit's before-numbers
#: were produced; see ``docs/audits/PMM_M1_CONDITIONING_2026_08_04.md`` S4.
PMM_CONICAL_PERLAYER_ORDER_CAP = True


def _conical_jones_farfield(S11, S21, order_x, order_y, kxv, kyv, kx0, ky0,
                            eps_sup, eps_sub, return_modal=False):
    """The shared conical far-field tail (drive unit ``Ex``/``Ey``; build
    ``R``/``T`` per incident polarization + the ``(2, 2)`` zeroth-order
    reflection Jones) -- the :func:`pmm_jones_2d` convention verbatim, reused by
    the isotropic and tensor native-conical entry points.

    Inputs are INTERNAL (``exp(+i w t)``) gauge; the returned Jones is PUBLIC
    (conjugated here).  ``return_modal=True`` (AUDIT_DYNAMETA_CONSUMER_API_GAPS
    B) additionally returns the per-order amplitude dict in the PUBLIC gauge
    (``rx``/``ry``/``tx``/``ty`` each ``(2, N)``, rows keyed to incident lab
    ``E_x``/``E_y``; the ``kz`` entries below are public-gauge already --
    ``conj(eps_internal)`` IS the public permittivity)."""
    Nf = len(order_x)
    p0 = int(np.where((order_x == 0) & (order_y == 0))[0][0])
    delta = ((order_x == 0) & (order_y == 0)).astype(_C)
    kz_inc = float(np.real(_kz_forward2(np.conj(eps_sup), kx0, ky0)))
    kz_ref_f = _kz_forward2(np.conj(eps_sup), kxv, kyv)
    kz_trn_f = _kz_forward2(np.conj(eps_sub), kxv, kyv)
    safe_r = np.where(np.abs(kz_ref_f) < 1e-12, 1.0, kz_ref_f)
    safe_t = np.where(np.abs(kz_trn_f) < 1e-12, 1.0, kz_trn_f)
    R_rows, T_rows, j_cols = [], [], []
    amp = {k: np.zeros((2, Nf), dtype=_C) for k in ("rx", "ry", "tx", "ty")}
    for col, (ex0, ey0) in enumerate(((1.0, 0.0), (0.0, 1.0))):
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
        amp["rx"][col], amp["ry"][col] = np.conj(rx), np.conj(ry)
        amp["tx"][col], amp["ty"][col] = np.conj(tx), np.conj(ty)
    orders = np.stack([order_x, order_y], axis=1)
    out = (orders, np.stack(R_rows), np.stack(T_rows),
           np.stack(j_cols, axis=1))
    if return_modal:
        modal = dict(orders=np.asarray(order_x).copy(), p0=p0,
                     kx=np.asarray(kxv).copy(), ky=np.asarray(kyv).copy(),
                     kz_ref=np.asarray(kz_ref_f).copy(),
                     kz_trn=np.asarray(kz_trn_f).copy(),
                     kz_inc=kz_inc, kx0=float(kx0), ky0=float(ky0), **amp)
        return out + (modal,)
    return out


def _tile_has_offplane_public(tensors):
    """True if any (3, 3) tensor in ``tensors`` carries out-of-plane coupling
    (``xz/yz/zx/zy``) -- conjugation-invariant, so PUBLIC-gauge safe."""
    for M in tensors:
        M = np.asarray(M, dtype=_C)
        scale = max(float(np.max(np.abs(M))), 1.0)
        off = max(abs(M[0, 2]), abs(M[1, 2]), abs(M[2, 0]), abs(M[2, 1]))
        if off > 1e-12 * scale:
            return True
    return False


def _conical_nodal_solve(period, layer_specs, eps_sup, eps_sub, wavelength,
                         theta, phi, degree, n_el, grade, n_orders,
                         min_feature=0.0, label="pmm conical (nodal)",
                         return_modal=False, layer_grids="shared"):
    """PURE-NODAL (no projection floor) conical multilayer cascade for
    PATTERNED in-plane stacks -- the fix for
    ``AUDIT_PMM_CONICAL_PATTERNED_TENSOR_BUG_2026_07_12``.

    The previous conical route built PATTERNED layer modes by projecting the
    spectral-element operators onto a truncated Fourier basis
    (:func:`~lumenairy.elements.pmm.twod_jones._tensor_layer_modes` /
    ``_layer_modes_projected``).  That operator-level compression carries a
    RESOLUTION-INDEPENDENT systematic error (~3e-3 Jones for a mild dielectric
    grating; the gap saturates in ``far_field_orders`` and GROWS with
    ``degree``), so the ``ky0 = 0`` degenerate limit never reduced to the
    converged classical solve -- for SCALAR gratings as well as tensors.

    This path instead runs the CLASSICAL (validated, spectrally-convergent)
    nodal machinery end-to-end, generalized to conical incidence:

    * per-layer modes from :func:`~lumenairy.elements.pmm._core._sem_modes_tensor`
      with the DIMENSIONAL ``ky0`` (uniform layers/half-spaces use the same
      eigensolver on uniform operators -- exact for constants);
    * the same union-grid + interface/propagation Redheffer cascade as
      :meth:`PMMStack._solve`;
    * far field by projecting the nodal solution onto Rayleigh orders
      (:func:`~lumenairy.elements.pmm._core._sem_fourier_projection`) and
      applying the conical vector far-field math in the PUBLIC ``exp(-i w t)``
      gauge (no conjugations -- the classical machinery is public end-to-end).

    At ``ky0 == 0`` every added operator term vanishes, so this reduces
    BIT-IDENTICALLY (modal content) to the classical solve -- the audit's
    degenerate-limit gate.  ``layer_specs`` is a list of
    ``(depth_m, widths, tensors)`` with ``widths`` the segment width fractions
    and ``tensors`` the PUBLIC ``(3, 3)`` permittivities.  IN-PLANE tensors
    only (callers reject out-of-plane patterned cells loudly).
    """
    from ._core import (
        _build_sem_tensor_segments,
        _guarded_lstsq,
        _interface_smatrix_mortar,
        _n_propagating_orders,
        _pmm_union_grid,
        _redheffer_star_rect,
        _sem_cross_mass,
        _sem_fourier_projection,
        _sem_mass_exact,
        _sem_modes_tensor,
        _tensor3_dict,
    )
    from ._core import (
        _interface_smatrix as _ifc_n,
    )
    from ._core import (
        _propagation_star as _prop_star_n,
    )
    from ._core import (
        _redheffer_star as _star_n,
    )

    eps_sup = _C(eps_sup)
    eps_sub = _C(eps_sub)
    nre = float(np.real(np.sqrt(eps_sup)))
    kx0 = nre * np.sin(theta) * np.cos(phi)            # normalized (n_eff)
    ky0 = nre * np.sin(theta) * np.sin(phi)
    _require_propagating_incidence(label, np.conj(eps_sup),
                                   kx0 ** 2 + ky0 ** 2)

    # ---- union nodal grid across layers (interface matching needs ONE grid) --
    segs_layers = [
        [(float(w), np.asarray(e, dtype=_C)) for w, e in zip(widths, tensors)]
        for (_d, widths, tensors) in layer_specs
    ]
    uwidths, layer_eps_u = _pmm_union_grid(
        segs_layers, (min_feature / period) if min_feature else None)
    nU = len(uwidths)

    nlay = len(segs_layers)
    # ---- PER-LAYER window grids, built BEFORE the far-field budget ---------
    # T3-3 (M1, 2026-08-04).  This block used to sit AFTER the order cap, and
    # the cap was computed from ``nU`` -- the FULL-UNION cell count -- on both
    # paths.  On the per-layer path the half-spaces live on the WINDOW grids,
    # whose cell count is a fraction of ``nU`` (~nU/6 on the audit device), so
    # the cap OVER-STATED the nodal capacity, the ``m_prop > cap`` raise could
    # not fire, and ``_sem_fourier_projection`` then built a projector with
    # more Rayleigh orders than the grid has nodes.  ``Hsup`` went rank
    # deficient and the least squares below returned a build-dependent
    # null-space draw -- silently, because a null-space component of ``cinc``
    # is invisible to ``Hsup`` and so leaves R + T = 1 intact.
    #
    # Every sibling per-layer path already clamps to the WINDOW half-spaces and
    # raises: ``PMMStack._solve_vertical_perlayer`` (stack.py, ``cap =
    # min(n0, nN)`` from ``mats_sup/sub["n_glob"]``), the per-layer sweep
    # ``_solve_vs_wavelength_perlayer``, and the JAX twin (min over the two end
    # grids' capacity).  Conical was the only outlier.
    per_layer = layer_grids == "per-layer"
    t_sup = _tensor3_dict(eps_sup * np.eye(3))
    t_sub = _tensor3_dict(eps_sub * np.eye(3))
    grid_of = mats_sup = mats_sub = None
    if per_layer:
        grid_of = []
        for i in range(nlay):
            js = [j for j in (i - 1, i, i + 1) if 0 <= j < nlay]
            uw_i, rows_i = _pmm_union_grid(
                [segs_layers[j] for j in js],
                (min_feature / period) if min_feature else None)
            grid_of.append((np.asarray(uw_i, dtype=float),
                            rows_i[js.index(i)]))
        mats_sup = _build_sem_tensor_segments(
            period, grid_of[0][0], [t_sup] * len(grid_of[0][0]),
            degree, n_el, grade)
        mats_sub = _build_sem_tensor_segments(
            period, grid_of[-1][0], [t_sub] * len(grid_of[-1][0]),
            degree, n_el, grade)

    # far-field order budget: capped by the nodal capacity of the grids the
    # HALF-SPACES actually live on, and the grid must resolve every propagating
    # order (classical-path parity).
    if per_layer and PMM_CONICAL_PERLAYER_ORDER_CAP:
        cap = (min(int(mats_sup["n_glob"]), int(mats_sub["n_glob"])) - 1) // 2
    else:
        cap = (nU * n_el * degree - 1) // 2
    n_orders = max(1, min(int(n_orders), cap))
    _idx = [float(np.real(np.sqrt(eps_sup))), float(np.real(np.sqrt(eps_sub)))]
    for eps_u in layer_eps_u:
        for e in eps_u:
            _idx.append(float(np.real(np.sqrt(np.max(np.abs(np.diag(e)))))))
    m_prop = _n_propagating_orders(period, wavelength, max(_idx))
    if m_prop > cap:
        raise ValueError(
            f"{label}: the spectral-element grid resolves only {2 * cap + 1} "
            f"orders but {2 * m_prop + 1} propagate; raise degree / "
            f"elements_per_region.")

    ox = np.arange(-n_orders, n_orders + 1)
    order_x = ox
    order_y = np.zeros_like(ox)
    # grazing-safe wavelength (Re(eps) drives the cutoffs -- gauge-immaterial)
    eps_reals = [eps_sup, eps_sub]
    for eps_u in layer_eps_u:
        eps_reals.extend(complex(v) for e in eps_u
                         for v in np.diag(np.asarray(e, dtype=_C)))
    wl = _grazing_safe_wavelength(float(wavelength), kx0, ky0, order_x,
                                  order_y, period, period, eps_reals)
    k0 = 2.0 * np.pi / wl
    kxv = kx0 + order_x * (wl / period)
    kyv = ky0 + 0.0 * order_x                          # constant ky0

    # ---- nodal operators + modes (PUBLIC gauge; DIMENSIONAL kx0/ky0) --------
    kx0_dim = kx0 * k0
    ky0_dim = ky0 * k0
    if per_layer:
        # ---- PER-LAYER grids + interface mortar (audit R-6, conical v2) ----
        # Identical construction to PMMStack._solve_vertical_perlayer: each
        # layer's grid = its own walls + its two NEIGHBOURS' (the interface-
        # conforming window, via _pmm_union_grid over the 3-layer window with
        # the window-local min_feature), coupled by the exact L2 mortar.  Only
        # the eigensolve differs (ky0-shifted) -- the mortar is geometry-level
        # and gauge-free, so it drops in unchanged.
        # ``grid_of`` / ``mats_sup`` / ``mats_sub`` were built ABOVE, because
        # the far-field order cap has to be derived from them (T3-3).
        mats_by, lmodes = [], []
        _eig_memo = {}
        for w_i, eps_row in grid_of:
            ck = (w_i.tobytes(),
                  tuple(np.asarray(e, dtype=_C).tobytes() for e in eps_row))
            hit = _eig_memo.get(ck)
            if hit is None:
                m = _build_sem_tensor_segments(
                    period, w_i, [_tensor3_dict(e) for e in eps_row],
                    degree, n_el, grade)
                hit = (m, _sem_modes_tensor(m, k0, kx0_dim, ky0=ky0_dim))
                _eig_memo[ck] = hit
            mats_by.append(hit[0])
            lmodes.append(hit[1])
        wkeys = [w.tobytes() for w, _r in grid_of]
        Wsup, Vsup, _ls, _qs = _sem_modes_tensor(mats_sup, k0, kx0_dim,
                                                 ky0=ky0_dim)
        Wsub, Vsub, _lb, _qb = _sem_modes_tensor(mats_sub, k0, kx0_dim,
                                                 ky0=ky0_dim)
        _mass_memo = {}

        def _mass(i):
            M = _mass_memo.get(wkeys[i])
            if M is None:
                M = _sem_mass_exact(mats_by[i])
                _mass_memo[wkeys[i]] = M
            return M

        def _ifc_pl(Wa, Va, Wb, Vb, ia, ib):
            if ia is None or ib is None or wkeys[ia] == wkeys[ib]:
                return _ifc_n(Wa, Va, Wb, Vb)
            return _interface_smatrix_mortar(
                Wa, Va, Wb, Vb, _mass(ia), _mass(ib),
                _sem_cross_mass(mats_by[ia], mats_by[ib]))

        S = _ifc_pl(Wsup, Vsup, lmodes[0][0], lmodes[0][1], None, 0)
        for i in range(nlay):
            S = _prop_star_n(S, lmodes[i][2], k0 * layer_specs[i][0])
            if i < nlay - 1:
                nxt = _ifc_pl(lmodes[i][0], lmodes[i][1],
                              lmodes[i + 1][0], lmodes[i + 1][1], i, i + 1)
            else:
                nxt = _ifc_pl(lmodes[i][0], lmodes[i][1], Wsub, Vsub,
                              i, None)
            S = _redheffer_star_rect(S, nxt)
        S11, _S12, S21, _S22 = S
        n_glob_sup = mats_sup["n_glob"]
        n_glob_sub = mats_sub["n_glob"]
    else:
        layer_mats = [
            _build_sem_tensor_segments(period, uwidths,
                                       [_tensor3_dict(e) for e in eps_u],
                                       degree, n_el, grade)
            for eps_u in layer_eps_u]
        mats_sup = _build_sem_tensor_segments(period, uwidths, [t_sup] * nU,
                                              degree, n_el, grade)
        mats_sub = _build_sem_tensor_segments(period, uwidths, [t_sub] * nU,
                                              degree, n_el, grade)
        n_glob_sup = n_glob_sub = mats_sup["n_glob"]

        Wsup, Vsup, _ls, _qs = _sem_modes_tensor(mats_sup, k0, kx0_dim,
                                                 ky0=ky0_dim)
        Wsub, Vsub, _lb, _qb = _sem_modes_tensor(mats_sub, k0, kx0_dim,
                                                 ky0=ky0_dim)
        lmodes = []
        _eig_memo = {}
        for eps_u, m in zip(layer_eps_u, layer_mats):
            ck = tuple(np.asarray(e, dtype=_C).tobytes() for e in eps_u)
            hit = _eig_memo.get(ck)
            if hit is None:
                hit = _sem_modes_tensor(m, k0, kx0_dim, ky0=ky0_dim)
                _eig_memo[ck] = hit
            lmodes.append(hit)

        # ---- symmetric Redheffer cascade (in-plane +/-q symmetry) ----------
        ifc = [_ifc_n(Wsup, Vsup, lmodes[0][0], lmodes[0][1])]
        for i in range(1, nlay):
            ifc.append(_ifc_n(lmodes[i - 1][0], lmodes[i - 1][1],
                              lmodes[i][0], lmodes[i][1]))
        ifc.append(_ifc_n(lmodes[-1][0], lmodes[-1][1], Wsub, Vsub))
        S = ifc[0]
        for i in range(nlay):
            S = _prop_star_n(S, lmodes[i][2], k0 * layer_specs[i][0])
            S = _star_n(S, ifc[i + 1])
        S11, _S12, S21, _S22 = S

    # ---- conical vector far field (PUBLIC gauge: no conjugations) ----------
    Tp_sup = _sem_fourier_projection(ox, period, mats_sup)
    Tp_sub = (Tp_sup if mats_sub is mats_sup
              else _sem_fourier_projection(ox, period, mats_sub))
    Hsup = np.vstack([Tp_sup @ Wsup[:n_glob_sup, :],
                      Tp_sup @ Wsup[n_glob_sup:, :]])
    Hsub = np.vstack([Tp_sub @ Wsub[:n_glob_sub, :],
                      Tp_sub @ Wsub[n_glob_sub:, :]])
    Nf = len(ox)
    p0 = int(np.where(ox == 0)[0][0])
    delta = (ox == 0).astype(_C)
    # PUBLIC-gauge forward kz, evaluated directly on the PUBLIC eps (the
    # branch with Im(kz) >= 0 = forward decay, matching the classical path's
    # _kz_forward(eps_public) and RCWA's modal kz_ref).  The earlier
    # conj(_kz_forward2(conj(eps))) "gauge map" flipped the EVANESCENT branch
    # to Im < 0 for lossless media (conj of +i|kz|); the flux math below uses
    # Re() and the Re > 0 masks only, so R/T/jones are bit-identical either
    # way -- but the per_order_amplitudes modal export (B) reports these kz
    # and must carry the public decaying branch.
    kz_inc = float(np.real(_kz_forward2(eps_sup, kx0, ky0)))
    kz_ref_f = _kz_forward2(eps_sup, kxv, kyv)
    kz_trn_f = _kz_forward2(eps_sub, kxv, kyv)
    safe_r = np.where(np.abs(kz_ref_f) < 1e-12, 1.0, kz_ref_f)
    safe_t = np.where(np.abs(kz_trn_f) < 1e-12, 1.0, kz_trn_f)
    R_rows, T_rows, j_cols = [], [], []
    amp = {k: np.zeros((2, Nf), dtype=_C) for k in ("rx", "ry", "tx", "ty")}
    for col, (ex0, ey0) in enumerate(((1.0, 0.0), (0.0, 1.0))):
        long_inc = (kx0 * ex0 + ky0 * ey0)
        einc_sq = 1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0 else 1.0
        rhs = np.concatenate([ex0 * delta, ey0 * delta])
        cinc = _guarded_lstsq(
            Hsup, rhs, "pmm conical far-field Rayleigh projection")
        r = Hsup @ (S11 @ cinc)
        t = Hsub @ (S21 @ cinc)
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
        j_cols.append(np.stack([rx[p0], ry[p0]]))
        # this cascade is PUBLIC gauge end-to-end -> amplitudes need no conj
        amp["rx"][col], amp["ry"][col] = rx, ry
        amp["tx"][col], amp["ty"][col] = tx, ty
    orders = np.stack([order_x, order_y], axis=1)
    out = (orders, np.stack(R_rows), np.stack(T_rows),
           np.stack(j_cols, axis=1))
    if return_modal:
        # AUDIT_DYNAMETA_CONSUMER_API_GAPS B: per-order complex tangential
        # amplitudes (PUBLIC exp(-iwt) gauge), the RCWAResult modal contract.
        modal = dict(orders=np.asarray(order_x).copy(), p0=p0,
                     kx=np.asarray(kxv).copy(), ky=np.asarray(kyv).copy(),
                     kz_ref=np.asarray(kz_ref_f).copy(),
                     kz_trn=np.asarray(kz_trn_f).copy(),
                     kz_inc=kz_inc, kx0=float(kx0), ky0=float(ky0),
                     wavelength=float(wl), **amp)
        return out + (modal,)
    return out


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
    # PATTERNED grating -> the pure-nodal conical cascade
    # (AUDIT_PMM_CONICAL_PATTERNED_TENSOR_BUG_2026_07_12): the Fourier-
    # projected route below carries a RESOLUTION-INDEPENDENT systematic error
    # for any patterned cell (scalar included -- the bug is in the operator
    # projection, not the tensor factorization), so its ky0=0 degenerate limit
    # never reduces to the classical solve.  ``formulation`` is inert on this
    # path (the nodal build resolves walls geometrically -- no factorization
    # rule).  A UNIFORM cell keeps the exact analytic Fourier path below.
    duty = float(duty_cycle)
    if 0.0 < duty < 1.0 and _C(eps_ridge) != _C(eps_groove):
        I3 = np.eye(3, dtype=_C)
        return _conical_nodal_solve(
            period, [(depth, [duty, 1.0 - duty],
                      [_C(eps_ridge) * I3, _C(eps_groove) * I3])],
            _C(n_superstrate) ** 2, _C(n_substrate) ** 2, wavelength,
            theta, phi, degree, elements_per_region, grade, n_orders,
            label="pmm_jones_1d_conical")
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
    # Suite-standard incidence guard (D1): reject a gain / evanescent
    # superstrate before the kz_inc-normalised far field would silently
    # negate or NaN the efficiencies -- the guard every other PMM/RCWA entry
    # runs (2-D core twod.py:752).
    _require_propagating_incidence("pmm_jones_1d_conical", eps_sup,
                                   kx0 ** 2 + ky0 ** 2)

    ox = np.arange(-n_orders, n_orders + 1)
    oy = np.array([0])
    order_x = np.tile(ox, len(oy))
    order_y = np.repeat(oy, len(ox))
    # Nudge the wavelength off any EXACT Rayleigh cutoff -- of the half-spaces
    # OR a grating constituent -- so kz_inc-normalisation and the interface
    # S-matrix stay well-conditioned (a grazing layer mode otherwise crashes
    # the interface solve); no-op away from a cutoff, +/-m symmetry preserved.
    eps_reals = [eps_sup, eps_sub] + [complex(e) for e in
                                      np.asarray(eps_tile).ravel()]
    wl = _grazing_safe_wavelength(float(wavelength), kx0, ky0, order_x,
                                  order_y, period, period, eps_reals)
    k0 = 2.0 * np.pi / wl
    kxv = kx0 + order_x * (wl / period)
    kyv = ky0 + order_y * (wl / period)                  # == ky0 (constant)

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
    # PATTERNED tensor cell -> the pure-nodal conical cascade
    # (AUDIT_PMM_CONICAL_PATTERNED_TENSOR_BUG_2026_07_12): the projected route
    # below produced a resolution-independent ~3.5 deg retardance error for a
    # patterned anisotropic layer (its ky0=0 limit never reduced to the
    # converged classical tensor solve).  IN-PLANE tensors only -- a PATTERNED
    # out-of-plane cell is rejected loudly (the old path returned silently
    # wrong numbers for it); a UNIFORM cell of any tensor keeps the exact
    # Fourier path below (Berreman-validated, incl. out-of-plane).
    if len(x_walls) > 0:
        tens = [np.asarray(tile[s, 0], dtype=_C)
                for s in range(tile.shape[0])]
        if _tile_has_offplane_public(tens):
            raise NotImplementedError(
                "pmm_jones_1d_conical_tensor: a PATTERNED cell whose tensor "
                "carries out-of-plane coupling (eps_xz/yz/zx/zy) is not "
                "supported at conical incidence -- the previous projected "
                "route returned silently-wrong retardance for patterned "
                "cells (AUDIT_PMM_CONICAL_PATTERNED_TENSOR_BUG_2026_07_12) "
                "and the nodal conical cascade is in-plane only.  Use the "
                "classical mount (phi=0, full out-of-plane support), a "
                "UNIFORM out-of-plane layer (Berreman-exact here), or "
                "PMM2DStackPure.")
        walls = np.concatenate([[0.0], np.asarray(x_walls, dtype=float),
                                [period]])
        widths = list(np.diff(walls) / period)
        return _conical_nodal_solve(
            period, [(depth, widths, tens)],
            _C(n_superstrate) ** 2, _C(n_substrate) ** 2, wavelength,
            theta, phi, degree, elements_per_region, grade, n_orders,
            label="pmm_jones_1d_conical_tensor")
    tile_i = np.conj(tile)                          # INTERNAL exp(+iwt) gauge

    eps_sup = np.conj(_C(n_superstrate) ** 2)
    eps_sub = np.conj(_C(n_substrate) ** 2)
    ax = _build_axis(period, x_walls, degree, elements_per_region, grade)
    ay = _build_axis(period, [], degree, elements_per_region, grade)  # y uniform

    nre = float(np.real(np.sqrt(eps_sup)))
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)
    # Suite-standard incidence guard (D1): reject a gain / evanescent
    # superstrate (kz_inc-normalisation would silently negate/NaN the far
    # field) -- mirrors every other PMM/RCWA entry.
    _require_propagating_incidence("pmm_jones_1d_conical_tensor", eps_sup,
                                   kx0 ** 2 + ky0 ** 2)

    ox = np.arange(-n_orders, n_orders + 1)
    oy = np.array([0])
    order_x = np.tile(ox, len(oy))
    order_y = np.repeat(oy, len(ox))
    # Grazing-safe wavelength (D1): include the tensor layer's DIAGONAL
    # permittivities as constituent cutoff-setters alongside the half-spaces,
    # so a grazing layer mode can't crash the generalized S-matrix.  Re() is
    # conjugation-invariant, so the public ``tile`` values are fine here.
    eps_reals = ([eps_sup, eps_sub]
                 + [complex(e) for e in np.einsum("...ii->...i", tile).ravel()])
    wl = _grazing_safe_wavelength(float(wavelength), kx0, ky0, order_x,
                                  order_y, period, period, eps_reals)
    k0 = 2.0 * np.pi / wl
    kxv = kx0 + order_x * (wl / period)
    kyv = ky0 + order_y * (wl / period)                  # == ky0 (constant)

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
