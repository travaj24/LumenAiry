"""
lumenairy.elements.pmm._jax_twod_jones -- JAX twin of pmm_jones_2d.
===================================================================

Differentiable forward solve for :func:`pmm_jones_2d` (the anisotropic 2-D
hybrid PMM with Jones output) on JAX inputs: gradients flow through the
per-region (3, 3) permittivity tensor VALUES (real and imaginary parts),
``n_substrate`` / ``n_superstrate``, ``depth``, ``wavelength`` and
``theta`` / ``phi``.

Geometry scope (mirrors the scalar cell twin
:func:`~lumenairy.elements.pmm._jax_twod._pmm_efficiency_2d_cell_jax`): the
spectral-element walls come from a CONCRETE ``region_layout`` (an int grid of
the same ``(Sx, Sy)`` shape labelling the regions); a traced
``eps_tensor_cell`` supplies the region tensor VALUES (read at each region's
first pixel).  A traced array cannot define the exact walls -- that is
data-dependent control flow (``_cell_to_walls_tile`` compares pixel values).

Out-of-plane routing -- the routing-bug lesson from the RCWA-2D twin (commit
b6953d1).  Under a trace the tensor is a Tracer whose out-of-plane structure is
invisible (``to_numpy`` fails), so keying the in-plane-vs-out-of-plane branch on
the tensor VALUES would silently diverge between the concrete forward and the
gradient (the exact v5.20.1 RCWA bug: a ~30%-wrong gradient with no error).
This twin therefore ALWAYS drives the full-3x3 GENERATOR path
(:func:`~lumenairy.elements.rcwa._core._layer_eigenmodes_tensor` with the
off-plane cross-blocks ``A``/``B``), which is EXACT for an in-plane tensor (the
off-plane blocks vanish -> ``A = B = 0``, the pointwise ``e_zz``-Schur reduction
is a no-op, and ``G = [[0, P], [Q, 0]]`` reproduces ``eig(P@Q)``) and correct
for a genuinely out-of-plane one.  Forward and gradient thus take the SAME
branch.  COST: the generator eig is ``4*Nf`` (vs the concrete in-plane path's
``2*Nf``, ~8x slower) -- prefer modest ``n_orders`` for in-plane gradient design
loops (the NumPy / concrete forward keeps the fast ``2*Nf`` path).

BOTH-AXES-PATTERNED cells only.  A cell patterned along BOTH axes couples every
Fourier order and lifts the modal degeneracy, so ``jnp.linalg.eig`` is
well-conditioned and the twin reproduces the NumPy solve to MACHINE PRECISION
(~1e-13, verified in-plane AND out-of-plane at well-conditioned truncations).  A
cell UNIFORM ALONG AN AXIS (a fully uniform single-region cell, or a
1-axis-patterned "stripe") leaves that axis's orders uncoupled -> a degenerate
spectrum for which ``jnp.linalg.eig`` returns an ill-conditioned eigenbasis
(~1e-3 error; the same degeneracy the symmetric branch cures with an iso-uniform
blend, which the full-3x3 generator lacks).  The public entry RAISES for it and
points at the 1-D differentiable solvers (``pmm_jones_1d`` / ``rcwa_jones_1d``
for a 1-D grating, ``berreman_jones_1d`` for a uniform anisotropic layer -- the
exact, analytic planar tool).

The forward/backward modal split uses the trace-safe ``jnp.argsort`` flux
selector ``_select_forward_flux_jax`` (already wired into
``_layer_eigenmodes_tensor``'s jax branch); the generalized S-matrix cascade
(``_interface_smatrix_general`` / ``_propagation_smatrix_general``) is
backend-agnostic.  Caveats (mirroring the scalar twin): x64 required;
``jnp.linalg.eig`` is CPU-only; ``stabilize`` is rejected (host-side degree
scan, handled at the public entry); ``_check_energy`` is skipped (a traced
non-reciprocal out-of-plane cell can be physically ``R + T != 1``); the
concrete-only incidence + Wood-anomaly guards run only when the source values
are concrete (a traced source skips them -- gradients valid only between order
cutoffs).  Loss convention: PUBLIC ``Im(eps) > 0`` for loss, the conjugation
bridge is internal, the Jones matrix is conjugated back at extraction.
"""
from __future__ import annotations

import numpy as np

from ...backend import is_jax_array  # noqa: F401  (re-exported for the dispatch)
from ._jax_twod import _host_incidence_guard, _static_prep_cell

_C = np.complex128


def _pmm_jones_2d_cell_jax(period_x, period_y, eps_tensor_cell, region_layout,
                           n_substrate, n_superstrate, depth, wavelength, *,
                           theta, phi, degree, elements_per_strip, grade,
                           n_orders, formulation):
    """Differentiable ``pmm_jones_2d`` on a CONCRETE ``region_layout`` (walls) +
    a TRACED ``eps_tensor_cell`` (region tensor values).  Returns
    ``(orders(N,2), R_eff(2,N), T_eff(2,N), jones(2,2))`` matching the NumPy
    :func:`pmm_jones_2d` contract."""
    import jax.numpy as jnp

    from ..rcwa import _require_jax_x64
    from ..rcwa._core import (
        _interface_smatrix_general,
        _layer_eigenmodes_tensor,
        _modes_to_M,
        _propagation_smatrix_general,
        _redheffer_star,
    )
    _require_jax_x64("pmm_jones_2d")
    cj = jnp.complex128

    st = _static_prep_cell(float(period_x), float(period_y), region_layout,
                           int(degree), int(elements_per_strip), bool(grade),
                           int(n_orders))
    # concrete-only incidence + Wood-anomaly guards (a traced source value skips
    # them); pass the PUBLIC tensor DIAGONAL as the Wood constituent set (mirrors
    # the NumPy path's eps_reals -- diagonal only; the guard Re()s them).
    diag_candidate = jnp.asarray(eps_tensor_cell)[..., [0, 1, 2], [0, 1, 2]]
    wavelength = _host_incidence_guard(
        "pmm_jones_2d", st, float(period_x), float(period_y),
        wavelength, theta, phi, n_superstrate, n_substrate, (diag_candidate,))

    # ---- traced per-region tensor values (loss bridge: PUBLIC Im>0 -> conj) --
    eps_c = jnp.asarray(eps_tensor_cell).astype(cj)
    reg = jnp.stack([eps_c[i, j] for (i, j) in st["first_idx"]])   # (nreg, 3, 3)
    reg = jnp.conj(reg)
    Wreg = jnp.asarray(st["Wreg"], cj)                             # (nreg, Ndof)
    Tp = jnp.asarray(st["Tp"], cj)
    Tpinv = jnp.asarray(st["Tpinv"], cj)
    Gx0F = jnp.asarray(st["Gx0F"], cj)
    Gy0F = jnp.asarray(st["Gy0F"], cj)
    IprojF = jnp.asarray(st["IprojF"], cj)
    order_x = st["order_x"]
    order_y = st["order_y"]
    Nf = order_x.size

    def _proj(nodal):
        # (Tp * v) @ Tpinv == Tp @ diag(v) @ Tpinv  (the GLL nodal mass is
        # diagonal, so a multiply-by-component operator is a nodal VECTOR)
        return (Tp * nodal[None, :]) @ Tpinv

    def _nodal(regvals):
        # region values -> projected nodal vector, LINEAR in the traced values
        # (Wreg[r] is the partition-of-unity mass of region r); exactly the
        # NumPy _tensor_layer_modes _nodal()/_proj() sum over strip cells.
        return regvals @ Wreg

    # pointwise e_zz-Schur reduction of the in-plane 2x2 (per region), BEFORE
    # projection (Li 2003; the "gen2 trap" is doing it spectrally).  For an
    # in-plane cell the off-plane entries are zero -> this is a no-op.
    izz = reg[:, 2, 2]

    def _te(a, b):
        return reg[:, a, b] - reg[:, a, 2] * reg[:, 2, b] / izz

    CxxF = _proj(_nodal(_te(0, 0)))
    CxyF = _proj(_nodal(_te(0, 1)))
    CyxF = _proj(_nodal(_te(1, 0)))
    CyyF = _proj(_nodal(_te(1, 1)))
    if formulation == "li":
        # the generator computes Ez_inv = inv(EZZ); feeding inv(proj(1/ezz))
        # makes Ez_inv == the projected multiply-by-1/ezz (inverse-rule)
        EZZ = jnp.linalg.inv(_proj(_nodal(1.0 / izz)))
    else:
        EZZ = _proj(_nodal(izz))                    # laurent (rcwa_jones_2d mirror)
    # RAW off-plane components (the generator cross-blocks + Ez elimination use
    # them directly); zero for an in-plane cell -> A = B = 0.
    EZX = _proj(_nodal(reg[:, 2, 0]))
    EZY = _proj(_nodal(reg[:, 2, 1]))
    EXZ = _proj(_nodal(reg[:, 0, 2]))
    EYZ = _proj(_nodal(reg[:, 1, 2]))

    # ---- incidence geometry -------------------------------------------------
    eps_sup = jnp.conj(jnp.asarray(n_superstrate).astype(cj) ** 2)
    eps_sub = jnp.conj(jnp.asarray(n_substrate).astype(cj) ** 2)
    k0 = 2.0 * jnp.pi / wavelength
    nre = jnp.real(jnp.sqrt(eps_sup))
    kx0 = nre * jnp.sin(theta) * jnp.cos(phi)
    ky0 = nre * jnp.sin(theta) * jnp.sin(phi)
    ox_j = jnp.asarray(order_x)
    oy_j = jnp.asarray(order_y)
    kxv = kx0 + ox_j * (wavelength / period_x)
    kyv = ky0 + oy_j * (wavelength / period_y)

    GxF = Gx0F / k0 + kx0 * IprojF
    GyF = Gy0F / k0 + ky0 * IprojF

    # ---- layer modes: ALWAYS the full-3x3 generator (see module docstring) --
    Wf, Vf, lam_f, Wb, Vb, lam_b = _layer_eigenmodes_tensor(
        GxF, GyF, CxxF, CxyF, CyxF, CyyF, EZZ,
        EZX=EZX, EZY=EZY, EXZ=EXZ, EYZ=EYZ)

    # ---- isotropic half-space modes (analytic Rayleigh, W = I) --------------
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
        eps_I = eps * jnp.eye(Nf, dtype=cj)
        Q = jnp.block([[Kx @ Ky, eps_I - Kx @ Kx],
                       [Ky @ Ky - eps_I, -Ky @ Kx]])
        W = jnp.eye(2 * Nf, dtype=cj)
        V = Q @ jnp.diag(_inv_lam(lam))
        return W, V, lam

    Wsup, Vsup, _ls = _homog(eps_sup)
    Wsub, Vsub, _lb = _homog(eps_sub)

    # ---- generalized S-matrix cascade (sup | layer | sub) -------------------
    Msup = _modes_to_M(Wsup, Vsup, Wsup, -Vsup)
    Msub = _modes_to_M(Wsub, Vsub, Wsub, -Vsub)
    Ml = _modes_to_M(Wf, Vf, Wb, Vb)
    S = _interface_smatrix_general(Msup, Ml)
    S = _redheffer_star(
        S, _propagation_smatrix_general(lam_f, lam_b, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix_general(Ml, Msub))
    S11, _S12, S21, _S22 = S

    # ---- far-field flux tail (mirror _pmm_jones_2d_at) ----------------------
    kz_inc = jnp.real(_kz_fwd(jnp.conj(eps_sup), kx0, ky0))
    kz_ref_f = _kz_fwd(jnp.conj(eps_sup), kxv, kyv)
    kz_trn_f = _kz_fwd(jnp.conj(eps_sub), kxv, kyv)
    safe_r = jnp.where(jnp.abs(kz_ref_f) < 1e-12, 1.0, kz_ref_f)
    safe_t = jnp.where(jnp.abs(kz_trn_f) < 1e-12, 1.0, kz_trn_f)
    safe_inc = jnp.where(kz_inc == 0, 1.0, kz_inc)
    delta = jnp.asarray(((order_x == 0) & (order_y == 0)).astype(_C))
    p0 = int(np.where((order_x == 0) & (order_y == 0))[0][0])

    R_rows, T_rows, j_cols = [], [], []
    for (ex0, ey0) in ((1.0, 0.0), (0.0, 1.0)):
        long_inc = kx0 * ex0 + ky0 * ey0
        einc_sq = 1.0 + (long_inc / safe_inc) ** 2   # incident sec^2 inflation
        cinc = jnp.concatenate([ex0 * delta, ey0 * delta])
        r = S11 @ cinc
        t = S21 @ cinc
        rx, ry = r[:Nf], r[Nf:]
        tx, ty = t[:Nf], t[Nf:]
        rz = -(kxv * rx + kyv * ry) / safe_r
        tz = -(kxv * tx + kyv * ty) / safe_t
        Re = jnp.real(kz_ref_f / kz_inc) * (jnp.abs(rx) ** 2 + jnp.abs(ry) ** 2
                                            + jnp.abs(rz) ** 2) / einc_sq
        Te = jnp.real(kz_trn_f / kz_inc) * (jnp.abs(tx) ** 2 + jnp.abs(ty) ** 2
                                            + jnp.abs(tz) ** 2) / einc_sq
        R_rows.append(jnp.where(jnp.real(kz_ref_f) > 0, jnp.real(Re), 0.0))
        T_rows.append(jnp.where(jnp.real(kz_trn_f) > 0, jnp.real(Te), 0.0))
        # PUBLIC-convention Jones: conjugate back out of the internal gauge
        j_cols.append(jnp.stack([jnp.conj(rx[p0]), jnp.conj(ry[p0])]))

    R_eff = jnp.stack(R_rows)
    T_eff = jnp.stack(T_rows)
    jones_reflection = jnp.stack(j_cols, axis=1)
    orders2d = np.stack([order_x, order_y], axis=1)
    return orders2d, R_eff, T_eff, jones_reflection
