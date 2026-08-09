"""Differentiable (JAX) ``PMMStack.solve`` -- the MULTILAYER twin of the
single-layer ``_pmm_jones_1d_jax`` surface (same building blocks: frozen NumPy
geometry, eps-linear traced assembly, the gauge-stable custom-VJP eig, a pure
matmul Redheffer cascade, and the closed-form min-norm far-field projection).

Scope (matches the single-layer twin's in-plane vertical surface):

* ALL-VERTICAL stacks of IN-PLANE (possibly anisotropic) layers -- scalar or
  ``(3, 3)`` tensors whose out-of-plane components are zero.  Slanted layers,
  ``stabilize``, ``retain_internal`` and dispersive / material-key layers
  raise upstream (in :meth:`PMMStack.solve`'s dispatch).
* Differentiable w.r.t. every segment permittivity (real AND imaginary part,
  scalar or in-plane tensor entries), every layer ``thickness``, the
  half-space indices, ``wavelength`` and the incidence ``angle``.
* Segment WIDTHS are static (frozen geometry): the union grid, the element
  boundaries and the Fourier projection are concrete NumPy constants -- the
  same restriction as the shipped single-layer Jones twin (the duty
  moving-mesh rebuild exists only on the scalar ``pmm_efficiency_1d`` path).
* A traced ``(3, 3)`` tensor cannot be inspected for out-of-plane coupling
  (concretizing would sever the trace), so its ``xz / yz / zx / zy`` entries
  are IGNORED -- exactly the single-layer twin's ``_t3`` contract.  Concrete
  tensors ARE inspected and raise on out-of-plane coupling.

The order set / far-field sizing mirrors the NumPy ``PMMStack.solve`` policy
(concrete numbers only -- a traced wavelength or permittivity falls back to
the ``far_field_orders`` floor with the documented Wood-anomaly caveat:
gradients are valid only BETWEEN order cutoffs).
"""
from __future__ import annotations

import numpy as np

from ._core import (
    _forward_growth_flip,
    _gll_nodes_weights,
    _jeps_is_passive,
    _jpmm_fourier_projection,
    _jpmm_order_set,
    _jpmm_sem_modes_tensor,
    _jtensor_is_passive,
    _kz_forward,
    _l2g_periodic,
    _lagrange_derivative_matrix,
    _mass_flux_threshold,
    _pmm_union_grid,
    _require_concrete_wavelength,
    _segment_elem_bnds,
)

__all__ = ["_pmm_stack_solve_jax"]


# ---------------------------------------------------------------------------
# frozen geometry: the stack's union grid as a multi-region SEM topology
# ---------------------------------------------------------------------------

def _jstack_static(period, uwidths, degree, n_el_per_region, grade):
    """Frozen multi-region SEM topology on the stack's UNION grid -- the
    multi-region analogue of :func:`_jpmm_build_static`.  ``cell[e]`` is the
    union-cell index of element ``e`` (the slot into ``layer_eps_union``).
    Built through the SAME :func:`_segment_elem_bnds` the NumPy assembler
    uses, so the reversed region layout, the grading and the lone-element
    midpoint split -- hence ``n_glob`` / ``l2g`` / the quadrature -- all
    match the NumPy stack exactly."""
    ref_nodes, ref_w = _gll_nodes_weights(degree)
    Dref = _lagrange_derivative_matrix(ref_nodes)
    bnds = _segment_elem_bnds(period, list(uwidths),
                              list(range(len(uwidths))), n_el_per_region,
                              grade)
    n_el = len(bnds)
    l2g, n_glob = _l2g_periodic(n_el, degree)
    Mloc = np.zeros((n_el, degree + 1))
    Kloc = np.zeros((n_el, degree + 1, degree + 1))
    Cloc = np.zeros((n_el, degree + 1, degree + 1))
    for e, (xl, xr, _c) in enumerate(bnds):
        J = 0.5 * (xr - xl)
        wel = ref_w * J
        Dphys = Dref / J
        Mloc[e] = wel
        Kloc[e] = (Dphys.T * wel) @ Dphys
        Cloc[e] = wel[:, None] * Dphys
    return dict(l2g=l2g, n_glob=n_glob, n_el=n_el, degree=degree,
                ref_nodes=ref_nodes, ref_w=ref_w, Dref=Dref,
                Mloc=Mloc, Kloc=Kloc, Cloc=Cloc,
                elem_bnds=[(xl, xr) for (xl, xr, _c) in bnds],
                cell=np.asarray([c for (_xl, _xr, c) in bnds], dtype=int))


# ---------------------------------------------------------------------------
# traced assembly: per-element union-cell tensors (eps-LINEAR throughout)
# ---------------------------------------------------------------------------

def _jstack_assemble(static, jnp, t_cells):
    """Traced multi-region tensor Galerkin assembly -- the body of
    :func:`_jpmm_assemble_tensor` with the binary ``t_of[region[e]]`` lookup
    replaced by the union-cell map ``t_cells[cell[e]]``.  Every tensor entry
    enters its mass LINEARLY (the only nonlinearity, the wall-normal
    ``[[1/exx]]^-1`` inverse rule, is applied later in the modal solve), so
    the eps gradients flow."""
    cj = jnp.complex128
    n_glob, n_el = static["n_glob"], static["n_el"]
    l2g, cell = static["l2g"], static["cell"]
    Mloc = jnp.asarray(static["Mloc"], cj)
    Kloc = jnp.asarray(static["Kloc"], cj)
    Cloc = jnp.asarray(static["Cloc"], cj)
    Z = jnp.zeros((n_glob, n_glob), cj)
    mass = {k: Z for k in ("one", "eyy", "exy_xx", "eyx_xx", "schur",
                           "inv_xx", "inv_ezz")}
    stiff = {k: Z for k in ("one", "inv_ezz")}
    conv = {k: Z for k in ("one", "inv_ezz")}
    for e in range(n_el):
        t = t_cells[int(cell[e])]
        exx, exy, eyx = t["exx"], t["exy"], t["eyx"]
        eyy, ezz = t["eyy"], t["ezz"]
        iez = 1.0 / ezz
        idx = l2g[e]
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        Ml = jnp.diag(Mloc[e])
        Kl = Kloc[e]
        Cl = Cloc[e]
        mass["one"] = mass["one"].at[ii, jj].add(Ml)
        mass["eyy"] = mass["eyy"].at[ii, jj].add(eyy * Ml)
        mass["exy_xx"] = mass["exy_xx"].at[ii, jj].add((exy / exx) * Ml)
        mass["eyx_xx"] = mass["eyx_xx"].at[ii, jj].add((eyx / exx) * Ml)
        mass["schur"] = mass["schur"].at[ii, jj].add(
            (eyy - eyx * exy / exx) * Ml)
        mass["inv_xx"] = mass["inv_xx"].at[ii, jj].add((1.0 / exx) * Ml)
        mass["inv_ezz"] = mass["inv_ezz"].at[ii, jj].add(iez * Ml)
        stiff["one"] = stiff["one"].at[ii, jj].add(Kl)
        stiff["inv_ezz"] = stiff["inv_ezz"].at[ii, jj].add(iez * Kl)
        conv["one"] = conv["one"].at[ii, jj].add(Cl)
        conv["inv_ezz"] = conv["inv_ezz"].at[ii, jj].add(iez * Cl)
    return dict(mass=mass, stiff=stiff, conv=conv, S0=mass["one"],
                n_glob=n_glob,
                # see :func:`_core._jpmm_assemble_tensor` -- the traced
                # assembler carries the passivity verdict of the tensors it
                # consumed, because it has no concrete element table.
                passive=_jtensor_is_passive(list(t_cells), jnp))


def _jstack_geo_ops(static):
    """GEOMETRY-ONLY global operators (weight-1 mass / stiffness / convection)
    assembled once in concrete NumPy -- the eps-free ingredients of the shared
    half-space eig (audit P3-27).  These are exactly the ``one``-keyed outputs
    of :func:`_jstack_assemble` (no eps enters them), so building them host-side
    skips the traced per-element scatter-add loop entirely."""
    n_glob, n_el = static["n_glob"], static["n_el"]
    l2g = static["l2g"]
    S0 = np.zeros((n_glob, n_glob), dtype=np.complex128)
    K = np.zeros_like(S0)
    Cw = np.zeros_like(S0)
    for e in range(n_el):
        ix = np.ix_(l2g[e], l2g[e])
        S0[ix] += np.diag(static["Mloc"][e])
        K[ix] += static["Kloc"][e]
        Cw[ix] += static["Cloc"][e]
    return S0, K, Cw


def _jstack_modes_uniform(S0, mu, w, jnp, eps):
    """Uniform-isotropic half-space modes from the SHARED geometric eig -- the
    traced twin of :func:`_core._sem_modes_uniform` (v5.14.2 backlog A2,
    ported per audit P3-27).  For a uniform ``eps`` the coupled operator
    collapses exactly to ``Mbig(eps) = eps I - blockdiag(Kx2, Kx2)``, so the
    eps-free ``(mu, w)`` eig of ``Kx2`` serves every half-space with the
    ANALYTIC spectrum shift ``q^2 = eps - mu`` (eps gradients flow through the
    shift; k0/kx0 gradients through the one shared eig).  Same return contract
    and z-Poynting forward selector as :func:`_jpmm_sem_modes_tensor`."""
    cj = jnp.complex128
    n = w.shape[0]
    q2 = eps - mu
    q = jnp.sqrt(jnp.concatenate([q2, q2]))
    Zn = jnp.zeros((n, n), cj)
    W2 = jnp.block([[w, Zn], [Zn, w]])
    # Q @ W2 for the uniform medium: Q = [[0, eps I - Kx2], [-eps I, 0]] and
    # Kx2 w = w diag(mu)  =>  (eps I - Kx2) w = w diag(q2)
    QW = jnp.block([[Zn, w * q2[None, :]], [-eps * w, Zn]])
    # --- the _jpmm_sem_modes_tensor forward selector, verbatim policy -------
    lam0 = -1j * q
    safe0 = jnp.where(jnp.abs(lam0) < 1e-12, 1e-12, lam0)
    V0 = QW * (1.0 / safe0)[None, :]
    SVt = S0 @ jnp.conj(V0[:n])             # S0 conj(Hx)
    SVb = S0 @ jnp.conj(V0[n:])             # S0 conj(Hy)
    flux = jnp.imag(jnp.einsum("in,in->n", W2[:n], SVb)
                    - jnp.einsum("in,in->n", W2[n:], SVt))
    thr = _mass_flux_threshold(flux, W2, SVt, SVb, n, jnp)
    prop = jnp.abs(flux) > thr                        # W7 B2: unit-safe cut
    flip = _forward_growth_flip(flux, q, thr, prop, jnp,
                                _jeps_is_passive(eps, jnp))
    q = jnp.where(flip, -q, q)
    lam = -1j * q
    safe = jnp.where(jnp.abs(lam) < 1e-12, 1e-12, lam)
    V2 = QW * (1.0 / safe)[None, :]
    return W2, V2, lam, q


# ---------------------------------------------------------------------------
# the traced multilayer solve
# ---------------------------------------------------------------------------

def _concrete_real(v):
    """``float(Re(v))`` when ``v`` is concrete, else ``None`` (a traced value
    has no concrete real part -- the caller falls back to a static policy)."""
    try:
        return float(np.real(np.asarray(v)))
    except Exception:
        return None


def _concrete_index(eps):
    """``Re(sqrt(eps))`` from a CONCRETE complex eps (the propagating-order
    count policy of the NumPy stack), else ``None`` for a traced eps."""
    try:
        return float(np.real(np.sqrt(complex(np.asarray(eps)))))
    except Exception:
        return None


def _grazing_guard_concrete(n_sup, angle):
    """Host-side mirror of the NumPy stack's grazing/evanescent-incidence
    raise (``_core._assemble_jones_farfield``), run ONLY when ``n_sup`` and
    ``angle`` are concrete (audit P2-10): the twin divides the flux
    normalization by ``kz_inc``, so at (near-)grazing incidence it returned
    NaN R/T where the NumPy stack raises.  Same tolerance (``|kz_inc| <
    1e-9``) and message as the NumPy guard.

    A TRACED ``n_sup`` / ``angle`` (a ``jax.grad`` / ``jax.jit`` Tracer has no
    concrete host value) SKIPS the guard -- concretizing would sever the
    trace (the ``_reject_jax_offplane`` tracer contract); the sibling
    ``_jax_stack2d`` degrades identically."""
    try:
        nsup_c = complex(np.asarray(n_sup))
        angle_c = float(np.real(np.asarray(angle)))
    except Exception:               # Tracer: no concrete value -> skip
        return
    kx0n = float(np.real(nsup_c)) * np.sin(angle_c)
    kz_inc = float(np.real(_kz_forward(nsup_c ** 2, np.array([kx0n]))[0]))
    if abs(kz_inc) < 1e-9:
        raise ValueError(
            "pmm: grazing/evanescent incidence (kz_inc ~ 0) -- the incident wave "
            "carries ~no z-flux, so the R/T flux normalization is ill-defined.  "
            "Reduce the incidence angle below grazing.")


def _pmm_stack_solve_jax(stack):
    """The JAX twin of the all-vertical in-plane branch of
    :meth:`PMMStack.solve`.  Consumes the stack's stored (possibly traced)
    layers / half-spaces / source and returns ``(orders, R(2,N), T(2,N),
    jones(2,2))`` as jnp arrays.  All gating (slant / OOP / stabilize /
    retain_internal / dispersive) happens in the dispatcher."""
    import jax.numpy as jnp

    from ..rcwa import _jax_eig_stable, _require_jax_x64
    _require_jax_x64("PMMStack.solve")
    cj = jnp.complex128

    wl = stack._src["wl"]
    angle = stack._src["angle"]
    # W7 F-E: the propagating-order SET is selected from the wavelength, so a
    # TRACED wl silently solved a smaller order set than the NumPy path (and
    # jit returned a different array length than un-jitted).
    _require_concrete_wavelength(
        wl, "PMMStack.solve",
        "PMMStack.solve_vs_wavelength (NumPy) or a loop over concrete "
        "wavelengths")
    # concrete-only grazing/evanescent-incidence guard (audit P2-10; a traced
    # n_sup / angle skips it -- see _grazing_guard_concrete)
    _grazing_guard_concrete(stack.n_sup, angle)
    n_sup = jnp.asarray(stack.n_sup, cj)
    n_sub = jnp.asarray(stack.n_sub, cj)
    eps_sup = n_sup ** 2
    eps_sub = n_sub ** 2
    wl_t = jnp.asarray(wl)
    k0 = 2.0 * jnp.pi / wl_t

    # ---- frozen geometry (concrete widths; eps objects pass through) ------
    uwidths, layer_eps_u = _pmm_union_grid(
        [L[1] for L in stack._layers], stack.min_feature / stack.period)
    static = _jstack_static(stack.period, uwidths, stack.degree, stack.n_el,
                            stack.grade)

    # ---- CONCRETE order set (mirrors the NumPy stack's n_max policy: max
    # over BOTH in-plane diagonal components of every cell + the half-space
    # indices; traced values drop out -> the far_field_orders floor) --------
    n_vals = []
    for eps_u in layer_eps_u:
        for e in eps_u:
            for comp in (e[0, 0], e[1, 1]):
                v = _concrete_index(comp)
                if v is not None:
                    n_vals.append(v)
    for nv in (_concrete_real(stack.n_sup), _concrete_real(stack.n_sub)):
        if nv is not None:
            n_vals.append(nv)
    n_max = max(n_vals) if n_vals else 1.0
    wl_c = _concrete_real(wl)
    if wl_c is None:
        wl_c = float("inf")
    orders = _jpmm_order_set(static, stack.period, wl_c, n_max, stack.ffo,
                             stack.degree, "PMMStack.solve")
    N = len(orders)
    Tp = jnp.asarray(_jpmm_fourier_projection(orders, stack.period, static),
                     cj)

    # ---- oblique Bloch wavenumber (traced when angle / n_sup / wl is) -----
    # Skip the convection ONLY for the python literal 0.0 (normal incidence,
    # byte-equal to the normal branch); a TRACED zero still differentiates.
    from ...backend import is_jax_array
    if is_jax_array(angle) or float(angle) != 0.0:
        kx0 = jnp.real(n_sup) * jnp.sin(jnp.asarray(angle)) * k0
    else:
        kx0 = 0.0

    # ---- half-space + layer modes (gauge-stable custom-VJP eig) -----------
    eig = _jax_eig_stable()

    def _t5(M):
        M = jnp.asarray(M, cj)
        if M.ndim == 0:
            return dict(exx=M, exy=0.0 * M, eyx=0.0 * M, eyy=M, ezz=M)
        return dict(exx=M[0, 0], exy=M[0, 1], eyx=M[1, 0], eyy=M[1, 1],
                    ezz=M[2, 2])

    # SHARED-EIG half-spaces (audit P3-27; the NumPy stack's v5.14.2 backlog-A2
    # optimization ported to the twin).  Both UNIFORM half-spaces reduce to
    # ``Mbig(eps) = eps I - blockdiag(Kx2, Kx2)`` with the eps-free geometric
    # ``Kx2 = S0^-1 (K - i kx0 (C - C^T) + kx0^2 S0) / k0^2``, so ONE n x n
    # custom-VJP eig (traced through k0 / kx0) replaces the two full traced
    # assemblies + two dense 2n x 2n eigs; the eps / n_sup / n_sub gradients
    # flow ANALYTICALLY through the spectrum shift q^2 = eps - mu.
    S0g, Kg, Cwg = _jstack_geo_ops(static)
    S0j = jnp.asarray(S0g, cj)
    op = jnp.asarray(Kg, cj)
    if not (isinstance(kx0, float) and kx0 == 0.0):
        Cwj = jnp.asarray(Cwg, cj)
        op = op - 1j * kx0 * (Cwj - Cwj.T) + (kx0 * kx0) * S0j
    Kx2 = (1.0 / (k0 * k0)) * (jnp.linalg.inv(S0j) @ op)
    mu_geo, w_geo = eig(Kx2)
    Wsup, Vsup, _ls, _qs = _jstack_modes_uniform(S0j, mu_geo, w_geo, jnp,
                                                 eps_sup)
    Wsub, Vsub, _lb, _qb = _jstack_modes_uniform(S0j, mu_geo, w_geo, jnp,
                                                 eps_sub)

    lmodes = []
    for i, (thk, _segs, _slant) in enumerate(stack._layers):
        mats = _jstack_assemble(static, jnp,
                                [_t5(e) for e in layer_eps_u[i]])
        Wl, Vl, lam, _q = _jpmm_sem_modes_tensor(mats, jnp, eig, k0, kx0)
        lmodes.append((Wl, Vl, lam, jnp.asarray(thk)))

    # ---- Redheffer cascade (verbatim the single-layer twin's blocks) ------
    def _ismat(Wa, Va, Wb, Vb):
        a = jnp.linalg.solve(Wb, Wa)
        b = jnp.linalg.solve(Vb, Va)
        apb, amb = a + b, a - b
        iapb = jnp.linalg.inv(apb)
        return (-iapb @ amb, 2.0 * iapb,
                0.5 * (apb - amb @ iapb @ amb), amb @ iapb)

    def _psmat(lam, k0_L):
        m = lam.shape[0]
        X = jnp.diag(jnp.exp(-lam * k0_L))
        Z = jnp.zeros((m, m), cj)
        return (Z, X, X, Z)

    def _star(SA, SB):
        A11, A12, A21, A22 = SA
        B11, B12, B21, B22 = SB
        m = A11.shape[0]
        eye = jnp.eye(m, dtype=cj)
        D = jnp.linalg.inv(eye - B11 @ A22)
        F = jnp.linalg.inv(eye - A22 @ B11)
        return (A11 + A12 @ D @ B11 @ A21, A12 @ D @ B12,
                B21 @ F @ A21, B22 + B21 @ F @ A22 @ B12)

    nlay = len(lmodes)
    S = _ismat(Wsup, Vsup, lmodes[0][0], lmodes[0][1])
    for i, (Wl, Vl, lam, thk) in enumerate(lmodes):
        S = _star(S, _psmat(lam, k0 * thk))
        if i == nlay - 1:
            S = _star(S, _ismat(Wl, Vl, Wsub, Vsub))
        else:
            S = _star(S, _ismat(Wl, Vl, lmodes[i + 1][0], lmodes[i + 1][1]))
    S11, _S12, S21, _S22 = S

    # ---- far field (verbatim the single-layer twin's tail; mirrors
    # _assemble_jones_farfield with the differentiable min-norm pinv) -------
    n_glob = static["n_glob"]
    Gv = 2.0 * np.pi / stack.period

    def _project(Wmodes):
        return jnp.vstack([Tp @ Wmodes[:n_glob, :], Tp @ Wmodes[n_glob:, :]])

    Hsup = _project(Wsup)
    Hsub = _project(Wsub)

    orders_j = jnp.asarray(orders)
    kx = (kx0 + orders_j * Gv) / k0

    def _kzf(eps, kxv):
        val = jnp.sqrt(jnp.asarray(eps - kxv ** 2, dtype=cj))
        return jnp.where(val.imag < 0.0, -val, val)

    kz_sup = _kzf(eps_sup, kx)
    kz_sub = _kzf(eps_sub, kx)
    kx0n = kx0 / k0
    kz_inc = jnp.real(_kzf(eps_sup, jnp.asarray(kx0n, cj)))
    safe_r = jnp.where(jnp.abs(kz_sup) < 1e-12, 1.0, kz_sup)
    safe_t = jnp.where(jnp.abs(kz_sub) < 1e-12, 1.0, kz_sub)

    # Differentiable minimum-norm least squares (jnp.linalg.lstsq's VJP NaNs
    # on the underdetermined stacked Hsup): x = A^H (A A^H)^-1 b.
    mrows, ncols = Hsup.shape
    if mrows <= ncols:
        AAH_inv = jnp.linalg.inv(Hsup @ Hsup.conj().T)
        pinv = Hsup.conj().T @ AAH_inv
    else:
        AHA_inv = jnp.linalg.inv(Hsup.conj().T @ Hsup)
        pinv = AHA_inv @ Hsup.conj().T

    m0 = int(np.where(orders == 0)[0][0])
    rows_R, rows_T, jcols = [], [], []
    for col in range(2):                    # 0 = incident Ex, 1 = incident Ey
        rhs = jnp.zeros(2 * N, cj).at[(col * N) + m0].set(1.0)
        cinc = pinv @ rhs
        r_ord = Hsup @ (S11 @ cinc)
        t_ord = Hsub @ (S21 @ cinc)
        rx, ry = r_ord[:N], r_ord[N:]
        tx, ty = t_ord[:N], t_ord[N:]
        rz = -(kx * rx) / safe_r
        tz = -(kx * tx) / safe_t
        if col == 0:
            flux_inc = kz_inc * (1.0 + (kx0n / kz_inc) ** 2)
        else:
            flux_inc = kz_inc
        Re = jnp.real(kz_sup) * (jnp.abs(rx) ** 2 + jnp.abs(ry) ** 2
                                 + jnp.abs(rz) ** 2) / flux_inc
        Te = jnp.real(kz_sub) * (jnp.abs(tx) ** 2 + jnp.abs(ty) ** 2
                                 + jnp.abs(tz) ** 2) / flux_inc
        # ``Re(kz) > 0`` cut-off mask, matching the NumPy twin (audit M7).
        rows_R.append(jnp.where(jnp.real(kz_sup) > 0.0, jnp.real(Re), 0.0))
        rows_T.append(jnp.where(jnp.real(kz_sub) > 0.0, jnp.real(Te), 0.0))
        jcols.append(jnp.stack([rx[m0], ry[m0]]))
    R_eff = jnp.stack(rows_R)
    T_eff = jnp.stack(rows_T)
    jones = jnp.stack(jcols, axis=1)        # (2,2): columns = incident Ex / Ey
    return jnp.asarray(orders), R_eff, T_eff, jones

def _pmm_stack_solve_jax_perlayer(stack):
    """The JAX twin of the PER-LAYER all-vertical in-plane branch (audit R-6).

    Geometry is CONCRETE and frozen exactly as in the NumPy per-layer path:
    the neighbour-window grids, the exact interface masses / cross-masses and
    the two half-space Rayleigh projectors are NumPy constants; only the
    eps-dependent operators, eigs, mortar solves, cascade and far field trace.
    Gradients flow to every traced eps / thickness / half-space index / angle;
    the wavelength must be concrete (W7 F-E, same as the shared twin)."""
    import jax.numpy as jnp

    from ...backend import is_jax_array
    from ..rcwa import _jax_eig_stable, _require_jax_x64
    from ._core import (
        PMM_MORTAR_SEPARABLE,
        _build_sem_tensor_segments,
        _kron2_apply,
        _perlayer_window_grids,
        _sem_cross_mass_cached,
        _sem_fourier_projection,
        _sem_mass_exact_cached,
        _tensor3_dict,
    )
    _require_jax_x64("PMMStack.solve")
    cj = jnp.complex128

    wl = stack._src["wl"]
    angle = stack._src["angle"]
    _require_concrete_wavelength(
        wl, "PMMStack.solve",
        "PMMStack.solve_vs_wavelength (NumPy) or a loop over concrete "
        "wavelengths")
    _grazing_guard_concrete(stack.n_sup, angle)
    n_sup = jnp.asarray(stack.n_sup, cj)
    n_sub = jnp.asarray(stack.n_sub, cj)
    eps_sup = n_sup ** 2
    eps_sub = n_sub ** 2
    wl_t = jnp.asarray(wl)
    k0 = 2.0 * jnp.pi / wl_t

    # ---- frozen PER-LAYER geometry (concrete widths; eps pass through) ----
    nlay = len(stack._layers)
    grid_of = _perlayer_window_grids(
        [L[1] for L in stack._layers], stack.min_feature / stack.period,
        stack.window_halfwidth)
    wkeys = [w.tobytes() for w, _r in grid_of]
    statics = [_jstack_static(stack.period, w_i, stack.degree, stack.n_el,
                              stack.grade) for w_i, _r in grid_of]
    # geometry-only NumPy mats per grid (masses / cross-masses / projectors
    # ignore eps entirely)
    _eye3 = np.eye(3)
    geo_mats = [_build_sem_tensor_segments(
        stack.period, w_i, [_tensor3_dict(_eye3)] * len(w_i),
        stack.degree, stack.n_el, stack.grade) for w_i, _r in grid_of]
    # M3 / N-3 items 1 + 3: the 1-D blocks are stored (and CACHED across
    # traces by ``_sem_*_cached``) and applied SEPARABLY -- the twin no longer
    # materialises four ``kron(I_2, .)`` operators per mortar, which were 4x
    # the bytes and 2x the flops of the blockwise form (see the exactness
    # argument above ``_kron2_apply`` in ``_core.py``).
    #
    # The fail-before switch reaches HERE TOO.  It is read once at TRACE time
    # (a Python bool, never a traced value), so the traced graph is one arm or
    # the other and nothing data-dependent is branched on.  Measured: without
    # this gate the switch was NumPy-only and the JAX twin moved 5.6e-17 with
    # items 3+4 nominally OFF -- a fail-before that does not fail before is
    # worse than none.
    _sep = bool(PMM_MORTAR_SEPARABLE)
    _eye2 = np.eye(2)
    Ms = {}
    for i in range(nlay):
        if wkeys[i] not in Ms:
            M = _sem_mass_exact_cached(geo_mats[i])
            Ms[wkeys[i]] = M if _sep else np.kron(_eye2, M)
    crosses = {}
    for i in range(nlay - 1):
        if wkeys[i] != wkeys[i + 1]:
            C = _sem_cross_mass_cached(geo_mats[i], geo_mats[i + 1])
            crosses[i] = ((C.T, C) if _sep
                          else (np.kron(_eye2, C.T), np.kron(_eye2, C)))

    # ---- CONCRETE order set (min over the two end grids' capacity) --------
    n_vals = []
    for _w, row in grid_of:
        for e in row:
            # tracer-safe: use .ndim (attribute access) and never np.asarray
            # a possibly-traced entry (TracerArrayConversionError)
            comps = ((e[0, 0], e[1, 1])
                     if getattr(e, "ndim", 0) == 2 else (e, e))
            for comp in comps:
                v = _concrete_index(comp)
                if v is not None:
                    n_vals.append(v)
    for nv in (_concrete_real(stack.n_sup), _concrete_real(stack.n_sub)):
        if nv is not None:
            n_vals.append(nv)
    n_max = max(n_vals) if n_vals else 1.0
    wl_c = _concrete_real(wl)
    if wl_c is None:
        wl_c = float("inf")
    o0 = _jpmm_order_set(statics[0], stack.period, wl_c, n_max, stack.ffo,
                         stack.degree, "PMMStack.solve")
    oN = _jpmm_order_set(statics[-1], stack.period, wl_c, n_max, stack.ffo,
                         stack.degree, "PMMStack.solve")
    orders = o0 if len(o0) <= len(oN) else oN
    N = len(orders)
    Tp_sup = jnp.asarray(
        _sem_fourier_projection(np.asarray(orders), stack.period,
                                geo_mats[0]), cj)
    Tp_sub = jnp.asarray(
        _sem_fourier_projection(np.asarray(orders), stack.period,
                                geo_mats[-1]), cj)

    if is_jax_array(angle) or float(angle) != 0.0:
        kx0 = jnp.real(n_sup) * jnp.sin(jnp.asarray(angle)) * k0
    else:
        kx0 = 0.0

    eig = _jax_eig_stable()

    def _t5(M):
        M = jnp.asarray(M, cj)
        if M.ndim == 0:
            return dict(exx=M, exy=0.0 * M, eyx=0.0 * M, eyy=M, ezz=M)
        return dict(exx=M[0, 0], exy=M[0, 1], eyx=M[1, 0], eyy=M[1, 1],
                    ezz=M[2, 2])

    def _uniform_modes(static_x, eps_x):
        S0g, Kg, Cwg = _jstack_geo_ops(static_x)
        S0j = jnp.asarray(S0g, cj)
        op = jnp.asarray(Kg, cj)
        if not (isinstance(kx0, float) and kx0 == 0.0):
            Cwj = jnp.asarray(Cwg, cj)
            op = op - 1j * kx0 * (Cwj - Cwj.T) + (kx0 * kx0) * S0j
        Kx2 = (1.0 / (k0 * k0)) * (jnp.linalg.inv(S0j) @ op)
        mu_geo, w_geo = eig(Kx2)
        return _jstack_modes_uniform(S0j, mu_geo, w_geo, jnp, eps_x)

    Wsup, Vsup, _ls, _qs = _uniform_modes(statics[0], eps_sup)
    Wsub, Vsub, _lb, _qb = _uniform_modes(statics[-1], eps_sub)

    lmodes = []
    for i, (thk, _segs, _slant) in enumerate(stack._layers):
        mats = _jstack_assemble(statics[i], jnp,
                                [_t5(e) for e in grid_of[i][1]])
        Wl, Vl, lam, _q = _jpmm_sem_modes_tensor(mats, jnp, eig, k0, kx0)
        lmodes.append((Wl, Vl, lam, jnp.asarray(thk)))

    # ---- cascade: plain interface on conforming pairs, mortar otherwise ---
    def _ismat(Wa, Va, Wb, Vb):
        a = jnp.linalg.solve(Wb, Wa)
        b = jnp.linalg.solve(Vb, Va)
        apb, amb = a + b, a - b
        iapb = jnp.linalg.inv(apb)
        return (-iapb @ amb, 2.0 * iapb,
                0.5 * (apb - amb @ iapb @ amb), amb @ iapb)

    def _imort(Wa, Va, Wb, Vb, i):
        Cba, Cab = crosses[i]
        Ma = jnp.asarray(Ms[wkeys[i]], cj)
        Mb = jnp.asarray(Ms[wkeys[i + 1]], cj)
        if _sep:
            A = jnp.linalg.solve(_kron2_apply(Mb, Wb, jnp),
                                 _kron2_apply(jnp.asarray(Cba, cj), Wa, jnp))
            B = jnp.linalg.solve(_kron2_apply(Ma, Va, jnp),
                                 _kron2_apply(jnp.asarray(Cab, cj), Vb, jnp))
            BA = B @ A
            eye_a = jnp.eye(BA.shape[0], dtype=cj)
            # one factorisation, both right-hand sides (the NumPy twin's shape)
            X = jnp.linalg.solve(eye_a + BA,
                                 jnp.concatenate((eye_a - BA, B), axis=1))
            nc = eye_a.shape[1]
            S11 = X[:, :nc]
            S12 = 2.0 * X[:, nc:]
        else:                       # fail-before: the pre-M3 traced arithmetic
            A = jnp.linalg.solve(Mb @ Wb, jnp.asarray(Cba, cj) @ Wa)
            B = jnp.linalg.solve(Ma @ Va, jnp.asarray(Cab, cj) @ Vb)
            BA = B @ A
            eye_a = jnp.eye(BA.shape[0], dtype=cj)
            iba = jnp.linalg.inv(eye_a + BA)
            S11 = iba @ (eye_a - BA)
            S12 = 2.0 * (iba @ B)
        S21 = A @ (eye_a + S11)
        S22 = A @ S12 - jnp.eye(A.shape[0], dtype=cj)
        return (S11, S12, S21, S22)

    def _psmat(lam, k0_L):
        m = lam.shape[0]
        X = jnp.diag(jnp.exp(-lam * k0_L))
        Z = jnp.zeros((m, m), cj)
        return (Z, X, X, Z)

    def _star_rect(SA, SB):
        # The inverses stay inverses, mirroring the NumPy twin -- see
        # ``_redheffer_star_rect``'s docstring for the identity pin that
        # refused the solve.  The common subexpressions are named once (the
        # bit-identical half of M3 / N-3 item 4).
        A11, A12, A21, A22 = SA
        B11, B12, B21, B22 = SB
        m = A22.shape[0]
        eye = jnp.eye(m, dtype=cj)
        AD = A12 @ jnp.linalg.inv(eye - B11 @ A22)
        BF = B21 @ jnp.linalg.inv(eye - A22 @ B11)
        return (A11 + AD @ B11 @ A21, AD @ B12,
                BF @ A21, B22 + BF @ A22 @ B12)

    def _ifc(Wa, Va, Wb, Vb, i):
        # i = the LOWER layer index of an interior interface; None = half-space
        if i is None or wkeys[i] == wkeys[i + 1]:
            return _ismat(Wa, Va, Wb, Vb)
        return _imort(Wa, Va, Wb, Vb, i)

    S = _ismat(Wsup, Vsup, lmodes[0][0], lmodes[0][1])
    for i, (Wl, Vl, lam, thk) in enumerate(lmodes):
        S = _star_rect(S, _psmat(lam, k0 * thk))
        if i == nlay - 1:
            S = _star_rect(S, _ismat(Wl, Vl, Wsub, Vsub))
        else:
            S = _star_rect(S, _ifc(Wl, Vl, lmodes[i + 1][0],
                                   lmodes[i + 1][1], i))
    S11, _S12, S21, _S22 = S

    # ---- far field (per-grid projectors; otherwise the shared twin tail) --
    n0g = statics[0]["n_glob"]
    nNg = statics[-1]["n_glob"]
    Gv = 2.0 * np.pi / stack.period
    Hsup = jnp.vstack([Tp_sup @ Wsup[:n0g, :], Tp_sup @ Wsup[n0g:, :]])
    Hsub = jnp.vstack([Tp_sub @ Wsub[:nNg, :], Tp_sub @ Wsub[nNg:, :]])
    orders_j = jnp.asarray(orders)
    kx = (kx0 + orders_j * Gv) / k0

    def _kzf(eps, kxv):
        val = jnp.sqrt(jnp.asarray(eps - kxv ** 2, dtype=cj))
        return jnp.where(val.imag < 0.0, -val, val)

    kz_sup = _kzf(eps_sup, kx)
    kz_sub = _kzf(eps_sub, kx)
    kx0n = kx0 / k0
    kz_inc = jnp.real(_kzf(eps_sup, jnp.asarray(kx0n, cj)))
    safe_r = jnp.where(jnp.abs(kz_sup) < 1e-12, 1.0, kz_sup)
    safe_t = jnp.where(jnp.abs(kz_sub) < 1e-12, 1.0, kz_sub)
    mrows, ncols = Hsup.shape
    if mrows <= ncols:
        AAH_inv = jnp.linalg.inv(Hsup @ Hsup.conj().T)
        pinv = Hsup.conj().T @ AAH_inv
    else:
        AHA_inv = jnp.linalg.inv(Hsup.conj().T @ Hsup)
        pinv = AHA_inv @ Hsup.conj().T
    m0 = int(np.where(np.asarray(orders) == 0)[0][0])
    rows_R, rows_T, jcols = [], [], []
    for col in range(2):
        rhs = jnp.zeros(2 * N, cj).at[(col * N) + m0].set(1.0)
        cinc = pinv @ rhs
        r_ord = Hsup @ (S11 @ cinc)
        t_ord = Hsub @ (S21 @ cinc)
        rx, ry = r_ord[:N], r_ord[N:]
        tx, ty = t_ord[:N], t_ord[N:]
        rz = -(kx * rx) / safe_r
        tz = -(kx * tx) / safe_t
        if col == 0:
            flux_inc = kz_inc * (1.0 + (kx0n / kz_inc) ** 2)
        else:
            flux_inc = kz_inc
        Re = jnp.real(kz_sup) * (jnp.abs(rx) ** 2 + jnp.abs(ry) ** 2
                                 + jnp.abs(rz) ** 2) / flux_inc
        Te = jnp.real(kz_sub) * (jnp.abs(tx) ** 2 + jnp.abs(ty) ** 2
                                 + jnp.abs(tz) ** 2) / flux_inc
        rows_R.append(jnp.where(jnp.real(kz_sup) > 0.0, jnp.real(Re), 0.0))
        rows_T.append(jnp.where(jnp.real(kz_sub) > 0.0, jnp.real(Te), 0.0))
        jcols.append(jnp.stack([rx[m0], ry[m0]]))
    R_eff = jnp.stack(rows_R)
    T_eff = jnp.stack(rows_T)
    jones = jnp.stack(jcols, axis=1)
    return jnp.asarray(orders), R_eff, T_eff, jones

