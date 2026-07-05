"""Differentiable (JAX) twin of the Berreman 4x4 planar multilayer solve.

Mirrors :func:`lumenairy.elements.berreman.berreman_jones_1d` /
:meth:`BerremanStack.solve` on JAX inputs: gradients flow through every layer
permittivity tensor (real AND imaginary entries), every layer thickness, the
half-space indices, the wavelength and the incidence ``angle`` / ``phi``.  Layer
ORDER and count are static (as for any stack), but every numeric value is
traced.

The one trace-hazard a 4x4 modal method has -- the forward/backward mode split
-- is done with a STABLE ``jnp.argsort`` on a forwardness key (the gathered
eigen-values/-vectors carry the gradient; the integer permutation is constant),
so it traces cleanly under ``grad`` / ``jit`` / ``vmap``.  This is what lets the
Berreman twin succeed where the PMM out-of-plane path could not: that path's
host-side (NumPy) argsort severed the graph; a jnp argsort does not.

Built on the SAME generalized scattering-matrix cascade as the NumPy path
(``_interface_smatrix_general`` etc. are backend-generic) and the rcwa
gauge-stable custom-VJP eig.  PUBLIC convention throughout (raw eps).  x64
required; ``jnp.linalg.eig`` is CPU-only.
"""
from __future__ import annotations

import numpy as np

__all__ = ["_berreman_jones_1d_jax", "_berreman_stack_solve_jax"]

_C = np.complex128


def _delta_jax(eps, Kx, Ky, jnp):
    """Berreman 4x4 ``Delta`` built functionally in jnp (no in-place)."""
    cj = jnp.complex128
    exx, exy, exz = eps[0, 0], eps[0, 1], eps[0, 2]
    eyx, eyy, eyz = eps[1, 0], eps[1, 1], eps[1, 2]
    ezx, ezy, ezz = eps[2, 0], eps[2, 1], eps[2, 2]
    Kx = jnp.asarray(Kx, cj)
    Ky = jnp.asarray(Ky, cj)
    d = 1.0 / ezz
    Z = jnp.asarray(0.0, cj)
    rows = [
        [-Kx * ezx * d, -Kx * ezy * d, Kx * Ky * d, 1.0 - Kx * Kx * d],
        [-Ky * ezx * d, -Ky * ezy * d, Ky * Ky * d - 1.0, -Ky * Kx * d],
        [eyx - eyz * ezx * d + Kx * Ky, eyy - eyz * ezy * d - Kx * Kx,
         eyz * Ky * d, -eyz * Kx * d],
        [exz * ezx * d - exx + Ky * Ky, exz * ezy * d - exy - Kx * Ky,
         -exz * Ky * d, exz * Kx * d],
    ]
    return jnp.stack([jnp.stack([jnp.asarray(c, cj) + Z for c in r])
                      for r in rows])


def _layer_modes_jax(eps_tensor, Kx, Ky, jnp, eig):
    """Forward / backward modal blocks (jnp).  Returns ``(Wf, Vf, lamf, Wb,
    Vb, lamb)`` with the trace-safe argsort partition (forward = decaying /
    ``Im>0`` propagating; ``lam = -gam``)."""
    D = _delta_jax(eps_tensor, Kx, Ky, jnp)
    gam, Psi = eig(D)
    re, im = jnp.real(gam), jnp.imag(gam)
    tol = 1e-9 * jnp.maximum(jnp.max(jnp.abs(gam)), 1.0)
    is_fwd = jnp.where(re < -tol, True, jnp.where(re > tol, False, im > 0.0))
    order = jnp.argsort(jnp.where(is_fwd, 0, 1), stable=True)
    fwd, bwd = order[:2], order[2:]
    Wf = Psi[:2][:, fwd]
    Vf = Psi[2:][:, fwd]
    Wb = Psi[:2][:, bwd]
    Vb = Psi[2:][:, bwd]
    lamf = -gam[fwd]
    lamb = -gam[bwd]
    return Wf, Vf, lamf, Wb, Vb, lamb


def _flux_jax(E, Hmodal, jnp):
    Hx, Hy = -1j * Hmodal[0], -1j * Hmodal[1]
    return jnp.real(E[0] * jnp.conj(Hy) - E[1] * jnp.conj(Hx))


def _solve_jax(eps_layers, thicks, eps_sup, eps_sub, wavelength, Kx, Ky, jnp):
    from .rcwa import _jax_eig_stable
    from .rcwa._core import (
        _interface_smatrix_general,
        _modes_to_M,
        _propagation_smatrix_general,
        _redheffer_star,
    )
    cj = jnp.complex128
    eig = _jax_eig_stable()
    k0 = 2.0 * jnp.pi / wavelength
    I3 = jnp.eye(3, dtype=cj)

    Wf_s, Vf_s, lf_s, Wb_s, Vb_s, lb_s = _layer_modes_jax(
        eps_sup * I3, Kx, Ky, jnp, eig)
    Wf_b, Vf_b, lf_b, Wb_b, Vb_b, lb_b = _layer_modes_jax(
        eps_sub * I3, Kx, Ky, jnp, eig)
    M_sup = _modes_to_M(Wf_s, Vf_s, Wb_s, Vb_s)
    M_sub = _modes_to_M(Wf_b, Vf_b, Wb_b, Vb_b)

    modes = [_layer_modes_jax(e, Kx, Ky, jnp, eig) for e in eps_layers]
    Ms = [_modes_to_M(m[0], m[1], m[3], m[4]) for m in modes]
    nlay = len(modes)
    if nlay == 0:
        S = _interface_smatrix_general(M_sup, M_sub)
    else:
        S = _interface_smatrix_general(M_sup, Ms[0])
        for i in range(nlay):
            S = _redheffer_star(S, _propagation_smatrix_general(
                modes[i][2], modes[i][5], k0 * thicks[i]))
            nxt = M_sub if i == nlay - 1 else Ms[i + 1]
            S = _redheffer_star(S, _interface_smatrix_general(Ms[i], nxt))
    S11, _S12, S21, _S22 = S

    Jr_cols, Jt_cols, Rs, Ts = [], [], [], []
    for Einc in (jnp.array([1.0, 0.0], cj), jnp.array([0.0, 1.0], cj)):
        c_inc = jnp.linalg.solve(Wf_s, Einc)
        c_ref = S11 @ c_inc
        c_trn = S21 @ c_inc
        E_inc, H_inc = Wf_s @ c_inc, Vf_s @ c_inc
        E_ref, H_ref = Wb_s @ c_ref, Vb_s @ c_ref
        E_trn, H_trn = Wf_b @ c_trn, Vf_b @ c_trn
        Jr_cols.append(E_ref)
        Jt_cols.append(E_trn)
        F_inc = _flux_jax(E_inc, H_inc, jnp)
        Rs.append(-_flux_jax(E_ref, H_ref, jnp) / F_inc)
        Ts.append(_flux_jax(E_trn, H_trn, jnp) / F_inc)
    Jr = jnp.stack(Jr_cols, axis=1)
    Jt = jnp.stack(Jt_cols, axis=1)
    R = jnp.stack(Rs)
    T = jnp.stack(Ts)
    return R, T, Jr, Jt


def _tensor_is_offplane_jax(e, jnp):
    """True if a CONCRETE ``(3, 3)`` eps has out-of-plane coupling.  A concrete
    ``jax.Array`` (e.g. ``jnp.asarray`` of a fixed tensor) materialises fine; a
    TRACER (the eps itself being differentiated, under jit) raises on the
    ``np.asarray`` and returns False -- routing stays on the native path only
    when the tensor cannot be inspected."""
    try:
        M = np.asarray(e, dtype=_C)
    except Exception:
        return False
    if M.ndim != 2 or M.shape != (3, 3):
        return False
    off = max(abs(M[0, 2]), abs(M[1, 2]), abs(M[2, 0]), abs(M[2, 1]))
    diag = max(abs(M[0, 0]), abs(M[1, 1]), abs(M[2, 2]), 1.0)
    return bool(off > 1e-12 * diag)


def _layer_M_gen_jax(e, kx0, ky0, jnp, eig):
    """Generalized (Li 2003) layer block ``(M, lam_f, lam_b)`` in jnp -- the
    out-of-plane-correct convention (flux-select forward + ``lam = +gam`` +
    ``[W; -V]`` symmetric isotropic layers), mirroring the NumPy
    ``berreman._offplane_condensed_M``.  Scalar / isotropic -> analytic
    homogeneous modes; tensor -> the Berreman ``Delta`` eigenmodes (identical to
    the ezz-Schur-condensed RCWA generator) with the trace-safe ``jnp.argsort``
    flux split.  ``kx0 / ky0`` scalar."""
    from .rcwa._core import (
        _homogeneous_eigenmodes,
        _modes_to_M,
        _sqrt_decay,
    )
    cj = jnp.complex128
    Kx1 = jnp.reshape(jnp.asarray(kx0, cj), (1, 1))
    Ky1 = jnp.reshape(jnp.asarray(ky0, cj), (1, 1))
    e = jnp.asarray(e, cj)
    is_iso = jnp.max(jnp.abs(e - e[0, 0] * jnp.eye(3, dtype=cj))) < 1e-12
    if bool(is_iso):
        W, V, kz = _homogeneous_eigenmodes(Kx1, Ky1, e[0, 0])
        kzc = kz.ravel()
        lam = _sqrt_decay(-jnp.concatenate([kzc, kzc]) ** 2)
        return _modes_to_M(W, V, W, -V), lam, -lam
    D = _delta_jax(e, kx0, ky0, jnp)
    gam, Psi = eig(D)
    E0, E1 = Psi[0], Psi[1]
    Hx, Hy = -1j * Psi[2], -1j * Psi[3]
    Sz = jnp.real(E0 * jnp.conj(Hy) - E1 * jnp.conj(Hx))
    gre = jnp.real(gam)
    mx = jnp.maximum(jnp.max(jnp.abs(Sz)), 1.0)
    carries = (jnp.abs(Sz) > 1e-9 * mx) & (jnp.abs(gre) <= 0.5)
    is_fwd = jnp.where(carries, Sz > 0.0, gre > 0.0)
    order = jnp.argsort(jnp.where(is_fwd, 0, 1), stable=True)
    fwd, bwd = order[:2], order[2:]
    M = _modes_to_M(Psi[:2][:, fwd], Psi[2:][:, fwd],
                    Psi[:2][:, bwd], Psi[2:][:, bwd])
    return M, gam[fwd], gam[bwd]


def _offplane_solve_jax(eps_layers, thicks, eps_sup, eps_sub, wl, kx0, ky0, jnp):
    """Differentiable generalized S-matrix solve for a stack containing an
    out-of-plane tensor layer (any incidence).  Returns ``(R, T, Jr, Jt)`` in the
    PUBLIC convention.  Mirrors ``berreman._offplane_oblique_solve``."""
    from .rcwa import _jax_eig_stable
    from .rcwa._core import (
        _forward_flux_kz,
        _homogeneous_eigenmodes,
        _interface_smatrix_general,
        _modes_to_M,
        _propagation_star_general,
        _redheffer_star,
        _sqrt_forward,
    )
    cj = jnp.complex128
    eig = _jax_eig_stable()
    k0 = 2.0 * jnp.pi / wl
    Kx1 = jnp.reshape(jnp.asarray(kx0, cj), (1, 1))
    Ky1 = jnp.reshape(jnp.asarray(ky0, cj), (1, 1))
    Wr, Vr, _kzr = _homogeneous_eigenmodes(Kx1, Ky1, jnp.conj(eps_sup))
    Wt, Vt, _kzt = _homogeneous_eigenmodes(Kx1, Ky1, jnp.conj(eps_sub))
    M_prev = _modes_to_M(Wr, Vr, Wr, -Vr)
    S = None
    for e, thk in zip(eps_layers, thicks):
        Ml, lf, lb = _layer_M_gen_jax(jnp.conj(e), kx0, ky0, jnp, eig)
        Si = _interface_smatrix_general(M_prev, Ml)
        S = Si if S is None else _redheffer_star(S, Si)
        S = _propagation_star_general(S, lf, lb, k0 * thk)
        M_prev = Ml
    S = _redheffer_star(
        S, _interface_smatrix_general(M_prev, _modes_to_M(Wt, Vt, Wt, -Vt)))
    S11, _S12, S21, _S22 = S
    kz_inc = jnp.real(_sqrt_forward(eps_sup - kx0 ** 2 - ky0 ** 2))
    kzrf = _forward_flux_kz(eps_sup, jnp.reshape(kx0, (1,)),
                            jnp.reshape(ky0, (1,)))[0]
    kztf = _forward_flux_kz(eps_sub, jnp.reshape(kx0, (1,)),
                            jnp.reshape(ky0, (1,)))[0]
    Jr_cols, Jt_cols, Rs, Ts = [], [], [], []
    for ex0, ey0 in ((1.0, 0.0), (0.0, 1.0)):
        r = S11 @ jnp.array([ex0, ey0], cj)
        t = S21 @ jnp.array([ex0, ey0], cj)
        rx, ry, tx, ty = r[0], r[1], t[0], t[1]
        longi = kx0 * ex0 + ky0 * ey0
        einc_sq = 1.0 + (longi / kz_inc) ** 2
        rz = -(kx0 * rx + ky0 * ry) / kzrf
        tz = -(kx0 * tx + ky0 * ty) / kztf
        Rs.append(jnp.real(kzrf / kz_inc)
                  * (jnp.abs(rx) ** 2 + jnp.abs(ry) ** 2 + jnp.abs(rz) ** 2)
                  / einc_sq)
        Ts.append(jnp.real(kztf / kz_inc)
                  * (jnp.abs(tx) ** 2 + jnp.abs(ty) ** 2 + jnp.abs(tz) ** 2)
                  / einc_sq)
        Jr_cols.append(jnp.stack([jnp.conj(rx), jnp.conj(ry)]))
        Jt_cols.append(jnp.stack([jnp.conj(tx), jnp.conj(ty)]))
    return (jnp.stack(Rs), jnp.stack(Ts),
            jnp.stack(Jr_cols, axis=1), jnp.stack(Jt_cols, axis=1))


def _prep(layers, n_substrate, n_superstrate, wavelength, angle, phi, theta):
    import jax.numpy as jnp

    from .rcwa import _require_jax_x64
    _require_jax_x64("berreman_jones_1d")
    cj = jnp.complex128
    if theta is not None:
        angle = theta

    def _t3(e):
        M = jnp.asarray(e, cj)
        if M.ndim == 0:
            return M * jnp.eye(3, dtype=cj)
        return M
    eps_layers = [_t3(e) for e, _t in layers]
    thicks = [jnp.asarray(t) for _e, t in layers]
    eps_sup = jnp.asarray(n_superstrate, cj) ** 2
    eps_sub = jnp.asarray(n_substrate, cj) ** 2
    nre = jnp.real(jnp.asarray(n_superstrate, cj))
    Kx = nre * jnp.sin(jnp.asarray(angle)) * jnp.cos(jnp.asarray(phi))
    Ky = nre * jnp.sin(jnp.asarray(angle)) * jnp.sin(jnp.asarray(phi))
    return jnp, eps_layers, thicks, eps_sup, eps_sub, wavelength, Kx, Ky


def _berreman_jones_1d_jax(layers, n_substrate, n_superstrate, wavelength,
                           *, angle=0.0, phi=0.0, theta=None):
    jnp, eps_layers, thicks, eps_sup, eps_sub, wl, Kx, Ky = _prep(
        layers, n_substrate, n_superstrate, wavelength, angle, phi, theta)
    # A CONCRETELY-detectable out-of-plane tensor routes to the generalized
    # (out-of-plane-correct) S-matrix -- the native cascade below is ~2% off in
    # that regime (see berreman._offplane_oblique_solve).  The generalized path
    # is correct at ALL incidences, so no obliqueness test is needed (which also
    # sidesteps a traced angle).  Traced-eps / non-off-plane layers stay on the
    # native path (unchanged); a traced-eps out-of-plane stack cannot be detected
    # and falls through -- document via the NumPy entry's guard removal note.
    if any(_tensor_is_offplane_jax(e, jnp) for e, _t in layers):
        return _offplane_solve_jax(eps_layers, thicks, eps_sup, eps_sub,
                                   jnp.asarray(wl), Kx, Ky, jnp)
    R, T, Jr, Jt = _solve_jax(eps_layers, thicks, eps_sup, eps_sub,
                              jnp.asarray(wl), Kx, Ky, jnp)
    return R, T, Jr, Jt


def _berreman_stack_solve_jax(stack):
    src = stack._src
    layers = [(e, t) for t, e in stack._layers]
    R, T, Jr, Jt = _berreman_jones_1d_jax(
        layers, stack.n_sub, stack.n_sup, src["wl"],
        angle=src["angle"], phi=src["phi"])
    return R, T, Jr
