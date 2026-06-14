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
