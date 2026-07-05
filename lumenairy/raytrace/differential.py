"""Differential ray transfer -- per-beamlet / per-ray ABCD Jacobians.

The reusable primitive underneath the *per-surface* forms of the phase-space
propagators.  Along each base ray of a bundle it returns the ray-transfer
Jacobian ``[[A, B], [C, D]]`` (2x2 blocks) of the real (aberrated) trace, by
central finite differences.  The complex-beam-parameter propagators consume it:

* Gaussian Beam Decomposition (``propagators.gbd``): evolves each beamlet's
  tensor ``Q`` via the generalized Collins relation
  ``Q_out = (C + D Q)(A + B Q)^{-1}`` and its amplitude via
  ``1/sqrt(det(A + B Q))`` -- capturing off-axis astigmatism / higher-order
  aberrations that a single whole-system paraxial ABCD cannot.
* (future) Maslov phase-space: the same differential transfer supplies the
  Hessian propagation.

Phase space is **unreduced** ``(x, y, ux, uy)`` with slopes ``ux = L/N``,
``uy = M/N`` (matching ``propagators.gbd.apply_abcd_to_beamlets``), referenced
to surface **vertex** planes.  The finite-difference Jacobian bakes Snell's law
and the glass indices in automatically (no reduced ``n*u`` bookkeeping), so the
on-axis 2x2 meridional block reproduces
``raytrace.system_abcd_prescription`` to ~1e-8.

Validated in ``tests/unit/test_gbd_feature_complete.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .trace import _make_bundle, trace


@dataclass
class DifferentialTransfer:
    """Result of :func:`ray_transfer_jacobian`.

    Attributes
    ----------
    jacobian : ndarray
        ``(N_rays, 4, 4)`` composite input->output ray-transfer Jacobian, OR
        ``(N_surfaces, N_rays, 4, 4)`` per-surface local transfers when
        ``per_surface=True``.  Row/col order is ``(x, y, ux, uy)``; the 2x2
        blocks are ``A = J[..., 0:2, 0:2]`` (dx_out/dx_in),
        ``B = J[..., 0:2, 2:4]``, ``C = J[..., 2:4, 0:2]``,
        ``D = J[..., 2:4, 2:4]``.
    x, y, ux, uy : ndarray
        ``(N_rays,)`` base-ray state at the output vertex (unreduced slopes).
    opd : ndarray
        ``(N_rays,)`` base-ray accumulated optical path length [m] at output.
    alive : ndarray of bool
        ``(N_rays,)`` False for rays that vignetted / TIR'd / missed.
    """
    jacobian: np.ndarray
    x: np.ndarray
    y: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    opd: np.ndarray
    alive: np.ndarray


def _slopes_to_dirs(ux, uy):
    inv = 1.0 / np.sqrt(1.0 + ux * ux + uy * uy)
    return ux * inv, uy * inv, inv


def ray_transfer_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    surfaces: List,
    wavelength: float,
    *,
    per_surface: bool = False,
    h_pos: float = 1e-6,
    h_slope: float = 5e-5,
) -> DifferentialTransfer:
    """Differential ray-transfer Jacobian along each base ray.

    Parameters
    ----------
    x, y : ndarray ``(N,)``
        Base-ray transverse positions at the input vertex [m].
    ux, uy : ndarray ``(N,)``
        Base-ray paraxial slopes ``L/N``, ``M/N`` at the input.
    surfaces : list of Surface
        As from :func:`raytrace.surfaces_from_prescription`.  Each surface's
        ``thickness`` is the axial distance to the next vertex.
    wavelength : float
        Vacuum wavelength [m].
    per_surface : bool, default False
        ``False`` -> composite input->output ``(N, 4, 4)`` Jacobian.
        ``True``  -> per-surface local transfers ``(N_surf, N, 4, 4)`` obtained
        from the cumulative Jacobians ``J_local_k = J_cum_k @ inv(J_cum_{k-1})``
        (needed for the per-surface complex-parameter amplitude accumulation,
        where each near-identity local factor keeps the ``sqrt(det)`` branch
        unambiguous).
    h_pos, h_slope : float
        Central-FD steps for the position and slope perturbations.  The result
        is insensitive across several decades (the trace is smooth); the
        defaults sit in the truncation-limited regime for mm-scale optics.

    Returns
    -------
    DifferentialTransfer
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ux = np.asarray(ux, dtype=np.float64)
    uy = np.asarray(uy, dtype=np.float64)
    n = x.shape[0]
    steps = (h_pos, h_pos, h_slope, h_slope)

    # Build 9N rays: base + (+/-h) in each of x, y, ux, uy.  Group g runs
    # 0=base, then (dim, sign) pairs 1..8.
    base = (x, y, ux, uy)
    cols = [base]
    for dim in range(4):
        for sign in (+1.0, -1.0):
            pert = [a.copy() for a in base]
            pert[dim] = pert[dim] + sign * steps[dim]
            cols.append(tuple(pert))
    xx = np.concatenate([c[0] for c in cols])
    yy = np.concatenate([c[1] for c in cols])
    uxx = np.concatenate([c[2] for c in cols])
    uyy = np.concatenate([c[3] for c in cols])
    L, M, _ = _slopes_to_dirs(uxx, uyy)
    rb = _make_bundle(xx, yy, L, M, wavelength)

    of = 'all' if per_surface else 'last'
    res = trace(rb, surfaces, wavelength, output_filter=of)

    def _state(bundle):
        return (bundle.x, bundle.y, bundle.L / bundle.N, bundle.M / bundle.N,
                bundle.opd, bundle.alive)

    def _cum_jac(bundle):
        sx, sy, sux, suy, _, _ = _state(bundle)
        g = [(sx[i * n:(i + 1) * n], sy[i * n:(i + 1) * n],
              sux[i * n:(i + 1) * n], suy[i * n:(i + 1) * n])
             for i in range(9)]
        J = np.zeros((n, 4, 4))
        for dim in range(4):
            gp = g[1 + 2 * dim]
            gm = g[2 + 2 * dim]
            d = 2.0 * steps[dim]
            for outr in range(4):
                J[:, outr, dim] = (gp[outr] - gm[outr]) / d
        return J

    if not per_surface:
        img = res.image_rays
        Jc = _cum_jac(img)
        bx, by, bux, buy, bopd, balive = _state(img)
        return DifferentialTransfer(
            jacobian=Jc, x=bx[:n], y=by[:n], ux=bux[:n], uy=buy[:n],
            opd=bopd[:n], alive=np.asarray(balive[:n], bool))

    # per-surface: cumulative J at each surface -> local transfers
    hist = res.ray_history
    cum = [np.broadcast_to(np.eye(4), (n, 4, 4)).copy()]  # J at input = I
    for hb in hist:
        cum.append(_cum_jac(hb))
    locals_ = np.stack([cum[k + 1] @ np.linalg.inv(cum[k])
                        for k in range(len(hist))], axis=0)  # (Nsurf,n,4,4)
    final = hist[-1]
    bx, by, bux, buy, bopd, balive = _state(final)
    return DifferentialTransfer(
        jacobian=locals_, x=bx[:n], y=by[:n], ux=bux[:n], uy=buy[:n],
        opd=bopd[:n], alive=np.asarray(balive[:n], bool))


def ray_transfer_jacobian_jax(x, y, ux, uy, prescription, wavelength):
    """Differentiable per-ray composite ABCD Jacobian via ``jax`` autodiff.

    The JAX twin of :func:`ray_transfer_jacobian` (composite input->output):
    the 4x4 ``(x, y, ux, uy)`` ray-transfer Jacobian of the JAX-traceable
    :func:`raytrace.trace_jax`, computed by ``jax.jacfwd`` (EXACT, not finite
    differences) and vmapped over rays.  Because it is built on ``trace_jax``,
    the whole thing is ``jax.grad`` / ``jax.jit`` friendly and differentiable
    with respect to the ray state -- and, via
    :func:`raytrace.trace_jax_with_params`, with respect to prescription
    parameters (radii, thicknesses) -- enabling gradient-based lens design on
    the per-surface GBD.

    .. warning::
       ``jax_trace._transfer_jax`` propagates ``x_new = x + L * thickness``
       using the direction cosine ``L`` rather than the paraxial slope
       ``u = L / N``, so the ``B``-block is scaled by ~``N`` and deviates at
       high NA.  Use the NumPy finite-difference :func:`ray_transfer_jacobian`
       as the accuracy reference; this path is for **gradients / optimization**
       (and is exact at low NA, where ``L ~ u``).

    Returns
    -------
    jax array ``(N, 4, 4)``
        Per-ray composite ray-transfer Jacobian (row/col order (x, y, ux, uy)).
    """
    import jax
    import jax.numpy as jnp

    from .jax_trace import make_jax_ray_state, trace_jax

    def _out_state(s4):
        xx, yy, uxx, uyy = s4[0], s4[1], s4[2], s4[3]
        inv = 1.0 / jnp.sqrt(1.0 + uxx * uxx + uyy * uyy)
        st = make_jax_ray_state(
            jnp.reshape(xx, (1,)), jnp.reshape(yy, (1,)),
            jnp.zeros((1,)), jnp.reshape(uxx * inv, (1,)),
            jnp.reshape(uyy * inv, (1,)), jnp.reshape(inv, (1,)))
        r = trace_jax(st, prescription, wavelength)
        return jnp.stack([r.x[0], r.y[0], r.L[0] / r.N[0], r.M[0] / r.N[0]])

    s4 = jnp.stack([jnp.asarray(x), jnp.asarray(y),
                    jnp.asarray(ux), jnp.asarray(uy)], axis=-1)
    return jax.vmap(jax.jacfwd(_out_state))(s4)


__all__ = ['DifferentialTransfer', 'ray_transfer_jacobian',
           'ray_transfer_jacobian_jax']
