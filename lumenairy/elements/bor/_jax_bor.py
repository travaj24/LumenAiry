"""Differentiable (JAX) twin of the axisymmetric BOR-PMM stack solve.

Mirrors :meth:`lumenairy.elements.bor.BORStack.solve` on JAX inputs: gradients
flow through each traced LAYER permittivity / ring index (the half-spaces stay
concrete).  Layer order/count and the radial grid are static; the eps-dependent
assembly, the staggered ``K x = q^2 B x`` eigensolve, the modal field
reconstruction, and the Redheffer cascade are all traced.

The geometry-only staggered stencils (the two-grid derivative / average
operators, the node Laplacian) are precomputed ONCE in NumPy and frozen into the
trace; only the eps-dependent ``K``/``B``, the eig (via the rcwa gauge-stable
custom-VJP ``_jax_eig_stable`` on the equilibrated fold ``inv(Be) Ke`` -- the
"generalized eig VJP" is a phantom, the fold is a plain standard eig), and the
per-mode field/flux reconstruction (vectorized, with a trace-safe ``where``
forward-orientation) are traced.  x64 required; ``jnp.linalg.eig`` is CPU-only.

The reflected / transmitted SET of propagating orders is a data-dependent
integer set that cannot be materialized under a trace, so -- like the RCWA/PMM
twins -- ``R``/``T`` are returned as full-``2N`` per-mode arrays with the
non-propagating modes masked to 0 by a trace-safe ``where`` (the mask is a
measure-zero step, constant under small perturbations, so the gradient flows
through the amplitudes).  A scalar design loss (``sum(R)`` total reflectance,
one order's ``R``) is differentiable; the TOTAL ``sum(R)`` / ``sum(T)`` match the
NumPy solve (order-/gauge-invariant).
"""
from __future__ import annotations

import numpy as np

from .coupled_radial_eigensolver import _fd_grid_staggered


def _jbor_static(m, Rbig, N):
    """Geometry-only staggered stencils (NumPy, frozen into the trace)."""
    r_n, r_f, h, Dn2f, Df2n, An2f, Af2n = _fd_grid_staggered(Rbig, N)
    mrn, mrf = m / r_n, m / r_f
    A_n2f = Dn2f + np.diag(1.0 / r_f) @ An2f
    A_f2n = Df2n + np.diag(1.0 / r_n) @ Af2n
    Lm = np.diag(1.0 / r_n) @ Df2n @ np.diag(r_f) @ Dn2f - np.diag(mrn ** 2)
    return dict(N=N, h=h, r_n=r_n, r_f=r_f, Dn2f=Dn2f, Df2n=Df2n,
                A_n2f=A_n2f, A_f2n=A_f2n, Lm=Lm, mrn=mrn, mrf=mrf)


def _jbor_layer_modes(static, eps_node, k0, eig, jnp):
    """Forward modal basis ``(W, V, q)`` of one layer from its nodal eps
    (jnp twin of ``zcascade._layer_modes_staggered``)."""
    cj = jnp.complex128
    N, h = static["N"], static["h"]
    Dn2f = jnp.asarray(static["Dn2f"], cj)
    Df2n = jnp.asarray(static["Df2n"], cj)
    A_n2f = jnp.asarray(static["A_n2f"], cj)
    A_f2n = jnp.asarray(static["A_f2n"], cj)
    Lm = jnp.asarray(static["Lm"], cj)
    mrn = jnp.asarray(static["mrn"], cj)
    mrf = jnp.asarray(static["mrf"], cj)
    wq_n = jnp.asarray(static["r_n"] * h, cj)
    wq_f = jnp.asarray(static["r_f"] * h, cj)
    Imat = jnp.eye(N, dtype=cj)
    eps_node = jnp.asarray(eps_node, cj)
    # wall-normal (face) eps = harmonic mean of the straddling nodes (reduces to
    # the node value inside a homogeneous ring); last face = last node.
    inv = 1.0 / eps_node
    eps_face = jnp.concatenate([2.0 / (inv[:-1] + inv[1:]), eps_node[-1:]])

    Mz = Lm + k0 ** 2 * jnp.diag(eps_node)
    Lei = jnp.linalg.inv(Mz)
    Phi_r = Lei @ (1j * A_f2n)
    Phi_p = Lei @ (-jnp.diag(mrn))
    B = jnp.block([[Imat + 1j * Dn2f @ Phi_r, 1j * Dn2f @ Phi_p],
                   [-jnp.diag(mrn) @ Phi_r,   Imat - jnp.diag(mrn) @ Phi_p]])
    K = jnp.block([[k0 ** 2 * jnp.diag(eps_face) - jnp.diag(mrf ** 2),
                    -1j * jnp.diag(mrf) @ A_n2f],
                   [-1j * Df2n @ jnp.diag(mrf),
                    k0 ** 2 * jnp.diag(eps_node) + Df2n @ A_n2f]])

    # equilibrated folded standard eig (mirrors _fast_geig)
    dB = jnp.sqrt(jnp.abs(jnp.diag(B)))
    dB = jnp.where(dB > 0, 1.0 / dB, 1.0).astype(cj)
    Ke = (dB[:, None] * K) * dB[None, :]
    Be = (dB[:, None] * B) * dB[None, :]
    q2, z = eig(jnp.linalg.solve(Be, Ke))
    Vm = dB[:, None] * z
    q = jnp.sqrt(q2)
    Er, Ephi = Vm[:N, :], Vm[N:, :]              # faces / nodes, (N, nm)

    def _hfields(qr):
        Ez = qr[None, :] * (Lei @ (1j * A_f2n @ Er - mrn[:, None] * Ephi))
        hr = (1.0 / k0) * (mrn[:, None] * Ez - qr[None, :] * Ephi)      # nodes
        hphi = (1.0 / k0) * (qr[None, :] * Er + 1j * (Dn2f @ Ez))       # faces
        return hr, hphi

    def _flux(hr, hphi):
        return jnp.real(jnp.sum(Er * jnp.conj(hphi) * wq_f[:, None], axis=0)
                        - jnp.sum(Ephi * jnp.conj(hr) * wq_n[:, None], axis=0))

    hr, hphi = _hfields(q)
    Pz = _flux(hr, hphi)
    propagating = jnp.abs(jnp.imag(q)) < 1e-9 * jnp.maximum(
        jnp.abs(jnp.real(q)), 1e-300)
    flip = jnp.where(propagating, Pz < 0.0, jnp.imag(q) < 0.0)
    q = jnp.where(flip, -q, q)                    # orient forward (+z)
    hr, hphi = _hfields(q)
    Pz = _flux(hr, hphi)

    # flux-normalize (propagating), else field-norm (evanescent) -- safe divides
    fnrm = (jnp.sum(jnp.abs(Er) ** 2 * wq_f[:, None], axis=0)
            + jnp.sum(jnp.abs(Ephi) ** 2 * wq_n[:, None], axis=0))
    strong = jnp.abs(Pz) > 1e-10 * jnp.real(fnrm)
    s_flux = 1.0 / jnp.sqrt(jnp.where(strong, jnp.abs(Pz), 1.0))
    s_norm = 1.0 / jnp.sqrt(jnp.sum(jnp.abs(Er) ** 2 + jnp.abs(Ephi) ** 2,
                                    axis=0) + 1e-300)
    s = jnp.where(strong, s_flux, s_norm).astype(cj)
    W = jnp.concatenate([Er * s[None, :], Ephi * s[None, :]], axis=0)
    V = jnp.concatenate([hr * s[None, :], hphi * s[None, :]], axis=0)
    return W, V, q


# --- S-matrix primitives (jnp twins of zcascade's; backward == [W; -V]) ------
def _jbor_ismat(Wa, Va, Wb, Vb, jnp):
    a = jnp.linalg.solve(Wb, Wa)
    b = jnp.linalg.solve(Vb, Va)
    apb, amb = a + b, a - b
    iapb = jnp.linalg.inv(apb)
    return (-iapb @ amb, 2.0 * iapb,
            0.5 * (apb - amb @ iapb @ amb), amb @ iapb)


def _jbor_psmat(q, L, jnp):
    X = jnp.diag(jnp.exp(1j * q * L))
    Z = jnp.zeros_like(X)
    return (Z, X, X, Z)


def _jbor_star(SA, SB, jnp):
    A11, A12, A21, A22 = SA
    B11, B12, B21, B22 = SB
    Imat = jnp.eye(A11.shape[0], dtype=jnp.complex128)
    D = jnp.linalg.inv(Imat - B11 @ A22)
    F = jnp.linalg.inv(Imat - A22 @ B11)
    return (A11 + A12 @ D @ B11 @ A21, A12 @ D @ B12,
            B21 @ F @ A21, B22 + B21 @ F @ A22 @ B12)


def _jax_bor_stack_solve(stack):
    """Differentiable ``BORStack.solve``.  Returns the same dict as the NumPy
    path but with ``R``/``T`` as full-``2N`` per-mode arrays masked to 0 off the
    propagating set (so a scalar loss differentiates; ``sum(R)``/``sum(T)`` match
    the NumPy totals)."""
    import jax.numpy as jnp

    from ..rcwa import _jax_eig_stable, _require_jax_x64
    _require_jax_x64("BORStack.solve")
    if stack.k0 is None:
        raise RuntimeError("call set_source(...) before solve()")
    eig = _jax_eig_stable()
    k0 = float(stack.k0)
    static = _jbor_static(stack.m, stack.Rbig, stack.N)
    N, r_n = stack.N, static["r_n"]
    eps_sup = complex(stack.eps_sup)             # half-spaces stay CONCRETE
    eps_sub = complex(stack.eps_sub)

    def _modes(eps_node):
        return _jbor_layer_modes(static, eps_node, k0, eig, jnp)

    sup = _modes(jnp.full((N,), eps_sup, dtype=jnp.complex128))
    sub = (sup if eps_sub == eps_sup
           else _modes(jnp.full((N,), eps_sub, dtype=jnp.complex128)))
    mids = [(thk, _modes(jbuild(r_n, jnp)))
            for thk, jbuild, _raw in stack._jax_layers]

    if mids:
        S = _jbor_ismat(sup[0], sup[1], mids[0][1][0], mids[0][1][1], jnp)
        for i, (thk, L) in enumerate(mids):
            S = _jbor_star(S, _jbor_psmat(L[2], thk, jnp), jnp)
            nxt = mids[i + 1][1] if i + 1 < len(mids) else sub
            S = _jbor_star(S, _jbor_ismat(L[0], L[1], nxt[0], nxt[1], jnp), jnp)
    else:
        S = _jbor_ismat(sup[0], sup[1], sub[0], sub[1], jnp)
    S11, _S12, S21, _S22 = S
    q_sup, q_sub = sup[2], sub[2]

    def _mask(q, eps):
        # AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13: real-axis floor at
        # 1e-6 (guards only the q ~ 0 degenerate point).  The old 0.05 was an
        # angular cutoff that dropped genuinely propagating near-grazing
        # orders.
        #
        # S1-16 (audit AUDIT_V5_24_2): shares the {imag, real-floor,
        # index-ceiling} core with bor_stack.solve's prop() and
        # bor_solve._physical_propagating.  Like the bor_stack twin it carries
        # NO reldiv leg BY DESIGN -- the JAX BOR path is staggered-only
        # (div-conforming, spurious-free), so reldiv is structurally 0; only
        # bor_solve's optional NODAL basis needs it.
        qn = q / k0
        return ((jnp.abs(jnp.imag(qn)) < 5e-5) & (jnp.real(qn) > 1e-6)
                & (jnp.real(jnp.sqrt(eps)) - jnp.real(qn) > -5e-10))

    inc = _mask(q_sup, eps_sup).astype(jnp.complex128)     # propagating in super
    out = _mask(q_sub, eps_sub).astype(jnp.complex128)     # propagating in sub
    # R[j] = sum_jp |S11[jp,j]|^2 over propagating reflected orders jp (masked);
    # meaningful for incident orders j where inc[j]; masked to 0 elsewhere.
    R = jnp.real(jnp.sum(jnp.abs(S11) ** 2 * inc[:, None], axis=0)) * jnp.real(inc)
    T = jnp.real(jnp.sum(jnp.abs(S21) ** 2 * out[:, None], axis=0)) * jnp.real(inc)
    return dict(R=R, T=T, energy=R + T, S=S, q_sup=q_sup, q_sub=q_sub,
                inc_mask=jnp.real(inc), out_mask=jnp.real(out))
