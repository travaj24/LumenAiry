"""Differentiable (JAX) twin of the SEM-basis axisymmetric BOR-PMM solve.

Mirrors :meth:`BORStack._solve_sem` on JAX inputs.  Gradients flow through
each traced SEGMENT permittivity (scalar or the diagonal cylindrical
``(eps_rr, eps_phiphi, eps_zz)`` triple -- the LC-director design knob),
traced uniform-layer ``eps=`` values, and traced layer thicknesses.  Walls,
meshes, the azimuthal order and the half-spaces stay concrete.

The split exploits the SEM assembly's structure: permittivity enters the
Galerkin blocks ONLY as three per-element scalars multiplying geometry-only
element mass matrices (``Krr_eps``, ``Kpp_eps``, ``Mz_eps`` in
``sem_radial._assemble``); everything else -- stiffness/coupling blocks, the
axis/PEC keep masks, the field-recovery operators, the cross-tested mortar
projections -- is geometry, precomputed ONCE in NumPy on placeholder meshes
and frozen into the trace.  The traced part is: the three eps-weighted mass
assemblies, the ``E_z`` Schur elimination ``Lei = inv(-S_z + k0^2 Mz_eps)``,
the equilibrated-fold eigensolve (the rcwa gauge-stable custom-VJP
``_jax_eig_stable``, exactly the ``_jax_bor`` recipe), the modal field/flux
reconstruction, the mortar interface algebra, and the Redheffer cascade
(S-matrix primitives reused from ``_jax_bor``).

MESH TOPOLOGY IS VALUE-DEPENDENT on this basis: the wavelength-resolution cap
sizes elements from the LOCAL refractive index, which a traced eps cannot
provide under a trace.  The rule, per segment-eps component: a plain number
uses its actual value; a JAX value uses ``BORStack(n_mesh_cap=...)`` when set
(an UPPER bound -- the mesh can only over-resolve, so it is safe and keeps
the mesh IDENTICAL across AD/FD evaluations of a design loop), else its
concrete value when one is extractable (a non-traced jnp array), else raises.

Deliberately NOT on this path (both warn/raise on the NumPy path too):
``retain_internal``/``layer_absorption``, and the longitudinal-resonance
UNREDUCED-QZ fallback (QZ has no traceable VJP; detune k0 -- the NumPy path
warns at a resonance, so a traced sweep through one is detectable there).

Like the RCWA/PMM/FD-BOR twins, the propagating ORDER SET cannot be
materialized under a trace, so ``R``/``T`` return as full per-mode arrays
masked to 0 off the propagating set; the TOTAL ``sum(R)``/``sum(T)`` match
the NumPy solve and a scalar design loss differentiates.
"""
from __future__ import annotations

import numpy as np

from ...backend import is_jax_array as _is_jax_array
from ._jax_bor import _jbor_ismat, _jbor_psmat, _jbor_star
from .radial_eigensolver import _lagrange_vals_derivs
from .sem_radial import SemRadialMesh, _assemble, _keeps, _overlap


def _conc(v):
    """Concrete complex value of ``v``, or None when it is a JAX tracer."""
    if not _is_jax_array(v):
        return complex(v)
    try:
        return complex(np.asarray(v))
    except TypeError:                    # TracerArrayConversionError subclass
        return None


def _tri_nloc(tri, n_mesh_cap):
    """Local |n| for the wavelength-resolution cap (module-docstring rule)."""
    n = 1e-3
    for t in tri:
        if _is_jax_array(t):
            if n_mesh_cap is not None:
                n = max(n, n_mesh_cap)
                continue
            c = _conc(t)
            if c is None:
                raise NotImplementedError(
                    "BORStack(basis='sem'): a traced segment eps reached the "
                    "mesh builder, whose wavelength-resolution cap needs a "
                    "numeric refractive index.  Pass BORStack(n_mesh_cap=...) "
                    "-- a concrete UPPER bound on |n| over the traced values "
                    "(safe: the mesh can only over-resolve).")
            n = max(n, abs(np.sqrt(c).real))
        else:
            n = max(n, abs(np.sqrt(complex(t)).real))
    return n


def _build_bnd(stack, wall_list, segs, k0):
    """(breakpoints, per-element raw eps triples) for one layer -- the traced
    mirror of ``_solve_sem.build_mesh`` (keep in LOCKSTEP with it; the
    forward-parity test pins the two producing identical meshes on concrete
    input)."""
    bnd = np.concatenate([[0.0], np.asarray(wall_list, float), [stack.Rbig]])
    bnd = np.unique(bnd)
    keep = np.concatenate([[True], np.diff(bnd) > 1e-12 * stack.Rbig])
    bnd = bnd[keep]
    if abs(bnd[-1] - stack.Rbig) > 0:
        bnd[-1] = stack.Rbig
    k = stack.elements_per_segment
    if k > 1:
        if stack.grade:
            sloc = 0.5 * (1.0 - np.cos(np.pi * np.arange(k + 1) / k))
        else:
            sloc = np.linspace(0.0, 1.0, k + 1)
        out = [bnd[0]]
        for lo, hi in zip(bnd[:-1], bnd[1:]):
            out.extend(lo + (hi - lo) * sloc[1:])
        bnd = np.asarray(out)
    DPW = 8.0
    edges = np.asarray([rs for rs, _t in segs])
    tris = [t for _rs, t in segs]
    out = [float(bnd[0])]
    for lo, hi in zip(bnd[:-1], bnd[1:]):
        jdx = int(np.searchsorted(edges, 0.5 * (lo + hi), side="left"))
        n_loc = _tri_nloc(tris[min(jdx, len(tris) - 1)], stack.n_mesh_cap)
        lam_loc = 2.0 * np.pi / (k0 * n_loc)
        max_el = stack.degree * lam_loc / DPW
        nsplit = max(1, int(np.ceil((hi - lo) / max_el)))
        out.extend(np.linspace(lo, hi, nsplit + 1)[1:])
    bnd = np.asarray(out)
    eps_el = []
    for lo, hi in zip(bnd[:-1], bnd[1:]):
        jdx = int(np.searchsorted(edges, 0.5 * (lo + hi), side="left"))
        eps_el.append(tris[min(jdx, len(tris) - 1)])
    return bnd, eps_el


def _equalize(pairs):
    """Joint (breakpoints, raw-eps) twin of ``sem_radial.equalize_meshes`` --
    same deterministic largest-element midpoint split, applied to the raw
    (possibly traced) per-element eps lists in lockstep."""
    target = max(len(b) - 1 for b, _e in pairs)
    out = []
    for b, eps in pairs:
        b, eps = list(b), list(eps)
        while len(b) - 1 < target:
            widths = np.diff(np.asarray(b))
            j = int(np.argmax(widths))
            b.insert(j + 1, b[j] + widths[j] / 2.0)
            eps.insert(j, eps[j])
        out.append((np.asarray(b), eps))
    return out


def _sem_static(bnd, degree, m):
    """All geometry-only (eps-independent) operators of one layer's mesh,
    NumPy, frozen into the trace.  Built on a PLACEHOLDER unit-eps mesh --
    ``_assemble``'s eps-weighted outputs are discarded and reassembled traced
    from the per-element mass stacks."""
    mesh = SemRadialMesh(bnd, [(1.0,) * 3] * (len(bnd) - 1), degree)
    ops = _assemble(mesh, m, 1.0)                 # k0 unused by the geometry
    ip, iz = _keeps(mesh, m)
    m00s, m11s, g0s, g1s = [], [], [], []
    for e in range(mesh.ne):
        a, _bb, J = mesh.el(e)
        r_real = a + (mesh.xq + 1.0) * J
        r, s_pml = mesh.stretch(r_real)           # identity (no stack PML)
        w = mesh.wq * J * s_pml
        wr = w * r
        m00s.append((mesh.V0q * wr[:, None]).T @ mesh.V0q)
        m11s.append((mesh.V1q * wr[:, None]).T @ mesh.V1q)
        g0s.append(mesh.g0(e))
        g1s.append(mesh.g1(e))
    r0, w0 = mesh.nodes0()
    _r1, w1 = mesh.nodes1()
    n0, n1 = mesh.n0, mesh.n1
    # E_z/r recovery operator (the sem_layer_modes axis-limit rule)
    EZR = np.zeros((n0, n0), dtype=complex)
    pos = np.where(r0 > 0.0)[0]
    EZR[pos, pos] = 1.0 / r0[pos]
    if m != 0:
        _a0, _b0, J0 = mesh.el(0)
        Daxis = _lagrange_vals_derivs(mesh.ref0, mesh.ref0[:1])[1] / J0
        EZR[0, :] = 0.0
        EZR[0, g0s[0]] = Daxis[0, :]
    # E_z' at the V1 (Gauss) nodes, per element (rows disjoint -> assignment)
    GD = np.zeros((n1, n0), dtype=complex)
    for e in range(mesh.ne):
        _a, _b, J = mesh.el(e)
        GD[np.ix_(g1s[e], g0s[e])] = mesh.D0g / J
    return dict(mesh=mesh, ip=ip, iz=iz, n0=n0, n1=n1, ne=mesh.ne,
                g0=g0s, g1=g1s, m00=m00s, m11=m11s,
                M_rr=ops["M_rr"], N_rr=ops["N_rr"], M_pp=ops["M_pp"],
                S_A=ops["S_A"], S_z=ops["S_z"], C1=ops["C1"],
                D_zr=ops["D_zr"], Q=ops["Q"],
                EZR=EZR, GD=GD, w0=w0, w1=w1)


def _jsem_layer_modes(st, tri_list, m, k0, eig, jnp):
    """Traced modal basis ``(W, V, q)`` of one SEM layer (jnp twin of
    ``sem_radial.sem_layer_modes``, Schur-elimination branch)."""
    cj = jnp.complex128
    n0, n1 = st["n0"], st["n1"]
    ip, iz = st["ip"], st["iz"]
    er = [jnp.asarray(t[0]).astype(cj) for t in tri_list]
    ep = [jnp.asarray(t[1]).astype(cj) for t in tri_list]
    ez = [jnp.asarray(t[2]).astype(cj) for t in tri_list]
    Krr = jnp.zeros((n1, n1), cj)
    Kpp = jnp.zeros((n0, n0), cj)
    Mze = jnp.zeros((n0, n0), cj)
    for e in range(st["ne"]):
        i0, i1 = st["g0"][e], st["g1"][e]
        Krr = Krr.at[i1[:, None], i1[None, :]].add(
            er[e] * jnp.asarray(st["m11"][e], cj))
        m00 = jnp.asarray(st["m00"][e], cj)
        Kpp = Kpp.at[i0[:, None], i0[None, :]].add(ep[e] * m00)
        Mze = Mze.at[i0[:, None], i0[None, :]].add(ez[e] * m00)
    S_z = jnp.asarray(st["S_z"], cj)
    Mz = (-S_z + k0 * k0 * Mze)[np.ix_(iz, iz)]
    Lei = jnp.linalg.inv(Mz)
    Dzr = jnp.asarray(st["D_zr"][iz, :], cj)
    Qzp = jnp.asarray(st["Q"][np.ix_(iz, ip)], cj)
    C1p = jnp.asarray(st["C1"][:, ip], cj)
    K = jnp.block([
        [k0 * k0 * Krr - m * m * jnp.asarray(st["N_rr"], cj), -1j * m * C1p],
        [+1j * m * C1p.T,
         (k0 * k0 * Kpp - jnp.asarray(st["S_A"], cj))[np.ix_(ip, ip)]]])
    LD, LQ = Lei @ Dzr, Lei @ Qzp
    B = jnp.block([
        [jnp.asarray(st["M_rr"], cj) + Dzr.T @ LD, -1j * m * (Dzr.T @ LQ)],
        [+1j * m * (Qzp.T @ LD),
         jnp.asarray(st["M_pp"][np.ix_(ip, ip)], cj) + m * m * (Qzp.T @ LQ)]])
    # equilibrated folded standard eig (the _fast_geig / _jax_bor recipe)
    dB = jnp.sqrt(jnp.abs(jnp.diag(B)))
    dB = jnp.where(dB > 0, 1.0 / dB, 1.0).astype(cj)
    Ke = (dB[:, None] * K) * dB[None, :]
    Be = (dB[:, None] * B) * dB[None, :]
    q2, z = eig(jnp.linalg.solve(Be, Ke))
    Vm = dB[:, None] * z
    q = jnp.sqrt(q2)
    Er, Ephi_red = Vm[:n1, :], Vm[n1:, :]
    nm = q.shape[0]
    Ephi_full = jnp.zeros((n0, nm), cj).at[ip].set(Ephi_red)
    Ez_red = Lei @ (-1j * (Dzr @ Er) - m * (Qzp @ Ephi_red))
    Ez_full = jnp.zeros((n0, nm), cj).at[iz].set(Ez_red)   # E_z PER UNIT q
    Ez_over_r = jnp.asarray(st["EZR"], cj) @ Ez_full
    dEz_g = jnp.asarray(st["GD"], cj) @ Ez_full
    w0 = jnp.asarray(st["w0"])
    w1 = jnp.asarray(st["w1"])

    def _hfields(qv):
        hr = (m * (Ez_over_r * qv[None, :]) - qv[None, :] * Ephi_full) / k0
        hphi = (qv[None, :] * Er + 1j * (dEz_g * qv[None, :])) / k0
        return hr, hphi

    def _flux(hr, hphi):
        return jnp.real(
            jnp.sum(Er * jnp.conj(hphi) * w1[:, None], axis=0)
            - jnp.sum(Ephi_full * jnp.conj(hr) * w0[:, None], axis=0))

    hr, hphi = _hfields(q)
    Pz = _flux(hr, hphi)
    prop = jnp.abs(jnp.imag(q)) < 1e-9 * jnp.maximum(
        jnp.abs(jnp.real(q)), 1e-300)
    flip = jnp.where(prop, Pz < 0.0, jnp.imag(q) < 0.0)
    q = jnp.where(flip, -q, q)                    # orient forward (+z)
    hr, hphi = _hfields(q)
    Pz = _flux(hr, hphi)
    fnrm = (jnp.sum(jnp.abs(Er) ** 2 * w1[:, None], axis=0)
            + jnp.sum(jnp.abs(Ephi_full) ** 2 * w0[:, None], axis=0))
    s = jnp.where(jnp.abs(Pz) > 1e-10 * fnrm,
                  1.0 / jnp.sqrt(jnp.abs(Pz) + 1e-300),
                  1.0 / (jnp.sqrt(fnrm) + 1e-300)).astype(cj)
    W = jnp.concatenate([Er * s[None, :], (Ephi_full * s[None, :])[ip]],
                        axis=0)
    V = jnp.concatenate([(hr * s[None, :])[ip], hphi * s[None, :]], axis=0)
    return W, V, q


def _ifc_static(sta, stb):
    """Geometry of one cross-mesh mortar interface (the NumPy projection
    matrices of ``sem_radial.sem_interface_smatrix``); None when the meshes
    coincide (equal-mesh interfaces use the plain ``_jbor_ismat``)."""
    ma, mb = sta["mesh"], stb["mesh"]
    if (ma.p == mb.p and ma.b.shape == mb.b.shape
            and np.array_equal(ma.b, mb.b)):
        return None
    ipa, ipb = sta["ip"], stb["ip"]
    G1_ba = _overlap(ma, mb, "v1")
    G0_ba = _overlap(ma, mb, "v0")
    P1_ba = np.linalg.solve(_overlap(mb, mb, "v1"), G1_ba)
    M0b = _overlap(mb, mb, "v0")[np.ix_(ipb, ipb)]
    P0_ba = np.linalg.solve(M0b, G0_ba[np.ix_(ipb, ipa)])
    P1_ab = np.linalg.solve(_overlap(ma, ma, "v1"), G1_ba.T)
    M0a = _overlap(ma, ma, "v0")[np.ix_(ipa, ipa)]
    P0_ab = np.linalg.solve(M0a, G0_ba[np.ix_(ipb, ipa)].T)
    return dict(P1_ba=P1_ba, P0_ba=P0_ba, P1_ab=P1_ab, P0_ab=P0_ab,
                n1a=sta["n1"], n0pb=stb["ip"].size)


def _jsem_ismat(pack, A, B, jnp):
    """Traced interface S-matrix (cross-tested Galerkin mortar when the
    meshes differ; the algebra of ``sem_interface_smatrix``, verbatim)."""
    Wa, Va, _qa = A
    Wb, Vb, _qb = B
    if pack is None:
        return _jbor_ismat(Wa, Va, Wb, Vb, jnp)
    cj = jnp.complex128
    n1a, n0pb = pack["n1a"], pack["n0pb"]
    Wa_b = jnp.concatenate([jnp.asarray(pack["P1_ba"], cj) @ Wa[:n1a],
                            jnp.asarray(pack["P0_ba"], cj) @ Wa[n1a:]], axis=0)
    Vb_a = jnp.concatenate([jnp.asarray(pack["P0_ab"], cj) @ Vb[:n0pb],
                            jnp.asarray(pack["P1_ab"], cj) @ Vb[n0pb:]], axis=0)
    alpha = jnp.linalg.solve(Wb, Wa_b)
    gamma = jnp.linalg.solve(Va, Vb_a)
    ga = gamma @ alpha
    eye = jnp.eye(ga.shape[0], dtype=cj)
    inv_iga = jnp.linalg.inv(eye + ga)
    S11 = inv_iga @ (eye - ga)
    S12 = 2.0 * (inv_iga @ gamma)
    S21 = alpha @ (eye + S11)
    S22 = alpha @ S12 - eye
    return (S11, S12, S21, S22)


def _jax_sem_stack_solve(stack):
    """Differentiable ``BORStack(basis='sem').solve``.  Returns the FD-twin
    dict: full per-mode ``R``/``T`` masked to 0 off the propagating set (the
    TOTAL ``sum(R)``/``sum(T)`` match the NumPy solve), plus ``S``,
    ``q_sup``/``q_sub`` and the masks."""
    import jax.numpy as jnp

    from ..rcwa import _jax_eig_stable, _require_jax_x64
    _require_jax_x64("BORStack.solve")
    if stack.k0 is None:
        raise RuntimeError("call set_source(...) before solve()")
    eig = _jax_eig_stable()
    k0 = float(stack.k0)
    m = stack.m
    wl_now = 2.0 * np.pi / k0
    sem_layers = []
    for i, sc in enumerate(stack._sem_layers):
        if sc is None:
            raise ValueError(
                f"BORStack(basis='sem'): layer {i} was added with a spec the "
                "SEM basis cannot mesh on the traced path -- re-add it with "
                "segments=[(r_out, eps), ...] or eps=.")
        walls_i, segs_i = sc
        resolved = []
        for r_out, tri in segs_i:
            resolved.append((r_out, tuple(
                complex(t(wl_now)) if callable(t)
                else (t if _is_jax_array(t) else complex(t))
                for t in tri)))
        sem_layers.append((walls_i, tuple(resolved)))
    walls = [sc[0] for sc in sem_layers]
    win = []
    for i in range(len(walls)):
        u = set(walls[i])
        if i > 0:
            u |= set(walls[i - 1])
        if i + 1 < len(walls):
            u |= set(walls[i + 1])
        win.append(sorted(u))
    sup_walls = sorted(set(walls[0])) if walls else []
    sub_walls = sorted(set(walls[-1])) if walls else []
    eps_sup = complex(stack.eps_sup)              # half-spaces stay CONCRETE
    eps_sub = complex(stack.eps_sub)
    sup_segs = ((stack.Rbig, (eps_sup,) * 3),)
    sub_segs = ((stack.Rbig, (eps_sub,) * 3),)
    pairs = ([_build_bnd(stack, sup_walls, sup_segs, k0)]
             + [_build_bnd(stack, win[i], sem_layers[i][1], k0)
                for i in range(len(walls))]
             + [_build_bnd(stack, sub_walls, sub_segs, k0)])
    pairs = _equalize(pairs)
    statics = [_sem_static(b, stack.degree, m) for b, _e in pairs]

    def _modes(st, eps_el):
        return _jsem_layer_modes(st, eps_el, m, k0, eig, jnp)

    sup = _modes(statics[0], pairs[0][1])
    same_hs = (eps_sub == eps_sup
               and np.array_equal(statics[0]["mesh"].b, statics[-1]["mesh"].b))
    sub = sup if same_hs else _modes(statics[-1], pairs[-1][1])
    mids = [(thk, _modes(statics[1 + i], pairs[1 + i][1]))
            for i, (thk, _jb, _raw) in enumerate(stack._jax_layers)]
    if mids:
        S = _jsem_ismat(_ifc_static(statics[0], statics[1]), sup, mids[0][1],
                        jnp)
        for i, (thk, L) in enumerate(mids):
            S = _jbor_star(S, _jbor_psmat(L[2], thk, jnp), jnp)
            if i + 1 < len(mids):
                nxt, st_n = mids[i + 1][1], statics[2 + i]
            else:
                nxt, st_n = sub, statics[-1]
            S = _jbor_star(S, _jsem_ismat(_ifc_static(statics[1 + i], st_n),
                                          L, nxt, jnp), jnp)
    else:
        S = _jsem_ismat(_ifc_static(statics[0], statics[-1]), sup, sub, jnp)
    S11, _S12, S21, _S22 = S
    q_sup, q_sub = sup[2], sub[2]

    def _mask(q, eps):
        # the _solve_sem prop() gate (P2-06 + the 2026-07-13 cutoff audit)
        qn = q / k0
        return ((jnp.abs(jnp.imag(qn)) < 5e-5) & (jnp.real(qn) > 1e-6)
                & (np.sqrt(eps).real - jnp.real(qn) > -5e-10))

    inc = _mask(q_sup, eps_sup).astype(jnp.complex128)
    out = _mask(q_sub, eps_sub).astype(jnp.complex128)
    R = jnp.real(jnp.sum(jnp.abs(S11) ** 2 * inc[:, None], axis=0)) \
        * jnp.real(inc)
    T = jnp.real(jnp.sum(jnp.abs(S21) ** 2 * out[:, None], axis=0)) \
        * jnp.real(inc)
    return dict(R=R, T=T, energy=R + T, S=S, q_sup=q_sup, q_sub=q_sub,
                inc_mask=jnp.real(inc), out_mask=jnp.real(out))
