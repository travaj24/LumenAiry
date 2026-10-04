"""D3: EME eig-based mode twins (``ref_2d_modes`` / ``ref_2d_modes_vector``
on JAX input -> ``eme._jax_modes``) at an exactly degenerate cluster.

    python d3_eme.py

Configuration: a UNIFORM eps = 4 square Bloch cell, Lx = Ly = 1, k0 = 2 pi,
kx0 = ky0 = 0 (scalar Nx = Ny = 6 -> A 36x36; vector Nx = Ny = 4 -> G 64x64):
translation + C4v symmetry => exact clusters (plane waves (+-1,0),(0,+-1) ...).
Symmetry-BREAKING parameter x: eps[1, 2] = 4 + x (one off-centre pixel).
Symmetry-KEEPING control: k0.

The consumer here is the SORTED EIGENVALUE LIST (no eigenvectors).  Outputs:
the first K sorted qz^2 individually + the SUM over each complete cluster
inside the first K (a symmetric, smooth function of the cluster).

1. spectrum of the eig at x = 0 (n, max|lam|, min gap, members <= 1e-12).
2. AD (jit(jacrev)) vs a Richardson central FD of the twin's own concrete
   forward (premise ratios), per output; and at offsets x0 = 4e-6, 4e-4
   (1e-6, 1e-4 relative) with h scaled by x0 (stay on one side of the kink).
3. the first-order in-cluster block Bc = (V^-1 dA V)[c, c] in the LAPACK basis
   (dA by central difference of the captured matrix): AD of a member must
   equal diag(Bc) (basis dependent), the one-sided FD slopes are eig(Bc)
   (basis invariant), the central FD is their mean pairing.
4. control: d/dk0 at x = 0.
"""
from _h import dump, jax, jnp, ladder, min_rel_gap, n_pairs_below, np, rel

import lumenairy.elements.eme._jax_modes as JM
from lumenairy.elements.eme import ref_2d_modes, ref_2d_modes_vector

L_, K0, E0, PIX = 1.0, 2 * np.pi, 4.0, (1, 2)
CFG = {"scalar": (ref_2d_modes, 6, 9), "vector": (ref_2d_modes_vector, 4, 20)}
_orig = JM._jax_eig_stable
out = {"config": dict(L=L_, k0=K0, eps=E0, pixel=PIX), "fam": {}}


def qz2(kind, x, k0=K0, xp=jnp):
    fn, n, _K = CFG[kind]
    e = np.full((n, n), E0, dtype=complex)
    m = np.zeros((n, n))
    m[PIX] = 1.0
    eps = xp.asarray(e) + x * xp.asarray(m)
    return fn(eps, L_, L_, n, n, k0)


def capture(kind, x):
    got = []

    def factory():
        e = _orig()

        def eig(A, tau_rel=1e-12):
            lam, V = e(A, tau_rel)
            if not isinstance(lam, jax.core.Tracer):
                got.append((np.asarray(A), np.asarray(lam), np.asarray(V)))
            return lam, V
        return eig
    JM._jax_eig_stable = factory
    try:
        q = np.asarray(qz2(kind, jnp.asarray(x)))
    finally:
        JM._jax_eig_stable = _orig
    return q, got[0]


def clusters(q, K, tol=1e-9):
    s = np.max(np.abs(q))
    cl, cur = [], [0]
    for i in range(1, K):
        if abs(q[i] - q[cur[-1]]) <= tol * s:
            cur.append(i)
        else:
            cl.append(cur)
            cur = [i]
    cl.append(cur)
    # keep only clusters complete inside the first K
    last = cl[-1]
    if last and abs(q[K] - q[last[-1]]) <= tol * s:
        cl = cl[:-1]
    return cl


for kind in ("scalar", "vector"):
    _fn, _n, K = CFG[kind]
    q0, (A0, lam0, V0) = capture(kind, 0.0)
    cl = clusters(q0, K)
    rec = {"spectrum": {"n": int(lam0.size),
                        "max_abs": float(np.max(np.abs(lam0))),
                        "min_rel_gap": min_rel_gap(lam0),
                        "members_below_1e-12": n_pairs_below(lam0, 1e-12),
                        "members_below_1e-9": n_pairs_below(lam0, 1e-9)},
           "output_clusters": cl, "q_first_K": q0[:K + 1].tolist()}
    print(kind, "spectrum", rec["spectrum"], "clusters", cl, flush=True)

    def y(x, k0=K0, cl=cl, kind=kind, K=K):
        q = qz2(kind, x, k0)
        return jnp.concatenate([q[:K], jnp.stack([jnp.sum(q[jnp.asarray(c)])
                                                  for c in cl])])

    def yc(x, y=y):          # concrete twin forward for the FD oracle
        return np.asarray(y(jnp.asarray(x)))

    g_fn = jax.jit(jax.jacrev(lambda x: y(x)))
    names = [f"q{i}" for i in range(K)] + [f"sum{c}" for c in cl]
    sym_idx = list(range(K, K + len(cl)))
    member_idx = [i for c in cl if len(c) > 1 for i in c]
    for x0 in (0.0, 4e-6, 4e-4):
        g = np.asarray(g_fn(x0))
        scale = 1.0 if x0 == 0.0 else x0
        _rows, fd, rat = ladder(yc, x0, scale=scale)
        r = {"AD": g.tolist(), "FD": fd.tolist(), "names": names,
             "premise": np.asarray(rat).ravel().round(3).tolist(),
             "rel_err_all": rel(g, fd),
             "rel_err_cluster_sums": rel(g[sym_idx], fd[sym_idx]),
             "rel_err_degenerate_members":
                 rel(g[member_idx], fd[member_idx]) if member_idx else None}
        rec[f"x0_{x0!r}"] = r
        print(kind, "x0", x0, "rel all %.2e sums %.2e members %s" % (
            r["rel_err_all"], r["rel_err_cluster_sums"],
            r["rel_err_degenerate_members"]), "premise(sums)",
            np.asarray(rat)[:, sym_idx].ravel().round(2).tolist(), flush=True)
    # 3. in-cluster first-order block in the LAPACK basis
    hb = 1e-6
    _q, (Ap, _l, _v) = capture(kind, hb)
    _q, (Am, _l, _v) = capture(kind, -hb)
    dA = (Ap - Am) / (2 * hb)
    Bfull = np.linalg.solve(V0, dA @ V0)
    s = np.max(np.abs(lam0))
    blocks = []
    if kind == "scalar":
        lamq = lam0.real
    else:
        lamq = (-(lam0 ** 2)).real
    for c in cl:
        if len(c) < 2:
            continue
        sel = np.nonzero(np.abs(lamq - q0[c[0]]) <= 1e-9 * s * (
            1 if kind == "scalar" else s))[0]
        subs = []                      # split into eig-level clusters
        for i in sel:
            for sb in subs:
                if abs(lam0[i] - lam0[sb[0]]) <= 1e-9 * s:
                    sb.append(i)
                    break
            else:
                subs.append([i])
        dg, ev, offn, nrm = [], [], 0.0, 0.0
        for sb in subs:
            Bc = Bfull[np.ix_(sb, sb)]
            if kind == "vector":   # d qz2 = -2 gam d gam
                Bc = -2.0 * lam0[sb][:, None] * Bc
            dg += np.diag(Bc).real.tolist()
            ev += np.linalg.eigvals(Bc).real.tolist()
            offn += np.linalg.norm(Bc - np.diag(np.diag(Bc))) ** 2
            nrm += np.linalg.norm(Bc) ** 2
        hs = 1e-7
        fp = (yc(hs) - yc(0.0)) / hs
        fm = (yc(0.0) - yc(-hs)) / hs
        blocks.append({"cluster": c, "size_in_eig": int(sel.size),
                       "eig_subclusters": [len(sb) for sb in subs],
                       "diag_Bc_sorted": np.sort(dg).tolist(),
                       "eig_Bc_sorted": np.sort(ev).tolist(),
                       "offdiag_norm_rel": float(np.sqrt(offn / max(
                           nrm, 1e-300))),
                       "AD_members_sorted": np.sort(np.asarray(
                           rec["x0_0.0"]["AD"])[c]).tolist(),
                       "FD_forward_members": fp[c].tolist(),
                       "FD_backward_members": fm[c].tolist(),
                       "FD_central_members": np.asarray(
                           rec["x0_0.0"]["FD"])[c].tolist()})
        print(kind, "block", blocks[-1], flush=True)
    rec["blocks"] = blocks
    # 4. control d/dk0 at x = 0
    gk = np.asarray(jax.jit(jax.jacrev(lambda k: y(0.0, k)))(K0))
    _rows, fdk, ratk = ladder(lambda k: np.asarray(y(jnp.asarray(0.0),
                                                     jnp.asarray(k))), K0)
    rec["control_k0"] = {"AD": gk.tolist(), "FD": fdk.tolist(),
                         "rel_err": rel(gk, fdk),
                         "premise": np.asarray(ratk).ravel().round(3).tolist()}
    print(kind, "control k0 rel %.2e" % rel(gk, fdk), flush=True)
    out["fam"][kind] = rec
print(dump("d3_eme.json", out))
