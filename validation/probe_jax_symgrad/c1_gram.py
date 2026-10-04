"""C1: why the 1-D PMM twin hands the rule its PENCIL (A, B) and not the
fold B^-1 A.  The same rule, the same eig, with the problems passed folded
(Euclidean Gram for the cluster lift) vs as the pencil (Gram B, the mass
matrix): d / d(angle) of R, T of the +-1 orders vs the NumPy Richardson FD
at angles where the half-space pairs are NEAR-degenerate (split / max|lam|
= 4.8e-3 x angle, inside the rule's gap_rel = 1e-6 below ~2e-4 rad).

    python c1_gram.py
"""
from _h import dump, jax, jnp, ladder, np, rel

import lumenairy.elements.rcwa._core as RC
from lumenairy.elements.pmm import pmm_efficiency_1d

P, WL = 1.2, 1.0
o = np.asarray(pmm_efficiency_1d(P, 2.0, 1.0, 1.45, 1.0, 0.45, 0.5, WL,
                                 degree=12, stabilize=False)[0])
IDX = [int(np.nonzero(o == m)[0][0]) for m in (1, -1)]


def f(t, xp, pol):
    _o, R, T = pmm_efficiency_1d(P, xp.asarray(2.0 + 0j), 1.0, 1.45, 1.0,
                                 0.45, 0.5, WL, angle=t, polarization=pol,
                                 degree=12, stabilize=False)
    return xp.concatenate([xp.stack([R[i] for i in IDX]),
                           xp.stack([T[i] for i in IDX])])


_orig = RC._jax_eig_cluster_adjoint


def folded(eig_fn, problems, consumer, **kw):
    probs = tuple((jnp.linalg.solve(G, L), None) for L, G in problems)
    n = probs[0][0].shape[0]
    return _orig(lambda A, _G: eig_fn(A, jnp.eye(n, dtype=A.dtype)), probs,
                 consumer, **kw)


out = {}
for pol in ("te", "tm"):
    for a in (0.0, 1e-7, 1e-6, 1e-5, 1e-4):
        _r, fd, rat = ladder(lambda t: np.asarray(f(t, np, pol)), a)
        rec = {"FD": fd.tolist(),
               "premise": np.asarray(rat).ravel().round(2).tolist()}
        for name, fn in (("pencil", _orig), ("fold", folded)):
            RC._jax_eig_cluster_adjoint = fn
            try:
                g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp, pol)))(
                    a))
            finally:
                RC._jax_eig_cluster_adjoint = _orig
            rec[name] = {"AD": g.tolist(), "rel_err": rel(g, fd)}
        out[f"{pol}_{a!r}"] = rec
        print(pol, "angle %.0e" % a, "pencil %.2e fold %.2e" % (
            rec["pencil"]["rel_err"], rec["fold"]["rel_err"]), flush=True)
print(dump("c1_gram.json", out))
