"""A1: the RCWA JAX path at a four-fold symmetric pixel cell -- what is
degenerate, and is the wrong symmetry-breaking gradient the degenerate-
cluster class?

    python a1_rcwa.py

The cell (the E3 round-2 fixture): 15 x 15 pixels, centre block eps 4, side
blocks 1.5, corners 1; t added to the two x-side blocks (breaks the
90-degree rotation, keeps both mirrors).  ``rcwa_efficiency_2d``,
n_orders 3 x 3, P = 1.2, wl = 1, depth 0.45, n_sup 1.45, n_sub 1.

1. SPECTRUM: the one eig of this path is the layer's ``P @ Q`` (the
   half-spaces are analytic); its pairwise gaps at the symmetric cell and
   at offsets delta (t = delta) -- clusters, members, distance of the
   spectrum to the sqrt branch point lam^2 = 0.
2. OFFSET SWEEP: the cell moved off symmetry by delta in {0, 1e-12 .. 1e-2}
   (delta added to the x-side blocks); AD vs a premise-checked NumPy
   Richardson FD of d R / d t, d T / d t ((0,0), (+-1,0)) at t = delta.
2b. LOWER-SYMMETRY OFFSETS: the same sweep with the offset added to ONE
   x-side block only (the y-mirror survives) and as a fixed random pixel
   pattern (no symmetry left: the split members belong to no symmetry
   sector, so their eigenvectors need not be orthogonal -- the case where a
   Euclidean-Gram lift can push eigenvalues off the real axis).
3. GAUGE: the eig replaced by one whose basis inside every exact cluster
   (gap <= 1e-12 max|lam|) is a random unitary rotation of LAPACK's (two
   seeds): forward change and gradient change at delta = 0.
"""
from _h import dump, jax, jnp, ladder, min_rel_gap, n_pairs_below, np, rel

import lumenairy.elements.rcwa._core as RC
import lumenairy.elements.rcwa.twod as RT
from lumenairy.elements.rcwa import rcwa_efficiency_2d

P, WL = 1.2, 1.0
base3 = np.array([[1.0, 1.5, 1.0], [1.5, 4.0, 1.5], [1.0, 1.5, 1.0]])
xs3 = np.zeros((3, 3))
xs3[0, 1] = xs3[2, 1] = 1.0
base, dirn = (np.kron(a, np.ones((5, 5))) for a in (base3, xs3))
o, _R, _T = rcwa_efficiency_2d(P, P, base.astype(complex), 1.45, 1.0, 0.45,
                               WL, n_orders_x=3, n_orders_y=3)
o = np.asarray(o)
IDX = [int(np.nonzero((o[:, 0] == a) & (o[:, 1] == b))[0][0])
       for a, b in ((0, 0), (1, 0), (-1, 0))]


one = np.zeros((3, 3))
one[0, 1] = 1.0
OFFSETS = {"one_block": np.kron(one, np.ones((5, 5))),
           "random": np.random.default_rng(7).uniform(0.0, 1.0, (15, 15))}


def f(t, xp, pol, off=0.0, kind="one_block"):
    eps = (xp.asarray(base) + t * xp.asarray(dirn)
           + off * xp.asarray(OFFSETS[kind])).astype(complex)
    _o, R, T = rcwa_efficiency_2d(P, P, eps, 1.45, 1.0, 0.45, WL,
                                  polarization=pol, n_orders_x=3,
                                  n_orders_y=3)
    return xp.concatenate([xp.stack([R[i] for i in IDX]),
                           xp.stack([T[i] for i in IDX])])


_orig_eig_for = RC._eig_for
captured = []


def capturing(xp):
    e = _orig_eig_for(xp)
    if xp is np:
        def eig(A):
            lam, V = e(A)
            captured.append(np.asarray(lam))
            return lam, V
        return eig
    return e


out = {"spectrum": {}, "offset": {}, "offset_generic": {}, "gauge": {}}
# 1. spectrum
for d in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
    captured.clear()
    RC._eig_for = capturing
    try:
        f(d, np, "te")
    finally:
        RC._eig_for = _orig_eig_for
    lam2 = captured[0]
    rec = {"n": int(lam2.size), "max_abs": float(np.max(np.abs(lam2))),
           "min_rel_gap": min_rel_gap(lam2),
           "members_below_1e-6": n_pairs_below(lam2, 1e-6),
           "members_below_1e-12": n_pairs_below(lam2, 1e-12),
           "min_abs_rel_to_branch_point": float(np.min(np.abs(lam2))
                                                / np.max(np.abs(lam2)))}
    out["spectrum"][repr(d)] = rec
    print("spectrum delta", d, rec, flush=True)

# 2. offset sweep
for pol in ("te", "tm"):
    g_fn = jax.jit(jax.jacrev(lambda t, pol=pol: f(t, jnp, pol)))
    for d in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
        g = np.asarray(g_fn(d))
        _rows, fd, rat = ladder(lambda t, pol=pol: np.asarray(f(t, np, pol)),
                                d)
        rec = {"AD": g.tolist(), "FD": fd.tolist(),
               "premise": np.asarray(rat).ravel().round(2).tolist(),
               "abs_err": float(np.max(np.abs(g - fd))),
               "rel_err": rel(g, fd)}
        out["offset"][f"{pol}_{d!r}"] = rec
        print(pol, "delta %.0e" % d, "rel err %.2e" % rec["rel_err"],
              "premise", rec["premise"][:3], flush=True)


# 2b. lower-symmetry offsets
for kind, pol in [(k, p) for k in OFFSETS for p in ("te", "tm")]:
    for d in (1e-10, 1e-8, 1e-6, 1e-4):
        g = np.asarray(jax.jit(jax.jacrev(
            lambda t, pol=pol, d=d, kind=kind: f(t, jnp, pol, d, kind)))(0.0))
        _rows, fd, rat = ladder(lambda t, pol=pol, d=d, kind=kind: np.asarray(
            f(t, np, pol, d, kind)), 0.0)
        captured.clear()
        RC._eig_for = capturing
        try:
            f(0.0, np, pol, d, kind)
        finally:
            RC._eig_for = _orig_eig_for
        rec = {"AD": g.tolist(), "FD": fd.tolist(),
               "premise": np.asarray(rat).ravel().round(2).tolist(),
               "rel_err": rel(g, fd), "min_rel_gap": min_rel_gap(captured[0]),
               "members_below_1e-6": n_pairs_below(captured[0], 1e-6)}
        out["offset_generic"][f"{kind}_{pol}_{d!r}"] = rec
        print(kind, pol, "delta %.0e" % d, "rel err %.2e" % rec["rel_err"],
              "gap %.1e" % rec["min_rel_gap"], "premise", rec["premise"][:3],
              flush=True)

# 3. gauge (the eig is patched where each tree reads it: _core's
# _layer_eigenmodes before the fix, twod's entry after it)
def rotating(seed):
    orig = RC._jax_eig_stable()

    def eig(A, tau_rel=RC._EIG_TAU_REL):
        lam, V = orig(A, tau_rel)
        n = lam.shape[0]
        s = jnp.max(jnp.abs(lam))
        K = (jnp.abs(lam[:, None] - lam[None, :]) <= 1e-12 * s) | jnp.eye(
            n, dtype=bool)
        rng = np.random.default_rng(seed * 1000 + n)
        X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Y = jnp.where(K, X, 0.0)
        w, Q = jnp.linalg.eigh(jnp.conj(Y).T @ Y)
        U = Y @ ((Q * (1.0 / jnp.sqrt(w))[None, :]) @ jnp.conj(Q).T)
        return lam, V @ jax.lax.stop_gradient(U)
    return eig


for pol in ("te", "tm"):
    vals, grads = [], []
    for seed in (None, 1, 2):
        if seed is not None:
            ef = rotating(seed)
            RC._eig_for = RT._eig_for = (
                lambda xp, ef=ef: ef if xp is not np else _orig_eig_for(xp))
        try:
            vals.append(np.asarray(jax.jit(lambda t, pol=pol: f(
                t, jnp, pol))(0.0)))
            grads.append(np.asarray(jax.jit(jax.jacrev(
                lambda t, pol=pol: f(t, jnp, pol)))(0.0)))
        finally:
            RC._eig_for = RT._eig_for = _orig_eig_for
    sc = float(np.max(np.abs(grads[0])))
    rec = {"grad_change_rel": [float(np.max(np.abs(g - grads[0]))) / sc
                               for g in grads[1:]],
           "value_change": [float(np.max(np.abs(v - vals[0])))
                            for v in vals[1:]],
           "AD": [g.tolist() for g in grads]}
    out["gauge"][pol] = rec
    print("gauge", pol, rec["grad_change_rel"], rec["value_change"],
          flush=True)
print(dump("a1_rcwa.json", out))
