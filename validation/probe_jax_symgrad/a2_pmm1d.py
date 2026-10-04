"""A2: the 1-D PMM twin's d / d(angle) at EXACTLY normal incidence -- which
eig is degenerate, and is the wrong gradient the degenerate-cluster class or
something else on that path?

    python a2_pmm1d.py

``pmm_efficiency_1d`` (JAX), P 1.2, ridge n 2 / groove 1, n_sup 1.45,
n_sub 1, depth 0.45, duty 0.5, wl 1, degree 12, stabilize=False; R and T of
the +-1 orders.

1. SPECTRUM of the three eigs of the path (the layer, the superstrate, the
   substrate fold eig(B^-1 A)) at angle 0 and off it: gaps, members, the
   distance to the sqrt branch point q^2 = 0.
2. ANGLE SWEEP: AD vs a premise-checked NumPy Richardson FD at angle
   0, 1e-9, 1e-7, 1e-5, 1e-3; relative AND absolute error; the mirror
   identity d R_{+1} = -d R_{-1} at 0.
3. GAUGE, PER EIG SITE: the basis inside every exact cluster
   (gap <= 1e-12 max|lam|) of ONE site (layer / sup / sub) rotated by a
   random unitary (two seeds): forward change, gradient change.
4. OTHER CANDIDATES: the NumPy Wood nudge (_wood_safe_wl_1d) at every FD
   abscissa (identity?); NumPy stabilize=True vs False at 0 and +-h
   (same oracle?); the forward-branch selector's mask at the AD point
   (any mode inside the band?).
"""
import warnings

from _h import dump, jax, jnp, ladder, min_rel_gap, n_pairs_below, np, rel

import lumenairy.elements.rcwa as RCpkg
from lumenairy.elements.pmm import _core as PC, pmm_efficiency_1d

P, WL = 1.2, 1.0
o, _R, _T = pmm_efficiency_1d(P, 2.0, 1.0, 1.45, 1.0, 0.45, 0.5, WL,
                              degree=12, stabilize=False)
o = np.asarray(o)
IDX = [int(np.nonzero(o == m)[0][0]) for m in (1, -1)]


def f(t, xp, pol, stabilize=False):
    _o, R, T = pmm_efficiency_1d(P, xp.asarray(2.0 + 0j), 1.0, 1.45, 1.0,
                                 0.45, 0.5, WL, angle=t, polarization=pol,
                                 degree=12, stabilize=stabilize)
    return xp.concatenate([xp.stack([R[i] for i in IDX]),
                           xp.stack([T[i] for i in IDX])])


_orig = RCpkg._jax_eig_stable
SITES = ("layer", "sup", "sub")
out = {"spectrum": {}, "sweep": {}, "gauge": {}, "other": {}}

# 1. spectrum (eager JAX: the eigenvalues are concrete)
for pol in ("te", "tm"):
    for a in (0.0, 1e-7, 1e-5, 1e-3):
        got = []

        def cap():
            e = _orig()

            def eig(A, tau_rel=1e-12):
                lam, V = e(A, tau_rel)
                if not isinstance(lam, jax.core.Tracer):   # skip eval_shape
                    got.append(np.asarray(lam))
                return lam, V
            return eig
        RCpkg._jax_eig_stable = cap
        try:
            f(jnp.asarray(a), jnp, pol)
        finally:
            RCpkg._jax_eig_stable = _orig
        for s, lam in zip(SITES, got):
            rec = {"n": int(lam.size), "max_abs": float(np.max(np.abs(lam))),
                   "min_rel_gap": min_rel_gap(lam),
                   "members_below_1e-6": n_pairs_below(lam, 1e-6),
                   "members_below_1e-12": n_pairs_below(lam, 1e-12),
                   "min_abs_rel_to_branch_point":
                       float(np.min(np.abs(lam)) / np.max(np.abs(lam)))}
            out["spectrum"][f"{pol}_{a!r}_{s}"] = rec
            print("spectrum", pol, a, s, rec, flush=True)

# 2. angle sweep
for pol in ("te", "tm"):
    g_fn = jax.jit(jax.jacrev(lambda t, pol=pol: f(t, jnp, pol)))
    for a in (0.0, 1e-9, 1e-7, 1e-5, 1e-3):
        g = np.asarray(g_fn(a))
        _rows, fd, rat = ladder(lambda t, pol=pol: np.asarray(f(t, np, pol)),
                                a)
        rec = {"AD": g.tolist(), "FD": fd.tolist(),
               "premise": np.asarray(rat).ravel().round(2).tolist(),
               "abs_err": float(np.max(np.abs(g - fd))),
               "scale": float(np.max(np.abs(fd))),
               "rel_err": rel(g, fd),
               "mirror_defect_AD": float(np.max(np.abs(g[0::2] + g[1::2]))),
               "mirror_defect_FD": float(np.max(np.abs(fd[0::2]
                                                       + fd[1::2])))}
        out["sweep"][f"{pol}_{a!r}"] = rec
        print(pol, "angle %.0e" % a, "rel %.2e abs %.2e scale %.2e" % (
            rec["rel_err"], rec["abs_err"], rec["scale"]),
            "mirror AD %.1e" % rec["mirror_defect_AD"],
            "premise", rec["premise"][:2], flush=True)


# 3. gauge per site
def rotating(site_idx, seed):
    def factory():
        e = _orig()
        count = [0]

        def eig(A, tau_rel=1e-12):
            lam, V = e(A, tau_rel)
            k = count[0] % 3
            count[0] += 1
            if k != site_idx:
                return lam, V
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
    return factory


for pol in ("te", "tm"):
    g0 = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp, pol)))(0.0))
    v0 = np.asarray(jax.jit(lambda t: f(t, jnp, pol))(0.0))
    for si, s in enumerate(SITES):
        ch_g, ch_v = [], []
        for seed in (1, 2):
            RCpkg._jax_eig_stable = rotating(si, seed)
            try:
                v = np.asarray(jax.jit(lambda t: f(t, jnp, pol))(0.0))
                g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp, pol)))(
                    0.0))
            finally:
                RCpkg._jax_eig_stable = _orig
            ch_g.append(float(np.max(np.abs(g - g0)) / np.max(np.abs(g0))))
            ch_v.append(float(np.max(np.abs(v - v0))))
        out["gauge"][f"{pol}_{s}"] = {"grad_change_rel": ch_g,
                                      "value_change": ch_v}
        print("gauge", pol, s, ch_g, ch_v, flush=True)

# 4. other candidates
wood = {}
for a in (0.0, 1e-3, -1e-3, 1e-4, -1e-4, 3e-4, -3e-4):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        w = PC._wood_safe_wl_1d(WL, a, 1.45, P, [1.45 ** 2, 1.0, 4.0, 1.0],
                                3, fn_name="probe")
    wood[repr(a)] = float(w) - WL
out["other"]["wood_nudge_minus_wl"] = wood
stab = {}
for pol in ("te", "tm"):
    for a in (0.0, 1e-4, -1e-4):
        stab[f"{pol}_{a!r}"] = float(np.max(np.abs(
            np.asarray(f(a, np, pol, stabilize=True))
            - np.asarray(f(a, np, pol, stabilize=False)))))
out["other"]["stabilize_true_minus_false"] = stab
print("wood", wood, "stab", stab, flush=True)
print(dump("a2_pmm1d.json", out))
