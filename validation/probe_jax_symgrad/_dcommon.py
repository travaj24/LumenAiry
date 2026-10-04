"""Shared pieces of the d4-d7 probes (other JAX twins of _jax_eig_stable).

* ``capture(fn)``: run ``fn()`` (pass a ``jax.jit``-ed call, so the route is
  the one the gradient takes) with every ``_jax_eig_stable`` eig's
  eigenvalues recorded by a host callback -> list of spectrum records.
* ``rotated(seed)``: a context in which every eig returns its basis inside
  each EXACT cluster (gap <= 1e-12 max|lam|) rotated by a random unitary AND
  the library rule runs in that basis -- forward must not move; the gradient
  of a correct (basis-independent) rule must not move either.
* ``sweep(...)``: AD (jit(jacrev)) vs the premise-checked Richardson FD at a
  list of abscissae, plus the forward parity of the AD function against the
  FD oracle at x0 and x0 +- 1e-3.
"""
import contextlib

from _h import jax, jnp, ladder, min_rel_gap, n_pairs_below, np, rel

import lumenairy.elements.rcwa as RCpkg
import lumenairy.elements.rcwa._core as RCcore

_orig = RCpkg._jax_eig_stable


def _install(factory):
    """Install an eig factory where every tree reads it: the package
    attribute (the PRE tree's twins import it from there at call time) and
    the ``_core`` global (the routed twins' ``_jax_twin_eig`` calls it)."""
    RCpkg._jax_eig_stable = factory
    RCcore._jax_eig_stable = factory


def spec(lam):
    lam = np.asarray(lam)
    return {"n": int(lam.size), "max_abs": float(np.max(np.abs(lam))),
            "min_rel_gap": min_rel_gap(lam),
            "members_below_1e-12": n_pairs_below(lam, 1e-12),
            "members_below_1e-8": n_pairs_below(lam, 1e-8),
            "members_below_1e-6": n_pairs_below(lam, 1e-6)}


def capture(fn):
    got = []

    def factory():
        e = _orig()

        def eig(A, *a, **k):
            lam, V = e(A, *a, **k)
            # a host callback records the CONCRETE eigenvalues of the route
            # the traced (jit / grad) call takes, in call order
            jax.debug.callback(lambda v: got.append(np.asarray(v)), lam,
                               ordered=True)
            return lam, V
        return eig
    _install(factory)
    try:
        jax.block_until_ready(fn())
        jax.effects_barrier()
    finally:
        _install(_orig)
    return [spec(lam) for lam in got]


def _cluster_unitary(lam, seed):
    n = lam.shape[0]
    s = jnp.max(jnp.abs(lam))
    K = (jnp.abs(lam[:, None] - lam[None, :]) <= 1e-12 * s) | jnp.eye(
        n, dtype=bool)
    rng = np.random.default_rng(seed * 1000 + n)
    X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Y = jnp.where(K, X, 0.0)
    w, Q = jnp.linalg.eigh(jnp.conj(Y).T @ Y)
    return jax.lax.stop_gradient(
        Y @ ((Q * (1.0 / jnp.sqrt(w))[None, :]) @ jnp.conj(Q).T))


@contextlib.contextmanager
def rotated(seed):
    """Emulate "LAPACK returned another basis inside each exact cluster":
    the primal returns ``V U`` (U unitary, block-diagonal on the clusters
    with gap <= 1e-12 max|lam|) AND the library's own backward rule
    (``_jax_eig_stable().bwd``) runs on the residuals ``(lam, V U)`` -- i.e.
    the rule works in the rotated basis, exactly as if eig had produced it.
    (Rotating AFTER a plain eig call is NOT this: the rule would still work
    in LAPACK's basis and only the cotangent would be re-expressed.)"""
    from functools import partial
    base = _orig()

    @partial(jax.custom_vjp, nondiff_argnums=(1,))
    def eig_rot(A, tau_rel):
        lam, V = base(A, tau_rel)
        return lam, V @ _cluster_unitary(lam, seed)

    def fwd(A, tau_rel):
        out = eig_rot(A, tau_rel)
        return out, out

    def bwd(tau_rel, res, cot):
        return base.bwd(tau_rel, res, cot)

    eig_rot.defvjp(fwd, bwd)

    def factory():
        def eig(A, tau_rel=1e-12):
            return eig_rot(A, tau_rel)
        return eig
    _install(factory)
    try:
        yield
    finally:
        _install(_orig)


def premise_of(rat, fd, rows):
    """Premise ratios of the components carrying >= 1e-3 of max|FD|."""
    rat = np.asarray(rat)            # (n_steps-2, n_out)
    big = np.abs(fd) >= 1e-3 * np.max(np.abs(fd))
    vals = rat[:, big].ravel()
    return {"ratios_big_components": np.round(vals, 3).tolist(),
            "median": float(np.median(vals)) if vals.size else None}


def sweep(f_jax, f_np, xs, scale=1.0, mirror_pairs=None, label=""):
    g_fn = jax.jit(jax.jacrev(f_jax))
    out = {}
    for x in xs:
        g = np.asarray(g_fn(jnp.asarray(x)))
        rows, fd, rat = ladder(f_np, x, scale=scale)
        rec = {"AD": g.tolist(), "FD": fd.tolist(),
               "premise": premise_of(rat, fd, rows),
               "abs_err": float(np.max(np.abs(g - fd))),
               "scale": float(np.max(np.abs(fd))),
               "rel_err": rel(g, fd)}
        if mirror_pairs is not None:
            a = np.asarray([g[i] + g[j] for i, j in mirror_pairs])
            b = np.asarray([fd[i] + fd[j] for i, j in mirror_pairs])
            rec["mirror_defect_AD_rel"] = float(np.max(np.abs(a))
                                                / np.max(np.abs(g)))
            rec["mirror_defect_FD_rel"] = float(np.max(np.abs(b))
                                                / np.max(np.abs(fd)))
        out[repr(x)] = rec
        print(label, "x=%r" % x, "rel %.3e abs %.3e scale %.3e" % (
            rec["rel_err"], rec["abs_err"], rec["scale"]),
            "premise med", rec["premise"]["median"],
            ("mirror AD %.2e FD %.2e" % (rec["mirror_defect_AD_rel"],
                                          rec["mirror_defect_FD_rel"])
             if mirror_pairs is not None else ""), flush=True)
    return out


def parity(f_jax, f_np, x0, h=1e-3):
    fj = jax.jit(f_jax)
    return float(max(np.max(np.abs(np.asarray(fj(jnp.asarray(x)))
                                   - np.asarray(f_np(x))))
                     for x in (x0, x0 + h, x0 - h)))


def gauge(f_jax, x0, seeds=(1, 2)):
    g0 = np.asarray(jax.jit(jax.jacrev(f_jax))(jnp.asarray(x0)))
    v0 = np.asarray(jax.jit(f_jax)(jnp.asarray(x0)))
    gch, vch = [], []
    for s in seeds:
        with rotated(s):    # fresh lambdas: no reuse of a cached trace
            v = np.asarray(jax.jit(lambda t: f_jax(t))(jnp.asarray(x0)))
            g = np.asarray(jax.jit(jax.jacrev(lambda t: f_jax(t)))(
                jnp.asarray(x0)))
        gch.append(float(np.max(np.abs(g - g0)) / np.max(np.abs(g0))))
        vch.append(float(np.max(np.abs(v - v0))))
    return {"grad_change_rel": gch, "value_change": vch}
