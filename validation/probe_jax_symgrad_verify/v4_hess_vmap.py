"""V4: jax.hessian and jax.vmap(jax.grad) through the rule, against their
un-vmapped / FD counterparts, at the symmetric point (a cluster: the
lifted branch) and away from it (no cluster), RCWA (x-widening of the C4v
post, TE, n_orders 2) and the 1-D PMM twin (angle, TE and TM).

Loss L(t) = w . f(t) with a fixed random weight vector w over all R, T.
Hessian oracle: Richardson FD of the AD gradient (correct off the symmetric
point: V1 / V2) at +-h, h = 2e-2 .. 5e-3 (RCWA) / 4e-3 .. 1e-3 (PMM), with
the h^2 premise; and a second difference of the NumPy forward.
"""
from _fix import pmm1d_f, rcwa_f
from _v import dump, fd_halving, jax, jnp, np, rel

out = {}


def run(name, f, x0s, h0):
    n = np.asarray(f(0.0, np)).size
    w = np.random.default_rng(5).standard_normal(n)

    def L(t):
        return jnp.dot(jnp.asarray(w), f(t, jnp))

    def Ln(t):
        return float(np.dot(w, np.asarray(f(t, np))))
    gfun = jax.jit(jax.grad(L))
    for x0 in x0s:
        rec = {}
        try:
            H = float(jax.hessian(L)(x0))
        except Exception as e:  # noqa: BLE001
            H = float("nan")
            rec["hessian_error"] = repr(e)[:300]
        Hfd, prem = fd_halving(lambda t: np.array([float(gfun(t))]), x0, h0)
        # second difference of the NumPy forward, Richardson
        rows = []
        for h in (h0, h0 / 2, h0 / 4):
            rows.append((Ln(x0 + h) - 2 * Ln(x0) + Ln(x0 - h)) / h ** 2)
        H2 = (4 * rows[2] - rows[1]) / 3
        rec.update({"hessian": H, "fd_of_ad_grad": float(Hfd[0]),
                    "premise": prem, "second_diff_numpy": H2,
                    "rel_vs_fd_of_grad": abs(H - Hfd[0]) / abs(Hfd[0]),
                    "rel_vs_second_diff": abs(H - H2) / abs(H2)})
        out[f"{name}_hess_x{x0:g}"] = rec
        print(name, x0, rec, flush=True)
    xs = jnp.asarray([0.0, 1e-3, -2e-3, 1e-2])
    gv = np.asarray(jax.jit(jax.vmap(jax.grad(L)))(xs))
    gs = np.asarray([float(gfun(x)) for x in xs])
    out[f"{name}_vmap_grad"] = {"vmapped": gv.tolist(), "single": gs.tolist(),
                                "rel": rel(gv, gs)}
    print(name, "vmap", out[f"{name}_vmap_grad"], flush=True)


run("rcwa_x_te", rcwa_f("x", "te", n_orders=2), (0.0, 0.05), 2e-2)
run("pmm1d_angle_te", pmm1d_f("te"), (0.0, 0.1), 4e-3)
run("pmm1d_angle_tm", pmm1d_f("tm"), (0.0, 0.1), 4e-3)
dump("v4_hess_vmap", out)
