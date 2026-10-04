"""V5: does jax.hessian run through an eig-entering parameter (eps for
RCWA, angle for the 1-D PMM twin) on this tree?  Run once per tree."""
from _fix import pmm1d_f, rcwa_f
from _v import dump, jax, jnp, np

out = {}
for name, f, x0 in (("rcwa_x_te_sym", rcwa_f("x", "te", n_orders=2), 0.0),
                    ("rcwa_corner_te_conical",
                     rcwa_f("corner", "te", n_orders=2, theta=0.2, phi=0.3),
                     0.0),
                    ("pmm1d_angle_te_0.1", pmm1d_f("te"), 0.1),
                    ("pmm1d_index_te", pmm1d_f("te", "index"), 0.0)):
    w = np.random.default_rng(5).standard_normal(np.asarray(f(0.0, np)).size)

    def L(t, f=f, w=w):
        return jnp.dot(jnp.asarray(w), f(t, jnp))
    try:
        out[name] = float(jax.hessian(L)(x0))
    except Exception as e:  # noqa: BLE001
        out[name] = type(e).__name__ + ": " + str(e)[:120]
    print(name, out[name], flush=True)
dump("v5_hess_tree", out)
