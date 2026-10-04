"""The most symmetric point of a patterned layer: an exactly UNIFORM
(isotropic) eps_cell, differentiated in a pattern direction (topology
optimisation from a uniform start).  RCWAStack (JAX): a uniform layer over a
cross layer.  AD at t = 0 vs AD at t = +-1e-6 (the eig branch) vs NumPy FD.
python p9_uniform.py <tag>   (LUMROOT selects the tree)"""
import json
import sys

from _vc import BUILD, fd_rich, jax, jnp, np, premise_ok, rel

from lumenairy.elements.rcwa import RCWAStack

S = 24
M = np.zeros((S, S), bool)
M[8:16, :] = True
M[:, 8:16] = True
RND = np.random.default_rng(11).uniform(-1, 1, (S, S))
RND0 = RND - RND.mean()                      # zero-mean pattern
out = {"build": BUILD}


def make(theta, phi, eps_u, pattern, second=True, eps_key="eps_cell"):
    def f(t, xp):
        st = RCWAStack(1.3, period_y=1.3, n_superstrate=1.0, n_substrate=1.5,
                       n_orders=3, n_orders_y=3)
        u = xp.asarray(np.full((S, S), eps_u)) + t * xp.asarray(pattern)
        if eps_key == "eps_tensor_cell":
            u = u[:, :, None, None] * xp.eye(3, dtype=complex)[None, None]
        st.add_layer(0.25, **{eps_key: u})
        if second:
            st.add_layer(0.2, eps_cell=xp.asarray(np.where(M, 4.0 + 0j,
                                                           1.44 + 0j)))
        st.set_source(1.0, theta=theta, phi=phi)
        _o, R, T = st.solve().efficiencies()
        return xp.concatenate([xp.ravel(R), xp.ravel(T)])
    return f


rows = []
for name, args in (
        ("conical_two_layers", (0.2, 0.3, 2.25 + 0.05j, RND0)),
        ("normal_two_layers", (0.0, 0.0, 2.25 + 0.0j, RND0)),
        ("normal_two_layers_tensorcell", (0.0, 0.0, 2.25 + 0.0j, RND0, True,
                                          "eps_tensor_cell")),
        ("normal_single_layer", (0.0, 0.0, 2.25 + 0.0j, RND0, False))):
    f = make(*args)
    fd, ratio, ex = fd_rich(lambda t: f(t, np), 0.0, floor=1e-2)
    jg = jax.jit(jax.jacrev(lambda t: f(t, jnp)))
    g0 = np.asarray(jg(0.0))
    gp = np.asarray(jg(1e-6))
    gm = np.asarray(jg(-1e-6))
    r = dict(case=name, err_ad_at_0=rel(g0, fd),
             err_ad_at_pm1e6=rel(0.5 * (gp + gm), fd),
             ad0_vs_ad_pm=rel(g0, 0.5 * (gp + gm)),
             premise=premise_ok(ratio, ex), gmax=float(np.max(np.abs(fd))),
             ratio=[float(ratio.min()), float(np.median(ratio)),
                    float(ratio.max())])
    print(r, flush=True)
    rows.append(r)
out["rows"] = rows
json.dump(out, open(f"p9_uniform_{sys.argv[1]}_{BUILD}.json", "w"),
          indent=1)
