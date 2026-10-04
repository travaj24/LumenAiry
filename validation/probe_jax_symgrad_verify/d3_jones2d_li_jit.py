"""D3: rcwa_jones_2d(formulation='li') under jax.jit: a TRACED tensor cell
routes to the general (out-of-plane) cascade, whose operator build has no
'li' branch (direct rule for every block).  Compare the jitted forward with
the eager JAX forward, NumPy 'li' and NumPy 'laurent'."""
import _fix
from _v import dump, jax, jnp, np, rel

from lumenairy.elements.rcwa import rcwa_jones_2d

out = {}
for cname, base in (("c4v_post", _fix.BASE), ("c1_random", 2.25 + 0.6 * _fix.RAND)):
    cell = (base[..., None, None] * np.eye(3)).astype(complex)

    def f(c, form, **kw):
        o, R, T, J = rcwa_jones_2d(0.9, 0.9, c, 1.52, 1.0, 0.37, 1.0,
                                   n_orders_x=2, n_orders_y=2,
                                   formulation=form, **kw)
        return np.concatenate([np.ravel(np.asarray(R)),
                               np.ravel(np.asarray(T))]).real
    eager = f(jnp.asarray(cell), "li")
    jitted = np.asarray(jax.jit(lambda c: jnp.concatenate([
        jnp.ravel(x) for x in rcwa_jones_2d(
            0.9, 0.9, c, 1.52, 1.0, 0.37, 1.0, n_orders_x=2, n_orders_y=2,
            formulation="li")[1:3]]).real)(jnp.asarray(cell)))
    np_li = f(cell, "li", symmetry=False)
    np_la = f(cell, "laurent", symmetry=False)
    rec = {"eager_vs_numpy_li": rel(eager, np_li),
           "jit_vs_numpy_li": rel(jitted, np_li),
           "jit_vs_numpy_laurent": rel(jitted, np_la),
           "li_vs_laurent_numpy": rel(np_li, np_la)}
    out[cname] = rec
    print(cname, rec, flush=True)
dump("d3_jones2d_li_jit", out)
