"""The API examples of the CHANGELOG entry and of the BUILD doc section 2.5,
run as written except for n_modes (3 here, for speed).  Output
f11_api_examples.json."""
import jax
import jax.numpy as jnp
from _e3common import dump

from lumenairy.elements.pmm import Circle, PMM2DStackPure, pmm_jones_2d_staggered

jax.config.update("jax_enable_x64", True)
NM = 3
orders, R, T, J = pmm_jones_2d_staggered(
    1.2, 1.2, None, 1.45, 1.0, 0.5, 1.0, n_modes=NM, n_orders=2,
    shapes=[Circle(0.6, 0.6, 0.36, eps=4.0)], background_eps=1.0,
    backend="jax")
st = PMM2DStackPure(1.2, 1.2, n_superstrate=1.0, n_substrate=1.45,
                    n_modes=NM, n_orders=2, backend="jax")
st.add_layer(0.5, shapes=[Circle(0.6, 0.6, 0.36, eps=4.0)], background_eps=1.0)
st.set_source(1.0)
p0 = st.jax_twin().p0


def t00(r):
    p = st.jax_params()
    p["layers"][0]["shapes"] = [Circle(0.6, 0.6, r, eps=4.0)]
    return st.solve(params=p)[2][0, p0]


dT_dr = jax.jit(jax.grad(t00))(0.36)


def loss(v):
    p = st.jax_params()
    p["layers"][0]["shapes"] = [Circle(0.6, 0.6, 0.36, eps=v[0] + 1j * v[1])]
    p["layers"][0]["thickness"] = v[2]
    p["n_substrate"] = v[3]
    return st.solve(params=p)[2][0, p0]


g = jax.jit(jax.grad(loss))(jnp.array([4.0, 0.0, 0.5, 1.45]))
out = {"T00_entry": float(T[0, 12]), "dT_dr": float(dT_dr),
       "grad_eps_re_im_depth_nsub": [float(x) for x in g]}
dump("f11_api_examples.json", out)
print(out)
