"""Does 're-create the jitted function after changing the switch' (the
setter's docstring) take effect?  jax.jit caches compiled programs keyed by
the wrapped Python callable: a NEW jax.jit wrapper of the SAME callable
reuses the program traced under the OLD setting."""
import json

from _vc import BUILD, fd_rich, jax, jnp, np, rel

from lumenairy.backend import jax_cluster_rule
from lumenairy.elements.berreman import berreman_jones_1d

X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], complex)


def berr(d, xp):
    e = (3.0 + 0.1j) * xp.eye(3, dtype=complex) + d * xp.asarray(X)
    if xp is jnp:
        R, T, Jr, _ = berreman_jones_1d(
            [(e, jnp.asarray(0.25e-6))], jnp.asarray(1.5 + 0j),
            jnp.asarray(1.0 + 0j), jnp.asarray(0.9e-6), angle=0.3)
    else:
        R, T, Jr, _ = berreman_jones_1d([(e, 0.25e-6)], 1.5, 1.0, 0.9e-6,
                                        angle=0.3)
    Jr = xp.ravel(Jr)
    return xp.concatenate([xp.ravel(R), xp.ravel(T), xp.real(Jr),
                           xp.imag(Jr)])


fd = fd_rich(lambda d: berr(d, np), 0.0)[0]
out = {"build": BUILD}
for first, second in ((False, True), (True, False)):
    g = jax.jacrev(lambda d: berr(d, jnp))      # ONE callable
    with jax_cluster_rule(first):
        e1 = rel(np.asarray(jax.jit(g)(0.0)), fd)
    with jax_cluster_rule(second):
        e2 = rel(np.asarray(jax.jit(g)(0.0)), fd)      # a NEW jit wrapper
        e3 = rel(np.asarray(jax.jit(jax.jacrev(lambda d: berr(d, jnp)))(
            0.0)), fd)                                  # a new callable
    out[f"{first}->{second}"] = dict(first=e1, new_wrapper_same_callable=e2,
                                     new_callable=e3)
    print(first, "->", second, out[f"{first}->{second}"], flush=True)
json.dump(out, open(f"p11_jit_recreate_{BUILD}.json", "w"), indent=1)
