"""B2: JAX forward bytes of the two ROUTED entries (eager and jax.jit),
PRE vs POST: rcwa_efficiency_2d on the verifier fixtures (C4v TE / TM /
li, lossy, conical C1, weak near-degenerate C1) and pmm_efficiency_1d
(traced angle 0 / 0.2, TE / TM).  Run once per tree with LUM_TREE / VTAG."""
import hashlib

from _fix import BASE, RAND, post
from _v import dump, jax, jnp, np

from lumenairy.elements.pmm import pmm_efficiency_1d
from lumenairy.elements.rcwa import rcwa_efficiency_2d


def h(obj):
    m = hashlib.sha256()
    for a in obj:
        a = np.asarray(a)
        m.update(str(a.shape).encode())
        m.update(np.ascontiguousarray(a).tobytes())
    return m.hexdigest()


out = {}
for name, cell, kw in [
        ("c4v_te", BASE, {}), ("c4v_tm", BASE, dict(polarization="tm")),
        ("c4v_li_tm", BASE, dict(formulation="li", polarization="tm")),
        ("lossy_te", BASE + 0.8j * post, {}),
        ("c1_conical_tm", 2.25 + 0.3 * RAND, dict(polarization="tm",
                                                  theta=0.2, phi=0.3)),
        ("c1weak_conical_te", 2.25 + 1e-2 * RAND, dict(theta=0.2, phi=0.3))]:
    c = jnp.asarray(cell.astype(complex))
    out["rcwa2d_eager_" + name] = h(rcwa_efficiency_2d(
        0.9, 0.9, c, 1.52, 1.0, 0.37, 1.0, n_orders_x=2, n_orders_y=2,
        **kw)[1:])
    out["rcwa2d_jit_" + name] = h(jax.jit(lambda c, kw=kw: rcwa_efficiency_2d(
        0.9, 0.9, c, 1.52, 1.0, 0.37, 1.0, n_orders_x=2, n_orders_y=2,
        **kw)[1:])(c))
for pol in ("te", "tm"):
    for ang in (0.0, 0.2):
        def f(a, pol=pol):
            return pmm_efficiency_1d(0.85, jnp.asarray(2.3 + 0j), 1.35, 1.6,
                                     1.0, 0.31, 0.4, 1.0, angle=a,
                                     polarization=pol, degree=10,
                                     stabilize=False)[1:]
        out[f"pmm1d_eager_{pol}_{ang}"] = h(f(jnp.asarray(ang)))
        out[f"pmm1d_jit_{pol}_{ang}"] = h(jax.jit(f)(jnp.asarray(ang)))
for k, v in out.items():
    print(k, v[:16])
dump("b2_bytes_jax", out)
