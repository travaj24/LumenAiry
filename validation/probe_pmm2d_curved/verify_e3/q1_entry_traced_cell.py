"""Q1: pmm_jones_2d_staggered(backend='jax') with a TRACED eps_cell / mu_cell:
is the traced value used (value equal to NumPy at that value; AD equal to the
NumPy FD), or does the uniform stand-in leak?"""
from _ve3 import WL, P, amax, dump, jax, jnp, ladder, np

from lumenairy.elements.pmm import pmm_jones_2d_staggered as PJ

out = {}
cell = np.array([[4.0, 1.0], [1.0, 1.0]], complex)
M = 3


def f_np(e):
    c = cell.copy()
    c[0, 0] = e
    return PJ(P, P, c, 1.45, 1.0, 0.4, WL, degree=M, n_orders=2)


def f_jx(e):
    c = jnp.asarray(cell).at[0, 0].set(e)
    return PJ(P, P, c, 1.45, 1.0, 0.4, WL, degree=M, n_orders=2,
              backend="jax")


# value with a TRACED (but numerically identical) cell, via jit
o, R, T, J = f_np(4.0)
Tj = jax.jit(lambda e: f_jx(e)[2])(4.0)
out["eps_cell_traced_value_vs_numpy"] = amax(Tj, T)
g = float(jax.grad(lambda e: f_jx(e)[2][0, 12].real)(4.0))
rows, rich, ch, rat = ladder(lambda e: f_np(e)[2][0, 12].real, 4.0,
                             [1e-3, 3e-4, 1e-4])
out["eps_cell_AD"] = g
out["eps_cell_FDnumpy"] = float(rich)
out["eps_cell_rel"] = abs(g - float(rich)) / abs(float(rich))

# magnetic: traced mu_cell
mcell = np.array([[1.5, 1.0], [1.0, 1.0]], complex)


def g_np(m):
    c = mcell.copy()
    c[0, 0] = m
    return PJ(P, P, cell, 1.45, 1.0, 0.4, WL, degree=M, n_orders=2,
              mu_cell=c)


def g_jx(m):
    c = jnp.asarray(mcell).at[0, 0].set(m)
    return PJ(P, P, cell, 1.45, 1.0, 0.4, WL, degree=M, n_orders=2,
              mu_cell=c, backend="jax")


o, R, T, J = g_np(1.5)
try:
    Tj = jax.jit(lambda m: g_jx(m)[2])(1.5)
    out["mu_cell_traced_value_vs_numpy"] = amax(Tj, T)
    g = float(jax.grad(lambda m: g_jx(m)[2][0, 12].real)(1.5))
    rows, rich, ch, rat = ladder(lambda m: g_np(m)[2][0, 12].real, 1.5,
                                 [1e-3, 3e-4, 1e-4])
    out["mu_cell_AD"] = g
    out["mu_cell_FDnumpy"] = float(rich)
    out["mu_cell_rel"] = abs(g - float(rich)) / abs(float(rich))
except Exception as exc:  # noqa: BLE001 -- probe records the failure
    out["mu_cell_error"] = repr(exc)[:400]
# all-ones traced mu (stand-in = ones; non-magnetic collapse?)
try:
    T1 = PJ(P, P, cell, 1.45, 1.0, 0.4, WL, degree=M, n_orders=2,
            mu_cell=mcell)[2]
    Tj = jax.jit(lambda s: PJ(P, P, cell, 1.45, 1.0, 0.4, WL, degree=M,
                              n_orders=2, mu_cell=jnp.asarray(mcell) * s,
                              backend="jax")[2])(1.0)
    out["mu_cell_traced_scale_value_vs_numpy"] = amax(Tj, T1)
except Exception as exc:  # noqa: BLE001
    out["mu_cell_scale_error"] = repr(exc)[:400]
print(out)
dump("q1_entry_traced_cell.json", out)
