"""D2: rcwa_jones_2d formulation='li' on JAX input: forward parity with
NumPy, and AD vs an FD of the JAX EAGER forward itself (so the oracle is
the same code path), at the C4v cell (t = 0) and off it (t = 1e-4, 0.05),
plus a C1 cell (no symmetry at all)."""
import _fix
from _v import dump, fd_halving, jax, jnp, np, rel

from lumenairy.elements.rcwa import rcwa_jones_2d

iso_cell = (_fix.BASE[..., None, None] * np.eye(3)).astype(complex)
c1_cell = ((2.25 + 0.6 * _fix.RAND)[..., None, None] * np.eye(3)).astype(
    complex)
DX = (_fix.D_X[..., None, None] * np.eye(3)).astype(complex)
DC = (_fix.D_CORNER[..., None, None] * np.eye(3)).astype(complex)
out = {}
for cname, cell0, D in (("c4v_xwiden", iso_cell, DX), ("c1_corner", c1_cell,
                                                        DC)):
    for form in ("laurent", "li"):
        def fr(t, xp):
            cell = xp.asarray(cell0) + t * xp.asarray(D)
            extra = {} if xp is jnp else {"symmetry": False}
            o, R, T, J = rcwa_jones_2d(0.9, 0.9, cell, 1.52, 1.0, 0.37, 1.0,
                                       n_orders_x=2, n_orders_y=2,
                                       formulation=form, **extra)
            return xp.concatenate([xp.ravel(xp.asarray(R)),
                                   xp.ravel(xp.asarray(T))]).real
        for x0 in (0.0, 1e-4, 0.05):
            fwd = rel(np.asarray(fr(x0, jnp)), fr(x0, np))
            g = np.asarray(jax.jit(jax.jacrev(lambda t: fr(t, jnp)))(x0))
            fdn, pn = fd_halving(lambda t: fr(t, np), x0)
            fdj, pj = fd_halving(lambda t: np.asarray(fr(jnp.asarray(t), jnp)),
                                 x0)
            rec = {"fwd_jax_vs_numpy": fwd, "ad_vs_fd_numpy": rel(g, fdn),
                   "ad_vs_fd_jax": rel(g, fdj), "fd_jax_vs_fd_numpy":
                   rel(fdj, fdn), "premise_numpy": pn, "premise_jax": pj}
            out[f"{cname}_{form}_{x0:g}"] = rec
            print(cname, form, x0, {k: (f"{v:.3g}" if isinstance(v, float)
                                        else v) for k, v in rec.items()},
                  flush=True)
dump("d2_jones2d_li", out)
