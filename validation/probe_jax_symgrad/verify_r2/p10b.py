"""Timing / premise of candidate (smaller) test fixtures: PMMStack 2 layers
degree 8 (shared, per-layer), hybrid two patterned layers degree 5, and the
PMMStack near a Rayleigh anomaly (does the p1 defect reach the stack?)."""
import time

from _vc import fd_rich, jax, jnp, np, premise_ok, rel

from lumenairy.backend import jax_cluster_rule
from lumenairy.elements.pmm import PMMStack


def pmmstack(grids, wl=0.8e-6, period=1.0e-6, nsub=1.5, deg=8, pack="all",
             nsup=1.0):
    IDX = []

    def f(t, xp):
        st = PMMStack(period, n_substrate=nsub, n_superstrate=nsup,
                      degree=deg, layer_grids=grids)
        st.add_layer(0.2e-6, segments=[(0.15, 3.0), (0.2, 6.0 + 0.3j),
                                       (0.15, 3.0), (0.5, 1.5)])
        st.add_layer(0.1e-6, segments=[(1.0, 2.1)])
        st.set_source(wl, angle=t)
        o, R, T, J = st.solve()
        if pack == "pm1":
            if not IDX:
                oc = np.asarray(o)
                IDX.extend(int(np.nonzero(oc == m)[0][0]) for m in (1, -1))
            idx = IDX
            return xp.concatenate([xp.stack([xp.asarray(R)[p][i]
                                             for p in (0, 1) for i in idx]),
                                   xp.stack([xp.asarray(T)[p][i]
                                             for p in (0, 1) for i in idx])])
        J = xp.ravel(xp.asarray(J))
        return xp.concatenate([xp.ravel(xp.asarray(R)),
                               xp.ravel(xp.asarray(T)), xp.real(J),
                               xp.imag(J)])
    return f


def run(name, f, x0=0.0, hs=(1e-3, 3e-4, 1e-4), floor=1e-3, scale=1.0):
    t0 = time.perf_counter()
    fd, ratio, ex = fd_rich(lambda t: f(t, np), x0, hs=hs, floor=floor,
                            scale=scale)
    t1 = time.perf_counter()
    g = {}
    for on in (True, False):
        with jax_cluster_rule(on):
            g[on] = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(x0))
    t2 = time.perf_counter()
    mir = float(np.max(np.abs(g[True][0::2] + g[True][1::2]))
                / np.max(np.abs(g[True])))
    mir_fd = float(np.max(np.abs(fd[0::2] + fd[1::2])) / np.max(np.abs(fd)))
    print(dict(case=name, on=rel(g[True], fd), off=rel(g[False], fd),
               premise=premise_ok(ratio, ex), rmin=float(ratio.min()),
               rmax=float(ratio.max()), t_fd=round(t1 - t0, 1),
               t_ad=round(t2 - t1, 1), mirror_ad=mir, mirror_fd=mir_fd),
          flush=True)


# Rayleigh anomaly of the m=1 order in a substrate n=1.0: wl = P*(1+1e-4)
for d in (1e-3, 1e-4):
    run(f"pmmstack_shared_rayleigh_{d}",
        pmmstack("shared", wl=1.0e-6 * (1 + d), nsub=1.0, nsup=1.45, deg=10,
                 pack="pm1"), hs=tuple(min(1e-3, d * 1e-2) * s
                                       for s in (1.0, 0.3, 0.1)))

