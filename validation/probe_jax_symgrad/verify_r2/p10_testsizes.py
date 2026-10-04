"""Timing / premise of candidate (smaller) test fixtures: PMMStack 2 layers
degree 8 (shared, per-layer), hybrid two patterned layers degree 5, and the
PMMStack near a Rayleigh anomaly (does the p1 defect reach the stack?)."""
import time

from _vc import fd_rich, jax, jnp, np, premise_ok, rel

from lumenairy.backend import jax_cluster_rule
from lumenairy.elements.pmm import PMM2DStackHybrid, PMMStack


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

S8 = 8
C1 = np.full((S8, S8), 1.0 + 0j)
C1[0:4, 0:4] = 3.0
C2 = np.full((S8, S8), 2.0 + 0j)
C2[1:3, 1:3] = 4.0 + 0.2j
LAY1 = np.zeros((S8, S8), np.int64)
for i in range(4):
    for j in range(4):
        LAY1[i, j] = 1 + 4 * i + j
LAY2 = np.zeros((S8, S8), np.int64)
for i in range(1, 3):
    for j in range(1, 3):
        LAY2[i, j] = 1 + 2 * (i - 1) + (j - 1)


def hyb(t):
    st = PMM2DStackHybrid(1.1e-6, n_substrate=1.5, n_superstrate=1.0,
                          degree=5, n_orders=3, formulation="laurent")
    st.add_layer(0.25e-6, eps_cell=jnp.asarray(C1).at[0, 0].add(t),
                 region_layout=LAY1)
    st.add_layer(0.15e-6, eps_cell=jnp.asarray(C2).at[1, 1].add(0.7 * t),
                 region_layout=LAY2)
    st.set_source(0.95e-6, theta=0.0)
    _o, R, T, J = st.solve()
    J = jnp.ravel(J)
    return jnp.concatenate([jnp.ravel(R), jnp.ravel(T), jnp.real(J),
                            jnp.imag(J)])


fwd = jax.jit(hyb)
run("hybrid_two_patterned_deg5",
    lambda t, xp: np.asarray(fwd(jnp.asarray(t))) if xp is np else hyb(t),
    hs=(1e-2, 3e-3, 1e-3), floor=1e-2)
