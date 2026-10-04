"""Claim (1) with the VERIFIER's own fixtures: every routed family at a
symmetric point, AD (jit(jacrev)) vs a premise-checked Richardson FD of the
NumPy (or concrete) forward; rule ON (default) and OFF (the defect, as a
discriminator).  python p2_families.py <case> [<case> ...]"""
import json
import sys
import time

from _vc import BUILD, fd_rich, jax, jnp, np, premise_ok, rel

from lumenairy.backend import jax_cluster_rule

CASES = {}


def case(fn):
    CASES[fn.__name__] = fn
    return fn


# ---------------------------------------------------------------- RCWA
S = 24
CROSS = np.ones((S, S), complex)
CROSS[8:16, :] = 0.0
CROSS[:, 8:16] = 0.0
ARMX = np.zeros((S, S))
ARMX[8:16, 16:24] = 1.0         # the +x half-arm only: breaks C4v -> mirror y
RND = np.random.default_rng(11).uniform(-1, 1, (S, S))


def cross_eps(eps_a, t, xp, dirn=ARMX):
    base = xp.asarray(np.where(CROSS == 0.0, eps_a, 1.44 + 0j))
    return base + t * xp.asarray(dirn)


def _pack2(R, T, xp):
    return xp.concatenate([xp.ravel(R), xp.ravel(T)])


@case
def rcwa_jones_2d_cross_lossy():
    from lumenairy.elements.rcwa import rcwa_jones_2d

    def f(t, xp):
        e = cross_eps(6.25 + 0.8j, t, xp)
        tens = e[:, :, None, None] * xp.eye(3, dtype=complex)[None, None]
        _o, R, T, _J = rcwa_jones_2d(1.3, 1.3, tens, 1.5, 1.0, 0.3, 1.0,
                                     n_orders_x=3, n_orders_y=3)
        return _pack2(R, T, xp)
    return f, 0.0


@case
def rcwa_jones_2d_cross_li_uniaxial():
    # 'li' on a uniaxial (ezz != exx = eyy) symmetric cell: the traced li route
    from lumenairy.elements.rcwa import rcwa_jones_2d
    U = np.diag([1.0, 1.0, 1.2]).astype(complex)

    def f(t, xp):
        e = cross_eps(5.0 + 0j, t, xp)
        tens = e[:, :, None, None] * xp.asarray(U)[None, None]
        _o, R, T, _J = rcwa_jones_2d(1.3, 1.3, tens, 1.5, 1.0, 0.3, 1.0,
                                     n_orders_x=3, n_orders_y=3,
                                     formulation="li")
        return _pack2(R, T, xp)
    return f, 0.0


@case
def rcwa_efficiency_2d_cross_te_tm():
    from lumenairy.elements.rcwa import rcwa_efficiency_2d

    def f(t, xp):
        e = cross_eps(4.0 + 0.3j, t, xp)
        out = []
        for pol in ("te", "tm"):
            _o, R, T = rcwa_efficiency_2d(1.3, 1.3, e, 1.5, 1.0, 0.3, 1.0,
                                          polarization=pol, n_orders_x=3,
                                          n_orders_y=3)
            out += [R, T]
        return xp.concatenate([xp.ravel(a) for a in out])
    return f, 0.0


def _stack(layers, theta, phi, nox=5):
    from lumenairy.elements.rcwa import RCWAStack
    st = RCWAStack(1.3, period_y=1.3, n_superstrate=1.0, n_substrate=1.5,
                   n_orders=nox, n_orders_y=nox)
    for d, kw in layers:
        st.add_layer(d, **kw)
    st.set_source(1.0, theta=theta, phi=phi)
    _o, R, T = st.solve().efficiencies()
    return R, T


@case
def rcwastack_conical_uniform_cluster():
    # conical incidence (0.2, 0.3): a UNIFORM patterned layer (its TE/TM
    # pairs are degenerate at any angle) perturbed by a random pattern, over
    # a cross layer
    def f(t, xp):
        u = xp.asarray(np.full((S, S), 2.25 + 0.05j)) + t * xp.asarray(RND)
        R, T = _stack([(0.25, dict(eps_cell=u)),
                       (0.2, dict(eps_cell=cross_eps(4.0 + 0j, 0.0, xp)))],
                      0.2, 0.3, nox=3)
        return _pack2(R, T, xp)
    return f, 0.0


@case
def rcwastack_normal_uniform_8fold():
    # normal incidence, uniform layer: orders (+-1,0),(0,+-1) x 2 pols are an
    # 8-member cluster (and (+-1,+-1) another); a random pattern splits them
    def f(t, xp):
        u = xp.asarray(np.full((S, S), 3.0 + 0j)) + t * xp.asarray(RND)
        R, T = _stack([(0.3, dict(eps_cell=u))], 0.0, 0.0, nox=3)
        return _pack2(R, T, xp)
    return f, 0.0


@case
def rcwastack_two_patterned_cross():
    def f(t, xp):
        R, T = _stack([(0.2, dict(eps_cell=cross_eps(6.25 + 0j, t, xp))),
                       (0.1, dict(eps=2.1)),
                       (0.15, dict(eps_cell=cross_eps(3.0 + 0.2j, 0.5 * t,
                                                      xp)))],
                      0.0, 0.0, nox=3)
        return _pack2(R, T, xp)
    return f, 0.0


@case
def rcwastack_keep_symmetry():
    # a symmetry-KEEPING parameter: all four arms together (C4v kept)
    allarms = (CROSS == 0.0).astype(float)

    def f(t, xp):
        R, T = _stack([(0.2, dict(eps_cell=cross_eps(6.25 + 0j, t, xp,
                                                     dirn=allarms)))],
                      0.0, 0.0, nox=3)
        return _pack2(R, T, xp)
    return f, 0.0


# ------------------------------------------------------------ Berreman
def _berr(t, xp, angle, eps0=3.0 + 0.1j,
          dirn=((1, 0, 0.5), (0, -1, 0), (0.5, 0, 0))):
    from lumenairy.elements.berreman import berreman_jones_1d
    D = np.asarray(dirn, complex)
    e1 = eps0 * xp.eye(3, dtype=complex) + t * xp.asarray(D)
    e2 = 2.0 * xp.eye(3, dtype=complex) + 0.5 * t * xp.asarray(D.T)
    if xp is jnp:
        R, T, Jr, Jt = berreman_jones_1d(
            [(e1, jnp.asarray(0.25e-6)), (e2, jnp.asarray(0.15e-6))],
            jnp.asarray(1.5 + 0j), jnp.asarray(1.0 + 0j), jnp.asarray(0.9e-6),
            angle=angle)
    else:
        R, T, Jr, Jt = berreman_jones_1d([(e1, 0.25e-6), (e2, 0.15e-6)],
                                         1.5, 1.0, 0.9e-6, angle=angle)
    J = xp.concatenate([xp.ravel(Jr), xp.ravel(Jt)])
    return xp.concatenate([xp.ravel(R), xp.ravel(T), xp.real(J), xp.imag(J)])


@case
def berreman_two_iso_lossy_normal():
    return (lambda t, xp: _berr(t, xp, 0.0)), 0.0


@case
def berreman_two_iso_lossy_oblique():
    return (lambda t, xp: _berr(t, xp, 0.35)), 0.0


# ------------------------------------------------------------ 1-D PMM
ERJ = np.diag([5.0 + 0.2j, 4.0 + 0.2j, 4.5 + 0.2j])
EGJ = 1.2 * np.eye(3, dtype=complex)


@case
def pmm_jones_1d_lossy_aniso():
    from lumenairy.elements.pmm import pmm_jones_1d

    def f(t, xp):
        _o, R, T, J = pmm_jones_1d(0.9, ERJ, EGJ, 1.5, 1.0, 0.3, 0.4, 1.0,
                                   angle=t, degree=10, stabilize=False)
        J = xp.ravel(xp.asarray(J))
        return xp.concatenate([xp.ravel(xp.asarray(R)),
                               xp.ravel(xp.asarray(T)), xp.real(J),
                               xp.imag(J)])
    return f, 0.0


@case
def pmm_jones_1d_accidental_0p2():
    # the builder's fixture at 0.2 rad: an ACCIDENTAL Mbig pair 3.1e-7 of
    # the spectrum apart (build record 10.6), no symmetry; d / d angle there
    from lumenairy.elements.pmm import pmm_jones_1d
    ER, EG = 4.0 * np.eye(3, dtype=complex), np.eye(3, dtype=complex)

    def f(t, xp):
        _o, R, T, _J = pmm_jones_1d(1.2, ER, EG, 1.0, 1.45, 0.45, 0.5, 1.0,
                                    angle=t, degree=12, stabilize=False)
        return xp.concatenate([xp.ravel(xp.asarray(R)),
                               xp.ravel(xp.asarray(T))])
    return f, 0.2


@case
def pmm_efficiency_1d_lossy_te_tm():
    from lumenairy.elements.pmm import pmm_efficiency_1d

    def f(t, xp):
        out = []
        for pol in ("te", "tm"):
            _o, R, T = pmm_efficiency_1d(0.9, xp.asarray(5.0 + 0.2j), 1.2,
                                         1.5, 1.0, 0.3, 0.4, 1.0, angle=t,
                                         polarization=pol, degree=10,
                                         stabilize=False)
            out += [R, T]
        return xp.concatenate([xp.ravel(xp.asarray(a)) for a in out])
    return f, 0.0


def _pmmstack(grids):
    from lumenairy.elements.pmm import PMMStack

    def f(t, xp):
        st = PMMStack(1.0e-6, n_substrate=1.5, n_superstrate=1.0, degree=10,
                      layer_grids=grids)
        st.add_layer(0.25e-6, segments=[(0.5, 4.0), (0.5, 1.0)])
        st.add_layer(0.1e-6, segments=[(1.0, 2.1)])
        st.add_layer(0.2e-6, segments=[(0.15, 3.0), (0.2, 6.0 + 0.3j),
                                       (0.15, 3.0), (0.5, 1.5)])
        st.set_source(0.8e-6, angle=t)
        _o, R, T, J = st.solve()
        J = xp.ravel(xp.asarray(J))
        return xp.concatenate([xp.ravel(xp.asarray(R)),
                               xp.ravel(xp.asarray(T)), xp.real(J),
                               xp.imag(J)])
    return f, 0.0


@case
def pmmstack_shared_3layer_lossy():
    return _pmmstack("shared")


@case
def pmmstack_perlayer_3layer_lossy():
    return _pmmstack("per-layer")


# ------------------------------------------------------------ hybrid 2-D
@case
def hybrid_two_patterned_traced_layout():
    from lumenairy.elements.pmm import PMM2DStackHybrid
    S8 = 8
    c1 = np.full((S8, S8), 1.0 + 0j)
    c1[0:4, 0:4] = 3.0
    c2 = np.full((S8, S8), 2.0 + 0j)
    c2[1:3, 1:3] = 4.0 + 0.2j
    lay1 = np.zeros((S8, S8), np.int64)
    for i in range(4):
        for j in range(4):
            lay1[i, j] = 1 + 4 * i + j
    lay2 = np.zeros((S8, S8), np.int64)
    for i in range(1, 3):
        for j in range(1, 3):
            lay2[i, j] = 1 + 2 * (i - 1) + (j - 1)

    def solve(t):
        st = PMM2DStackHybrid(1.1e-6, n_substrate=1.5, n_superstrate=1.0,
                              degree=7, n_orders=3, formulation="laurent")
        st.add_layer(0.25e-6, eps_cell=jnp.asarray(c1).at[0, 0].add(t),
                     region_layout=lay1)
        st.add_layer(0.1e-6, eps=2.1)
        st.add_layer(0.15e-6, eps_cell=jnp.asarray(c2).at[1, 1].add(0.7 * t),
                     region_layout=lay2)
        st.set_source(0.95e-6, theta=0.0)
        _o, R, T, J = st.solve()
        J = jnp.ravel(J)
        return jnp.concatenate([jnp.ravel(R), jnp.ravel(T), jnp.real(J),
                                jnp.imag(J)])
    fwd = jax.jit(solve)

    def f(t, xp):
        if xp is np:
            return np.asarray(fwd(jnp.asarray(t)))
        return solve(t)
    return f, 0.0, 10.0


def run(name):
    spec = CASES[name]()
    f, x0 = spec[0], spec[1]
    scale = spec[2] if len(spec) > 2 else 1.0
    t0 = time.perf_counter()
    fd, ratio, ex = fd_rich(lambda t: f(t, np), x0, scale=scale)
    t1 = time.perf_counter()
    res = {}
    for on in (True, False):
        with jax_cluster_rule(on):
            g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(x0))
        res[on] = g
    t2 = time.perf_counter()
    big = np.abs(fd) >= 1e-3 * np.max(np.abs(fd))
    r = dict(case=name, x0=x0, err_on=rel(res[True], fd),
             err_off=rel(res[False], fd), premise=premise_ok(ratio, ex),
             ratio=[float(ratio.min()), float(ratio.max()),
                    float(np.median(ratio))], n_big=int(big.sum()),
             gmax=float(np.max(np.abs(fd))), t_fd=t1 - t0, t_ad=t2 - t1)
    print(r, flush=True)
    return r


if __name__ == "__main__":
    names = sys.argv[1:] or list(CASES)
    rows = [run(n) for n in names]
    tag = names[0] if len(names) == 1 else "all"
    json.dump({"build": BUILD, "rows": rows},
              open(f"p2_families_{tag}_{BUILD}.json", "w"), indent=1)
