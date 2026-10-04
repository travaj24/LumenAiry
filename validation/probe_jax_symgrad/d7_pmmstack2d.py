"""D7: the hybrid 2-D PMM STACK JAX twin (pmm/_jax_stack2d.py,
``_pmm_stack2d_solve_jax``) -- ``PMM2DStackHybrid.solve`` with a traced
theta, d / d(theta) at EXACTLY normal incidence on symmetric cells.

Eig site ~229 (``_modes_projected``: ``eig(P @ Q)`` of a patterned layer in
the projected Fourier basis); half-spaces and uniform layers are analytic
(``_homog``, no eig).

Stack: P 1.2 um square, 6 x 6 pixel cell, t 0.3 um; then a uniform layer
eps 2.25, t 0.08 um; n_sup 1.45, n_sub 1.0; wl 1 um; degree 7, n_orders 3.
Cells (walls at 0 and P/2 on a patterned axis, so the strip MESH is as
symmetric as the cell; an off-centre inclusion such as [1:4, 1:4] gives
strips 1/6, 3/6, 2/6 whose mesh is NOT mirror symmetric -- no exact
cluster and the mirror identity broken by the discretization at 5.6e-2):
  c4v_li       eps 4 on [0:3, 0:3] (C4v), formulation 'li' (default), phi 0
  c4v_laurent  the same cell, formulation 'laurent', phi 0
  1dasym_phi90 eps 4 on rows [0:2, :], 2.25 on row 2 (an x-ASYMMETRIC
               profile, uniform along y), 'li', phi = pi/2 (conical: theta
               moves ky0, which splits the +-n_y pairs at FIRST order; the
               y-mirror makes R/T of the x-orders even in theta, the
               cross-polarized zeroth-order Jones entries odd -- nonzero
               because no x-mirror pins them to 0.  With a SYMMETRIC x
               profile every output is even in theta at phi = pi/2.)
  1dx_phi0     eps 4 on rows [0:3, :] (symmetric, uniform along y), 'li',
               phi 0 (kx0 does not split the +-n_y pairs)
  traced_corner_{laurent,li}
               the C4v cell as a TRACED eps_cell with a 9-region layout on
               the inclusion (C4v-symmetric walls), d / d(eps of the corner
               pixel region) at theta 0 -- symmetry-breaking (C4v -> the
               diagonal mirror); FD oracle = concrete forward of the twin.
Outputs: R and T of the (+-1, 0) orders for incident Ex and Ey (8) + Re/Im
of the zeroth-order reflection Jones (8) = 16; mirror identity (phi 0)
d R_(+1,0) / d theta = - d R_(-1,0) / d theta (phi pi/2: both are 0).
Control: d / d(t) at theta 0.
"""
from _dcommon import capture, gauge, parity, sweep
from _h import dump, jax, jnp, np

from lumenairy.elements.pmm import PMM2DStackHybrid

P, WL, S = 1.2e-6, 1.0e-6, 6
C4 = np.full((S, S), 1.0 + 0j)
C4[0:3, 0:3] = 4.0
C1 = np.full((S, S), 1.0 + 0j)
C1[0:3, :] = 4.0
CA = np.full((S, S), 1.0 + 0j)      # x-ASYMMETRIC profile, uniform along y
CA[0:2, :] = 4.0
CA[2:3, :] = 2.25
CFG = {"c4v_li": (C4, "li", 0.0),
       "c4v_laurent": (C4, "laurent", 0.0),
       "1dasym_phi90": (CA, "li", np.pi / 2),
       "1dx_phi0": (C1, "li", 0.0)}
MIRROR = [(0, 1), (2, 3), (4, 5), (6, 7)]


def solve(cell, form, phi, theta, t1=0.3e-6):
    st = PMM2DStackHybrid(P, n_substrate=1.0, n_superstrate=1.45, degree=7,
                          n_orders=3, formulation=form)
    st.add_layer(t1, eps_cell=cell)
    st.add_layer(0.08e-6, eps=2.25)
    st.set_source(WL, theta=theta, phi=phi)
    return st.solve()


def pair_idx(cell, form, phi):
    o = np.asarray(solve(cell, form, phi, 0.0)[0])
    return [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == 0))[0][0])
            for m in (1, -1)]


def pack(R, T, J, idx, xp):
    J = xp.ravel(J)
    return xp.concatenate([xp.stack([R[p][i] for p in (0, 1) for i in idx]),
                           xp.stack([T[p][i] for p in (0, 1) for i in idx]),
                           xp.real(J), xp.imag(J)])


def f_theta(cfg, idx, xp):
    def f(a):
        _o, R, T, J = solve(*cfg, a if xp is jnp else float(a))
        return pack(R, T, J, idx, xp)
    return f


def f_t1(cfg, idx, xp):
    def f(t):
        _o, R, T, J = solve(*cfg, jnp.asarray(0.0) if xp is jnp else 0.0,
                            t1=(t if xp is jnp else float(t)))
        return pack(R, T, J, idx, xp)
    return f


out = {"spectrum": {}, "sweep": {}}
for name, cfg in CFG.items():
    idx = pair_idx(*cfg)
    for a in (0.0, 1e-5):
        sp = capture(lambda a=a, cfg=cfg, idx=idx: jax.jit(
            f_theta(cfg, idx, jnp))(jnp.asarray(a)))
        out["spectrum"][f"{name}_{a!r}"] = sp
        print("spectrum", name, a, [(r["n"], round(r["max_abs"], 3),
                                     "%.1e" % r["min_rel_gap"],
                                     r["members_below_1e-12"],
                                     r["members_below_1e-8"]) for r in sp],
              flush=True)
    fj, fn = f_theta(cfg, idx, jnp), f_theta(cfg, idx, np)
    rec = {"idx": idx, "parity": parity(fj, fn, 0.0)}
    rec["sweep"] = sweep(fj, fn, (0.0, 1e-5, 1e-3), mirror_pairs=MIRROR,
                         label=name + " theta")
    rec["gauge"] = gauge(fj, 0.0)
    print("parity", rec["parity"], "gauge", rec["gauge"], flush=True)
    fj, fn = f_t1(cfg, idx, jnp), f_t1(cfg, idx, np)
    rec["t1_sweep"] = sweep(fj, fn, (0.3e-6,), scale=0.3e-6,
                            label=name + " t1")
    rec["t1_gauge"] = gauge(fj, 0.3e-6)
    print("t1 gauge", rec["t1_gauge"], flush=True)
    out["sweep"][name] = rec

# ---- TRACED region cell: a C4v-symmetric MESH and cell, one corner pixel's
# region differentiated (breaks C4v down to the diagonal mirror).  The
# region layout puts walls at 0, 1, 2, 3 (x 1/6 P) on both axes, which is
# mirror / C4v symmetric about the inclusion centre (pixel 1.5).  The FD
# oracle is the CONCRETE forward of the same traced-branch twin (jit at
# delta +- h): a NumPy cell cannot carry the region layout (its walls would
# come from the pixel values, a different mesh).
LAY = np.zeros((S, S), dtype=np.int64)
for i in range(3):
    for j in range(3):
        LAY[i, j] = 1 + 3 * i + j


def solve_traced(delta, form, theta=0.0):
    cell = jnp.asarray(C4).at[0, 0].add(delta)
    st = PMM2DStackHybrid(P, n_substrate=1.0, n_superstrate=1.45, degree=7,
                          n_orders=3, formulation=form)
    st.add_layer(0.3e-6, eps_cell=cell, region_layout=LAY)
    st.add_layer(0.08e-6, eps=2.25)
    st.set_source(WL, theta=theta)
    return st.solve()


idx = pair_idx(C4, "laurent", 0.0)
for form in ("laurent", "li"):
    name = f"traced_corner_{form}"

    def fj(d, form=form):
        _o, R, T, J = solve_traced(d, form)
        return pack(R, T, J, idx, jnp)
    fwd = jax.jit(fj)

    def fn(d, fwd=fwd):
        return np.asarray(fwd(jnp.asarray(d)))
    for d in (0.0, 1e-4):
        # a FRESH jit per capture (a cached trace would keep the callback of
        # the first capture)
        sp = capture(lambda d=d, form=form: jax.jit(lambda t: fj(t, form))(
            jnp.asarray(d)))
        out["spectrum"][f"{name}_{d!r}"] = sp
        print("spectrum", name, d, [(r["n"], round(r["max_abs"], 3),
                                     "%.1e" % r["min_rel_gap"],
                                     r["members_below_1e-12"],
                                     r["members_below_1e-8"]) for r in sp],
              flush=True)
    rec = {"idx": idx}
    # h = 1e-2, 3e-3, 1e-3 (scale 10): at 1e-3..1e-4 the rung changes of
    # this nearly-linear response sit at the forward noise floor
    rec["sweep"] = sweep(fj, fn, (0.0, 1e-4, 1e-2), scale=10.0,
                         label=name + " delta")
    rec["gauge"] = gauge(fj, 0.0)
    print("gauge", rec["gauge"], flush=True)
    out["sweep"][name] = rec
print(dump("d7_pmmstack2d.json", out))
