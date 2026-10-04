"""H: FORWARD BYTES of the round-2 routed twins, PRE vs POST.

SHA-256 of the forward outputs, one fixture per routed family, NumPy eager
(where the entry accepts NumPy), JAX eager and JAX under ``jax.jit`` with the
fixture parameter TRACED; ``SG_TAG=pre`` on the base tree, ``SG_TAG=r2post``
on the fixed one, compared by ``h_compare.py``.

  berreman_exy_d{0,1e-3}  d4 stack (eps 2.56 I + d exy | 2.1), normal inc.
      numpy          : NumPy berreman_jones_1d
      jax_eager      : the routed ``_offplane_solve_jax`` called eagerly
                       (the off-plane solve a TRACED tensor takes)
      jax_eager_api  : public entry on a concrete jnp tensor (in-plane ->
                       ``_solve_jax``, the unrouted control)
      jax_jit        : public entry under jit with d traced (-> off-plane)
  pmmjones1d_{0,0.2}      d5 fixture, angle traced (jit) / jnp (eager)
  pmmstack_{shared_2L,shared_spacer,perlayer_3L}  d6 fixtures, angle 0
  stack2d_traced_{laurent,li}  d7 traced-layout corner cell, delta 0
      (no NumPy: a NumPy cell cannot carry the region layout)

    python h_fwd_bytes.py
"""
import hashlib
import traceback

from _h import dump, jax, jnp, np

from lumenairy.elements.berreman import berreman_jones_1d
from lumenairy.elements.pmm import PMM2DStackHybrid, PMMStack, pmm_jones_1d


def sha(*objs):
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(objs):
        h.update(np.ascontiguousarray(np.asarray(leaf)).tobytes())
    return h.hexdigest()


def flat(*objs):
    return np.concatenate([np.ravel(np.asarray(x, dtype=complex))
                           for x in jax.tree_util.tree_leaves(objs)])


# ---------------------------------------------------------------- Berreman
WL, T1, T2, E0 = 1e-6, 0.3e-6, 0.1e-6, 2.56
XEXY = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], complex)


def bm_np(d):
    return berreman_jones_1d([(E0 * np.eye(3) + d * XEXY, T1), (2.1, T2)],
                             1.0, 1.45, WL)


def bm_layers_j(d):
    e = E0 * jnp.eye(3, dtype=complex) + d * jnp.asarray(XEXY)
    return [(e, jnp.asarray(T1)), (jnp.asarray(2.1 + 0j), jnp.asarray(T2))]


def bm_api(d):
    return berreman_jones_1d(bm_layers_j(d), jnp.asarray(1.0 + 0j),
                             jnp.asarray(1.45 + 0j), jnp.asarray(WL))


def bm_offplane(d):
    from lumenairy.elements._berreman_jax import _offplane_solve_jax, _prep
    j, el, th, es, eb, wl, Kx, Ky = _prep(
        bm_layers_j(d), jnp.asarray(1.0 + 0j), jnp.asarray(1.45 + 0j),
        jnp.asarray(WL), 0.0, 0.0, None)
    return _offplane_solve_jax(el, th, es, eb, jnp.asarray(wl), Kx, Ky, j)


def bm_fixture(d):
    def run():
        rec = {}
        o = bm_np(d)
        rec["numpy"] = sha(o)
        ref = flat(o)
        rec["jax_eager"] = sha(bm_offplane(jnp.asarray(d)))
        rec["jax_eager_api"] = sha(bm_api(jnp.asarray(d)))
        jo = jax.jit(bm_api)(jnp.asarray(d))
        rec["jax_jit"] = sha(jo)
        rec["max_abs_jit_minus_numpy"] = float(np.max(np.abs(flat(jo) - ref)))
        return rec
    return run


# ---------------------------------------------------------- pmm_jones_1d
P1, WL1 = 1.2, 1.0
ER, EG = 4.0 * np.eye(3, dtype=complex), np.eye(3, dtype=complex)


def pj(angle):
    return pmm_jones_1d(P1, ER, EG, 1.0, 1.45, 0.45, 0.5, WL1, angle=angle,
                        degree=12, stabilize=False)


def pj_fixture(a):
    def run():
        o, R, T, J = pj(float(a))
        rec = {"numpy": sha(o, R, T, J)}
        ref = flat(R, T, J)
        _o, R, T, J = pj(jnp.asarray(a))
        rec["jax_eager"] = sha(R, T, J)
        jo = jax.jit(lambda t: pj(t)[1:])(jnp.asarray(a))
        rec["jax_jit"] = sha(jo)
        rec["max_abs_jit_minus_numpy"] = float(np.max(np.abs(flat(jo) - ref)))
        return rec
    return run


# ------------------------------------------------------------- PMMStack
P6, WL6 = 1.2e-6, 1.0e-6
L1 = [(0.5, 4.0), (0.5, 1.0)]
L2 = [(0.1, 1.0), (0.3, 2.25), (0.6, 1.0)]
L3 = [(0.2, 1.0), (0.1, 4.0), (0.7, 1.0)]
CFG6 = {"shared_2L": ("shared", [(0.3e-6, L1), (0.15e-6, L2)]),
        "shared_spacer": ("shared", [(0.3e-6, L1), (0.1e-6, [(1.0, 2.25)])]),
        "perlayer_3L": ("per-layer", [(0.3e-6, L1), (0.15e-6, L2),
                                      (0.1e-6, L3)])}


def ps(grids, layers, angle):
    st = PMMStack(P6, n_substrate=1.0, n_superstrate=1.45, degree=12,
                  layer_grids=grids)
    for t, segs in layers:
        st.add_layer(t, segments=segs)
    st.set_source(WL6, angle=angle)
    return st.solve()


def ps_fixture(grids, layers):
    def run():
        o, R, T, J = ps(grids, layers, 0.0)
        rec = {"numpy": sha(o, R, T, J)}
        ref = flat(R, T, J)
        _o, R, T, J = ps(grids, layers, jnp.asarray(0.0))
        rec["jax_eager"] = sha(R, T, J)
        jo = jax.jit(lambda a: ps(grids, layers, a)[1:])(jnp.asarray(0.0))
        rec["jax_jit"] = sha(jo)
        rec["max_abs_jit_minus_numpy"] = float(np.max(np.abs(flat(jo) - ref)))
        return rec
    return run


# ------------------------------------------------------ hybrid 2-D stack
S = 6
C4 = np.full((S, S), 1.0 + 0j)
C4[0:3, 0:3] = 4.0
LAY = np.zeros((S, S), dtype=np.int64)
for _i in range(3):
    for _j in range(3):
        LAY[_i, _j] = 1 + 3 * _i + _j


def s2(delta, form):
    cell = jnp.asarray(C4).at[0, 0].add(delta)
    st = PMM2DStackHybrid(P6, n_substrate=1.0, n_superstrate=1.45, degree=7,
                          n_orders=3, formulation=form)
    st.add_layer(0.3e-6, eps_cell=cell, region_layout=LAY)
    st.add_layer(0.08e-6, eps=2.25)
    st.set_source(WL6, theta=0.0)
    return st.solve()


def s2_fixture(form):
    def run():
        _o, R, T, J = s2(jnp.asarray(0.0), form)
        rec = {"jax_eager": sha(R, T, J)}
        ref = flat(R, T, J)
        jo = jax.jit(lambda d: s2(d, form)[1:])(jnp.asarray(0.0))
        rec["jax_jit"] = sha(jo)
        rec["max_abs_jit_minus_eager"] = float(np.max(np.abs(flat(jo) - ref)))
        return rec
    return run


FIX = {"berreman_exy_d0": bm_fixture(0.0),
       "berreman_exy_d1e-3": bm_fixture(1e-3),
       "pmmjones1d_0": pj_fixture(0.0),
       "pmmjones1d_0.2": pj_fixture(0.2)}
for _k, (_g, _l) in CFG6.items():
    FIX[f"pmmstack_{_k}"] = ps_fixture(_g, _l)
for _f in ("laurent", "li"):
    FIX[f"stack2d_traced_{_f}"] = s2_fixture(_f)

out = {}
for name, run in FIX.items():
    try:
        out[name] = run()
    except Exception as e:          # recorded, not hidden
        out[name] = {"error": repr(e),
                     "traceback": traceback.format_exc()[-2000:]}
    print(name, out[name], flush=True)
print(dump("h_fwd_bytes.json", out))
