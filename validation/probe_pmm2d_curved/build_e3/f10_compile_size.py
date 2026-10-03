"""F-E3-2: the traced size of a twin solve and its compile times.

    python f10_compile_size.py M [jit]

Counts the jaxpr equations of the rectangle's R00(w) (traced width, M) per
stage -- the traced-map Jacobian at the nodes, the node weights, one shadow
assembly, the cofactor far field, the fold guard, the whole solve -- and,
with 'jit', the first-call (compile + run) times of the forward and of the
gradient; plus the compile time of single LAPACK kernels on a 162 x 162
complex matrix (the floor every solve / inverse / eig in the graph pays).
Output f10_compile_size_M<M>.json.
"""
import sys
import time
from collections import Counter

import numpy as np
from _e3common import TS, StagJaxTwin, dump, jax, jnp, rect_stack, rect_traced_map, tic

from lumenairy.elements.pmm._jax_twod_staggered import _min_detj, _shadow
from lumenairy.elements.rcwa import _jax_eig_stable

M = int(sys.argv[1])
JIT = len(sys.argv) > 2 and sys.argv[2] == "jit"
tw = StagJaxTwin(rect_stack(M), geometry="mapped")
ref = tw.sol_h
p0 = tw.p0


def mk(x):
    return rect_traced_map(tw.cmap_ref, x)


def n(fun):
    return len(jax.make_jaxpr(fun)(0.5).jaxpr.eqns)


def full(x):
    return tw.solve(cmap=mk(x))[1][0, p0]


cells = jnp.ones((3, 3)) * (1 + 0j)
out = {"M": M, "eqns": {
    "jacobian": n(lambda x: TS._stag_map_node_jacobian(
        ref.bx, ref.by, mk(x), ref._qrule, ref._qrule.tensor[0], jnp)[0][0]),
    "weights": n(lambda x: TS._stag_map_weights(
        ref.bx, ref.by, mk(x), cells, ref._qrule, xp=jnp)["e11"].t),
    "shadow_assembly": n(lambda x: _shadow(ref, jnp, cells, None,
                                           mk(x)).Lmat),
    "far_field": n(lambda x: TS._far_projector_mapped(
        tw.bx, tw.by, tw.ox, tw.oy, tw.a0x, tw.a0y, mk(x), xp=jnp,
        nq_cells=tw.far_nq)[0]),
    "fold_guard": n(lambda x: _min_detj(ref, mk(x), jnp)),
    "full_solve": n(full)}}
out["full_solve_primitives"] = Counter(
    e.primitive.name for e in jax.make_jaxpr(full)(0.5).jaxpr.eqns
).most_common(12)
if JIT:
    t = tic()
    float(jax.jit(full)(0.5))
    out["jit_forward_first_s"] = tic() - t
    t = tic()
    float(jax.jit(jax.grad(full))(0.5))
    out["jit_grad_first_s"] = tic() - t
A = np.random.default_rng(0).random((162, 162)) * (1 + 1j)
lap = {}
for nm, f in (("eig", lambda a: jnp.sum(jnp.abs(jnp.linalg.eig(a)[0]))),
              ("solve", lambda a: jnp.sum(jnp.abs(jnp.linalg.solve(a, a)))),
              ("inv", lambda a: jnp.sum(jnp.abs(jnp.linalg.inv(a)))),
              ("eig_stable_grad", jax.grad(
                  lambda a: jnp.real(jnp.sum(_jax_eig_stable()(a)[0]))))):
    t = time.perf_counter()
    jax.jit(f).lower(A).compile()
    lap[nm] = time.perf_counter() - t
out["lapack_compile_s"] = lap
dump(f"f10_compile_size_M{M}.json", out)
print(out)
