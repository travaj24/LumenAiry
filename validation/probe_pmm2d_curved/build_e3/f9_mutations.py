"""E3-9 MUTATION MATRIX: each engineered defect of the twin and the gate that
catches it.

    python f9_mutations.py M

  cofactor   -- the JAX far field without the cofactor (traced path only):
                forward parity on the mapped circle (E3-2 bar 5e-12);
  nq_reread  -- the frozen far-field node count re-read inside the trace:
                the jit gate (a concretization error);
  gauge_host -- the forward-branch gauge decided on the host from traced
                data (the NumPy branch flip on a tracer): the jit gate;
  plain_eig  -- _jax_eig_stable replaced by jnp.linalg.eig: E3-5 (JAX refuses
                non-symmetric eigenvector derivatives -> an error) and, with
                enable_eigvec_derivs=True, the centred-circle d/dcx;
  no_guard   -- the topology / fold poison removed: E3-4 (a fold returns a
                finite, wrong number instead of NaN).
Output f9_mutations_M<M>.json.
"""
import sys

from _e3common import TS, WL, P, PMM2DStackPure, absd, dump, jax, jnp, np  # noqa

import lumenairy.elements.pmm._jax_twod_staggered as JT
from lumenairy.elements.pmm import Circle

M = int(sys.argv[1])
out = {"M": M}


def mk(backend="jax"):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                        n_orders=2, backend=backend)
    st.add_layer(0.5, shapes=[Circle(0.6, 0.6, 0.36, 4.0)],
                 background_eps=1.0)
    st.set_source(WL)
    return st


o, R, T, J = mk("numpy").solve()
st = mk()
tw = st.jax_twin()
p0 = tw.p0


def f(r, cx=0.6):
    p = tw.params()
    p["layers"][0]["shapes"] = [Circle(cx, 0.6, r, 4.0)]
    return st.solve(params=p)


_o, R0, T0, J0 = f(jnp.asarray(0.36))
out["clean_parity"] = [absd(R0, R), absd(T0, T), absd(J0, J)]

# cofactor
orig_far = TS._far_projector_mapped


class NoCof:
    def __init__(self, m):
        self.m = m

    def __getattr__(self, k):
        return getattr(self.m, k)

    def geom(self, sx, sy, U, V):
        X, Y, xu, xv, yu, yv = self.m.geom(sx, sy, U, V)
        one = jnp.ones_like(xu)
        return X, Y, one, 0 * xv, 0 * yu, one


def nocof(bx, by, ox, oy, a0x, a0y, cmap, xp=np, **kw):
    if xp is not np:
        cmap = NoCof(cmap)
    return orig_far(bx, by, ox, oy, a0x, a0y, cmap, xp=xp, **kw)


TS._far_projector_mapped = nocof
try:
    _o, R1, T1, J1 = f(jnp.asarray(0.36))
    out["cofactor"] = [absd(R1, R), absd(T1, T), absd(J1, J)]
finally:
    TS._far_projector_mapped = orig_far


def jit_err(fun):
    try:
        jax.jit(fun)(0.36)
        return "NO ERROR"
    except Exception as exc:                                # noqa: BLE001
        return type(exc).__name__


def reread(*a, **k):
    k.pop("nq_cells", None)
    return orig_far(*a, **k)


TS._far_projector_mapped = reread
try:
    out["nq_reread"] = jit_err(lambda r: f(r)[2][0, p0])
finally:
    TS._far_projector_mapped = orig_far
flip = TS._forward_branch_flip
TS._forward_branch_flip = lambda q, xp=np: flip(q)
try:
    out["gauge_host"] = jit_err(lambda r: f(r)[2][0, p0])
finally:
    TS._forward_branch_flip = flip

# plain eig
orig_eig = JT._stag_geneig_jax
g_ok = float(jax.jit(jax.grad(lambda c: f(0.36, c)[2][0, p0]))(0.6))
out["dT_dcx_centred_clean"] = g_ok
JT._stag_geneig_jax = lambda L, G, tau_rel=None: jnp.linalg.eig(
    jnp.linalg.solve(G, L))
try:
    try:
        jax.jit(jax.grad(lambda c: f(0.36, c)[2][0, p0]))(0.6)
        out["plain_eig"] = "NO ERROR"
    except NotImplementedError as exc:
        out["plain_eig"] = "NotImplementedError: " + str(exc)[:90]
    JT._stag_geneig_jax = lambda L, G, tau_rel=None: jax.lax.linalg.eig(
        jnp.linalg.solve(G, L), compute_left_eigenvectors=False,
        enable_eigvec_derivs=True)
    out["dT_dcx_centred_unregularised"] = float(jax.jit(jax.grad(
        lambda c: f(0.36, c)[2][0, p0]))(0.6))
    out["dT_dr_unregularised"] = float(jax.jit(jax.grad(
        lambda r: f(r)[2][0, p0]))(0.36))
finally:
    JT._stag_geneig_jax = orig_eig
out["dT_dr_clean"] = float(jax.jit(jax.grad(lambda r: f(r)[2][0, p0]))(0.36))

# no guard
orig_min = JT._min_detj
JT._min_detj = lambda *a, **k: jnp.asarray(1.0)
orig_tsm = JT._traced_shape_merge


def tsm(*a, **k):
    cm, cells, mus, ok = orig_tsm(*a, **k)
    return cm, cells, mus, jnp.asarray(True)


JT._traced_shape_merge = tsm
try:
    out["no_guard_fold_value"] = float(jax.jit(
        lambda r: f(r)[2][0, p0])(0.62))
finally:
    JT._min_detj = orig_min
    JT._traced_shape_merge = orig_tsm
out["guard_fold_value"] = float(jax.jit(lambda r: f(r)[2][0, p0])(0.62))
dump(f"f9_mutations_M{M}.json", out)
print(out)
