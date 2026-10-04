"""V6: the build record's section 2.1 says 'ALL 50 eigenvalues' of the
layer operator P @ Q of its fixture sit in exact pairs.  Which operator is
that?  The builder's probe captured the NumPy eig with symmetry='auto' (the
even-parity fold, N + 1 = 50).  Here: the operator the JAX path actually
hands the rule (spied), its size and its exact pairs; and the NumPy full
solve (symmetry=False)."""
from _fix import clusters
from _v import dump, jnp, np

import lumenairy.elements.rcwa._core as RC
import lumenairy.elements.rcwa.twod as RT
from lumenairy.elements.rcwa import rcwa_efficiency_2d

b3 = np.array([[1.0, 1.5, 1.0], [1.5, 4.0, 1.5], [1.0, 1.5, 1.0]])
base = np.kron(b3, np.ones((5, 5)))
calls = []
orig = RT._jax_eig_cluster_adjoint


def spy(eig_fn, problems, consumer, **kw):
    calls.append(problems)
    return orig(eig_fn, problems, consumer, **kw)


RT._jax_eig_cluster_adjoint = spy
rcwa_efficiency_2d(1.2, 1.2, jnp.asarray(base.astype(complex)), 1.0, 1.45,
                   0.45, 1.0, n_orders_x=3, n_orders_y=3)
RT._jax_eig_cluster_adjoint = orig
A = np.asarray(calls[-1][0][0])
lam = np.linalg.eigvals(A)
out = {"jax_operator_size": int(A.shape[0]),
       "jax_exact_pair_members": int(sum(len(c) for c in clusters(lam, 1e-12))),
       "jax_cluster_sizes": sorted({len(c) for c in clusters(lam, 1e-6)})}
got = []
oe = RC._eig_for


def cap(xp):
    e = oe(xp)

    def g(M):
        r = e(M)
        got.append(np.asarray(r[0]))
        return r
    return g


for sym in ("auto", False):
    got.clear()
    RC._eig_for = cap
    rcwa_efficiency_2d(1.2, 1.2, base.astype(complex), 1.0, 1.45, 0.45, 1.0,
                       n_orders_x=3, n_orders_y=3, symmetry=sym)
    RC._eig_for = oe
    lam = got[0]
    out[f"numpy_symmetry_{sym}_size"] = int(lam.size)
    out[f"numpy_symmetry_{sym}_exact_pair_members"] = int(
        sum(len(c) for c in clusters(lam, 1e-12)))
print(out)
dump("v6_builder_spectrum", out)
