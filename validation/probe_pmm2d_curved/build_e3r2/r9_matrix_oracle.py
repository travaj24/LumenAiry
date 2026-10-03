"""R9: the matrix-function oracle behind
``test_e3r2_no_eig_level_rule_can_see_the_in_cluster_block``: a random
similarity of diag(1, 1, 2, 3), L = Re tr(expm(A + t B) X), d L / d t at 0
through the eigenpairs with the cluster rule and without it, vs
``jax.scipy.linalg.expm``.

    python r9_matrix_oracle.py
"""
from _r2 import dump, jax, jnp, np
from jax.scipy.linalg import expm

from lumenairy.elements.rcwa._core import (
    _jax_eig_cluster_adjoint,
    _jax_eig_stable,
)


def via_eig(A, X, gap):
    def consumer(eigs):
        lam, V = eigs[0]
        return jnp.real(jnp.trace(V @ jnp.diag(jnp.exp(lam))
                                  @ jnp.linalg.inv(V) @ X))
    return _jax_eig_cluster_adjoint(lambda L, G: _jax_eig_stable()(L),
                                    [(A, None)], consumer, gap_rel=gap)


rng = np.random.default_rng(7)
c = lambda: jnp.asarray(rng.standard_normal((4, 4))  # noqa: E731
                        + 1j * rng.standard_normal((4, 4)))
Q = c()
A1 = Q @ jnp.diag(jnp.asarray([1.0, 1.0, 2.0, 3.0], dtype=complex)) \
    @ jnp.linalg.inv(Q)
B, X = c(), c()
gt = float(jax.grad(lambda t: jnp.real(jnp.trace(expm(A1 + t * B) @ X)))(0.0))
gr = float(jax.grad(lambda t: via_eig(A1 + t * B, X, None))(0.0))
gp = float(jax.grad(lambda t: via_eig(A1 + t * B, X, 0.0))(0.0))
out = {"oracle": gt, "rule": gr, "plain": gp,
       "rule_rel": abs(gr - gt) / abs(gt), "plain_rel": abs(gp - gt) / abs(gt)}
print(out)
print(dump("r9_matrix_oracle.json", out))
