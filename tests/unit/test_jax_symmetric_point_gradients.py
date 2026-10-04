"""Reverse-mode gradients of the RCWA JAX path and the 1-D PMM twin at and
near SYMMETRIC points, where eigenvalues of the solve are (near-)degenerate
-- the discriminating measurements of
``docs/audits/BUILD_JAX_SYMMETRIC_POINT_GRADIENTS_2026_10_03.md``.

The defect (pre-existing, found by Phase E3 round 2 of the curved-cell
campaign): at an exactly degenerate eigenvalue cluster no reverse rule
written at the eig boundary can be right for a parameter that splits the
cluster, because the cotangent the eig receives carries only the diagonal of
the consumer's response; the gradient then depends on the basis LAPACK picked
inside the cluster.  The fix routes the eig(s) and everything downstream of
them through ``rcwa._core._jax_eig_cluster_adjoint`` (the rule of the pure
staggered 2-D PMM twin).  The exact-point gradients vs a finite difference
are gated in ``test_pmm2d_staggered_curved_e3.py`` (the two flipped strict
xfails); this file pins what those do not:

* the ROOT CAUSE as a property: with the rule, the RCWA gradient does not
  depend on the basis inside the clusters; with the rule switched off it
  does (the gauge test of round 2, applied to the RCWA path);
* NEAR-degenerate clusters (inside the rule's gap_rel = 1e-6 of the
  spectrum but not exact): a no-symmetry offset of the RCWA cell, and the
  1-D PMM twin a little off normal incidence, where a Euclidean-Gram lift
  of the folded operator was measured wrong by 4.7e3 (TE) / 7.0e4 (TM)
  relative -- the reason the twin hands the rule its PENCIL (A, B);
* an FD-free oracle at exactly normal incidence: the mirror identity
  d R_{+1} / d angle = - d R_{-1} / d angle.

Every bar is derived from measurements of 2026-10-03 on both builds
(Windows 11 / CPython 3.14.6 / jax 0.11.0 and WSL / CPython 3.12.3 / jax
0.10.2), probe JSON under ``validation/probe_jax_symgrad/``, stated next to
the assertion with its gap on both sides.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.backend import JAX_AVAILABLE  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed"),
    pytest.mark.filterwarnings("ignore:.*energy closure.*"),
]

_P, _WL = 1.2, 1.0


@pytest.fixture(autouse=True, scope="module")
def _jax_x64_for_this_module():
    """Hold ``jax_enable_x64`` on for this file and RESTORE it (the JAX x64
    isolation contract: the flag must not leak to the next module)."""
    if not JAX_AVAILABLE:
        yield
        return
    import jax
    previous = bool(getattr(jax.config, "jax_enable_x64", False))
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _fd_premised(fj, x0):
    """Richardson FD (h = 3e-4, 1e-4) with its h^2 PREMISE asserted first:
    the ratio of successive rung changes on (1e-3, 3e-4, 1e-4) must be the
    h^2 value 11.375 (bar [10, 13]; measured 11.2 .. 11.6 on every entry,
    both builds)."""
    rows = [(np.asarray(fj(x0 + h)) - np.asarray(fj(x0 - h))) / (2 * h)
            for h in (1e-3, 3e-4, 1e-4)]
    c1, c2 = np.abs(rows[0] - rows[1]), np.abs(rows[1] - rows[2])
    ratio = c1 / np.maximum(c2, 1e-300)
    assert np.all((ratio > 10.0) & (ratio < 13.0)), ratio
    return (9.0 * rows[2] - rows[1]) / 8.0


def _rel(g, fd):
    return float(np.max(np.abs(g - fd)) / np.max(np.abs(fd)))


# ---------------------------------------------------------------- RCWA cell
_BASE3 = np.array([[1.0, 1.5, 1.0], [1.5, 4.0, 1.5], [1.0, 1.5, 1.0]])
_XS3 = np.zeros((3, 3))
_XS3[0, 1] = _XS3[2, 1] = 1.0
_BASE, _DIRN = (np.kron(a, np.ones((5, 5))) for a in (_BASE3, _XS3))
#: a fixed pixel pattern with no symmetry at all
_RANDOM = np.random.default_rng(7).uniform(0.0, 1.0, (15, 15))


def _rcwa_f(pol, offset=0.0):
    from lumenairy.elements.rcwa import rcwa_efficiency_2d
    o = np.asarray(rcwa_efficiency_2d(_P, _P, _BASE.astype(complex), 1.45,
                                      1.0, 0.45, _WL, n_orders_x=3,
                                      n_orders_y=3)[0])
    idx = [int(np.nonzero((o[:, 0] == a) & (o[:, 1] == b))[0][0])
           for a, b in ((0, 0), (1, 0), (-1, 0))]

    def f(t, xp):
        eps = (xp.asarray(_BASE) + t * xp.asarray(_DIRN)
               + offset * xp.asarray(_RANDOM)).astype(complex)
        _o, R, T = rcwa_efficiency_2d(_P, _P, eps, 1.45, 1.0, 0.45, _WL,
                                      polarization=pol, n_orders_x=3,
                                      n_orders_y=3)
        return xp.concatenate([xp.stack([R[i] for i in idx]),
                               xp.stack([T[i] for i in idx])])
    return f


def _rotating_eig(seed):
    """The library's eig with the basis inside every EXACT cluster
    (gap <= 1e-12 max|lam|) rotated by a seeded random unitary (the polar
    factor of a cluster-masked Gaussian).  The forward values cannot move
    beyond round-off (the solve is basis-invariant); a correct reverse pass
    cannot move either."""
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.rcwa._core import _jax_eig_stable
    orig = _jax_eig_stable()

    def eig(A):
        lam, V = orig(A)
        n = lam.shape[0]
        s = jnp.max(jnp.abs(lam))
        K = (jnp.abs(lam[:, None] - lam[None, :]) <= 1e-12 * s) | jnp.eye(
            n, dtype=bool)
        rng = np.random.default_rng(seed * 1000 + n)
        X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Y = jnp.where(K, X, 0.0)
        w, Q = jnp.linalg.eigh(jnp.conj(Y).T @ Y)
        U = Y @ ((Q * (1.0 / jnp.sqrt(w))[None, :]) @ jnp.conj(Q).T)
        return lam, V @ jax.lax.stop_gradient(U)
    return eig


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_rcwa_jax_gradient_does_not_depend_on_the_basis_inside_a_cluster(
        pol, monkeypatch):
    """THE ROOT CAUSE, as a property.  At the four-fold symmetric cell every
    eigenvalue of the layer operator is doubly degenerate; the eig's basis
    inside each pair is replaced by two seeded random rotations.

    * forward: unchanged to round-off (measured <= 5.1e-15; bar 1e-12);
    * rule ON: the symmetry-breaking gradient unchanged (measured
      4.4e-11 .. 3.7e-10 relative on Windows; bar 1e-8, >= 27x above);
    * rule OFF (``_EIG_CLUSTER_GAP_REL = 0``, the plain composition, i.e.
      the gradient before the fix): it moves -- that dependence on LAPACK's
      basis WAS the defect (measured 9.1e-3 .. 0.12 relative, both seeds,
      both polarizations; bar 1e-4, >= 90x below)."""
    import jax
    import jax.numpy as jnp

    import lumenairy.elements.rcwa._core as RC
    import lumenairy.elements.rcwa.twod as RT
    f = _rcwa_f(pol)
    orig_for = RT._eig_for

    def run():
        v = np.asarray(jax.jit(lambda t: f(t, jnp))(0.0))
        g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
        return v, g

    out = {}
    for rule, gap in (("on", RC._EIG_CLUSTER_GAP_REL), ("off", 0.0)):
        monkeypatch.setattr(RC, "_EIG_CLUSTER_GAP_REL", gap)
        monkeypatch.setattr(RT, "_eig_for", orig_for)
        v0, g0 = run()
        moves = []
        for seed in (1, 2):
            ef = _rotating_eig(seed)
            monkeypatch.setattr(RT, "_eig_for", lambda xp, ef=ef: (
                ef if xp is not np else orig_for(xp)))
            v, g = run()
            assert np.max(np.abs(v - v0)) < 1e-12
            moves.append(_rel(g, g0))
        out[rule] = max(moves)
    assert out["on"] < 1e-8, out
    assert out["off"] > 1e-4, out


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_rcwa_jax_gradient_near_a_symmetric_cell_without_any_symmetry(pol):
    """A NEAR-degenerate case: the cell offset by 1e-6 x a fixed random
    pixel pattern (no symmetry left), which splits every pair by 9.9e-10 of
    the spectrum -- inside the rule's gap_rel (1e-6) and below its lift
    (1e-7), and with split members that belong to no symmetry sector (their
    eigenvectors need not be orthogonal).  AD vs the NumPy FD: measured
    1.2e-10 (TE) / 3.6e-10 (TM) relative; bar 1e-7 as for the exact
    point."""
    import jax
    import jax.numpy as jnp
    f = _rcwa_f(pol, offset=1e-6)
    g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
    fd = _fd_premised(lambda t: f(t, np), 0.0)
    assert _rel(g, fd) < 1e-7


# ------------------------------------------------------------ 1-D PMM twin
def _pmm1d_f(pol):
    from lumenairy.elements.pmm import pmm_efficiency_1d
    o = np.asarray(pmm_efficiency_1d(_P, 2.0, 1.0, 1.45, 1.0, 0.45, 0.5, _WL,
                                     degree=12, stabilize=False)[0])
    idx = [int(np.nonzero(o == m)[0][0]) for m in (1, -1)]

    def f(t, xp):
        _o, R, T = pmm_efficiency_1d(_P, xp.asarray(2.0 + 0j), 1.0, 1.45,
                                     1.0, 0.45, 0.5, _WL, angle=t,
                                     polarization=pol, degree=12,
                                     stabilize=False)
        return xp.concatenate([xp.stack([R[i] for i in idx]),
                               xp.stack([T[i] for i in idx])])
    return f


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_pmm1d_jax_angle_gradient_obeys_the_mirror_identity_at_normal(pol):
    """FD-free oracle: the grating is mirror-symmetric, so at exactly normal
    incidence d X_{+1} / d angle = - d X_{-1} / d angle for X = R, T.  The
    AD defect of the sum: measured <= 8.6e-12 after the fix (6.7e-3 TE /
    3.3e-2 TM before it), on gradients of size 0.07 .. 0.86; bar 1e-8
    relative to the largest entry."""
    import jax
    import jax.numpy as jnp
    f = _pmm1d_f(pol)
    g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
    assert np.max(np.abs(g[0::2] + g[1::2])) < 1e-8 * np.max(np.abs(g))


@pytest.mark.parametrize("angle", [1e-9, 1e-5])
@pytest.mark.parametrize("pol", ["te", "tm"])
def test_pmm1d_jax_angle_gradient_near_normal_incidence(pol, angle):
    """Off normal incidence the half-space pairs split by 4.8e-3 x angle of
    the spectrum: 4.8e-12 at 1e-9 rad (below the plain VJP's resolution:
    7e-7 / 1.6e-6 relative error before the fix) and 4.8e-8 at 1e-5 rad
    (inside gap_rel, comparable to the lift: the folded operator with a
    Euclidean-Gram lift was wrong there by 4.7e3 / 7.0e4).  AD vs the NumPy
    FD, measured <= 2.6e-9 relative after the fix, both builds; bar 1e-7."""
    import jax
    import jax.numpy as jnp
    f = _pmm1d_f(pol)
    g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(angle))
    fd = _fd_premised(lambda t: f(t, np), angle)
    assert _rel(g, fd) < 1e-7
