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

Round 2 (2026-10-04, build record section 10) adds one pin per family
routed through ``rcwa._core._jax_cluster_routed`` (``rcwa_jones_2d``,
``RCWAStack``, the Berreman off-plane cascade, ``pmm_jones_1d``,
``PMMStack``, the hybrid ``PMM2DStack``), the library switch
``lumenairy.backend.jax_cluster_rule`` (wrong when off at a symmetric point,
inert elsewhere, structurally absent from a vmapped gradient when off), and
two defects found on the way (``rcwa_jones_2d(formulation='li')`` on a
traced tensor solved Laurent; ``RCWAStack`` leaked tracers through its mode
cache).

Every bar is derived from measurements of 2026-10-03 / 04 on both builds
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


def _fd_premised(fj, x0, scale=1.0, floor=1e-3):
    """Richardson FD (h = 3e-4, 1e-4, times ``scale``) with its h^2 PREMISE
    asserted first: on the components carrying >= ``floor`` of max |FD| the
    ratio of successive rung changes on (1e-3, 3e-4, 1e-4) must be the h^2
    value 11.375 (bar [10, 13]; measured 11.1 .. 11.7 on every such entry of
    every fixture here, both builds)."""
    rows = [(np.asarray(fj(x0 + h * scale)) - np.asarray(fj(x0 - h * scale)))
            / (2 * h * scale) for h in (1e-3, 3e-4, 1e-4)]
    c1, c2 = np.abs(rows[0] - rows[1]), np.abs(rows[1] - rows[2])
    fd = (9.0 * rows[2] - rows[1]) / 8.0
    big = np.abs(fd) >= floor * np.max(np.abs(fd))
    ratio = c1[big] / np.maximum(c2[big], 1e-300)
    assert np.all((ratio > 10.0) & (ratio < 13.0)), ratio
    return fd


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


def _rotated_eig_factory(seed):
    """A replacement for ``rcwa._core._jax_eig_stable`` (the factory every
    routed twin's eig calls) whose eig returns its basis inside every EXACT
    cluster (gap <= 1e-12 max|lam|) rotated by a seeded random unitary (the
    polar factor of a cluster-masked Gaussian) AND runs the library's own
    eig VJP on the rotated residuals -- i.e. what LAPACK returning another
    basis looks like.  The forward values cannot move beyond round-off (the
    solve is basis-invariant); a correct reverse pass cannot move either."""
    from functools import partial

    import jax
    import jax.numpy as jnp

    from lumenairy.elements.rcwa._core import _jax_eig_stable
    base = _jax_eig_stable()

    def unitary(lam):
        n = lam.shape[0]
        s = jnp.max(jnp.abs(lam))
        K = (jnp.abs(lam[:, None] - lam[None, :]) <= 1e-12 * s) | jnp.eye(
            n, dtype=bool)
        rng = np.random.default_rng(seed * 1000 + n)
        X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Y = jnp.where(K, X, 0.0)
        w, Q = jnp.linalg.eigh(jnp.conj(Y).T @ Y)
        return jax.lax.stop_gradient(
            Y @ ((Q * (1.0 / jnp.sqrt(w))[None, :]) @ jnp.conj(Q).T))

    @partial(jax.custom_vjp, nondiff_argnums=(1,))
    def eig_rot(A, tau_rel=1e-12):
        lam, V = base(A, tau_rel)
        return lam, V @ unitary(lam)

    def fwd(A, tau_rel):
        out = eig_rot(A, tau_rel)
        return out, out

    def bwd(tau_rel, res, cot):
        return base.bwd(tau_rel, res, cot)

    eig_rot.defvjp(fwd, bwd)
    return lambda: eig_rot


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_rcwa_jax_gradient_does_not_depend_on_the_basis_inside_a_cluster(
        pol, monkeypatch):
    """THE ROOT CAUSE, as a property.  At the four-fold symmetric cell every
    eigenvalue of the layer operator is doubly degenerate; the eig's basis
    inside each pair is replaced by two seeded random rotations.

    * forward: unchanged to round-off (bar 1e-12; measured <= 5e-15);
    * rule ON: the symmetry-breaking gradient unchanged (bar 1e-8;
      measured 3.3e-11 .. 2.7e-10 Windows, 1.0e-10 .. 5.2e-10 WSL);
    * rule OFF (``lumenairy.backend.jax_cluster_rule(False)``, the plain
      composition, i.e. the gradient before the fix): it moves -- that
      dependence on LAPACK's basis WAS the defect (bar 1e-4; measured
      9.4e-2 .. 0.71 Windows, 2.9e-2 .. 0.14 WSL)."""
    import jax
    import jax.numpy as jnp

    import lumenairy.elements.rcwa._core as RC
    from lumenairy.backend import jax_cluster_rule
    f = _rcwa_f(pol)
    orig = RC._jax_eig_stable

    def run():
        v = np.asarray(jax.jit(lambda t: f(t, jnp))(0.0))
        g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
        return v, g

    out = {}
    for rule in (True, False):
        with jax_cluster_rule(rule):
            monkeypatch.setattr(RC, "_jax_eig_stable", orig)
            v0, g0 = run()
            moves = []
            for seed in (1, 2):
                monkeypatch.setattr(RC, "_jax_eig_stable",
                                    _rotated_eig_factory(seed))
                v, g = run()
                assert np.max(np.abs(v - v0)) < 1e-12
                moves.append(_rel(g, g0))
            monkeypatch.setattr(RC, "_jax_eig_stable", orig)
        out[rule] = max(moves)
    assert out[True] < 1e-8, out
    assert out[False] > 1e-4, out


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


# ===================================================== round 2: every twin
# Each family's symmetric configuration has an EXACT degenerate eigenvalue
# cluster that the parameter splits (counted in the build record, section
# 10).  Bar 1e-7 on AD vs the premise-checked FD, as for the round-1 pins:
# the measured envelopes after the fix are listed per test (both builds),
# the defects before it were 0.30 .. 6.6 -- >= 6 decades above the bar.

def _rcwa_cell_tensor(t, xp):
    eps = (xp.asarray(_BASE) + t * xp.asarray(_DIRN)).astype(complex)
    return eps[:, :, None, None] * xp.eye(3, dtype=complex)[None, None]


def _rcwa_idx():
    from lumenairy.elements.rcwa import rcwa_efficiency_2d
    o = np.asarray(rcwa_efficiency_2d(_P, _P, _BASE.astype(complex), 1.45,
                                      1.0, 0.45, _WL, n_orders_x=3,
                                      n_orders_y=3)[0])
    return [int(np.nonzero((o[:, 0] == a) & (o[:, 1] == b))[0][0])
            for a, b in ((0, 0), (1, 0), (-1, 0))]


def test_rcwa_jones_2d_symmetry_breaking_gradient_at_a_symmetric_cell():
    """``rcwa_jones_2d`` (JAX tensor cell eps * I), the four-fold cell, t on
    the two x-side blocks; R, T of (0,0), (+-1,0) for both incident
    polarizations.  Before: 0.34 / 0.38 (Windows / WSL, the same cell with the half-spaces
    swapped, probe g1); after: 2.0e-10 / 4.4e-10."""
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.rcwa import rcwa_jones_2d
    idx = _rcwa_idx()

    def f(t, xp):
        _o, R, T, _J = rcwa_jones_2d(_P, _P, _rcwa_cell_tensor(t, xp), 1.45,
                                     1.0, 0.45, _WL, n_orders_x=3,
                                     n_orders_y=3)
        return xp.concatenate([xp.ravel(xp.stack([R[:, i] for i in idx])),
                               xp.ravel(xp.stack([T[:, i] for i in idx]))])
    g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
    fd = _fd_premised(lambda t: f(t, np), 0.0)
    assert _rel(g, fd) < 1e-7


def test_rcwa_stack_symmetry_breaking_gradient_at_a_symmetric_cell():
    """``RCWAStack`` on JAX input: the four-fold cell over a uniform spacer
    (eps 2.25, 0.1); t on the two x-side blocks; R, T of (0,0), (+-1,0) for
    both incident polarizations.  Before: 0.13 / 0.14 (probe g1, half-spaces
    swapped); after: 2.3e-10 / 4.7e-10 (Windows / WSL)."""
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.rcwa import RCWAStack
    idx = _rcwa_idx()

    def f(t, xp):
        st = RCWAStack(_P, period_y=_P, n_superstrate=1.0, n_substrate=1.45,
                       n_orders=3, n_orders_y=3)
        st.add_layer(0.45, eps_cell=(xp.asarray(_BASE) + t * xp.asarray(
            _DIRN)).astype(complex))
        st.add_layer(0.1, eps=2.25)
        st.set_source(_WL, theta=0.0, phi=0.0)
        res = st.solve()
        _o, R, T = res.efficiencies()
        return xp.concatenate([xp.ravel(xp.stack([R[:, i] for i in idx])),
                               xp.ravel(xp.stack([T[:, i] for i in idx]))])
    g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
    fd = _fd_premised(lambda t: f(t, np), 0.0)
    assert _rel(g, fd) < 1e-7


def test_berreman_traced_tensor_gradient_at_an_isotropic_layer():
    """``berreman_jones_1d`` with a traced (3, 3) tensor (the off-plane /
    Li-2003 route): n_sup 1.45 | eps 2.56 I + d (E_xy + E_yx), 0.3 um |
    eps 2.1, 0.1 um | n_sub 1, normal incidence, d / d d at d = 0 -- the
    isotropic layer's two exactly degenerate pairs are split by the
    in-plane anisotropy.  Outputs R, T and Re / Im of the reflection Jones.
    Before: 0.99 (both builds); after: 1.5e-9 / 1.7e-9 (Windows / WSL)."""
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.berreman import berreman_jones_1d
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], complex)

    def f(d, xp):
        e = 2.56 * xp.eye(3, dtype=complex) + d * xp.asarray(X)
        if xp is jnp:
            R, T, Jr, _Jt = berreman_jones_1d(
                [(e, jnp.asarray(0.3e-6)), (jnp.asarray(2.1 + 0j),
                                            jnp.asarray(0.1e-6))],
                jnp.asarray(1.0 + 0j), jnp.asarray(1.45 + 0j),
                jnp.asarray(1e-6))
        else:
            R, T, Jr, _Jt = berreman_jones_1d([(e, 0.3e-6), (2.1, 0.1e-6)],
                                              1.0, 1.45, 1e-6)
        Jr = xp.ravel(Jr)
        return xp.concatenate([xp.ravel(R), xp.ravel(T), xp.real(Jr),
                               xp.imag(Jr)])
    g = np.asarray(jax.jit(jax.jacrev(lambda d: f(d, jnp)))(0.0))
    fd = _fd_premised(lambda d: f(d, np), 0.0)
    assert _rel(g, fd) < 1e-7


def _pmm_jones_pack(R, T, idx, xp):
    return xp.concatenate([xp.stack([R[p][i] for p in (0, 1) for i in idx]),
                           xp.stack([T[p][i] for p in (0, 1) for i in idx])])


def test_pmm_jones_1d_angle_gradient_at_normal_incidence():
    """``pmm_jones_1d`` (JAX), ridge 4 I / groove I, P 1.2, duty 0.5, depth
    0.45, n_sup 1.45, degree 12: d / d(angle) at exactly 0 of R, T of the
    +-1 orders for both incident polarizations (the half-spaces' shared
    geometric eig carries the +-m pairs).  Before: 2.7 (both builds); after:
    1.3e-10 / 6.0e-11 (Windows / WSL)."""
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.pmm import pmm_jones_1d
    ER, EG = 4.0 * np.eye(3, dtype=complex), np.eye(3, dtype=complex)
    o = np.asarray(pmm_jones_1d(_P, ER, EG, 1.0, 1.45, 0.45, 0.5, _WL,
                                degree=12, stabilize=False)[0])
    idx = [int(np.nonzero(o == m)[0][0]) for m in (1, -1)]

    def f(t, xp):
        _o, R, T, _J = pmm_jones_1d(_P, ER, EG, 1.0, 1.45, 0.45, 0.5, _WL,
                                    angle=t, degree=12, stabilize=False)
        return _pmm_jones_pack(R, T, idx, xp)
    g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
    fd = _fd_premised(lambda t: f(float(t), np), 0.0)
    assert _rel(g, fd) < 1e-7


@pytest.mark.parametrize("grids", ["shared", "per-layer"])
def test_pmm_stack_angle_gradient_at_normal_incidence(grids):
    """``PMMStack`` (JAX), P 1.2 um: a symmetric binary layer (eps 4 / 1,
    duty 0.5, 0.3 um) over a UNIFORM spacer (eps 2.25, 0.1 um -- its Mbig is
    degenerate throughout), n_sup 1.45, n_sub 1, degree 12, wl 1 um;
    d / d(angle) at exactly 0 of R, T of the +-1 orders, both incident
    polarizations; shared and per-layer grids.  Before: 1.05 (shared, this
    fixture) and 6.6 (per-layer, a three-layer fixture); after:
    8.3e-11 / 9.1e-11 (shared) and 8.0e-11 / 8.5e-11 (per-layer), Windows /
    WSL."""
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.pmm import PMMStack

    def solve(angle):
        st = PMMStack(1.2e-6, n_substrate=1.0, n_superstrate=1.45, degree=12,
                      layer_grids=grids)
        st.add_layer(0.3e-6, segments=[(0.5, 4.0), (0.5, 1.0)])
        st.add_layer(0.1e-6, segments=[(1.0, 2.25)])
        st.set_source(1.0e-6, angle=angle)
        return st.solve()
    o = np.asarray(solve(0.0)[0])
    idx = [int(np.nonzero(o == m)[0][0]) for m in (1, -1)]

    def f(a, xp):
        _o, R, T, _J = solve(a)
        return _pmm_jones_pack(R, T, idx, xp)
    g = np.asarray(jax.jit(jax.jacrev(lambda a: f(a, jnp)))(0.0))
    fd = _fd_premised(lambda a: f(float(a), np), 0.0)
    assert _rel(g, fd) < 1e-7


def test_pmm2d_hybrid_stack_traced_layout_gradient_at_a_symmetric_cell():
    """``PMM2DStackHybrid`` (JAX, 'laurent'), a 6 x 6 cell with a C4v
    inclusion (eps 4 on [0:3, 0:3]) as a TRACED eps_cell with a 9-region
    layout (C4v-symmetric walls), over a uniform layer; d / d(eps of the
    corner region) at 0 (C4v -> the diagonal mirror; every eigenvalue of the
    patterned layer in a pair).  Outputs R, T of (+-1, 0), both incident
    polarizations, and Re / Im of the zeroth-order reflection Jones.  The FD
    oracle is the concrete forward of the same twin (a NumPy cell cannot
    carry the region layout), steps 1e-2 .. 1e-3 (the response is nearly
    linear; smaller rungs sit at the forward noise).  Before: 0.30 (Windows)
    / 0.35 (WSL); after: 2.3e-10 / 2.0e-10."""
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.pmm import PMM2DStackHybrid
    S = 6
    C4 = np.full((S, S), 1.0 + 0j)
    C4[0:3, 0:3] = 4.0
    LAY = np.zeros((S, S), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            LAY[i, j] = 1 + 3 * i + j

    def solve(cell, layout=None):
        st = PMM2DStackHybrid(1.2e-6, n_substrate=1.0, n_superstrate=1.45,
                              degree=7, n_orders=3, formulation="laurent")
        kw = {} if layout is None else {"region_layout": layout}
        st.add_layer(0.3e-6, eps_cell=cell, **kw)
        st.add_layer(0.08e-6, eps=2.25)
        st.set_source(1.0e-6, theta=0.0)
        return st.solve()
    o = np.asarray(solve(C4)[0])
    idx = [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == 0))[0][0])
           for m in (1, -1)]

    def f(d):
        _o, R, T, J = solve(jnp.asarray(C4).at[0, 0].add(d), LAY)
        J = jnp.ravel(J)
        return jnp.concatenate([_pmm_jones_pack(R, T, idx, jnp),
                                jnp.real(J), jnp.imag(J)])
    fwd = jax.jit(f)
    g = np.asarray(jax.jit(jax.jacrev(f))(0.0))
    fd = _fd_premised(lambda d: np.asarray(fwd(jnp.asarray(d))), 0.0,
                      scale=10.0)
    assert _rel(g, fd) < 1e-7


# ================================================ the library-wide switch
def _count_eig(jaxpr):
    """Number of ``eig`` primitives in a jaxpr, through every sub-jaxpr
    (cond branches, custom-VJP calls, pjit)."""
    import jax
    import jax.extend.core  # noqa: F401
    n = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "eig":
            n += 1
        for v in eqn.params.values():
            for sub in (v if isinstance(v, (tuple, list)) else (v,)):
                if isinstance(sub, jax.extend.core.ClosedJaxpr):
                    n += _count_eig(sub.jaxpr)
                elif isinstance(sub, jax.extend.core.Jaxpr):
                    n += _count_eig(sub)
    return n


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_the_switch_off_is_wrong_at_a_symmetric_point_and_inert_elsewhere(
        pol):
    """``lumenairy.backend.jax_cluster_rule(False)`` -- the ONE switch of
    the rule -- made loud, two-sidedly:

    * at the four-fold cell (an exact cluster) the gradient with the switch
      OFF is WRONG against the FD (> 1e-3; measured 0.23 / 0.28 TE, 0.39 / 0.47 TM) and ON
      is right (< 1e-7);
    * away from the symmetry (the cell offset by 1e-2: no pair closer than
      2e-5 of the spectrum) ON and OFF give the same gradient (< 1e-12
      relative; measured <= 4e-16)."""
    import jax
    import jax.numpy as jnp

    from lumenairy.backend import jax_cluster_rule, jax_cluster_rule_enabled
    assert jax_cluster_rule_enabled()            # the default
    f = _rcwa_f(pol)
    fd = _fd_premised(lambda t: f(t, np), 0.0)
    g = {}
    for on in (True, False):
        with jax_cluster_rule(on):
            gj = jax.jit(jax.jacrev(lambda t: f(t, jnp)))
            g[on] = (np.asarray(gj(0.0)), np.asarray(gj(1e-2)))
    assert jax_cluster_rule_enabled()            # restored
    assert _rel(g[True][0], fd) < 1e-7
    assert _rel(g[False][0], fd) > 1e-3
    assert _rel(g[False][1], g[True][1]) < 1e-12


def test_the_switch_off_removes_the_rule_from_a_vmapped_gradient():
    """What OFF costs, structurally: the jaxpr of a VMAPPED gradient with
    the switch off holds exactly the one eig of the plain solve (the
    gradient before the rule); with it on, the rule's lifted branch adds its
    four lifted eigs.  The measured cost is in the build record."""
    import jax
    import jax.numpy as jnp

    from lumenairy.backend import jax_cluster_rule
    f = _rcwa_f("te")
    ts = jnp.asarray([0.0, 1e-2, 2e-2])

    def loss(t):
        return jnp.sum(f(t, jnp) * jnp.arange(1.0, 7.0))
    n = {}
    for on in (True, False):
        with jax_cluster_rule(on):
            n[on] = _count_eig(jax.make_jaxpr(jax.vmap(jax.grad(loss)))(
                ts).jaxpr)
    assert n[False] == 1, n
    assert n[True] >= 1 + 4, n


def test_a_vmapped_batch_mixing_symmetric_and_generic_points_is_exact():
    """Under ``jax.vmap`` the rule decides its branch on the OR over the
    batch (``rcwa._core._batch_any``): a batch with the symmetric cell (a
    cluster) and offset cells (none) runs the lifted branch for all of them,
    which must equal the per-point jitted gradients (probe f1, a batch [0, 1e-3, 2e-3, 3e-3]: 6.5e-13 / 1.3e-12 absolute
    on gradients of order 0.1, Windows / WSL)."""
    import jax
    import jax.numpy as jnp
    f = _rcwa_f("tm")

    def loss(t):
        return jnp.sum(f(t, jnp) * jnp.arange(1.0, 7.0))
    ts = jnp.asarray([0.0, 1e-2, 2e-2])
    gv = np.asarray(jax.jit(jax.vmap(jax.grad(loss)))(ts))
    gs = np.asarray([jax.jit(jax.grad(loss))(t) for t in ts])
    assert np.max(np.abs(gv - gs)) < 1e-10 * np.max(np.abs(gs))


# ==================================== found on the way (verifier P1-1, g1)
def test_rcwa_jones_2d_li_on_a_traced_tensor_is_li_at_a_symmetric_cell():
    """P1-1 of the verification (pre-existing): ``rcwa_jones_2d(formulation=
    'li')`` on a TRACED tensor took the general path, which built every
    block by the direct rule -- it solved LAURENT.  The traced call now
    selects the in-plane Li operators on the tensor's value
    (``_core._tensor_inplane_mask``), the same choice the concrete call
    makes.  At the four-fold cell: the jitted forward equals NumPy 'li' to
    round-off (bar 1e-12; measured 5.6e-16 / 1.2e-15) and the symmetry-breaking 'li'
    gradient matches the 'li' FD (bar 1e-7; measured 2.6e-10 / 3.2e-10).  The no-
    symmetry cell is the verifier's pin."""
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.rcwa import rcwa_jones_2d
    idx = _rcwa_idx()

    def f(t, xp):
        _o, R, T, _J = rcwa_jones_2d(_P, _P, _rcwa_cell_tensor(t, xp), 1.45,
                                     1.0, 0.45, _WL, n_orders_x=3,
                                     n_orders_y=3, formulation="li")
        return xp.concatenate([xp.ravel(xp.stack([R[:, i] for i in idx])),
                               xp.ravel(xp.stack([T[:, i] for i in idx]))])
    fwd = np.asarray(jax.jit(lambda t: f(t, jnp))(0.0))
    assert np.max(np.abs(fwd - np.asarray(f(0.0, np)))) < 1e-12
    g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
    fd = _fd_premised(lambda t: f(t, np), 0.0)
    assert _rel(g, fd) < 1e-7


def test_rcwa_stack_jit_then_grad_then_eager_does_not_leak_tracers():
    """Pre-existing (found by the round-2 measurement): ``RCWAStack`` cached
    its half-space modes in a module-level cache even inside a ``jax.jit``
    trace, where they are tracers, so ``jit(f)`` followed by
    ``jit(grad(f))`` or an eager ``f`` raised UnexpectedTracerError.  Now
    a traced call computes them uncached; the three calls agree."""
    import jax
    import jax.numpy as jnp

    import lumenairy.elements.rcwa._core as RC
    from lumenairy.elements.rcwa import RCWAStack
    RC._clear_rcwa_caches()

    def f(t):
        st = RCWAStack(_P, period_y=_P, n_superstrate=1.0, n_substrate=1.45,
                       n_orders=3, n_orders_y=3)
        st.add_layer(0.45, eps_cell=(jnp.asarray(_BASE) + t * jnp.asarray(
            _DIRN)).astype(complex))
        st.set_source(_WL, theta=0.0, phi=0.0)
        return jnp.sum(st.solve().efficiencies()[1])
    v = float(jax.jit(f)(1e-2))
    g = float(jax.jit(jax.grad(f))(1e-2))
    assert np.isfinite(g)
    assert abs(float(f(1e-2)) - v) < 1e-12


# ============================================ round 3: the switch's semantics
@pytest.mark.parametrize("value,expect", [("0", "False"), ("off", "False"),
                                          ("YES", "True"), ("", "True"),
                                          ("maybe", "ValueError")])
def test_the_switch_environment_value_is_parsed_or_refused(value, expect):
    """``LUMENAIRY_JAX_CLUSTER_RULE`` is read once at import: the eight
    recognised spellings set the process value; anything else REFUSES to
    import (it used to leave the rule on silently)."""
    import subprocess
    import sys
    env = dict(os.environ, LUMENAIRY_JAX_CLUSTER_RULE=value)
    code = ("from lumenairy.backend import jax_cluster_rule_enabled as e; "
            "print(e())")
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=300)
    if expect == "ValueError":
        assert r.returncode != 0 and "LUMENAIRY_JAX_CLUSTER_RULE" in r.stderr
    else:
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip().splitlines()[-1] == expect


def test_a_switch_scope_in_one_thread_does_not_reach_another():
    """The scope is THREAD-local (and part of JAX's trace key): a thread
    holding ``jax_cluster_rule(False)`` does not turn the rule off for a
    thread that traces at the same time."""
    import threading

    from lumenairy.backend import jax_cluster_rule, jax_cluster_rule_enabled
    seen = {}
    inside, release = threading.Event(), threading.Event()

    def off():
        with jax_cluster_rule(False):
            seen["off"] = jax_cluster_rule_enabled()
            inside.set()
            release.wait(30)

    t = threading.Thread(target=off)
    t.start()
    inside.wait(30)
    seen["main"] = jax_cluster_rule_enabled()
    release.set()
    t.join(30)
    assert seen == {"off": False, "main": True}
    assert jax_cluster_rule_enabled()
