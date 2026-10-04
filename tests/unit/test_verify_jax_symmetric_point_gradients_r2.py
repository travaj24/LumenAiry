"""Independent verification of ROUND 2 of the JAX symmetric-point gradient
fix (``docs/audits/BUILD_JAX_SYMMETRIC_POINT_GRADIENTS_2026_10_03.md``
section 10; record ``docs/audits/VERIFY_JAX_SYMMETRIC_POINT_GRADIENTS_R2_
2026_10_04.md``).  Verifier: Claude Opus 5.5 (``claude-opus-5-5``).

Every fixture here is the verifier's own (different cells, contrasts, order
counts, layer counts, losses and incidences from the builder's).  Confirmed
claims are passing two-sided pins; each DEFECT is a strict xfail that flips
when it is fixed.  Readings quoted next to each bar are Windows 11 / CPython
3.14.6 / jax 0.11.0 and WSL / CPython 3.12.3 / jax 0.10.2, 2026-10-04, probe
JSON under ``validation/probe_jax_symgrad/verify_r2/``.

"FD" is a Richardson-extrapolated central difference of the NumPy (or
concrete) forward on three rungs (h, 0.3 h, 0.1 h) whose h^2 premise --
the ratio of successive rung changes equals (h1^2-h2^2)/(h2^2-h3^2) = 11.375
to 12 % on every component above a floor of max |FD| -- is ASSERTED before
the FD is used.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.backend import JAX_AVAILABLE  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed"),
    pytest.mark.filterwarnings("ignore:.*energy closure.*"),
]

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


@pytest.fixture(autouse=True, scope="module")
def _jax_x64_for_this_module():
    """Hold ``jax_enable_x64`` on for this file and RESTORE it."""
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


@pytest.fixture(autouse=True)
def _rule_default_on():
    """Every test starts and ends with the library default (rule ON)."""
    from lumenairy.backend import set_jax_cluster_rule
    set_jax_cluster_rule(True)
    yield
    set_jax_cluster_rule(True)


def _fd(fn, x0, hs=(1e-3, 3e-4, 1e-4), floor=1e-2, scale=1.0):
    rows = [(np.asarray(fn(x0 + h * scale)) - np.asarray(fn(x0 - h * scale)))
            / (2 * h * scale) for h in hs]
    h1, h2, h3 = hs
    expect = (h1 ** 2 - h2 ** 2) / (h2 ** 2 - h3 ** 2)
    fd = rows[2] + (rows[2] - rows[1]) * (h3 ** 2 / (h2 ** 2 - h3 ** 2))
    big = np.abs(fd) >= floor * np.max(np.abs(fd))
    ratio = (np.abs(rows[0] - rows[1])[big]
             / np.maximum(np.abs(rows[1] - rows[2])[big], 1e-300))
    assert np.all(np.abs(ratio / expect - 1.0) < 0.12), (ratio, expect)
    return fd


#: FD rungs for the RCWA cells: on (1e-3, 3e-4, 1e-4) the NumPy forward's
#: round-off near the split clusters (~eps / splitting) exceeds the rung
#: changes of the nearly linear response and the h^2 premise cannot be
#: shown; on (1e-2, 3e-3, 1e-3) it holds on every component above 1e-2 of
#: max |FD| (verifier, 2026-10-04, Windows), and the Richardson remainder is
#: O(h^4) ~ 1e-12 of the gradient.
_HS_RCWA = (1e-2, 3e-3, 1e-3)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _ad(f, x0, on=True):
    import jax
    import jax.numpy as jnp

    from lumenairy.backend import jax_cluster_rule
    with jax_cluster_rule(on):
        return np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(x0))


# ------------------------------------------------------------- fixtures
_S = 24
_CROSS = np.zeros((_S, _S), bool)
_CROSS[8:16, :] = True
_CROSS[:, 8:16] = True
_ARMX = np.zeros((_S, _S))
_ARMX[8:16, 16:24] = 1.0          # +x half-arm: C4v -> one mirror
_RND0 = np.random.default_rng(11).uniform(-1, 1, (_S, _S))
_RND0 = _RND0 - _RND0.mean()


def _cross(eps_a, xp, t=0.0, dirn=_ARMX):
    return (xp.asarray(np.where(_CROSS, eps_a, 1.44 + 0j))
            + t * xp.asarray(dirn))


def _berreman(angle, n_layers=2):
    from lumenairy.elements.berreman import berreman_jones_1d
    D = np.array([[1, 0, 0.5], [0, -1, 0], [0.5, 0, 0]], complex)

    def f(t, xp):
        e1 = (3.0 + 0.1j) * xp.eye(3, dtype=complex) + t * xp.asarray(D)
        e2 = 2.0 * xp.eye(3, dtype=complex) + 0.5 * t * xp.asarray(D.T)
        lay = [(e1, 0.25e-6), (e2, 0.15e-6)][:n_layers]
        if xp is not np:
            R, T, Jr, Jt = berreman_jones_1d(
                [(e, xp.asarray(d)) for e, d in lay], xp.asarray(1.5 + 0j),
                xp.asarray(1.0 + 0j), xp.asarray(0.9e-6), angle=angle)
        else:
            R, T, Jr, Jt = berreman_jones_1d(lay, 1.5, 1.0, 0.9e-6,
                                             angle=angle)
        J = xp.concatenate([xp.ravel(Jr), xp.ravel(Jt)])
        return xp.concatenate([xp.ravel(R), xp.ravel(T), xp.real(J),
                               xp.imag(J)])
    return f


# =============================================== claim (1), own fixtures
def test_r2v_berreman_two_lossy_isotropic_layers_oblique():
    """Two lossy isotropic layers (3.0+0.1j, 2.0), 0.35 rad, a direction
    mixing in-plane anisotropy and eps_xz (off-plane route); R, T, Re/Im of
    both Jones matrices.  ON vs FD: 4.4e-10 (Windows) / 4.6e-10 (WSL); OFF
    (the plain composition, the gradient before the fix): 0.45 / 0.45.
    Bars 1e-7 and 1e-2."""
    f = _berreman(0.35)
    fd = _fd(lambda t: f(t, np), 0.0)
    assert _rel(_ad(f, 0.0, True), fd) < 1e-7
    assert _rel(_ad(f, 0.0, False), fd) > 1e-2


def test_r2v_pmm_jones_1d_lossy_anisotropic_ridge_at_normal_incidence():
    """P 0.9, ridge diag(5, 4, 4.5) + 0.2j (in-plane anisotropic, lossy,
    mirror kept), groove 1.2, duty 0.4, depth 0.3, n_sub 1.5, n_sup 1,
    degree 10; d / d(angle) at 0 of every R, T and Re/Im J.  ON 1.7e-10 /
    1.7e-10, OFF 0.24 / 0.24.  Bars 1e-7 and 1e-2."""
    from lumenairy.elements.pmm import pmm_jones_1d
    ER = np.diag([5.0 + 0.2j, 4.0 + 0.2j, 4.5 + 0.2j])

    def f(t, xp):
        _o, R, T, J = pmm_jones_1d(0.9, ER, 1.2 * np.eye(3, dtype=complex),
                                   1.5, 1.0, 0.3, 0.4, 1.0, angle=t,
                                   degree=10, stabilize=False)
        J = xp.ravel(xp.asarray(J))
        return xp.concatenate([xp.ravel(xp.asarray(R)),
                               xp.ravel(xp.asarray(T)), xp.real(J),
                               xp.imag(J)])
    fd = _fd(lambda t: f(float(t), np), 0.0)
    assert _rel(_ad(f, 0.0, True), fd) < 1e-7
    assert _rel(_ad(f, 0.0, False), fd) > 1e-2


def test_r2v_rcwa_jones_2d_lossy_cross_cell():
    """24 x 24 cross cell (arms 6.25+0.8j, background 1.44), n_orders 3 (7 x 7 = 49 orders),
    n_sub 1.5 / n_sup 1, depth 0.3; t on the +x half-arm only.  ON 5.4e-10
    (Windows), OFF 0.042.  Bars 1e-7 and 1e-3."""
    from lumenairy.elements.rcwa import rcwa_jones_2d

    def f(t, xp):
        e = _cross(6.25 + 0.8j, xp, t)
        tens = e[:, :, None, None] * xp.eye(3, dtype=complex)[None, None]
        _o, R, T, _J = rcwa_jones_2d(1.3, 1.3, tens, 1.5, 1.0, 0.3, 1.0,
                                     n_orders_x=3, n_orders_y=3)
        return xp.concatenate([xp.ravel(R), xp.ravel(T)])
    fd = _fd(lambda t: f(t, np), 0.0, hs=_HS_RCWA)
    assert _rel(_ad(f, 0.0, True), fd) < 1e-7
    assert _rel(_ad(f, 0.0, False), fd) > 1e-3


def test_r2v_rcwastack_two_patterned_layers_and_a_spacer():
    """RCWAStack (JAX): cross 6.25 (t on its +x half-arm) | spacer 2.1 |
    lossy cross 3.0+0.2j (0.5 t), n_orders 3 (7 x 7 = 49 orders), normal incidence.  ON
    5.2e-10 (Windows), OFF 0.21.  Bars 1e-7 and 1e-2."""
    from lumenairy.elements.rcwa import RCWAStack

    def f(t, xp):
        st = RCWAStack(1.3, period_y=1.3, n_superstrate=1.0, n_substrate=1.5,
                       n_orders=3, n_orders_y=3)
        st.add_layer(0.2, eps_cell=_cross(6.25 + 0j, xp, t))
        st.add_layer(0.1, eps=2.1)
        st.add_layer(0.15, eps_cell=_cross(3.0 + 0.2j, xp, 0.5 * t))
        st.set_source(1.0, theta=0.0, phi=0.0)
        _o, R, T = st.solve().efficiencies()
        return xp.concatenate([xp.ravel(R), xp.ravel(T)])
    fd = _fd(lambda t: f(t, np), 0.0, hs=_HS_RCWA)
    assert _rel(_ad(f, 0.0, True), fd) < 1e-7
    assert _rel(_ad(f, 0.0, False), fd) > 1e-2


@pytest.mark.parametrize("grids", ["shared", "per-layer"])
def test_r2v_pmmstack_lossy_three_region_layer_over_a_spacer(grids):
    """PMMStack (JAX), P 1.0 um: a layer [3.0 | 6.0+0.3j | 3.0 | 1.5]
    (mirror-symmetric about 0.25 P) over a uniform 2.1 spacer, n_sub 1.5,
    degree 8, wl 0.8 um; d / d(angle) at 0 of R, T, Re/Im J.  p10: ON
    5.7e-10 / 5.4e-10 (shared / per-layer), OFF 1.58 (Windows).  Bars 1e-7
    and (shared only, to keep the test short) 1e-2."""
    from lumenairy.elements.pmm import PMMStack

    def f(t, xp):
        st = PMMStack(1.0e-6, n_substrate=1.5, n_superstrate=1.0, degree=8,
                      layer_grids=grids)
        st.add_layer(0.2e-6, segments=[(0.15, 3.0), (0.2, 6.0 + 0.3j),
                                       (0.15, 3.0), (0.5, 1.5)])
        st.add_layer(0.1e-6, segments=[(1.0, 2.1)])
        st.set_source(0.8e-6, angle=t)
        _o, R, T, J = st.solve()
        J = xp.ravel(xp.asarray(J))
        return xp.concatenate([xp.ravel(xp.asarray(R)),
                               xp.ravel(xp.asarray(T)), xp.real(J),
                               xp.imag(J)])
    fd = _fd(lambda t: f(float(t), np), 0.0)
    assert _rel(_ad(f, 0.0, True), fd) < 1e-7
    if grids == "shared":
        assert _rel(_ad(f, 0.0, False), fd) > 1e-2


def test_r2v_hybrid_stack_two_patterned_layers_traced_layouts():
    """PMM2DStackHybrid ('laurent', degree 5, 3 orders): two patterned
    layers, each a TRACED eps_cell with a region layout and the same C4v
    centre (an eps-3 square, a lossy 4+0.2j square), t on one corner region
    of each.  FD of the concrete jitted forward on (1e-2, 3e-3, 1e-3).
    p10: ON 9.5e-11, OFF 0.096 (Windows).  Bars 1e-7 and 1e-2."""
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.pmm import PMM2DStackHybrid
    c1 = np.full((8, 8), 1.0 + 0j)
    c1[0:4, 0:4] = 3.0
    c2 = np.full((8, 8), 2.0 + 0j)
    c2[1:3, 1:3] = 4.0 + 0.2j
    lay1 = np.zeros((8, 8), np.int64)
    lay2 = np.zeros((8, 8), np.int64)
    for i in range(4):
        for j in range(4):
            lay1[i, j] = 1 + 4 * i + j
    for i in range(1, 3):
        for j in range(1, 3):
            lay2[i, j] = 2 * i + j - 2

    def solve(t):
        st = PMM2DStackHybrid(1.1e-6, n_substrate=1.5, n_superstrate=1.0,
                              degree=5, n_orders=3, formulation="laurent")
        st.add_layer(0.25e-6, eps_cell=jnp.asarray(c1).at[0, 0].add(t),
                     region_layout=lay1)
        st.add_layer(0.15e-6, eps_cell=jnp.asarray(c2).at[1, 1].add(0.7 * t),
                     region_layout=lay2)
        st.set_source(0.95e-6, theta=0.0)
        _o, R, T, J = st.solve()
        J = jnp.ravel(J)
        return jnp.concatenate([jnp.ravel(R), jnp.ravel(T), jnp.real(J),
                                jnp.imag(J)])
    fwd = jax.jit(solve)
    fd = _fd(lambda t: np.asarray(fwd(jnp.asarray(t))), 0.0,
             hs=(1e-2, 3e-3, 1e-3))

    def f(t, xp):
        return solve(t)
    assert _rel(_ad(f, 0.0, True), fd) < 1e-7
    assert _rel(_ad(f, 0.0, False), fd) > 1e-2


# ========================== claim (1): probes for NEW breakage (none found)
def test_r2v_a_symmetry_keeping_parameter_is_exact_with_the_rule_on_and_off():
    """t on ALL four arms of the cross (C4v kept: the clusters stay
    degenerate and the parameter does not split them).  ON and OFF agree
    (2.3e-14 relative on Windows, the lifted average of a non-splitting
    direction) and both match the FD (2.5e-10).  Bars 1e-9 / 1e-7."""
    from lumenairy.elements.rcwa import RCWAStack

    def f(t, xp):
        st = RCWAStack(1.3, period_y=1.3, n_superstrate=1.0, n_substrate=1.5,
                       n_orders=3, n_orders_y=3)
        st.add_layer(0.2, eps_cell=_cross(6.25 + 0j, xp, t,
                                          dirn=_CROSS.astype(float)))
        st.set_source(1.0, theta=0.0, phi=0.0)
        _o, R, T = st.solve().efficiencies()
        return xp.concatenate([xp.ravel(R), xp.ravel(T)])
    fd = _fd(lambda t: f(t, np), 0.0)
    g_on, g_off = _ad(f, 0.0, True), _ad(f, 0.0, False)
    assert _rel(g_on, g_off) < 1e-9
    assert _rel(g_on, fd) < 1e-7


def test_r2v_an_accidental_near_degeneracy_is_unharmed():
    """pmm_jones_1d at 0.2 rad (the builder's grating; build record 10.6:
    an accidental Mbig pair 3.1e-7 of the spectrum apart, no symmetry): the
    lifted branch runs, and changes nothing -- ON vs OFF 1e-15 class,
    both vs FD 2.0e-11 (Windows).  Bars 1e-9 / 1e-7."""
    from lumenairy.elements.pmm import pmm_jones_1d
    ER, EG = 4.0 * np.eye(3, dtype=complex), np.eye(3, dtype=complex)

    def f(t, xp):
        _o, R, T, _J = pmm_jones_1d(1.2, ER, EG, 1.0, 1.45, 0.45, 0.5, 1.0,
                                    angle=t, degree=12, stabilize=False)
        return xp.concatenate([xp.ravel(xp.asarray(R)),
                               xp.ravel(xp.asarray(T))])
    fd = _fd(lambda t: f(float(t), np), 0.2)
    g_on, g_off = _ad(f, 0.2, True), _ad(f, 0.2, False)
    assert _rel(g_on, g_off) < 1e-9
    assert _rel(g_on, fd) < 1e-7


def _uniform_over_cross(theta, phi, eps_u):
    from lumenairy.elements.rcwa import RCWAStack

    def f(t, xp):
        st = RCWAStack(1.3, period_y=1.3, n_superstrate=1.0, n_substrate=1.5,
                       n_orders=3, n_orders_y=3)
        st.add_layer(0.25, eps_cell=xp.asarray(np.full((_S, _S), eps_u))
                     + t * xp.asarray(_RND0))
        st.add_layer(0.2, eps_cell=_cross(4.0 + 0j, xp))
        st.set_source(1.0, theta=theta, phi=phi)
        _o, R, T = st.solve().efficiencies()
        return xp.concatenate([xp.ravel(R), xp.ravel(T)])
    return f


def test_r2v_eight_member_clusters_just_off_the_uniform_point():
    """A cluster with MORE than two members: a uniform eps 2.25 layer at
    normal incidence has the orders (+-1, 0), (0, +-1) x 2 polarizations
    (8 members) and (+-1, +-1) x 2 (8) degenerate; at t = +-1e-6 x a
    zero-mean random pattern they split by ~1e-6 x the pattern, INSIDE the
    rule's gap_rel, so the lifted branch handles 8-member clusters.  The
    mean of AD(+1e-6) and AD(-1e-6) vs the FD at 0: 4.1e-9 (Windows).
    Bar 1e-7.  (AD AT t = 0 is a defect: the strict xfail below.)"""
    f = _uniform_over_cross(0.0, 0.0, 2.25 + 0j)
    fd = _fd(lambda t: f(t, np), 0.0, hs=_HS_RCWA)
    g = 0.5 * (_ad(f, 1e-6) + _ad(f, -1e-6))
    assert _rel(g, fd) < 1e-7


# ============================== claim (1): vmap / jacrev / eager / hessian
def test_r2v_vmapped_gradient_on_a_mixed_batch_with_the_rule_on_and_off():
    """One-layer Berreman (oblique 0.35, lossy), batch d = [0, 1e-2, 2e-2]
    of a weighted sum: ON, every member equals the FD (<= 1e-10 class,
    Windows 1.9e-11 .. 8.8e-11 on the p5 fixture); OFF, the symmetric
    member is wrong (1.65 on p5) while the others are right.  Bars 1e-7 and
    1e-2."""
    import jax
    import jax.numpy as jnp

    from lumenairy.backend import jax_cluster_rule
    f = _berreman(0.35, n_layers=1)
    w = np.arange(1.0, 21.0)

    def loss(d):
        return jnp.sum(f(d, jnp) * w)
    ds = (0.0, 1e-2, 2e-2)
    fds = [float(np.sum(_fd(lambda d: f(d, np), x) * w)) for x in ds]
    out = {}
    for on in (True, False):
        with jax_cluster_rule(on):
            out[on] = np.asarray(jax.jit(jax.vmap(jax.grad(loss)))(
                jnp.asarray(ds)))
    err = {on: [abs(a - b) / abs(b) for a, b in zip(out[on], fds)]
           for on in out}
    assert max(err[True]) < 1e-7, err
    assert err[False][0] > 1e-2 and max(err[False][1:]) < 1e-7, err


def test_r2v_eager_jacrev_at_the_symmetric_point():
    """EAGER (non-jit) jax.jacrev of the two-layer Berreman fixture at 0:
    the first (recording) pass sees the concrete second layer as isotropic
    and takes the analytic branch, the traced record / replay take the eig
    branch (``_jax_cluster_routed``'s concreteness note) -- the gradient
    must still match the FD.  Bar 1e-7; measured as the jitted one."""
    import jax
    import jax.numpy as jnp
    f = _berreman(0.35)
    fd = _fd(lambda t: f(t, np), 0.0)
    g = np.asarray(jax.jacrev(lambda t: f(t, jnp))(jnp.asarray(0.0)))
    assert _rel(g, fd) < 1e-7


def test_r2v_hessian_scope_is_as_the_changelog_states():
    """CHANGELOG scope note: ``jax.hessian`` works for a parameter that does
    not enter an eig (a thickness) and raises NotImplementedError for one
    that does (here the in-plane anisotropy) -- reverse mode only."""
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.berreman import berreman_jones_1d
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], complex)

    def loss(d, thk):
        e = (3.0 + 0.1j) * jnp.eye(3, dtype=complex) + d * jnp.asarray(X)
        R, T, _Jr, _Jt = berreman_jones_1d(
            [(e, thk)], jnp.asarray(1.5 + 0j), jnp.asarray(1.0 + 0j),
            jnp.asarray(0.9e-6), angle=0.35)
        return jnp.sum(R)
    h = jax.hessian(lambda thk: loss(0.0, thk))(jnp.asarray(0.25e-6))
    assert np.isfinite(float(h))
    with pytest.raises(NotImplementedError):
        jax.hessian(lambda d: loss(d, jnp.asarray(0.25e-6)))(0.0)


# ====================================== claim B: the record / replay helper
def _toy():
    """A basis-invariant consumer of an eig (the matrix exponential) at a
    non-normal matrix with two exactly degenerate pairs; t splits them."""
    import scipy.linalg as sla
    rng = np.random.default_rng(5)
    lam0 = np.diag([1.0, 1.0, 2.0, 3.5, 3.5]).astype(complex)
    Q = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    A0 = Q @ lam0 @ np.linalg.inv(Q)
    C = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    W = rng.standard_normal((5, 5))

    def truth(t):
        return float(np.real(np.sum(sla.expm(A0 + t * C) * W)))
    h = 1e-4
    g = (8 * (truth(h) - truth(-h)) - (truth(2 * h) - truth(-2 * h))) / (
        12 * h)
    return A0, C, W, g


def _expm(A):
    import jax.numpy as jnp

    import lumenairy.elements.rcwa._core as RC
    lam, V = RC._jax_twin_eig(A)
    return V @ jnp.diag(jnp.exp(lam)) @ jnp.linalg.inv(V)


def _routed_toy(A0, C, W, hook=None):
    import jax.numpy as jnp

    import lumenairy.elements.rcwa._core as RC

    def f(t):
        A = jnp.asarray(A0) + t * jnp.asarray(C)

        def solve():
            if hook is not None:
                hook()
            return jnp.real(jnp.sum(_expm(A) * W))
        return RC._jax_cluster_routed(solve)
    return f


def test_r2v_routed_helper_on_a_toy_and_nested():
    """The helper itself, on a 5 x 5 non-normal matrix with two exact pairs
    and the matrix exponential as consumer: ON 7.0e-10 relative (jit and
    eager), OFF 1.31 (both builds).  A routed solve NESTED inside another
    routed solve: 7.3e-10.  Bars 1e-7 / 1e-2."""
    import jax
    import jax.numpy as jnp
    import scipy.linalg as sla

    import lumenairy.elements.rcwa._core as RC
    from lumenairy.backend import jax_cluster_rule
    A0, C, W, g = _toy()
    f = _routed_toy(A0, C, W)
    assert abs(float(jax.jit(jax.grad(f))(0.0)) - g) < 1e-7 * abs(g)
    assert abs(float(jax.grad(f)(0.0)) - g) < 1e-7 * abs(g)
    with jax_cluster_rule(False):
        assert abs(float(jax.jit(jax.grad(f))(0.0)) - g) > 1e-2 * abs(g)
    B0 = np.diag([2.0, 2.0, 0.5]).astype(complex)
    CB = np.random.default_rng(6).standard_normal((3, 3)) + 0j

    def nested(t):
        A = jnp.asarray(A0) + t * jnp.asarray(C)
        B = jnp.asarray(B0) + t * jnp.asarray(CB)

        def inner():
            return jnp.real(jnp.sum(_expm(B)))

        def outer():
            x = RC._jax_cluster_routed(inner)
            return jnp.real(jnp.sum(_expm(A) * W)) + x ** 2
        return RC._jax_cluster_routed(outer)

    def nested_np(t):
        return (float(np.real(np.sum(sla.expm(A0 + t * C) * W)))
                + float(np.real(np.sum(sla.expm(B0 + t * CB)))) ** 2)
    h = 1e-4
    gn = (8 * (nested_np(h) - nested_np(-h))
          - (nested_np(2 * h) - nested_np(-2 * h))) / (12 * h)
    assert abs(float(jax.jit(jax.grad(nested))(0.0)) - gn) < 1e-7 * abs(gn)
    assert RC._JAX_TWIN_EIG_ROUTE.get() is None


@pytest.mark.parametrize("bad_pass", [1, 2, 3])
def test_r2v_an_exception_in_any_pass_leaves_no_route_behind(bad_pass):
    """An exception raised by the solve in its recording pass (1), its
    traced re-recording (2) or the replay (3) propagates, the route context
    variable is reset, and the next routed gradient is exact."""
    import jax

    import lumenairy.elements.rcwa._core as RC
    A0, C, W, g = _toy()
    calls = [0]

    def hook():
        calls[0] += 1
        if calls[0] == bad_pass:
            raise ValueError("boom")
    with pytest.raises(ValueError, match="boom"):
        jax.jit(jax.grad(_routed_toy(A0, C, W, hook)))(0.0)
    assert RC._JAX_TWIN_EIG_ROUTE.get() is None
    f = _routed_toy(A0, C, W)
    assert abs(float(jax.jit(jax.grad(f))(0.0)) - g) < 1e-7 * abs(g)


def test_r2v_fewer_eigs_on_the_replay_raise_the_named_error():
    """A solve that takes one eig FEWER on its replay than in its traced
    recording (state outside the trace) is refused loudly with the helper's
    own RuntimeError."""
    import jax
    import jax.numpy as jnp

    import lumenairy.elements.rcwa._core as RC
    A0, C, W, _g = _toy()
    calls = [0]

    def f(t):
        A = jnp.asarray(A0) + t * jnp.asarray(C)

        def solve():
            calls[0] += 1
            n = 1 if calls[0] == 3 else 2
            return sum(jnp.real(jnp.sum(_expm(A) * W)) for _ in range(n))
        return RC._jax_cluster_routed(solve)
    with pytest.raises(RuntimeError, match="eigs on replay"):
        jax.jit(jax.grad(f))(0.0)


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT P3-1 (verifier r2): a solve that takes MORE eigs on the replay "
    "than recorded surfaces as a bare StopIteration from next(it) in "
    "_jax_cluster_routed's replay (rcwa/_core.py:5090), not the helper's "
    "named RuntimeError; flips when the replay raises that error itself"))
def test_r2v_more_eigs_on_the_replay_raise_the_named_error():
    import jax
    import jax.numpy as jnp

    import lumenairy.elements.rcwa._core as RC
    A0, C, W, _g = _toy()
    calls = [0]

    def f(t):
        A = jnp.asarray(A0) + t * jnp.asarray(C)

        def solve():
            calls[0] += 1
            n = 3 if calls[0] == 3 else 2
            acc = 0.0
            for _ in range(n):
                acc = acc + jnp.real(jnp.sum(_expm(A) * W))
            return acc
        return RC._jax_cluster_routed(solve)
    with pytest.raises(RuntimeError, match="eigs on replay"):
        jax.jit(jax.grad(f))(0.0)


def test_r2v_an_eig_inside_an_inner_vmap_or_scan_is_refused_loudly():
    """A solve whose eig sits inside its OWN jax.vmap / lax.scan would hand
    the recorder a tracer of that inner transform; it must fail loudly (it
    does: a leaked-trace error), never return a silent gradient."""
    import jax
    import jax.numpy as jnp

    import lumenairy.elements.rcwa._core as RC
    A0, C, W, _g = _toy()
    for kind in ("vmap", "scan"):
        def f(t, kind=kind):
            As = jnp.stack([jnp.asarray(A0) + t * jnp.asarray(C)] * 2)

            def solve():
                if kind == "vmap":
                    Es = jax.vmap(_expm)(As)
                else:
                    Es = jax.lax.scan(lambda c, A: (c, _expm(A)), 0.0, As)[1]
                return jnp.real(jnp.sum(Es[0] * W))
            return RC._jax_cluster_routed(solve)
        with pytest.raises(Exception, match="[Ll]eak"):
            jax.jit(jax.grad(f))(0.0)
        assert RC._JAX_TWIN_EIG_ROUTE.get() is None


def test_r2v_threads_tracing_routed_solves_concurrently():
    """The route lives in a ContextVar: four threads tracing jit(grad) of
    the routed toy at once each get the exact gradient."""
    import jax
    A0, C, W, g = _toy()
    res = {}

    def work(i):
        try:
            res[i] = float(jax.jit(jax.grad(_routed_toy(A0, C, W)))(0.0))
        except Exception as e:  # noqa: BLE001
            res[i] = e
    th = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert all(isinstance(v, float) and abs(v - g) < 1e-7 * abs(g)
               for v in res.values()), res


# ============================================ claims (3), (4), (5)
def _tens_aniso(xp, tilt=False):
    e = np.where(_CROSS, 5.0 + 0.2j, 1.44 + 0j)
    t = e[:, :, None, None] * np.eye(3, dtype=complex)[None, None]
    t = t.copy()
    t[:, :, 1, 1] *= 0.8
    t[:, :, 0, 1] = t[:, :, 1, 0] = np.where(_CROSS, 0.3, 0.0)
    if tilt:
        t[:, :, 0, 2] = t[:, :, 2, 0] = np.where(_CROSS, 0.4, 0.0)
    return xp.asarray(t)


@pytest.mark.parametrize("tilt", [False, True])
def test_r2v_traced_li_route_equals_the_numpy_li_route_conical(tilt):
    """rcwa_jones_2d(formulation='li'), conical (0.2, 0.3), n_orders 3 (7 x 7 = 49 orders):
    an IN-PLANE anisotropic cell with eps_xy = 0.3 (so the off-diagonal
    blocks matter) -- the jitted call equals NumPy 'li' (5.2e-15, Windows;
    it was the Laurent answer, 6.3e-3 away, on the base tree); an
    OUT-OF-PLANE (eps_xz = 0.4) cell -- jitted = NumPy 'li' (2.3e-14) and
    NumPy 'li' = NumPy 'laurent' (the general path is the direct rule for
    both, as on the base tree).  Bar 1e-12 / 1e-11; the 'li' vs 'laurent'
    gap for the in-plane cell (> 1e-4) shows the comparison can see it."""
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.rcwa import rcwa_jones_2d

    def call(t, form):
        return rcwa_jones_2d(1.3, 1.3, t, 1.5, 1.0, 0.3, 1.0, theta=0.2,
                             phi=0.3, n_orders_x=3, n_orders_y=3,
                             formulation=form)

    def pack(r):
        return np.concatenate([np.ravel(np.asarray(a)) for a in r[1:]])
    li_np = pack(call(_tens_aniso(np, tilt), "li"))
    lau_np = pack(call(_tens_aniso(np, tilt), "laurent"))
    li_jit = pack(jax.jit(lambda t: call(t, "li"))(_tens_aniso(jnp, tilt)))
    if tilt:
        assert np.max(np.abs(li_jit - li_np)) < 1e-11
        assert np.max(np.abs(li_np - lau_np)) == 0.0
    else:
        assert np.max(np.abs(li_jit - li_np)) < 1e-12
        assert np.max(np.abs(li_np - lau_np)) > 1e-4


def test_r2v_rcwastack_cache_leak_sequence_is_fixed():
    """Claim (4), own fixture (cross 6.25 + t ARMX, theta 0.1): jit(f), then
    jit(grad(f)), then eager f.  On the base tree 7c0bc8bd the second and
    third raise UnexpectedTracerError and the module cache holds tracers
    (reproduced, both builds); on HEAD they agree (5.6e-17) and the cache
    holds none."""
    import jax
    import jax.numpy as jnp

    import lumenairy.elements.rcwa._core as RC
    from lumenairy.elements.rcwa import RCWAStack
    RC._clear_rcwa_caches()

    def f(t):
        st = RCWAStack(1.3, period_y=1.3, n_superstrate=1.0, n_substrate=1.5,
                       n_orders=3, n_orders_y=3)
        st.add_layer(0.2, eps_cell=_cross(6.25 + 0j, jnp, t))
        st.set_source(1.0, theta=0.1, phi=0.0)
        return jnp.sum(st.solve().efficiencies()[1])
    v = float(jax.jit(f)(1e-2))
    assert np.isfinite(float(jax.jit(jax.grad(f))(1e-2)))
    assert abs(float(f(1e-2)) - v) < 1e-12
    assert not any(isinstance(a, jax.core.Tracer)
                   for val in RC._HOMOG_CACHE.values()
                   for a in (val if isinstance(val, tuple) else (val,)))


def _enabled_in_subprocess(pre, post=""):
    code = (f"import os,sys;sys.path.insert(0,{_REPO!r});{pre}"
            "import lumenairy.backend as B;" + post +
            "print(B.jax_cluster_rule_enabled())")
    env = {k: v for k, v in os.environ.items()
           if k != "LUMENAIRY_JAX_CLUSTER_RULE"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, stdin=subprocess.DEVNULL,
                       timeout=300)
    return r.stdout.strip().splitlines()[-1]


def test_r2v_env_var_sets_the_process_default_at_import_only():
    """LUMENAIRY_JAX_CLUSTER_RULE is read ONCE, at import (the process
    default): '0' / 'OFF' before import turn the rule off; setting it after
    import changes nothing (the setter is the runtime control)."""
    assert _enabled_in_subprocess(
        "os.environ['LUMENAIRY_JAX_CLUSTER_RULE']='0';") == "False"
    assert _enabled_in_subprocess(
        "os.environ['LUMENAIRY_JAX_CLUSTER_RULE']='OFF';") == "False"
    assert _enabled_in_subprocess(
        "", "os.environ['LUMENAIRY_JAX_CLUSTER_RULE']='0';") == "True"


def test_r2v_context_manager_restores_on_exception_and_nesting():
    from lumenairy.backend import jax_cluster_rule, jax_cluster_rule_enabled
    with pytest.raises(KeyError):
        with jax_cluster_rule(False):
            assert not jax_cluster_rule_enabled()
            raise KeyError("x")
    assert jax_cluster_rule_enabled()
    with jax_cluster_rule(False):
        with jax_cluster_rule(True):
            assert jax_cluster_rule_enabled()
        assert not jax_cluster_rule_enabled()
    assert jax_cluster_rule_enabled()


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT P3-2 (verifier r2): the switch is one module global "
    "(backend/array.py _JAX_CLUSTER_RULE) and the context manager restores "
    "the value it saw on entry, so two scopes left out of order -- what two "
    "THREADS each using `with jax_cluster_rule(...)` do -- leave the process "
    "with the rule OFF; flips if the setting becomes context-local (or the "
    "scopes otherwise cannot clobber each other)"))
def test_r2v_interleaved_switch_scopes_do_not_leave_the_rule_off():
    from lumenairy.backend import jax_cluster_rule, jax_cluster_rule_enabled
    a, b = jax_cluster_rule(False), jax_cluster_rule(True)
    a.__enter__()
    b.__enter__()
    a.__exit__(None, None, None)
    b.__exit__(None, None, None)
    assert jax_cluster_rule_enabled()


def test_r2v_a_jitted_function_keeps_its_setting_per_compiled_signature():
    """What 'a jax.jit-compiled function keeps the setting it was compiled
    with' means, measured (one-layer Berreman, symmetric point): the SAME
    compiled signature keeps ON after the switch goes OFF (exact, 9.4e-9 on
    p5), but the same jitted function RETRACED for a new input dtype under
    OFF takes OFF (wrong, 1.6).  Bars 1e-7 / 1e-2."""
    import jax
    import jax.numpy as jnp

    from lumenairy.backend import jax_cluster_rule
    f = _berreman(0.35, n_layers=1)
    fd = _fd(lambda d: f(d, np), 0.0)
    g = jax.jit(jax.jacrev(lambda d: f(d, jnp)))
    with jax_cluster_rule(True):
        assert _rel(np.asarray(g(0.0)), fd) < 1e-7
    with jax_cluster_rule(False):
        assert _rel(np.asarray(g(0.0)), fd) < 1e-7
        g32 = np.asarray(g(jnp.asarray(0.0, jnp.float32))).astype(float)
        assert _rel(g32, fd) > 1e-2


# =================================================== claim (7): controls
def test_r2v_controls_rcwa_1d_and_bor_are_exact_at_their_symmetric_points():
    """Not routed, measured correct: rcwa_efficiency_1d (lossy ridge
    2.6+0.05j, 13 orders) d / d(angle) at exactly 0 -- 7.3e-11 TE; BORStack
    (R 2.5, m = 2, N 48, ring 0.7 / 0.4) d / d(n_ring) at the radially
    homogeneous point n_ring = n_g = 2.2 -- 2.3e-10.  Identical bytes (value
    and gradient) on the base tree, both builds (p8).  Bar 1e-7."""
    import warnings

    import jax.numpy as jnp

    from lumenairy.elements.bor.bor_stack import BORStack
    from lumenairy.elements.rcwa import rcwa_efficiency_1d

    def r1(t, xp):
        _o, R, T = rcwa_efficiency_1d(0.9, xp.asarray(2.6 + 0.05j), 1.2, 1.5,
                                      1.0, 0.35, 0.45, 1.0, angle=t,
                                      polarization="te", n_orders=13)
        return xp.concatenate([xp.ravel(R), xp.ravel(T)])
    assert _rel(_ad(r1, 0.0), _fd(lambda t: r1(float(t), np), 0.0)) < 1e-7

    def bor(nr, xp):
        s = BORStack(2.5, 2, N=48, n_superstrate=1.3, n_substrate=1.6)
        s.add_layer(0.4, rings=(0.7, 0.4, nr, 2.2))
        s.set_source(wavelength=2 * np.pi / 2.4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = s.solve()
        return xp.concatenate([xp.ravel(xp.asarray(r["R"])),
                               xp.ravel(xp.asarray(r["T"]))])
    fd = _fd(lambda t: np.asarray(bor(jnp.asarray(t), jnp)), 2.2)
    assert _rel(_ad(bor, 2.2), fd) < 1e-7


# ====================================================== DEFECTS (strict)
_RA_ER = 4.0 * np.eye(3, dtype=complex)


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT P1-1 (verifier r2): _jax_cluster_routed passes ONE anchor (0) "
    "for every problem (rcwa/_core.py:5104-5106; Kx2 eig pmm/_core.py:4357, "
    "routed at pmm/_core.py:4507, pmm/_jax_stack.py:459/746), but the "
    "half-space geometric eig Kx2 of "
    "pmm_jones_1d / PMMStack feeds q = sqrt(eps_half - mu), non-smooth at "
    "mu = eps_half (a Rayleigh anomaly).  At normal incidence the +-m pair "
    "of Kx2 is a cluster and the unshortened lift crosses that branch "
    "point: d/dangle wrong by 1.8e-5 / 1.7e-3 / 19 at 0.1 / 0.03 / 0.01 % "
    "from the m = 1 anomaly (both builds); FD-free: the mirror identity "
    "dR_{+1} = -dR_{-1} broken by 0.50 (FD: 1.1e-7).  Patching the anchors "
    "to (0, eps_sub, eps_sup) gives 1.2e-6 -- flips when the Kx2 problem "
    "carries its own anchors"))
def test_r2v_pmm_jones_1d_mirror_identity_near_a_rayleigh_anomaly():
    """P 1.2, ridge 4 / groove 1, n_sub 1.0, n_sup 1.45, degree 12, wl =
    1.2 (1 + 1e-4): the m = +-1 substrate orders are 1e-4 from grazing.
    The grating is mirror-symmetric, so at exactly normal incidence
    d X_{+1} / d angle = - d X_{-1} / d angle (X = R, T, both incident
    polarizations).  Bar 1e-4 of the largest entry: the FD obeys it to
    1.1e-7, the anchor-patched rule to ~1e-6; AD today 0.50."""
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.pmm import pmm_jones_1d
    wl = 1.2 * (1.0 + 1e-4)
    o = np.asarray(pmm_jones_1d(1.2, _RA_ER, np.eye(3, dtype=complex), 1.0,
                                1.45, 0.45, 0.5, wl, degree=12,
                                stabilize=False)[0])
    idx = [int(np.nonzero(o == m)[0][0]) for m in (1, -1)]

    def f(t):
        _o, R, T, _J = pmm_jones_1d(1.2, _RA_ER, np.eye(3, dtype=complex),
                                    1.0, 1.45, 0.45, 0.5, wl, angle=t,
                                    degree=12, stabilize=False)
        return jnp.concatenate([
            jnp.stack([R[p][i] for p in (0, 1) for i in idx]),
            jnp.stack([T[p][i] for p in (0, 1) for i in idx])])
    g = np.asarray(jax.jit(jax.jacrev(f))(0.0))
    assert np.max(np.abs(g[0::2] + g[1::2])) < 1e-4 * np.max(np.abs(g))


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT P1-2 (verifier r2; pre-existing, identical on the base tree "
    "7c0bc8bd): RCWAStack (JAX) at an EXACTLY uniform scalar eps_cell "
    "layer differentiated in a pattern direction.  rcwa/_core.py:2785-2792 "
    "(_layer_eigenmodes) selects the analytic uniform modes with "
    "xp.where(uniform, ...), so at t = 0 the gradient flows through W = I "
    "and EPS[0, 0] only and misses the first-order coupling of the pattern "
    "to the other layer's diffracted orders; the eig branch, which the "
    "cluster rule now makes exact, is discarded.  Measured: AD(0) vs FD "
    "8.5 (normal) / 5.7 (conical 0.2, 0.3), both builds; AD(+-1e-6) vs FD "
    "4e-9 / 2e-9.  Flips when the uniform select stops severing the "
    "gradient (e.g. the eig branch through the cluster rule)"))
def test_r2v_rcwastack_uniform_layer_pattern_gradient_at_the_uniform_point():
    """Topology optimisation from a uniform start: a uniform eps 2.25 layer
    (t x a zero-mean random pattern) over a cross layer, normal incidence.
    The gradient AT t = 0 must equal the limit of the gradients next to it
    (mean of AD(+-1e-6), itself = FD to 4e-9).  Bar 1e-5 relative; today
    8.5."""
    f = _uniform_over_cross(0.0, 0.0, 2.25 + 0j)
    g0 = _ad(f, 0.0)
    g_near = 0.5 * (_ad(f, 1e-6) + _ad(f, -1e-6))
    assert _rel(g0, g_near) < 1e-5


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT P1-1, second family: PMMStack shares the half-space geometric "
    "Kx2 eig (pmm/_jax_stack.py) and its anchor-0 lift: near the m = 1 "
    "substrate Rayleigh anomaly d/dangle at normal incidence is wrong by "
    "3.4e-3 at 0.1 % and 3.3 at 0.01 % (p10b, Windows); mirror identity "
    "broken by 6.6e-2 (FD 4.7e-8)"))
def test_r2v_pmmstack_mirror_identity_near_a_rayleigh_anomaly():
    """PMMStack, P 1 um, layer [3 | 6+0.3j | 3 | 1.5] (mirror about 0.25 P)
    over a 2.1 spacer, n_sup 1.45, n_sub 1.0, degree 10, wl = 1 um
    (1 + 1e-4).  Mirror identity of the +-1 orders at normal incidence, bar
    1e-4 of the largest entry (FD 4.7e-8; AD today 6.6e-2)."""
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.pmm import PMMStack
    wl = 1.0e-6 * (1.0 + 1e-4)

    def solve(t):
        st = PMMStack(1.0e-6, n_substrate=1.0, n_superstrate=1.45, degree=10)
        st.add_layer(0.2e-6, segments=[(0.15, 3.0), (0.2, 6.0 + 0.3j),
                                       (0.15, 3.0), (0.5, 1.5)])
        st.add_layer(0.1e-6, segments=[(1.0, 2.1)])
        st.set_source(wl, angle=t)
        return st.solve()
    o = np.asarray(solve(0.0)[0])
    idx = [int(np.nonzero(o == m)[0][0]) for m in (1, -1)]

    def f(t):
        _o, R, T, _J = solve(t)
        return jnp.concatenate([
            jnp.stack([R[p][i] for p in (0, 1) for i in idx]),
            jnp.stack([T[p][i] for p in (0, 1) for i in idx])])
    g = np.asarray(jax.jit(jax.jacrev(f))(0.0))
    assert np.max(np.abs(g[0::2] + g[1::2])) < 1e-4 * np.max(np.abs(g))


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT P2-1 (verifier r2): set_jax_cluster_rule's docstring says to "
    "'re-create the jitted function after changing it', but jax.jit caches "
    "the compiled program by the wrapped CALLABLE, so a new jax.jit wrapper "
    "of the same callable reuses the program traced under the old setting: "
    "compiled OFF, switched back ON and re-wrapped, the gradient at a "
    "symmetric point stays WRONG (1.6, both builds; p11).  Flips when the "
    "setting is part of JAX's trace cache key (or the re-wrap otherwise "
    "takes the new setting)"))
def test_r2v_rewrapping_a_jitted_callable_takes_the_new_setting():
    import jax
    import jax.numpy as jnp

    from lumenairy.backend import jax_cluster_rule
    f = _berreman(0.35, n_layers=1)
    fd = _fd(lambda d: f(d, np), 0.0)
    g = jax.jacrev(lambda d: f(d, jnp))
    with jax_cluster_rule(False):
        jax.jit(g)(0.0)
    with jax_cluster_rule(True):
        assert _rel(np.asarray(jax.jit(g)(0.0)), fd) < 1e-7
