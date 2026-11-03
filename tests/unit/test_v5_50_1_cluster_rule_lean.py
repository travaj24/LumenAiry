"""The lean path of the degenerate-cluster gradient rule (L-2 of the DynaMeta
bridge audit, 2026-11-02; ``rcwa._core``, "The closure conversions of the
rule, without their per-call overhead").

5.50.0 ran every eager gradient through the rule with ``jax.closure_convert``
under ``jax.checking_leaks()`` (19% of the GATE D gradient) and evaluated the
re-record jaxpr in full (a whole extra solve, eigs included), then decomposed
every matrix a third time in the rule's forward, and its reverse pass built a
fresh ``lax.cond`` per eager gradient (a new XLA compilation and ~19 MB of
RSS held by JAX's primitive cache each time, on every RCWA solve).  The lean
path takes the same uncached conversion without leak checking, drops dead
WHOLE equations eagerly only (every closed-over constant kept), shares the
eager eigs (``_JaxEigMemo``) and branches in Python on a concrete predicate.
The branch evaluates op by op instead of as one compiled program, so eager
gradients are no longer bit-identical to 5.50.0: they move by round-off,
which a firing cluster's lifted stencil amplifies by ~1 / split_rel -- the
same amount by which 5.50.0's OWN eager and jit gradients differ (up to
9e-9 relative on the symmetric cell, ~1e-13 elsewhere).  The 5.50.0 path
stays reachable (``_CLUSTER_RULE_LEAN = False``) and every case compares the
two in one process:

* ``jax.jit``: the LOWERED PROGRAM TEXT is identical (identical program ->
  identical compiled code), and the values are bit-identical;
* eager ``jax.grad``, eager ``jax.vmap`` of it, and the lean path's eager
  vs jit: within 4x 5.50.0's own eager-vs-jit spread (1e-14 floor).

Cases: a four-fold RCWA cell at its symmetric point (the cluster FIRES and the
lifted branch runs), the 1-D ``PMMStack`` at normal incidence (a pencil with a
lift Gram, cluster fires), the Berreman cascade with a concrete isotropic
layer (the case that needs the re-record: the eager first pass takes the
analytic branch, the replay the eig -- whose isotropic layer is a degenerate
cluster, so the lifted branch runs), an RCWA stack whose second patterned
layer is uniform (the analytic-shortcut select: ``has_select``; the uniform
layer is a cluster too), DynaMeta's GATE D slab (no cluster at all), and an
RCWA stack with two patterned layers of the SAME cell shape (a memo keyed on
shape and dtype alone would hand the second layer the first one's eigs).
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


@pytest.fixture(autouse=True, scope="module")
def _jax_x64_for_this_module():
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


@pytest.fixture()
def lean_switch():
    from lumenairy.elements.rcwa import _core
    previous = _core._CLUSTER_RULE_LEAN
    _core._CLUSTER_RULE_LEAN = True
    # every lean-path test must exercise the LEAN path: a failed self-test
    # would make it the 5.50.0 path and these tests vacuous
    assert _core._lean_impl(), "the lean path fell back (self-test failed)"
    assert _core._LEAN_SELFTEST is True
    yield _core
    _core._CLUSTER_RULE_LEAN = previous


_BASE3 = np.array([[1.0, 1.5, 1.0], [1.5, 4.0, 1.5], [1.0, 1.5, 1.0]])
_XS3 = np.zeros((3, 3))
_XS3[0, 1] = _XS3[2, 1] = 1.0
_BASE, _DIRN = (np.kron(a, np.ones((3, 3))) for a in (_BASE3, _XS3))
_D3 = np.zeros((3, 3))
_D3[0, 0] = _D3[2, 2] = 1.0
_DIAG = np.kron(_D3, np.ones((3, 3)))   # breaks the four-fold symmetry too
_PIL = np.zeros((9, 9))
_PIL[2:6, 3:7] = 1.0


def _sym_rcwa(t, terms=False):
    import jax.numpy as jnp

    from lumenairy.elements.rcwa import RCWAStack
    st = RCWAStack(1.2, period_y=1.2, n_superstrate=1.0, n_substrate=1.45,
                   n_orders=2, n_orders_y=2)
    st.add_layer(0.45, eps_cell=(jnp.asarray(_BASE)
                                 + t * jnp.asarray(_DIRN)).astype(complex))
    st.add_layer(0.1, eps=2.25)
    st.set_source(1.0, theta=0.0, phi=0.0)
    _o, R, T = st.solve().efficiencies()
    w = jnp.asarray(np.linspace(0.3, 1.0, R.shape[1]))
    if terms:
        return jnp.concatenate([R[0] * w, R[1] * w[::-1], 0.5 * T[0] * w])
    return jnp.sum(R[0] * w) + jnp.sum(R[1] * w[::-1]) + 0.5 * jnp.sum(T[0] * w)


def _pmm_stack(a, terms=False):
    import jax.numpy as jnp

    from lumenairy.elements.pmm import PMMStack
    st = PMMStack(1.2e-6, n_substrate=1.0, n_superstrate=1.45, degree=8)
    st.add_layer(0.3e-6, segments=[(0.5, 4.0), (0.5, 1.0)])
    st.add_layer(0.1e-6, segments=[(1.0, 2.25)])
    st.set_source(1.0e-6, angle=a)
    _o, R, T, _J = st.solve()
    w = jnp.asarray(np.linspace(0.2, 1.0, R.shape[1]))
    if terms:
        return jnp.concatenate([R[0] * w, T[1] * w[::-1]])
    return jnp.sum(R[0] * w) + jnp.sum(T[1] * w[::-1])


def _berreman_mixed(re, terms=False):
    import jax.numpy as jnp

    from lumenairy import berreman_jones_1d
    th = np.deg2rad(35.0)
    no2, ne2 = 1.5 ** 2, 1.75 ** 2
    c, s = np.cos(th), np.sin(th)
    M = np.array([[ne2 * c * c + no2 * s * s, (ne2 - no2) * c * s, 0],
                  [(ne2 - no2) * c * s, ne2 * s * s + no2 * c * c, 0],
                  [0, 0, no2]], complex)
    e = jnp.asarray(M).at[0, 0].set(re + 0.05j)
    R, T, _Jr, _Jt = berreman_jones_1d(
        [(e, jnp.asarray(180e-9)), (jnp.asarray(2.1 + 0j),
                                    jnp.asarray(100e-9))],
        jnp.asarray(1.5 + 0j), jnp.asarray(1.0 + 0j), jnp.asarray(1.55e-6),
        angle=jnp.asarray(0.2))
    if terms:
        return jnp.stack([R[0], 0.5 * T[1]])
    return R[0] + 0.5 * T[1]


def _berreman_uniaxial(ne, terms=False):
    """DynaMeta's lumenairy_berreman_jax GATE D slab (no cluster)."""
    import jax.numpy as jnp

    from lumenairy import berreman_jones_1d
    eps = jnp.asarray([[ne ** 2, 0.0, 0.0], [0.0, 2.25, 0.0], [0.0, 0.0, 2.25]],
                      dtype=jnp.complex128)
    R, _T, _Jr, _Jt = berreman_jones_1d([(eps, 220e-9)], 1.5 + 0j, 1.0 + 0j,
                                        jnp.asarray(1.55e-6), angle=0.0,
                                        phi=0.0)
    if terms:
        return jnp.reshape((jnp.real(R[0]) - 0.05) ** 2, (1,))
    return (jnp.real(R[0]) - 0.05) ** 2


def _rcwa_uniform(p, terms=False):
    import jax.numpy as jnp

    from lumenairy.elements.rcwa import RCWAStack
    cell = (1.0 + (p - 1.0) * jnp.asarray(_PIL)).astype(jnp.complex128)
    uni = (jnp.ones((9, 9)) * (2.0 + 0.1 * p)).astype(jnp.complex128)
    st = RCWAStack(1.2, period_y=1.2, n_superstrate=1.0, n_substrate=1.45,
                   n_orders=2, n_orders_y=2)
    st.add_layer(0.45, eps_cell=cell)
    st.add_layer(0.1, eps_cell=uni)
    st.set_source(1.0, theta=0.1, phi=0.2)
    _o, R, T = st.solve().efficiencies()
    if terms:
        return jnp.concatenate([R[0], 0.5 * T[1], 0.25 * R[1]])
    return jnp.sum(R[0]) + 0.5 * jnp.sum(T[1]) + 0.25 * jnp.sum(R[1])


def _rcwa_two_layers(p, terms=False):
    import jax.numpy as jnp

    from lumenairy.elements.rcwa import RCWAStack
    cell = (1.0 + (p - 1.0) * jnp.asarray(_PIL)).astype(jnp.complex128)
    cell2 = (2.25 + (p - 2.0) * jnp.asarray(_PIL.T)).astype(jnp.complex128)
    st = RCWAStack(1.2, period_y=1.2, n_superstrate=1.0, n_substrate=1.45,
                   n_orders=2, n_orders_y=2)
    st.add_layer(0.3, eps_cell=cell)
    st.add_layer(0.2, eps_cell=cell2)
    st.set_source(1.0, theta=0.1, phi=0.2)
    _o, R, T = st.solve().efficiencies()
    if terms:
        return jnp.concatenate([R[0], 0.5 * T[1], 0.25 * R[1]])
    return jnp.sum(R[0]) + 0.5 * jnp.sum(T[1]) + 0.25 * jnp.sum(R[1])


def _close(a, b):
    """Every element within 4 ulp or 1e-14 relative (NaN where NaN)."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    m = ~np.isnan(a)
    d = np.abs(a[m] - b[m])
    ulp = d / np.spacing(np.maximum(np.abs(a[m]), np.abs(b[m])))
    return bool(np.all((ulp <= 4) | (d <= 1e-14 * np.abs(b[m]))))


_BAR_NO_CLUSTER = 1e-12
_BAR_CLUSTER = 1e-7


def _cancellation_scale(f, x0):
    """sum_m |d term_m / dp| at x0: every test loss is a SUM of weighted
    per-order efficiencies (``f(t, terms=True)`` returns the vector of its
    terms), and round-off in the gradient scales with the size of those
    term gradients, not with their sum -- a sum that cancels (the two
    same-shape layers: gradient -0.003 from terms an order larger) inflates
    an error relative to the gradient itself."""
    import jax
    import jax.numpy as jnp
    J = jax.jit(jax.jacrev(lambda t: f(t, terms=True)))(jnp.asarray(x0))
    return float(np.sum(np.abs(np.asarray(J))))


def _abs_dev(a, b):
    a, b = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
    return float(np.max(np.abs(a - b)))


#: name: (loss, x0, vmap points, the cluster's lifted branch must run)
_CASES = {
    "rcwa_symmetric_cell": (_sym_rcwa, 0.0, (0.0, 0.0), True),
    "pmm_stack_normal": (_pmm_stack, 0.0, (0.0, 0.0), True),
    "berreman_concrete_isotropic_layer": (_berreman_mixed, 2.6, (2.5, 2.7),
                                          True),
    "rcwa_uniform_select": (_rcwa_uniform, 4.0, (3.8, 4.2), True),
    "berreman_uniaxial_normal": (_berreman_uniaxial, 1.6,
                                 (1.58, 1.62), False),
    "rcwa_two_same_shape_layers": (_rcwa_two_layers, 4.0, (3.8, 4.2), False),
}


def _run(f, x0, xs, lifts):
    import jax
    import jax.numpy as jnp
    fe = lambda t: f(t)  # noqa: E731  -- a fresh callable: forces a retrace
    x = jnp.asarray(x0)
    lifts[0] = 0
    ge = np.asarray(jax.grad(fe)(x))
    n_lift = lifts[0]          # eager: counts only a lifted branch that RAN
    gv = np.asarray(jax.vmap(jax.grad(fe))(jnp.asarray(xs)))
    low = jax.jit(jax.grad(fe)).lower(x)
    return ge, gv, low.as_text(), np.asarray(low.compile()(x)), n_lift


@pytest.mark.parametrize("case", sorted(_CASES))
def test_lean_path_gradients_agree_with_the_5_50_0_path(
        case, lean_switch, monkeypatch):
    _core = lean_switch
    f, x0, xs, fires = _CASES[case]
    lifts = [0]
    lift0 = _core._eig_cluster_lift

    def counting_lift(*a, **k):
        lifts[0] += 1
        return lift0(*a, **k)
    monkeypatch.setattr(_core, "_eig_cluster_lift", counting_lift)
    out = {}
    for lean in (False, True):
        _core._CLUSTER_RULE_LEAN = lean
        out[lean] = _run(f, x0, xs, lifts)
        assert (out[lean][4] > 0) == fires, (case, lean, out[lean][4])
    old, new = out[False], out[True]
    assert old[2] == new[2], "jit: the lowered program changed"
    assert np.array_equal(old[3], new[3]), ("jit", old[3], new[3])
    # FIXED bars on a CANCELLATION-AWARE scale (not 5.50.0's own
    # eager-vs-jit spread, which can be near zero on one input and platform
    # and 8x larger on the next): the deviation is divided by
    # sum_m |d term_m / dp| (_cancellation_scale), the size of the per-order
    # term gradients the loss sums.  Without a firing cluster the two
    # evaluation orders differ by round-off only, so 1e-12; at a firing
    # cluster the lifted stencil amplifies round-off by ~1 / split_rel =
    # 1e7, i.e. ~eps_mach / split_rel ~ 2e-9, so 1e-7 ~ 100 x eps_mach /
    # split_rel.  Accuracy against the truth is gated separately
    # (test_..._against_an_extrapolated_reference).
    bar = _BAR_CLUSTER if fires else _BAR_NO_CLUSTER
    scale = _cancellation_scale(f, x0)
    for what, a, b in (("eager", new[0], old[0]), ("vmap", new[1], old[1]),
                       ("lean eager vs jit", new[0], new[3])):
        dev = _abs_dev(a, b) / scale
        assert dev <= bar, (what, dev, bar, scale)
    assert np.all(np.isfinite(new[0]))


def _concrete_eig_calls(monkeypatch, fn):
    import jax
    import jax.numpy as jnp
    n = [0]
    eig0 = jnp.linalg.eig

    def counting_eig(a, *k, **kw):
        if not isinstance(a, jax.core.Tracer):
            n[0] += 1
        return eig0(a, *k, **kw)
    monkeypatch.setattr(jnp.linalg, "eig", counting_eig)
    fn()
    monkeypatch.setattr(jnp.linalg, "eig", eig0)
    return n[0]


def test_an_eager_gradient_without_a_cluster_decomposes_each_matrix_once(
        lean_switch, monkeypatch):
    """5.50.0: three LAPACK eigs per problem (first pass, the full
    evaluation of the re-record jaxpr, the rule's forward); lean: one, as
    with the rule off (the GATE D slab: no cluster, no lifted eigs).  Counted are the Python-level ``jnp.linalg.eig`` calls
    on concrete matrices, which do not see the re-record jaxpr's eig (it is
    evaluated as a staged ``eig`` primitive): 2 per problem before, 1 now."""
    import jax
    import jax.numpy as jnp

    import lumenairy
    _core = lean_switch
    x = jnp.asarray(1.6)
    g = jax.grad(_berreman_uniaxial)
    with lumenairy.jax_cluster_rule(False):
        n_off = _concrete_eig_calls(monkeypatch, lambda: g(x))
    counts = {}
    for lean in (False, True):
        _core._CLUSTER_RULE_LEAN = lean
        counts[lean] = _concrete_eig_calls(monkeypatch, lambda: g(x))
    assert n_off > 0
    assert counts[True] == n_off, (counts, n_off)
    assert counts[False] == 2 * n_off, (counts, n_off)


def test_the_lean_path_never_calls_closure_convert(lean_switch, monkeypatch):
    """``jax.closure_convert`` keys a STRONG LRU on the per-call function;
    5.50.0 kept it off that cache with ``checking_leaks``.  The lean path
    does not call it at all (eager, vmap and jit)."""
    import jax
    import jax.numpy as jnp
    lean_switch._CLUSTER_RULE_LEAN = True
    assert lean_switch._lean_impl()     # the one-time self-test uses it

    def refuse(*_a, **_k):
        raise AssertionError("closure_convert called on the lean path")
    monkeypatch.setattr(jax, "closure_convert", refuse)
    fe = lambda t: _berreman_mixed(t)  # noqa: E731
    jax.grad(fe)(jnp.asarray(2.6))
    jax.vmap(jax.grad(fe))(jnp.asarray([2.5, 2.7]))
    jax.jit(jax.grad(fe))(jnp.asarray(2.6))


def test_the_eager_regime_is_detected_and_staging_is_not(lean_switch):
    import jax
    import jax.numpy as jnp
    _core = lean_switch
    _core._CLUSTER_RULE_LEAN = True
    seen = []

    def f(x):
        seen.append(_core._jax_eager_regime())
        return jnp.sin(x) ** 2

    x = jnp.asarray(0.3)
    jax.grad(f)(x)
    jax.vmap(jax.grad(f))(jnp.asarray([0.1, 0.2]))
    assert seen == [True, True]
    jax.jit(jax.grad(f))(x)
    jax.grad(jax.jit(f))(x)
    jax.jit(jax.vmap(jax.grad(f)))(jnp.asarray([0.1, 0.2]))
    jax.make_jaxpr(jax.grad(f))(x)
    jax.grad(jax.checkpoint(f))(x)
    jax.lax.scan(lambda c, t: (c + f(t), None), 0.0, jnp.asarray([0.1, 0.2]))
    jax.grad(lambda t: jax.lax.cond(t > 0, f, f, t))(x)
    assert len(seen) >= 9 and seen[2:] == [False] * (len(seen) - 2), seen
    assert _core._jax_eager_regime() is True
    _core._CLUSTER_RULE_LEAN = False
    assert _core._jax_eager_regime() is False


def test_the_eig_memo_matches_on_bits_not_on_equality():
    """-0.0 == +0.0, but the memo's key is the exact bytes."""
    import jax.numpy as jnp

    from lumenairy.elements.rcwa._core import _JaxEigMemo
    m = _JaxEigMemo()
    a = jnp.asarray(np.array([[1.0, 0.0], [0.0, 2.0]], complex))
    b = jnp.asarray(np.array([[1.0, -0.0], [0.0, 2.0]], complex))
    m.store(a, ("A",))
    assert m.lookup(a) == ("A",)
    assert m.lookup(jnp.asarray(np.asarray(a).copy())) == ("A",)
    assert m.lookup(b) is None
    assert m.lookup(a.astype(jnp.complex64)) is None
    c = jnp.asarray(np.array([[1.0, 0.5], [0.0, 2.0]], complex))
    assert m.lookup(c) is None                 # same shape and dtype


def test_a_dead_use_of_a_closed_over_value_gives_the_5_50_0_cotangent(
        lean_switch):
    """Verifier finding (2026-11-03): a consumer closing over c = sqrt(q) and
    using it only in dead code.  5.50.0 hands c an instantiated ZERO
    cotangent, which d sqrt / dq = inf at q = 0 turns into NaN -- eagerly and
    under jit.  A DCE that also dropped the constant gave 0.0 eagerly (and
    NaN under jit); every constant is now kept, so eager == jit == 5.50.0."""
    import jax
    import jax.numpy as jnp
    _core = lean_switch
    L0 = jnp.asarray(np.array([[2.0, 0.3], [0.3, 3.0]], complex))

    def f(p, q):
        c = jnp.sqrt(q)
        L = L0 * (1.0 + p)

        def consumer(eigs):
            lam, _V = eigs[0]
            _dead = c * lam
            return jnp.sum(jnp.abs(lam) ** 2)
        return _core._jax_eig_cluster_adjoint(_core._jax_twin_eig_plain,
                                              [(L, None)], consumer)
    for q in (0.0, 0.5):
        args = (jnp.asarray(0.2), jnp.asarray(q))
        vals = []
        for lean in (False, True):
            _core._CLUSTER_RULE_LEAN = lean
            fe = lambda a, b: f(a, b)  # noqa: E731
            # both arguments: p makes the eig argument traced (the rule
            # applies), q is the closed-over value's producer
            vals.append(float(jax.grad(fe, argnums=(0, 1))(*args)[1]))
            vals.append(float(jax.jit(jax.grad(fe, argnums=(0, 1)))(*args)[1]))
        if q == 0.0:
            assert all(np.isnan(v) for v in vals), vals
        else:
            assert all(v == 0.0 for v in vals), vals


def test_an_eager_gradient_compiles_nothing_new(lean_switch):
    """The reverse pass's per-gradient ``lax.cond`` compiled a fresh XLA
    program on every eager gradient (and kept it in JAX's strong primitive
    cache: the 5.50.0 memory growth on every RCWA solve).  After a warm-up,
    eager gradients through the select path, a cluster that fires, and an
    eager vmap add no compilation and no primitive-callable entry."""
    import jax
    import jax.numpy as jnp
    from jax._src import util as jutil
    lean_switch._CLUSTER_RULE_LEAN = True

    def counts():
        out = {}
        for c, name in list(jutil._caches.items()):
            for key in ("_cached_compilation", "xla_primitive_callable"):
                if key in name:
                    out[key] = out.get(key, 0) + c.cache_info().currsize
        return out
    calls = [
        lambda i: jax.grad(_rcwa_uniform)(jnp.asarray(4.0 + 1e-3 * i)),
        lambda i: jax.grad(_sym_rcwa)(jnp.asarray(0.0)),
        lambda i: jax.vmap(jax.grad(_rcwa_uniform))(
            jnp.asarray([3.9, 4.1 + 1e-3 * i])),
    ]
    for call in calls:
        call(0)
        call(1)
    before = counts()
    assert len(before) == 2, "JAX cache names changed: update this test"
    for i in range(2, 5):
        for call in calls:
            call(i)
    assert counts() == before, (before, counts())


def test_a_failed_self_test_falls_back_with_one_warning(lean_switch,
                                                         monkeypatch):
    import jax
    import jax.numpy as jnp
    _core = lean_switch
    assert _core._lean_impl()                   # passes on this JAX
    assert _core._LEAN_SELFTEST is True

    def broken(_impl):
        raise TypeError("changed private API")
    monkeypatch.setattr(_core, "_lean_selftest_body", broken)
    monkeypatch.setattr(_core, "_LEAN_SELFTEST", None)
    with pytest.warns(RuntimeWarning, match="self-test"):
        assert _core._lean_impl() is False
    assert _core._lean_impl() is False          # no second run, no warning
    assert _core._jax_eager_regime() is False
    assert _core._concrete_bool(jnp.asarray(True)) is None
    g = np.asarray(jax.grad(_berreman_uniaxial)(jnp.asarray(1.6)))
    _core._CLUSTER_RULE_LEAN = False
    g_old = np.asarray(jax.grad(_berreman_uniaxial)(jnp.asarray(1.6)))
    assert np.array_equal(g, g_old)            # exactly the 5.50.0 path


def test_a_failing_lean_conversion_falls_back_for_that_call(lean_switch,
                                                            monkeypatch):
    import jax
    import jax.numpy as jnp
    _core = lean_switch
    _core._CLUSTER_RULE_LEAN = True
    ref = np.asarray(jax.grad(_berreman_uniaxial)(jnp.asarray(1.6)))

    def broken(*_a, **_k):
        raise AttributeError("changed private API")
    monkeypatch.setattr(_core, "_closure_convert_lean", broken)
    g = np.asarray(jax.grad(_berreman_uniaxial)(jnp.asarray(1.6)))
    assert _close(g, ref)


def _sym_rcwa_diag(t):
    """The verifier's loss c: the symmetric cell perturbed DIAGONALLY."""
    import jax.numpy as jnp

    from lumenairy.elements.rcwa import RCWAStack
    st = RCWAStack(1.2, period_y=1.2, n_superstrate=1.0, n_substrate=1.45,
                   n_orders=2, n_orders_y=2)
    st.add_layer(0.45, eps_cell=(jnp.asarray(_BASE)
                                 + t * jnp.asarray(_DIAG)).astype(complex))
    st.add_layer(0.1, eps=2.25)
    st.set_source(1.0, theta=0.0, phi=0.0)
    _o, R, T = st.solve().efficiencies()
    w = jnp.asarray(np.cos(np.arange(R.shape[1])))
    return jnp.sum(R[0] * w) - 0.3 * jnp.sum(T[1] * w ** 2) + jnp.sum(T[0])


def _extrapolated_derivative(f, x0):
    """d f / dx at x0 from central differences of the (jitted) forward at
    h = 0.08 / 2^k, k = 0..6, Richardson-extrapolated in h^2 (Neville
    table); the reference is the median of five well-converged entries and
    its spread their largest distance from it.  The points x0 +- h are off
    the symmetric point, so no degenerate cluster enters the reference."""
    import jax
    import jax.numpy as jnp
    fj = jax.jit(f)
    hs = [0.08 / 2 ** k for k in range(7)]
    D = [(float(fj(jnp.asarray(x0 + h))) - float(fj(jnp.asarray(x0 - h))))
         / (2 * h) for h in hs]
    T = [D]
    for k in range(1, len(D)):
        r = (hs[0] / hs[1]) ** (2 * k)
        prev = T[-1]
        T.append([(r * prev[i + 1] - prev[i]) / (r - 1)
                  for i in range(len(prev) - 1)])
    cands = np.asarray([T[2][2], T[2][3], T[3][1], T[3][2], T[2][4]])
    ref = float(np.median(cands))
    return ref, float(np.max(np.abs(cands - ref)))


@pytest.mark.parametrize("loss", [_sym_rcwa, _sym_rcwa_diag],
                         ids=["xsides", "diagonal"])
def test_firing_cluster_gradients_against_an_extrapolated_reference(
        loss, lean_switch):
    """ACCURACY, not only closeness to 5.50.0: at the symmetric point (the
    cluster fires) the lean eager, eager-vmap and jit gradients and the
    5.50.0 eager gradient all agree with a Richardson-extrapolated central
    difference to the rule's round-off level (measured: every mode, old and
    new, within 4.4e-9 .. 3.4e-8 of the reference; bar 1e-7)."""
    import jax
    import jax.numpy as jnp
    _core = lean_switch
    ref, spread = _extrapolated_derivative(loss, 0.0)
    # the reference's own spread: measured 1.0e-9 (x sides) / 2.4e-9
    # (diagonal) relative -- 10x below the bar it is used against
    assert spread <= 1e-8 * abs(ref), (ref, spread)
    x = jnp.asarray(0.0)
    vals = {
        "eager": float(jax.grad(loss)(x)),
        "vmap": float(jax.vmap(jax.grad(loss))(jnp.asarray([0.0, 0.0]))[0]),
        "jit": float(jax.jit(jax.grad(loss))(x)),
    }
    _core._CLUSTER_RULE_LEAN = False
    vals["5.50.0 eager"] = float(jax.grad(lambda t: loss(t))(x))
    errs = {k: abs(v - ref) / abs(ref) for k, v in vals.items()}
    assert all(e <= _BAR_CLUSTER for e in errs.values()), (ref, errs)


class _ConsumerBoom(Exception):
    pass


def test_an_exception_of_the_consumer_propagates_and_it_runs_once(
        lean_switch):
    """An exception raised by the converted function itself is not a
    failure of the lean machinery: it propagates as the SAME object, and the
    function is not re-run through the 5.50.0 conversion (a solve may count
    its passes -- the r2v route tests do)."""
    import jax
    import jax.numpy as jnp
    _core = lean_switch
    L0 = jnp.asarray(np.array([[2.0, 0.3], [0.3, 3.0]], complex))
    boom = _ConsumerBoom("consumer")
    calls = [0]

    def f(p):
        L = L0 * (1.0 + p)

        def consumer(eigs):
            calls[0] += 1
            raise boom
        return _core._jax_eig_cluster_adjoint(_core._jax_twin_eig_plain,
                                              [(L, None)], consumer)
    with pytest.raises(_ConsumerBoom) as info:
        jax.grad(f)(jnp.asarray(0.2))
    assert info.value is boom
    assert calls[0] == 1, calls
