"""The lean path of the degenerate-cluster gradient rule (L-2 of the DynaMeta
bridge audit, 2026-11-02; ``rcwa._core``, "The closure conversions of the
rule, without their per-call overhead").

5.50.0 ran every eager gradient through the rule with ``jax.closure_convert``
under ``jax.checking_leaks()`` (19% of the GATE D gradient) and evaluated the
re-record jaxpr in full (a whole extra solve, eigs included), then decomposed
every matrix a third time in the rule's forward.  The lean path takes the same
uncached conversion without leak checking, drops dead WHOLE equations
eagerly only, and shares the eager eigs (``_JaxEigMemo``).  It must not move
a gradient by one bit, so the 5.50.0 path stays reachable
(``_CLUSTER_RULE_LEAN = False``) and every case here compares the two in one
process with ``np.array_equal``:

* eager ``jax.grad`` and eager ``jax.vmap`` of it: the values;
* ``jax.jit``: the LOWERED PROGRAM TEXT (identical program -> identical
  compiled code -> identical bits), plus the values.

Cases: a four-fold RCWA cell at its symmetric point (the cluster FIRES and the
lifted branch runs), the 1-D ``PMMStack`` at normal incidence (a pencil with a
lift Gram, cluster fires), the Berreman cascade with a concrete isotropic
layer (the case that needs the re-record: the eager first pass takes the
analytic branch, the replay the eig -- whose isotropic layer is a degenerate
cluster, so the lifted branch runs), an RCWA stack whose second patterned
layer is uniform (the analytic-shortcut select: ``has_select``; the uniform
layer is a cluster too), and DynaMeta's GATE D slab (no cluster at all).
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
    yield _core
    _core._CLUSTER_RULE_LEAN = previous


_BASE3 = np.array([[1.0, 1.5, 1.0], [1.5, 4.0, 1.5], [1.0, 1.5, 1.0]])
_XS3 = np.zeros((3, 3))
_XS3[0, 1] = _XS3[2, 1] = 1.0
_BASE, _DIRN = (np.kron(a, np.ones((3, 3))) for a in (_BASE3, _XS3))
_PIL = np.zeros((9, 9))
_PIL[2:6, 3:7] = 1.0


def _sym_rcwa(t):
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
    return jnp.sum(R[0] * w) + jnp.sum(R[1] * w[::-1]) + 0.5 * jnp.sum(T[0] * w)


def _pmm_stack(a):
    import jax.numpy as jnp

    from lumenairy.elements.pmm import PMMStack
    st = PMMStack(1.2e-6, n_substrate=1.0, n_superstrate=1.45, degree=8)
    st.add_layer(0.3e-6, segments=[(0.5, 4.0), (0.5, 1.0)])
    st.add_layer(0.1e-6, segments=[(1.0, 2.25)])
    st.set_source(1.0e-6, angle=a)
    _o, R, T, _J = st.solve()
    w = jnp.asarray(np.linspace(0.2, 1.0, R.shape[1]))
    return jnp.sum(R[0] * w) + jnp.sum(T[1] * w[::-1])


def _berreman_mixed(re):
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
    return R[0] + 0.5 * T[1]


def _berreman_uniaxial(ne):
    """DynaMeta's lumenairy_berreman_jax GATE D slab (no cluster)."""
    import jax.numpy as jnp

    from lumenairy import berreman_jones_1d
    eps = jnp.asarray([[ne ** 2, 0.0, 0.0], [0.0, 2.25, 0.0], [0.0, 0.0, 2.25]],
                      dtype=jnp.complex128)
    R, _T, _Jr, _Jt = berreman_jones_1d([(eps, 220e-9)], 1.5 + 0j, 1.0 + 0j,
                                        jnp.asarray(1.55e-6), angle=0.0,
                                        phi=0.0)
    return (jnp.real(R[0]) - 0.05) ** 2


def _rcwa_uniform(p):
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
    return jnp.sum(R[0]) + 0.5 * jnp.sum(T[1]) + 0.25 * jnp.sum(R[1])


#: name: (loss, x0, vmap points, the cluster's lifted branch must run)
_CASES = {
    "rcwa_symmetric_cell": (_sym_rcwa, 0.0, (0.0, 0.0), True),
    "pmm_stack_normal": (_pmm_stack, 0.0, (0.0, 0.0), True),
    "berreman_concrete_isotropic_layer": (_berreman_mixed, 2.6, (2.5, 2.7),
                                          True),
    "rcwa_uniform_select": (_rcwa_uniform, 4.0, (3.8, 4.2), True),
    "berreman_uniaxial_normal": (lambda ne: _berreman_uniaxial(ne), 1.6,
                                 (1.58, 1.62), False),
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
def test_lean_path_gradients_are_bit_identical_to_the_5_50_0_path(
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
    assert np.array_equal(old[0], new[0]), ("eager", old[0], new[0])
    assert np.array_equal(old[1], new[1]), ("vmap", old[1], new[1])
    assert old[2] == new[2], "jit: the lowered program changed"
    assert np.array_equal(old[3], new[3]), ("jit", old[3], new[3])
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
    assert seen[2:] == [False, False, False]
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
