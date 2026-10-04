"""INDEPENDENT VERIFICATION of the symmetric-point JAX gradient fix
(``docs/audits/BUILD_JAX_SYMMETRIC_POINT_GRADIENTS_2026_10_03.md``; verifier
record ``docs/audits/VERIFY_JAX_SYMMETRIC_POINT_GRADIENTS_2026_10_04.md``).

The verifier's OWN fixtures, chosen to differ from the builder's (15 x 15
pixels, P 1.2, n_orders 3, a 1-D grating P 1.2 / n 2 / 1):

* RCWA: a 24 x 24 pixel cell, P 0.9, wl 1, a square post (pixels 7..16,
  eps 5) in eps 1.8 -- four-fold (C4v) symmetric -- depth 0.37, n_sup 1,
  n_sub 1.52, n_orders 2 (25 orders; the layer operator P @ Q is 50 x 50, of
  which 26 eigenvalues sit in 13 exact pairs).  Directions: ``x`` widens the
  post along x (keeps both mirrors, breaks the rotation), ``corner`` adds one
  pixel off a corner (breaks every symmetry), ``keep`` scales the whole post
  (keeps C4v).
* 1-D PMM: P 0.85, ridge n 2.3 / groove n 1.35, duty 0.4, depth 0.31,
  n_sup 1, n_sub 1.6, wl 1, degree 10, ``stabilize=False``.

Oracle: the NumPy solve (``symmetry=False`` for RCWA: the same full solve the
JAX path runs), central differences at h, h/2, h/4 and Richardson of the last
two; the h^2 PREMISE (ratio of successive rung changes = 4) is asserted
before any comparison.  Error = max |AD - FD| / max |FD| over all R and T.

Confirmed claims are passing pins with bars that have measured decades on
both sides (Windows 11 / CPython 3.14.6 / jax 0.11.0 and WSL / CPython
3.12.3 / jax 0.10.2, 2026-10-04, probes in
``validation/probe_jax_symgrad_verify/``).  DEFECTS found by the
verification are STRICT xfails that flip when fixed.
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
    pytest.mark.filterwarnings("ignore::UserWarning"),
]


@pytest.fixture(autouse=True, scope="module")
def _jax_x64_for_this_module():
    """Hold ``jax_enable_x64`` on for this file and restore it after."""
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


# ------------------------------------------------------------- the oracle
def _fd(fn, x0, h0):
    """Richardson central FD with the h -> h/2 PREMISE asserted: the ratio of
    successive rung changes (max over the outputs) must be the h^2 value 4
    (bar [3.5, 4.5]; measured 3.76 .. 4.08 on every fixture below, both
    builds)."""
    rows = [(np.asarray(fn(x0 + h), float) - np.asarray(fn(x0 - h), float))
            / (2 * h) for h in (h0, h0 / 2, h0 / 4)]
    c1 = np.max(np.abs(rows[0] - rows[1]))
    c2 = np.max(np.abs(rows[1] - rows[2]))
    assert 3.5 < c1 / c2 < 4.5, (c1, c2)
    return (4.0 * rows[2] - rows[1]) / 3.0


def _rel(g, fd):
    return float(np.max(np.abs(g - fd)) / np.max(np.abs(fd)))


def _ad(f, x0, rule=True):
    """jit(jacrev) of ``f(t, jnp)`` at x0; ``rule=False`` switches the
    cluster rule off at trace time (``_EIG_CLUSTER_GAP_REL = 0``: the plain
    composition = the gradient before the fix), restored afterwards."""
    import jax
    import jax.numpy as jnp

    import lumenairy.elements.rcwa._core as RC
    gap0 = RC._EIG_CLUSTER_GAP_REL
    if not rule:
        RC._EIG_CLUSTER_GAP_REL = 0.0
    try:
        return np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(x0))
    finally:
        RC._EIG_CLUSTER_GAP_REL = gap0


# ------------------------------------------------------------ RCWA fixture
_S, _PX = 24, 0.9
_POST = np.zeros((_S, _S))
_POST[7:17, 7:17] = 1.0
_BASE = 1.8 + 3.2 * _POST
_DX = np.zeros((_S, _S))
_DX[6, 7:17] = _DX[17, 7:17] = 1.0
_DCORNER = np.zeros((_S, _S))
_DCORNER[6, 6] = 1.0
_RAND = np.random.default_rng(1234).uniform(-1.0, 1.0, (_S, _S))
_DIRS = {"x": _DX, "corner": _DCORNER, "keep": _POST}


def _rcwa_f(direction, pol, base=None, theta=0.0, phi=0.0,
            formulation="laurent", loss=0.0):
    from lumenairy.elements.rcwa import rcwa_efficiency_2d
    b = (_BASE if base is None else base).astype(complex) + 1j * loss * _POST
    D = _DIRS[direction]

    def f(t, xp):
        eps = (xp.asarray(b) + t * xp.asarray(D)).astype(complex)
        kw = {} if xp is not np else {"symmetry": False}
        _o, R, T = rcwa_efficiency_2d(_PX, _PX, eps, 1.52, 1.0, 0.37, 1.0,
                                      theta=theta, phi=phi,
                                      polarization=pol, n_orders_x=2,
                                      n_orders_y=2, formulation=formulation,
                                      **kw)
        return xp.concatenate([R, T]).real
    return f


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_rcwa_c4v_symmetry_breaking_gradient_is_exact_and_was_not(pol):
    """Claim (1), RCWA, on the verifier's C4v cell (x-widening of the post).
    Rule ON: measured 1.6e-10 (TE) / 7.1e-11 (TM) on Windows and
    2.9e-10 / 6.2e-11 on WSL; bar 1e-7.  Rule OFF (= the pre-fix tree
    5ea82b44, checked equal to all printed digits): 0.27 (TE) / 0.073 (TM)
    on Windows, 0.28 / 0.076 on WSL; bar 1e-3, so the fixture really does
    exercise the defect (73x below the smallest reading; the reading is
    LAPACK-basis-dependent, 0.028 .. 0.30 over 7 x 7 orders and both
    builds)."""
    f = _rcwa_f("x", pol)
    fd = _fd(lambda t: f(t, np), 0.0, 2e-2)
    assert _rel(_ad(f, 0.0), fd) < 1e-7
    assert _rel(_ad(f, 0.0, rule=False), fd) > 1e-3


@pytest.mark.parametrize("case", ["lossy_te", "lossy_tm", "li_te", "li_tm",
                                  "corner_te"])
def test_rcwa_symmetric_cell_variants_are_exact(case):
    """Claim (1) on variants the builder did not gate: a LOSSY post
    (eps 5 + 0.8i), the 'li' formulation (tensor eigensolver route), and a
    direction that breaks every symmetry.  Measured Windows / WSL: lossy
    1.6e-10 / 1.1e-10 (TE), 3.7e-11 / 7.3e-11 (TM); li 1.1e-10 / 2.0e-10
    (TE), 1.2e-10 / 4.5e-11 (TM); corner 5.3e-10 / 4.4e-10 (FD resolution
    6.8e-10 / 6.1e-10).  Pre-fix (rule off): lossy 0.31 / 0.30, 0.070 /
    0.066; li 0.32 / 0.56, 0.057 / 0.10; corner 0.13 / 0.35.  Bar 1e-7."""
    kind, pol = case.rsplit("_", 1)
    if kind == "lossy":
        f = _rcwa_f("x", pol, loss=0.8)
    elif kind == "li":
        f = _rcwa_f("x", pol, formulation="li")
    else:
        f = _rcwa_f("corner", pol)
    fd = _fd(lambda t: f(t, np), 0.0, 2e-2)
    assert _rel(_ad(f, 0.0), fd) < 1e-7


def test_rcwa_symmetry_keeping_direction_is_exact_with_and_without_rule():
    """A parameter that KEEPS the four-fold symmetry (the post's eps) leaves
    every pair degenerate; the plain VJP is exact there and the rule must
    not change that.  Measured 1.3e-11 (Windows) / 1.6e-11 (WSL) with the
    rule on and off; bar 1e-7 on both arms."""
    f = _rcwa_f("keep", "te")
    fd = _fd(lambda t: f(t, np), 0.0, 2e-2)
    assert _rel(_ad(f, 0.0), fd) < 1e-7
    assert _rel(_ad(f, 0.0, rule=False), fd) < 1e-7


def test_rcwa_conical_gradient_is_exact():
    """Conical incidence (theta 0.2, phi 0.3) on the C4v cell, corner
    direction: no cluster is present, the rule's plain branch runs.
    Measured 2.0e-9 / 2.5e-9 (Windows / WSL; FD resolution 2.0e-9 /
    2.5e-9); bar 1e-7."""
    f = _rcwa_f("corner", "te", theta=0.2, phi=0.3)
    fd = _fd(lambda t: f(t, np), 0.0, 2e-2)
    assert _rel(_ad(f, 0.0), fd) < 1e-7


def test_rcwa_accidental_near_degeneracy_without_symmetry_is_exact():
    """NEAR-degenerate clusters that are not symmetry-forced: a weak
    (holographic-contrast) grating eps 2.25 + 0.01 x a fixed random pattern
    at conical incidence has 23 pairs closer than the rule's gap_rel = 1e-6
    of the spectrum, none exact, 7 of them propagating.  The rule lifts
    them.  AD vs FD, corner direction: measured 3.9e-9 / 1.5e-8 (Windows /
    WSL; FD resolution 4.5e-9 / 8.7e-9), equal to the rule-off value; bar
    1e-6 (the near-degenerate splitting is up to 9.9e-7 of the spectrum,
    the W9 envelope at that splitting is ~1e-8)."""
    f = _rcwa_f("corner", "te", base=2.25 + 1e-2 * _RAND, theta=0.2,
                phi=0.3)
    fd = _fd(lambda t: f(t, np), 0.0, 2e-2)
    assert _rel(_ad(f, 0.0), fd) < 1e-6


def test_rcwa_lift_moves_propagating_layer_roots_across_the_cut_harmlessly():
    """The build record says the RCWA operator needs no Gram because split
    members are orthogonal or 'lifted cleanly'.  Measured on the weak
    no-symmetry conical grating above: the Euclidean lift of a cluster of
    DISTINCT eigenvalues gives COMPLEX shifts (|Im| up to 1.5 d) and moves
    propagating layer roots across the sqrt cut (16 member-evaluations
    change branch over the four stencil points).  The gradient is still
    right (test above) because a LAYER mode's branch is a forward/backward
    re-labelling the S-matrix is invariant under (lam -> -lam, the
    ``_sqrt_decay`` docstring), unlike the 1-D twin's HALF-SPACE modes where
    the same thing produced 4.7e3.  This pins the mechanism the record's
    rationale omits: >= 1 branch change (measured 16 on both builds) and a
    complex shift above 0.1 d (measured 1.54 d)."""
    import jax.numpy as jnp

    import lumenairy.elements.rcwa._core as RC
    import lumenairy.elements.rcwa.twod as RT
    from lumenairy.elements.rcwa import rcwa_efficiency_2d
    calls = []
    orig = RT._jax_eig_cluster_adjoint

    def spy(eig_fn, problems, consumer, **kw):
        calls.append((eig_fn, problems, kw))
        return orig(eig_fn, problems, consumer, **kw)
    RT._jax_eig_cluster_adjoint = spy
    try:
        rcwa_efficiency_2d(_PX, _PX, jnp.asarray(
            (2.25 + 1e-2 * _RAND).astype(complex)), 1.52, 1.0, 0.37, 1.0,
            theta=0.2, phi=0.3, n_orders_x=2, n_orders_y=2)
    finally:
        RT._jax_eig_cluster_adjoint = orig
    eig_fn, ((A, G),), kw = calls[-1]
    lam, V = eig_fn(A, G)
    dN, anyc = RC._eig_cluster_lift(lam, V, G, RC._EIG_CLUSTER_GAP_REL,
                                    RC._EIG_CLUSTER_SPLIT_REL,
                                    kw["anchors"][0])
    assert bool(anyc)
    lam = np.asarray(lam)
    d = RC._EIG_CLUSTER_SPLIT_REL * np.max(np.abs(lam))
    r0 = np.asarray(RC._sqrt_decay(jnp.asarray(lam), xp=jnp))
    close = np.abs(lam[:, None] - lam[None, :]) <= (
        RC._EIG_CLUSTER_GAP_REL * np.max(np.abs(lam)))
    np.fill_diagonal(close, False)
    members = np.nonzero(np.any(close, axis=1))[0]
    jumps, im_max = 0, 0.0
    for t in (1.0, -1.0, 2.0, -2.0):
        lt = np.asarray(eig_fn(A + t * dN, G)[0])
        rt = np.asarray(RC._sqrt_decay(jnp.asarray(lt), xp=jnp))
        for i in members:
            j = int(np.argmin(np.abs(lt - lam[i])))
            im_max = max(im_max, abs((lt[j] - lam[i]).imag) / d)
            jumps += int(abs(rt[j] + r0[i]) < abs(rt[j] - r0[i]))
    assert jumps >= 1 and im_max > 0.1, (jumps, im_max)


def test_rcwa_clusters_of_up_to_eight_members_are_exact():
    """More than two members per cluster: a near-uniform C4v cell
    (eps 2.25 + 1e-4 x post) chains exact E pairs and distinct eigenvalues
    of the uniform medium's eight-fold multiplets into clusters of up to 8
    members.  x direction, error on the DIFFRACTED orders (whose O(delta^2)
    efficiencies would hide in a max over the zero order): measured
    2.1e-10 / 1.5e-10 (TE, Windows / WSL), rule off 8.3e-6 / 9.8e-6; bar
    1e-6."""
    f = _rcwa_f("x", "te", base=2.25 + 1e-4 * _POST)
    fd = _fd(lambda t: f(t, np), 0.0, 2e-2)
    g = _ad(f, 0.0)
    from lumenairy.elements.rcwa import rcwa_efficiency_2d
    # drop the two (0, 0) entries (R and T)
    o = np.asarray(rcwa_efficiency_2d(_PX, _PX, _BASE.astype(complex), 1.52,
                                      1.0, 0.37, 1.0, n_orders_x=2,
                                      n_orders_y=2)[0])
    zero = int(np.nonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    nz = np.ones(2 * len(o), bool)
    nz[[zero, len(o) + zero]] = False
    assert _rel(g[nz], fd[nz]) < 1e-6


@pytest.mark.parametrize("which", ["rcwa", "pmm1d"])
def test_vmapped_gradient_equals_the_unbatched_gradients(which):
    """``jax.vmap(jax.grad)`` (where the cluster flag is batched, so the
    rule's cond becomes a select and BOTH branches run) against per-point
    ``jax.grad`` over [0, 1e-3, -2e-3, 1e-2], the first point on the
    symmetric configuration: RCWA C4v x / TE and the 1-D twin d / d(angle) /
    TM.  Measured 3.2e-11 / 6.0e-13 (Windows), 1.5e-10 / 1.7e-13 (WSL);
    bar 1e-8."""
    import jax
    import jax.numpy as jnp
    xs = jnp.asarray([0.0, 1e-3, -2e-3, 1e-2])
    f = _rcwa_f("x", "te") if which == "rcwa" else _pmm1d_f("tm")
    n = np.asarray(f(0.0, np)).size
    w = jnp.asarray(np.random.default_rng(5).standard_normal(n))

    def L(t):
        return jnp.dot(w, f(t, jnp))
    gv = np.asarray(jax.jit(jax.vmap(jax.grad(L)))(xs))
    g1 = jax.jit(jax.grad(L))
    gs = np.asarray([float(g1(x)) for x in xs])
    assert _rel(gv, gs) < 1e-8


def test_hessian_through_an_eig_parameter_is_refused_on_both_paths():
    """The build record says ``jax.hessian`` 'keeps working'.  It does only
    for a parameter downstream of the eig (the depth, its own probe): for
    a parameter that enters the eig (eps_cell, the angle) forward-over-
    reverse raises NotImplementedError (JAX's non-symmetric eigenvector
    JVP), on this tree AND on the pre-fix tree 5ea82b44 (measured, both
    builds) -- so it is a scope statement, not a regression."""
    import jax
    import jax.numpy as jnp
    for f, x0 in ((_rcwa_f("x", "te"), 0.0), (_pmm1d_f("te"), 0.1)):
        n = np.asarray(f(0.0, np)).size
        w = jnp.asarray(np.random.default_rng(5).standard_normal(n))
        with pytest.raises(NotImplementedError):
            jax.hessian(lambda t, f=f, w=w: jnp.dot(w, f(t, jnp)))(x0)


# ----------------------------------------------------------- 1-D PMM twin
def _pmm1d_f(pol, which="angle", loss=0.0):
    from lumenairy.elements.pmm import pmm_efficiency_1d

    def f(t, xp):
        nr = xp.asarray(2.3 + 1j * loss + 0j)
        ang = t if which == "angle" else 0.0
        if which == "index":
            nr = nr + t
        _o, R, T = pmm_efficiency_1d(0.85, nr, 1.35, 1.6, 1.0, 0.31, 0.4, 1.0,
                                     angle=ang, polarization=pol, degree=10,
                                     stabilize=False)
        return xp.concatenate([xp.asarray(R), xp.asarray(T)]).real
    return f


def _mirror_defect(g):
    """|d X_{+m} + d X_{-m}| over the mirror pairs, relative to max |g|."""
    from lumenairy.elements.pmm import pmm_efficiency_1d
    o = np.asarray(pmm_efficiency_1d(0.85, 2.3, 1.35, 1.6, 1.0, 0.31, 0.4,
                                     1.0, degree=10, stabilize=False)[0])
    n = o.size
    worst = 0.0
    for m in o[o > 0]:
        i, j = int(np.nonzero(o == m)[0][0]), int(np.nonzero(o == -m)[0][0])
        worst = max(worst, abs(g[i] + g[j]), abs(g[n + i] + g[n + j]))
    return worst / np.max(np.abs(g))


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_pmm1d_angle_gradient_at_exactly_normal_is_exact_and_was_not(pol):
    """Claim (1), 1-D twin, on the verifier's grating.  Rule ON: measured
    4.8e-10 / 4.7e-10 (TE, Windows / WSL), 6.1e-11 / 5.7e-11 (TM); bar 1e-7;
    FD-free mirror identity <= 2.2e-11 both builds, bar 1e-8.  Rule OFF
    (= the pre-fix tree, checked): 1.6 (TE) / 0.89 (TM) on both; bar
    1e-2."""
    f = _pmm1d_f(pol)
    fd = _fd(lambda t: f(t, np), 0.0, 4e-3)
    g = _ad(f, 0.0)
    assert _rel(g, fd) < 1e-7
    assert _mirror_defect(g) < 1e-8
    assert _rel(_ad(f, 0.0, rule=False), fd) > 1e-2


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_pmm1d_pencil_is_right_where_the_fold_is_wrong(pol, monkeypatch):
    """Claim (3), the pencil: at 1e-5 rad off normal (half-space pairs split
    inside gap_rel) the rule handed the FOLDED operator B^-1 A with a
    Euclidean Gram is wrong by 5.6e3 (TE) / 2.6e3 (TM) relative (Windows;
    builder: 4.7e3 / 7.0e4 on its grating); the shipped pencil (A, B) reads
    4.2e-10 / 4.5e-11.  WSL identical to two digits.  Bars: fold > 1e1,
    pencil < 1e-7."""
    import jax.numpy as jnp

    import lumenairy.elements.rcwa._core as RC
    f = _pmm1d_f(pol)
    fd = _fd(lambda t: f(t, np), 1e-5, 4e-3)
    assert _rel(_ad(f, 1e-5), fd) < 1e-7
    orig = RC._jax_eig_cluster_adjoint

    def fold(eig_fn, problems, consumer, **kw):
        probs = tuple((jnp.linalg.solve(B, A), None) for A, B in problems)
        return orig(lambda A, _G: eig_fn(A, jnp.eye(A.shape[0],
                                                   dtype=A.dtype)),
                    probs, consumer, **kw)
    monkeypatch.setattr(RC, "_jax_eig_cluster_adjoint", fold)
    assert _rel(_ad(f, 1e-5), fd) > 1e1


def test_pmm1d_lossy_angle_gradient_at_exactly_normal_is_exact():
    """A LOSSY ridge (n 2.3 + 0.05i) at exactly normal incidence, d / d
    angle: measured 1.0e-9 (TE, both builds; rule off 4.1).  Bar 1e-7."""
    f = _pmm1d_f("te", loss=0.05)
    fd = _fd(lambda t: f(t, np), 0.0, 4e-3)
    assert _rel(_ad(f, 0.0), fd) < 1e-7


def test_pmm1d_symmetry_keeping_parameter_is_exact_with_and_without_rule():
    """The ridge index (keeps the mirror) at normal incidence: measured
    2.6e-11 (TE, both builds) with the rule on and off.  Bar 1e-7 on
    both."""
    f = _pmm1d_f("te", which="index")
    fd = _fd(lambda t: f(t, np), 0.0, 4e-3)
    assert _rel(_ad(f, 0.0), fd) < 1e-7
    assert _rel(_ad(f, 0.0, rule=False), fd) < 1e-7


# ---------------------------------------- the twins NOT changed by the fix
def _jones1d_f():
    from lumenairy.elements.pmm import pmm_jones_1d
    ER = 2.3 ** 2 * np.eye(3, dtype=complex)
    EG = 1.35 ** 2 * np.eye(3, dtype=complex)

    def f(t, xp):
        import jax.numpy as jnp
        _o, R, T, _J = pmm_jones_1d(0.85, ER, EG, 1.6, 1.0, 0.31, 0.4, 1.0,
                                    angle=(jnp.asarray(t) if xp is not np
                                           else float(t)),
                                    degree=10, stabilize=False)
        return xp.concatenate([xp.ravel(xp.asarray(R)),
                               xp.ravel(xp.asarray(T))]).real
    return f


def test_pmm_jones_1d_angle_gradient_off_normal_is_exact():
    """Control for the defect below: 1e-5 rad off normal the unrouted
    ``pmm_jones_1d`` twin is right (measured 3.5e-10, both builds); bar 1e-7."""
    f = _jones1d_f()
    fd = _fd(lambda t: f(t, np), 1e-5, 4e-3)
    assert _rel(_ad(f, 1e-5), fd) < 1e-7


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT (claim 5, re-measured): pmm_jones_1d d / d(angle) at exactly "
    "normal incidence is wrong -- measured 0.97 relative on the verifier "
    "grating, both builds (builder: 2.7 on its own); the half-space Kx^2 pairs are not "
    "routed through the cluster rule.  Flips when the twin is routed."))
def test_pmm_jones_1d_angle_gradient_at_exactly_normal():
    f = _jones1d_f()
    fd = _fd(lambda t: f(t, np), 0.0, 4e-3)
    assert _rel(_ad(f, 0.0), fd) < 1e-7


@pytest.mark.parametrize("m", [0, 2])
def test_bor_sem_anisotropy_gradient_at_an_isotropic_layer_is_exact(m):
    """Claim (5) 'correct' family, re-measured: BORStack(basis='sem'),
    Rbig 2.5, n_hs 1.3, eps 4.5, thk 0.4, degree 6, d / d(in-plane
    anisotropy) at the isotropic layer.  Measured 2.9e-11 (m 0) / 2.8e-11
    (m 2), both builds; bar 1e-7."""
    import warnings

    import jax
    import jax.numpy as jnp

    from lumenairy.elements.bor.bor_stack import BORStack

    def solve(tri):
        s = BORStack(2.5, m, n_superstrate=1.3, n_substrate=1.3, basis="sem",
                     degree=6, n_mesh_cap=2.6)
        s.add_layer(0.4, segments=[(2.5, tri)])
        s.set_source(wavelength=2 * np.pi / 2.2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return s.solve()
    r0 = solve(jnp.asarray([4.5, 4.5, 4.5], dtype=complex))
    inc = np.nonzero(np.asarray(r0["inc_mask"]) > 0.5)[0][:3]

    def fb(d):
        d = jnp.asarray(d).astype(jnp.complex128)
        r = solve(jnp.stack([4.5 + d, 4.5 - d, 4.5 + 0.0 * d]))
        R, T = jnp.asarray(r["R"]), jnp.asarray(r["T"])
        return jnp.concatenate([jnp.stack([jnp.sum(R), jnp.sum(T)]), R[inc],
                                T[inc]]).real
    g = np.asarray(jax.jit(jax.jacrev(fb))(0.0))
    fd = _fd(lambda x: np.asarray(fb(jnp.asarray(x))), 0.0, 2e-2)
    assert _rel(g, fd) < 1e-7


_ISO = (_BASE[..., None, None] * np.eye(3)).astype(complex)


def _jones2d_f(D, formulation="laurent", cell=None):
    from lumenairy.elements.rcwa import rcwa_jones_2d
    c0 = _ISO if cell is None else cell

    def f(t, xp):
        kw = {} if xp is not np else {"symmetry": False}
        _o, R, T, _J = rcwa_jones_2d(_PX, _PX, xp.asarray(c0)
                                     + t * xp.asarray(D), 1.52, 1.0, 0.37,
                                     1.0, n_orders_x=2, n_orders_y=2,
                                     formulation=formulation, **kw)
        return xp.concatenate([xp.ravel(xp.asarray(R)),
                               xp.ravel(xp.asarray(T))]).real
    return f


_DX_T = (_DX[..., None, None] * np.eye(3)).astype(complex)


def test_rcwa_jones_2d_laurent_off_symmetry_is_exact():
    """Control for the defect below: 1e-4 off the C4v cell the unrouted
    ``rcwa_jones_2d`` (Laurent) gradient is right (measured 2.9e-11 /
    1.8e-11, Windows / WSL); bar 1e-7."""
    f = _jones2d_f(_DX_T)
    fd = _fd(lambda t: f(t, np), 1e-4, 2e-2)
    assert _rel(_ad(f, 1e-4), fd) < 1e-7


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT (unrouted RCWA entry, not measured by the builder): "
    "rcwa_jones_2d on JAX input at the C4v cell, x-widening of the post "
    "(isotropic tensors), is wrong by 6.1e-2 (Windows) / 4.4e-2 (WSL) "
    "relative, build-dependent -- the "
    "same degenerate-cluster class; its traced path (the general 4N "
    "generator cascade) is not routed through the rule.  Flips when routed."))
def test_rcwa_jones_2d_symmetry_breaking_gradient_at_the_c4v_cell():
    f = _jones2d_f(_DX_T)
    fd = _fd(lambda t: f(t, np), 0.0, 2e-2)
    assert _rel(_ad(f, 0.0), fd) < 1e-7


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT (unrouted RCWA entry): d / d(eps_xy) of rcwa_jones_2d at the "
    "isotropic C4v cell -- every central difference vanishes to 3.6e-13 "
    "(the true derivative is zero), AD returns 0.088 (Windows) / 0.083 "
    "(WSL), the "
    "Berreman-like class.  Flips when routed."))
def test_rcwa_jones_2d_eps_xy_gradient_at_an_isotropic_c4v_cell():
    XY = np.zeros((_S, _S, 3, 3), complex)
    XY[..., 0, 1] = XY[..., 1, 0] = _POST
    f = _jones2d_f(XY)
    # the oracle: the truth is zero to the FD's own resolution
    d = (np.asarray(f(1e-2, np)) - np.asarray(f(-1e-2, np))) / 2e-2
    assert np.max(np.abs(d)) < 1e-9
    assert np.max(np.abs(_ad(f, 0.0))) < 1e-6


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT, P1, PRE-EXISTING (identical on 5ea82b44): rcwa_jones_2d("
    "formulation='li') on a TRACED tensor cell (jax.jit / grad) routes to "
    "the general out-of-plane cascade (twod.py, the traced_tensor branch), "
    "whose operator build has no 'li' branch -- it silently solves the "
    "LAURENT formulation.  Measured: jitted forward = NumPy laurent to "
    "5e-15 and 7.6e-2 away from NumPy / eager li (C4v cell; 3.5e-4 on a C1 "
    "cell), and the 'li' gradient is 21-62 % wrong at ANY cell.  Flips when "
    "the traced path honours 'li'."))
def test_rcwa_jones_2d_li_under_jit_solves_li():
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.rcwa import rcwa_jones_2d

    def fw(c):
        _o, R, T, _J = rcwa_jones_2d(_PX, _PX, c, 1.52, 1.0, 0.37, 1.0,
                                     n_orders_x=2, n_orders_y=2,
                                     formulation="li")
        return jnp.concatenate([jnp.ravel(R), jnp.ravel(T)]).real
    eager = np.asarray(fw(jnp.asarray(_ISO)))
    jitted = np.asarray(jax.jit(fw)(jnp.asarray(_ISO)))
    assert _rel(jitted, eager) < 1e-10


@pytest.mark.xfail(strict=True, reason=(
    "DEFECT, P1, PRE-EXISTING: the rcwa_jones_2d 'li' gradient on a cell "
    "with NO symmetry (eps 2.25 + 0.6 x random) is wrong by 0.24 relative "
    "(both builds, both trees) -- it is the Laurent model's gradient (see "
    "the jit test above).  Flips when the traced path honours 'li'."))
def test_rcwa_jones_2d_li_gradient_on_a_cell_without_symmetry():
    cell = ((2.25 + 0.6 * _RAND)[..., None, None] * np.eye(3)).astype(
        complex)
    D = (_DCORNER[..., None, None] * np.eye(3)).astype(complex)
    f = _jones2d_f(D, formulation="li", cell=cell)
    fd = _fd(lambda t: f(t, np), 0.0, 2e-2)
    assert _rel(_ad(f, 0.0), fd) < 1e-7
