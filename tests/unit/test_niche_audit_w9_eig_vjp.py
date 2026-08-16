"""W9 audit -- the shared custom-VJP ``eig`` (:func:`_jax_eig_stable`) and its
DEGENERACY behaviour.

Background (measured 2026-07-27).  The eigenvector cotangent factor is exactly
``F_ij = 1 / conj(lam_j - lam_i)``; the Lorentzian ``F = D / (|D|^2 + eps)``
exists only to keep that finite when two eigenvalues collide.  ``eps`` used to
be an ABSOLUTE ``1e-10``, but the eigenvalues of these modal operators are
dimensionful (``max|lam|`` ~ 6e2 on the PMM spectral-element fold, ~3e1 on the
RCWA ``P@Q`` fold, ~1 on the Berreman 4x4), so an absolute floor corrupted a
scale-dependent window: it was wrong whenever ``|D| <~ sqrt(eps) = 1e-5``, which
on the PMM fold is a RELATIVE splitting of only 1.6e-8.  Symptom on the reported
probe (``pmm_efficiency_1d`` TE, ``d(sum R)/d(theta)`` at 1e-6 rad off normal):
AD ``4.217e-05`` against FD ``1.755e-06`` -- a 24x error.  The floor is now
SPECTRUM-RELATIVE (``_EIG_TAU_REL`` * ``max|lam|``), which is the exact formula
wherever the splitting is numerically resolved.

KNOWN LIMIT, pinned here deliberately: at an EXACT (symmetry-enforced)
degeneracy no choice of ``F`` can be right.  For a matrix-function loss
``L = tr(g(A) X)`` -- which is what a layer ``V exp(i q d) V^-1`` is -- the
eigenvector cotangent carries ``M_ij = (g(lam_j) - g(lam_i)) Y_ji`` with
``Y = V^-1 X V``, so the physical factor is the divided difference
``M_ij / D_ij -> g'(lam) Y_ji``.  When ``lam_i == lam_j`` EXACTLY, ``M_ij`` is
identically zero and ``Y_ji`` is simply absent from the cotangent; ``eig``
itself is not differentiable there (``V`` jumps by a direction-dependent
in-subspace rotation).  It bites only where the perturbation's intra-cluster
block is non-diagonal -- for the PMM 1-D fold ``A = eps I - Lop/k0^2`` that is
``d/d(angle)`` at EXACTLY normal incidence and nothing else.

CI note: these are gradient / eigensolver outputs, so every bound is a
calibrated inequality with >= 10x headroom over the measured value -- never an
equality.  The only ``==`` here is the primal-vs-``jnp.linalg.eig`` bit-identity
(same process, same call, so exact equality is well defined).
"""
import functools

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.scipy.linalg import expm  # noqa: E402

from lumenairy.elements.rcwa import _core as _rc  # noqa: E402
from lumenairy.elements.rcwa._core import _EIG_TAU_REL, _jax_eig_stable  # noqa: E402

_CJ = jnp.complex128


def _degenerate_splitting_eig(tau_rel, splitting):
    """A LIVE copy of ``_jax_eig_stable``'s custom-VJP ``eig``, verbatim except
    for two injected knobs, for the fail-before below.

    ``FIX_RUNNER_PINS_2026_08_12`` S7.  It is a reference IMPLEMENTATION rather
    than a stored number, so both arms are the same round trip in the same
    process and nothing in the comparison can drift with the platform.

    * ``splitting`` scales ``D = lam_j - lam_i`` for pairs that are ALREADY
      degenerate (``|D| < 1e-9`` of the spectrum scale).  Well-separated pairs
      are left alone, so the physical part of the gradient -- everything the
      clean and sweep arms measure -- is untouched.  What it emulates is a
      LAPACK build that resolves a symmetry-enforced degeneracy more exactly
      than this one does, which is the runner, not the physics: the splitting
      of an exactly degenerate pair is pure round-off and nothing else.
    * ``tau_rel`` is the spectrum-relative floor, ``0`` for the unfloored arm.
    """
    @functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
    def _eig(A, tau=tau_rel, split=splitting):
        lam, V = jnp.linalg.eig(A)
        return lam, V

    def _fwd(A, tau, split):
        lam, V = jnp.linalg.eig(A)
        return (lam, V), (lam, V)

    def _bwd(tau, split, res, cot):
        lam, V = res
        lam_bar, V_bar = cot
        raw = lam[None, :] - lam[:, None]
        n = lam.shape[0]
        offdiag = 1.0 - jnp.eye(n, dtype=raw.dtype)
        scale = jnp.max(jnp.abs(lam))
        scale = jnp.where(scale > 0, scale, 1.0)
        D = jnp.where(jnp.abs(raw) < 1e-9 * scale, raw * split, raw)
        floor = (tau * scale) ** 2
        denom = jnp.abs(D) ** 2 + floor
        F = jnp.where(offdiag != 0, D / jnp.where(denom == 0, 1.0, denom), 0.0)
        Vinv = jnp.linalg.inv(V)
        VinvH = jnp.conj(Vinv).T
        VH = jnp.conj(V).T
        Mmat = VH @ jnp.conj(V_bar)
        inner = jnp.diag(jnp.conj(lam_bar)) + F * Mmat
        return (jnp.conj(VinvH @ inner @ VH),)

    _eig.defvjp(_fwd, _bwd)
    return _eig

# ---------------------------------------------------------------- probes ---

_P1 = dict(period=0.8e-6, depth=0.35e-6, duty=0.45, wl=0.633e-6,
           n_r=2.0, n_g=1.0, n_sub=1.52, n_sup=1.0)


def _pmm1(angle, pol="te", n_r=None, depth=None):
    from lumenairy.elements.pmm import pmm_efficiency_1d
    o, R, T = pmm_efficiency_1d(
        _P1["period"],
        jnp.asarray(_P1["n_r"] if n_r is None else n_r, _CJ),
        jnp.asarray(_P1["n_g"], _CJ), jnp.asarray(_P1["n_sub"], _CJ),
        jnp.asarray(_P1["n_sup"], _CJ),
        jnp.asarray(_P1["depth"] if depth is None else depth),
        _P1["duty"], _P1["wl"], angle=angle, polarization=pol, degree=12,
        stabilize=False)
    return jnp.sum(R)


def _rcwa1(angle, pol="te"):
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    out = rcwa_efficiency_1d(
        _P1["period"], jnp.asarray(_P1["n_r"] ** 2 + 0j),
        jnp.asarray(_P1["n_g"] ** 2 + 0j), jnp.asarray(_P1["n_sub"] + 0j),
        jnp.asarray(_P1["n_sup"] + 0j), jnp.asarray(_P1["depth"]),
        _P1["duty"], jnp.asarray(_P1["wl"]), angle=angle, polarization=pol,
        n_orders=7)
    return jnp.sum(out[1])


def _central(f, x0, h):
    return float((f(jnp.asarray(x0 + h)) - f(jnp.asarray(x0 - h))) / (2.0 * h))


# =========================================================== the fix itself

def test_eig_tau_rel_is_a_relative_floor_between_noise_and_physics():
    """``_EIG_TAU_REL`` must sit above the LAPACK eigenvalue rounding floor
    (measured 2.6e-17 relative on the PMM half-space fold, ~1e-15 on a
    cond(V)~30 constructed case) and below the smallest splitting the physics
    needs (4.9e-11 relative at 1e-8 rad off normal)."""
    assert 1e-15 < _EIG_TAU_REL < 1e-11


def test_custom_vjp_does_not_perturb_the_primal():
    """The custom VJP must leave the FORWARD solve bit-identical to a plain
    ``jnp.linalg.eig``.  Same process / same call -> exact equality is legal."""
    eig = _jax_eig_stable()
    rng = np.random.default_rng(4711)
    for n in (4, 9):
        A = jnp.asarray(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
        lam_c, V_c = eig(A)
        lam_r, V_r = jnp.linalg.eig(A)
        assert np.array_equal(np.asarray(lam_c), np.asarray(lam_r))
        assert np.array_equal(np.asarray(V_c), np.asarray(V_r))


def test_eig_vjp_is_scale_equivariant():
    """THE semantic of the fix.  ``L`` below is mathematically INDEPENDENT of
    the scaling ``c`` (the eig of ``c A`` has eigenvalues ``c lam`` and the same
    eigenvectors, and the loss divides them straight back out), so the gradient
    must be too.  An ABSOLUTE broadening floor breaks that -- its significance
    relative to the spectrum changes by ``c^2`` -- while a spectrum-relative
    floor preserves it exactly."""
    eig = _jax_eig_stable()
    rng = np.random.default_rng(90210)
    n = 6
    S = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    X = jnp.asarray(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    d = np.array([1.0, 1.0 + 3e-7, 2.3, -0.7, 3.1, -1.9])   # near-degenerate
    A0 = jnp.asarray(S @ np.diag(d) @ np.linalg.inv(S))

    def loss(A0_, c):
        lam, V = eig(c * A0_)
        E = V @ jnp.diag(jnp.exp(lam / c)) @ jnp.linalg.inv(V)
        t = jnp.trace(E @ X)
        return jnp.real(t * jnp.conj(t))

    g1 = np.asarray(jax.grad(loss)(A0, 1.0))
    ref = np.linalg.norm(g1)
    for c in (1e-3, 1e3):
        gc = np.asarray(jax.grad(loss)(A0, c))
        # Measured 1.1e-08 (c=1e-3) / 7.0e-09 (c=1e3) -- that residual is the
        # FORWARD rounding of eig(c A) amplified by the 3e-7 splitting, not the
        # VJP.  The absolute floor gave 7.8e-04, i.e. ~1e5x larger.
        assert np.linalg.norm(gc - g1) / ref < 1e-6, f"c={c}"


def test_eig_vjp_stays_jit_and_vmap_compatible():
    """The relative floor reads ``max|lam|`` off a TRACED array, so it must not
    introduce a host branch.  ``vmap`` must also give each batch element its
    OWN scale (an absolute floor could not) -- pinned by requiring the batched
    gradient to equal the looped one on a batch of differently-scaled
    matrices."""
    eig = _jax_eig_stable()
    rng = np.random.default_rng(7)
    X = jnp.asarray(rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)))

    def loss(A):
        # RESOLVENT (A - z I)^-1, not expm: gauge-invariant like the oracle but
        # overflow-free across the 1e6 dynamic range of the batch below.
        lam, V = eig(A)
        E = V @ jnp.diag(1.0 / (lam - (7.0 + 5.0j))) @ jnp.linalg.inv(V)
        t = jnp.trace(E @ X)
        return jnp.real(t * jnp.conj(t))

    A = jnp.asarray(rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)))
    g = jax.grad(loss)(A)
    assert np.allclose(jax.jit(jax.grad(loss))(A), g, rtol=0, atol=1e-12)

    batch = jnp.stack([A, 1e3 * A, 1e-3 * A])       # three different scales
    gv = jax.vmap(jax.grad(loss))(batch)
    assert gv.shape == (3, 5, 5)
    for i in range(3):
        assert np.allclose(gv[i], jax.grad(loss)(batch[i]), rtol=0, atol=1e-12)
    assert np.allclose(jax.jit(jax.vmap(jax.grad(loss)))(batch), gv,
                       rtol=0, atol=1e-12)


def test_eig_vjp_degenerate_edge_cases_stay_finite():
    """A zero matrix drives ``max|lam|`` to 0 (the ``scale`` fallback) and every
    off-diagonal ``D`` to 0 at once -- the worst case for the floor.  It must
    return finite, not NaN."""
    eig = _jax_eig_stable()
    for A in (jnp.zeros((3, 3), _CJ), jnp.eye(4, dtype=_CJ)):
        g = jax.grad(lambda a: jnp.real(jnp.sum(eig(a)[0])))(A)
        assert np.all(np.isfinite(np.asarray(g)))


# ============================ entrywise oracle on CONSTRUCTED degeneracies ==
#
# L(A) = |tr(expm(A) X)|^2 computed (1) through the eig and (2) through
# jax.scipy.linalg.expm, whose VJP is known-correct.  Comparing dL/dA entrywise
# is an EXACT oracle -- no finite differences.  The loss is GAUGE-INVARIANT by
# construction (it sees only the matrix function, never the eigenvector basis),
# so the LAPACK-dependent choice of basis inside a degenerate subspace cannot
# affect it -- which is what makes this pin portable.

_ORACLE_SEED = 20260727


def _oracle_case(split, n=6, mult=2, eig=None):
    """``eig`` overrides the shipped custom-VJP eig, so a caller can run the
    IDENTICAL oracle through an injected variant (see
    :func:`_degenerate_splitting_eig`) and compare the two in one process."""
    rng = np.random.default_rng(_ORACLE_SEED)
    S = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    S = S / np.linalg.norm(S) * 3.0
    X = jnp.asarray(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    d = np.array([1.0, 1.0, 2.3, -0.7, 3.1, -1.9])[:n]
    for j in range(1, mult):
        d[j] = d[0] + j * split
    A = jnp.asarray(S @ np.diag(d) @ np.linalg.inv(S))
    eig = _jax_eig_stable() if eig is None else eig

    def L_expm(a):
        t = jnp.trace(expm(a) @ X)
        return jnp.real(t * jnp.conj(t))

    def L_eig(a):
        lam, V = eig(a)
        E = V @ jnp.diag(jnp.exp(lam)) @ jnp.linalg.inv(V)
        t = jnp.trace(E @ X)
        return jnp.real(t * jnp.conj(t))

    g_true = np.asarray(jax.grad(L_expm)(A))
    g_eig = np.asarray(jax.grad(L_eig)(A))
    return np.linalg.norm(g_eig - g_true) / np.linalg.norm(g_true)


@pytest.mark.parametrize("split,bound", [
    (1.0,   1e-9),      # measured 1.4e-14 (absolute floor: 2.3e-09)
    (1e-2,  1e-9),      # measured 4.2e-14 (absolute floor: 7.4e-07)
    (1e-4,  1e-7),      # measured 2.4e-11 (absolute floor: 7.3e-03)
    (1e-6,  1e-5),      # measured 1.4e-09 (absolute floor: 7.3e-01)
])
def test_oracle_resolved_splitting_is_exact(split, bound):
    """Wherever the splitting is numerically RESOLVED the eig VJP must match
    the ``expm`` VJP entrywise.  The pre-fix absolute floor missed by up to
    73% at split 1e-6 -- five to six orders outside these bounds."""
    assert _oracle_case(split) < bound


@pytest.mark.parametrize("outlier", [1.0, 1e3, 1e6])
def test_oracle_accuracy_depends_only_on_the_RELATIVE_splitting(outlier):
    """``max|lam|`` (not a per-pair scale) is the right normaliser: the LAPACK
    eigenvalue error is set by the global norm ~``eps_mach * ||A||``, so a
    splitting is resolvable exactly when it is large RELATIVE to the whole
    spectrum.  Pinned by hiding a tiny near-degenerate cluster under an outlier
    eigenvalue 1x / 1e3x / 1e6x larger and requiring the SAME accuracy at the
    same ``split / max|lam|`` -- a scale-free envelope, which an absolute floor
    could never provide."""
    rng = np.random.default_rng(31337)
    n = 6
    S = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    S = S / np.linalg.norm(S) * 3.0
    X = jnp.asarray(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    eig = _jax_eig_stable()
    split = 1e-7 * max(outlier, 3.1e-3)          # fixed RELATIVE splitting 1e-7
    d = np.array([1e-3, 1e-3 + split, 2.3e-3, -0.7e-3, 3.1e-3, outlier])
    A = jnp.asarray(S @ np.diag(d) @ np.linalg.inv(S))
    sc = max(outlier, 1.0)

    def L_expm(a):
        t = jnp.trace(expm(a / sc) @ X)
        return jnp.real(t * jnp.conj(t))

    def L_eig(a):
        lam, V = eig(a)
        E = V @ jnp.diag(jnp.exp(lam / sc)) @ jnp.linalg.inv(V)
        t = jnp.trace(E @ X)
        return jnp.real(t * jnp.conj(t))

    g_true = np.asarray(jax.grad(L_expm)(A))
    g_eig = np.asarray(jax.grad(L_eig)(A))
    err = np.linalg.norm(g_eig - g_true) / np.linalg.norm(g_true)
    assert err < 1e-5, f"outlier={outlier}: {err:.3e}"   # measured ~1.2e-08


@pytest.mark.parametrize("mult", [2, 3])
def test_oracle_exact_degeneracy_is_bounded_not_correct(mult):
    """The KNOWN LIMIT, pinned so it cannot silently change.

    At an EXACT degeneracy the divided-difference factor is absent from the
    cotangent (see the module docstring), so the eig-route gradient is NOT the
    true one -- measured 0.16-0.75 relative for EVERY variant (floored,
    unfloored, absolute).  What the relative floor DOES guarantee is that the
    error stays BOUNDED: it caps ``|F|`` at ``1/(2 tau max|lam|)`` instead of
    dividing by pure rounding noise.

    2026-08-15 (``FIX_RUNNER_PINS_2_2026_08_15`` S2 sibling sweep).  The
    boundedness used to rest on the absolute ``err < 5.0`` alone, which is
    6.7x over the worst variant ever recorded (0.75) -- inside one decade of a
    per-build quantity, and this file's own headline finding is what that
    costs.  The load-bearing arm is now COMPARATIVE and measured in this same
    process: the same exactly-degenerate case, run through a live copy of the
    VJP with the floor REMOVED and the (already round-off) splitting shrunk by
    1e-8, is eight decades worse.  Measured 2026-08-15::

        mult   shipped (floored)   unfloored + splitting x 1e-8   ratio
        2      0.3131 [W] [M]      3.619e+07 [W] 3.619e+07 [M]    1.2e8
        3      0.4377 [W] [M]      6.192e+07 [W] 4.160e+07 [M]    1.4e8 / 9.5e7

    The shipped readings are IDENTICAL on both mounts (the fixture is a
    constructed matrix with a fixed seed), so the regime bar below is kept as
    the O(1) sanity it always was -- the eig route is missing ONE of several
    comparable terms, so its relative error is O(1) by construction -- while
    the decades of separation carry the actual claim.
    """
    err = _oracle_case(0.0, mult=mult)
    assert np.isfinite(err)
    # regime: one missing term among several comparable ones -> O(1), never
    # a blow-up.  Measured 0.3131 / 0.4377 on both mounts (0.16-0.75 across
    # all recorded variants); 5.0 is 6.7x the worst of those.
    assert err < 5.0
    # ...and the floor is what buys that.  Same case, same process, floor off.
    unfloored = _oracle_case(
        0.0, mult=mult,
        eig=_degenerate_splitting_eig(tau_rel=0.0, splitting=1e-8))
    assert unfloored > 1e4 * err, (
        f"removing the spectrum-relative floor left the exactly-degenerate "
        f"gradient error at {unfloored:.4g} against the floored {err:.4g} -- "
        f"only {unfloored / err:.4g}x.  The floor is supposed to be the "
        f"difference between an O(1) wrong answer and dividing by rounding "
        f"noise (measured ~1e8x); if it is not, either the injector no "
        f"longer reaches the degenerate block or the floor stopped binding")


# ================================================= the reported physical probe

def test_pmm1d_near_normal_angle_gradient_matches_fd():
    """THE regression pin for the reported defect.  At 1e-6 rad off normal the
    +/-m splitting is 3.2e-06 absolute (4.9e-09 relative), which the absolute
    floor swamped: AD 4.217e-05 vs FD 1.755e-06, a 24x error.  Post-fix the
    measured gap is 1.2e-10."""
    ad = float(jax.grad(_pmm1)(jnp.asarray(1e-6)))
    fd = _central(_pmm1, 1e-6, 1e-5)
    assert abs(ad - fd) < 1e-8                 # measured 1.2e-10 (>= 80x)
    assert abs(ad - fd) < 0.02 * abs(fd)       # pre-fix ratio was 24x


@pytest.mark.parametrize("theta,bound", [
    (1e-6, 1e-4),      # measured 2.5e-06  (absolute floor: 2.3e+01)
    (1e-5, 1e-4),      # measured 2.6e-06  (absolute floor: 1.3e-02)
])
def test_pmm1d_near_normal_angle_gradient_is_linear_in_theta(theta, bound):
    """FD-FREE oracle -- the most CI-portable statement available here.

    ``R(theta)`` is smooth and EVEN about normal incidence, so ``dR/dtheta`` is
    exactly LINEAR in theta as theta -> 0, with slope ``d2R/dtheta2``.  Taking
    that curvature from the always-clean theta=1e-4 point, the AD slope at
    smaller theta must reproduce ``curvature * theta``.  No finite differences,
    no step-size tuning, no cancellation -- and it is precisely the quantity
    the absolute floor destroyed (23x error at 1e-6 rad)."""
    curvature = float(jax.grad(_pmm1)(jnp.asarray(1e-4))) / 1e-4
    exact = curvature * theta
    ad = float(jax.grad(_pmm1)(jnp.asarray(theta)))
    assert abs(ad - exact) < bound * abs(exact)


def test_pmm1d_off_normal_angle_gradient_no_regression():
    """The already-clean off-normal gradient must not move.  Measured
    |AD - FD| / |FD| = 8.2e-06 (FD truncation dominated) both pre- and
    post-fix.

    2026-08-12 (``docs/audits/FIX_JAX_NAN_PINS_2026_08_12.md`` S3).  This read
    ``nan`` on WSL py3.12 / jax 0.10.2 -- and ONLY there -- until the
    incident-amplitude projection stopped going through ``jnp.linalg.lstsq``.
    The projection matrix carries a structurally repeated singular value, the
    SVD gradient divides by the square difference of two singular values with
    a guard on the diagonal only, and whether that difference underflows to
    zero is a per-LAPACK round-off fact.  Post-fix, both mounts read
    0.0363046947 against a Richardson-extrapolated FD oracle of the same
    (3.5e-11 [W] / 2.2e-11 [M] relative).  See the fail-before below."""
    ad = float(jax.grad(_pmm1)(jnp.asarray(0.3)))
    fd = _central(_pmm1, 0.3, 3e-4)
    assert abs(ad - fd) < 1e-4 * abs(fd)       # measured 8.2e-06 (12x)


def test_the_lstsq_projection_route_is_refuted_on_a_degenerate_projection(
        monkeypatch):
    """THE FAIL-BEFORE for the 2026-08-12 projection fix, driven both ways
    (``docs/audits/FIX_JAX_NAN_PINS_2026_08_12.md`` S3).

    The defect is NOT the eig VJP this file is about: ``_EIG_TAU_REL`` at
    1e-12 / 1e-10 / 1e-8 / 1e-6 all still gave ``nan``.  It is
    :func:`~lumenairy.elements.pmm._core._jpmm_min_norm_projection`'s
    predecessor -- ``jnp.linalg.lstsq``, whose gradient is the SVD JVP and so
    carries ``1 / (s_i^2 - s_j^2)`` with only the DIAGONAL guarded.

    Three claims, none of which depends on reproducing one build's round-off:

    (a) EXPOSURE, read off the SHIPPED projection.  ``Hsup`` is not
        incidentally degenerate: 3 of its 21 singular values sit on
        ``1 / sqrt(n_glob)`` (measured cluster spread 3.6e-16 [M] /
        1.4e-16 [W]), so the minimum relative gap is round-off -- 8.0e-16 [M] /
        1.3e-16 [W] at theta = 0.3, 2.0e-14 / 1.9e-14 at 0.29.  The
        normal-incidence branch (a python-literal ``kx0 = 0.0``) carries no
        such cluster: 5.4e-04 on both mounts, six decades away, which is why
        the defect is OBLIQUE-only;

    (b) MECHANISM, deterministic on every build.  On an EXACTLY degenerate
        projection of the same shape (all 420 off-diagonal ``s_i^2 - s_j^2``
        identically 0) the pre-fix route's gradient is not finite and the
        shipped route's is -- and is RIGHT, against a central difference of
        the same loss along a fixed direction;

    (c) FORWARD NULLITY.  The two routes are the same map: they return the
        same minimum-norm solution on the degenerate injector (measured
        7.3e-15 on |x| ~ 1) and on the live oblique projection (3e-15), so the
        fix changes the GRADIENT and not the answer.
    """
    from lumenairy.elements.pmm import _core as pmm

    seen = {}
    _real = pmm._jpmm_min_norm_projection

    def _capture(A, b, xp):
        seen.setdefault("A", A)
        seen.setdefault("b", b)
        return _real(A, b, xp)

    monkeypatch.setattr(pmm, "_jpmm_min_norm_projection", _capture)

    def _live(theta):
        """The projection the SHIPPED solve builds, PRIMAL only (so it is a
        concrete array and not a tracer)."""
        seen.clear()
        float(_pmm1(theta))
        return np.asarray(seen["A"]), np.asarray(seen["b"])

    def _rel_gap(A):
        s = np.sort(np.linalg.svd(A, compute_uv=False))[::-1]
        return float(np.min(np.abs(np.diff(s))) / s[0]), s

    # ---- (a) exposure, on the shipped object ------------------------------
    A03, b03 = _live(jnp.asarray(0.3))
    gap, s = _rel_gap(A03)
    n_glob = A03.shape[1]
    tie = 1.0 / np.sqrt(n_glob)
    assert gap < 1e-10, f"no repeated singular value: min rel gap {gap:.3e}"
    assert abs(s[int(np.argmin(np.abs(np.diff(s))))] - tie) < 1e-12
    assert int(np.sum(np.abs(s - tie) < 1e-12)) >= 3        # measured 3 of 21
    gap29, _s29 = _rel_gap(_live(jnp.asarray(0.29))[0])
    assert gap29 < 1e-10                                    # measured 2.0e-14
    assert _rel_gap(_live(0.0)[0])[0] > 1e-6                # measured 5.4e-04

    # ---- (b) the mechanism, on an EXACTLY degenerate projection -----------
    m, n = A03.shape
    A_deg = jnp.asarray(0.7 * np.eye(m, n), _CJ)
    sd = np.asarray(jnp.linalg.svd(A_deg, full_matrices=False,
                                   compute_uv=False))
    off = ~np.eye(len(sd), dtype=bool)
    assert int(np.sum(np.subtract.outer(sd ** 2, sd ** 2)[off] == 0.0)) == \
        int(off.sum()), "the injector is not exactly degenerate on this build"
    rng = np.random.default_rng(1013)
    bd = jnp.asarray(rng.normal(size=m) + 1j * rng.normal(size=m), _CJ)
    Dd = jnp.asarray(rng.normal(size=(m, n))
                     + 1j * rng.normal(size=(m, n)), _CJ)

    def _loss(t):
        x = pmm._jpmm_min_norm_projection(A_deg + t * Dd, bd, jnp)
        return jnp.sum(jnp.abs(x) ** 2)

    ad = float(jax.grad(_loss)(jnp.asarray(0.0)))
    fd = _central(_loss, 0.0, 1e-6)
    assert np.isfinite(ad)
    assert abs(ad - fd) < 1e-6 * abs(fd)        # finite AND right
    x_fix = np.asarray(pmm._jpmm_min_norm_projection(A_deg, bd, jnp))
    x_live_fix = np.asarray(
        pmm._jpmm_min_norm_projection(jnp.asarray(A03), jnp.asarray(b03), jnp))

    monkeypatch.setattr(pmm, "PMM_JAX_MINNORM_PROJECTION", False)
    assert not np.isfinite(float(jax.grad(_loss)(jnp.asarray(0.0)))), (
        "the pre-fix lstsq route returned a finite gradient on an EXACTLY "
        "degenerate projection -- the injector no longer reaches the defect, "
        "so this fail-before is vacuous")

    # ---- (c) forward nullity ---------------------------------------------
    # The two routes are DIFFERENT algorithms solving the same least-squares
    # problem, so "the same answer" can only mean "to the conditioning of the
    # problem".  The bar is therefore DERIVED here rather than pinned: a
    # backward-stable solve of a system with condition number ``kappa``
    # agrees to ~``eps * kappa`` relative, and 1e2 of headroom over that is
    # the claim (FIX_RUNNER_PINS_2_2026_08_15 S2 sibling sweep).  The frozen
    # 1e-13 it replaces measured 2.85e-15 [W] / 3.23e-15 [M] on the live
    # oblique projection -- 35x / 31x, inside one decade of a round-off
    # quantity.  cond(A03) = 5.196 and cond(A_deg) = 1 on both mounts, so the
    # derived bars come out at 1.2e-13 and 2.2e-14 and the measured values
    # sit 42x and 128x under them.
    _EPS = float(np.finfo(np.float64).eps)

    def _nullity_bar(A):
        return 1e2 * _EPS * float(np.linalg.cond(np.asarray(A)))

    x_pre = np.asarray(pmm._jpmm_min_norm_projection(A_deg, bd, jnp))
    x_live_pre = np.asarray(
        pmm._jpmm_min_norm_projection(jnp.asarray(A03), jnp.asarray(b03), jnp))
    r_deg = np.max(np.abs(x_fix - x_pre)) / np.max(np.abs(x_fix))
    r_live = np.max(np.abs(x_live_fix - x_live_pre)) / np.max(
        np.abs(x_live_fix))
    assert r_deg < _nullity_bar(A_deg), (
        f"degenerate injector: routes differ by {r_deg:.3e} relative against "
        f"a conditioning-derived {_nullity_bar(A_deg):.3e}")
    assert r_live < _nullity_bar(A03), (
        f"live oblique projection: routes differ by {r_live:.3e} relative "
        f"against a conditioning-derived {_nullity_bar(A03):.3e}")


@pytest.mark.parametrize("name,x0,h,bound", [
    ("n_r", 2.0, 1e-7, 1e-5),          # measured 2.1e-08 relative
    ("depth", 0.35e-6, 1e-10, 1e-3),   # measured 4.6e-06 (FD curvature bound)
])
def test_pmm1d_design_gradients_at_exactly_normal_incidence_are_clean(
        name, x0, h, bound):
    """The exact-degeneracy limit does NOT touch the design gradients.  The
    half-space fold is ``A = eps I - Lop/k0^2``: ``d/d(eps)`` gives ``U = I``
    and ``d/d(depth)`` does not perturb the half-space matrices at all, so
    their intra-cluster blocks are diagonal and the missing cross-block is
    never multiplied.  Only ``d/d(angle)`` injects the convection
    ``-1j kx0 (C - C^T)``, whose intra-cluster block is not diagonal."""
    f = (lambda v: _pmm1(0.0, n_r=v)) if name == "n_r" \
        else (lambda v: _pmm1(0.0, depth=v))
    ad = float(jax.grad(f)(jnp.asarray(x0)))
    fd = _central(f, x0, h)
    assert abs(ad - fd) < bound * abs(fd)


#: How many decades above the truth's own RESOLUTION the value at exactly
#: normal incidence must sit to count as still WRONG, and how far the finite
#: difference itself may sit above that resolution before the fixture is
#: disowned.  ``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S2, 2026-08-15.
#:
#: WHY THIS IS NOT A MAGNITUDE RATIO ANY MORE.  The 2026-08-12 remediation
#: replaced an absolute bar with two ratios (``|AD(0)|`` against the clean
#: near-normal gradient, and against the observable's physical gradient
#: scale).  Both still carry ``|AD(0)|``, and ``|AD(0)|`` is THE per-build
#: fact here -- it is the size of the eigenvector JUMP, and which direction
#: ``V`` jumps inside an exactly degenerate subspace is what a LAPACK build
#: is entitled to choose:
#:
#:     build                             TE AD(0)      AD(0)/clean
#:     authoring box (2026-07-27)       -2.221e-03      1.3e3
#:     Windows py3.14 / numpy 2.4.4     +7.793e-03      4.4e3
#:     WSL py3.12 / numpy 2.5.1         +1.506e-02      8.6e3
#:     CI ubuntu py3.12 (2026-08-12)    -2.664e-02      1.5e4
#:     CI ubuntu, numpy 2.5-era wheel   +2.910e-05      1.66e1   <- FAILED
#:
#: Three decades of spread and two sign flips, against a bar of 100.  The
#: envelope arm was in the same shape: ``|AD(0)|/sweep`` measured 1.333 [W]
#: against a bar of 30 (22x headroom) but 0.0179 [M] -- a 75x spread across
#: two builds of one source.
#:
#: THE DISCRIMINATOR IS NOW PRESENCE OF WRONGNESS.  Mirror symmetry forces the
#: true derivative to be EXACTLY 0 at normal incidence, so the question "is AD
#: still wrong?" needs no magnitude at all -- only the resolution at which
#: "zero" can be asserted.  For a function that is EVEN about 0 the central
#: difference is analytically zero at every step size (all truncation terms
#: cancel identically), so its whole reading is float64 cancellation and the
#: resolution is the derived ``eps_mach * |R(0)| / h``.  VERIFIED 2026-08-15:
#: the FD reading scales as 1/h over three decades of h on both mounts
#: (TE: 6.9e-13 / 2.1e-11 / 8.3e-10 at h = 1e-5 / 1e-6 / 1e-7) and lands at
#: 1.38x [W] and 0.46x [M] of the derived floor.
#:
#: ``_THETA0_WRONGNESS_FACTOR`` then separates the two regimes by decades in
#: BOTH directions, measured as ``|AD(0) - FD(0)| / resolution``:
#:
#:     still defective   1.9e6 (the failing runner) .. 2.0e10   [5 builds]
#:     defect FIXED      3.3e-4 (rcwa1d TE) .. 7.9e-5 (TM)      [2 mounts]
#:
#: 1e3 sits 3.3 decades below the smallest defective reading ever seen and
#: 6.5 decades above the largest clean one, so it can neither burn a tag nor
#: pass quietly on the day the upstream defect is closed.
_THETA0_WRONGNESS_FACTOR = 1e3
#: ...and the fixture's own symmetry, checked against the same resolution.
#: Measured |FD(0)|/resolution = 1.38 / 6.64 [W, te/tm], 0.46 / 0.35 [M];
#: 1e3 is 150x over the worst of those.
_THETA0_SYMMETRY_FACTOR = 1e3


def _zero_derivative_resolution(f, h):
    """The finest nonzero derivative a central difference of ``f`` at 0 can be
    told apart from zero, on THIS build: the float64 cancellation floor
    ``eps_mach * |f(0)| / h``.

    Exact for this fixture because ``R(theta)`` is EVEN about normal
    incidence, so ``f(h) - f(-h)`` is analytically zero and carries no
    truncation term at all -- every bit the quotient returns is round-off.
    """
    return float(np.finfo(np.float64).eps) * abs(float(f(jnp.asarray(0.0)))) / h


def _theta0_scores(f):
    """``(AD at exactly 0, FD at 0, |AD| at a clean 1e-6, max |AD| over a
    physical sweep, the FD's own zero-resolution)`` for an angle loss.

    Every number is measured in ONE process on the running build.  ``clean``
    and ``sweep`` are no longer asserted on -- they are the printed
    adjudication in the failure message, kept because a reader who trips this
    pin needs to know how big the wrong value was, not just that it was wrong.
    """
    ad0 = float(jax.grad(f)(jnp.asarray(0.0)))
    fd0 = _central(f, 0.0, 1e-6)
    clean = abs(float(jax.grad(f)(jnp.asarray(1e-6))))
    sweep = max(abs(float(jax.grad(f)(jnp.asarray(t))))
                for t in (0.05, 0.1, 0.2, 0.3, 0.5))
    return ad0, fd0, clean, sweep, _zero_derivative_resolution(f, 1e-6)


def _score_theta0_defect(f, name):
    """Score the OPEN eig-VJP degeneracy defect at exactly normal incidence.
    Returns the measurements.  Split out so the fail-before below can run the
    identical claims against a solver that does NOT carry the defect."""
    ad0, fd0, clean, sweep, res = _theta0_scores(f)
    # (0) the SYMMETRY oracle: the true derivative is zero, and the FD says so
    # to within its own derived cancellation floor.
    assert abs(fd0) < _THETA0_SYMMETRY_FACTOR * res, (
        f"{name}: d(sum R)/d(theta) is forced to ZERO at normal incidence by "
        f"mirror symmetry, but the finite difference reads {fd0:.4e} -- "
        f"{abs(fd0) / res:.4g}x its own float64 cancellation floor "
        f"({res:.3e}).  The fixture is not symmetric and nothing below means "
        f"what it says")
    assert np.isfinite(ad0), (
        f"{name}: AD at exactly normal incidence returned {ad0!r}.  The "
        f"spectrum-relative floor exists to keep this FINITE")
    # (1) THE DEFECT, still present: AD disagrees with the truth by decades
    # more than the truth's own resolution.  A future fix must make this fail.
    scale = max(res, abs(fd0))
    assert abs(ad0 - fd0) > _THETA0_WRONGNESS_FACTOR * scale, (
        f"{name}: AD at exactly normal incidence is {ad0:.4e}, which agrees "
        f"with the symmetry-forced truth (FD {fd0:.4e}) to within "
        f"{abs(ad0 - fd0) / scale:.4g}x the resolution at which that truth "
        f"can be asserted ({scale:.3e}) -- the eigenvector-VJP degeneracy no "
        f"longer corrupts it.  That is the KNOWN LIMIT being CLOSED: re-pin "
        f"this test against the fix and drop the theta >= 1e-8 advice from "
        f"the module docstring.  Do NOT loosen the factor.  "
        f"[adjudication: |AD(0)| is {abs(ad0) / max(clean, 1e-300):.4g}x the "
        f"clean AD at 1e-6 rad ({clean:.4e}) and {abs(ad0) / sweep:.4g}x the "
        f"largest gradient on a physical sweep ({sweep:.4e})]")
    return ad0, fd0, clean, sweep, res


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_pmm1d_angle_gradient_at_exactly_zero_is_an_OPEN_defect(pol):
    """The KNOWN LIMIT (see the module docstring), pinned as a DEFECT.
    A future fix should make this test fail.

    Mirror symmetry forces ``d(sum R)/d(theta) = 0`` at exactly normal
    incidence, but the exactly degenerate +/-m pair makes the eig
    non-differentiable there, so AD returns a finite WRONG value.  This has
    never been a correctness pin; it fences the value.

    2026-08-12 (``docs/audits/FIX_RUNNER_PINS_2026_08_12.md`` S7) fenced it
    with an ABSOLUTE bar -- 1e-2 (TE) -- and the ubuntu py3.12 JAX job read
    0.026644.  That bar could not have held, because the fenced quantity is
    pure round-off garbage whose MAGNITUDE is a per-build fact.  ADJUDICATED
    with the floor's own knob: the floored value is INSENSITIVE to the
    degenerate pair's numerical splitting (shrinking it by 1e-8 in a live copy
    of the VJP moves AD(0) from 7.792836e-03 to 7.793047e-03), so it is not a
    ``1/D`` reading at all -- it is the eigenvector JUMP, and which direction
    ``V`` jumps is exactly what a LAPACK build is entitled to choose.

    2026-08-15 (``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S2).  The
    2026-08-12 fix restated both halves as RATIOS against arms measured in the
    same process -- but both ratios still had ``|AD(0)|`` in the numerator, so
    both still asserted a magnitude the runners disagree about, and a numpy
    2.5-era ubuntu wheel duly read 16.6x against the bar of 100.  The defect
    arm is now a discriminator with no magnitude in it at all: AD must
    DISAGREE with the symmetry-forced truth (exactly 0) by decades more than
    the resolution at which that truth can be asserted, which is derived from
    the finite difference's own float64 cancellation floor.  See
    ``_THETA0_WRONGNESS_FACTOR`` for the two-sided regime separation.

    The boundedness half moved out to
    :func:`test_the_theta0_floor_bounds_the_wrong_value_but_the_unfloored_vjp_does_not`,
    where it is stated as the MECHANISM it actually is (the floored value does
    not move when the degenerate pair's splitting is shrunk; the unfloored one
    scales as 1/splitting) rather than as a ratio against a physical sweep.

    (Correction to the pre-2026-08-12 docstring, which recorded the unfloored
    value as 7.7x WORSE: on this build it is 2.28x BETTER -- floored
    +7.793e-03 against unfloored -3.411e-03.  Which of the two is larger at an
    exact degeneracy is not a property the floor controls; the boundedness
    under a shrinking splitting is.)

    Use ``theta >= 1e-8`` rad if the angle derivative itself is the objective.
    """
    _score_theta0_defect(lambda a: _pmm1(a, pol=pol), f"pmm1d {pol}")


def test_the_theta0_defect_pin_fires_when_the_defect_is_absent():
    """THE FAIL-BEFORE for the defect pin above.

    ``FIX_RUNNER_PINS_2026_08_12`` S7; discriminator restated
    ``FIX_RUNNER_PINS_2_2026_08_15`` S2.

    The same claims are scored against ``rcwa1d``, which solves the identical
    grating with ANALYTIC homogeneous half-space modes and therefore has no
    degeneracy at normal incidence (its theta = 0 gradient is machine-zero,
    8.9e-14 [W] / 6.5e-14 [M], pinned by the control test below).  That is
    what "the eig-VJP degeneracy was fixed" looks like from outside, and the
    pin must FAIL on it, telling the reader to re-pin rather than passing
    quietly.

    This arm is what makes the new discriminator two-sided, and it is
    MEASURED, not assumed: scored against its own resolution, ``rcwa1d``
    reads ``|AD(0) - FD(0)| / resolution`` = 1.0 [W] / 1.0 [M] against a bar
    of 1e3, i.e. it fails by three decades.  The defective ``pmm1d`` arm
    passes the same bar by three to seven decades in the other direction.
    """
    with pytest.raises(AssertionError, match="re-pin this test"):
        _score_theta0_defect(lambda a: _rcwa1(a, pol="te"), "rcwa1d te")


def test_the_theta0_floor_bounds_the_wrong_value_but_the_unfloored_vjp_does_not():
    """The BOUNDEDNESS half of the known limit, stated as the mechanism it is.

    ``FIX_RUNNER_PINS_2_2026_08_15`` S2.  Until 2026-08-15 this was a ratio
    of ``|AD(0)|`` to the observable's largest physical gradient, with a bar
    of 30 -- and TM measured 1.333 on Windows against 0.0179 on WSL, a 75x
    cross-build spread on a bar with 22x headroom.  The claim it was standing
    in for is not about how big the wrong value is; it is that the
    spectrum-relative floor is what keeps ``F = D / (|D|^2 + eps)`` from
    dividing by pure rounding noise at an exact degeneracy.  That is a LAW,
    and a law can be measured directly.

    Both arms use a LIVE copy of the library's eig VJP, verbatim except for
    two knobs (:func:`_degenerate_splitting_eig`), installed over
    ``_JAX_EIG_STABLE`` so the whole solve runs through it.  The knobs shrink
    the numerical splitting of pairs that are ALREADY degenerate (``|D| <
    1e-9`` of the spectrum scale, so the well-separated pairs -- and hence the
    physical part of the gradient -- are untouched) and optionally drop the
    floor.  A LAPACK build that resolves a symmetry-enforced degeneracy more
    exactly is exactly what this emulates.

    * FLOORED, the shipped behaviour: shrinking the splitting by 1e-8 leaves
      ``AD(0)`` where it was.  Measured relative move 2.73e-05 [W] /
      2.73e-06 [M] -- the wrong value is the eigenvector JUMP, not a ``1/D``
      reading, and it is BOUNDED.
    * UNFLOORED: the same shrink scales ``AD(0)`` by exactly ``1/splitting``.
      Measured ``|AD(0)| at 1e-8 / |AD(0)| at 1e-4`` = 1.0000e+04 on BOTH
      mounts, against a bar of 1e3.  That is the whole of "unbounded", and it
      is exact arithmetic (``F -> 1/conj(D)``), not a calibrated magnitude.

    Neither arm contains an absolute bar or a cross-arm magnitude, so neither
    can move with the runner.
    """
    orig = _rc._JAX_EIG_STABLE
    try:
        # FLOORED: insensitive to the splitting -> bounded.
        def _floored(split):
            _rc._JAX_EIG_STABLE = _degenerate_splitting_eig(
                tau_rel=_EIG_TAU_REL, splitting=split)
            return float(jax.grad(lambda a: _pmm1(a, pol="te"))(
                jnp.asarray(0.0)))

        base = _floored(1.0)
        shrunk = _floored(1e-8)
        assert abs(base) > 0.0
        moved = abs(shrunk - base) / abs(base)
        assert moved < 1e-2, (
            f"the FLOORED AD(0) moved by {moved:.3e} relative when the "
            f"degenerate pair's splitting was shrunk by 1e-8 ({base:.6e} -> "
            f"{shrunk:.6e}).  It is supposed to be insensitive to that "
            f"splitting -- if it is not, the wrong value IS a 1/D reading "
            f"and the floor is not doing what this test says it does")

        # UNFLOORED: scales as 1/splitting -> unbounded.
        def _unfloored(split):
            _rc._JAX_EIG_STABLE = _degenerate_splitting_eig(tau_rel=0.0,
                                                            splitting=split)
            return float(jax.grad(lambda a: _pmm1(a, pol="te"))(
                jnp.asarray(0.0)))

        u4 = _unfloored(1e-4)
        u8 = _unfloored(1e-8)
        blowup = abs(u8) / abs(u4)
        assert blowup > 1e3, (
            f"with the floor REMOVED, shrinking an already-degenerate pair's "
            f"splitting by 1e-4 scaled AD(0) by only {blowup:.4g}x "
            f"({u4:.4e} -> {u8:.4e}); F = 1/conj(D) makes that scaling exact, "
            f"so anything but ~1e4 means the injector no longer reaches the "
            f"degenerate block and this arm is vacuous")
        # ...and the floored arm really is the same solve, not a different one
        assert abs(u4) > abs(base), (
            "the unfloored injection did not even exceed the shipped value; "
            "the two arms are not exercising the same degeneracy")
    finally:
        _rc._JAX_EIG_STABLE = orig


# ================================================== sweep clean-map fences ==

@pytest.mark.parametrize("pol", ["te", "tm"])
def test_rcwa1d_normal_incidence_control_stays_clean(pol):
    """The no-regression CONTROL.  RCWA uses ANALYTIC homogeneous half-space
    modes, so its single eig has no degeneracy at normal incidence (measured
    min relative gap 4.6e-04) and its theta=0 gradient is machine-zero:
    measured 1.1e-13 (TE) / 2.6e-15 (TM), unchanged by the fix."""
    ad = float(jax.grad(lambda a: _rcwa1(a, pol=pol))(jnp.asarray(0.0)))
    assert abs(ad) < 1e-7                       # measured <= 1.1e-13


def test_rcwa1d_off_normal_control_stays_clean():
    ad = float(jax.grad(_rcwa1)(jnp.asarray(0.3)))
    fd = _central(_rcwa1, 0.3, 3e-4)
    assert abs(ad - fd) < 1e-4 * abs(fd)        # measured 5.8e-07


def test_berreman_exact_kz_degeneracy_is_benign():
    """The Berreman 4x4 has EXACTLY degenerate ``kz`` pairs at every incidence
    (measured off-diagonal gap identically 0.0), yet its gradients are clean --
    the existence proof that an exact degeneracy only hurts when the
    perturbation's intra-cluster block is non-diagonal.  Swept over a
    shrinking birefringence down to zero; measured <= 4.0e-10 relative."""
    from lumenairy.elements.berreman import berreman_jones_1d
    no2 = 2.1

    def f(t, delta):
        eps = jnp.diag(jnp.asarray([no2 + delta, no2, no2], dtype=_CJ))
        R, T, Jr, Jt = berreman_jones_1d(
            [(eps, t)], jnp.asarray(1.5 + 0j), jnp.asarray(1.0 + 0j),
            jnp.asarray(0.633e-6), angle=jnp.asarray(0.4),
            phi=jnp.asarray(0.5))
        return jnp.sum(jnp.asarray(R))

    for delta in (1e-5, 0.0):                   # 0.0 == exactly isotropic
        ad = float(jax.grad(f)(jnp.asarray(180e-9), delta))
        fd = _central(lambda t: f(t, delta), 180e-9, 1e-13)
        assert abs(ad - fd) < 1e-6 * abs(fd), f"delta={delta}"


def test_eme_symmetric_cell_degenerate_pair_is_clean():
    """EME at ``kx0 = ky0 = 0`` on a 4-fold symmetric cell has EXACTLY
    degenerate mode pairs (measured gap 7.1e-13 between modes 1 and 2).  A
    cluster-COMPLETE window over such a pair is a smooth gauge-invariant loss
    and its gradient is clean: measured 3.3e-11 relative."""
    from lumenairy.elements.eme import ref_2d_modes
    nx = ny = 6
    e0 = np.full((nx, ny), 2.0)
    e0[2:4, 2:4] = 6.0

    def f(epix):
        e = jnp.asarray(e0)
        for i in (2, 3):
            for j in (2, 3):
                e = e.at[i, j].set(epix)
        s = ref_2d_modes(e, 1.0, 1.0, nx, ny, 8.0, kx0=0.0, ky0=0.0)
        return s[1] + s[2]                      # the degenerate pair, complete

    ad = float(jax.grad(f)(jnp.asarray(6.0)))
    fd = _central(f, 6.0, 1e-4)
    assert abs(ad - fd) < 1e-6 * abs(fd)        # measured 3.3e-11


def test_bor_axisymmetric_gradient_is_clean():
    """BOR is axisymmetric; its radial fold showed no degeneracy on the swept
    geometries.  Measured 2.0e-08 relative, unchanged by the fix."""
    from lumenairy.elements.bor.bor_stack import BORStack
    wl = 2 * np.pi / 2.0

    def f(nr):
        s = BORStack(Rbig=3.0, m=1, N=56, n_superstrate=1.4142,
                     n_substrate=1.4142)
        s.add_layer(0.5, rings=(0.8, 0.5, nr, 1.414))
        return jnp.sum(jnp.asarray(s.set_source(wavelength=wl).solve()["R"]))

    ad = float(jax.grad(f)(jnp.asarray(2.449)))
    fd = _central(f, 2.449, 1e-6)
    assert abs(ad - fd) < 1e-5 * abs(fd)        # measured 2.0e-08


def test_pmm2d_near_normal_angle_gradient_improved():
    """The PMM 2-D twin at normal incidence: the spectral-element mesh breaks
    the exact +/-m pairing, so the splitting IS resolved and the fix applies in
    full -- measured |AD - FD| / |FD| 4.1e-05 -> 1.1e-06 (37x)."""
    from lumenairy.elements.pmm import pmm_efficiency_2d
    p, wl, dep = 0.6e-6, 0.55e-6, 0.25e-6
    xb = (0.2 * p, 0.6 * p)

    def f(theta):
        o, R, T = pmm_efficiency_2d(p, p, jnp.asarray(6.0 + 0j, _CJ), 1.0,
                                    xb, xb, 1.5, 1.0, jnp.asarray(dep), wl,
                                    theta=theta, degree=5, n_orders=2,
                                    polarization="te")
        return jnp.sum(R)

    ad = float(jax.grad(f)(jnp.asarray(0.0)))
    fd = _central(f, 0.0, 1e-6)
    assert abs(ad - fd) < 1e-4 * abs(fd)        # measured 1.1e-06
