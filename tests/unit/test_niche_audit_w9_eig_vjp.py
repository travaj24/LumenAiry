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


def _oracle_case(split, n=6, mult=2):
    rng = np.random.default_rng(_ORACLE_SEED)
    S = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    S = S / np.linalg.norm(S) * 3.0
    X = jnp.asarray(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    d = np.array([1.0, 1.0, 2.3, -0.7, 3.1, -1.9])[:n]
    for j in range(1, mult):
        d[j] = d[0] + j * split
    A = jnp.asarray(S @ np.diag(d) @ np.linalg.inv(S))
    eig = _jax_eig_stable()

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
    dividing by pure rounding noise.  Dropping the floor made the physical
    probe 7.7x worse (pmm 1-D theta=0: 1.71e-02 vs 2.22e-03), so this bound is
    a real fence around removing it."""
    err = _oracle_case(0.0, mult=mult)
    assert err < 5.0
    assert np.isfinite(err)


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
    post-fix."""
    ad = float(jax.grad(_pmm1)(jnp.asarray(0.3)))
    fd = _central(_pmm1, 0.3, 3e-4)
    assert abs(ad - fd) < 1e-4 * abs(fd)       # measured 8.2e-06 (12x)


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


#: The exactly-degenerate point's two RATIO bars, both between arms measured
#: in one process on the running build (``FIX_RUNNER_PINS_2026_08_12`` S7).
#:
#: ``_THETA0_DEFECT_RATIO`` -- how far above the CLEAN near-normal gradient the
#: wrong value at exactly zero must sit for the defect to count as still
#: present.  ``AD(theta)`` is exactly linear in theta just off normal (the
#: sibling test pins that), so the correct value at theta = 0 is 0 and the
#: clean ``AD(1e-6)`` is the scale a fixed eig-VJP would return something near.
#: Measured 4.4e3 (TE) / 1.6e5 (TM), i.e. 44x / 1580x headroom.
#:
#: ``_THETA0_ENVELOPE_RATIO`` -- how far above the observable's OWN physical
#: gradient scale the wrong value may sit before it counts as blown up.
#: Measured 2.5e-2 (TE) / 1.33 (TM), i.e. 1200x / 22x headroom, and the
#: injected blow-up in the fail-before reads 3.6e2 / 1.9e4.
_THETA0_DEFECT_RATIO = 100.0
_THETA0_ENVELOPE_RATIO = 30.0


def _theta0_scores(f):
    """``(AD at exactly 0, FD at 0, |AD| at a clean 1e-6, max |AD| over a
    physical sweep)`` for a one-argument angle loss.

    Every number is measured in ONE process on the running build, so the
    ratios taken from them carry no runner arithmetic at all.
    """
    ad0 = float(jax.grad(f)(jnp.asarray(0.0)))
    fd0 = _central(f, 0.0, 1e-6)
    clean = abs(float(jax.grad(f)(jnp.asarray(1e-6))))
    sweep = max(abs(float(jax.grad(f)(jnp.asarray(t))))
                for t in (0.05, 0.1, 0.2, 0.3, 0.5))
    return ad0, fd0, clean, sweep


def _score_theta0_defect(f, name):
    """Score the OPEN eig-VJP degeneracy defect at exactly normal incidence.
    Returns the four measurements.  Split out so the fail-before below can run
    the identical claims against a solver that does NOT carry the defect."""
    ad0, fd0, clean, sweep = _theta0_scores(f)
    # (0) the SYMMETRY oracle: the true derivative is zero.  1e-6 sits ~1e5
    # above float64's own floor for this difference quotient
    # (eps * R / h ~ 7e-12) and ~1e5 below the wrong value; measured 2.8e-11.
    assert abs(fd0) < 1e-6, (
        f"{name}: d(sum R)/d(theta) is forced to ZERO at normal incidence by "
        f"mirror symmetry, but the finite difference reads {fd0:.4e} -- the "
        f"fixture is not symmetric and nothing below means what it says")
    assert np.isfinite(ad0), (
        f"{name}: AD at exactly normal incidence returned {ad0!r}.  The "
        f"spectrum-relative floor exists to keep this FINITE")
    # (1) THE DEFECT, still present.  A future fix should make this fail.
    assert abs(ad0) > _THETA0_DEFECT_RATIO * clean, (
        f"{name}: AD at exactly normal incidence is {ad0:.4e}, only "
        f"{abs(ad0) / max(clean, 1e-300):.4g}x the clean AD at 1e-6 rad "
        f"({clean:.4e}), so the eigenvector-VJP degeneracy no longer corrupts "
        f"it.  That is the KNOWN LIMIT being CLOSED -- re-pin this test "
        f"against the fix (and drop the theta >= 1e-8 advice from the module "
        f"docstring); do not loosen the ratio")
    # (2) THE ENVELOPE: bounded, comparatively.  Not blown up.
    assert abs(ad0) < _THETA0_ENVELOPE_RATIO * sweep, (
        f"{name}: AD at exactly normal incidence is {ad0:.4e}, "
        f"{abs(ad0) / sweep:.4g}x the largest gradient this observable "
        f"attains anywhere on a physical angle sweep ({sweep:.4e}).  The "
        f"wrong value is allowed to be wrong; it is not allowed to be "
        f"UNBOUNDED, which is what an unfloored 1/D on a degenerate pair does")
    return ad0, fd0, clean, sweep


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_pmm1d_angle_gradient_at_exactly_zero_is_an_OPEN_defect(pol):
    """The KNOWN LIMIT (see the module docstring), pinned as a DEFECT.
    A future fix should make this test fail.

    Mirror symmetry forces ``d(sum R)/d(theta) = 0`` at exactly normal
    incidence, but the exactly degenerate +/-m pair makes the eig
    non-differentiable there, so AD returns a finite WRONG value.  This has
    never been a correctness pin; it fences the value.

    2026-08-12 (``docs/audits/FIX_RUNNER_PINS_2026_08_12.md`` S7).  It fenced
    it with an ABSOLUTE bar -- 1e-2 (TE) -- and the ubuntu py3.12 JAX job read
    0.026644.  That bar could not have held, because the fenced quantity is
    pure round-off garbage whose MAGNITUDE is a per-build fact:

        build                                TE AD(0)
        authoring box (2026-07-27)          -2.221e-03
        Windows py3.14 / numpy 2.4.4        +7.793e-03
        CI ubuntu py3.12                    -2.664e-02

    -- a decade of spread and a sign flip, on a bar with 1.28x headroom.
    ADJUDICATED with the floor's own knob: the floored value is INSENSITIVE to
    the degenerate pair's numerical splitting (shrinking it by 1e-8 in a live
    copy of the VJP moves AD(0) from 7.792836e-03 to 7.793047e-03), so it is
    not a ``1/D`` reading at all -- it is the eigenvector JUMP, and which
    direction ``V`` jumps is exactly what a LAPACK build is entitled to
    choose.  The same probe shows what the floor IS for: with the floor
    removed, that 1e-8 shrink takes AD(0) to -1.12e+06.

    So both halves are re-stated as RATIOS between arms measured in the same
    process (see ``_THETA0_DEFECT_RATIO`` / ``_THETA0_ENVELOPE_RATIO``):
    the value must still be WRONG by orders against the clean near-normal
    gradient, and must still be BOUNDED against the observable's own physical
    gradient scale.  Both are invariant to the splitting the runners disagree
    about.

    (Correction to the pre-2026-08-12 docstring, which recorded the unfloored
    value as 7.7x WORSE: on this build it is 2.28x BETTER -- floored
    +7.793e-03 against unfloored -3.411e-03.  Which of the two is larger at an
    exact degeneracy is not a property the floor controls; the boundedness
    under a shrinking splitting is.)

    Use ``theta >= 1e-8`` rad if the angle derivative itself is the objective.
    """
    _score_theta0_defect(lambda a: _pmm1(a, pol=pol), f"pmm1d {pol}")


def test_the_theta0_defect_pin_fires_when_the_defect_is_absent_or_unbounded():
    """THE FAIL-BEFORE for the defect pin above, driven in BOTH directions.

    ``FIX_RUNNER_PINS_2026_08_12`` S7.

    (a) DEFECT ABSENT -- the same claims are scored against ``rcwa1d``, which
        solves the identical grating with ANALYTIC homogeneous half-space
        modes and therefore has no degeneracy at normal incidence (its
        theta = 0 gradient is machine-zero, 1.1e-13, pinned by the control
        test below).  That is what "the eig-VJP degeneracy was fixed" looks
        like from outside, and the pin must FAIL on it, telling the reader to
        re-pin rather than passing quietly.

    (b) VALUE UNBOUNDED -- a LIVE copy of the library's eig VJP, verbatim
        except for two knobs, installed over ``_JAX_EIG_STABLE`` so the whole
        solve runs through it.  The knobs shrink the numerical splitting of
        pairs that are ALREADY degenerate (``|D| < 1e-9`` of the spectrum
        scale -- the well-separated pairs, and hence the physical part of the
        gradient, are untouched) and drop the floor.  A LAPACK build that
        resolves a symmetry-enforced degeneracy more exactly is precisely what
        this emulates; it is the runner, not the physics.  Measured with the
        floor OFF and the splitting shrunk by 1e-4 / 1e-8::

            splitting x    AD(0) TE       AD(0)/sweep TE   AD(0)/sweep TM
            1              -3.411e-03     1.1e-02          6.1e-01
            1e-4           -1.120e+02     3.6e+02          1.9e+04
            1e-8           -1.120e+06     3.6e+06          1.9e+08

        The clean and sweep arms read 1.7541e-06 / 3.1315e-01 in every row,
        i.e. the injector really is targeted.  With the floor ON the same
        1e-8 shrink leaves AD(0) at 7.793047e-03 -- which is both the fail-
        before for the envelope and the measurement that says the floor is
        load-bearing.
    """
    # (a) the defect, absent
    with pytest.raises(AssertionError, match="re-pin this test"):
        _score_theta0_defect(lambda a: _rcwa1(a, pol="te"), "rcwa1d te")

    # (b) the value, unbounded
    orig = _rc._JAX_EIG_STABLE
    try:
        _rc._JAX_EIG_STABLE = _degenerate_splitting_eig(tau_rel=0.0,
                                                        splitting=1e-4)
        with pytest.raises(AssertionError, match="UNBOUNDED"):
            _score_theta0_defect(lambda a: _pmm1(a, pol="te"),
                                 "pmm1d te [unfloored, splitting x 1e-4]")
        # ... and the SHIPPED floor holds the same injection bounded
        _rc._JAX_EIG_STABLE = _degenerate_splitting_eig(tau_rel=_EIG_TAU_REL,
                                                        splitting=1e-8)
        _score_theta0_defect(lambda a: _pmm1(a, pol="te"),
                             "pmm1d te [floored, splitting x 1e-8]")
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
