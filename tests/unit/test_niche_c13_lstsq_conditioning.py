"""``LSTSQ_CONDITIONING_STEPDOWN`` -- the traced fits' least-squares solve is
CERTIFIED against the data instead of assumed well conditioned.

WHAT THE DEFECT IS.  ``_solve_lstsq_thread_safe`` solves every traced
Chebyshev / coordinate / eikonal fit through the NORMAL EQUATIONS,
``G x = A^T b`` with ``G = A^T A``, on the argument -- written into its own
docstring -- that ``A`` is "a well-conditioned normalised tensor-Chebyshev /
monomial Vandermonde ... so squaring the condition number in ``G`` is safe".
That argument is sound for the CONCENTRIC, unweighted fits it was written for
and false for the weighted ones D1/D7 introduced: with
``_FIT_DISC_OUTSIDE_WEIGHT_REL`` = 1e-8 the rows carry two scales, the in-disc
rows alone do not determine a total-degree-10 basis over the whole launch box,
and ``cond(A)`` = 1.4e10 was MEASURED on niche D4's own chain.  Squared, that
is past float64: the Cholesky answer is then an arbitrary draw from the
numerical null space, its fit residual runs 15-23x above the attainable
minimum on one BLAS build and 1.05x on the other, and
``test_niche_d4_dgrating::test_matches_the_manual_hand_split`` read 5.93e-07
on one build against 8.80e-02 on the other RUN ALONE.  See
``docs/audits/C13_DEGREE6_CONDITIONING_2026_08_03.md``.

WHAT IS PINNED HERE, on self-contained synthetic fixtures:

  1. the DEFECT is real and it is the weighting, not the basis -- an unweighted
     fit in the same basis is well conditioned and a two-scale weighted one is
     numerically singular, with a normal-equations residual many times the
     attainable minimum and BUILD-DEPENDENT;
  2. the step-down returns the BETTER fit, measured on the data, and its answer
     is the QR one to the bit;
  3. a fit the normal equations already get right is BITWISE UNCHANGED -- the
     screen skips it -- which is what every byte-identity contract in niches
     C1/C6/C8/C9 rests on;
  4. TIES go to the shipped path, so the margin cannot silently reroute a
     healthy solve;
  5. the QR route never takes ``gelsd`` (B7's ban survives), and it carries
     both RHS shapes;
  6. the screen is scale-INVARIANT (it equilibrates), so a fit cannot be
     rerouted for carrying inconvenient units;
  7. the flag's OFF state is the exact fail-before;
  8. niche C10's own degree-6 residual-eikonal fit is NOT the ill-conditioned
     one -- it passes the screen -- which is the measurement that moved the
     defect off C10 and onto the solver.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements import _lens_traced as LT


# ==========================================================================
# fixtures -- two design matrices in the SAME basis, one healthy, one not
# ==========================================================================
def _cheb_design(x, y, order):
    terms = [(a, c) for a in range(order + 1) for c in range(order + 1 - a)]
    Vx = np.polynomial.chebyshev.chebvander(x, order)
    Vy = np.polynomial.chebyshev.chebvander(y, order)
    return np.ascontiguousarray(
        np.column_stack([Vx[:, a] * Vy[:, c] for a, c in terms]))


def _truth(x, y):
    return 0.7 * x ** 3 - 0.2 * y ** 4 + 0.05 * np.sin(3 * x * y)


@pytest.fixture(scope='module')
def healthy():
    """Unweighted order-8 Chebyshev fit over the whole box -- the fit
    ``_solve_lstsq_thread_safe``'s normal-equations argument was written for."""
    rng = np.random.default_rng(3)
    x = rng.uniform(-1.0, 1.0, 4000)
    y = rng.uniform(-1.0, 1.0, 4000)
    A = _cheb_design(x, y, 8)
    return A, _truth(x, y)


@pytest.fixture(scope='module')
def two_scale():
    """The D1/D7 WEIGHTED branch in miniature: an order-10 basis over the whole
    box, but the samples that carry weight 1 sit in a small OFF-CENTRE disc and
    everything else is down-weighted, exactly as
    ``_FIT_DISC_OUTSIDE_WEIGHT_REL`` arranges.  ``cond(A)`` ~ 3.8e7 here, so
    ``cond(G)`` ~ 1.4e15 -- the same mechanism as the design's 1.4e10 / 1.9e20,
    at a size a unit test can afford."""
    rng = np.random.default_rng(7)
    n_in = n_out = 900
    th = rng.uniform(0.0, 2.0 * np.pi, n_in)
    rr = 0.12 * np.sqrt(rng.uniform(0.0, 1.0, n_in))
    x = np.concatenate([0.55 + rr * np.cos(th), rng.uniform(-1.0, 1.0, n_out)])
    y = np.concatenate([0.20 + rr * np.sin(th), rng.uniform(-1.0, 1.0, n_out)])
    A = _cheb_design(x, y, 10)
    w = np.concatenate([np.ones(n_in), np.full(n_out, 1e-6)])
    return A * w[:, None], _truth(x, y) * w


def _resid(A, b, x):
    return float(np.linalg.norm(b - A @ x))


def _normal_equations_answer(A, b):
    """The pre-C13 solve, reproduced here so the fail-before is asserted rather
    than inherited from the flag."""
    from scipy.linalg import cho_factor, cho_solve
    G = A.T @ A
    return cho_solve(cho_factor(G, check_finite=False), A.T @ b,
                     check_finite=False)


# ==========================================================================
# 1 -- the constants, and the defect they describe
# ==========================================================================
class TestTheContract:

    def test_the_flag_ships_on_with_the_documented_constants(self):
        assert LT.LSTSQ_CONDITIONING_STEPDOWN is True
        assert LT._LSTSQ_GRAM_RCOND_MIN == 1e-8
        assert LT._LSTSQ_RESID_MARGIN == 1e-6

    def test_the_basis_is_healthy_and_the_weighting_is_what_kills_it(
            self, healthy, two_scale):
        """(1) The defect is not the Chebyshev basis and not the ORDER -- it is
        the two-scale weighting.  Same construction, same library, one screens
        clear by six orders and the other is singular AGAINST FLOAT64 ITSELF.

        2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md, D2): the
        weighted line was ``rc_t < 1e-14``, a hand-picked absolute on an
        SVD-derived reciprocal condition number.  MEASURED rc_t = 9.913e-16
        (Win py3.14/np2.4.4) / 9.595e-16 (WSL py3.12/np2.5.1) -- 10.1x on a
        number whose whole content is "smaller than round-off", and 3% of
        cross-build motion in an SVD of a numerically singular Gram is not a
        bound anyone derived.  A rank threshold in float64 IS ``n * eps``
        (LAPACK's own default rcond for an n-column problem), so state it that
        way: with n = 66 the bar is 1.4655e-14 and the measurements clear it by
        14.8x / 15.3x.  Two-sided: the bar is DERIVED from the fixture's own
        column count rather than chosen, it cannot be softened by a BLAS build,
        and a Gram that were merely ill-conditioned rather than singular (rcond
        anywhere above round-off) fails it.  The ``< _LSTSQ_GRAM_RCOND_MIN``
        line below still carries the shipped-screen claim with 7 decades."""
        A_h, _ = healthy
        A_t, _ = two_scale
        rc_h = LT._gram_rcond(A_h.T @ A_h)
        rc_t = LT._gram_rcond(A_t.T @ A_t)
        assert rc_h > 1e-4, rc_h
        assert rc_h >= LT._LSTSQ_GRAM_RCOND_MIN
        n_t = A_t.shape[1]
        assert rc_t < n_t * np.finfo(float).eps, (rc_t, n_t)
        assert rc_t < LT._LSTSQ_GRAM_RCOND_MIN

    def test_the_normal_equations_miss_the_minimum_on_the_weighted_fit(
            self, two_scale):
        """(1) The consequence, in the currency a fit is DEFINED by: the
        normal-equations answer does not minimise ``||b - A x||``, and it misses
        by more than float64 can excuse.

        2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md, D1): this was
        ``r_ne > 10.0 * r_qr`` -- a MAGNITUDE RATIO on a quantity this module's
        own docstring calls "an arbitrary draw from the numerical null space".
        MEASURED r_ne / r_qr = 28.435 (Win py3.14/np2.4.4) vs 22.664 (WSL
        py3.12/np2.5.1): a 25% cross-build spread leaving 2.27x of headroom at
        worst, with ALL the motion in the numerator (r_qr is bit-identical,
        1.714987e-09, on both builds).  And the module docstring records the
        SAME ratio at 1.05x on one BLAS build for the production chain -- so 10x
        pins a per-build magnitude, not the defect.

        THE BUILD-FREE FORM.  ``r_qr`` is the least-squares MINIMUM, so the
        normal-equations answer is WRONG iff its residual exceeds that minimum
        by more than round-off.  The residual NORM of a least-squares problem is
        conditioned by ``cond(A)`` -- NOT by ``cond(G) = cond(A)^2``, which is
        what governs the SOLUTION -- so any backward-stable solve lands within
        ``r_min * (1 + c n eps cond(A))``.  Derive that in-build and assert the
        Cholesky answer sits outside it: presence-of-wrongness, not
        size-of-wrongness.  ``cond(A) = 3.783140e+07`` (bit-identical on both
        builds) and n = 66, so with c = 10 the excusable excess is 5.54e-06 --
        the bar sits 6 parts per million above the attainable minimum.

        TWO-SIDED.  Above: the measurements clear it by 28.43x / 22.66x, and
        even the production chain's WORST recorded build (1.05x) would clear it
        by 1.05x, so no BLAS/LAPACK can turn this into a false failure.  Below:
        the bar is 1 + 5.5e-06, so a solve that actually REACHED the minimum --
        the outcome this test denies -- fails it.  The ``excusable`` guard keeps
        the bound from being defanged should ``cond(A)`` ever drift."""
        A, b = two_scale
        r_ne = _resid(A, b, _normal_equations_answer(A, b))
        r_qr = _resid(A, b, LT._solve_lstsq_qr(A, b))
        n = A.shape[1]
        excusable = r_qr * (1.0 + 10.0 * n * np.finfo(float).eps
                            * float(np.linalg.cond(A)))
        assert excusable < 1.001 * r_qr, excusable      # the bound is still tight
        assert r_ne > excusable, (r_ne, r_qr, excusable)


# ==========================================================================
# 2 -- the step-down
# ==========================================================================
class TestTheStepDown:

    def test_it_returns_the_qr_answer_on_the_weighted_fit(self, two_scale):
        """(2) THE fix: where the Gram is singular the shipped call returns the
        backward-stable answer, to the bit."""
        A, b = two_scale
        got = LT._solve_lstsq_thread_safe(A, b)
        assert np.array_equal(got, LT._solve_lstsq_qr(A, b))
        assert _resid(A, b, got) < _resid(A, b,
                                          _normal_equations_answer(A, b))

    def test_the_healthy_fit_is_bitwise_unchanged(self, healthy):
        """(3) THE null contract.  A fit the screen clears returns the
        historical bits -- not 'agrees to 1e-12', the same float64 words -- so
        every byte-identity pin in C1/C6/C8/C9 is untouched."""
        A, b = healthy
        assert np.array_equal(LT._solve_lstsq_thread_safe(A, b),
                              _normal_equations_answer(A, b))

    def test_the_flag_off_is_the_exact_fail_before(self, two_scale,
                                                   monkeypatch):
        """(7) With the flag down the function is the pre-C13 one, bit for bit,
        ON THE CASE THE FIX EXISTS FOR -- which is what makes the flag a usable
        fail-before rather than a description of one."""
        A, b = two_scale
        monkeypatch.setattr(LT, 'LSTSQ_CONDITIONING_STEPDOWN', False)
        assert np.array_equal(LT._solve_lstsq_thread_safe(A, b),
                              _normal_equations_answer(A, b))

    def test_a_tie_keeps_the_shipped_path(self, two_scale, monkeypatch):
        """(4) The margin is one-sided.  Hand the step-down a 'stable' answer
        that is NOT better and the normal-equations bits must come back --
        otherwise every screened solve would reroute on noise, and two builds
        could disagree about which."""
        A, b = two_scale
        ne = _normal_equations_answer(A, b)
        # a candidate whose residual ties the incumbent to well inside the
        # margin: the incumbent itself, nudged in the last bits.
        monkeypatch.setattr(LT, '_solve_lstsq_qr',
                            lambda _A, _b: ne * (1.0 + 1e-15))
        assert np.array_equal(LT._solve_lstsq_thread_safe(A, b), ne)

    def test_it_is_deterministic(self, two_scale):
        """(2) Two calls, same bits.  The defect this closes was a solve whose
        answer depended on the BLAS build; a solve whose answer depends on
        anything but the data is the same class of defect."""
        A, b = two_scale
        first = LT._solve_lstsq_thread_safe(A, b)
        second = LT._solve_lstsq_thread_safe(A.copy(), b.copy())
        assert np.array_equal(first, second)

    def test_the_step_down_does_not_degrade_a_screened_but_healthy_solve(
            self, two_scale):
        """(2) The step-down can only IMPROVE the fit it is allowed to change:
        whatever it returns fits at least as well as what it replaced, on every
        fixture in this file."""
        A, b = two_scale
        assert _resid(A, b, LT._solve_lstsq_thread_safe(A, b)) <= _resid(
            A, b, _normal_equations_answer(A, b))


# ==========================================================================
# 3 -- the QR route itself
# ==========================================================================
class TestTheQrRoute:

    def test_it_matches_lstsq_on_a_well_posed_problem(self):
        rng = np.random.default_rng(11)
        A = rng.standard_normal((300, 12))
        b = rng.standard_normal(300)
        ref, *_ = np.linalg.lstsq(A, b, rcond=None)
        assert np.allclose(LT._solve_lstsq_qr(A, b), ref, rtol=0, atol=1e-12)

    def test_it_carries_both_rhs_shapes(self):
        rng = np.random.default_rng(12)
        A = rng.standard_normal((200, 7))
        b1 = rng.standard_normal(200)
        b2 = rng.standard_normal((200, 3))
        assert LT._solve_lstsq_qr(A, b1).shape == (7,)
        assert LT._solve_lstsq_qr(A, b2).shape == (7, 3)
        ref, *_ = np.linalg.lstsq(A, b2, rcond=None)
        assert np.allclose(LT._solve_lstsq_qr(A, b2), ref, rtol=0, atol=1e-12)

    def test_it_never_reaches_gelsd_on_a_full_rank_matrix(self, two_scale,
                                                          monkeypatch):
        """(5) B7's ban.  ``np.linalg.lstsq`` is LAPACK ``gelsd``, whose
        OpenBLAS OpenMP pool deadlocks nested inside JAX's; the step-down must
        not be a way back to it.  Detonate ``lstsq`` and the route still
        works."""
        A, b = two_scale

        def _boom(*_a, **_k):
            raise AssertionError('the QR route took the gelsd path')

        monkeypatch.setattr(np.linalg, 'lstsq', _boom)
        x = LT._solve_lstsq_qr(A, b)
        assert np.all(np.isfinite(x))
        # and the whole shipped entry point, on the case that reroutes
        assert np.all(np.isfinite(LT._solve_lstsq_thread_safe(A, b)))


# ==========================================================================
# 4 -- the screen
# ==========================================================================
class TestTheScreen:

    def test_it_is_invariant_to_column_scaling(self, healthy):
        """(6) The screen equilibrates, so it is a statement about the FIT and
        not about the units its columns happen to carry.  Scale the columns
        over sixteen orders: the raw Gram condition number goes to nonsense and
        the screen does not move."""
        A, _ = healthy
        s = np.logspace(-8.0, 8.0, A.shape[1])
        G0 = A.T @ A
        G1 = (A * s).T @ (A * s)
        assert np.linalg.cond(G1) > 1e15                 # scaling wrecks it
        assert LT._gram_rcond(G1) == pytest.approx(LT._gram_rcond(G0),
                                                   rel=1e-6)

    def test_a_scaled_healthy_fit_still_returns_the_shipped_bits(self,
                                                                 healthy):
        """(6) and its consequence: column scaling alone must not push a fit
        onto the step-down, or the null contract would depend on units."""
        A, b = healthy
        s = np.logspace(-6.0, 6.0, A.shape[1])
        As = A * s
        assert np.array_equal(LT._solve_lstsq_thread_safe(As, b),
                              _normal_equations_answer(As, b))

    @pytest.mark.parametrize('bad', ['nan', 'inf', 'indefinite', 'nonsquare'])
    def test_it_returns_zero_for_a_gram_it_cannot_read(self, bad):
        """A screen that cannot read its input must fail TOWARDS the stable
        solver, never away from it."""
        G = np.eye(4)
        if bad == 'nan':
            G[1, 1] = np.nan
        elif bad == 'inf':
            G[2, 0] = np.inf
        elif bad == 'indefinite':
            G[3, 3] = -1.0
        else:
            G = np.eye(4)[:, :3]
        assert LT._gram_rcond(G) == 0.0

    def test_an_exactly_rank_deficient_fit_still_returns_a_finite_answer(self):
        """The end of the chain: a design matrix with a dead column screens to
        zero, the QR's ``R`` is exactly singular, and the last-resort
        minimum-norm branch has to catch it.  A fit that cannot be solved must
        still not return NaN into a ray launch."""
        rng = np.random.default_rng(21)
        A = rng.standard_normal((200, 6))
        A[:, 3] = 0.0                      # a basis term that is dead on the
        A[:, 5] = A[:, 1]                  # samples, and an exact duplicate
        b = rng.standard_normal(200)
        assert LT._gram_rcond(A.T @ A) == 0.0
        x = LT._solve_lstsq_thread_safe(A, b)
        assert x.shape == (6,)
        assert np.all(np.isfinite(x))


# ==========================================================================
# 5 -- niche C10 is exonerated
# ==========================================================================
class TestTheResidualEikonalFitIsNotTheIllConditionedOne:

    def test_the_degree_six_residual_fit_passes_the_screen(self):
        """(8) The attribution that closes C13.  C10's degree-6 fit is what
        MOVED the failure, and the natural reading was that its 27-term
        gradient basis had gone singular.  It has not: its own Gram screens
        clear by six orders, and it screens clear at EVERY degree the cap
        allows.  The singular fit is the one downstream of it."""
        seen = []
        real = LT._solve_lstsq_thread_safe

        def _spy(A, b):
            seen.append(LT._gram_rcond(np.asarray(A).T @ np.asarray(A)))
            return real(A, b)

        n, dx, lam = 96, 4.0e-6, 1.31e-6
        ax = (np.arange(n) - n / 2) * dx
        X, Y = np.meshgrid(ax, ax, indexing='xy')
        w = 8.0 * dx
        k0 = 2.0 * np.pi / lam
        # a smooth residual eikonal with genuine r^6 content, which is the
        # term C10 exists to carry
        a = 3.0e-8 * ((X / (6 * dx)) ** 2 + (Y / (6 * dx)) ** 2) ** 3
        E = np.exp(-(X ** 2 + Y ** 2) / w ** 2) * np.exp(1j * k0 * a)
        try:
            LT._solve_lstsq_thread_safe = _spy
            for deg in (4, 5, 6):
                seen.clear()
                fit = LT._fit_residual_eikonal(
                    E, None, lam, dx, dx, (0.0, 0.0), w, degree=deg)
                assert fit is not None
                assert fit.diag['degree'] == deg
                assert len(seen) == 1
                assert seen[0] >= LT._LSTSQ_GRAM_RCOND_MIN, (deg, seen[0])
        finally:
            LT._solve_lstsq_thread_safe = real
